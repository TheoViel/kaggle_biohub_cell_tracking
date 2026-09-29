"""Dump every fork the edge chain proposes, labelled against the annotation.

The division classifier in this package asks "does this annotated cell divide at
this frame?" over GT cells, with negatives mined from the dividing cells' own
lineages. This builds the other dataset: **the forks the tracking chain actually
proposes**, labelled by whether they are real.

Why that is a different and better-posed question. `main_edge_postprocess`'s ILP
finds far more divisions than `propose_divisions` -- on fold 0, 18 of 26 against
4 -- and drowns them, 442 false forks against 6. Recall is not the constraint;
precision is, and the thing that needs rejecting is a *specific proposal the
pipeline makes*, not an arbitrary cell. So the negatives here are the ILP's own
false forks rather than mined ones: hard by construction, and exactly the
distribution the classifier meets at inference.

Labels come from the annotation and only the annotation:

* **1** -- the fork's parent matches (within `max_distance` um, `SCALE`-aware)
  an annotated GT node that divides.
* **0** -- it matches an annotated GT node that has a child and does not divide.
  The child requirement mirrors `count_matched_pred_divisions`: a matched GT
  node with no children is the end of the annotation, where "did it divide?" is
  unanswerable, and the metric excludes those from its FP tally too.
* **-1** -- no annotated GT node within range. The annotation is sparse, so
  this is most of them -- 64 351 forks on fold 0 against 460 scored ones. These
  are *unlabelled, not negative*: calling them 0 would train the model that the
  unannotated majority never divides, which the data does not say. They are
  written anyway because the threshold at inference has to be applied to every
  fork, not only the scored ones.

One row per (parent, second child) pair, centred on the parent -- the same
convention as `DivisionDataset`, whose positives are GT nodes with out-degree
>= 2, so crops and labels mean the same thing in both datasets.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import io
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main_edge_postprocess as MEP  # noqa: E402
from params import SCALE, TRAIN_DIR  # noqa: E402

torch.set_num_threads(1)

# The candidate-producing configuration: ILP forks grafted onto the tuned
# backbone plus `propose_divisions` on top, which is the widest pool measured
# (18 tp / 442 fp on fold 0). A classifier can only remove, so generate wide.
FORK_CFG = dict(
    ilp=1, ilp_disappearance=0.75, ilp_threshold=0.4,
    ilp_div_forks=1, ilp_fork_division=0.3,
    # Only `require_divergence` gates ILP forks at all; leave it off so the
    # classifier sees the proposals rather than a geometric pre-filter's
    # survivors.
    div_require_divergence=0,
    div_global_frac_cap=1.0, div_frame_frac_cap=1.0,
    divisions=1,
)

COLUMNS = ("t", "z", "y", "x", "label", "parent_dist_um", "sister_dist_um",
           "child_prob", "sister_prob", "n_children",
           # The two daughters' positions, in the same voxel frame as the
           # parent. The scalars above summarise the proposal; these let a
           # dataset *look* at it -- centre the crop on the pair, or rotate so
           # the sister axis is canonical. Both children are at t+1.
           "c1_z", "c1_y", "c1_x", "c2_z", "c2_y", "c2_x")
# Columns 5:10 are the model's scalar features. Named rather than implied,
# because appending to COLUMNS silently changed `table[:, 5:]` for every
# consumer the first time it was done.
FEATURES = ("parent_dist_um", "sister_dist_um", "child_prob", "sister_prob",
            "n_children")


def gt_division_flags(geff_path):
    """`(nodes, divides, has_child)` for one annotated stack.

    `division_cls.dataset.gt_lineage` returns the first two but not the third,
    and the third is what separates "annotated and did not divide" from "the
    annotation stops here, so the question is unanswerable".
    """
    import tracksdata as td

    graph = td.graph.IndexedRXGraph.from_geff(Path(geff_path))
    graph = graph[0] if isinstance(graph, tuple) else graph
    rows = list(graph.node_attrs().iter_rows(named=True))
    index = {int(r["node_id"]): i for i, r in enumerate(rows)}
    nodes = np.array([[r["t"], r["z"], r["y"], r["x"]] for r in rows], np.float32)
    n_out = np.zeros(len(rows), np.int32)
    for e in graph.edge_attrs().iter_rows(named=True):
        n_out[index[int(e["source_id"])]] += 1
    return nodes, n_out >= 2, n_out >= 1


def _label_forks(parents_zyx, gt_nodes, gt_divides, gt_has_child, max_distance=7.0,
                 max_dt=1):
    """Label each fork parent against the annotation. See the module docstring.

    ``max_dt`` is how many frames away a GT division may sit and still be the
    event this fork is proposing. **It is not a tolerance, it is the metric's
    own semantics.** `score_divisions` never asks the predicted fork to be at
    the GT dividing node's frame: it matches nodes to the GT division subgraph
    and credits any fork inside a weakly-connected component that spans the
    one-node stage and the two-node stage. A chain that splits the track one
    frame early or late still recovers the division and still scores a TP.

    Labelling frame-exactly (``max_dt=0``, the old behaviour) therefore calls a
    large slice of real divisions negative, and the classifier is trained to
    reject exactly the forks the metric would have rewarded. Measured on the v4
    harvest pool, over 151 annotated divisions:

        within 7 um and +/-0 frames :  57   <- what the labeller used to see
        within 7 um and +/-1 frames :  76
        within 7 um and +/-2 frames :  82
        the metric, nothing gated   :  81 TP

    So ~24 divisions sat in the pool labelled 0. 1 is the default because it
    recovers most of the gap while keeping the match unambiguous; 2 buys six
    more and starts pulling in neighbouring events.

    A dividing node wins ties: if any GT node within reach divides, this fork is
    proposing that event. Otherwise the nearest match decides, and a matched
    node with no child leaves the fork unlabelled -- the annotation stops there,
    so the question is unanswerable rather than answered "no".
    """
    from scipy.spatial import cKDTree

    if not len(gt_nodes):
        return np.full(len(parents_zyx), -1, np.int8)
    scale = np.asarray(SCALE)
    labels = np.full(len(parents_zyx), -1, np.int8)
    gt_t = gt_nodes[:, 0].astype(int)
    trees = {}
    for t in np.unique(gt_t):
        rows = np.flatnonzero(gt_t == t)
        trees[int(t)] = (rows, cKDTree(gt_nodes[rows, 1:] * scale))

    for t in np.unique(parents_zyx[:, 0]).astype(int):
        rows = np.flatnonzero(parents_zyx[:, 0].astype(int) == t)
        pts = parents_zyx[rows, 1:] * scale
        best = np.full(len(rows), np.inf)
        best_gt = np.full(len(rows), -1, np.int64)
        div_hit = np.zeros(len(rows), bool)
        for dt in range(-max_dt, max_dt + 1):
            entry = trees.get(int(t) + dt)
            if entry is None:
                continue
            gt_rows, tree = entry
            d, idx = tree.query(pts)
            hit = d <= max_distance
            matched = gt_rows[idx]
            # A dividing match claims the fork outright, at any offset in range.
            claim = hit & gt_divides[matched]
            div_hit |= claim
            better = hit & (d < best)
            best[better] = d[better]
            best_gt[better] = matched[better]
        lab = np.full(len(rows), -1, np.int8)
        seen = best_gt >= 0
        lab[seen] = np.where(gt_has_child[best_gt[seen]], 0, -1).astype(np.int8)
        lab[div_hit] = 1
        labels[rows] = lab
    return labels


def forks_table(coords, edges):
    """`(coords, edges)` -> one row per (parent, non-top child), label column 0.

    Split out of `_one` so that anything re-deriving the pool under a different
    chain config -- `sweep_candidates.py` -- enumerates forks the same way this
    does. Two implementations of "what counts as a fork" would be two different
    datasets wearing one name, and the classifier's scores are keyed on the
    rows this produces.
    """
    edges = np.asarray(edges, float).reshape(-1, 4)
    if not len(edges):
        return np.zeros((0, len(COLUMNS)), np.float32)

    src = edges[:, 0].astype(int)
    order = np.argsort(-edges[:, 2], kind="stable")
    children: dict[int, list[int]] = {}
    for k in order:
        children.setdefault(int(src[k]), []).append(int(k))

    scale = np.asarray(SCALE)
    rows = []
    for parent, ks in children.items():
        if len(ks) < 2:
            continue
        first = int(edges[ks[0], 1])
        for k in ks[1:]:
            q = int(edges[k, 1])
            rows.append((
                *coords[parent],                                   # t, z, y, x
                0.0,                                               # label, filled below
                float(np.linalg.norm((coords[parent, 1:] - coords[q, 1:]) * scale)),
                float(np.linalg.norm((coords[first, 1:] - coords[q, 1:]) * scale)),
                float(edges[k, 2]), float(edges[ks[0], 2]), float(len(ks)),
                *coords[first, 1:], *coords[q, 1:],                # c1 zyx, c2 zyx
            ))
    if not rows:
        return np.zeros((0, len(COLUMNS)), np.float32)
    return np.asarray(rows, np.float32)


def label_table(table, stem, data_dir=TRAIN_DIR, max_dt=1):
    """The label column for a fork table: 1 divides, 0 does not, -1 unannotated."""
    if not len(table):
        return np.zeros(0, np.float32)
    geff = Path(data_dir) / f"{stem}.geff"
    if not geff.exists():
        return np.full(len(table), -1, np.float32)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        gt_nodes, gt_divides, gt_has_child = gt_division_flags(geff)
    return _label_forks(table[:, :4], gt_nodes, gt_divides, gt_has_child,
                        max_dt=max_dt)


def gt_fork_rows(table, geff_path, max_distance=7.0, max_dt=1):
    """Fork rows for the annotated divisions the chain never proposed.

    The pool only ever contains divisions the ILP found a fork for, so the
    classifier trains on a biased 38% of them and never sees the appearance of
    the ones the chain misses -- which are the ones worth recovering. Measured
    on the v4 harvest: 151 annotated divisions, 92 with the parent *detected*
    but no fork proposed at its frame, only 2 undetected. The signal is in the
    images; the proposer is what does not reach it.

    So the missing divisions are added straight from the annotation: parent and
    both daughters at their GT positions. This is the same move
    `edge_transformer/data.py` makes with ``with_gt`` -- append the annotated
    nodes no detection reached -- and it carries the same warning: these rows
    are training signal only. They are not in the chain's pool, so a model
    scored on them would be answering a question the pipeline never asks.

    The geometry columns are computed from the annotation. The two probability
    columns have no meaning for a fork the model never proposed -- there is no
    edge, so there is no edge probability. They are **sampled from the stack's
    own fork probabilities** rather than set to a constant or to NaN.

    NaN was the first attempt and it is wrong: ``fuse_features`` configurations
    (``f5``, ``c48f5``) read these columns, and the NaN propagates through the
    model into `average_precision_score`, which raises. Off by default in the
    base config is not the same as unused.

    A constant is worse than useless: every appended row is a positive, so a
    fixed value is a memorisable tell that the row is GT-appended, and a fused
    model would learn the tell instead of the appearance. Sampling from the
    observed distribution leaves the column uninformative about the label,
    which is what "this fork has no probability evidence" should look like.
    """
    import tracksdata as td
    from scipy.spatial import cKDTree

    geff = Path(geff_path)
    if not geff.exists():
        return np.zeros((0, len(COLUMNS)), np.float32)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        graph = td.graph.IndexedRXGraph.from_geff(geff)
        graph = graph[0] if isinstance(graph, tuple) else graph
        rows = list(graph.node_attrs().iter_rows(named=True))
        index = {int(r["node_id"]): i for i, r in enumerate(rows)}
        nodes = np.array([[r["t"], r["z"], r["y"], r["x"]] for r in rows], np.float32)
        kids: dict[int, list[int]] = {}
        for e in graph.edge_attrs().iter_rows(named=True):
            kids.setdefault(index[int(e["source_id"])], []).append(index[int(e["target_id"])])

    scale = np.asarray(SCALE)
    have = table[:, :4] if len(table) else np.zeros((0, 4), np.float32)
    # Seeded per stack so the pool is reproducible from the command alone.
    rng = np.random.default_rng(abs(hash(Path(geff_path).stem)) % (2 ** 32))
    ip, isis = list(COLUMNS).index("child_prob"), list(COLUMNS).index("sister_prob")
    child_pool = (table[:, ip][np.isfinite(table[:, ip])] if len(table)
                  else np.array([], np.float32))
    sister_pool = (table[:, isis][np.isfinite(table[:, isis])] if len(table)
                   else np.array([], np.float32))
    if not len(child_pool):
        child_pool = np.array([0.5], np.float32)
    if not len(sister_pool):
        sister_pool = np.array([0.5], np.float32)
    out = []
    for i, ks in kids.items():
        if len(ks) < 2:
            continue
        parent = nodes[i]
        t = int(parent[0])
        if len(have):
            sel = have[np.abs(have[:, 0].astype(int) - t) <= max_dt]
            if len(sel) and cKDTree(sel[:, 1:] * scale).query(parent[1:] * scale)[0] <= max_distance:
                continue            # the chain already proposes this one
        c1, c2 = nodes[ks[0]], nodes[ks[1]]
        out.append((
            *parent, 1.0,
            float(np.linalg.norm((parent[1:] - c2[1:]) * scale)),
            float(np.linalg.norm((c1[1:] - c2[1:]) * scale)),
            float(rng.choice(child_pool)), float(rng.choice(sister_pool)), 2.0,
            *c1[1:], *c2[1:],
        ))
    if not out:
        return np.zeros((0, len(COLUMNS)), np.float32)
    return np.asarray(out, np.float32)


def _one(stem_pred):
    stem, pred_dir, data_dir, max_dt, append_gt, pp_overrides = stem_pred
    z = np.load(Path(pred_dir) / f"{stem}.npz")
    cfg = dict(MEP.DEFAULTS)
    cfg.update(out_dir="", geff_dir="", no_score=True, assign=1)
    cfg.update(FORK_CFG)
    if pp_overrides:
        # The pool is whatever chain proposes it, and FORK_CFG is only *our*
        # chain. A partner's branch reaches div_J 0.4146 against our 0.2557 on
        # the same annotation with the ILP off and the geometric proposer wide
        # open, so the pool has to be expressible here or it cannot be trained
        # on or compared against.
        cfg.update(pp_overrides)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        coords, edges = MEP.postprocess(
            z["coords"].astype(np.float64), np.empty((0, 4)),
            np.asarray(z["candidates"], float).reshape(-1, 5), cfg)
    table = forks_table(coords, edges)
    if len(table):
        table[:, 4] = label_table(table, stem, data_dir, max_dt=max_dt)
    if append_gt:
        extra = gt_fork_rows(table, Path(data_dir) / f"{stem}.geff", max_dt=max_dt)
        if len(extra):
            table = np.concatenate([table, extra]) if len(table) else extra
    return stem, table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preds_dir", required=True,
                   help="folder of <stem>.npz edge predictions")
    p.add_argument("--data_dir", default=TRAIN_DIR)
    p.add_argument("--out", required=True, help="output .npz")
    p.add_argument("--workers", type=int, default=28)
    p.add_argument("--embryo", default=None, help="prefix filter, e.g. 44b6")
    p.add_argument("--label_dt", type=int, default=1,
                   help="frames a GT division may sit from the fork and still "
                        "be the event it proposes. The metric matches whole "
                        "components and never demands the same frame, so 0 "
                        "(the old behaviour) labels real divisions negative: "
                        "57 positives against 76 at 1 and 82 at 2, out of 151")
    p.add_argument("--pp_overrides", default=None,
                   help="JSON of main_edge_postprocess settings applied on top "
                        "of FORK_CFG, so a pool from a different chain can be "
                        "built and compared. Recorded in the output.")
    p.add_argument("--append_gt", action="store_true",
                   help="add a positive row for every annotated division the "
                        "chain proposes no fork for. Training signal only -- "
                        "these are not in the pool the pipeline scores")
    args = p.parse_args()

    pred_dir = Path(args.preds_dir)
    stems = sorted(q.stem for q in pred_dir.glob("*.npz")
                   if args.embryo is None or q.stem.startswith(args.embryo))
    print(f"{len(stems)} stacks from {pred_dir}", flush=True)

    overrides = json.loads(args.pp_overrides) if args.pp_overrides else None
    jobs = [(s, str(pred_dir), str(args.data_dir), args.label_dt, args.append_gt,
             overrides) for s in stems]
    stem_col, tables = [], []
    with Pool(args.workers) as pool:
        for i, (stem, table) in enumerate(pool.imap_unordered(_one, jobs), 1):
            if len(table):
                stem_col += [stem] * len(table)
                tables.append(table)
            if i % 20 == 0 or i == len(stems):
                print(f"  {i}/{len(stems)} stacks, {sum(len(t) for t in tables)} forks",
                      flush=True)

    table = np.concatenate(tables) if tables else np.zeros((0, len(COLUMNS)), np.float32)
    labels = table[:, 4].astype(np.int8)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, stem=np.asarray(stem_col), table=table,
                        columns=np.asarray(COLUMNS),
                        pp_overrides=np.asarray(args.pp_overrides or ""))
    print(f"\n{len(table)} forks -> {out}   (label_dt={args.label_dt}, "
          f"append_gt={args.append_gt})")
    print(f"  labelled positive : {(labels == 1).sum()}")
    print(f"  labelled negative : {(labels == 0).sum()}")
    print(f"  unlabelled        : {(labels == -1).sum()}")
    for e in ("44b6", "6bba"):
        m = np.asarray([s.startswith(e) for s in stem_col], bool)
        if m.any():
            print(f"  {e}: {(labels[m] == 1).sum()} pos, {(labels[m] == 0).sum()} neg, "
                  f"{(labels[m] == -1).sum()} unlabelled")


if __name__ == "__main__":
    main()
