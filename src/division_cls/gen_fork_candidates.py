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


def _label_forks(parents_zyx, gt_nodes, gt_divides, gt_has_child, max_distance=7.0):
    """Label each fork parent against the annotation. See the module docstring."""
    from scipy.spatial import cKDTree

    if not len(gt_nodes):
        return np.full(len(parents_zyx), -1, np.int8)
    scale = np.asarray(SCALE)
    labels = np.full(len(parents_zyx), -1, np.int8)
    # Match within a frame: a GT node at another timepoint is a different event.
    for t in np.unique(parents_zyx[:, 0]).astype(int):
        rows = np.flatnonzero(parents_zyx[:, 0].astype(int) == t)
        gt_rows = np.flatnonzero(gt_nodes[:, 0].astype(int) == t)
        if not len(gt_rows):
            continue
        tree = cKDTree(gt_nodes[gt_rows, 1:] * scale)
        d, idx = tree.query(parents_zyx[rows, 1:] * scale)
        hit = d <= max_distance
        matched = gt_rows[idx[hit]]
        # A matched node with no child is the end of the annotation: the metric
        # leaves those out of its FP tally, so they stay unlabelled here too.
        lab = np.where(gt_divides[matched], 1,
                       np.where(gt_has_child[matched], 0, -1)).astype(np.int8)
        labels[rows[hit]] = lab
    return labels


def _one(stem_pred):
    stem, pred_dir, data_dir = stem_pred
    z = np.load(Path(pred_dir) / f"{stem}.npz")
    cfg = dict(MEP.DEFAULTS)
    cfg.update(out_dir="", geff_dir="", no_score=True, assign=1)
    cfg.update(FORK_CFG)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        coords, edges = MEP.postprocess(
            z["coords"].astype(np.float64), np.empty((0, 4)),
            np.asarray(z["candidates"], float).reshape(-1, 5), cfg)
    edges = np.asarray(edges, float).reshape(-1, 4)
    if not len(edges):
        return stem, np.zeros((0, len(COLUMNS)), np.float32)

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
        return stem, np.zeros((0, len(COLUMNS)), np.float32)
    table = np.asarray(rows, np.float32)

    geff = Path(data_dir) / f"{stem}.geff"
    if geff.exists():
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            gt_nodes, gt_divides, gt_has_child = gt_division_flags(geff)
        table[:, 4] = _label_forks(table[:, :4], gt_nodes, gt_divides, gt_has_child)
    else:
        table[:, 4] = -1
    return stem, table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preds_dir", required=True,
                   help="folder of <stem>.npz edge predictions")
    p.add_argument("--data_dir", default=TRAIN_DIR)
    p.add_argument("--out", required=True, help="output .npz")
    p.add_argument("--workers", type=int, default=28)
    p.add_argument("--embryo", default=None, help="prefix filter, e.g. 44b6")
    args = p.parse_args()

    pred_dir = Path(args.preds_dir)
    stems = sorted(q.stem for q in pred_dir.glob("*.npz")
                   if args.embryo is None or q.stem.startswith(args.embryo))
    print(f"{len(stems)} stacks from {pred_dir}", flush=True)

    jobs = [(s, str(pred_dir), str(args.data_dir)) for s in stems]
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
                        columns=np.asarray(COLUMNS))
    print(f"\n{len(table)} forks -> {out}")
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
