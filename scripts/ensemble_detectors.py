#!/usr/bin/env python
"""Ensemble two detectors by appending the detections the base one missed.

Node set, per frame:

    N = BASE(th_base)  U  { a in ADD(th_add) : dist(a, BASE) > d_add }

Edges are rebuilt by nearest-neighbour linking between consecutive frames,
gated at ``--max-disp`` um. That is sound here because the GT frame-to-frame
displacement is p50 1.2 / p99 6.6 um against a ~8.3 um nearest-neighbour cell
spacing, so the linking is well determined and the resulting edge Jaccard reads
mainly as a verdict on the *node set* -- which is what an ensemble of detectors
changes. It also sidesteps the fact that a merged node set has no edge-head
output for the appended nodes.

Scoring is the repo's own ``evaluate`` / ``per_sample_metrics`` / ``summarise``,
macro-averaged over embryo groups exactly as ``main_predict`` does, so ``score``
is comparable to a run's cv_score. A hand-rolled reimplementation was tried
first and rejected: it ranked 13/1 and 23/6 the opposite way round to the real
metric. The node-count adjustment ``J * (1 - 0.1 * (N_pred - N_total)/N_total)``
is what makes over-detection cost something, and it is why the sweep prefers
thresholds *above* each model's solo optimum -- recall alone would pick a much
looser cut.

``d_add`` should not go below the NMS radius the predictions were made at:
appending a node closer than that recreates exactly the pair the base model's
own NMS removed. Measured on r6 predictions, the optimum sits at 7 um and the
score is flat from 7 to 12, so the choice is not delicate as long as it clears
the radius.

Predictions are read from each run's stored ``preds_<tag>`` folder; a higher
threshold is applied by filtering on ``node_probs``, which is exact for an
already-NMS'd set (a peak that survives a higher cut still suppresses whatever
it suppressed before).

Measured on 199 OOF samples, r6 predictions, NN linking:

    13/1 alone @ 0.40                 0.7658
    23/6 alone @ 0.45                 0.7522
    base=13/1 0.45, add=23/6 0.40, d=7   0.7738   (+0.0081)

    python scripts/ensemble_detectors.py --base logs/2026-09-13/1 \
        --add logs/2026-09-23/6 --cfg 0.45 0.40 7 --all
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import logging
import warnings
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from params import SCALE  # noqa: E402

logging.getLogger("tracksdata").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

TRAIN = REPO / "input" / "train"
SC = np.array(SCALE)
RUN_A: Path | None = None   # base run; set by main() from --base
RUN_B: Path | None = None   # run appended from; set by main() from --add


def nn_edges(coords: np.ndarray, max_disp: float, mutual: bool) -> np.ndarray:
    """Link each node to its nearest neighbour in the next frame."""
    t = np.round(coords[:, 0]).astype(int)
    by_t = {int(f): np.where(t == f)[0] for f in np.unique(t)}
    out = []
    for f in sorted(by_t):
        i, j = by_t.get(f), by_t.get(f + 1)
        if j is None or not len(i) or not len(j):
            continue
        src, tgt = coords[i][:, 1:4] * SC, coords[j][:, 1:4] * SC
        d_f, n_f = cKDTree(tgt).query(src, k=1)
        ok = d_f <= max_disp
        if mutual:
            # Only keep a link when the target's own nearest is the source; a
            # one-sided NN lets two nodes collapse onto the same successor,
            # which the metric counts as an extra valid-but-wrong edge.
            _, n_b = cKDTree(src).query(tgt, k=1)
            ok &= n_b[n_f] == np.arange(len(src))
        if ok.any():
            out.append(np.stack([i[ok.nonzero()[0]], j[n_f[ok]]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 2), int)


def merge_idx(ca, pa, cb, pb, th_a, th_b, d_add):
    """Which rows of each source survive, and the order they end up in.

    Returns ``(b_idx, a_idx, order)``: indices into the base arrays, indices
    into the appended arrays, and the permutation that sorts the concatenation
    ``[base[b_idx], add[a_idx]]`` by frame. Indices rather than coordinates so
    every per-node array (node_probs, annot_probs, coords_f) can be carried
    through the same selection -- scoring and saving then cannot disagree.
    """
    b_idx = np.nonzero(pb >= th_b)[0]
    a_idx = np.nonzero(pa >= th_a)[0]

    if len(a_idx) and len(b_idx) and d_add > 0:
        b = cb[b_idx]
        t_b = np.round(b[:, 0]).astype(int)
        t_a = np.round(ca[a_idx][:, 0]).astype(int)
        keep = np.zeros(len(a_idx), bool)
        for f in np.unique(t_a):
            sel = t_a == f
            bb = b[t_b == f]
            if not len(bb):
                keep[sel] = True
                continue
            d, _ = cKDTree(bb[:, 1:4] * SC).query(ca[a_idx[sel]][:, 1:4] * SC, k=1)
            keep[sel] = d > d_add
        a_idx = a_idx[keep]
    # Otherwise every row above th_a is kept: either the base is empty (which
    # is how the single-model baselines are built, th_b > 1), or d_add <= 0,
    # which means "append unconditionally".

    t = np.concatenate([cb[b_idx][:, 0], ca[a_idx][:, 0]]) if len(b_idx) or len(a_idx) \
        else np.zeros(0)
    return b_idx, a_idx, np.argsort(t, kind="stable")


def merge(ca, pa, cb, pb, th_a, th_b, d_add):
    """The merged coordinate array alone, for scoring."""
    b_idx, a_idx, order = merge_idx(ca, pa, cb, pb, th_a, th_b, d_add)
    if not len(b_idx) and not len(a_idx):
        return np.zeros((0, 4))
    return np.concatenate([cb[b_idx], ca[a_idx]])[order]


def _radius_of(tag: str) -> float | None:
    """The NMS radius a preds_<tag> folder was produced at, e.g. r6_0.30 -> 6."""
    m = re.match(r"r(\d+(?:\.\d+)?)_", tag)
    return float(m.group(1)) if m else None


def _measured_for(a: Path, b: Path, radius: float, th: float) -> dict:
    """This pair's own row from the consensus sweep, if it was measured.

    Hardcoding one pair's numbers here was wrong: the block was copied into
    every blend regardless of which runs it actually came from.
    """
    out = {"_note": "repo metric, NN-linked edges -- comparable to other rows "
                    "of the same sweep, not to an edge-head cv_score",
           "source": "output/consensus_sweep.tsv"}
    tsv = REPO / "output" / "consensus_sweep.tsv"
    if not tsv.exists():
        return out
    want = {str(a.relative_to(REPO / "logs")), str(b.relative_to(REPO / "logs"))}
    for line in tsv.read_text().splitlines()[1:]:
        f = line.split("\t")
        if len(f) < 10 or f[1] not in want:
            continue
        if abs(float(f[3]) - radius) > 1e-9 or abs(float(f[4]) - th) > 1e-9:
            continue
        out["this_pair"] = {"node_recall": float(f[5]), "score": float(f[6]),
                            "edge_jaccard": float(f[7]), "ratio": float(f[9])}
    return out


def consensus_clusters(ca, pa, cb, pb, radius, th_cluster=0.30):
    """Cluster both models' detections; return fused nodes and their provenance.

    Greedy, seeded by the strongest remaining detection -- the same traversal
    greedy NMS and ``centroid_merge`` use, so a cluster is always one radius
    around a real high-confidence point and cannot chain across a dense region
    the way single linkage would.

    The fused score is the mean of the two models' probabilities *with a zero
    filled in for the model that did not fire*. That is the whole idea: a cell
    both models found at 0.40 keeps 0.40, while one model alone at 0.40 drops
    to 0.20, so a lone detection has to be twice as confident to survive the
    same cut. Every append recipe was capped by the node-count penalty at ratio
    ~0.12; this is what gets underneath it without giving up recall.

    Returns ``(coords, score, rep_src, rep_idx, n_src)`` -- the fused centroid,
    the fused score, which row of which source was the cluster's strongest
    member (so per-node columns that cannot be averaged, like annot_probs, come
    from a real detection rather than being invented), and how many of the two
    models contributed to the cluster.
    """
    a_keep = np.nonzero(pa >= th_cluster)[0]
    b_keep = np.nonzero(pb >= th_cluster)[0]
    if not len(a_keep) and not len(b_keep):
        return (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int),
                np.zeros(0, int), np.zeros(0, int))

    pts = np.concatenate([ca[a_keep], cb[b_keep]])
    prob = np.concatenate([pa[a_keep], pb[b_keep]])
    src = np.concatenate([np.zeros(len(a_keep), int), np.ones(len(b_keep), int)])
    row = np.concatenate([a_keep, b_keep])

    out_c, out_s, out_src, out_row, out_n = [], [], [], [], []
    frames = np.round(pts[:, 0]).astype(int)
    for f in np.unique(frames):
        m = frames == f
        P, Q, S, R = pts[m][:, 1:4], prob[m], src[m], row[m]
        tree = cKDTree(P * SC)
        alive = np.ones(len(P), bool)
        for i in np.argsort(-Q):
            if not alive[i]:
                continue
            # Queried per seed, not batched for the whole frame: greedy only
            # ever removes points, so roughly half these points are suppressed
            # before they are ever seeded and their neighbour list is never
            # needed. Batching measured 5x slower for identical output.
            idx = np.asarray(tree.query_ball_point(P[i] * SC, radius), dtype=int)
            idx = idx[alive[idx]]
            w = Q[idx]
            out_c.append(np.concatenate([[f], (P[idx] * w[:, None]).sum(0) / w.sum()]))
            out_s.append(sum(w[S[idx] == t].max() if (S[idx] == t).any() else 0.0
                             for t in (0, 1)) / 2.0)
            out_src.append(int(S[i])); out_row.append(int(R[i]))
            out_n.append(int(len(np.unique(S[idx]))))
            alive[idx] = False
    order = np.argsort([c[0] for c in out_c], kind="stable")
    return (np.asarray(out_c)[order], np.asarray(out_s)[order],
            np.asarray(out_src)[order], np.asarray(out_row)[order],
            np.asarray(out_n)[order])


def save_consensus(out_dir: Path, base: Path, add: Path, tag: str,
                   names: list[str], radius: float, th: float,
                   th_cluster: float, n_jobs: int = 32) -> dict:
    """Write the fused node sets in the source preds_<tag> layout.

    Unlike the append blend this is not a selection of source rows, so the
    columns are built rather than indexed:

      coords / coords_f  the confidence-weighted centroid -- coords_f keeps the
                         fraction, coords is the rounded voxel, which is what
                         the two columns already mean in the source files.
      node_probs         the fused consensus score, NOT either model's raw
                         probability. It is on a different scale to a
                         single-model file: re-thresholding this column is a
                         different operation and 0.5 here means "both models
                         agreed at 0.5", not "one model said 0.5".
      annot_probs        carried from the cluster's strongest member, since an
                         average of it would not correspond to any detection.
      edges              empty (0, 4); main_edge.py recomputes them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(n):
        db = np.load(base / f"preds_{tag}" / f"{n}.npz")
        da = np.load(add / f"preds_{tag}" / f"{n}.npz")
        c, s, rsrc, rrow, nsrc = consensus_clusters(
            da["coords"].astype(float), da["node_probs"].astype(float),
            db["coords"].astype(float), db["node_probs"].astype(float),
            radius, th_cluster,
        )
        keep = s >= th
        c, s, rsrc, rrow, nsrc = (c[keep], s[keep], rsrc[keep],
                                  rrow[keep], nsrc[keep])
        # Genuine provenance: how many of the two models contributed a
        # detection to the cluster. (An earlier version compared the fused
        # score against cluster_th, which every surviving node clears by
        # construction and so always read 100%.)
        agreed = int((nsrc == 2).sum())
        annot = np.array([ (da if t == 0 else db)["annot_probs"][r]
                           for t, r in zip(rsrc, rrow) ], dtype=np.float32) \
            if len(c) else np.zeros(0, np.float32)
        np.savez_compressed(
            out_dir / f"{n}.npz",
            coords=np.rint(c).astype(db["coords"].dtype),
            coords_f=c.astype(np.float32),
            node_probs=s.astype(np.float32),
            annot_probs=annot,
            edges=np.zeros((0, 4), dtype=db["edges"].dtype),
        )
        return n, int(len(c)), agreed

    got = Parallel(n_jobs=n_jobs, backend="loky")(delayed(one)(n) for n in names)
    n_nodes = sum(k for _, k, _ in got)
    n_agreed = sum(a for _, _, a in got)
    per_sample = {n: {"n_nodes": k, "n_both_models": a} for n, k, a in got}
    return {"n_samples": len(names), "n_nodes": n_nodes,
            "n_found_by_both": n_agreed,
            "frac_found_by_both": n_agreed / n_nodes if n_nodes else float("nan"),
            "per_sample": per_sample}


def node_arrays(db, da) -> list[str]:
    """Per-node array names present in *both* sources, so both can be indexed.

    Anything whose first axis matches that file's node count is a per-node
    column and travels with the node: node_probs, annot_probs, coords_f, and
    the ``embeddings_contrastive_*`` features `main_edge.py` needs. A key only
    one run carries is dropped -- half a column would be worse than none.
    """
    def per_node(d):
        n = len(d["coords"])
        return {k for k in d.files
                if k != "edges" and np.asarray(d[k]).shape[:1] == (n,)}
    return sorted(per_node(db) & per_node(da))


def save_merged(out_dir: Path, base: Path, add: Path, tag: str,
                names: list[str], th_base: float, th_add: float, d_add: float,
                command: str) -> dict:
    """Write one npz per sample, in the source preds_<tag> layout.

    Every per-node array is carried over from whichever model contributed the
    node, with its original dtype, so a consumer cannot tell a merged file from
    a single-model one by its schema.

    `edges` is written as an empty (0, 4) array. The key is kept so the layout
    matches, but the merged node set has no edge-head output for the appended
    nodes and NN links are a scoring device, not a prediction -- writing them
    here would pass off a placeholder as a model output.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    src = {}
    for run in (base, add):
        src[run] = {n: np.load(run / f"preds_{tag}" / f"{n}.npz") for n in names}

    n_base = n_add = 0
    carried = None
    per_sample = {}
    for n in names:
        db, da = src[base][n], src[add][n]
        b_idx, a_idx, order = merge_idx(
            da["coords"].astype(float), da["node_probs"].astype(float),
            db["coords"].astype(float), db["node_probs"].astype(float),
            th_add, th_base, d_add,
        )
        keys = node_arrays(db, da)
        if carried is None:
            carried = keys
        elif keys != carried:
            raise SystemExit(
                f"{n}: per-node arrays {keys} differ from {carried} seen "
                "earlier; the folder would be inconsistent")
        out = {}
        for k in keys:
            out[k] = np.concatenate([db[k][b_idx], da[k][a_idx]])[order]
        out["edges"] = np.zeros((0, 4), dtype=db["edges"].dtype)
        np.savez_compressed(out_dir / f"{n}.npz", **out)
        n_base += len(b_idx); n_add += len(a_idx)
        per_sample[n] = {"n_nodes": int(len(b_idx) + len(a_idx)),
                         "from_base": int(len(b_idx)), "from_add": int(len(a_idx))}

    total = n_base + n_add
    return {"n_samples": len(names), "n_nodes": total, "arrays": carried,
            "n_from_base": n_base, "n_from_add": n_add,
            "frac_from_add": n_add / total if total else float("nan"),
            "per_sample": per_sample}


_GT = {}
_META = None


def gt_graph(name):
    if name not in _GT:
        import tracksdata as td
        g = td.graph.IndexedRXGraph.from_geff(TRAIN / f"{name}.geff")
        _GT[name] = g[0] if isinstance(g, tuple) else g
    return _GT[name]


def n_total(name):
    global _META
    if _META is None:
        import pandas as pd
        p = TRAIN.parent / "metadata.csv"
        _META = pd.read_csv(p).set_index("sample_name") if p.exists() else None
    if _META is not None and name in _META.index:
        return float(_META.loc[name, "estimated_number_of_nodes"])
    return float("nan")


def score_one(name, coords, edges):
    from tracking.post_process import build_graph
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics
    g = build_graph(coords, [(int(s), int(t), 1.0, 0.0) for s, t in edges])
    gt = gt_graph(name)
    res = evaluate(g, gt, scale=SCALE)
    return per_sample_metrics(res, n_total(name), node_recall(g, gt))


def macro(rows_by_name):
    """main_predict's aggregation: summarise per embryo, then mean the groups."""
    from tracking_cellmot.metrics import summarise
    by_emb = defaultdict(list)
    for name, row in rows_by_name.items():
        by_emb[name.split("_")[0]].append(row)
    per = {e: summarise(v) for e, v in by_emb.items()}
    out = {}
    for k in ("score", "edge_jaccard", "adj_edge_jaccard", "node_recall"):
        out[k] = float(np.nanmean([s[k] for s in per.values()]))
    out["n_pred"] = float(np.mean([r["num_pred_nodes"] for r in rows_by_name.values()]))
    out["ratio"] = float(np.nanmean(
        [r["total_node_ratio"] for r in rows_by_name.values()]))
    return out, per


def run_cfg(names, data, th_a, th_b, d_add, max_disp, mutual, n_jobs):
    def one(name):
        ca, pa, cb, pb = data[name]
        c = merge(ca, pa, cb, pb, th_a, th_b, d_add)
        if not len(c):
            return name, None
        return name, score_one(name, c, nn_edges(c, max_disp, mutual))

    got = Parallel(n_jobs=n_jobs, backend="loky")(delayed(one)(n) for n in names)
    rows = {n: r for n, r in got if r is not None}
    return macro(rows)


def load(base: Path, add: Path, tag: str, names: list[str]) -> dict:
    """(base_coords, base_probs, add_coords, add_probs) per sample.

    `merge` treats its *second* pair as the base, so the pairs are stored in
    that order and no caller has to remember the convention.
    """
    out = {}
    for n in names:
        da = np.load(add / f"preds_{tag}" / f"{n}.npz")
        db = np.load(base / f"preds_{tag}" / f"{n}.npz")
        out[n] = (da["coords"].astype(float), da["node_probs"].astype(float),
                  db["coords"].astype(float), db["node_probs"].astype(float))
    return out


def stratified(names: list[str], n: int) -> list[str]:
    """An even slice of each embryo, so a sweep subset spans both."""
    by_emb = defaultdict(list)
    for x in names:
        by_emb[x.split("_")[0]].append(x)
    picks = []
    for v in by_emb.values():
        step = max(1, len(v) // max(1, n // len(by_emb)))
        picks += v[::step][: n // len(by_emb)]
    return sorted(picks)


HDR = (f"{'config':<36}{'score':>9}{'edge_jac':>11}{'adj_jac':>11}"
       f"{'node_rec':>11}{'n_pred':>10}{'ratio':>9}")


def show(label: str, m: dict, ref: float | None = None) -> None:
    delta = f"   {m['score'] - ref:+.4f}" if ref is not None else ""
    print(f"{label:<36}{m['score']:>9.4f}{m['edge_jaccard']:>11.4f}"
          f"{m['adj_edge_jaccard']:>11.4f}{m['node_recall']:>11.4f}"
          f"{m['n_pred']:>10.0f}{m['ratio']:>9.3f}{delta}")


def main() -> None:
    global RUN_A, RUN_B
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="log folder of the base detector")
    p.add_argument("--add", required=True, help="log folder whose extras get appended")
    p.add_argument("--tag", default="r6_0.30",
                   help="preds_<tag> folder to read; use the lowest threshold "
                        "available, higher ones are reached by filtering")
    p.add_argument("--cfg", nargs=3, type=float, metavar=("TH_BASE", "TH_ADD", "D_ADD"),
                   help="score this one configuration instead of sweeping")
    p.add_argument("--sweep", action="store_true", help="grid search instead")
    p.add_argument("--n", type=int, default=24, help="samples used by --sweep")
    p.add_argument("--all", action="store_true",
                   help="score on every sample, and split swept vs unseen")
    p.add_argument("--solo", nargs=2, type=float, default=(0.40, 0.45),
                   metavar=("TH_BASE", "TH_ADD"),
                   help="thresholds for the two single-model baselines")
    p.add_argument("--max-disp", type=float, default=10.0,
                   help="um; NN links longer than this are dropped")
    p.add_argument("--mutual", action="store_true",
                   help="require the NN link to be reciprocal")
    p.add_argument("--jobs", type=int, default=32)
    p.add_argument("--consensus", nargs=2, type=float, metavar=("RADIUS", "TH"),
                   help="use the consensus fusion instead of the append recipe: "
                        "cluster both models within RADIUS um, score each cluster "
                        "by the zero-filled mean of the two probabilities, keep "
                        ">= TH")
    p.add_argument("--cluster-th", type=float, default=0.30,
                   help="per-model cut applied before clustering (consensus only)")
    p.add_argument("--no-score", action="store_true",
                   help="skip the metric entirely and just write the blend")
    p.add_argument("--save", action="store_true",
                   help="write the merged nodes, ready for "
                        "`main_edge.py --nodes_folder <dir>`")
    p.add_argument("--out-root", default="ensemble",
                   help="where blends live; the folder is named "
                        "<version>_<base_exp>_<add_exp>")
    p.add_argument("--version", default="v1")
    p.add_argument("--out", help="explicit output directory, overriding the "
                                 "--out-root/--version naming")
    args = p.parse_args()

    RUN_A = Path(args.add) if Path(args.add).is_absolute() else REPO / args.add
    RUN_B = Path(args.base) if Path(args.base).is_absolute() else REPO / args.base

    names_all = sorted(q.stem for q in (RUN_B / f"preds_{args.tag}").glob("*.npz"))
    if not names_all:
        raise SystemExit(f"no predictions under {RUN_B}/preds_{args.tag}")
    swept = stratified(names_all, args.n)
    data = ({} if args.no_score else
            load(RUN_B, RUN_A, args.tag, names_all if args.all else swept))

    if args.sweep:
        print(f"sweep on {len(swept)} samples, tag {args.tag}\n")
        print("SINGLE MODELS under the same NN linking")
        print(HDR); print("-" * 97)
        solo = {}
        for th in (0.30, 0.40, 0.45, 0.50, 0.55, 0.60):
            solo[("base", th)] = run_cfg(swept, data, 1.01, th, 0.0,
                                         args.max_disp, args.mutual, args.jobs)[0]
            show(f"base alone @ {th:.2f}", solo[("base", th)])
        print()
        for th in (0.30, 0.40, 0.45, 0.50, 0.55, 0.60):
            solo[("add", th)] = run_cfg(swept, data, th, 1.01, 0.0,
                                        args.max_disp, args.mutual, args.jobs)[0]
            show(f"add alone @ {th:.2f}", solo[("add", th)])
        best_solo = max(solo.values(), key=lambda m: m["score"])["score"]

        print("\n\nENSEMBLE")
        print(HDR); print("-" * 97)
        res = {}
        for th_base, th_add, d_add in itertools.product(
            (0.40, 0.45, 0.50, 0.55), (0.35, 0.40, 0.45, 0.50, 0.55),
            (6.0, 7.0, 8.0, 10.0, 12.0),
        ):
            res[(th_base, th_add, d_add)] = run_cfg(
                swept, data, th_add, th_base, d_add,
                args.max_disp, args.mutual, args.jobs)[0]
        for k in sorted(res, key=lambda k: -res[k]["score"])[:14]:
            show(f"base={k[0]:.2f} add={k[1]:.2f} d={k[2]:g}", res[k], best_solo)
        best = max(res, key=lambda k: res[k]["score"])
        print(f"\n-> best {res[best]['score']:.4f} at th_base={best[0]} "
              f"th_add={best[1]} d_add={best[2]:g}  "
              f"(best single {best_solo:.4f}, {res[best]['score'] - best_solo:+.4f})")
        return

    if args.consensus:
        radius, th = args.consensus
        if args.out:
            out_dir = Path(args.out)
            out_dir = out_dir if out_dir.is_absolute() else REPO / out_dir
        else:
            def exp(r: Path) -> str:
                return "_".join(r.relative_to(REPO / "logs").parts)
            out_dir = (REPO / args.out_root /
                       f"{args.version}_{exp(RUN_B)}_{exp(RUN_A)}"
                       f"_consensus_r{radius:g}_th{th:.2f}")
        stats = save_consensus(out_dir, RUN_B, RUN_A, args.tag, names_all,
                               radius, th, args.cluster_th, n_jobs=args.jobs)
        settings = {
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "command": "python " + " ".join(sys.argv),
            "recipe": ("consensus fusion: cluster both models' detections within "
                       "radius_um (greedy, strongest-first); node = "
                       "confidence-weighted centroid; score = mean of the two "
                       "models' probabilities with 0 filled in for a model that "
                       "did not fire; keep score >= th"),
            "model_a": str(RUN_B.relative_to(REPO)),
            "model_b": str(RUN_A.relative_to(REPO)),
            "source_preds": f"preds_{args.tag}",
            "nms_radius_um": _radius_of(args.tag),
            "radius_um": radius, "th": th, "cluster_th": args.cluster_th,
            "voxel_scale_um": list(SCALE),
            "counts": {k: v for k, v in stats.items() if k != "per_sample"},
            "consumed_by": ("python main_edge.py --nodes_folder "
                            f"../{out_dir.relative_to(REPO)}/"),
            "measured_199_oof": _measured_for(RUN_B, RUN_A, radius, th),
            "notes": [
                "node_probs is the FUSED consensus score, not either model's raw "
                "probability. 0.5 here means 'both models agreed at 0.5', not "
                "'one model said 0.5' -- do not compare it to a single-model file "
                "or re-threshold it as if it were one.",
                "coords/coords_f are confidence-weighted centroids of the two "
                "models' positions, so they match neither source exactly.",
                "annot_probs is carried from each cluster's strongest member; "
                "averaging it would not correspond to any real detection.",
                "edges is an empty (0, 4) array; main_edge.py recomputes them.",
                "no embeddings_* column: 2026-09-24/5's preds folder has none. "
                "Run inference_contrastive.py against this folder before "
                "main_edge.py.",
            ],
            "per_sample": stats["per_sample"],
        }
        (out_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        print(f"-> wrote {stats['n_samples']} npz + settings.json to {out_dir}")
        print(f"   {stats['n_nodes']} nodes, {stats['frac_found_by_both']:.1%} "
              f"found by both models")
        return

    if not args.cfg:
        raise SystemExit("pass --cfg TH_BASE TH_ADD D_ADD, or --sweep")

    th_base, th_add, d_add = args.cfg
    rows = [
        (f"base alone @ {args.solo[0]:.2f}", 1.01, args.solo[0], 0.0),
        (f"add alone @ {args.solo[1]:.2f}", args.solo[1], 1.01, 0.0),
        (f"ensemble {th_base:.2f}/{th_add:.2f} d={d_add:g}", th_add, th_base, d_add),
    ]
    splits = [("ALL (%d)" % len(names_all), names_all)] if args.all else [
        ("SWEPT (%d)" % len(swept), swept)]
    if args.all:
        unseen = [n for n in names_all if n not in set(swept)]
        splits = [("SWEPT (%d)" % len(swept), swept),
                  ("UNSEEN (%d)" % len(unseen), unseen)] + splits

    scores = {}
    for label, split in ([] if args.no_score else splits):
        print(f"\n{label}")
        print(HDR); print("-" * 97)
        ref = None
        for name, ta, tb, da_ in rows:
            m = run_cfg(split, data, ta, tb, da_, args.max_disp,
                        args.mutual, args.jobs)[0]
            if ref is None:
                ref = m["score"]
            show(name, m, ref)
            scores.setdefault(label, {})[name] = {
                k: (None if v != v else round(float(v), 6)) for k, v in m.items()}

    if not (args.save or args.out):
        return

    if args.out:
        out_dir = Path(args.out)
        out_dir = out_dir if out_dir.is_absolute() else REPO / out_dir
    else:
        # v1_<base experiment>_<add experiment>, experiments named by their log
        # folder: logs/2026-09-13/1 -> 2026-09-13_1.
        def exp(run: Path) -> str:
            return "_".join(run.relative_to(REPO / "logs").parts)
        out_dir = (REPO / args.out_root /
                   f"{args.version}_{exp(RUN_B)}_{exp(RUN_A)}")
    # Save on every sample the source runs have, not just whatever split was
    # scored: a saved artefact that silently held a subset would be a trap.
    stats = save_merged(out_dir, RUN_B, RUN_A, args.tag, names_all,
                        th_base, th_add, d_add, " ".join(sys.argv))
    settings = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": "python " + " ".join(sys.argv),
        "recipe": ("nodes = BASE(node_probs >= th_base) + "
                   "ADD(node_probs >= th_add) further than d_add um from every "
                   "kept BASE node, per frame"),
        "base_run": str(RUN_B.relative_to(REPO)),
        "add_run": str(RUN_A.relative_to(REPO)),
        "source_preds": f"preds_{args.tag}",
        "nms_radius_um": _radius_of(args.tag),
        "th_base": th_base, "th_add": th_add, "d_add_um": d_add,
        "voxel_scale_um": list(SCALE),
        "counts": {k: v for k, v in stats.items() if k != "per_sample"},
        "consumed_by": ("python main_edge.py --nodes_folder "
                        f"../{out_dir.relative_to(REPO)}/"),
        "scores": scores,
        "scoring": {
            "metric": "repo evaluate/per_sample_metrics/summarise, macro over "
                      "embryo groups (same as main_predict cv_score)",
            "edges": f"nearest-neighbour linking, max_disp {args.max_disp} um, "
                     f"mutual={args.mutual} -- used ONLY for scoring",
        },
        "notes": [
            "edges in the npz files are an empty (0, 4) array: the key is kept "
            "so the layout matches preds_<tag>, but the merged node set has no "
            "edge-head output for the appended nodes.",
            "node_probs are each source model's raw values and are NOT on a "
            "common scale -- the two runs' probability distributions differ by "
            "~0.067 at the median, so do not re-threshold this file globally.",
        ],
        "per_sample": stats["per_sample"],
    }
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"\n-> wrote {stats['n_samples']} npz + settings.json to {out_dir}")
    print(f"   {stats['n_nodes']} nodes: {stats['n_from_base']} from base, "
          f"{stats['n_from_add']} from add ({stats['frac_from_add']:.1%})")


if __name__ == "__main__":
    sys.exit(main())
