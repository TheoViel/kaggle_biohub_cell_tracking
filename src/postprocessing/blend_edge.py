"""Blend several edge runs into one candidate graph, on a shared node set.

    cd src/postprocessing
    python blend_edge.py --runs ../logs/edge_run/2026-09-13/0 \
                                ../logs/edge_run/2026-09-14/1 \
                                ../logs/edge_run/2026-09-15/0 \
                         --embryo 44b6 --out ../logs/edge_run/blend/0/preds_0.35

Paths are relative to ``src/``, not to this folder: ``common`` chdirs there so
``params``' own relative paths resolve, exactly as the other scripts here do.

Each run's npz carries `candidates` -- every scored pair with its probability,
before any threshold (`main_edge_predict._candidate_array`). That array, not the
committed `edges`, is the thing worth blending: the threshold and the assignment
are then re-decided once, over the fused scores, instead of once per run and
then voted on.

The obstacle is that the runs do not share a node set. Each was trained and
predicted over its own `nodes_folder`, so a candidate is a pair of *row indices*
into an array that differs run to run. They are close but not equal: across the
runs in `logs/edge_run/2026-09-13..15`, 66-80% of nodes coincide on the exact
voxel and 97-99% fall within 2 um of a node in another run. So one run is taken
as the reference, every other run's rows are matched into it by nearest
neighbour within `--tol_um`, and candidates whose endpoints do not both match
are dropped -- that run simply abstains on those pairs.

Matching is safe at this tolerance because the detector's NMS already separates
any two nodes of one set by its radius (6-7 um), which is several times the
tolerance: the nearest node in the reference is either the same cell or nothing.

Keeping the *reference's* node set rather than their union is a metric decision,
not a convenience. The score is
`adj_edge_jaccard = edge_jaccard * (1 - 0.1 * (N_pred - N_total) / N_total)`,
so nodes are charged for whether or not they carry an edge, and the union of
six node sets is roughly six times N_total. Ultrack can union its foreground
maps because its ILP then *selects* a non-overlapping subset and the cells it
emits are one per cell; nothing downstream here does that, so the union would be
paid for in full. `multi_hypothesis.py` is the arm that builds the selector.

The fusion rules differ in how much they let the other runs move the reference:

  mean_logit  the unweighted mean of log-odds over the runs that saw the pair.
  mean_prob   the unweighted mean of probabilities.
  median      the median, which ignores one run's outlier.
  max         the most optimistic run wins; recall-biased.
  anchored    `(1-w)` of the reference's log-odds plus `w` of the others' mean,
              falling back to the plain mean where the reference abstains. This
              is the rule to reach for when the runs are not equally trusted,
              and at `w = 0` it is the reference alone.

A pair one run never proposed is *not* scored as zero. The candidate sets are
built by kNN inside a radius from each run's own nodes, so a pair missing from
run B is usually a pair B's node set could not express, not one B rejected.
Filling those with zero is what makes a blend lose: measured on fold 0 it costs
0.03-0.05 adjusted Jaccard, far more than blending gains.

What blending is worth here
---------------------------
Nothing measurable, and the reason is worth recording before anyone spends a
week on it. Measured over all 71 fold-0 stacks (`44b6`), six runs from
`logs/edge_run/2026-09-13..15`, 2 um matching. Raw, before post-processing:

    best single run (2026-09-13/0)              adjJ 0.8516
    best blend (anchored w=0.25, thr 0.35)      adjJ 0.8525   (+0.0009)
    union of the six runs' committed edges      adjJ 0.8431   (-0.0085)
    majority vote over committed edges          adjJ 0.8381   (-0.0135)

After `main_edge_postprocess.py`, which is where the score actually lives, the
sign flips and the magnitude does not:

    2026-09-13/0  preds_0.45_pp                 adjJ 0.8807
    blend + pp, threshold swept 0.30 .. 0.50    adjJ 0.8797 / 0.8793 / 0.8802
                                                     / 0.8803 / 0.8738

The blend's best point is 0.8803 against the reference's 0.8807. Flat across the
whole threshold curve, so it is not a mistuned cutoff.

The ceiling explains why. Keeping only the true positives out of a candidate set
-- a perfect selector, no false positives, the best any re-linking of that set
could do:

    reference run's candidates alone            adjJ 0.9649
    all six runs' candidates pooled             adjJ 0.9657   (+0.0008)

The reference run's own candidate graph already holds 19411 of the 19826
ground-truth edges. Pooling five more runs finds 17 more. Coverage is not what
is missing: every run proposes essentially the same pairs, because the pairs are
kNN inside a radius over node sets that agree to within 2 um. The gap from 0.965
to 0.881 is selection among candidates one run already has, which is why the
post-processing chain is worth +0.03 and blending is worth nothing.

Read those deltas against the noise. Paired over the 71 stacks, the standard
error of a difference is 0.0011 between a blend and its own reference (the arms
are nearly the same graph) and 0.003-0.004 between two independent runs. So
~0.002 is the smallest credible difference for the first kind and ~0.007 for the
second -- which also means the run ranking in `Experiments.md` is, across the
last three days, inside fold 0's noise: 2026-09-13/0 beats 2026-09-14/1 by
0.0004 (t = 0.33) and 2026-09-13/3 by 0.0032 (t = 0.95). Anything this size
needs both embryos (199 stacks), not fold 0.

Where the headroom actually is, on the same 71 stacks and the same runs:

  * selection *within* one run's candidates -- oracle 0.9649 against 0.8807
    delivered. This is the 0.08, and `edge_pp`'s assign/recover/ILP stages are
    what attack it.
  * choosing *which run* to trust per stack -- an oracle that picks the best
    run's post-processed graph for each stack scores 0.9007 against 0.8815 for
    the best fixed run, so +0.019 is on the table. Unsupervised proxies get a
    twentieth of it: committed edges per node +0.0012, mean edge probability
    +0.0006, agreement with the other runs +0.0000 (it just finds the best run).
    All three are inside the noise floor above, but the oracle is not, and a
    selector is a different and more promising object than an average.
  * pooling *node* hypotheses, which is Ultrack's actual claim -- see
    `multi_hypothesis.py`, where the ILP that makes a union affordable lives.

So this script exists to make the first question answerable and to take the free
fraction of a point if a later run set ever has more diversity than this one --
not because it is the lever. `--oracle` prints the ceilings above.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: F401,E402  (chdir to src/ and set the path)

import pandas as pd  # noqa: E402
import tracksdata as td  # noqa: E402

from params import SCALE, TRAIN_DIR  # noqa: E402
from postprocessing.fast_score import (  # noqa: E402
    GroundTruth,
    match_nodes,
    roll_up,
    row_from_counts,
)

warnings.filterwarnings("ignore")

SCALE_ARR = np.asarray(SCALE, dtype=np.float64)
EPS = 1e-6
FUSIONS = ("mean_logit", "mean_prob", "median", "max", "anchored")


# --------------------------------------------------------------------------
# node correspondence
# --------------------------------------------------------------------------

def node_map(src: np.ndarray, ref: np.ndarray, tol_um: float) -> np.ndarray:
    """``row in src -> row in ref``, -1 where nothing is within ``tol_um``.

    Per frame and nearest-neighbour rather than a global assignment: NMS keeps
    any two nodes of one set at least its radius apart, several times this
    tolerance, so the nearest reference node is either this cell or no cell and
    a solver would have nothing left to decide.
    """
    out = np.full(len(src), -1, np.int64)
    order = np.argsort(ref[:, 0], kind="stable")
    frames = ref[order, 0]
    for t in np.unique(src[:, 0]):
        lo, hi = np.searchsorted(frames, [t, t + 1])
        ri = order[lo:hi]
        if not len(ri):
            continue
        si = np.flatnonzero(src[:, 0] == t)
        d, j = cKDTree(ref[ri, 1:] * SCALE_ARR).query(src[si, 1:] * SCALE_ARR, k=1)
        ok = d <= tol_um
        out[si[ok]] = ri[j[ok]]
    return out


def candidates_in_ref(coords, cands, ref, tol_um, is_ref):
    """A run's candidates as ``(pairs (K, 2) in reference rows, prob (K,), dist (K,))``.

    Pairs with an unmatched endpoint are dropped, which is this run abstaining
    on them -- see the module docstring on why that is not a zero.
    """
    cands = np.asarray(cands, np.float64).reshape(-1, 5)
    a, b = cands[:, 0].astype(np.int64), cands[:, 1].astype(np.int64)
    p, d = cands[:, 2], cands[:, 3]
    if not is_ref:
        m = node_map(coords, ref, tol_um)
        a, b = m[a], m[b]
        ok = (a >= 0) & (b >= 0)
        a, b, p, d = a[ok], b[ok], p[ok], d[ok]
    return np.stack([a, b], 1), np.clip(p, EPS, 1 - EPS), d


def edges_in_ref(coords, edges, ref, tol_um, is_ref) -> np.ndarray:
    """A run's *committed* edges as ``(M, 2)`` reference rows, for the vote arms."""
    e = np.asarray(edges, np.float64).reshape(-1, 4)
    a, b = e[:, 0].astype(np.int64), e[:, 1].astype(np.int64)
    if not is_ref:
        m = node_map(coords, ref, tol_um)
        a, b = m[a], m[b]
        ok = (a >= 0) & (b >= 0)
        a, b = a[ok], b[ok]
    return np.stack([a, b], 1)


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------

def _pack(pairs: np.ndarray) -> np.ndarray:
    return pairs[:, 0].astype(np.int64) * (1 << 32) + pairs[:, 1].astype(np.int64)


def _unpack(keys: np.ndarray) -> np.ndarray:
    return np.stack([(keys >> 32).astype(np.int64),
                     (keys & 0xFFFFFFFF).astype(np.int64)], 1)


def fuse(per_run, mode: str = "mean_logit", weight: float = 0.25):
    """Several runs' reference-space candidates into one ``(pairs, prob, dist)``.

    ``per_run`` is a list of ``(pairs, prob, dist)``, the first being the
    reference. Runs that never proposed a pair are absent from its average, not
    counted as a zero.
    """
    packed = [_pack(pr) for pr, _, _ in per_run]
    keys = np.unique(np.concatenate(packed)) if packed else np.zeros(0, np.int64)
    n_runs = len(per_run)
    prob = np.full((len(keys), n_runs), np.nan)
    dist = np.full(len(keys), np.nan)
    for j, (pk, (_, p, d)) in enumerate(zip(packed, per_run)):
        idx = np.searchsorted(keys, pk)
        prob[idx, j] = p
        dist[idx] = np.where(np.isnan(dist[idx]), d, dist[idx])

    seen = ~np.isnan(prob)
    logit = np.log(prob / (1 - prob))
    if mode == "mean_prob":
        fused = np.nanmean(prob, 1)
    elif mode == "median":
        fused = np.nanmedian(prob, 1)
    elif mode == "max":
        fused = np.nanmax(prob, 1)
    elif mode == "mean_logit":
        fused = 1 / (1 + np.exp(-np.nanmean(logit, 1)))
    elif mode == "anchored":
        others = (np.nanmean(np.where(seen[:, 1:], logit[:, 1:], np.nan), 1)
                  if n_runs > 1 else np.full(len(keys), np.nan))
        mix = np.where(np.isnan(others), logit[:, 0],
                       (1 - weight) * logit[:, 0] + weight * others)
        # Where the reference abstains there is no anchor; the crowd is all
        # there is, so fall back to their plain mean rather than dropping the
        # pair -- it is a pair the reference's *node set* could not express.
        mix = np.where(seen[:, 0], mix, np.nanmean(logit, 1))
        fused = 1 / (1 + np.exp(-mix))
    else:
        raise ValueError(f"unknown --fuse {mode!r}; one of {FUSIONS}")

    ok = np.isfinite(fused)
    return _unpack(keys[ok]), fused[ok], np.nan_to_num(dist[ok]), seen.sum(1)[ok]


# --------------------------------------------------------------------------
# linking
# --------------------------------------------------------------------------

def link(coords: np.ndarray, pairs: np.ndarray, prob: np.ndarray,
         dist: np.ndarray, threshold: float):
    """``link_video``'s rule -- Hungarian per consecutive frame pair.

    The cost is ``log(threshold) - log(p)``, negative exactly where the model
    beats the threshold, and not linking is free. Kept identical to
    ``edge_transformer.predict.link_video`` so a blended folder and a single
    run's folder are the same object scored the same way.
    """
    if not len(pairs):
        return np.zeros((0, 4), np.float64)
    frame = coords[:, 0].astype(int)
    ta, tb = frame[pairs[:, 0]], frame[pairs[:, 1]]
    adj = tb == ta + 1
    pairs, prob, dist, ta = pairs[adj], prob[adj], dist[adj], ta[adj]
    log_th = float(np.log(min(max(threshold, 1e-6), 1 - 1e-6)))
    cost_all = log_th - np.log(np.clip(prob, 1e-9, 1 - 1e-9))

    out = []
    for t in np.unique(ta):
        sel = ta == t
        par, ri = np.unique(pairs[sel, 0], return_inverse=True)
        ch, ci = np.unique(pairs[sel, 1], return_inverse=True)
        cost = np.zeros((len(par), len(ch)), np.float32)
        pr = np.zeros_like(cost)
        di = np.zeros_like(cost)
        cost[ri, ci] = cost_all[sel]
        pr[ri, ci] = prob[sel]
        di[ri, ci] = dist[sel]
        r, c = linear_sum_assignment(cost)
        keep = cost[r, c] < 0          # a pair exactly at the threshold is free
        r, c = r[keep], c[keep]
        out.append(np.stack([par[r], ch[c], pr[r, c], di[r, c]], 1))
    return np.concatenate(out).astype(np.float64) if out else np.zeros((0, 4))


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

class Scorer:
    """Edge TP/FP for one node set, vectorised over edge arrays.

    ``fast_score`` counts with a Python loop per edge set; a sweep here scores
    a dozen arms per stack, so the node matching is done once and the counting
    is a lookup.
    """

    def __init__(self, coords, gt: GroundTruth, n_total: float):
        self.gt, self.n_total, self.n_nodes = gt, n_total, len(coords)
        m = match_nodes(coords, gt)
        self.g = np.full(len(coords), -1, np.int64)
        for row, node in m.items():
            self.g[row] = node
        self.has_child = np.isin(self.g, list(gt.has_child))
        self.has_parent = np.isin(self.g, list(gt.has_parent))
        self.node_recall = (len(set(m.values())) / gt.n_nodes
                            if gt.n_nodes else float("nan"))

    def flags(self, e):
        """``(valid, is_tp)`` -- the metric's own rule, see ``fast_score``."""
        e = np.asarray(e).reshape(-1, e.shape[1] if np.ndim(e) > 1 else 1)
        if not len(e):
            return np.zeros(0, bool), np.zeros(0, bool)
        a, b = e[:, 0].astype(int), e[:, 1].astype(int)
        valid = self.has_child[a] | self.has_parent[b]
        ga, gb = self.g[a], self.g[b]
        tp = np.fromiter(((int(x), int(y)) in self.gt.pairs for x, y in zip(ga, gb)),
                         bool, len(ga))
        return valid, tp & valid

    def row(self, e, n_nodes=None):
        valid, tp_mask = self.flags(e)
        tp = int(tp_mask.sum())
        return row_from_counts(tp, int(valid.sum()) - tp, self.gt.n_edges - tp,
                               n_nodes or self.n_nodes, self.n_total, self.node_recall)


def load_gt(sample: str, meta: pd.DataFrame):
    g = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{sample}.geff")
    g = g[0] if isinstance(g, tuple) else g
    n_total = float(meta.loc[meta.sample_name == sample,
                             "estimated_number_of_nodes"].iloc[0])
    return GroundTruth(g), n_total


# --------------------------------------------------------------------------
# per-stack work
# --------------------------------------------------------------------------

def blend_sample(job: tuple[str, dict]):
    """One stack: fuse, link, optionally write, and score the arms asked for."""
    sample, cfg = job
    have = [d for d in (Path(x) for x in cfg["preds"])
            if (d / f"{sample}.npz").exists()]
    if not have:
        return sample, {}
    data = [np.load(d / f"{sample}.npz") for d in have]
    ref_coords = data[0]["coords"].astype(np.int64)

    per_run = [candidates_in_ref(d["coords"].astype(np.int64), d["candidates"],
                                 ref_coords, cfg["tol_um"], i == 0)
               for i, d in enumerate(data)]
    pairs, prob, dist, _ = fuse(per_run, cfg["fuse"], cfg["weight"])
    edges = link(ref_coords, pairs, prob, dist, cfg["threshold"])

    if cfg["out"]:
        out = Path(cfg["out"])
        out.mkdir(parents=True, exist_ok=True)
        frame = ref_coords[:, 0].astype(np.int64)
        cand = np.stack([pairs[:, 0], pairs[:, 1], prob, dist,
                         frame[pairs[:, 1]] - frame[pairs[:, 0]]], 1)
        np.savez_compressed(
            out / f"{sample}.npz",
            coords=ref_coords.astype(np.int16),
            edges=edges.reshape(-1, 4),
            candidates=cand.astype(np.float64),
        )

    if cfg["no_score"]:
        return sample, {}

    gt, n_total = load_gt(sample, pd.read_csv("../input/metadata.csv"))
    s = Scorer(ref_coords, gt, n_total)
    rows = {f"blend {cfg['fuse']}": s.row(edges[:, :2])}

    if cfg["arms"]:
        for name, mode in (("mean_logit", "mean_logit"), ("mean_prob", "mean_prob"),
                           ("median", "median"), ("max", "max")):
            p, q, dd, _ = fuse(per_run, mode, cfg["weight"])
            rows[f"blend {name}"] = s.row(link(ref_coords, p, q, dd, cfg["threshold"])[:, :2])
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            p, q, dd, _ = fuse(per_run, "anchored", w)
            rows[f"blend anchored w={w}"] = s.row(
                link(ref_coords, p, q, dd, cfg["threshold"])[:, :2])
        # single runs, each on its own node set -- the number to beat
        for d, path in zip(data, have):
            c = d["coords"].astype(np.int64)
            tag = "/".join(path.parts[-3:-1])
            rows[f"single {tag}"] = Scorer(c, gt, n_total).row(
                np.asarray(d["edges"]).reshape(-1, 4)[:, :2], len(c))
        # committed-edge votes, for the record
        ref_e = [edges_in_ref(d["coords"].astype(np.int64), d["edges"],
                              ref_coords, cfg["tol_um"], i == 0)
                 for i, d in enumerate(data)]
        sets = [set(map(tuple, e)) for e in ref_e]
        cnt = Counter()
        for st in sets:
            cnt.update(st)
        for name, keep in (("union", set().union(*sets)),
                           ("majority", {k for k, v in cnt.items() if v > len(sets) / 2})):
            rows[f"vote {name}"] = s.row(np.asarray(sorted(keep), np.int64).reshape(-1, 2))

    if cfg["oracle"]:
        # A perfect selector over a candidate set: every true positive, no false
        # positive. The ceiling of any re-linking of that set, and the number
        # that says whether pooling runs adds coverage at all.
        _, tp = s.flags(pairs)
        rows["ORACLE pooled candidates"] = s.row(pairs[tp])
        p0, _, _ = per_run[0]
        _, tp0 = s.flags(p0)
        rows["ORACLE ref candidates"] = s.row(p0[tp0])

    return sample, rows


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=None,
                   help="edge_run folders; their --preds_name subfolder is used")
    p.add_argument("--preds", nargs="+", default=None,
                   help="explicit preds_* folders, instead of --runs")
    p.add_argument("--preds_name", default="preds_0.45",
                   help="which preds_* folder of each --runs entry to read")
    p.add_argument("--embryo", default="44b6",
                   help="fold 0 is 44b6 (71 stacks); a preds folder holds both "
                        "embryos, so name the one you mean")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--fuse", default="anchored", choices=FUSIONS)
    p.add_argument("--weight", type=float, default=0.25,
                   help="--fuse anchored: how far the others may move the reference")
    p.add_argument("--threshold", type=float, default=0.35,
                   help="link threshold for the blended graph. Lower than a "
                        "single run's 0.45: averaging shrinks probabilities "
                        "toward the middle, so the same cutoff is stricter")
    p.add_argument("--tol_um", type=float, default=2.0,
                   help="node matching tolerance; well under the detector's "
                        "6-7 um NMS radius, so a match is unambiguous")
    p.add_argument("--out", default=None,
                   help="write blended npz here, in main_edge_predict's format, "
                        "so main_edge_postprocess.py runs over it unchanged")
    p.add_argument("--arms", action="store_true", help="score every fusion rule")
    p.add_argument("--oracle", action="store_true",
                   help="print the perfect-selector ceiling of the candidate sets")
    p.add_argument("--no_score", action="store_true")
    p.add_argument("--jobs", type=int, default=16)
    a = p.parse_args()

    if a.preds:
        preds = [Path(x) for x in a.preds]
    elif a.runs:
        preds = [Path(x) / a.preds_name for x in a.runs]
    else:
        p.error("one of --runs or --preds is required")
    for d in preds:
        if not d.is_dir():
            p.error(f"not a folder: {d}")

    samples = sorted(f.stem for f in preds[0].glob("*.npz"))
    if a.embryo:
        samples = [s for s in samples if s.split("_")[0] == a.embryo]
    if a.limit:
        samples = samples[: a.limit]

    print(f"-> {len(samples)} stacks, {len(preds)} runs, reference = {preds[0]}")
    for d in preds[1:]:
        print(f"   + {d}")
    print(f"   fuse={a.fuse} weight={a.weight} threshold={a.threshold} tol={a.tol_um}um")

    cfg = dict(preds=[str(d) for d in preds], fuse=a.fuse, weight=a.weight,
               threshold=a.threshold, tol_um=a.tol_um, out=a.out, arms=a.arms,
               oracle=a.oracle, no_score=a.no_score)
    with Pool(a.jobs) as pool:
        results = pool.map(blend_sample, [(s, cfg) for s in samples])

    if a.out:
        print(f"-> npz: {a.out}")
    if a.no_score:
        return

    acc: dict[str, list] = {}
    for _, rows in results:
        for k, v in rows.items():
            acc.setdefault(k, []).append(v)
    ranked = sorted(((roll_up(v)["adj_edge_jaccard"], k, roll_up(v))
                     for k, v in acc.items()), reverse=True)
    best_single = max((s for s, k, _ in ranked if k.startswith("single")), default=None)

    print(f"\n{'arm':<32}{'tp':>7}{'fp':>7}{'fn':>7}{'J':>9}{'adjJ':>9}{'delta':>9}")
    for adj, k, ru in ranked:
        d = f"{adj - best_single:+.4f}" if best_single is not None else ""
        print(f"{k:<32}{ru['edge_tp']:>7}{ru['edge_fp']:>7}{ru['edge_fn']:>7}"
              f"{ru['edge_jaccard']:>9.4f}{adj:>9.4f}{d:>9}")
    if best_single is not None:
        print("\ndelta is against the best single run on its own node set.")


if __name__ == "__main__":
    main()
