"""Ultrack's first idea, for a point detector: never commit to one node set.

    cd src/postprocessing
    python multi_hypothesis.py --sets ../logs/2026-09-14/4/preds_r6_0.40 \
                                      ../logs/2026-09-14/7/preds_r7_0.40 --limit 5

`main_predict.py` writes eighteen node sets per run -- radius x threshold x
head -- and the sweep keeps one and deletes the other seventeen. Ultrack
(Bragantini et al., Nat. Methods 2025) argues that choice is the mistake: hold
every candidate as a *hypothesis*, make mutually incompatible ones exclusive,
and let temporal consistency decide which survives. Their ablation is the
claim's strongest form -- combining four gamma settings beat the best single
setting on the metric that setting was chosen for (TRA 0.924 against 0.860,
false negatives 69 against 145).

Their hypotheses are nested segments from an ultrametric contour map, and the
exclusion rule is containment: a segment and the segment it sits inside cannot
both be chosen. There are no contours here, only cell centres, so the rule has
to be restated. Two candidates in the same frame closer than `--conflict_um`
are taken to be two explanations of one cell and made mutually exclusive.
That threshold has a principled value rather than a tuned one: NMS already
guarantees no single node set contains two detections within its radius, so at
`conflict_um` equal to the smallest radius pooled, a surviving pair is one that
*no* contributing hypothesis would have allowed on its own.

Everything else is Ultrack's ILP as `tracksdata` already implements it --
`node_weight` on selection, costs on appearance, disappearance and division,
flow conservation, and `bulk_add_overlaps` for the exclusion constraints
(their equation 7). What this file adds is the pooling and the conflict graph.

Scoring is the competition metric, so an arm here is directly comparable to a
`scores_*.json` written by `main_predict.py` or `main_postprocess.py`.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import warnings
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tracksdata as td
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: F401,E402  (chdir to src/ and set the path)

from params import EMBRYOS, SCALE, TRAIN_DIR  # noqa: E402
from tracking.post_process import build_graph  # noqa: E402
from tracking_cellmot.metrics import (  # noqa: E402
    evaluate,
    macro_summarise,
    node_recall,
    per_sample_metrics,
    summarise,
)

warnings.filterwarnings("ignore")
SCALE_ARR = np.array(SCALE, dtype=np.float64)


# --------------------------------------------------------------------------
# pooling
# --------------------------------------------------------------------------

def pool_hypotheses(paths: list[Path], sample: str):
    """Every candidate node and link from several node sets, merged.

    Returns ``(coords, prob, votes, edges)``:

    coords : (N, 4) int   the union of the sets' detections
    prob   : (N,)         the detector's probability, max over the sets
    votes  : (N,)         how many sets proposed this exact voxel
    edges  : (M, 3)       (src, tgt, prob), max over the sets

    Identical coordinates across sets are the *same* hypothesis, not competing
    ones, so they merge rather than conflict -- and how many sets found it is
    itself evidence, which is what `votes` carries. Competing hypotheses are
    the near-but-not-equal ones, and those are handled by the conflict graph.
    """
    index: dict[tuple, int] = {}
    coords: list[tuple] = []
    prob: list[float] = []
    votes: list[int] = []
    emb: list[np.ndarray] = []
    edge_prob: dict[tuple[int, int], float] = {}

    for path in paths:
        f = path / f"{sample}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        c = d["coords"].astype(np.int64)
        p = d["node_probs"].astype(np.float64) if "node_probs" in d.files else np.ones(len(c))
        # NaN is "this set carries no probability"; treat it as neutral rather
        # than letting it poison the max.
        p = np.nan_to_num(p, nan=0.5)
        ekey = next((k for k in d.files if k.startswith("embeddings")), None)
        e = d[ekey].astype(np.float32) if ekey else None

        local = np.empty(len(c), dtype=np.int64)
        for i, row in enumerate(map(tuple, c)):
            j = index.get(row)
            if j is None:
                j = len(coords)
                index[row] = j
                coords.append(row)
                prob.append(p[i])
                votes.append(0)
                emb.append(None if e is None else e[i])
            else:
                prob[j] = max(prob[j], p[i])
            votes[j] += 1
            local[i] = j

        for src, tgt, pr, _ in np.asarray(d["edges"], float).reshape(-1, 4):
            a, b = local[int(src)], local[int(tgt)]
            if a == b:
                continue
            key = (int(a), int(b))
            if pr > edge_prob.get(key, -1.0):
                edge_prob[key] = float(pr)

    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 4)
    edges = np.array([[a, b, p] for (a, b), p in edge_prob.items()], dtype=np.float64)
    emb_arr = (np.stack(emb) if emb and emb[0] is not None else None)
    return coords, np.asarray(prob), np.asarray(votes), edges.reshape(-1, 3), emb_arr


def candidate_edges(coords, emb, model_edges, cfg):
    """Associations over the *pooled* candidate set, Ultrack's way.

    The first version of this pooled each node set's own edges, and that is
    wrong in a way worth recording: a set's edges only ever join that set's own
    nodes. Once the exclusion constraint makes the solver pick set A's node at
    t and set B's node at t+1 -- which is exactly what it is there to do --
    there is no edge between them, because that pair never existed in either
    set. Chains shatter, the survivors are isolated, and the solver drops them.
    Measured over 199 stacks: node recall 0.9567 -> 0.8401 and cv 0.8395 ->
    0.7694.

    Ultrack does not inherit associations either. They merge the input maps,
    re-derive the hypotheses, and then compute the associations once over the
    pooled set -- 2k nearest neighbours by centroid inside a radius, scored
    IoU^gamma (Methods, "Candidate segmentation association"). There are no
    masks here so IoU is unavailable, but the contrastive embedding answers the
    same question -- is this the same cell one frame later -- and it is defined
    for every pooled node, which is the property that matters.

    `assoc`:
      "model"  the committed edges only. Correct for one set, broken for many.
      "embed"  cosine over every kNN candidate. Uniform, so a pair is not
               favoured for happening to fall inside one set.
      "hybrid" the model's probability where it exists, cosine elsewhere.
               Mixes two scales -- the model saturates near 1 where cosine does
               not -- so it quietly prefers within-set pairs. Kept to measure
               that, not because it is principled.
    """
    mode = cfg.get("assoc", "model")
    if mode == "model" or emb is None:
        return model_edges

    known = {(int(a), int(b)): p for a, b, p in model_edges}
    by_t: dict[int, np.ndarray] = {}
    for t in np.unique(coords[:, 0]):
        by_t[int(t)] = np.flatnonzero(coords[:, 0] == t)

    out = []
    k = int(cfg["knn"])
    for t in sorted(by_t):
        src, tgt = by_t.get(t), by_t.get(t + 1)
        if src is None or tgt is None or not len(src) or not len(tgt):
            continue
        p_src = coords[src, 1:].astype(np.float64) * SCALE_ARR
        p_tgt = coords[tgt, 1:].astype(np.float64) * SCALE_ARR
        tree = cKDTree(p_src)
        dist, idx = tree.query(p_tgt, k=min(k, len(src)),
                               distance_upper_bound=cfg["max_link_um"])
        dist = np.atleast_2d(dist.T).T
        idx = np.atleast_2d(idx.T).T
        for j_local in range(len(tgt)):
            for c in range(idx.shape[1]):
                i_local = idx[j_local, c]
                if i_local >= len(src):          # cKDTree's "no neighbour" pad
                    continue
                i, j = int(src[i_local]), int(tgt[j_local])
                score = float(np.dot(emb[i], emb[j]))
                score = max(0.0, min(1.0, score))
                if mode == "hybrid":
                    score = max(score, known.get((i, j), 0.0))
                out.append([i, j, score])
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


def conflict_pairs(coords: np.ndarray, conflict_um: float) -> list[list[int]]:
    """Pairs of same-frame candidates too close to both be real cells.

    The point-detector stand-in for Ultrack's containment rule. Physical
    distance, so the anisotropy is respected -- in voxels the same separation
    means different things on z and on y/x.
    """
    if conflict_um <= 0:
        return []
    out: list[list[int]] = []
    by_t: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(coords[:, 0]):
        by_t[int(t)].append(i)
    for idx in by_t.values():
        idx_arr = np.asarray(idx)
        pts = coords[idx_arr, 1:].astype(np.float64) * SCALE_ARR
        tree = cKDTree(pts)
        for a, b in tree.query_pairs(conflict_um):
            out.append([int(idx_arr[a]), int(idx_arr[b])])
    return out


# --------------------------------------------------------------------------
# the ILP
# --------------------------------------------------------------------------

def solve(coords, prob, votes, edges, cfg):
    """Ultrack's formulation over the pooled hypotheses. -> (coords, edges)."""
    n_sets = max(1, int(votes.max()) if len(votes) else 1)
    keep = edges[:, 2] >= cfg["edge_threshold"] if len(edges) else np.zeros(0, bool)
    edges = edges[keep] if len(edges) else edges
    if not len(edges):
        return coords[:0], np.zeros((0, 4))

    # gamma-sharpening, Ultrack's w(i,j) = IoU^gamma with gamma = 4. The same
    # move on a probability: it widens the gap between a link the model is sure
    # of and one it merely prefers, so the solver stops treating 0.95 and 0.60
    # as nearly equivalent.
    w_edge = -cfg["edge_weight"] * np.power(edges[:, 2], cfg["gamma"])

    graph = td.graph.InMemoryGraph()
    import polars as pl

    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    graph.add_node_attr_key("node_w", pl.Float64, 0.0)

    # Consensus and confidence, both in [0, 1]. Ultrack scores no node term at
    # all -- a spurious detection is rejected because it pays appearance +
    # disappearance and earns no association. That still holds here; this only
    # tilts the choice *between* competing hypotheses, and `--node_weight 0`
    # turns it off to recover their formulation exactly.
    score = cfg["vote_mix"] * (votes / n_sets) + (1 - cfg["vote_mix"]) * prob
    node_w = -cfg["node_weight"] * score

    node_ids = graph.bulk_add_nodes([
        {"t": int(t), "z": float(z), "y": float(y), "x": float(x), "node_w": float(w)}
        for (t, z, y, x), w in zip(coords, node_w)
    ])
    graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
    graph.add_edge_attr_key("edge_w", pl.Float64, 0.0)
    graph.bulk_add_edges([
        {"source_id": node_ids[int(a)], "target_id": node_ids[int(b)],
         "edge_prob": float(p), "edge_w": float(w)}
        for (a, b, p), w in zip(edges, w_edge)
    ])

    pairs = conflict_pairs(coords, cfg["conflict_um"])
    if pairs:
        graph.bulk_add_overlaps([[node_ids[a], node_ids[b]] for a, b in pairs])

    solver = td.solvers.ILPSolver(
        edge_weight=td.EdgeAttr("edge_w"),
        node_weight=td.NodeAttr("node_w"),
        appearance_weight=cfg["appearance"],
        disappearance_weight=cfg["disappearance"],
        division_weight=cfg["division"],
        num_threads=1,
        timeout=cfg["timeout"],
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        solved = solver.solve(graph)

    na = solved.node_attrs()
    sel = np.asarray(na[td.DEFAULT_ATTR_KEYS.NODE_ID]).astype(int)
    ea = solved.edge_attrs()
    if td.DEFAULT_ATTR_KEYS.SOLUTION in ea.columns:
        ea = ea.filter(ea[td.DEFAULT_ATTR_KEYS.SOLUTION])

    # `node_ids` is the identity here (bulk_add_nodes over a fresh graph), but
    # invert it rather than assume so: a future backend numbering nodes any
    # other way would otherwise silently scramble the coordinates.
    back = {nid: i for i, nid in enumerate(node_ids)}
    rows = np.array(sorted(back[int(s)] for s in sel), dtype=np.int64)
    remap = -np.ones(len(coords), dtype=np.int64)
    remap[rows] = np.arange(len(rows))

    out_edges = []
    for s, t, p in zip(ea["source_id"], ea["target_id"], ea["edge_prob"]):
        a, b = remap[back[int(s)]], remap[back[int(t)]]
        if a >= 0 and b >= 0:
            out_edges.append([a, b, float(p), 0.0])
    return coords[rows], np.asarray(out_edges, dtype=np.float64).reshape(-1, 4)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

_GT: dict[str, object] = {}
_META = None


def apply_pp(coords, edges, sample):
    """`main_postprocess.py`'s chain, on whatever produced `coords`/`edges`.

    The pipeline's headline number is post-processed, so an arm scored raw is
    not comparable to it. Running the same chain over both sides is what makes
    "does pooling hypotheses help" answerable rather than confounded with
    "does this arm happen to prune more".
    """
    import main_postprocess as MP
    from stages import run_divisions, run_pipeline, run_pp_public

    coords = np.asarray(coords, float)
    edges = np.asarray(edges, float).reshape(-1, 4)
    if not len(edges):
        return coords, edges
    with contextlib.redirect_stdout(io.StringIO()):
        coords, edges = run_pipeline(coords, edges, recentre=None)
        coords, edges = run_pp_public(coords, edges, sample=sample, **MP.PP_PUBLIC_TUNED)
        coords, edges = run_divisions(coords, edges, **MP.DIVISIONS_TUNED)
    return coords, edges


def _n_total(sample: str) -> float:
    global _META
    import pandas as pd
    if _META is None:
        p = Path(TRAIN_DIR).parent / "metadata.csv"
        _META = pd.read_csv(p).set_index("sample_name") if p.exists() else pd.DataFrame()
    if sample in getattr(_META, "index", []):
        return float(_META.loc[sample, "estimated_number_of_nodes"])
    return float("nan")


def score_one(sample: str, coords, edges) -> dict:
    if sample not in _GT:
        r = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{sample}.geff")
        _GT[sample] = r[0] if isinstance(r, tuple) else r
    gt = _GT[sample]
    # An arm can select nothing at all -- a mis-scaled association score makes
    # every link cost more than it earns, and the solver correctly returns the
    # empty graph. `evaluate` then registers no match and `node_recall` raises
    # KeyError('match_node_id') deep inside tracksdata, which killed a whole
    # sweep rather than scoring that arm 0. A degenerate result is a result.
    if len(coords) == 0 or len(np.asarray(edges).reshape(-1, 4)) == 0:
        # Every key `summarise` / `macro_summarise` read, not just the ones a
        # caller happens to print -- an incomplete guard row turns a degenerate
        # arm into a KeyError that kills the whole sweep, which it did twice.
        return {"edge_tp": 0, "edge_fp": 0, "edge_fn": 0, "edge_jaccard": 0.0,
                "adj_edge_jaccard": 0.0, "division_jaccard": 0.0, "score": 0.0,
                "node_recall": 0.0, "division_tp": 0, "division_fp": 0,
                "division_fn": 0, "n_nodes_pred": int(len(coords)),
                "num_pred_nodes": int(len(coords)),
                "n_nodes_estimated": _n_total(sample),
                "total_node_ratio": 0.0, "n_edges_pred": 0,
                "sample_name": sample, "embryo": sample.split("_")[0],
                "fold": EMBRYOS.index(sample.split("_")[0])
                if sample.split("_")[0] in EMBRYOS else None}
    with contextlib.redirect_stdout(io.StringIO()):
        g = build_graph(np.round(np.asarray(coords)).astype(np.int64), [
            (int(a), int(b), float(p), float(d))
            for a, b, p, d in np.asarray(edges, float).reshape(-1, 4)])
    row = per_sample_metrics(evaluate(g, gt, scale=SCALE), _n_total(sample), node_recall(g, gt))
    one = summarise([row])
    row["division_jaccard"] = one["division_jaccard"]
    row["score"] = one["score"]
    row["sample_name"] = sample
    row["embryo"] = sample.split("_")[0]
    row["fold"] = EMBRYOS.index(row["embryo"]) if row["embryo"] in EMBRYOS else None
    row["n_nodes_pred"] = len(coords)
    return row


def _job(arg):
    sample, set_paths, cfg = arg
    paths = [Path(p) for p in set_paths]
    coords, prob, votes, edges, emb = pool_hypotheses(paths, sample)
    n_pool = len(coords)
    edges = candidate_edges(coords, emb, edges, cfg)
    coords, edges = solve(coords, prob, votes, edges, cfg)
    if cfg.get("pp"):
        coords, edges = apply_pp(coords, edges, sample)
    row = score_one(sample, coords, edges)
    row["n_pool"] = n_pool
    return row


def _job_raw(arg):
    """One node set, exactly as `main_predict` wrote it -- the baseline."""
    sample, set_path, pp = arg
    d = np.load(Path(set_path) / f"{sample}.npz")
    coords = d["coords"].astype(np.int64)
    edges = np.asarray(d["edges"], float).reshape(-1, 4)
    if pp:
        coords, edges = apply_pp(coords, edges, sample)
    row = score_one(sample, coords, edges)
    row["n_pool"] = len(d["coords"])
    return row


DEFAULTS = dict(
    conflict_um=5.0,
    edge_threshold=0.05,
    gamma=4.0,
    edge_weight=1.0,
    node_weight=0.0,
    vote_mix=0.5,
    appearance=0.0,
    disappearance=1.5,
    division=1.0,
    timeout=900.0,
    pp=0,
    assoc="model",
    knn=5,
    max_link_um=15.0,
)


def report(name: str, rows: list[dict], ref: float | None = None) -> float:
    groups, macro = macro_summarise(rows)
    cv = macro["score"]
    d = f"{cv - ref:+.4f}" if ref is not None else "     "
    print(f"{name[:40]:<42}{cv:>9.4f}{d:>9}{macro['adj_edge_jaccard']:>9.4f}"
          f"{macro['division_jaccard']:>8.4f}{macro['node_recall']:>9.4f}"
          f"{np.mean([r['n_nodes_pred'] for r in rows]):>10.0f}"
          f"{np.mean([r.get('n_pool', r['n_nodes_pred']) for r in rows]):>10.0f}",
          flush=True)
    return cv


HEADER = (f"{'arm':<42}{'cv':>9}{'delta':>9}{'adj':>9}{'div':>8}"
          f"{'recall':>9}{'nodes':>10}{'pool':>10}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", nargs="+", required=True,
                   help="the preds_* folders to pool as hypotheses")
    p.add_argument("--baseline", default=None,
                   help="the node set to compare against; default = the first --sets entry")
    p.add_argument("--samples", nargs="*", default=None, help="explicit stack names")
    p.add_argument("--embryo", default=None,
                   help="keep only this embryo's stacks. A preds folder holds "
                        "every fold's validation stacks at once, so globbing it "
                        "silently scores both embryos -- name the one you mean")
    p.add_argument("--limit", type=int, default=None, help="first N stacks")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--sweep", default="",
                   help="'k=v;k2=v2|k=v3' arms on top of DEFAULTS")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    p.add_argument("--out", default=None, help="write the selected npz here")
    p.add_argument("--skip_baselines", action="store_true",
                   help="do not recompute the baseline and single-set arms; "
                        "use when they are already measured on this exact split")
    a = p.parse_args()

    sets = [Path(s) for s in a.sets]
    baseline = Path(a.baseline) if a.baseline else sets[0]
    samples = a.samples or sorted(f.stem for f in baseline.glob("*.npz"))
    if a.embryo:
        samples = [s for s in samples if s.split("_")[0] == a.embryo]
    if a.limit:
        samples = samples[: a.limit]
    print(f"-> {len(samples)} stacks, {len(sets)} hypothesis sets, {a.jobs} workers")
    for s in sets:
        print(f"   {s}")

    base = {k: getattr(a, k) for k in DEFAULTS}
    arms = [{}]
    if a.sweep:
        arms = []
        for arm in a.sweep.split("|"):
            kw = {}
            for part in arm.split(";"):
                if part.strip():
                    k, v = part.split("=", 1)
                    kw[k.strip()] = json.loads(v.strip())
            arms.append(kw)

    print(f"\n{HEADER}")
    if a.skip_baselines:
        ref = None
    else:
        with Pool(a.jobs) as pool:
            rows = list(pool.imap_unordered(
                _job_raw, [(s, str(baseline), 0) for s in samples]))
        ref = report(f"baseline raw  {baseline.name}", rows)
    if a.pp and not a.skip_baselines:
        with Pool(a.jobs) as pool:
            rows = list(pool.imap_unordered(
                _job_raw, [(s, str(baseline), 1) for s in samples]))
        report(f"baseline +PP  {baseline.name}", rows, ref)

    # The same ILP over one node set: separates "a solver helps" from
    # "several hypotheses help", which is the whole question.
    if not a.skip_baselines:
      with Pool(a.jobs) as pool:
        rows = list(pool.imap_unordered(
            _job, [(s, [str(baseline)], base) for s in samples]))
      report(f"ILP, 1 set ({baseline.name})", rows, ref)

    for kw in arms:
        cfg = {**base, **kw}
        with Pool(a.jobs) as pool:
            rows = list(pool.imap_unordered(
                _job, [(s, [str(x) for x in sets], cfg) for s in samples]))
        label = "ILP, %d sets" % len(sets) + (
            "  " + ",".join(f"{k}={v}" for k, v in kw.items()) if kw else "")
        report(label, rows, ref)

        if a.out:
            out = Path(a.out)
            out.mkdir(parents=True, exist_ok=True)
            for s in samples:
                coords, prob, votes, edges, emb = pool_hypotheses(sets, s)
                edges = candidate_edges(coords, emb, edges, cfg)
                coords, edges = solve(coords, prob, votes, edges, cfg)
                np.savez_compressed(
                    out / f"{s}.npz",
                    coords=np.round(coords).astype(np.int16),
                    edges=np.asarray(edges, np.float64).reshape(-1, 4))
            print(f"-> npz: {out}")


if __name__ == "__main__":
    main()
