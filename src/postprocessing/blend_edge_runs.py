"""Blend two edge runs into one candidate graph, then post-process as usual.

    cd src/postprocessing
    python blend_edge_runs.py --a ../logs/edge_run/2026-09-13/0 \
                              --b ../logs/edge_run/2026-09-11/2 --limit 5

``--a`` / ``--b`` take *run* folders, not prediction folders: each run holds a
threshold sweep and several post-processing attempts, and which of them is the
run's best is recorded rather than guessable. See ``best_preds_dir``.

Two edge runs disagree in two different ways and the distinction decides what a
blend can do:

* **Same node set, different model.** `2026-09-13/0` and `2026-09-14/5` sit on
  `2026-09-13/1/preds_r6_0.40` with different contrastive embeddings, so their
  candidate *pairs* are identical (151,974 in both) and only the probabilities
  differ. Union node recall gain: exactly 0. A blend here is a probability
  ensemble and nothing else.
* **Different node sets.** `2026-09-13/0` (cv 0.8699) and `2026-09-11/2`
  (0.8658) come from different detectors, and their union reaches node recall
  0.8933 against 0.8742 for the better one alone -- **+0.0191 of headroom**.

Node recall is the thing to move first: the edge metric only scores links whose
*both* endpoints match a ground-truth node, so a cell neither run detected is
two edges nobody can win, and the linker is already at ~95% of what its node set
makes reachable.

The naive union doubles the node count (2.02x), which the metric's node-count
adjustment would eat outright. Merging near-duplicates fixes that, because most
of the union is the same cells seen twice:

    merge_um   nodes vs A   union recall   gain over A
        0         2.02x        0.8933        +0.0191
        1         1.25x        0.8924        +0.0182
        2         1.06x        0.8899        +0.0157
        4         1.05x        0.8895        +0.0153

2 um keeps 82% of the recall headroom for 6% more nodes, and is the default.

Merging by *proximity* rather than identity is also what makes the associations
transfer, which is where an earlier attempt at this went wrong: pooling two
runs' edges while keeping their nodes distinct leaves a chain that switches
between the runs with no edge to switch along, so the chains shatter and the
solver drops the pieces (measured: node recall 0.9567 -> 0.8401). Once a cell
is *one* node, both runs' candidates land on it and the graph stays connected.

Divisions are deliberately not the blend's problem. The ILP runs with forks
disabled and the existing `main_edge_postprocess` chain -- `assign -> recover ->
divisions -> prune -> linefit`, on its tuned defaults -- is applied afterwards,
so the division stage sees the blended graph and does its usual job.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: F401,E402  (chdir to src/ and set the path)

import main_edge_postprocess as MEP  # noqa: E402
from params import SCALE  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
SCALE_ARR = np.array(SCALE, dtype=np.float64)


# --------------------------------------------------------------------------
# which predictions a run's own scores say are its best
# --------------------------------------------------------------------------

def best_preds_dir(run_dir: Path) -> tuple[Path, float, str]:
    """The candidate folder behind a run's highest-scoring result.

    A run writes one ``scores_preds_<th>.json`` per swept link threshold and one
    more per post-processing attempt (``_pp``, ``_pp2``, ``_pp_best`` ...), and
    the best is usually a post-processed one -- on ``edge_run/2026-09-13/0`` the
    raw thresholds top out at 0.8174 while ``scores_preds_0.45_pp2.json`` reads
    0.8699. Picking by filename would therefore pick the wrong thing, and
    picking the best *raw* score would pick the wrong threshold: 0.45 wins after
    post-processing but only by 0.0002 before it.

    So rank every scores file by ``cv_score`` and follow the winner's own
    ``preds_dir`` field, which names the folder it was computed from. For a
    ``_pp`` file that field is the *input* -- the raw candidate folder -- which
    is exactly what a blend needs, since only those npz carry the
    ``candidates`` array. Nothing here has to know the naming convention.

    ``_pl`` files are skipped: a pseudo-label pass is scored on its own training
    embryos and its number is not comparable.

    Returns ``(preds_dir, cv, scores_filename)``.
    """
    run_dir = Path(run_dir)
    best = None
    for f in sorted(run_dir.glob("scores_*.json")):
        if "_pl" in f.stem:
            continue
        try:
            blob = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        cv = blob.get("cv_score")
        # NaN loses every comparison, so it would silently rank first or last
        # depending which way the test is written; drop it explicitly.
        if cv is None or cv != cv:
            continue
        if best is None or cv > best[0]:
            best = (float(cv), f, blob.get("preds_dir"))
    if best is None:
        raise SystemExit(f"no scored result under {run_dir}")

    cv, f, named = best
    cand = []
    if named:
        cand.append(run_dir / Path(named).name)
    # Fall back to the filename only if the json predates the `preds_dir` field.
    stem = f.stem[len("scores_"):]
    cand.append(run_dir / stem)
    for d in cand:
        if d.is_dir() and any(d.glob("*.npz")):
            if not _has_candidates(d):
                raise SystemExit(
                    f"{d} has no `candidates` array -- blending needs the raw "
                    f"predictions main_edge_predict.py wrote with "
                    f"--dump_candidates, not a post-processed folder"
                )
            return d, cv, f.name
    raise SystemExit(f"{f.name} scores {cv:.4f} but its predictions are gone")


def _has_candidates(d: Path) -> bool:
    one = next(iter(sorted(d.glob("*.npz"))), None)
    return one is not None and "candidates" in np.load(one).files


# --------------------------------------------------------------------------
# merging
# --------------------------------------------------------------------------

def merge_nodes(ca: np.ndarray, cb: np.ndarray, merge_um: float, merge_pos: str = "a"):
    """Fold B's detections into A's where they name the same cell.

    Returns ``(coords, map_a, map_b)`` where the maps send each run's original
    row to a row of ``coords``. Greedy nearest-neighbour inside a frame: B's
    node joins A's if it is within ``merge_um``, otherwise it is a new node.
    Greedy rather than optimal because the alternative is a Hungarian solve per
    frame for a decision that is nearly always unambiguous -- at 2 um the
    nearest A node is either obviously the same cell or 5+ um away.
    """
    if merge_um <= 0:
        return (np.concatenate([ca, cb]), np.arange(len(ca)),
                np.arange(len(ca), len(ca) + len(cb)))
    out = [ca.copy()]
    map_b = np.full(len(cb), -1, dtype=np.int64)
    nxt = len(ca)
    for t in np.unique(cb[:, 0]):
        bi = np.flatnonzero(cb[:, 0] == t)
        ai = np.flatnonzero(ca[:, 0] == t)
        if not len(ai):
            map_b[bi] = np.arange(nxt, nxt + len(bi))
            nxt += len(bi)
            out.append(cb[bi])
            continue
        d, j = cKDTree(ca[ai][:, 1:] * SCALE_ARR).query(cb[bi][:, 1:] * SCALE_ARR, k=1)
        hit = d <= merge_um
        map_b[bi[hit]] = ai[j[hit]]
        if merge_pos == "mean" and hit.any():
            # Average the two runs' estimate of the same cell. GT coordinates
            # are integers so there is no sub-voxel truth to recover, but the
            # 7 um match is on the *predicted* position, and two independent
            # estimates of one centre average closer to it than either.
            tgt = ai[j[hit]]
            out[0][tgt, 1:] = 0.5 * (out[0][tgt, 1:] + cb[bi[hit]][:, 1:])
        fresh = bi[~hit]
        if len(fresh):
            map_b[fresh] = np.arange(nxt, nxt + len(fresh))
            nxt += len(fresh)
            out.append(cb[fresh])
    return np.concatenate(out), np.arange(len(ca)), map_b


def blend_candidates(cand_a, cand_b, map_a, map_b, n_nodes, cfg):
    """One candidate array over the merged nodes.

    A pair both runs scored gets its probabilities combined by ``prob_blend``:

      max      optimistic -- either model vouching for a link is enough
      mean     the ensemble average
      min      both must agree, which is precision-first
      noisy_or 1 - (1-pa)(1-pb), the independent-evidence reading. Two models
               at 0.6 become 0.84, which is the only rule here that can exceed
               both inputs, and the one that matches "two independent detectors
               saw it" if you believe the two are independent. They are not
               entirely -- same architecture, same node grid -- so this is the
               optimistic end.
      low_margin_consensus
               the public notebooks' rule (`912`, `LOW_MARGIN_MAX` 0.35,
               `SECONDARY_EDGE_WEIGHT` 0.15). B contributes only where A is
               genuinely unsure *and* the two agree on which parent a child
               belongs to:

                   u = clamp((margin_max - margin_A(child)) / margin_max, 0, 1)
                   w = secondary_weight * u
                   w = 0 where A and B disagree on the child's best parent
                   p = (1 - w) * pa + w * pb

               `margin_A(child)` is the gap between the best and second-best
               parent probability A assigns that child. Where A is confident
               the margin is wide, u is 0 and A's probability is untouched; a
               disagreement is resolved in A's favour rather than averaged.
               Unlike every other rule here this is not pairwise -- it needs
               each child's whole column -- which is why `blend_candidates`
               computes margins before combining.

    A pair only one run proposed is kept at ``solo_scale`` times its own
    probability. Below 1 that encodes "a link only one model even considered is
    worth less", which matters because the two runs contribute unequal numbers
    of candidates once the node sets differ.
    """
    def remap(cand, mapping):
        c = np.asarray(cand, float).reshape(-1, 5)
        if not len(c):
            return c
        c = c.copy()
        c[:, 0] = mapping[c[:, 0].astype(int)]
        c[:, 1] = mapping[c[:, 1].astype(int)]
        return c

    ra, rb = remap(cand_a, map_a), remap(cand_b, map_b)
    acc: dict[tuple[int, int], list] = {}
    for c, which in ((ra, 0), (rb, 1)):
        for src, tgt, p, d, flag in c:
            a, b = int(src), int(tgt)
            if a == b or a < 0 or b < 0:
                continue                       # a merge can collapse a pair
            e = acc.setdefault((a, b), [None, None, d, flag])
            if e[which] is None or p > e[which]:
                e[which] = float(p)
            e[2] = min(e[2], float(d))
            e[3] = max(e[3], float(flag))

    rule = cfg["prob_blend"]
    margin = argmax_a = argmax_b = None
    if rule == "low_margin_consensus":
        # Per *child*: A's top-two parent gap, and each run's preferred parent.
        best_a: dict[int, list[float]] = {}
        top_a: dict[int, tuple[float, int]] = {}
        top_b: dict[int, tuple[float, int]] = {}
        for (a, b), (pa, pb, _, _) in acc.items():
            if pa is not None:
                best_a.setdefault(b, []).append(pa)
                if pa > top_a.get(b, (-1.0, -1))[0]:
                    top_a[b] = (pa, a)
            if pb is not None and pb > top_b.get(b, (-1.0, -1))[0]:
                top_b[b] = (pb, a)
        margin = {}
        for child, ps in best_a.items():
            ps = sorted(ps, reverse=True)
            # A single candidate means no competition, so A is as sure as it
            # can be and the secondary must not intervene.
            margin[child] = ps[0] - ps[1] if len(ps) > 1 else 1.0
        argmax_a = {c: v[1] for c, v in top_a.items()}
        argmax_b = {c: v[1] for c, v in top_b.items()}

    out = np.empty((len(acc), 5), dtype=np.float64)
    for i, ((a, b), (pa, pb, d, flag)) in enumerate(acc.items()):
        if pa is None or pb is None:
            p = (pa if pa is not None else pb) * cfg["solo_scale"]
        elif rule == "max":
            p = max(pa, pb)
        elif rule == "mean":
            p = 0.5 * (pa + pb)
        elif rule == "min":
            p = min(pa, pb)
        elif rule == "noisy_or":
            p = 1.0 - (1.0 - pa) * (1.0 - pb)
        elif rule == "low_margin_consensus":
            mmax = cfg["margin_max"]
            u = min(max((mmax - margin.get(b, 1.0)) / max(mmax, 1e-9), 0.0), 1.0)
            w = cfg["secondary_weight"] * u
            if argmax_a.get(b, -1) != argmax_b.get(b, -2):
                w = 0.0                      # disagreement -> keep A
            p = (1.0 - w) * pa + w * pb
        else:
            raise ValueError(f"unknown prob_blend {rule!r}")
        out[i] = (a, b, p, d, flag)
    return out


def node_support(coords, cand, cfg):
    """Per-node evidence, for the ILP's `node_weight` and for consensus gating.

    ``votes`` is how many runs proposed the node; ``best_in`` / ``best_out`` the
    strongest candidate link entering and leaving it. A node no link wants is
    one the blend should be free to drop.
    """
    n = len(coords)
    best_in = np.zeros(n)
    best_out = np.zeros(n)
    for a, b, p, *_ in cand:
        a, b = int(a), int(b)
        if p > best_out[a]:
            best_out[a] = p
        if p > best_in[b]:
            best_in[b] = p
    return best_in, best_out


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def select_nodes(coords, cand, votes, cfg):
    """Which merged nodes to keep. Returns a boolean mask.

    Four strategies, cheapest first. The ILP is the Ultrack one; the others are
    here because a blend that needs a solver to beat a threshold is a blend
    worth being suspicious of.

      a_only     keep A's node set exactly, so B contributes only its opinion
                 of A's links. The pure probability ensemble, and the control
                 that says how much of any gain is the extra nodes rather than
                 the extra model
      union      keep everything the merge produced
      consensus  keep only nodes both runs proposed -- precision-first, and the
                 direct test of whether agreement between runs is informative
      supported  drop nodes with no candidate link above `support_min` on
                 either side: the cheap stand-in for the ILP's appearance /
                 disappearance economics
      ilp        tracksdata's ILPSolver over the blended graph, forks disabled
    """
    mode = cfg["select"]
    n = len(coords)
    if mode == "union":
        return np.ones(n, bool)
    if mode == "a_only":
        mask = np.zeros(n, bool)
        mask[: cfg["_n_a"]] = True
        return mask
    if mode == "consensus":
        return votes >= 2
    best_in, best_out = node_support(coords, cand, cfg)
    if mode == "supported":
        return np.maximum(best_in, best_out) >= cfg["support_min"]
    if mode != "ilp":
        raise ValueError(f"unknown select {mode!r}")

    import polars as pl
    import tracksdata as td
    keep = cand[:, 2] >= cfg["ilp_edge_min"]
    c = cand[keep]
    if not len(c):
        return np.ones(n, bool)
    graph = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        graph.add_node_attr_key(k, pl.Float64, -999999.0)
    graph.add_node_attr_key("node_w", pl.Float64, 0.0)
    # Reward keeping a node the runs agree on; the ILP otherwise has no reason
    # to prefer a corroborated detection to a lone one.
    w = -cfg["ilp_node_weight"] * (votes / max(votes.max(), 1))
    ids = graph.bulk_add_nodes([
        {"t": int(t), "z": float(z), "y": float(y), "x": float(x), "node_w": float(ww)}
        for (t, z, y, x), ww in zip(coords, w)])
    graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
    graph.add_edge_attr_key("edge_w", pl.Float64, 0.0)
    graph.bulk_add_edges([
        {"source_id": ids[int(a)], "target_id": ids[int(b)],
         "edge_prob": float(p), "edge_w": float(-(p ** cfg["ilp_gamma"]))}
        for a, b, p, *_ in c])
    solver = td.solvers.ILPSolver(
        edge_weight=td.EdgeAttr("edge_w"), node_weight=td.NodeAttr("node_w"),
        appearance_weight=cfg["ilp_appearance"],
        disappearance_weight=cfg["ilp_disappearance"],
        # Forks are the post-processing's job, not the blend's -- a cost this
        # high provably never divides (see main_edge_postprocess.ilp_division).
        division_weight=10.0,
        num_threads=1, timeout=cfg["ilp_timeout"])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        solved = solver.solve(graph)
    sel = np.asarray(solved.node_attrs()[td.DEFAULT_ATTR_KEYS.NODE_ID]).astype(int)
    back = {nid: i for i, nid in enumerate(ids)}
    mask = np.zeros(n, bool)
    mask[[back[int(s)] for s in sel]] = True
    return mask


def blend(sample, dir_a, dir_b, cfg):
    """(coords, edges, cand) for one stack, after merge, blend and selection."""
    za = np.load(Path(dir_a) / f"{sample}.npz")
    zb = np.load(Path(dir_b) / f"{sample}.npz")
    ca, cb = za["coords"].astype(np.float64), zb["coords"].astype(np.float64)
    if cfg.get("swap"):
        ca, cb = cb, ca
        za, zb = zb, za
    coords, map_a, map_b = merge_nodes(ca, cb, cfg["merge_um"],
                                       cfg.get("merge_pos", "a"))
    cfg = {**cfg, "_n_a": len(ca)}       # `a_only` needs to know where A ends
    votes = np.zeros(len(coords), np.int64)
    np.add.at(votes, map_a, 1)
    np.add.at(votes, map_b, 1)
    cand = blend_candidates(za["candidates"], zb["candidates"], map_a, map_b,
                            len(coords), cfg)
    mask = select_nodes(coords, cand, votes, cfg)
    if not mask.all():
        remap = -np.ones(len(coords), np.int64)
        remap[np.flatnonzero(mask)] = np.arange(int(mask.sum()))
        coords = coords[mask]
        ok = (remap[cand[:, 0].astype(int)] >= 0) & (remap[cand[:, 1].astype(int)] >= 0)
        cand = cand[ok]
        cand[:, 0] = remap[cand[:, 0].astype(int)]
        cand[:, 1] = remap[cand[:, 1].astype(int)]
    return coords, np.empty((0, 4)), cand


DEFAULTS = dict(
    merge_um=2.0,
    prob_blend="max",
    solo_scale=1.0,
    select="union",
    support_min=0.05,
    ilp_edge_min=0.05,
    ilp_node_weight=0.0,
    ilp_gamma=1.0,
    ilp_appearance=0.0,
    ilp_disappearance=1.5,
    ilp_timeout=600.0,
    merge_pos="a",
    swap=0,
    # low_margin_consensus only. Public: 0.35 and 0.15.
    margin_max=0.35,
    secondary_weight=0.15,
)


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def _spread(run_dir: Path, scores_name: str, n: int) -> list[str]:
    """``n`` stacks spanning the run's difficulty range, not the first ``n``.

    The first-``n`` habit cost a day: a subset taken alphabetically off
    `2026-09-14/4` turned out to be the easy end of fold 0 -- baseline node
    recall 0.9935 against 0.9574 over all 199, and no divisions at all -- and a
    broken method measured +0.0166 on it before failing on the full set. Same
    idiom as `exp_edge_pp.py --spread`.
    """
    per = json.loads((run_dir / scores_name).read_text()).get("per_sample", {})
    if not per:
        return sorted(p.stem for p in (run_dir / "preds_0.45").glob("*.npz"))[:n]
    ranked = sorted((v.get("score", 0.0), k) for k, v in per.items())
    idx = [round(i * (len(ranked) - 1) / max(n - 1, 1)) for i in range(n)]
    return [ranked[i][1] for i in idx]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="run folder of the stronger model")
    p.add_argument("--b", required=True, help="run folder of the second model")
    p.add_argument("--preds_a", default=None, help="override A's predictions folder")
    p.add_argument("--preds_b", default=None, help="override B's predictions folder")
    p.add_argument("--limit", type=int, default=None, help="how many stacks")
    p.add_argument("--samples", nargs="*", default=None)
    p.add_argument("--first", action="store_true",
                   help="take the first --limit stacks instead of a difficulty "
                        "spread. Faster to reason about, and misleading -- see _spread")
    p.add_argument("--jobs", type=int, default=10)
    p.add_argument("--out", default=None, help="write the blended npz here")
    p.add_argument("--no_score", action="store_true")
    for k, v in DEFAULTS.items():
        if k.startswith("_"):
            continue
        p.add_argument(f"--{k}", type=type(v), default=v)
    a = p.parse_args()

    da, cva, fa = ((Path(a.preds_a), float("nan"), "given") if a.preds_a
                   else best_preds_dir(Path(a.a)))
    db, cvb, fb = ((Path(a.preds_b), float("nan"), "given") if a.preds_b
                   else best_preds_dir(Path(a.b)))
    print(f"-> A: {da}   cv {cva:.4f}  ({fa})")
    print(f"-> B: {db}   cv {cvb:.4f}  ({fb})")
    if cvb > cva:
        print("   note: B scores higher than A. A is the merge base and keeps its "
              "coordinates, and base order is worth ~0.002 -- consider swapping.")

    cfg = {k: getattr(a, k) for k in DEFAULTS if not k.startswith("_")}
    samples = a.samples
    if samples is None:
        allnames = sorted(p_.stem for p_ in da.glob("*.npz"))
        n = a.limit or len(allnames)
        samples = (allnames[:n] if a.first or not fa.startswith("scores_")
                   else _spread(Path(a.a), fa, n))
    print(f"-> {len(samples)} stacks, {cfg}\n")

    import main_edge_postprocess as _MEP
    pp = dict(_MEP.DEFAULTS)
    pp.update(out_dir="", geff_dir="", no_score=False, assign=1)

    # The worker reads module globals rather than taking a closure: a closure
    # cannot be pickled, and the alternative -- passing the config per task --
    # would ship the two prediction folders' paths with every stack.
    global _JOB
    _JOB = (da, db, cfg, pp)
    with Pool(a.jobs) as pool:
        results = list(pool.imap_unordered(_blend_one, samples))

    if a.out:
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        for sample, coords, edges in results:
            np.savez_compressed(out / f"{sample}.npz",
                                coords=np.round(np.asarray(coords, np.float64)).astype(np.int16),
                                edges=np.asarray(edges, np.float64).reshape(-1, 4))
        print(f"-> npz: {out}")

    if a.no_score:
        return
    from multi_hypothesis import score_one
    from tracking_cellmot.metrics import macro_summarise
    rows = [score_one(s, c, e) for s, c, e in results]
    _, m = macro_summarise(rows)
    print(f"blend  cv {m['score']:.4f}   adj {m['adj_edge_jaccard']:.4f}   "
          f"div {m['division_jaccard']:.4f}   node_recall {m['node_recall']:.4f}   "
          f"nodes {np.mean([r['n_nodes_pred'] for r in rows]):.0f}")


_JOB = None


def _blend_one(sample):
    """One stack: blend, post-process. Set up by `main` through `_JOB`."""
    da, db, cfg, pp = _JOB
    import main_edge_postprocess as _MEP
    coords, edges, cand = blend(sample, da, db, cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        coords, edges = _MEP.postprocess(coords, edges, cand, pp)
    return sample, coords, edges


if __name__ == "__main__":
    main()
