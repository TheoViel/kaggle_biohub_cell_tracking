"""Search `main_edge_postprocess.py`'s knobs on an edge model's candidate dump.

    cd src/postprocessing
    python tune_edge_pp.py                                   # latest edge run
    python tune_edge_pp.py --preds_dir ../logs/edge_run/2026-09-14/2   # a run
    python tune_edge_pp.py --preds_dir ../logs/edge_run/2026-09-11/0/preds_0.45
    python tune_edge_pp.py --sweep "min_track=4,5,6,7"       # one knob, explicit
    python tune_edge_pp.py --arms "|min_track=5;div_min_prob=0.3"   # free arms

`common` chdirs to `src/`, so a relative `--preds_dir` is relative to `src/` --
`../logs/...` -- exactly as `main_edge_postprocess.py` takes it, and not to the
`src/postprocessing/` you are standing in.

With no `--sweep`/`--arms` it runs the built-in coordinate descent: the knobs in
`SEARCH` in order, each walked away from the current best, the winner kept, then
a second pass so a knob can move again once the others have. That is how the
`DEFAULTS` block in `main_edge_postprocess.py` was tuned by hand, made
repeatable -- retuning is needed on every new checkpoint, because these knobs
trade against how much the detector over-detects and that changes run to run.

*Always start from `DEFAULTS`.* Earlier versions resumed from the best-scoring
`scores_*.json` in the run folder, which was cheaper -- a knob already right
cost two arms instead of a walk. It also compounds: each retune began where the
last one's fit landed, so whatever that search took from the noise was inherited
rather than re-derived, and a value that won once was never asked again. The
search is five knobs now and can afford to re-derive all of them. `--preds_dir`
still takes either the candidate folder or the run folder above it; given a run
folder the best score file is read only to name the candidate folder it used.

What keeps the search cheap, given the old version scored every value of every
knob on every pass and took hours:

*Walk, do not sweep.* A knob is a line search along its grid, not a sweep of it:
score one step either side of where it sits, and if a direction wins, keep
stepping that way until it stops winning. A knob that is already right costs
two arms rather than the whole grid, and one that is wrong still crosses the
grid -- it just pays per step it actually takes. The grids in `SEARCH` are step
sizes rather than boundaries: a walk still improving at the end of its grid
keeps going at the width of the outermost interval, until it stops improving or
reaches the knob's `LIMITS` entry, because a grid that runs out mid-climb is a
grid drawn too narrow and stopping there would report its edge as an optimum.
The grids must stay sorted; a starting value that is not on its grid is
inserted into the walk rather than snapped to a neighbour.

The division gates in `SWEEP` are the exception, and are swept whole and then
refined at half a step. They are a conjunction, so each one reads flat while
the others are still rejecting and then falls off a cliff -- a shape a line
search reads as a local optimum and stops on. `sweep_knob` says more.

The ILP node filter is off unless `--ilp`, whatever the starting config says.
It is the first stage of the chain and reads none of the knobs below it, so
with `--ilp` it is solved once per stack and the filtered graph is cached for
every arm after that: the search pays for the solve once rather than per arm.
Its own knobs are left alone unless `--tune_ilp`, which visits them first --
they decide which nodes exist, and everything below is fitted to what survives
-- and costs one solve per stack per value tried.

`postprocessing/ilp_cache.py` does the same freeze to disk, and is still the
better move when the ILP settings are settled and the tuning is not: it writes
a predictions folder, so the solve is paid once across every future search
rather than once per search.

It scores exactly what `main_edge_postprocess.py` scores -- same `postprocess`
chain, same `evaluate`, same macro roll-up -- but writes no npz and no geff, so
an arm costs the scoring and nothing else.

When the search lands it runs `main_edge_postprocess.py` itself, over every
stack rather than the tuned fold, which is what produces the artefacts and the
two-embryo score. `--no_run` prints the command instead.

Two final runs, always: `<preds>_pp_<tag>_tuned` without the ILP and
`<preds>_pp_<tag>_tuned_ilp` with it. The names mean exactly that and nothing
else, whichever way the search was invoked. Which one wins has reversed between
checkpoints, so it is a measurement worth having on disk every time rather than
only when the search happened to run without the filter; and because the filter
does overlapping work with `recover` and `min_track`, an arm that only won by
leaning on their absence tends not to survive it. When the search never looked
at the `ilp_*` knobs the ILP arm takes them from `ILP_AFTER`; under `--tune_ilp`
it keeps what the search found.

The reported number is the competition score, macro-averaged: one vote per
embryo, `adj_edge_jaccard + 0.1 * division_jaccard`. The per-embryo columns are
there because that is the bar `README_edge.md` sets and the macro mean hides
it: a knob that buys 0.006 on 44b6 and sells 0.001 on 6bba shows up as a gain,
and on two embryos that is one embryo's worth of evidence, not two. `--both`
makes the rule binding rather than advisory.

That argument only has force on stacks the knobs were not fitted to, so the
search runs on one fold -- `--fold 0`, the `44b6` stacks -- and leaves the other
embryo out. A dozen knobs against 199 stacks, each worth thousandths, is enough
rope to fit the noise; the held-out embryo is what distinguishes a knob that
generalised from one that memorised. Score the winner on `--fold all` when the
search lands. The cv printed under `--fold` is one embryo and is not the
two-embryo number `Experiments.md` ranks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tracksdata as td

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: F401,E402  (chdir to src/ and set the path)
import edge_pp  # noqa: E402
import torch  # noqa: E402

import main_edge_postprocess as MEP  # noqa: E402
from params import EMBRYOS, LOG_FOLDER, SCALE, TRAIN_DIR  # noqa: E402
from tracking.post_process import build_graph  # noqa: E402
from tracking_cellmot.metrics import (  # noqa: E402
    evaluate,
    macro_summarise,
    node_recall,
    per_sample_metrics,
    summarise,
)

warnings.filterwarnings("ignore")
# `recover`'s drift estimate goes through torch, whose intra-op pool sizes
# itself from the core count and ignores OMP_NUM_THREADS -- with one worker per
# core that is a 3x oversubscription and the search runs slower than serial.
torch.set_num_threads(1)


# Coordinate descent visits these in order, walking each grid from where the
# starting config sits. Ordering is not cosmetic: the `recover_*` knobs move the
# node set and come before the `div_*` ones that read it, because a division is
# proposed from detections that survived pruning, so tuning the gates first
# tunes them against a graph that is about to change underneath them.
#
# Five knobs, down from twelve -- see `RETIRED` below for what went and why.
#
# Each list is the knob's step sizes, walked from the current value outwards, so
# it must stay sorted and its spacing is the resolution of the search: a value
# the grid steps over is a value this cannot reach. Its ends are not limits --
# the walk continues past them at the outermost spacing, up to `LIMITS` -- so a
# grid only has to cover where the answer usually is, not where it could be.
SEARCH = [
    # Deliberately two values each: the nine tuned configs put gap at 2 or 3
    # in eight of nine and the residual at 5.0 or 6.0 in six of nine, so the
    # product is four combinations and the search is exhaustive over it
    # rather than a walk that has to be trusted to have looked the right way.
    ("recover_max_gap", [2, 3]),
    ("recover_max_residual", [5.0, 6.0]),
    ("div_min_prob", [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]),
    ("div_diverge_um", [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]),
    ("div_symmetry_tau", [0.0, 0.3, 0.45, 0.6]),
]

# Dropped from the search on 2026-09-18, after tabulating the settings of the
# five best CV-ranked runs and the four fold-0-tuned ones (Experiments.md). 21
# of 32 parameters are identical across all nine, and four more are effectively
# so -- they differ in at most one config out of nine:
#
#   div_max_sister_um   14.0 everywhere but one run at 12.0
#   div_max_parent_um   11.0 everywhere but one run at 7.0
#   min_track           5 in seven of nine
#   recover_min_dup     3.0, and the fold-0-tuned runs that moved it went *down*
#                       to 2.0, which is outside the old grid anyway
#
# Sweeping a knob that nine independent tunings agree on spends the budget
# re-deriving a constant. The five that remain are the ones that genuinely
# move, and `div_min_prob` is the clearest: it splits by protocol rather than by
# checkpoint (0.20-0.35 tuned on both embryos, 0.30-0.375 tuned on fold 0 and
# measured held-out), which is the signature of a knob that matters and is easy
# to overfit.
#
# Restore any of them by moving the line back into SEARCH -- the machinery is
# unchanged and `--knobs` still names them individually.
RETIRED = [
    ("min_track", [3, 4, 5, 6, 7, 8, 9]),
    ("recover_min_dup", [2.0, 3.0, 4.0, 5.0, 6.0]),
    ("div_max_sister_um", [12.0, 14.0, 16.0]),
    ("div_max_parent_um", [7.0, 9.0, 11.0]),
    ("threshold", [0.35, 0.40, 0.45]),
    ("linefit", [0.0, 0.4, 0.6, 0.8]),
    ("vel_weight", [0.0, 0.02, 0.05]),
]

# Swept whole rather than walked -- see `sweep_knob` for why these and not the
# others. They are the division gates, which act as a conjunction and so read
# flat until they do not, which is exactly the shape a line search mistakes for
# a local optimum.
SWEEP = {"div_min_prob", "div_diverge_um", "div_symmetry_tau"}

# Swept as a product rather than one after the other. `recover_max_residual` is
# the drift-corrected motion allowed across a gap and `recover_max_gap` is how
# many frames that gap may span, so the pair describes one quantity: how far a
# cell is allowed to have moved while it was missing. A line search sees neither.
# It walks the gap at the *old* residual, where the longer gaps it tries are
# rejected because the residual will not admit the motion they imply, backs off,
# then walks the residual against the short gap it just settled on. Each knob
# looks locally flat while the pair that would have won is never scored.
#
# The nine tuned configs in Experiments.md bear that out: their (gap, residual)
# combinations are (2,4.0), (2,6.0), (2,6.0), (2,7.0), (3,5.0), (3,5.0), (3,5.0),
# (3,6.0), (4,6.0) -- the residual spans 4.0-7.0 at gap 2 alone, so there is no
# single residual at which walking the gap would find the right answer.
PAIRS = [("recover_max_gap", "recover_max_residual")]

# Only visited under `--tune_ilp`, and first when they are: the filter runs
# before everything else and every knob below is fitted to the node set it
# leaves behind, so moving it after them invalidates them. Walked, not swept --
# a solve is the expensive arm even cached, since each new value pays for one
# per stack.
ILP_SEARCH = [
    ("ilp_threshold", [0.2, 0.3, 0.4, 0.5, 0.6]),
    ("ilp_disappearance", [0.25, 0.5, 0.75, 1.0, 1.5, 3.0]),
    ("ilp_division", [0.5, 0.75, 1.0]),
    ("ilp_node_weight", [0.0, 0.5, 1.0]),
]

# How far a knob may be walked past the end of its grid, since the grid is a
# step size and not a boundary. These are the values that are meaningless rather
# than merely untried -- a probability outside (0, 1), a negative distance, a
# `min_track` of zero frames -- plus a far end loose enough that hitting it says
# the knob ran away rather than that the limit was chosen too tight. A knob
# absent here gets (0, inf).
LIMITS = {
    "threshold": (0.05, 0.95),
    "linefit": (0.0, 1.0),
    "vel_weight": (0.0, 0.5),
    "min_track": (1, 40),
    "recover_max_gap": (0, 20),
    "recover_max_residual": (0.5, 20.0),
    "recover_min_dup": (0.5, 20.0),
    "div_min_prob": (0.02, 0.95),
    "div_diverge_um": (0.0, 20.0),
    # Bounded by 2 by construction -- it is a ratio of two sister distances --
    # so anything at or above that is the gate turned off.
    "div_symmetry_tau": (0.0, 2.0),
    "div_max_sister_um": (2.0, 40.0),
    "div_max_parent_um": (2.0, 40.0),
    "ilp_threshold": (0.05, 0.95),
    "ilp_disappearance": (0.0, 10.0),
    "ilp_division": (0.0, 2.0),
    "ilp_node_weight": (0.0, 5.0),
}

# `recover_max_gap=0` is how the sweep expresses "turn the stage off": the flag
# `main_edge_postprocess.py` takes for that is `--no_recover`, and `recover=0`
# is what the cfg dict wants.
_OFF = {"recover_max_gap": ("recover", 0)}

_GT: dict[str, object] = {}
_PRED: dict[str, tuple] = {}
# `(stack, every ilp_* knob) -> the filtered graph`. See `_filtered`.
_ILP: dict[tuple, tuple] = {}

# What the solve depends on, and so what has to change before it is re-run.
# `ilp_div_forks` and `ilp_fork_division` are in here because the fork harvest
# is a second solve inside the same call.
_ILP_KEYS = ("ilp_threshold", "ilp_edge_weight", "ilp_appearance",
             "ilp_disappearance", "ilp_division", "ilp_node_weight",
             "ilp_keep_edges", "ilp_div_forks", "ilp_fork_division")

# `(stack, ilp keys, prefix keys) -> (coords, edges)` after assign/recover/
# through. See `_prefixed`.
_PREFIX: dict[tuple, tuple] = {}

# Everything `postprocess_prefix` reads. The division gates, `min_track` and
# `linefit` are deliberately absent: that they do not appear here is what makes
# the halfway point reusable across their arms.
_PREFIX_KEYS = ("assign", "threshold", "vel_weight", "vel_momentum",
                "recover", "recover_max_gap", "recover_max_residual",
                "recover_min_dup", "through", "through_max_gap")

# A worker holds this many prefixes at once, across all stacks, and two is
# nearly all of the benefit: `_job` runs every arm of a call on one stack before
# moving to the next, so the reuse being bought is within a call, where the arms
# of a suffix knob all share one prefix. Unbounded, a `threshold` or `recover_*`
# sweep would keep one graph per value tried per stack in every worker, and
# those are the configs that will never be asked for again.
_PREFIX_MAX = 2


def _resolve_cfg(cfg: dict) -> dict:
    cfg = dict(cfg)
    for key, (flag, when) in _OFF.items():
        if cfg.get(key) == when:
            cfg[flag] = 0
    return cfg


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _filtered(path, cfg, graph0):
    """The graph the rest of the chain runs on, with the ILP solved at most once.

    The filter is the first stage of the chain and reads none of the knobs below
    it, so its output is a function of the stack and the `ilp_*` settings alone
    -- at 9-16 s a stack against ~1 s for everything after it, re-solving it per
    arm is the whole cost of the search. Solve on first sight, keep the filtered
    arrays, and hand every later arm the same three arrays it would have
    rebuilt. Keyed by the settings rather than just the stack, so `--tune_ilp`
    still works: an ILP knob that moves is a new key and pays for one solve per
    stack, and moving back is free.

    Returns `(coords, edges, cand, forks)` and leaves `cfg` alone; `_ilp_forks`
    is part of what the solve produces, so it is cached with the rest rather
    than left in the cfg the caller passed in.
    """
    if not cfg.get("ilp"):
        return (*graph0, None)
    key = (path,) + tuple(cfg.get(k) for k in _ILP_KEYS)
    if key not in _ILP:
        probe = dict(cfg)
        coords, edges, cand = MEP.ilp_filter_nodes(
            *(a.copy() for a in graph0), probe)
        _ILP[key] = (coords, edges, cand, probe.get("_ilp_forks"))
    return _ILP[key]


def _prefixed(path, cfg, graph):
    """`(coords, edges)` after assign/recover/through, solved at most once.

    The same argument as `_filtered`, one stage lower down and worth more often:
    the prefix is ~70% of an arm (`recover`'s drift estimate and `assign`'s
    per-frame-pair solves), and every arm of a division gate, of `min_track` and
    of `linefit` -- which is most of the search, and all of the swept knobs --
    leaves it untouched.

    Only `coords` and `edges` are cached: `cand` is the one array the prefix
    does not rewrite once the ILP has had it, so it comes back from `_filtered`.
    """
    key = (path,) + tuple(cfg.get(k) for k in _ILP_KEYS + _PREFIX_KEYS)
    if key not in _PREFIX:
        coords, edges, _ = MEP.postprocess_prefix(
            graph[0].copy(), graph[1].copy(), graph[2], cfg)
        if len(_PREFIX) >= _PREFIX_MAX:
            # Oldest out. Python dicts keep insertion order, so this is the
            # first key inserted, which is the one least likely to be the config
            # the current knob's arms share.
            del _PREFIX[next(iter(_PREFIX))]
        _PREFIX[key] = (coords, edges)
    return _PREFIX[key]


def _job(arg):
    """Every arm, on one stack. The load and the GT parse are the expensive
    part and both are per-stack, so arms are batched inside the worker rather
    than the sweep being re-run per arm."""
    path, cfgs = arg
    sample = Path(path).stem
    if path not in _PRED:
        _PRED[path] = edge_pp.load_pred(path)
    coords0, edges0, cand0 = _PRED[path]
    if sample not in _GT:
        res = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{sample}.geff")
        _GT[sample] = res[0] if isinstance(res, tuple) else res
    gt = _GT[sample]
    n_total = MEP._n_total(sample)

    rows = []
    for cfg in cfgs:
        # `ilp` is cleared because the filter has already run: `postprocess`
        # would otherwise solve it again on the graph it just produced.
        filtered = _filtered(path, cfg, (coords0, edges0, cand0))
        cand, forks = filtered[2], filtered[3]
        if cfg.get("ilp"):
            cfg = {**cfg, "ilp": 0, "_ilp_forks": forks}
        # Copied on the way out: the suffix stages rewrite what they are given,
        # and the cache is handing every arm the same two arrays.
        pre = _prefixed(path, cfg, filtered[:3])
        coords, edges = MEP.postprocess_suffix(
            pre[0].copy(), pre[1].copy(), cand, cfg)
        graph = build_graph(np.round(np.asarray(coords)).astype(np.int64), [
            (int(s), int(t), float(p), float(d))
            for s, t, p, d in np.asarray(edges, float).reshape(-1, 4)])
        row = per_sample_metrics(
            # mark_gt: the sweep reads the counts, and the per-edge mask this
            # would write onto the GT graph is for viewers.
            evaluate(graph, gt, scale=SCALE, mark_gt=False),
            n_total, node_recall(graph, gt))
        one = summarise([row])
        row["division_jaccard"] = one["division_jaccard"]
        row["score"] = one["score"]
        row["sample_name"] = sample
        row["embryo"] = sample.split("_")[0]
        row["fold"] = EMBRYOS.index(row["embryo"]) if row["embryo"] in EMBRYOS else None
        rows.append(row)
    return rows


_POOL = None


def _pool(jobs):
    """One pool for the whole search, not one per call.

    Everything a worker caches -- the npz, the parsed GT, the ILP solve -- lives
    in that worker's globals, so a pool per `score_arms` threw all three away
    between knobs and re-read them on the next one. The search is dozens of
    calls over the same stacks, so the pool has to outlive them.
    """
    global _POOL
    if _POOL is None:
        _POOL = Pool(jobs)
    return _POOL


def _close_pool():
    global _POOL
    if _POOL is not None:
        _POOL.terminate()
        _POOL = None


def score_arms(paths, cfgs, jobs):
    """One summary per arm: macro metrics plus the per-embryo split."""
    per_cfg = [[] for _ in cfgs]
    for rows in _pool(jobs).imap_unordered(
        _job, [(str(p), [_resolve_cfg(c) for c in cfgs]) for p in paths]
    ):
        for i, row in enumerate(rows):
            per_cfg[i].append(row)
    out = []
    for rows in per_cfg:
        groups, macro = macro_summarise(rows)
        out.append({"macro": macro, "per_embryo": groups})
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

HEADER = (f"{'arm':<44}{'cv':>9}{'delta':>9}{'adj':>9}{'div':>8}{'rec':>8}"
          f"{'44b6':>9}{'6bba':>9}")


def _emb(summary, key):
    return summary["per_embryo"].get(key, {}).get("score", float("nan"))


def _line(name, summary, ref):
    m = summary["macro"]
    embryos = sorted(summary["per_embryo"])
    cells = "".join(f"{_emb(summary, e):>9.4f}" for e in embryos[:2])
    return (f"{name[:43]:<44}{m['score']:>9.4f}{m['score'] - ref:>+9.4f}"
            f"{m['adj_edge_jaccard']:>9.4f}{m['division_jaccard']:>8.4f}"
            f"{m['node_recall']:>8.4f}{cells}")


def _beats(summary, base, both, min_gain=0.0):
    """Does this arm win? Macro always; on both embryos too when `--both`.

    `min_gain` is what stops the walk from crawling along a flat grid: a knob
    the chain barely reads -- `div_min_prob` once the gates above it have
    already rejected everything -- otherwise scores a few float bits better at
    every step and gets followed to the end of its grid, one arm at a time, for
    a move that rounds to nothing.
    """
    if summary["macro"]["score"] <= base["macro"]["score"] + min_gain:
        return False
    if not both:
        return True
    return all(_emb(summary, e) >= _emb(base, e) - 1e-9
               for e in summary["per_embryo"])


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------

def _grid(values, current):
    """The knob's grid, with the current value in it.

    A config resumed from a previous retune can sit off the grid -- the last
    search may have run a different one, or the value may be hand-set. Snapping
    it to the nearest grid point would move the knob before anything has been
    measured, which is a silent regression of the starting config, so the value
    is inserted instead and the walk steps off it.
    """
    if current is None:
        return list(values)
    return sorted(set(values) | {current})


def sweep_knob(paths, cfg, best, knob, values, jobs, both, min_gain):
    """Sweep one knob's whole grid, then refine around the winner.

    The line search is a bet that a knob's score is single-peaked along its
    grid, so one step tells you which way the peak is. The division gates break
    that bet. They are a conjunction -- a fork has to pass all of them -- so
    loosening one while the others still reject changes nothing at all, and the
    knob reads flat right up until the step where it suddenly does not. That is
    a plateau with a cliff off the end of it, and a walk stops on the plateau.
    `DEFAULTS` records the shape from the other side: `div_symmetry_tau` is
    "effectively binary rather than a knob", 0 and 1.5 identical with the entire
    cost in the middle.

    So: score the whole grid at once -- one call, and arms are batched per
    stack, so the marginal arm is cheap next to the per-stack load the call pays
    anyway -- and then walk *half a step* either side of whichever value won.
    The sweep finds which basin, the refinement asks whether the grid is too
    coarse to sit at the bottom of it, which is the part of "these need more
    tuning" a fixed grid cannot answer.
    """
    cur = cfg.get(knob)
    grid = _grid(values, cur)
    ref = best["macro"]["score"]
    trials = [v for v in grid if v != cur]
    # A current value sitting at an end of the grid is only swept on one side,
    # so the knob would never be tried in the direction the grid does not cover.
    # One step out, on the same terms as the walk.
    i = grid.index(cur)
    for d in (-1, 1):
        if not 0 <= i + d < len(grid):
            trials.append(_next_along(grid, cur, d, knob))
    trials = [v for v in dict.fromkeys(trials) if v is not None]
    grid = sorted(set(grid) | set(trials))
    if not trials:
        return cfg, best, False
    results = score_arms(paths, [{**cfg, knob: v} for v in trials], jobs)
    for v, r in zip(trials, results):
        print(_line(f"  {knob}={v}", r, ref))

    wins = [(r, v) for v, r in zip(trials, results) if _beats(r, best, both, min_gain)]
    if not wins:
        return cfg, best, False
    best, v = max(wins, key=lambda w: w[0]["macro"]["score"])
    cfg = {**cfg, knob: v}

    # Refine around the winner. Inward: half a step either side, and only on a
    # continuous scale -- bisecting `min_track` asks for 5.5 frames, and integer
    # grids are already at their resolution. Outward: if the winner is the end
    # of the grid, the next step past it, because a sweep whose best value is
    # its own edge has not been shown a peak, only a boundary.
    i = grid.index(v)
    probes = []
    if all(isinstance(g, float) for g in grid):
        probes = [round((grid[j] + v) / 2, 6)
                  for j in (i - 1, i + 1) if 0 <= j < len(grid)]
    for d in (-1, 1):
        if not 0 <= i + d < len(grid):
            probes.append(_next_along(grid, v, d, knob))
    probes = [p for p in dict.fromkeys(probes) if p is not None and p != v]
    if probes:
        results = score_arms(paths, [{**cfg, knob: p} for p in probes], jobs)
        for p, r in zip(probes, results):
            print(_line(f"  {knob}={p}", r, ref))
        wins = [(r, p) for p, r in zip(probes, results)
                if _beats(r, best, both, min_gain)]
        if wins:
            best, p = max(wins, key=lambda w: w[0]["macro"]["score"])
            cfg = {**cfg, knob: p}
    return cfg, best, True


def _next_along(grid, cur, direction, knob):
    """The next value from `cur` going `direction`, or None at the limit.

    Inside the grid this is the adjacent grid point, because the grids are not
    all evenly spaced -- `linefit` goes 0.0, 0.4, 0.6, 0.8 -- and the spacing is
    a statement about where the interesting values are.

    Past the end it keeps going at the width of the outermost interval. The grid
    is the step size, not a boundary: a walk that is still improving when it
    runs out of grid has found that the grid was drawn too narrow, and stopping
    there reports the edge as an optimum. `LIMITS` is where it actually stops.
    """
    if cur in grid:
        i = grid.index(cur) + direction
        if 0 <= i < len(grid):
            return grid[i]
    if len(grid) < 2:
        return None
    spacing = abs(grid[-1] - grid[-2] if direction > 0 else grid[1] - grid[0])
    if not spacing:
        return None
    nxt = cur + direction * spacing
    # Integer knobs stay integers: `min_track` counts frames.
    nxt = int(nxt) if isinstance(cur, int) and float(spacing).is_integer() \
        else round(float(nxt), 6)
    lo, hi = LIMITS.get(knob, (0.0, float("inf")))
    return nxt if lo <= nxt <= hi else None


def walk_knob(paths, cfg, best, knob, values, jobs, both, min_gain):
    """Line-search one knob: probe both neighbours, then follow the winner.

    Returns `(cfg, best, moved)`. Costs two arms when the knob is already at a
    local optimum, which is the common case on a retune and the reason this is
    not a sweep. Each accepted step raises the bar for the next one -- the walk
    stops at the first step that does not improve on the step before it, not on
    where the knob started -- and it stops on that, not on running out of grid.
    """
    grid = _grid(values, cfg.get(knob))
    cur = cfg.get(knob)
    ref = best["macro"]["score"]

    trials = [(d, v) for d in (-1, 1)
              if (v := _next_along(grid, cur, d, knob)) is not None]
    if not trials:
        return cfg, best, False
    results = score_arms(paths, [{**cfg, knob: v} for _, v in trials], jobs)
    for (_, v), r in zip(trials, results):
        print(_line(f"  {knob}={v}", r, ref))

    wins = [(r, d, v) for (d, v), r in zip(trials, results)
            if _beats(r, best, both, min_gain)]
    if not wins:
        return cfg, best, False
    best, step, cur = max(wins, key=lambda w: w[0]["macro"]["score"])
    cfg = {**cfg, knob: cur}

    # Keep going the way that won: one step at a time, because a knob that wants
    # to move usually wants to move further than one step, and leaving that to
    # the next pass would pay for the neighbour arm a second time.
    while (nxt := _next_along(grid, cur, step, knob)) is not None:
        r = score_arms(paths, [{**cfg, knob: nxt}], jobs)[0]
        print(_line(f"  {knob}={nxt}", r, ref))
        if not _beats(r, best, both, min_gain):
            break
        cfg, best, cur = {**cfg, knob: nxt}, r, nxt
    return cfg, best, True


def sweep_pair(paths, cfg, best, pair, grids, jobs, both, min_gain):
    """Score the full product of two coupled knobs and keep the best combination.

    Same bet as `sweep_knob`, one dimension up, and for a stronger reason: these
    two describe one quantity -- how far a cell may have moved while it was
    missing -- so a walk tunes the gap at the old residual, rejects the longer
    gaps because the residual will not admit the motion they imply, then tunes
    the residual against the short gap it just settled on. The pair that would
    have won is never scored. The product is four arms, batched per stack, so it
    costs less than the two walks it replaces.
    """
    ka, kb = pair
    ga = _grid(grids[0], cfg.get(ka))
    gb = _grid(grids[1], cfg.get(kb))
    cur = (cfg.get(ka), cfg.get(kb))
    ref = best["macro"]["score"]

    trials = [(a, b) for a in ga for b in gb if (a, b) != cur]
    if not trials:
        return cfg, best, False
    results = score_arms(paths, [{**cfg, ka: a, kb: b} for a, b in trials], jobs)
    for (a, b), r in zip(trials, results):
        print(_line(f"  {ka}={a},{kb}={b}", r, ref))

    wins = [(r, ab) for ab, r in zip(trials, results) if _beats(r, best, both, min_gain)]
    if not wins:
        return cfg, best, False
    best, (a, b) = max(wins, key=lambda w: w[0]["macro"]["score"])
    cfg = {**cfg, ka: a, kb: b}

    # No refinement and no stepping outside the grid. Both grids are two values
    # wide and chosen to be the whole question -- a bisection to 5.5 or a probe
    # out to 7.0 would put back exactly the values this is meant to stop
    # testing. `sweep_knob` extends because its grids are step sizes; these are
    # an enumeration, so the product above is already exhaustive.
    return cfg, best, True


def _units(knobs):
    """`knobs` regrouped so a PAIR whose members are both present is one unit.

    A pair only fires when both members survived `--knobs` / `--no_div`; asking
    for one of them alone still walks it, rather than silently dragging its
    partner back into a search the caller excluded it from.
    """
    have = {k for k, _ in knobs}
    paired = {k for pair in PAIRS for k in pair if set(pair) <= have}
    grids = dict(knobs)
    out, done = [], set()
    for k, v in knobs:
        if k in done:
            continue
        pair = next((p for p in PAIRS if k in p and set(p) <= have), None)
        if pair and k in paired:
            out.append((pair, [grids[p] for p in pair]))
            done.update(pair)
        else:
            out.append(((k,), [v]))
            done.add(k)
    return out


def coordinate_descent(paths, base_cfg, jobs, passes, both, knobs, min_gain):
    cfg = dict(base_cfg)
    best = score_arms(paths, [cfg], jobs)[0]
    print(f"\nstart  cv={best['macro']['score']:.4f}\n")

    for p in range(passes):
        moved = False
        for unit, grids in _units(knobs):
            was = {k: cfg.get(k) for k in unit}
            if len(unit) > 1:
                mode, label = "pair", " + ".join(unit)
            else:
                mode = "sweep" if unit[0] in SWEEP else "walk"
                label = unit[0]
            print(f"pass {p + 1}  {label}  "
                  f"(current {', '.join(str(v) for v in was.values())}, {mode})")
            print(HEADER)
            print(_line("  keep " + ",".join(str(v) for v in was.values()),
                        best, best["macro"]["score"]))
            if len(unit) > 1:
                cfg, best, won = sweep_pair(
                    paths, cfg, best, unit, grids, jobs, both, min_gain)
            else:
                search = sweep_knob if unit[0] in SWEEP else walk_knob
                cfg, best, won = search(
                    paths, cfg, best, unit[0], grids[0], jobs, both, min_gain)
            if won:
                moved = True
                print("  -> " + ", ".join(f"{k}: {was[k]} -> {cfg[k]}" for k in unit)
                      + f"  cv -> {best['macro']['score']:.4f}")
            print()
        if not moved:
            print(f"pass {p + 1}: nothing moved, stopping\n")
            break
    return cfg, best


def _best_scored(run_dir: Path):
    """The best-scoring `scores_*.json` in a run folder: `(path, score, json)`.

    `main_edge_postprocess.py` writes one of these beside every output folder it
    produces, each carrying the full `settings` it ran with, so the run folder
    already records its own best strategy and there is no separate state to
    keep. Score files from an unpostprocessed `preds_*` are skipped: they have
    no `settings` and are not a strategy to resume from.
    """
    best = None
    for path in sorted(run_dir.glob("scores_*.json")):
        try:
            blob = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(blob.get("settings"), dict):
            continue
        score = blob.get("cv_score", blob.get("macro", {}).get("score"))
        if score is None:
            continue
        if best is None or score > best[1]:
            best = (path, score, blob)
    return best


def _resolve_start(folder: Path):
    """`(preds_dir, base_cfg, note)` for a candidate folder or a run folder.

    The starting config is **always** `MEP.DEFAULTS`.

    It used to resume from the best-scoring `scores_*.json` in the run folder,
    which was cheaper -- a knob already right cost two arms instead of a walk --
    but it compounds: each retune starts where the last one's fit landed and can
    only move from there, so whatever the previous search took from the noise is
    inherited rather than re-derived, and a value that won once on 199 stacks is
    never asked the question again. With the search down to five knobs that
    saving is not worth buying overfitting with.

    A run folder is one with no npz of its own; the best score file is still
    read, but only to name the candidate folder it was measured on, so
    `--preds_dir <run>` still resolves without saying which `preds_*` to use.
    """
    base = dict(MEP.DEFAULTS)
    is_preds = any(folder.glob("*.npz"))
    run_dir = folder if not is_preds else folder.parent
    note = "from DEFAULTS"
    if not is_preds:
        scored = _best_scored(run_dir)
        if scored is not None:
            path, score, blob = scored
            folder = Path(blob.get("preds_dir") or MEP._best_preds_dir(run_dir)[0])
            note += f"; candidates {folder.name} (from {path.name}, cv {score:.4f})"
        else:
            folder, why = MEP._best_preds_dir(run_dir)
            note += f"; candidates {folder.name} ({why})"
    return folder, base, note


def _parse_arms(spec):
    """`a=1;b=2|a=3` -> [{'a': 1, 'b': 2}, {'a': 3}]. An empty arm is the base."""
    arms = []
    for arm in spec.split("|"):
        kw = {}
        for part in arm.split(";"):
            if part.strip():
                k, v = part.split("=", 1)
                kw[k.strip()] = json.loads(v.strip())
        arms.append(kw)
    return arms


# What the ILP is re-enabled at for the confirmation run. Not tuned and not
# meant to be: these are the settings the 2026-09-16 retune landed on, and the
# point of the second run is that the filter has repeatedly been worth turning
# on over knobs tuned without it -- see the DEFAULTS block in
# `main_edge_postprocess.py`. Leaving it off during the search keeps the arms at
# ~1 s a stack instead of 9-16, and putting it back afterwards has behaved like
# a good enough surrogate for having searched behind it.
ILP_AFTER = {"ilp": 1, "ilp_disappearance": 0.75, "ilp_threshold": 0.4}


def _launch(preds_dir, cfg, ilp_on, fold):
    """Run `main_edge_postprocess.py` on the winning config, over every stack.

    The search ran on one fold; this is the run that produces the artefacts and
    the two-embryo score, so it is deliberately not limited to the fold that was
    tuned on -- the held-out embryo is the point.

    **Both arms always run**, whichever way the search went: `<tag>_tuned`
    without the filter and `<tag>_tuned_ilp` with it. Which of the two wins is a
    measurement, not a default -- the filter and the `recover`/`min_track` block
    do overlapping work, and which way that falls has reversed between
    checkpoints -- so the answer should be on disk either way rather than only
    when the search happened to run without it.

    That also makes the two names mean one thing forever: `_tuned` is the chain
    without the ILP and `_tuned_ilp` is the chain with it, regardless of how the
    search was invoked. They used to mean "as searched" and "the other one",
    which is how a file called `..._f0_tuned_ilp` came to hold a run with
    `settings.ilp = 0`.

    The ILP arm keeps whatever `ilp_*` values the search settled on when it ran
    behind the solver (`--tune_ilp`), and falls back to `ILP_AFTER` only when the
    search never looked at them -- otherwise turning the filter back on would
    also throw away the tuning of its own knobs.

    The fold leads the name -- `preds_0.45_pp_f0_tuned` -- because it qualifies
    the score rather than the chain: these knobs saw one embryo, and the folder
    should say which, since the run then scores on both. It leads rather than
    trails because `_versioned` appends its index to the end of the name, and
    `..._tuned_f0` re-run would read as `..._tuned_f02`, which looks like a fold
    and is not one.

    Nothing is passed to make it overwrite, so a second run of the same config
    lands beside the first rather than on top of it; `_versioned` moves the npz
    folder, the geff folder and the scores json together.
    """
    tag = f"f{fold}" if fold != "all" else "fall"
    ilp_cfg = {**cfg, "ilp": 1} if ilp_on else {**cfg, **ILP_AFTER}
    runs = [(f"{tag}_tuned", {**cfg, "ilp": 0}),
            (f"{tag}_tuned_ilp", ilp_cfg)]

    for name, c in runs:
        cmd = [sys.executable, "main_edge_postprocess.py",
               "--preds_dir", str(preds_dir), "--name", name]
        cmd += _flags(c, MEP.DEFAULTS).split()
        print("\n" + "=" * 78)
        print("-> " + " ".join(cmd[1:]), flush=True)
        # check=False: one arm's artefacts are worth keeping even if the other
        # dies, and the ILP arm is the more likely to -- it needs a solver the
        # plain chain does not.
        rc = subprocess.run(cmd).returncode
        if rc:
            print(f"-> {name}: exited {rc}", flush=True)


def _flags(cfg, defaults):
    """The `main_edge_postprocess.py` command line for a config."""
    out = []
    for k, v in cfg.items():
        if k not in defaults:
            continue
        # `ilp` is emitted whatever the default says. It is the one setting the
        # search deliberately differs from `DEFAULTS` on -- off while searching,
        # back on for the second arm -- so "same as default, omit" is exactly
        # wrong for it. It was `store_true` on the other side while the default
        # was 0, so once the default became 1 the `_ilp` arm stopped being given
        # the flag and silently reran the plain chain under a name saying it had
        # not: `2026-09-14/1/scores_preds_0.45_pp_f0_tuned_ilp.json` is
        # byte-identical to its non-ILP sibling and records `settings.ilp = 0`.
        if k == "ilp":
            out.append("--ilp" if v else "--no-ilp")
            continue
        if defaults[k] == v:
            continue
        if k == "recover" and v == 0:
            out.append("--no_recover")
        elif k == "divisions" and v == 0:
            out.append("--no_divisions")
        elif k in ("recover", "divisions"):
            continue
        else:
            out.append(f"--{k} {v:g}" if isinstance(v, float) else f"--{k} {v}")
    return " ".join(o for o in out if o)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds_dir", default=None,
                   help="the npz main_edge_predict.py wrote with "
                        "--dump_candidates, or the run folder above it; "
                        "default = the best-scoring preds_* "
                        "of the latest run under logs/edge_run/")
    p.add_argument("--ilp", action="store_true",
                   help="run the ILP node filter, once per stack, cached for "
                        "the rest of the search. Off by default whatever the "
                        "starting config says: a solve is 9-16 s a stack "
                        "against ~1 s for the rest of the chain, so the first "
                        "pass over the stacks is slow even cached")
    p.add_argument("--tune_ilp", action="store_true",
                   help="also tune the ILP's own knobs (implies --ilp). Each "
                        "new value costs a solve per stack, so this is the "
                        "expensive search, not the default one")
    p.add_argument("--base", default="",
                   help="'k=v;k2=v2' starting point, applied last")
    p.add_argument("--sweep", default="",
                   help="'knob=v1,v2,v3' -- sweep one knob and stop")
    p.add_argument("--arms", default="",
                   help="'k=v;k2=v2|k=v3' -- score these arms and stop")
    p.add_argument("--passes", type=int, default=2,
                   help="coordinate-descent passes (default %(default)s)")
    p.add_argument("--min_gain", type=float, default=1e-4,
                   help="cv a move must gain to be taken (default %(default)s, "
                        "the last digit the report prints)")
    p.add_argument("--both", action="store_true",
                   help="only accept a move that does not lose on either "
                        "embryo -- the bar README_edge.md sets")
    p.add_argument("--no_div", "--no-div", action="store_true",
                   help="leave the division gates where they are. They are the "
                        "swept knobs, so this is also the cheap search")
    p.add_argument("--knobs", default=None,
                   help="comma-separated subset of the SEARCH knobs to visit")
    p.add_argument("--fold", default="0",
                   help="tune on one fold's embryo (default %(default)s = "
                        f"{EMBRYOS[0]}); 'all' for every stack. Tuning on all "
                        "of them fits the knobs to the same stacks the result "
                        "is read off, and the knobs are worth thousandths")
    p.add_argument("--jobs", type=int, default=24)
    p.add_argument("--limit", type=int, default=None,
                   help="cap the stacks. For a smoke test only: the knobs are "
                        "worth thousandths and a subset cannot resolve that")
    p.add_argument("--out", default=None, help="write the winning config as json")
    p.add_argument("--no_run", "--no-run", action="store_true",
                   help="print the main_edge_postprocess.py command for the "
                        "winning config instead of running it. By default the "
                        "search ends by running it over every stack, plus a "
                        "second run with the ILP re-enabled when the search "
                        "had it off")
    a = p.parse_args()

    if a.preds_dir:
        folder = Path(a.preds_dir)
    else:
        folder, why = MEP._best_preds_dir(Path(LOG_FOLDER) / "edge_run")
        print(f"-> no --preds_dir given, using {folder}  ({why})")

    preds_dir, base, note = _resolve_start(folder)
    print(f"-> start  : {note}")
    base.update(out_dir="", geff_dir="", no_score=False, assign=1)
    ilp = a.ilp or a.tune_ilp
    if not ilp and base.get("ilp"):
        print("-> ilp    : on in the starting config, forced off "
              "(--ilp to run it, cached; --tune_ilp to tune it)")
    elif ilp:
        print(f"-> ilp    : on, solved once per stack and cached"
              f"{', and tuned' if a.tune_ilp else ''}")
    base["ilp"] = 1 if ilp else 0
    if a.base:
        base.update(_parse_arms(a.base)[0])

    paths = sorted(preds_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"No npz files under {preds_dir}")

    # One fold by default. Every knob here is worth thousandths, and fitting
    # them on the same stacks the number is then read off spends exactly that
    # much on nothing -- the held-out embryo is what says whether a move was a
    # tuning gain or a fitting one. Scoring the winner on `--fold all` at the
    # end is the check; tuning there is not.
    if a.fold != "all":
        try:
            embryo = EMBRYOS[int(a.fold)]
        except (ValueError, IndexError):
            raise SystemExit(f"--fold must be 'all' or one of "
                             f"{list(range(len(EMBRYOS)))}, got {a.fold!r}")
        held = [p for p in paths if p.stem.split("_")[0] != embryo]
        paths = [p for p in paths if p.stem.split("_")[0] == embryo]
        if not paths:
            found = sorted({p.stem.split("_")[0] for p in held})
            raise SystemExit(f"no {embryo} stacks under {preds_dir} "
                             f"(found {', '.join(found) or 'nothing'})")
        print(f"-> fold     : {a.fold} = {embryo}, {len(paths)} stacks "
              f"({len(held)} held out -- the cv below is one embryo, so it is "
              f"not the two-embryo number Experiments.md ranks)")

    paths = paths[: a.limit]
    print(f"-> {len(paths)} stacks, {a.jobs} workers")

    if a.sweep or a.arms:
        if a.sweep:
            knob, values = a.sweep.split("=", 1)
            arms = [{knob.strip(): json.loads(v)} for v in values.split(",")]
        else:
            arms = _parse_arms(a.arms)
        cfgs = [{**base, **kw} for kw in arms]
        results = score_arms(paths, cfgs, a.jobs)
        ref = results[0]["macro"]["score"]
        print(f"\n{HEADER}")
        for kw, r in zip(arms, results):
            print(_line(",".join(f"{k}={v}" for k, v in kw.items()) or "(base)", r, ref))
        return

    # ILP first: it decides which nodes exist, and every knob after it is fitted
    # to what it leaves behind.
    knobs = (ILP_SEARCH if a.tune_ilp else []) + SEARCH
    if a.knobs:
        want = [k.strip() for k in a.knobs.split(",")]
        # `--knobs` may name a RETIRED one. The default search skips them because
        # nine independent tunings agree on their values, but "this checkpoint is
        # different" is exactly the case worth being able to test without editing
        # the file -- so naming one explicitly puts it back in, in SEARCH order
        # followed by whatever retired knobs were asked for.
        available = knobs + [kv for kv in RETIRED if kv[0] not in {k for k, _ in knobs}]
        unknown = set(want) - {k for k, _ in available}
        if unknown:
            raise SystemExit(f"unknown knob(s): {', '.join(sorted(unknown))}"
                             + ("" if a.tune_ilp else "  (--tune_ilp for the "
                                                      "ilp_* knobs)"))
        knobs = [(k, v) for k, v in available if k in want]

    if a.no_div:
        # By prefix rather than by a list, so a gate added to SEARCH later is
        # covered without having to remember this flag exists.
        knobs = [(k, v) for k, v in knobs if not k.startswith("div_")]
        if not knobs:
            raise SystemExit("--no_div leaves nothing to tune")

    cfg, best = coordinate_descent(
        paths, base, a.jobs, a.passes, a.both, knobs, a.min_gain)

    print("=" * len(HEADER))
    print(f"best cv = {best['macro']['score']:.4f}   "
          f"adj={best['macro']['adj_edge_jaccard']:.4f}  "
          f"div={best['macro']['division_jaccard']:.4f}  "
          f"recall={best['macro']['node_recall']:.4f}")
    for emb in sorted(best["per_embryo"]):
        print(f"  {emb}: {_emb(best, emb):.4f}")
    changed = {k: v for k, v in cfg.items()
               if k in MEP.DEFAULTS and MEP.DEFAULTS[k] != v}
    print("\nchanged vs DEFAULTS: "
          + (", ".join(f"{k}: {MEP.DEFAULTS[k]} -> {v}" for k, v in changed.items())
             or "nothing"))
    if a.out:
        Path(a.out).write_text(json.dumps(
            {k: v for k, v in cfg.items()
             if k not in ("out_dir", "geff_dir", "no_score")}, indent=4))
        print(f"-> config  : {a.out}")

    tag = f"f{a.fold}" if a.fold != "all" else "fall"
    if a.no_run:
        print(f"\npython main_edge_postprocess.py --preds_dir {preds_dir} "
              f"--name {tag}_tuned {_flags(cfg, MEP.DEFAULTS)}".rstrip())
        if not ilp:
            print(f"python main_edge_postprocess.py --preds_dir {preds_dir} "
                  f"--name {tag}_tuned_ilp "
                  f"{_flags({**cfg, **ILP_AFTER}, MEP.DEFAULTS)}".rstrip())
        return

    # The pool is idle from here and the run below wants the cores.
    _close_pool()
    _launch(preds_dir, cfg, ilp, a.fold)


if __name__ == "__main__":
    main()
