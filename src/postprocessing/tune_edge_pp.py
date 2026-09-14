"""Search `main_edge_postprocess.py`'s knobs on an edge model's candidate dump.

    cd src/postprocessing
    python tune_edge_pp.py                                   # latest edge run
    python tune_edge_pp.py --preds_dir ../logs/edge_run/2026-09-11/0/preds_0.45
    python tune_edge_pp.py --sweep "min_track=4,5,6,7"       # one knob, explicit
    python tune_edge_pp.py --arms "|min_track=5;div_min_prob=0.3"   # free arms

`common` chdirs to `src/`, so a relative `--preds_dir` is relative to `src/` --
`../logs/...` -- exactly as `main_edge_postprocess.py` takes it, and not to the
`src/postprocessing/` you are standing in.

With no `--sweep`/`--arms` it runs the built-in coordinate descent: the knobs in
`SEARCH` in order, each swept around the current best, the winner kept, then a
second pass so a knob can move again once the others have. That is how the
`DEFAULTS` block in `main_edge_postprocess.py` was tuned by hand, made
repeatable -- retuning is needed on every new checkpoint, because these knobs
trade against how much the detector over-detects and that changes run to run.

It scores exactly what `main_edge_postprocess.py` scores -- same `postprocess`
chain, same `evaluate`, same macro roll-up -- but writes no npz and no geff, so
an arm costs the scoring and nothing else. Once the search lands, re-run
`main_edge_postprocess.py` with the printed flags to get the artefacts.

The reported number is the competition score, macro-averaged: one vote per
embryo, `adj_edge_jaccard + 0.1 * division_jaccard`. The per-embryo columns are
there because that is the bar `README_edge.md` sets and the macro mean hides
it: a knob that buys 0.006 on 44b6 and sells 0.001 on 6bba shows up as a gain,
and on two embryos that is one embryo's worth of evidence, not two. `--both`
makes the rule binding rather than advisory.
"""

from __future__ import annotations

import argparse
import json
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


# Coordinate descent visits these in order. Ordering is not cosmetic: the ones
# that move the node set (`recover`, `min_track`) come before the ones that read
# it (`divisions`), because a division is proposed from detections that survived
# pruning, so tuning the gates first tunes them against a graph that is about to
# change underneath them.
#
# `threshold`, `linefit` and `vel_weight` are last and deliberately narrow. Each
# retune so far has left all three where they are, and each is a stage-level
# switch rather than a knob -- turning `linefit` off costs ~0.010 and no
# intermediate value has ever won -- so they are here to catch a checkpoint that
# breaks the pattern, not because they are expected to move.
SEARCH = [
    ("min_track", [3, 4, 5, 6, 7, 8, 9]),
    ("recover_max_gap", [0, 1, 2, 3, 4, 5]),
    ("recover_max_residual", [3.0, 4.0, 5.0, 6.0]),
    ("recover_min_dup", [3.0, 4.0, 5.0, 6.0]),
    ("div_min_prob", [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]),
    ("div_diverge_um", [2.5, 3.0, 3.5, 4.0, 4.5]),
    ("div_symmetry_tau", [0.0, 0.3, 0.6]),
    ("div_max_sister_um", [12.0, 14.0, 16.0]),
    ("div_max_parent_um", [7.0, 9.0, 11.0]),
    ("threshold", [0.35, 0.40, 0.45]),
    ("linefit", [0.0, 0.4, 0.6, 0.8]),
    ("vel_weight", [0.0, 0.02, 0.05]),
]

# `recover_max_gap=0` is how the sweep expresses "turn the stage off": the flag
# `main_edge_postprocess.py` takes for that is `--no_recover`, and `recover=0`
# is what the cfg dict wants.
_OFF = {"recover_max_gap": ("recover", 0)}

_GT: dict[str, object] = {}
_PRED: dict[str, tuple] = {}


def _resolve_cfg(cfg: dict) -> dict:
    cfg = dict(cfg)
    for key, (flag, when) in _OFF.items():
        if cfg.get(key) == when:
            cfg[flag] = 0
    return cfg


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _job(arg):
    """Every arm, on one stack. The load and the GT parse are the expensive
    part and both are per-stack, so arms are batched inside the worker rather
    than the sweep being re-run per arm."""
    path, cfgs = arg
    sample = Path(path).stem
    if path not in _PRED:
        _PRED[path] = edge_pp.load_pred(path)
    coords0, edges0, cand = _PRED[path]
    if sample not in _GT:
        res = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{sample}.geff")
        _GT[sample] = res[0] if isinstance(res, tuple) else res
    gt = _GT[sample]
    n_total = MEP._n_total(sample)

    rows = []
    for cfg in cfgs:
        coords, edges = MEP.postprocess(
            coords0.copy(), edges0.copy(), cand.copy(), cfg)
        graph = build_graph(np.round(np.asarray(coords)).astype(np.int64), [
            (int(s), int(t), float(p), float(d))
            for s, t, p, d in np.asarray(edges, float).reshape(-1, 4)])
        row = per_sample_metrics(
            evaluate(graph, gt, scale=SCALE), n_total, node_recall(graph, gt))
        one = summarise([row])
        row["division_jaccard"] = one["division_jaccard"]
        row["score"] = one["score"]
        row["sample_name"] = sample
        row["embryo"] = sample.split("_")[0]
        row["fold"] = EMBRYOS.index(row["embryo"]) if row["embryo"] in EMBRYOS else None
        rows.append(row)
    return rows


def score_arms(paths, cfgs, jobs):
    """One summary per arm: macro metrics plus the per-embryo split."""
    per_cfg = [[] for _ in cfgs]
    with Pool(jobs) as pool:
        for rows in pool.imap_unordered(
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


def _beats(summary, base, both):
    """Does this arm win? Macro always; on both embryos too when `--both`."""
    if summary["macro"]["score"] <= base["macro"]["score"]:
        return False
    if not both:
        return True
    return all(_emb(summary, e) >= _emb(base, e) - 1e-9
               for e in summary["per_embryo"])


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------

def coordinate_descent(paths, base_cfg, jobs, passes, both, knobs):
    cfg = dict(base_cfg)
    best = score_arms(paths, [cfg], jobs)[0]
    print(f"\nstart  cv={best['macro']['score']:.4f}\n")

    for p in range(passes):
        moved = False
        for knob, values in knobs:
            trials = [v for v in values if v != cfg.get(knob)]
            if not trials:
                continue
            arms = []
            for v in trials:
                c = dict(cfg)
                c[knob] = v
                arms.append(c)
            print(f"pass {p + 1}  {knob}  (current {cfg.get(knob)})")
            print(HEADER)
            print(_line(f"  keep {cfg.get(knob)}", best, best["macro"]["score"]))
            results = score_arms(paths, arms, jobs)
            win, win_v = None, None
            for v, r in zip(trials, results):
                mark = ""
                if _beats(r, best, both) and (
                        win is None or r["macro"]["score"] > win["macro"]["score"]):
                    win, win_v, mark = r, v, ""
                print(_line(f"  {knob}={v}", r, best["macro"]["score"]) + mark)
            if win is not None:
                print(f"  -> {knob}: {cfg.get(knob)} -> {win_v}  "
                      f"cv {best['macro']['score']:.4f} -> {win['macro']['score']:.4f}")
                cfg[knob], best, moved = win_v, win, True
            print()
        if not moved:
            print(f"pass {p + 1}: nothing moved, stopping\n")
            break
    return cfg, best


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


def _flags(cfg, defaults):
    """The `main_edge_postprocess.py` command line for a config."""
    out = []
    for k, v in cfg.items():
        if k not in defaults or defaults[k] == v:
            continue
        if k == "recover" and v == 0:
            out.append("--no_recover")
        elif k == "divisions" and v == 0:
            out.append("--no_divisions")
        elif k in ("recover", "divisions", "ilp"):
            continue
        else:
            out.append(f"--{k} {v:g}" if isinstance(v, float) else f"--{k} {v}")
    return " ".join(out)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds_dir", default=None,
                   help="the npz main_edge_predict.py wrote with "
                        "--dump_candidates; default = the best-scoring preds_* "
                        "of the latest run under logs/edge_run/")
    p.add_argument("--base", default="",
                   help="'k=v;k2=v2' starting point, on top of "
                        "main_edge_postprocess.DEFAULTS")
    p.add_argument("--sweep", default="",
                   help="'knob=v1,v2,v3' -- sweep one knob and stop")
    p.add_argument("--arms", default="",
                   help="'k=v;k2=v2|k=v3' -- score these arms and stop")
    p.add_argument("--passes", type=int, default=2,
                   help="coordinate-descent passes (default %(default)s)")
    p.add_argument("--both", action="store_true",
                   help="only accept a move that does not lose on either "
                        "embryo -- the bar README_edge.md sets")
    p.add_argument("--knobs", default=None,
                   help="comma-separated subset of the SEARCH knobs to visit")
    p.add_argument("--jobs", type=int, default=24)
    p.add_argument("--limit", type=int, default=None,
                   help="cap the stacks. For a smoke test only: the knobs are "
                        "worth thousandths and a subset cannot resolve that")
    p.add_argument("--out", default=None, help="write the winning config as json")
    a = p.parse_args()

    if a.preds_dir:
        preds_dir = Path(a.preds_dir)
    else:
        preds_dir, why = MEP._best_preds_dir(Path(LOG_FOLDER) / "edge_run")
        print(f"-> no --preds_dir given, using {preds_dir}  ({why})")

    base = dict(MEP.DEFAULTS)
    base.update(out_dir="", geff_dir="", no_score=False, assign=1)
    if a.base:
        base.update(_parse_arms(a.base)[0])

    paths = sorted(preds_dir.glob("*.npz"))[: a.limit]
    if not paths:
        raise SystemExit(f"No npz files under {preds_dir}")
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

    knobs = SEARCH
    if a.knobs:
        want = [k.strip() for k in a.knobs.split(",")]
        knobs = [(k, v) for k, v in SEARCH if k in want]
        unknown = set(want) - {k for k, _ in SEARCH}
        if unknown:
            raise SystemExit(f"unknown knob(s): {', '.join(sorted(unknown))}")

    cfg, best = coordinate_descent(paths, base, a.jobs, a.passes, a.both, knobs)

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
    print(f"\npython main_edge_postprocess.py --preds_dir {preds_dir} "
          f"{_flags(cfg, MEP.DEFAULTS)}".rstrip())

    if a.out:
        Path(a.out).write_text(json.dumps(
            {k: v for k, v in cfg.items()
             if k not in ("out_dir", "geff_dir", "no_score")}, indent=4))
        print(f"-> config  : {a.out}")


if __name__ == "__main__":
    main()
