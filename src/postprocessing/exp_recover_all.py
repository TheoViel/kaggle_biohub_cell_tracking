"""The gap-recovery stage over every validation stack, against the gate.

One sample is not enough to set ``recover_max_residual``: the ground truth is
~280 annotated nodes per stack, so a single stack moves in steps of one link.

Run:  OMP_NUM_THREADS=1 python src/postprocessing/exp_recover_all.py --jobs 24
"""

import argparse
from pathlib import Path

import sweep
from common import ROOT
from stages import run_pipeline, run_pp_public
from sweep import add_common_args, stack_names

DEFAULT_PRED_DIR = ROOT / "logs" / "2026-08-26" / "4" / "preds_0.25"
PP_BEST = dict(filter_short_tracks=1, min_track_len=4, relink_tight_um=6.5)


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--pred_dir", default=str(DEFAULT_PRED_DIR))
    ap.add_argument("--grid", type=float, nargs="*", default=[6.0],
                    help="max_residual values")
    ap.add_argument("--gaps", type=int, nargs="*", default=[1],
                    help="max_gap values — the longest run of missing frames filled")
    ap.add_argument("--passes", type=int, nargs="*", default=[1],
                    help="rounds of recover + relink")
    ap.add_argument("--with_pp", action="store_true",
                    help="also layer pp_public(tuned) on both sides")
    args = ap.parse_args()

    sweep.PRED_DIR = Path(args.pred_dir)
    names = stack_names(args.limit, args.spread)
    print(f"{len(names)} stacks from {sweep.PRED_DIR}\n")

    settings = {
        f"recover r={r} gap<={g} x{p}":
            dict(recover=1, recover_max_residual=r, recover_max_gap=g, recover_passes=p)
        for r in args.grid for g in args.gaps for p in args.passes
    }

    recipes = {"raw": [], "pipeline": [(run_pipeline, dict(recover=0))]}
    for label, kw in settings.items():
        recipes[f"pipeline+{label}"] = [(run_pipeline, kw)]

    if args.with_pp:
        recipes["pipeline -> pp"] = [
            (run_pipeline, dict(recover=0)), (run_pp_public, PP_BEST)]
        for label, kw in settings.items():
            recipes[f"pipeline+{label} -> pp"] = [
                (run_pipeline, kw), (run_pp_public, PP_BEST)]

    table, results = sweep.run(recipes, names, jobs=args.jobs, tag="recover")

    base = "pipeline"
    print(f"\nagainst {base}:")
    for label in recipes:
        if label in ("raw", base):
            continue
        delta = [results[s][label]["adj_edge_jaccard"]
                 - results[s][base]["adj_edge_jaccard"]
                 for s in results if label in results[s] and base in results[s]]
        better = sum(1 for d in delta if d > 0.005)
        worse = sum(1 for d in delta if d < -0.005)
        row = next(r for r in table if r["config"] == label)
        ref = next(r for r in table if r["config"] == base)
        print(f"  {label:<28s} {row['adj_edge_jaccard'] - ref['adj_edge_jaccard']:+.4f}  "
              f"better {better:>2d}  worse {worse:>2d}")


if __name__ == "__main__":
    main()
