"""Does restoring the detector's spacing floor pay? All 71 stacks.

Two places break it (see suppress.py): ``recover`` inserts nodes closer to an
existing detection than the detector's own NMS ever would, and ``pp_public``
smooths existing ones on top of each other. So both the fix and its cause are
tested here: the ``recover_min_dup`` that governs the first, and a suppression
pass after the second.

Run:  OMP_NUM_THREADS=1 python src/postprocessing/exp_suppress.py --jobs 24
"""

import argparse
from pathlib import Path

import sweep
from common import ROOT
from stages import run_pipeline, run_pp_public, run_suppress
from sweep import add_common_args, stack_names

DEFAULT_PRED_DIR = ROOT / "logs" / "2026-08-20" / "20" / "preds_0.20"
PP_BEST = dict(filter_short_tracks=1, min_track_len=4, relink_tight_um=6.5)


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--pred_dir", default=str(DEFAULT_PRED_DIR))
    ap.add_argument("--radius", type=float, nargs="*", default=[3.0, 5.0, 7.0])
    ap.add_argument("--min_dup", type=float, nargs="*", default=[7.0])
    ap.add_argument("--centroid", action="store_true",
                    help="also try averaging the merged positions")
    args = ap.parse_args()

    sweep.PRED_DIR = Path(args.pred_dir)
    names = stack_names(args.limit, args.spread)
    print(f"{len(names)} stacks from {sweep.PRED_DIR}\n")

    base = [(run_pipeline, {}), (run_pp_public, PP_BEST)]
    recipes = {
        "pipeline -> pp": base,
    }
    # 1. suppression after pp_public, at several radii
    for r in args.radius:
        recipes[f"pipeline -> pp -> supp {r:g}"] = base + [(run_suppress, dict(radius=r))]
        if args.centroid:
            recipes[f"pipeline -> pp -> supp {r:g} cen"] = base + [
                (run_suppress, dict(radius=r, position="centroid"))]
    # 2. stop `recover` breaking the floor in the first place
    for m in args.min_dup:
        kw = dict(recover_min_dup=m)
        recipes[f"pipeline(dup {m:g}) -> pp"] = [(run_pipeline, kw), (run_pp_public, PP_BEST)]
        for r in args.radius:
            recipes[f"pipeline(dup {m:g}) -> pp -> supp {r:g}"] = [
                (run_pipeline, kw), (run_pp_public, PP_BEST), (run_suppress, dict(radius=r))]

    table, results = sweep.run(recipes, names, jobs=args.jobs, tag="suppress")

    ref = "pipeline -> pp"
    base_row = next(r for r in table if r["config"] == ref)
    print(f"\nagainst {ref} ({base_row['adj_edge_jaccard']:.4f}):")
    for label in recipes:
        if label == ref:
            continue
        delta = [results[s][label]["adj_edge_jaccard"] - results[s][ref]["adj_edge_jaccard"]
                 for s in results if label in results[s] and ref in results[s]]
        row = next(r for r in table if r["config"] == label)
        print(f"  {label:<44s} {row['adj_edge_jaccard'] - base_row['adj_edge_jaccard']:+.4f}  "
              f"better {sum(1 for d in delta if d > 0.005):>2d}  "
              f"worse {sum(1 for d in delta if d < -0.005):>2d}")


if __name__ == "__main__":
    main()
