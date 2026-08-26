"""Every chain worth reporting, on all 71 validation stacks.

Each idea is measured twice — from the raw graph, so its own contribution is
visible, and on top of the re-linking pipeline, so it is clear whether it adds
anything or merely does the same job again.

Run:  python src/postprocessing/exp_final.py --jobs 24
"""

import argparse

from stages import cap_children, run_ilp, run_pipeline, run_pp_public
from sweep import add_common_args, run, stack_names

# Tuned on a 10-stack subset spanning the difficulty range (exp_pp_public.py
# --tune). motion_relink is left on: it is what pp_public mostly is, and
# turning it off costs 0.05.
PP_BEST = dict(filter_short_tracks=1, min_track_len=4, relink_tight_um=6.5)
ILP_BEST = dict(appear=0.02, disappear=0.02, division=1.0, timeout=900.0)
ILP_NOTEBOOK = dict(appear=0.1, disappear=0.1, division=1.0, timeout=900.0)


def main():
    args = add_common_args(argparse.ArgumentParser()).parse_args()
    names = stack_names(args.limit, args.spread)
    print(f"{len(names)} stacks\n")

    recipes = {
        "raw": [],
        # 1. the zero-division prior on its own
        "raw -> cap1": [(cap_children, dict(mode="best"))],
        # 2. the notebook's ILP, as written and tuned
        "raw -> ILP(notebook)": [(run_ilp, ILP_NOTEBOOK)],
        "raw -> ILP(tuned)": [(run_ilp, ILP_BEST)],
        # 3. pp_public, as shipped and tuned
        "raw -> pp_public(default)": [(run_pp_public, {})],
        "raw -> pp_public(tuned)": [(run_pp_public, PP_BEST)],
        # the pipeline, and each idea layered on it
        "pipeline": [(run_pipeline, {})],
        "pipeline -> ILP(tuned)": [(run_pipeline, {}), (run_ilp, ILP_BEST)],
        "pipeline -> pp_public(default)": [(run_pipeline, {}), (run_pp_public, {})],
        "pipeline -> pp_public(tuned)": [(run_pipeline, {}), (run_pp_public, PP_BEST)],
        "pipeline -> pp_public(tuned) -> ILP(tuned)": [
            (run_pipeline, {}), (run_pp_public, PP_BEST), (run_ilp, ILP_BEST)],
    }
    table, results = run(recipes, names, jobs=args.jobs, tag="final")

    base = next(r for r in table if r["config"] == "raw")
    print(f"\nagainst raw ({base['adj_edge_jaccard']:.4f}):")
    for r in table:
        if r["config"] == "raw":
            continue
        delta = [results[s][r["config"]]["adj_edge_jaccard"]
                 - results[s]["raw"]["adj_edge_jaccard"]
                 for s in results if r["config"] in results[s]]
        better = sum(1 for d in delta if d > 0.005)
        worse = sum(1 for d in delta if d < -0.005)
        print(f"  {r['config']:<44s} {r['adj_edge_jaccard']-base['adj_edge_jaccard']:+.4f}  "
              f"better {better:>2d}  worse {worse:>2d}")


if __name__ == "__main__":
    main()
