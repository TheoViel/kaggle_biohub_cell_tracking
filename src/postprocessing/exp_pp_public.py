"""Ablate tracking.pp_public's knobs, from the raw graph and on top of the pipeline.

On the one stack looked at earlier it made things worse (0.531 -> 0.505), and
its own statistics said why: motion_relink replaced 33 879 of 33 895 edges with
its own geometric links, throwing the model's opinion away. This tests that
across the run and asks which of its other passes are worth keeping.

Every knob is set explicitly on each call, not left to the environment, so an
ablation measures only the switch it names.

Run:  python src/postprocessing/exp_pp_public.py --limit 10 --spread
"""

import argparse

from stages import PP_PUBLIC_DEFAULTS, run_pipeline, run_pp_public
from sweep import add_common_args, run, stack_names

# On/off switches worth turning the other way, and why.
ABLATIONS = {
    "no motion_relink": dict(motion_relink=0),
    "single_child_repair": dict(single_child_repair=1),      # the zero-division prior
    "no safe_divisions": dict(safe_divisions=0),             # it adds divisions
    "no gap_close": dict(gap_close=0),
    "no gap2": dict(gap2=0),
    "no linefit": dict(linefit=0),
    "filter_short_tracks(4)": dict(filter_short_tracks=1, min_track_len=4),
    "division_geometry_filter": dict(division_geometry_filter=1),
    "edge_max=9um": dict(edge_max_um=9.0),
}


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--tune", action="store_true",
                    help="coordinate-wise grid over pp_public's knobs")
    ap.add_argument("--combined", action="store_true",
                    help="skip the ablation; test the combinations that won")
    args = ap.parse_args()
    names = stack_names(args.limit, args.spread)
    print(f"{len(names)} stacks\n")

    if args.tune:
        # Coordinate-wise around the defaults, on top of the pipeline — the
        # chain the ablation picked out. motion_relink is what pp_public
        # mostly *is*, so its two radii and its learned-score bonus are the
        # knobs with anything to give.
        grid = {
            "min_track_len": (3, 4, 6, 8),
            "relink_tight_um": (5.0, 5.95, 6.5, 7.0),
            "relink_relaxed_um": (8.0, 10.0, 12.0),
            "relink_learned_bonus": (0.4, 0.76, 1.2, 2.0),
            "linefit_weight": (0.5, 0.74, 0.9),
        }
        base = dict(filter_short_tracks=1)
        recipes = {
            "pipeline": [(run_pipeline, {})],
            "pipeline -> pp_public(default)": [(run_pipeline, {}), (run_pp_public, {})],
        }
        for knob, values in grid.items():
            for v in values:
                recipes[f"pipeline -> pp_public {knob}={v:g}"] = [
                    (run_pipeline, {}), (run_pp_public, {**base, knob: v})
                ]
        run(recipes, names, jobs=args.jobs, tag="pp_public_tune")
        return

    if args.combined:
        best = dict(motion_relink=0, single_child_repair=1, safe_divisions=0,
                    gap2=0, filter_short_tracks=1, min_track_len=4)
        recipes = {
            "raw": [],
            "pipeline": [(run_pipeline, {})],
            "raw -> pp_public(default)": [(run_pp_public, {})],
            "raw -> pp_public(best)": [(run_pp_public, best)],
            "pipeline -> pp_public(best)": [(run_pipeline, {}), (run_pp_public, best)],
            "raw -> pp_public(best) -> pipeline": [(run_pp_public, best), (run_pipeline, {})],
        }
        run(recipes, names, jobs=args.jobs, tag="pp_public_combined")
        return

    recipes = {
        "raw": [],
        "pipeline": [(run_pipeline, {})],
        "raw -> pp_public(default)": [(run_pp_public, {})],
        "pipeline -> pp_public(default)": [(run_pipeline, {}), (run_pp_public, {})],
    }
    for label, knobs in ABLATIONS.items():
        recipes[f"raw -> pp_public + {label}"] = [(run_pp_public, knobs)]

    run(recipes, names, jobs=args.jobs, tag="pp_public_ablation")


if __name__ == "__main__":
    main()
