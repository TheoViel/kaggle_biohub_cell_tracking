"""Does the grid-centring correction survive the rest of the chain?

Run:  python src/postprocessing/exp_recentre.py --jobs 24
"""

import argparse
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "postprocessing"))
os.chdir(SRC)

from postprocessing.pipeline import RECENTRE_LEGACY  # noqa: E402
from stages import cap_children, run_pipeline, run_pp_public  # noqa: E402
from sweep import add_common_args, run, stack_names  # noqa: E402

PP_BEST = dict(filter_short_tracks=1, min_track_len=4, relink_tight_um=6.5)
OFF = dict(recentre=None)          # the control: leave the corner convention
ON = dict(recentre=RECENTRE_LEGACY)


def main():
    args = add_common_args(argparse.ArgumentParser()).parse_args()
    names = stack_names(args.limit, args.spread)
    print(f"{len(names)} stacks\n")

    recipes = {
        "raw": [],
        "raw -> cap1": [(cap_children, dict(mode="best"))],
        "pipeline, no recentre": [(run_pipeline, OFF)],
        "pipeline + recentre": [(run_pipeline, ON)],
        "pipeline, no recentre -> pp_public": [(run_pipeline, OFF), (run_pp_public, PP_BEST)],
        "pipeline + recentre -> pp_public": [(run_pipeline, ON), (run_pp_public, PP_BEST)],
    }
    run(recipes, names, jobs=args.jobs, tag="recentre")


if __name__ == "__main__":
    main()
