"""The zero-division prior, from the raw graph and on top of the pipeline.

Across the 71 validation stacks the annotation has 26 dividing nodes in 19 826
edges (0.13%), while 7.7% of predicted edges are second children. Forbidding
divisions therefore risks almost nothing and removes a great deal.

The pipeline's re-linking already enforces it — every frame pair is solved as a
one-to-one assignment — so the raw-graph rows are the ones that measure the
prior on its own, and the pipeline rows confirm there is nothing left to take.

Run:  python src/postprocessing/exp_zero_div.py [--limit N] [--spread]
"""

import argparse

from stages import cap_children, run_pipeline
from sweep import add_common_args, run, stack_names


def main():
    args = add_common_args(argparse.ArgumentParser()).parse_args()
    names = stack_names(args.limit, args.spread)
    print(f"{len(names)} stacks\n")

    recipes = {
        "raw": [],
        "raw -> cap1(best)": [(cap_children, dict(mode="best"))],
        "raw -> cap1(assign)": [(cap_children, dict(mode="assign"))],
        "pipeline": [(run_pipeline, {})],
        "pipeline -> cap1(best)": [(run_pipeline, {}), (cap_children, dict(mode="best"))],
    }
    run(recipes, names, jobs=args.jobs, tag="zero_div")


if __name__ == "__main__":
    main()
