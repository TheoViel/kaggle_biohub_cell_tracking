"""The notebook's ILP over many stacks, with its weights tuned.

The notebook runs it at edge=-1, appearance=disappearance=0.1, division=1 on a
single stack. Two things are worth separating here: whether the solver helps at
all once it is measured across the run, and whether its gain survives on top of
the re-linking pipeline — both are, in the end, deciding which edges to keep.

The division weight is swept up to a value that forbids divisions outright, so
the zero-division prior can be expressed inside the objective rather than as a
separate pass.

Run:  python src/postprocessing/exp_ilp_all.py --limit 10 --spread     # tune
      python src/postprocessing/exp_ilp_all.py --best                  # winners on all 71
"""

import argparse

from stages import cap_children, run_ilp, run_pipeline
from sweep import add_common_args, run, stack_names

# Cost of leaving a track end open, against edge costs that live in [-1, 0].
APPEAR = (0.02, 0.05, 0.1, 0.2, 0.4)
DIVISION = (1.0, 100.0)


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--best", action="store_true",
                    help="skip the grid; run the tuned settings on every stack")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()
    names = stack_names(args.limit, args.spread)
    print(f"{len(names)} stacks\n")

    if args.best:
        recipes = {
            "raw": [],
            "raw -> cap1": [(cap_children, dict(mode="best"))],
            "raw -> ILP(a=0.1,d=1)": [(run_ilp, dict(appear=0.1, disappear=0.1,
                                                     division=1.0, timeout=args.timeout))],
            "raw -> ILP(tuned)": [(run_ilp, dict(appear=BEST_APPEAR, disappear=BEST_APPEAR,
                                                 division=BEST_DIV, timeout=args.timeout))],
            "pipeline": [(run_pipeline, {})],
            "pipeline -> ILP(tuned)": [(run_pipeline, {}),
                                       (run_ilp, dict(appear=BEST_APPEAR, disappear=BEST_APPEAR,
                                                      division=BEST_DIV, timeout=args.timeout))],
            "raw -> ILP(tuned) -> pipeline": [
                (run_ilp, dict(appear=BEST_APPEAR, disappear=BEST_APPEAR,
                               division=BEST_DIV, timeout=args.timeout)),
                (run_pipeline, {})],
        }
        run(recipes, names, jobs=args.jobs, tag="ilp_best")
        return

    recipes = {"raw": [], "pipeline": [(run_pipeline, {})]}
    for div in DIVISION:
        for a in APPEAR:
            recipes[f"raw -> ILP(a={a:g},div={div:g})"] = [
                (run_ilp, dict(appear=a, disappear=a, division=div, timeout=args.timeout))
            ]
    for a in APPEAR:
        recipes[f"pipeline -> ILP(a={a:g},div=100)"] = [
            (run_pipeline, {}),
            (run_ilp, dict(appear=a, disappear=a, division=100.0, timeout=args.timeout)),
        ]
    run(recipes, names, jobs=args.jobs, tag="ilp_grid")


BEST_APPEAR = 0.1
BEST_DIV = 100.0

if __name__ == "__main__":
    main()
