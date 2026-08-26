"""Is the ILP's gain anything more than the zero-division constraint?

Both score +0.030 on the raw graph, which is suspicious. If the solver's chosen
edge set is essentially the one a greedy one-child cap produces, then the ILP is
a 2000x more expensive way to state the same prior, and the interesting question
is what else it could be given to decide.

Run:  python src/postprocessing/probe_ilp_vs_cap.py
"""

from collections import Counter
from pathlib import Path

import numpy as np

from common import ROOT
from params import TRAIN_DIR
from stages import cap_children, run_ilp
from sweep import stack_names

PRED_DIR = ROOT / "logs" / "2026-08-20" / "20" / "preds"


def main():
    for name in stack_names(3, spread=True):
        d = np.load(PRED_DIR / f"{name}.npz", allow_pickle=True)
        coords = d["coords"].astype(np.int64)
        edges = d["edges"].astype(np.float64).reshape(-1, 4)

        _, cap = cap_children(coords, edges, mode="best")
        _, ilp = run_ilp(coords, edges, appear=0.02, disappear=0.02, division=1.0)

        raw_set = {(int(a), int(b)) for a, b in edges[:, :2]}
        cap_set = {(int(a), int(b)) for a, b in cap[:, :2]}
        ilp_set = {(int(a), int(b)) for a, b in ilp[:, :2]}

        shared = len(cap_set & ilp_set)
        ilp_div = sum(v - 1 for v in Counter(int(a) for a, _ in ilp_set).values() if v > 1)
        print(f"{name}")
        print(f"    raw {len(raw_set)} edges -> cap1 {len(cap_set)}, ILP {len(ilp_set)}")
        print(f"    the two agree on {shared} edges "
              f"({100*shared/max(len(cap_set | ilp_set),1):.1f}% of their union)")
        print(f"    ILP kept {len(ilp_set - raw_set)} edges the raw graph did not have")
        print(f"    dividing nodes left by the ILP: {ilp_div}")


if __name__ == "__main__":
    main()
