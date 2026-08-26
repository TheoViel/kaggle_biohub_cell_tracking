"""Facts needed to size the next three experiments.

How many divisions does the ground truth really contain across all 71 stacks,
what does the zero-division prior cost, and how long does one ILP solve take.

Run:  python src/postprocessing/probe.py
"""

import contextlib
import io
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tracksdata as td

from common import ROOT, to_graph
from params import TRAIN_DIR

PRED_DIR = ROOT / "logs" / "2026-08-20" / "20" / "preds"


def main():
    names = [p.stem for p in sorted(PRED_DIR.glob("*.npz"))
             if (Path(TRAIN_DIR) / f"{p.stem}.geff").exists()]

    total_nodes = total_edges = total_div = 0
    per_stack = []
    for name in names:
        r = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{name}.geff")
        g = r[0] if isinstance(r, tuple) else r
        deg = Counter(g.out_degree(g.node_ids()))
        div = sum(v for k, v in deg.items() if k > 1)
        total_nodes += g.num_nodes()
        total_edges += g.num_edges()
        total_div += div
        per_stack.append((name, g.num_nodes(), g.num_edges(), div))

    print(f"--- ground truth over {len(names)} stacks ---")
    print(f"    nodes {total_nodes}, edges {total_edges}, dividing nodes {total_div}")
    print(f"    stacks with any division: "
          f"{sum(1 for _, _, _, d in per_stack if d)}/{len(per_stack)}")
    print(f"    a zero-division prior can lose at most {total_div} of {total_edges} "
          f"GT edges ({100*total_div/total_edges:.2f}%)")
    worst = sorted(per_stack, key=lambda r: -r[3])[:5]
    print(f"    most divisions: {[(n, d) for n, _, _, d in worst]}")

    # --- how much of the predicted graph the prior would remove ---
    extra = kept = 0
    for name in names[:10]:
        d = np.load(PRED_DIR / f"{name}.npz", allow_pickle=True)
        e = d["edges"].astype(float).reshape(-1, 4)
        out_deg = Counter(e[:, 0].astype(int))
        extra += sum(v - 1 for v in out_deg.values() if v > 1)
        kept += len(e)
    print(f"\n--- predicted graphs (first 10 stacks) ---")
    print(f"    {extra} of {kept} edges are second children ({100*extra/kept:.1f}%)")

    # --- ILP timing on one stack ---
    name = names[0]
    d = np.load(PRED_DIR / f"{name}.npz", allow_pickle=True)
    graph = to_graph(d["coords"].astype(np.int64), d["edges"].astype(float).reshape(-1, 4))
    print(f"\n--- ILP timing on {name} ({graph.num_nodes()} nodes, "
          f"{graph.num_edges()} edges) ---")
    solver = td.solvers.ILPSolver(
        edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
        appearance_weight=0.1, disappearance_weight=0.1, division_weight=1.0,
        num_threads=1,
    )
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve(graph).detach()
    print(f"    {time.time() - t0:.1f}s with 1 thread")


if __name__ == "__main__":
    main()
