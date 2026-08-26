"""Sweep the re-linker's two knobs against the baseline.

Run:  python src/postprocessing/exp_relink.py
"""

import numpy as np

from common import load_gt, load_prediction, save, score, to_graph
from relink import relink


def main():
    gt_graph, n_total = load_gt()
    coords, edges = load_prediction()

    rows = [score(to_graph(coords, edges), gt_graph, n_total, "baseline (raw npz)")]

    print()
    for max_distance in (7.0, 8.0, 9.0, 10.0, 12.0):
        e = relink(coords, edges, max_distance=max_distance, prob_weight=0.0)
        rows.append(score(to_graph(coords, e), gt_graph, n_total,
                          f"relink d<={max_distance:.0f} pw=0"))

    print()
    for prob_weight in (0.0, 2.0, 5.0, 10.0, 20.0):
        e = relink(coords, edges, max_distance=9.0, prob_weight=prob_weight)
        rows.append(score(to_graph(coords, e), gt_graph, n_total,
                          f"relink d<=9 pw={prob_weight:g}"))

    print()
    for passes in (1, 2, 3, 5, 8):
        e = relink(coords, edges, max_distance=9.0, prob_weight=5.0, passes=passes)
        rows.append(score(to_graph(coords, e), gt_graph, n_total,
                          f"relink d<=9 pw=5 passes={passes}"))

    print()
    for flow_k in (10, 20, 30, 50, 80):
        e = relink(coords, edges, max_distance=9.0, prob_weight=5.0, flow_k=flow_k)
        rows.append(score(to_graph(coords, e), gt_graph, n_total,
                          f"relink d<=9 pw=5 k={flow_k}"))

    save(rows, "relink_sweep")
    best = max(rows, key=lambda r: r["adj_edge_jaccard"])
    print(f"\nbest: {best['label']}  adjJ={best['adj_edge_jaccard']:.4f}")


if __name__ == "__main__":
    main()
