"""Sweep the model-scored assignment.

Run:  python src/postprocessing/exp_model.py
"""

import numpy as np

from common import RESULTS, SAMPLE, load_gt, load_prediction, save, score, to_graph
from exp_selective import prune_short_tracks, tracks_of
from model_assign import assign, load_candidates


def main():
    gt_graph, n_total = load_gt()
    coords, edges = load_prediction()
    by_pair = load_candidates(RESULTS / f"{SAMPLE}_edge_candidates.npz")

    rows = [score(to_graph(coords, edges), gt_graph, n_total, "baseline (raw npz)")]

    print("\n--- assignment on the model alone ---")
    e = assign(coords, by_pair)
    rows.append(score(to_graph(coords, e[:, :4]), gt_graph, n_total, "assign (model only)"))

    print("\n--- mixing in the flow-corrected distance ---")
    for fw in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
        e = assign(coords, by_pair, flow_weight=fw, max_distance=10.0)
        rows.append(score(to_graph(coords, e[:, :4]), gt_graph, n_total,
                          f"assign fw={fw:g} d<=10"))

    print("\n--- distance gate ---")
    for d in (6.0, 7.0, 8.0, 9.0, 10.0, 12.0):
        e = assign(coords, by_pair, flow_weight=0.1, max_distance=d)
        rows.append(score(to_graph(coords, e[:, :4]), gt_graph, n_total,
                          f"assign fw=0.1 d<={d:g}"))

    save(rows, "model_assign_sweep")
    best = max(rows, key=lambda r: r["adj_edge_jaccard"])
    print(f"\nbest so far: {best['label']}  adjJ={best['adj_edge_jaccard']:.4f}")


if __name__ == "__main__":
    main()
