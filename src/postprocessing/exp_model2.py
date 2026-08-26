"""Sweep the no-link price, then the flow mix, then margin and track pruning.

Run:  python src/postprocessing/exp_model2.py
"""

import numpy as np

from common import RESULTS, SAMPLE, load_gt, load_prediction, save, score, to_graph
from exp_selective import prune_short_tracks
from model_assign import assign, load_candidates


def main():
    gt_graph, n_total = load_gt()
    coords, edges = load_prediction()
    by_pair = load_candidates(RESULTS / f"{SAMPLE}_edge_candidates.npz")

    rows = [score(to_graph(coords, edges), gt_graph, n_total, "baseline (raw npz)")]

    print("\n--- price of leaving a cell unlinked (model only, cost = -log p) ---")
    for c in (0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 3.0, np.inf):
        e = assign(coords, by_pair, no_link=c)
        rows.append(score(to_graph(coords, e[:, :4]), gt_graph, n_total,
                          f"assign no_link={c:g} (p>{np.exp(-c):.2f})"))

    print("\n--- with the flow term ---")
    for fw in (0.05, 0.1, 0.2, 0.3):
        for c in (1.5, 2.0, 2.5, 3.0):
            e = assign(coords, by_pair, flow_weight=fw, max_distance=10.0, no_link=c)
            rows.append(score(to_graph(coords, e[:, :4]), gt_graph, n_total,
                              f"assign fw={fw:g} no_link={c:g}"))

    save(rows, "model_assign_nolink")
    best = max(rows, key=lambda r: r["adj_edge_jaccard"])
    print(f"\nbest so far: {best['label']}  adjJ={best['adj_edge_jaccard']:.4f}")


if __name__ == "__main__":
    main()
