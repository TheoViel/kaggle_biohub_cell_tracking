"""Sweep the combined tracker.

Run:  python src/postprocessing/exp_track.py
"""

import numpy as np

from common import RESULTS, SAMPLE, load_gt, load_prediction, save, score, to_graph
from model_assign import load_candidates
from track import track


def main():
    gt_graph, n_total = load_gt()
    coords, edges = load_prediction()
    by_pair = load_candidates(RESULTS / f"{SAMPLE}_edge_candidates.npz")

    rows = [score(to_graph(coords, edges), gt_graph, n_total, "baseline (raw npz)")]

    def run(label, **kw):
        e = track(coords, by_pair, seed_edges=edges, **kw)
        rows.append(score(to_graph(coords, e[:, :4]), gt_graph, n_total, label))
        return e

    print("\n--- motion weight (model_weight=1) ---")
    for mw in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
        run(f"track mw={mw:g}", motion_weight=mw, no_link=3.0)

    print("\n--- model weight, motion fixed at 0.5 ---")
    for kw in (0.0, 0.25, 0.5, 1.0, 2.0):
        run(f"track model_w={kw:g}", model_weight=kw, motion_weight=0.5, no_link=3.0)

    print("\n--- how much to trust a cell's own velocity ---")
    for vw in (0.0, 0.3, 0.6, 0.9):
        run(f"track vel={vw:g}", motion_weight=0.5, velocity_weight=vw, no_link=3.0)

    print("\n--- no-link price ---")
    for nl in (1.5, 2.0, 2.5, 3.0, 4.0, 6.0):
        run(f"track no_link={nl:g}", motion_weight=0.5, no_link=nl)

    print("\n--- distance gate ---")
    for d in (6.0, 7.0, 8.0, 9.0, 11.0):
        run(f"track d<={d:g}", motion_weight=0.5, no_link=3.0, max_distance=d)

    save(rows, "track_sweep")
    best = max(rows, key=lambda r: r["adj_edge_jaccard"])
    print(f"\nbest: {best['label']}  adjJ={best['adj_edge_jaccard']:.4f}")


if __name__ == "__main__":
    main()
