"""Sweep the tracker with cost in microns: no_link is then a distance in um.

cost = motion_weight * displacement_um + model_weight * (-log p)

With motion_weight = 1 the units are microns, so no_link reads directly as
"link only if the cell is predicted to within this distance", and model_weight
reads as "microns of geometric slack one nat of log-probability buys".

Run:  python src/postprocessing/exp_track2.py
"""

import numpy as np

from common import RESULTS, SAMPLE, load_gt, load_prediction, save, score, to_graph
from exp_selective import prune_short_tracks
from model_assign import load_candidates
from track import track


def main():
    gt_graph, n_total = load_gt()
    coords, edges = load_prediction()
    by_pair = load_candidates(RESULTS / f"{SAMPLE}_edge_candidates.npz")

    rows = [score(to_graph(coords, edges), gt_graph, n_total, "baseline (raw npz)")]

    def run(label, **kw):
        kw.setdefault("motion_weight", 1.0)
        kw.setdefault("model_weight", 0.0)
        kw.setdefault("velocity_weight", 0.0)
        kw.setdefault("max_distance", 25.0)
        e = track(coords, by_pair, seed_edges=edges, **kw)
        rows.append(score(to_graph(coords, e[:, :4]), gt_graph, n_total, label))
        return e

    print("\n--- motion only: no_link is a distance in um ---")
    for nl in (6.0, 7.0, 8.0, 9.0, 10.0, 12.0):
        run(f"motion only, no_link={nl:g}um", no_link=nl)

    print("\n--- how many model candidates to keep ---")
    for k in (1, 2, 3, 5, 8, 12):
        run(f"motion only, top_k={k}, no_link=8", top_k=k, no_link=8.0)

    print("\n--- adding the model score (um of slack per nat) ---")
    for mw in (0.0, 0.5, 1.0, 2.0, 4.0):
        run(f"model_w={mw:g}um/nat, no_link=8", model_weight=mw, no_link=8.0)

    print("\n--- a cell's own velocity ---")
    for vw in (0.0, 0.25, 0.5, 0.75):
        run(f"velocity={vw:g}, no_link=8", velocity_weight=vw, no_link=8.0)

    print("\n--- passes ---")
    for p in (1, 2, 3, 5):
        run(f"passes={p}, no_link=8", passes=p, no_link=8.0)

    save(rows, "track_sweep2")
    best = max(rows, key=lambda r: r["adj_edge_jaccard"])
    print(f"\nbest: {best['label']}  adjJ={best['adj_edge_jaccard']:.4f}")


if __name__ == "__main__":
    main()
