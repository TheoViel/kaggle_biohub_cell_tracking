"""Fine grid over the levers that survived: relink, margin gate, track pruning.

Order matters. Tracks are cut into pieces by the margin gate, so pruning by
track length after gating throws away the two halves of a track that was only
broken at one uncertain link. Pruning first, gating second keeps them.

Run:  python src/postprocessing/exp_grid.py
"""

import numpy as np

from common import load_gt, load_prediction, save, score, to_graph
from exp_selective import prune_short_tracks
from relink import relink


def main():
    gt_graph, n_total = load_gt()
    coords, edges = load_prediction()
    rows = [score(to_graph(coords, edges), gt_graph, n_total, "baseline (raw npz)", verbose=True)]

    print()
    linked = {d: relink(coords, edges, max_distance=d) for d in (7.5, 8.0, 8.5)}

    for d, e in linked.items():
        for track_cut in (1, 5, 8, 10, 12, 15):
            c, kept = prune_short_tracks(coords, e, track_cut)
            for m_cut in (0.0, 1.0, 2.0, 3.0):
                final = kept[kept[:, 4] >= m_cut]
                rows.append(score(
                    to_graph(c, final[:, :4]), gt_graph, n_total,
                    f"d={d:g} track>={track_cut} margin>={m_cut:g}", verbose=False,
                ))

    rows.sort(key=lambda r: -r["adj_edge_jaccard"])
    print("--- top 15 ---")
    from common import fmt
    for r in rows[:15]:
        print(fmt(r))
    save(rows, "grid")


if __name__ == "__main__":
    main()
