"""Gate the re-linked edges on motion margin AND model probability.

Run:  python src/postprocessing/exp_gate.py
"""

import numpy as np

from common import RESULTS, SAMPLE, fmt, load_gt, load_prediction, save, score, to_graph
from confidence import annotate
from exp_selective import prune_short_tracks
from model_assign import load_candidates
from relink import relink


def main():
    gt_graph, n_total = load_gt()
    coords, edges = load_prediction()
    by_pair = load_candidates(RESULTS / f"{SAMPLE}_edge_candidates.npz")

    rows = [score(to_graph(coords, edges), gt_graph, n_total, "baseline (raw npz)")]

    linked = relink(coords, edges, max_distance=7.5)
    rich = annotate(coords, linked, by_pair)   # cols: src tgt prob dist margin model_p residual
    print(f"\nre-linked {len(rich)} edges; model ranked "
          f"{int((rich[:,5]>0).sum())} of them ({100*(rich[:,5]>0).mean():.0f}%)")

    print("\n--- each signal alone ---")
    for cut in (0.0, 0.1, 0.2, 0.3, 0.4):
        e = rich[rich[:, 5] >= cut]
        rows.append(score(to_graph(coords, e[:, :4]), gt_graph, n_total,
                          f"relink + model_p>={cut:g}"))
    print()
    for cut in (0.0, 2.0, 3.0, 4.0):
        e = rich[rich[:, 4] >= cut]
        rows.append(score(to_graph(coords, e[:, :4]), gt_graph, n_total,
                          f"relink + margin>={cut:g}"))
    print()
    for cut in (10.0, 4.0, 3.0, 2.0, 1.5):
        e = rich[rich[:, 6] <= cut]
        rows.append(score(to_graph(coords, e[:, :4]), gt_graph, n_total,
                          f"relink + residual<={cut:g}"))

    print("\n--- combined, with track pruning ---")
    combos = []
    for track_cut in (5, 8, 10, 12):
        c, kept = prune_short_tracks(coords, rich, track_cut)
        for m_cut in (0.0, 2.0, 3.0):
            for p_cut in (0.0, 0.15, 0.25, 0.35):
                for r_cut in (99.0, 4.0, 3.0):
                    e = kept[(kept[:, 4] >= m_cut) & (kept[:, 5] >= p_cut) & (kept[:, 6] <= r_cut)]
                    combos.append(score(
                        to_graph(c, e[:, :4]), gt_graph, n_total,
                        f"track>={track_cut} margin>={m_cut:g} p>={p_cut:g} res<={r_cut:g}",
                        verbose=False,
                    ))
    combos.sort(key=lambda r: -r["adj_edge_jaccard"])
    for r in combos[:12]:
        print(fmt(r))

    save(rows + combos, "gate")
    best = max(rows + combos, key=lambda r: r["adj_edge_jaccard"])
    print(f"\nbest: {best['label']}  adjJ={best['adj_edge_jaccard']:.4f}")


if __name__ == "__main__":
    main()
