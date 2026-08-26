"""Add duplicate-track suppression to the pipeline and re-sweep.

Run:  python src/postprocessing/exp_dedup.py
"""

import numpy as np

from common import RESULTS, SAMPLE, fmt, load_gt, load_prediction, save, score, to_graph
from confidence import annotate
from dedup import apply_mask, suppress_parallel
from exp_selective import prune_short_tracks
from model_assign import load_candidates
from relink import relink


def main():
    gt_graph, n_total = load_gt()
    coords, edges = load_prediction()
    by_pair = load_candidates(RESULTS / f"{SAMPLE}_edge_candidates.npz")

    rows = [score(to_graph(coords, edges), gt_graph, n_total, "baseline (raw npz)")]

    linked = relink(coords, edges, max_distance=7.5)
    rich = annotate(coords, linked, by_pair)

    print("\n--- how many nodes duplicate suppression removes ---")
    for radius in (5.0, 6.0, 7.0, 8.0):
        for overlap in (0.5, 0.7, 0.9):
            keep = suppress_parallel(coords, rich, radius=radius, overlap=overlap)
            print(f"    radius={radius:g} overlap={overlap:g}: keeps {int(keep.sum()):>6d} "
                  f"({100*keep.mean():.0f}%), estimated {n_total:.0f}")

    print("\n--- scored ---")
    out = []
    for radius in (5.0, 6.0, 7.0, 8.0):
        for overlap in (0.5, 0.7, 0.9):
            keep = suppress_parallel(coords, rich, radius=radius, overlap=overlap)
            c, e = apply_mask(coords, rich, keep)
            for track_cut in (1, 5, 8):
                c2, e2 = prune_short_tracks(c, e, track_cut)
                for m_cut in (0.0, 2.0, 3.0):
                    for r_cut in (99.0, 4.0):
                        f = e2[(e2[:, 4] >= m_cut) & (e2[:, 6] <= r_cut)]
                        out.append(score(
                            to_graph(c2, f[:, :4]), gt_graph, n_total,
                            f"dedup(r={radius:g},o={overlap:g}) track>={track_cut} "
                            f"margin>={m_cut:g} res<={r_cut:g}",
                            verbose=False,
                        ))
    out.sort(key=lambda r: -r["adj_edge_jaccard"])
    for r in out[:15]:
        print(fmt(r))

    save(rows + out, "dedup")
    best = max(rows + out, key=lambda r: r["adj_edge_jaccard"])
    print(f"\nbest: {best['label']}  adjJ={best['adj_edge_jaccard']:.4f}")


if __name__ == "__main__":
    main()
