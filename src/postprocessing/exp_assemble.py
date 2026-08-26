"""Prune on assembled track span, then gate. Run: python src/postprocessing/exp_assemble.py"""

import numpy as np

from assemble import prune_by_span
from common import RESULTS, SAMPLE, fmt, load_gt, load_prediction, save, score, to_graph
from confidence import annotate
from dedup import apply_mask
from model_assign import load_candidates
from relink import relink


def main():
    gt_graph, n_total = load_gt()
    coords, edges = load_prediction()
    by_pair = load_candidates(RESULTS / f"{SAMPLE}_edge_candidates.npz")

    rows = [score(to_graph(coords, edges), gt_graph, n_total, "baseline (raw npz)")]
    linked = relink(coords, edges, max_distance=7.5)
    rich = annotate(coords, linked, by_pair)

    print("\n--- nodes surviving a span cut, tracks assembled across gaps ---")
    for max_gap in (1, 2, 3):
        for span in (10, 20, 30, 40, 50):
            keep = prune_by_span(coords, rich, span, max_gap=max_gap)
            print(f"    gap<={max_gap} span>={span:>2d}: keeps {int(keep.sum()):>6d} "
                  f"({100*keep.mean():.0f}%)  target {n_total:.0f}")

    print("\n--- scored ---")
    out = []
    for max_gap in (2, 3):
        for span in (10, 20, 30, 40, 50, 60):
            keep = prune_by_span(coords, rich, span, max_gap=max_gap)
            c, e = apply_mask(coords, rich, keep)
            for m_cut in (0.0, 2.0, 3.0, 4.0):
                for r_cut in (99.0, 4.0, 3.5):
                    f = e[(e[:, 4] >= m_cut) & (e[:, 6] <= r_cut)]
                    out.append(score(
                        to_graph(c, f[:, :4]), gt_graph, n_total,
                        f"gap<={max_gap} span>={span} margin>={m_cut:g} res<={r_cut:g}",
                        verbose=False,
                    ))
    out.sort(key=lambda r: -r["adj_edge_jaccard"])
    for r in out[:15]:
        print(fmt(r))

    save(rows + out, "assemble")
    best = max(rows + out, key=lambda r: r["adj_edge_jaccard"])
    print(f"\nbest: {best['label']}  adjJ={best['adj_edge_jaccard']:.4f}")


if __name__ == "__main__":
    main()
