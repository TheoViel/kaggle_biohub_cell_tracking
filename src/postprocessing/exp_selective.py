"""Two levers that follow from the metric's own arithmetic.

With every ground-truth node detected, J reduces to ``TP / (85 + FP)``, and a
ground-truth link we get wrong costs *two* false positives — one for the parent
we linked elsewhere, one for the child we linked from elsewhere. Differentiating
that, a link is worth making only when it is better than ~54% likely to be
right. Below that, declining to link scores higher than guessing.

    lever 1  margin gating   — decline the links the assignment nearly tied on
    lever 2  track pruning   — drop the short tracks, which are mostly spurious
                               detections, and with them the node-count penalty

Run:  python src/postprocessing/exp_selective.py
"""

from collections import defaultdict

import numpy as np

from common import load_gt, load_prediction, save, score, to_graph
from relink import relink

BEST_DISTANCE = 8.0


def tracks_of(n_nodes, edges):
    """Connected components of the one-to-one graph — i.e. the tracks."""
    parent = list(range(n_nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, t in edges[:, :2].astype(int):
        ra, rb = find(s), find(t)
        if ra != rb:
            parent[ra] = rb
    groups = defaultdict(list)
    for i in range(n_nodes):
        groups[find(i)].append(i)
    return list(groups.values())


def prune_short_tracks(coords, edges, min_length):
    """Drop every node whose track spans fewer than min_length frames."""
    keep = np.zeros(len(coords), dtype=bool)
    for comp in tracks_of(len(coords), edges):
        if len(comp) >= min_length:
            keep[comp] = True
    index = np.full(len(coords), -1, dtype=int)
    index[keep] = np.arange(keep.sum())
    kept_edges = edges[keep[edges[:, 0].astype(int)] & keep[edges[:, 1].astype(int)]].copy()
    kept_edges[:, 0] = index[kept_edges[:, 0].astype(int)]
    kept_edges[:, 1] = index[kept_edges[:, 1].astype(int)]
    return coords[keep], kept_edges


def main():
    gt_graph, n_total = load_gt()
    coords, edges = load_prediction()
    rows = [score(to_graph(coords, edges), gt_graph, n_total, "baseline (raw npz)")]

    linked = relink(coords, edges, max_distance=BEST_DISTANCE)
    rows.append(score(to_graph(coords, linked[:, :4]), gt_graph, n_total,
                      f"relink d<={BEST_DISTANCE:.0f}"))

    print("\n--- lever 1: decline the links the assignment nearly tied on ---")
    margin = linked[:, 4]
    finite = margin[np.isfinite(margin)]
    print(f"    margin (um): med={np.median(finite):.2f} "
          f"p10={np.percentile(finite,10):.2f} p25={np.percentile(finite,25):.2f}")
    for cut in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
        kept = linked[margin >= cut]
        rows.append(score(to_graph(coords, kept[:, :4]), gt_graph, n_total,
                          f"relink + margin>={cut:g}um"))

    print("\n--- lever 2: drop the short tracks ---")
    comp = tracks_of(len(coords), linked)
    sizes = np.array([len(c) for c in comp])
    print(f"    {len(comp)} tracks; sizes med={np.median(sizes):.0f} max={sizes.max()}")
    for cut in (2, 3, 5, 8, 12, 20, 30):
        n_kept = sizes[sizes >= cut].sum()
        print(f"    min length {cut:>2d}: keeps {n_kept:>6d} nodes "
              f"({100*n_kept/len(coords):.0f}%), estimated {n_total:.0f}")
    for cut in (2, 3, 5, 8, 12, 20):
        c, e = prune_short_tracks(coords, linked, cut)
        rows.append(score(to_graph(c, e[:, :4]), gt_graph, n_total,
                          f"relink + track>={cut}"))

    print("\n--- both ---")
    for m_cut in (0.5, 1.0, 1.5, 2.0):
        for t_cut in (5, 12, 20):
            kept = linked[margin >= m_cut]
            c, e = prune_short_tracks(coords, kept, t_cut)
            rows.append(score(to_graph(c, e[:, :4]), gt_graph, n_total,
                              f"relink + margin>={m_cut:g} + track>={t_cut}"))

    save(rows, "selective")
    best = max(rows, key=lambda r: r["adj_edge_jaccard"])
    print(f"\nbest: {best['label']}  adjJ={best['adj_edge_jaccard']:.4f}")


if __name__ == "__main__":
    main()
