"""Assemble tracks across missed frames, to decide which nodes are worth keeping.

Pruning by track length is the only node reduction that survived — raising the
detection threshold destroys recall long before it removes the excess. But it
is blunted by fragmentation: a real cell whose link was declined at one frame
becomes two short tracks and gets pruned, taking a ground-truth node with it.

So close the gaps *for grouping only*. A gap-closing edge would itself score as
a false positive — the ground truth links consecutive frames, so an edge from t
to t+2 can never match one, while still counting against any node it touches.
The joins are therefore used to decide track identity and never emitted.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from params import SCALE

SCALE_ARR = np.asarray(SCALE)


def _components(n_nodes, pairs):
    parent = np.arange(n_nodes)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, t in pairs:
        ra, rb = find(int(s)), find(int(t))
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n_nodes)])


def gap_joins(coords, edges, max_gap=3, step_um=7.0):
    """Extra (a, b) pairs joining a track's end to a later track's start.

    Only ends and starts are eligible, so this never gives a node a second
    parent or a second child; it re-attaches pieces of one cell's history.
    """
    coords = np.asarray(coords)
    edges = np.asarray(edges, dtype=float).reshape(-1, edges.shape[1])
    pos_um = coords[:, 1:].astype(float) * SCALE_ARR

    has_child = np.zeros(len(coords), bool)
    has_parent = np.zeros(len(coords), bool)
    has_child[edges[:, 0].astype(int)] = True
    has_parent[edges[:, 1].astype(int)] = True

    rows_by_t = {int(t): np.flatnonzero(coords[:, 0] == t) for t in np.unique(coords[:, 0])}
    joins = []
    used_end, used_start = set(), set()

    for gap in range(2, max_gap + 1):
        for t in sorted(rows_by_t):
            if t + gap not in rows_by_t:
                continue
            ends = np.array([r for r in rows_by_t[t]
                             if not has_child[r] and r not in used_end], dtype=int)
            starts = np.array([r for r in rows_by_t[t + gap]
                               if not has_parent[r] and r not in used_start], dtype=int)
            if not len(ends) or not len(starts):
                continue
            # A cell moves at most about one spacing per frame, so the reach
            # grows with the gap rather than being a fixed radius.
            reach = step_um * gap
            d = np.linalg.norm(pos_um[ends][:, None, :] - pos_um[starts][None, :, :], axis=-1)
            cost = np.where(d <= reach, d, 1e6)
            if not (cost < 1e6).any():
                continue
            n, m = cost.shape
            big = np.full((n + m, n + m), reach)
            big[:n, :m] = np.minimum(cost, reach)
            big[n:, m:] = 0.0
            rows, cols = linear_sum_assignment(big)
            ok = (rows < n) & (cols < m)
            rows, cols = rows[ok], cols[ok]
            keep = cost[rows, cols] < reach
            for a, b in zip(ends[rows[keep]], starts[cols[keep]]):
                joins.append((int(a), int(b)))
                used_end.add(int(a))
                used_start.add(int(b))
    return joins


def prune_by_span(coords, edges, min_span, max_gap=3, step_um=7.0):
    """Keep only nodes whose assembled track spans at least min_span frames."""
    coords = np.asarray(coords)
    edges = np.asarray(edges, dtype=float).reshape(-1, edges.shape[1])
    pairs = list(map(tuple, edges[:, :2].astype(int))) + gap_joins(
        coords, edges, max_gap=max_gap, step_um=step_um
    )
    label = _components(len(coords), pairs)

    order = np.argsort(label, kind="stable")
    sorted_label = label[order]
    bounds = np.flatnonzero(np.r_[True, sorted_label[1:] != sorted_label[:-1], True])
    keep = np.zeros(len(coords), bool)
    for a, b in zip(bounds[:-1], bounds[1:]):
        rows = order[a:b]
        ts = coords[rows, 0]
        if ts.max() - ts.min() + 1 >= min_span:
            keep[rows] = True
    return keep
