"""Attach every re-linked edge the evidence available about it.

The re-linker decides on motion; the model knows appearance. Neither alone
separates the links we get right from the ones we get wrong well enough — but a
link both agree on is a different proposition from one only motion likes.

Columns added per edge: the model's probability for that exact pair (0 when the
model did not rank it at all), and the link's residual against the local flow.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from params import SCALE

SCALE_ARR = np.asarray(SCALE)


def model_prob_lookup(by_pair):
    """(parent row, child row) -> probability, from the candidate dump."""
    prob = {}
    for t, per_child in by_pair.items():
        for child, cands in per_child.items():
            for parent, p in cands:
                prob[(int(parent), int(child))] = float(p)
    return prob


def annotate(coords, edges, by_pair, flow_k=30):
    """edges (m, 5) -> (m, 7): + model probability, + flow residual in um."""
    coords = np.asarray(coords)
    pos_um = coords[:, 1:].astype(float) * SCALE_ARR
    edges = np.asarray(edges, dtype=float).reshape(-1, 5)
    prob = model_prob_lookup(by_pair)

    model_p = np.array([prob.get((int(s), int(t)), 0.0) for s, t in edges[:, :2]])

    residual = np.zeros(len(edges))
    rows_by_t = {int(t): np.flatnonzero(coords[:, 0] == t) for t in np.unique(coords[:, 0])}
    src_t = coords[edges[:, 0].astype(int), 0].astype(int)
    for t in sorted(rows_by_t):
        if t + 1 not in rows_by_t:
            continue
        idx = np.flatnonzero(src_t == t)
        if not len(idx):
            continue
        here = rows_by_t[t]
        local = {int(r): i for i, r in enumerate(here)}
        src_um = pos_um[here]
        anchor = pos_um[edges[idx, 0].astype(int)]
        disp = pos_um[edges[idx, 1].astype(int)] - anchor
        k = min(flow_k, len(idx))
        _, near = cKDTree(anchor).query(anchor, k=k)
        flow = np.median(disp[near.reshape(len(idx), -1)], axis=1)
        residual[idx] = np.linalg.norm(disp - flow, axis=1)

    return np.column_stack([edges, model_p, residual])
