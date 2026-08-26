"""Suppress tracks that shadow another track — the same cell detected twice.

The detector returns 38k nodes where the dataset estimates 24k, and non-maximum
suppression cannot tell the difference: the duplicates sit a full NMS radius
apart, because that is exactly what NMS leaves behind when a cell is bright
enough to peak twice. 47 of the 89 annotated cells have more than one detection
within the matching radius.

Over a single frame the two are indistinguishable. Over a track they are not:
a duplicate follows its original for the whole of its life, staying within a
cell's width the entire time. That is the signature this looks for, and it is
worth removing for two reasons — the node-count adjustment, and the fact that a
duplicate competes for the parent its original needed.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from params import SCALE

SCALE_ARR = np.asarray(SCALE)


def track_labels(n_nodes, edges):
    """Connected-component id per node — the tracks of a one-to-one graph."""
    parent = np.arange(n_nodes)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, t in np.asarray(edges)[:, :2].astype(int):
        ra, rb = find(s), find(t)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n_nodes)])


def suppress_parallel(coords, edges, radius=6.0, overlap=0.6, keep_longer=True):
    """Drop the weaker of every pair of tracks that shadow each other.

    Parameters
    ----------
    radius : float
        Microns. Two nodes of different tracks this close in the same frame
        count as shadowing for that frame.
    overlap : float
        Fraction of the shorter track's frames that must be shadowed before it
        is called a duplicate.

    Returns
    -------
    Boolean keep-mask over nodes.
    """
    coords = np.asarray(coords)
    edges = np.asarray(edges, dtype=float).reshape(-1, edges.shape[1] if np.ndim(edges) > 1 else 5)
    label = track_labels(len(coords), edges)

    length = defaultdict(int)
    for lab in label:
        length[lab] += 1

    shadow = defaultdict(int)
    pos_um = coords[:, 1:].astype(float) * SCALE_ARR
    for t in np.unique(coords[:, 0]):
        rows = np.flatnonzero(coords[:, 0] == t)
        if len(rows) < 2:
            continue
        pairs = cKDTree(pos_um[rows]).query_pairs(radius, output_type="ndarray")
        if not len(pairs):
            continue
        a = label[rows[pairs[:, 0]]]
        b = label[rows[pairs[:, 1]]]
        for la, lb in zip(a, b):
            if la != lb:
                shadow[(min(la, lb), max(la, lb))] += 1

    # Strongest track wins; process the most-shadowed pairs first so a cluster
    # of duplicates collapses onto one survivor rather than half of each.
    dropped = set()
    for (la, lb), count in sorted(shadow.items(), key=lambda kv: -kv[1]):
        if la in dropped or lb in dropped:
            continue
        short = min(length[la], length[lb])
        if short == 0 or count / short < overlap:
            continue
        loser = (la if length[la] < length[lb] else lb) if keep_longer else (
            la if length[la] > length[lb] else lb
        )
        dropped.add(loser)

    return ~np.isin(label, list(dropped)) if dropped else np.ones(len(coords), bool)


def apply_mask(coords, edges, keep):
    """Reindex a (coords, edges) pair down to the kept nodes."""
    edges = np.asarray(edges, dtype=float)
    index = np.full(len(coords), -1, dtype=int)
    index[keep] = np.arange(int(keep.sum()))
    alive = keep[edges[:, 0].astype(int)] & keep[edges[:, 1].astype(int)]
    kept = edges[alive].copy()
    kept[:, 0] = index[kept[:, 0].astype(int)]
    kept[:, 1] = index[kept[:, 1].astype(int)]
    return coords[keep], kept
