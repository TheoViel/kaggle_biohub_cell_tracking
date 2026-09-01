"""The pipeline up to the gate, shared by training and inference.

The gate judges the links that survive re-linking and short-track pruning, so
that is the state features must be computed in. Pruning renumbers the nodes,
which means the tracker's own edges have to be renumbered with them — otherwise
the "did the model score this pair" feature is looked up against stale indices
and silently becomes noise.

Both build.py and gate.py go through here so the two cannot drift apart.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from postprocessing.pipeline import DEFAULTS
from postprocessing.relink import relink


def _components(n_nodes, edges):
    parent = np.arange(n_nodes)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, t in edges[:, :2].astype(int):
        ra, rb = find(s), find(t)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n_nodes)])


def prepare(coords, npz_edges, **kw):
    """-> (coords, linked, npz_edges), all in one consistent index space.

    ``linked`` is the re-linker's (L, 5): source, target, probability, distance,
    margin. ``npz_edges`` is the tracker's own (M, 4), renumbered to match and
    with any edge touching a pruned node dropped.
    """
    cfg = {**DEFAULTS, **kw}
    coords = np.asarray(coords)
    npz_edges = np.asarray(npz_edges, dtype=float).reshape(-1, 4)

    linked = relink(coords, npz_edges, max_distance=cfg["max_distance"],
                    flow_k=cfg["flow_k"], passes=cfg["passes"])
    if len(linked) == 0:
        return coords, linked, npz_edges

    label = _components(len(coords), linked)
    size = defaultdict(int)
    for lab in label:
        size[lab] += 1
    keep = np.array([size[lab] >= cfg["min_track"] for lab in label])
    if keep.all():
        return coords, linked, npz_edges

    index = np.full(len(coords), -1, dtype=int)
    index[keep] = np.arange(int(keep.sum()))

    def remap(arr):
        arr = np.asarray(arr, dtype=float)
        alive = keep[arr[:, 0].astype(int)] & keep[arr[:, 1].astype(int)]
        out = arr[alive].copy()
        out[:, 0] = index[out[:, 0].astype(int)]
        out[:, 1] = index[out[:, 1].astype(int)]
        return out

    return coords[keep], remap(linked), remap(npz_edges)
