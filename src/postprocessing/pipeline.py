"""The post-processing pipeline that came out of the experiments.

Everything here runs from the saved npz alone — no model, no images.

The order is fixed by what each step needs from the one before:

1. **Re-link** every frame pair as a global assignment on flow-corrected
   distance. The tracker picks each child's parent independently, so children
   compete for parents and up to two children can take the same one; the
   ground truth in this data has no divisions at all, so every second child is
   a guaranteed false positive. Solving the pair jointly fixes both.

2. **Prune short tracks.** The links now form chains, so a track's length is
   meaningful, and a chain a few frames long is a run of spurious detections.
   This is also the only node reduction that survives: raising the detection
   threshold instead destroys recall (at 0.4 the node count halves but only
   71% of ground-truth cells are still found), and the excess detections do
   not shadow each other closely enough for duplicate suppression to catch.

3. **Decline the uncertain links.** The metric charges two false positives for
   a link to the wrong cell — one for the parent linked elsewhere, one for the
   child linked from elsewhere — against one true positive for a right one.
   Differentiating J = TP/(n_gt + FP) says a link is only worth making when it
   is better than about 55% likely to be right. Two signals say when it is
   not: the assignment margin, and the link's residual against the local flow.

Pruning comes before gating: the gate cuts tracks in two, and pruning after it
would throw away both halves of a track that was only uncertain at one frame.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from params import SCALE
from relink import relink

SCALE_ARR = np.asarray(SCALE)

# Tuned on all 71 validation stacks (src/postprocessing/tune.py), not on the one hard
# sample: settings picked on that sample alone gain 0.13 there and lose 0.02
# across the run. Note where they landed — the two confidence gates are all but
# off. Nearly all of the gain is the re-linking; the gates trade true positives
# for false ones at about 1:2, which is worth it only on a stack whose linking
# is already poor.
DEFAULTS = dict(
    max_distance=7.0,      # um, gate on flow-corrected displacement
    min_track=3,           # nodes; shorter chains are dropped entirely
    min_margin=0.0,        # um the winner must beat the runner-up by
    max_residual=np.inf,   # um a link may depart from its neighbours' motion
    flow_k=30,
    passes=3,
)


def _flow_residual(coords, edges, flow_k=30):
    """How far each link departs from the motion of its neighbourhood, in um."""
    pos_um = coords[:, 1:].astype(float) * SCALE_ARR
    residual = np.zeros(len(edges))
    src_t = coords[edges[:, 0].astype(int), 0].astype(int)
    for t in np.unique(src_t):
        idx = np.flatnonzero(src_t == t)
        anchor = pos_um[edges[idx, 0].astype(int)]
        disp = pos_um[edges[idx, 1].astype(int)] - anchor
        k = min(flow_k, len(idx))
        _, near = cKDTree(anchor).query(anchor, k=k)
        flow = np.median(disp[near.reshape(len(idx), -1)], axis=1)
        residual[idx] = np.linalg.norm(disp - flow, axis=1)
    return residual


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


def _prune_short(coords, edges, min_track):
    label = _components(len(coords), edges)
    size = defaultdict(int)
    for lab in label:
        size[lab] += 1
    keep = np.array([size[lab] >= min_track for lab in label])

    index = np.full(len(coords), -1, dtype=int)
    index[keep] = np.arange(int(keep.sum()))
    alive = keep[edges[:, 0].astype(int)] & keep[edges[:, 1].astype(int)]
    kept = edges[alive].copy()
    kept[:, 0] = index[kept[:, 0].astype(int)]
    kept[:, 1] = index[kept[:, 1].astype(int)]
    return coords[keep], kept


def postprocess(coords, edges, **kw):
    """Raw tracker output -> (coords, edges) ready for build_graph.

    Returns edges as (source row, target row, probability, distance um).
    """
    cfg = {**DEFAULTS, **kw}
    coords = np.asarray(coords)

    linked = relink(
        coords, edges,
        max_distance=cfg["max_distance"],
        flow_k=cfg["flow_k"],
        passes=cfg["passes"],
    )
    if len(linked) == 0:
        return coords, np.empty((0, 4))

    coords, linked = _prune_short(coords, linked, cfg["min_track"])
    if len(linked) == 0:
        return coords, np.empty((0, 4))

    residual = _flow_residual(coords, linked, cfg["flow_k"])
    keep = (linked[:, 4] >= cfg["min_margin"]) & (residual <= cfg["max_residual"])
    return coords, linked[keep][:, :4]
