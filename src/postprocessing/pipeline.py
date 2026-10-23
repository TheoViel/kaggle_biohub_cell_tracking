"""The post-processing pipeline that came out of the experiments.

Everything here runs from the saved npz alone — no model, no images.

The order is fixed by what each step needs from the one before:

1. **Re-link** every frame pair as a global assignment on flow-corrected
   distance. The tracker picks each child's parent independently, so children
   compete for parents and up to two children can take the same one; the
   ground truth in this data has no divisions at all, so every second child is
   a guaranteed false positive. Solving the pair jointly fixes both.

2. **Recover the detections dropped for a single frame** (`recover.py`). A
   track that ends at `u` and one that begins at `u + 2` are the same cell far
   more often than not, and the node the detector missed at `u + 1` cost two
   ground-truth edges that nothing else here can win back. The decision is made
   on displacement with the field's own drift removed — over two frames that
   drift is the same size as the gate.

3. **Prune short tracks.** The links now form chains, so a track's length is
   meaningful, and a chain a few frames long is a run of spurious detections.
   This is also the only node reduction that survives: raising the detection
   threshold instead destroys recall (at 0.4 the node count halves but only
   71% of ground-truth cells are still found), and the excess detections do
   not shadow each other closely enough for duplicate suppression to catch.

4. **Decline the uncertain links.** The metric charges two false positives for
   a link to the wrong cell — one for the parent linked elsewhere, one for the
   child linked from elsewhere — against one true positive for a right one.
   Differentiating J = TP/(n_gt + FP) says a link is only worth making when it
   is better than about 55% likely to be right. Two signals say when it is
   not: the assignment margin, and the link's residual against the local flow.

Recovery comes before pruning, and pruning before gating: the gate cuts tracks
in two, and pruning after it would throw away both halves of a track that was
only uncertain at one frame — and pruning before recovery would throw away the
fragments recovery exists to join.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from params import SCALE
from recover import recover
from relink import relink

SCALE_ARR = np.asarray(SCALE)

# Tuned on all 71 validation stacks, not on the one hard
# sample: settings picked on that sample alone gain 0.13 there and lose 0.02
# across the run. Note where they landed — the two confidence gates are all but
# off. Nearly all of the gain is the re-linking; the gates trade true positives
# for false ones at about 1:2, which is worth it only on a stack whose linking
# is already poor.
# The detection grid is 4x coarser than the image in y and x, and a grid cell is
# reported at its corner rather than its centre (tracking/predict.py multiplied
# by unet_scaling; tracking/targets.py assigns with floor). That is a fixed
# -1.5 voxel error in y and x. Predictions saved before that was fixed carry it,
# and shifting them by this puts the detections back on the cells: the mean
# offset against the annotation goes from 0.83 um to 0.22 um and the score from
# 0.8065 to 0.8130. Do NOT apply it to predictions made after the fix.
RECENTRE_LEGACY = (0.0, 1.5, 1.5)


def detect_recentre(coords, max_step=8):
    """The shift a saved npz still needs, read off the coordinates themselves.

    Applying the correction twice would be as wrong as not applying it, and
    which is needed depends on when the npz was written. Fortunately the two
    cases are distinguishable: a corner-referenced axis has every coordinate at
    a multiple of the grid step, a centred one has every coordinate offset by
    half a step. So look at the residues.

    Returns a (z, y, x) shift in image voxels — zeros when nothing is needed.
    """
    coords = np.asarray(coords)
    shift = np.zeros(3)
    for axis in range(3):
        values = np.unique(coords[:, axis + 1].astype(np.int64))
        if len(values) < 2:
            continue
        step, residue = 1, 0
        for candidate in (8, 4, 2):
            if candidate > max_step:
                continue
            r = values % candidate
            if len(np.unique(r)) == 1:
                step, residue = candidate, int(r[0])
                break
        if step > 1 and residue == 0:
            shift[axis] = (step - 1) / 2.0
    return shift

DEFAULTS = dict(
    max_distance=7.0,      # um, gate on flow-corrected displacement
    min_track=3,           # nodes; shorter chains are dropped entirely
    min_margin=0.0,        # um the winner must beat the runner-up by
    max_residual=np.inf,   # um a link may depart from its neighbours' motion
    flow_k=30,
    passes=3,
    # "auto" reads the grid convention off the coordinates and shifts only a
    # corner-referenced npz. Pass None to disable, or an explicit (z, y, x).
    recentre="auto",
    # Put back the detections dropped for a frame or two (recover.py). +0.0161
    # over the 71 stacks, 42 better / 6 worse, and +0.0063 still survives under
    # pp_public. Costs a second relink pass: 5.5 -> 11.1 s/stack.
    recover=1,
    recover_max_residual=5.0,   # um of own motion across a one-frame gap, scaled by span
    recover_max_gap=3,          # longest run of missing frames to fill
    # A second round is worth +0.0002 for 50% more time — available, not default.
    recover_passes=1,           # rounds of recover + relink
    recover_min_dup=4.0,        # um; do not insert beside an existing detection
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
    coords = np.asarray(coords, dtype=np.float64)
    shift = cfg["recentre"]
    if isinstance(shift, str):
        shift = detect_recentre(coords)
    if shift is not None and np.any(shift):
        # A uniform translation, so nothing downstream changes except which
        # detection each annotated cell matches to.
        coords = coords.copy()
        coords[:, 1:] += np.asarray(shift, dtype=np.float64)

    linked = relink(
        coords, edges,
        max_distance=cfg["max_distance"],
        flow_k=cfg["flow_k"],
        passes=cfg["passes"],
    )
    if len(linked) == 0:
        return coords, np.empty((0, 4))

    if cfg["recover"]:
        # After re-linking, because a gap is only visible once the links are
        # chains; before pruning, because the fragments a gap-fill joins are
        # exactly the short ones pruning would throw away.
        #
        # Repeated, because filling a gap moves the assignment: a parent handed
        # to a recovered node frees the child that had taken it, and the ends
        # and starts that leaves behind are gaps the previous round could not
        # see. Iteration stops on its own once nothing new is proposed.
        for _ in range(max(int(cfg["recover_passes"]), 1)):
            coords, n_added = recover(
                coords, linked,
                max_residual=cfg["recover_max_residual"],
                max_gap=cfg["recover_max_gap"],
                min_dup_dist=cfg["recover_min_dup"],
            )
            if not n_added:
                break
            # The recovered nodes carry no links of their own; re-running the
            # assignment wires them in the same way every other frame pair was
            # decided, rather than by a probability invented here.
            linked = relink(
                coords, edges,
                max_distance=cfg["max_distance"],
                flow_k=cfg["flow_k"],
                passes=cfg["passes"],
            )

    coords, linked = _prune_short(coords, linked, cfg["min_track"])
    if len(linked) == 0:
        return coords, np.empty((0, 4))

    residual = _flow_residual(coords, linked, cfg["flow_k"])
    keep = (linked[:, 4] >= cfg["min_margin"]) & (residual <= cfg["max_residual"])
    return coords, linked[keep][:, :4]
