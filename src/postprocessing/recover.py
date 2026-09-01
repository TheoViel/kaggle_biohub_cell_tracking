"""Recover the detections a track is missing, from the frames on either side.

The detector drops a cell for a frame or two fairly often — at threshold 0.25
node recall over the 71 validation stacks is 0.984, and every missed node costs
*two* ground-truth edges: the link into it and the link out of it. Nothing
downstream can repair that, because the node the links need does not exist.

So: find a track that ends at frame ``u`` and one that begins at ``u + L + 1``,
decide they are the same cell, and put ``L`` nodes back in between.

The decision has to be made on *drift-corrected* displacement. A sample can
slide bodily between frames — one video here moves a steady -1.6 um in Z every
frame — and over a two-frame gap that is 3.2 um of pure translation before the
cell has moved at all, against a gate of the same order. ``estimate_drift``
(model_zoo/simple_node_transformer.py) takes the whole-field translation out by
matching each child to its nearest parent and taking the median offset, which
is what makes the remaining displacement the cell's own motion.

The metric makes this a cheap bet. A recovered node that is right is worth up
to two true positives and removes up to two false ones; a recovered node that
is wrong adds a node to ``N_pred``, worth about 1e-5 of the node-count
adjustment, plus whatever links it steals. See postprocessing/README.md.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from model_zoo.simple_node_transformer import estimate_drift
from params import SCALE

SCALE_ARR = np.asarray(SCALE)

DEFAULTS = dict(
    max_residual=5.0,      # um, gate on the drift-corrected two-frame displacement
    max_gap=3,             # longest run of missing frames to fill
    min_dup_dist=4.0,      # um; do not insert next to a detection that already exists
    require_history=True,  # both ends must be part of a track, not isolated nodes
    drift_iter=3,
    drift_refine=True,
)


def _rows_by_t(coords):
    if len(coords) == 0:
        return {}
    t = coords[:, 0].astype(int)
    order = np.argsort(t, kind="stable")
    t_sorted = t[order]
    bounds = np.searchsorted(t_sorted, np.arange(t_sorted[0], t_sorted[-1] + 2))
    return {
        int(tt): order[bounds[i]:bounds[i + 1]]
        for i, tt in enumerate(range(int(t_sorted[0]), int(t_sorted[-1]) + 1))
    }


def _refine_drift(src_um, tgt_um, drift, tol=2.0):
    """Sharpen the median estimate to the mean of its inliers.

    ``estimate_drift`` returns a median over nearest-neighbour offsets, so on a
    grid whose step is 1.625 um in z the answer is quantised to about that —
    a quarter of the matching radius. Averaging the offsets that agree with it
    recovers the sub-voxel part without letting the disagreeing ones vote.
    """
    if len(src_um) == 0 or len(tgt_um) == 0:
        return drift
    _, nn = cKDTree(src_um).query(tgt_um - drift, k=1)
    offset = tgt_um - src_um[nn]
    inlier = np.linalg.norm(offset - drift, axis=1) <= tol
    return offset[inlier].mean(axis=0) if inlier.sum() >= 5 else drift


def drift_table(coords, n_iter=3, refine=True):
    """``D[t]``: the whole-field translation in um carrying frame t onto t+1.

    Frames the estimator cannot speak for (either side empty) fall back to the
    previous frame's answer, which beats the zero it would otherwise return.
    """
    pos_um = coords[:, 1:].astype(np.float64) * SCALE_ARR
    rows = _rows_by_t(coords)
    frames = sorted(rows)
    drift = {}
    last = np.zeros(3)
    for t in frames[:-1]:
        src = pos_um[rows.get(t, np.empty(0, int))]
        tgt = pos_um[rows.get(t + 1, np.empty(0, int))]
        if len(src) == 0 or len(tgt) == 0:
            drift[t] = last
            continue
        d = estimate_drift(
            torch.from_numpy(src).unsqueeze(0),
            torch.from_numpy(tgt).unsqueeze(0),
            None,
            None,
            n_iter=n_iter,
        )[0].numpy()
        if refine:
            d = _refine_drift(src, tgt, d)
        drift[t] = last = d
    return drift


def find_gaps(coords, links, drift=None, **kw):
    """Chain ends and starts ``L + 1`` frames apart that are plausibly one cell.

    Gap lengths are tried shortest first, and an end or a start consumed by a
    short gap is not offered to a longer one: where a one-frame explanation
    exists it is the better one, and letting a two-frame bridge swallow the same
    endpoints would hide it.

    Returns ``(pairs, positions)`` — ``pairs`` is (n, 2) rows into ``coords``
    (the parent, then the child), ``positions`` a list of (L, 3) arrays in um,
    one row per frame from ``u + 1`` to ``u + L``.
    """
    cfg = {**DEFAULTS, **kw}
    coords = np.asarray(coords, dtype=np.float64)
    links = np.asarray(links, dtype=float)
    links = links.reshape(-1, links.shape[-1]) if links.size else np.empty((0, 5))
    pos_um = coords[:, 1:] * SCALE_ARR
    rows = _rows_by_t(coords)
    if drift is None:
        drift = drift_table(coords, n_iter=cfg["drift_iter"], refine=cfg["drift_refine"])

    src_rows = links[:, 0].astype(int)
    tgt_rows = links[:, 1].astype(int)
    has_child = np.zeros(len(coords), dtype=bool)
    has_parent = np.zeros(len(coords), dtype=bool)
    has_child[src_rows] = True
    has_parent[tgt_rows] = True
    used = np.zeros(len(coords), dtype=bool)

    pairs, positions = [], []
    for gap in range(1, int(cfg["max_gap"]) + 1):
        # The gate is a speed, not a distance: a cell crossing a two-frame gap
        # has had twice as long to move as one crossing a single frame.
        limit = cfg["max_residual"] * (gap + 1) / 2.0

        for u in sorted(rows):
            span = list(range(u, u + gap + 1))          # the frame steps crossed
            if u + gap + 1 not in rows or any(t not in drift for t in span):
                continue
            # Where the field itself goes, step by step and in total.
            step = np.cumsum([drift[t] for t in span], axis=0)   # (gap+1, 3)
            total = step[-1]

            ends = rows[u][~has_child[rows[u]] & ~used[rows[u]]]
            tgt = rows[u + gap + 1]
            starts = tgt[~has_parent[tgt] & ~used[tgt]]
            if cfg["require_history"]:
                # An isolated node is a spurious detection more often than the
                # end of a track, and bridging two of them invents a whole one.
                ends = ends[has_parent[ends]]
                starts = starts[has_child[starts]]
            if len(ends) == 0 or len(starts) == 0:
                continue

            # What is left after the field's own translation: the cell's own
            # motion, which is what the gate is about.
            residual = np.linalg.norm(
                pos_um[starts][None, :, :] - (pos_um[ends] + total)[:, None, :],
                axis=-1,
            )
            gated = residual > limit
            if gated.all():
                continue
            work = np.where(gated, 1e6, residual)
            ri, ci = linear_sum_assignment(work)
            keep = ~gated[ri, ci]
            ri, ci = ri[keep], ci[keep]

            for i, j in zip(ri, ci):
                a, b = int(ends[i]), int(starts[j])
                # Straight line in the frame the drift is removed in, then the
                # drift put back: with nothing but two endpoints, an even share
                # of the leftover motion per frame is the honest guess.
                motion = (pos_um[b] - pos_um[a]) - total
                pts = np.stack([
                    pos_um[a] + step[k - 1] + motion * k / (gap + 1)
                    for k in range(1, gap + 1)
                ])

                # A cell that was detected in between was not missed; if its
                # links are wrong that is relink's problem, and a node on top of
                # another one only steals the ground-truth match. One collision
                # sinks the whole bridge — a shorter gap is the right reading.
                if cfg["min_dup_dist"] > 0 and _collides(
                    pts, rows, pos_um, u, gap, cfg["min_dup_dist"]
                ):
                    continue

                used[a] = used[b] = True
                pairs.append((a, b))
                positions.append(pts)

    if not pairs:
        return np.empty((0, 2), dtype=int), []
    return np.asarray(pairs, dtype=int), positions


def _collides(pts, rows, pos_um, u, gap, min_dist):
    for k in range(1, gap + 1):
        here = rows.get(u + k)
        if here is None or len(here) == 0:
            continue
        if np.linalg.norm(pos_um[here] - pts[k - 1], axis=1).min() <= min_dist:
            return True
    return False


def recover(coords, links, **kw):
    """Append the nodes every gap found is missing. Links are not touched.

    Returns ``(coords, n_added)``. The new rows go on the end, so every index
    into the old ``coords`` still means what it did. Re-run ``relink`` after
    this to wire the new nodes in — deciding links is its job, and it will do it
    with the same assignment the rest of the video got, instead of a synthetic
    probability invented here.
    """
    coords = np.asarray(coords, dtype=np.float64)
    pairs, positions = find_gaps(coords, links, **kw)
    if len(pairs) == 0:
        return coords, 0
    new = []
    for (a, _b), pts in zip(pairs, positions):
        t0 = coords[a, 0]
        for k, p in enumerate(pts, start=1):
            new.append(np.concatenate([[t0 + k], p / SCALE_ARR]))
    return np.vstack([coords, np.asarray(new)]), len(new)
