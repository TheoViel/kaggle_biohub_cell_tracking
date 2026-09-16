"""Put back the detections a track *ends* on, not the ones it skips over.

`recover.py` bridges: it needs a track that stops at ``u`` and another that
starts at ``u + L + 1``, and it interpolates between two observed endpoints.
That is the safe half of the problem and it is the smaller half. Measured over
199 out-of-fold stacks of `logs/2026-09-13/1/preds_r6_0.40`, of 7,490 ground
truth nodes the detector missed:

    both neighbours detected (a gap -- what `recover` can bridge)    800   10.7%
    exactly one neighbour detected (a track *end*)                 2,450   32.7%
    neither                                                        4,240   56.6%

So the one-sided misses outnumber the bridgeable ones three to one, and nothing
in the chain addresses them. This module does.

The premise is that a cell does not stop existing. A track that ends in the
middle of a video ended for one of three reasons: the cell left the imaged
volume, the cell divided or died, or the detector missed it. Only the last is
an error, and the first is the one that is cheap to rule out -- it is a
question about where the track ended, not about what happened next.

Three stages, each of which can be switched off:

1. **Global motion first.** The whole sample can slide bodily between frames --
   one video here moves a steady -1.6 um in Z every frame -- and over a single
   frame that is most of the gate a cell's own motion is judged against. Every
   displacement below is measured after `estimate_drift` removes the field
   translation, which is the decomposition into global plus individual motion
   that makes the rest meaningful.

2. **Re-link before inventing.** An orphaned parent at ``t`` and an unparented
   node at ``t+1`` that sit on top of each other once drift is removed are a
   *linking* failure, not a detection one. Those are matched by Hungarian
   assignment on the drift-corrected distances -- the cheapest possible repair,
   because the node already exists and costs nothing more in the node-count
   adjustment.

3. **Extrapolate what is left.** A parent still unmatched gets its own velocity
   extrapolated from the last few frames of its track, and a node is inserted
   where that says it should be. The velocity is fitted, not differenced, so
   one noisy position does not throw the prediction.

The metric makes this an asymmetric bet, which is why it is worth taking. A
recovered node that is right is worth up to two true-positive edges and removes
up to two false negatives; one that is wrong costs a node in ``N_pred`` -- about
1e-5 of the node-count adjustment -- plus whatever links it steals. The gates
below exist to stop the third stage running away, not because a single bad
insertion is expensive.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from model_zoo.simple_node_transformer import estimate_drift

DEFAULTS = dict(
    # Stage 2: how far apart a parent and a free child may be, after drift
    # removal, and still be called the same cell. The linker's own gate is
    # 7 um; this is deliberately tighter, because a pair this obvious is one
    # the linker should already have found and a looser gate here would be
    # inventing links rather than repairing them.
    relink_um=5.0,
    # Stage 3: the extrapolation.
    extrapolate=1,
    # Frames of history the velocity is fitted over. Below `min_history` the
    # track is too short to extrapolate from and is left alone -- a two-node
    # track's "velocity" is one noisy difference.
    fit_window=4,
    min_history=3,
    # Reject the insertion if an existing detection is already this close: the
    # cell is there, the link is what is missing, and stage 2 either found it
    # or had a reason not to.
    min_dup_um=4.0,
    # How far the fitted velocity is allowed to say the cell went in one frame.
    # p99 of the true frame-to-frame displacement is 7.6 um; past that the fit
    # is describing a linking error upstream, not a cell.
    max_step_um=8.0,
    # Residual of the velocity fit, in um. A track whose own history the line
    # does not explain is not one to extrapolate from.
    max_fit_residual_um=3.0,
    # Consecutive frames a single track may be extrapolated through. One is the
    # honest default: with no evidence on the far side, every further step
    # compounds the last one's error.
    max_consecutive=1,
    # Volume margin, in um. A track that ends this close to the edge of the
    # imaged volume ended for a legitimate reason and is left alone.
    boundary_um=10.0,
    # Snap to a sub-threshold detection instead of inventing a coordinate.
    #
    # Inventing one is what the first version did, and it is wasteful: the
    # detector has already looked at that voxel and reported what it saw, just
    # below the threshold the sweep committed to. `preds_r6_0.30` holds the
    # detections `0.40` discarded. A candidate that the detector proposed *and*
    # that an independent track extrapolation lands on is a far better bet than
    # a coordinate computed from the track alone -- two sources agreeing rather
    # than one.
    #
    # `snap_um` is how far the extrapolation may reach for such a candidate.
    # `require_pool` refuses to insert at all when none is found, which is the
    # precision/recall knob: the measured problem is not finding the misses but
    # the hit rate per insertion.
    snap_um=4.0,
    require_pool=1,
    # Stage 4: the link that went to the wrong cell.
    #
    # Tracing the three misses on 44b6_0b24845f killed the assumption stages 2
    # and 3 rest on. The parent of the missed cell was detected and was *not*
    # orphaned -- it had a child, 8.1 um away, because with the true child
    # missing the assignment gave it the nearest impostor instead. A missing
    # detection does not always end a track; more often it bends one.
    #
    # So the signal is the cost of the match, not its absence. After drift
    # removal a parent matched further than `suspicious_um` is one whose real
    # child is probably missing: inserting a node where the track says it
    # should be lets that parent match at ~0 and frees the impostor to start
    # its own track, which is the "total assignment cost falls when a good
    # candidate is added" test. p90 of true frame-to-frame displacement is
    # 3.6 um and p99 is 7.6, so 6.0 sits between "moved" and "implausible".
    suspicious=1,
    suspicious_um=6.0,
    # A mis-link costs the next frame its parent, so misses arrive in runs --
    # t=39 then t=40 on that stack. Nodes inserted at t+1 are therefore made
    # visible when frame t+1 is processed, so a recovered cell can itself be
    # extrapolated from.
    chain=1,
)


def _fit_velocity(track_um: np.ndarray, max_window: int):
    """Least-squares velocity over a track's last frames, and its residual.

    Returns ``(velocity, residual)`` in um per frame. Fitted rather than
    differenced: the last two positions are each one detection away from being
    wrong, and a line through four of them is not.
    """
    pts = track_um[-max_window:]
    n = len(pts)
    t = np.arange(n, dtype=np.float64)
    t = t - t.mean()
    denom = float((t * t).sum())
    if n < 2 or denom == 0:
        return np.zeros(3), 0.0
    vel = (t[:, None] * (pts - pts.mean(axis=0))).sum(axis=0) / denom
    pred = pts.mean(axis=0) + t[:, None] * vel
    residual = float(np.linalg.norm(pts - pred, axis=1).mean())
    return vel, residual


def _drift(coords, rows_t, rows_t1):
    """Field translation from frame t to t+1, in grid voxels."""
    if not len(rows_t) or not len(rows_t1):
        return np.zeros(3)
    a = torch.from_numpy(coords[rows_t, 1:].astype(np.float32))[None]
    b = torch.from_numpy(coords[rows_t1, 1:].astype(np.float32))[None]
    return estimate_drift(a, b, None, None).numpy()[0].astype(np.float64)


def terminal_recover(coords, edges, scale, cfg=None, stats=None, pool=None):
    """``(coords, edges)`` with terminal misses re-linked or inserted.

    ``coords`` is (N, 4) ``[t, z, y, x]`` in grid voxels and ``scale`` converts
    a voxel to microns per axis, so every gate below is physical.

    ``pool`` is an optional (M, 4) array of sub-threshold candidates -- a
    lower-threshold prediction of the same run -- that stage 3 snaps to instead
    of inventing a position. See ``snap_um`` / ``require_pool``.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 4)
    edges = np.asarray(edges, dtype=np.float64).reshape(-1, 4)
    if not len(coords) or not len(edges):
        return coords, edges
    s = np.asarray(scale, dtype=np.float64)

    n = len(coords)
    frames = coords[:, 0].astype(int)
    t_max = int(frames.max())
    rows_by_t = {t: list(np.flatnonzero(frames == t)) for t in range(t_max + 1)}
    live = [coords]          # grows as nodes are inserted; see `chain`

    def pos(i):
        """Position of node ``i``, original or inserted."""
        return (coords[i, 1:] if i < n else extra[i - n, 1:])

    # Where the volume ends, from the detections themselves: a track that stops
    # at the edge of the imaged region stopped for a reason that is not a miss.
    lo = coords[:, 1:].min(axis=0) * s
    hi = coords[:, 1:].max(axis=0) * s

    pool_trees: dict[int, tuple] = {}
    if pool is not None and len(pool):
        pool = np.asarray(pool, dtype=np.float64).reshape(-1, 4)
        pf = pool[:, 0].astype(int)
        for t_ in np.unique(pf):
            rows = np.flatnonzero(pf == t_)
            pool_trees[int(t_)] = (cKDTree(pool[rows, 1:] * s), rows)

    parent_of = {}
    children = {i: [] for i in range(n)}
    for a, b, *_ in edges:
        children[int(a)].append(int(b))
        parent_of[int(b)] = int(a)

    def history(i):
        """The track ending at node ``i``, oldest first, in um."""
        ch = [i]
        while len(ch) < cfg["fit_window"] and ch[-1] in parent_of:
            ch.append(parent_of[ch[-1]])
        pts = np.stack([pos(k) for k in ch[::-1]]) * s
        return pts, len(ch)

    extra = np.zeros((0, 4))
    new_coords = [coords]
    new_edges = [edges]
    added = 0
    relinked = 0
    extrapolated = 0
    snapped_n = 0
    drop_edges: set[tuple[int, int]] = set()
    next_id = n
    # How many frames in a row each track has been carried without evidence.
    carried: dict[int, int] = {}

    for t in range(t_max):
        rows_t, rows_t1 = rows_by_t.get(t, []), rows_by_t.get(t + 1, [])
        if not len(rows_t):
            continue
        rows_t = np.asarray(rows_t, int); rows_t1 = np.asarray(rows_t1, int)
        all_pos = np.concatenate([coords, extra]) if len(extra) else coords
        drift = _drift(all_pos, rows_t, rows_t1) if len(rows_t1) else np.zeros(3)

        orphans = [int(i) for i in rows_t if not children.get(int(i))]
        # Stage 4: a parent whose one child sits implausibly far away, once the
        # field translation is removed, is treated exactly like an orphan --
        # its real child is missing, and the link it has is the symptom.
        if cfg["suspicious"]:
            for i in rows_t:
                i = int(i)
                kids = children.get(i, [])
                if len(kids) != 1:
                    continue                    # a division is not a mis-link
                d_um = np.linalg.norm(((pos(i) + drift) - pos(kids[0])) * s)
                if d_um > cfg["suspicious_um"]:
                    orphans.append(i)
        if not orphans:
            continue
        # Away from the boundary only.
        keep = []
        for i in orphans:
            p = pos(i) * s
            if np.all(p - lo > cfg["boundary_um"]) and np.all(hi - p > cfg["boundary_um"]):
                keep.append(i)
        orphans = keep
        if not orphans:
            continue

        # --- stage 2: an existing node with no parent is the cheapest repair ---
        free = [int(j) for j in rows_t1 if j not in parent_of] if len(rows_t1) else []
        matched: set[int] = set()
        if free:
            P = (np.stack([pos(i) for i in orphans]) + drift) * s
            C = np.stack([pos(j) for j in free]) * s
            cost = np.linalg.norm(P[:, None, :] - C[None, :, :], axis=-1)
            big = cfg["relink_um"] * 10.0
            gated = np.where(cost <= cfg["relink_um"], cost, big)
            ri, ci = linear_sum_assignment(gated)
            for a, b in zip(ri, ci):
                if gated[a, b] < big:
                    i, j = orphans[a], free[b]
                    new_edges.append(np.array([[i, j, 0.5, float(cost[a, b])]]))
                    parent_of[j] = i
                    children[i].append(j)
                    matched.add(i)
                    relinked += 1

        if not cfg["extrapolate"]:
            continue

        # --- stage 3: nothing to link to, so put the detection back ---
        tree = (cKDTree(np.stack([pos(j) for j in rows_t1]) * s)
                if len(rows_t1) else None)
        for i in orphans:
            if i in matched:
                continue
            if carried.get(i, 0) >= cfg["max_consecutive"]:
                continue
            track_um, seen = history(i)
            if seen < cfg["min_history"]:
                continue
            vel, residual = _fit_velocity(track_um, cfg["fit_window"])
            if residual > cfg["max_fit_residual_um"]:
                continue
            # The cell's own motion plus the field's. `vel` is fitted on
            # positions that already contain past drift, so this is an
            # approximation, and `max_step_um` is what keeps it honest.
            step = vel + drift * s
            if np.linalg.norm(step) > cfg["max_step_um"]:
                continue
            p_um = pos(i) * s + step

            # Prefer a detection the model actually made near the prediction.
            snapped = False
            if cfg["snap_um"] > 0 and (t + 1) in pool_trees:
                ptree, prows = pool_trees[t + 1]
                d_pool, k_pool = ptree.query(p_um[None], k=1)
                if float(d_pool[0]) <= cfg["snap_um"]:
                    p_um = pool[prows[int(k_pool[0])], 1:] * s
                    snapped = True
            if cfg["require_pool"] and not snapped:
                continue

            if tree is not None and len(rows_t1):
                d, _ = tree.query(p_um[None], k=1)
                if float(d[0]) < cfg["min_dup_um"]:
                    continue                     # the cell is there; see stage 2
            if np.any(p_um - lo < 0) or np.any(hi - p_um < 0):
                continue

            row = np.empty((1, 4))
            row[0, 0] = t + 1
            row[0, 1:] = p_um / s
            new_coords.append(row)
            extra = np.concatenate([extra, row])
            # The mis-link is now wrong twice over -- the parent has its real
            # child back, and the impostor is somebody else's cell. Drop it.
            for bad in list(children.get(i, [])):
                drop_edges.add((i, bad))
                parent_of.pop(bad, None)
            children[i] = []
            if cfg["chain"]:
                rows_by_t.setdefault(t + 1, []).append(next_id)
            new_edges.append(np.array([[i, next_id, 0.5, float(np.linalg.norm(step))]]))
            children[i].append(next_id)
            parent_of[next_id] = i
            carried[next_id] = carried.get(i, 0) + 1
            children[next_id] = []
            next_id += 1
            added += 1
            extrapolated += 1
            snapped_n += int(snapped)

    out_coords = np.concatenate(new_coords) if len(new_coords) > 1 else coords
    out_edges = np.concatenate(new_edges) if len(new_edges) > 1 else edges
    if drop_edges:
        keep = np.array([(int(a), int(b)) not in drop_edges
                         for a, b, *_ in out_edges], bool)
        out_edges = out_edges[keep]
    if stats is not None:
        stats.update(relinked=relinked, extrapolated=extrapolated, added=added,
                     snapped=snapped_n)
    return out_coords, out_edges
