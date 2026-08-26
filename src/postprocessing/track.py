"""A tracker over the saved detections: model candidates, motion scoring.

The two signals fail in different places, which is the reason to combine them.

* The model knows appearance. Its candidate list is excellent — every one of
  the 85 ground-truth links sits in its top 12 — but its *ranking* puts the
  true parent second whenever it errs.
* Motion knows the field. The embryo slides bodily between frames and cells
  move with their neighbours, so a flow-corrected distance sorts candidates
  well; but on its own it proposes links the appearance model would never make.

So: take the candidate set from the model, score it with motion, and decide the
frame pair globally with an assignment that is allowed to leave cells unlinked.

Motion is predicted per cell, not just per neighbourhood: once a cell has a
link behind it, its own velocity says more about where it is going than the
median of its neighbours does. Frames are processed in order so that velocity
is available from the pass currently being built.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from params import SCALE

SCALE_ARR = np.asarray(SCALE)
FORBIDDEN = 1e6


def _local_flow(src_um, tgt_um, links, k=30):
    if len(links) == 0:
        return np.zeros_like(src_um)
    anchor = src_um[links[:, 0]]
    disp = tgt_um[links[:, 1]] - anchor
    k = min(k, len(links))
    _, near = cKDTree(anchor).query(src_um, k=k)
    return np.median(disp[near.reshape(len(src_um), -1)], axis=1)


def _solve(cost, no_link):
    """Assignment in which leaving a node unlinked costs ``no_link``."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.empty((0, 2), dtype=int)
    big = np.full((n + m, n + m), float(no_link))
    big[:n, :m] = np.minimum(cost, no_link)
    big[n:, m:] = 0.0
    rows, cols = linear_sum_assignment(big)
    ok = (rows < n) & (cols < m)
    rows, cols = rows[ok], cols[ok]
    keep = cost[rows, cols] < no_link
    return np.stack([rows[keep], cols[keep]], axis=1)


def track(
    coords,
    by_pair,
    seed_edges=None,
    top_k=12,
    model_weight=1.0,
    motion_weight=0.25,
    velocity_weight=0.6,
    no_link=2.5,
    max_distance=9.0,
    flow_k=30,
    passes=2,
    margin=0.0,
):
    """Link the whole video.

    Parameters
    ----------
    by_pair : {t: {child row: [(parent row, prob), ...]}}
        The model's candidates, from dump_edge_scores.
    top_k : int
        How far down the model's list to consider. Every ground-truth link is
        within 12, so more than that only adds ways to be wrong.
    model_weight, motion_weight : float
        Cost is ``model_weight * -log p + motion_weight * displacement_um``.
    velocity_weight : float
        How far to trust a cell's own velocity over its neighbourhood's flow
        when predicting where it went. 0 uses the neighbourhood alone.
    no_link : float
        Cost of leaving a cell unlinked. The metric charges two false positives
        for a wrong link against one true positive for a right one, so this
        wants to sit where a link is about 55% likely to be right.
    margin : float
        Drop assignments that beat the runner-up by less than this.

    Returns
    -------
    (m, 5) array of (source row, target row, prob, distance um, margin).
    """
    coords = np.asarray(coords)
    pos_um = coords[:, 1:].astype(float) * SCALE_ARR
    frames = sorted({int(t) for t in np.unique(coords[:, 0])})
    rows_by_t = {t: np.flatnonzero(coords[:, 0] == t) for t in frames}
    local_of = {t: {int(r): i for i, r in enumerate(rows)} for t, rows in rows_by_t.items()}

    # Candidate cost from the model, per frame pair, in local indices.
    model_cost, candidate = {}, {}
    for t in frames[:-1]:
        if t + 1 not in rows_by_t:
            continue
        n, m = len(rows_by_t[t]), len(rows_by_t[t + 1])
        cost = np.full((n, m), FORBIDDEN)
        for child, cands in by_pair.get(t, {}).items():
            j = local_of[t + 1].get(int(child))
            if j is None:
                continue
            for parent, prob in sorted(cands, key=lambda pp: -pp[1])[:top_k]:
                i = local_of[t].get(int(parent))
                if i is not None:
                    cost[i, j] = -np.log(max(prob, 1e-12))
        model_cost[t] = cost
        candidate[t] = cost < FORBIDDEN

    # Seed the flow with whatever links we already have — the tracker's own
    # output if given, otherwise the model's argmax.
    links = {}
    for t in model_cost:
        c, cand = model_cost[t], candidate[t]
        best = np.argmin(np.where(cand, c, np.inf), axis=0)
        ok = cand[best, np.arange(c.shape[1])]
        links[t] = np.stack([best[ok], np.flatnonzero(ok)], axis=1)
    if seed_edges is not None:
        seed_edges = np.asarray(seed_edges, dtype=float).reshape(-1, 4)
        seeded = {t: [] for t in model_cost}
        for s, tg in seed_edges[:, :2].astype(int):
            ts = int(coords[s, 0])
            if ts in seeded and int(coords[tg, 0]) == ts + 1:
                seeded[ts].append((local_of[ts][s], local_of[ts + 1][tg]))
        for t, pairs in seeded.items():
            if pairs:
                links[t] = np.asarray(pairs, dtype=int)

    out = None
    for _ in range(max(passes, 1)):
        new_links, records = {}, []
        velocity = {}
        for t in sorted(model_cost):
            here, there = rows_by_t[t], rows_by_t[t + 1]
            src_um, tgt_um = pos_um[here], pos_um[there]

            flow = _local_flow(src_um, tgt_um, links[t], k=flow_k)
            predicted = src_um + flow
            if velocity_weight and t in velocity:
                own, has = velocity[t]
                blended = velocity_weight * own + (1 - velocity_weight) * flow
                predicted = src_um + np.where(has[:, None], blended, flow)

            d = np.linalg.norm(predicted[:, None, :] - tgt_um[None, :, :], axis=-1)
            cost = np.where(
                candidate[t], model_weight * model_cost[t] + motion_weight * d, FORBIDDEN
            )
            cost[d > max_distance] = FORBIDDEN

            chosen = _solve(cost, no_link)
            new_links[t] = chosen

            rival = np.where(cost < no_link, cost, np.inf)
            if len(chosen):
                picked = cost[chosen[:, 0], chosen[:, 1]]
                rival[chosen[:, 0], chosen[:, 1]] = np.inf
                gap = np.minimum(
                    rival[:, chosen[:, 1]].min(axis=0), rival[chosen[:, 0], :].min(axis=1)
                ) - picked
                for (i, j), g in zip(chosen, gap):
                    records.append((
                        int(here[i]), int(there[j]),
                        float(np.exp(-model_cost[t][i, j])),
                        float(np.linalg.norm(pos_um[here[i]] - pos_um[there[j]])),
                        float(g),
                    ))
                # Carry each linked cell's own displacement into the next pair.
                own = np.zeros((len(there), 3))
                has = np.zeros(len(there), dtype=bool)
                own[chosen[:, 1]] = tgt_um[chosen[:, 1]] - src_um[chosen[:, 0]]
                has[chosen[:, 1]] = True
                velocity[t + 1] = (own, has)

        links = new_links
        out = np.asarray(records, dtype=float).reshape(-1, 5)

    if margin > 0:
        out = out[out[:, 4] >= margin]
    return out
