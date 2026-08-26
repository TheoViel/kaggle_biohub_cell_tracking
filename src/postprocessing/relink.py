"""Re-link the detections with a one-to-one assignment under estimated flow.

Why this shape:

* The ground truth is one-to-one — in this sample no annotated cell has two
  children or two parents — while the tracker emits up to two children per
  node. Every extra child is a guaranteed false positive.
* The tracker picks each child's parent greedily and independently, so two
  children can claim the same parent and a cell can be handed a parent that
  another cell had a much better claim on.
* Where the tracker was wrong, the true parent is essentially always the
  *second* nearest candidate once the frame's flow is removed (rank 1 in 13 of
  17 cases, within the top 5 in all 17). The right answer is reachable; it just
  needs a global decision rather than a per-child one.

So: estimate the field's motion from the links themselves, correct for it, and
solve a rectangular assignment per frame pair.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from params import SCALE

SCALE_ARR = np.asarray(SCALE)
NO_LINK = 1e6


def local_flow(src_um, tgt_um, links, k=30):
    """Per-source displacement, taken as the median of its k nearest links.

    Cells sit in tissue and move with their neighbours, so a cell's own nearest
    match is a far noisier estimate of where it went than its neighbourhood's
    consensus. The median keeps a handful of wrong links from dragging it.
    """
    flow = np.zeros_like(src_um)
    if len(links) == 0:
        return flow
    anchor = src_um[links[:, 0]]
    disp = tgt_um[links[:, 1]] - anchor
    k = min(k, len(links))
    _, near = cKDTree(anchor).query(src_um, k=k)
    return np.median(disp[near.reshape(len(src_um), -1)], axis=1)


def assign_pair(
    src_um,
    tgt_um,
    seed_links,
    prob=None,
    max_distance=9.0,
    prob_weight=0.0,
    passes=3,
    flow_k=30,
):
    """One-to-one links between two frames, in local (row) indices.

    Parameters
    ----------
    seed_links : (n, 2) int
        Links to seed the flow estimate — the tracker's own output.
    prob : dict or None
        (source row, target row) -> model probability, for the candidate edges
        the tracker scored. Where given, ``prob_weight`` * prob is subtracted
        from the assignment cost so a link the model liked survives a slightly
        worse geometric fit.
    max_distance : float
        Flow-corrected distance, in microns, beyond which a pair is not a
        candidate at all. Cells do appear and disappear, so leaving a cell
        unlinked has to remain an option.
    """
    if len(src_um) == 0 or len(tgt_um) == 0:
        return np.empty((0, 2), dtype=int)

    links = np.asarray(seed_links, dtype=int).reshape(-1, 2)
    for _ in range(max(passes, 1)):
        flow = local_flow(src_um, tgt_um, links, k=flow_k)
        cost = np.linalg.norm((src_um + flow)[:, None, :] - tgt_um[None, :, :], axis=-1)
        gated = cost > max_distance
        work = cost.copy()
        if prob_weight and prob:
            for (i, j), p in prob.items():
                work[i, j] -= prob_weight * p
        # A forbidden pair must never be chosen, but the solver needs a finite
        # square-ish matrix; a cost no assignment would prefer does the job.
        work[gated] = NO_LINK
        rows, cols = linear_sum_assignment(work)
        keep = ~gated[rows, cols]
        links = np.stack([rows[keep], cols[keep]], axis=1)

    # How clearly the winner won, per link: the gap to the child's next best
    # parent. A link chosen by a hair is the one the tracker gets wrong — the
    # true parent, where we err, is almost always the runner-up.
    margin = np.full(len(links), np.inf)
    if len(links):
        chosen = cost[links[:, 0], links[:, 1]]
        rival = cost.copy()
        rival[links[:, 0], links[:, 1]] = np.inf
        best_other_parent = rival[:, links[:, 1]].min(axis=0)
        best_other_child = rival[links[:, 0], :].min(axis=1)
        margin = np.minimum(best_other_parent, best_other_child) - chosen
    return links, margin


def relink(coords, edges, max_distance=9.0, prob_weight=0.0, passes=3, flow_k=30):
    """Re-link a whole video. Returns edges as (src row, tgt row, prob, dist).

    Node coordinates are left untouched: this decides links only.
    """
    coords = np.asarray(coords)
    pos_um = coords[:, 1:].astype(float) * SCALE_ARR
    rows_by_t = {int(t): np.flatnonzero(coords[:, 0] == t) for t in np.unique(coords[:, 0])}

    edges = np.asarray(edges, dtype=float).reshape(-1, 4)
    src_rows = edges[:, 0].astype(int)
    tgt_rows = edges[:, 1].astype(int)

    out = []
    for t in sorted(rows_by_t):
        if t + 1 not in rows_by_t:
            continue
        here, there = rows_by_t[t], rows_by_t[t + 1]
        local_src = {int(r): i for i, r in enumerate(here)}
        local_tgt = {int(r): i for i, r in enumerate(there)}

        mask = np.isin(src_rows, here) & np.isin(tgt_rows, there)
        idx = np.flatnonzero(mask)
        seed = np.array(
            [[local_src[src_rows[i]], local_tgt[tgt_rows[i]]] for i in idx], dtype=int
        ).reshape(-1, 2)
        prob = {
            (local_src[src_rows[i]], local_tgt[tgt_rows[i]]): edges[i, 2] for i in idx
        }

        links, margin = assign_pair(
            pos_um[here], pos_um[there], seed, prob=prob,
            max_distance=max_distance, prob_weight=prob_weight,
            passes=passes, flow_k=flow_k,
        )
        for (i, j), m in zip(links, margin):
            gi, gj = int(here[i]), int(there[j])
            out.append((
                gi, gj,
                float(prob.get((int(i), int(j)), 0.5)),
                float(np.linalg.norm(pos_um[gi] - pos_um[gj])),
                float(m),
            ))
    return np.asarray(out, dtype=float).reshape(-1, 5)
