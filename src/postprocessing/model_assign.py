"""Global one-to-one assignment over the model's own candidate scores.

The tracker hands every child its most likely parent independently. That is the
wrong decision rule for a problem where parents are a scarce resource: two
children can take the same parent, and a child can take a parent another child
had a far stronger claim to. Solving the frame pair as an assignment problem
puts that right, and the model's runner-up — which is where the true parent sits
whenever the argmax is wrong — becomes reachable.

Costs are -log p, so the assignment maximises the product of the chosen edges'
probabilities. A flow term can be mixed in: the model scores a pair on
appearance and local geometry, and knows nothing about the fact that the whole
field slides between frames.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from params import SCALE

SCALE_ARR = np.asarray(SCALE)
FORBIDDEN = 1e6


def load_candidates(path):
    """(t, child row, parent row, prob) rows -> per frame pair, per child."""
    cand = np.load(path)["candidates"]
    by_pair = defaultdict(lambda: defaultdict(list))
    for t, child, parent, prob in cand:
        by_pair[int(t)][int(child)].append((int(parent), float(prob)))
    return by_pair


def _local_flow(src_um, tgt_um, links, k=30):
    if len(links) == 0:
        return np.zeros_like(src_um)
    anchor = src_um[links[:, 0]]
    disp = tgt_um[links[:, 1]] - anchor
    k = min(k, len(links))
    _, near = cKDTree(anchor).query(src_um, k=k)
    return np.median(disp[near.reshape(len(src_um), -1)], axis=1)


def _solve(cost, no_link):
    """Assignment where leaving a node unlinked is an option, priced at no_link.

    A plain rectangular solve must match every row, which here means a spurious
    detection — and there are more of those than real cells — can take the
    parent a real cell needed. Padding both sides with dummies that cost
    ``no_link`` lets a node go unmatched whenever no real partner beats that
    price, which is also exactly the decision rule the metric wants: a wrong
    link costs two false positives, so a coin-toss link is worse than none.
    """
    n, m = cost.shape
    big = np.full((n + m, n + m), no_link, dtype=float)
    big[:n, :m] = np.minimum(cost, no_link)
    big[n:, m:] = 0.0
    rows, cols = linear_sum_assignment(big)
    keep = (rows < n) & (cols < m) & (cost[rows % max(n, 1), cols % max(m, 1)] < no_link)
    return rows[keep], cols[keep]


def assign(
    coords,
    by_pair,
    flow_weight=0.0,
    max_distance=np.inf,
    min_prob=0.0,
    margin=0.0,
    no_link=np.inf,
    flow_k=30,
    passes=2,
):
    """Assign parents to children over the whole video.

    Parameters
    ----------
    flow_weight : float
        Microns of flow-corrected displacement traded against one nat of
        log-probability. 0 uses the model alone.
    max_distance : float
        Flow-corrected distance gate, in microns. Applied only when
        ``flow_weight`` is in use, since without a flow estimate the raw
        distance says little.
    min_prob : float
        Candidates below this probability are not considered at all.
    margin : float
        Decline any assignment whose cost advantage over the child's next best
        parent is smaller than this, in cost units. A ground-truth link we get
        wrong costs two false positives against one true positive gained, so
        near-ties are better left unlinked than guessed.

    Returns
    -------
    (m, 5) array of (source row, target row, prob, distance um, margin).
    """
    coords = np.asarray(coords)
    pos_um = coords[:, 1:].astype(float) * SCALE_ARR
    rows_by_t = {int(t): np.flatnonzero(coords[:, 0] == t) for t in np.unique(coords[:, 0])}

    out = []
    for t in sorted(by_pair):
        if t not in rows_by_t or t + 1 not in rows_by_t:
            continue
        here, there = rows_by_t[t], rows_by_t[t + 1]
        local_src = {int(r): i for i, r in enumerate(here)}
        local_tgt = {int(r): i for i, r in enumerate(there)}
        src_um, tgt_um = pos_um[here], pos_um[there]

        model_cost = np.full((len(here), len(there)), FORBIDDEN)
        for child, cands in by_pair[t].items():
            j = local_tgt.get(int(child))
            if j is None:
                continue
            for parent, prob in cands:
                i = local_src.get(int(parent))
                if i is not None and prob >= min_prob:
                    model_cost[i, j] = -np.log(max(prob, 1e-12))
        candidate = model_cost < FORBIDDEN
        if not candidate.any():
            continue

        # Seed the flow from the model's own argmax, then iterate: each round
        # re-estimates the field's motion from the links just chosen.
        links = np.stack(
            [np.argmin(np.where(candidate, model_cost, np.inf), axis=0),
             np.arange(len(there))], axis=1,
        )
        links = links[candidate[links[:, 0], links[:, 1]]]

        barrier = min(no_link, FORBIDDEN)
        for _ in range(max(passes, 1) if flow_weight else 1):
            cost = model_cost.copy()
            if flow_weight:
                flow = _local_flow(src_um, tgt_um, links, k=flow_k)
                d = np.linalg.norm((src_um + flow)[:, None, :] - tgt_um[None, :, :], axis=-1)
                cost = np.where(candidate, model_cost + flow_weight * d, FORBIDDEN)
                cost[d > max_distance] = FORBIDDEN
            rows, cols = _solve(cost, barrier)
            links = np.stack([rows, cols], axis=1).reshape(-1, 2)

        if len(links) == 0:
            continue

        chosen = cost[links[:, 0], links[:, 1]]
        rival = np.where(cost < barrier, cost, np.inf)
        rival[links[:, 0], links[:, 1]] = np.inf
        gap = np.minimum(
            rival[:, links[:, 1]].min(axis=0), rival[links[:, 0], :].min(axis=1)
        ) - chosen
        for (i, j), g in zip(links, gap):
            if g < margin:
                continue
            gi, gj = int(here[i]), int(there[j])
            out.append((
                gi, gj,
                float(np.exp(-model_cost[i, j])),
                float(np.linalg.norm(pos_um[gi] - pos_um[gj])),
                float(g),
            ))

    return np.asarray(out, dtype=float).reshape(-1, 5)
