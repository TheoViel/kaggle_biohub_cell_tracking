"""Per-link features and labels, from a saved npz alone.

The links this describes are the ones the re-linker proposes — those are the
decisions the gate has to make. For each one we want everything that might say
whether it is right, without using the annotation.

Labels come from the annotation and exist only for the links the *metric* can
see: a link is scored only when its source matched a ground-truth node that has
a child, or its target matched one that has a parent. That is a small subset —
about a hundred links per stack out of tens of thousands — but it is exactly the
population the score is computed over, so training on it is not a distribution
shift. Links the metric cannot see are unlabelled, and dropping them changes
nothing, so the gate is free to apply the same rule to them.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from fast_score import GroundTruth, edge_flags, match_nodes
from params import SCALE

SCALE_ARR = np.asarray(SCALE)

FEATURE_NAMES = [
    "model_prob",         # the tracker's own probability for this exact pair
    "has_model_prob",     # 0 when the re-linker invented the pair
    "margin_um",          # how far the assignment's winner beat the runner-up
    "residual_um",        # departure from the neighbourhood's motion
    "dist_um",            # raw displacement
    "dist_over_flow",     # displacement relative to the frame's median motion
    "rank",               # rank of this parent by flow-corrected distance
    "rank_frac",          # the same, normalised by the number of candidates
    "n_rivals_6um",       # other children within 6um of this parent
    "src_nn_um",          # distance to the source's nearest neighbour: crowding
    "tgt_nn_um",
    "len_before",         # nodes in the track upstream of the source
    "len_after",          # nodes downstream of the target
    "track_len",          # the whole track this link sits in
    "vel_change_um",      # |this step - the source's previous step|
    "vel_cos",            # cosine between them; 0 when the source has no history
    "has_velocity",
    "frame_nodes",        # detections in the source frame
    "t_frac",             # position in the video
]


def _tracks(n_nodes, edges):
    """Union-find over a one-to-one graph: component id per node."""
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


def link_features(coords, npz_edges, linked):
    """(n_links, n_features) for the re-linker's output.

    Parameters
    ----------
    coords : (N, 4) int   [t, z, y, x]
    npz_edges : (M, 4)    the tracker's own edges, for the model probability
    linked : (L, 5)       the re-linker's output: src, tgt, prob, dist, margin
    """
    coords = np.asarray(coords)
    linked = np.asarray(linked, dtype=float).reshape(-1, 5)
    if len(linked) == 0:
        return np.empty((0, len(FEATURE_NAMES)))

    pos_um = coords[:, 1:].astype(float) * SCALE_ARR
    src = linked[:, 0].astype(int)
    tgt = linked[:, 1].astype(int)
    n_nodes = len(coords)

    # --- the tracker's own probability, where it had an opinion on this pair ---
    model_prob = {}
    for a, b, p, _ in np.asarray(npz_edges, dtype=float).reshape(-1, 4):
        model_prob[(int(a), int(b))] = float(p)
    prob = np.array([model_prob.get((int(a), int(b)), 0.0) for a, b in zip(src, tgt)])
    has_prob = np.array([(int(a), int(b)) in model_prob for a, b in zip(src, tgt)], float)

    # --- per-frame quantities ---
    rows_by_t = {int(t): np.flatnonzero(coords[:, 0] == t) for t in np.unique(coords[:, 0])}
    src_t = coords[src, 0].astype(int)

    rank = np.zeros(len(linked))
    rank_frac = np.zeros(len(linked))
    residual = np.zeros(len(linked))
    dist_over_flow = np.ones(len(linked))
    n_rivals = np.zeros(len(linked))
    src_nn = np.zeros(len(linked))
    tgt_nn = np.zeros(len(linked))
    frame_nodes = np.zeros(len(linked))

    for t in np.unique(src_t):
        idx = np.flatnonzero(src_t == t)
        here, there = rows_by_t[t], rows_by_t.get(t + 1)
        if there is None:
            continue
        src_um, tgt_um = pos_um[here], pos_um[there]
        local_src = {int(r): i for i, r in enumerate(here)}
        local_tgt = {int(r): i for i, r in enumerate(there)}

        # Flow from the links themselves, the same estimate the re-linker used.
        anchor = pos_um[src[idx]]
        disp = pos_um[tgt[idx]] - anchor
        k = min(30, len(idx))
        _, near = cKDTree(anchor).query(anchor, k=k)
        flow_at_link = np.median(disp[near.reshape(len(idx), -1)], axis=1)
        residual[idx] = np.linalg.norm(disp - flow_at_link, axis=1)
        median_step = float(np.median(np.linalg.norm(disp, axis=1))) or 1.0
        dist_over_flow[idx] = np.linalg.norm(disp, axis=1) / median_step

        # Rank of the chosen parent among all parents, after flow correction.
        _, near_all = cKDTree(anchor).query(src_um, k=k)
        flow_all = np.median(disp[near_all.reshape(len(here), -1)], axis=1)
        d = np.linalg.norm((src_um + flow_all)[:, None, :] - tgt_um[None, :, :], axis=-1)
        order = np.argsort(d, axis=0)
        rank_of = np.empty_like(order)
        rows = np.arange(len(here))[:, None]
        np.put_along_axis(rank_of, order, np.broadcast_to(rows, order.shape), axis=0)
        for n, i in enumerate(idx):
            a, b = local_src[int(src[i])], local_tgt[int(tgt[i])]
            rank[i] = rank_of[a, b]
            rank_frac[i] = rank_of[a, b] / max(len(here) - 1, 1)

        if len(src_um) > 1:
            dd, _ = cKDTree(src_um).query(src_um, k=2)
            nn_src = dd[:, 1]
        else:
            nn_src = np.full(len(src_um), np.inf)
        if len(tgt_um) > 1:
            dd, _ = cKDTree(tgt_um).query(tgt_um, k=2)
            nn_tgt = dd[:, 1]
        else:
            nn_tgt = np.full(len(tgt_um), np.inf)

        # How contested this parent was: children sitting near it.
        tree_tgt = cKDTree(tgt_um)
        rivals = np.array([len(tree_tgt.query_ball_point(p, 6.0)) for p in src_um], float)

        for i in idx:
            a, b = local_src[int(src[i])], local_tgt[int(tgt[i])]
            src_nn[i] = nn_src[a]
            tgt_nn[i] = nn_tgt[b]
            n_rivals[i] = max(rivals[a] - 1.0, 0.0)
            frame_nodes[i] = len(here)

    # --- track context ---
    label = _tracks(n_nodes, linked)
    members = defaultdict(list)
    for node, lab in enumerate(label):
        members[lab].append(node)
    before = np.zeros(n_nodes)
    after = np.zeros(n_nodes)
    track_size = np.zeros(n_nodes)
    for lab, nodes in members.items():
        nodes = sorted(nodes, key=lambda n: coords[n, 0])
        for i, node in enumerate(nodes):
            before[node] = i
            after[node] = len(nodes) - 1 - i
            track_size[node] = len(nodes)

    # --- velocity continuity: does this step follow the source's previous one ---
    parent_of = {int(b): int(a) for a, b in zip(src, tgt)}
    vel_change = np.zeros(len(linked))
    vel_cos = np.zeros(len(linked))
    has_vel = np.zeros(len(linked))
    for i, (a, b) in enumerate(zip(src, tgt)):
        prev = parent_of.get(int(a))
        if prev is None:
            continue
        prev_step = pos_um[a] - pos_um[prev]
        step = pos_um[b] - pos_um[a]
        vel_change[i] = float(np.linalg.norm(step - prev_step))
        n1, n2 = np.linalg.norm(prev_step), np.linalg.norm(step)
        if n1 > 1e-6 and n2 > 1e-6:
            vel_cos[i] = float(np.dot(prev_step, step) / (n1 * n2))
        has_vel[i] = 1.0

    t_max = max(float(coords[:, 0].max()), 1.0)

    out = np.column_stack([
        prob, has_prob, linked[:, 4], residual, linked[:, 3], dist_over_flow,
        rank, rank_frac, n_rivals,
        np.minimum(src_nn, 50.0), np.minimum(tgt_nn, 50.0),
        before[src], after[tgt], track_size[src],
        vel_change, vel_cos, has_vel,
        frame_nodes, src_t / t_max,
    ])
    return np.nan_to_num(out, nan=0.0, posinf=50.0, neginf=-50.0)


def link_labels(coords, linked, gt_graph):
    """(in_play, is_tp) per link — which links the metric scores, and their truth."""
    gt = GroundTruth(gt_graph)
    match = match_nodes(np.asarray(coords), gt)
    valid, is_tp = edge_flags(np.asarray(linked, dtype=float).reshape(-1, 5), match, gt)
    return valid, is_tp
