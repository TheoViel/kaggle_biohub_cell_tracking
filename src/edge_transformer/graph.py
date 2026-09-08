"""Turning a window of nodes into the candidate graph HOCT scores.

Three steps, in this order, and the order matters:

1. **Drift correction.** A sample can slide bodily between frames -- one video
   here moves a steady -1.6 um in z every frame. That translation biases every
   distance in the window, so it is removed first, before anything reads a
   distance. Reused from the tracking pipeline (``estimate_drift``), where it
   was applied inside the model; here it moves up to graph construction because
   the candidate set itself is chosen by distance, and correcting after the
   candidates were picked would leave them chosen under the bias.

2. **Candidates.** Each node's ``n_neighbors`` nearest partners in each
   reachable later frame, within ``max_distance_um``, unioned over both
   directions -- a link that is a child's 8th-nearest parent may still be that
   parent's nearest child, and taking only one direction silently drops it.

3. **Features.** Per node: time, position, appearance. Per edge: the pairwise
   quantities an MLP over two node tokens would otherwise have to rediscover --
   distance, appearance similarity, and the *ranks* of both among the competing
   candidates. The tracking pipeline's experience is the argument for the ranks:
   a nearest-neighbour baseline already resolves ~90% of links here, so stating
   "this is the nearest parent, and the runner-up is 40% further" outright
   leaves the model only the job of deciding when to disagree with proximity.
"""

from __future__ import annotations

import numpy as np
import torch

from model_zoo.simple_node_transformer import estimate_drift

# One row per edge feature, in the order ``edge_features`` stacks them. Named
# so the model's input width is derived rather than declared, and so a feature
# can be added without hunting for the constant that has to change with it.
EDGE_FEATURE_NAMES = (
    "distance",           # scaled by max_distance_um
    "delta_t",
    "cosine_sim",         # appearance; 0 when either endpoint has no embedding
    "parent_rank",        # rank of this parent among the child's candidates
    "is_nearest_parent",
    "child_rank",         # rank of this child among the parent's candidates
    "is_nearest_child",
    "sim_parent_rank",    # the same two ranks, by appearance instead
    "is_best_parent",
    "sim_child_rank",
    "is_best_child",
    "excess_distance",    # log1p((d - d_min) / d_min) for the child
)

# Appended when ``use_conf_score``: the detection head's probability at each
# endpoint, as written into the nodes npz by ``main_predict.py``.
#
# The minimum is stated outright rather than left to the MLP because it is the
# quantity the link actually turns on. An edge is only as good as its weaker
# endpoint -- a confident parent linked to a barely-detected child is exactly
# the case worth doubting -- and min is not something two features and a ReLU
# recover cleanly. The two raw values stay alongside it so the model can still
# tell *which* end is the shaky one, which the minimum alone hides.
CONF_EDGE_FEATURE_NAMES = (
    "conf_src",
    "conf_tgt",
    "conf_min",
)


def edge_feature_names(use_conf_score: bool = False) -> tuple[str, ...]:
    """The edge features in the order ``edge_features`` stacks them."""
    return EDGE_FEATURE_NAMES + (CONF_EDGE_FEATURE_NAMES if use_conf_score else ())


def drift_between_frames(pos_um: np.ndarray, frame_idx: np.ndarray, n_frames: int) -> np.ndarray:
    """Per-frame translation carrying every frame onto the window's first.

    Returns ``(n_frames, 3)``, the offset to *subtract* from each frame's
    positions. Frame 0 is the reference, so its row is zero; the rest accumulate
    the consecutive estimates, which is what makes a window of more than two
    frames come out in one common reference rather than pairwise ones.
    """
    drift = np.zeros((n_frames, 3), np.float32)
    for k in range(n_frames - 1):
        a = pos_um[frame_idx == k]
        b = pos_um[frame_idx == k + 1]
        if len(a) == 0 or len(b) == 0:
            drift[k + 1] = drift[k]
            continue
        step = estimate_drift(
            torch.from_numpy(a).unsqueeze(0), torch.from_numpy(b).unsqueeze(0),
            None, None,
        )[0].numpy()
        drift[k + 1] = drift[k] + step
    return drift


def build_candidates(
    pos_um: np.ndarray,           # (N, 3) drift-corrected microns
    frame_idx: np.ndarray,        # (N,) 0-based frame within the window
    n_frames: int,
    n_neighbors: int = 5,
    max_distance_um: float = 20.0,
    max_dt: int = 1,
) -> np.ndarray:                  # (E, 2) int64, into the window's node axis
    """Candidate links: parent (earlier frame) -> child (later frame)."""
    pairs: list[np.ndarray] = []
    rows = {k: np.flatnonzero(frame_idx == k) for k in range(n_frames)}

    for a in range(n_frames):
        for b in range(a + 1, min(a + max_dt, n_frames - 1) + 1):
            ra, rb = rows[a], rows[b]
            if not len(ra) or not len(rb):
                continue
            d = np.linalg.norm(pos_um[ra][:, None] - pos_um[rb][None], axis=2)
            near = d <= max_distance_um

            keep = np.zeros_like(near)
            k = min(n_neighbors, len(rb))
            cols = np.argpartition(d, k - 1, axis=1)[:, :k]      # per parent
            np.put_along_axis(keep, cols, True, axis=1)
            k = min(n_neighbors, len(ra))
            rows_ = np.argpartition(d, k - 1, axis=0)[:k]        # per child
            np.put_along_axis(keep, rows_, True, axis=0)

            i, j = np.nonzero(keep & near)
            if len(i):
                pairs.append(np.stack([ra[i], rb[j]], axis=1))

    if not pairs:
        return np.zeros((0, 2), np.int64)
    return np.concatenate(pairs).astype(np.int64)


def _ranks(values: np.ndarray, groups: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Rank each value within its group, ascending, plus the group minimum.

    Returns ``(rank_normalised, group_min)`` where the rank is 0 for the group's
    best and 1 for its worst, so the feature does not change meaning when a node
    happens to have more candidates than another.
    """
    order = np.lexsort((values, groups))
    rank = np.empty(len(values), np.float32)
    size = np.bincount(groups, minlength=n_groups).astype(np.float32)
    # Position within the run of each group, in the lexsorted order.
    starts = np.concatenate([[0], np.cumsum(size).astype(int)[:-1]])
    rank[order] = np.arange(len(values)) - starts[groups[order]]
    denom = np.maximum(size[groups] - 1.0, 1.0)

    group_min = np.full(n_groups, np.inf, np.float32)
    np.minimum.at(group_min, groups, values)
    return rank / denom, group_min


def edge_features(
    edges: np.ndarray,            # (E, 2)
    pos_um: np.ndarray,           # (N, 3)
    frame_idx: np.ndarray,        # (N,)
    emb: np.ndarray | None,       # (N, D) L2-normalised, or None
    n_nodes: int,
    max_distance_um: float = 20.0,
    max_dt: int = 1,
    conf: np.ndarray | None = None,   # (N,) detection probability, or None
) -> np.ndarray:                  # (E, len(edge_feature_names(...))) float32
    """The pairwise features handed to the model alongside the node tokens.

    ``conf`` adds :data:`CONF_EDGE_FEATURE_NAMES`. Nothing here penalises a weak
    endpoint by hand: the targets already do that job, since a node the detector
    was unsure of is more often a false positive, and its candidate edges are
    more often the negatives among a real node's competitors. Stating the
    confidences lets the model find that; hard-coding a discount would fix the
    exchange rate between "unsure detection" and "good geometry" at a number
    nobody measured.
    """
    src, tgt = edges[:, 0], edges[:, 1]
    dist = np.linalg.norm(pos_um[src] - pos_um[tgt], axis=1).astype(np.float32)
    delta_t = (frame_idx[tgt] - frame_idx[src]).astype(np.float32)

    if emb is None:
        cos = np.zeros(len(edges), np.float32)
    else:
        e = emb.astype(np.float32)
        cos = (e[src] * e[tgt]).sum(1).astype(np.float32)

    # Ranks by distance, and by appearance, each in both directions. "Parent
    # rank" answers "how good is this parent for that child, relative to the
    # child's other options" -- which is what the linker actually decides.
    parent_rank, child_min = _ranks(dist, tgt, n_nodes)
    child_rank, _ = _ranks(dist, src, n_nodes)
    if emb is None:
        # With no appearance there is nothing to rank: ties would be broken by
        # node index, handing the model an index-shaped feature to overfit.
        sim_parent_rank = np.zeros(len(edges), np.float32)
        sim_child_rank = np.zeros(len(edges), np.float32)
    else:
        sim_parent_rank, _ = _ranks(-cos, tgt, n_nodes)
        sim_child_rank, _ = _ranks(-cos, src, n_nodes)

    best = child_min[tgt]
    excess = np.log1p(np.maximum(dist - best, 0.0) / np.maximum(best, 1e-3))

    columns = [
        dist / max_distance_um,
        delta_t / max(max_dt, 1),
        cos,
        parent_rank,
        (parent_rank == 0).astype(np.float32),
        child_rank,
        (child_rank == 0).astype(np.float32),
        sim_parent_rank,
        (emb is not None) * (sim_parent_rank == 0).astype(np.float32),
        sim_child_rank,
        (emb is not None) * (sim_child_rank == 0).astype(np.float32),
        excess,
    ]
    if conf is not None:
        c = np.asarray(conf, np.float32)
        columns += [c[src], c[tgt], np.minimum(c[src], c[tgt])]

    return np.stack(columns, axis=1).astype(np.float32)


def node_features(
    pos_um: np.ndarray,           # (N, 3) drift-corrected
    frame_idx: np.ndarray,        # (N,)
    emb: np.ndarray | None,       # (N, D)
    n_frames: int,
    pos_scale: float = 20.0,
    conf: np.ndarray | None = None,   # (N,) detection probability, or None
) -> np.ndarray:                  # (N, 4 + D [+ 1]) float32
    """Time, position and appearance, as one vector per node.

    Position appears here *and* as the rotary position the attention reads. The
    two are not redundant: the rotary encoding only ever sees differences, so a
    node's own absolute place in the volume -- near the edge of the field, deep
    in z -- reaches the model through this feature alone.

    ``conf`` appends the detection probability. It is a node property, so it
    belongs on the token as well as on the edges built from it: the node blocks
    run before the edge ones, and a node that is probably not a cell should be
    able to say so to its neighbourhood before any link is scored.
    """
    t = (frame_idx.astype(np.float32) / max(n_frames - 1, 1))[:, None]
    parts = [t, pos_um.astype(np.float32) / pos_scale]
    if emb is not None:
        parts.append(emb.astype(np.float32))
    if conf is not None:
        parts.append(np.asarray(conf, np.float32).reshape(-1, 1))
    return np.concatenate(parts, axis=1).astype(np.float32)


def node_feature_dim(embed_dim: int, use_conf_score: bool = False) -> int:
    """Width of ``node_features``: t, z, y, x, the embedding, then confidence."""
    return 4 + int(embed_dim) + int(bool(use_conf_score))
