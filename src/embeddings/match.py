"""Linking frames with the embeddings: ASCENT's serial bipartite matcher.

Three pieces, all from the paper:

* **Hungarian assignment** on cosine cost, frame to frame. One-to-one by
  construction, which a greedy nearest-neighbour pass is not -- and the
  competition's own matcher is one-to-one too, so a greedy linker would be
  optimising a different problem than it is scored on.
* **A dynamic cost threshold**, ``0.5 * (mean(top-k costs) + mean(all costs))``
  with k the number of active tracks. A fixed cutoff cannot work here: cosine
  costs drift with how crowded the frame is, so a threshold that is right for
  180 cells rejects everything at 790.
* **An EMA track embedding**, momentum 0.5. A track is matched against its
  running appearance rather than only its last frame, so one bad frame does not
  lose the track.

Two properties worth knowing before trusting the output.

*Divisions.* A one-to-one assignment cannot represent a cell becoming two, so a
daughter starts a new track. That is the paper's formulation -- it tracks
neurons, whose roster is fixed -- and it is the wrong shape for dividing cells.
Anything scored on divisions needs the link set repaired afterwards.

*The threshold is relative, not absolute.* It is built from the frame's own cost
distribution, so it always admits roughly the better half of the assignment.
Measured on unrelated embeddings it still accepts 38 of 40 pairs. It is a good
way to reject the worst matches within a frame and no defence at all against a
frame that should not have been matched -- a shot-noise frame, a dropped
timepoint. Pass ``threshold`` explicitly where that matters.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def dynamic_threshold(cost: np.ndarray, n_active: int) -> float:
    """``0.5 * (mean(top-k costs) + mean(all costs))``, k = active tracks.

    "top-k" is read as the k *best* (lowest) costs, those being the matches the
    frame is confident about: the threshold then sits halfway between what a
    good match costs here and what an average one does, which is what makes it
    track the frame's own difficulty.
    """
    flat = cost.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return float("inf")
    k = int(np.clip(n_active, 1, flat.size))
    best = np.partition(flat, k - 1)[:k]
    return float(0.5 * (best.mean() + flat.mean()))


def match_frames(
    emb_prev: np.ndarray,     # (P, D) L2-normalised track embeddings
    emb_next: np.ndarray,     # (C, D) L2-normalised detection embeddings
    n_active: int | None = None,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One frame-to-frame assignment.

    Returns ``(rows, cols, threshold)``: index pairs that were accepted, and
    the cutoff used. Pairs above the cutoff are dropped, which is how a track
    ends and how a detection starts a new one.
    """
    if len(emb_prev) == 0 or len(emb_next) == 0:
        return np.zeros(0, int), np.zeros(0, int), float("inf")

    cost = 1.0 - emb_prev @ emb_next.T          # cosine distance in [0, 2]
    if threshold is None:
        threshold = dynamic_threshold(cost, n_active if n_active is not None else len(emb_prev))

    rows, cols = linear_sum_assignment(cost)
    keep = cost[rows, cols] <= threshold
    return rows[keep], cols[keep], threshold


def link_sequence(
    embeddings: list[np.ndarray],
    coords: list[np.ndarray] | None = None,
    momentum: float = 0.5,
    threshold: float | None = None,
) -> list[tuple[int, int, int, int]]:
    """Link a whole video frame by frame.

    Args:
        embeddings: one ``(N_t, D)`` array per frame, L2-normalised.
        momentum: EMA on a matched track's embedding, the paper's alpha = 0.5.
        threshold: fixed cutoff, or None for the dynamic one per frame.

    Returns edges as ``(t, i, t + 1, j)`` index pairs into the per-frame arrays,
    which is the form the tracking pipeline's ``build_graph`` already reads.
    """
    edges: list[tuple[int, int, int, int]] = []
    if not embeddings:
        return edges

    # A track carries its own running embedding; the index it points at is the
    # detection it matched in the previous frame.
    track_emb = embeddings[0].copy()
    track_idx = np.arange(len(embeddings[0]))

    for t in range(len(embeddings) - 1):
        nxt = embeddings[t + 1]
        rows, cols, _ = match_frames(track_emb, nxt, n_active=len(track_emb),
                                     threshold=threshold)
        for r, c in zip(rows, cols):
            edges.append((t, int(track_idx[r]), t + 1, int(c)))

        # Matched tracks carry forward, smoothed; unmatched detections start
        # their own. Tracks that went unmatched are dropped rather than coasted:
        # without an appearance update they drift, and a stale embedding
        # matching later is worse than a broken track.
        new_emb = nxt.copy()
        new_idx = np.arange(len(nxt))
        for r, c in zip(rows, cols):
            blended = momentum * track_emb[r] + (1.0 - momentum) * nxt[c]
            norm = np.linalg.norm(blended)
            new_emb[c] = blended / norm if norm > 0 else nxt[c]
        track_emb, track_idx = new_emb, new_idx

    return edges
