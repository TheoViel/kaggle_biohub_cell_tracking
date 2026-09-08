"""Which candidate edges the ground truth can speak about, and what it says.

The annotation covers roughly one cell a frame against ~260 detections, so the
central question is not "is this edge real" but "does the ground truth *know*".
Marking every unlabelled candidate as a negative would drown the handful of
positives in noise that is mostly wrong -- an unannotated cell still went
somewhere, and that link is a candidate too.

The rule used here is the one the tracking pipeline's ``edge_loss`` arrives at
from the other end, where it masks the loss to rows and columns the GT touches:

    an edge is supervised iff its source has a known successor, or its target
    has a known predecessor.

That is exactly the set where a label is derivable without further assumption.
If cell *i* is annotated as going to *j*, then every other candidate out of *i*
is a negative -- we know where it went. If *j* is annotated as coming from *i*,
every other candidate into *j* is a negative -- we know where it came from. A
cell annotated with no link at all (a track ending, or the annotation simply
stopping) states nothing, and its edges stay unsupervised rather than being
called negatives.

One more condition, easy to miss and quietly damaging: the positive edge has to
be **in the candidate set**. If the true link was never proposed -- outside
``max_distance_um``, or beaten out by ``n_neighbors`` closer nodes -- then
marking the source supervised would turn its whole fan of candidates into
negatives and teach the model that this cell has no successor. Those sources and
targets are left unsupervised instead, and the loss never sees them. The
fraction lost this way is the ceiling on what the model can learn, so
``candidate_recall`` reports it.
"""

from __future__ import annotations

import numpy as np


def edge_targets(
    edges: np.ndarray,                      # (E, 2) window-node indices
    links: list[tuple[int, int]],           # GT (source node, target node) pairs
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """``(target, supervised, n_missed)`` for the window's candidate edges.

    ``target`` is 1.0 on the ground-truth links and 0.0 elsewhere; ``supervised``
    marks the edges the loss may read. ``n_missed`` counts GT links absent from
    the candidate set, which is reported rather than hidden because it bounds
    the achievable recall.
    """
    target = np.zeros(len(edges), np.float32)
    supervised = np.zeros(len(edges), bool)
    if not len(edges) or not links:
        return target, supervised, len(links)

    # Edge lookup by (source, target). A window holds ~1e3 candidates and a
    # handful of links, so a dict beats any vectorised search.
    row_of = {(int(a), int(b)): e for e, (a, b) in enumerate(edges)}

    has_out = np.zeros(n_nodes, bool)
    has_in = np.zeros(n_nodes, bool)
    n_missed = 0
    for a, b in links:
        e = row_of.get((int(a), int(b)))
        if e is None:
            n_missed += 1
            continue
        target[e] = 1.0
        # Set only for links that made it into the candidate set -- see the
        # module docstring.
        has_out[int(a)] = True
        has_in[int(b)] = True

    supervised = has_out[edges[:, 0]] | has_in[edges[:, 1]]
    return target, supervised, n_missed


def window_links(
    video,
    frames: list[int],
    node_of: dict[tuple[int, int], int],
) -> list[tuple[int, int]]:
    """The window's ground-truth links, as window-node index pairs.

    ``video.links[t]`` holds ``[source row, target row, src_unique]`` where the
    rows index frame ``t`` and ``t + 1``'s detections in their own order;
    ``node_of`` maps ``(frame, row)`` onto this window's node axis. Rows whose
    node was dropped by the per-frame cap are skipped.

    ``src_unique`` is ignored here. It marks a dividing parent, whose two
    daughters are *both* correct -- an ambiguity for a loss that has to name one
    positive, but not for this one: each edge is scored on its own, so both
    daughter edges are simply positives.
    """
    out: list[tuple[int, int]] = []
    for k, t in enumerate(frames[:-1]):
        rows = video.links.get(int(t))
        if rows is None:
            continue
        for src_row, tgt_row in rows[:, :2]:
            a = node_of.get((k, int(src_row)))
            b = node_of.get((k + 1, int(tgt_row)))
            if a is not None and b is not None:
                out.append((a, b))
    return out
