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


def _label(edges, links, n_nodes, row_of):
    """``(target, supervised, n_missed)`` from one set of links.

    The rule is the module docstring's, and it does not care whether the links
    came from the annotation or from a previous round's predictions -- only
    what they are worth does, which is the caller's business.
    """
    target = np.zeros(len(edges), np.float32)
    if not links:
        return target, np.zeros(len(edges), bool), 0

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

    return target, has_out[edges[:, 0]] | has_in[edges[:, 1]], n_missed


def edge_targets(
    edges: np.ndarray,                      # (E, 2) window-node indices
    links: list[tuple[int, int]],           # GT (source node, target node) pairs
    n_nodes: int,
    pl_links: list[tuple[int, int]] | None = None,
    pl_weight: float = 0.0,
    division_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """``(target, supervised, weight, n_missed)`` for the window's candidates.

    ``target`` is 1.0 on the links and 0.0 elsewhere; ``supervised`` marks the
    edges the loss may read; ``weight`` is what each is worth to it.
    ``n_missed`` counts *ground-truth* links absent from the candidate set,
    which is reported rather than hidden because it bounds achievable recall.

    Pseudo-labels are folded in at ``pl_weight``, and the annotation overrides
    them everywhere it speaks. That precedence is belt to ``pseudo_links``'
    braces: it already dropped every pseudo-label the ground truth contradicts,
    but a *derived* verdict can still collide -- an edge can be a negative under
    the annotation because its target's parent is known, and a positive under a
    pseudo-label whose own source is unconstrained. The ground truth wins those,
    and at full weight.
    """
    target = np.zeros(len(edges), np.float32)
    supervised = np.zeros(len(edges), bool)
    weight = np.zeros(len(edges), np.float32)
    if not len(edges) or not (links or pl_links):
        return target, supervised, weight, len(links)

    # Edge lookup by (source, target). A window holds ~1e3 candidates and a
    # handful of links, so a dict beats any vectorised search.
    row_of = {(int(a), int(b)): e for e, (a, b) in enumerate(edges)}

    target, supervised, n_missed = _label(edges, links, n_nodes, row_of)
    weight = supervised.astype(np.float32)

    if pl_links and pl_weight > 0:
        pl_target, pl_sup, _ = _label(edges, pl_links, n_nodes, row_of)
        # Only where the annotation is silent, so nothing it says is overwritten
        # and nothing it says is charged the reduced weight.
        only_pl = pl_sup & ~supervised
        target = np.where(only_pl, pl_target, target)
        weight = np.where(only_pl, np.float32(pl_weight), weight)
        supervised = supervised | pl_sup

    # A dividing parent's two links, priced up. Divisions are 0.13% of the
    # annotation -- 151 across both embryos -- so at their natural frequency the
    # model has no reason to learn them, and it does not: measured on the
    # predictions, the *true* second child of a division scores a median 0.368
    # where an ordinary link clears 0.45, which is why the assignment drops it
    # and why post-processing has to put divisions back by hand
    # (postprocessing/README_edge.md, "Divisions"). Both links are already
    # labelled 1.0 above -- nothing here teaches "one child only" -- this only
    # says the rare case is worth more when it is wrong.
    #
    # Only the positives: raising a fork parent's negatives too would emphasise
    # the same decision from the other side and mostly reinforce not-linking,
    # which is the direction that already dominates.
    if division_weight != 1.0 and links:
        n_out: dict[int, int] = {}
        for a, _ in links:
            n_out[int(a)] = n_out.get(int(a), 0) + 1
        forks = {a for a, c in n_out.items() if c >= 2}
        if forks:
            is_fork = np.fromiter((int(a) in forks for a in edges[:, 0]),
                                  bool, len(edges))
            weight = np.where(is_fork & (target > 0), weight * np.float32(division_weight),
                              weight)

    return target, supervised, weight, n_missed


def window_links(
    video,
    frames: list[int],
    node_of: dict[tuple[int, int], int],
    links_by_frame: dict[int, "np.ndarray"] | None = None,
) -> list[tuple[int, int]]:
    """The window's links, as window-node index pairs.

    ``video.links[t]`` holds ``[source row, target row, src_unique]`` where the
    rows index frame ``t`` and ``t + 1``'s detections in their own order;
    ``node_of`` maps ``(frame, row)`` onto this window's node axis. Rows whose
    node was dropped by the per-frame cap are skipped.

    ``links_by_frame`` overrides the source, so the same slicing serves
    ``video.pl_links``, which is the identical structure with two columns.

    ``src_unique`` is ignored here. It marks a dividing parent, whose two
    daughters are *both* correct -- an ambiguity for a loss that has to name one
    positive, but not for this one: each edge is scored on its own, so both
    daughter edges are simply positives.
    """
    source = video.links if links_by_frame is None else links_by_frame
    out: list[tuple[int, int]] = []
    for k, t in enumerate(frames[:-1]):
        rows = source.get(int(t))
        if rows is None:
            continue
        for src_row, tgt_row in rows[:, :2]:
            a = node_of.get((k, int(src_row)))
            b = node_of.get((k + 1, int(tgt_row)))
            if a is not None and b is not None:
                out.append((a, b))
    return out
