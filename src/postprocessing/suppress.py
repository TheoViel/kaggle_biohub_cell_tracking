"""Put back the spacing the detector guaranteed and post-processing threw away.

Greedy NMS at inference leaves a hard floor between detections in a frame — on
this run 7.0 um requested, 7.08 um observed, with *not one* pair below it. Two
stages then move nodes: ``recover`` inserts interpolated ones, and
``pp_public``'s linefit smoothing shifts existing ones. Neither knows about the
floor, and between them 14–21% of nodes end up with a neighbour inside 7 um and
about 1% inside 5 um, down to nodes sitting on top of each other.

That is the failure mode `analysis_worst.py` isolates. When one cell carries two
detections, the metric ties the annotation to one of them and the track may run
through the other; the link is then charged as a false positive at both ends
even though it is physically right. Node recall does not notice — both copies
are inside the 7 um matching radius — and the score loses twice.

Which copy to keep barely matters, and cannot be decided anyway: over the six
worst stacks the annotation's own match sits on the longer track only 54.7% of
the time, a coin flip. What matters is that **one** of them survives, because
then the matcher and the tracker cannot disagree. So this suppresses rather than
chooses, and rewires the loser's links onto the winner instead of dropping them,
which keeps both tracks whole.

**MEASURED, AND IT DOES NOT PAY — this is not in the pipeline.** Over all 71
stacks: -0.0002 at a 5 um radius (3 stacks better, 6 worse) and -0.068 at 7 um.
The reasoning above is sound and the target is wrong. What costs the score is
two detections within 7 um *of the annotation*, and those sit 7-14 um apart from
**each other** — exactly what a 7 um NMS is built to allow. A radius that
reaches them is a radius that deletes cells: 9-11 um drives the rival rate to
zero and node recall from ~99% to 41-78%. With cells 8.0 um apart and a 7 um
matching radius, every detection set that covers them has rivals inside it.

Kept because the measurement is worth more than the stage: see
postprocessing/README.md, *Cutting the rival density*.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from dedup import track_labels
from params import SCALE

SCALE_ARR = np.asarray(SCALE)

DEFAULTS = dict(
    radius=7.0,          # um; the detector's own NMS radius
    position="keep",     # "keep" the winner's coordinates, or their "centroid"
    max_parents=1,
    max_children=1,
)


def _resolve(coords, edges, radius, position):
    """Winner per frame, and the map from every suppressed node onto its winner."""
    pos_um = coords[:, 1:] * SCALE_ARR
    label = track_labels(len(coords), edges)
    strength = np.bincount(label, minlength=label.max() + 1)[label]

    t = coords[:, 0].astype(int)
    merge = np.arange(len(coords))
    keep = np.ones(len(coords), dtype=bool)
    new_pos = pos_um.copy()

    for tt in np.unique(t):
        rows = np.flatnonzero(t == tt)
        if len(rows) < 2:
            continue
        tree = cKDTree(pos_um[rows])
        # Longest track first: not because it is right — it is a coin flip — but
        # because it makes the outcome deterministic and keeps the node that has
        # the most links to preserve.
        order = rows[np.lexsort((rows, -strength[rows]))]
        alive = np.ones(len(rows), dtype=bool)
        local = {int(r): i for i, r in enumerate(rows)}
        for r in order:
            i = local[int(r)]
            if not alive[i]:
                continue
            group = [j for j in tree.query_ball_point(pos_um[r], radius)
                     if alive[j] and j != i]
            if not group:
                continue
            for j in group:
                alive[j] = False
                keep[rows[j]] = False
                merge[rows[j]] = r
            if position == "centroid":
                members = [r] + [rows[j] for j in group]
                new_pos[r] = pos_um[members].mean(axis=0)
    return keep, merge, new_pos


def _rewire(edges, merge, keep, pos_um, max_parents, max_children):
    """Edges of suppressed nodes moved onto their winner, then capped in degree."""
    if len(edges) == 0:
        return edges
    src = merge[edges[:, 0].astype(int)]
    tgt = merge[edges[:, 1].astype(int)]
    prob = edges[:, 2]
    alive = keep[src] & keep[tgt] & (src != tgt)
    src, tgt, prob = src[alive], tgt[alive], prob[alive]
    if not len(src):
        return np.empty((0, 4))

    # Two copies of a cell can carry the same link twice; keep the better one.
    order = np.lexsort((-prob, tgt, src))
    src, tgt, prob = src[order], tgt[order], prob[order]
    first = np.ones(len(src), dtype=bool)
    first[1:] = (src[1:] != src[:-1]) | (tgt[1:] != tgt[:-1])
    src, tgt, prob = src[first], tgt[first], prob[first]

    # The ground truth has no divisions and one parent per cell; a merge can
    # break both, so re-impose them, best link first.
    n_par: dict[int, int] = {}
    n_chi: dict[int, int] = {}
    out = []
    for k in np.argsort(-prob):
        s, g = int(src[k]), int(tgt[k])
        if n_chi.get(s, 0) >= max_children or n_par.get(g, 0) >= max_parents:
            continue
        n_chi[s] = n_chi.get(s, 0) + 1
        n_par[g] = n_par.get(g, 0) + 1
        out.append((s, g, float(prob[k]),
                    float(np.linalg.norm(pos_um[s] - pos_um[g]))))
    return np.asarray(out, dtype=float).reshape(-1, 4)


def suppress_rivals(coords, edges, **kw):
    """Restore the per-frame spacing floor. Returns (coords, edges), reindexed."""
    cfg = {**DEFAULTS, **kw}
    coords = np.asarray(coords, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64).reshape(-1, 4)
    if cfg["radius"] <= 0 or len(coords) == 0:
        return coords, edges

    keep, merge, new_pos = _resolve(coords, edges, cfg["radius"], cfg["position"])
    if keep.all():
        return coords, edges

    out = coords.copy()
    out[:, 1:] = new_pos / SCALE_ARR
    linked = _rewire(edges, merge, keep, new_pos,
                     cfg["max_parents"], cfg["max_children"])

    index = np.full(len(coords), -1, dtype=int)
    index[keep] = np.arange(int(keep.sum()))
    if len(linked):
        linked[:, 0] = index[linked[:, 0].astype(int)]
        linked[:, 1] = index[linked[:, 1].astype(int)]
    return out[keep], linked
