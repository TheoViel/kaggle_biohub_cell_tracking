"""Find divisions by running time backwards, where they are merges.

A cell cannot appear from nothing. Read forward, a track that *starts* in the
middle of a video started for one of three reasons: the cell drifted in from
outside the imaged volume, it was born from a division, or the detector missed
it earlier and the track is a fragment. Read backwards, the second case is a
**merge** -- two trajectories running into one -- and merges are much easier to
test for than births, because two tracks converging is a statement about
*direction*, not just proximity.

That is the difference from `edge_pp.propose_divisions`, which is the public
946 repair plus a probability gate. Its evidence is distances: parent to
candidate, parent to first child, sister to sister, and a forward divergence
check at ``t+2``. Distance alone cannot separate a real division from a cell
that happens to have a neighbour, which is why that stage runs at roughly 4
true positives against 16 false ones.

Here the test is that the two daughters' own trajectories, extrapolated
*backwards*, intersect -- they must arrive at the same place at ``t-1``, and
that place must be where the parent actually is. A cell and an unrelated
neighbour travel on their own headings and their backward extrapolations do not
converge; two halves of a division do, because they were one object a frame
earlier.

Three quantities, all in microns and all after the field translation is removed
(`estimate_drift`), since over one frame the whole-sample drift is most of the
budget a cell's own motion is judged against:

    converge  ||b1 - b2||        the two backward extrapolations meeting
    to_parent max(||b1 - p||, ||b2 - p||)   and meeting *at the parent*
    closing   ||d1 - d2|| at t+1 minus at t   sisters separating going forward

`closing` is the same information `require_divergence` uses, expressed as a
rate rather than a threshold at a fixed offset, so it survives a division whose
daughters separate slowly.
"""

from __future__ import annotations

import numpy as np
import torch
from collections import defaultdict

from model_zoo.simple_node_transformer import estimate_drift

# `estimate_drift` goes through torch, whose intra-op pool sizes itself from the
# core count and ignores OMP_NUM_THREADS. Under a worker pool that is a massive
# oversubscription -- measured at 11 cores per worker across 18 workers, which
# turned a 3-minute sweep into a 35-minute one that had produced nothing. Same
# line, same reason, as main_edge_postprocess.py and tune_edge_pp.py.
torch.set_num_threads(1)

DEFAULTS = dict(
    # How far back a daughter's trajectory is fitted before it is extrapolated.
    # Two is the minimum that defines a direction; three is steadier and still
    # available for most starts, since a fragment shorter than that is usually
    # noise rather than a daughter.
    fit_window=3,
    min_track=2,
    # The intersection test.
    converge_um=4.0,        # the two backward extrapolations must meet
    to_parent_um=6.0,       # and must meet where the parent is
    # Sisters must be separating forward -- the reverse of converging backward.
    # 0 disables, which is the ablation that isolates what the backward test
    # adds over the forward one.
    min_closing_um=0.0,
    # A parent may only gain a second child; it must already have exactly one.
    # Anything else is a lineage error rather than a division.
    max_sister_um=14.0,
    # Divisions are rare -- 26 in the 71 fold-0 stacks -- so an unbounded
    # proposer is a false-positive generator. Same caps as `propose_divisions`,
    # applied to the ranked list so the best survive.
    frame_frac_cap=0.0076,
    global_frac_cap=0.00375,
    boundary_um=10.0,
)


def _fit_back(pts_um: np.ndarray):
    """Extrapolate one frame *before* the first of ``pts_um``.

    ``pts_um`` is a track's first frames, oldest first. Least squares rather
    than a difference: a daughter's first positions are each one detection away
    from being wrong, and the whole test is that two of these extrapolations
    land on the same voxel.
    """
    n = len(pts_um)
    if n == 1:
        return pts_um[0], 0.0
    t = np.arange(n, dtype=np.float64)
    tc = t - t.mean()
    denom = float((tc * tc).sum())
    if denom == 0:
        return pts_um[0], 0.0
    vel = (tc[:, None] * (pts_um - pts_um.mean(axis=0))).sum(axis=0) / denom
    pred = pts_um.mean(axis=0) + tc[:, None] * vel
    resid = float(np.linalg.norm(pts_um - pred, axis=1).mean())
    # position at t = -1, i.e. one frame before the track's first node
    return pts_um.mean(axis=0) + (-1.0 - t.mean()) * vel, resid


def propose_reverse_divisions(coords, edges, scale, cfg=None, stats=None):
    """Add a second child to parents whose two daughters converge backwards."""
    cfg = {**DEFAULTS, **(cfg or {})}
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 4)
    edges = np.asarray(edges, dtype=np.float64).reshape(-1, 4)
    if not len(coords) or not len(edges):
        return edges
    s = np.asarray(scale, dtype=np.float64)

    frames = coords[:, 0].astype(int)
    t_max = int(frames.max())
    rows_by_t = {t: np.flatnonzero(frames == t) for t in range(t_max + 1)}

    children = defaultdict(list)
    parent_of = {}
    for a, b, *_ in edges:
        children[int(a)].append(int(b))
        parent_of[int(b)] = int(a)

    lo = coords[:, 1:].min(axis=0) * s
    hi = coords[:, 1:].max(axis=0) * s

    def forward_track(i, k):
        """``i`` and up to ``k-1`` of its descendants, oldest first."""
        ch = [i]
        while len(ch) < k and len(children.get(ch[-1], [])) == 1:
            ch.append(children[ch[-1]][0])
        return ch

    proposals = []
    for t in range(1, t_max + 1):
        rows_t = rows_by_t.get(t, [])
        rows_prev = rows_by_t.get(t - 1, [])
        if not len(rows_t) or not len(rows_prev):
            continue
        a = torch.from_numpy(coords[rows_prev, 1:].astype(np.float32))[None]
        b = torch.from_numpy(coords[rows_t, 1:].astype(np.float32))[None]
        drift = estimate_drift(a, b, None, None).numpy()[0].astype(np.float64) * s

        # Track starts: a cell that appeared. Backwards, these are the merges.
        starts = [int(j) for j in rows_t if int(j) not in parent_of]
        if not starts:
            continue
        # Parents that could take a second child, and their existing daughter.
        cands = [(int(i), children[int(i)][0]) for i in rows_prev
                 if len(children.get(int(i), [])) == 1]
        if not cands:
            continue
        p_pos = np.stack([coords[i, 1:] for i, _ in cands]) * s

        for d2 in starts:
            p2 = coords[d2, 1:] * s
            if np.any(p2 - lo < cfg["boundary_um"]) or np.any(hi - p2 < cfg["boundary_um"]):
                continue                      # drifted in from outside
            tr2 = forward_track(d2, cfg["fit_window"])
            if len(tr2) < cfg["min_track"]:
                continue
            b2, _ = _fit_back(coords[tr2, 1:] * s)
            b2 = b2 - drift                   # into the parent frame's coordinates

            # Nearest plausible parents by the *extrapolated* position, not by
            # the daughter's own: where the trajectory says it came from.
            d_par = np.linalg.norm(p_pos - b2, axis=1)
            order = np.argsort(d_par)[:5]
            for oi in order:
                if d_par[oi] > cfg["to_parent_um"]:
                    break
                par, d1 = cands[oi]
                if d1 == d2:
                    continue
                p1 = coords[d1, 1:] * s
                if np.linalg.norm(p1 - p2) > cfg["max_sister_um"]:
                    continue
                tr1 = forward_track(d1, cfg["fit_window"])
                if len(tr1) < cfg["min_track"]:
                    continue
                b1, _ = _fit_back(coords[tr1, 1:] * s)
                b1 = b1 - drift

                converge = float(np.linalg.norm(b1 - b2))
                to_par = max(float(np.linalg.norm(b1 - coords[par, 1:] * s)),
                             float(d_par[oi]))
                if converge > cfg["converge_um"] or to_par > cfg["to_parent_um"]:
                    continue
                if cfg["min_closing_um"] > 0:
                    # sisters must be separating as time runs forward
                    if len(tr1) > 1 and len(tr2) > 1:
                        sep0 = np.linalg.norm(p1 - p2)
                        sep1 = np.linalg.norm(
                            (coords[tr1[1], 1:] - coords[tr2[1], 1:]) * s)
                        if sep1 - sep0 < cfg["min_closing_um"]:
                            continue
                # rank by how well the two trajectories actually intersect
                proposals.append((converge + to_par, t, par, d2))
                break

    # Caps, on the ranked list: the best-supported proposals survive.
    proposals.sort()
    n_frames = t_max + 1
    per_frame = defaultdict(int)
    frame_cap = max(1, int(cfg["frame_frac_cap"] * len(coords) / max(n_frames, 1)))
    global_cap = max(1, int(cfg["global_frac_cap"] * len(coords)))
    taken = []
    used_child = set()
    for _, t, par, d2 in proposals:
        if len(taken) >= global_cap:
            break
        if per_frame[t] >= frame_cap or d2 in used_child:
            continue
        if len(children.get(par, [])) >= 2:
            continue
        taken.append((par, d2))
        children[par].append(d2)
        used_child.add(d2)
        per_frame[t] += 1

    if stats is not None:
        stats.update(considered=len(proposals), added=len(taken))
    if not taken:
        return edges
    new = np.array([[a, b, 0.5, 0.0] for a, b in taken], dtype=np.float64)
    return np.concatenate([edges, new])


def intersection_score(coords, parent, d1, d2, scale, drift_um=None, fit_window=3):
    """How well two daughters' backward trajectories meet, and meet at ``parent``.

    Returns ``(converge_um, to_parent_um)``. This is the same test
    `propose_reverse_divisions` runs, exposed so it can be used the other way
    round: as a *filter* on proposals somebody else generated.

    That turned out to be where it belongs. Used as a generator it has to scan
    every track start -- thousands per stack against 26 divisions in all of fold
    0 -- and at a threshold loose enough to catch a real division it proposes
    hundreds of false ones. Used as a filter it only ever sees candidates that
    have already passed the forward stage's gates, where the prior is far less
    hostile.
    """
    s = np.asarray(scale, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    children = intersection_score._children
    def fwd(i, k):
        ch = [i]
        while len(ch) < k and len(children.get(ch[-1], [])) == 1:
            ch.append(children[ch[-1]][0])
        return ch
    b1, _ = _fit_back(coords[fwd(d1, fit_window), 1:] * s)
    b2, _ = _fit_back(coords[fwd(d2, fit_window), 1:] * s)
    if drift_um is not None:
        b1 = b1 - drift_um; b2 = b2 - drift_um
    p = coords[parent, 1:] * s
    return (float(np.linalg.norm(b1 - b2)),
            max(float(np.linalg.norm(b1 - p)), float(np.linalg.norm(b2 - p))))


def frame_drifts(coords, scale):
    """Field translation for every frame pair, in um. Shared across thresholds."""
    coords = np.asarray(coords, float).reshape(-1, 4)
    s = np.asarray(scale, float)
    frames = coords[:, 0].astype(int)
    rows = {t: np.flatnonzero(frames == t) for t in np.unique(frames)}
    out = {}
    for t in rows:
        if t - 1 in rows and len(rows[t]) and len(rows[t - 1]):
            a = torch.from_numpy(coords[rows[t - 1], 1:].astype(np.float32))[None]
            b = torch.from_numpy(coords[rows[t], 1:].astype(np.float32))[None]
            out[t] = estimate_drift(a, b, None, None).numpy()[0].astype(np.float64) * s
    return out


def filter_divisions(coords, edges_before, edges_after, scale, cfg=None, stats=None,
                     drifts=None):
    """Drop forks the backward-intersection test does not support.

    ``edges_after`` is the forward stage's output; anything in it that is not in
    ``edges_before`` is a proposed fork. Each is re-tested on whether the two
    daughters' backward extrapolations converge, and at the parent.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    before = {(int(a), int(b)) for a, b, *_ in np.asarray(edges_before, float).reshape(-1, 4)}
    after = np.asarray(edges_after, float).reshape(-1, 4)
    coords = np.asarray(coords, float).reshape(-1, 4)
    s = np.asarray(scale, float)

    children = defaultdict(list)
    for a, b, *_ in after:
        children[int(a)].append(int(b))
    intersection_score._children = children

    if drifts is None:
        drifts = frame_drifts(coords, s)

    keep = np.ones(len(after), bool)
    checked = dropped = 0
    for k, (a, b, *_ ) in enumerate(after):
        key = (int(a), int(b))
        if key in before:
            continue
        par = int(a); d2 = int(b)
        kids = [c for c in children[par] if c != d2]
        if not kids:
            continue
        checked += 1
        t = int(coords[d2, 0])
        conv, to_par = intersection_score(coords, par, kids[0], d2, s,
                                          drifts.get(t), cfg["fit_window"])
        if conv > cfg["converge_um"] or to_par > cfg["to_parent_um"]:
            keep[k] = False
            dropped += 1
    if stats is not None:
        stats.update(checked=checked, dropped=dropped)
    return after[keep]
