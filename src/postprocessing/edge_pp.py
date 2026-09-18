"""Post-processing for the edge transformer's *candidate* graph.

`main_edge_predict.py --dump_candidates` writes, beside the committed edges,
every candidate the model scored: `(src row, tgt row, prob, distance um, dt)`.
That changes what post-processing can do. The stages in `pipeline.py` were
written for the detector's npz, where the only edges present are the ones the
tracker already chose and the probabilities are barely informative — so
`relink` throws them away and re-decides on flow-corrected geometry alone
(`prob_weight` is 0). Here the probability *is* the signal: the model reaches
AP 0.948 on held-out frames, which is a better ranking than distance gives.

So the stages here re-decide the assignment rather than replace it, and every
one of them is a function of the candidate set:

`assign`        the frame-pair Hungarian, at any threshold, with an optional
                flow-residual penalty mixed into the log-odds cost.
`recover_nodes` the gap filler from `recover.py`, then a re-assignment that
                wires the new nodes in on geometry (the model never scored
                them, so it has no opinion).
`prune_short`   drop chains shorter than `min_track`; this is the one stage
                that moves the node-count adjustment as well as the Jaccard.
`bridge_tracks` join a chain that ends at `u` to one that starts at `u + 1`
                when the assignment declined the link but the geometry is
                unambiguous -- the tracklet-level pass, without new nodes.

Everything runs from the npz alone: no model, no images, no GPU.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from embeddings.match import dynamic_threshold
from params import SCALE

SCALE_ARR = np.asarray(SCALE)
NO_LINK = 1e6


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_pred(path):
    """`(coords, edges, candidates)` from a `--dump_candidates` npz.

    `coords` comes back as float64 -- the file stores int16 to stay small, but
    every stage downstream works in microns and `recover` writes sub-voxel
    positions, so widening once here keeps the rest of the module honest.
    """
    data = np.load(path, allow_pickle=True)
    coords = data["coords"].astype(np.float64)
    edges = data["edges"].astype(np.float64).reshape(-1, 4)
    cand = (data["candidates"].astype(np.float64).reshape(-1, 5)
            if "candidates" in data.files else np.empty((0, 5)))
    return coords, edges, cand


def pos_um(coords):
    return np.asarray(coords)[:, 1:].astype(float) * SCALE_ARR


def rows_by_frame(coords):
    t = np.asarray(coords)[:, 0].astype(int)
    return {int(tt): np.flatnonzero(t == tt) for tt in np.unique(t)}


# --------------------------------------------------------------------------
# the assignment
# --------------------------------------------------------------------------

def _log_odds(p, threshold, kind="logit"):
    """The cost of committing a link, negative exactly when it is worth making.

    Both forms cross zero at `p == threshold`, so they accept and reject the
    same pairs in isolation; they differ in how much a *confident* link
    outweighs a marginal one, which is what decides a contested assignment.

    `"logprob"` is `link_video`'s own cost, `log(threshold) - log(p)`. It reads
    as a log-likelihood ratio but is not one: under a one-to-one constraint the
    alternative to committing an edge is *declining* it, which costs
    `log(1 - p)`, and dropping that term flattens the scale. Between p = 0.45
    and p = 0.99 it spans a factor of 7.8, where the odds span a factor of 24 --
    so the solver trades a near-certain link for two mediocre ones far too
    cheaply.

    `"logit"` is the corrected form, `logit(threshold) - logit(p)`, the log
    posterior odds of the edge against its absence. Measured, it changes 3-9
    edges in 36 000 and the score by 0.0000: the candidate graph is sparse
    enough (5 per node) that the sign decides almost every pair, not the
    magnitude. `"logprob"` stays the default so `assign` keeps reproducing
    `link_video` exactly, which is this module's regression test.
    """
    p = np.clip(p, 1e-9, 1 - 1e-9)
    th = min(max(threshold, 1e-6), 1 - 1e-6)
    if kind == "logprob":
        return float(np.log(th)) - np.log(p)
    if kind != "logit":
        raise ValueError(f"unknown cost {kind!r}; use 'logit' or 'logprob'")
    logit_th = float(np.log(th) - np.log1p(-th))
    return logit_th - (np.log(p) - np.log1p(-p))


def load_embeddings(path):
    """The L2-normalised node embeddings for a sample, or None.

    `notebooks/CompareCV.ipynb` links on the cosine between these and gains
    +0.05 cv over the *node* transformer's own edges. They are worth having
    here for the same reason -- except that this model already reads them as
    node features, which is what `emb_weight` below is for measuring.

    The embeddings npz and the prediction npz are built from the same detection
    set, so row `i` is the same node in both; this checks that rather than
    assuming it.
    """
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path) as d:
        if "embeddings" not in d.files:
            return None
        emb = d["embeddings"].astype(np.float32)
        coords = d["coords"]
    emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
    return emb, np.asarray(coords)


def assign(coords, cand, threshold=0.4, flow_weight=0.0, flow_k=30,
           flow_passes=2, max_residual=np.inf, emb=None, emb_weight=0.0,
           emb_momentum=0.0, vel_weight=0.0, vel_momentum=0.5,
           vel_max_residual=np.inf, vel_seed_drift=True,
           cost='logprob', match='hungarian', fork_threshold=None):
    """One-to-one links per consecutive frame pair, over the scored candidates.

    Parameters
    ----------
    coords : (N, 4) array
        The video's nodes, as t / z / y / x in grid voxels.
    cand : (K, 5) array
        Every scored candidate: source row, target row, probability, distance
        in microns, and the frame step. Only `dt == 1` is linked -- a longer
        step produces an edge the metric does not score and a track it reads
        as two.
    threshold : float
        The probability a link must beat to be worth making at all. The metric
        charges one false positive for a wrong link against one true positive
        for a right one, so the break-even is around 0.45 -- see
        postprocessing/README.md, "What the metric actually rewards".
    flow_weight : float
        Microns of flow-corrected residual, converted to log-odds at this rate
        and added to the cost. 0 leaves the decision entirely to the model,
        which is what the sweep chose -- see README_edge.md.
    flow_k : int
        Links in the neighbourhood the local flow is the median of.
    flow_passes : int
        Rounds of re-estimating that flow from the assignment it produced.
    max_residual : float
        Microns of flow-corrected residual beyond which a candidate is refused
        outright, whatever the model thinks.
    emb : (N, D) array or None
        L2-normalised node embeddings, from `load_embeddings`.
    emb_weight : float
        Blend the cosine into the link probability, as
        `(1 - w) * p + w * (cos + 1) / 2` -- `CompareCV.ipynb`'s
        `w_emb * sim_emb + w_trans * p_trans`, in probability space so that the
        threshold keeps meaning. 0 leaves the decision to the model.
    emb_momentum : float
        Carry an EMA of each track's embedding forward instead of comparing
        against the parent detection's own: `m * track + (1 - m) * detection`,
        re-normalised, which is the ASCENT linker's track state. 0 compares
        detection to detection.
    vel_weight : float
        Log-odds charged per micron a link departs from where the parent's own
        momentum says it should go -- `velocity_link`'s state, arbitrating the
        model's cost instead of replacing it. This is the version of "linear
        assignment with momentum" that keeps the model as the primary signal:
        the probability decides, and the track's history breaks the ties it
        cannot. 0 leaves the decision to the model alone.
    vel_momentum : float
        EMA on that velocity, `m * v + (1 - m) * displacement`.
    vel_max_residual : float
        Microns of velocity residual beyond which a candidate is refused.
    vel_seed_drift : bool
        Give a track with no history yet the field's own drift rather than zero
        -- the embryo slides bodily, so zero is a wrong guess, not a neutral one.
    fork_threshold : float or None
        The probability a *second* child must beat for a matched parent to
        fork. `None` keeps the pure one-to-one assignment, which cannot express
        a division at all and scores a flat zero on the metric's division term.
        Set it *below* `threshold`, not above: the true second daughter runs a
        median 0.368 where the assignment commits at 0.4, which is exactly why
        it is dropped. The child must be one the Hungarian left unclaimed and a
        parent may fork once, so nothing is stolen. Distinct from `match="child"`, which removes the constraint
        instead of pricing it -- measured at 17 division false positives for 0
        true ones.
    """
    cand = np.asarray(cand, dtype=float).reshape(-1, 5)
    coords = np.asarray(coords)
    if len(cand) == 0:
        return np.empty((0, 4))

    p_um = pos_um(coords)
    frame_of = coords[:, 0].astype(int)
    step = cand[:, 4].astype(int)
    cand = cand[step == 1]
    if len(cand) == 0:
        return np.empty((0, 4))

    src = cand[:, 0].astype(int)
    tgt = cand[:, 1].astype(int)
    order = np.argsort(frame_of[src], kind="stable")
    src, tgt, prob, dist = src[order], tgt[order], cand[order, 2], cand[order, 3]
    t_of = frame_of[src]
    bounds = np.searchsorted(t_of, np.unique(t_of))
    bounds = np.append(bounds, len(t_of))

    if emb is not None and emb_weight and len(emb) < len(coords):
        # `recover` appends nodes the contrastive model never saw, so an
        # embedding array shorter than `coords` means somebody re-linked a
        # recovered graph with the cosine term still on. Indexing would go out
        # of bounds a few rows later; say what is wrong instead.
        raise ValueError(
            f"embeddings cover {len(emb)} of {len(coords)} nodes; recovered "
            "nodes have none, so the cosine term cannot be used after recover"
        )
    # The EMA track state, when asked for: a track's embedding is a blend of
    # every detection along it, not just the last one.
    track_emb = None
    if emb is not None and emb_weight and emb_momentum:
        track_emb = np.asarray(emb, np.float32).copy()

    vel = has_vel = drift = None
    if vel_weight or np.isfinite(vel_max_residual):
        vel = np.zeros((len(coords), 3))
        has_vel = np.zeros(len(coords), bool)
        if vel_seed_drift:
            from recover import drift_table
            drift = drift_table(coords)

    out = []
    for b in range(len(bounds) - 1):
        lo, hi = bounds[b], bounds[b + 1]
        a_rows, b_rows = src[lo:hi], tgt[lo:hi]
        p, d = prob[lo:hi], dist[lo:hi]

        parents, pi = np.unique(a_rows, return_inverse=True)
        children, ci = np.unique(b_rows, return_inverse=True)

        if emb is not None and emb_weight:
            parent_vec = (track_emb if track_emb is not None else emb)[a_rows]
            sim = (parent_vec * emb[b_rows]).sum(1)
            p = (1 - emb_weight) * p + emb_weight * (sim + 1) / 2
        base = _log_odds(p, threshold, cost)

        vel_resid = v_used = None
        if vel is not None:
            t_here = int(coords[a_rows[0], 0])
            seed_v = np.asarray((drift or {}).get(t_here, np.zeros(3)), float)
            v_used = np.where(has_vel[a_rows][:, None], vel[a_rows],
                              seed_v[None, :])
            vel_resid = np.linalg.norm(
                p_um[b_rows] - (p_um[a_rows] + v_used), axis=1)
            base = base + vel_weight * vel_resid
        # The flow term needs a set of links to estimate the field's motion
        # from. The model's own above-threshold candidates are the seed; two
        # passes then re-estimate it from what the assignment actually chose.
        penalty = np.zeros(len(p))
        if flow_weight or np.isfinite(max_residual):
            seed = base < 0
            if seed.sum() < 3:
                seed = np.ones(len(p), bool)
            resid = _residual(p_um, a_rows, b_rows, seed, flow_k)
            for _ in range(max(int(flow_passes) - 1, 0)):
                chosen = _solve_pair(base + flow_weight * resid, pi, ci,
                                     len(parents), len(children),
                                     resid > max_residual)
                if chosen.sum() < 3:
                    break
                resid = _residual(p_um, a_rows, b_rows, chosen, flow_k)
            penalty = flow_weight * resid
            forbid = resid > max_residual
        else:
            forbid = np.zeros(len(p), bool)
        if vel_resid is not None:
            forbid = forbid | (vel_resid > vel_max_residual)

        keep = _commit(base + penalty, a_rows, b_rows, pi, ci, len(parents),
                       len(children), forbid, match)

        # A second child for a parent the assignment already matched, when the
        # model scores it above `fork_threshold`.
        #
        # This is *not* `match="child"`, which drops the one-to-one constraint
        # outright and was measured at 17 division false positives for 0 true
        # ones. The constraint is kept and a fork is priced: the extra link must
        # still have negative cost after the penalty, the child must be one the
        # Hungarian left free, and a parent may fork once. The metric pays 0.1 x
        # division Jaccard and charges a wrong link as one edge false positive
        # among ~19 000, so the penalty is what sets that exchange rate.
        if fork_threshold is not None:
            taken_child = np.zeros(len(coords), bool)
            taken_child[b_rows[keep]] = True
            matched_parent = np.zeros(len(coords), bool)
            matched_parent[a_rows[keep]] = True
            forked = np.zeros(len(coords), bool)
            # Highest probability first, and a plain threshold rather than a
            # penalty added to the log-odds. A penalty can only make a fork
            # *rarer* than an ordinary link, and the difficulty is the opposite:
            # the true second daughter runs a median 0.368 where the assignment
            # commits at 0.4. A penalty of even 0.5 log-odds would demand
            # p > 0.66 and fire never -- measured, it committed nothing.
            for i in np.argsort(-p):
                if p[i] <= fork_threshold:
                    break
                a, b = a_rows[i], b_rows[i]
                if forbid[i] or keep[i] or taken_child[b] or not matched_parent[a] \
                        or forked[a]:
                    continue
                keep[i] = True
                taken_child[b] = True
                forked[a] = True

        for i in np.flatnonzero(keep):
            out.append((a_rows[i], b_rows[i], p[i], d[i]))

        if vel is not None:
            # Only a matched track has a velocity; an unmatched detection keeps
            # `has_vel` false and falls back to the field drift next frame.
            kept = np.flatnonzero(keep)
            moved = p_um[b_rows[kept]] - p_um[a_rows[kept]]
            vel[b_rows[kept]] = vel_momentum * v_used[kept] + \
                (1 - vel_momentum) * moved
            has_vel[b_rows[kept]] = True

        if track_emb is not None:
            kept = np.flatnonzero(keep)
            blended = (emb_momentum * track_emb[a_rows[kept]]
                       + (1 - emb_momentum) * emb[b_rows[kept]])
            norm = np.linalg.norm(blended, axis=1, keepdims=True)
            track_emb[b_rows[kept]] = np.where(norm > 0, blended / np.maximum(norm, 1e-8),
                                               emb[b_rows[kept]])

    return np.asarray(out, dtype=float).reshape(-1, 4)


def _residual(p_um, a_rows, b_rows, seed, flow_k):
    """Each candidate's displacement minus its neighbourhood's, in microns."""
    anchor = p_um[a_rows[seed]]
    disp = p_um[b_rows[seed]] - anchor
    if len(anchor) == 0:
        return np.linalg.norm(p_um[b_rows] - p_um[a_rows], axis=1)
    k = min(flow_k, len(anchor))
    _, near = cKDTree(anchor).query(p_um[a_rows], k=k)
    flow = np.median(disp[near.reshape(len(a_rows), -1)], axis=1)
    return np.linalg.norm(p_um[b_rows] - p_um[a_rows] - flow, axis=1)


def _commit(cost, a_rows, b_rows, pi, ci, n_parents, n_children, forbid, match):
    """Which candidates of this frame pair to commit to.

    `link_video` solves a one-to-one assignment, on the grounds that the ground
    truth has no divisions so a second child is a guaranteed false positive.
    That argument is about the *annotation*; the predictions are over-detected
    by ~8%, so two detections of one cell both legitimately want the same
    parent, and exclusivity forces one of them to steal a different parent or
    go unlinked. `"child"` drops the constraint on the parent side -- every
    child takes its best parent, a parent may keep several children -- which is
    the rule the ground truth's "at most one parent" actually implies.
    """
    if match == "hungarian":
        return _solve_pair(cost, pi, ci, n_parents, n_children, forbid)
    if match not in ("child", "parent"):
        raise ValueError(f"unknown match {match!r}; use hungarian, child or parent")

    keep = np.zeros(len(cost), bool)
    usable = np.flatnonzero(~forbid & (cost < 0))
    if not len(usable):
        return keep
    # Best (lowest cost) candidate per child, or per parent.
    side = b_rows[usable] if match == "child" else a_rows[usable]
    order = np.lexsort((cost[usable], side))
    first = np.ones(len(order), bool)
    first[1:] = side[order][1:] != side[order][:-1]
    keep[usable[order[first]]] = True
    return keep


def _solve_pair(cost, pi, ci, n_parents, n_children, forbid):
    """Hungarian over one frame pair; returns a keep mask over the candidates.

    Pairs that were never candidates, and those `forbid` rules out, are given a
    cost no assignment would prefer rather than a large finite one: every row
    must be assigned something, so a finite dummy would make the solver
    maximise the *number* of links before caring about their quality.
    """
    dense = np.zeros((n_parents, n_children), np.float64)
    where = np.full((n_parents, n_children), -1, np.int64)
    usable = ~forbid
    dense[pi[usable], ci[usable]] = cost[usable]
    where[pi[usable], ci[usable]] = np.flatnonzero(usable)
    r, c = linear_sum_assignment(dense)
    keep = np.zeros(len(cost), bool)
    picked = where[r, c]
    picked = picked[(picked >= 0) & (dense[r, c] < 0)]
    keep[picked] = True
    return keep


# --------------------------------------------------------------------------
# track-level stages
# --------------------------------------------------------------------------

def _components(n_nodes, edges):
    parent = np.arange(n_nodes)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, t in np.asarray(edges)[:, :2].astype(int):
        ra, rb = find(s), find(t)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n_nodes)])


def track_length(coords, edges):
    """Per node, the number of nodes in the chain it belongs to."""
    label = _components(len(coords), edges)
    size = np.bincount(label, minlength=len(coords))
    return size[label]


def prune_short(coords, edges, min_track=3, keep_index=False):
    """Drop every chain of fewer than `min_track` nodes, edges and all.

    This is the only node reduction that pays, and it moves two terms at once:
    a spurious chain contributes false-positive edges *and* inflates `N_pred`,
    which the score penalises at 0.1 per unit of relative excess.
    """
    if min_track <= 1 or len(coords) == 0:
        return (coords, edges, np.arange(len(coords))) if keep_index else (coords, edges)
    keep = track_length(coords, edges) >= min_track
    index = np.full(len(coords), -1, np.int64)
    index[keep] = np.arange(int(keep.sum()))
    edges = np.asarray(edges, float).reshape(-1, 4)
    alive = keep[edges[:, 0].astype(int)] & keep[edges[:, 1].astype(int)]
    kept = edges[alive].copy()
    kept[:, 0] = index[kept[:, 0].astype(int)]
    kept[:, 1] = index[kept[:, 1].astype(int)]
    out = (coords[keep], kept)
    return (*out, np.flatnonzero(keep)) if keep_index else out


def propose_divisions(
    coords,
    edges,
    cand,
    min_prob=0.2,
    max_parent_um=9.0,
    max_existing_child_um=10.0,
    max_sister_um=14.0,
    symmetry_tau=0.6,
    diverge_um=2.25,
    require_divergence=True,
    require_mutual_nn=True,
    frame_frac_cap=0.0076,
    global_frac_cap=0.00375,
    div_prob=None,
    div_prob_min=0.0,
    div_prob_rank=True,
    steal_margin=None,
    return_stats=False,
):
    """Give a parent a second child where the geometry says a cell divided.

    The assignment is a one-to-one matching, so it cannot express a division at
    all: the score is ``adj_edge_jaccard + 0.1 * division_jaccard`` and this
    pipeline was scoring a flat 0 on the second term, with `division_fp` at 0 --
    not wrong divisions, *no* divisions. This is the stage that proposes them.

    The gate stack is the public "safe division" repair (see PublicChanges.md,
    the 946 Harmonic Fusion notebook), which reached the strongest public score
    of the family. A parent already holding exactly one committed child may take
    a second one when every one of these holds:

    * the second child is **free** -- nothing has claimed it as a child. A node
      with two parents is an invalid lineage, so a claimed node can only be
      taken by *moving* the link, never by adding a second one: that is
      ``steal_margin``, off by default and described below.
    * ``min_prob`` -- the model's own probability for the parent/second-child
      pair. **This gate is ours, not the notebook's**: 946 proposes on geometry
      alone (it has no scored candidate graph), while here every pair carries a
      probability, and on the true second daughters it runs a median 0.368
      against a threshold the assignment set at 0.45. That is exactly why they
      were dropped, and it is the sharpest signal available here.
    * ``max_parent_um`` / ``max_existing_child_um`` / ``max_sister_um`` --
      parent-to-candidate, parent-to-first-child and sister-to-sister distances.
    * ``require_mutual_nn`` -- the candidate must be the free node *nearest the
      first child*, which is what makes a pair of sisters rather than a cell
      and an unrelated neighbour.
    * ``require_divergence`` -- both sisters must themselves have a child at
      ``t + 2``, and those grandchildren must be at least ``diverge_um`` further
      apart than the sisters are. A real division separates; two detections of
      one cell do not.
    * ``symmetry_tau`` -- the parent must sit between its daughters, not beside
      one of them: ``|d1 - d2| / mean(d1, d2) <= tau``.

    ``div_prob`` is the division classifier's opinion of the *parent*, one entry
    per node from `main_division_predict.py` (`nan` where unscored). It enters
    twice and both matter:

    * as a gate, ``div_prob_min`` -- and this is the only signal here that is
      not geometry or the edge model's link probability, which is why it is
      worth the extra pass. `README_edge.md` establishes that precision is the
      binding constraint and that no rearrangement of the existing features
      supplies it; the classifier reads the image, where the mitotic figure
      actually is.
    * as the ranking, ``div_prob_rank`` -- the caps below admit only a handful
      of proposals per frame, so *which* ones they admit is most of the
      decision. Ordering by ``parent_dist + 0.15 * sister_dist`` prefers the
      tightest pair; ordering by the classifier prefers the one that looks like
      a division.

    A parent with no score (`nan`) is treated as unscored rather than as zero:
    it falls back to the geometric ordering and is only gated if
    ``div_prob_min`` is set, so a partially-scored folder degrades to the old
    behaviour instead of silently dropping every division.

    Then the caps: ``frame_frac_cap`` of the eligible parents in a frame and
    ``global_frac_cap`` of all nodes, proposals taken in order of
    ``parent_dist + 0.15 * sister_dist``. Divisions are rare -- 26 in 71 stacks
    for 44b6, 125 in 128 for 6bba -- so an uncapped stage that fires on 1% of
    parents is wrong whatever its gates say.

    ``steal_margin`` lifts the free-child restriction, at a price. With it set,
    a node the assignment already gave to another parent may be taken as the
    second daughter, and the edge that held it is **deleted** -- the lineage
    stays valid because the link moves rather than duplicates. The price is
    that probability difference: a steal is proposed only when

        p(new parent -> q)  -  p(old parent -> q)  >=  steal_margin

    so 0.0 takes any pair the model likes at least as much as the assignment
    did, and a positive value demands the model actively prefer the fork. The
    old parent is left a track end, which is the cost the caps and the gates
    then have to justify.

    **MEASURED NEGATIVE on fold 0 -- off, and kept for the result.** It was
    built to lift a ceiling it does not lift, and the measurement says why.

    All 26 annotated divisions are reachable after `recover`: every one has a
    parent holding exactly one committed child, and that child is a true
    daughter. In 12 of them the *other* daughter is already claimed, which is
    exactly this knob's case -- and taking it moves nothing:

        arm                                    forks    divisions in pool
        production gates                        6 481          6
        steal 0.0 / 0.1 / -0.1              6.5k-6.9k          6
        ungated gates, no steal                52 537         11
        ungated + steal 0.0                    52 710         11
        steal -1.0 + min_prob 0.02 + ungated  454 012          8

    Two gates bind independently on those 12, and opening one alone reaches
    none of them: the model scores the true second daughter a median **0.139**
    against `min_prob` 0.3 (9 of 12 rejected), while scoring the assignment's
    competing edge for the same node 0.713, so the price rejects 9 of 12 too.

    Opening both is worse than not stealing, and the mechanism is intrinsic:
    this stage only considers a parent holding exactly one child, and every
    steal *deletes* some other parent's link. At a permissive price thousands
    of tracks lose their only child and stop being eligible to propose at all,
    taking true divisions with them -- 11 down to 8. Stealing for one division
    costs the eligibility of others. (`min_prob` 0.02 without stealing also
    scores 11, so the floor is not what loses them; the steal is.)

    The binding constraint is upstream of every knob here: the edge model does
    not believe these links. Nothing in post-processing can rank what it scores
    at 0.139 against a 0.713 alternative.

    Runs *before* ``prune_short``: attaching a free detection to a parent puts
    it in that parent's component, so a second daughter that would have been
    pruned as a fragment survives on the strength of the link.
    """
    coords = np.asarray(coords)
    edges = np.asarray(edges, float).reshape(-1, 4)
    stats = {k: 0 for k in (
        "eligible_parents", "free_children", "prob_rejected", "distance_rejected",
        "mutual_nn_rejected", "divergence_rejected", "symmetry_rejected",
        # divergence splits two ways and they want different fixes: a pair whose
        # sisters simply do not separate enough, and a pair where one sister has
        # no committed child at t+2 to measure the separation with at all.
        "divergence_no_grandchild", "divergence_too_close",
        "prob_cls_rejected", "proposals", "capped", "added",
        # `steal_margin` only. `steal_candidates` counts claimed nodes that
        # reached the margin test at all, so a zero there means the stage never
        # saw a claimed node worth trying rather than that the price was wrong.
        "steal_candidates", "steal_margin_rejected", "stolen",
    )}
    cls = None if div_prob is None else np.asarray(div_prob, float).ravel()
    if cls is not None and len(cls) != len(coords):
        raise ValueError(
            f"div_prob covers {len(cls)} of {len(coords)} nodes; it must be the "
            "array main_division_predict.py wrote for *this* prediction npz"
        )
    if len(edges) == 0 or len(coords) == 0:
        return (edges, stats) if return_stats else edges

    p_um = pos_um(coords)
    frame_of = coords[:, 0].astype(int)
    src = edges[:, 0].astype(int)
    tgt = edges[:, 1].astype(int)

    # One committed child per parent is what the Hungarian produces; a parent
    # with none is a track end and has nothing to divide from.
    n_children = np.bincount(src, minlength=len(coords))
    only_child = np.full(len(coords), -1, np.int64)
    only_child[src[n_children[src] == 1]] = tgt[n_children[src] == 1]
    has_parent = np.zeros(len(coords), bool)
    has_parent[tgt] = True
    # Which edge row holds each claimed node, so a steal can delete exactly the
    # link it replaces. Last writer wins, which matches `has_parent`: a node
    # claimed twice is already malformed and either row is as wrong as the other.
    parent_row = np.full(len(coords), -1, np.int64)
    parent_row[tgt] = np.arange(len(edges))

    # The model's opinion on every pair it scored, for the min_prob gate.
    cand = np.asarray(cand, float).reshape(-1, 5)
    cand = cand[cand[:, 4] == 1] if len(cand) else cand
    prob_of = {}
    for s, t, pr in zip(cand[:, 0].astype(int), cand[:, 1].astype(int), cand[:, 2]):
        prob_of[(s, t)] = pr

    by_frame = rows_by_frame(coords)
    committed = {(int(a), int(b)) for a, b in zip(src, tgt)}
    global_cap = max(1, int(round(len(coords) * global_frac_cap)))

    added = []
    used_parents, used_children = set(), set()
    stolen_rows: set[int] = set()
    for t in sorted(by_frame):
        if len(added) >= global_cap:
            break
        rows_next = by_frame.get(t + 1)
        if rows_next is None:
            continue
        parents = [int(r) for r in by_frame[t]
                   if only_child[r] >= 0 and frame_of[only_child[r]] == t + 1]
        # With a steal price, a claimed node is a candidate too -- and it has to
        # be in the KD-tree as well, or `require_mutual_nn` would keep answering
        # with the nearest *free* node and the steal would never be reachable
        # through the gate that matters most.
        free = [int(r) for r in rows_next
                if not has_parent[r] or steal_margin is not None]
        if not parents or not free:
            continue
        stats["eligible_parents"] += len(parents)
        stats["free_children"] += sum(1 for r in free if not has_parent[r])

        free_pos = p_um[free]
        tree = cKDTree(free_pos)
        frame_cap = max(1, int(round(len(parents) * frame_frac_cap)))

        proposals = []
        for parent in parents:
            child = int(only_child[parent])
            child_dist = float(np.linalg.norm(p_um[parent] - p_um[child]))
            if child_dist > max_existing_child_um:
                continue
            # 946 queries with the *first child's* position, not the parent's:
            # the second daughter is the free node nearest its sister.
            if require_mutual_nn:
                # The node nearest the first child -- one free, and with
                # `steal_margin` also one claimed, as *separate* candidates.
                #
                # Both halves of that are load-bearing. The first child is itself
                # a claimed node at distance zero from the query point, so a
                # plain nearest query answers "the child", the `q == child` guard
                # drops it and the stage proposes nothing. And taking simply the
                # nearest non-child node lets a claimed one occupy the single
                # slot this gate allows, so enabling steals *removes* free-node
                # proposals that would have passed -- measured as forks going
                # 1257 -> 913 with the price attached. A steal has to be an extra
                # option, never a substitute.
                k = min(8, len(free))
                _, idx = tree.query(p_um[child], k=k)
                queries, want_free, want_claimed = [], True, steal_margin is not None
                for i in np.atleast_1d(idx):
                    q_i = free[int(i)]
                    if q_i == child:
                        continue
                    if not has_parent[q_i] and want_free:
                        queries.append(q_i)
                        want_free = False
                    elif has_parent[q_i] and want_claimed:
                        queries.append(q_i)
                        want_claimed = False
                    if not want_free and not want_claimed:
                        break
            else:
                queries = [free[i] for i in tree.query_ball_point(p_um[parent], max_parent_um)]

            for q in queries:
                if q == child or (parent, q) in committed:
                    continue
                prob = prob_of.get((parent, q), 0.0)
                if prob < min_prob:
                    stats["prob_rejected"] += 1
                    continue
                # A claimed node costs the link that holds it. Priced here,
                # before the geometry gates, so `steal_margin_rejected` counts
                # pairs the price turned down rather than ones the geometry
                # would have refused anyway.
                steal_from = -1
                if has_parent[q]:
                    row = int(parent_row[q])
                    old_parent = int(edges[row, 0]) if row >= 0 else -1
                    if row < 0 or old_parent == parent:
                        continue
                    stats["steal_candidates"] += 1
                    if prob - float(edges[row, 2]) < steal_margin:
                        stats["steal_margin_rejected"] += 1
                        continue
                    steal_from = row
                # The classifier's view of the parent. `nan` means unscored --
                # not "not dividing" -- so it passes the gate and falls back to
                # the geometric ranking below.
                p_cls = float(cls[parent]) if cls is not None else float("nan")
                if div_prob_min > 0 and p_cls == p_cls and p_cls < div_prob_min:
                    stats["prob_cls_rejected"] += 1
                    continue
                parent_dist = float(np.linalg.norm(p_um[parent] - p_um[q]))
                sister_dist = float(np.linalg.norm(p_um[child] - p_um[q]))
                if parent_dist > max_parent_um or sister_dist > max_sister_um:
                    stats["distance_rejected"] += 1
                    continue
                if require_divergence:
                    g1, g2 = only_child[child], only_child[q]
                    if g1 < 0 or g2 < 0 or frame_of[g1] != t + 2 or frame_of[g2] != t + 2:
                        stats["divergence_rejected"] += 1
                        stats["divergence_no_grandchild"] += 1
                        continue
                    if float(np.linalg.norm(p_um[g1] - p_um[g2])) - sister_dist < diverge_um:
                        stats["divergence_rejected"] += 1
                        stats["divergence_too_close"] += 1
                        continue
                if symmetry_tau > 0:
                    denom = max((child_dist + parent_dist) / 2.0, 1e-6)
                    if abs(child_dist - parent_dist) / denom > symmetry_tau:
                        stats["symmetry_rejected"] += 1
                        continue
                # Lower sorts first. Geometry gives `parent_dist + 0.15 *
                # sister_dist`; the classifier gives `-p_cls`, so a confident
                # division outranks a merely tight pair.
                order_key = (-p_cls if (div_prob_rank and p_cls == p_cls)
                             else parent_dist + 0.15 * sister_dist)
                proposals.append((order_key, parent, q, prob, parent_dist,
                                  steal_from))

        stats["proposals"] += len(proposals)
        proposals.sort(key=lambda item: item[0])
        added_here = 0
        for _, parent, q, prob, parent_dist, steal_from in proposals:
            if len(added) >= global_cap or added_here >= frame_cap:
                stats["capped"] += 1
                break
            # `has_parent[q]` is not a refusal for a steal -- being claimed is
            # the case it exists for -- but `used_children` still is: two
            # parents cannot both take the same node in one pass.
            if q in used_children or parent in used_parents:
                continue
            if has_parent[q] and steal_from < 0:
                continue
            if steal_from >= 0:
                if steal_from in stolen_rows:
                    continue        # already moved this frame; do not double-spend
                stolen_rows.add(steal_from)
                stats["stolen"] += 1
            added.append((parent, q, prob, parent_dist))
            used_children.add(q)
            used_parents.add(parent)
            has_parent[q] = True
            added_here += 1

    stats["added"] = len(added)
    if stolen_rows:
        # Delete before appending: the rows the steals replaced are indices into
        # the array as it was, and appending first would shift nothing but makes
        # the intent harder to read.
        edges = edges[np.setdiff1d(np.arange(len(edges)),
                                   np.fromiter(stolen_rows, np.int64))]
    if added:
        edges = np.vstack([edges, np.asarray(added, float)])
    return (edges, stats) if return_stats else edges


def merge_shadow_forks(
    coords,
    edges,
    radius=7.0,
    min_shadow=3,
    shadow_frac=0.6,
    max_sister_um=14.0,
    global_frac_cap=0.00375,
    return_stats=False,
):
    """Rejoin a division that the detector split into two parallel tracks.

    `merge_duplicate_parents` had the right diagnosis and the wrong instrument.
    In 34 of the 151 annotated divisions the second daughter is already claimed
    by a *duplicate detection of the same parent cell*, so the fork exists in
    the graph but no node carries it -- and a one-frame distance test cannot
    tell that duplicate from an ordinary neighbour, because both sit at 6-11 um.
    Merging on distance alone fired 10 508 times for 34 opportunities.

    Time tells them apart. A cell found twice produces two tracks that **shadow
    each other** -- within a cell's width, frame after frame, for as long as both
    detections survive. That is `dedup.suppress_parallel`'s signature, and it is
    the one this uses.

    What makes it a *division* rather than a plain duplicate is how the
    shadowing **ends**. When the cell divides, one copy follows one daughter and
    the other copy follows the other, and the two tracks separate for good. So:

        ...a-a-a-a-a          two tracks within `radius` for >= min_shadow
        ...b-b-b-b-b          consecutive frames, then at t+1
                   \\
                    a'         they part, and a' and b' are within
                    b'         max_sister_um of each other

    `shadow_frac` is what keeps that from firing on any two neighbours that
    happen to drift together: the run has to be most of the shorter track's
    life, not merely `min_shadow` frames of it.

    The last shadowed frame is the division. The loser's child link is re-pointed
    onto the winner, which then forks; the loser's own history is left dangling
    and `prune_short` collects it, which also removes the duplicate detections
    from the node count.

    Nothing is invented: both edges keep the probability the model gave them.
    """
    coords = np.asarray(coords)
    edges = np.asarray(edges, float).reshape(-1, 4)
    stats = {k: 0 for k in ("shadow_pairs", "too_short", "no_split",
                            "sister_rejected", "merged")}
    if len(edges) == 0 or len(coords) == 0:
        return (edges, stats) if return_stats else edges

    p_um = pos_um(coords)
    frame_of = coords[:, 0].astype(int)
    src, tgt = edges[:, 0].astype(int), edges[:, 1].astype(int)
    label = _components(len(coords), edges)
    track_len = np.bincount(label, minlength=len(coords))[label]

    n_children = np.bincount(src, minlength=len(coords))
    only_child = np.full(len(coords), -1, np.int64)
    single = n_children[src] == 1
    only_child[src[single]] = tgt[single]
    edge_of_parent = np.full(len(coords), -1, np.int64)
    edge_of_parent[src[single]] = np.flatnonzero(single)

    by_frame = rows_by_frame(coords)
    # Which frames each pair of tracks shadows in, and which nodes carried it.
    shadow: dict[tuple[int, int], dict[int, tuple[int, int]]] = {}
    for t, rows in by_frame.items():
        if len(rows) < 2:
            continue
        pairs = cKDTree(p_um[rows]).query_pairs(radius, output_type="ndarray")
        for i, j in pairs:
            u, v = int(rows[i]), int(rows[j])
            lu, lv = int(label[u]), int(label[v])
            if lu == lv:
                continue
            key = (lu, lv) if lu < lv else (lv, lu)
            shadow.setdefault(key, {})[t] = (u, v) if lu < lv else (v, u)

    global_cap = max(1, int(round(len(coords) * global_frac_cap)))
    moved: dict[int, int] = {}
    spent: set[int] = set()

    for (la, lb), frames in sorted(shadow.items(), key=lambda kv: -len(kv[1])):
        if len(moved) >= global_cap:
            break
        stats["shadow_pairs"] += 1
        ts = sorted(frames)
        # the run of consecutive shadowed frames that ends the shadowing
        run_end = ts[-1]
        run = 1
        while run < len(ts) and ts[-1 - run] == run_end - run:
            run += 1
        # The run must also be most of the pair's shared life. Two unrelated
        # neighbours drift within `radius` for a few frames all the time; a
        # duplicate shadows its original for essentially the whole overlap.
        span_a = int(track_len[frames[ts[0]][0]])
        span_b = int(track_len[frames[ts[0]][1]])
        if run < min_shadow or run < shadow_frac * min(span_a, span_b):
            stats["too_short"] += 1
            continue
        t = run_end
        a_t, b_t = frames[t]
        if la in spent or lb in spent:
            continue
        ca, cb = int(only_child[a_t]), int(only_child[b_t])
        # Both copies must carry the parent forward, into different nodes, and
        # those must be the frame after -- that pair is the division.
        if ca < 0 or cb < 0 or ca == cb:
            stats["no_split"] += 1
            continue
        if frame_of[ca] != t + 1 or frame_of[cb] != t + 1:
            stats["no_split"] += 1
            continue
        sister = float(np.linalg.norm(p_um[ca] - p_um[cb]))
        if sister > max_sister_um or sister <= radius:
            # <= radius means they never actually parted: a plain duplicate,
            # not a division, and dropping one is `prune_short`'s business.
            stats["sister_rejected"] += 1
            continue
        keep_node, drop_node = (a_t, b_t)
        if track_len[b_t] > track_len[a_t]:
            keep_node, drop_node = b_t, a_t
        row = int(edge_of_parent[drop_node])
        if row < 0:
            continue
        moved[row] = keep_node
        spent.update((la, lb))
        stats["merged"] += 1

    if moved:
        edges = edges.copy()
        for row, keep in moved.items():
            edges[row, 0] = keep
            edges[row, 3] = float(np.linalg.norm(
                p_um[keep] - p_um[int(edges[row, 1])]))
    return (edges, stats) if return_stats else edges


def merge_duplicate_parents(
    coords,
    edges,
    max_dup_um=7.0,
    max_parent_um=9.0,
    max_sister_um=14.0,
    symmetry_tau=0.6,
    diverge_um=3.0,
    require_divergence=True,
    require_mutual=True,
    global_frac_cap=0.00375,
    return_stats=False,
):
    """Pull a division that got split across two duplicate detections onto one node.

    `propose_divisions` can only use a second daughter nothing has claimed, and
    `diag_divisions.py` says that is where most of the recall goes: of 151
    annotated divisions, 59 fail because the second daughter already has a
    parent. Those are not assignment mistakes -- the model gives the rival a
    median probability of 0.719 for her against our parent's 0.016 -- and in
    **34 of the 59 the rival is itself a detection of the same annotated parent
    cell**. The graph is over-detected, the dividing cell was found twice, and
    the two copies took one daughter each. The fork is already there; it is just
    spread over two nodes, and the metric wants one node of out-degree 2.

    So this moves the second copy's child onto the first copy rather than
    inventing a link. Both parents keep the probability the model gave their own
    child, which is the point: no link here is a guess.

    **Measured, and it does not work -- off by default.** The premise is sound
    and the execution is not: "the same cell, found twice" is not recoverable
    from geometry at these radii. The rival pairs sit at p25 6.3 um and median
    8.1, and so do thousands of genuinely distinct neighbours, so a 7 um merge
    fires 10 508 times across 199 stacks for 34 reachable divisions. Requiring
    both parents *and* both daughters to be mutual nearest neighbours cuts that
    to 6 324 and division false positives from 114 to 85, still 200x too loose.
    On the fold: the stage alone is -0.0001, and in front of `propose_divisions`
    it takes +0.0114 down to +0.0077. Kept because the diagnosis behind it is
    solid and a real duplicate test -- `dedup.suppress_parallel`'s shadow-track
    signature, which watches a pair over its whole life rather than one frame --
    could still make it pay. Do not switch it on without one.

    ``max_dup_um`` is what makes a pair of parents "the same cell". It should sit
    near the detector's NMS radius -- two peaks it left standing are, by
    construction, at least that far apart -- and the measured separation of these
    rival pairs is p25 6.3 um, median 8.1.

    The loser keeps its own history and simply ends there; `prune_short` removes
    it if what is left is a fragment, which also takes a duplicate detection out
    of the node count.

    Returns the edges with those child links re-pointed. Runs before
    `propose_divisions`, which then works on what is genuinely unclaimed.
    """
    coords = np.asarray(coords)
    edges = np.asarray(edges, float).reshape(-1, 4)
    stats = {k: 0 for k in (
        "pairs_considered", "not_mutual", "sister_rejected", "parent_rejected",
        "divergence_rejected", "symmetry_rejected", "merged",
    )}
    if len(edges) == 0 or len(coords) == 0:
        return (edges, stats) if return_stats else edges

    p_um = pos_um(coords)
    frame_of = coords[:, 0].astype(int)
    src = edges[:, 0].astype(int)
    tgt = edges[:, 1].astype(int)

    n_children = np.bincount(src, minlength=len(coords))
    only_child = np.full(len(coords), -1, np.int64)
    single = n_children[src] == 1
    only_child[src[single]] = tgt[single]
    # Which edge row carries each parent's single child, so it can be re-pointed.
    edge_of_parent = np.full(len(coords), -1, np.int64)
    edge_of_parent[src[single]] = np.flatnonzero(single)

    by_frame = rows_by_frame(coords)
    global_cap = max(1, int(round(len(coords) * global_frac_cap)))
    moved, spent = {}, set()

    for t in sorted(by_frame):
        if len(moved) >= global_cap:
            break
        parents = [int(r) for r in by_frame[t]
                   if only_child[r] >= 0 and frame_of[only_child[r]] == t + 1]
        if len(parents) < 2:
            continue
        pairs = cKDTree(p_um[parents]).query_pairs(max_dup_um, output_type="ndarray")
        # "Same cell, found twice" is not a distance -- at these radii thousands
        # of genuinely distinct cells qualify, and merging on distance alone
        # costs 114 division false positives for 7 true ones (README_edge.md).
        # Both copies must be each other's nearest neighbour among *all* nodes of
        # the frame, and so must the two daughters, which is what a duplicated
        # detection and its sister pair look like and a crowded neighbourhood
        # does not.
        nn_at = {}
        if require_mutual and pairs.size:
            for tt in (t, t + 1):
                rows_tt = by_frame.get(tt)
                if rows_tt is None or len(rows_tt) < 2:
                    nn_at[tt] = {}
                    continue
                dists, near = cKDTree(p_um[rows_tt]).query(p_um[rows_tt], k=2)
                nn_at[tt] = {int(rows_tt[i]): int(rows_tt[near[i, 1]])
                             for i in range(len(rows_tt))}

        cand_pairs = []
        for i, j in pairs:
            a, b = parents[i], parents[j]
            ca, cb = int(only_child[a]), int(only_child[b])
            if ca == cb:
                continue
            if require_mutual:
                if nn_at.get(t, {}).get(a) != b or nn_at.get(t, {}).get(b) != a:
                    stats["not_mutual"] += 1
                    continue
                if nn_at.get(t + 1, {}).get(ca) != cb or nn_at.get(t + 1, {}).get(cb) != ca:
                    stats["not_mutual"] += 1
                    continue
            cand_pairs.append((float(np.linalg.norm(p_um[a] - p_um[b])), a, b, ca, cb))
        cand_pairs.sort()
        stats["pairs_considered"] += len(cand_pairs)

        for _, a, b, ca, cb in cand_pairs:
            if len(moved) >= global_cap:
                break
            if a in spent or b in spent or ca in spent or cb in spent:
                continue
            sister_dist = float(np.linalg.norm(p_um[ca] - p_um[cb]))
            if sister_dist > max_sister_um:
                stats["sister_rejected"] += 1
                continue
            if require_divergence:
                g1, g2 = only_child[ca], only_child[cb]
                if g1 < 0 or g2 < 0 or frame_of[g1] != t + 2 or frame_of[g2] != t + 2:
                    stats["divergence_rejected"] += 1
                    continue
                if float(np.linalg.norm(p_um[g1] - p_um[g2])) - sister_dist < diverge_um:
                    stats["divergence_rejected"] += 1
                    continue
            # Whichever copy sits between the two daughters becomes the fork;
            # the other is the duplicate that gives its child up.
            keep, drop = (a, b)
            da = max(np.linalg.norm(p_um[a] - p_um[ca]), np.linalg.norm(p_um[a] - p_um[cb]))
            db = max(np.linalg.norm(p_um[b] - p_um[ca]), np.linalg.norm(p_um[b] - p_um[cb]))
            if db < da:
                keep, drop = b, a
            d1 = float(np.linalg.norm(p_um[keep] - p_um[ca]))
            d2 = float(np.linalg.norm(p_um[keep] - p_um[cb]))
            if max(d1, d2) > max_parent_um:
                stats["parent_rejected"] += 1
                continue
            if symmetry_tau > 0:
                denom = max((d1 + d2) / 2.0, 1e-6)
                if abs(d1 - d2) / denom > symmetry_tau:
                    stats["symmetry_rejected"] += 1
                    continue
            row = int(edge_of_parent[drop])
            if row < 0:
                continue
            moved[row] = keep
            spent.update((a, b, ca, cb))
            stats["merged"] += 1

    if moved:
        edges = edges.copy()
        for row, keep in moved.items():
            edges[row, 0] = keep
            edges[row, 3] = float(np.linalg.norm(
                p_um[keep] - p_um[int(edges[row, 1])]))
    return (edges, stats) if return_stats else edges


def bridge_tracks(coords, edges, max_um=6.0, drift=None, min_len=2):
    """Join a chain ending at `u` to one starting at `u + 1`, on geometry alone.

    The assignment declines a link whenever the model scores it below the
    threshold, and it is right to -- but a *chain* that ends where another
    begins, one frame apart and with nothing else competing for either end, is
    a different piece of evidence than a single edge in isolation. This adds
    exactly those links: both endpoints already belong to a track, both are
    free, and the drift-corrected distance between them is under `max_um`.

    No new nodes, unlike `recover_nodes` -- this repairs a chain the detector
    did find and the linker cut.
    """
    coords = np.asarray(coords, float)
    edges = np.asarray(edges, float).reshape(-1, 4)
    if len(coords) == 0:
        return edges
    p_um = pos_um(coords)
    rows = rows_by_frame(coords)
    length = track_length(coords, edges)

    has_child = np.zeros(len(coords), bool)
    has_parent = np.zeros(len(coords), bool)
    has_child[edges[:, 0].astype(int)] = True
    has_parent[edges[:, 1].astype(int)] = True

    if drift is None:
        from recover import drift_table
        drift = drift_table(coords)

    added = []
    for u in sorted(rows):
        if u + 1 not in rows or u not in drift:
            continue
        here, there = rows[u], rows[u + 1]
        ends = here[~has_child[here] & (length[here] >= min_len)]
        starts = there[~has_parent[there] & (length[there] >= min_len)]
        if not len(ends) or not len(starts):
            continue
        d = np.linalg.norm(
            p_um[starts][None, :, :] - (p_um[ends] + drift[u])[:, None, :], axis=-1
        )
        gated = d > max_um
        if gated.all():
            continue
        ri, ci = linear_sum_assignment(np.where(gated, NO_LINK, d))
        for i, j in zip(ri, ci, strict=True):
            if not gated[i, j]:
                added.append((ends[i], starts[j], 0.5, float(d[i, j])))
    if not added:
        return edges
    return np.vstack([edges, np.asarray(added, float).reshape(-1, 4)])


def linefit_smooth(coords, edges, window=2, weight=0.74):
    """Move each node toward a line fitted through its own track.

    `pp_public` does this and it is the one part of that module worth its
    runtime here. The point is not that the detector jitters -- smoothing a
    node against its neighbours in *time* was measured and makes the median
    localisation error worse (`analysis_jitter.py`). The point is the node's
    own track: a cell that is tracked over five frames gives five readings of a
    trajectory that is locally straight, and the fit averages the detection
    error down without averaging the motion away.

    A node needs at least three points on an unbranched stretch of its own
    chain, so isolated nodes and two-node chains are left exactly where they
    are. At a track end the window is one-sided and the fit extrapolates, which
    is what `pp_public` does too.

    This is `linefit_smooth_output_graph` from `tracking/pp_public.py`, done in
    one pass over arrays rather than a Python loop per node -- 0.1 s a stack
    against that module's 38.
    """
    coords = np.asarray(coords, float).copy()
    edges = np.asarray(edges, float).reshape(-1, 4)
    if weight <= 0 or window <= 0 or len(edges) == 0:
        return coords

    n = len(coords)
    src = edges[:, 0].astype(int)
    tgt = edges[:, 1].astype(int)
    # Only unbranched steps count: a node with two children (or two parents)
    # has no single track to fit along, and the fit must not cross the branch.
    n_child = np.bincount(src, minlength=n)
    n_parent = np.bincount(tgt, minlength=n)
    step = (coords[tgt, 0] - coords[src, 0]) == 1
    ok = step & (n_child[src] == 1) & (n_parent[tgt] == 1)
    succ = np.full(n, -1, np.int64)
    pred = np.full(n, -1, np.int64)
    succ[src[ok]] = tgt[ok]
    pred[tgt[ok]] = src[ok]

    pos = coords[:, 1:]
    # Walk out along the chain in both directions, one step at a time, so the
    # whole video's neighbourhoods are gathered in 2 * window steps.
    offsets = [np.zeros(n, np.int64)]
    rows = [np.arange(n)]
    for table, sign in ((pred, -1), (succ, 1)):
        current = np.arange(n)
        alive = np.ones(n, bool)
        for k in range(1, int(window) + 1):
            nxt = np.where(alive, table[current], -1)
            alive = nxt >= 0
            current = np.where(alive, nxt, current)
            # A node whose chain ran out parks on row 0 with offset 0; the mask
            # below reads the offset, so it contributes to nothing.
            offsets.append(np.where(alive, sign * k, 0))
            rows.append(np.where(alive, current, 0))
            if not alive.any():
                break

    dt = np.stack(offsets, axis=1).astype(float)          # (n, m)
    idx = np.stack(rows, axis=1)                          # (n, m)
    mask = dt != 0
    mask[:, 0] = True                                     # the node itself

    count = mask.sum(axis=1)
    usable = count >= 3
    w = mask.astype(float)
    mean_dt = (dt * w).sum(1) / np.maximum(count, 1)
    y = pos[idx]                                          # (n, m, 3)
    mean_y = (y * w[:, :, None]).sum(1) / np.maximum(count, 1)[:, None]
    ddt = (dt - mean_dt[:, None]) * w
    var = (ddt * ddt).sum(1)
    cov = (ddt[:, :, None] * (y - mean_y[:, None, :])).sum(1)
    slope = np.zeros_like(cov)
    good = usable & (var > 0)
    slope[good] = cov[good] / var[good, None]
    fitted = mean_y - slope * mean_dt[:, None]            # the line at dt = 0

    w_blend = float(np.clip(weight, 0.0, 1.0))
    coords[good, 1:] = (1 - w_blend) * pos[good] + w_blend * fitted[good]
    return coords


def velocity_link(coords, cand=None, momentum=0.5, max_um=7.0, prob_weight=0.0,
                  seed_drift=True, drift=None, threshold=0.4):
    """Serial linear assignment against a per-track velocity, carried with momentum.

    The ASCENT linker's shape -- walk the frames in order, solve one assignment
    per pair, carry a state forward on the matched tracks -- with the state
    being the cell's **own motion** rather than its appearance:

        v[track] <- m * v[track] + (1 - m) * (pos_child - pos_parent)
        predicted =  pos_parent + v[track]
        cost[i, j] = || pos_next[j] - predicted[i] ||

    This is not the neighbourhood flow `relink.py` corrects by, and not the
    appearance EMA `CompareCV.ipynb` carries. `postprocessing/README.md` records
    per-cell velocity as tried and worse ("tracks fragment too often for a
    cell's own history to be available when it is needed") -- but that was
    measured against the detector's own links, where a track is a handful of
    frames. On this graph the median track is 12 nodes and the p75 is 21, so
    the history the idea needs now exists.

    A track that has just started has no velocity of its own. `seed_drift`
    gives it the field's, from `recover.drift_table`, which is the right prior:
    the embryo slides bodily by up to 7 um a frame, so zero is not a neutral
    guess, it is a wrong one.

    `prob_weight` optionally lets the model buy slack -- microns of distance
    forgiven per unit of probability -- so the two signals can be traded off.
    0 is the pure motion model.
    """
    coords = np.asarray(coords, float)
    p_um = pos_um(coords)
    rows = rows_by_frame(coords)
    n = len(coords)

    if drift is None and seed_drift:
        from recover import drift_table
        drift = drift_table(coords)
    drift = drift or {}

    by_t = {}
    if prob_weight and cand is not None and len(cand):
        cand = np.asarray(cand, float).reshape(-1, 5)
        cand = cand[cand[:, 4].astype(int) == 1]
        frame_of = coords[:, 0].astype(int)
        for a, b, p, _d, _dt in cand:
            by_t.setdefault(int(frame_of[int(a)]), []).append((int(a), int(b), p))

    vel = np.zeros((n, 3))
    has_vel = np.zeros(n, bool)

    out = []
    for t in sorted(rows):
        if t + 1 not in rows:
            continue
        here, there = rows[t], rows[t + 1]
        if not len(here) or not len(there):
            continue

        seed = np.asarray(drift.get(t, np.zeros(3)), float)
        v = np.where(has_vel[here][:, None], vel[here], seed[None, :])
        predicted = p_um[here] + v

        cost = np.linalg.norm(p_um[there][None, :, :] - predicted[:, None, :],
                              axis=-1)
        if prob_weight:
            cost = cost - prob_weight * _prob_matrix(here, there, by_t.get(t, ()))
        gated = cost > max_um
        if gated.all():
            continue
        ri, ci = linear_sum_assignment(np.where(gated, NO_LINK, cost))
        keep = ~gated[ri, ci]
        ri, ci = ri[keep], ci[keep]

        for i, j in zip(ri, ci, strict=True):
            s, t_row = int(here[i]), int(there[j])
            out.append((s, t_row, 1.0, float(np.linalg.norm(p_um[t_row] - p_um[s]))))

        # Only the matched tracks carry a velocity forward; a detection that
        # started its own track has no history yet and falls back to the seed.
        moved = p_um[there[ci]] - p_um[here[ri]]
        vel[there[ci]] = momentum * v[ri] + (1 - momentum) * moved
        has_vel[there[ci]] = True

    return np.asarray(out, dtype=float).reshape(-1, 4)


def ascent_link(coords, emb, cand=None, w_emb=0.5, w_trans=0.5, momentum=0.5,
                threshold=None, max_um=0.0):
    """`notebooks/CompareCV.ipynb`'s linker, as a graph builder.

    A different decision rule from `assign`, in four ways that all matter:

    * **Dense.** The cost is `emb_prev @ emb_next.T` over *every* pair in the
      frame, not just the model's 5 nearest neighbours within 20 um. It can
      therefore propose a link the candidate graph never contained -- worth
      0.2% of ground-truth links, by `diag_edge.py`'s count.
    * **A track state, not a detection.** The parent side of the cost is an EMA
      along the track, `m * track + (1 - m) * detection`, so a cell is compared
      against everything it has looked like rather than against one frame.
    * **A dynamic threshold.** `embeddings.match.dynamic_threshold` puts the
      cutoff halfway between what a *good* match costs in this frame and what
      an average one does, so it tracks the frame's own difficulty instead of
      being a constant the way `assign`'s probability threshold is.
    * **Segment-wise.** Frames are walked in contiguous runs, so a hole in the
      detection set ends the tracks rather than linking across it.

    `w_emb` and `w_trans` blend the cosine with the model's probability
    (`cand`, zero where the model did not score the pair). `w_trans=0` is the
    notebook's "contrastive" row, `w_emb=0` uses the model alone under this
    rule, and the default is its "combined".

    `max_um` optionally refuses a link longer than that, which the dense cost
    otherwise has no reason to: nothing in a cosine knows how far a cell is
    allowed to move. 0 leaves it off, as the notebook has it.
    """
    coords = np.asarray(coords)
    emb = np.asarray(emb, np.float32)
    p_um = pos_um(coords)
    rows = rows_by_frame(coords)
    frames = sorted(rows)

    # Bucketed by the source frame once: the candidate set has 255k rows and
    # walking all of it per frame pair would cost more than the matching does.
    by_t = {}
    if w_trans and cand is not None and len(cand):
        cand = np.asarray(cand, float).reshape(-1, 5)
        cand = cand[cand[:, 4].astype(int) == 1]
        frame_of = coords[:, 0].astype(int)
        for a, b, p, _d, _dt in cand:
            by_t.setdefault(int(frame_of[int(a)]), []).append((int(a), int(b), p))

    out = []
    for segment in _contiguous(frames):
        if len(segment) < 2:
            continue
        track_emb = emb[rows[segment[0]]].copy()
        for k in range(len(segment) - 1):
            here, there = rows[segment[k]], rows[segment[k + 1]]
            if not len(here) or not len(there):
                track_emb = emb[there].copy()
                continue

            sim = track_emb @ emb[there].T
            if w_trans:
                sim = w_emb * sim + w_trans * _prob_matrix(
                    here, there, by_t.get(segment[k], ()))
            elif w_emb != 1.0:
                sim = w_emb * sim
            cost = 1.0 - sim
            far = None
            if max_um:
                far = np.linalg.norm(
                    p_um[there][None, :, :] - p_um[here][:, None, :], axis=-1) > max_um

            # The threshold is a statistic *of the cost matrix*, so the gated
            # pairs have to be invisible to it rather than merely expensive:
            # `dynamic_threshold` drops non-finite entries, but a large finite
            # stand-in would drag its mean up and let everything through.
            cut = threshold
            if cut is None:
                cut = dynamic_threshold(
                    cost if far is None else np.where(far, np.inf, cost),
                    len(track_emb))
            # The solver needs finite costs, so it gets the stand-in -- and the
            # gated pairs are dropped again after it has chosen.
            ri, ci = linear_sum_assignment(
                cost if far is None else np.where(far, NO_LINK, cost))
            keep = cost[ri, ci] <= cut
            if far is not None:
                keep &= ~far[ri, ci]
            ri, ci = ri[keep], ci[keep]

            for i, j in zip(ri, ci, strict=True):
                s, t = int(here[i]), int(there[j])
                out.append((s, t, float(1.0 - cost[i, j]),
                            float(np.linalg.norm(p_um[t] - p_um[s]))))

            # The next frame's parents: the matched ones carry their track
            # forward, the unmatched ones start a track of their own.
            nxt = emb[there].copy()
            blended = momentum * track_emb[ri] + (1 - momentum) * emb[there][ci]
            norm = np.linalg.norm(blended, axis=1, keepdims=True)
            nxt[ci] = np.where(norm > 0, blended / np.maximum(norm, 1e-8),
                               emb[there][ci])
            track_emb = nxt

    return np.asarray(out, dtype=float).reshape(-1, 4)


def _contiguous(frames):
    """`frames` split into runs of consecutive integers."""
    segments = []
    for t in frames:
        if segments and t == segments[-1][-1] + 1:
            segments[-1].append(t)
        else:
            segments.append([t])
    return segments


def _prob_matrix(here, there, triples):
    """The model's probability for each pair of the frame, 0 where unscored.

    The candidate set is sparse -- 5 per node -- so this fills from the
    candidates rather than walking the dense pair list.
    """
    mat = np.zeros((len(here), len(there)), np.float32)
    row = {int(a): i for i, a in enumerate(here)}
    col = {int(b): j for j, b in enumerate(there)}
    for a, b, p in triples:
        i, j = row.get(a), col.get(b)
        if i is not None and j is not None:
            mat[i, j] = p
    return mat


def drop_long(edges, max_um):
    """Refuse a link longer than `max_um`, whatever the model scored it.

    The candidate graph reaches 20 um and a cell moves a median 2 um a frame,
    so the tail of that range is a link between two different cells however
    confident the model is. `pp_public` cuts at 14.2.
    """
    edges = np.asarray(edges, float).reshape(-1, 4)
    if not max_um or len(edges) == 0:
        return edges
    return edges[edges[:, 3] <= max_um]


def link_through_isolated(coords, edges, max_gap=2, max_um=5.0, min_len=2,
                          drift=None):
    """Join two chains through detections that are already there but unlinked.

    `recover_nodes` refuses a bridge whose interpolated position lands on an
    existing detection, and it is right to: a node on top of another one only
    steals the ground-truth match. But the refusal throws the case away, and it
    is the *easier* case -- the detector did find the cell, the linker just left
    it isolated because the model scored both of its links below the threshold.
    Wiring the chain through it costs no new node at all, and turns two
    declined links into two true positives.

    An intermediate node has to be genuinely free (no parent and no child) for
    this: stealing a linked node out of somebody else's track is the ranking
    decision the assignment already made, and it is not this stage's business.
    """
    coords = np.asarray(coords, float)
    edges = np.asarray(edges, float).reshape(-1, 4)
    if len(coords) == 0:
        return edges
    p_um = pos_um(coords)
    rows = rows_by_frame(coords)
    length = track_length(coords, edges)

    has_child = np.zeros(len(coords), bool)
    has_parent = np.zeros(len(coords), bool)
    if len(edges):
        has_child[edges[:, 0].astype(int)] = True
        has_parent[edges[:, 1].astype(int)] = True
    free = ~has_child & ~has_parent

    if drift is None:
        from recover import drift_table
        drift = drift_table(coords)

    added = []
    for gap in range(1, int(max_gap) + 1):
        # The gate is a speed: a cell crossing a two-frame hole has had twice
        # as long to move as one crossing a single frame.
        limit = max_um * (gap + 1) / 2.0
        for u in sorted(rows):
            span = list(range(u, u + gap + 1))
            if u + gap + 1 not in rows or any(t not in drift for t in span):
                continue
            step = np.cumsum([drift[t] for t in span], axis=0)
            total = step[-1]

            here, there = rows[u], rows[u + gap + 1]
            ends = here[~has_child[here] & has_parent[here] & (length[here] >= min_len)]
            starts = there[~has_parent[there] & has_child[there]
                           & (length[there] >= min_len)]
            if not len(ends) or not len(starts):
                continue
            residual = np.linalg.norm(
                p_um[starts][None, :, :] - (p_um[ends] + total)[:, None, :], axis=-1)
            gated = residual > limit
            if gated.all():
                continue
            ri, ci = linear_sum_assignment(np.where(gated, NO_LINK, residual))
            for i, j in zip(ri, ci, strict=True):
                if gated[i, j]:
                    continue
                a, b = int(ends[i]), int(starts[j])
                motion = (p_um[b] - p_um[a]) - total
                chain = [a]
                for k in range(1, gap + 1):
                    want = p_um[a] + step[k - 1] + motion * k / (gap + 1)
                    pool = rows[u + k][free[rows[u + k]]]
                    if not len(pool):
                        break
                    d = np.linalg.norm(p_um[pool] - want, axis=1)
                    if d.min() > max_um:
                        break
                    chain.append(int(pool[int(d.argmin())]))
                if len(chain) != gap + 1:
                    continue
                chain.append(b)
                for s, t in itertools.pairwise(chain):
                    added.append((s, t, 0.5, float(np.linalg.norm(p_um[t] - p_um[s]))))
                # Consume both ends and every node the chain runs through, so a
                # longer gap cannot reuse them.
                has_child[chain[:-1]] = True
                has_parent[chain[1:]] = True
                free[chain[1:-1]] = False
    if not added:
        return edges
    return np.vstack([edges, np.asarray(added, float).reshape(-1, 4)])


def recover_nodes(coords, edges, cand, max_gap=3, max_residual=5.0,
                  min_dup=4.0, threshold=0.4, passes=1, assign_kw=None):
    """Fill the frames a track is missing, then re-decide the links.

    `recover.recover` proposes the nodes; the re-assignment then has to wire
    them in, and the model never scored them -- so the recovered nodes get
    geometric candidates of their own (nearest neighbours in the neighbouring
    frames, at a probability that puts them just above the threshold) while
    every original pair keeps the probability the model gave it.

    `passes` repeats the pair: filling a gap moves the assignment, and the ends
    and starts that leaves behind are gaps the previous round could not see. It
    stops on its own once a round proposes nothing.

    Returns `(coords, edges)`, with the recovered nodes appended to `coords`.
    """
    from recover import drift_table, recover

    coords = np.asarray(coords, float)
    cand = np.asarray(cand, float).reshape(-1, 5)
    for _ in range(max(int(passes), 1)):
        n_before = len(coords)
        # recover() reads a (n, 5) link array; the fifth column is the margin
        # and it only ever indexes the first two.
        links = np.column_stack([edges[:, :4], np.zeros(len(edges))]) if len(edges) \
            else np.empty((0, 5))
        coords, n_added = recover(coords, links, max_gap=max_gap,
                                  max_residual=max_residual, min_dup_dist=min_dup)
        if not n_added:
            break
        drift = drift_table(coords)
        cand = np.vstack([cand, _geometric_candidates(
            coords, np.arange(n_before, len(coords)), drift, threshold=threshold)])
        edges = assign(coords, cand, threshold=threshold, **(assign_kw or {}))
    return coords, edges


def _geometric_candidates(coords, new_rows, drift, threshold=0.4,
                          k=5, max_um=10.0):
    """Candidate edges into and out of the recovered nodes, scored on distance.

    The model cannot speak for a node it never saw, so these carry a synthetic
    probability: 1 at zero drift-corrected distance, falling to the threshold
    at `max_um`. That puts a recovered node's best neighbour ahead of an
    unlinked cell without letting it outbid a link the model is sure of.
    """
    p_um = pos_um(coords)
    rows = rows_by_frame(coords)
    new = {int(r) for r in new_rows}
    out = []
    for u in sorted(rows):
        if u + 1 not in rows:
            continue
        here, there = rows[u], rows[u + 1]
        touches = np.array([int(r) in new for r in here])
        touches_next = np.array([int(r) in new for r in there])
        if not touches.any() and not touches_next.any():
            continue
        shift = drift.get(u, np.zeros(3))
        tree = cKDTree(p_um[there])
        kk = min(k, len(there))
        d, idx = tree.query(p_um[here] + shift, k=kk)
        d = np.atleast_2d(d.reshape(len(here), kk))
        idx = np.atleast_2d(idx.reshape(len(here), kk))
        for i, a in enumerate(here):
            for j in range(kk):
                b = there[idx[i, j]]
                if int(a) not in new and int(b) not in new:
                    continue
                if d[i, j] > max_um:
                    continue
                p = threshold + (1.0 - threshold) * (1.0 - d[i, j] / max_um)
                out.append((a, b, min(p, 0.999), float(d[i, j]), 1))
    return np.asarray(out, float).reshape(-1, 5)
