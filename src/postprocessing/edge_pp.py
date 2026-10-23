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

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

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

    `coords_f` is preferred when the folder carries it: widening the rounded
    column recovers nothing, and the detection head's sub-voxel offsets are a
    mean 0.33 voxel that every micron gate and every image crop below this
    point would otherwise be reading through a quantiser. Folders written
    before `main_edge_predict.py` carried the column fall back to `coords`, so
    an old run still loads and still reproduces its own numbers.
    """
    data = np.load(path, allow_pickle=True)
    coords = (data["coords_f"] if "coords_f" in data.files
              else data["coords"]).astype(np.float64)
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
        which is what the sweep chose.
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
        `(1 - w) * p + w * (cos + 1) / 2`, the blend
        `w_emb * sim_emb + w_trans * p_trans` in probability space so that the
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

    The gate stack is the public "safe division" repair (the 946 Harmonic
    Fusion public notebook), which reached the strongest public score
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
    per node (`nan` where unscored). It enters
    twice and both matter:

    * as a gate, ``div_prob_min`` -- and this is the only signal here that is
      not geometry or the edge model's link probability, which is why it is
      worth the extra pass. Precision is the
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
            "array scored for *this* prediction npz"
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



def linefit_smooth(coords, edges, window=2, weight=0.74):
    """Move each node toward a line fitted through its own track.

    `pp_public` does this and it is the one part of that module worth its
    runtime here. The point is not that the detector jitters -- smoothing a
    node against its neighbours in *time* was measured and makes the median
    localisation error worse. The point is the node's
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
