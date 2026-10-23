"""Gate the ILP's forks as division proposals. MEASURED NEGATIVE -- off by default.

Keep this for the result, not the stage. `propose_divisions` and the ILP attack
the division term from opposite ends, and the ILP wins decisively on recall:
on the 21 fold-0 stacks holding all 26 GT divisions it finds 11 against the
proposer's 4. It has the structural advantage, because it decides the whole
lineage and the fork together under flow conservation and can take a fork whose
second edge the Hungarian never offered, where `propose_divisions` can only
bolt a second child onto an assignment already fixed without it.

It does not know what a division looks like. A fork is payable whenever two
edge rewards beat `ilp_division`, so it emits thousands, and the extra recall
never pays for the precision. Full fold 0, 71 stacks, against the tuned chain:

    arm                          cv       adj     div_jac   tp   fp   fn
    propose_divisions         0.9010    0.8885    0.1250     4    6   22
    ILP forks + caps          0.8968    0.8885    0.0833     6   46   20
    ILP forks, ungated        0.8948    0.8873    0.0758    10  106   16
    ILP forks + PP divisions  0.8808    0.8770    0.0385    18  442    8

Division precision is 40% for the proposer and 4-12% for every ILP arm. Since
``division_jaccard = tp / (tp + fp + fn)`` and ``fn`` is bounded by 26, the FPs
dominate long before the recall helps: the best ILP arm trades 2 extra TP for
40 extra FP.

The 21-stack subset said +0.0013 for the ungated arm and the full set says
-0.0061, and the gap is not noise -- it is the sampling. Those 50 excluded
stacks contain no GT division, so they can contribute division FP and never
division TP; they added 65 FP and 0 TP. Selecting stacks *by the thing being
measured* biases the measurement, and for divisions the bias is large enough to
flip the sign. Measure division changes on the whole fold.

The stage itself is a faithful port of the proposer's gates applied as a filter
-- probability, parent/sister distance, mutual nearest neighbour, divergence at
t+2, sister symmetry, and the frame/global caps -- deleting the second child
wherever a gate fails. Of those, only ``require_divergence`` has any effect on
ILP forks: `min_prob`, `symmetry_tau` and `require_mutual_nn` were bit-identical
when added to an ungated run, `min_prob=0.25` being inert because
``ilp_threshold=0.4`` already exceeds it.

Two differences from the proposer that the direction of the operation forces:

* **Which child is the "second" one.** `propose_divisions` knows, because it
  just added it. Here both children arrive on equal footing, so the
  higher-probability edge is treated as the committed child and every other
  child is a candidate second daughter. That is the same convention the
  divergence test needs to pick grandchildren.
* **`require_mutual_nn` is checked, not used to generate.** The proposer queries
  the KD-tree for the free node nearest the first child; here the candidate is
  given, so the test becomes "is this candidate the nearest node to its sister
  among the frame's other available nodes", which is the same predicate.

Out-degree above 2 is truncated to the best two children before gating: a
three-way split is not a cell division, and the ILP does emit them at low
`ilp_division`.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .edge_pp import pos_um, rows_by_frame

DEFAULTS = dict(
    enabled=1,
    # Same meanings, and same tuned values, as `propose_divisions`.
    min_prob=0.25,
    max_parent_um=11.0,
    max_existing_child_um=10.0,
    max_sister_um=14.0,
    symmetry_tau=0.6,
    diverge_um=4.0,
    require_divergence=1,
    require_mutual_nn=1,
    frame_frac_cap=0.0076,
    global_frac_cap=0.00375,
    # Truncate a >2-way split rather than rejecting it outright.
    max_children=2,
)


def inject_forks(coords, edges, forks, cand=None):
    """Add the ILP's second-child links to a graph built by another matcher.

    The ILP's lineage scores 0.0177 *worse* on ``adj_edge_jaccard`` than
    assign+recover, so adopting it wholesale pays for its divisions with edges.
    This takes only the forks: every ``(parent, child)`` the solver forked on is
    appended to the existing graph where the child is still free and the parent
    already has a committed child, leaving the tuned backbone untouched.

    Node indices survive the chain up to ``prune_short`` -- ``recover_nodes``
    appends, ``link_through_isolated`` and ``propose_divisions`` only add edges
    -- so ``forks`` may be collected at ILP time and spent here.
    """
    coords = np.asarray(coords)
    edges = np.asarray(edges, float).reshape(-1, 4)
    if not len(forks):
        return edges
    p_um = pos_um(coords)
    n = len(coords)
    src = edges[:, 0].astype(int) if len(edges) else np.empty(0, int)
    tgt = edges[:, 1].astype(int) if len(edges) else np.empty(0, int)
    has_parent = np.zeros(n, bool)
    has_parent[tgt] = True
    n_children = np.bincount(src, minlength=n)
    committed = set(zip(src.tolist(), tgt.tolist()))

    prob_of = {}
    if cand is not None:
        cand = np.asarray(cand, float).reshape(-1, 5)
        for s, t, pr in zip(cand[:, 0].astype(int), cand[:, 1].astype(int), cand[:, 2]):
            prob_of[(int(s), int(t))] = float(pr)

    add = []
    for p, q in forks:
        p, q = int(p), int(q)
        if p >= n or q >= n or has_parent[q] or n_children[p] < 1 or (p, q) in committed:
            continue
        add.append([p, q, prob_of.get((p, q), 0.5),
                    float(np.linalg.norm(p_um[p] - p_um[q]))])
        has_parent[q] = True
    return np.vstack([edges, np.asarray(add, float)]) if add else edges


def filter_forks(coords, edges, cfg=None, return_stats=False):
    """Delete the second child of every fork that fails the division gates.

    ``edges`` is (M, 4) ``(src, tgt, prob, dist_um)``. Returns the same array
    with the rejected second-child rows removed, so every surviving fork is one
    the gate stack accepts.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    coords = np.asarray(coords)
    edges = np.asarray(edges, float).reshape(-1, 4)
    stats = {k: 0 for k in (
        "forks_in", "forks_out", "prob_rejected", "distance_rejected",
        "mutual_nn_rejected", "divergence_rejected", "symmetry_rejected",
        "capped", "truncated",
    )}
    if not cfg["enabled"] or len(edges) == 0 or len(coords) == 0:
        return (edges, stats) if return_stats else edges

    p_um = pos_um(coords)
    frame_of = coords[:, 0].astype(int)
    src = edges[:, 0].astype(int)
    tgt = edges[:, 1].astype(int)

    # Children per parent, best-first: the top edge is the committed child and
    # the rest are candidate second daughters.
    children: dict[int, list[int]] = {}
    for k in np.argsort(-edges[:, 2], kind="stable"):
        children.setdefault(int(src[k]), []).append(int(k))
    forks = {p: ks for p, ks in children.items() if len(ks) >= 2}
    stats["forks_in"] = len(forks)
    if not forks:
        return (edges, stats) if return_stats else edges

    # The committed child of any node, for the t+2 divergence test. A forked
    # node contributes its best child, which is the daughter the grandchild
    # test should follow.
    best_child = np.full(len(coords), -1, np.int64)
    for p, ks in children.items():
        best_child[p] = tgt[ks[0]]

    # Nearest-neighbour structure per frame, for `require_mutual_nn`. The
    # proposer queries a tree of *free* nodes, because at proposal time the
    # second daughter has no parent yet. Here it already has one -- the fork
    # under test -- so the candidate set is "free, or free but for a fork",
    # i.e. every node that is not some parent's committed first child. Building
    # this over all frame nodes instead is far stricter than the proposer and
    # rejected 9 of the ILP's 11 true divisions.
    committed_child = np.zeros(len(coords), bool)
    for p, ks in children.items():
        committed_child[tgt[ks[0]]] = True
    by_frame = rows_by_frame(coords)
    trees = {}
    for t, rows in by_frame.items():
        rows = np.asarray([int(r) for r in rows if not committed_child[int(r)]], np.int64)
        if len(rows):
            trees[t] = (rows, cKDTree(p_um[rows]))

    drop: set[int] = set()
    proposals = []           # (order_key, frame, edge_idx)
    for parent, ks in forks.items():
        keep_k = ks[0]
        child = int(tgt[keep_k])
        child_dist = float(np.linalg.norm(p_um[parent] - p_um[child]))
        t = int(frame_of[parent])
        extra = ks[1:]
        if len(extra) > cfg["max_children"] - 1:
            # Rank by probability is already the sort order, so the tail is the
            # weakest of the split.
            stats["truncated"] += len(extra) - (cfg["max_children"] - 1)
            drop.update(extra[cfg["max_children"] - 1:])
            extra = extra[:cfg["max_children"] - 1]

        for k in extra:
            q = int(tgt[k])
            prob = float(edges[k, 2])
            if child_dist > cfg["max_existing_child_um"]:
                stats["distance_rejected"] += 1
                drop.add(int(k))
                continue
            if prob < cfg["min_prob"]:
                stats["prob_rejected"] += 1
                drop.add(int(k))
                continue
            parent_dist = float(np.linalg.norm(p_um[parent] - p_um[q]))
            sister_dist = float(np.linalg.norm(p_um[child] - p_um[q]))
            if parent_dist > cfg["max_parent_um"] or sister_dist > cfg["max_sister_um"]:
                stats["distance_rejected"] += 1
                drop.add(int(k))
                continue
            if cfg["require_mutual_nn"]:
                rows, tree = trees.get(int(frame_of[q]), (None, None))
                if tree is None:
                    drop.add(int(k))
                    continue
                # The sister is a committed child, so it is not in the tree and
                # k=1 is the proposer's query exactly.
                _, idx = tree.query(p_um[child])
                if int(rows[int(idx)]) != q:
                    stats["mutual_nn_rejected"] += 1
                    drop.add(int(k))
                    continue
            if cfg["require_divergence"]:
                g1, g2 = int(best_child[child]), int(best_child[q])
                if (g1 < 0 or g2 < 0 or frame_of[g1] != t + 2 or frame_of[g2] != t + 2
                        or float(np.linalg.norm(p_um[g1] - p_um[g2])) - sister_dist
                        < cfg["diverge_um"]):
                    stats["divergence_rejected"] += 1
                    drop.add(int(k))
                    continue
            if cfg["symmetry_tau"] > 0:
                denom = max((child_dist + parent_dist) / 2.0, 1e-6)
                if abs(child_dist - parent_dist) / denom > cfg["symmetry_tau"]:
                    stats["symmetry_rejected"] += 1
                    drop.add(int(k))
                    continue
            proposals.append((parent_dist + 0.15 * sister_dist, t, int(k)))

    # The caps, applied exactly as the proposer applies them: tightest pairs
    # first, `frame_frac_cap` of the frame's forked parents and
    # `global_frac_cap` of all nodes.
    global_cap = max(1, int(round(len(coords) * cfg["global_frac_cap"])))
    per_frame_forks: dict[int, int] = {}
    for parent in forks:
        per_frame_forks[int(frame_of[parent])] = per_frame_forks.get(int(frame_of[parent]), 0) + 1
    proposals.sort(key=lambda item: item[0])
    kept_total = 0
    kept_frame: dict[int, int] = {}
    for _, t, k in proposals:
        frame_cap = max(1, int(round(per_frame_forks.get(t, 0) * cfg["frame_frac_cap"])))
        if kept_total >= global_cap or kept_frame.get(t, 0) >= frame_cap:
            stats["capped"] += 1
            drop.add(k)
            continue
        kept_total += 1
        kept_frame[t] = kept_frame.get(t, 0) + 1

    stats["forks_out"] = kept_total
    if drop:
        edges = edges[np.setdiff1d(np.arange(len(edges)), np.fromiter(drop, np.int64))]
    return (edges, stats) if return_stats else edges
