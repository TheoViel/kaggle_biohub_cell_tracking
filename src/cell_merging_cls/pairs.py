"""Which pairs of detections to ask about, and which of them the annotation labels.

The unit here is a **pair of detections in one frame**, and the question is
whether they are two cells or one cell found twice. Candidates are every pair
closer than `max_pair_um`; the work in this module is deciding which of those
the ground truth can label, and refusing to guess about the rest.

**Why the labels cannot come from matching alone.** The annotation covers one
lineage per stack -- a median of 10 cells in a frame against ~5 000 detections,
223x more detections than annotated nodes overall. So a detection that matches
no annotated node is almost always a *real, unannotated cell*, not a duplicate.
The rule "unmatched, therefore spurious" would label essentially every cell in
the embryo a duplicate. It is not used here, and it is the single mistake this
module exists to avoid.

What the annotation *can* say, exactly:

* two detections charged to the **same** annotated cell are one cell found
  twice -- a positive;
* two detections charged to **different** annotated cells are two cells -- a
  negative, and after a division these are the sisters, which is the hardest
  and most important negative there is;
* a pair with either member unattributed is **unlabelled**, and is dropped
  rather than assumed.

That leaves a clean, small labelled set inside a large unlabelled pool, which
is the honest shape of this problem.

**Charging a detection to a cell.** Each detection goes to its *nearest*
annotated node within `dup_radius`, so a detection between two annotated cells
is not counted for both. The radius is what makes a positive: two centres of
genuinely distinct cells cannot be much closer than a cell radius, because
cells do not interpenetrate, so a second detection inside `dup_radius` of an
annotated cell is a duplicate of it rather than a neighbour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from params import SCALE

SCALE_ARR = np.asarray(SCALE, float)

# What the pair classifier is ever asked about. 14 um is `edge_pp`'s
# `max_sister_um`: beyond it a pair is not a division candidate either, so a
# merge there could not help the thing this is for.
MAX_PAIR_UM = 14.0
# What counts as "the same annotated cell". Cells are ~8.5 um across, so two
# distinct centres this close would have to overlap; measured duplicate pairs
# sit at median 6.7 um and the annotated cells they belong to are within 5.
DUP_RADIUS_UM = 5.0


def gt_graph(geff_path):
    """`(nodes, parent_of, divides)` for one annotated stack.

    `nodes` is (N, 4) t/z/y/x, `parent_of` gives each node's predecessor or -1,
    and `divides` marks the nodes with two or more children. `parent_of` is what
    `division_cls.dataset.gt_lineage` does not carry and what sister pairs need:
    two nodes are sisters exactly when they share a parent.
    """
    import tracksdata as td

    graph = td.graph.IndexedRXGraph.from_geff(Path(geff_path))
    graph = graph[0] if isinstance(graph, tuple) else graph

    rows = list(graph.node_attrs().iter_rows(named=True))
    index = {int(r["node_id"]): i for i, r in enumerate(rows)}
    nodes = np.array([[r["t"], r["z"], r["y"], r["x"]] for r in rows], np.float32)

    parent_of = np.full(len(rows), -1, np.int64)
    n_out = np.zeros(len(rows), np.int32)
    for e in graph.edge_attrs().iter_rows(named=True):
        s, t = index[int(e["source_id"])], index[int(e["target_id"])]
        n_out[s] += 1
        parent_of[t] = s
    return nodes, parent_of, n_out >= 2


def charge_to_gt(det_zyx_um, det_t, gt_zyx_um, gt_t, radius=DUP_RADIUS_UM):
    """`owner[i]` = index of the annotated node detection `i` belongs to, or -1.

    Nearest-annotated-node within `radius`, computed per frame. Nearest rather
    than "within radius of", so a detection sitting between two annotated cells
    is charged once and cannot manufacture a duplicate out of two neighbours.
    """
    owner = np.full(len(det_zyx_um), -1, np.int64)
    for t in np.unique(gt_t):
        gi = np.flatnonzero(gt_t == t)
        di = np.flatnonzero(det_t == t)
        if len(di) == 0 or len(gi) == 0:
            continue
        dist, which = cKDTree(gt_zyx_um[gi]).query(det_zyx_um[di], k=1)
        hit = dist <= radius
        owner[di[hit]] = gi[which[hit]]
    return owner


def labelled_pairs(coords, geff_path, max_pair_um=MAX_PAIR_UM,
                   dup_radius=DUP_RADIUS_UM, rng=None, neg_per_pos=None,
                   unlabelled_per_pos=20.0):
    """`(row_a, row_b, label, kind)` pairs for one stack.

    `kind` is "dup", "sister", "distinct" or "unlabelled".

    The first three come from the annotation and are exact. The fourth needs its
    own justification, because it is one step away from the mistake this module
    exists to avoid.

    **Unlabelled pairs are used as negatives, and the direction matters.** The
    trap is "this detection matches no annotated node, therefore it is a
    duplicate" -- at 223 detections per annotated node that would call the whole
    embryo spurious. Going the other way is a different claim with a measurable
    error rate: of the ~675k pairs within 9 um, roughly 3.5% are duplicates
    (154 confirmed among the 0.65% of detections the annotation covers,
    extrapolated), so a pair drawn at random from the unlabelled pool is a
    genuine pair of distinct cells about 96.5% of the time.

    Taking them is worth that 3.5% label noise for two reasons. Without them the
    training set is *positive-heavy* -- two annotated cells within 14 um of each
    other are rarer than duplicates, so the labelled set runs about 2:1 the
    wrong way against an inference population that is ~28:1 negative, and a
    model fitted to that prior over-fires. And the labelled negatives are all
    drawn from one lineage, so they show the model a single neighbourhood; the
    unlabelled pool is every cell density in the embryo, which is what it will
    actually be asked about.

    Set `unlabelled_per_pos=0` to train on the exact labels alone.
    """
    coords = np.asarray(coords)
    det_um = coords[:, 1:].astype(float) * SCALE_ARR
    det_t = coords[:, 0].astype(int)

    nodes, parent_of, _ = gt_graph(geff_path)
    gt_um = nodes[:, 1:].astype(float) * SCALE_ARR
    gt_t = nodes[:, 0].astype(int)

    owner = charge_to_gt(det_um, det_t, gt_um, gt_t, dup_radius)

    out, unlabelled = [], []
    for t in np.unique(det_t):
        di = np.flatnonzero(det_t == t)
        if len(di) < 2:
            continue
        pairs = cKDTree(det_um[di]).query_pairs(max_pair_um, output_type="ndarray")
        if not len(pairs):
            continue
        # Classified with array ops rather than a Python loop: there are ~160k
        # pairs per stack once the unlabelled ones are in scope, and 199 stacks
        # of that is minutes of interpreter time before the first training step.
        a, b = di[pairs[:, 0]], di[pairs[:, 1]]
        oa, ob = owner[a], owner[b]
        known = (oa >= 0) & (ob >= 0)
        dup = known & (oa == ob)
        pa = np.where(known, parent_of[np.clip(oa, 0, None)], -1)
        pb = np.where(known, parent_of[np.clip(ob, 0, None)], -1)
        sister = known & ~dup & (pa >= 0) & (pa == pb)
        distinct = known & ~dup & ~sister

        out.extend((int(x), int(y), 1, "dup") for x, y in zip(a[dup], b[dup]))
        out.extend((int(x), int(y), 0, "sister") for x, y in zip(a[sister], b[sister]))
        out.extend((int(x), int(y), 0, "distinct")
                   for x, y in zip(a[distinct], b[distinct]))
        if unlabelled_per_pos:
            un = ~known
            unlabelled.extend((int(x), int(y), 0, "unlabelled")
                              for x, y in zip(a[un], b[un]))

    rng = np.random.default_rng() if rng is None else rng
    if neg_per_pos and out:
        out = _cap_negatives(out, float(neg_per_pos), rng)
    if unlabelled_per_pos and unlabelled:
        n_pos = sum(1 for p in out if p[2] == 1)
        keep = int(round(float(unlabelled_per_pos) * max(n_pos, 1)))
        if len(unlabelled) > keep:
            pick = rng.choice(len(unlabelled), size=keep, replace=False)
            unlabelled = [unlabelled[k] for k in pick]
        out = out + unlabelled
    return out


def _cap_negatives(pairs, neg_per_pos, rng):
    """Thin ordinary negatives to `neg_per_pos` per positive; keep every sister.

    Sisters are never dropped. They are ~0.1% of the negatives and 100% of the
    damage a false merge can do, so subsampling them to keep a ratio tidy would
    be optimising the loss against the thing that matters.
    """
    rng = np.random.default_rng() if rng is None else rng
    pos = [p for p in pairs if p[2] == 1]
    sis = [p for p in pairs if p[2] == 0 and p[3] == "sister"]
    oth = [p for p in pairs if p[2] == 0 and p[3] == "distinct"]
    keep = int(round(neg_per_pos * max(len(pos), 1)))
    if len(oth) > keep:
        oth = [oth[k] for k in rng.choice(len(oth), size=keep, replace=False)]
    return pos + sis + oth


def division_probe(coords, geff_path, dup_radius=DUP_RADIUS_UM,
                   max_pair_um=MAX_PAIR_UM):
    """The evaluation set the annotated divisions define, with no ambiguity.

    A division names one cell that is *one* cell at frame `t` and *two* cells at
    `t+1`, at a separation where the two cases are otherwise indistinguishable:

    * every pair of detections on the **parent** is a duplicate (label 1);
    * the pair of detections on the **two daughters** is two cells (label 0).

    Nothing here depends on a radius choice being generous or a negative pool
    being representative -- the annotation states both answers outright. It is
    also exactly the regime the model exists for, so it is the honest probe to
    watch while training, and the sister half is the one that must not fail.
    """
    coords = np.asarray(coords)
    det_um = coords[:, 1:].astype(float) * SCALE_ARR
    det_t = coords[:, 0].astype(int)

    nodes, parent_of, divides = gt_graph(geff_path)
    gt_um = nodes[:, 1:].astype(float) * SCALE_ARR
    gt_t = nodes[:, 0].astype(int)
    owner = charge_to_gt(det_um, det_t, gt_um, gt_t, dup_radius)

    by_owner: dict[int, list[int]] = {}
    for row, o in enumerate(owner):
        if o >= 0:
            by_owner.setdefault(int(o), []).append(row)

    out = []
    for p in np.flatnonzero(divides):
        # the parent is one cell: every pair of detections on it is a duplicate
        rows = by_owner.get(int(p), [])
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if np.linalg.norm(det_um[rows[i]] - det_um[rows[j]]) <= max_pair_um:
                    out.append((rows[i], rows[j], 1, "parent"))
        # the daughters are two cells: their detections must not be merged
        kids = np.flatnonzero(parent_of == p)
        for a in range(len(kids)):
            for b in range(a + 1, len(kids)):
                for ra in by_owner.get(int(kids[a]), []):
                    for rb in by_owner.get(int(kids[b]), []):
                        if np.linalg.norm(det_um[ra] - det_um[rb]) <= max_pair_um:
                            out.append((ra, rb, 0, "sister"))
    return out
