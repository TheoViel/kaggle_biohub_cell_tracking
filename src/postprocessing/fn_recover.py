"""Put back the detections the segmenter missed, from the tracks' own motion.

    cd src/postprocessing
    python fn_recover.py --nodes ../logs/2026-09-13/1/preds_r6_0.40 --arms

Paths are relative to ``src/``: ``common`` chdirs there, as the other scripts
here do.

A missed detection is expensive twice over. The node is worth nothing by itself
-- the metric scores edges -- but the cell it belonged to loses the link into it
*and* the link out of it, so one missing node costs two ground-truth edges, and
a run of `L` missing nodes costs `L + 1`. Nothing downstream repairs it: the
linker cannot connect through a node that does not exist.

The premise is the one thing a tracker knows that a detector does not: **a cell
detected at `t` does not stop existing at `t + 1`.** It divides, it leaves the
imaged volume, or it is still there and was missed. So every track that ends
mid-video is either an exit or a false negative, and the two are separable --
an exit happens at the boundary, a false negative happens anywhere.

Where the cell is
-----------------
Motion decomposes into what the whole field does and what the cell does within
it. The field's share is large here: the global translation between adjacent
frames averages 0.5 to 2.9 um across these stacks, against a 7 um node-matching
radius, and one stack slides steadily in z. `drift_table` takes it out by
matching each child to its nearest parent and taking the median offset, which is
insensitive to the cells that entered, left, or genuinely moved.

What is left is the cell's own motion, and measuring it against the 28 places in
these stacks where a tracked cell's next node is a known miss says something
worth recording -- the residual is not worth extrapolating:

    estimator                     median err   within 7 um
    global drift only                  4.96 um       85.7%
    drift + own velocity (1 step)      6.13 um       71.4%
    drift + own velocity (3 steps)     5.29 um       78.6%
    local flow, k = 16 neighbours      4.23 um       78.6%
    two-sided interpolation            3.17 um       96.0%

Per-cell velocity makes the estimate *worse*. Over one frame a cell moves less
than the detector's own jitter, so a velocity estimated from one or three steps
is mostly noise, and adding it to a good global estimate only adds variance. A
local flow field -- the median displacement of the k nearest tracked neighbours,
the cheap stand-in for the vector fields Ultrack computes -- has a better median
but a worse tail, and on 28 points that is not a difference worth having.

The row that matters is the last one. Where the cell is seen again on the far
side of the gap, sharing the leftover motion evenly across the missing frames
puts the node inside the matching radius 96% of the time. So a gap with an
anchor at both ends is a far better bet than an open-ended extrapolation, and
the two are proposed by separate stages here.

Which gaps to fill
------------------
The stages over-propose on purpose and let one test decide. `propose_bridges`
(delegated to `recover.find_gaps`) pairs a track that ends at `t` with one that
starts at `t + L + 1`; `propose_open` extrapolates a track that ends and never
resumes, in either time direction. Together they offer far more nodes than can
be right -- 5 to 10% of all detections fail to continue into the next frame,
most of them spurious detections blinking out.

The test is the assignment itself, which is the third thing the pointers name:
*a good candidate lowers the total cost of the matching.* Rather than gate each
proposal on its own distance, the candidates for a frame pair are added to that
frame's assignment problem -- the unexplained detections at `t` against the
candidates at `t + 1`, cost being drift-corrected distance, with not linking
charged at `lambda_um` -- and only the ones the solver links are kept. The
sources include the candidates accepted at `t - 1`, without which no bridge
longer than one frame can survive: its second node's only possible parent is its
first, which does not exist until this loop puts it there.

`corroborate` is the other filter: a prediction says where a cell should be, and
a different segmentation run of the same images says whether anything is there.

What each of them is worth
--------------------------
Five fold-0 stacks with node recall below 1 (`44b6_e35b117d`, `2f31fc2f`,
`5f15d135`, `e57ff5c6`, `996155de`), nodes from `logs/2026-09-13/1/preds_r6_0.40`,
graph rebuilt by drift-corrected assignment after every change. 64 of 1210
annotated nodes are missed to begin with.

    arm                          nodes added   found   node recall   FN edges    adjJ
    baseline                               0       0        0.9438        168  0.7457
    bridges                            11220      31        0.9727        106  0.7892
    bridges, cost test                 11217      31        0.9727        106  0.7892
    bridges, corroborated               3084      13        0.9569        139  0.7706
    bridges + open                     20586      38        0.9788        115  0.7600
    open only                           8837      21        0.9614        136  0.7563

Only the two-sided stage pays, and it pays well: a 37% cut in false-negative
edges for +0.0435. Everything else trades in the wrong direction, and the reason
is the same in each case. `adj_edge_jaccard` charges 0.1 x the node excess, and
these node sets are *already* 41% over the dataset's estimated cell count, so a
speculative node is never free -- it is paid for whether or not a cell was there,
and if it lands near an annotated cell it also competes for that cell's match.

Which makes node recall the wrong thing to maximise, and this is the result worth
carrying away: `bridges + open` reaches the best node recall in the table, 0.9788
against 0.9727, and scores 0.029 *worse*. Six more cells found, 9000 more nodes
and 29 more false-positive edges to find them.

End to end, through the real pipeline
------------------------------------
The table above scores the repaired node set with a drift-corrected assignment,
which is the graph this module builds and not the one the project ships. To
check that the gain survives the real thing, both strategies were run all the
way through: `inference_contrastive.py` to embed the appended nodes (they are
new rows, so the whole row-aligned embedding array is rewritten), then
`main_edge_predict.py` with the fold-0 checkpoint of `edge_run/2026-09-13/0`
-- the run trained on exactly these nodes -- then `main_edge_postprocess.py` at
its defaults. Same five stacks throughout.

    arm         stage         nodes      NR    tp   fp   fn       J    adjJ      cv
    baseline    edge model   156134  0.9438  1022  109  165  0.7886  0.7588  0.7588
    baseline    + post-proc  149467  0.9351  1054   79  133  0.8325  0.8049  0.8382
    max adjJ    edge model   167354  0.9727  1081  103  106  0.8380  0.7990  0.7990
    max adjJ    + post-proc  159906  0.9680  1096   81   91  0.8644  0.8290  0.8623
    max recall  edge model   176720  0.9788  1074  128  113  0.8167  0.7730  0.7730
    max recall  + post-proc  169614  0.9725  1086  115  101  0.8341  0.7936  0.8269

`bridges` is worth +0.0241 adjusted Jaccard on *these five stacks* end to end
(0.8049 -> 0.8290), cutting false-negative edges 133 -> 91. But the five were
picked for having the worst node recall in the fold, and that gain does not
survive on the rest of it. See below; do not quote the 0.0241.

Validated on all 71 fold-0 stacks
---------------------------------
Same pipeline, every `44b6` stack, 3,059,598 nodes with 128,425 recovered
(4.2%, mean 1809 a stack). The baseline reproduces `edge_run/2026-09-13/0`'s
published 44b6 numbers exactly, 0.8512 raw and 0.8807 post-processed, so the
harness is the pipeline and not a re-implementation of it.

    stage         arm         NR       tp    fp    fn        J     adjJ
    edge model    baseline  0.9886  18354  1430  1472   0.8635   0.8512
    edge model    bridges   0.9932  18623  1372  1203   0.8785   0.8620
    + post-proc   baseline  0.9841  18701  1156  1125   0.8913   0.8807
    + post-proc   bridges   0.9884  18807  1199  1019   0.8945   0.8803

Before post-processing this is a real result: **+0.0108, paired t = +3.26** over
the 71 stacks, winning on 56% of them, with false negatives down 18% and false
positives down as well. After post-processing it is **-0.0004, t = -0.55**,
winning on 34%. False negatives keep falling (1125 -> 1019) but 43 false
positives arrive with them.

The reason is that `main_edge_postprocess.py` runs `assign -> recover -> prune
-> linefit`, and its `recover` stage fills the same gaps from the same motion,
one stage later and on the edge graph rather than the node set. The two are
substitutes, and `--no_recover` measures it:

                       pp with recover   pp --no_recover   what pp's recover adds
    baseline                    0.8807            0.8727                  +0.0080
    bridges upstream            0.8803            0.8783                  +0.0020

Read the other way: recovering nodes first is worth +0.0055 when the
post-processing chain is not already doing it, and -0.0004 when it is. It
captures about 70% of what the existing stage captures, and nothing on top.

Retuning `recover_*` does not rescue it
---------------------------------------
The obvious objection to the table above is that `edge_pp`'s `recover_*` knobs
were tuned on a different run (`edge_run/2026-09-11/0/preds_0.45`), assuming
nothing upstream had filled the gaps -- so the 43 extra false positives might be
a mistuning artefact rather than a real redundancy. `tune_edge_pp.py
--knobs recover_max_gap,recover_max_residual,recover_min_dup,min_track --passes 2`
was therefore run over *both* arms, since tuning only the repaired one would
compare a fresh config against a stale one and manufacture the gain.

    arm         shipped DEFAULTS      retuned      what retuning adds
                  cv      adjJ      cv      adjJ         cv
    baseline    0.8948  0.8807   0.9012  0.8838       +0.0064
    bridges     0.8949  0.8803   0.8988  0.8832       +0.0039

Tuned against tuned the baseline is still ahead, by +0.0024 cv and +0.0006 adjJ,
so the redundancy is real and retuning does not rescue the stage.

The knob that moved differently is exactly the one the substitution predicts.
Both arms wanted `recover_max_residual` 4.0 -> 5.0 and `recover_min_dup`
5.0 -> 3.0, identically. Only `recover_max_gap` split: the baseline reaches
*further*, 3 -> 4, and the arm with nodes already recovered pulls *back*, 3 -> 2.
The two stages are doing one job, and whichever runs first leaves the other less
to do.

One incidental result worth acting on separately: retuning those three knobs is
worth **+0.0064 cv on the baseline** (0.8948 -> 0.9012), which has nothing to do
with false negatives -- the shipped defaults are simply stale for this
checkpoint. Tuned and scored on the same 71 stacks of one embryo, so confirm it
on `6bba` before believing the size of it.

The repaired arm also loses on divisions specifically (div jaccard 0.1569
against 0.1739), which is not something this module reasons about: a recovered
node next to a real one is a second candidate daughter.

So this is not a stage to add to the shipped chain. Two things could still
change that:

  * **Retrain the edge model on the repaired node set.** Every number above was
    produced with the fold-0 checkpoint of `edge_run/2026-09-13/0`, which was
    trained on the unrepaired nodes -- the recovered ones are out of
    distribution for it, and the +0.0108 at the model stage is what survives
    that handicap. This is the only untested explanation left for the gap
    between the model stage and the end.
  * **Node sets with worse recall than this one.** The five worst stacks in the
    fold do gain end to end where the fold as a whole does not, so the value is
    there when there is more to recover -- on a detector at node recall 0.94
    rather than 0.99.

Three more things that did not work, so they need not be tried again:

  * **Per-cell velocity.** See the table above -- it is worse than drift alone.
  * **Corroboration by *other runs*.** A different run detects 42 of the 64
    missed cells -- the same detector at another NMS radius recovers only 5, so
    the diversity is in the training rather than the suppression -- but it does
    not help. Requiring a second run to confirm a prediction cuts the inserted
    nodes 3.6x and the recovered cells 31 -> 13 (adjJ 0.7706), and requiring two
    runs to agree (`--min_votes 2`) cuts them to 2. The corroborating detection
    and the motion prediction are each within ~5 um of the cell and so up to
    10 um from each other, wider than the 7 um that would make them one cell.
    Taking the union of the node sets outright is worse still: node recall 0.9809
    for +0.006, with the node excess going 0.41 -> 0.62.

    Corroborating against the detector's *own sub-threshold pool* is a different
    and better proposition, and `--extra ../logs/<run>/preds_r6_0.30` is how to
    ask for it: same network, same NMS radius, just the peaks the 0.40 cutoff
    discarded, so a candidate sits where the detector actually saw something
    rather than where another training run put its own centroid. It recovers 20
    cells to the cross-run 13 for the same number of insertions, and is 2.7x more
    node-efficient than plain bridges (4063 nodes for 20 cells against 11220 for
    31). It still scores lower -- 0.7774 against 0.7892 -- because on these
    stacks the eleven extra cells are worth more than the seven thousand extra
    nodes cost. That ordering is what would flip first on a node set less
    over-detected than this one, which is the arm to re-run there.
  * **Iterating.** A second and third pass of the bridge stage add 284 nodes and
    no cells: the gaps a first pass leaves are the ones with no anchor, and
    another pass does not give them one.

The cost test is a no-op at these settings -- identical score, three fewer nodes
-- which is itself worth knowing. It is not that the test is weak; it is that
`find_gaps`' speed gate already rejects everything the assignment would have
rejected. It earns its place only against a looser proposer, which is what
`--open_steps` and `--stages open` give it.

`--arms` scores every stage and rule; `--extra` supplies the other runs' node
folders for the corroboration arms.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: F401,E402  (chdir to src/ and set the path)

import pandas as pd  # noqa: E402
import tracksdata as td  # noqa: E402

from params import SCALE, TRAIN_DIR  # noqa: E402
from postprocessing.fast_score import (  # noqa: E402
    GroundTruth,
    match_nodes,
    roll_up,
    row_from_counts,
)

warnings.filterwarnings("ignore")

SCALE_ARR = np.asarray(SCALE, dtype=np.float64)

DEFAULTS = dict(
    link_um=8.0,        # gate on the drift-corrected step, for the base graph
    lambda_um=6.0,      # what not linking costs, in the same units as distance
    max_gap=5,          # longest run of missing frames a bridge may fill
    max_residual=5.0,   # um/frame of own motion a bridge may ask for
    open_steps=2,       # frames an open-ended extrapolation may run on
    min_history=2,      # frames of track behind an end before it may be extended
    min_dup_um=4.0,     # do not insert next to a detection that already exists
    border_um=6.0,      # a prediction this close to the volume edge is an exit
    margin_um=0.0,      # extra improvement a candidate must show to be kept
    snap_um=5.0,        # how far a prediction may reach for a second run's detection
    min_votes=1,        # how many *distinct* other runs must corroborate it
)


# --------------------------------------------------------------------------
# frames, drift, and the base graph
# --------------------------------------------------------------------------

def rows_by_t(coords: np.ndarray) -> dict[int, np.ndarray]:
    """``frame -> rows of coords``, every frame in range present (possibly empty)."""
    t = coords[:, 0].astype(int)
    order = np.argsort(t, kind="stable")
    ts = t[order]
    out, lo = {}, 0
    for tt in range(int(ts[0]), int(ts[-1]) + 1):
        hi = int(np.searchsorted(ts, tt + 1))
        out[tt] = order[lo:hi]
        lo = hi
    return out


def drift_table(coords: np.ndarray, n_iter: int = 4, tol: float = 2.0) -> dict[int, np.ndarray]:
    """``D[t]``: the whole-field translation in um carrying frame t onto t + 1.

    Centroid difference to seed, then rounds of "match each child to its nearest
    parent, take the median offset" -- the centroid moves with whatever entered
    or left the field, the median does not. A last pass averages the offsets
    that agree with the median, which recovers the sub-voxel part that a median
    over a 1.625 um z-grid cannot express.

    A frame the estimator cannot speak for inherits the previous answer, which
    beats the zero it would otherwise return.
    """
    pos = coords[:, 1:].astype(np.float64) * SCALE_ARR
    rows = rows_by_t(coords)
    out, last = {}, np.zeros(3)
    for t in sorted(rows)[:-1]:
        a, b = pos[rows[t]], pos[rows.get(t + 1, np.empty(0, int))]
        if not len(a) or not len(b):
            out[t] = last
            continue
        d = b.mean(0) - a.mean(0)
        for _ in range(n_iter):
            _, nn = cKDTree(a).query(b - d, k=1)
            d = np.median(b - a[nn], axis=0)
        _, nn = cKDTree(a).query(b - d, k=1)
        off = b - a[nn]
        ok = np.linalg.norm(off - d, axis=1) <= tol
        if ok.sum() >= 5:
            d = off[ok].mean(0)
        out[t] = last = d
    return out


def solve_pair(src_um, tgt_um, lambda_um):
    """Assignment for one frame pair, with not-linking priced at ``lambda_um``.

    Returns ``(cost, assignment)``: the total, and ``src index -> tgt index`` for
    the pairs that were worth linking. Charging a fixed price for leaving a
    source unlinked is what makes the total comparable between problems with
    different numbers of targets -- which is the whole point, since that is the
    comparison a candidate node has to win.
    """
    if not len(src_um) or not len(tgt_um):
        return lambda_um * len(src_um), {}
    d = np.linalg.norm(src_um[:, None, :] - tgt_um[None, :, :], axis=-1)
    # A source may always decline: a dummy column per source, priced at lambda.
    dummy = np.full((len(src_um), len(src_um)), np.inf)
    np.fill_diagonal(dummy, lambda_um)
    work = np.concatenate([d, dummy], axis=1)
    work = np.where(np.isfinite(work), work, 1e9)
    r, c = linear_sum_assignment(work)
    real = c < len(tgt_um)
    return float(work[r, c].sum()), {int(i): int(j) for i, j in zip(r[real], c[real])}


def base_graph(coords, drift, link_um):
    """Pairwise assignment on drift-corrected positions -- the graph to repair."""
    pos = coords[:, 1:].astype(np.float64) * SCALE_ARR
    rows = rows_by_t(coords)
    out = []
    for t in sorted(rows)[:-1]:
        a, b = rows[t], rows.get(t + 1, np.empty(0, int))
        if not len(a) or not len(b):
            continue
        cost = np.linalg.norm(
            (pos[a] + drift.get(t, 0.0))[:, None, :] - pos[b][None, :, :], axis=-1)
        gate = cost > link_um
        r, c = linear_sum_assignment(np.where(gate, 1e6, cost))
        keep = ~gate[r, c]
        out.append(np.stack([a[r[keep]], b[c[keep]]], 1))
    return np.concatenate(out) if out else np.zeros((0, 2), np.int64)


# --------------------------------------------------------------------------
# proposing
# --------------------------------------------------------------------------

def _ends_and_starts(coords, links):
    has_child = np.zeros(len(coords), bool)
    has_parent = np.zeros(len(coords), bool)
    has_child[links[:, 0]] = True
    has_parent[links[:, 1]] = True
    return has_child, has_parent


def _history(links, n_nodes):
    """``row -> how many frames of track sit behind it``, capped at a small n."""
    parent = np.full(n_nodes, -1, np.int64)
    parent[links[:, 1]] = links[:, 0]
    depth = np.zeros(n_nodes, np.int64)
    # Rows are in frame order within a chain, but not globally, so walk.
    for r in range(n_nodes):
        d, cur = 0, r
        while d < 8:
            cur = parent[cur]
            if cur < 0:
                break
            d += 1
        depth[r] = d
    return depth


def propose_bridges(coords, links, drift, cfg):
    """Candidates between a track that ends at ``t`` and one that starts later.

    Delegated to `postprocessing.recover.find_gaps`, which already does exactly
    this and does it better than the version that used to live here: it requires
    a track on *both* sides rather than only behind the end, and it drops a whole
    bridge when any of its nodes would land on an existing detection rather than
    dropping that node and keeping a broken chain. Tuned side by side on these
    five stacks the two differ by 0.006 adjusted Jaccard in its favour, at 2000
    fewer inserted nodes for the same 31 recovered cells, so there is no reason
    for a second implementation.

    What is kept here is what `recover` does not offer: the proposals, rather
    than the finished node array, so `accept` can put them through the cost test
    and the corroboration check.
    """
    from postprocessing.recover import find_gaps

    pairs, positions = find_gaps(
        np.asarray(coords, np.float64), links,
        drift=drift, max_gap=int(cfg["max_gap"]),
        max_residual=cfg["max_residual"], min_dup_dist=cfg["min_dup_um"],
        require_history=cfg["min_history"] >= 1,
    )
    out = []
    for (a, _b), pts in zip(pairs, positions):
        t0 = int(coords[a, 0])
        for k, pt in enumerate(pts, start=1):
            out.append((t0 + k, pt, int(a)))
    return out


def propose_open(coords, links, drift, cfg):
    """Candidates for a track that ends and never resumes -- the open-ended case.

    The bridge stage needs an anchor on the far side; a cell the detector loses
    and never finds again has none, and eleven of the sixty-three missed nodes in
    the sample used to develop this are exactly that -- a ground-truth track
    whose *first* frames are missed, or whose last are.

    So the track is carried forward on the field's motion alone, for at most
    ``open_steps`` frames, and only where the prediction lands inside the imaged
    volume. A prediction near the boundary is not a false negative: that is the
    cell leaving, which is the other thing a disappearance can mean.

    Run on time-reversed coordinates this proposes the mirror case, a track that
    begins mid-video with nothing before it.
    """
    pos = coords[:, 1:].astype(np.float64) * SCALE_ARR
    rows = rows_by_t(coords)
    has_child, _ = _ends_and_starts(coords, links)
    depth = _history(links, len(coords))
    lo = pos.min(0) + cfg["border_um"]
    hi = pos.max(0) - cfg["border_um"]
    last_t = max(rows)
    out = []

    for t in sorted(rows)[:-1]:
        e = rows[t][~has_child[rows[t]]]
        e = e[depth[e] >= cfg["min_history"]]
        if not len(e):
            continue
        p = pos[e]
        for k in range(1, int(cfg["open_steps"]) + 1):
            if t + k > last_t:
                break
            p = p + drift.get(t + k - 1, np.zeros(3))
            inside = np.all((p >= lo) & (p <= hi), axis=1)
            if not inside.any():
                break
            for i in np.flatnonzero(inside):
                out.append((t + k, p[i], int(e[i])))
    return out


# --------------------------------------------------------------------------
# accepting
# --------------------------------------------------------------------------

def corroborate(cand, ex, cfg, require):
    """Replace a predicted position with a second run's detection at that spot.

    A prediction says *where* a cell should be; another segmentation run of the
    same images says whether anything is actually there. Across these stacks a
    different run detects 42 of the 64 cells this one misses, and different runs
    fail on different cells -- the same detector at another NMS radius recovers
    only 5, so the diversity is in the training, not the suppression.

    Taking the whole union is the wrong way to use that: it raises node recall to
    0.981 but carries every one of the other runs' spurious detections too, and
    on these stacks the node excess goes 0.41 -> 0.62 and eats the gain. Asking
    only where a track says a cell is missing keeps the recall and almost none of
    the cost.

    ``ex`` is ``(positions, set_id)``. ``min_votes`` counts *distinct* runs, not
    detections: one run's spurious blob corroborates nothing, two runs that both
    see something the base run missed are evidence.

    ``require`` drops predictions nothing corroborates, which is the strict
    reading -- invent no node, only reinstate one some detector already found.
    """
    if ex is None or not len(ex[0]):
        return cand[:0] if require else cand
    pos, who = ex
    tree = cKDTree(pos)
    votes = np.zeros(len(cand), int)
    best = np.full(len(cand), -1)
    for i, near in enumerate(tree.query_ball_point(cand, cfg["snap_um"])):
        if not near:
            continue
        votes[i] = len(set(who[j] for j in near))
        d = np.linalg.norm(pos[near] - cand[i], axis=1)
        best[i] = near[int(np.argmin(d))]
    hit = (votes >= cfg["min_votes"]) & (best >= 0)
    if require:
        return pos[best[hit]]
    out = cand.copy()
    out[hit] = pos[best[hit]]
    return out


def accept(coords, links, drift, proposals, cfg, rule="cost", extra=None,
           require_extra=False):
    """Which proposed nodes to keep. Returns ``(T, 3)`` positions in um and their t.

    ``rule``:
      ``"all"``   keep everything that does not sit on an existing detection.
      ``"cost"``  add the frame's candidates to that frame's assignment problem
                  and keep only the ones the solver links. A candidate then has
                  to beat every competing explanation, including leaving the
                  cell unlinked at ``lambda_um``, and two ends cannot both be
                  given the same hole.
    """
    pos = coords[:, 1:].astype(np.float64) * SCALE_ARR
    rows = rows_by_t(coords)
    has_child, _ = _ends_and_starts(coords, links)

    by_t: dict[int, list] = {}
    for t, p, src in proposals:
        by_t.setdefault(int(t), []).append((p, src))

    kept_t, kept_p = [], []
    # Frames are walked in order, not just the ones with proposals, so a frame
    # that accepts nothing breaks the chain rather than silently carrying the
    # previous frame's candidates forward across a hole.
    accepted_prev = np.zeros((0, 3))
    for t in range(min(rows), max(rows) + 1):
        items = by_t.get(t)
        if not items:
            accepted_prev = np.zeros((0, 3))
            continue
        cand = np.asarray([p for p, _ in items], np.float64)
        here = rows.get(t, np.empty(0, int))
        # Never insert on top of a detection that exists: the cell was not
        # missed, and a second node there only competes for its match.
        if len(here) and cfg["min_dup_um"] > 0:
            d, _ = cKDTree(pos[here]).query(cand, k=1)
            cand = cand[d > cfg["min_dup_um"]]
        if len(cand) and extra is not None:
            cand = corroborate(cand, extra.get(t), cfg, require_extra)
            # Snapping moves a prediction by up to snap_um, which can land it
            # back on a base detection; the dup rule has to hold after the move.
            if len(cand) and len(here) and cfg["min_dup_um"] > 0:
                d, _ = cKDTree(pos[here]).query(cand, k=1)
                cand = cand[d > cfg["min_dup_um"]]
        if not len(cand):
            accepted_prev = np.zeros((0, 3))
            continue
        # Two proposals for the same hole are one hole.
        if len(cand) > 1:
            keep = np.ones(len(cand), bool)
            order = np.argsort(cand[:, 0])
            tree = cKDTree(cand)
            for i in order:
                if not keep[i]:
                    continue
                for j in tree.query_ball_point(cand[i], cfg["min_dup_um"]):
                    if j != i:
                        keep[j] = False
            cand = cand[keep]

        if rule == "all":
            kept_t += [t] * len(cand)
            kept_p.append(cand)
            continue


        # The sources in doubt: real detections at t-1 that nothing continues,
        # *plus* the candidates accepted at t-1. The second half is what makes a
        # multi-frame bridge possible at all -- its second node's only possible
        # parent is its first node, which does not exist until this loop puts it
        # there, and testing it against the real detections alone rejects every
        # bridge longer than one frame.
        prev = rows.get(t - 1, np.empty(0, int))
        src_um = pos[prev[~has_child[prev]]] if len(prev) else np.zeros((0, 3))
        if len(accepted_prev):
            src_um = np.concatenate([src_um, accepted_prev])
        if not len(src_um):
            accepted_prev = np.zeros((0, 3))
            continue
        src_um = src_um + drift.get(t - 1, np.zeros(3))
        _, sol = solve_pair(src_um, cand, cfg["lambda_um"] - cfg["margin_um"])
        take = sorted(set(sol.values()))
        accepted_prev = cand[take] if take else np.zeros((0, 3))
        if take:
            kept_t += [t] * len(take)
            kept_p.append(cand[take])

    if not kept_p:
        return np.zeros((0, 4), np.float64)
    p = np.concatenate(kept_p)
    return np.concatenate([np.asarray(kept_t, np.float64)[:, None], p / SCALE_ARR], 1)


def add_nodes(coords, new_rows):
    """Append recovered nodes. Old row indices keep their meaning."""
    if not len(new_rows):
        return coords
    return np.vstack([coords, np.round(new_rows).astype(coords.dtype)])


def reverse_time(coords):
    """Coordinates with time flipped, so a forward stage sees the backward case."""
    out = coords.copy()
    out[:, 0] = coords[:, 0].max() - coords[:, 0]
    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

class Scorer:
    def __init__(self, coords, gt: GroundTruth, n_total: float):
        self.gt, self.n_total = gt, n_total
        m = match_nodes(coords.astype(np.int64), gt)
        self.g = np.full(len(coords), -1, np.int64)
        for r, n in m.items():
            self.g[r] = n
        self.hc = np.isin(self.g, list(gt.has_child))
        self.hp = np.isin(self.g, list(gt.has_parent))
        self.matched = set(m.values())
        self.node_recall = len(self.matched) / gt.n_nodes if gt.n_nodes else float("nan")

    def row(self, edges, n_nodes):
        e = np.asarray(edges).reshape(-1, 2)
        if not len(e):
            return row_from_counts(0, 0, self.gt.n_edges, n_nodes,
                                   self.n_total, self.node_recall)
        a, b = e[:, 0], e[:, 1]
        valid = self.hc[a] | self.hp[b]
        tp = np.fromiter(((int(x), int(y)) in self.gt.pairs
                          for x, y in zip(self.g[a], self.g[b])), bool, len(a))
        tp = int((tp & valid).sum())
        return row_from_counts(tp, int(valid.sum()) - tp, self.gt.n_edges - tp,
                               n_nodes, self.n_total, self.node_recall)


def load(sample, nodes_dir, meta):
    g = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{sample}.geff")
    g = g[0] if isinstance(g, tuple) else g
    coords = np.load(Path(nodes_dir) / f"{sample}.npz")["coords"].astype(np.int64)
    n_total = float(meta.loc[meta.sample_name == sample,
                             "estimated_number_of_nodes"].iloc[0])
    return GroundTruth(g), coords, n_total


def extra_by_frame(base_coords, extra_coords, cfg):
    """``frame -> (M, 3) um`` detections other runs have that this one does not.

    Only the ones the base set does not already cover: a detection both runs
    found is not a recovery, and leaving it in would let a prediction snap to a
    node that is already there.
    """
    if not extra_coords:
        return None
    base_pos = base_coords[:, 1:].astype(np.float64) * SCALE_ARR
    base_rows = rows_by_t(base_coords)
    out: dict[int, list] = {}
    for i, c in enumerate(extra_coords):
        for t, rws in rows_by_t(c).items():
            if not len(rws):
                continue
            p = c[rws, 1:].astype(np.float64) * SCALE_ARR
            here = base_rows.get(t, np.empty(0, int))
            if len(here):
                d, _ = cKDTree(base_pos[here]).query(p, k=1)
                p = p[d > cfg["min_dup_um"]]
            if len(p):
                out.setdefault(t, []).append((p, np.full(len(p), i)))
    return {t: (np.concatenate([a for a, _ in v]),
                np.concatenate([b for _, b in v])) for t, v in out.items()}


def repair(coords, cfg, stages=("bridge", "open"), rule="cost", both_ways=True,
           extra_coords=None, require_extra=False):
    """The whole chain: base graph -> propose -> accept -> re-link.

    Returns ``(coords, links, n_added)`` with the recovered nodes appended.
    """
    drift = drift_table(coords)
    links = base_graph(coords, drift, cfg["link_um"])
    props = []
    if "bridge" in stages:
        props += propose_bridges(coords, links, drift, cfg)
    if "open" in stages:
        props += propose_open(coords, links, drift, cfg)
        if both_ways:
            rev = reverse_time(coords)
            rdrift = drift_table(rev)
            rlinks = base_graph(rev, rdrift, cfg["link_um"])
            tmax = int(coords[:, 0].max())
            for t, p, src in propose_open(rev, rlinks, rdrift, cfg):
                props.append((tmax - int(t), p, src))
    extra = extra_by_frame(coords, extra_coords, cfg) if extra_coords else None
    new = accept(coords, links, drift, props, cfg, rule=rule,
                 extra=extra, require_extra=require_extra)
    out = add_nodes(coords, new)
    if len(new):
        d2 = drift_table(out)
        links = base_graph(out, d2, cfg["link_um"])
    return out, links, len(new)


# --------------------------------------------------------------------------

ARMS = {
    "baseline (no recovery)":      dict(stages=()),
    "bridges":                     dict(stages=("bridge",), rule="all"),
    "bridges, cost test":          dict(stages=("bridge",), rule="cost"),
    "bridges + open":              dict(stages=("bridge", "open"), rule="all"),
    "bridges + open, cost test":   dict(stages=("bridge", "open"), rule="cost"),
    "open only":                   dict(stages=("open",), rule="cost"),
    # need --extra; skipped when none is given
    "bridges, snapped":            dict(stages=("bridge",), rule="cost",
                                        needs_extra=True),
    "bridges, corroborated":       dict(stages=("bridge",), rule="cost",
                                        needs_extra=True, require_extra=True),
    "bridges + open, corroborated": dict(stages=("bridge", "open"), rule="cost",
                                         needs_extra=True, require_extra=True),
}


def _job(task):
    sample, nodes_dir, extra_dirs, cfg, arms = task
    meta = pd.read_csv("../input/metadata.csv")
    gt, coords, n_total = load(sample, nodes_dir, meta)
    extra = [np.load(Path(d) / f"{sample}.npz")["coords"].astype(np.int64)
             for d in extra_dirs]
    base = Scorer(coords, gt, n_total)
    out = {}
    for name, kw in arms.items():
        kw = dict(kw)
        needs = kw.pop("needs_extra", False)
        if needs and not extra:
            continue
        if needs:
            kw["extra_coords"] = extra
        if not kw.get("stages"):
            drift = drift_table(coords)
            links = base_graph(coords, drift, cfg["link_um"])
            c, n = coords, 0
        else:
            c, links, n = repair(coords, cfg, **kw)
        s = Scorer(c, gt, n_total)
        row = s.row(links, len(c))
        row["_added"] = n
        row["_gained"] = len(s.matched) - len(base.matched)
        row["_missing"] = gt.n_nodes - len(s.matched)
        out[name] = row
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nodes", default="../logs/2026-09-13/1/preds_r6_0.40",
                   help="a seg model's node folder (the edge runs' nodes_folder)")
    p.add_argument("--samples", nargs="*", default=None,
                   help="default = five fold-0 stacks with node recall < 1")
    p.add_argument("--extra", nargs="*", default=[],
                   help="other seg runs' node folders, used to corroborate a "
                        "prediction with a detection something actually found")
    p.add_argument("--arms", action="store_true", help="score every stage separately")
    p.add_argument("--jobs", type=int, default=8)
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    a = p.parse_args()

    samples = a.samples or ["44b6_e35b117d", "44b6_2f31fc2f", "44b6_5f15d135",
                            "44b6_e57ff5c6", "44b6_996155de"]
    cfg = {k: getattr(a, k) for k in DEFAULTS}
    default = ("baseline (no recovery)", "bridges")
    arms = ARMS if a.arms else {k: ARMS[k] for k in default}
    print(f"-> {len(samples)} stacks from {a.nodes}")
    for d in a.extra:
        print(f"   corroborating with {d}")
    print("   " + "  ".join(f"{k}={v}" for k, v in cfg.items()))

    with Pool(a.jobs) as pool:
        res = pool.map(_job, [(s, a.nodes, a.extra, cfg, arms) for s in samples])

    acc = {}
    for r in res:
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    print(f"\n{'arm':<30}{'added':>8}{'+GT':>6}{'miss':>6}{'NR':>8}"
          f"{'tp':>6}{'fp':>5}{'fn':>5}{'J':>8}{'adjJ':>8}")
    base = None
    for name in arms:
        rows = acc.get(name)
        if not rows:
            continue
        ru = roll_up(rows)
        if base is None:
            base = ru["adj_edge_jaccard"]
        print(f"{name:<30}{sum(r['_added'] for r in rows):>8}"
              f"{sum(r['_gained'] for r in rows):>6}{sum(r['_missing'] for r in rows):>6}"
              f"{ru['node_recall']:>8.4f}{ru['edge_tp']:>6}{ru['edge_fp']:>5}"
              f"{ru['edge_fn']:>5}{ru['edge_jaccard']:>8.4f}{ru['adj_edge_jaccard']:>8.4f}"
              f"   {ru['adj_edge_jaccard']-base:+.4f}")


if __name__ == "__main__":
    main()
