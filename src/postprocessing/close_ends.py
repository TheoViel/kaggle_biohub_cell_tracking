"""Finish the tracks that stop for no reason, on a post-processed edge graph.

    cd src/postprocessing
    python close_ends.py --preds_dir ../logs/edge_run/2026-09-15/0/preds_0.45_pp --arms

Paths are relative to ``src/``: ``common`` chdirs there, as the other scripts
here do.

A cell does not stop existing. A track that ends in the middle of a video ended
because the cell left the imaged volume, or because the graph is wrong -- the
detector missed the cell, or something else took its successor. Counting the
ends of `edge_run/2026-09-15/0/preds_0.45_pp` over ten fold-0 stacks, of 13803
ends before the last frame:

    at the volume boundary (a real exit)      7488   54.2%
    a node nearby, already claimed            4894   35.5%
    nothing within 8 um                       1123    8.1%
    a node nearby and unclaimed                298    2.2%

The last row is this module. It is the smallest of the four and the only one
that is cheap to act on: the successor exists, nobody else wants it, and the
link simply was not made.

Which of those ends cost anything
---------------------------------
Almost none of them do, because the annotation is sparse -- one to four cells a
frame against several hundred detections -- and the metric only sees an edge
whose source or target matched an annotated cell. Over the same ten stacks the
graph loses 101 ground-truth edges, and they break down as:

    parent linked elsewhere, true child taken     31   30.7%
    parent linked elsewhere, true child free      17   16.8%
    track ended at the parent                     19   18.8%
    a node the detector never found               34   33.7%

Only the third row is what this module fixes, and the first is a warning. For
each of the 31 stolen cells the thief is *geometrically nearer* to the true
child than the true parent is -- under drift alone and under drift plus the
track's own fitted velocity, in all 31 cases. So no re-assignment on geometry
recovers them; they are the model's ranking being wrong, not the linker's, and
a swap rule built on motion would make the same choice the linker already made.
That is measured, not assumed, and it is why this module only *adds* links and
never moves one.

The 19 that are reachable look like this: the true child is the nearest node to
the parent's drift-corrected prediction in 79% of cases and in the nearest three
in 100%, at a median 6.1 um. So the geometry was unambiguous and the link was
refused anyway -- `edge_pp.assign` decides on the model's probability against
`threshold`, and a pair the model scored below it is not linked however obvious
it looks. This is the geometric override of that decision, applied only where
the assignment left both ends spare.

Why adding links is nearly free
-------------------------------
The same sparsity that makes most ends worthless makes most repairs costless. An
edge between two unannotated detections is invisible to the score: it is neither
a true positive nor a false one. So the arms below add hundreds of links and
move the counts by single digits.

What it is worth: nothing, on 71 stacks
---------------------------------------
`resolve` re-solves each frame pair with a price on a source going unlinked --
the hypothesis as an optimisation term, and the one thing `edge_pp.assign` does
not have, since there a track ending costs the objective nothing at all.
`close_ends` only links spare ends to spare nodes and never moves an existing
link. On the ten development stacks `resolve term=3` looked like +0.0014. On all
71 fold-0 stacks of `edge_run/2026-09-15/0/preds_0.45_pp`:

    arm                       added     tp    fp    fn     adjJ    delta      t
    baseline                      0  18680  1158  1146   0.8800        -      -
    close ends 6um              987  18687  1161  1139   0.8802  +0.0002   1.30
    close ends 8um             2695  18692  1169  1134   0.8801  +0.0001   1.22
    resolve term=2             4603  18693  1172  1133   0.8800  +0.0000  -0.44
    resolve term=3             4970  18695  1175  1131   0.8800  -0.0000  -0.65
    resolve term=6, border 12  7288  18685  1186  1141   0.8790  -0.0009  -0.96

Paired per stack, nothing clears |t| = 2. Every arm converts a few false
negatives and buys about as many false positives doing it: `close ends 6um` is
+7 true positives for +3 false ones, which is the best of them and is +0.0002.
The 0.0014 on ten stacks was ten stacks' worth of noise.

The diagnosis is the result here, not the repair. Three measurements say why
nothing in this file can work, and they are the useful part:

  * the thief is **nearer** than the true parent in all 31 contested cases, so
    no geometric rule picks the right one;
  * the thief is a different cell, not a duplicate detection -- median 8.2 um
    away, with a *longer* track (31 frames against 25) -- so suppressing it is
    not available either;
  * 18 of the 19 plain track ends have their successor already claimed, so the
    spare-to-spare rule has almost nothing to work on.

Together those say the remaining losses are the model's *ranking* being wrong
between two plausible parents, not the linker's arithmetic. Fixing them needs
evidence this file does not have -- appearance rather than position. Two places
to look, neither tried here:

  * the contrastive embeddings, as a term in the assignment cost rather than
    only as a model input, which is what Ultrack's colour-feature gate does
    (Methods, "Color features association"): reject a candidate whose appearance
    is more than z standard deviations from the parent's;
  * `main_edge_postprocess.py --ilp`, which is off in this run's settings and is
    the same idea solved globally over the video rather than per frame pair --
    it has an explicit `ilp_disappearance` cost, so the hypothesis is already
    expressible there, over the full candidate graph rather than the committed
    edges this module is restricted to.

The second is the cheaper experiment and should come first. Note that it wants
running from `preds_0.45`, not from this folder: `linefit` has already moved the
node coordinates by the time a `_pp` graph is written, so only 13% of its nodes
map back onto the candidate array by voxel.
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
    max_um=8.0,        # how far a link may reach, after the field motion is out
    border_um=0.0,     # skip ends this close to the volume edge (0 = do not skip)
    beta=0.0,          # weight on the track's own fitted velocity; 0 = drift only
    history=4,         # frames of track the velocity is fitted over
    min_history=0,     # frames of track behind an end before it may be extended
    passes=1,          # repeat: a closed end can expose the next one
    term_um=6.0,       # what ending a track costs, in the units of distance
    appear_um=6.0,     # what a node with no parent costs
    prob_weight=0.0,   # weight on the model's log-odds where it committed an edge
)


# --------------------------------------------------------------------------

def rows_by_t(coords):
    t = coords[:, 0].astype(int)
    order = np.argsort(t, kind="stable")
    ts = t[order]
    out, lo = {}, 0
    for tt in range(int(ts[0]), int(ts[-1]) + 1):
        hi = int(np.searchsorted(ts, tt + 1))
        out[tt] = order[lo:hi]
        lo = hi
    return out


def drift_table(coords, n_iter=4, tol=2.0):
    """``D[t]``: whole-field translation in um carrying frame t onto t + 1.

    Median of nearest-neighbour offsets rather than the centroid difference,
    which moves with whatever entered or left the field; a final pass averages
    the offsets that agree with the median, for the sub-voxel part.
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


def _chains(edges, n):
    """``parent[row]``, taking the first parent where a merge somehow exists."""
    parent = np.full(n, -1, np.int64)
    src, tgt = edges[:, 0].astype(int), edges[:, 1].astype(int)
    seen = np.zeros(n, bool)
    for a, b in zip(src, tgt):
        if not seen[b]:
            parent[b] = a
            seen[b] = True
    return parent


def predict(pos, rows_sel, parent, drift, t, cfg):
    """Where each of ``rows_sel`` should be at ``t + 1``, and its history depth.

    Drift alone by default. ``beta > 0`` adds the track's own mean velocity with
    the field motion already removed, which is off because it does not help:
    over one frame a cell moves less than the detector's jitter, so the velocity
    is mostly noise, and on the cases this module targets it changes nothing.
    """
    g = drift.get(t, np.zeros(3))
    out = pos[rows_sel] + g
    depth = np.zeros(len(rows_sel), np.int64)
    for i, r in enumerate(rows_sel):
        chain, cur = [int(r)], int(r)
        for _ in range(cfg["history"]):
            cur = int(parent[cur])
            if cur < 0:
                break
            chain.append(cur)
        depth[i] = len(chain) - 1
        if cfg["beta"] and len(chain) > 1:
            vs = [(pos[chain[k]] - pos[chain[k + 1]]) - drift.get(t - k - 1, np.zeros(3))
                  for k in range(len(chain) - 1)]
            out[i] = out[i] + cfg["beta"] * np.mean(vs, axis=0)
    return out, depth


def close_ends(coords, edges, cfg):
    """Link every spare track end to a spare node in the next frame.

    One Hungarian per frame pair over the *spare* rows only -- ends with no
    child against nodes with no parent -- so nothing already decided is touched
    and no node gains a second parent. Returns ``(edges, n_added)``.
    """
    coords = np.asarray(coords)
    edges = np.asarray(edges, np.float64).reshape(-1, 4)
    pos = coords[:, 1:].astype(np.float64) * SCALE_ARR
    rows = rows_by_t(coords)
    lo = pos.min(0) + cfg["border_um"]
    hi = pos.max(0) - cfg["border_um"]
    added_total = 0

    for _ in range(int(cfg["passes"])):
        drift = drift_table(coords)
        parent = _chains(edges, len(coords))
        has_child = np.zeros(len(coords), bool)
        has_child[edges[:, 0].astype(int)] = True
        has_parent = np.zeros(len(coords), bool)
        has_parent[edges[:, 1].astype(int)] = True
        new = []
        for t in sorted(rows)[:-1]:
            here, nxt = rows[t], rows.get(t + 1, np.empty(0, int))
            if not len(here) or not len(nxt):
                continue
            ends = here[~has_child[here]]
            free = nxt[~has_parent[nxt]]
            if not len(ends) or not len(free):
                continue
            pred, depth = predict(pos, ends, parent, drift, t, cfg)
            keep = depth >= cfg["min_history"]
            if cfg["border_um"] > 0:
                keep &= np.all((pred >= lo) & (pred <= hi), axis=1)
            ends, pred = ends[keep], pred[keep]
            if not len(ends):
                continue
            cost = np.linalg.norm(pred[:, None, :] - pos[free][None, :, :], axis=-1)
            gate = cost > cfg["max_um"]
            if gate.all():
                continue
            r, c = linear_sum_assignment(np.where(gate, 1e6, cost))
            ok = ~gate[r, c]
            for i, j in zip(r[ok], c[ok]):
                # Probability 1.0 and the true distance, matching what the rest
                # of the chain writes for a link it decided on geometry.
                new.append((ends[i], free[j], 1.0, float(cost[i, j])))
        if not new:
            break
        edges = np.vstack([edges, np.asarray(new, np.float64)])
        added_total += len(new)
    return edges, added_total


def resolve(coords, edges, cfg):
    """Re-solve every frame pair with a price on a track *ending*.

    `close_ends` can only use what the assignment left spare, and the diagnosis
    says that is not where the losses are: of the 49 ground-truth children this
    graph gets wrong, 49 are already claimed by some other track. Fixing those
    means taking a child away from whoever has it, which no rule that only adds
    links can do.

    What makes that possible in principle is that the existing assignment never
    paid for the consequence. `edge_pp.assign` is a Hungarian per frame pair in
    which leaving a source unlinked is free, so a track ending costs the
    objective nothing and the solver will happily end one to gain a slightly
    better link elsewhere. The premise here is that ending a track is not free:
    a cell that does not leave the volume has a successor. `term_um` is that
    price, in the same units as distance -- a source that goes unlinked is
    charged as if it had linked that far -- and `appear_um` the matching price
    for a node at t+1 with no parent.

    With both prices set the frame pair is a balanced assignment again, and the
    solver will move a child from one parent to another whenever doing so lets
    a track survive that would otherwise have stopped. Costs are geometric,
    drift-corrected, with the model's own probability mixed in at `prob_weight`
    through its log-odds, since the committed edges carry it.

    Divisions are preserved rather than re-derived: after the one-to-one solve,
    an original edge whose target ends up unparented is put back if its source
    already has a child, which is what a division looks like here.
    """
    coords = np.asarray(coords)
    edges = np.asarray(edges, np.float64).reshape(-1, 4)
    pos = coords[:, 1:].astype(np.float64) * SCALE_ARR
    rows = rows_by_t(coords)
    drift = drift_table(coords)
    parent = _chains(edges, len(coords))
    lo = pos.min(0) + cfg["border_um"]
    hi = pos.max(0) - cfg["border_um"]

    prob_of = {(int(a), int(b)): p for a, b, p, _ in edges}
    out, moved = [], 0
    for t in sorted(rows)[:-1]:
        here, nxt = rows[t], rows.get(t + 1, np.empty(0, int))
        if not len(here) or not len(nxt):
            continue
        pred, _ = predict(pos, here, parent, drift, t, cfg)
        cost = np.linalg.norm(pred[:, None, :] - pos[nxt][None, :, :], axis=-1)
        if cfg["prob_weight"]:
            # A committed edge is evidence; turn its probability into a discount
            # on the geometric cost so the model's ranking still counts.
            for i, a in enumerate(here):
                for j, b in enumerate(nxt):
                    pr = prob_of.get((int(a), int(b)))
                    if pr is not None:
                        cost[i, j] -= cfg["prob_weight"] * float(
                            np.log(min(max(pr, 1e-6), 1 - 1e-6) / (1 - min(max(pr, 1e-6), 1 - 1e-6))))
        cost = np.where(cost > cfg["max_um"], 1e6, cost)

        # A source at the boundary may end for free: that is the cell leaving.
        term = np.full(len(here), cfg["term_um"], np.float64)
        if cfg["border_um"] > 0:
            term[~np.all((pred >= lo) & (pred <= hi), axis=1)] = 0.0
        n_s, n_t = len(here), len(nxt)
        big = np.full((n_s + n_t, n_t + n_s), 1e7)
        big[:n_s, :n_t] = cost
        np.fill_diagonal(big[:n_s, n_t:], term)                # source unlinked
        np.fill_diagonal(big[n_s:, :n_t], cfg["appear_um"])     # target unparented
        big[n_s:, n_t:] = 0.0
        r, c = linear_sum_assignment(big)
        for i, j in zip(r, c):
            if i < n_s and j < n_t and cost[i, j] < 1e6:
                a, b = int(here[i]), int(nxt[j])
                pr = prob_of.get((a, b))
                if pr is None:
                    moved += 1
                out.append((a, b, 1.0 if pr is None else pr,
                            float(np.linalg.norm(pos[b] - pos[a]))))

    new = np.asarray(out, np.float64).reshape(-1, 4)
    # Put divisions back: a target nobody now parents, whose original source
    # already has a child, was the second child of a division.
    has_parent = np.zeros(len(coords), bool)
    if len(new):
        has_parent[new[:, 1].astype(int)] = True
    has_child = np.zeros(len(coords), bool)
    if len(new):
        has_child[new[:, 0].astype(int)] = True
    back = [e for e in edges
            if not has_parent[int(e[1])] and has_child[int(e[0])]]
    if back:
        new = np.vstack([new, np.asarray(back, np.float64)])
    return new, moved


# --------------------------------------------------------------------------

class Scorer:
    def __init__(self, coords, gt: GroundTruth, n_total):
        self.gt, self.n_total = gt, n_total
        m = match_nodes(coords.astype(np.int64), gt)
        self.g = np.full(len(coords), -1, np.int64)
        for r, n in m.items():
            self.g[r] = n
        self.hc = np.isin(self.g, list(gt.has_child))
        self.hp = np.isin(self.g, list(gt.has_parent))
        self.node_recall = (len(set(m.values())) / gt.n_nodes
                            if gt.n_nodes else float("nan"))

    def row(self, edges, n_nodes):
        e = np.asarray(edges).reshape(-1, 4) if np.ndim(edges) > 1 else np.zeros((0, 4))
        if not len(e):
            return row_from_counts(0, 0, self.gt.n_edges, n_nodes,
                                   self.n_total, self.node_recall)
        a, b = e[:, 0].astype(int), e[:, 1].astype(int)
        valid = self.hc[a] | self.hp[b]
        tp = np.fromiter(((int(x), int(y)) in self.gt.pairs
                          for x, y in zip(self.g[a], self.g[b])), bool, len(a))
        tp = int((tp & valid).sum())
        return row_from_counts(tp, int(valid.sum()) - tp, self.gt.n_edges - tp,
                               n_nodes, self.n_total, self.node_recall)


def load(sample, preds_dir, meta):
    z = np.load(Path(preds_dir) / f"{sample}.npz")
    coords = z["coords"].astype(np.int64)
    edges = np.asarray(z["edges"], np.float64).reshape(-1, 4)
    g = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{sample}.geff")
    g = g[0] if isinstance(g, tuple) else g
    n_total = float(meta.loc[meta.sample_name == sample,
                             "estimated_number_of_nodes"].iloc[0])
    return GroundTruth(g), coords, edges, n_total


ARMS = {
    "baseline":                  None,
    "close ends 6um":            dict(max_um=6.0),
    "close ends 8um":            dict(max_um=8.0),
    "close ends 8um, border 10": dict(max_um=8.0, border_um=10.0),
    "resolve term=2":            dict(_mode="resolve", term_um=2.0),
    "resolve term=3":            dict(_mode="resolve", term_um=3.0),
    "resolve term=4":            dict(_mode="resolve", term_um=4.0),
    "resolve term=3, border 8":  dict(_mode="resolve", term_um=3.0, border_um=8.0),
    "resolve term=3, border 12": dict(_mode="resolve", term_um=3.0, border_um=12.0),
    "resolve term=4, border 12": dict(_mode="resolve", term_um=4.0, border_um=12.0),
    "resolve term=6, border 12": dict(_mode="resolve", term_um=6.0, border_um=12.0),
    "resolve term=3, b12, +prob": dict(_mode="resolve", term_um=3.0, border_um=12.0,
                                       prob_weight=1.0),
}


def _job(task):
    sample, preds_dir, cfg, arms = task
    meta = pd.read_csv("../input/metadata.csv")
    gt, coords, edges, n_total = load(sample, preds_dir, meta)
    s = Scorer(coords, gt, n_total)
    out = {}
    for name, kw in arms.items():
        if kw is None:
            e, n = edges, 0
        else:
            kw = dict(kw)
            fn = resolve if kw.pop("_mode", None) == "resolve" else close_ends
            e, n = fn(coords, edges, {**cfg, **kw})
        row = s.row(e, len(coords))
        row["_added"] = n
        out[name] = row
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds_dir",
                   default="../logs/edge_run/2026-09-15/0/preds_0.45_pp")
    p.add_argument("--embryo", default="44b6")
    p.add_argument("--limit", type=int, default=10,
                   help="first N stacks; 0 for all of them")
    p.add_argument("--arms", action="store_true")
    p.add_argument("--jobs", type=int, default=10)
    p.add_argument("--out", default=None, help="write the repaired npz here")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    a = p.parse_args()

    d = Path(a.preds_dir)
    samples = sorted(f.stem for f in d.glob(f"{a.embryo}_*.npz"))
    if a.limit:
        samples = samples[: a.limit]
    cfg = {k: getattr(a, k) for k in DEFAULTS}
    arms = ARMS if a.arms else {"baseline": None,
                                f"close ends, {cfg['max_um']:g}um": {}}
    print(f"-> {len(samples)} stacks from {d}")
    print("   " + "  ".join(f"{k}={v}" for k, v in cfg.items()))

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        meta = pd.read_csv("../input/metadata.csv")
        for s in samples:
            z = np.load(d / f"{s}.npz")
            coords = z["coords"].astype(np.int64)
            e, _ = close_ends(coords, np.asarray(z["edges"], np.float64).reshape(-1, 4), cfg)
            np.savez_compressed(out / f"{s}.npz", coords=z["coords"], edges=e)
        print(f"-> npz: {out}")

    with Pool(a.jobs) as pool:
        res = pool.map(_job, [(s, str(d), cfg, arms) for s in samples])
    acc = {}
    for r in res:
        for k, v in r.items():
            acc.setdefault(k, []).append(v)

    base = roll_up(acc["baseline"])["adj_edge_jaccard"]
    b_per = np.array([r["adj_edge_jaccard"] for r in acc["baseline"]], float)
    print(f"\n{'arm':<28}{'added':>8}{'tp':>7}{'fp':>6}{'fn':>6}{'J':>9}{'adjJ':>9}"
          f"{'delta':>9}{'t':>7}{'wins':>7}")
    for name in arms:
        rows = acc[name]
        ru = roll_up(rows)
        d = np.array([r["adj_edge_jaccard"] for r in rows], float) - b_per
        ok = np.isfinite(d)
        se = d[ok].std(ddof=1) / np.sqrt(ok.sum()) if ok.sum() > 1 else float("nan")
        t = d[ok].mean() / se if se and se > 0 else float("nan")
        print(f"{name:<28}{sum(r['_added'] for r in rows):>8}{ru['edge_tp']:>7}"
              f"{ru['edge_fp']:>6}{ru['edge_fn']:>6}{ru['edge_jaccard']:>9.4f}"
              f"{ru['adj_edge_jaccard']:>9.4f}{ru['adj_edge_jaccard']-base:>+9.4f}"
              f"{t:>7.2f}{(d[ok] > 0).mean():>7.0%}")
    print("\nt and wins are paired per stack against the baseline; on 71 stacks "
          "a |t| below ~2 is not a result.")


if __name__ == "__main__":
    main()
