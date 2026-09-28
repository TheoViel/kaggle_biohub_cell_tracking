#!/usr/bin/env python
"""Drop a fork when another fork sits within N frames of it on the same lineage.

    python scripts/ancestry_spacing.py --preds_dir /tmp/g28_one_44b6 --frames 3

A cell does not divide twice in three frames. So two forks on one ancestral
chain, that close together, are at least one false positive -- and the division
metric charges every one of them that lands near annotation. This walks up the
parent chain from each fork and, where it meets another fork inside `frames`,
keeps the more confident one.

Ported from the `suppress_close_ancestral_forks` in a teammate's division
branch, which reaches div_J 0.4146 where ours reaches 0.2557 on the same
annotation. Two differences from that implementation, both forced by running
*after* the gate rather than inside the proposer:

* **Confidence is the second child's edge probability, not the classifier's.**
  Theirs ranks by the fused mitosis probability, which is the right key; ours
  has already been thresholded away by the time the graph is written. The edge
  probability is a proxy and a worse one -- it is what the geometric proposer
  gated on, not what the classifier believes.
* **No fork is protected.** Theirs protects forks that predate the division
  stage and only arbitrates new proposals; the gated output does not record
  which were which, so every fork competes.

Both make this a lower bound on what the idea is worth, not a reproduction.

Note what this is *not*: a spatial dedup. Forks within 7 um and +/-1 frame of
each other are 0.4% of ours, which is why a proximity filter found nothing. A
lineage proposing forks at t and t+2 is not spatially clustered at all.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "postprocessing"))

import edge_pp  # noqa: E402
import multi_hypothesis as M  # noqa: E402
from tracking_cellmot.metrics import summarise  # noqa: E402


def space_one(coords, edges, frames):
    """Return `edges` with the losing fork of each close ancestral pair removed."""
    edges = np.asarray(edges, float).reshape(-1, 4)
    if not len(edges):
        return edges, 0
    n = len(coords)
    src = edges[:, 0].astype(int)
    tgt = edges[:, 1].astype(int)

    parent = np.full(n, -1, np.int64)
    parent[tgt] = src
    outdeg = np.bincount(src, minlength=n)
    forks = np.flatnonzero(outdeg >= 2)
    if len(forks) < 2:
        return edges, 0

    # A fork's confidence and the row that would be dropped: its *weakest*
    # outgoing edge, which is the second child the proposer added.
    weakest_row, conf = {}, {}
    for p in forks:
        rows = np.flatnonzero(src == p)
        k = rows[np.argmin(edges[rows, 2])]
        weakest_row[int(p)] = int(k)
        conf[int(p)] = float(edges[k, 2])

    fork_set = set(int(p) for p in forks)
    conflicts = {p: set() for p in fork_set}
    for child in fork_set:
        anc = int(parent[child])
        while anc >= 0 and (coords[child, 0] - coords[anc, 0]) <= frames:
            if anc in fork_set:
                conflicts[child].add(anc)
                conflicts[anc].add(child)
            anc = int(parent[anc])

    accepted, drop = set(), []
    for p in sorted(fork_set, key=lambda v: (-conf[v], v)):
        if conflicts[p] & accepted:
            drop.append(weakest_row[p])
        else:
            accepted.add(p)
    if not drop:
        return edges, 0
    return edges[np.setdiff1d(np.arange(len(edges)), np.asarray(drop, int))], len(drop)


def run_stack(arg):
    stem, pred_dir, frames_list, min_track = arg
    z = np.load(Path(pred_dir) / f"{stem}.npz")
    coords = np.asarray(
        z["coords_exact"] if "coords_exact" in z.files else z["coords"], float)
    edges = np.asarray(z["edges"], float).reshape(-1, 4)
    out = []
    for f in frames_list:
        e, n_drop = (edges, 0) if f == 0 else space_one(coords, edges, f)
        c, e = edge_pp.prune_short(coords, e, min_track)
        r = M.score_one(stem, c, e)
        r["_frames"] = f
        r["_dropped"] = n_drop
        out.append(r)
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds_dir", nargs="+", required=True,
                   help="gated output folder(s); several are pooled, which is "
                        "how an out-of-fold pair of embryos is scored together")
    p.add_argument("--frames", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    p.add_argument("--min_track", type=int, default=5)
    p.add_argument("--jobs", type=int, default=24)
    a = p.parse_args()

    jobs = [(q.stem, str(d), list(a.frames), a.min_track)
            for d in a.preds_dir for q in sorted(Path(d).glob("*.npz"))]
    got = Parallel(n_jobs=a.jobs)(delayed(run_stack)(j) for j in jobs)
    by = {}
    for rows in got:
        for r in rows:
            by.setdefault(r["_frames"], []).append(r)

    print(f"{len(jobs)} stacks\n")
    print(f"{'frames':>7}{'cv':>9}{'adj':>9}{'divJ':>8}{'44b6':>8}{'6bba':>8}"
          f"{'tp':>5}{'fp':>5}{'fn':>5}{'dropped':>9}")
    for f in sorted(by):
        rows = by[f]
        per = {e: summarise([r for r in rows if r["embryo"] == e])
               for e in sorted({r["embryo"] for r in rows})}
        cv = float(np.mean([per[e]["score"] for e in per]))
        adj = float(np.mean([per[e]["adj_edge_jaccard"] for e in per]))
        dj = float(np.mean([per[e]["division_jaccard"] for e in per]))
        tp = sum(per[e]["division_tp"] for e in per)
        fp = sum(per[e]["division_fp"] for e in per)
        fn = sum(per[e]["division_fn"] for e in per)
        d = sum(r["_dropped"] for r in rows)
        j4 = per.get("44b6", {}).get("division_jaccard", float("nan"))
        j6 = per.get("6bba", {}).get("division_jaccard", float("nan"))
        tag = "off" if f == 0 else str(f)
        print(f"{tag:>7}{cv:>9.4f}{adj:>9.4f}{dj:>8.4f}{j4:>8.4f}{j6:>8.4f}"
              f"{tp:>5}{fp:>5}{fn:>5}{d:>9}")


if __name__ == "__main__":
    sys.exit(main())
