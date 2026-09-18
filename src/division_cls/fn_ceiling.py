"""How many GT divisions could any fork proposer reach? The ceiling, per stage.

    cd src && python division_cls/fn_ceiling.py --embryo 44b6

Tuning the proposer only pays where the proposal is possible. A GT division is
reachable as a fork only if the detector found the parent *and* both daughters,
and the edge model scored a candidate from that parent to each of them. Each of
those is a separate loss, and they are attributable:

    annotated divisions                     26 on fold 0
      parent detected                       ... the detector's loss
      + both daughters detected             ... the detector's loss
      + a candidate edge to each daughter   ... the edge model's loss
      = reachable                           the ceiling for any div_* tuning

Whatever `propose_divisions` and the classifier then do, they are dividing up
the last row. If it is much below 26, the division term is detector-limited and
no amount of gate loosening will find the rest.

Matching is the metric's own rule: nearest GT node within `max_distance` um,
`SCALE`-aware, so "detected" here means what `evaluate` would count.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tracksdata as td

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "postprocessing"))

from params import SCALE, TRAIN_DIR  # noqa: E402


def _gt_divisions(geff_path):
    """`[(parent_tzyx, child1_tzyx, child2_tzyx), ...]` for every GT fork."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        res = td.graph.IndexedRXGraph.from_geff(Path(geff_path))
    g = res[0] if isinstance(res, tuple) else res
    ids = g.node_ids()
    attrs = g.node_attrs(attr_keys=["t", "z", "y", "x"])
    pos = {int(i): np.array([attrs["t"][k], attrs["z"][k], attrs["y"][k], attrs["x"][k]], float)
           for k, i in enumerate(ids)}
    out = []
    for i in ids:
        succ = list(g.successors(int(i))) if hasattr(g, "successors") else []
        if len(succ) >= 2:
            out.append((pos[int(i)], pos[int(succ[0])], pos[int(succ[1])]))
    return out


def _near(point, coords, max_um):
    """Index of the nearest detection within `max_um`, or -1. Same frame only."""
    same_t = np.flatnonzero(coords[:, 0] == point[0])
    if not len(same_t):
        return -1
    d = np.linalg.norm((coords[same_t, 1:] - point[1:]) * np.asarray(SCALE), axis=1)
    k = int(np.argmin(d))
    return int(same_t[k]) if d[k] <= max_um else -1


def _assigned(job):
    """Why a reachable division is not proposed, against the assigned graph.

    `_one` bounds what the detector and edge model make possible. This asks the
    next question: of the divisions that *are* reachable, what did `assign` do
    with the parent, because `propose_divisions` reads the committed graph and
    not the candidates. It only ever considers a parent holding **exactly one**
    committed child, and measures the sister gates against that child -- so a
    parent left childless, already forked, or holding some third node is
    skipped before any gate is consulted, and no threshold reaches it.

    Needs the cached prefix from `sweep_candidates.py` (ILP -> assign ->
    recover), which is the graph the proposer is handed.
    """
    stem, prefix_dir, data_dir, max_um = job
    z = np.load(Path(prefix_dir) / f"{stem}.npz")
    coords = z["coords"].astype(float)
    edges = z["edges"].astype(float).reshape(-1, 4)
    src, tgt = edges[:, 0].astype(int), edges[:, 1].astype(int)
    n_children = np.bincount(src, minlength=len(coords))
    kids: dict[int, list[int]] = {}
    for s, t in zip(src, tgt):
        kids.setdefault(int(s), []).append(int(t))
    claimed = np.zeros(len(coords), bool)
    claimed[tgt] = True

    got = dict(reachable=0, parent_childless=0, parent_already_forked=0,
               child_not_a_daughter=0, proposable=0, other_daughter_claimed=0)
    for parent, c1, c2 in _gt_divisions(Path(data_dir) / f"{stem}.geff"):
        pi = _near(parent, coords, max_um)
        k1, k2 = _near(c1, coords, max_um), _near(c2, coords, max_um)
        if pi < 0 or k1 < 0 or k2 < 0:
            continue
        got["reachable"] += 1
        n = n_children[pi]
        if n == 0:
            got["parent_childless"] += 1
        elif n >= 2:
            got["parent_already_forked"] += 1
        elif kids[pi][0] not in (k1, k2):
            got["child_not_a_daughter"] += 1
        else:
            got["proposable"] += 1
            other = k2 if kids[pi][0] == k1 else k1
            got["other_daughter_claimed"] += int(claimed[other])
    return got


def _one(job):
    stem, pred_dir, data_dir, max_um = job
    z = np.load(Path(pred_dir) / f"{stem}.npz")
    coords = np.asarray(z["coords"], float)
    cand = np.asarray(z["candidates"], float).reshape(-1, 5)
    # (source, target) pairs the edge model scored at all.
    pairs = {(int(a), int(b)) for a, b, *_ in cand}

    got = dict(divisions=0, parent=0, both_children=0, reachable=0)
    for parent, c1, c2 in _gt_divisions(Path(data_dir) / f"{stem}.geff"):
        got["divisions"] += 1
        pi = _near(parent, coords, max_um)
        if pi < 0:
            continue
        got["parent"] += 1
        k1, k2 = _near(c1, coords, max_um), _near(c2, coords, max_um)
        if k1 < 0 or k2 < 0:
            continue
        got["both_children"] += 1
        if (pi, k1) in pairs and (pi, k2) in pairs:
            got["reachable"] += 1
    return got


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preds_dir", default="../logs/edge_run/2026-09-13/0/preds_0.45")
    p.add_argument("--data_dir", default=TRAIN_DIR)
    p.add_argument("--embryo", default="44b6")
    p.add_argument("--max_distance", type=float, default=7.0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--prefix_dir", default="../logs/fork_cls/sweep/prefix",
                   help="cached ILP->assign->recover graphs; adds the "
                        "breakdown of why a reachable division is not proposed")
    a = p.parse_args()

    stems = sorted(q.stem for q in Path(a.preds_dir).glob("*.npz")
                   if q.stem.startswith(a.embryo))
    jobs = [(s, a.preds_dir, a.data_dir, a.max_distance) for s in stems]
    total = dict(divisions=0, parent=0, both_children=0, reachable=0)
    with Pool(a.workers) as pool:
        for got in pool.imap_unordered(_one, jobs):
            for k in total:
                total[k] += got[k]

    n = max(total["divisions"], 1)
    print(f"\n{a.embryo}: {len(stems)} stacks")
    for k, label in (("divisions", "annotated divisions"),
                     ("parent", "parent detected"),
                     ("both_children", "+ both daughters detected"),
                     ("reachable", "+ candidate edge to each  = CEILING")):
        print(f"  {label:42s} {total[k]:4d}   {total[k] / n:6.1%}")

    if a.prefix_dir and Path(a.prefix_dir).exists():
        jobs = [(s, a.prefix_dir, a.data_dir, a.max_distance) for s in stems]
        tot = dict(reachable=0, parent_childless=0, parent_already_forked=0,
                   child_not_a_daughter=0, proposable=0,
                   other_daughter_claimed=0)
        with Pool(a.workers) as pool:
            for got in pool.imap_unordered(_assigned, jobs):
                for k in tot:
                    tot[k] += got[k]
        print(f"\n  what the assignment did with the {tot['reachable']} reachable:")
        for k, label in (("parent_childless", "parent has no committed child"),
                         ("parent_already_forked", "parent already has 2+"),
                         ("child_not_a_daughter", "its one child is neither daughter"),
                         ("proposable", "one child, and it IS a daughter  = PROPOSABLE")):
            print(f"    {label:46s} {tot[k]:4d}")
        print(f"      of those, the other daughter is already claimed "
              f"{tot['other_daughter_claimed']:3d}  (what `div_steal_margin` addresses)")


if __name__ == "__main__":
    main()
