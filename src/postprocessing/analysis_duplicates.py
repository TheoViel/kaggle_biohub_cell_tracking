"""The duplicate detections that steal links, and whether they are separable.

`analysis_worst.py` splits the lost ground-truth links into a thief that is a
*different cell* and a thief that is *another detection of the same cell*, and
finds the second kind is the majority on the worst stacks and the only kind on
the best ones. That failure is not a tracking error at all: the link is
physically right, and the score charges for it because the matcher tied the
ground-truth node to the other copy.

This asks whether the copies can be told apart well enough to merge, which
would fix the link and remove a node from the count at the same time. For every
duplicate steal it reports what the two copies are to each other — the same
track, or two tracks, and if two, how long they run side by side and how far
apart.

Run:  python src/postprocessing/analysis_duplicates.py --log_folder ../logs/2026-08-20/20/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from analysis_worst import MATCH_UM, SCALE_ARR, gt_edges, load_stack, match_nodes
from common import ROOT
from dedup import track_labels


def _rows_by_t(coords):
    t = coords[:, 0].astype(int)
    return {int(tt): np.flatnonzero(t == tt) for tt in np.unique(t)}


def pair_stats(a, b, coords, pos_um, label, rows_by_t):
    """What two co-located detections are to each other, over their whole lives."""
    if label[a] == label[b]:
        return dict(same_track=True)
    ta = np.flatnonzero(label == label[a])
    tb = np.flatnonzero(label == label[b])
    fa = {int(coords[i, 0]): i for i in ta}
    fb = {int(coords[i, 0]): i for i in tb}
    shared = sorted(set(fa) & set(fb))
    if not shared:
        return dict(same_track=False, overlap=0, len_a=len(ta), len_b=len(tb))
    sep = np.array([np.linalg.norm(pos_um[fa[t]] - pos_um[fb[t]]) for t in shared])
    return dict(
        same_track=False,
        overlap=len(shared),
        span=max(len(fa), len(fb)),
        len_a=len(ta),
        len_b=len(tb),
        sep_median=float(np.median(sep)),
        sep_max=float(sep.max()),
    )


def analyse(sample, npz_path):
    coords, edges, gt_graph = load_stack(npz_path, sample)
    g2p = match_nodes(coords, edges, gt_graph)
    parent_of = {int(t): int(s) for s, t, *_ in edges}
    pos_um = coords[:, 1:] * SCALE_ARR
    rows_by_t = _rows_by_t(coords)
    label = track_labels(len(coords), edges)

    gt_attrs = gt_graph.node_attrs(attr_keys=["z", "y", "x"])
    gt_um = {
        int(i): np.array(p) * SCALE_ARR
        for i, p in zip(gt_graph.node_ids(),
                        np.column_stack([gt_attrs[c].to_numpy() for c in "zyx"]))
    }

    cases = []
    for gs, gt_ in gt_edges(gt_graph):
        ps, pt = g2p.get(gs), g2p.get(gt_)
        if ps is None or pt is None:
            continue
        got = parent_of.get(pt)
        if got is None or got == ps:
            continue
        if np.linalg.norm(pos_um[got] - gt_um[int(gs)]) > MATCH_UM:
            continue                                   # a different cell, not a copy
        st = pair_stats(ps, got, coords, pos_um, label, rows_by_t)
        st["gap_um"] = float(np.linalg.norm(pos_um[ps] - pos_um[got]))
        cases.append(st)

    # How common are co-located pairs at all, and are they parallel tracks?
    close, parallel = 0, 0
    for t, rows in rows_by_t.items():
        if len(rows) < 2:
            continue
        pairs = cKDTree(pos_um[rows]).query_pairs(MATCH_UM)
        close += len(pairs)
        for i, j in pairs:
            parallel += label[rows[i]] != label[rows[j]]
    return cases, dict(n_nodes=len(coords), close=close, parallel=parallel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_folder", default=str(ROOT / "logs" / "2026-08-20" / "20"))
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    log_folder = Path(args.log_folder)
    scores = json.loads((log_folder / "scores_pp.json").read_text())
    preds_pp = log_folder / "preds_pp"
    worst = sorted(scores["per_sample"].items(), key=lambda kv: kv[1]["score"])[: args.n]

    allc, tot = [], defaultdict(int)
    print(f"{'sample':<16s}{'dupes':>7s}{'same':>6s}{'2trk':>6s}{'gap_um':>8s}"
          f"{'ovlp':>6s}{'sep':>7s}{'shortr':>7s}{'close':>8s}{'/node':>7s}")
    for name, _row in worst:
        cases, g = analyse(name, preds_pp / f"{name}.npz")
        allc += cases
        for k, v in g.items():
            tot[k] += v
        same = sum(c["same_track"] for c in cases)
        two = [c for c in cases if not c["same_track"] and c.get("overlap", 0) > 0]
        print(f"{name:<16s}{len(cases):>7d}{same:>6d}{len(two):>6d}"
              f"{np.median([c['gap_um'] for c in cases]) if cases else float('nan'):>8.2f}"
              f"{np.median([c['overlap'] for c in two]) if two else float('nan'):>6.0f}"
              f"{np.median([c['sep_median'] for c in two]) if two else float('nan'):>7.2f}"
              f"{np.median([min(c['len_a'], c['len_b']) for c in two]) if two else float('nan'):>7.0f}"
              f"{g['close']:>8d}{g['close'] / g['n_nodes']:>7.3f}")

    two = [c for c in allc if not c["same_track"] and c.get("overlap", 0) > 0]
    print(f"\n{len(allc)} duplicate steals over the {args.n} worst stacks")
    print(f"  same track          : {sum(c['same_track'] for c in allc)}")
    print(f"  two tracks, overlap : {len(two)}")
    print(f"  two tracks, disjoint: {len(allc) - len(two) - sum(c['same_track'] for c in allc)}")
    if two:
        ov = np.array([c["overlap"] for c in two])
        sep = np.array([c["sep_median"] for c in two])
        short = np.array([min(c["len_a"], c["len_b"]) for c in two])
        frac = np.array([c["overlap"] / c["span"] for c in two])
        for name, arr in (("overlap frames", ov), ("median separation um", sep),
                          ("shorter track len", short), ("overlap / span", frac)):
            print(f"  {name:<22s} p25 {np.percentile(arr, 25):.2f}  "
                  f"median {np.median(arr):.2f}  p75 {np.percentile(arr, 75):.2f}")
    print(f"\n  co-located pairs within {MATCH_UM} um: {tot['close']} "
          f"over {tot['n_nodes']} nodes ({tot['close'] / tot['n_nodes']:.3f}/node), "
          f"{tot['parallel'] / max(tot['close'], 1):.1%} of them on two different tracks")


if __name__ == "__main__":
    main()
