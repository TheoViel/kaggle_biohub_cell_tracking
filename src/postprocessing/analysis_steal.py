"""What the geometry of a stolen link actually looks like, in raw microns.

The rank statistics say the true parent is the second or third nearest
candidate once the frame's flow is removed, and that the parent we picked
instead is several microns closer. Two very different situations produce that,
and they call for opposite fixes:

* the annotated cell **moved further than its neighbours did**, so a
  displacement prior built from the neighbourhood points at the wrong cell — the
  prior is the problem;
* the two candidates are a **near tie** and the flow correction tips it the
  wrong way — the tie-break is the problem.

So measure, per stolen ground-truth link, the distances nothing is corrected in:
how far the annotation itself moved between the two frames, how far the
detection we linked sits from the child, and how far the true parent does.

Run:  python src/postprocessing/analysis_steal.py --log_folder ../logs/2026-08-20/20/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_worst import MATCH_UM, SCALE_ARR, gt_edges, load_stack, match_nodes
from common import ROOT


def analyse(sample, npz_path):
    coords, edges, gt_graph = load_stack(npz_path, sample)
    g2p = match_nodes(coords, edges, gt_graph)
    parent_of = {int(t): int(s) for s, t, *_ in edges}
    pos_um = coords[:, 1:] * SCALE_ARR

    attrs = gt_graph.node_attrs(attr_keys=["z", "y", "x"])
    gt_um = {
        int(i): np.array(p) * SCALE_ARR
        for i, p in zip(gt_graph.node_ids(),
                        np.column_stack([attrs[c].to_numpy() for c in "zyx"]))
    }

    t = coords[:, 0].astype(int)
    rows_by_t = {int(tt): np.flatnonzero(t == tt) for tt in np.unique(t)}
    frame_of = {int(i): int(coords[i, 0]) for i in range(len(coords))}

    def nearest(gid):
        """Closest detection to a ground-truth node, matched or not."""
        here = rows_by_t.get(frame_of_gt[int(gid)])
        if here is None or not len(here):
            return np.nan
        return float(np.linalg.norm(pos_um[here] - gt_um[int(gid)], axis=1).min())

    gt_t = gt_graph.node_attrs(attr_keys=["t"])["t"].to_numpy()
    frame_of_gt = {int(i): int(tt) for i, tt in zip(gt_graph.node_ids(), gt_t)}

    rows = []
    for gs, gt_ in gt_edges(gt_graph):
        ps, pt = g2p.get(gs), g2p.get(gt_)
        if ps is None or pt is None:
            continue
        got = parent_of.get(pt)
        stolen = got is not None and got != ps
        rows.append(dict(
            stolen=stolen,
            # how far the annotation itself moved over the frame
            gt_step=float(np.linalg.norm(gt_um[int(gt_)] - gt_um[int(gs)])),
            # the link we should have made, between the detections that matched
            true_link=float(np.linalg.norm(pos_um[pt] - pos_um[ps])),
            # the link we did make
            made_link=float(np.linalg.norm(pos_um[pt] - pos_um[got])) if stolen else np.nan,
            loc_parent=float(np.linalg.norm(pos_um[ps] - gt_um[int(gs)])),
            loc_child=float(np.linalg.norm(pos_um[pt] - gt_um[int(gt_)])),
            # the best the detector did on that cell, whether or not it was the
            # detection the matcher tied to it
            near_parent=nearest(gs),
            near_child=nearest(gt_),
        ))
    return rows


def summarise(rows, label):
    if not rows:
        print(f"{label:<12s} (none)")
        return
    def q(key):
        a = np.array([r[key] for r in rows], dtype=float)
        a = a[~np.isnan(a)]
        if not len(a):
            return f"{'-':>6s}{'-':>7s}{'-':>7s}"
        return f"{np.percentile(a, 25):>6.2f}{np.median(a):>7.2f}{np.percentile(a, 75):>7.2f}"
    print(f"{label:<12s}{len(rows):>6d}  " + "  ".join(
        q(k) for k in ("gt_step", "true_link", "made_link", "loc_parent",
                       "loc_child", "near_parent", "near_child")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_folder", default=str(ROOT / "logs" / "2026-08-20" / "20"))
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    log_folder = Path(args.log_folder)
    scores = json.loads((log_folder / "scores_pp.json").read_text())
    preds_pp = log_folder / "preds_pp"
    ranked = sorted(scores["per_sample"].items(), key=lambda kv: kv[1]["score"])

    for group, items in (("worst", ranked[: args.n]), ("best", ranked[-args.n:])):
        allr = []
        for name, _ in items:
            allr += analyse(name, preds_pp / f"{name}.npz")
        print(f"\n=== {group} {args.n} ===   p25 / median / p75, microns")
        print(f"{'':<12s}{'n':>6s}  {'gt_step':>20s}{'true_link':>21s}"
              f"{'made_link':>21s}{'loc_parent':>21s}{'loc_child':>21s}"
              f"{'near_parent':>21s}{'near_child':>21s}")
        summarise([r for r in allr if r["stolen"]], "stolen")
        summarise([r for r in allr if not r["stolen"]], "kept")


if __name__ == "__main__":
    main()
