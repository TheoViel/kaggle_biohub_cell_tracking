"""Is the localisation error temporal jitter, and does smoothing take it out?

`analysis_steal.py` shows that on a lost link the annotated cell moved 2.3 um
while the two detections matched to it sit 8.5 um apart, each about 4 um off the
annotation, with nothing closer available. Detection error is therefore about
twice the cell's own step and half the spacing between cells — and, since the
two endpoints disagree, it is *not* a fixed offset per cell. It re-rolls every
frame.

That is testable and it is worth testing before proposing anything: if the error
is independent frame to frame, averaging a cell's position along its own track
must reduce it by roughly the square root of the window, and the detections
would land on the cells. If the error is instead a stable per-cell offset,
smoothing changes nothing and the story is wrong.

Reports, over the ground-truth nodes: the error of the detection matched to each
one, and the error after replacing that detection's position with a local fit
along its track.

Run:  python src/postprocessing/analysis_jitter.py --log_folder ../logs/2026-08-20/20/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_worst import SCALE_ARR, load_stack, match_nodes
from common import ROOT
from dedup import track_labels


def smooth_tracks(coords, edges, window, mode="line"):
    """Positions replaced by a local fit along each track, in um.

    ``mean`` is a centred moving average — the right estimator if the error is
    independent noise on a slowly moving cell. ``line`` fits a straight line in
    time over the window and reads off the middle, which additionally survives a
    cell that is moving steadily through it.
    """
    pos_um = coords[:, 1:] * SCALE_ARR
    label = track_labels(len(coords), edges)
    out = pos_um.copy()
    order = np.argsort(label, kind="stable")
    bounds = np.flatnonzero(np.diff(label[order])) + 1
    for chunk in np.split(order, bounds):
        if len(chunk) < 2:
            continue
        t = coords[chunk, 0]
        s = np.argsort(t)
        chunk, t = chunk[s], t[s]
        p = pos_um[chunk]
        for k in range(len(chunk)):
            lo, hi = max(0, k - window), min(len(chunk), k + window + 1)
            if hi - lo < 2:
                continue
            if mode == "mean":
                out[chunk[k]] = p[lo:hi].mean(axis=0)
            else:
                tt = t[lo:hi] - t[k]
                A = np.stack([tt, np.ones_like(tt)], axis=1)
                coef, *_ = np.linalg.lstsq(A, p[lo:hi], rcond=None)
                out[chunk[k]] = coef[1]          # the fit evaluated at t[k]
    return out


def analyse(sample, npz_path, windows, mode):
    coords, edges, gt_graph = load_stack(npz_path, sample)
    g2p = match_nodes(coords, edges, gt_graph)
    attrs = gt_graph.node_attrs(attr_keys=["z", "y", "x"])
    gt_um = {
        int(i): np.array(p) * SCALE_ARR
        for i, p in zip(gt_graph.node_ids(),
                        np.column_stack([attrs[c].to_numpy() for c in "zyx"]))
    }
    pairs = [(p, gt_um[int(g)]) for g, p in g2p.items()]
    if not pairs:
        return {}

    rows, targets = np.array([p for p, _ in pairs]), np.array([t for _, t in pairs])
    raw = coords[:, 1:] * SCALE_ARR
    out = {"raw": np.linalg.norm(raw[rows] - targets, axis=1)}
    for w in windows:
        sm = smooth_tracks(coords, edges, w, mode=mode)
        out[f"w{w}"] = np.linalg.norm(sm[rows] - targets, axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_folder", default=str(ROOT / "logs" / "2026-08-20" / "20"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--windows", type=int, nargs="*", default=[1, 2, 3, 5])
    ap.add_argument("--mode", default="line", choices=("line", "mean"))
    args = ap.parse_args()

    log_folder = Path(args.log_folder)
    scores = json.loads((log_folder / "scores_pp.json").read_text())
    preds_pp = log_folder / "preds_pp"
    ranked = sorted(scores["per_sample"].items(), key=lambda kv: kv[1]["score"])

    for group, items in (("worst", ranked[: args.n]), ("best", ranked[-args.n:])):
        acc = {}
        for name, _ in items:
            for k, v in analyse(name, preds_pp / f"{name}.npz",
                                args.windows, args.mode).items():
                acc.setdefault(k, []).append(v)
        print(f"\n=== {group} {args.n} === localisation error vs annotation, um "
              f"({args.mode} fit)")
        print(f"{'':<8s}{'median':>9s}{'mean':>9s}{'p75':>9s}{'p90':>9s}{'>3um':>9s}")
        for k, vs in acc.items():
            a = np.concatenate(vs)
            print(f"{k:<8s}{np.median(a):>9.2f}{a.mean():>9.2f}"
                  f"{np.percentile(a, 75):>9.2f}{np.percentile(a, 90):>9.2f}"
                  f"{(a > 3).mean():>9.1%}")


if __name__ == "__main__":
    main()
