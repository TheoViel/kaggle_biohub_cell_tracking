"""Does filling one-frame detection gaps pay? Diagnosis first, then the score.

    python src/postprocessing/exp_recover.py                 # diagnose + score, one sample
    python src/postprocessing/exp_recover.py --diagnose       # just the ceiling
    python src/postprocessing/exp_recover.py --sweep          # over the gate

The diagnosis answers the only question worth asking before tuning anything:
of the ground-truth nodes the detector missed, how many are bracketed by a
track that ends the frame before and one that starts the frame after — that is
the ceiling on what this strategy can recover — and how close does the
interpolated position land.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common import ROOT, load_gt_graph, score, to_graph
from params import SCALE, TRAIN_DIR
from pipeline import DEFAULTS as PP_DEFAULTS
from pipeline import detect_recentre, postprocess
from recover import DEFAULTS as REC_DEFAULTS
from recover import drift_table, find_gaps
from relink import relink

SCALE_ARR = np.asarray(SCALE)

SAMPLE = "44b6_267148e4"
PRED_DIR = ROOT / "logs" / "2026-08-26" / "4" / "preds_0.25"
MATCH_UM = 7.0


def load_pred(sample, pred_dir):
    d = np.load(Path(pred_dir) / f"{sample}.npz", allow_pickle=True)
    coords = d["coords"].astype(np.float64)
    edges = d["edges"].astype(np.float64).reshape(-1, 4)
    shift = detect_recentre(coords)
    if shift is not None and np.any(shift):
        coords = coords.copy()
        coords[:, 1:] += np.asarray(shift, dtype=np.float64)
        print(f"recentred by {tuple(shift)}")
    return coords, edges


def load_gt_nodes(sample):
    """Ground-truth nodes as (n, 4) [t, z, y, x] — the csv stores y before z."""
    df = pd.read_csv(f"{TRAIN_DIR}/../processed/nodes/{sample}.csv")
    return df[["t", "z", "y", "x"]].to_numpy(dtype=np.float64), df["node_id"].to_numpy()


def _per_frame(coords):
    t = coords[:, 0].astype(int)
    return {int(tt): np.flatnonzero(t == tt) for tt in np.unique(t)}


def missed_gt(coords, gt, match_um=MATCH_UM):
    """Which ground-truth nodes have no detection within the matching radius."""
    pos_um = coords[:, 1:] * SCALE_ARR
    rows = _per_frame(coords)
    dist = np.full(len(gt), np.inf)
    for i, g in enumerate(gt):
        idx = rows.get(int(g[0]))
        if idx is None or len(idx) == 0:
            continue
        dist[i] = np.linalg.norm(pos_um[idx] - g[1:] * SCALE_ARR, axis=1).min()
    return dist > match_um, dist


def diagnose(coords, edges, gt, cfg):
    print(f"\nnodes={len(coords)}  frames={int(coords[:, 0].max()) + 1}  gt_nodes={len(gt)}")

    drift = drift_table(coords, n_iter=cfg["drift_iter"], refine=cfg["drift_refine"])
    dmag = np.linalg.norm(np.array([drift[t] for t in sorted(drift)]), axis=1)
    print(
        f"drift per frame (um): median {np.median(dmag):.2f}  "
        f"p90 {np.percentile(dmag, 90):.2f}  max {dmag.max():.2f}"
    )

    links = relink(
        coords, edges,
        max_distance=PP_DEFAULTS["max_distance"],
        flow_k=PP_DEFAULTS["flow_k"],
        passes=PP_DEFAULTS["passes"],
    )
    print(f"relinked edges: {len(links)}")

    miss, dist = missed_gt(coords, gt)
    print(f"\nGT nodes with no detection within {MATCH_UM} um: {miss.sum()} / {len(gt)}")
    for g in gt[miss]:
        print(f"    t={int(g[0]):>3d}  z={g[1]:.0f} y={g[2]:.0f} x={g[3]:.0f}")

    pairs, positions = find_gaps(coords, links, drift=drift, **{
        k: cfg[k] for k in REC_DEFAULTS
    })
    if len(pairs) == 0:
        print("\nproposed nodes: 0")
        return

    # One row per proposed node, not per gap: a gap of L contributes L of them.
    prop_t, pos_um = [], []
    for (a, _b), pts in zip(pairs, positions):
        prop_t.extend(int(coords[a, 0]) + k for k in range(1, len(pts) + 1))
        pos_um.extend(pts)
    prop_t, pos_um = np.asarray(prop_t), np.asarray(pos_um)
    by_len = np.bincount([len(p) for p in positions])[1:]
    print(f"\nproposed nodes: {len(pos_um)} over {len(pairs)} gaps"
          f"  (gaps of length 1..: {by_len.tolist()})")

    hit = np.zeros(miss.sum(), dtype=bool)
    hit_dist = np.full(miss.sum(), np.inf)
    for i, g in enumerate(gt[miss]):
        same = prop_t == int(g[0])
        if same.any():
            d = np.linalg.norm(pos_um[same] - g[1:] * SCALE_ARR, axis=1)
            hit_dist[i] = d.min()
            hit[i] = d.min() <= MATCH_UM
    print(
        f"missed GT nodes recovered: {hit.sum()} / {miss.sum()}"
        + (f"   (distances: {np.array2string(hit_dist, precision=1)})" if len(hit_dist) else "")
    )

    # How many proposals land on a GT node at all — the rest are unverifiable
    # here (the annotation covers ~1% of the cells), not necessarily wrong.
    gt_by_t = _per_frame(gt)
    on_gt = 0
    for k, p in enumerate(pos_um):
        idx = gt_by_t.get(int(prop_t[k]))
        if idx is not None and len(idx):
            if np.linalg.norm(gt[idx][:, 1:] * SCALE_ARR - p, axis=1).min() <= MATCH_UM:
                on_gt += 1
    print(f"proposals landing on any GT node: {on_gt} / {len(pos_um)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=SAMPLE)
    ap.add_argument("--pred_dir", default=str(PRED_DIR))
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--max_gap", type=int, default=PP_DEFAULTS["recover_max_gap"])
    ap.add_argument("--passes", type=int, default=PP_DEFAULTS["recover_passes"])
    args = ap.parse_args()

    coords, edges = load_pred(args.sample, args.pred_dir)
    gt, _ = load_gt_nodes(args.sample)
    diagnose(coords, edges, gt, {
        **REC_DEFAULTS,
        "max_residual": PP_DEFAULTS["recover_max_residual"],
        "max_gap": args.max_gap,
    })
    if args.diagnose:
        return

    print()
    score(to_graph(coords, edges), *load_gt_graph(args.sample), "raw")

    # recentre is already applied by load_pred, so the pipeline must not do it twice.
    base = dict(recentre=None)
    c0, e0 = postprocess(coords, edges, **base, recover=0)
    score(to_graph(c0, e0), *load_gt_graph(args.sample), "pipeline")

    settings = [(PP_DEFAULTS["recover_max_residual"], args.max_gap, args.passes)]
    if args.sweep:
        settings = [(r, g, p)
                    for g in (1, 2, 3) for p in (1, 2, 3)
                    for r in ([4.0, 6.0, 8.0] if (g, p) == (1, 1) else [6.0])]
    for r, g, p in settings:
        c1, e1 = postprocess(
            coords, edges, **base, recover=1,
            recover_max_residual=r, recover_max_gap=g, recover_passes=p,
        )
        added = len(c1) - len(c0)
        score(to_graph(c1, e1), *load_gt_graph(args.sample),
              f"+recover r={r} gap<={g} x{p} ({added:+d})")


if __name__ == "__main__":
    main()
