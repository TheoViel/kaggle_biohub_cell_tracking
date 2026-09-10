"""Oracle upper bound on deduplication.

The detector fires twice on 2.85% of annotated cells (307 of 10 770 measured
over 40 stacks at `preds_0.50`). This asks what removing those is worth, by
using the annotation itself as a perfect duplicate detector:

    for each annotated cell, of the detections within `radius` keep the one
    nearest the annotation and delete the rest

then re-runs the shipped post-processing chain and scores it. No model can beat
this on the annotated cells, so whatever it returns is the ceiling for the
`adj_edge_jaccard` half of a dedup model -- and if the ceiling is small, the
model is not worth building.

    python exp_oracle_dedup.py --preds_dir ../../logs/edge_run/2026-09-08/13/preds_0.50
    python ../main_edge_postprocess.py --preds_dir <out>

What it does *not* measure: duplicates of the ~99.5% of cells the annotation
does not cover. Those cost the score only indirectly, by competing for links
with an annotated cell, so this is the direct effect alone and the true ceiling
is somewhat higher.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cell_merging_cls.merge import apply_merges  # noqa: E402

SCALE = np.array([1.625, 0.40625, 0.40625])


def duplicates_for_stack(coords, gt, radius=5.0):
    """Row indices of `coords` that are surplus detections of an annotated cell.

    Each detection is charged to its *nearest* annotated cell within `radius`,
    so a detection sitting between two annotated cells is not counted twice.
    The nearest detection to each annotated cell survives; the rest are the
    duplicates.
    """
    det_um = coords[:, 1:].astype(float) * SCALE
    det_t = coords[:, 0].astype(int)
    gt_um = gt[["z", "y", "x"]].values * SCALE
    gt_t = gt["t"].values.astype(int)

    drop: list[int] = []
    onto = np.full(len(coords), -1, np.int64)
    for t in np.unique(gt_t):
        gi = np.flatnonzero(gt_t == t)
        di = np.flatnonzero(det_t == t)
        if len(di) == 0:
            continue
        # nearest annotated cell for every detection, and its distance
        dist, which = cKDTree(gt_um[gi]).query(det_um[di], k=1)
        near = dist <= radius
        if not near.any():
            continue
        for k in range(len(gi)):
            mine = np.flatnonzero(near & (which == k))
            if len(mine) < 2:
                continue
            keep = mine[np.argmin(dist[mine])]
            for m in mine:
                if m != keep:
                    drop.append(int(di[m]))
                    onto[int(di[m])] = int(di[keep])
    return np.array(sorted(set(drop)), dtype=int), onto


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preds_dir", required=True)
    p.add_argument("--nodes_dir", default="../../input/processed/nodes")
    p.add_argument("--out_dir", default=None, help="default <preds_dir>_oracledd")
    p.add_argument("--radius", type=float, default=5.0)
    p.add_argument("--merge", action="store_true",
                   help="re-point the duplicate's links onto the survivor instead "
                        "of discarding them (see cell_merging_cls.merge)")
    args = p.parse_args()

    preds = Path(args.preds_dir)
    suffix = "_oraclemg" if args.merge else "_oracledd"
    out = Path(args.out_dir) if args.out_dir else preds.with_name(preds.name + suffix)
    out.mkdir(parents=True, exist_ok=True)
    nodes = Path(args.nodes_dir)

    n_drop = n_nodes = n_stacks = 0
    for f in sorted(preds.glob("*.npz")):
        with np.load(f, allow_pickle=True) as d:
            coords, edges = d["coords"], d["edges"]
            cand = d["candidates"] if "candidates" in d.files else np.empty((0, 5))
        gt_path = nodes / f"{f.stem}.csv"
        if gt_path.exists():
            drop, onto = duplicates_for_stack(coords, pd.read_csv(gt_path), args.radius)
        else:
            drop, onto = np.empty(0, int), np.full(len(coords), -1, np.int64)
        # The graph surgery is `cell_merging_cls.merge`, shared with the model
        # pipeline so the oracle and the classifier are measured through
        # exactly the same code path.
        c2, e2, k2 = apply_merges(coords, edges, cand, drop,
                                  onto=onto if args.merge else None)
        np.savez_compressed(out / f.name, coords=c2.astype(np.int16),
                            edges=e2.astype(np.float64), candidates=k2.astype(np.float64))
        n_drop += len(drop)
        n_nodes += len(coords)
        n_stacks += 1

    print(f"{n_stacks} stacks -> {out}")
    print(f"removed {n_drop} of {n_nodes} detections ({100 * n_drop / max(n_nodes, 1):.3f}%)")


if __name__ == "__main__":
    main()
