"""Is the systematic offset a grid-centring bug, and does fixing it help?

The detection grid is 4x coarser than the image in y and x. Training assigns a
ground-truth cell to a grid cell with ``floor(gt / scaling)``
(tracking/targets.py), and inference maps the grid back with a plain
``coords *= unet_scaling`` (tracking/predict.py). So a grid cell spanning image
voxels [4g, 4g+3] is reported at 4g — its corner — when the cell it represents
is on average at 4g + 1.5, its centre.

That predicts a fixed offset of -(scaling - 1) / 2 voxels per axis: nothing in z
(scaling 1), and -1.5 voxels = -0.61 um in y and x. This checks the prediction
against the measured offset, and then scores the correction.

A uniform shift of every node cannot change the re-linking — the relative
geometry is untouched — so the only thing it moves is which detection each
annotated cell matches to. That makes it a clean experiment.

Run:  python src/postprocessing/check_centring.py [--jobs 24]
"""

import argparse
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "postprocessing"))
os.chdir(SRC)

from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tracksdata as td  # noqa: E402

from fast_score import GroundTruth, match_nodes, metrics, roll_up  # noqa: E402
from params import SCALE, TRAIN_DIR  # noqa: E402
from postprocessing.pipeline import postprocess  # noqa: E402

PRED_DIR = SRC.parent / "logs" / "2026-08-20" / "20" / "preds"
SCALE_ARR = np.asarray(SCALE)
UNET_SCALING = np.array([1.0, 4.0, 4.0])

# What the floor-vs-centre mismatch predicts, in image voxels.
PREDICTED_SHIFT = (UNET_SCALING - 1.0) / 2.0        # [0, 1.5, 1.5]

# Shifts to try, in image voxels (z, y, x).
CANDIDATES = {
    "none": np.zeros(3),
    "predicted (grid centre)": PREDICTED_SHIFT,
    "half of predicted": PREDICTED_SHIFT / 2,
    "predicted + z": PREDICTED_SHIFT + np.array([0.14, 0.0, 0.0]),
}

_META = None


def offsets_for(coords, gt):
    """Detection-minus-annotation vectors, in um, for the matched nodes."""
    match = match_nodes(coords, gt)
    index = {int(nid): i for i, nid in enumerate(gt.ids)}
    pos_um = coords[:, 1:].astype(float) * SCALE_ARR
    return np.array([pos_um[row] - gt.xyz[index[int(g)]] for row, g in match.items()])


def work(name):
    global _META
    if _META is None:
        _META = pd.read_csv("../input/metadata.csv").set_index("sample_name")
    n_total = float(_META.loc[name, "estimated_number_of_nodes"])

    data = np.load(PRED_DIR / f"{name}.npz", allow_pickle=True)
    coords = data["coords"].astype(np.float64)
    edges = data["edges"].astype(np.float64).reshape(-1, 4)
    result = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{name}.geff")
    gt = GroundTruth(result[0] if isinstance(result, tuple) else result)

    # The pipeline is run once: a uniform translation cannot change its output.
    pp_coords, pp_edges = postprocess(coords, edges)

    out = {}
    for label, shift in CANDIDATES.items():
        moved = coords.copy()
        moved[:, 1:] += shift
        out[f"offset::{label}"] = offsets_for(moved, gt)

        moved_pp = np.asarray(pp_coords, dtype=np.float64).copy()
        moved_pp[:, 1:] += shift
        out[f"score::{label}"] = metrics(moved_pp, pp_edges, gt, n_total)
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    names = [p.stem for p in sorted(PRED_DIR.glob("*.npz"))
             if (Path(TRAIN_DIR) / f"{p.stem}.geff").exists()][: args.limit]
    print(f"{len(names)} stacks\n")

    with Pool(args.jobs) as pool:
        results = dict(pool.imap_unordered(work, names))

    print("--- the prediction ---")
    print(f"    grid scaling {UNET_SCALING.astype(int)}  ->  a cell is reported at its "
          f"corner, {PREDICTED_SHIFT} voxels short of its centre")
    print(f"    which in microns is {np.round(PREDICTED_SHIFT * SCALE_ARR, 3)}\n")

    print("--- measured offset (detection minus annotation, um) ---")
    print(f"    {'shift applied':<26s} {'z':>8s} {'y':>8s} {'x':>8s} "
          f"{'|mean|':>8s} {'median dist':>12s}")
    for label in CANDIDATES:
        off = np.concatenate([r[f"offset::{label}"] for r in results.values()
                              if len(r[f"offset::{label}"])])
        mean = off.mean(axis=0)
        dist = np.linalg.norm(off, axis=1)
        print(f"    {label:<26s} {mean[0]:>+8.3f} {mean[1]:>+8.3f} {mean[2]:>+8.3f} "
              f"{np.linalg.norm(mean):>8.3f} {np.median(dist):>12.3f}")

    print("\n--- does correcting it move the score? ---")
    print(f"    {'shift applied':<26s} {'adjJ':>8s} {'J':>8s} {'tp':>7s} "
          f"{'fp':>6s} {'fn':>6s} {'recall':>8s}")
    for label in CANDIDATES:
        rows = [r[f"score::{label}"] for r in results.values()]
        s = roll_up(rows)
        print(f"    {label:<26s} {s['adj_edge_jaccard']:>8.4f} {s['edge_jaccard']:>8.4f} "
              f"{s['edge_tp']:>7d} {s['edge_fp']:>6d} {s['edge_fn']:>6d} "
              f"{s['node_recall']:>8.4f}")


if __name__ == "__main__":
    main()
