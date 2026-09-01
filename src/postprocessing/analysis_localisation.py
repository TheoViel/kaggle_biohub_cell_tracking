"""How far the detections sit from the cells they are supposed to be, and which way.

Two different failures hide in one number. A detection can be *imprecise* — off
the annotated centre by a random amount — or *biased*, off in the same direction
every time. The first is a detector-quality problem; the second is a bug or a
convention mismatch somewhere in the target construction, and is correctable by
measuring it once on the training split and subtracting.

Both matter for the score for the same reason: cells here sit about 7 um apart
and the metric matches within 7 um, so a detection several microns off centre
can be matched to a ground-truth cell it is not actually tracking.

Run:  python src/postprocessing/analysis_localisation.py [--jobs 24]
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
import tracksdata as td  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from fast_score import GroundTruth, match_nodes  # noqa: E402
from params import SCALE, TRAIN_DIR  # noqa: E402
from postprocessing.pipeline import postprocess  # noqa: E402

PRED_DIR = SRC.parent / "logs" / "2026-08-20" / "20" / "preds"
SCALE_ARR = np.asarray(SCALE)


def work(name):
    data = np.load(PRED_DIR / f"{name}.npz", allow_pickle=True)
    coords = data["coords"].astype(np.int64)
    edges = data["edges"].astype(np.float64).reshape(-1, 4)

    result = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{name}.geff")
    gt = GroundTruth(result[0] if isinstance(result, tuple) else result)

    match = match_nodes(coords, gt)
    gt_index = {int(nid): i for i, nid in enumerate(gt.ids)}

    offsets, crowd = [], []
    pos_um = coords[:, 1:].astype(float) * SCALE_ARR
    for pred_row, gt_id in match.items():
        g = gt.xyz[gt_index[int(gt_id)]]
        offsets.append(pos_um[pred_row] - g)

    # How many detections sit inside the matching radius of each annotated cell.
    for t, gt_rows in gt.by_t.items():
        pred_rows = np.flatnonzero(coords[:, 0] == t)
        if not len(pred_rows):
            continue
        tree = cKDTree(pos_um[pred_rows])
        for r in gt_rows:
            crowd.append(len(tree.query_ball_point(gt.xyz[r], 7.0)))

    # Do badly localised endpoints lose their link? Uses the pipeline's own links.
    c2, e2 = postprocess(coords, edges)
    m2 = match_nodes(c2, gt)
    err = {}
    p2 = c2[:, 1:].astype(float) * SCALE_ARR
    for pred_row, gt_id in m2.items():
        err[int(gt_id)] = float(np.linalg.norm(
            p2[pred_row] - gt.xyz[gt_index[int(gt_id)]]))
    linked = np.asarray(e2, dtype=float).reshape(-1, 4)
    # Which ground-truth links the pipeline reproduced, in ground-truth ids.
    reproduced = {(m2.get(int(s)), m2.get(int(t)))
                  for s, t in linked[:, :2].astype(int)}
    tp_err, fn_err = [], []
    for a, b in gt.pairs:
        ea, eb = err.get(int(a)), err.get(int(b))
        if ea is None or eb is None:
            continue                      # an endpoint was never detected
        target = tp_err if (int(a), int(b)) in reproduced else fn_err
        target.append(max(ea, eb))

    return name, (np.asarray(offsets), np.asarray(crowd, float),
                  np.asarray(tp_err), np.asarray(fn_err))


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

    offsets = np.concatenate([r[0] for r in results.values() if len(r[0])])
    crowd = np.concatenate([r[1] for r in results.values() if len(r[1])])
    tp_err = np.concatenate([r[2] for r in results.values() if len(r[2])])
    fn_err = np.concatenate([r[3] for r in results.values() if len(r[3])])

    dist = np.linalg.norm(offsets, axis=1)
    print(f"--- distance from a matched detection to its annotated cell "
          f"({len(dist):,} matches) ---")
    for q in (10, 25, 50, 75, 90):
        print(f"    p{q:<2d} = {np.percentile(dist, q):.2f} um")
    print(f"    within 2 um: {100 * (dist < 2).mean():.0f}%   "
          f"2-4 um: {100 * ((dist >= 2) & (dist < 4)).mean():.0f}%   "
          f">= 4 um: {100 * (dist >= 4).mean():.0f}%")

    print("\n--- is it biased, or just noisy? (detection minus annotation, um) ---")
    for i, axis in enumerate("zyx"):
        col = offsets[:, i]
        se = col.std() / np.sqrt(len(col))
        print(f"    {axis}: mean {col.mean():+.3f}  median {np.median(col):+.3f}  "
              f"sd {col.std():.2f}   (standard error {se:.3f})")
    bias = np.linalg.norm(offsets.mean(axis=0))
    print(f"    |mean offset| = {bias:.2f} um of the {np.median(dist):.2f} um median "
          f"-> {100 * bias / np.median(dist):.0f}% of the error is a fixed shift")
    print(f"    detection grid step: {np.round(np.array([1, 4, 4]) * SCALE_ARR, 3)} um "
          f"(so rounding alone contributes about "
          f"{np.linalg.norm(np.array([1, 4, 4]) * SCALE_ARR / 2) / np.sqrt(3):.2f} um)")

    print("\n--- competition for the match ---")
    print(f"    detections within 7 um of an annotated cell: "
          f"median {np.median(crowd):.0f}, mean {crowd.mean():.2f}")
    print(f"    annotated cells with more than one candidate: "
          f"{100 * (crowd > 1).mean():.0f}%")

    print("\n--- does bad localisation lose the link? ---")
    print(f"    worst endpoint error on links we got right  : "
          f"median {np.median(tp_err):.2f} um  (n={len(tp_err):,})")
    print(f"    worst endpoint error on links we missed     : "
          f"median {np.median(fn_err):.2f} um  (n={len(fn_err):,})")


if __name__ == "__main__":
    main()
