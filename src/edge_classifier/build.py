"""Build the training table: one row per re-linked edge, over every stack.

    python edge_classifier/build.py --jobs 24

Writes output/edge_classifier/dataset.npz with the feature matrix, the labels,
and the stack each row came from — the last is what keeps the cross-validation
honest, since two links from the same video are anything but independent.
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

from edge_classifier.features import FEATURE_NAMES, link_features, link_labels  # noqa: E402
from params import TRAIN_DIR  # noqa: E402
from edge_classifier.prepare import prepare  # noqa: E402

PRED_DIR = SRC.parent / "logs" / "2026-08-20" / "20" / "preds"
OUT = SRC.parent / "output" / "edge_classifier"


def work(name):
    gt_path = Path(TRAIN_DIR) / f"{name}.geff"
    data = np.load(PRED_DIR / f"{name}.npz", allow_pickle=True)
    coords = data["coords"].astype(np.int64)
    edges = data["edges"].astype(np.float64).reshape(-1, 4)

    # The gate judges the links that survive re-linking and pruning, so that is
    # the state the features must describe.
    coords, linked, edges = prepare(coords, edges)
    if len(linked) == 0:
        return name, None

    X = link_features(coords, edges, linked)

    result = td.graph.IndexedRXGraph.from_geff(gt_path)
    gt_graph = result[0] if isinstance(result, tuple) else result
    in_play, is_tp = link_labels(coords, linked, gt_graph)

    keep = np.flatnonzero(in_play)
    return name, (X[keep], is_tp[keep].astype(np.int8), len(linked))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    names = [p.stem for p in sorted(PRED_DIR.glob("*.npz"))
             if (Path(TRAIN_DIR) / f"{p.stem}.geff").exists()][: args.limit]
    print(f"{len(names)} stacks, {len(FEATURE_NAMES)} features\n")

    X_all, y_all, group_all = [], [], []
    with Pool(args.jobs) as pool:
        for i, (name, payload) in enumerate(pool.imap_unordered(work, names)):
            if payload is None:
                print(f"[{i + 1:>3d}/{len(names)}] {name:<16s} no links")
                continue
            X, y, n_links = payload
            X_all.append(X)
            y_all.append(y)
            group_all.append(np.full(len(X), name))
            print(f"[{i + 1:>3d}/{len(names)}] {name:<16s} {n_links:>7,d} links, "
                  f"{len(X):>5,d} labelled, {int(y.sum()):>5,d} correct "
                  f"({100 * y.mean():.1f}%)", flush=True)

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    groups = np.concatenate(group_all)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "dataset.npz", X=X, y=y, groups=groups,
                        feature_names=np.array(FEATURE_NAMES))
    print(f"\n{len(X):,} labelled links over {len(set(groups))} stacks, "
          f"{100 * y.mean():.1f}% correct")
    print(f"-> wrote {OUT / 'dataset.npz'}")


if __name__ == "__main__":
    main()
