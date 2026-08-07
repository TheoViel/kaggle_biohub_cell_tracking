#!/usr/bin/env python
"""Score :func:`~tracking_cellmot.segmentation.segment_cells_3d` against ground truth.

The competition's ground truth is *sparse*: a frame with hundreds of visible
cells may carry only a handful of annotated nodes. Recall is therefore
meaningful but precision is not — an unannotated detection is not necessarily a
false positive — so this script deliberately reports recall only.

Predicted seeds are matched to annotated ground-truth nodes with
:class:`~tracksdata.metrics.DistanceMatching` using optimal one-to-one assignment
on centroid distance in physical space (default ``--max-distance 7`` µm).
Unmatched predictions are ignored.

Metrics reported per frame:

``recall``
    Fraction of annotated ground-truth nodes that received a predicted match
    within ``--max-distance``.
``cover``
    Among *matched* ground-truth nodes, the fraction whose own voxel lies
    inside the specifically paired predicted instance (not just any instance).
``err``
    Median centroid distance in physical units over matched pairs only.
``merged``
    Among matched ground-truth nodes, how many share a predicted instance with
    another matched annotation (0 means no merges among annotated cells).
``vol``
    Median volume of *matched* instances as a multiple of an ideal sphere of
    ``--cell-diameter``.

Usage:
    python scripts/eval_segmentation.py --data-dir input/train --frames 0 30 60
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import tracksdata as td
import zarr
from tracksdata.metrics import DistanceMatching

from tracking_cellmot.io import list_datasets, open_dataset
from tracking_cellmot.segmentation import segment_cells_3d

from dataspec import DATASET_PATH


def ground_truth_centers(geff_path: Path, t: int) -> np.ndarray:
    """Return the ``(n, 3)`` array of annotated ``(z, y, x)`` centers at time ``t``."""
    result = td.graph.IndexedRXGraph.from_geff(str(geff_path))
    graph = result[0] if isinstance(result, tuple) else result
    nodes = graph.node_attrs()
    return nodes.filter(nodes["t"] == t).select(["z", "y", "x"]).to_numpy().astype(float)


def _build_frame_graph(centers: np.ndarray, t: int) -> td.graph.InMemoryGraph:
    """Build a single-frame node graph from ``(n, 3)`` ``(z, y, x)`` centers."""
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    if len(centers):
        graph.bulk_add_nodes([
            {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
            for z, y, x in centers
        ])
    return graph


def match_frame_nodes(
    seeds: np.ndarray,
    truth: np.ndarray,
    *,
    t: int,
    scale: tuple[float, float, float],
    max_distance: float,
) -> pl.DataFrame:
    """Match predicted seeds to sparse GT nodes and return paired attributes.

    Returns a DataFrame with one row per matched pair and columns:
    ``pred_idx``, ``gt_z``, ``gt_y``, ``gt_x``, ``distance``.
    """
    pred_graph = _build_frame_graph(seeds.astype(float), t)
    gt_graph = _build_frame_graph(truth, t)
    if pred_graph.num_nodes() == 0 or gt_graph.num_nodes() == 0:
        return pl.DataFrame(
            schema={
                "pred_idx": pl.Int64,
                "gt_z": pl.Float64,
                "gt_y": pl.Float64,
                "gt_x": pl.Float64,
                "distance": pl.Float64,
            }
        )

    # print(pred_graph.num_nodes(), gt_graph.num_nodes())

    matching = DistanceMatching(max_distance=max_distance, scale=scale)
    from tracksdata.options import get_options, set_options

    # Node-only graphs have no edges, so tracksdata's "No matching edges found"
    # warning is expected noise here.
    td_log = logging.getLogger("tracksdata.utils._logging")
    prev_level = td_log.level
    prev_show_progress = get_options().show_progress
    set_options(show_progress=False)
    td_log.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            from scipy.sparse import SparseEfficiencyWarning

            warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
            pred_graph.match(gt_graph, matching=matching)
    finally:
        td_log.setLevel(prev_level)
        set_options(show_progress=prev_show_progress)

    pred_attrs = pred_graph.node_attrs(
        attr_keys=[
            td.DEFAULT_ATTR_KEYS.NODE_ID,
            td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID,
            td.DEFAULT_ATTR_KEYS.MATCH_SCORE,
            "z",
            "y",
            "x",
        ]
    )
    gt_attrs = gt_graph.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "z", "y", "x"]
    ).rename(
        {
            "z": "gt_z",
            "y": "gt_y",
            "x": "gt_x",
            td.DEFAULT_ATTR_KEYS.NODE_ID: "gt_node_id",
        }
    )

    matched = pred_attrs.filter(
        pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID).is_not_null()
        & (pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID) != -1)
    ).with_columns(
        pl.col(td.DEFAULT_ATTR_KEYS.NODE_ID).alias("pred_idx"),
    ).join(
        gt_attrs,
        left_on=td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID,
        right_on="gt_node_id",
        how="inner",
    )

    scale_arr = np.asarray(scale, dtype=float)
    if len(matched) == 0:
        return pl.DataFrame(
            schema={
                "pred_idx": pl.Int64,
                "gt_z": pl.Float64,
                "gt_y": pl.Float64,
                "gt_x": pl.Float64,
                "distance": pl.Float64,
            }
        )

    pred_xyz = matched.select(["z", "y", "x"]).to_numpy() * scale_arr
    gt_xyz = matched.select(["gt_z", "gt_y", "gt_x"]).to_numpy() * scale_arr
    distances = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
    return matched.with_columns(pl.Series("distance", distances))


def score_frame(
    labels: np.ndarray,
    seeds: np.ndarray,
    truth: np.ndarray,
    voxel_size: tuple[float, float, float],
    max_distance: float,
    cell_diameter: float,
    *,
    t: int = 0,
) -> dict[str, float]:
    """Compare one segmented frame against the sparse ground truth for that frame."""
    scale = tuple(float(v) for v in voxel_size)
    n_truth = len(truth)
    pairs = match_frame_nodes(seeds, truth, t=t, scale=scale, max_distance=max_distance)
    n_matched = len(pairs)

    if n_matched:
        voxel = np.clip(np.rint(pairs.select(["gt_z", "gt_y", "gt_x"]).to_numpy()).astype(int), 0, np.array(labels.shape) - 1)
        pred_labels = pairs["pred_idx"].to_numpy() + 1
        hit_labels = labels[tuple(voxel.T)]
        paired_hits = hit_labels == pred_labels
        coverage = float(np.mean(paired_hits))
        n_cover_hits = int(paired_hits.sum())
        distances = pairs["distance"].to_numpy()
        median_error = float(np.median(distances))

        positive_hits = hit_labels[hit_labels > 0]
        if len(positive_hits):
            _, counts = np.unique(positive_hits, return_counts=True)
            merged = int(counts[counts > 1].sum() - np.count_nonzero(counts > 1))
        else:
            merged = 0

        max_label = max(int(labels.max()), int(pred_labels.max()))
        sizes = np.bincount(labels.ravel(), minlength=max_label + 1)
        matched_sizes = sizes[pred_labels]
        ideal_voxels = (4.0 / 3.0) * np.pi * (cell_diameter / 2.0) ** 3 / float(np.prod(scale))
        volume_ratio = float(np.median(matched_sizes) / ideal_voxels)
    else:
        coverage = 0.0
        n_cover_hits = 0
        median_error = float("nan")
        merged = 0
        volume_ratio = 0.0
        distances = np.array([], dtype=float)

    return {
        "n_truth": n_truth,
        "n_matched": n_matched,
        "n_cover_hits": n_cover_hits,
        "n_instances": int(labels.max()),
        "recall": float(n_matched / n_truth) if n_truth else float("nan"),
        "coverage": coverage,
        "median_error": median_error,
        "merged": merged,
        "volume_ratio": volume_ratio,
        "distances": distances,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATASET_PATH, help="Directory of .zarr/.geff pairs.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset stems; default is all of them.")
    parser.add_argument("--frames", type=int, nargs="*", default=[0, 30, 60], help="Timepoints to score.")
    parser.add_argument("--limit", type=int, default=3, help="Max datasets to score when --datasets is omitted.")
    parser.add_argument("--cell-diameter", type=float, default=6.0, help="Cell diameter in physical units.")
    parser.add_argument(
        "--max-distance",
        type=float,
        default=7.0,
        help="Maximum centroid distance for one-to-one node matching, in physical units.",
    )
    parser.add_argument(
        "--seed-prominence",
        type=float,
        default=0.05,
        help="Seed prominence passed to segment_cells_3d; <=0 selects fixed-radius suppression.",
    )
    args = parser.parse_args()
    seed_prominence = args.seed_prominence if args.seed_prominence > 0 else None

    if args.datasets:
        names = list(args.datasets)
    else:
        names = [p.name for p in list_datasets(args.data_dir, require_geff=True)][: args.limit]
    if not names:
        parser.error(f"No datasets with both .zarr and .geff found under {args.data_dir}")

    header = (
        f"{'dataset':22s} {'t':>3s} {'gt':>4s} {'match':>5s} {'inst':>5s} "
        f"{'recall':>7s} {'cover':>6s} {'err':>6s} {'merged':>6s} {'vol':>5s}"
    )
    print(header)
    print("-" * len(header))

    total_truth = 0
    total_matched = 0
    matched_cover_hits = 0
    all_distances: list[float] = []

    for name in names:
        scale = open_dataset(args.data_dir / name, load_image=False).scale
        images = zarr.open_group(str(args.data_dir / f"{name}.zarr"), mode="r")["0"]

        for t in args.frames:
            if t >= images.shape[0]:
                continue
            truth = ground_truth_centers(args.data_dir / f"{name}.geff", t)
            if len(truth) == 0:
                continue  # nothing annotated here, so recall is undefined

            frame = np.asarray(images[t])
            labels, seeds = segment_cells_3d(
                frame,
                voxel_size=scale,
                cell_diameter=args.cell_diameter,
                seed_prominence=seed_prominence,
            )
            row = score_frame(
                labels,
                seeds,
                truth,
                scale,
                args.max_distance,
                args.cell_diameter,
                t=t,
            )

            total_truth += row["n_truth"]
            total_matched += row["n_matched"]
            matched_cover_hits += row["n_cover_hits"]
            all_distances.extend(row["distances"].tolist())

            print(
                f"{name:22s} {t:3d} {row['n_truth']:4d} {row['n_matched']:5d} {row['n_instances']:5d} "
                f"{row['recall']:7.2f} {row['coverage']:6.2f} "
                f"{row['median_error']:6.2f} {row['merged']:6d} {row['volume_ratio']:4.1f}x"
            )

    if total_truth:
        pooled_recall = total_matched / total_truth
        pooled_coverage = matched_cover_hits / total_matched if total_matched else float("nan")
        pooled_error = float(np.median(all_distances)) if all_distances else float("nan")
        print("-" * len(header))
        print(
            f"{'TOTAL matched/gt':22s} {'':3s} {total_truth:4d} {total_matched:5d} {'':5s} "
            f"{pooled_recall:7.2f} {pooled_coverage:6.2f} {pooled_error:6.2f}"
        )


if __name__ == "__main__":
    main()
