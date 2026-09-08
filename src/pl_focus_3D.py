"""
FOCUS-3D inference over every training stack, run from src/.

Single GPU:
    python pl_focus_3D.py

Multi-GPU:
    torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=8 pl_focus_3D.py

Stacks are independent, so this is not DDP in the training sense: each rank
loads its own copy of the model and takes every world_size-th stack, with no
communication until the summary at the end. That is the whole reason it scales
linearly, and it is also why a rank that dies takes only its own stacks with it
-- rerun with --skip-existing and it picks up where it stopped.

Per stack it reports the two numbers that say whether segmentation is good
enough to track on:

    recall
        Share of the annotated ground-truth nodes that got a predicted centre
        within --max-distance um. The competition's ground truth is sparse -- a
        frame with hundreds of cells carries a handful of annotations -- so an
        unmatched prediction is not a false positive and precision is not
        reported.
    ratio
        Predicted cells over the estimated_number_of_nodes in metadata.csv,
        which is the closest thing to a count of what should be there. Above 1
        we are splitting cells or inventing them, below 1 we are merging or
        missing them. Recall stays blind to both, since it only ever looks at
        the annotated few.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from params import INPUT_FOLDER, IMG_FOLDER, NODES_FOLDER, OUTPUT_FOLDER, RESOLUTION, SCALE
from segmentation.focus3d import load_model
from segmentation.focus3d import infer_volume as focus3d_infer
from training.distributed import all_gather_lists, cleanup_distributed, init_distributed
from utils.metrics import match_frame_nodes
from utils.stitching import merge_close_cells_v2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default="../weights/focus3d/model_final_nuclei.pth",
        help="FOCUS-3D checkpoint.",
    )
    parser.add_argument(
        "--out",
        default=OUTPUT_FOLDER + "focus3d/",
        help="Where the per-stack predicted nodes are written.",
    )
    parser.add_argument("--frames", type=int, default=100, help="Frames per stack.")
    parser.add_argument(
        "--samples", type=int, default=0, help="Only the first N stacks, 0 for all."
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Leave stacks that already have a predictions file alone.",
    )
    # Segmentation
    parser.add_argument("--cell-radius", type=float, default=11.0)
    parser.add_argument("--score-thresh", type=float, default=0.7)
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    parser.add_argument("--min-edge-area", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    # Cleanup
    parser.add_argument("--min-size", type=int, default=200, help="0 to keep every cell.")
    parser.add_argument("--max-dist", type=float, default=5.0, help="Centroid merge distance, um.")
    # Scoring
    parser.add_argument("--max-distance", type=float, default=5.0, help="Match distance, um.")
    return parser.parse_args()


def cleanup_masks(masks, min_size, max_dist):
    """
    Merge cells split across patches, drop the ones too small to be a nucleus.

    Same two filters as the notebook, with the size pass done by ``bincount``
    rather than a comparison of the whole volume per label: over 19 900 frames
    the per-label loop costs more than the inference it follows.

    Args:
        masks (ndarray): 3D label volume, 0 is background.
        min_size (int): Smallest instance kept, in voxels. 0 keeps every cell.
        max_dist (float): Centroids closer than this, in microns, are one cell.

    Returns:
        tuple: The volume and its (n_cells, 3) centroids as (z, y, x).
    """
    ids = np.unique(masks)
    ids = ids[ids > 0]
    if not len(ids):
        return masks, np.empty((0, 3))

    seeds = np.round(np.asarray(ndimage.center_of_mass(masks > 0, masks, ids)))
    masks, seeds = merge_close_cells_v2(masks, seeds, max_dist=max_dist)

    if min_size:
        sizes = np.bincount(masks.ravel())
        small = np.flatnonzero(sizes < min_size)
        small = small[small > 0]
        if len(small):
            masks[np.isin(masks, small)] = 0

            ids = np.unique(masks)
            ids = ids[ids > 0]
            if not len(ids):
                return masks, np.empty((0, 3))
            seeds = np.round(np.asarray(ndimage.center_of_mass(masks > 0, masks, ids)))

    return masks, seeds


def run_stack(model, sample_name, nodes, args, infer_params, progress=False):
    """
    Segment every frame of one stack and score it against the annotations.

    Args:
        model: Model from load_model.
        sample_name (str): Stack to run.
        nodes (pd.DataFrame): The stack's annotated nodes, with t, z, y, x.
        args: Parsed command line.
        infer_params (dict): Passed to the FOCUS-3D inference.
        progress (bool, optional): Show a per-frame bar.

    Returns:
        tuple: {t: (n, 3) centres}, matched nodes, annotated nodes, cells found.
    """
    predicted_nodes, found, total, n_cells = {}, 0, 0, 0

    frames = range(args.frames)
    for t in tqdm(frames, desc=sample_name, disable=not progress, leave=False):
        img = np.load(IMG_FOLDER + f"{sample_name}_{t}.npy")

        masks = focus3d_infer(model, img, **infer_params)
        masks, seeds = cleanup_masks(masks, args.min_size, args.max_dist)

        predicted_nodes[t] = seeds
        n_cells += len(seeds)

        gt = nodes[nodes["t"] == t][["z", "y", "x"]].values
        if not len(gt):
            continue

        pairs, _, _ = match_frame_nodes(
            seeds, gt, t=t, scale=SCALE, max_distance=args.max_distance
        )
        found += len(pairs)
        total += len(gt)

    return predicted_nodes, found, total, n_cells


def main():
    args = parse_args()
    ctx = init_distributed()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(INPUT_FOLDER + "metadata.csv")
    if args.samples:
        metadata = metadata.head(args.samples)

    # Stacks are handed out round-robin rather than in contiguous blocks, so the
    # ranks stay in step even if cost drifts over the file order.
    mine = metadata.iloc[ctx.rank :: ctx.world_size]

    infer_params = {
        "z_ratio": RESOLUTION["z"] / RESOLUTION["x"],
        "cell_radius": args.cell_radius,
        "patch_size": (32, 96, 96),
        "stride": (24, 64, 64),
        "batch_size": args.batch_size,
        "score_thresh": args.score_thresh,
        "mask_thresh": args.mask_thresh,
        "min_edge_area": args.min_edge_area,
        "topk_postprocess": 500,
    }

    if ctx.is_main:
        print(
            f"FOCUS-3D over {len(metadata)} stacks x {args.frames} frames "
            f"on {ctx.world_size} process(es)",
            flush=True,
        )
        print(json.dumps({k: str(v) for k, v in infer_params.items()}), flush=True)

    model = load_model(args.weights, device=str(ctx.device))

    results = []
    for row in mine.itertuples():
        sample_name = row.sample_name
        path = out_dir / f"predicted_nodes_{sample_name}.json"
        if args.skip_existing and path.exists():
            print(f"[{sample_name}] skipped, {path.name} exists", flush=True)
            continue

        nodes = pd.read_csv(NODES_FOLDER + f"{sample_name}.csv")

        start = time.time()
        predicted_nodes, found, total, n_cells = run_stack(
            model, sample_name, nodes, args, infer_params, progress=ctx.is_main
        )
        elapsed = time.time() - start

        # An unannotated stack would make recall meaningless rather than zero.
        recall = found / total if total else float("nan")

        # The estimate covers the whole stack, so running a few frames has to be
        # compared against its share of it or every short run looks catastrophic.
        n_frames_total = int(str(row.shape).strip("()").split(",")[0])
        n_estimated = row.estimated_number_of_nodes * args.frames / n_frames_total
        ratio = n_cells / n_estimated

        print(
            f"[{sample_name}] node recall {recall:.3f} ({found}/{total})  "
            f"cells {n_cells} vs {n_estimated:.0f} estimated "
            f"(x{ratio:.2f})  {elapsed / 60:.1f} min",
            flush=True,
        )

        res = {
            "sample_name": sample_name,
            "recall": recall,
            "found": found,
            "total": total,
            "n_cells": n_cells,
            "n_estimated": n_estimated,
            "ratio": ratio,
            "nodes": {t: p.tolist() for t, p in predicted_nodes.items()}
        }
        with open(path, "w") as f:
            json.dump(res, f)
        results.append(res)

    results = all_gather_lists(results)

    if ctx.is_main and results:
        found = sum(r["found"] for r in results)
        total = sum(r["total"] for r in results)
        n_cells = sum(r["n_cells"] for r in results)
        n_estimated = sum(r["n_estimated"] for r in results)

        pd.DataFrame(results).sort_values("sample_name").to_csv(
            out_dir / "focus3d_stacks.csv", index=False
        )

        print(f"\n{len(results)} stacks", flush=True)
        print(f"- node recall: {found / max(total, 1):.3f} ({found}/{total})", flush=True)
        print(
            f"- cells: {n_cells} vs {n_estimated:.0f} estimated "
            f"(x{n_cells / max(n_estimated, 1):.2f})",
            flush=True,
        )
        print(f"- per stack: {out_dir / 'focus3d_stacks.csv'}", flush=True)

    cleanup_distributed()


if __name__ == "__main__":
    main()
