"""Sweep the detector's operating point, then link and score each node set.

The saved run detects 38k nodes where the dataset estimates 24k. That costs the
score twice: directly, through the node-count adjustment, and indirectly,
because a spurious detection competes for the parent a real cell needed — 47 of
the 89 ground-truth cells have more than one detection within the matching
radius.

The detection map is computed once and thresholded many ways, so the sweep
costs one forward pass over the video rather than one per setting.

Run:  python src/postprocessing/redetect.py
"""

import json

import numpy as np
import torch
import zarr

from common import RESULTS, SAMPLE, load_gt, save, score, to_graph
from main_predict import _reconcile_arch
from params import SCALE, TRAIN_DIR
from relink import relink
from tracking.main import build_model
from tracking.post_process import detect_cells_pooled, pool_kernel_from_um
from tracking_cellmot.io import open_dataset
from utils.logger import Config

LOG_FOLDER = "../logs/2026-08-20/20/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = RESULTS / f"{SAMPLE}_det_probs.npy"


def detection_maps():
    """Per-frame detection probability on the encoder grid, cached to disk."""
    if CACHE.exists():
        return np.load(CACHE), (1, 4, 4)

    cfg = Config(json.load(open(LOG_FOLDER + "args.json")))
    state = torch.load(LOG_FOLDER + "fold_0/ckpt_final.pt", map_location="cpu")
    state = state.get("model_state_dict", state)
    state = {k.replace("unet.module.", "unet.", 1): v for k, v in state.items()}
    cfg = _reconcile_arch(cfg, state)
    model = build_model(cfg, None)
    model.load_state_dict(state, strict=True)
    model.eval().to(DEVICE)

    ds = open_dataset(f"{TRAIN_DIR}/{SAMPLE}.zarr", normalize=False, load_image=False)
    arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low, q_high = float(ds.quantiles["0.001"]), float(ds.quantiles["0.999"])
    T = ds.image_shape[0]

    maps = []
    with torch.no_grad():
        # Windows of two frames, matching how the model was trained and run.
        for t in range(0, T, 2):
            idx = [t, min(t + 1, T - 1)]
            imgs = torch.from_numpy(np.stack([arr[i] for i in idx]).astype(np.float32))
            imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0, 4.0)
            imgs = imgs.unsqueeze(0).to(DEVICE)
            _, det = model.encode(imgs)
            for k, i in enumerate(idx):
                if i == len(maps):
                    maps.append(det[k][0, 0].sigmoid().cpu().numpy().astype(np.float16))
    maps = np.stack(maps)
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.save(CACHE, maps)
    return maps, (1, 4, 4)


def detect(maps, scaling, threshold, nms_radius_um):
    voxel_out = tuple(s * u for s, u in zip(SCALE, scaling))
    pool_k = pool_kernel_from_um(5.0, voxel_out)
    out = []
    for t in range(len(maps)):
        arr = detect_cells_pooled(
            torch.from_numpy(maps[t].astype(np.float32)),
            t, threshold, nms="greedy", pool_kernel=pool_k,
            voxel_size=voxel_out, nms_radius_um=nms_radius_um,
        )
        out.append(arr)
    coords = np.concatenate(out).astype(np.float64)
    coords[:, 1:] *= np.asarray(scaling, dtype=np.float64)
    return coords.astype(np.int64)


def main():
    gt_graph, n_total = load_gt()
    maps, scaling = detection_maps()
    print(f"detection maps {maps.shape}, grid scaling {scaling}\n")

    rows = []
    for threshold in (0.20, 0.30, 0.40, 0.50, 0.60):
        for radius in (7.0, 8.0, 9.0):
            coords = detect(maps, scaling, threshold, radius)
            edges = relink(coords, np.empty((0, 4)), max_distance=8.0, passes=4)
            row = score(to_graph(coords, edges[:, :4]), gt_graph, n_total,
                        f"det>={threshold:.2f} nms={radius:.0f}um")
            row["threshold"], row["nms_radius"] = threshold, radius
            rows.append(row)
        print()

    save(rows, "redetect")
    best = max(rows, key=lambda r: r["adj_edge_jaccard"])
    print(f"\nbest: {best['label']}  adjJ={best['adj_edge_jaccard']:.4f}")


if __name__ == "__main__":
    main()
