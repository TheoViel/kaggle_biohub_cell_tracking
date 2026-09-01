"""Inference for the 3D Cellpose segmenter, on real frames.

Real volumes are (64, 256, 256) and the backbone's stride chain divides that
exactly, so a frame goes through in one piece and there is no tiling seam to
reconcile. If one ever does not fit in memory, note that flows are a local
field -- a halo of a cell radius would do for the overlap -- but trajectories
are *not* local: a voxel near a seam can walk out of its tile. Reconstruction
would still have to run on the reassembled volume, never per tile.

The architecture is rebuilt from the ``unet_*`` entries a run recorded in its
``args.json``, the same names ``main_track.py`` uses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from model_zoo.models_flow import FLOW_HEADS, define_flow_model
from segmentation.dynamics import masks_from_flows
from tracking.data import resolve_normalization

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_FOLDER = PROJECT_ROOT / "input" / "processed" / "images"
NODES_FOLDER = PROJECT_ROOT / "input" / "processed" / "nodes"
SCALE = np.array([1.625, 0.40625, 0.40625], np.float32)


def build_model(cfg: dict, device="cuda"):
    """Rebuild the architecture a run recorded in its ``args.json``."""
    keys = (
        "unet_arch", "unet_out_channels", "unet_layers", "unet_pool_stride",
        "unet_n_anisotropic", "unet_n_decode_steps", "unet_temporal_attention",
        "unet_blocks_per_stage", "unet_stem", "unet_norm", "unet_hires_fusion",
        "in_channels",
    )
    kwargs = {k: cfg[k] for k in keys if k in cfg}
    for k in ("unet_layers", "unet_pool_stride", "unet_blocks_per_stage"):
        if kwargs.get(k) is not None:
            kwargs[k] = tuple(kwargs[k])
    kwargs["dropout"] = cfg.get("head_dropout", 0.0)
    return define_flow_model(**kwargs, heads=FLOW_HEADS).to(device)


def load_model(log_folder, checkpoint=None, device="cuda", fold=0):
    """Rebuild the model recorded in a run's ``args.json`` and load weights."""
    log_folder = Path(log_folder)
    cfg = json.load(open(log_folder / "args.json"))
    model = build_model(cfg, device)
    if checkpoint is None:
        # main_seg writes checkpoints into the log folder itself; the tracking
        # runs use a fold_N subfolder, so both layouts are accepted.
        found = sorted(log_folder.glob("ckpt_*.pt")) or sorted(
            (log_folder / f"fold_{fold}").glob("ckpt_*.pt")
        )
        if not found:
            raise FileNotFoundError(f"no ckpt_*.pt under {log_folder}")
        checkpoint = [f for f in found if f.name == "ckpt_final.pt"] or found
        checkpoint = checkpoint[-1]
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    return model.eval(), cfg


def normalize(volume: np.ndarray, normalization: dict | None = None) -> np.ndarray:
    """Quantile normalisation, the convention the model was trained under.

    Quantiles are taken per frame. Training crops are normalised by their own
    quantiles, so a per-frame reference is the closest match; the tracking
    loader's per-*video* statistics would put the model on a slightly different
    intensity scale than it ever saw.
    """
    norm = resolve_normalization(normalization)
    raw = volume.astype(np.float32)
    lo = float(np.quantile(raw, float(norm["low_q"])))
    hi = float(np.quantile(raw, float(norm["high_q"])))
    return np.clip((raw - lo) / (hi - lo + 1e-6), 0.0, float(norm["clamp_max"]))


@torch.no_grad()
def predict_volume(model, volume, normalization=None, device="cuda", amp=True,
                   already_normalized=False):
    """One volume -> ``(cellprob, flows)`` as numpy, at full resolution."""
    img = volume if already_normalized else normalize(volume, normalization)
    x = torch.from_numpy(np.ascontiguousarray(img))[None, None].to(device).float()
    with torch.amp.autocast("cuda", enabled=amp and device.startswith("cuda")):
        out = model(x)
    cellprob = torch.sigmoid(out["cellprob_logits"][0, 0].float()).cpu().numpy()
    flows = out["flows"][0].float().cpu().numpy()
    return cellprob, flows


def segment_volume(model, volume, normalization=None, device="cuda",
                   return_masks=False, **dynamics_kwargs):
    """Volume -> ``(centroids, cellprob, flows, instance_masks)``."""
    cellprob, flows = predict_volume(model, volume, normalization, device)
    instance, centroids = masks_from_flows(
        cellprob, flows, device=device, return_masks=return_masks, **dynamics_kwargs
    )
    return centroids, cellprob, flows, instance


def frame_path(sample: str, t: int) -> Path:
    return IMAGE_FOLDER / f"{sample}_{t}.npy"


def load_frame(sample: str, t: int) -> np.ndarray:
    return np.load(frame_path(sample, t))


def gt_nodes(sample: str, t: int | None = None) -> np.ndarray:
    """Annotated cell centres for a sample, as (N, 3) z/y/x voxel coordinates.

    The real ground truth is sparse -- a few percent of the cells -- so this is
    a recall reference, never a precision one.
    """
    import pandas as pd

    df = pd.read_csv(NODES_FOLDER / f"{sample}.csv")
    if t is not None:
        df = df[df["t"] == t]
    return df[["z", "y", "x"]].to_numpy(np.float32)
