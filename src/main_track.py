"""
Training entrypoint for the 3D cell tracking model, run from src/.

Single GPU:
    python main_track.py --epochs 50 --batch_size 8

Multi-GPU (DDP):
    torchrun --standalone --nproc_per_node=2 main_track.py --epochs 50 --batch_size 8

Cross-validation fold plus a model fit on every embryo:
    python main_track.py --epochs 50 --fullfit

batch_size and num_workers are per process.
"""

import argparse
import json
import os

from params import INPUT_FOLDER, LOG_FOLDER, TRAIN_DIR
from tracking.main import k_fold
from training.distributed import (
    broadcast_object,
    cleanup_distributed,
    init_distributed,
    seed_everything,
)
from utils.logger import create_logger, prepare_log_folder


DEFAULTS = {
    # Data
    "data_dir": TRAIN_DIR,
    "window_size": 2,
    "crop_size": None,  # (16, 64, 64),  # None,  # (16, 64, 64),     # Crop
    "selected_folds": [0, 1],  # , 1],
    "fullfit": True,
    "fullfit_n_val_videos": 1,     # leaky either way, so keep the eval cheap

    # Model
    "unet_arch": "resenc",          # "temporal" (two-conv) or "resenc" (nnU-Net ResEnc)
    "unet_out_channels": 32,
    "unet_layers": (32, 64, 128),
    "unet_pool_stride": (1, 2, 2),    # pool Y/X only; full-scale Z preserved
    "unet_n_anisotropic": 2,          # downsampling steps using unet_pool_stride
    "unet_n_decode_steps": 1,         # 1 decoder stage → output at 1/4 Y/X, full-scale Z
    "unet_temporal_attention": True,  # False = encode each frame independently
    "unet_blocks_per_stage": (3, 3, 3),    # (1, 3, 4) residual depth per stage;  None = two-conv blocks
    "unet_stem": "unshuffle",         # "unshuffle" keeps every voxel, "stride" drops 3 in 4
    "unet_norm": "batch",             # "batch" / "instance" / "group"
    "unet_hires_fusion": True,       # feed the skips finer than the head's grid into the head
    "transformer_hidden": 128,
    "transformer_heads": 4,
    "transformer_blocks": 4,
    "transformer_dropout": 0.1,
    "pos_feat_dim": 32,  # 32
    "edge_geom_feats": True,
    "edge_drift_correction": True,
    "edge_knn_k": None,  # 16,   # score each child only against its K nearest parents (None = all-vs-all)
    "unet_features": [],  # ["detection", "contrastive"]  # Node features the edge transformer sees

    # Normalisation. (x - q[low_q]) / (q[high_q] - q[low_q]), clamped
    "normalization": {"low_q": "0.001", "high_q": "0.999", "clamp_max": 4.0},

    # Augmentation
    "augmentations": {
        # Crop
        "crop_margin": 1.5,
        "crop_anchor_p": 0.9,
        # Frame gap inside a window
        "time_stride_max": 1,
        # Flips
        "flip_p": 0.5,
        "rot90_p": 0.5,
        # Affine - continuous rotate/scale/shift, resamples the crop
        "affine_p": 0.5,
        "rot_z_deg": 0,  # in-place
        "rot_xy_deg": 0,  # out-of-plane (<5.)
        "scale_range": (0.9, 1.45),
        "anisotropic_scale": 0.0,               # per-axis on top, i.e. z-spacing miscalibration
        "shift_um": 0.0,                        # sub-voxel, decorrelates cell centres from the grid
        "affine_interpolation": "bilinear",
        "affine_padding": "zeros",
        # Brightness - strength 0 = plain scalar shift instead of the heavy field
        "brightness_p": 0.9,
        "brightness_strength": 1.0,
        # Blur
        "blur_p": 0.25,
        "blur_sigma_um": 1.0,
        # Downscale then upscale
        "lowres_p": 0.25,             
        "lowres_max_factor": 2.0,
        # Noise
        "noise_p": 0.25,
        # Background leveling
        "bg_level_p": 0.25,
        "bg_level_range": (0.03, 0.30),
        # Dimming - drags a region's cells toward the background
        "dim_p": 0.5,
        "dim_factor_range": (0.25, 0.85),
        "dim_min_peak": 0.15,
        "dim_radius_um": (25.0, 70.0),
    },
    "mixup": {"p": 0.0, "alpha": 4.0},

    # Detection
    "detect_and_match_threshold": 0.2,
    "nms": "greedy",
    "pool_kernel_um": 5.0,  # for maxpool NMS
    "nms_radius_um": 7.0,   # for greedy NMS
    "max_dets_per_sample": 768,   # cap nodes per sample (bounds the N^2 edge matrix)
    "top_k_nodes": None,  # 64,            # min nodes/sample: below this, ignore the threshold & take the top-k peaks
    "offset_head": False,
    "offset_radius": 2,       # grid voxels a cell's vote reaches

    # Stage 1 - Synthetic data pretraining
    "synthetic_pretraining": {
        "enabled": True,
        "dir": "../input/synthetic/v4/",  # "../input/synth_out_3/biohub_synthetic",
        "proportion": 1,
        "weights": None,
        # Data
        "mosaic_p": 0.,
        "mosaic_jitter": 0.2,  # +/- fraction of img width/height the join wanders off centre, 0 = equal quarters
        "crop_size": (32, 128, 128),
        "max_nodes": 768,       # a synthetic frame holds up to ~656 cells
        # Training
        "iterations": 1000,
        "epochs": None,
        "batch_size": 16,
        "loss_config": {
            "det_use_weak_positives": False,
            "det_loss_weight": 1,
            "det_neg_weight": 1,
        },
        # Eval
        "n_val_videos": 5,
        "eval_every": 500,
    },

    # Weak positives
    "weak_positives": {
        "use": True,
        "max_nodes": 512,
        "ignore_max_nodes": 512,
        # Weak positives from image local maxima
        "threshold": 0.25,  # Multiplication factor for the image intensity peak threshold
        "nms_um": 5.0,
        "dir": None,
        # Which image-peak implementation to use.
        "strategy": "legacy",  # "v2" "legacy"  # v2 requires a lower threshold !
        "ignore_threshold": 0.25,  # v2 only - multiplier for the ignore bar
        "ignore_nms_um": 5.0,   # v2 only - exclusion radius for the ignore pass
        "ignore_floor": 0.0,    # v2 only - absolute floor on the ignore bar; not fold-symmetric
        # Weak positives from pl_folder
        "pl_folder": "../logs/2026-08-31/33/preds_pl_0.50/",  # "../logs/2026-08-28/13/preds_pl_0.40/",  # "../logs/2026-08-19/14/preds_pl/",
        "pl_ignore_folder": None,  # Expects lower threshold coords
        "ignore_min_dist_um": 7.0,  # the pl_ignore_folder path only
    },

    # Stage 1.5 - Pretraining
    "pretraining": {
        "enabled": True,
        "weights": None,  # "../logs/2026-08-31/30/fold_*/ckpt_pretrain.pt",
        "iterations": 1000,
        "epochs": None,
        "crop_size": (32, 128, 128),
        "batch_size": 16,
        "n_val_videos": 5,
        "eval_every": 500,
        "loss_config": {
            "det_use_weak_positives": True,
            "det_loss_weight": 1,
            "det_neg_weight": 0.05,
            "det_weak_weight": 0.2,        # To balance GT share of positive weight 
            "det_weak_smoothing": 0.01,      # smoothing factor for weak positives
        },
    },

    # Stage 2 - Contrastive - Built and trained only when "contrastive" is in unet_features.
    "contrastive": {
        "enabled": False,
        "weights": None,
        "epochs": 10,
        "crop_size": None,  # (16, 64, 64),
        "batch_size": 8,
        "augment": False,
        "lr": 1e-3,
        "temperature": 0.1,     # InfoNCE over cosine similarities
        "hard_negatives_k": None,  # None = every candidate
        "embed_dim": 64,        # node embedding width
        "unet_out_channels": 64,
        "unet_layers": (32, 64, 128),
    },

    # Stage 3 - Edge Transformer
    "edge": {
        "enabled": True,
        "transformer_only": True,
        "det_warmup_prop": 0.,
        "teacher_forcing_prop": 0,   # use GT points as nodes for this opening fraction of training
        "weights": None,
        "iterations": 200,
        "epochs": None,
        "batch_size": 16,
        "val_batch_size": 4,
        "eval_every": 200,
        "n_val_videos": 5,
        "crop_size": None,
    },

    "optimizer_config": {
        "name": "AdamW",
        "lr": 1e-3,
        "lr_unet": 1e-3,          # backbone: lower, it is pretrained by step 1
        "lr_transformer": 1e-3,   # edge head: still at init, needs the full lr
        "warmup_prop": 0.,
        "betas": (0.9, 0.999),
        "max_grad_norm": 1.0,
        "weight_decay": 0.  # 1e-2,
    },

    # Losses
    "loss_config": {
        "det_use_weak_positives": True,
        "det_loss_weight": 1,
        "det_neg_weight": 1,
        "det_weak_weight": 0.2,        # To balance GT share of positive weight 
        "det_weak_smoothing": 0.01,      # smoothing factor for weak positives
        "det_gaussian_sigma": 1.0,
        "det_ignore_radius": 0.0,      # Ignore radius around ignored_peaks
        "offset_loss_weight": 1.0,     # Offset loss weight
        "offset_beta": 1.0,            # SmoothL1 knee, grid voxels
    },

    # Eval
    "eval_thresholds": [0.2, 0.3, 0.4, 0.5],
    "eval_node_match_um": 3.0,

    # Multi-GPU
    "data_parallel": True,   # use nn.DataParallel on the UNet when >1 GPU visible
    "sync_bn": False,        # SyncBatchNorm under DDP
    "num_workers": 8,
    "fp16": True,

    # Logging & evaluation
    "log_path": LOG_FOLDER,
    "log_every": 100,
    "save_every": 500,

    # Distributed
    "seed": 0,
    "dist_backend": None,
}


def parse_config(defaults):
    def parse(value):
        return {**defaults, **json.loads(value)}
    return parse


def parse_shape(value):
    if value is None or str(value).strip().lower() in ("none", ""):
        return None
    return tuple(int(x) for x in str(value).strip("()[] ").split(","))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, value in DEFAULTS.items():
        if isinstance(value, bool):
            parser.add_argument(
                f"--{name}", action=argparse.BooleanOptionalAction, default=value
            )
        elif isinstance(value, dict):
            parser.add_argument(f"--{name}", type=parse_config(value), default=dict(value))
        elif isinstance(value, (tuple, list)) and any(isinstance(v, str) for v in value):
            # e.g. --unet_features '["detection", "contrastive"]'. type=list would
            # split the string into characters instead.
            parser.add_argument(f"--{name}", type=json.loads, default=list(value))
        elif isinstance(value, (tuple, list)) and all(isinstance(v, int) for v in (value or [])):
            parser.add_argument(f"--{name}", type=parse_shape, default=tuple(value) if value else None)
        elif value is None:
            parser.add_argument(f"--{name}", default=None)
        else:
            parser.add_argument(f"--{name}", type=type(value), default=value)
    return parser.parse_args()


def main():
    args = parse_args()

    ctx = init_distributed(args.dist_backend)
    seed_everything(getattr(args, "seed", 0), ctx.rank)

    try:
        log_folder = None
        if ctx.is_main:
            os.makedirs(args.log_path, exist_ok=True)
            log_folder = prepare_log_folder(args.log_path)
        args.output_dir = broadcast_object(log_folder, device=ctx.device)

        if ctx.is_main:
            print("-> Logging results to", args.output_dir, "\n")
            with open(args.output_dir + "args.json", "w") as f:
                json.dump(vars(args), f, indent=4, default=str)
            create_logger(directory=args.output_dir, name="logs.txt")
            if ctx.distributed:
                print(
                    f"- Distributed: {ctx.world_size} processes, "
                    f"batch_size={args.edge['batch_size']} each (stage 3)"
                )
            use_fp16 = args.fp16 and ctx.device.type == "cuda"
            print(f"- Precision: {'FP16 mixed precision' if use_fp16 else 'FP32'}")

        k_fold(args, ctx)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
