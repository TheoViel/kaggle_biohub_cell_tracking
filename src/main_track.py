"""
Training entrypoint for the 3D cell tracking model, run from src/.

Single GPU:
    python main_track.py --epochs 50 --batch_size 8

Multi-GPU (DDP):
    torchrun --standalone --nproc_per_node=2 main_track.py --epochs 50 --batch_size 8

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
    "n_val_videos": 10,            # number of videos to validate on
    "downscale_factor": 1,

    # Model
    "unet_out_channels": 32,
    "unet_layers": (32, 64, 128),
    "unet_pool_stride": (1, 2, 2),   # pool Y/X only; full-scale Z preserved
    "unet_n_decode_steps": 1,         # no decoder → output at 1/8 Y/X resolution
    "transformer_hidden": 128,
    "transformer_heads": 4,
    "transformer_blocks": 4,
    "transformer_dropout": 0.1,
    "pos_feat_dim": 32,  # 32
    "edge_geom_feats": True,  # feed distance / NN-rank straight to the pair MLP
    "edge_knn_k": None,  # 16,  # 16,   # score each child only against its K nearest parents (None = all-vs-all)
    # Node features the edge transformer sees, concatenated in this order.
    # Leave "contrastive" out and its UNet is never built (stage 2 is skipped);
    # an empty list leaves the transformer with geometry alone.
    "unet_features": ["contrastive"],  # ["detection", "contrastive"]

    # Data
    "brightness_strength": 1.,   # heavy photometric aug; 0 = plain scalar shift
    "mixup": {"p": 0.0, "alpha": 4.0},
    "weak_positives": {
        "use": True,
        "threshold": 1,  # Multiplication factor for the image intensity peak threshold
        "nms_um": 5.0,
        "max_nodes": 512,
        "dir": None,
    },

    # Detection
    "detect_and_match_threshold": 0.2,
    "nms": "greedy",
    "pool_kernel_um": 5.0,  # for maxpool NMS
    "nms_radius_um": 7.0,   # for greedy NMS
    "max_dets_per_sample": 768,   # cap nodes per sample (bounds the N^2 edge matrix)
    "top_k_nodes": None,  # 64,            # min nodes/sample: below this, ignore the threshold & take the top-k peaks

    # Stage 1 - Pretraining
    "pretraining": {
        "enabled": False,
        "weights": "../logs/2026-08-19/0/fold_0/ckpt_pretrain.pt",
        "epochs": 20,
        "crop_size": (16, 64, 64),
        "batch_size": 16,
    },

    # Stage 2 - Contrastive - Built and trained only when "contrastive" is in unet_features.
    "contrastive": {
        "weights": None,  # "../logs/2026-08-19/15/fold_0/ckpt_contrastive.pt",  # load one instead of training it
        "epochs": 20,
        "crop_size": None,  # (16, 64, 64),
        "batch_size": 16,
        "augment": True,
        "lr": 1e-3,
        "temperature": 0.1,     # InfoNCE over cosine similarities
        "embed_dim": 128,        # node embedding width
        "unet_out_channels": 128,
        "unet_layers": (32, 64, 128),
    },

    # Training
    "epochs": 100,
    "batch_size": 16,
    "val_batch_size": 4,
    "num_workers": 8,
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
    "fp16": True,

    "transformer_only": True,  # Freeze the UNet + detect head and train only the edge scorer, encoding
    "det_warmup_prop": 0.,
    "teacher_forcing_prop": 0,   # use GT points as nodes for this opening fraction of training

    # Losses
    "loss_config": {
        "det_use_weak_positives": True,   # image peaks count as (down-weighted) cells
        "det_loss_weight": 1,    # 0.01,
        "det_neg_weight": 0.1,
        "det_weak_weight": 0.5,        # weight for image-derived weak positives
        "det_weak_smoothing": 0.05,    # smoothing factor for weak positives
        "det_gaussian_sigma": 1.0,
    },

    # Multi-GPU
    "data_parallel": True,   # use nn.DataParallel on the UNet when >1 GPU visible
    "sync_bn": False,        # SyncBatchNorm under DDP

    # Logging & evaluation
    "log_path": LOG_FOLDER,
    "log_every": 100,
    "save_every": 1000,
    "eval_every": 100,          # 0 = no mid-training evaluation

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
                print(f"- Distributed: {ctx.world_size} processes, batch_size={args.batch_size} each")
            use_fp16 = args.fp16 and ctx.device.type == "cuda"
            print(f"- Precision: {'FP16 mixed precision' if use_fp16 else 'FP32'}")

        k_fold(args, ctx)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
