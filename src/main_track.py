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

from params import LOG_FOLDER, TRAIN_DIR
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
    "crop_size": (16, 64, 64),     # Crop
    "n_val_videos": 1,            # number of videos to validate on

    # Model
    "unet_out_channels": 32,
    "unet_layers": (32, 64, 128),
    "unet_pool_stride": (1, 2, 2),   # pool Y/X only; full-scale Z preserved
    "unet_n_decode_steps": 0,         # no decoder → output at 1/4 Y/X resolution
    "transformer_hidden": 128,
    "transformer_heads": 4,
    "transformer_blocks": 4,

    # Detection
    "pool_kernel_um": 5.0,        # NMS kernel size in microns
    "detect_and_match_threshold": 0.1,
    "max_dets_per_sample": None,

    # Training
    "epochs": 100,
    "batch_size": 4,
    "val_batch_size": 4,
    "num_workers": 8,
    "optimizer_config": {
        "name": "AdamW",
        "lr": 1e-3,
        "warmup_prop": 0.,
        "betas": (0.9, 0.999),
        "max_grad_norm": 1.0,
        "weight_decay": 0.  # 1e-2,
    },
    "fp16": True,
    "det_warmup_prop": 0.,

    # Losses
    "loss_config": {
        "det_loss_weight": 1.,
        "det_neg_weight": 0.1,
        "det_weak_weight": 0.5,        # weight for image-derived weak positives
        "det_weak_smoothing": 0.1,     # smoothing factor for weak positives
    },

    # Multi-GPU
    "data_parallel": True,   # use nn.DataParallel on the UNet when >1 GPU visible
    "sync_bn": False,        # SyncBatchNorm under DDP

    # Logging & evaluation
    "log_path": LOG_FOLDER,
    "log_every": 200,
    "save_every": 1000,
    "eval_every": 200,          # 0 = no mid-training evaluation

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
