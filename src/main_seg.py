"""
Training entrypoint for the 3D cell segmentation model, run from src/.

Single GPU:
    python main_seg.py --epochs 10 --batch_size 8

Multi-GPU, one process per GPU:
    torchrun --standalone --nproc_per_node=2 main_seg.py --epochs 10 --batch_size 4

batch_size and num_workers are per process, so the effective batch size is
batch_size * nproc_per_node. Results are logged to logs/<date>/<exp_id>/, which
rank 0 creates and shares with the other ranks.
"""

import argparse
import json
import os

from params import LOG_FOLDER, OUTPUT_FOLDER
from training.distributed import broadcast_object, init_distributed
from segmentation.train import train
from utils.logger import create_logger, prepare_log_folder


DEFAULTS = {
    # Data
    "data_dir": OUTPUT_FOLDER + "segs_3/",
    "use_cellpose": True,
    "sigma": 3.0,
    "patch_size": (64, 64, 64),  # D, H, W crop; None = full volume
    

    # Model
    "in_channels": 1,
    "decoder_name": "Unet",
    "encoder_name": "resnet18",
    "pretrained": True,

    # Augmentations
    # Only p, alpha and use_3d matter here: the batch-dict path ignores the class args
    "mixup": {
        "p": 0.0,
        "alpha": 0.4,
        "use_3d": True,
    },
    "cutmix": {
        "p": 0.0,
        "alpha": 0.4,
        "use_3d": True,
    },

    # Training
    "epochs": 10,
    "batch_size": 4,
    "num_workers": 8,
    "optimizer_config": {
        "name": "AdamW",
        "lr": 2e-3,
        "warmup_prop": 0.1,
        "betas": (0.9, 0.999),
        "max_grad_norm": 1.0,
        "weight_decay": 0.1,
    },
    "fp16": True,

    # Losses
    "w_fg": 1.0,
    "w_center": 1.0,
    "w_offset": 1.0,

    # Logging & evaluation
    "log_path": LOG_FOLDER,
    "log_every": 50,
    "save_every": 1000,
    "eval_every": 50,
    "eval_frames": 1,
    "eval_sample_name": "6bba_09961292",

    # Distributed
    "seed": 0,
    "sync_bn": True,
    "dist_backend": None,
}


def parse_config(defaults):
    """Build a parser for a JSON flag, so that overrides can be partial."""

    def parse(value):
        return {**defaults, **json.loads(value)}

    return parse


def parse_shape(value):
    """Parse a "D,H,W" flag, "none" meaning full volumes."""
    if value.strip().lower() in ("none", ""):
        return None
    return tuple(int(x) for x in value.strip("()[] ").split(","))


def parse_args():
    """Expose every entry of DEFAULTS as a command line flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name, value in DEFAULTS.items():
        if isinstance(value, bool):
            parser.add_argument(
                f"--{name}", action=argparse.BooleanOptionalAction, default=value
            )
        elif isinstance(value, dict):
            parser.add_argument(f"--{name}", type=parse_config(value), default=dict(value))
        elif isinstance(value, (tuple, list)):
            parser.add_argument(f"--{name}", type=parse_shape, default=tuple(value))
        elif value is None:
            parser.add_argument(f"--{name}", default=None)
        else:
            parser.add_argument(f"--{name}", type=type(value), default=value)
    return parser.parse_args()


def main():
    args = parse_args()

    ctx = init_distributed(args.dist_backend)

    log_folder = None
    if ctx.is_main:
        os.makedirs(args.log_path, exist_ok=True)
        log_folder = prepare_log_folder(args.log_path)
    args.output_dir = broadcast_object(log_folder, device=ctx.device)

    if ctx.is_main:
        print("-> Logging results to ", args.output_dir, "\n")
        with open(args.output_dir + "args.json", "w") as f:
            json.dump(vars(args), f, indent=4)
        create_logger(directory=args.output_dir, name="logs.txt")

    train(args)


if __name__ == "__main__":
    main()
