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
    # Data. Only the *static* volumes carry instance masks, so they are what a
    # flow model can train on; sequences store nodes and edges but no labels.
    "data_dir": "../input/synthetic/v4",
    # "diffusion" is the Cellpose field; "center" is the straight direction to
    # the centroid, exact for an ellipsoid and ~30x cheaper to build.
    "flow_mode": "diffusion",
    "patch_size": (16, 64, 64),  # D, H, W crop; None = full volume
    # An epoch is a number of crops, not a pass over the files: there are few
    # volumes and each one holds hundreds of cells.
    "epoch_size": 10000,
    "normalization": {"low_q": "0.001", "high_q": "0.999", "clamp_max": 4.0},
    "augmentations": {
        "flip_p": 0.5,
        "rot90_p": 0.5,
        "blur_p": 0.2,
        "blur_sigma_um": 1.0,
        "lowres_p": 0.2,
        "lowres_max_factor": 2.0,
        "brightness_p": 0.3,
        "brightness_strength": 0.1,
        "bg_level_p": 0.3,
        "bg_level_range": (0.03, 0.30),
    },
    
    # Model
    # The same backbone and the same argument names as main_track.py: the
    # tracking model is this U-Net with a detection head and a transformer on
    # top, so a geometry that works there transfers by copying the numbers, and
    # a checkpoint from here can be handed to a tracking run's
    # pretraining.weights -- the backbone is stored under "unet." either way.
    #
    # The one that differs on purpose is unet_temporal_attention: a static
    # volume is a window of one frame, so there is nothing for it to mix.
    "in_channels": 1,
    "unet_arch": "resenc",          # "temporal" (two-conv) or "resenc" (nnU-Net ResEnc)
    "unet_out_channels": 32,
    "unet_layers": (32, 64, 128),
    "unet_pool_stride": (1, 2, 2),    # pool Y/X only; full-scale Z preserved
    "unet_n_anisotropic": 2,          # downsampling steps using unet_pool_stride
    "unet_n_decode_steps": None,      # None = symmetric; the head upsamples the rest
    "unet_temporal_attention": False, # a still volume is a window of one frame
    "unet_blocks_per_stage": None,    # (1, 3, 4) residual depth per stage; None = two-conv
    "unet_stem": "unshuffle",         # "unshuffle" keeps every voxel, "stride" drops 3 in 4
    "unet_norm": "batch",             # "batch" / "instance" / "group"
    "unet_hires_fusion": False,       # feed the skips finer than the head's grid into the head
    "head_dropout": 0.0,

    # Training
    "epochs": 100,
    "batch_size": 8,
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
    "w_flow": 1.0,
    "w_cellprob": 1.0,

    # Logging & evaluation
    "log_path": LOG_FOLDER,
    "log_every": 500,
    "save_every": 1000,
    "eval_every": 1000,
    # Validation is on REAL stacks -- the model never trains on one, so this is
    # the only number that means anything. Cost is n_eval_samples *
    # eval_frames volume inferences plus their flow-following, ~2.5 s each on
    # one GPU and divided by the world size under torchrun.
    "n_eval_samples": 20,       # real stacks, split evenly between the embryos
    "eval_frames": 8,          # most densely annotated timepoints per stack
    "eval_samples": None,      # explicit names, overriding n_eval_samples
    "eval_embryos": None,      # restrict the draw, e.g. ("44b6",)
    # Scored in one pass: cell probability and flows do not depend on the
    # threshold, so the whole sweep costs one forward per frame plus one
    # reconstruction per threshold. Same idea as eval_thresholds in main_track.
    "eval_thresholds": (0.3, 0.4, 0.5, 0.6, 0.7),

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
