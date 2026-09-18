"""
Training entrypoint for NETr, the contrastive node-embedding model, run from src/.

Single GPU:
    python main_contrastive.py --iterations 2000

Multi-GPU (DDP):
    torchrun --standalone --nproc_per_node=2 main_contrastive.py --iterations 2000

``iterations`` is the fold-comparable step budget (folds see different
numbers of frames, so a fixed epoch count hands them unequal training); set
it and ``epochs`` falls back to unused. Leave it unset to train by ``epochs``.

An implementation of ASCENT (Han, Pritchard & Lu, bioRxiv 2025,
10.1101/2025.07.23.666425): one embedding per detected cell. In the paper's own
setting (``real_prop=0``) there is no annotation at all -- positives are two
augmented views of the same frame, so the supervision comes entirely from the
augmentation pipeline, and the embeddings are only ever asked to be invariant
to transforms rather than to a cell moving between frames.

``real_prop`` mixes in pairs of *consecutive* frames matched by the ground-truth
graph, which is the task at inference. The annotation is thin -- ~1 cell a frame
-- so only the annotated rows are scored and every other detection serves as a
negative; see ``embeddings/dataset.py``.

The key argument is ``coords_log_folder``: a predictions folder written by the
tracking pipeline, holding ``<sample>.npz`` with a ``coords`` array of
``(t, z, y, x)`` rows. Those detections define the crops. Training on the
pipeline's own nodes rather than on ground truth is deliberate -- these are the
nodes the matcher will be asked to link, spurious ones included, and the real
ground truth annotates only ~3% of cells anyway.

batch_size and num_workers are per process; batch size counts *frames*, since
NT-Xent needs every cell of a frame at once to have negatives.
"""

import argparse
import json
import os

from embeddings.main import k_fold
from params import LOG_FOLDER, TRAIN_DIR
from training.distributed import (
    broadcast_object,
    cleanup_distributed,
    init_distributed,
    seed_everything,
)
from utils.logger import create_logger, prepare_log_folder


DEFAULTS = {
    # Global
    # "coords_log_folder": "../logs/2026-09-07/4/preds_r6_0.50/",
    # "coords_log_folder": "../logs/2026-09-09/4/preds_r6_0.40/",
    "coords_log_folder": "../logs/2026-09-13/1/preds_r6_0.40/",
    "data_dir": TRAIN_DIR,
    "selected_folds": [0, 1],
    "fullfit": True,

    # Data
    "max_nodes": 256,  # 256
    "min_nodes": 8,
    "max_frames_per_video": None,   # cap for debugging
    "patch_zyx": (3, 48, 48),  # (channels, height, width)
    "n_frames": 1,
    "real_prop": 0.5,  # Share of samples served as a *real* (t, t+1)

    # Augmentations
    "augmentations": {
        "jitter_um": 3.0,
        "rot_p": 0.75,
        "rot_deg": 45.0,
        "elastic_p": 0.75,
        "elastic_um": 1.0,
        "elastic_sigma_um": 5.0,
        "scale_p": 0.75,
        "scale_range": (0.85, 1.18),
        "intensity_p": 0.8,
        "brightness": 0.2,
        "contrast": 0.2,
        "gamma": 0.2,
        "noise_std": 0.,
        "dropout_p": 0.,
    },

    # Model
    # "backbone": "vit_small_patch16_224.dino",
    # vit_base, not vit_small: every contrastive run since 2026-09-08 has
    # used it, including `2026-09-09/1` whose embeddings the best edge chain
    # reads (`embeddings_contrastive_2026-09-09_1`). The small default was
    # left over from the first run.
    "backbone": "vit_base_patch16_224.dino",
    # "backbone": "vit_small_patch8_224.dino",
    # "backbone": "xcit_tiny_12_p8_224.fb_dist_in1k",
    "pretrained": True,
    "frame_aggregation": "mean",   # "mean" (paper default) | "bigru"
    "channels_last": True,         # layout only; big for the ConvNeXt backbone
    "min_tokens": 3,
    # Recompute the backbone's activations in the backward rather than storing
    # them. The patch encoder runs ~25k times a step, so it holds essentially
    # all the memory: 71 GB -> 13 GB at full frames, for ~27% more time. Turn
    # off if you have headroom, or raise batch_size/max_nodes with it on.
    "grad_checkpointing": True,
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 4,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "embed_dim": None,  # None = d_model
    "use_visual": True,
    "use_position": False,
    "use_transformer": True,
    "weights": None,  # resume

    # Training. The paper: Adam, 1e-4, 4 frames, 20 epochs.
    # `iterations` is the fold-comparable step budget (folds see different
    # numbers of frames); set it and `epochs` becomes a fallback, unused.
    "iterations": 1000,
    "epochs": None,
    "batch_size": 16,
    "num_workers": 8,
    "fp16": True,
    "optimizer_config": {
        "name": "AdamW",
        "lr": 1e-4,
        "warmup_prop": 0.05,
        "betas": (0.9, 0.999),
        "max_grad_norm": 1.0,
        "weight_decay": 0.0,
    },

    # Loss
    "temperature": 0.05,

    # Matching, used by the inference side (embeddings/match.py)
    "ema_momentum": 0.5,            # the paper's alpha
    "match_threshold": None,        # None = the paper's dynamic threshold

    # Logging & evaluation
    "log_path": LOG_FOLDER + "contrastive/",
    "log_every": 100,
    "save_every": 500,
    "eval_every": 100,
    "n_val_videos": 5,
    "max_val_pairs": None,

    # Distributed
    "seed": 0,
    "sync_bn": False,
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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, value in DEFAULTS.items():
        if isinstance(value, bool):
            parser.add_argument(
                f"--{name}", action=argparse.BooleanOptionalAction, default=value
            )
        elif isinstance(value, dict):
            parser.add_argument(f"--{name}", type=parse_config(value), default=dict(value))
        elif isinstance(value, (tuple, list)) and any(isinstance(v, str) for v in value):
            parser.add_argument(f"--{name}", type=json.loads, default=list(value))
        elif isinstance(value, (tuple, list)) and all(isinstance(v, int) for v in (value or [])):
            parser.add_argument(f"--{name}", type=parse_shape,
                                default=tuple(value) if value else None)
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

        # Saved into every checkpoint so inference rebuilds the same geometry.
        args.model_config = {
            k: getattr(args, k) for k in (
                "d_model", "n_layers", "n_heads", "dim_feedforward", "dropout",
                "embed_dim", "backbone", "patch_zyx", "n_frames",
                "use_visual", "use_position", "use_transformer",
                "frame_aggregation", "min_tokens",
            )
        }

        if ctx.is_main:
            print("-> Logging results to", args.output_dir, "\n")
            with open(args.output_dir + "args.json", "w") as f:
                json.dump(vars(args), f, indent=4, default=str)
            create_logger(directory=args.output_dir, name="logs.txt")
            print(f"- Detections   : {args.coords_log_folder}")
            if ctx.distributed:
                print(f"- Distributed  : {ctx.world_size} processes, "
                      f"batch_size={args.batch_size} frames each")
            use_fp16 = args.fp16 and ctx.device.type == "cuda"
            print(f"- Precision    : {'FP16 mixed precision' if use_fp16 else 'FP32'}")

        k_fold(args, ctx)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
