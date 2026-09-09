"""
Training entrypoint for the division classifier, run from src/.

Single GPU:
    python main_division.py --iterations 2000

Multi-GPU (DDP):
    torchrun --standalone --nproc_per_node=8 main_division.py --iterations 2000

Asks one question about an annotated cell: **does it divide at this frame?**
The input is what `notebooks/Divisions.ipynb` plots -- a short clip of crops
centred on the cell, `n_frames` frames of time, `patch_z` z-slices each -- and
the output is one logit.

Why a separate model rather than a term in the edge transformer. The score is
`adj_edge_jaccard + 0.1 * division_jaccard` and the chain converts about one
annotated division in seven; `postprocessing/README_edge.md` shows that no
geometric post-processing recovers the rest, because the *probabilities* do not
separate a true second daughter from a free neighbour. Divisions are 0.13% of
the links, so the edge model has little reason to learn them. A classifier that
looks at the image directly is the other way at it, and its output is a node
prior `edge_pp.propose_divisions` can gate on.

What it is not: a solution to the label count. There are 151 annotated
divisions across both embryos, and the honest split is by embryo, so a fold
trains on 26 or 125 of them. Read the validation AP against `chance_ap`
(the positive rate) rather than in absolute terms, and treat a fold that
disagrees with the other as noise -- that has already happened once here, with
the candidate classifier in `postprocessing/div_classifier.py`.
"""

import argparse
import json
import os

from division_cls.main import k_fold
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
    "data_dir": TRAIN_DIR,
    "selected_folds": [0, 1],
    "fullfit": False,

    # Data. 32 voxels is 13 um in y/x, so the cell (~8.5 um) plus the sliver of
    # neighbourhood that says whether it just split; 3 z-slices become the
    # backbone's input channels, as in the contrastive pipeline.
    "patch_yx": 32,
    "patch_z": 3,
    "n_frames": 3,                  # odd, centred on the labelled frame

    # Negatives. "track" takes them from the lineages that contain a division,
    # so the model is asked *when* a cell divides rather than *which* cell does
    # -- and the imbalance falls from 1:880 to about 1:12. See dataset.py.
    "negatives": "track",           # "track" | "stack" | "all"
    "neg_per_pos": 12.0,
    "exclude_window": 1,            # frames flanking a division, dropped
    "positive_multiplier": 4,       # oversampling, training split only
    "pos_weight": 1.0,              # on top of oversampling; usually leave at 1
    "max_stacks": None,

    "augmentations": {
        "jitter_um": 1.5,
        "rot_p": 0.75,
        "rot_deg": 180.0,           # no canonical orientation for a division
        "elastic_p": 0.5,
        "elastic_um": 1.0,
        "elastic_sigma_um": 5.0,
        "scale_p": 0.5,
        "scale_range": (0.9, 1.1),
        "intensity_p": 0.8,
        "brightness": 0.2,
        "contrast": 0.2,
        "gamma": 0.2,
        "noise_std": 0.0,
        "dropout_p": 0.0,
    },

    # Model
    "backbone": "convnext_tiny",
    # "backbone": "maxvit_tiny_tf_224",   # hybrid conv/attention, needs img_size
    "pretrained": True,
    "min_tokens": 4,                # keep a >=4x4 map for the transformer
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 4,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "grad_checkpointing": False,
    "weights": None,

    # Training
    "iterations": 2000,
    "epochs": None,
    "batch_size": 32,
    "val_batch_size": 64,
    "num_workers": 8,
    "fp16": True,
    "optimizer_config": {
        "name": "AdamW",
        "lr": 3e-4,
        "lr_encoder": 3e-5,         # the pretrained backbone, moved gently
        "warmup_prop": 0.05,
        "betas": (0.9, 0.999),
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
    },

    # Logging
    "log_path": LOG_FOLDER + "division/",
    "log_every": 50,
    "eval_every": 250,

    # Distributed
    "seed": 0,
    "sync_bn": False,
    "dist_backend": None,
}


def parse_config(defaults):
    def parse(value):
        return {**defaults, **json.loads(value)}
    return parse


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, value in DEFAULTS.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction,
                                default=value)
        elif isinstance(value, dict):
            parser.add_argument(f"--{name}", type=parse_config(value), default=dict(value))
        elif isinstance(value, (tuple, list)):
            parser.add_argument(f"--{name}", type=json.loads, default=list(value))
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
                "backbone", "patch_yx", "patch_z", "n_frames", "d_model",
                "n_layers", "n_heads", "dim_feedforward", "dropout", "min_tokens",
            )
        }

        if ctx.is_main:
            print("-> Logging results to", args.output_dir, "\n")
            with open(args.output_dir + "args.json", "w") as f:
                json.dump(vars(args), f, indent=4, default=str)
            create_logger(directory=args.output_dir, name="logs.txt")
            print(f"- Crops     : {args.n_frames} frames x {args.patch_z} z "
                  f"x {args.patch_yx}^2, negatives from {args.negatives!r}")
            print(f"- Precision : {'FP16' if args.fp16 else 'FP32'}")

        k_fold(args, ctx)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
