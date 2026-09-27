"""
Training entrypoint for HOCT, the edge transformer, run from src/.

Single GPU:
    python main_edge.py --iterations 2000

Multi-GPU (DDP):
    torchrun --standalone --nproc_per_node=8 main_edge.py --iterations 2000

An implementation of the Higher-Order Cell Tracking Transformer (royerlab/hoct),
which ships inference-only with TorchScript weights; ``model_zoo/hoct.py`` is
the trainable source form, and reproduces the published ``general_v1`` to
float32 round-off when its weights are loaded into it.

The stage this trains is the last one: **the nodes already exist**. Detections
come from the tracking pipeline and their appearance features from NETr, both
already written to disk, and ``nodes_folder`` points at the npz folder holding
the two together (``coords`` and ``embeddings``, as ``inference_contrastive.py``
writes them). Nothing here opens a zarr or runs a convolution -- the job is
purely to decide which of the candidate links between fixed nodes are real.

Two things make that a different problem from the tracking pipeline's pair
scorer, and are the reason for this package:

* candidate edges are **tokens in a transformer of their own**, so a link is
  scored in the context of the links competing with it, and
* the ground truth is **partial** -- ~1 annotated cell a frame against ~260
  detections -- so only the edges the annotation can actually judge enter the
  loss. See ``edge_transformer/targets.py``.

batch_size and num_workers are per process; batch size counts *windows*.
"""

import argparse
import json
import os

from edge_transformer.main import k_fold
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
    # The best chain to date: this node set + the 2026-09-09_1 embeddings is
    # `edge_run/2026-09-13/0`, cv 0.8719 at `preds_0.45_pp_ilp` -- the top row
    # of Experiments.md. Its detector is `logs/2026-09-13/1`, whose own defaults
    # main_track.py now carries.
    "nodes_folder": "../logs/2026-09-13/1/preds_r6_0.40/",   # <- cv 0.8719
    # "nodes_folder": "../logs/2026-09-14/4/preds_r6_0.40/",   # bigger crops

    # "embeddings_key": "embeddings_contrastive_2026-09-09_1",
    "embeddings_key": "embeddings_contrastive_2026-09-26_0",
    # "embeddings_key": "embeddings_contrastive_2026-09-14_0",  # 3 frames, 2026-09-13/1/preds_r6_0.40
    
    "data_dir": TRAIN_DIR,
    "selected_folds": [0, 1] , # [0, 1],
    "fullfit": True,

    # Data
    "window_size": 2,               # frames per sample; 2 = one scored frame pair
    "max_nodes_per_frame": 1024,    # 512
    "n_train_videos": None,         # cap, for debugging
    "n_val_videos": 10,             # also how many videos the full metric scores
    "use_embeddings": True,         # False = geometry alone, as an ablation

    # Divisions are 0.13% of the annotated links (151 across both embryos), so
    # at their natural frequency the model does not learn them: the true second
    # child of a division scores a median 0.368 against an ordinary link's 0.45,
    # the assignment drops it, and the division term of the score -- a full 0.1
    # of it -- comes back only through post-processing repair. This prices both
    # links of a dividing parent up in the loss.
    "division_weight": 1.0,

    # Pseudo-labels
    "pl_folder": None,  # "../logs/edge_run/2026-09-07/4/preds_pl_0.45/",
    "pl_weight": 0.,  # Weight in the loss

    # Augmentations
    "augment": True,
    "flip_p": 0.5,
    "rot_p": 0.5,
    "jitter_um": 0.3,
    "time_flip_p": 0.5,  # Run the window backwards

    # Candidate graph
    "n_neighbors": 5,               # per node, per frame pair, both directions
    "max_distance_um": 20.0,        # cells move ~1.9 um a frame, spaced ~8 um
    "max_dt": 1,                    # >1 proposes gap-closing edges as context
    "edge_drift_correction": True,  # Correct the estimated field translation between frames
    "use_conf_score": True,  # Node confidence fts

    # Model
    "d_model": 288,                 # the published model's; 4 heads x 72
    "n_heads": 4,
    "n_node_blocks": 4,
    "n_edge_blocks": 4,
    "mlp_ratio": 2.0,
    "dropout": 0.1,
    "input_dropout": 0.1,           # Dropout1d over whole input channels
    "use_edge_feats": True,         # distance / similarity / rank features
    # Rotary positions
    "pos_scale": 20.0,  # unit the coordinates are divided by;
    "min_wavelength_um": 4.0,  # the frequency set is initialised to span these two wavelengths and is learnable from there.
    "max_wavelength_um": 200.0,
    "attn_max_distance_um": 40.0,  # Wider than the candidate radius - an edge should see the neighbourhood
    # The published model's Householder reflection on q and k. The same
    # orthogonal matrix is applied to both, so it cancels exactly in the
    # attention logits -- kept for weight compatibility, free to turn off.
    "householder": True,
    "weights": None,

    # Training
    "iterations": 2000,
    "epochs": None,
    "batch_size": 8,
    "val_batch_size": 8,
    "num_workers": 8,
    "fp16": True,
    "focal_gamma": 2.0,
    "pos_weight": 1.0,
    "rank_weight": 0.,  # 0.25   # Weight on the competitor-ranking term in `losses.competitor_rank_loss`.
    "degree_weight": 0.0,   # Weight on the structural term in `losses.degree_loss`
    "link_threshold": 0.4,  # for the reported precision/recall and linking
    "optimizer_config": {
        "name": "AdamW",
        "lr": 3e-4,
        "warmup_prop": 0.05,
        "betas": (0.9, 0.999),
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
    },

    # Logging & evaluation
    "log_path": LOG_FOLDER + "edge_run/",
    "log_every": 100,
    "save_every": 1000,
    "eval_every": 500,
    "video_eval_every": 1000,

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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
                "d_model", "n_heads", "n_node_blocks", "n_edge_blocks", "mlp_ratio",
                "dropout", "use_edge_feats", "pos_scale",
                "min_wavelength_um", "max_wavelength_um", "attn_max_distance_um",
                "householder", "window_size", "n_neighbors", "max_distance_um",
                "max_dt", "edge_drift_correction", "use_embeddings",
                "use_conf_score", "max_nodes_per_frame",
                # Carried so inference reads the same run's embeddings: a
                # checkpoint trained on one NETr run scores nothing useful on
                # another's vectors, and both can sit in the same npz.
                "embeddings_key",
            )
        }

        if ctx.is_main:
            print("-> Logging results to", args.output_dir, "\n")
            with open(args.output_dir + "args.json", "w") as f:
                json.dump(vars(args), f, indent=4, default=str)
            create_logger(directory=args.output_dir, name="logs.txt")
            print(f"- Nodes        : {args.nodes_folder}")
            if ctx.distributed:
                print(f"- Distributed  : {ctx.world_size} processes, "
                      f"batch_size={args.batch_size} windows each")
            use_fp16 = args.fp16 and ctx.device.type == "cuda"
            print(f"- Precision    : {'mixed precision' if use_fp16 else 'FP32'}")

        k_fold(args, ctx)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
