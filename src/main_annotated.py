"""Train the "is this detection annotated" classifier, run from src/.

    python main_annotated.py --iterations 2000
    torchrun --standalone --nproc_per_node=8 main_annotated.py --iterations 2000

One question about a **single detection** in a post-processed graph: is this one
of the cells the annotation covers, and linked the way the annotation says?

Why it exists. `adj_edge_jaccard = edge_jaccard * (1 - 0.1 * node_excess)`
charges for every node whether or not the metric ever looks at it, and the
annotation looks at very few: 0.466% of detections are positive under this
label. So a ranking that puts the annotated cells on top lets the graph be
trimmed from the bottom, dropping node count -- which the metric charges -- while
leaving the edges it scores untouched.

Read the result carefully. A negative here is, 214 times out of 215, a perfectly
real cell that the annotators did not follow; this label is not "spurious
detection" and a model fitted to it is not a detector of them. The annotated
cells are also spatially clustered, so a crop model can score well by learning
where the lineage sits rather than what its cells look like. The split is by
embryo for exactly that reason -- see `annotated_cls/labels.py`.
"""

import argparse
import json
import os

from annotated_cls.main import k_fold
from params import LOG_FOLDER, TRAIN_DIR
from training.distributed import (
    broadcast_object,
    cleanup_distributed,
    init_distributed,
    seed_everything,
)
from utils.logger import create_logger, prepare_log_folder


DEFAULTS = {
    "data_dir": TRAIN_DIR,
    # The *post-processed* graph, not the raw detections: the label is "carries
    # a link the annotation agrees with", which does not exist until something
    # has linked them.
    "preds_dir": LOG_FOLDER + "edge_run/2026-09-15/0/preds_0.45_pp",
    "selected_folds": [0, 1],
    "fullfit": False,

    # Data
    "patch_yx": 32,
    "patch_z": 9,
    "n_frames": 1,              # one detection, one moment: no clip by default
    # 214:1 raw. 30:1 keeps the epoch honest without pretending the prior is
    # balanced; `pos_weight` is left at 1 so the scores stay calibrated to it.
    "neg_per_pos": 30.0,
    "val_neg_per_pos": 30.0,
    "positive_multiplier": 1,
    "pos_weight": 1.0,
    "max_stacks": None,
    "val_max_stacks": 20,

    "augmentations": {
        "jitter_um": 1.0,
        "rot_p": 0.5, "rot_deg": 180.0,
        "elastic_p": 0.6, "elastic_um": 1.4, "elastic_sigma_um": 5.0,
        "scale_p": 0.6, "scale_range": (0.85, 1.18),
        "intensity_p": 0.8, "brightness": 0.25, "contrast": 0.25, "gamma": 0.25,
        "noise_std": 0.0, "dropout_p": 0.0,
    },

    # Model
    "backbone": "convnext_tiny",
    "pretrained": True,
    "dropout": 0.1,
    "grad_checkpointing": False,

    # Optimisation
    "iterations": 2000,
    "batch_size": 64,
    "val_batch_size": 128,
    "num_workers": 8,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "warmup_prop": 0.05,
    "max_grad_norm": 1.0,
    "fp16": True,

    "log_path": LOG_FOLDER + "annotated_run/",
    "log_every": 100,
    "eval_every": 250,
    "seed": 0,
    "dist_backend": None,
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        if k in ("augmentations", "selected_folds", "fullfit"):
            continue
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=lambda s: str(s).lower() != "false", default=v)
        elif v is None:
            p.add_argument(f"--{k}", default=None)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    p.add_argument("--selected_folds", type=lambda s: [int(x) for x in s.split(",")],
                   default=DEFAULTS["selected_folds"])
    p.add_argument("--fullfit", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    for k, v in DEFAULTS.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    ctx = init_distributed(args.dist_backend)
    seed_everything(int(args.seed), ctx.rank)

    output_dir = None
    if ctx.is_main:
        output_dir = prepare_log_folder(args.log_path)
        os.makedirs(output_dir, exist_ok=True)
    output_dir = broadcast_object(output_dir, device=ctx.device)
    args.output_dir = output_dir

    if ctx.is_main:
        create_logger(directory=output_dir, name="logs.txt")
        print(f"-> output {output_dir}")
        print(f"-> preds  {args.preds_dir}")
        with open(os.path.join(output_dir, "args.json"), "w") as f:
            json.dump({k: getattr(args, k, None) for k in DEFAULTS}, f, indent=4,
                      default=str)

    try:
        k_fold(args, ctx)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
