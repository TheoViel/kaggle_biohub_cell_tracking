"""Training entrypoint for the merge (deduplication) classifier, run from src/.

    python main_merging.py --iterations 2000
    torchrun --standalone --nproc_per_node=8 main_merging.py --iterations 2000

Asks one question about a **pair of detections in one frame**: are these two
cells, or one cell the detector found twice?

Why it exists. The detector doubles 2.85% of annotated cells, and 23% of the 151
annotated divisions -- a dividing cell is bilobed, which is exactly the shape
that makes a peak finder fire twice. `postprocessing/README_edge.md` records
that geometry alone cannot separate that duplicate from an ordinary neighbour
(a 7 um merge fires 10 508 times for 34 opportunities, because rival pairs sit
at median 8.1 um and so do thousands of genuine neighbours). This reads the
pixels instead.

**What it is worth.** An oracle that uses the annotation as a perfect duplicate
detector and merges what it finds scores +0.0129 cv (0.8513 -> 0.8642), which is
more than the whole shipped division-gate chain (+0.0114). The gain is entirely
in `adj_edge_jaccard` (+0.0165); `division_jaccard` *falls* (-0.0368), so the
division stage needs re-tuning after a merge rather than being left alone. See
`cell_merging_cls/README.md`.

**Merge, do not delete.** The same oracle scores only +0.0050 if the duplicate
is deleted instead of merged, because in the fork bucket the duplicate is the
copy carrying the second daughter -- deleting it deletes the division. The
model's output is therefore a *pairing*, and `main_merging_predict.py` re-points
links onto the survivor.
"""

import argparse
import json
import os

from cell_merging_cls.main import k_fold
from cell_merging_cls.pairs import DUP_RADIUS_UM, MAX_PAIR_UM
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
    # Detections. Pairs are built from what the detector actually produced,
    # because that is what the model is asked about at inference; building them
    # from annotated nodes would train on a cleanliness that does not exist.
    "preds_dir": LOG_FOLDER + "edge_run/2026-09-08/13/preds_0.50",
    "selected_folds": [0, 1],
    "fullfit": False,

    # Data
    "patch_yx": 32,
    # 9, not the division classifier's 3. 61% of duplicate pairs are separated
    # more in z than in the plane (median 3 z-slices), because the voxel is 4x
    # coarser in z and the detector splits one cell across two planes. A 3-slice
    # crop reaches +-1 and would not contain the second detection.
    "patch_z": 9,
    "n_frames": 3,                  # odd, centred on the pair's frame

    # Which pairs are asked about, and which the annotation can label.
    "max_pair_um": MAX_PAIR_UM,     # edge_pp's max_sister_um
    "dup_radius": DUP_RADIUS_UM,    # "the same annotated cell"
    "neg_per_pos": 8.0,             # ordinary annotated negatives per duplicate
    # Close pairs the annotation cannot label, taken as negatives. ~96.5% of
    # them are genuinely two cells, and without them the training set runs 2:1
    # positive against an inference population that is ~28:1 negative. See
    # `pairs.labelled_pairs` for why this direction is sound and the reverse
    # ("unmatched, therefore duplicate") is not.
    "unlabelled_per_pos": 20.0,
    "positive_multiplier": 4,
    # Sisters are ~0.1% of the negatives and 100% of the damage a false merge
    # can do, so they are repeated rather than left to the sampler.
    "sister_multiplier": 4,
    "pos_weight": 1.0,
    "max_stacks": None,
    # The annotated divisions as an unambiguous held-out probe: the parent is
    # one cell, the two daughters are two. See `pairs.division_probe`.
    "probe": True,

    # Augmentation. Same knobs as `division_cls`, with three changes forced by
    # the oriented crop: rotation is off (it would undo the canonicalisation
    # that puts the pair on +x), and the two label-changing division knobs are
    # off because they mean nothing here. Flips stay on -- across the axis is a
    # mirror, along it swaps the two members, and the label is symmetric in them.
    "augmentations": {
        "jitter_um": 1.0,
        "rot_p": 0.0,
        "rot_deg": 0.0,
        "elastic_p": 0.6,
        "elastic_um": 1.4,
        "elastic_sigma_um": 5.0,
        "scale_p": 0.6,
        "scale_range": (0.85, 1.18),
        "intensity_p": 0.8,
        "brightness": 0.25,
        "contrast": 0.25,
        "gamma": 0.25,
        "noise_std": 0.0,
        "dropout_p": 0.0,
        "flip_p": 0.5,
        "rot90_p": 0.0,
        "z_jitter": 1,
        "frame_shift_um": 0.4,
        "frame_brightness": 0.06,
        "frame_contrast": 0.06,
        "shot_noise": 0.04,
        "blur_p": 0.25,
        "blur_sigma": (0.4, 1.1),
        "cutout_p": 0.25,
        "cutout_frac": (0.1, 0.3),
        "time_reverse_p": 0.0,
        "drop_context_p": 0.0,
    },

    # Model
    "backbone": "convnext_tiny",
    "pretrained": True,
    "min_tokens": 4,
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 4,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "geom_dim": 32,
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
        "lr_encoder": 3e-5,
        "warmup_prop": 0.05,
        "betas": (0.9, 0.999),
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
    },

    # Logging
    "log_path": LOG_FOLDER + "merging/",
    "log_every": 50,
    # Larger than `division_cls`'s 250 on purpose. Validation here is every
    # labelled pair in the held-out embryo, and fold 1 holds out the big one:
    # 118k pairs, each a 3x9x32x32 read, is ~6 minutes a pass. At 250 that is
    # more time in validation than in training.
    "eval_every": 1000,

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

        args.model_config = {
            k: getattr(args, k) for k in (
                "backbone", "patch_yx", "patch_z", "n_frames", "d_model",
                "n_layers", "n_heads", "dim_feedforward", "dropout", "min_tokens",
                "geom_dim", "max_pair_um",
            )
        }

        if ctx.is_main:
            print("-> Logging results to", args.output_dir, "\n")
            with open(args.output_dir + "args.json", "w") as f:
                json.dump(vars(args), f, indent=4, default=str)
            create_logger(directory=args.output_dir, name="logs.txt")
            print(f"- Pairs     : within {args.max_pair_um} um, crops "
                  f"{args.n_frames} frames x {args.patch_z} z x {args.patch_yx}^2, "
                  f"oriented on the pair axis")
            print(f"- Detections: {args.preds_dir}")
            print(f"- Precision : {'FP16' if args.fp16 else 'FP32'}")

        k_fold(args, ctx)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
