"""Run one detector over a folder of stacks without ground truth (the test split).

`main_predict.py` predicts out of fold on the train split; this runs a single
checkpoint (the fold_fullfit by default) over any folder of `.zarr` stacks and
writes one `<stack>.npz` per stack with `coords`, `edges` and `node_probs`, the
layout the merge and NETr read. Run from src/, like every entry point:

    torchrun --standalone --nproc_per_node=4 infer_detector.py \
        --log_folder ../logs/2026-09-13/1/ --det_threshold 0.45 \
        --nms_radius_um 6 --out_dir ../output/inference/preds_base/
"""

import warnings
warnings.filterwarnings("ignore")

import json
import torch
import argparse
import numpy as np
from pathlib import Path

from tracking.main import build_model
from tracking_cellmot.io import list_datasets
from tracking.predict import PredictConfig, predict_video_thresholds
from training.distributed import cleanup_distributed, init_distributed
from params import TEST_DIR
from utils.logger import Config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log_folder", required=True, help="run folder holding args.json and the checkpoint")
    p.add_argument("--ckpt", default="fold_fullfit/ckpt_final.pt",
                   help="checkpoint, relative to --log_folder; searched for if not found there")
    p.add_argument("--det_threshold", type=float, default=0.40,
                   help="detection threshold; pick it from the run's cv_score")
    p.add_argument("--nms_radius_um", type=float, default=None,
                   help="greedy-NMS radius. Overrides the training run's own "
                        "value, which is a training default and not the one the "
                        "downstream node set was chosen at -- preds_r6_0.40 "
                        "means radius 6, while args.json records 7")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--limit", type=int, default=None, help="cap the stacks (debugging)")
    p.add_argument("--no_fp16", action="store_true",
                   help="run the detector in fp32. Measured on an H100, where "
                        "fp32 convs are really TF32 so the gap is at its "
                        "narrowest: the UNet forward is 54%% of this stage and "
                        "fp16 takes it 3.34s -> 2.53s per 30 frames (-24%%), "
                        "total -12%%. A T4/P100 has no TF32, so expect more "
                        "there. It moves detections: 6682 nodes against 6683 "
                        "on one stack, threshold crossings in the last digit")
    p.add_argument("--data_dir", default=TEST_DIR, help="folder of .zarr stacks")
    return p.parse_args()


def resolve_ckpt(log_folder: Path, ckpt: str) -> Path:
    """The checkpoint, wherever the weights dataset happens to have put it.

    A run folder keeps checkpoints under fold_*/; a weights dataset is often
    flattened to the root instead. Rather than making the caller know which,
    take the path as given and fall back to a search by file name.
    """
    direct = log_folder / ckpt
    if direct.exists():
        return direct
    found = sorted(log_folder.rglob(Path(ckpt).name))
    if not found:
        available = sorted(str(f.relative_to(log_folder)) for f in log_folder.rglob("*.pt"))
        raise FileNotFoundError(
            f"{ckpt} not found under {log_folder}. Checkpoints present: {available}"
        )
    return found[0]


def main():
    args = parse_args()
    ctx = init_distributed(None)
    device = ctx.device

    # Every window is the same shape, so cuDNN's autotuner pays for itself on
    # the first few and wins on the rest. Worth ~0.4s per 30 frames here on top
    # of fp16; on a fixed-shape 3D conv stack it is free money.
    torch.backends.cudnn.benchmark = True

    paths = sorted(list_datasets(Path(args.data_dir), require_geff=False))
    if args.limit:
        paths = paths[: args.limit]

    log_folder = Path(args.log_folder)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = resolve_ckpt(log_folder, args.ckpt)

    cfg = Config(json.load(open(log_folder / "args.json")))

    # Inference settings come from the training run wherever the run fixes
    # them, so the node sets the edge head sees here are the ones it was fitted
    # against. Only the detection threshold is chosen separately -- it is swept
    # after training, and the value below is the sweep's answer.
    pred_cfg = PredictConfig
    pred_cfg.det_threshold = args.det_threshold
    pred_cfg.normalization = getattr(cfg, "normalization", None)
    pred_cfg.nms = getattr(cfg, "nms", pred_cfg.nms)
    pred_cfg.nms_radius_um = (
        args.nms_radius_um if args.nms_radius_um is not None
        else getattr(cfg, "nms_radius_um", pred_cfg.nms_radius_um)
    )
    pred_cfg.pool_kernel_um = getattr(cfg, "pool_kernel_um", pred_cfg.pool_kernel_um)
    pred_cfg.use_offset = bool(getattr(cfg, "offset_head", False))

    window_size = getattr(cfg, "window_size", 2)
    downscale_factor = getattr(cfg, "downscale_factor", 1)

    if ctx.is_main:
        print(f"-> log_folder     : {log_folder}")
        print(f"-> checkpoint     : {ckpt_path.relative_to(log_folder)}")
        print(f"-> threshold      : {pred_cfg.det_threshold}")
        print(f"-> nms            : {pred_cfg.nms} @ {pred_cfg.nms_radius_um} um")
        print(f"-> normalization  : {pred_cfg.normalization}")
        print(f"-> window_size    : {window_size}")
        print(f"-> stacks         : {len(paths)}")
        print(f"-> writing to     : {out_dir}")
        if ctx.distributed:
            print(f"-> distributed    : {ctx.world_size} processes")
        print()

    state = torch.load(ckpt_path, map_location="cpu")
    state = state.get("model_state_dict", state)
    state = {k.replace("unet.module.", "unet.", 1): v for k, v in state.items()}

    model = build_model(cfg, None)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    # Shard stacks across ranks; each writes its own npz files.
    mine = paths[ctx.rank :: ctx.world_size]

    for i, path in enumerate(mine):
        print(f"[rank {ctx.rank}] {i + 1}/{len(mine)}  -  {path.name}", flush=True)

        # predict_video_thresholds rather than predict_video: the latter drops
        # the per-node detection probability, and the edge model reads it as a
        # feature (`use_conf_score`). Without a `node_probs` array its dataset
        # refuses to build, so the pipeline would fail two stages later.
        # autocast here rather than inside predict_video_thresholds: that
        # function is also the training-time evaluator, and its numbers are
        # compared against training losses computed in fp32.
        with torch.autocast("cuda", torch.float16,
                            enabled=not args.no_fp16 and device.type == "cuda"):
            results, _ = predict_video_thresholds(
                model,
                path,
                thresholds=[pred_cfg.det_threshold],
                device=device,
                cfg=pred_cfg,
                window_size=window_size,
                downscale_factor=downscale_factor,
                progress=ctx.is_main,
                return_det_logits=False,
                nms_radii=[pred_cfg.nms_radius_um],
            )
        # Keyed (head, radius, threshold), five values. "det" is the plain
        # detection head; this detector also carries an annotation head, and
        # the node set the edge model was fitted on came from "det", so asking
        # for the other one here would score its candidates against a node set
        # it never saw. The trailing two -- the annotation probability and the
        # unrounded coordinates -- are for consumers that still have fusing to
        # do, which this pipeline does not.
        coords, edges, node_probs, _annot, _coords_f = results[
            ("det", float(pred_cfg.nms_radius_um), float(pred_cfg.det_threshold))
        ]

        np.savez_compressed(
            out_dir / f"{path.name}.npz",
            coords=coords,
            edges=np.asarray(edges, dtype=np.float64).reshape(-1, 4),
            # Row-aligned with coords, exactly as main_predict.py writes it.
            node_probs=np.asarray(node_probs, dtype=np.float32).reshape(-1),
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
