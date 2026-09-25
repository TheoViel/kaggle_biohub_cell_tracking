#!/usr/bin/env python
"""Pseudo-label competition-format crops that have no ground truth.

``main_predict.py`` cannot be pointed at these: it discovers datasets under
``TRAIN_DIR`` with ``require_geff=True`` and scores each against its ``.geff``.
Unlabelled crops have neither, so this drives the same model and the same
``predict_video_thresholds`` over an explicit directory and writes the same
npz layout, minus the evaluation.

The normalisation comes from the *training run's* config, not from defaults --
the detector's thresholds only mean anything against the intensity mapping it
was fitted on, which is why ``--log_folder``'s ``args.json`` is read rather
than reconstructed (the same reasoning as main_predict's `pred_cfg`).

    python scripts/pseudo_label_crops.py \
        --log_folder ../logs/2026-09-13/1 --data_dir ../input/ultrack_crops \
        --det_threshold 0.6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tracking.main import build_model  # noqa: E402
from tracking.predict import PredictConfig, predict_video_thresholds  # noqa: E402


def load_model(log_folder: Path, fold: int, ckpt: str | None, device):
    """The trained model plus the config it was fitted with."""
    cfg = SimpleNamespace(**json.load(open(log_folder / "args.json")))
    ckpt_path = Path(ckpt) if ckpt else log_folder / f"fold_{fold}" / "ckpt_final.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    state = state.get("model_state_dict", state)
    # Same key fix main_predict applies: DDP leaves "unet.module." behind.
    state = {k.replace("unet.module.", "unet.", 1): v for k, v in state.items()}
    model = build_model(cfg, None)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, cfg, ckpt_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log_folder", default=str(REPO / "logs" / "2026-09-13" / "1"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--data_dir", default=str(REPO / "input" / "ultrack_crops"))
    p.add_argument("--out_dir", default=None, help="default <log_folder>/preds_pl_<tag>")
    p.add_argument("--det_threshold", type=float, default=0.6)
    p.add_argument("--nms_radius_um", type=float, default=PredictConfig.nms_radius_um)
    p.add_argument("--window_size", type=int, default=2)
    p.add_argument("--n_tta", type=int, choices=(1, 4, 8), default=1)
    p.add_argument("--annot_head", action="store_true", help="also write the annot head's nodes")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_folder = Path(args.log_folder)
    data_dir = Path(args.data_dir)
    paths = sorted(data_dir.glob("*.zarr"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no .zarr crops under {data_dir}")

    model, cfg, ckpt_path = load_model(log_folder, args.fold, args.ckpt, device)

    pred_cfg = PredictConfig()
    pred_cfg.n_tta = args.n_tta
    pred_cfg.det_threshold = args.det_threshold
    pred_cfg.nms_radius_um = args.nms_radius_um
    # The training run's mapping, not PredictConfig's default.
    pred_cfg.normalization = getattr(cfg, "normalization", None)

    heads = ("det", "annot") if args.annot_head else ("det",)
    base = Path(args.out_dir) if args.out_dir else (
        log_folder / f"preds_pl_ultrack_{args.det_threshold:.2f}")
    out_dirs = {"det": base,
                "annot": base.with_name(base.name + "_annot_head")}
    for head in heads:
        out_dirs[head].mkdir(parents=True, exist_ok=True)
    print(f"-> model        : {ckpt_path}")
    print(f"-> normalisation: {pred_cfg.normalization}")
    print(f"-> crops        : {len(paths)} under {data_dir}")
    print(f"-> threshold    : {args.det_threshold}  nms {args.nms_radius_um} um")
    print(f"-> heads        : {', '.join(heads)}")
    print(f"-> writing to   : {', '.join(str(out_dirs[h]) for h in heads)}\n", flush=True)

    started, n_nodes = time.time(), []
    for i, path in enumerate(paths, 1):
        name = path.stem
        out_path = out_dirs[heads[0]] / f"{name}.npz"
        if args.skip_existing and out_path.exists():
            continue
        per_key, _ = predict_video_thresholds(
            model, path, [args.det_threshold], device=device, cfg=pred_cfg,
            window_size=args.window_size, progress=False, return_det_logits=False,
            nms_radii=[args.nms_radius_um], heads=heads,
        )
        # One directory per head, mirroring main_predict's `_annot_head` suffix.
        # Reading only the "det" key would silently discard the annotation
        # head's node set while still paying for it.
        for head in heads:
            coords, edges, node_probs, annot_probs, coords_f = per_key[
                (head, args.nms_radius_um, args.det_threshold)
            ]
            target = out_dirs[head] / f"{name}.npz"
            np.savez_compressed(
                target,
                coords=coords,
                edges=np.asarray(edges, dtype=np.float64).reshape(-1, 4),
                node_probs=np.asarray(node_probs, dtype=np.float32).reshape(-1),
                annot_probs=np.asarray(annot_probs, dtype=np.float32).reshape(-1),
                coords_f=np.asarray(coords_f, dtype=np.float32).reshape(-1, 4),
            )
            if head == heads[0]:
                n_nodes.append(len(coords))
        if i % 25 == 0 or i == len(paths):
            el = time.time() - started
            print(f"[{i}/{len(paths)}] median {np.median(n_nodes):.0f} nodes/crop, "
                  f"{el/i:.1f}s each, {el/60:.1f} min", flush=True)

    n = np.array(n_nodes)
    print(f"\nDONE {len(n)} crops -> {', '.join(str(out_dirs[h]) for h in heads)}")
    if len(n):
        frames = 4
        print(f"  nodes/crop: p10 {np.quantile(n,.1):.0f}  p50 {np.median(n):.0f}  "
              f"p90 {np.quantile(n,.9):.0f}   (~{np.median(n)/frames:.0f} per frame)")


if __name__ == "__main__":
    sys.exit(main())
