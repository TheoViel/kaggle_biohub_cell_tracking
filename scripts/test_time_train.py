#!/usr/bin/env python
"""Test-time training: adapt a trained model to one embryo using its own output.

The motivation is the embryo domain shift -- a model fitted on one embryo reads
the next one's intensity statistics slightly wrong. This adapts to the target
embryo without any label: the model predicts over the target videos, those
predictions become the training target, and a few hundred low-LR steps follow.
Nothing here touches the ground truth, so the identical procedure runs at test
time; the GT is used only to *score* the before/after, never to fit.

What the target actually is
---------------------------
The stored ``preds_<tag>/<sample>.npz`` the run already wrote: ``coords`` as
the nodes and ``edges`` as the transition matrices, both the model's own. They
are re-thresholded on ``node_probs`` / edge probability so the fit is anchored
on the confident part of the model's own belief rather than on all of it --
self-training on marginal detections reinforces exactly the errors that need
fixing. Note the cost of that choice: a detection below ``--node-threshold``
is not merely unused, it is absent from the target, so the loss actively calls
it background. Raising the threshold therefore buys purity by teaching the
model to suppress its own borderline peaks, which is a real behaviour change
and not a neutral filter. An ignore band over the uncertain range would avoid
it and is the obvious extension; it is not implemented here.

The weak-positive term the training runs use is switched off: weak positives
are a *detector's* belief injected alongside the annotation, and here the
detector's belief is already the whole target. Leaving it on would double-count
it -- and with nothing supplied the loss asserts anyway.

Why it can work at all despite the target being the model's own output
----------------------------------------------------------------------
Two mechanisms, and neither needs the labels to be better than the model:
  * the normalisation layers see the target embryo's statistics in train mode,
    which is plain BN/domain adaptation and is where most of the effect
    usually comes from (`--norm-only` isolates it);
  * the loss is evaluated on random crops of the target volume, so consistent
    detections get reinforced across contexts while inconsistent ones average
    out -- self-distillation, not new information.

It can also do nothing, or drift: a fixed point of "fit your own output" is
your own output. Read the printed before/after, not the loss.

    python scripts/test_time_train.py --log_folder logs/2026-09-13/1 --fold 0 \
        --preds preds_r6_0.40 --steps 200
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from params import TRAIN_DIR  # noqa: E402
from tracking.data import (  # noqa: E402
    FrameWindowData, VideoMeta, _k_step_transition, resolve_normalization,
)
from tracking.eval import evaluate_videos  # noqa: E402
from tracking.losses import TrackingLoss  # noqa: E402
from tracking.main import build_loader, build_model  # noqa: E402
from tracking.targets import build_matched_edge_targets  # noqa: E402
from tracking_cellmot.io import open_dataset  # noqa: E402
from utils.logger import Config  # noqa: E402


# ---------------------------------------------------------------- the target


def pseudo_windows(
    zarr_path: Path,
    npz_path: Path,
    window_size: int,
    node_threshold: float,
    edge_threshold: float,
    normalization: dict | None,
    max_windows: int | None,
    rng: np.random.Generator,
) -> tuple[VideoMeta, list[FrameWindowData]] | None:
    """One video's windows, with the model's own detections standing in for GT.

    Mirrors ``load_dataset_windows`` but reads the nodes and links out of a
    predictions npz instead of a ``.geff``. ``require_tracks=False`` is the
    point: at test time there is no geff to require.
    """
    ds = open_dataset(zarr_path, normalize=False, require_tracks=False,
                      load_image=False)
    norm = resolve_normalization(normalization)
    if norm["scope"] == "video":
        missing = [q for q in (norm["low_q"], norm["high_q"]) if q not in ds.quantiles]
        if missing:
            raise ValueError(f"{zarr_path} lacks stored quantiles {missing}")

    meta = VideoMeta(
        zarr_path=ds.zarr_path,
        image_shape=ds.image_shape,
        voxel_size=ds.scale,
        q_low=float(ds.quantiles[norm["low_q"]]) if norm["scope"] == "video" else float("nan"),
        q_high=float(ds.quantiles[norm["high_q"]]) if norm["scope"] == "video" else float("nan"),
        norm=norm,
    )

    d = np.load(npz_path)
    coords = d["coords"].astype(np.float32)
    probs = d["node_probs"].astype(np.float32).reshape(-1)
    keep = probs >= node_threshold
    # Edge endpoints index the *unfiltered* rows, so they have to be remapped
    # before anything is dropped.
    remap = np.full(len(coords), -1, np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    coords, probs = coords[keep], probs[keep]

    succ: dict[int, list[int]] = {}
    e = d["edges"].reshape(-1, 4)
    if len(e):
        s, t = remap[e[:, 0].astype(int)], remap[e[:, 1].astype(int)]
        ok = (s >= 0) & (t >= 0) & (e[:, 2] >= edge_threshold)
        for a, b in zip(s[ok].tolist(), t[ok].tolist()):
            succ.setdefault(a, []).append(b)

    frames = np.round(coords[:, 0]).astype(int)
    by_t = {int(f): np.where(frames == f)[0] for f in np.unique(frames)}

    starts = [t for t in sorted(by_t)
              if all((t + i) in by_t and len(by_t[t + i]) for i in range(window_size))]
    if not starts:
        return None
    if max_windows is not None and len(starts) > max_windows:
        # Spread over the video rather than taking the head: the first frames
        # of a stack are not representative of the whole of it.
        starts = [starts[i] for i in
                  rng.choice(len(starts), size=max_windows, replace=False)]

    windows = []
    for t0 in sorted(starts):
        ids = [by_t[t0 + i] for i in range(window_size)]
        windows.append(FrameWindowData(
            t_start=int(t0), n_frames=window_size, stride=1,
            coords=[torch.from_numpy(coords[j][:, 1:4]) for j in ids],
            node_counts=[int(len(j)) for j in ids],
            targets=[_k_step_transition(ids[i], ids[i + 1], succ, 1)
                     for i in range(window_size - 1)],
        ))
    return meta, windows


# ------------------------------------------------------------------ the step


def train_step(model, batch, criterion, args, device, window_size,
               use_amp=False, amp_dtype=torch.float16):
    """One optimisation step, structured exactly as `tracking.train.fit` does.

    The autocast is not optional in practice: the training runs use it
    (``cfg.fp16``), and without it a batch of 4 whole-window crops OOMs an 80 GB
    card in the backward.
    """
    from tracking.fit_utils import encode_nodes

    imgs = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
    coords = batch["coords"].to(device, non_blocking=True)
    masks = batch["masks"].to(device, non_blocking=True)
    targets = batch["targets"].to(device, non_blocking=True)
    image_shape = tuple(batch["image_shape"][0].tolist())
    voxel_size = tuple(batch["voxel_size"][0].tolist())

    ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)
    with ctx:
      (frame_det, det_logits, unet_scaling, offset_maps,
     det_logits_ds, det_stride, annot_logits) = encode_nodes(
          model, imgs, coords, masks, image_shape, voxel_size, args,
          teacher_forcing=False, detect=True,
      )

      edge_logits, edge_targets, det_mask_pairs = [], [], []
      for i in range(window_size - 1):
          ns, nt = frame_det[i][0].shape[1], frame_det[i + 1][0].shape[1]
          edge_targets.append(build_matched_edge_targets(
              frame_det[i][3], frame_det[i + 1][3], targets[:, i], ns, nt))
          edge_logits.append(model.predict_edges(
              unet_feat_src=frame_det[i][4], unet_feat_tgt=frame_det[i + 1][4],
              coords_src=frame_det[i][0], coords_tgt=frame_det[i + 1][0],
              pos_feat_src=frame_det[i][1], pos_feat_tgt=frame_det[i + 1][1],
              mask_src=frame_det[i][2], mask_tgt=frame_det[i + 1][2],
          ))
          det_mask_pairs.append((frame_det[i][2], frame_det[i + 1][2]))

      return criterion(
          det_logits,
          [coords[:, i] for i in range(window_size)],
          [masks[:, i] for i in range(window_size)],
          edge_logits, edge_targets, det_mask_pairs,
          unet_scaling=unet_scaling,
          det_stride=det_stride,
          det_logits_ds_per_frame=det_logits_ds,
          annot_logits_per_frame=annot_logits,
          voxel_size=voxel_size,
          offset_maps_per_frame=offset_maps,
      )


def lr_at(step: int, total: int, base: float, warmup: int) -> float:
    """Linear warmup then cosine decay.

    The warmup is not cosmetic here. The model starts at a point that already
    fits the target almost exactly -- it produced it -- so the first gradients
    are small and noisy, and a full-size step on them is the most likely way to
    knock a good model off its own optimum before the adaptation has begun.
    """
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log_folder", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--preds", default="preds_r6_0.40",
                   help="folder under --log_folder holding this model's own "
                        "predictions on the target videos")
    p.add_argument("--embryo", default=None,
                   help="adapt on this embryo only; default = every sample the "
                        "preds folder covers")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--crop_size", type=int, nargs=3, default=None)
    p.add_argument("--windows-per-sample", type=int, default=1,
                   help="windows drawn per video; >=1 keeps every sample in")
    p.add_argument("--node-threshold", type=float, default=0.5,
                   help="detections at or above this become positives")
    p.add_argument("--edge-threshold", type=float, default=0.5)
    p.add_argument("--norm-only", action="store_true",
                   help="adapt only normalisation affines -- the conservative "
                        "variant, and the one that isolates the domain effect")
    p.add_argument("--eval-thresholds", type=float, nargs="+", default=(0.4, 0.5))
    p.add_argument("--eval-n", type=int, default=None,
                   help="score on this many videos (default: all)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--out", default=None, help="write the adapted weights here")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_folder = Path(args.log_folder)
    log_folder = log_folder if log_folder.is_absolute() else REPO / log_folder
    cfg = Config(json.load(open(log_folder / "args.json")))

    preds_dir = log_folder / args.preds
    names = sorted(q.stem for q in preds_dir.glob("*.npz"))
    if args.embryo:
        names = [n for n in names if n.split("_")[0] == args.embryo]
    if not names:
        raise SystemExit(f"no predictions under {preds_dir}")

    ckpt = Path(args.ckpt) if args.ckpt else log_folder / f"fold_{args.fold}" / "ckpt_final.pt"
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("model_state_dict", state)
    state = {k.replace("unet.module.", "unet.", 1): v for k, v in state.items()}
    model = build_model(cfg, None)
    model.load_state_dict(state, strict=True)
    model.to(device)

    train_dir = Path(TRAIN_DIR)
    train_dir = train_dir if train_dir.is_absolute() else REPO / "input" / "train"
    window_size = int(getattr(cfg, "window_size", 2))
    crop = tuple(args.crop_size) if args.crop_size else (
        tuple(getattr(cfg, "crop_size", None) or (32, 128, 128)))

    print(f"-> model     : {ckpt}")
    print(f"-> target    : {len(names)} videos, pseudo-labels from {preds_dir.name}")
    print(f"-> thresholds: node {args.node_threshold}, edge {args.edge_threshold}")
    print(f"-> schedule  : {args.steps} steps, lr {args.lr:g}, warmup {args.warmup}, "
          f"batch {args.batch_size}, crop {crop}, no augmentation")
    print(f"-> adapting  : {'normalisation affines only' if args.norm_only else 'all weights'}",
          flush=True)

    video_data, n_win = [], 0
    for n in names:
        got = pseudo_windows(
            train_dir / f"{n}.zarr", preds_dir / f"{n}.npz", window_size,
            args.node_threshold, args.edge_threshold,
            getattr(cfg, "normalization", None), args.windows_per_sample, rng,
        )
        if got is None:
            print(f"   ! {n}: no usable window, skipped")
            continue
        video_data.append(got)
        n_win += len(got[1])
    print(f"-> pool      : {n_win} windows over {len(video_data)} videos "
          f"({args.steps * args.batch_size} samples will be drawn)\n", flush=True)

    # evaluate_videos builds "<data_dir>/<path.name>.geff", so it wants stems,
    # and it reads data_dir off the config -- which stores it relative to src/.
    cfg.data_dir = str(train_dir)
    eval_paths = sorted(train_dir / n for n in names)
    if args.eval_n:
        eval_paths = eval_paths[: args.eval_n]

    def score(tag):
        model.eval()
        torch.cuda.empty_cache()
        r = evaluate_videos(model, eval_paths, cfg, device=device,
                            thresholds=tuple(args.eval_thresholds))
        for th in sorted(r):
            m = r[th]
            print(f"   {tag:<8} thr {th:g}  node_rec {m['node_recall']:.4f}  "
                  f"score {m['score']:.4f}", flush=True)
        return r

    print(f"scoring before adaptation ({len(eval_paths)} videos)")
    before = score("BEFORE")

    # No augmentation, no synthetic, no zebrahub -- the point is to see this
    # embryo as it is, not a perturbed version of it.
    from main_track import get_augs_config
    _, loader, _ = build_loader(
        data_dir=str(train_dir), embryos=[], window_size=window_size,
        batch_size=args.batch_size, num_workers=args.num_workers,
        downscale_factor=1, augment=False, shuffle=True,
        aug_cfg=get_augs_config(0), crop_size=crop, video_data=video_data,
        weak_positives_cfg={"use": False}, distributed=False,
        normalization=getattr(cfg, "normalization", None),
    )

    loss_cfg = dict(getattr(cfg, "loss_config", {}) or {})
    loss_cfg["det_use_weak_positives"] = False
    criterion = TrackingLoss(**loss_cfg)
    if args.norm_only:
        for q in model.parameters():
            q.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, (torch.nn.modules.batchnorm._BatchNorm,
                              torch.nn.GroupNorm, torch.nn.InstanceNorm3d,
                              torch.nn.LayerNorm)):
                for q in m.parameters(recurse=False):
                    q.requires_grad_(True)
    params = [q for q in model.parameters() if q.requires_grad]
    print(f"\n{sum(q.numel() for q in params)/1e6:.2f}M trainable parameters")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    use_amp = bool(getattr(cfg, "fp16", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    torch.cuda.empty_cache()

    model.train()
    step, it = 0, iter(loader)
    running = []
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        loss, breakdown = train_step(model, batch, criterion, cfg, device,
                                     window_size, use_amp=use_amp)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        running.append(float(loss.item()))
        step += 1
        if step % 25 == 0 or step == args.steps:
            print(f"  step {step:>4}/{args.steps}  lr {opt.param_groups[0]['lr']:.2e}  "
                  f"loss {np.mean(running[-25:]):.4f}  "
                  + "  ".join(f"{k}={v:.3f}" for k, v in breakdown.items()), flush=True)

    print(f"\nscoring after adaptation")
    after = score("AFTER")

    print(f"\n{'threshold':>10}{'node_rec':>22}{'score':>22}")
    print("-" * 56)
    for th in sorted(before):
        b, a = before[th], after[th]
        print(f"{th:>10g}"
              f"{b['node_recall']:>12.4f} ->{a['node_recall']:>8.4f}"
              f"{b['score']:>12.4f} ->{a['score']:>8.4f}"
              f"   {a['score'] - b['score']:+.4f}")

    if args.out:
        out = Path(args.out)
        out = out if out.is_absolute() else REPO / out
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, out)
        print(f"\n-> adapted weights: {out}")


if __name__ == "__main__":
    sys.exit(main())
