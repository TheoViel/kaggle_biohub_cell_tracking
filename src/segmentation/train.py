"""
Training script for CellSegNet.

Usage (from project root):
    python -m src.segmentation.train \
        --data_dir output/segs \
        --output_dir output/models \
        --epochs 200 \
        --lr 3e-4 \
        --sigma 3.0
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import CellSegDataset
from .model import CellSegNet
from .losses import CellSegLoss


def build_loader(data_dir, sigma, augment, patch_size, batch_size, num_workers, use_cellpose):
    ds = CellSegDataset(data_dir, sigma=sigma, augment=augment, patch_size=patch_size, use_cellpose=use_cellpose)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def train(args):
    if args.log_every < 1:
        raise ValueError("--log_every must be at least 1")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup_ratio must be in [0, 1)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_fp16 = args.fp16 and device.type == "cuda"
    if args.fp16 and not use_fp16:
        print("FP16 requested but CUDA is unavailable; using FP32")
    else:
        print(f"Precision: {'FP16 mixed precision' if use_fp16 else 'FP32'}")

    patch_size = (
        tuple(int(x) for x in args.patch_size.split(",")) if args.patch_size else None
    )

    train_loader = build_loader(
        args.data_dir,
        args.sigma,
        augment=True,
        patch_size=patch_size,
        use_cellpose=args.use_cellpose,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Dataset: {len(train_loader.dataset)} volumes, patch_size={patch_size}")

    model = CellSegNet(
        features=tuple(int(x) for x in args.features.split(",")),
        in_channels=2 if args.use_cellpose else 1,
    ).to(device)
    print(f"Model parameters: {model.count_parameters() / 1e6:.1f}M")

    criterion = CellSegLoss(
        w_fg=args.w_fg,
        w_center=args.w_center,
        w_offset=args.w_offset,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if available
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    scheduler_state = None
    latest = sorted(output_dir.glob("ckpt_epoch*.pt"))
    if latest:
        ckpt = torch.load(latest[-1], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt.get("global_step", start_epoch * len(train_loader))
        best_loss = ckpt.get("loss", best_loss)
        if ckpt.get("scheduler_type") == "linear_warmup":
            scheduler_state = ckpt.get("scheduler_state_dict")
        print(f"Resumed from {latest[-1].name} (epoch {start_epoch})")

    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_multiplier(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, (total_steps - step) / decay_steps)

    for group in optimizer.param_groups:
        group.setdefault("initial_lr", args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_multiplier,
        last_epoch=global_step - 1,
    )
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    interval_losses = {"fg": 0.0, "center": 0.0, "offset": 0.0, "total": 0.0}
    interval_steps = 0
    interval_start = time.time()
    last_avg = {"total": best_loss}

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {"fg": 0.0, "center": 0.0, "offset": 0.0, "total": 0.0}

        for batch_idx, batch in enumerate(train_loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            targets = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
                if k != "image"
            }

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_fp16,
            ):
                preds = model(image)
                loss, breakdown = criterion(preds, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            for k in epoch_losses:
                epoch_losses[k] += breakdown[k]
                interval_losses[k] += breakdown[k]
            interval_steps += 1

            if global_step % args.log_every == 0:
                avg_interval = {
                    k: value / interval_steps for k, value in interval_losses.items()
                }
                elapsed = time.time() - interval_start
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch {epoch+1:4d}/{args.epochs} - "
                    f"step {global_step:6d}/{total_steps} | "
                    f"lr={lr:.2e} | "
                    f"loss={avg_interval['total']:.3f} | "
                    f"fg={avg_interval['fg']:.3f} - "
                    f"center={avg_interval['center']:.3f} - "
                    f"offset={avg_interval['offset']:.3f}"
                )
                interval_losses = {
                    "fg": 0.0,
                    "center": 0.0,
                    "offset": 0.0,
                    "total": 0.0,
                }
                interval_steps = 0
                interval_start = time.time()

        n = len(train_loader)
        avg = {k: v / n for k, v in epoch_losses.items()}
        last_avg = avg

        is_best = avg["total"] < best_loss
        if is_best:
            best_loss = avg["total"]
        if is_best or (epoch + 1) % args.save_every == 0:
            tag = "best" if is_best else f"epoch{epoch+1:04d}"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_type": "linear_warmup",
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg["total"],
                    "args": vars(args),
                },
                output_dir / f"ckpt_{tag}.pt",
            )

    torch.save(
        {
            "epoch": args.epochs,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_type": "linear_warmup",
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": last_avg["total"],
            "args": vars(args),
        },
        output_dir / "ckpt_final.pt",
    )
    print(f"Training complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="output/segs")
    parser.add_argument("--output_dir", default="output/models")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FP16 mixed-precision training on CUDA (default: enabled)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Fraction of total steps used for linear LR warmup",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Gaussian sigma for heatmap targets (voxels)",
    )
    parser.add_argument(
        "--features",
        default="32,64,128,256,256",
        help="Comma-separated encoder feature counts per level",
    )
    parser.add_argument(
        "--patch_size",
        default=None,
        help="Random crop D,H,W e.g. 48,192,192. None = full volume",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--w_fg", type=float, default=1.0)
    parser.add_argument("--w_center", type=float, default=1.0)
    parser.add_argument(
        "--w_offset",
        type=float,
        default=0.05,
        help="Offset loss weight — small because offsets are in raw voxel units",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Print running losses every N optimizer steps",
    )
    parser.add_argument("--save_every", type=int, default=50)
    args = parser.parse_args()
    train(args)
