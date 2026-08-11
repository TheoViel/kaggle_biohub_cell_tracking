"""
Training script for CellSegNet.

Usage (from project root):
    python -m src.segmentation.train \
        --data_dir output/segs \
        --output_dir output/models \
        --epochs 200 \
        --lr 3e-4 \
        --sigma 3.0

Multi-GPU (one process per GPU):
    torchrun --standalone --nproc_per_node=4 -m src.segmentation.train \
        --data_dir output/segs \
        --output_dir output/models \
        --batch_size 2

--batch_size is per process, so the effective batch is batch_size * nproc_per_node.
--num_workers is also per process.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .dataset import CellSegDataset
from .model import CellSegNet
from .losses import CellSegLoss
from .eval import evaluate
from .distributed import (
    all_reduce_mean,
    cleanup_distributed,
    init_distributed,
    seed_everything,
    unwrap_model,
    worker_init_fn,
)


def build_loader(
    data_dir,
    sigma,
    augment,
    patch_size,
    batch_size,
    num_workers,
    use_cellpose,
    ctx=None,
    seed=0,
):
    """Build the loader and, under torchrun, the sampler that shards it."""
    ds = CellSegDataset(data_dir, sigma=sigma, augment=augment, patch_size=patch_size, use_cellpose=use_cellpose)

    sampler = None
    if ctx is not None and ctx.distributed:
        sampler = DistributedSampler(
            ds,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=augment,
            seed=seed,
        )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=augment and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
    return loader, sampler


def train(args):
    if args.log_every < 1:
        raise ValueError("--log_every must be at least 1")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup_ratio must be in [0, 1)")

    ctx = init_distributed(getattr(args, "dist_backend", None))
    seed_everything(getattr(args, "seed", 0), ctx.rank)

    def log(*message):
        if ctx.is_main:
            print(*message)

    device = ctx.device
    log(f"Using device: {device}")
    if ctx.distributed:
        log(f"Distributed: {ctx.world_size} processes, batch_size={args.batch_size} each")
    use_fp16 = args.fp16 and device.type == "cuda"
    if args.fp16 and not use_fp16:
        log("FP16 requested but CUDA is unavailable; using FP32")
    else:
        log(f"Precision: {'FP16 mixed precision' if use_fp16 else 'FP32'}")

    patch_size = (
        tuple(int(x) for x in args.patch_size.split(",")) if args.patch_size else None
    )

    train_loader, train_sampler = build_loader(
        args.data_dir,
        args.sigma,
        augment=True,
        patch_size=patch_size,
        use_cellpose=args.use_cellpose,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ctx=ctx,
        seed=getattr(args, "seed", 0),
    )
    log(f"Dataset: {len(train_loader.dataset)} volumes, patch_size={patch_size}")

    model = CellSegNet(
        features=tuple(int(x) for x in args.features.split(",")),
        in_channels=args.in_channels,
    ).to(device)
    log(f"Model parameters: {model.count_parameters() / 1e6:.1f}M")

    if ctx.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    core_model = unwrap_model(model)

    criterion = CellSegLoss(
        w_fg=args.w_fg,
        w_center=args.w_center,
        w_offset=args.w_offset,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    output_dir = Path(args.output_dir)
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if available
    start_epoch = 0
    global_step = 0

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

    interval_losses = {"fg": 0.0, "center": 0.0, "offset": 0.0, "total": 0.0}
    interval_steps = 0
    interval_start = time.time()

    log('\n-> Starting training...\n')
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_losses = {"fg": 0.0, "center": 0.0, "offset": 0.0, "total": 0.0}

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(train_loader, start=1):
                image = batch["image"].to(device, non_blocking=True)
                targets = {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch.items()
                    if k != "image"
                }

                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                    preds = model(image)
                    loss, breakdown = criterion(preds, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 12.0)
                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()

                model.zero_grad(set_to_none=True)

                if scale == scaler.get_scale():
                    scheduler.step()
                global_step += 1

                for k in epoch_losses:
                    epoch_losses[k] += breakdown[k]
                    interval_losses[k] += breakdown[k]
                interval_steps += 1

                if global_step % args.log_every == 0:
                    avg_interval = all_reduce_mean(
                        {k: value / interval_steps for k, value in interval_losses.items()},
                        device,
                    )
                    elapsed = time.time() - interval_start
                    lr = optimizer.param_groups[0]["lr"]
                    message = (
                        f"epoch {epoch + 1:02d}/{args.epochs:02d}  (step {global_step:04d}/{total_steps})  "
                        f"lr={lr:.1e}   train_loss={avg_interval['total']:.3f}   t={elapsed:.0f}s"
                    )
                    if (args.eval_every > 0 and global_step % args.eval_every == 0 and ctx.is_main):
                        precision, recall, f1 = evaluate(core_model, args)
                        message += f"   det_f1={f1:.3f}   det_precision={precision:.3f}   det_recall={recall:.3f}"
                    log(message)
                    interval_losses = {
                        "fg": 0.0,
                        "center": 0.0,
                        "offset": 0.0,
                        "total": 0.0,
                    }
                    interval_steps = 0
                    interval_start = time.time()

                if global_step % args.log_every == 0 and ctx.is_main:
                    torch.save(
                        {"model_state_dict": core_model.state_dict()},
                        output_dir / f"ckpt_{global_step:05d}.pt",
                    )

        if ctx.is_main:
            torch.save(
                {"model_state_dict": core_model.state_dict()},
                output_dir / "ckpt_final.pt",
            )
            log('\n-> Saved weights to ', output_dir / "ckpt_final.pt")
            log(f"\nDone ! ")
    finally:
        cleanup_distributed()


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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per process; the effective batch is this times the process count",
    )
    parser.add_argument(
        "--use_cellpose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add Cellpose probabilities as a second model input channel",
    )
    parser.add_argument("--in_channels", type=int, default=1)
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Dataloader workers per process",
    )
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
    parser.add_argument(
        "--eval_every",
        type=int,
        default=0,
        help="Evaluate every N optimizer steps; 0 disables evaluation",
    )
    parser.add_argument("--eval_frames", type=int, default=10)
    parser.add_argument("--eval_sample_name", default="6bba_09961292")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed; each rank offsets it so augmentations differ",
    )
    parser.add_argument(
        "--dist_backend",
        default=None,
        help="Process group backend (default: nccl on CUDA, gloo otherwise)",
    )
    args = parser.parse_args()
    train(args)
