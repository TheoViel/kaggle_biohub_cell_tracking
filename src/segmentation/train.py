import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from segmentation.dataset import CellSegDataset
from segmentation.losses import CellSegLoss
from segmentation.eval import evaluate

from training.optim import define_optimizer, define_scheduler
from training.distributed import (
    all_reduce_mean,
    cleanup_distributed,
    init_distributed,
    seed_everything,
    unwrap_model,
    worker_init_fn,
)

from model_zoo.models_seg import CELL_SEG_HEADS, define_model
from data.mix import Cutmix, Mixup


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
    ctx = init_distributed(getattr(args, "dist_backend", None))
    seed_everything(getattr(args, "seed", 0), ctx.rank)

    def log(*message):
        if ctx.is_main:
            print(*message)

    device = ctx.device
    if ctx.distributed:
        log(f"- Distributed: {ctx.world_size} processes, batch_size={args.batch_size} each")
    use_fp16 = args.fp16 and device.type == "cuda"
    log(f"- Precision: {'FP16 mixed precision' if use_fp16 else 'FP32'}")

    # Dataset
    train_loader, train_sampler = build_loader(
        args.data_dir,
        args.sigma,
        augment=True,
        patch_size=args.patch_size,
        use_cellpose=args.use_cellpose,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ctx=ctx,
        seed=getattr(args, "seed", 0),
    )
    log(f"- Training with {len(train_loader.dataset)} volumes  (patch_size={args.patch_size})")

    # Model
    model = define_model(
        decoder_name=args.decoder_name,
        encoder_name=args.encoder_name,
        n_channels=args.in_channels,
        pretrained=args.pretrained,
        heads=CELL_SEG_HEADS,
        use_3d=True,
        verbose=int(ctx.is_main),
    ).to(device)
    log(f"- Model: {args.decoder_name} / {args.encoder_name}  (n_parameters={model.count_parameters() / 1e6:.1f}M)")

    if ctx.distributed:
        if getattr(args, "sync_bn", False):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
        )
    core_model = unwrap_model(model)

    # Loss
    criterion = CellSegLoss(
        w_fg=args.w_fg,
        w_center=args.w_center,
        w_offset=args.w_offset,
    )

    # Optimizer
    optimizer = define_optimizer(
        model,
        args.optimizer_config["name"],
        lr=args.optimizer_config["lr"],
        lr_encoder=args.optimizer_config.get("lr_encoder", args.optimizer_config["lr"]),
        betas=args.optimizer_config["betas"],
        weight_decay=args.optimizer_config["weight_decay"],
    )
    max_grad_norm = args.optimizer_config.get("max_grad_norm", 0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # Scheduler
    scheduler = define_scheduler(
        optimizer, train_loader, args.epochs, args.optimizer_config["warmup_prop"]
    )

    # Batch-level augmentation
    mixup = Mixup(**args.mixup)
    cutmix = Cutmix(**args.cutmix)

    output_dir = Path(args.output_dir)
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    interval_losses = {"fg": 0.0, "center": 0.0, "offset": 0.0, "total": 0.0}
    interval_steps = 0
    interval_start = time.time()

    log('\n-> Starting training...\n')
    try:
        global_step = 0
        for epoch in range(args.epochs):
            model.train()
            epoch_losses = {"fg": 0.0, "center": 0.0, "offset": 0.0, "total": 0.0}

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(train_loader, start=1):
                batch = {
                    k: v.to(device, non_blocking=True) for k, v in batch.items()
                }

                batch = mixup(batch)
                batch = cutmix(batch)

                image = batch["image"]
                targets = {k: v for k, v in batch.items() if k != "image"}

                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                    preds = model(image)
                    loss, breakdown = criterion(preds, targets)

                scaler.scale(loss).backward()
                if max_grad_norm:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()

                model.zero_grad(set_to_none=True)

                if scaler.get_scale() >= scale:  # skipped steps must not advance the lr
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
                        f"epoch {epoch + 1:02d}/{args.epochs:02d}  (step {global_step:04d})  "
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

                if args.save_every > 0 and global_step % args.save_every == 0 and ctx.is_main:
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
