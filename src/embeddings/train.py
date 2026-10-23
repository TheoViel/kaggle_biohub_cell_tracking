"""The NETr training loop, shaped like ``tracking.train.fit``."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DistributedSampler

from embeddings.eval import evaluate_retrieval
from embeddings.losses import alignment_uniformity, nt_xent
from training.distributed import all_reduce_mean, unwrap_model
from training.optim import define_optimizer, define_scheduler

# top1_real / top1_synth are accumulated as (sum, count) pairs rather than as
# ratios: a rank whose interval held no real pair has no ratio to average, and
# a nan in an all_reduce poisons every rank's number.
LOSS_KEYS = ("loss", "top1", "top1_real", "n_real", "top1_synth", "n_synth")


def fit(
    model: nn.Module,
    train_loader,
    args,
    ctx,
    train_sampler: DistributedSampler | None = None,
    val_embryos: list[str] | None = None,
) -> None:
    """Train NETr with NT-Xent on two augmented views per frame."""

    def log(*message):
        if ctx.is_main:
            print(*message)

    device = ctx.device
    use_fp16 = getattr(args, "fp16", True) and device.type == "cuda"

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

    # Step budget, mirroring tracking.train.fit: `iterations` is the
    # fold-comparable one (folds see different numbers of frames), `epochs`
    # is the fallback when it is unset.
    steps_per_epoch = max(1, len(train_loader))
    iterations = int(getattr(args, "iterations", 0) or 0)
    if iterations > 0:
        total_steps = iterations
        n_epochs = -(-total_steps // steps_per_epoch)  # ceil
    else:
        n_epochs = int(args.epochs)
        total_steps = n_epochs * steps_per_epoch

    scheduler = define_scheduler(
        optimizer, train_loader, n_epochs, args.optimizer_config["warmup_prop"],
        total_steps=total_steps,
    )

    output_dir = Path(args.output_dir)
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    core = unwrap_model(model)

    def save(name):
        torch.save(
            {"model_state_dict": core.state_dict(), "model_config": args.model_config},
            output_dir / name,
        )

    # Accumulated on the GPU, not in Python floats. A float() on a CUDA tensor
    # blocks until the step has finished, so reading the loss every iteration
    # serialises the host against the device and throws away the queue depth
    # that lets the next batch's H2D copy overlap this step's compute. These are
    # only ever read at log time.
    def zero_interval():
        return {k: torch.zeros((), device=device) for k in LOSS_KEYS}

    interval = zero_interval()
    steps = 0
    start = time.time()
    global_step = 0

    log("\n-> Starting training...\n")
    for epoch in range(n_epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            patches1 = batch["patches_1"].to(device, non_blocking=True)
            patches2 = batch["patches_2"].to(device, non_blocking=True)
            if not use_fp16:
                # The loader always ships fp16; widen again for the FP32 path.
                patches1, patches2 = patches1.float(), patches2.float()
            coords1 = batch["coords_1"].to(device, non_blocking=True)
            coords2 = batch["coords_2"].to(device, non_blocking=True)
            mask1 = batch["mask_1"].to(device, non_blocking=True)
            mask2 = batch["mask_2"].to(device, non_blocking=True)
            pos12 = batch["pos_1to2"].to(device, non_blocking=True)
            pos21 = batch["pos_2to1"].to(device, non_blocking=True)
            is_real = batch["is_real"].to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_fp16
            ):
                z1 = model(patches1, coords1, mask1)
                z2 = model(patches2, coords2, mask2)
                out = nt_xent(
                    z1.float(), z2.float(), mask1, mask2, pos12, pos21,
                    temperature=args.temperature,
                )
                loss, top1 = out.loss, out.top1

            scaler.scale(loss).backward()
            if max_grad_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            model.zero_grad(set_to_none=True)
            if scaler.get_scale() >= scale:   # a skipped step must not advance the lr
                scheduler.step()
            global_step += 1

            interval["loss"] += loss.detach()
            interval["top1"] += top1.detach()
            # Split by pair kind: the two say different things, and the real
            # one is the only one measured on cells that actually moved.
            scored = out.sample_rows > 0
            for key, sel in (("real", scored & is_real), ("synth", scored & ~is_real)):
                interval[f"top1_{key}"] += (out.sample_top1 * sel).sum()
                interval[f"n_{key}"] += sel.sum()
            steps += 1

            is_last_step = global_step >= total_steps

            if global_step % args.log_every == 0 or is_last_step:
                avg = all_reduce_mean(
                    {k: float(v) / steps for k, v in interval.items()}, device
                )
                align, unif = alignment_uniformity(z1.float(), z2.float(), pos12)
                msg = (
                    f"epoch {epoch + 1:02d}/{n_epochs:02d}  (step {global_step:04d}/{total_steps})   "
                    f"lr={optimizer.param_groups[0]['lr']:.1e}   "
                    f"loss={avg['loss']:.4f}   top1={avg['top1']:.4f}"
                )
                # Both are means over the same ranks, so their ratio is the
                # global one; only the count needs guarding.
                for key in ("synth", "real"):
                    if avg[f"n_{key}"] > 0:
                        msg += f"   top1_{key}={avg[f'top1_{key}'] / avg[f'n_{key}']:.4f}"
                interval = zero_interval()
                steps = 0

                eval_due = (
                    val_embryos and args.eval_every > 0
                    and (global_step % args.eval_every == 0 or is_last_step)
                )
                if eval_due:
                    summary, _ = evaluate_retrieval(
                        model, args, val_embryos, ctx=ctx,
                        n_val_videos=args.n_val_videos,
                        max_pairs_per_video=args.max_val_pairs,
                        verbose=False,
                    )
                    msg += (
                        f"   val_top1={summary['top1']:.4f}   geom_top1={summary['top1_nn']:.4f}"
                    )
                    model.train()
                msg += f"   t={int(time.time() - start):03d}s"
                start = time.time()
                log(msg)

            if args.save_every > 0 and global_step % args.save_every == 0 and ctx.is_main:
                save(f"ckpt_{global_step:05d}.pt")

            # The budget can land mid-epoch, so stop here rather than at the
            # loader's end.
            if is_last_step:
                break

        if global_step >= total_steps:
            break

    if ctx.is_main:
        save("ckpt_final.pt")
        log("\n-> Saved weights to ", output_dir / "ckpt_final.pt")
