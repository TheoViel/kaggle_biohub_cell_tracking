"""The HOCT training loop, shaped like ``tracking.train.fit``."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from edge_transformer.eval import evaluate, evaluate_videos
from edge_transformer.losses import edge_loss
from training.distributed import all_reduce_mean, unwrap_model
from training.optim import define_optimizer, define_scheduler

LOSS_KEYS = ("loss", "degree", "n_supervised", "n_positive")


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    args,
    ctx,
    train_sampler: DistributedSampler | None = None,
    val_loader: DataLoader | None = None,
    val_dataset=None,
) -> None:
    """Train the edge transformer.

    ``val_loader`` gives the cheap window-level numbers every ``eval_every``
    steps; ``val_dataset`` gives the expensive full-video competition metrics,
    run every ``video_eval_every`` steps because it costs seconds a video rather
    than milliseconds a window.
    """

    def log(*message):
        if ctx.is_main:
            print(*message)

    device = ctx.device
    use_amp = bool(getattr(args, "fp16", True)) and device.type == "cuda"
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    core = unwrap_model(model)

    optimizer = define_optimizer(
        model,
        args.optimizer_config["name"],
        lr=args.optimizer_config["lr"],
        lr_encoder=args.optimizer_config.get("lr_encoder", args.optimizer_config["lr"]),
        betas=args.optimizer_config["betas"],
        weight_decay=args.optimizer_config["weight_decay"],
    )
    max_grad_norm = args.optimizer_config.get("max_grad_norm", 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not use_bf16)

    # `iterations` is the fold-comparable budget: the folds train on different
    # embryos and one holds nearly twice the videos of the other, so a fixed
    # epoch count hands them unequal numbers of gradient steps and their scores
    # stop being comparable. `epochs` is the fallback when it is unset.
    steps_per_epoch = max(1, len(train_loader))
    iterations = int(getattr(args, "iterations", 0) or 0)
    if iterations > 0:
        total_steps = iterations
        n_epochs = -(-total_steps // steps_per_epoch)
    else:
        n_epochs = int(args.epochs)
        total_steps = n_epochs * steps_per_epoch

    scheduler = define_scheduler(
        optimizer, train_loader, n_epochs,
        args.optimizer_config["warmup_prop"], total_steps=total_steps,
    )

    output_dir = Path(args.output_dir)
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    def save(name):
        torch.save(
            {"model_state_dict": core.state_dict(),
             "model_config": getattr(args, "model_config", {})},
            output_dir / name,
        )

    # Accumulated on the GPU: a float() on a CUDA tensor blocks the host until
    # the step has finished, and doing that every iteration serialises the host
    # against the device. Read only at log time.
    def zero_interval():
        return {k: torch.zeros((), device=device) for k in LOSS_KEYS}

    interval = zero_interval()
    steps = 0
    start = time.time()
    global_step = 0

    log("\n-> Starting training...\n")
    for epoch in range(n_epochs):
        model.train()
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            node_feats = batch["node_feats"].to(device, non_blocking=True)
            node_pos = batch["node_pos"].to(device, non_blocking=True)
            edge_pos = batch["edge_pos"].to(device, non_blocking=True)
            edge_index = batch["edge_index"].to(device, non_blocking=True)
            node_mask = batch["node_mask"].to(device, non_blocking=True)
            edge_mask = batch["edge_mask"].to(device, non_blocking=True)
            edge_feats = batch["edge_feats"].to(device, non_blocking=True)
            target = batch["edge_target"].to(device, non_blocking=True)
            supervised = batch["edge_supervised"].to(device, non_blocking=True)
            edge_weight = batch["edge_weight"].to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                out = model(
                    node_feats, node_pos, edge_pos, edge_index,
                    node_mask, edge_mask, edge_feats=edge_feats,
                )
                result = edge_loss(
                    out.edge_logits, target, supervised,
                    gamma=float(getattr(args, "focal_gamma", 2.0)),
                    pos_weight=float(getattr(args, "pos_weight", 1.0)),
                    rank_weight=float(getattr(args, "rank_weight", 0.0)),
                    degree_weight=float(getattr(args, "degree_weight", 0.0)),
                    edge_index=edge_index,
                    edge_mask=edge_mask,
                    weight=edge_weight,
                    n_nodes=node_feats.shape[1],
                )
                loss = result.loss

            scaler.scale(loss).backward()
            grad_norm = torch.tensor(0.0, device=device)
            if max_grad_norm:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # GradScaler only auto-skips non-finite gradients when fp16 loss
            # scaling is on; bf16 runs with the scaler disabled, so guard here.
            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            model.zero_grad(set_to_none=True)
            if scaler.get_scale() >= scale:   # a skipped step must not advance the lr
                scheduler.step()
            global_step += 1

            interval["loss"] += loss.detach()
            # Logged apart from the total: it is unsupervised and on a quite
            # different scale from the focal term, so a weight that looks small
            # can still dominate, and the log should say so.
            if result.degree is not None:
                interval["degree"] += result.degree.detach()
            interval["n_supervised"] += result.n_supervised
            interval["n_positive"] += result.n_positive
            steps += 1

            is_last_step = global_step >= total_steps

            if global_step % args.log_every == 0 or is_last_step:
                avg = all_reduce_mean(
                    {k: float(v) / steps for k, v in interval.items()}, device
                )
                interval = zero_interval()
                steps = 0

                eval_due = (
                    getattr(args, "eval_every", 0) > 0
                    and (global_step % args.eval_every == 0 or is_last_step)
                )
                val_msg = ""
                if eval_due and val_loader is not None:
                    m = evaluate(core, val_loader, device,
                                 threshold=float(getattr(args, "link_threshold", 0.5)),
                                 fp16=use_amp)
                    val_msg += f"   val_loss={m['loss']:.4f}  AP={m['ap']:.4f}  P={m['precision']:.3f} R={m['recall']:.3f}"
                    model.train()

                video_due = (
                    getattr(args, "video_eval_every", 0) > 0
                    and val_dataset is not None
                    and (global_step % args.video_eval_every == 0 or is_last_step)
                )
                if video_due:
                    v = evaluate_videos(
                        core, val_dataset, args.data_dir, device,
                        threshold=float(getattr(args, "link_threshold", 0.5)),
                        n_videos=(
                            int(args.n_val_videos)
                            if getattr(args, "n_val_videos", None) is not None
                            else None
                        ),
                        fp16=use_amp,
                    )
                    if v:
                        val_msg += f"   edge_jac={v['edge_jaccard']:.4f}  adj_edge_jac={v['adj_edge_jaccard']:.4f}"
                    model.train()

                msg = (
                    f"epoch {epoch + 1:02d}/{n_epochs:02d}  "
                    f"(step {global_step:04d}/{total_steps})   "
                    f"lr={optimizer.param_groups[0]['lr']:.1e}   "
                    f"loss={avg['loss']:.4f}   "
                ) + (
                    f"deg={avg['degree']:.3f}   "
                    if float(getattr(args, "degree_weight", 0.0)) else ""
                ) + (
                    f"t={int(time.time() - start):02d}s"
                ) + val_msg
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
        log("\n-> Saved weights to", output_dir / "ckpt_final.pt")
