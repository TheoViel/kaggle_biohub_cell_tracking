"""Training loop for the cell tracking model."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from tracking.losses import TrackingLoss
from tracking.targets import detect_and_match, build_matched_edge_targets
from tracking.eval import evaluate

from training.optim import define_optimizer, define_scheduler
from training.distributed import all_reduce_mean, unwrap_model

from model_zoo.models_track import UNetNodeTransformer


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    args,
    ctx,
    train_sampler: DistributedSampler | None = None,
    val_loader: DataLoader | None = None,
) -> None:
    """Run the tracking training loop on a prebuilt model and dataloader."""

    def log(*message):
        if ctx.is_main:
            print(*message)

    device = ctx.device
    use_amp = args.fp16 and device.type == "cuda"
    # bf16 has the same exponent range as fp32, so unlike fp16 it can't
    # overflow to inf inside the model's own forward pass (e.g. from a
    # large-but-legitimate edge-loss gradient spike) — no loss scaling needed.
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    core_model: UNetNodeTransformer = unwrap_model(model)

    # Loss
    criterion = TrackingLoss(**args.loss_config)

    # Optimizer
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

    # Scheduler
    scheduler = define_scheduler(
        optimizer, train_loader, args.epochs, args.optimizer_config["warmup_prop"]
    )

    output_dir = Path(args.output_dir)
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    interval_losses = {"det": 0.0, "edge": 0.0, "total": 0.0}
    interval_steps = 0
    interval_start = time.time()

    log("\n-> Starting tracking training...\n")
    global_step = 0
    best_score = 0.0
    total_steps = args.epochs * len(train_loader)

    for epoch in range(args.epochs):
        model.train()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            imgs = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
            coords = batch["coords"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            image_shape = tuple(batch["image_shape"][0].tolist())
            voxel_size = tuple(batch["voxel_size"][0].tolist())

            bs, window_size, d, h, w = imgs.shape

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                unet_out, det_logits = core_model.encode(imgs)

                d_out, h_out, w_out = unet_out.shape[-3:]
                unet_scaling = (d // d_out, h // h_out, w // w_out)

                frame_det = []
                for i in range(window_size):
                    det_coords, det_pos_embed, det_mask, matches = detect_and_match(
                        det_logits[i].sigmoid(),
                        coords[:, i],
                        masks[:, i],
                        image_shape,
                        voxel_size=voxel_size,
                        unet_scaling=unet_scaling,
                        pool_kernel_um=args.pool_kernel_um,
                        frame_index=i,
                        window_size=window_size,
                        det_threshold=args.detect_and_match_threshold,
                        max_dets_per_sample=args.max_dets_per_sample,
                    )

                    unet_feat = core_model._index_features(
                        unet_out[:, i], det_coords, det_mask,
                    )
                    frame_det.append((det_coords, det_pos_embed, det_mask, matches, unet_feat))

                edge_logits_list = []
                edge_targets_list = []
                det_mask_pairs = []

                if global_step / total_steps > args.det_warmup_prop:
                    for i in range(window_size - 1):
                        ns = frame_det[i][0].shape[1]  # number of source nodes
                        nt = frame_det[i + 1][0].shape[1]  # number of target nodes

                        pair_target = build_matched_edge_targets(
                            frame_det[i][3],  # matches at frame i
                            frame_det[i + 1][3],  # matches at frame i+1
                            targets[:, i],  # gt transition matrix at frame i
                            ns,  # number of source nodes
                            nt,  # number of target nodes
                        )  # which node pairs are linked

                        # Forward of the edge scorer - score each node pair
                        edge_log = core_model.predict_edges(
                            unet_feat_src=frame_det[i][4],  # frame i UNet features
                            unet_feat_tgt=frame_det[i + 1][4],  # frame i+1 UNet features
                            coords_src=frame_det[i][0],  # frame i coords
                            coords_tgt=frame_det[i + 1][0],  # frame i+1 coords
                            pos_feat_src=frame_det[i][1],  # frame i pos-embed
                            pos_feat_tgt=frame_det[i + 1][1],  # frame i+1 pos-embed
                            mask_src=frame_det[i][2],  # frame i mask
                            mask_tgt=frame_det[i + 1][2],  # frame i+1 mask
                        )
                        edge_logits_list.append(edge_log)
                        edge_targets_list.append(pair_target)
                        det_mask_pairs.append((frame_det[i][2], frame_det[i + 1][2]))

                loss, breakdown = criterion(
                    det_logits,
                    [coords[:, i] for i in range(window_size)],
                    [masks[:, i] for i in range(window_size)],
                    edge_logits_list,
                    edge_targets_list,
                    det_mask_pairs,
                    imgs_per_frame=[imgs[:, i] for i in range(window_size)],
                    unet_scaling=unet_scaling,
                )

            scaler.scale(loss).backward()
            grad_norm = torch.tensor(0.0)
            if max_grad_norm:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # GradScaler only auto-skips non-finite grads when fp16 loss
            # scaling is active; bf16 runs with scaler disabled, so guard
            # explicitly here too (e.g. against a stray NaN/inf batch).
            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
            else:
                log(f"- Skipping step {global_step}: non-finite grad norm ({grad_norm.item()})")
            scale = scaler.get_scale()
            scaler.update()
            model.zero_grad(set_to_none=True)

            if scaler.get_scale() >= scale:
                scheduler.step()
            global_step += 1

            for k in interval_losses:
                interval_losses[k] += breakdown[k]
            interval_steps += 1

            if global_step % args.log_every == 0:
                avg = all_reduce_mean(
                    {k: v / interval_steps for k, v in interval_losses.items()},
                    device,
                )
                elapsed = time.time() - interval_start
                lr = optimizer.param_groups[0]["lr"]
                message = (
                    f"epoch {epoch + 1:02d}/{args.epochs:02d}  (step {global_step:04d})  "
                    f"lr={lr:.1e}   loss={avg['total']:.4f}  "
                    f"det={avg['det']:.4f}  edge={avg['edge']:.4f}  t={elapsed:.0f}s"
                )
                if (
                    getattr(args, "eval_every", 0) > 0
                    and global_step % args.eval_every == 0
                    and val_loader is not None
                ):
                    val_loss, val_acc, val_recall = evaluate(
                        core_model, val_loader, args, device=device,
                    )
                    if ctx.is_main:
                        score = val_acc * val_recall
                        message += (
                            f"   val_loss={val_loss:.4f}  "
                            f"val_acc={val_acc:.3f}  val_recall={val_recall:.3f}"
                        )
                        if score > best_score:
                            best_score = score
                            torch.save(
                                {"model_state_dict": core_model.state_dict(),
                                 "model_config": _model_config(args)},
                                output_dir / "ckpt_best.pt",
                            )
                    model.train()
                log(message)
                interval_losses = {"det": 0.0, "edge": 0.0, "total": 0.0}
                interval_steps = 0
                interval_start = time.time()

            if args.save_every > 0 and global_step % args.save_every == 0 and ctx.is_main:
                torch.save(
                    {"model_state_dict": core_model.state_dict()},
                    output_dir / f"ckpt_{global_step:05d}.pt",
                )

    if ctx.is_main:
        torch.save(
            {"model_state_dict": core_model.state_dict(),
             "model_config": _model_config(args)},
            output_dir / "ckpt_final.pt",
        )
        log("\n-> Saved weights to", output_dir / "ckpt_final.pt")
        log("\nDone!")


def _model_config(args) -> dict:
    return {
        "unet_out_channels": args.unet_out_channels,
        "unet_layers": list(args.unet_layers),
        "unet_pool_stride": list(getattr(args, "unet_pool_stride", [2, 2, 2])),
        "unet_n_decode_steps": getattr(args, "unet_n_decode_steps", None),
        "transformer_hidden": args.transformer_hidden,
        "transformer_heads": args.transformer_heads,
        "transformer_blocks": args.transformer_blocks,
        "pool_kernel_um": args.pool_kernel_um,
        "detect_and_match_threshold": args.detect_and_match_threshold,
        "max_dets_per_sample": args.max_dets_per_sample,
        "window_size": args.window_size,
    }
