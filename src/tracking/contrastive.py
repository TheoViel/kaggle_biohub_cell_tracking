"""Stage 2: train a second UNet to tell cells apart.

The detection UNet is trained to answer "is a cell here", which gives two
look-alike neighbours the *same* feature — correct for detection, useless for
deciding which of them a cell became one frame later. This stage trains a
separate UNet whose only job is that second question: a cell's embedding at t
must be closer to its own embedding at t + 1 than to any other candidate's.

Detection is frozen throughout, so the node coordinates are constant and get
cached exactly as in transformer-only training; only the embedding UNet moves.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from tracking.fit_utils import (
    collate_frame_det,
    encode_nodes,
    set_frozen_eval,
    split_frame_det,
)
from tracking.targets import build_matched_edge_targets
from training.distributed import all_reduce_mean, unwrap_model
from training.optim import define_scheduler


def _hard_negative_mask(
    coords_src: torch.Tensor,   # (B, N_src, 3) detection-grid coords
    coords_tgt: torch.Tensor,   # (B, N_tgt, 3)
    mask_src: torch.Tensor,     # (B, N_src) bool
    gt_parent: torch.Tensor,    # (B, N_tgt) index of the true parent
    voxel_size: torch.Tensor | None,  # (3,) um per grid voxel; None = raw voxels
    k: int,
) -> torch.Tensor:              # (B, N_src, N_tgt) bool — keep in the denominator
    """Each child's ``k`` nearest candidate parents, plus its true one.

    The negatives that matter are the ones the tracker could actually confuse:
    a cell 100 um away is ruled out by geometry no matter what it looks like,
    but it still takes a share of the InfoNCE denominator. Keeping only near
    candidates spends the embedding's capacity on the decision that is
    genuinely ambiguous.

    The true parent is force-kept even when it falls outside the k nearest —
    dropping it would leave the positive at -inf and the loss infinite.

    Without ``voxel_size`` the ranking is done in raw grid voxels, which agrees
    with the physical one as long as the detection grid is isotropic (it is,
    by construction of unet_pool_stride — see _knn_keep).
    """
    scale = 1.0 if voxel_size is None else voxel_size
    d = torch.cdist(coords_tgt.float() * scale, coords_src.float() * scale)
    d = d.masked_fill(~mask_src.unsqueeze(1), float("inf"))     # (B, N_tgt, N_src)

    k = min(k, coords_src.shape[1])
    keep = torch.zeros_like(d, dtype=torch.bool)
    keep.scatter_(-1, d.topk(k, dim=-1, largest=False).indices, True)
    keep.scatter_(-1, gt_parent.unsqueeze(-1), True)
    return keep.transpose(1, 2)


def contrastive_loss(
    embed_src: torch.Tensor,   # (B, N_src, D) L2-normalised
    embed_tgt: torch.Tensor,   # (B, N_tgt, D) L2-normalised
    target: torch.Tensor,      # (B, N_src, N_tgt) GT links in node-index space
    mask_src: torch.Tensor,    # (B, N_src) bool
    mask_tgt: torch.Tensor,    # (B, N_tgt) bool
    temperature: float = 0.1,
    coords_src: torch.Tensor | None = None,
    coords_tgt: torch.Tensor | None = None,
    voxel_size: torch.Tensor | None = None,
    hard_negatives_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """InfoNCE over candidate parents, plus the top-1 retrieval rate.

    Every child node with a GT parent is one classification problem: pick that
    parent out of the detected nodes in the previous frame. Softmax runs over
    the parent axis, matching ``edge_loss``, so the two stages optimise the
    same decision — one with cosine similarity, one with a learned scorer.

    With ``hard_negatives_k`` the loss only ranks each child against its k
    nearest candidates; ``top1`` is always measured against *all* of them, so
    the reported number stays comparable across settings.

    Returns ``(loss, top1, n_children)``; ``top1`` is the fraction of children
    whose GT parent is the most similar candidate, which is the number to watch.
    """
    sim = torch.bmm(embed_src, embed_tgt.transpose(1, 2)) / temperature
    sim = sim.masked_fill(~mask_src.unsqueeze(2), float("-inf"))

    supervised = (target.sum(dim=1) > 0) & mask_tgt      # (B, N_tgt)
    gt_parent = target.argmax(dim=1)                     # (B, N_tgt)
    n = int(supervised.sum())
    if n == 0:
        # Keep the graph alive so DDP still sees a gradient for every parameter.
        return sim.nan_to_num().sum() * 0.0, torch.zeros((), device=sim.device), 0

    # Reported on the full candidate set, whatever the loss ranks against.
    top1 = ((sim.argmax(dim=1) == gt_parent) & supervised).sum() / n

    sim_loss = sim
    if hard_negatives_k is not None and coords_src is not None:
        keep = _hard_negative_mask(
            coords_src, coords_tgt, mask_src, gt_parent, voxel_size, hard_negatives_k
        )
        sim_loss = sim.masked_fill(~keep, float("-inf"))

    # Children with no valid parent leave an all -inf column, whose softmax is
    # NaN; they are unsupervised, but NaN * 0 would still spread.
    log_p = torch.log_softmax(sim_loss, dim=1).nan_to_num()
    picked = log_p.gather(1, gt_parent.unsqueeze(1)).squeeze(1)   # (B, N_tgt)
    loss = -(picked * supervised).sum() / n

    return loss, top1, n


def fit_contrastive(
    model: torch.nn.Module,
    train_loader: DataLoader,
    args,
    ctx,
    train_sampler: DistributedSampler | None = None,
    val_loader: DataLoader | None = None,
) -> None:
    """Train the embedding UNet; everything else is frozen."""

    def log(*message):
        if ctx.is_main:
            print(*message)

    config = args.contrastive
    device = ctx.device
    use_amp = args.fp16 and device.type == "cuda"
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    core_model = unwrap_model(model)

    trainable = ("embed_unet.", "embed_head.")
    for name, param in core_model.named_parameters():
        param.requires_grad_(name.startswith(trainable))
    # Gradients alone would leave the frozen detector's BatchNorm statistics
    # drifting on every forward, changing the node coordinates mid-stage.
    set_frozen_eval(core_model, trainable)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config["lr"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not use_bf16)
    scheduler = define_scheduler(optimizer, train_loader, config["epochs"], 0.0)
    max_grad_norm = args.optimizer_config.get("max_grad_norm", 1.0)

    # Node coordinates come from the frozen detector, so they never change.
    node_cache: dict[str, list] = {}
    grid_voxel_size: torch.Tensor | None = None

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"- Contrastive: training embed UNet ({n_train / 1e6:.1f}M params), "
        f"temperature={config['temperature']}, dim={core_model.embed_dim}")

    global_step = 0
    interval = {"loss": 0.0, "top1": 0.0, "steps": 0}
    interval_start = time.time()
    total_steps = config["epochs"] * len(train_loader)

    for epoch in range(config["epochs"]):
        model.train()
        set_frozen_eval(core_model, trainable)
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            imgs = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
            coords = batch["coords"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            image_shape = tuple(batch["image_shape"][0].tolist())
            voxel_size = tuple(batch["voxel_size"][0].tolist())
            window_size = imgs.shape[1]

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                names = batch["name"]
                if any(name not in node_cache for name in names):
                    with torch.no_grad():
                        fresh, _, scaling = encode_nodes(
                            core_model, imgs, coords, masks, image_shape, voxel_size, args, node_features=False,
                        )
                    for b, name in enumerate(names):
                        node_cache[name] = split_frame_det(fresh, b)
                    if grid_voxel_size is None:
                        # Node coords live on the detection grid, so distances
                        # need the grid's voxel size, not the image's.
                        grid_voxel_size = torch.tensor(
                            [v * s_ for v, s_ in zip(voxel_size, scaling)],
                            device=device, dtype=torch.float32,
                        )
                frame_det = collate_frame_det([node_cache[n] for n in names])

                embed = core_model.encode_embed(imgs)      # (B, W, D, *spatial)
                node_embed = [
                    core_model._index_features(embed[:, i], frame_det[i][0], frame_det[i][2])
                    for i in range(window_size)
                ]

                loss, top1, n_children = 0.0, 0.0, 0
                for i in range(window_size - 1):
                    pair_target = build_matched_edge_targets(
                        frame_det[i][3],
                        frame_det[i + 1][3],
                        targets[:, i],
                        frame_det[i][0].shape[1],
                        frame_det[i + 1][0].shape[1],
                    )
                    pair_loss, pair_top1, n = contrastive_loss(
                        node_embed[i],
                        node_embed[i + 1],
                        pair_target,
                        frame_det[i][2],
                        frame_det[i + 1][2],
                        temperature=config["temperature"],
                        coords_src=frame_det[i][0],
                        coords_tgt=frame_det[i + 1][0],
                        voxel_size=grid_voxel_size,
                        hard_negatives_k=config.get("hard_negatives_k"),
                    )
                    loss = loss + pair_loss
                    top1 = top1 + float(pair_top1.detach()) * n
                    n_children += n
                loss = loss / max(window_size - 1, 1)

            scaler.scale(loss).backward()
            if max_grad_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            interval["loss"] += float(loss.detach())
            interval["top1"] += top1 / max(n_children, 1)
            interval["steps"] += 1

            is_last = global_step >= total_steps
            if global_step % args.log_every == 0 or is_last:
                avg = all_reduce_mean(
                    {k: interval[k] / interval["steps"] for k in ("loss", "top1")},
                    device,
                )
                message = (
                    f"epoch {epoch + 1:02d}/{config['epochs']:02d}  (step {global_step:05d})  "
                    f"  lr={optimizer.param_groups[0]['lr']:.1e}  loss={avg['loss']:.4f}  top1={avg['top1']:.3f}"
                )
                if val_loader is not None:
                    val_top1 = evaluate_contrastive(core_model, val_loader, args, device)
                    message += f"   val_top1={val_top1:.3f}"
                    model.train()
                    set_frozen_eval(core_model, trainable)
                message += f"  t={time.time() - interval_start:.0f}s"
                log(message)
                interval = {"loss": 0.0, "top1": 0.0, "steps": 0}
                interval_start = time.time()

    if ctx.is_main:
        torch.save(
            {"model_state_dict": core_model.state_dict()},
            Path(args.output_dir) / "ckpt_contrastive.pt",
        )


@torch.no_grad()
def evaluate_contrastive(model, loader: DataLoader, args, device) -> float:
    """Fraction of GT children whose true parent is their nearest embedding."""
    model.eval()
    n_right, n_total = 0, 0

    for batch in loader:
        imgs = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        image_shape = tuple(batch["image_shape"][0].tolist())
        voxel_size = tuple(batch["voxel_size"][0].tolist())
        window_size = imgs.shape[1]

        frame_det, _, _ = encode_nodes(
            model, imgs, coords, masks, image_shape, voxel_size, args,
            nms=getattr(args, "nms", "greedy"),
            nms_radius_um=getattr(args, "nms_radius_um", 7.0),
            node_features=False,
        )
        embed = model.encode_embed(imgs)
        node_embed = [
            model._index_features(embed[:, i], frame_det[i][0], frame_det[i][2])
            for i in range(window_size)
        ]

        for i in range(window_size - 1):
            pair_target = build_matched_edge_targets(
                frame_det[i][3], frame_det[i + 1][3], targets[:, i],
                frame_det[i][0].shape[1], frame_det[i + 1][0].shape[1],
            )
            _, top1, n = contrastive_loss(
                node_embed[i], node_embed[i + 1], pair_target,
                frame_det[i][2], frame_det[i + 1][2],
                temperature=args.contrastive["temperature"],
            )
            n_right += float(top1) * n
            n_total += n

    totals = all_reduce_mean({"right": n_right, "total": n_total}, device)
    model.train()
    return totals["right"] / max(totals["total"], 1e-6)
