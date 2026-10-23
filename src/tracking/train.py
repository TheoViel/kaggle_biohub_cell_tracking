"""Training loop for the cell tracking model."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from data.mix import Mixup
from tracking.losses import TrackingLoss
from tracking.targets import build_matched_edge_targets
from tracking.eval import evaluate, evaluate_videos
from tracking.fit_utils import (
    collate_frame_det,
    encode_nodes,
    encode_nodes_precomputed,
    fts_cache_dir,
    set_frozen_eval,
    load_fts_cache,
    loader_without_images,
    save_fts_cache,
    split_frame_det,
)

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
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    core_model: UNetNodeTransformer = unwrap_model(model)

    # Which of the three training stages this call is running, for the log
    # line and the checkpoint names. k_fold sets it per stage.
    stage = getattr(args, "stage", "edge")

    # Full-video evaluation, used in place of the window-level one when the
    # caller supplies videos to run it on. k_fold sets it for the detection
    # stage and for the edge stage, with args.eval_edge_mode picking how links
    # are built in each: nearest while the edge head is at init, the
    # transformer once it is trained.
    eval_video_paths = list(getattr(args, "eval_video_paths", None) or [])
    eval_thresholds = tuple(
        getattr(args, "eval_thresholds", None) or (0.2, 0.3, 0.4)
    )

    # Loss. offset_radius comes from the model flag, not loss_config: the head
    # is bounded by it, so a target built at a different radius would ask for a
    # vote the head cannot make.
    loss_config = dict(args.loss_config)
    loss_config["offset_radius"] = int(getattr(args, "offset_radius", 2))
    criterion = TrackingLoss(**loss_config)

    # Transformer-only mode
    transformer_only = getattr(args, "transformer_only", False)
    fts_cache: dict[str, list] = {}
    val_fts_cache: dict[str, list] = {}
    cache_dir = fts_cache_dir(args, getattr(args, "fold", 0))
    # Set explicitly rather than only when freezing: an earlier stage may have
    # left its own parameters frozen, and this stage owns the decision.
    if transformer_only:
        for name, param in core_model.named_parameters():
            param.requires_grad_(name.startswith("transformer."))
        set_frozen_eval(core_model, ("transformer",))
        # log(f"- Transformer-only: UNet + detect head frozen, node features cached in {cache_dir}")
        # Both UNets are frozen, so features computed by an earlier run against
        # the same weights and detection config are still exact.
        fts_cache = load_fts_cache(cache_dir, "train", args, ctx, device)
        val_fts_cache = load_fts_cache(cache_dir, "val", args, ctx, device)
        # if fts_cache or val_fts_cache:
        #     log(f"- Loaded node features from {cache_dir} "
        #         f"({len(fts_cache)} train, {len(val_fts_cache)} val windows)")
    else:
        for param in core_model.parameters():
            param.requires_grad_(True)

    # Freeze the UNet for the opening fraction of the stage, then hand it back.
    #
    # Stage 3 starts with a *randomly initialised* transformer -- `load_backbone`
    # filters `transformer.` out of whatever checkpoint seeded the backbone --
    # so on the first steps the edge loss is nearly pure noise. Training the two
    # together from step 0 pushes that noise straight into a pretrained
    # detector at `lr_unet`, and it does not survive: node recall fell from
    # ~0.98 to 0.465 inside 100 steps at 9e-4, then spent the rest of the run
    # climbing back. The warmup lets the head become worth listening to before
    # anything it says reaches the backbone.
    #
    # Not the same knob as `optimizer_config.warmup_prop`, which ramps the
    # learning rate for everything. This one is about *who* is being trained.
    tf_warmup = float(getattr(args, "transformer_warmup_prop", 0.0) or 0.0)
    backbone_frozen = bool(tf_warmup > 0.0) and not transformer_only
    if backbone_frozen:
        for name, param in core_model.named_parameters():
            param.requires_grad_(name.startswith("transformer."))
        # eval() as well as requires_grad_: BatchNorm's running statistics are
        # buffers updated on the forward pass, so a detector frozen by
        # gradients alone still drifts -- see `set_frozen_eval`.
        set_frozen_eval(core_model, ("transformer",))

    # Mixup
    mix_config = getattr(args, "mixup", None) or {}
    mixer = None
    if mix_config.get("p", 0.0) > 0:
        mixer = Mixup(p=mix_config["p"], alpha=mix_config.get("alpha", 4.0))
        # log(f"- Mixup: p={mix_config['p']}, alpha={mix_config.get('alpha', 4.0)}")

    # Optimizer
    lr = args.optimizer_config["lr"]
    lr_unet = args.optimizer_config.get("lr_unet", lr)
    lr_transformer = args.optimizer_config.get("lr_transformer", lr)
    optimizer = define_optimizer(
        model,
        args.optimizer_config["name"],
        lr=lr,
        lr_encoder=args.optimizer_config.get("lr_encoder", lr),
        lr_by_name={
            "transformer": lr_transformer,
            "unet": lr_unet,
            "detect_head": lr_unet,
            # A detection-side head, so it follows the backbone's rate rather
            # than the edge scorer's.
            "offset_head": lr_unet,
            "annot_head": lr_unet,
        },
        betas=args.optimizer_config["betas"],
        weight_decay=args.optimizer_config["weight_decay"],
    )
    max_grad_norm = args.optimizer_config.get("max_grad_norm", 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not use_bf16)

    # Step budget. `iterations` is the fold-comparable one: the folds train on
    # different embryos and 6bba has 128 videos against 44b6's 71, so a fixed
    # epoch count silently hands one fold ~1.8x the gradient steps of the
    # other and their scores stop being comparable. `epochs` is the fallback
    # when iterations is unset, so older configs keep their behaviour.
    steps_per_epoch = max(1, len(train_loader))
    iterations = int(getattr(args, "iterations", 0) or 0)
    if iterations > 0:
        total_steps = iterations
        # Ceil: the last pass over the loader stops early, mid-epoch.
        n_epochs = -(-total_steps // steps_per_epoch)
    else:
        n_epochs = int(args.epochs)
        total_steps = n_epochs * steps_per_epoch

    # Scheduler
    scheduler = define_scheduler(
        optimizer,
        train_loader,
        n_epochs,
        args.optimizer_config["warmup_prop"],
        total_steps=total_steps,
    )

    output_dir = Path(args.output_dir)
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    # "offset" is absent on a model without the head, so the keys are taken
    # from each breakdown rather than fixed here.
    interval_losses = {"det": 0.0, "edge": 0.0, "total": 0.0}
    interval_steps = 0
    interval_start = time.time()

    global_step = 0
    val_cache_saved = bool(val_fts_cache)
    # log(f"- Training for {total_steps} steps "
    #     f"({steps_per_epoch} per epoch, {n_epochs} epoch(s)"
    #     f"{', iteration-bounded' if iterations > 0 else ''})")

    for epoch in range(n_epochs):
        model.train()
        if transformer_only or backbone_frozen:
            set_frozen_eval(core_model, ("transformer",))

        # Once the cache covers every window this rank will ever be given, the
        # volumes are dead weight. Verified rather than assumed: a sampler that
        # re-partitions across epochs (DistributedSampler with shuffle=True)
        # would hand this rank windows it never encoded, and they would reach
        # the UNet as empty placeholders. Checked every epoch, so a cache loaded
        # from disk skips image loading from the very first one.
        if transformer_only and train_loader.dataset.load_images:
            n_shard = len(train_loader.sampler or train_loader.dataset)
            if len(fts_cache) >= n_shard:
                train_loader = loader_without_images(train_loader)
                save_fts_cache(fts_cache, cache_dir, "train", args, ctx)

        # SubsetRandomSampler (the stable-shard path) reshuffles on its own.
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            imgs = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
            coords = batch["coords"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            weak_pos = batch.get("weak_positives")
            if weak_pos is not None:
                weak_pos = weak_pos.to(device, non_blocking=True)
            ignore_pos = batch.get("ignore_positives")
            if ignore_pos is not None:
                ignore_pos = ignore_pos.to(device, non_blocking=True)
            image_shape = tuple(batch["image_shape"][0].tolist())
            voxel_size = tuple(batch["voxel_size"][0].tolist())

            if mixer is not None:
                imgs, coords, masks, targets, weak_pos = mixer.mix_track(
                    imgs, coords, masks, targets, weak_pos
                )

            bs, window_size, d, h, w = imgs.shape

            # Nodes from a predictions folder: no U-Net, no detection, no
            # augmentation to keep in step with. Set on the stage's config as
            # `coordinates_folder`, and the loader has already read them.
            pre_coords = batch.get("pre_coords")
            precomputed_nodes = pre_coords is not None and pre_coords.shape[-2] > 0
            if precomputed_nodes:
                pre_coords = pre_coords.to(device, non_blocking=True)
                pre_emb = batch["pre_emb"].to(device, non_blocking=True)

            progress = global_step / max(total_steps, 1)
            if backbone_frozen and progress >= tf_warmup:
                backbone_frozen = False
                for param in core_model.parameters():
                    param.requires_grad_(True)
                # Restores train mode on the modules `set_frozen_eval` put into
                # eval; the optimiser already holds these parameters, since
                # `define_optimizer` collects them all regardless of
                # `requires_grad`, so they simply start moving again.
                model.train()
            teacher_forcing = progress < getattr(args, "teacher_forcing_prop", 0.0)
            # `backbone_frozen` forces the edges on for the same reason
            # `transformer_only` does: with the UNet frozen the edge head is the
            # only thing that can learn, so a step without an edge loss has no
            # trainable parameter in it at all. That is not merely wasted work
            # -- the loss comes back with no `grad_fn` and `backward()` raises.
            # It bites on step 0 of every run, where `progress` is 0.0 and
            # `0.0 > det_warmup_prop` is false even at the default of 0.0.
            edges_enabled = (
                transformer_only
                or backbone_frozen
                or (global_step / total_steps > args.det_warmup_prop)
            )

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if precomputed_nodes:
                    frame_det = encode_nodes_precomputed(
                        core_model, pre_coords, pre_emb, coords, masks,
                        image_shape, voxel_size, args,
                    )
                    # Nothing on the detection side is being supervised: there
                    # is no heatmap and no U-Net in the graph at all.
                    det_logits, unet_scaling, offset_maps = [], None, None
                    det_logits_ds, det_stride, annot_logits = None, None, None
                    div_logits = None
                elif transformer_only:
                    names = batch["name"]
                    if any(name not in fts_cache for name in names):
                        with torch.no_grad():
                            fresh = encode_nodes(
                                core_model, imgs, coords, masks, image_shape,
                                voxel_size, args,
                            )[0]
                        for b, name in enumerate(names):
                            fts_cache[name] = split_frame_det(fresh, b)
                    frame_det = collate_frame_det([fts_cache[n] for n in names])
                    # The detector is frozen here, so neither the heatmap nor
                    # the vote is being supervised.
                    det_logits, unet_scaling, offset_maps = [], None, None
                    det_logits_ds, det_stride, annot_logits = None, None, None
                    div_logits = None
                else:
                    (frame_det, det_logits, unet_scaling, offset_maps,
                     det_logits_ds, det_stride, annot_logits,
                     div_logits) = encode_nodes(
                        core_model,
                        imgs,
                        coords,
                        masks,
                        image_shape,
                        voxel_size,
                        args,
                        teacher_forcing=teacher_forcing,
                        detect=edges_enabled,
                    )

                edge_logits_list = []
                edge_targets_list = []
                det_mask_pairs = []

                if edges_enabled:
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
                        # Present only on the precomputed path; the U-Net path's
                        # six-tuple has no embeddings to offer.
                        emb_s = frame_det[i][6] if len(frame_det[i]) > 6 else None
                        emb_t = frame_det[i + 1][6] if len(frame_det[i + 1]) > 6 else None
                        val_s = frame_det[i][7] if len(frame_det[i]) > 7 else None
                        val_t = frame_det[i + 1][7] if len(frame_det[i + 1]) > 7 else None

                        edge_log = core_model.predict_edges(
                            unet_feat_src=frame_det[i][4],  # frame i UNet features
                            unet_feat_tgt=frame_det[i + 1][4],  # frame i+1 UNet features
                            coords_src=frame_det[i][0],  # frame i coords
                            coords_tgt=frame_det[i + 1][0],  # frame i+1 coords
                            pos_feat_src=frame_det[i][1],  # frame i pos-embed
                            pos_feat_tgt=frame_det[i + 1][1],  # frame i+1 pos-embed
                            mask_src=frame_det[i][2],  # frame i mask
                            mask_tgt=frame_det[i + 1][2],  # frame i+1 mask
                            emb_src=emb_s,
                            emb_tgt=emb_t,
                            emb_valid_src=val_s,
                            emb_valid_tgt=val_t,
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
                    unet_scaling=unet_scaling,
                    # The heatmap's own grid, finer than the node grid when the
                    # backbone decodes past it; equal otherwise.
                    det_stride=det_stride,
                    det_logits_ds_per_frame=det_logits_ds,
                    annot_logits_per_frame=annot_logits,
                    div_logits_per_frame=div_logits,
                    # The splat is sized in grid voxels; the loss needs the
                    # physical voxel to keep it round on a non-cubic grid.
                    voxel_size=voxel_size,
                    weak_pos_per_frame=(
                        [weak_pos[:, i] for i in range(window_size)]
                        if weak_pos is not None else None
                    ),
                    ignore_pos_per_frame=(
                        [ignore_pos[:, i] for i in range(window_size)]
                        if ignore_pos is not None else None
                    ),
                    offset_maps_per_frame=offset_maps,
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
            # else:
            #     log(f"- Skipping step {global_step}: non-finite grad norm ({grad_norm.item()})")
            scale = scaler.get_scale()
            scaler.update()
            model.zero_grad(set_to_none=True)

            if scaler.get_scale() >= scale:
                scheduler.step()
            global_step += 1

            for k, v in breakdown.items():
                interval_losses[k] = interval_losses.get(k, 0.0) + v
            interval_steps += 1

            is_last_step = global_step >= total_steps

            if global_step % args.log_every == 0 or is_last_step:
                avg = all_reduce_mean(
                    {k: v / interval_steps for k, v in interval_losses.items()},
                    device,
                )
                lrs = sorted({g["lr"] for g in optimizer.param_groups})
                lr_str = "/".join(f"{v:.1e}" for v in lrs)

                # Evaluate
                val_message = ""
                eval_due = (
                    getattr(args, "eval_every", 0) > 0
                    and (global_step % args.eval_every == 0 or is_last_step)
                )
                # Full-video inference runs the U-Net's own detection, which a
                # model taking its nodes (and their embeddings) from a folder
                # has no way to feed. Scoring it would need the predict path to
                # read those folders too -- see the note in predict.py. The
                # window-level `evaluate` above does use the precomputed nodes
                # and still runs, so the stage is not flying blind.
                if eval_due and eval_video_paths:
                    # Score the real thing: full-video inference through the
                    # actual predict path, then the competition metrics. The
                    # window-level numbers saturate -- node recall at the 7 um
                    # tolerance cannot rank two detectors. k_fold sets the
                    # paths per stage, and eval_edge_mode with them: nearest
                    # links while the edge head is still at init (stage 1),
                    # the transformer once it is trained (stage 3).
                    per_th = evaluate_videos(
                        core_model,
                        eval_video_paths,
                        args,
                        device=device,
                        thresholds=eval_thresholds,
                    )
                    val_message = "".join(
                        f"  [thr {t:g}] node_rec={m['node_recall']:.3f}"
                        f" edge_jac={m['edge_jaccard']:.3f}"
                        f" score={m['score']:.3f}"
                        for t, m in sorted(per_th.items())
                    )
                    model.train()
                    if transformer_only or backbone_frozen:
                        set_frozen_eval(core_model, ("transformer",))
                elif eval_due and val_loader is not None:
                    (val_loss, val_edge_jaccard, val_node_recall,
                     val_adjust, val_edge_recall, val_node_recall_tight) = evaluate(
                        core_model,
                        val_loader,
                        args,
                        device=device,
                        eval_edges=edges_enabled,
                        transformer_only=transformer_only,
                        fts_cache=val_fts_cache,
                    )
                    # evaluate() walks the whole loader, so one pass fills the
                    # val cache completely.
                    if transformer_only and not val_cache_saved:
                        save_fts_cache(val_fts_cache, cache_dir, "val", args, ctx)
                        val_cache_saved = True
                    tight_um = float(getattr(args, "eval_node_match_um", 0.0) or 0.0)
                    val_message = "".join(
                        f"  [thr {t:g}] node_rec={val_node_recall[t]:.3f}"
                        + (f" rec@{tight_um:g}um={val_node_recall_tight[t]:.3f}" if val_node_recall_tight else "")
                        + f" adjust={val_adjust[t]:.3f}"
                        + ("" if stage == "seg" else  f" edge_jac={val_edge_jaccard[t]:.3f}")
                        for t in sorted(val_node_recall)
                    )
                    if val_loss:
                        val_message = f"  val_loss={val_loss:.4f}" + val_message

                    model.train()
                    if transformer_only or backbone_frozen:
                        set_frozen_eval(core_model, ("transformer",))

                # After the evaluation, so its cost shows up in the timing.
                elapsed = time.time() - interval_start
                log(
                    f"epoch {epoch + 1:02d}/{n_epochs:02d}  (step {global_step:04d}/{total_steps})  "
                    f"lr={lr_str}  loss={avg['total']:.4f}"
                    + (f" (det={avg['det']:.3f}"
                       + (f" ds={avg['det_ds']:.3f}" if "det_ds" in avg else "")
                       + (f" ann={avg['annot']:.3f}" if "annot" in avg else "")
                       + (f" off={avg['offset']:.3f}" if "offset" in avg else "")
                       + ")"
                       if any(k in avg for k in ("offset", "det_ds", "annot")) else "")
                    + f"  t={elapsed:.0f}s \t{val_message}"
                )
                interval_losses = dict.fromkeys(interval_losses, 0.0)
                interval_steps = 0
                interval_start = time.time()

            if args.save_every > 0 and global_step % args.save_every == 0 and ctx.is_main:
                torch.save(
                    {"model_state_dict": core_model.state_dict()},
                    output_dir / f"ckpt_stage_{stage}_{global_step:05d}.pt",
                )

            # The budget can land mid-epoch, so stop here rather than at the
            # loader's end. is_last_step above has already run the final log,
            # evaluation and checkpoint for this step.
            if is_last_step:
                break

        if global_step >= total_steps:
            break
