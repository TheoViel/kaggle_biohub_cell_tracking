import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model_zoo.models_flow import FLOW_HEADS, define_flow_model
from segmentation.dataset import SyntheticStaticDataset
from segmentation.eval import evaluate, select_eval_samples
from segmentation.losses import CellposeLoss, flow_error
from training.distributed import (
    all_reduce_mean,
    cleanup_distributed,
    init_distributed,
    seed_everything,
    unwrap_model,
    worker_init_fn,
)
from training.optim import define_optimizer, define_scheduler

LOSS_KEYS = ("flow", "cellprob", "total")


# The architecture arguments, shared verbatim with main_track.py.
UNET_KEYS = (
    "unet_arch", "unet_out_channels", "unet_layers", "unet_pool_stride",
    "unet_n_anisotropic", "unet_n_decode_steps", "unet_temporal_attention",
    "unet_blocks_per_stage", "unet_stem", "unet_norm", "unet_hires_fusion",
)


def _model_config(args) -> dict:
    """The define_flow_model arguments that fix the architecture.

    Saved into every checkpoint so inference rebuilds the same geometry, and
    so a tracking run can read off the backbone config a set of weights was
    trained under.
    """
    cfg = {k: getattr(args, k) for k in UNET_KEYS if hasattr(args, k)}
    for k in ("unet_layers", "unet_pool_stride", "unet_blocks_per_stage"):
        if cfg.get(k) is not None:
            cfg[k] = tuple(cfg[k])
    cfg["in_channels"] = args.in_channels
    cfg["dropout"] = getattr(args, "head_dropout", 0.0)
    return cfg


def build_loader(args, ctx=None):
    """The loader and, under torchrun, the sampler that shards it."""
    ds = SyntheticStaticDataset(
        args.data_dir,
        crop_size=args.patch_size,
        epoch_size=args.epoch_size,
        augment=True,
        flow_mode=args.flow_mode,
        normalization=getattr(args, "normalization", None),
        aug_cfg=getattr(args, "augmentations", None),
    )
    # Built once here rather than inside the dataloader workers, which would
    # redo it every epoch -- and would each want the GPU it runs on.
    ds.prepare(n_jobs=min(8, len(ds.files)))

    sampler = None
    if ctx is not None and ctx.distributed:
        sampler = DistributedSampler(
            ds, num_replicas=ctx.world_size, rank=ctx.rank,
            shuffle=True, seed=getattr(args, "seed", 0),
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
    return loader, sampler, ds


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

    train_loader, train_sampler, ds = build_loader(args, ctx)
    log(f"- Training on {len(ds.files)} static volumes, {args.epoch_size} crops/epoch "
        f"(patch_size={args.patch_size}, flows={args.flow_mode})")
    if len(ds.files) < 50:
        log(f"  ! only {len(ds.files)} volumes: raise N_STATIC in SyntheticData.ipynb "
            f"— crops from a handful of fields do not cover the imaging regimes")

    # A None default in DEFAULTS becomes a plain string flag, so --eval_samples
    # arrives as "a,b" rather than a sequence; list() on that would iterate
    # characters and ask for a stack named "4".
    def _names(value):
        if not value:
            return None
        return [s for s in value.split(",") if s] if isinstance(value, str) else list(value)

    eval_samples = _names(args.eval_samples) or select_eval_samples(
        args.n_eval_samples, _names(getattr(args, "eval_embryos", None))
    )
    log(f"- Validating on {len(eval_samples)} real stacks x {args.eval_frames} frames: "
        f"{', '.join(eval_samples)}")

    model_config = _model_config(args)
    model = define_flow_model(**model_config, heads=FLOW_HEADS).to(device)
    log(f"- Model: FlowUNet / {args.unet_arch}  "
        f"(n_parameters={model.count_parameters() / 1e6:.1f}M)")

    if ctx.distributed:
        if getattr(args, "sync_bn", False):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    core_model = unwrap_model(model)

    criterion = CellposeLoss(w_flow=args.w_flow, w_cellprob=args.w_cellprob)

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
    scheduler = define_scheduler(
        optimizer, train_loader, args.epochs, args.optimizer_config["warmup_prop"]
    )

    output_dir = Path(args.output_dir)
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    def save(name):
        torch.save(
            {"model_state_dict": core_model.state_dict(), "model_config": model_config},
            output_dir / name,
        )

    interval = dict.fromkeys(LOSS_KEYS, 0.0)
    interval_cos, interval_steps = 0.0, 0
    interval_start = time.time()

    log("\n-> Starting training...\n")
    try:
        global_step = 0
        for epoch in range(args.epochs):
            model.train()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            for batch in train_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                image = batch["image"]
                targets = {k: v for k, v in batch.items() if k != "image"}

                # No mixup or cutmix here: blending two images means blending
                # two flow fields, and the average of two unit vectors pointing
                # at different cells points at neither.
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_fp16
                ):
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
                if scaler.get_scale() >= scale:  # a skipped step must not advance the lr
                    scheduler.step()
                global_step += 1

                for k in LOSS_KEYS:
                    interval[k] += breakdown[k]
                interval_cos += flow_error(
                    {"flows": preds["flows"].float()}, targets
                )
                interval_steps += 1

                if global_step % args.log_every == 0:
                    avg = all_reduce_mean(
                        {k: v / interval_steps for k, v in interval.items()}, device
                    )
                    message = (
                        f"epoch {epoch + 1:02d}/{args.epochs:02d}  (step {global_step:05d})  "
                        f"lr={optimizer.param_groups[0]['lr']:.1e}   "
                        f"loss={avg['total']:.3f}  (flow {avg['flow']:.3f} "
                        f"cellprob {avg['cellprob']:.3f})   "
                        f"flow_cos={interval_cos / interval_steps:.3f}   "
                        f"t={time.time() - interval_start:.0f}s"
                    )


                    if args.eval_every > 0 and global_step % args.eval_every == 0:
                        summary, _ = evaluate(
                            core_model,
                            eval_samples,
                            max_frames=args.eval_frames,
                            normalization=getattr(args, "normalization", None),
                            device=str(device),
                            verbose=False,
                            ctx=ctx,
                            thresholds=args.eval_thresholds,
                        )
                        val_msg = ""
                        for _, r in summary.iterrows():
                            val_msg += f"[th={r['cellprob_threshold']:.2f}]  node_rec={r['node_recall']:.3f}  adj={r['adjustment']:.3f}"

                        message = message + "  " + val_msg

                    interval = dict.fromkeys(LOSS_KEYS, 0.0)
                    interval_cos, interval_steps = 0.0, 0
                    interval_start = time.time()
                    log(message)

                if args.save_every > 0 and global_step % args.save_every == 0 and ctx.is_main:
                    save(f"ckpt_{global_step:05d}.pt")

        if ctx.is_main:
            save("ckpt_final.pt")
            log("\n-> Saved weights to ", output_dir / "ckpt_final.pt")
            print()

        # Outside the is_main guard, for the same reason as above: the final
        # score is a collective too.
        summary, per_sample = evaluate(
            core_model,
            eval_samples,
            max_frames=args.eval_frames,
            normalization=getattr(args, "normalization", None),
            device=str(device),
            ctx=ctx,
            thresholds=args.eval_thresholds,
        )
        if ctx.is_main:
            log("")
            for _, r in summary.iterrows():
                log(f"- th {r['cellprob_threshold']:.2f}   "
                    f"node_recall {r['node_recall']:.4f}   "
                    f"adjustment {r['adjustment']:.4f}   "
                    f"nodes/frame {r['n_pred_per_frame']:.0f}")
            summary.to_csv(output_dir / "scores.csv", index=False)
            per_sample.to_csv(output_dir / "scores_per_sample.csv", index=False)
            log("\nDone !")
    finally:
        cleanup_distributed()
