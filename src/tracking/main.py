"""Dataset, model, and cross-validation orchestration for tracking training."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from model_zoo.models_track import define_track_model
from params import EMBRYOS
from tracking.dataset import TrackDataset, load_dataset_windows
from tracking.train import fit
from tracking_cellmot.io import list_datasets
from training.distributed import unwrap_model, worker_init_fn


def _paths_for_embryos(data_dir: str, embryos: list[str]) -> list[Path]:
    all_paths = list_datasets(Path(data_dir), require_geff=True)
    return [
        path
        for path in all_paths
        if any(path.name.startswith(embryo) for embryo in embryos)
    ]


def build_loader(
    data_dir: str,
    embryos: list[str],
    window_size: int,
    batch_size: int,
    num_workers: int,
    augment: bool,
    crop_size: tuple[int, int, int] | None = None,
    n_videos: int | None = None,
    ctx=None,
    seed: int = 0,
    distributed: bool | None = None,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Build a DataLoader for datasets belonging to the given embryos."""
    paths = _paths_for_embryos(data_dir, embryos)
    if not paths:
        raise ValueError(f"No datasets found for embryos {embryos} in {data_dir}")

    video_data = []
    for path in tqdm(paths, desc="Loading videos", disable=not ctx.is_main):
        video_meta, windows = load_dataset_windows(
            path,
            window_size=window_size,
        )
        video_data.append((video_meta, windows))

        if n_videos is not None:
            if len(video_data) >= n_videos:
                break

        # break   # for debugging

    dataset = TrackDataset(video_data, augment=augment, crop_size=crop_size)
    if distributed is None:
        distributed = ctx is not None and ctx.distributed

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=augment,
            seed=seed,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=augment and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader, sampler


def build_model(args, ctx):
    """Build and wrap the tracking model for the current process."""
    model = define_track_model(
        unet_out_channels=args.unet_out_channels,
        unet_layers=list(args.unet_layers),
        unet_pool_stride=tuple(args.unet_pool_stride) if hasattr(args, "unet_pool_stride") else 2,
        unet_n_decode_steps=getattr(args, "unet_n_decode_steps", None),
        transformer_hidden=args.transformer_hidden,
        transformer_heads=args.transformer_heads,
        transformer_blocks=args.transformer_blocks,
    )

    if ctx is None:
        return model
    model.to(ctx.device)

    if (
        getattr(args, "data_parallel", False)
        and not ctx.distributed
        and ctx.device.type == "cuda"
        and torch.cuda.device_count() > 1
    ):
        model.unet = nn.DataParallel(model.unet)
        if ctx.is_main:
            print(f"- DataParallel: UNet across {torch.cuda.device_count()} GPUs")
    elif ctx.distributed:
        if getattr(args, "sync_bn", False):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model,
            device_ids=[ctx.device.index] if ctx.device.type == "cuda" else None,
        )
    return model


def k_fold(args, ctx) -> None:
    """Run two-fold leave-one-embryo-out cross-validation."""
    base_output_dir = args.output_dir
    crop_size = tuple(args.crop_size) if args.crop_size is not None else None

    for fold, val_embryo in enumerate(EMBRYOS):
        train_embryos = [embryo for embryo in EMBRYOS if embryo != val_embryo]
        val_embryos = [val_embryo]

        fold_dir = os.path.join(base_output_dir, f"fold_{fold}")
        args.output_dir = fold_dir
        args.train_split = train_embryos
        args.val_split = val_embryos

        if ctx.is_main:
            os.makedirs(fold_dir, exist_ok=True)
            print(f"\n{'=' * 60}")
            print(f"Fold {fold}: train={train_embryos}  val={val_embryos}")
            print(f"{'=' * 60}\n")

        train_loader, train_sampler = build_loader(
            args.data_dir,
            embryos=train_embryos,
            window_size=args.window_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=True,
            crop_size=crop_size,
            ctx=ctx,
            seed=getattr(args, "seed", 0),
        )
        val_loader, _ = build_loader(
            args.data_dir,
            embryos=val_embryos,
            window_size=args.window_size,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            augment=False,
            crop_size=None,
            n_videos=args.n_val_videos,
            ctx=ctx,
            seed=getattr(args, "seed", 0),
        )
        if ctx.is_main:
            print()
            print(f"- Training with {len(train_loader.dataset)} windows")
            print(f"- Validating with {len(val_loader.dataset)} windows")

        model = build_model(args, ctx)
        if ctx.is_main:
            n_params = unwrap_model(model).count_parameters()
            print(
                f"- Model: UNetNodeTransformer  "
                f"(n_parameters={n_params / 1e6:.1f}M)"
            )

        fit(
            model,
            train_loader,
            args,
            ctx,
            train_sampler=train_sampler,
            val_loader=val_loader,
        )

        del model, train_loader, train_sampler, val_loader
        if ctx.device.type == "cuda":
            torch.cuda.empty_cache()

        break
