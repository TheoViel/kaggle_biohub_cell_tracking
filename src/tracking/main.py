"""Dataset, model, and cross-validation orchestration for tracking training."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, SubsetRandomSampler

from model_zoo.models_track import define_track_model
from params import EMBRYOS
from tracking.dataset import TrackDataset, load_dataset_windows
from tracking.contrastive import fit_contrastive
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
    downscale_factor: int,
    augment: bool,
    weak_positives_cfg: dict | None = None,
    brightness_strength: float = 0.0,
    crop_size: tuple[int, int, int] | None = None,
    n_videos: int | None = None,
    video_data=None,
    ctx=None,
    seed: int = 0,
    distributed: bool | None = None,
    shuffle: bool | None = None,
    stable_shard: bool = False,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Build a DataLoader for datasets belonging to the given embryos.

    ``shuffle`` defaults to ``augment``; pass it explicitly to shuffle an
    unaugmented loader (transformer-only training caches per-sample features,
    so it wants fixed samples but a fresh batch order every epoch).

    ``stable_shard`` keeps each rank on the same windows for the whole run,
    which a per-rank cache needs; the order within a rank still varies.
    """
    if shuffle is None:
        shuffle = augment
    if video_data is None:
        paths = _paths_for_embryos(data_dir, embryos)
        if not paths:
            raise ValueError(f"No datasets found for embryos {embryos} in {data_dir}")

        video_data = []
        for path in paths:
            video_meta, windows = load_dataset_windows(path, window_size=window_size)
            video_data.append((video_meta, windows))

            if n_videos is not None:
                if len(video_data) >= n_videos:
                    break
    
    dataset = TrackDataset(
        video_data,
        downscale_factor=downscale_factor,
        augment=augment,
        crop_size=crop_size,
        weak_positives_cfg=weak_positives_cfg,
        brightness_strength=brightness_strength,
    )

    if distributed is None:
        distributed = ctx is not None and ctx.distributed

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=shuffle and not stable_shard,
            seed=seed,
        )
        if stable_shard and shuffle:
            # A shuffling DistributedSampler re-partitions on every set_epoch,
            # so a rank sees different windows each epoch — fatal for a per-rank
            # feature cache. Take the unshuffled (hence fixed, and equal-length
            # across ranks) shard, then shuffle only inside it.
            sampler = SubsetRandomSampler(list(sampler))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return video_data, loader, sampler


def build_model(args, ctx):
    """Build and wrap the tracking model for the current process."""
    model = define_track_model(
        unet_out_channels=args.unet_out_channels,
        unet_layers=list(args.unet_layers),
        unet_pool_stride=(
            tuple(args.unet_pool_stride) if hasattr(args, "unet_pool_stride") else 2
        ),
        unet_n_decode_steps=getattr(args, "unet_n_decode_steps", None),
        transformer_hidden=args.transformer_hidden,
        transformer_heads=args.transformer_heads,
        transformer_blocks=args.transformer_blocks,
        transformer_dropout=args.transformer_dropout,
        edge_knn_k=getattr(args, "edge_knn_k", None),
        edge_geom_feats=getattr(args, "edge_geom_feats", True),
        pos_feat_dim=args.pos_feat_dim,
        contrastive=getattr(args, "contrastive", None),
        unet_features=tuple(getattr(args, "unet_features", ("detection",)) or ()),
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


def load_weights(
    model,
    path: str,
    fold: int,
    ctx,
    prefixes: tuple[str, ...] = ("unet.", "detect_head."),
) -> None:
    """Load the segmentation backbone from a checkpoint, in place.

    Only tensors under ``prefixes`` are taken — the pretraining stage trains
    detection alone, so the transformer in such a checkpoint is untouched from
    init and carries no information. Skipping it also means the edge
    architecture can change freely between the pretraining run and this one.

    Tensors whose shape no longer matches are skipped as well, so a backbone
    config change degrades to a partial load rather than an exception.

    ``path`` may contain a ``{fold}`` placeholder, so a single config entry can
    point at the per-fold checkpoints a previous run wrote.
    """
    path = path.format(fold=fold) if "{fold}" in path else path
    ckpt = torch.load(path, map_location=ctx.device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)

    core = unwrap_model(model)
    current = core.state_dict()
    to_load, mismatched = {}, []
    for key, tensor in state.items():
        if not key.startswith(prefixes):
            continue
        if key not in current:
            continue
        if current[key].shape != tensor.shape:
            mismatched.append(key)
            continue
        to_load[key] = tensor

    core.load_state_dict(to_load, strict=False)

    if ctx.is_main:
        n_backbone = sum(1 for k in current if k.startswith(prefixes))
        print(f"- Loaded {len(to_load)}/{n_backbone} backbone tensors ({'|'.join(prefixes)}) from {path}")
        if len(to_load) < n_backbone:
            print(f"  WARNING: {n_backbone - len(to_load)} not loaded", end="")
            print(f", {len(mismatched)} shape mismatches" if mismatched else "")


def k_fold(args, ctx) -> None:
    """Run two-fold leave-one-embryo-out cross-validation."""
    base_output_dir = args.output_dir
    crop_size = tuple(args.crop_size) if args.crop_size is not None else None

    for fold, val_embryo in enumerate(EMBRYOS):
        train_embryos = [embryo for embryo in EMBRYOS if embryo != val_embryo]
        val_embryos = [val_embryo]

        fold_dir = os.path.join(base_output_dir, f"fold_{fold}")
        args.output_dir = fold_dir
        args.fold = fold
        args.train_split = train_embryos
        args.val_split = val_embryos

        if ctx.is_main:
            os.makedirs(fold_dir, exist_ok=True)
            print(f"\n{'=' * 60}")
            print(f"Fold {fold}: train={train_embryos}  val={val_embryos}")
            print(f"{'=' * 60}\n")

        train_video_data, train_loader, train_sampler = build_loader(
            args.data_dir,
            embryos=train_embryos,
            window_size=args.window_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            downscale_factor=args.downscale_factor,
            # Transformer-only training reuses cached node features, so the
            # images must not change between epochs — but the batch order still
            # should, and the cache is per sample, so shuffling stays on.
            augment=not getattr(args, "transformer_only", False),
            shuffle=True,
            stable_shard=getattr(args, "transformer_only", False),
            weak_positives_cfg=getattr(args, "weak_positives", {}),
            brightness_strength=getattr(args, "brightness_strength", 0.0),
            crop_size=crop_size,
            # n_videos=10,  # DEBUG
            ctx=ctx,
            seed=getattr(args, "seed", 0),
        )
        _, val_loader, _ = build_loader(
            args.data_dir,
            embryos=val_embryos,  # train_embryos,
            window_size=args.window_size,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            downscale_factor=args.downscale_factor,
            augment=False,
            weak_positives_cfg={},  # don't bother with that
            crop_size=None,
            n_videos=args.n_val_videos,
            ctx=ctx,
            seed=getattr(args, "seed", 0),
        )
        if ctx.is_main:
            print(f"- Training with {len(train_loader.dataset)} windows")
            print(f"- Validating with {len(val_loader.dataset)} windows")

        model = build_model(args, ctx)
        if ctx.is_main:
            n_params = unwrap_model(model).count_parameters()
            print(f"- Model: UNetNodeTransformer  (n_parameters={n_params / 1e6:.1f}M)")

        # --- Stage 1: pretraining the detection UNet ---
        pretraining = args.pretraining
        weights = pretraining.get("weights")
        if weights:
            # Reuse a previous run's pretraining instead of redoing step 1.
            load_weights(model, weights, fold, ctx)

        if pretraining["enabled"] and not weights:
            _, seg_loader, seg_sampler = build_loader(
                args.data_dir,
                embryos=train_embryos,
                window_size=args.window_size,
                batch_size=pretraining["batch_size"],
                num_workers=args.num_workers,
                downscale_factor=args.downscale_factor,
                augment=True,
                video_data=train_video_data,  # save reloading the videos
                weak_positives_cfg=getattr(args, "weak_positives", {}),
                brightness_strength=getattr(args, "brightness_strength", 0.0),
                crop_size=pretraining["crop_size"],
                ctx=ctx,
                seed=getattr(args, "seed", 0),
            )
            seg_args = copy.copy(args)
            for k, v in pretraining.items():
                setattr(seg_args, k, v)
            seg_args.save_every = -1  # do not save
            seg_args.det_warmup_prop = 1.0
            # Stage 1 is the one stage that *does* train the detection UNet,
            # so it must not inherit stage 3's freeze-everything flag.
            seg_args.transformer_only = False
            seg_args.optimizer_config = {
                **args.optimizer_config,
                "lr_unet": args.optimizer_config["lr"],
                "lr_transformer": args.optimizer_config["lr"],
            }

            if ctx.is_main:
                print(
                    f"\n-> Stage 1: Pretraining the detection UNet on "
                    f"{seg_loader.dataset.crop_size} crops, "
                    f"{len(seg_loader.dataset)} windows, "
                    f"{seg_args.epochs} epochs\n"
                )

            fit(
                model,
                seg_loader,
                seg_args,
                ctx,
                train_sampler=seg_sampler,
                val_loader=val_loader,
            )

            if ctx.is_main:
                torch.save(
                    {"model_state_dict": unwrap_model(model).state_dict()},
                    os.path.join(fold_dir, "ckpt_pretrain.pt"),
                )

            del seg_loader, seg_sampler
            if ctx.device.type == "cuda":
                torch.cuda.empty_cache()

        # --- Stage 2: contrastive embedding UNet, on the frozen detections ---
        # Runs only when the transformer asks for those features; otherwise the
        # branch was never built.
        contrastive = getattr(args, "contrastive", {}) or {}
        use_contrastive = "contrastive" in (getattr(args, "unet_features", ()) or ())

        weights = contrastive.get("weights") if use_contrastive else None
        if weights:
            load_weights(
                model, weights, fold, ctx,
                prefixes=("embed_unet.", "embed_head."),
            )
        if use_contrastive and not weights:
            _, emb_loader, emb_sampler = build_loader(
                args.data_dir,
                embryos=train_embryos,
                window_size=args.window_size,
                batch_size=contrastive["batch_size"],
                num_workers=args.num_workers,
                downscale_factor=args.downscale_factor,
                augment=contrastive.get("augment", True),
                video_data=train_video_data,  # save reloading the videos
                weak_positives_cfg={},        # detection is frozen here
                brightness_strength=getattr(args, "brightness_strength", 0.0),
                crop_size=contrastive.get("crop_size"),
                ctx=ctx,
                seed=getattr(args, "seed", 0),
                stable_shard=True,            # node coords are cached per rank
            )
            if ctx.is_main:
                print(
                    f"\n-> Stage 2: Contrastive training the embedding UNet on "
                    f"{contrastive['crop_size']} crops, "
                    f"{len(emb_loader.dataset)} windows, "
                    f"{contrastive['epochs']} epochs\n"
                )
            fit_contrastive(
                model, emb_loader, args, ctx,
                train_sampler=emb_sampler, val_loader=val_loader,
            )
            del emb_loader, emb_sampler
            if ctx.device.type == "cuda":
                torch.cuda.empty_cache()

        # --- Stage 3: edge transformer on frozen coordinates + embeddings ---
        if ctx.is_main:
            print("\n-> Stage 3: Edge transformer training\n")

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
