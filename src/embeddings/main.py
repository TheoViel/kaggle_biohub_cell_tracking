"""Loader, model and cross-validation orchestration, mirroring ``tracking.main``."""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, DistributedSampler

from embeddings.dataset import EmbeddingDataset
from embeddings.model import NETr
from embeddings.train import fit
from params import EMBRYOS
from training.distributed import unwrap_model, worker_init_fn


def _opt_int(value):
    """A None-defaulted numeric flag, which argparse hands back as a string.

    DEFAULTS entries that default to None get no ``type=``, so ``--max_frames
    _per_video 6`` arrives as "6". Coercing at the point of use keeps the
    config declarative and stops a string reaching a slice.
    """
    if value is None or value == "" or str(value).lower() == "none":
        return None
    return int(value)


def build_loader(
    args,
    embryos: list[str],
    augment: bool = True,
    shuffle: bool | None = None,
    ctx=None,
) -> tuple[DataLoader, DistributedSampler | None, EmbeddingDataset]:
    """A loader over the frames of ``coords_log_folder``."""
    dataset = EmbeddingDataset(
        args.coords_log_folder,
        args.data_dir,
        embryos=embryos,
        max_nodes=args.max_nodes,
        min_nodes=args.min_nodes,
        patch_zyx=tuple(args.patch_zyx),
        n_frames=int(getattr(args, "n_frames", 1) or 1),
        aug_cfg=getattr(args, "augmentations", None),
        augment=augment,
        max_frames=_opt_int(getattr(args, "max_frames_per_video", None)),
        real_prop=float(getattr(args, "real_prop", 0.0) or 0.0),
    )
    if shuffle is None:
        shuffle = augment

    sampler = None
    if ctx is not None and ctx.distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=ctx.world_size, rank=ctx.rank,
            shuffle=shuffle, seed=getattr(args, "seed", 0),
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
    return loader, sampler, dataset


def build_model(args, ctx):
    """Build NETr and wrap it for the current process."""
    model = NETr(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        patch_z=int(args.patch_zyx[0]),
        patch_size=int(args.patch_zyx[1]),
        embed_dim=_opt_int(args.embed_dim),
        backbone=args.backbone,
        pretrained=args.pretrained,
        use_visual=bool(getattr(args, "use_visual", True)),
        use_position=bool(getattr(args, "use_position", True)),
        use_transformer=bool(getattr(args, "use_transformer", True)),
        frame_aggregation=getattr(args, "frame_aggregation", "mean"),
        channels_last=bool(getattr(args, "channels_last", True)),
        min_tokens=int(getattr(args, "min_tokens", 3) or 0),
        grad_checkpointing=bool(getattr(args, "grad_checkpointing", False)),
        verbose=int(ctx.is_main) if ctx is not None else 0,
    )
    device = ctx.device if ctx is not None else "cpu"
    model = model.to(device)
    if model.channels_last:
        # Weights have to move too, or every conv would convert them back on
        # each call. Only 4-D parameters are affected; the transformer and the
        # MLPs are untouched.
        model = model.to(memory_format=torch.channels_last)
    if ctx is not None and ctx.distributed:
        if getattr(args, "sync_bn", False):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    return model


def _training_runs(args) -> list[dict]:
    """The models to train, in the tracking pipeline's fold convention.

    Fold ``i`` holds out ``EMBRYOS[i]``. Nothing here is supervised, so the
    held-out embryo is not protecting a label -- it is protecting the *frames*,
    so that a top-1 number is measured on cells the embeddings never saw.
    """
    runs = []
    for fold, val_embryo in enumerate(EMBRYOS):
        if fold in args.selected_folds:
            runs.append({
                "name": f"fold_{fold}",
                "fold": fold,
                "train_embryos": [e for e in EMBRYOS if e != val_embryo],
                "val_embryos": [val_embryo],
                "leaky_val": False,
            })
    if getattr(args, "fullfit", False):
        runs.append({
            "name": "fold_fullfit",
            "fold": 0,
            "train_embryos": list(EMBRYOS),
            "val_embryos": [EMBRYOS[0]],
            "leaky_val": True,
        })
    return runs


def _train_one(args, ctx, run: dict, base_output_dir: str) -> None:
    fold_dir = os.path.join(base_output_dir, run["name"])
    args.output_dir = fold_dir
    args.fold = run["fold"]

    if ctx.is_main:
        os.makedirs(fold_dir, exist_ok=True)
        print(f"\n{'=' * 60}")
        print(f"{run['name']}: train={run['train_embryos']}  val={run['val_embryos']}")
        if run["leaky_val"]:
            print("  validation overlaps training -- the score is leaky, not a metric")
        print(f"{'=' * 60}\n")

    train_loader, train_sampler, train_ds = build_loader(
        args, run["train_embryos"], augment=True, ctx=ctx
    )
    if ctx.is_main and args.eval_every > 0:
        print(f"- Validating on {args.n_val_videos} video(s) of "
              f"{run['val_embryos']}: top-1 GT link retrieval")
    if ctx.is_main:
        n_frames = int(getattr(args, "n_frames", 1) or 1)
        pz = int(args.patch_zyx[0])
        # z coverage is the number worth seeing: 1.625 um a slice, and a
        # nucleus spans only a handful of them.
        print(f"- Training on {len(train_ds)} frames, "
              f"<= {args.max_nodes} cells each, patch {tuple(args.patch_zyx)}"
              + (f" x {n_frames} slabs ({n_frames * pz} z-slices, "
                 f"{n_frames * pz * 1.625:.0f} um)" if n_frames > 1 else ""))
        if train_ds.real_prop > 0:
            # The realised share is below the setting wherever the ground truth
            # has no usable link at t -- those frames fall back to two views of
            # themselves rather than being dropped.
            print(f"- Pairs: {train_ds.real_prop:.0%} real (t, t+1) requested, "
                  f"{train_ds.real_coverage:.0%} of frames can serve one "
                  f"-> ~{train_ds.real_prop * train_ds.real_coverage:.0%} of samples")

    model = build_model(args, ctx)
    if ctx.is_main:
        core = unwrap_model(model)
        branches = "+".join(
            b for b, on in (("visual", core.use_visual), ("position", core.use_position)) if on
        )
        if not core.use_transformer:
            branches += ", no frame attention"
        print(f"- Model: NETr {args.backbone} [{branches}]"
              f"(n_parameters={core.count_parameters() / 1e6:.1f}M)")

    if args.weights:
        state = torch.load(args.weights, map_location=ctx.device, weights_only=True)
        unwrap_model(model).load_state_dict(state.get("model_state_dict", state))
        if ctx.is_main:
            print(f"- Loaded weights from {args.weights}")

    fit(model, train_loader, args, ctx, train_sampler=train_sampler,
        val_embryos=run["val_embryos"])

    del model, train_loader, train_sampler
    if ctx.device.type == "cuda":
        torch.cuda.empty_cache()


def k_fold(args, ctx) -> None:
    """Train the selected folds, and the fullfit model when asked for."""
    base_output_dir = args.output_dir
    runs = _training_runs(args)
    if ctx.is_main and len(runs) > 1:
        print(f"\n-> Training {len(runs)} models: {', '.join(r['name'] for r in runs)}")

    for run in runs:
        _train_one(args, ctx, run, base_output_dir)
        args.output_dir = base_output_dir

    if ctx.is_main:
        print("\nDone!")
