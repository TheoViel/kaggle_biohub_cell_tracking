"""Dataset, model, and cross-validation orchestration for tracking training."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, SubsetRandomSampler

from model_zoo.models_track import define_track_model
from params import EMBRYOS
from tracking.data import resolve_normalization
from tracking.dataset import TrackDataset, load_dataset_windows
from tracking.synthetic import (
    MixedDataset,
    SyntheticDataset,
    find_embryo_sequences,
    find_sequences,
)
from tracking.contrastive import fit_contrastive
from tracking.fit_utils import resolve_run_path
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
    aug_cfg: dict | None = None,
    normalization: dict | None = None,
    crop_size: tuple[int, int, int] | None = None,
    n_videos: int | None = None,
    video_data=None,
    ctx=None,
    seed: int = 0,
    distributed: bool | None = None,
    shuffle: bool | None = None,
    stable_shard: bool = False,
    synthetic_cfg: dict | None = None,
    nodes_cfg: dict | None = None,
    load_edges: bool = True,
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
    # Frame gaps wider than 1 are extra training windows, so they are built
    # only for a loader that augments — validation stays on consecutive frames.
    stride_max = int((aug_cfg or {}).get("time_stride_max", 1)) if augment else 1
    strides = tuple(range(1, max(1, stride_max) + 1))

    if video_data is None:
        paths = _paths_for_embryos(data_dir, embryos)
        if not paths:
            raise ValueError(f"No datasets found for embryos {embryos} in {data_dir}")

        video_data = []
        for path in paths:
            video_meta, windows = load_dataset_windows(
                path,
                window_size=window_size,
                strides=strides,
                normalization=normalization,
            )
            video_data.append((video_meta, windows))

            if n_videos is not None:
                if len(video_data) >= n_videos:
                    break

    # Synthetic sequences, mixed in at a fixed proportion. Built first so the
    # real dataset can be padded to a common max_nodes: the two are collated
    # into the same batch, and synthetic frames hold ~285 cells against a
    # handful of annotated ones in a real window.
    synth_cfg = synthetic_cfg or {}
    synth = None
    if (
        synth_cfg.get("synthetic_dir")
        and synth_cfg.get("synthetic_proportion", 0.0) > 0
    ):
        files = find_sequences(synth_cfg["synthetic_dir"])
        # Its own cap, not n_videos: that one counts real videos, and the two
        # sets are unrelated in size (1000 sequences against ~100 videos).
        if synth_cfg.get("max_sequences"):
            files = files[: int(synth_cfg["max_sequences"])]
        synth = SyntheticDataset(
            files,
            window_size=window_size,
            max_nodes=int(synth_cfg.get("max_nodes", 768)),
            crop_size=crop_size,
            augment=augment,
            aug_cfg=aug_cfg,
            normalization=normalization,
            mosaic_p=float(synth_cfg.get("mosaic_p", 0.0)),
            mosaic_jitter=float(synth_cfg.get("mosaic_jitter", 0.2)),
            load_edges=load_edges,
        )

    # Zebrahub crops: real images from the public release, tiled into the
    # synthetic set's format (see data.zebrahub), so the same loader reads them.
    # They are annotated densely, which is what the real half is short of, but
    # they are real tissue rather than rendered -- hence their own proportion
    # rather than being thrown in with the synthetic files. No mosaic: stitching
    # four unrelated pieces of an embryo together is a tell on real images.
    zeb_cfg_embryos = synth_cfg.get("zebrahub_embryos") or ()
    zeb = None
    if (
        synth_cfg.get("zebrahub_dir")
        and zeb_cfg_embryos
        and synth_cfg.get("zebrahub_proportion", 0.0) > 0
    ):
        zeb_max_nodes = int(synth_cfg.get("max_nodes", 768))
        zeb_files = find_embryo_sequences(
            synth_cfg["zebrahub_dir"], zeb_cfg_embryos, max_nodes=zeb_max_nodes
        )
        zeb = SyntheticDataset(
            zeb_files,
            window_size=window_size,
            max_nodes=zeb_max_nodes,
            crop_size=crop_size,
            augment=augment,
            aug_cfg=aug_cfg,
            normalization=normalization,
            load_edges=load_edges,
        )

    dataset = TrackDataset(
        video_data,
        max_nodes=(
            int(synth_cfg.get("max_nodes", 768))
            if synth is not None or zeb is not None
            else None
        ),
        downscale_factor=downscale_factor,
        augment=augment,
        crop_size=crop_size,
        weak_positives_cfg=weak_positives_cfg,
        aug_cfg=aug_cfg,
        clamp_max=resolve_normalization(normalization)["clamp_max"],
        nodes_cfg=nodes_cfg,
        load_edges=load_edges,
    )
    sources = []
    if synth is not None:
        sources.append((synth, float(synth_cfg["synthetic_proportion"])))
        if ctx is None or ctx.is_main:
            print(
                f"- Synthetic: {len(files)} sequences from {synth_cfg['synthetic_dir']}, "
                f"{synth_cfg['synthetic_proportion']:.0%} of training samples"
            )
    if zeb is not None:
        sources.append((zeb, float(synth_cfg["zebrahub_proportion"])))
        if ctx is None or ctx.is_main:
            print(
                f"- Zebrahub: {len(zeb_files)} crops from {list(zeb_cfg_embryos)}, "
                f"{synth_cfg['zebrahub_proportion']:.0%} of training samples"
            )
    if sources:
        dataset = MixedDataset(dataset, sources)

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
        unet_arch=getattr(args, "unet_arch", "temporal"),
        unet_out_channels=args.unet_out_channels,
        unet_layers=list(args.unet_layers),
        unet_pool_stride=(
            tuple(args.unet_pool_stride) if hasattr(args, "unet_pool_stride") else 2
        ),
        unet_n_anisotropic=getattr(args, "unet_n_anisotropic", 2),
        unet_n_decode_steps=getattr(args, "unet_n_decode_steps", None),
        unet_n_extra_decode_steps=int(getattr(args, "unet_n_extra_decode_steps", 0) or 0),
        unet_temporal_attention=bool(getattr(args, "unet_temporal_attention", True)),
        # "temporal_v2" only; ignored by the other backbones.
        unet_temporal_fusion_at=tuple(
            getattr(args, "unet_temporal_fusion_at", ("head",))
        ),
        unet_temporal_match_channels=int(getattr(args, "unet_temporal_match_channels", 32)),
        unet_temporal_taps=tuple(getattr(args, "unet_temporal_taps", (5, 3, 3))),
        unet_temporal_dilation=getattr(args, "unet_temporal_dilation", "auto"),
        unet_temporal_agree=getattr(args, "unet_temporal_agree", "mean"),
        unet_temporal_max_neighbours=getattr(args, "unet_temporal_max_neighbours", None),
        unet_temporal_motion_prior=float(getattr(args, "unet_temporal_motion_prior", 1.0)),
        unet_temporal_flow_smooth_iters=int(
            getattr(args, "unet_temporal_flow_smooth_iters", 3)
        ),
        unet_temporal_flow_channels=int(
            getattr(args, "unet_temporal_flow_channels", 8)
        ),
        unet_temporal_flow_hidden_channels=int(
            getattr(args, "unet_temporal_flow_hidden_channels", 32)
        ),
        unet_temporal_value_channels=int(
            getattr(args, "unet_temporal_value_channels", 8)
        ),
        unet_spatial_attention=getattr(args, "unet_spatial_attention", "none"),
        unet_spatial_attention_stages=int(
            getattr(args, "unet_spatial_attention_stages", 1)
        ),
        unet_spatial_attention_heads=int(getattr(args, "unet_spatial_attention_heads", 4)),
        unet_spatial_attention_reduction=getattr(
            args, "unet_spatial_attention_reduction", 4
        ),
        unet_spatial_attention_window=tuple(
            getattr(args, "unet_spatial_attention_window", (4, 8, 8))
        ),
        # Defaults here are the pre-upgrade architecture, so a run logged before
        # these keys existed still rebuilds the model its checkpoint was saved from.
        unet_blocks_per_stage=getattr(args, "unet_blocks_per_stage", None),
        unet_stem=getattr(args, "unet_stem", "stride"),
        unet_norm=getattr(args, "unet_norm", "batch"),
        unet_hires_fusion=bool(getattr(args, "unet_hires_fusion", False)),
        # "timm25d" only; ignored by the other backbones.
        unet_timm_backbone=getattr(
            args, "unet_timm_backbone", "convnext_nano.in12k_ft_in1k"
        ),
        unet_timm_slice_size=int(getattr(args, "unet_timm_slice_size", 5)),
        unet_timm_pretrained=bool(getattr(args, "unet_timm_pretrained", True)),
        unet_timm_drop_path_rate=float(getattr(args, "unet_timm_drop_path_rate", 0.0)),
        unet_timm_max_levels=getattr(args, "unet_timm_max_levels", None),
        transformer_hidden=args.transformer_hidden,
        transformer_heads=args.transformer_heads,
        transformer_blocks=args.transformer_blocks,
        transformer_dropout=args.transformer_dropout,
        edge_knn_k=getattr(args, "edge_knn_k", None),
        edge_geom_feats=getattr(args, "edge_geom_feats", True),
        edge_contrastive_feats=bool(getattr(args, "edge_contrastive_feats", False)),
        contrastive_feat_dim=int(getattr(args, "contrastive_feat_dim", 0) or 0),
        edge_attention=bool(getattr(args, "edge_attention", True)),
        edge_drift_correction=getattr(args, "edge_drift_correction", False),
        pos_feat_dim=args.pos_feat_dim,
        contrastive=getattr(args, "contrastive", None),
        unet_features=tuple(getattr(args, "unet_features", ("detection",)) or ()),
        offset_head=bool(getattr(args, "offset_head", False)),
        offset_radius=int(getattr(args, "offset_radius", 2)),
        annot_head=bool(getattr(args, "annot_head", False)),
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
    prefixes: tuple[str, ...] = (
        "unet.", "detect_head.", "detect_head_fine.", "offset_head.", "annot_head.",
    ),
    run_name: str | None = None,
) -> None:
    """Load the segmentation backbone from a checkpoint, in place.

    Only tensors under ``prefixes`` are taken — the pretraining stage trains
    detection alone, so the transformer in such a checkpoint is untouched from
    init and carries no information. Skipping it also means the edge
    architecture can change freely between the pretraining run and this one.

    Tensors whose shape no longer matches are skipped as well, so a backbone
    config change degrades to a partial load rather than an exception.

    ``detect_head_fine`` is named separately because ``startswith`` does not see
    it inside ``detect_head.`` — the dot is what makes them different prefixes.
    A new detection-side module that is not listed here loads silently as random
    weights, so the log below names every module the filter dropped rather than
    only counting the ones it kept.

    ``path`` may contain ``fold_*`` (or the older ``{fold}`` placeholder), so a
    single config entry can point at the per-run checkpoints a previous run
    wrote — see ``resolve_run_path``.
    """
    path = resolve_run_path(path, fold, run_name)
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
        print(
            f"- Loaded {len(to_load)}/{n_backbone} backbone tensors ({'|'.join(prefixes)}) from {path}"
        )
        if len(to_load) < n_backbone:
            print(f"  WARNING: {n_backbone - len(to_load)} not loaded", end="")
            print(f", {len(mismatched)} shape mismatches" if mismatched else "")
        # Weights the checkpoint has and this model wants, dropped only because
        # no prefix names them. The count above cannot see these — it filters
        # the model by the same prefixes — so they would otherwise stay at init
        # with the log reporting a clean 184/184.
        dropped = sorted({
            key.split(".")[0]
            for key in state
            if key in current and not key.startswith(prefixes)
        })
        if dropped:
            print(f"  Skipped modules (not in prefixes): {', '.join(dropped)}")


def _stage_aug(args, name: str) -> dict | None:
    """The augmentation block this stage runs with.

    ``main_track`` resolves each stage's ``aug_strength`` into a concrete block
    before training starts; a stage that named no rung falls back to the
    run-level one. Synthetic data is what motivates per-stage strength: the
    model is in-domain on it and will happily fit the generator's own
    regularities, so it is worth corrupting far harder than the real frames,
    whose photometry is already whatever the microscope made it.
    """
    stage_block = _stage(args, name).get("augmentations")
    if stage_block is not None:
        return stage_block
    return getattr(args, "augmentations", None)


def _stage(args, name: str) -> dict:
    """One training stage's settings, as a plain dict.

    Each stage owns its own block in the config -- ``pretraining``,
    ``contrastive``, ``edge`` -- so that batch size, crop size and schedule can
    differ between them without three sets of top-level names shadowing each
    other. Reading them through here keeps a missing block from turning into an
    AttributeError halfway through a run.
    """
    return dict(getattr(args, name, None) or {})


def _apply_stage(args, name: str):
    """A copy of ``args`` with one stage's settings promoted to attributes.

    ``fit`` and the loss/eval code read flat attributes (``args.epochs``,
    ``args.transformer_only``, ...), so a stage is run by flattening its block
    on top of a copy. It has to be a copy: ``k_fold`` reuses one ``args``
    across folds, and a stage that edited it in place would hand its settings
    to the next fold's stage 1.
    """
    stage_args = copy.copy(args)
    for key, value in _stage(args, name).items():
        # A stage's dict entries are overrides, not replacements. A stage that
        # names two of loss_config's keys means "those two, everything else as
        # configured" -- replacing the whole dict would drop the rest and
        # TrackingLoss would be called without them.
        base = getattr(args, key, None)
        if isinstance(value, dict) and isinstance(base, dict):
            value = {**base, **value}
        setattr(stage_args, key, value)

    # Optimiser keys named directly in a stage block override `optimizer_config`
    # for that stage alone. They need naming here because the dict-merge above
    # cannot reach them: `lr` is a flat key in a stage block and a nested one in
    # `optimizer_config`, so setting it in a stage would otherwise land on
    # `stage_args.lr`, which nothing reads.
    #
    # This is the difference between "finetune the backbone gently" and "train
    # everything gently". A pretraining stage resuming from someone else's
    # checkpoint wants a low rate and a long warmup; stage 3, whose transformer
    # is at init because `load_weights` drops `transformer.`, emphatically does
    # not -- at 1e-4 with half the budget as warmup it reaches edge_jac 0.35
    # where 1e-3 reaches 0.75. Moving the global config to suit the first stage
    # is what broke the second in 2026-09-17/4 and /6.
    block = _stage(args, name)
    tuned = {k: block[k] for k in ("lr", "warmup_prop", "weight_decay")
             if block.get(k) is not None}
    if tuned:
        stage_args.optimizer_config = {**args.optimizer_config, **tuned}

    stage_args.stage = name
    return stage_args


def _training_runs(args) -> list[dict]:
    """The models to train, in order: the cross-validation fold, then any fullfit.

    A fullfit model sees every embryo, so it has no held-out data to score against
    and its validation numbers are leaky by construction — it exists to be the
    stronger model, not the measured one. Fold 0's score is the honest one.
    """
    runs = []
    for fold, val_embryo in enumerate(EMBRYOS):
        if args.selected_folds is None:
            continue
        if fold in args.selected_folds:
            runs.append(
                {
                    "name": f"fold_{fold}",
                    "fold": fold,
                    "train_embryos": [e for e in EMBRYOS if e != val_embryo],
                    "val_embryos": [val_embryo],
                    "n_val_videos": int(_stage(args, "edge").get("n_val_videos", 5)),
                    "leaky_val": False,
                }
            )

    if getattr(args, "fullfit", False):
        runs.append(
            {
                "name": "fold_fullfit",
                # Nothing is held out, so there is no fold index of its own.
                # ``fold`` stays 0 for the legacy {fold} placeholder and for
                # anything keyed on an integer; a ``fold_*`` weights path
                # resolves to this run's own "fold_fullfit" directory instead.
                "fold": 0,
                "train_embryos": list(EMBRYOS),
                "val_embryos": [EMBRYOS[0]],
                # The score is leaky whatever it is computed on, so spend as little
                # time on it as possible and keep it only as a training-progress trace.
                "n_val_videos": getattr(args, "fullfit_n_val_videos", 1),
                "leaky_val": True,
            }
        )
    return runs


def _check_pseudo_labels(args, runs: list[dict]) -> None:
    """Fail before any training starts if a run's pseudo-labels are missing.

    TrackDataset raises per sample, so a gap would otherwise only surface once
    that run's first epoch reads it -- after everything before it has trained.
    The gap is easy to hit with fullfit: a --pl pass covers one fold's training
    embryos, while fullfit trains on all of them.
    """
    cfg = getattr(args, "weak_positives", {}) or {}
    if not cfg.get("use") or not cfg.get("pl_folder"):
        return

    folder = Path(cfg["pl_folder"])
    for run in runs:
        paths = _paths_for_embryos(args.data_dir, run["train_embryos"])
        missing = [p.name for p in paths if not (folder / f"{p.name}.npz").exists()]
        if missing:
            raise FileNotFoundError(
                f"{run['name']} trains on {run['train_embryos']} but "
                f"{len(missing)} of its {len(paths)} videos have no pseudo-labels in "
                f"{folder} (e.g. {missing[0]}). Run main_predict.py --pl over those "
                f"embryos, or unset weak_positives.pl_folder."
            )


def _check_stage_weights(args, runs: list[dict]) -> None:
    """Fail before any training starts if a configured weights path is missing.

    A stale ``weights`` entry otherwise surfaces only when its stage is
    reached, which on a three-run sweep can be an hour of synthetic
    pretraining after the mistake was made. The path is resolved per run
    because ``fold_*`` expands to the run's own directory.
    """
    for stage in ("synthetic_pretraining", "pretraining", "contrastive", "edge"):
        cfg = _stage(args, stage)
        path = cfg.get("weights")
        if not path:
            continue
        for run in runs:
            resolved = resolve_run_path(path, run["fold"], run["name"])
            if not Path(resolved).exists():
                raise FileNotFoundError(
                    f"{stage}.weights points at {resolved}, which does not exist "
                    f"(run {run['name']}). Set it to null to start the stage from "
                    f"scratch, or fix the path."
                )


def k_fold(args, ctx) -> None:
    """Train the cross-validation fold, and the fullfit model when asked for."""
    base_output_dir = args.output_dir

    runs = _training_runs(args)
    _check_pseudo_labels(args, runs)
    _check_stage_weights(args, runs)

    # Carries the synthetic checkpoint from the first run to the rest. Stage 1
    # sees no real data at proportion 1, so it has no fold to be specific to
    # and there is nothing to gain from paying for it once per fold.
    shared: dict[str, str] = {}

    if ctx.is_main and len(runs) > 1:
        print(f"\n-> Training {len(runs)} models: {', '.join(r['name'] for r in runs)}")

    for run in runs:
        _train_one(args, ctx, run, base_output_dir, shared)
        args.output_dir = base_output_dir

    if ctx.is_main:
        print("\nDone!")


# What stage 3 inherits from the pretraining blocks when it asks for mixed-in
# samples. The proportions are deliberately not here: each stage says how much
# it wants, the source is configured once.
_EDGE_INHERITED = (
    "synthetic_dir", "max_sequences", "max_nodes", "mosaic_p", "mosaic_jitter",
    "zebrahub_dir", "zebrahub_embryos",
)


def _resolve_edge_sources(args, edge_cfg: dict, nodes_cfg: dict | None) -> dict:
    """The synthetic/zebrahub config for stage 3, resolved against the earlier stages.

    Only the proportions have to be set on the edge block; the directories, the
    embryo list, the mosaic and the caps fall back to ``synthetic_pretraining``
    and then ``pretraining``, so a source is configured in one place and each
    stage just says how much of it to mix in.

    The point in stage 3 is edges, not detections: these crops carry their own
    lineage, so they supervise the transformer on divisions and crossings at a
    density the competition annotation never reaches (a handful of annotated
    cells a frame against ~285 synthetic or up to 786 real zebrahub ones).
    """
    synthetic_proportion = float(edge_cfg.get("synthetic_proportion", 0.0) or 0.0)
    zebrahub_proportion = float(edge_cfg.get("zebrahub_proportion", 0.0) or 0.0)
    if synthetic_proportion <= 0 and zebrahub_proportion <= 0:
        return {}

    base = dict(getattr(args, "synthetic_pretraining", {}) or {})
    if not base.get("synthetic_dir") and not base.get("zebrahub_dir"):
        base.update(getattr(args, "pretraining", {}) or {})
    resolved = {k: v for k, v in base.items() if k in _EDGE_INHERITED}
    # `is not None` matters: the edge block declares `synthetic_dir: None` as
    # its default, and a plain update would overwrite the value inherited from
    # `synthetic_pretraining` with it -- leaving the run with a proportion and
    # nowhere to draw from.
    resolved.update(
        {k: v for k, v in edge_cfg.items() if v is not None and k in _EDGE_INHERITED}
    )
    resolved["synthetic_proportion"] = synthetic_proportion
    resolved["zebrahub_proportion"] = zebrahub_proportion

    if synthetic_proportion > 0 and not resolved.get("synthetic_dir"):
        raise ValueError(
            "edge.synthetic_proportion is set but no synthetic_dir was found "
            "in the edge block, synthetic_pretraining or pretraining")
    if zebrahub_proportion > 0 and not (
        resolved.get("zebrahub_dir") and resolved.get("zebrahub_embryos")
    ):
        raise ValueError(
            "edge.zebrahub_proportion is set but no zebrahub_dir/zebrahub_embryos "
            "were found in the edge block, synthetic_pretraining or pretraining")
    if nodes_cfg:
        # Real samples would take their nodes -- and their contrastive
        # embeddings -- off disk while the mixed-in ones carry their own, so the
        # two halves of the batch would carry different node features. Fail here
        # rather than at the first collate.
        raise ValueError(
            "edge.synthetic_proportion/zebrahub_proportion cannot be combined with "
            f"{'/'.join(sorted(nodes_cfg))}: those read nodes off disk for "
            "the real videos and the mixed-in crops have none")
    return resolved


def _train_one(
    args, ctx, run: dict, base_output_dir: str, shared: dict | None = None
) -> None:
    """Train one model end to end -- pretraining, contrastive, edge transformer."""
    # The train/val loaders below belong to stage 3, so they take their shape
    # from the edge stage's own settings. Anything the stage does not name
    # falls back to the top-level default.
    edge_cfg = _stage(args, "edge")
    edge_crop = edge_cfg.get("crop_size", args.crop_size)
    crop_size = tuple(edge_crop) if edge_crop is not None else None
    transformer_only = bool(edge_cfg.get("transformer_only", False))
    downscale = int(getattr(args, "downscale_factor", 1) or 1)

    train_embryos = run["train_embryos"]
    val_embryos = run["val_embryos"]
    fold = run["fold"]

    fold_dir = os.path.join(base_output_dir, run["name"])
    args.output_dir = fold_dir
    args.fold = fold
    args.train_split = train_embryos
    args.val_split = val_embryos

    if ctx.is_main:
        os.makedirs(fold_dir, exist_ok=True)
        print(f"\n{'=' * 60}")
        print(f"{run['name']}: train={train_embryos}  val={val_embryos}")
        if run["leaky_val"]:
            print("  validation overlaps training -- the score is leaky, not a metric")
        print(f"{'=' * 60}\n")

    # Stage 3 nodes off disk. Both loaders get it: the val loader has to build
    # its nodes the same way or the score would measure a different model.
    nodes_cfg = {
        k: edge_cfg[k] for k in ("coordinates_folder", "contrastive_folder")
        if edge_cfg.get(k)
    }
    if nodes_cfg:
        nodes_cfg["max_nodes"] = getattr(args, "max_dets_per_sample", 768) or 768
    # evaluate_videos reads this to score whole videos off the same folders the
    # loader trains on, instead of running the U-Net's detection.
    args._nodes_cfg = nodes_cfg

    edge_synth = _resolve_edge_sources(args, edge_cfg, nodes_cfg)

    train_video_data, train_loader, train_sampler = build_loader(
        args.data_dir,
        embryos=train_embryos,
        window_size=args.window_size,
        batch_size=edge_cfg["batch_size"],
        num_workers=args.num_workers,
        downscale_factor=downscale,
        augment=not transformer_only,
        shuffle=True,
        stable_shard=transformer_only,
        # Transformer-only freezes the detector: det_logits comes back empty,
        # so the detection loss never runs and these points are never read.
        # Deriving them is ~20 ms/frame of pure dataloader cost. The
        # pretraining loader below does train detection and still needs them.
        # Pseudo-labelled detections, for the part of this stage that trains
        # the detector. `transformer_only` freezes it for the whole stage and
        # the loss never runs, so deriving them (~20 ms/frame) would be waste;
        # `det_warmup_prop` only *delays* the unfreeze, so they are still
        # needed -- the detection loss runs for the rest of the stage.
        weak_positives_cfg=(
            {} if transformer_only else getattr(args, "weak_positives", {})
        ),
        synthetic_cfg=edge_synth,
        aug_cfg=_stage_aug(args, "edge"),
        normalization=getattr(args, "normalization", None),
        crop_size=crop_size,
        # n_videos=10,  # DEBUG
        ctx=ctx,
        seed=getattr(args, "seed", 0),
        nodes_cfg=nodes_cfg,
    )

    def _val_loader(n_videos: int):
        return build_loader(
            args.data_dir,
            embryos=val_embryos,  # train_embryos,
            window_size=args.window_size,
            batch_size=edge_cfg["val_batch_size"],
            num_workers=args.num_workers,
            downscale_factor=downscale,
            augment=False,
            weak_positives_cfg={},  # don't bother with that
            normalization=getattr(args, "normalization", None),
            crop_size=None,
            n_videos=n_videos,
            ctx=ctx,
            seed=getattr(args, "seed", 0),
            nodes_cfg=nodes_cfg,
        )[1]

    val_loader = _val_loader(run["n_val_videos"])

    # Stage 1 evaluates the detector alone, which is cheap — no edge head, no
    # feature cache — so it can afford a wider validation set than the edge
    # stage, and detection metrics are noisy enough to want one.
    pretrain_n_val = int(
        (args.pretraining or {}).get("n_val_videos") or run["n_val_videos"]
    )
    pretrain_val_loader = (
        val_loader
        if pretrain_n_val == run["n_val_videos"]
        else _val_loader(pretrain_n_val)
    )

    if ctx.is_main:
        print(f"- Training with {len(train_loader.dataset)} windows")
        leak = " (leaky, also in training)" if run["leaky_val"] else ""
        print(
            f"- Validating with {len(val_loader.dataset)} windows from "
            f"{run['n_val_videos']} video(s){leak}"
        )
        if pretrain_val_loader is not val_loader:
            print(
                f"- Pretraining validates on {len(pretrain_val_loader.dataset)} windows "
                f"from {pretrain_n_val} video(s)"
            )

    model = build_model(args, ctx)
    if ctx.is_main:
        n_params = unwrap_model(model).count_parameters()
        print(f"- Model: UNetNodeTransformer  (n_parameters={n_params / 1e6:.1f}M)")

    # --- Stage 1: detection on synthetic data ---
    synth_cfg = _stage(args, "synthetic_pretraining")
    shared = {} if shared is None else shared

    shareable = float(synth_cfg.get("synthetic_proportion", 1)) >= 1.0
    synth_cfg_weights = synth_cfg.get("weights")
    synth_weights = synth_cfg_weights or shared.get("synthetic")
    if synth_weights:
        if ctx.is_main and synth_weights == shared.get("synthetic"):
            print(f"\n-> Stage 1: Synthetic weights reused from {synth_weights}\n")
        elif ctx.is_main:
            print(f"\n-> Stage 1: Synthetic weights initialised from {synth_weights}\n")
        load_weights(model, synth_weights, fold, ctx, run_name=run["name"])

    skip_synth_training = bool(shared.get("synthetic")) and not synth_cfg_weights
    if synth_cfg.get("enabled") and not skip_synth_training:
        synth_args = _apply_stage(args, "synthetic_pretraining")
        synth_args.stage = "seg"
        synth_args.det_warmup_prop = 1.0
        synth_args.transformer_only = False

        synth_args.eval_video_paths = _paths_for_embryos(args.data_dir, val_embryos)[
            : int(synth_cfg.get("n_val_videos") or 1)
        ]
        synth_args.eval_edge_mode = "nearest"
        # optimizer_config passes through as resolved: stage block over global.
        # `lr_unet` is not overridden here -- stage 3 freezes the UNet, so
        # pretraining is the only stage where it does anything, and forcing it
        # to `lr` made the setting unreachable.

        uses_weak = bool(synth_args.loss_config.get("det_use_weak_positives"))
        _, synth_loader, synth_sampler = build_loader(
            args.data_dir,
            embryos=train_embryos,
            window_size=args.window_size,
            batch_size=synth_cfg["batch_size"],
            num_workers=args.num_workers,
            downscale_factor=downscale,
            augment=True,
            video_data=train_video_data,  # save reloading the videos
            weak_positives_cfg=getattr(args, "weak_positives", {}) if uses_weak else {},
            synthetic_cfg=synth_cfg,
            aug_cfg=_stage_aug(args, "synthetic_pretraining"),
            normalization=getattr(args, "normalization", None),
            crop_size=synth_cfg["crop_size"],
            ctx=ctx,
            seed=getattr(args, "seed", 0),
            # det_warmup_prop=1.0 below: the edge head is never supervised in
            # this stage, so the transition matrices would be built, cropped
            # and collated only to be dropped.
            load_edges=False,
        )

        if ctx.is_main:
            schedule = (
                f"{synth_cfg['iterations']} iterations"
                if synth_cfg.get("iterations")
                else f"{synth_cfg.get('epochs')} epochs"
            )
            print(
                f"\n-> Stage 1: Synthetic pretraining on {synth_cfg['crop_size']} crops, "
                f"{synth_cfg['synthetic_proportion']:.0%} synthetic, {schedule}\n"
            )

        fit(
            model,
            synth_loader,
            synth_args,
            ctx,
            train_sampler=synth_sampler,
            val_loader=pretrain_val_loader,
        )

        synth_ckpt = os.path.join(fold_dir, "ckpt_synthetic.pt")
        if ctx.is_main:
            torch.save(
                {"model_state_dict": unwrap_model(model).state_dict()}, synth_ckpt
            )
        if shareable:
            if ctx.distributed:
                dist.barrier()
            shared["synthetic"] = synth_ckpt
        elif ctx.is_main:
            print(
                f"  (synthetic_proportion {synth_cfg['synthetic_proportion']:.0%} draws real "
                f"frames too, so this stage is refit per fold rather than shared)"
            )

        del synth_loader, synth_sampler
        if ctx.device.type == "cuda":
            torch.cuda.empty_cache()

    # --- Stage 1.5: pretraining the detection UNet on real data ---
    pretraining = args.pretraining
    weights = pretraining.get("weights")
    if weights:
        load_weights(model, weights, fold, ctx, run_name=run["name"])

    if pretraining["enabled"]:
        _, seg_loader, seg_sampler = build_loader(
            args.data_dir,
            embryos=train_embryos,
            window_size=args.window_size,
            batch_size=pretraining["batch_size"],
            num_workers=args.num_workers,
            downscale_factor=downscale,
            augment=True,
            video_data=train_video_data,  # save reloading the videos
            weak_positives_cfg=getattr(args, "weak_positives", {}),
            synthetic_cfg=pretraining,
            aug_cfg=_stage_aug(args, "pretraining"),
            normalization=getattr(args, "normalization", None),
            crop_size=pretraining["crop_size"],
            ctx=ctx,
            seed=getattr(args, "seed", 0),
            load_edges=False,  # detection only, as in stage 1
        )
        seg_args = _apply_stage(args, "pretraining")
        seg_args.stage = "seg"
        seg_args.det_warmup_prop = 1.0
        seg_args.transformer_only = False

        seg_args.eval_video_paths = _paths_for_embryos(args.data_dir, val_embryos)[
            : int(pretraining.get("n_val_videos") or 1)
        ]

        seg_args.eval_edge_mode = "nearest"
        # See stage 1: optimizer_config passes through as resolved.

        if ctx.is_main:
            synth_prop = float(pretraining.get("synthetic_proportion", 0.0) or 0.0)
            source = (
                f"real data + {synth_prop:.0%} synthetic"
                if pretraining.get("synthetic_dir") and synth_prop > 0
                else "real data"
            )
            print(
                f"\n-> Stage 1.5: Pretraining the detection UNet on {source}, "
                f"{seg_loader.dataset.crop_size} crops, "
                f"{len(seg_loader.dataset)} windows, "
                + (
                    f"{seg_args.iterations} iterations\n"
                    if getattr(seg_args, "iterations", None)
                    else f"{seg_args.epochs} epochs\n"
                )
            )

        fit(
            model,
            seg_loader,
            seg_args,
            ctx,
            train_sampler=seg_sampler,
            val_loader=pretrain_val_loader,
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
    contrastive = getattr(args, "contrastive", {}) or {}
    use_contrastive = bool(contrastive.get("enabled", True)) and "contrastive" in (
        getattr(args, "unet_features", ()) or ()
    )

    weights = contrastive.get("weights") if use_contrastive else None
    if weights:
        # Initialise the embedding UNet from a previous run; the stage still
        # trains on top of it below.
        load_weights(
            model,
            weights,
            fold,
            ctx,
            prefixes=("embed_unet.", "embed_head."),
            run_name=run["name"],
        )
    if use_contrastive:
        _, emb_loader, emb_sampler = build_loader(
            args.data_dir,
            embryos=train_embryos,
            window_size=args.window_size,
            batch_size=contrastive["batch_size"],
            num_workers=args.num_workers,
            downscale_factor=downscale,
            augment=contrastive.get("augment", True),
            video_data=train_video_data,  # save reloading the videos
            weak_positives_cfg={},  # detection is frozen here
            aug_cfg=_stage_aug(args, "contrastive"),
            normalization=getattr(args, "normalization", None),
            crop_size=contrastive.get("crop_size"),
            ctx=ctx,
            seed=getattr(args, "seed", 0),
            stable_shard=True,  # node coords are cached per rank
        )
        if ctx.is_main:
            print(
                f"\n-> Stage 2: Contrastive training the embedding UNet on "
                f"{contrastive['crop_size']} crops, "
                f"{len(emb_loader.dataset)} windows, "
                f"{contrastive['epochs']} epochs\n"
            )
        args.stage = "contrastive"
        fit_contrastive(
            model,
            emb_loader,
            args,
            ctx,
            train_sampler=emb_sampler,
            val_loader=val_loader,
        )
        del emb_loader, emb_sampler
        if ctx.device.type == "cuda":
            torch.cuda.empty_cache()

    # --- Stage 3: edge transformer on frozen coordinates + embeddings ---

    edge_weights = edge_cfg.get("weights")
    edge_enabled = bool(edge_cfg.get("enabled", True))
    if ctx.is_main and not edge_enabled:
        print("\n-> Stage 3: skipped (edge.enabled is false)\n")
    if ctx.is_main and edge_enabled:
        schedule = (
            f"{edge_cfg['iterations']} iterations"
            if edge_cfg.get("iterations")
            else f"{edge_cfg.get('epochs')} epochs"
        )
        frozen = ", backbone frozen" if transformer_only else ""
        init = f", initialised from {edge_weights}" if edge_weights else ""
        print(
            f"\n-> Stage 3: Edge transformer on {crop_size or 'full'} crops, "
            f"{len(train_loader.dataset)} windows, batch {edge_cfg['batch_size']}, "
            f"{schedule}{frozen}{init}\n"
        )

    if edge_weights:
        # `transformer.` is in the prefixes here and nowhere else. The default
        # drops it because the checkpoints the other stages resume from come
        # from detection-only training, where the transformer never left init
        # and copying it would only pin the edge architecture to that run's.
        # `edge.weights` is the opposite case: it names a previous *edge* stage,
        # whose transformer is the thing worth having, and dropping it silently
        # restarted the edge head from scratch while the log said the load was
        # clean -- the "Skipped modules" line was the only hint.
        #
        # A shape mismatch is still skipped and reported, so pointing this at a
        # run with a different `transformer_blocks`/`transformer_hidden`
        # degrades to a backbone-only load rather than raising.
        load_weights(
            model, edge_weights, fold, ctx, run_name=run["name"],
            prefixes=("unet.", "detect_head.", "detect_head_fine.",
                      "offset_head.", "annot_head.", "transformer."),
        )

    edge_args = _apply_stage(args, "edge")
    edge_args.eval_video_paths = _paths_for_embryos(args.data_dir, val_embryos)[
        : int(edge_cfg.get("n_val_videos") or 1)
    ]
    edge_args.eval_edge_mode = "transformer"

    if edge_enabled:
        fit(
            model,
            train_loader,
            edge_args,
            ctx,
            train_sampler=train_sampler,
            val_loader=val_loader,
        )

    if ctx.is_main:
        torch.save(
            {"model_state_dict": unwrap_model(model).state_dict()},
            os.path.join(fold_dir, "ckpt_final.pt"),
        )

        print("\n-> Saved model weights to: ", os.path.join(fold_dir, "ckpt_final.pt"))

    del model, train_loader, train_sampler, val_loader
    if ctx.device.type == "cuda":
        torch.cuda.empty_cache()
