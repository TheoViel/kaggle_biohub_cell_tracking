"""Loader, model and cross-validation orchestration, mirroring ``tracking.main``.

The fold convention is the tracking pipeline's: fold ``i`` holds out
``EMBRYOS[i]``, and ``fullfit`` adds a model trained on everything whose
validation score is leaky by construction.
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, DistributedSampler

from edge_transformer.data import load_videos
from edge_transformer.dataset import EdgeDataset, collate_graphs
from edge_transformer.train import fit
from model_zoo.hoct import HOCT
from params import EMBRYOS
from training.distributed import unwrap_model, worker_init_fn


def _opt_int(value):
    """A None-defaulted numeric flag, which argparse hands back as a string."""
    if value is None or value == "" or str(value).lower() == "none":
        return None
    return int(value)


def build_dataset(args, embryos: list[str], train: bool, n_videos=None,
                  verbose: bool = False) -> EdgeDataset:
    videos = load_videos(
        args.nodes_folder,
        args.data_dir,
        embryos=embryos,
        n_videos=_opt_int(n_videos),
        use_embeddings=bool(getattr(args, "use_embeddings", True)),
        embeddings_key=getattr(args, "embeddings_key", None) or "embeddings",
        # Training only. Validation must stay a clean read of the annotation --
        # pseudo-labels are this model's own output, so scoring against them
        # would measure agreement with a previous round, not with the truth.
        pl_folder=getattr(args, "pl_folder", None) if train else None,
        verbose=verbose,
    )
    return EdgeDataset(
        videos,
        window_size=int(args.window_size),
        max_nodes_per_frame=int(args.max_nodes_per_frame),
        n_neighbors=int(args.n_neighbors),
        max_distance_um=float(args.max_distance_um),
        max_dt=int(args.max_dt),
        pos_scale=float(args.pos_scale),
        drift_correction=bool(args.edge_drift_correction),
        use_conf_score=bool(getattr(args, "use_conf_score", False)),
        require_links=True,
        augment=train and bool(getattr(args, "augment", True)),
        flip_p=float(getattr(args, "flip_p", 0.5)),
        rot_p=float(getattr(args, "rot_p", 0.5)),
        jitter_um=float(getattr(args, "jitter_um", 0.3)),
        time_flip_p=float(getattr(args, "time_flip_p", 0.0)),
        pl_weight=float(getattr(args, "pl_weight", 0.0)) if train else 0.0,
        # Training only: validation must read the annotation as it is.
        division_weight=(float(getattr(args, "division_weight", 1.0))
                         if train else 1.0),
    )


def build_loader(
    args,
    dataset: EdgeDataset,
    batch_size: int,
    shuffle: bool,
    ctx=None,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if ctx is not None and ctx.distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=ctx.world_size, rank=ctx.rank,
            shuffle=shuffle, seed=getattr(args, "seed", 0),
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_graphs,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    return loader, sampler


def build_model(args, dataset: EdgeDataset, ctx):
    """Build HOCT for this dataset's feature widths and wrap it for the process."""
    model = HOCT(
        node_feat_dim=dataset.node_feat_dim,
        edge_feat_dim=dataset.edge_feat_dim if args.use_edge_feats else 0,
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_node_blocks=int(args.n_node_blocks),
        n_edge_blocks=int(args.n_edge_blocks),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        input_dropout=float(args.input_dropout),
        pos_scale=float(args.pos_scale),
        min_wavelength=float(args.min_wavelength_um),
        max_wavelength=float(args.max_wavelength_um),
        max_distance=float(args.attn_max_distance_um),
        householder=bool(args.householder),
    )
    device = ctx.device if ctx is not None else torch.device("cpu")
    model = model.to(device)
    if ctx is not None and ctx.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    return model


def _training_runs(args) -> list[dict]:
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

    train_ds = build_dataset(args, run["train_embryos"], train=True,
                             n_videos=getattr(args, "n_train_videos", None),
                             verbose=ctx.is_main and bool(getattr(args, "pl_folder", None)))
    val_ds = build_dataset(args, run["val_embryos"], train=False,
                           n_videos=getattr(args, "n_val_videos", None))
    train_loader, train_sampler = build_loader(
        args, train_ds, int(args.batch_size), shuffle=True, ctx=ctx
    )
    val_loader, _ = build_loader(
        args, val_ds, int(args.val_batch_size), shuffle=False, ctx=None
    )

    if ctx.is_main:
        print(f"- Training on {len(train_ds)} windows from {len(train_ds.videos)} videos, "
              f"validating on {len(val_ds)} from {len(val_ds.videos)}")
        features = (
            f"{train_ds.embed_dim}-d embeddings" if train_ds.embed_dim
            else "geometry only"
        )
        print(f"- Nodes: {args.nodes_folder} ({features})")
        print(f"- Candidates: {args.n_neighbors} neighbours within "
              f"{args.max_distance_um} um, dt <= {args.max_dt}"
              + ("  [drift-corrected]" if args.edge_drift_correction else ""))

    model = build_model(args, train_ds, ctx)
    if ctx.is_main:
        core = unwrap_model(model)
        print(f"- Model: HOCT [{args.n_node_blocks} node + {args.n_edge_blocks} edge "
              f"blocks, d={args.d_model}]  "
              f"(n_parameters={core.count_parameters() / 1e6:.1f}M)")

    if args.weights:
        state = torch.load(args.weights, map_location=ctx.device, weights_only=False)
        unwrap_model(model).load_state_dict(state.get("model_state_dict", state))
        if ctx.is_main:
            print(f"- Loaded weights from {args.weights}")

    fit(model, train_loader, args, ctx,
        train_sampler=train_sampler, val_loader=val_loader, val_dataset=val_ds)

    del model, train_loader, train_sampler, val_loader
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
