"""Loaders, model and the per-embryo cross-validation, mirroring `division_cls.main`."""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, DistributedSampler

from cell_merging_cls.dataset import MergePairDataset
from cell_merging_cls.model import MergeNet
from cell_merging_cls.train import fit
from params import EMBRYOS
from training.distributed import worker_init_fn


def _opt_int(value):
    if value is None or value == "" or str(value).lower() == "none":
        return None
    return int(value)


def build_loader(args, embryos, augment: bool, ctx=None, shuffle=None, probe=False):
    dataset = MergePairDataset(
        args.data_dir,
        args.preds_dir,
        embryos=embryos,
        patch_yx=int(args.patch_yx),
        patch_z=int(args.patch_z),
        n_frames=int(args.n_frames),
        max_pair_um=float(args.max_pair_um),
        dup_radius=float(args.dup_radius),
        neg_per_pos=float(args.neg_per_pos),
        unlabelled_per_pos=float(args.unlabelled_per_pos),
        # Oversampling is training-only: doing it in validation would move the
        # positive rate and flatter every ranking metric.
        positive_multiplier=int(args.positive_multiplier) if augment else 1,
        sister_multiplier=int(args.sister_multiplier) if augment else 1,
        augment=augment,
        aug_cfg=getattr(args, "augmentations", None),
        max_stacks=_opt_int(getattr(args, "max_stacks", None)),
        probe=probe,
        seed=int(getattr(args, "seed", 0)),
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
        batch_size=args.batch_size if augment else args.val_batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
    return loader, sampler, dataset


def build_model(args, ctx):
    model = MergeNet(
        backbone=args.backbone,
        pretrained=bool(args.pretrained),
        patch_yx=int(args.patch_yx),
        patch_z=int(args.patch_z),
        n_frames=int(args.n_frames),
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        dim_feedforward=int(args.dim_feedforward),
        dropout=float(args.dropout),
        min_tokens=int(args.min_tokens),
        geom_dim=int(args.geom_dim),
        grad_checkpointing=bool(getattr(args, "grad_checkpointing", False)),
        verbose=int(ctx.is_main) if ctx is not None else 1,
    )
    device = ctx.device if ctx is not None else "cpu"
    model = model.to(device)
    if ctx is not None and ctx.distributed:
        if getattr(args, "sync_bn", False):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    return model


def _training_runs(args) -> list[dict]:
    runs = []
    for fold, val_embryo in enumerate(EMBRYOS):
        if fold in args.selected_folds:
            runs.append({
                "name": f"fold_{fold}", "fold": fold,
                "train_embryos": [e for e in EMBRYOS if e != val_embryo],
                "val_embryos": [val_embryo], "leaky_val": False,
            })
    if getattr(args, "fullfit", False):
        runs.append({
            "name": "fold_fullfit", "fold": 0,
            "train_embryos": list(EMBRYOS), "val_embryos": [EMBRYOS[0]],
            "leaky_val": True,
        })
    return runs


def k_fold(args, ctx):
    base = args.output_dir
    results = {}
    for run in _training_runs(args):
        fold_dir = os.path.join(base, run["name"])
        args.output_dir = fold_dir
        args.fold = run["fold"]
        if ctx.is_main:
            os.makedirs(fold_dir, exist_ok=True)
            print(f"\n{'=' * 66}")
            print(f"{run['name']}: train={run['train_embryos']}  val={run['val_embryos']}")
            if run["leaky_val"]:
                print("  validation overlaps training -- leaky, not a metric")
            print(f"{'=' * 66}")

        train_loader, sampler, train_ds = build_loader(
            args, run["train_embryos"], augment=True, ctx=ctx)
        val_loader, _, val_ds = build_loader(
            args, run["val_embryos"], augment=False, ctx=None, shuffle=False)
        probe_loader = probe_ds = None
        if getattr(args, "probe", True):
            probe_loader, _, probe_ds = build_loader(
                args, run["val_embryos"], augment=False, ctx=None, shuffle=False,
                probe=True)
        if ctx.is_main:
            print(f"- train : {train_ds.summary}")
            print(f"- val   : {val_ds.summary}")
            if probe_ds is not None:
                print(f"- probe : {probe_ds.summary}")
            print()

        model = build_model(args, ctx)
        results[run["name"]] = fit(model, train_loader, args, ctx,
                                   train_sampler=sampler, val_loader=val_loader,
                                   probe_loader=probe_loader)
        del model
        torch.cuda.empty_cache()

    args.output_dir = base
    if ctx.is_main and results:
        print(f"\n{'=' * 66}\nSUMMARY\n{'=' * 66}")
        for name, m in results.items():
            if m:
                print(f"  {name:14s} ap {m.get('ap', float('nan')):.4f}  "
                      f"probe ap {m.get('probe_ap', float('nan')):.4f}  "
                      f"probe sisterFP@0.9 {m.get('probe_sister_fp@0.9', float('nan')):.4f}")
    return results
