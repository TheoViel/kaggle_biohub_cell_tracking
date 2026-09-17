"""Loaders, model and the per-embryo cross-validation, mirroring `embeddings.main`."""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, DistributedSampler

from division_cls.dataset import DivisionDataset
from division_cls.model import DivisionNet
from division_cls.synthetic import MixedDivisionDataset, SyntheticClipDataset
from division_cls.train import fit
from params import EMBRYOS
from training.distributed import worker_init_fn


def _opt_int(value):
    """A None-defaulted numeric flag, which argparse hands back as a string.

    DEFAULTS entries defaulting to None get no ``type=``, so ``--max_stacks 14``
    arrives as "14" and every comparison against it raises. Same helper, same
    reason, as ``embeddings/main.py``.
    """
    if value is None or value == "" or str(value).lower() == "none":
        return None
    return int(value)


def build_loader(args, embryos, augment: bool, ctx=None, shuffle=None):
    dataset = DivisionDataset(
        args.data_dir,
        embryos=embryos,
        patch_yx=int(args.patch_yx),
        patch_z=int(args.patch_z),
        n_frames=int(args.n_frames),
        negatives=args.negatives,
        neg_per_pos=float(args.neg_per_pos),
        exclude_window=int(args.exclude_window),
        # Only the training set is oversampled. Doing it in validation would
        # inflate the positive rate and make every ranking metric look better
        # than it is.
        positive_multiplier=int(args.positive_multiplier) if augment else 1,
        augment=augment,
        aug_cfg=getattr(args, "augmentations", None),
        max_stacks=_opt_int(getattr(args, "max_stacks", None)),
        seed=int(getattr(args, "seed", 0)),
        candidates=getattr(args, "candidates", None) or None,
        fork_align=getattr(args, "fork_align", "") or "",
    )
    # Synthetic crops, training split only. Validation must stay a clean read
    # of the annotation -- mixing generated positives into it would measure
    # agreement with the generator, not with the data.
    if augment and getattr(args, "synthetic_cache", None) and \
            float(getattr(args, "synthetic_proportion", 0.0) or 0.0) > 0:
        synth = SyntheticClipDataset(
            args.synthetic_cache, augment=True,
            positive_multiplier=int(args.positive_multiplier),
            seed=int(getattr(args, "seed", 0)))
        dataset = MixedDivisionDataset(
            dataset, synth, float(args.synthetic_proportion),
            seed=int(getattr(args, "seed", 0)))

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
    model = DivisionNet(
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
        grad_checkpointing=bool(getattr(args, "grad_checkpointing", False)),
        # Fusing the proposal's geometry only makes sense in candidate mode,
        # where a "sample" is a proposed fork rather than a cell.
        n_features=(5 if (getattr(args, "candidates", None)
                          and bool(getattr(args, "fuse_features", True))) else 0),
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
    """Fold `i` holds out `EMBRYOS[i]`, as everywhere else in this repo.

    The split has to be by embryo and not at random: a cell appears at many
    frames and its neighbours appear at all of them, so a random split puts
    near-copies of a training crop in validation and the number stops meaning
    anything.
    """
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
        if ctx.is_main:
            print(f"- train : {train_ds.summary}")
            print(f"- val   : {val_ds.summary}\n")

        model = build_model(args, ctx)
        # Fix the feature standardisation from the training split's own rows,
        # before any oversampling. `train_ds.features` is one row per distinct
        # sample; `train_ds.index` is the oversampled epoch and must not be used.
        target = model.module if hasattr(model, "module") else model
        if getattr(target, "n_features", 0) and getattr(train_ds, "features", None) is not None:
            target.set_feature_stats(train_ds.features)
            if ctx.is_main:
                print(f"- feat norm : mean {target.feat_mean.tolist()} "
                      f"std {target.feat_std.tolist()}\n")
        results[run["name"]] = fit(model, train_loader, args, ctx,
                                   train_sampler=sampler, val_loader=val_loader)
        del model
        torch.cuda.empty_cache()

    args.output_dir = base
    if ctx.is_main and results:
        print(f"\n{'=' * 66}\nSUMMARY\n{'=' * 66}")
        for name, m in results.items():
            if m:
                print(f"  {name:14s} ap {m.get('ap', float('nan')):.4f} "
                      f"(chance {m.get('chance_ap', float('nan')):.4f})  "
                      f"auc {m.get('auc', float('nan')):.3f}  "
                      f"prec@k {m.get('prec@k', float('nan')):.3f}")
    return results
