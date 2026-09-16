"""Loaders, model, and leave-one-embryo-out, mirroring `division_cls.main`.

The split is by embryo, which matters more here than anywhere else in the repo.
Every stack carries its own annotated lineage in its own part of the volume, so
a model that has learnt "the annotated cells are the ones over there" cannot
carry that to a held-out embryo. The out-of-fold AUC is therefore the whole
experiment: a high in-fold number would say nothing at all.
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, DistributedSampler

from annotated_cls.dataset import AnnotatedCellDataset
from annotated_cls.model import AnnotatedNet
from annotated_cls.train import fit
from params import EMBRYOS
from training.distributed import worker_init_fn


def _opt_int(v):
    if v is None or v == "" or str(v).lower() == "none":
        return None
    return int(v)


def build_loader(args, embryos, augment, ctx=None, shuffle=None):
    dataset = AnnotatedCellDataset(
        args.data_dir, args.preds_dir, embryos=embryos,
        patch_yx=int(args.patch_yx), patch_z=int(args.patch_z),
        n_frames=int(args.n_frames),
        # Validation keeps the harder, more honest mix: subsampling negatives
        # differently between fit and eval would move the operating point, and
        # AUC is read across the whole ranking.
        neg_per_pos=float(args.neg_per_pos if augment else args.val_neg_per_pos),
        positive_multiplier=int(args.positive_multiplier) if augment else 1,
        augment=augment, aug_cfg=getattr(args, "augmentations", None),
        # Validation is capped separately: the full held-out embryo is ~434k
        # crops at this negative ratio and every one is a zarr read, which would
        # cost more than the fit. A fixed subset of stacks keeps the AUC
        # comparable between evaluations.
        max_stacks=_opt_int(getattr(args, "max_stacks", None)) if augment
        else _opt_int(getattr(args, "val_max_stacks", None)
                      or getattr(args, "max_stacks", None)),
        seed=int(getattr(args, "seed", 0)),
    )
    if shuffle is None:
        shuffle = augment
    sampler = None
    if ctx is not None and ctx.distributed:
        sampler = DistributedSampler(dataset, num_replicas=ctx.world_size,
                                     rank=ctx.rank, shuffle=shuffle,
                                     seed=getattr(args, "seed", 0))
    loader = DataLoader(
        dataset, batch_size=args.batch_size if augment else args.val_batch_size,
        shuffle=shuffle and sampler is None, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, worker_init_fn=worker_init_fn,
    )
    return loader, sampler, dataset


def build_model(args, ctx):
    model = AnnotatedNet(
        backbone=args.backbone, pretrained=bool(args.pretrained),
        patch_yx=int(args.patch_yx), patch_z=int(args.patch_z),
        n_frames=int(args.n_frames), dropout=float(args.dropout),
        grad_checkpointing=bool(getattr(args, "grad_checkpointing", False)),
        verbose=int(ctx.is_main) if ctx is not None else 1,
    )
    device = ctx.device if ctx is not None else "cpu"
    model = model.to(device)
    if ctx is not None and ctx.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None)
    return model


def _runs(args):
    runs = []
    for fold, val_embryo in enumerate(EMBRYOS):
        if fold in args.selected_folds:
            runs.append({"name": f"fold_{fold}", "fold": fold,
                         "train_embryos": [e for e in EMBRYOS if e != val_embryo],
                         "val_embryos": [val_embryo], "leaky": False})
    if getattr(args, "fullfit", False):
        runs.append({"name": "fold_fullfit", "fold": 0,
                     "train_embryos": list(EMBRYOS),
                     "val_embryos": [EMBRYOS[0]], "leaky": True})
    return runs


def k_fold(args, ctx):
    base = args.output_dir
    results = {}
    for run in _runs(args):
        args.output_dir = os.path.join(base, run["name"])
        args.fold = run["fold"]
        if ctx.is_main:
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"\n{'=' * 66}")
            print(f"{run['name']}: train={run['train_embryos']}  val={run['val_embryos']}")
            if run["leaky"]:
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
        results[run["name"]] = fit(model, train_loader, args, ctx,
                                   train_sampler=sampler, val_loader=val_loader)
        del model
        torch.cuda.empty_cache()

    args.output_dir = base
    if ctx.is_main and results:
        print(f"\n{'=' * 66}\nSUMMARY\n{'=' * 66}")
        for name, m in results.items():
            if m:
                print(f"  {name:14s} AUC {m.get('auc', float('nan')):.4f}  "
                      f"ap {m.get('ap', float('nan')):.4f}")
    return results
