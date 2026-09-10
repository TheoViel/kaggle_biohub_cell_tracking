"""Fit loop for the merge classifier.

BCE, and metrics chosen for an asymmetric error cost. Merging two real cells
destroys a trajectory; missing a duplicate leaves one spurious track. `edge_pp`
already measured that asymmetry from the other side -- a distance-only merge
fired 10 508 times for 34 opportunities -- so the number to watch is not AP but
**how often a sister pair is called a duplicate at the threshold the
post-processing would actually run at**.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler

from training.distributed import unwrap_model
from training.optim import define_optimizer, define_scheduler

# Where the merge would be applied. Reported at several so the operating point
# is chosen from a curve rather than from one number.
THRESHOLDS = (0.5, 0.9, 0.99)


@torch.no_grad()
def evaluate(model, loader, device, use_fp16=True, thresholds=THRESHOLDS):
    """Ranking metrics, plus the sister false-merge rate at each threshold."""
    model.eval()
    scores, labels, sisters = [], [], []
    for batch in loader:
        clip = batch["clip"].to(device, non_blocking=True)
        geom = batch["geom"].to(device, non_blocking=True)
        with torch.autocast("cuda", torch.float16,
                            enabled=use_fp16 and device.type == "cuda"):
            logits = model(clip, geom)
        scores.append(logits.float().cpu())
        labels.append(batch["label"])
        sisters.append(batch["is_sister"])
    if not scores:
        return {}

    s = torch.sigmoid(torch.cat(scores)).numpy()
    y = torch.cat(labels).numpy().astype(int)
    sis = torch.cat(sisters).numpy().astype(bool)
    n_pos = int(y.sum())
    out = {"n": len(y), "n_pos": n_pos, "n_sister": int(sis.sum())}
    if n_pos == 0 or n_pos == len(y):
        return out

    from sklearn.metrics import average_precision_score, roc_auc_score

    out.update({
        "auc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
        "chance_ap": float(y.mean()),
    })
    for th in thresholds:
        fired = s >= th
        out[f"recall@{th}"] = float(fired[y == 1].mean()) if n_pos else float("nan")
        # The one that decides whether this is safe to switch on.
        out[f"sister_fp@{th}"] = (float(fired[sis].mean()) if sis.any()
                                  else float("nan"))
        neg = y == 0
        out[f"neg_fp@{th}"] = float(fired[neg].mean()) if neg.any() else float("nan")
    return out


def _fmt(m, prefix=""):
    if not m or "ap" not in m:
        return f"{prefix}(no positives: {m.get('n_pos', 0)} of {m.get('n', 0)})"
    return (f"{prefix}ap {m['ap']:.4f} (chance {m['chance_ap']:.4f})  "
            f"auc {m['auc']:.3f}  "
            f"rec@0.9 {m['recall@0.9']:.3f}  "
            f"sisterFP@0.9 {m['sister_fp@0.9']:.4f}  "
            f"[{m['n_pos']} pos, {m['n_sister']} sisters, of {m['n']}]")


def fit(model, train_loader, args, ctx, train_sampler=None, val_loader=None,
        probe_loader=None):
    def log(*msg):
        if ctx.is_main:
            print(*msg)

    device = ctx.device
    use_fp16 = getattr(args, "fp16", True) and device.type == "cuda"

    cfg = args.optimizer_config
    optimizer = define_optimizer(
        model, cfg["name"], lr=cfg["lr"],
        lr_encoder=cfg.get("lr_encoder", cfg["lr"]),
        betas=tuple(cfg["betas"]),
        weight_decay=cfg.get("weight_decay", 0.0),
    )
    iterations = int(getattr(args, "iterations", 0) or 0)
    epochs = int(getattr(args, "epochs", 0) or 0)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = iterations if iterations else epochs * steps_per_epoch
    scheduler = define_scheduler(
        optimizer, train_loader, epochs or 1,
        warmup_prop=cfg.get("warmup_prop", 0.05), total_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    pos_weight = float(getattr(args, "pos_weight", 1.0) or 1.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device) if pos_weight != 1.0 else None
    )

    step, epoch, running = 0, 0, []
    log(f"- {total_steps} steps, {steps_per_epoch} per epoch\n")
    while step < total_steps:
        model.train()
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            if step >= total_steps:
                break
            clip = batch["clip"].to(device, non_blocking=True)
            geom = batch["geom"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)

            with torch.autocast("cuda", torch.float16, enabled=use_fp16):
                loss = criterion(model(clip, geom), target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.get("max_grad_norm"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running.append(loss.item())
            step += 1

            if step % args.log_every == 0:
                log(f"step {step:>5d}/{total_steps}  "
                    f"loss {np.mean(running[-args.log_every:]):.4f}"
                    f"  lr {scheduler.get_last_lr()[0]:.2e}")
            if args.eval_every and step % args.eval_every == 0:
                if val_loader is not None:
                    log("    val   " + _fmt(evaluate(model, val_loader, device, use_fp16)))
                if probe_loader is not None:
                    log("    probe " + _fmt(evaluate(model, probe_loader, device, use_fp16)))
                model.train()
        epoch += 1

    if ctx.is_main and getattr(args, "output_dir", None):
        torch.save(
            {"model_state_dict": unwrap_model(model).state_dict(),
             "model_config": getattr(args, "model_config", {})},
            f"{args.output_dir}/ckpt_final.pt",
        )
        log(f"\n-> saved {args.output_dir}/ckpt_final.pt")

    out = {}
    if val_loader is not None:
        out = evaluate(model, val_loader, device, use_fp16)
        log("\nfinal val   " + _fmt(out))
    if probe_loader is not None:
        probe = evaluate(model, probe_loader, device, use_fp16)
        log("final probe " + _fmt(probe))
        out = {**out, **{f"probe_{k}": v for k, v in probe.items()}}
    return out
