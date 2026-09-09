"""Fit loop for the division classifier: BCE, and metrics that survive 1:12."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler

from training.distributed import unwrap_model
from training.optim import define_optimizer, define_scheduler


@torch.no_grad()
def evaluate(model, loader, device, use_fp16=True):
    """Average precision, AUC and precision@k on the held-out embryo.

    Accuracy is meaningless at this imbalance -- predicting "never" scores 92% --
    so the reported numbers are the ranking ones. `precision@k` takes k equal to
    the number of true divisions present, which is the operating point a
    post-processing gate would actually run at.
    """
    model.eval()
    scores, labels = [], []
    for batch in loader:
        clip = batch["clip"].to(device, non_blocking=True)
        with torch.autocast("cuda", torch.float16, enabled=use_fp16 and device.type == "cuda"):
            logits = model(clip)
        scores.append(logits.float().cpu())
        labels.append(batch["label"])
    if not scores:
        return {}

    s = torch.cat(scores).numpy()
    y = torch.cat(labels).numpy().astype(int)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return {"n": len(y), "n_pos": n_pos}

    from sklearn.metrics import average_precision_score, roc_auc_score

    order = np.argsort(-s)
    return {
        "n": len(y),
        "n_pos": n_pos,
        "auc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
        "chance_ap": float(y.mean()),
        "prec@k": float(y[order[:n_pos]].sum() / n_pos),
        "recall@2k": float(y[order[: 2 * n_pos]].sum() / n_pos),
    }


def fit(model, train_loader, args, ctx, train_sampler=None, val_loader=None):
    def log(*msg):
        if ctx.is_main:
            print(*msg)

    device = ctx.device
    use_fp16 = getattr(args, "fp16", True) and device.type == "cuda"

    cfg = args.optimizer_config
    optimizer = define_optimizer(
        model, cfg["name"], lr=cfg["lr"],
        # A pretrained backbone wants a gentler rate than a head initialised
        # from nothing; `lr_encoder` is how the rest of this repo says that.
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

    # Oversampling already lifts the positives; pos_weight on top would double
    # count. Left configurable because the two trade off and only a sweep says
    # where.
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
            target = batch["label"].to(device, non_blocking=True)

            with torch.autocast("cuda", torch.float16, enabled=use_fp16):
                loss = criterion(model(clip), target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.get("max_grad_norm"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running.append(loss.item())
            step += 1

            if step % args.log_every == 0:
                log(f"step {step:>5d}/{total_steps}  loss {np.mean(running[-args.log_every:]):.4f}"
                    f"  lr {scheduler.get_last_lr()[0]:.2e}")
            if val_loader is not None and args.eval_every and step % args.eval_every == 0:
                m = evaluate(model, val_loader, device, use_fp16)
                if m and ctx.is_main:
                    log(f"    val  ap {m.get('ap', float('nan')):.4f} "
                        f"(chance {m.get('chance_ap', float('nan')):.4f})  "
                        f"auc {m.get('auc', float('nan')):.3f}  "
                        f"prec@k {m.get('prec@k', float('nan')):.3f}  "
                        f"[{m['n_pos']} pos of {m['n']}]")
                model.train()
        epoch += 1

    if ctx.is_main and getattr(args, "output_dir", None):
        torch.save(
            {"model_state_dict": unwrap_model(model).state_dict(),
             "model_config": getattr(args, "model_config", {})},
            f"{args.output_dir}/ckpt_final.pt",
        )
        log(f"\n-> saved {args.output_dir}/ckpt_final.pt")

    if val_loader is not None:
        m = evaluate(model, val_loader, device, use_fp16)
        if m and ctx.is_main:
            log(f"\nfinal val: " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float)
                                             else f"{k} {v}" for k, v in m.items()))
        return m
    return {}
