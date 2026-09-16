"""The fit loop. AUC is the number to watch, and it is watched out of fold.

AUC rather than AP because the decision this feeds is a *ranking* one: nodes are
trimmed from the graph worst-first until the node-count adjustment stops paying,
so what matters is whether a positive outranks a negative, at every operating
point rather than at the top of the list. AP would weight the head of the
ranking, which is the part this never uses.

Chance is 0.5 whatever the imbalance, which matters here: the positive rate is
0.466% before subsampling and whatever `neg_per_pos` makes it afterwards, so a
prevalence-sensitive number could not be compared between the training mix and
the real one.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn


def _metrics(y, s):
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y).reshape(-1)
    s = np.asarray(s).reshape(-1)
    if y.min() == y.max():
        return {"auc": float("nan"), "ap": float("nan"), "pos_rate": float(y.mean())}
    return {
        "auc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
        "pos_rate": float(y.mean()),
    }


@torch.no_grad()
def evaluate(model, loader, device, fp16=True):
    model.eval()
    ys, ss = [], []
    for batch in loader:
        clip = batch["clip"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=fp16 and device.type == "cuda"):
            logits = model(clip)
        ss.append(logits.float().cpu().numpy())
        ys.append(batch["target"].numpy())
    model.train()
    if not ys:
        return {"auc": float("nan"), "ap": float("nan"), "pos_rate": float("nan")}
    return _metrics(np.concatenate(ys), np.concatenate(ss))


def fit(model, train_loader, args, ctx, train_sampler=None, val_loader=None):
    device = ctx.device
    main = ctx is None or ctx.is_main
    log = print if main else (lambda *a, **k: None)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(args.lr),
                            weight_decay=float(args.weight_decay))
    iterations = int(args.iterations)
    warmup = max(1, int(float(args.warmup_prop) * iterations))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warmup if it < warmup
        else 0.5 * (1 + np.cos(np.pi * (it - warmup) / max(1, iterations - warmup))))
    scaler = torch.amp.GradScaler(enabled=bool(args.fp16) and device.type == "cuda")
    pos_weight = torch.tensor(float(getattr(args, "pos_weight", 1.0)), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    it, epoch, t0, running = 0, 0, time.time(), []
    last = {}
    while it < iterations:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            if it >= iterations:
                break
            clip = batch["clip"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=bool(args.fp16) and device.type == "cuda"):
                loss = loss_fn(model(clip), target)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if getattr(args, "max_grad_norm", 0):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, float(args.max_grad_norm))
            scaler.step(opt)
            scaler.update()
            sched.step()
            running.append(float(loss.detach()))
            it += 1

            if it % int(args.log_every) == 0:
                log(f"  [{it:>5}/{iterations}] loss {np.mean(running[-100:]):.4f}  "
                    f"lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s")
            if val_loader is not None and it % int(args.eval_every) == 0:
                m = evaluate(model, val_loader, device, fp16=bool(args.fp16))
                last = m
                log(f"    val  AUC {m['auc']:.4f}   ap {m['ap']:.4f} "
                    f"(positive rate {m['pos_rate']:.4f})")
        epoch += 1

    if val_loader is not None:
        last = evaluate(model, val_loader, device, fp16=bool(args.fp16))
        log(f"  -> final val AUC {last['auc']:.4f}  ap {last['ap']:.4f}")
    if main:
        import os

        torch.save(
            (model.module if hasattr(model, "module") else model).state_dict(),
            os.path.join(args.output_dir, "ckpt_final.pt"))
    return last
