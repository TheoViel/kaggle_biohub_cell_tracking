"""Evaluation utilities for the cell tracking model."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from params import TRAIN_DIR

from tracking.targets import build_matched_edge_targets
from tracking.fit_utils import (
    collate_frame_det,
    encode_nodes,
    filter_frame_det,
    split_frame_det,
)


def _evaluate_pair(
    logits: torch.Tensor,  # (N_t, N_t1)
    target: torch.Tensor,  # (N_t, N_t1)
) -> tuple[float, int, int, int]:
    """Returns (focal_bce_loss, tp, fp, fn) for one unpadded pair.

    Links are picked by argmax over the parent axis — each child takes its
    best-scoring parent — matching ``predict.py``'s default
    ``edge_selection="argmax"``. A fixed probability cutoff is useless here:
    the softmax is normalised across all N candidate parents, so scores scale
    like 1/N and never approach 0.5 once a frame holds a few hundred
    detections, which pins the jaccard at 0 however well the model ranks.
    """
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    if not active_rows.any():
        return 0.0, 0, 0, 0

    from tracking.losses import edge_loss

    loss_val = edge_loss(logits, target).item()

    probs = torch.softmax(logits, dim=0)
    preds = torch.zeros_like(probs, dtype=torch.bool)
    preds[probs.argmax(dim=0), torch.arange(probs.shape[1], device=probs.device)] = True

    target_bool = target > 0.5
    mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)

    tp = (preds[mask] & target_bool[mask]).sum().item()
    fp = (preds[mask] & ~target_bool[mask]).sum().item()
    fn = (~preds[mask] & target_bool[mask]).sum().item()
    return loss_val, tp, fp, fn


_ESTIMATED_NODES: dict[str, float] | None = None


def _estimated_nodes() -> dict[str, float]:
    """sample_name -> estimated_number_of_nodes, from the dataset metadata."""
    global _ESTIMATED_NODES
    if _ESTIMATED_NODES is None:
        import pandas as pd

        path = Path(TRAIN_DIR).parent / "metadata.csv"
        df = pd.read_csv(path)
        _ESTIMATED_NODES = dict(
            zip(df["sample_name"], df["estimated_number_of_nodes"].astype(float))
        )
    return _ESTIMATED_NODES


def _sample_name(window_name: str) -> str:
    """Video name from a window key like ``44b6_0113de3b.zarr_t7``."""
    return window_name.split("_t")[0].removesuffix(".zarr")


def _adjustment_factor(frame_nodes: dict, n_frames: dict) -> float:
    """Mean of 1 - 0.1 * (T_pred - T_true) / T_true over the validated samples.

    Validation windows overlap and cover only part of a video — 50 windows of
    52 distinct frames out of 100 on this data — so summing detections over
    windows would both double-count frames and miss the rest. ``T_pred`` is
    instead the mean count over the *distinct* frames seen, scaled to the
    video's full length.
    """
    estimated = _estimated_nodes()
    per_sample: dict[str, list[int]] = {}
    for (sample, _t), n_det in frame_nodes.items():
        per_sample.setdefault(sample, []).append(n_det)

    factors = []
    for sample, counts in per_sample.items():
        t_true = estimated.get(sample)
        if not t_true:
            continue
        t_pred = sum(counts) / len(counts) * n_frames[sample]
        factors.append(1.0 - 0.1 * (t_pred - t_true) / t_true)
    return sum(factors) / len(factors) if factors else 0.0


def _gather_dicts(local: dict, world_size: int) -> dict:
    """Merge one dict per rank; later ranks win on a key, which is fine here
    because a (sample, frame) count is the same whichever rank produced it."""
    if world_size == 1 or not (dist.is_available() and dist.is_initialized()):
        return local
    parts: list[dict | None] = [None] * world_size
    dist.all_gather_object(parts, local)
    merged: dict = {}
    for part in parts:
        merged.update(part or {})
    return merged


def _all_reduce_sums(values: list[float], device: torch.device) -> list[float]:
    if not (dist.is_available() and dist.is_initialized()):
        return values
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    args,
    device: torch.device | None = None,
    eval_edges: bool = True,
    transformer_only: bool = False,
    fts_cache: dict[str, list] = None,
) -> tuple[float, dict, dict, dict, dict]:
    """Evaluate on a sharded validation loader across all ranks.

    Returns ``(avg_loss, edge_jaccard, node_recall, adjustment, edge_recall)``
    where every metric but the loss is keyed by detection threshold; the loss is
    reported at the configured one. ``adjustment`` is the competition's
    node-count penalty, 1 - 0.1 * (T_pred - T_true) / T_true, averaged over
    samples.

    Detection runs once, at ``args.detect_and_match_threshold``; every higher
    threshold is a subset of those peaks, so the sweep filters on each node's
    stored score instead of re-detecting. (Not bit-identical to re-running:
    greedy NMS on a smaller candidate set can keep a peak that a
    just-above-threshold neighbour would otherwise have suppressed.)
    """
    if device is None:
        device = next(model.parameters()).device

    if loader is None:
        return 0., {}, {}, {}, {}

    base = args.detect_and_match_threshold
    thresholds = sorted(
        {base} | {t for t in getattr(args, "eval_thresholds", ()) or () if t >= base}
    )

    model.eval()
    gt_total = 0
    gt_matched = {t: 0 for t in thresholds}
    total_loss = {t: 0.0 for t in thresholds}
    tp = {t: 0 for t in thresholds}
    fp = {t: 0 for t in thresholds}
    fn = {t: 0 for t in thresholds}
    n_pairs = {t: 0 for t in thresholds}
    # (sample, frame) -> detections, deduplicated: validation windows overlap.
    frame_nodes: dict[float, dict[tuple[str, int], int]] = {t: {} for t in thresholds}
    n_frames: dict[str, int] = {}

    for batch in loader:
        imgs = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        image_shape = tuple(batch["image_shape"][0].tolist())
        voxel_size = tuple(batch["voxel_size"][0].tolist())

        B, W, d, h, w = imgs.shape

        if transformer_only and fts_cache is not None:
            names = batch["name"]
            if any(name not in fts_cache for name in names):
                fresh = encode_nodes(
                    model, imgs, coords, masks, image_shape, voxel_size, args, nms=args.nms, nms_radius_um=args.nms_radius_um,
                )[0]
                for b, name in enumerate(names):
                    fts_cache[name] = split_frame_det(fresh, b)
            frame_det = collate_frame_det([fts_cache[n] for n in names])
        else:
            frame_det = encode_nodes(
                model, imgs, coords, masks, image_shape, voxel_size, args, nms=args.nms, nms_radius_um=args.nms_radius_um,
            )[0]

        names = [_sample_name(n) for n in batch["name"]]
        t_starts = batch["t_start"].tolist()
        video_len = batch["image_shape"][:, 0].tolist()
        for i in range(W):
            matches, det_mask, scores = frame_det[i][3], frame_det[i][2], frame_det[i][5]
            for b in range(B):
                gt_total += int(masks[b, i].sum().item())
                n_frames[names[b]] = int(video_len[b])
                for t in thresholds:
                    # Subset of the peaks already detected, by their own score.
                    keep = det_mask[b] & (scores[b] > t)
                    gt_matched[t] += int((matches[b][keep[: len(matches[b])]] >= 0).sum())
                    frame_nodes[t][(names[b], t_starts[b] + i)] = int(keep.sum())

        if not eval_edges:
            continue

        # One edge forward per threshold
        for t in thresholds:
            fd = frame_det if t == base else filter_frame_det(frame_det, t)
            for i in range(W - 1):
                pair_target = build_matched_edge_targets(
                    fd[i][3], fd[i + 1][3], targets[:, i],
                    fd[i][0].shape[1], fd[i + 1][0].shape[1],
                )
                pair_logits = model.predict_edges(
                    fd[i][4], fd[i + 1][4],
                    fd[i][0], fd[i + 1][0],
                    fd[i][1], fd[i + 1][1],
                    fd[i][2], fd[i + 1][2],
                )
                for b in range(B):
                    ns_b = int(fd[i][2][b].sum().item())
                    nt_b = int(fd[i + 1][2][b].sum().item())
                    pair_loss, pair_tp, pair_fp, pair_fn = _evaluate_pair(
                        pair_logits[b, :ns_b, :nt_b],
                        pair_target[b, :ns_b, :nt_b],
                    )
                    total_loss[t] += pair_loss
                    tp[t] += pair_tp
                    fp[t] += pair_fp
                    fn[t] += pair_fn
                    n_pairs[t] += 1

    # Aggregate: everything per threshold, in one reduction.
    n = len(thresholds)
    flat = _all_reduce_sums(
        [total_loss[t] for t in thresholds]
        + [tp[t] for t in thresholds]
        + [fp[t] for t in thresholds]
        + [fn[t] for t in thresholds]
        + [n_pairs[t] for t in thresholds]
        + [gt_matched[t] for t in thresholds]
        + [gt_total],
        device,
    )
    loss_a, tp_a, fp_a, fn_a, pairs_a, matched_a = (
        flat[i * n : (i + 1) * n] for i in range(6)
    )
    gt_total_all = flat[-1]

    edge_jaccard = {
        t: tp_a[i] / max(tp_a[i] + fp_a[i] + fn_a[i], 1.0) for i, t in enumerate(thresholds)
    }
    edge_recall = {
        t: tp_a[i] / max(tp_a[i] + fn_a[i], 1.0) for i, t in enumerate(thresholds)
    }
    node_recall = {
        t: matched_a[i] / max(gt_total_all, 1.0) for i, t in enumerate(thresholds)
    }
    avg_loss = loss_a[thresholds.index(base)] / max(pairs_a[thresholds.index(base)], 1.0)

    world = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    n_frames_all = _gather_dicts(n_frames, world)
    adjustment = {
        t: _adjustment_factor(_gather_dicts(frame_nodes[t], world), n_frames_all)
        for t in thresholds
    }
    return avg_loss, edge_jaccard, node_recall, adjustment, edge_recall
