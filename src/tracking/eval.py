"""Evaluation utilities for the cell tracking model."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from tracking.targets import build_matched_edge_targets
from tracking.fit_utils import collate_frame_det, encode_nodes, split_frame_det


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
) -> tuple[float, float, float]:
    """Evaluate on a sharded validation loader across all ranks.

    Returns (avg_loss, edge_jaccard, node_recall), reduced over the process group.
    """
    if device is None:
        device = next(model.parameters()).device

    if loader is None:
        return 0., 0., 0.

    model.eval()
    gt_total, gt_matched = 0, 0
    total_loss, tp, fp, fn, n_pairs = 0.0, 0, 0, 0, 0

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

        for i in range(W):
            matches = frame_det[i][3]
            for b in range(B):
                n_gt = int(masks[b, i].sum().item())
                n_matched = int((matches[b] >= 0).sum().item())
                gt_total += n_gt
                gt_matched += n_matched

        for i in range(W - 1):
            if not eval_edges:
                continue
            ns = frame_det[i][0].shape[1]
            nt = frame_det[i + 1][0].shape[1]
            pair_target = build_matched_edge_targets(
                frame_det[i][3],
                frame_det[i + 1][3],
                targets[:, i],
                ns,
                nt,
            )
            pair_logits = model.predict_edges(
                frame_det[i][4],
                frame_det[i + 1][4],
                frame_det[i][0],
                frame_det[i + 1][0],
                frame_det[i][1],
                frame_det[i + 1][1],
                frame_det[i][2],
                frame_det[i + 1][2],
            )

            for b in range(B):
                ns_b = int(frame_det[i][2][b].sum().item())
                nt_b = int(frame_det[i + 1][2][b].sum().item())
                pair_loss, pair_tp, pair_fp, pair_fn = _evaluate_pair(
                    pair_logits[b, :ns_b, :nt_b],
                    pair_target[b, :ns_b, :nt_b],
                )
                total_loss += pair_loss
                tp += pair_tp
                fp += pair_fp
                fn += pair_fn
                n_pairs += 1

    # Aggregate
    total_loss, tp, fp, fn, n_pairs, gt_matched, gt_total = _all_reduce_sums(
        [total_loss, tp, fp, fn, n_pairs, gt_matched, gt_total],
        device,
    )
    edge_jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    node_recall = gt_matched / max(gt_total, 1.0)
    avg_loss = total_loss / max(n_pairs, 1.0)
    return avg_loss, edge_jaccard, node_recall
