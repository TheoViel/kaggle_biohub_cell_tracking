"""Evaluation utilities for the cell tracking model."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from tracking.targets import detect_and_match, build_matched_edge_targets


def _evaluate_pair(
    logits: torch.Tensor,  # (N_t, N_t1)
    target: torch.Tensor,  # (N_t, N_t1)
) -> tuple[float, int, int]:
    """Returns (focal_bce_loss, n_correct, n_total) for one unpadded pair."""
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    if not active_rows.any():
        return 0.0, 0, 0

    from tracking.losses import edge_loss

    loss_val = edge_loss(logits, target).item()

    probs = torch.softmax(logits, dim=0)
    preds = (probs > 0.5).float()
    mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
    correct = int((preds[mask] == target[mask]).sum().item())
    total = int(mask.sum().item())
    return loss_val, correct, total


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
) -> tuple[float, float, float]:
    """Evaluate on a sharded validation loader across all ranks.

    Returns (avg_loss, edge_accuracy, node_recall), reduced over the process group.
    """
    if device is None:
        device = next(model.parameters()).device

    if loader is None:
        return 0.0, 0.0, 0.0

    model.eval()
    total_loss, correct, total, n_pairs = 0.0, 0, 0, 0
    gt_matched, gt_total = 0, 0

    for batch in loader:
        imgs = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        image_shape = tuple(batch["image_shape"][0].tolist())
        voxel_size = tuple(batch["voxel_size"][0].tolist())

        B, W, d, h, w = imgs.shape
        unet_out, det_logits = model.encode(imgs)

        d_out, h_out, w_out = unet_out.shape[-3:]
        unet_scaling = (d // d_out, h // h_out, w // w_out)

        frame_det = []
        for i in range(W):
            det_c, det_p, det_m, matches = detect_and_match(
                det_logits[i].sigmoid(),
                coords[:, i],
                masks[:, i],
                image_shape,
                voxel_size=voxel_size,
                unet_scaling=unet_scaling,
                pool_kernel_um=args.pool_kernel_um,
                frame_index=i,
                window_size=W,
                det_threshold=args.detect_and_match_threshold,
                max_dets_per_sample=args.max_dets_per_sample,
            )
            unet_feat = model._index_features(unet_out[:, i], det_c, det_m)
            frame_det.append((det_c, det_p, det_m, matches, unet_feat))

            for b in range(B):
                n_gt = int(masks[b, i].sum().item())
                n_matched = int((matches[b] >= 0).sum().item())
                gt_total += n_gt
                gt_matched += n_matched

        for i in range(W - 1):
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
                pair_loss, pair_correct, pair_total = _evaluate_pair(
                    pair_logits[b, :ns_b, :nt_b],
                    pair_target[b, :ns_b, :nt_b],
                )
                total_loss += pair_loss
                correct += pair_correct
                total += pair_total
                n_pairs += 1

    total_loss, correct, total, n_pairs, gt_matched, gt_total = _all_reduce_sums(
        [total_loss, correct, total, n_pairs, gt_matched, gt_total],
        device,
    )

    node_recall = gt_matched / max(gt_total, 1.0)
    edge_acc = correct / max(total, 1.0)
    avg_loss = total_loss / max(n_pairs, 1.0)
    return avg_loss, edge_acc, node_recall
