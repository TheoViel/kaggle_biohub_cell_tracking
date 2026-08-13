"""Loss functions for cell tracking.

Two terms:
  L = L_edge + det_loss_weight · L_det

L_det: per-voxel weighted BCE.
  - GT node voxels:    weight = 1          (target = 1)
  - Weak positives:    weight = weak_weight (target = 1)
  - Hard negatives:    weight = neg_weight  (target = 0)
  Weak positives are image local maxima brighter than half the mean GT intensity.
L_edge: focal-BCE on softmax edge probabilities, masked to annotated rows/cols.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _find_weak_positives(
    imgs: torch.Tensor,         # (B, Z_full, Y_full, X_full) normalised image
    coords: torch.Tensor,       # (B, max_nodes, 3) GT coords in output space
    mask: torch.Tensor,         # (B, max_nodes) bool
    spatial: tuple,             # (Z_out, Y_out, X_out) = det_logits spatial dims
    unet_scaling: tuple,        # (sz, sy, sx)
    nms_kernel: int = 3,
) -> torch.Tensor:              # (B, Z_out, Y_out, X_out) bool
    """Return a boolean mask of local-max peaks brighter than half the mean GT intensity.

    Steps:
      1. Downsample the raw image to match the detection output resolution.
      2. Compute the mean normalised intensity at GT cell locations.
      3. Return voxels that are local maxima AND exceed half that intensity.
    """
    B = imgs.shape[0]
    device = imgs.device
    sz, sy, sx = unet_scaling
    nt = mask.sum(dim=1).long()

    imgs_4d = imgs.detach().float().unsqueeze(1)   # (B, 1, Z_full, Y_full, X_full)
    if sz > 1 or sy > 1 or sx > 1:
        imgs_ds = F.avg_pool3d(imgs_4d, kernel_size=(sz, sy, sx), stride=(sz, sy, sx))
    else:
        imgs_ds = imgs_4d
    # imgs_ds: (B, 1, Z_out, Y_out, X_out)

    k = nms_kernel if nms_kernel % 2 == 1 else nms_kernel + 1
    pooled = F.max_pool3d(imgs_ds, k, stride=1, padding=k // 2)
    is_local_max = (imgs_ds == pooled).squeeze(1)  # (B, Z_out, Y_out, X_out)
    imgs_vol = imgs_ds.squeeze(1)                  # (B, Z_out, Y_out, X_out)

    weak = torch.zeros(B, *spatial, dtype=torch.uint8, device=device)
    for b in range(B):
        n = int(nt[b].item())
        if n == 0:
            continue
        zi = coords[b, :n, 0].long().clamp(0, spatial[0] - 1)
        yi = coords[b, :n, 1].long().clamp(0, spatial[1] - 1)
        xi = coords[b, :n, 2].long().clamp(0, spatial[2] - 1)
        threshold = imgs_vol[b, zi, yi, xi].mean() * 0.5

        # Dilate GT positions by the NMS radius to exclude their neighbourhood.
        gt_occ = torch.zeros(1, 1, *spatial, device=device)
        gt_occ[0, 0, zi, yi, xi] = 1.0
        near_gt = F.max_pool3d(gt_occ, k, stride=1, padding=k // 2)[0, 0] > 0

        weak[b] = (is_local_max[b] & (imgs_vol[b] > threshold) & ~near_gt).to(torch.uint8)

    return weak


def detection_loss(
    det_logits: torch.Tensor,       # (B, 1, Z, Y, X) raw logits
    coords: torch.Tensor,           # (B, max_nodes, 3) GT coords in output space
    mask: torch.Tensor,             # (B, max_nodes) bool
    neg_weight: float = 0.01,
    imgs: torch.Tensor | None = None,        # (B, Z_full, Y_full, X_full) full-scale
    unet_scaling: tuple | None = None,
    weak_weight: float = 0.3,
    weak_smoothing: float = 0.1,
) -> torch.Tensor:
    """Weighted BCE with three voxel categories.

    GT positives:    weight = 1
    Weak positives:  weight = weak_weight
    Negatives:       weight = neg_weight
    """
    B = det_logits.shape[0]
    spatial = det_logits.shape[2:]
    logits = det_logits[:, 0]       # (B, Z, Y, X)

    # --- GT target ---
    is_gt = torch.zeros_like(logits, dtype=torch.uint8)
    nt = mask.sum(dim=1).long()

    if unet_scaling is not None:  # adapt targets to output space
        coords[..., 0] //= unet_scaling[0]
        coords[..., 1] //= unet_scaling[1]
        coords[..., 2] //= unet_scaling[2]

    for b in range(B):
        n = int(nt[b].item())
        if n == 0:
            continue
        zi = coords[b, :n, 0].long().clamp(0, spatial[0] - 1)
        yi = coords[b, :n, 1].long().clamp(0, spatial[1] - 1)
        xi = coords[b, :n, 2].long().clamp(0, spatial[2] - 1)
        is_gt[b, zi, yi, xi] = 1

    # --- Weak positives ---
    if imgs is not None and unet_scaling is not None:
        weak_pos = _find_weak_positives(imgs, coords, mask, spatial, unet_scaling)
    else:
        weak_pos = torch.zeros_like(is_gt)

    # weak_smoothing for background, (1 - weak smoothing) for soft gt, 1 for GT
    target = torch.clamp(weak_pos * (1 - weak_smoothing) + is_gt, weak_smoothing, 1).float()

    # neg_weight background, weak_weight for soft gt, 1 for GT
    weight = torch.clamp(
        weak_pos * weak_weight + (1 - weak_pos) * neg_weight + is_gt, 0, 1
    )

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(15, 5))
    # frame = int(coords[0, 0, 0].item())
    # plt.subplot(1, 3, 1)
    # target[0, :, 0, 0] = 0
    # plt.imshow(target[0, frame].cpu().numpy())
    # plt.axis("off")
    # plt.title("Target")
    # plt.colorbar()
    # plt.subplot(1, 3, 2)
    # weight[0, :, 0, 0] = 0
    # plt.imshow(weight[0, frame].cpu().numpy())
    # plt.axis("off")
    # plt.title("Weight")
    # plt.colorbar()
    # plt.subplot(1, 3, 3)
    # plt.imshow(imgs[0, frame // unet_scaling[0]].cpu().numpy(), cmap="gray")
    # plt.axis("off")
    # plt.title("Image")
    # plt.show()

    # print(logits.max(), logits.min())
    # print(target.max(), target.min())
    # print(weight.max(), weight.min())

    # fp16 logits can overflow to inf under autocast; the BCE-with-logits
    # formula then subtracts two infinities (max(x,0) - x*target) and yields
    # NaN, so score in fp32 like edge_loss does.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        bce = F.binary_cross_entropy_with_logits(
            logits.float(), target.float(), weight=weight.float(), reduction="none",
        )

    # print(bce.max(), bce.min())
    # print(bce.size())

    return bce.mean()


def edge_loss(
    logits: torch.Tensor,   # (N_t, N_t1) edge logits (unbatched)
    target: torch.Tensor,   # (N_t, N_t1) GT transition matrix
) -> torch.Tensor:
    """Focal-BCE on softmax probs, restricted to annotated rows/columns.

    Softmax is over dim=0 (parent axis): permits 1→2 divisions, prevents
    many-parents merges.
    """
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    active = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
    if not active.any():
        # Stay attached to the graph: a fresh leaf tensor would leave the edge
        # scorer without gradients, which DDP rejects as an unused parameter.
        return logits.sum() * 0.0

    # binary_cross_entropy is unsafe under autocast, and the softmax here rules
    # out the fused *_with_logits variant, so score the pair in fp32.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float()
        target = target.float()
        probs = torch.softmax(logits, dim=0)
        bce = F.binary_cross_entropy(probs, target, reduction="none")
        p_t = probs * target + (1 - probs) * (1 - target)
        loss = ((1 - p_t) ** 2) * bce
        return loss[active].mean()


def batch_edge_loss(
    logits: torch.Tensor,   # (B, N_t, N_t1)
    target: torch.Tensor,   # (B, N_t, N_t1)
    mask_t: torch.Tensor,   # (B, N_t) bool
    mask_t1: torch.Tensor,  # (B, N_t1) bool
) -> torch.Tensor:
    """Compute edge loss over a batch by slicing out real (unpadded) regions."""
    B = logits.shape[0]
    losses = []
    for b in range(B):
        nt = int(mask_t[b].sum().item())
        nt1 = int(mask_t1[b].sum().item())
        losses.append(edge_loss(logits[b, :nt, :nt1], target[b, :nt, :nt1]))
    return torch.stack(losses).mean()


class TrackingLoss(nn.Module):
    """Combined detection + edge loss.

        L = L_edge + det_loss_weight · L_det

    Parameters
    ----------
    det_loss_weight : float
        Weight for L_det relative to L_edge.
    det_neg_weight : float
        Per-voxel weight for hard-negative voxels in the detection loss.
    det_weak_weight : float
        Per-voxel weight for weak-positive voxels (image local maxima).
    det_weak_smoothing : float
        Label smoothing applied to weak positives and background.
    """

    def __init__(
        self,
        det_loss_weight: float = 1.0,
        det_neg_weight: float = 0.01,
        det_weak_weight: float = 0.3,
        det_weak_smoothing: float = 0.1,
    ):
        super().__init__()
        self.det_loss_weight = det_loss_weight
        self.det_neg_weight = det_neg_weight
        self.det_weak_weight = det_weak_weight
        self.det_weak_smoothing = det_weak_smoothing

    def forward(
        self,
        det_logits_per_frame: list[torch.Tensor],   # W × (B, 1, Z, Y, X)
        gt_coords_per_frame: list[torch.Tensor],    # W × (B, max_nodes, 3)
        gt_mask_per_frame: list[torch.Tensor],      # W × (B, max_nodes) bool
        edge_logits_per_pair: list[torch.Tensor],   # (W-1) × (B, N_t, N_t1)
        edge_targets_per_pair: list[torch.Tensor],  # (W-1) × (B, N_t, N_t1)
        det_mask_per_pair: list[tuple],             # (W-1) × (mask_src, mask_tgt)
        imgs_per_frame: list[torch.Tensor] | None = None,   # W × (B, Z_full, Y_full, X_full)
        unet_scaling: tuple | None = None,
        progress: float = 1.0,   # fraction of training elapsed, in [0, 1]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        W = len(det_logits_per_frame)

        det_losses = [
            detection_loss(
                det_logits_per_frame[i],
                gt_coords_per_frame[i].clone(),
                gt_mask_per_frame[i],
                self.det_neg_weight,
                imgs=imgs_per_frame[i] if imgs_per_frame is not None else None,
                unet_scaling=unet_scaling,
                weak_weight=self.det_weak_weight,
                weak_smoothing=self.det_weak_smoothing,
            )
            for i in range(W)
        ]
        l_det = sum(det_losses) / W

        if edge_logits_per_pair:
            edge_losses = [
                batch_edge_loss(
                    edge_logits_per_pair[i],
                    edge_targets_per_pair[i],
                    det_mask_per_pair[i][0],
                    det_mask_per_pair[i][1],
                )
                for i in range(len(edge_logits_per_pair))
            ]
            l_edge = sum(edge_losses) / len(edge_losses)
        else:
            l_edge = torch.zeros((), device=det_logits_per_frame[0].device)

        total = l_edge + self.det_loss_weight * l_det

        breakdown = {
            "det":   l_det.item(),
            "edge":  l_edge.item(),
            "total": total.item(),
        }
        return total, breakdown
