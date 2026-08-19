"""Loss functions for cell tracking.

Two terms:
  L = L_edge + det_loss_weight · L_det

L_det: penalty-reduced focal loss (CenterNet) on a Gaussian heatmap.
  - GT peaks:          positive, weight 1
  - Weak-positive peaks: positive, weight = weak_weight
  - Everything else:   negative, scaled by neg_weight and discounted near peaks
  Weak positives are image local maxima brighter than half the mean GT intensity.
L_edge: focal-BCE on softmax edge probabilities, masked to annotated rows/cols.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Focal exponents, CenterNet's defaults: alpha sharpens the easy/hard split on
# the prediction, beta discounts negatives by how close they sit to a peak.
FOCAL_ALPHA = 2.0
FOCAL_BETA = 4.0


@torch.no_grad()
def gaussian_heatmap(
    coords: torch.Tensor,  # (B, max_nodes, 3) in detection-grid space
    mask: torch.Tensor,  # (B, max_nodes) bool
    spatial: tuple,  # (Z, Y, X)
    sigma: float,
) -> torch.Tensor:  # (B, Z, Y, X) float, peak 1.0 at each GT point
    """Splat an isotropic Gaussian at every GT point (CenterNet-style targets).

    Overlapping splats combine with max, so each GT point keeps a peak of
    exactly 1.0. ``sigma <= 0`` degenerates to single-voxel targets.

    Sigma is expressed in *detection-grid voxels* (the grid det_logits lives
    on), not microns.
    """
    B = coords.shape[0]
    device = coords.device
    Z, Y, X = spatial
    n_vox = B * Z * Y * X

    # One extra slot absorbs out-of-bounds / padded writes, then is dropped.
    heat = torch.zeros(n_vox + 1, device=device)

    r = max(1, int(math.ceil(3.0 * sigma))) if sigma > 0 else 0
    off = torch.arange(-r, r + 1, device=device)
    oz, oy, ox = torch.meshgrid(off, off, off, indexing="ij")
    oz, oy, ox = oz.reshape(-1), oy.reshape(-1), ox.reshape(-1)  # (K,)
    sq = (oz * oz + oy * oy + ox * ox).float()
    if sigma > 0:
        g = torch.exp(-sq / (2.0 * sigma * sigma))
    else:
        g = (sq == 0).float()

    zc = coords[..., 0].round().long().unsqueeze(-1) + oz  # (B, M, K)
    yc = coords[..., 1].round().long().unsqueeze(-1) + oy
    xc = coords[..., 2].round().long().unsqueeze(-1) + ox

    valid = (
        mask.unsqueeze(-1)
        & (zc >= 0)
        & (zc < Z)
        & (yc >= 0)
        & (yc < Y)
        & (xc >= 0)
        & (xc < X)
    )
    bidx = torch.arange(B, device=device).view(B, 1, 1).expand_as(zc)
    flat = ((bidx * Z + zc.clamp(0, Z - 1)) * Y + yc.clamp(0, Y - 1)) * X + xc.clamp(
        0, X - 1
    )
    flat = torch.where(valid, flat, torch.full_like(flat, n_vox))

    vals = g.view(1, 1, -1).expand_as(flat).reshape(-1)
    heat.scatter_reduce_(0, flat.reshape(-1), vals, reduce="amax")
    return heat[:n_vox].view(B, Z, Y, X)


def detection_loss(
    det_logits: torch.Tensor,  # (B, 1, Z, Y, X) raw logits
    coords: torch.Tensor,  # (B, max_nodes, 3) GT coords in output space
    mask: torch.Tensor,  # (B, max_nodes) bool
    neg_weight: float = 1e-3,
    unet_scaling: tuple | None = None,
    weak_weight: float = 0.3,
    weak_smoothing: float = 0.1,
    gaussian_sigma: float = 1.0,
    weak_positives: torch.Tensor | None = None,  # (B, N, 3), -1 padded
    use_weak_positives: bool = True,
) -> torch.Tensor:
    """Penalty-reduced focal loss on a Gaussian-splatted detection heatmap.

    Target
    ------
    GT points        1.0 at the peak, decaying as exp(-d^2 / 2 sigma^2)
    Weak positives   1 - weak_smoothing at the peak, same decay
    Background       0

    Positive term    at peak voxels only: 1 for GT, ``weak_weight`` for weak
    Negative term    every other voxel, scaled by ``neg_weight`` and discounted
                     by (1 - target)^beta so a peak's own skirt is cheap

    Normalised by the number of peaks, so the scale is independent of how many
    cells the crop holds.

    ``gaussian_sigma`` is in detection-grid voxels. It must be > 0 for the loss
    to shape a *peak*: at sigma = 0 the target is a lone voxel with no skirt,
    which says "this voxel is a cell" but nothing about being a local maximum,
    and the detector is free to answer with a blob.
    """
    spatial = det_logits.shape[2:]
    logits = det_logits[:, 0]  # (B, Z, Y, X)
    batch_size = logits.shape[0]

    if unet_scaling is not None:  # adapt targets to output space
        scale = torch.tensor(unet_scaling, dtype=coords.dtype, device=coords.device)
        coords = torch.div(coords, scale, rounding_mode="floor")

    # Peak 1.0 at each GT point, decaying across the Gaussian skirt.
    gt_heat = gaussian_heatmap(coords, mask, spatial, gaussian_sigma)
    heat = gt_heat
    # Weight of the *positive* term. Only a peak voxel is a positive; its own
    # skirt is a discounted negative (see below).
    pos_weight = (gt_heat >= 1.0).float()

    if use_weak_positives:
        assert weak_positives is not None, "use_weak_positives set with no weak_positives"
        valid = (weak_positives >= 0).all(dim=-1)
        wp = weak_positives
        if unet_scaling is not None:
            wp = torch.div(wp, scale, rounding_mode="floor")

        weak_heat = gaussian_heatmap(wp, valid, spatial, gaussian_sigma)
        # Weak peaks are cells too, so they must not be driven to 0 as
        # background; smoothing keeps their target just under a GT peak.
        heat = torch.maximum(gt_heat, weak_heat * (1.0 - weak_smoothing))
        weak_peak = (weak_heat >= 1.0) & (gt_heat < 1.0)
        pos_weight = pos_weight + weak_weight * weak_peak.float()

    # Penalty-reduced focal loss (CenterNet). Two properties matter here:
    #
    #  * The negative term runs over every non-peak voxel with no 1/n_neg
    #    rescaling. Normalising each class to unit mass — what a plain weighted
    #    BCE does — leaves the background holding `neg_weight` of total mass
    #    against 1.0 for the peaks. At neg_weight=1e-2 that makes predicting
    #    "cell" everywhere nearly free, and the detector saturates into a blob
    #    with no local maxima left to detect.
    #  * (1 - heat)^beta discounts voxels on a peak's skirt, so a near miss is
    #    cheap while the far background is not. That is what shapes an actual
    #    maximum at the cell instead of a plateau over it.
    pos = pos_weight > 0
    with torch.autocast(device_type=logits.device.type, enabled=False):
        p = torch.sigmoid(logits.float()).clamp(1e-4, 1.0 - 1e-4)
        heat = heat.float()

        pos_loss = -((1.0 - p) ** FOCAL_ALPHA) * torch.log(p) * pos_weight
        neg_loss = (
            -((1.0 - heat) ** FOCAL_BETA)
            * (p ** FOCAL_ALPHA)
            * torch.log(1.0 - p)
            * neg_weight
        )

        # Normalise per peak, so the loss scale does not depend on how many
        # cells a crop happens to contain.
        n_pos = pos.reshape(batch_size, -1).sum(dim=1).clamp(min=1).float()
        per_sample = (
            torch.where(pos, pos_loss, neg_loss).reshape(batch_size, -1).sum(dim=1)
        )
        return (per_sample / n_pos).mean()


def edge_loss(
    logits: torch.Tensor,  # (N_t, N_t1) edge logits (unbatched)
    target: torch.Tensor,  # (N_t, N_t1) GT transition matrix
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
        # kNN gating fills non-candidates with -inf, and -inf * 0 is NaN, so
        # zero those out before summing.
        finite = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
        return finite.sum() * 0.0

    # binary_cross_entropy is unsafe under autocast, and the softmax here rules
    # out the fused *_with_logits variant, so score the pair in fp32.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float()
        target = target.float()
        # kNN-gated entries come out of the softmax at exactly 0. That is fine
        # for target=0 (zero loss), but a gated-out true parent would hit
        # log(0) and pick up BCE's -100 clamp, so keep probs off the boundary.
        probs = torch.softmax(logits, dim=0).clamp(1e-6, 1.0 - 1e-6)
        bce = F.binary_cross_entropy(probs, target, reduction="none")
        p_t = probs * target + (1 - probs) * (1 - target)
        loss = ((1 - p_t) ** 2) * bce
        return loss[active].mean()


def batch_edge_loss(
    logits: torch.Tensor,  # (B, N_t, N_t1)
    target: torch.Tensor,  # (B, N_t, N_t1)
    mask_t: torch.Tensor,  # (B, N_t) bool
    mask_t1: torch.Tensor,  # (B, N_t1) bool
) -> torch.Tensor:
    """``edge_loss`` over a whole batch, masking padding instead of slicing it.

    Same value as looping ``edge_loss`` over the unpadded blocks, but without
    the per-sample ``.item()`` calls — each of those is a GPU sync, and there
    are 2B of them per pair.

    Padded parents are pushed to -inf before the softmax so they cannot take
    probability mass from the real ones, which is what slicing did implicitly.
    """
    B = logits.shape[0]
    real = mask_t.unsqueeze(2) & mask_t1.unsqueeze(1)          # (B, N_t, N_t1)
    target = target * real                                      # padding links nothing

    active_rows = target.sum(dim=2) > 0
    active_cols = target.sum(dim=1) > 0
    active = (active_rows.unsqueeze(2) | active_cols.unsqueeze(1)) & real

    # binary_cross_entropy is unsafe under autocast, and the softmax here rules
    # out the fused *_with_logits variant, so score the pairs in fp32.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float().masked_fill(~mask_t.unsqueeze(2), float("-inf"))
        target = target.float()
        # A child whose parents are all masked leaves an all -inf column, whose
        # softmax is NaN; it is never active, but NaN * 0 would still spread.
        probs = torch.nan_to_num(torch.softmax(logits, dim=1))
        # kNN-gated entries come out of the softmax at exactly 0. That is fine
        # for target=0 (zero loss), but a gated-out true parent would hit
        # log(0) and pick up BCE's -100 clamp, so keep probs off the boundary.
        probs = probs.clamp(1e-6, 1.0 - 1e-6)
        bce = F.binary_cross_entropy(probs, target, reduction="none")
        p_t = probs * target + (1 - probs) * (1 - target)
        loss = ((1 - p_t) ** 2) * bce

        n_active = active.reshape(B, -1).sum(dim=1)
        per_sample = (loss * active).reshape(B, -1).sum(dim=1) / n_active.clamp(min=1)
        # A sample with no annotated link contributes 0, as the loop did — and
        # the sum keeps the edge scorer attached to the graph for DDP.
        return per_sample.sum() / B


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
        Label smoothing applied to weak positives.
    det_gaussian_sigma : float
        Std-dev, in detection-grid voxels, of the Gaussian splatted at each
        GT point. 0 falls back to single-voxel targets.
    """

    def __init__(
        self,
        det_loss_weight: float = 1.0,
        det_neg_weight: float = 1e-3,
        det_weak_weight: float = 0.3,
        det_weak_smoothing: float = 0.1,
        det_gaussian_sigma: float = 1.0,
        det_use_weak_positives: bool = True,
    ):
        super().__init__()
        self.det_loss_weight = det_loss_weight
        self.det_neg_weight = det_neg_weight
        self.det_weak_weight = det_weak_weight
        self.det_weak_smoothing = det_weak_smoothing
        self.det_gaussian_sigma = det_gaussian_sigma
        self.det_use_weak_positives = det_use_weak_positives

    def forward(
        self,
        det_logits_per_frame: list[torch.Tensor],  # W × (B, 1, Z, Y, X)
        gt_coords_per_frame: list[torch.Tensor],  # W × (B, max_nodes, 3)
        gt_mask_per_frame: list[torch.Tensor],  # W × (B, max_nodes) bool
        edge_logits_per_pair: list[torch.Tensor],  # (W-1) × (B, N_t, N_t1)
        edge_targets_per_pair: list[torch.Tensor],  # (W-1) × (B, N_t, N_t1)
        det_mask_per_pair: list[tuple],  # (W-1) × (mask_src, mask_tgt)
        weak_pos_per_frame: (
            list[torch.Tensor] | None
        ) = None,  # W × (B, N, 3), -1 padded
        unet_scaling: tuple | None = None,
        progress: float = 1.0,  # fraction of training elapsed, in [0, 1]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        W = len(det_logits_per_frame)

        # An empty det_logits_per_frame means the detector is frozen
        # (transformer-only training): nothing to supervise on the voxel grid.
        if W:
            det_losses = [
                detection_loss(
                    det_logits_per_frame[i],
                    gt_coords_per_frame[i].clone(),
                    gt_mask_per_frame[i],
                    self.det_neg_weight,
                    unet_scaling=unet_scaling,
                    weak_weight=self.det_weak_weight,
                    weak_smoothing=self.det_weak_smoothing,
                    gaussian_sigma=self.det_gaussian_sigma,
                    weak_positives=(
                        weak_pos_per_frame[i] if weak_pos_per_frame is not None else None
                    ),
                    use_weak_positives=self.det_use_weak_positives,
                )
                for i in range(W)
            ]
            l_det = sum(det_losses) / W
        else:
            l_det = torch.zeros((), device=edge_logits_per_pair[0].device)

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
            "det": self.det_loss_weight * l_det.item(),
            "edge": l_edge.item(),
            "total": total.item(),
        }
        return total, breakdown
