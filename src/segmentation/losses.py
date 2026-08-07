import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def dice_loss(pred_prob, target, smooth=1.0):
    """Soft Dice loss; expects probabilities in [0, 1]."""
    flat_p = pred_prob.reshape(-1)
    flat_t = target.reshape(-1)
    intersection = (flat_p * flat_t).sum()
    return 1.0 - (2.0 * intersection + smooth) / (flat_p.sum() + flat_t.sum() + smooth)


def focal_loss(pred_logits, target, alpha=0.25, gamma=2.0):
    """
    Focal loss for dense heatmap targets in [0, 1].
    Uses the standard sigmoid focal form — soft targets naturally
    down-weight easy negatives (background far from any centroid).
    """
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
    pt  = torch.exp(-bce)                           # ≈ confidence of correct class
    return (alpha * (1.0 - pt) ** gamma * bce).mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class CellSegLoss(nn.Module):
    """
    Three-term loss matching the output heads of CellSegNet:

        L = λ_fg     · (Dice + BCE)       on foreground probability
          + λ_center · FocalLoss           on cell-centre heatmap
          + λ_offset · SmoothL1 (fg only)  on centroid-offset vectors

    Offsets are in raw voxel units (Δz, Δy, Δx), so λ_offset should be
    set smaller than λ_fg / λ_center to keep magnitudes balanced
    (typical cell radius ≈ 10–20 vx, BCE loss ≈ 0.5–1.0).
    """

    def __init__(
        self,
        w_fg: float = 1.0,
        w_center: float = 1.0,
        w_offset: float = 0.05,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth_l1_beta: float = 1.0,
    ):
        super().__init__()
        self.w_fg      = w_fg
        self.w_center  = w_center
        self.w_offset  = w_offset
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.beta = smooth_l1_beta

    def forward(self, preds, targets):
        fg_logits = preds['fg_logits'].squeeze(1)           # (B, D, H, W)
        hm_logits = preds['heatmap_logits'].squeeze(1)      # (B, D, H, W)
        offsets   = preds['offsets']                        # (B, 3, D, H, W)

        tgt_fg  = targets['fg']                             # (B, D, H, W) float [0,1]
        tgt_hm  = targets['heatmap']                        # (B, D, H, W) float [0,1]
        tgt_off = targets['offsets']                        # (B, 3, D, H, W) voxel units
        fg_mask = targets['fg_mask']                        # (B, D, H, W) bool

        # --- Foreground: Dice + BCE ---
        fg_prob  = torch.sigmoid(fg_logits)
        loss_bce = F.binary_cross_entropy_with_logits(fg_logits, tgt_fg)
        loss_dice = dice_loss(fg_prob, tgt_fg)
        loss_fg  = loss_bce + loss_dice

        # --- Heatmap: Focal ---
        # loss_center = focal_loss(hm_logits, tgt_hm, self.focal_alpha, self.focal_gamma)
        loss_center = F.binary_cross_entropy_with_logits(hm_logits, tgt_hm)

        # --- Offsets: SmoothL1 masked to foreground voxels ---
        if fg_mask.any():
            mask3 = fg_mask.unsqueeze(1).expand_as(offsets)
            loss_offset = F.smooth_l1_loss(offsets[mask3], tgt_off[mask3], beta=self.beta)
        else:
            loss_offset = offsets.sum() * 0.0   # same device / dtype, no gradient

        total = (
            self.w_fg * loss_fg
            + self.w_center * loss_center
            + self.w_offset * loss_offset
        )

        breakdown = {
            'fg':     loss_fg.item(),
            'center': loss_center.item(),
            'offset': loss_offset.item(),
            'total':  total.item(),
        }
        return total, breakdown
