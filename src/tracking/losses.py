"""Loss functions for cell tracking.

Three terms:
  L = L_edge + det_loss_weight · L_det + offset_loss_weight · L_offset

L_det: penalty-reduced focal loss (CenterNet) on a Gaussian heatmap.
  - GT peaks:          positive, weight 1
  - Weak-positive peaks: positive, weight = weak_weight
  - Everything else:   negative, scaled by neg_weight and discounted near peaks
  Weak positives are image local maxima brighter than half the mean GT intensity.
L_edge: focal-BCE on softmax edge probabilities, masked to annotated rows/cols.
L_offset: SmoothL1 on the centroid vote, over the voxels near a GT cell only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# How `sigma_per_axis` turns one scalar sigma into a per-axis one.
#   "fine"   physical width anchored on the finest axis -- the tightest the
#            grid can express, and what has been running since 2026-09-22.
#   "grid"   no restatement at all: sigma is in grid voxels on every axis, so
#            the splat is elongated along whichever axis has the coarser voxel.
#            This is what every run up to 2026-09-22 trained with.
#   "coarse" anchored on the *coarsest* axis: physically round like "fine", but
#            at the width "grid" gave z rather than the one it gave y/x.
SIGMA_ANCHORS = ("fine", "grid", "coarse")


def sigma_per_axis(
    sigma: float,
    unet_scaling: Sequence[float] | None,
    voxel_size: Sequence[float] | None,
    anchor: str = "fine",
) -> float | tuple[float, float, float]:
    """``sigma`` restated per axis so the splat is physically round.

    The detection grid is the image grid divided by ``unet_scaling``, so its
    voxel is ``voxel_size * unet_scaling`` microns on each axis. A single sigma
    on a non-cubic grid means a target stretched along whichever axis has the
    coarser voxel -- on the (1, 2, 2) head that was 1.625 um in z against
    0.8125 um in y/x, twice as tall as it was wide.

    The physical width is anchored on the *finest* axis, so making the grid
    finer tightens the target rather than leaving z to dictate it. On the
    (1, 2, 2) head that gives sigma (0.5, 1, 1) voxels -- 1.62 um across on all
    three axes, the sharpest the grid can express -- and z is narrowed to match
    y/x rather than y/x widened to match z.

    On a cubic grid this returns ``sigma`` unchanged, so the deep-supervision
    head at (1, 4, 4) -- where 4 y/x voxels of 0.40625 um make 1.625 um,
    exactly the z step -- trains as it always has.
    """
    if anchor not in SIGMA_ANCHORS:
        raise ValueError(f"sigma anchor must be one of {SIGMA_ANCHORS}, got {anchor!r}")
    if anchor == "grid" or unet_scaling is None or voxel_size is None:
        return sigma
    grid_um = np.asarray(voxel_size, dtype=np.float64) * np.asarray(unet_scaling, dtype=np.float64)
    if not np.isfinite(grid_um).all() or (grid_um <= 0).any():
        return sigma
    # microns, anchored on the finest axis ("fine") or the coarsest ("coarse")
    physical = sigma * (grid_um.min() if anchor == "fine" else grid_um.max())
    per_axis = physical / grid_um
    if np.allclose(per_axis, per_axis[0]):
        return float(per_axis[0])
    return tuple(float(v) for v in per_axis)

# Focal exponents, CenterNet's defaults: alpha sharpens the easy/hard split on
# the prediction, beta discounts negatives by how close they sit to a peak.
FOCAL_ALPHA = 2.0
FOCAL_BETA = 4.0
# gaussian_heatmap gives exp(-d^2 / 2 sigma^2), so thresholding above this is
# exactly d < sigma — it turns the splat into a hard ball of that radius.
_BALL_EDGE = math.exp(-0.5)


@torch.no_grad()
def gaussian_heatmap(
    coords: torch.Tensor,  # (B, max_nodes, 3) in detection-grid space
    mask: torch.Tensor,  # (B, max_nodes) bool
    spatial: tuple,  # (Z, Y, X)
    sigma: float | Sequence[float],
) -> torch.Tensor:  # (B, Z, Y, X) float, peak 1.0 at each GT point
    """Splat a Gaussian at every GT point (CenterNet-style targets).

    Overlapping splats combine with max, so each GT point keeps a peak of
    exactly 1.0. ``sigma <= 0`` degenerates to single-voxel targets.

    Sigma is expressed in *detection-grid voxels*, not microns, and may be one
    value or one per axis. Per-axis is what keeps the splat a fixed *physical*
    size on a grid whose voxels are not cubic: a head that refines y/x but not
    z has voxels 2x taller than they are wide, and a single sigma there means a
    target 2x taller than it is wide. See ``sigma_per_axis``.
    """
    B = coords.shape[0]
    device = coords.device
    Z, Y, X = spatial
    n_vox = B * Z * Y * X

    # One extra slot absorbs out-of-bounds / padded writes, then is dropped.
    heat = torch.zeros(n_vox + 1, device=device)

    sig = np.broadcast_to(np.asarray(sigma, dtype=np.float64).ravel(), (3,))
    radii = [max(1, int(math.ceil(3.0 * s))) if s > 0 else 0 for s in sig]
    offs = [torch.arange(-r, r + 1, device=device) for r in radii]
    oz, oy, ox = torch.meshgrid(*offs, indexing="ij")
    oz, oy, ox = oz.reshape(-1), oy.reshape(-1), ox.reshape(-1)  # (K,)
    if (sig > 0).all():
        # Normalised squared distance, so each axis is measured in its own
        # sigma and the skirt is an ellipsoid matching the voxel aspect.
        sq = sum(
            (o.float() / float(s)) ** 2 for o, s in zip((oz, oy, ox), sig)
        )
        g = torch.exp(-sq / 2.0)
    else:
        g = ((oz * oz + oy * oy + ox * ox) == 0).float()

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


def build_detection_target(
    spatial: tuple,  # (Z, Y, X), the detection-grid shape
    coords: torch.Tensor,  # (B, max_nodes, 3) GT coords in output space
    mask: torch.Tensor,  # (B, max_nodes) bool
    unet_scaling: tuple | None = None,
    weak_weight: float = 0.3,
    weak_smoothing: float = 0.1,
    gaussian_sigma: float = 1.0,
    sigma_anchor: str = "fine",
    weak_positives: torch.Tensor | None = None,  # (B, N, 3), -1 padded
    use_weak_positives: bool = True,
    weak_mode: str | None = None,  # "positive" | "ignore" | "negative"
    weak_ignore_radius: float = 0.0,  # grid voxels; 0 = gaussian_sigma
    ignore_positives: torch.Tensor | None = None,  # (B, N, 3), -1 padded
    ignore_radius: float = 0.0,  # detection-grid voxels; 0 disables
    voxel_size: Sequence[float] | None = None,  # image voxel, microns
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build the Gaussian-splatted target heatmap and per-voxel positive weight.

    Target
    ------
    GT points        1.0 at the peak, decaying as exp(-d^2 / 2 sigma^2)
    Weak positives   1 - weak_smoothing at the peak, same decay
    Ignore regions   no gradient at all — neither positive nor background
    Background       0

    Ignore regions
    --------------
    The annotation covers a few percent of the cells, and the weak positives
    only as many more as the labelling model was confident about. Everything
    left over is currently trained as background, which is a gradient telling
    the detector to suppress cells it can see. ``ignore_positives`` are the
    peaks that were too weak to accept but too plausible to deny: within
    ``ignore_radius`` grid voxels of one, the negative term is dropped, so
    those voxels simply say nothing either way.

    Returns ``(heat, pos_weight, ignore)``: ``heat`` is the target described
    above, ``pos_weight`` is 1 at a GT peak / ``weak_weight`` at a weak peak /
    0 elsewhere, and ``ignore`` (or ``None``) marks voxels within
    ``ignore_radius`` of an uncertain peak.

    ``gaussian_sigma`` is in detection-grid voxels. It must be > 0 for the loss
    to shape a *peak*: at sigma = 0 the target is a lone voxel with no skirt,
    which says "this voxel is a cell" but nothing about being a local maximum,
    and the detector is free to answer with a blob.
    """
    if unet_scaling is not None:  # adapt targets to output space
        scale = torch.tensor(unet_scaling, dtype=coords.dtype, device=coords.device)
        coords = torch.div(coords, scale, rounding_mode="floor")

    # Every splat below -- the target, the weak peaks, and both ignore balls --
    # uses this, so a "say nothing here" region is the same physical size as
    # the peak it stands in for rather than inheriting the voxel aspect.
    sigma = sigma_per_axis(gaussian_sigma, unet_scaling, voxel_size, sigma_anchor)

    # Peak 1.0 at each GT point, decaying across the Gaussian skirt.
    gt_heat = gaussian_heatmap(coords, mask, spatial, sigma)
    heat = gt_heat
    # Weight of the *positive* term. Only a peak voxel is a positive; its own
    # skirt is a discounted negative (see below).

    th = 0.5
    pos_weight = (gt_heat >= th).float()

    # `weak_mode` supersedes `use_weak_positives`; the old flag maps onto the
    # two modes it could express.
    mode = weak_mode or ("positive" if use_weak_positives else "negative")
    if mode not in ("positive", "ignore", "negative"):
        raise ValueError(f"weak_mode must be positive/ignore/negative, got {mode!r}")

    weak_ignore = None
    if mode == "ignore":
        # Neither a positive nor a negative: the annotation does not say these
        # are cells, and the detector can see that they are, so calling them
        # background is the one thing that is certainly wrong -- it is what
        # collapsed node recall from 0.99 to 0.75 in `logs/2026-09-18/6`.
        # Masking them leaves the positives exactly the annotation and the
        # negatives exactly the background, which is what "unweighted" means
        # here: they contribute no gradient at all.
        assert weak_positives is not None, "weak_mode='ignore' with no weak_positives"
        wp_valid = (weak_positives >= 0).all(dim=-1)
        if wp_valid.any():
            wp_i = weak_positives
            if unet_scaling is not None:
                wp_i = torch.div(wp_i, scale, rounding_mode="floor")
            # A ball like `ignore_positives`, not a Gaussian -- "say nothing
            # here" has no shape worth grading. Defaults to the splat width, so
            # the masked region matches what a positive would have covered.
            r = (sigma_per_axis(weak_ignore_radius, unet_scaling, voxel_size,
                                sigma_anchor)
                 if weak_ignore_radius else sigma)
            weak_ignore = gaussian_heatmap(wp_i, wp_valid, spatial, r) > _BALL_EDGE

    if mode == "positive":
        assert weak_positives is not None, "use_weak_positives set with no weak_positives"
        valid = (weak_positives >= 0).all(dim=-1)
        wp = weak_positives
        if unet_scaling is not None:
            wp = torch.div(wp, scale, rounding_mode="floor")

        weak_heat = gaussian_heatmap(wp, valid, spatial, sigma)
        # Weak peaks are cells too, so they must not be driven to 0 as
        # background; smoothing keeps their target just under a GT peak.
        heat = torch.maximum(gt_heat, weak_heat * (1.0 - weak_smoothing))
        weak_peak = (weak_heat >= th) & (gt_heat < th)
        pos_weight = pos_weight + weak_weight * weak_peak.float()

    # Voxels near an uncertain peak: keep them out of the negative term. A
    # ball, not a Gaussian — this is a "say nothing here" region, and its edge
    # carries no meaning worth shaping.
    ignore = None
    if ignore_radius > 0 and ignore_positives is not None:
        ig = ignore_positives
        if unet_scaling is not None:
            scale = torch.tensor(unet_scaling, dtype=coords.dtype, device=coords.device)
            ig = torch.div(ig, scale, rounding_mode="floor")
        ig_valid = (ignore_positives >= 0).all(dim=-1)
        if ig_valid.any():
            ignore = gaussian_heatmap(
                ig, ig_valid, spatial,
                sigma_per_axis(ignore_radius, unet_scaling, voxel_size,
                               sigma_anchor),
            ) > _BALL_EDGE

    # The two ignore sources are independent and both mean "no gradient here".
    if weak_ignore is not None:
        ignore = weak_ignore if ignore is None else (ignore | weak_ignore)

    return heat, pos_weight, ignore


def detection_loss(
    det_logits: torch.Tensor,  # (B, 1, Z, Y, X) raw logits
    coords: torch.Tensor,  # (B, max_nodes, 3) GT coords in output space
    mask: torch.Tensor,  # (B, max_nodes) bool
    neg_weight: float = 1e-3,
    unet_scaling: tuple | None = None,
    weak_weight: float = 0.3,
    weak_smoothing: float = 0.1,
    gaussian_sigma: float = 1.0,
    sigma_anchor: str = "fine",
    weak_positives: torch.Tensor | None = None,  # (B, N, 3), -1 padded
    use_weak_positives: bool = True,
    ignore_positives: torch.Tensor | None = None,  # (B, N, 3), -1 padded
    ignore_radius: float = 0.0,  # detection-grid voxels; 0 disables
    weak_mode: str | None = None,
    weak_ignore_radius: float = 0.0,
    pos_weight_scale: float = 1.0,
    voxel_size: Sequence[float] | None = None,  # image voxel, microns
) -> torch.Tensor:
    """Penalty-reduced focal loss on a Gaussian-splatted detection heatmap.

    Positive term    at peak voxels only: 1 for GT, ``weak_weight`` for weak
    Negative term    every other voxel, scaled by ``neg_weight`` and discounted
                     by (1 - target)^beta so a peak's own skirt is cheap

    Normalised by the number of peaks, so the scale is independent of how many
    cells the crop holds. See ``build_detection_target`` for how the target
    heatmap, positive weight, and ignore mask are constructed.

    ``pos_weight_scale`` multiplies the positive term. The normalisation is by
    peak count, not by voxel count, so the *negative* term is what scales with
    how sparse the positives are: drop from ~500 peaks a frame to the ~3 that
    are annotated and each surviving positive is charged 150x more background.
    A head supervised on the annotation alone therefore needs this above 1 to
    see its positives at all -- see ``annot_pos_scale`` in ``TrackingLoss``.
    """
    spatial = det_logits.shape[2:]
    logits = det_logits[:, 0]  # (B, Z, Y, X)
    batch_size = logits.shape[0]

    heat, pos_weight, ignore = build_detection_target(
        spatial,
        coords,
        mask,
        unet_scaling=unet_scaling,
        weak_weight=weak_weight,
        weak_smoothing=weak_smoothing,
        gaussian_sigma=gaussian_sigma,
        sigma_anchor=sigma_anchor,
        weak_positives=weak_positives,
        use_weak_positives=use_weak_positives,
        ignore_positives=ignore_positives,
        ignore_radius=ignore_radius,
        weak_mode=weak_mode,
        weak_ignore_radius=weak_ignore_radius,
        voxel_size=voxel_size,
    )

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

        pos_loss = -((1.0 - p) ** FOCAL_ALPHA) * torch.log(p) * pos_weight * pos_weight_scale
        neg_loss = (
            -((1.0 - heat) ** FOCAL_BETA)
            * (p ** FOCAL_ALPHA)
            * torch.log(1.0 - p)
            * neg_weight
        )

        per_voxel = torch.where(pos, pos_loss, neg_loss)
        if ignore is not None:
            # Positives win: a GT or weak peak inside an ignore ball is still
            # supervised, only the surrounding background falls silent.
            per_voxel = torch.where(pos | ~ignore, per_voxel, torch.zeros_like(per_voxel))

        # Normalise per peak, so the loss scale does not depend on how many
        # cells a crop happens to contain.
        n_pos = pos.reshape(batch_size, -1).sum(dim=1).clamp(min=1).float()
        per_sample = per_voxel.reshape(batch_size, -1).sum(dim=1)
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


def offset_loss(
    offset_map: torch.Tensor,   # (B, 3, Z, Y, X) predicted vote, grid voxels
    coords: torch.Tensor,       # (B, max_nodes, 3) GT coords in output space
    mask: torch.Tensor,         # (B, max_nodes) bool
    unet_scaling: tuple | None = None,
    radius: int = 2,
    beta: float = 1.0,
) -> torch.Tensor:
    """SmoothL1 on the centroid vote, over the voxels a GT cell votes for.

    Supervised on ground truth only. Weak positives are image intensity maxima,
    and they carry no opinion about where the *cell centre* is beyond the peak
    they already sit on — training the vote toward them would teach the head to
    reproduce the error it exists to remove. Sparse supervision is the trade:
    about three annotated cells a frame, each claiming ``(2r + 1)^3`` voxels.

    SmoothL1 rather than L2 because a vote that lands on the wrong cell is
    wrong by a whole cell spacing, and should not dominate the batch.
    """
    from tracking.targets import build_offset_target

    target, valid = build_offset_target(
        coords, mask, tuple(offset_map.shape[2:]),
        unet_scaling=unet_scaling, radius=radius,
    )
    if not bool(valid.any()):
        # Keep the head attached to the graph so DDP still sees its gradients.
        return offset_map.sum() * 0.0
    sel = valid.unsqueeze(1).expand_as(offset_map)
    return F.smooth_l1_loss(offset_map[sel], target[sel], beta=beta)


class TrackingLoss(nn.Module):
    """Combined detection + edge loss.

        L = L_edge
            + det_loss_weight · (L_det + det_ds_weight · L_det_ds)
            + offset_loss_weight · L_offset

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
    det_sigma_anchor : str
        How a scalar sigma is restated per axis on a non-cubic detection grid;
        see SIGMA_ANCHORS. "grid" reproduces every run before 2026-09-22.
    det_gaussian_sigma : float
        Std-dev, in detection-grid voxels, of the Gaussian splatted at each
        GT point. 0 falls back to single-voxel targets.
    det_ignore_radius : float
        Radius, in detection-grid voxels, of the ball around each uncertain
        peak where the negative term is dropped. 0 disables the mechanism, and
        it does nothing without an ``ignore_positives`` source — see
        ``weak_positives.pl_ignore_folder``.
    det_ds_weight : float
        Weight for the deep-supervision detection term, relative to L_det. Only
        applies on a model whose backbone decodes past the node grid, where the
        coarse head keeps being trained on the target it had before the extra
        stage was added.
    annot_loss_weight : float
        Weight for L_annot, the "would an annotator have labelled this cell"
        head. It is an auxiliary task on shared features, so this also bounds
        how far it is allowed to move the detector.
    annot_neg_weight : float
        Per-voxel negative weight for that head. Lower than the detector's,
        because its positives are ~150x rarer -- only the ground truth counts,
        never a weak positive.
    annot_pos_scale : float
        Multiplier on its positive term. The focal loss normalises by peak
        count, so a head with ~3 positives a frame instead of ~500 has each of
        them charged against the same background; without this above 1 the
        negatives simply drown it and the head learns to answer "no".
    offset_loss_weight : float
        Weight for L_offset. Ignored when the model has no offset head, in
        which case no vote is produced and the term is absent.
    offset_radius : int
        Grid voxels around a GT cell that vote for it. Must match the model's
        ``offset_radius``, which bounds what the head can output.
    offset_beta : float
        SmoothL1 transition point, in grid voxels.
    """

    def __init__(
        self,
        det_loss_weight: float = 1.0,
        det_neg_weight: float = 1e-3,
        det_weak_weight: float = 0.3,
        det_weak_smoothing: float = 0.1,
        det_gaussian_sigma: float = 1.0,
        det_sigma_anchor: str = "fine",
        det_use_weak_positives: bool = True,
        # "positive" (weak peaks are discounted positives), "ignore" (masked
        # out -- positives are the annotation alone, negatives the background
        # alone), or "negative" (weak peaks train as background). None keeps
        # `det_use_weak_positives`, so existing configs are unchanged.
        det_weak_mode: str | None = None,
        det_weak_ignore_radius: float = 0.0,
        det_ignore_radius: float = 0.0,
        det_ds_weight: float = 1.0,
        annot_loss_weight: float = 1.0,
        annot_neg_weight: float = 1e-2,
        annot_pos_scale: float = 1.0,
        div_loss_weight: float = 0.2,
        div_neg_weight: float = 1e-2,
        div_pos_scale: float = 1.0,
        offset_loss_weight: float = 1.0,
        offset_radius: int = 2,
        offset_beta: float = 1.0,
    ):
        super().__init__()
        self.det_loss_weight = det_loss_weight
        self.det_neg_weight = det_neg_weight
        self.det_weak_weight = det_weak_weight
        self.det_weak_smoothing = det_weak_smoothing
        self.det_gaussian_sigma = det_gaussian_sigma
        # Validated here, not at the first splat: sigma_per_axis only runs once
        # a batch has been loaded, so a typo would otherwise surface minutes
        # into a run instead of at construction.
        if det_sigma_anchor not in SIGMA_ANCHORS:
            raise ValueError(
                f"det_sigma_anchor must be one of {SIGMA_ANCHORS}, "
                f"got {det_sigma_anchor!r}"
            )
        self.det_sigma_anchor = det_sigma_anchor
        self.det_use_weak_positives = det_use_weak_positives
        self.det_weak_mode = det_weak_mode
        self.det_weak_ignore_radius = det_weak_ignore_radius
        self.det_ignore_radius = det_ignore_radius
        self.det_ds_weight = det_ds_weight
        self.annot_loss_weight = annot_loss_weight
        self.annot_neg_weight = annot_neg_weight
        self.annot_pos_scale = annot_pos_scale
        # 0.2 rather than 1.0: divisions are 0.13% of annotated links, so this
        # is a very rare positive on a shared trunk. The weight bounds how far
        # it may drag the detector, which is the whole risk of the head -- the
        # detection task is the one that has to keep working.
        self.div_loss_weight = div_loss_weight
        self.div_neg_weight = div_neg_weight
        self.div_pos_scale = div_pos_scale
        self.offset_loss_weight = offset_loss_weight
        self.offset_radius = offset_radius
        self.offset_beta = offset_beta

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
        ignore_pos_per_frame: (
            list[torch.Tensor] | None
        ) = None,  # W × (B, N, 3), -1 padded
        unet_scaling: tuple | None = None,
        offset_maps_per_frame: (
            list[torch.Tensor] | None
        ) = None,  # W × (B, 3, Z, Y, X), None when the head is off
        det_logits_ds_per_frame: (
            list[torch.Tensor] | None
        ) = None,  # W × (B, 1, Z, Y, X) on the node grid, None without deep supervision
        det_stride: tuple | None = None,  # image voxels per heatmap voxel
        annot_logits_per_frame: (
            list[torch.Tensor] | None
        ) = None,  # W × (B, 1, Z, Y, X) on det_logits' grid, None when the head is off
        div_logits_per_frame: (
            list[torch.Tensor] | None
        ) = None,  # W × (B, 1, Z, Y, X) on det_logits' grid, None when the head is off
        voxel_size: Sequence[float] | None = None,  # image voxel, microns
        progress: float = 1.0,  # fraction of training elapsed, in [0, 1]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        W = len(det_logits_per_frame)

        def _det_term(logits_per_frame, scaling):
            """The detection loss averaged over the window, on one grid.

            The GT points are the same either way; ``scaling`` is what places
            them, so the two heads are trained on the same cells splatted at
            each head's own resolution. That second target is the cost of the
            finer head, and it is why the coarse one can be kept unchanged.
            """
            return sum(
                detection_loss(
                    logits_per_frame[i],
                    gt_coords_per_frame[i].clone(),
                    gt_mask_per_frame[i],
                    self.det_neg_weight,
                    unet_scaling=scaling,
                    voxel_size=voxel_size,
                    weak_weight=self.det_weak_weight,
                    weak_smoothing=self.det_weak_smoothing,
                    gaussian_sigma=self.det_gaussian_sigma,
                    sigma_anchor=self.det_sigma_anchor,
                    weak_positives=(
                        weak_pos_per_frame[i] if weak_pos_per_frame is not None else None
                    ),
                    use_weak_positives=self.det_use_weak_positives,
                    weak_mode=self.det_weak_mode,
                    weak_ignore_radius=self.det_weak_ignore_radius,
                    ignore_positives=(
                        ignore_pos_per_frame[i]
                        if ignore_pos_per_frame is not None else None
                    ),
                    ignore_radius=self.det_ignore_radius,
                )
                for i in range(len(logits_per_frame))
            ) / len(logits_per_frame)

        # An empty det_logits_per_frame means the detector is frozen
        # (transformer-only training): nothing to supervise on the voxel grid.
        if W:
            # unet_scaling stays the node grid, which the offset head and the
            # deep-supervision heatmap both live on; only the main heatmap can
            # be finer, and det_stride is what says so.
            l_det = _det_term(
                det_logits_per_frame,
                unet_scaling if det_stride is None else det_stride,
            )
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
        }

        if annot_logits_per_frame:
            # Ground truth only: no weak positives, by construction. A weak
            # positive is a cell the *detector* believes in, which is precisely
            # the class this head exists to separate the annotation from -- feed
            # them in and the two heads collapse onto the same question.
            l_annot = sum(
                detection_loss(
                    annot_logits_per_frame[i],
                    gt_coords_per_frame[i].clone(),
                    gt_mask_per_frame[i],
                    self.annot_neg_weight,
                    unet_scaling=unet_scaling if det_stride is None else det_stride,
                    voxel_size=voxel_size,
                    weak_smoothing=self.det_weak_smoothing,
                    gaussian_sigma=self.det_gaussian_sigma,
                    sigma_anchor=self.det_sigma_anchor,
                    use_weak_positives=False,
                    pos_weight_scale=self.annot_pos_scale,
                )
                for i in range(len(annot_logits_per_frame))
            ) / len(annot_logits_per_frame)
            total = total + self.annot_loss_weight * l_annot
            breakdown["annot"] = self.annot_loss_weight * l_annot.item()

        if div_logits_per_frame:
            # Which annotated cells divide into the next frame. The transition
            # matrices already say so and nothing new has to be read: a row of
            # `edge_targets_per_pair[i]` carries one 1 per child, so a parent
            # that splits sums to >= 2. `_k_step_transition` documents that
            # shape -- "a cell that divides somewhere inside the gap reaches
            # both daughters and the row carries two ones".
            #
            # Ground truth only, like the annotation head: a weak positive has
            # no lineage, so its row is not evidence of anything either way.
            # The last frame of the window has no transition matrix after it
            # and is skipped rather than labelled negative -- "no successor
            # recorded" is not "did not divide".
            l_div, n_div = 0.0, 0
            for i in range(min(len(div_logits_per_frame), len(edge_targets_per_pair))):
                tgt = edge_targets_per_pair[i]              # (B, N_t, N_t1)
                mask_src = det_mask_per_pair[i][0]          # (B, N_t) real rows
                divides = (tgt.sum(dim=-1) >= 2) & mask_src.bool()
                coords = gt_coords_per_frame[i].clone()
                # Keep the dividing cells; the rest become padding rather than
                # negatives, exactly as `gt_mask_per_frame` marks absent nodes.
                keep = divides & gt_mask_per_frame[i].bool()
                l_div = l_div + detection_loss(
                    div_logits_per_frame[i],
                    coords,
                    keep,
                    self.div_neg_weight,
                    unet_scaling=unet_scaling if det_stride is None else det_stride,
                    voxel_size=voxel_size,
                    weak_smoothing=self.det_weak_smoothing,
                    gaussian_sigma=self.det_gaussian_sigma,
                    sigma_anchor=self.det_sigma_anchor,
                    use_weak_positives=False,
                    pos_weight_scale=self.div_pos_scale,
                )
                n_div += 1
            if n_div:
                l_div = l_div / n_div
                total = total + self.div_loss_weight * l_div
                breakdown["div"] = self.div_loss_weight * l_div.item()

        if det_logits_ds_per_frame:
            l_det_ds = _det_term(det_logits_ds_per_frame, unet_scaling)
            total = total + self.det_loss_weight * self.det_ds_weight * l_det_ds
            breakdown["det_ds"] = (
                self.det_loss_weight * self.det_ds_weight * l_det_ds.item()
            )

        if offset_maps_per_frame:
            offset_losses = [
                offset_loss(
                    offset_maps_per_frame[i],
                    gt_coords_per_frame[i],
                    gt_mask_per_frame[i],
                    unet_scaling=unet_scaling,
                    radius=self.offset_radius,
                    beta=self.offset_beta,
                )
                for i in range(len(offset_maps_per_frame))
            ]
            l_offset = sum(offset_losses) / len(offset_losses)
            total = total + self.offset_loss_weight * l_offset
            breakdown["offset"] = self.offset_loss_weight * l_offset.item()

        breakdown["total"] = total.item()
        return total, breakdown
