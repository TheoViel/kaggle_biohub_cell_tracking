"""Shared augmentation transforms for segmentation and tracking datasets."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Segmentation augmentations
# ---------------------------------------------------------------------------

def random_crop_seg(
    img: np.ndarray,
    fg: np.ndarray,
    heatmap: np.ndarray,
    offsets: np.ndarray,
    fg_mask: np.ndarray,
    patch_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    D, H, W = img.shape[-3:]
    pd, ph, pw = patch_size
    z0 = np.random.randint(0, max(1, D - pd))
    y0 = np.random.randint(0, max(1, H - ph))
    x0 = np.random.randint(0, max(1, W - pw))
    s = np.s_[z0:z0+pd, y0:y0+ph, x0:x0+pw]
    return (
        img[(slice(None),) + s], fg[s], heatmap[s],
        offsets[:, z0:z0+pd, y0:y0+ph, x0:x0+pw],
        fg_mask[s],
    )


def augment_seg(
    img: np.ndarray,
    fg: np.ndarray,
    heatmap: np.ndarray,
    offsets: np.ndarray,
    fg_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    for axis in range(3):
        if np.random.random() > 0.5:
            img = np.flip(img, axis=axis + 1).copy()
            fg = np.flip(fg, axis=axis).copy()
            heatmap = np.flip(heatmap, axis=axis).copy()
            offsets = np.flip(offsets, axis=axis + 1).copy()
            fg_mask = np.flip(fg_mask, axis=axis).copy()
            # Flipping spatial axis flips that offset component's sign
            offsets[axis] = -offsets[axis]

    scale = 1.0 + 0.15 * np.random.randn()
    shift = 0.05 * np.random.randn()
    img = np.clip(img * scale + shift, 0.0, 1.0)

    return img, fg, heatmap, offsets, fg_mask


# ---------------------------------------------------------------------------
# Tracking augmentations
# ---------------------------------------------------------------------------

def has_feasible_track_crop(
    coords: torch.Tensor,
    masks: torch.Tensor,
    targets: torch.Tensor,
    crop_size: tuple[int, int, int],
    spatial: tuple[int, int, int],
) -> bool:
    """Return whether a crop can contain both endpoints of a GT transition."""
    return _feasible_track_edges(
        coords, masks, targets, crop_size, spatial
    ).shape[0] > 0


def _feasible_track_edges(
    coords: torch.Tensor,
    masks: torch.Tensor,
    targets: torch.Tensor,
    crop_size: tuple[int, int, int],
    spatial: tuple[int, int, int],
) -> torch.Tensor:
    edges = torch.nonzero(targets > 0, as_tuple=False)
    if edges.shape[0] == 0:
        return edges

    frame, source_idx, target_idx = edges.unbind(dim=1)
    source_valid = masks[frame, source_idx]
    target_valid = masks[frame + 1, target_idx]

    voxels = torch.floor(coords).to(torch.long)
    source_voxels = voxels[frame, source_idx]
    target_voxels = voxels[frame + 1, target_idx]
    lower = torch.minimum(source_voxels, target_voxels)
    upper = torch.maximum(source_voxels, target_voxels)

    crop = torch.tensor(crop_size, dtype=torch.long, device=coords.device)
    image = torch.tensor(spatial, dtype=torch.long, device=coords.device)
    in_image = ((lower >= 0) & (upper < image)).all(dim=1)
    fits = ((upper - lower) < crop).all(dim=1)
    return edges[source_valid & target_valid & in_image & fits]


def random_crop_track(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    weak_coords: torch.Tensor,
    masks: torch.Tensor,
    targets: torch.Tensor,
    crop_size: tuple[int, int, int],
    rng: np.random.Generator,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    tuple[int, int, int],
]:
    """
    Crop around a GT transition, retaining its cells at both t and t + 1.

    Retained nodes are compacted to the front of each padded node dimension,
    and transition matrices are remapped to match the new node indices.

    ``weak_coords`` (W, N, 3, -1 padded) is shifted by the same origin and
    points falling outside the crop are dropped; it is defined in the
    uncropped frame and would otherwise silently end up misaligned. The
    origin is returned as well.
    """
    # coord-space spatial bounds derived from the (potentially full-scale) image
    coord_spatial = imgs.shape[1:]
    if len(crop_size) != 3 or any(size <= 0 for size in crop_size):
        raise ValueError(f"crop_size must contain three positive values, got {crop_size}")
    if any(crop > size for crop, size in zip(crop_size, coord_spatial)):
        raise ValueError(
            f"crop_size {crop_size} exceeds coord spatial shape {coord_spatial}"
        )

    feasible_edges = _feasible_track_edges(
        coords, masks, targets, crop_size, coord_spatial
    )
    if feasible_edges.shape[0] == 0:
        raise ValueError(
            "Cannot make a track-centred crop: no linked GT cells at t and t + 1 "
            f"fit inside crop_size {crop_size}"
        )

    edge_idx = int(rng.integers(feasible_edges.shape[0]))
    frame_idx, source_idx, target_idx = feasible_edges[edge_idx].tolist()
    source_voxel = torch.floor(coords[frame_idx, source_idx]).to(torch.long)
    target_voxel = torch.floor(coords[frame_idx + 1, target_idx]).to(torch.long)
    lower_anchor = torch.minimum(source_voxel, target_voxel)
    upper_anchor = torch.maximum(source_voxel, target_voxel)

    starts = []
    for lower, upper, image_size, crop in zip(
        lower_anchor.tolist(), upper_anchor.tolist(), coord_spatial, crop_size
    ):
        low = max(0, upper - crop + 1)
        high = min(lower, image_size - crop)
        starts.append(int(rng.integers(low, high + 1)))

    z0, y0, x0 = starts
    cz, cy, cx = crop_size
    imgs = imgs[:, z0:(z0+cz), y0:(y0+cy), x0:(x0+cx)]

    offset = coords.new_tensor(starts)
    shifted_coords = coords - offset
    voxels = torch.floor(coords).to(torch.long)
    starts_t = voxels.new_tensor(starts)
    ends_t = starts_t + voxels.new_tensor(crop_size)
    in_crop = ((voxels >= starts_t) & (voxels < ends_t)).all(dim=-1)
    keep = masks & in_crop

    new_coords = torch.zeros_like(coords)
    new_masks = torch.zeros_like(masks)
    kept_indices: list[torch.Tensor] = []
    for frame in range(coords.shape[0]):
        indices = torch.nonzero(keep[frame], as_tuple=False).squeeze(1)
        kept_indices.append(indices)
        count = indices.numel()
        if count:
            new_coords[frame, :count] = shifted_coords[frame, indices]
            new_masks[frame, :count] = True
    new_targets = torch.zeros_like(targets)
    for frame in range(targets.shape[0]):
        source = kept_indices[frame]
        target = kept_indices[frame + 1]
        if source.numel() and target.numel():
            block = targets[frame].index_select(0, source).index_select(1, target)
            new_targets[frame, :source.numel(), :target.numel()] = block

    # Same shift for the weak positives, dropping anything outside the crop.
    new_weak = torch.full_like(weak_coords, -1.0)
    off = weak_coords.new_tensor(starts)
    hi = weak_coords.new_tensor(crop_size)
    for frame in range(weak_coords.shape[0]):
        pts = weak_coords[frame]
        pts = pts[(pts >= 0).all(dim=-1)] - off
        pts = pts[((pts >= 0) & (pts < hi)).all(dim=-1)]
        n = min(pts.shape[0], new_weak.shape[1])
        if n:
            new_weak[frame, :n] = pts[:n]

    return imgs, new_coords, new_weak, new_masks, new_targets, (z0, y0, x0)


def brightness_augment(
    imgs: torch.Tensor,
    rng: np.random.Generator,
    shift_range: float = 0.1,
) -> torch.Tensor:
    shift = float(rng.uniform(-shift_range, shift_range))
    return (imgs + shift).clamp(0.0)


def _smooth_profile(
    n: int,
    rng: np.random.Generator,
    k: int = 3,
) -> torch.Tensor:
    """Smooth random 1-D profile in ~[0, 1], from ``k`` control points."""
    c = torch.as_tensor(rng.random(k), dtype=torch.float32)[None, None]
    return F.interpolate(c, size=n, mode="linear", align_corners=True)[0, 0]


def heavy_brightness_augment(
    imgs: torch.Tensor,            # (W, Z, Y, X) normalised
    rng: np.random.Generator,
    strength: float = 1.0,
    p: float = 0.9,
    noise: bool = False,
    max_spatial_contrast: float = 4.0,
    clamp_max: float = 4.0,
) -> torch.Tensor:
    """Aggressive photometric augmentation for uneven illumination and haze.

    Targets the failure where the detector fires across a diffuse bright
    region: it has learned that *absolute* brightness indicates a cell. These
    transforms decorrelate brightness from cell identity, so the only stable
    cue left is local peak structure.

    The illumination model is a separable field: an independent smooth profile
    per axis, combined as gain = gz * gy * gx and applied by broadcasting.
    That covers the artefacts light-sheet data actually shows — directional
    ramps, vignetting, depth attenuation — without ever materialising a
    volume-sized field, which is what made the dense version cost ~80 ms per
    sample. Every profile is smooth over tens of microns, so across a ~5 um
    cell the field is near constant: real cells stay local maxima and no new
    ones appear, which matters because the GT heatmap and the precomputed
    weak positives are both derived from the unaugmented image.

    Spatial variation is capped at ``max_spatial_contrast`` and exposure is set
    separately, so one side of the volume cannot be driven to saturation while
    the other is buried at noise level.
    """
    out = imgs.float()
    if strength <= 0 or rng.random() > p:
        return out.clamp(0.0, clamp_max)

    W, Z, Y, X = out.shape
    s = float(strength)

    # Signal amplitude (how far structure rises above its own background),
    # from a subsample. Additive terms are scaled by this rather than by
    # absolute brightness, which would let the haze dwarf the cells.
    flat = out.reshape(-1)
    sub = flat[:: max(1, flat.numel() // 100_000)]
    amp = float(
        (torch.quantile(sub, 0.999) - torch.quantile(sub, 0.5)).clamp_min(1e-4)
    )

    # --- separable multiplicative field ---
    # Each axis profile is normalised to median 1 and clamped so the product
    # spans at most max_spatial_contrast.
    c1 = float(max_spatial_contrast) ** (1.0 / 6.0)
    prof = []
    for n in (Z, Y, X):
        g = 1.0 + (0.8 * s) * (2.0 * _smooth_profile(n, rng, k=3) - 1.0)
        g = g / g.median().clamp_min(1e-6)
        prof.append(g.clamp(1.0 / c1, c1))
    gz, gy, gx = prof

    # depth attenuation folds into the z profile for free
    if rng.random() < 0.6:
        k = float(rng.uniform(0.0, 1.5 * s))
        z = torch.linspace(0, 1, Z)
        if rng.random() < 0.5:
            z = z.flip(0)
        gz = gz * torch.exp(-k * z)
        gz = (gz / gz.median().clamp_min(1e-6)).clamp(1.0 / c1, c1)

    gain = float(np.exp(rng.uniform(-0.6 * s, 0.6 * s)))

    # Collapse the separable factors into one (Z, Y, X) multiplier and one
    # additive field, then apply both in a single fused addcmul. Each pass over
    # the volume costs ~6 ms of pure memory bandwidth, so chaining the
    # broadcasts one at a time was the expense, not the arithmetic.
    mul = (gain * gz)[:, None, None] * gy[None, :, None] * gx[None, None, :]

    # Additive haze is non-negative: a negative offset would clip signal
    # against 0 and flatten it into plateaus, destroying every peak there.
    pedestal = float(rng.uniform(0.0, 0.15) * s * amp)
    if rng.random() < 0.8:
        hz = _smooth_profile(Z, rng, k=2) ** 2
        hy = _smooth_profile(Y, rng, k=2) ** 2
        hx = _smooth_profile(X, rng, k=2) ** 2
        add = (0.35 * s * amp) * (
            hz[:, None, None] * hy[None, :, None] * hx[None, None, :]
        )
        add += pedestal
    else:
        add = torch.full((1, 1, 1), pedestal)

    out = torch.addcmul(add.unsqueeze(0), out, mul.unsqueeze(0))

    # gamma — monotonic, so peak structure is preserved exactly. Asymmetric:
    # gamma < 1 lifts the background toward the peak and is the largest
    # contrast destroyer, so it is allowed only mildly.
    if rng.random() < 0.5:
        gamma = float(np.exp(rng.uniform(-0.2 * s, 0.5 * s)))
        peak = out.amax().clamp_min(1e-6)
        out = peak * (out / peak) ** gamma

    # per-frame gain jitter — bleaching / laser flicker
    if W > 1 and rng.random() < 0.5:
        j = torch.as_tensor(
            np.exp(rng.normal(0.0, 0.08 * s, size=W)), dtype=out.dtype
        )
        out = out * j[:, None, None, None]

    # Shot-like noise. Voxel-scale, unlike everything above, so it can shift a
    # peak off its labelled voxel; off by default and cheap to skip (a
    # volume-sized randn is ~30 ms).
    if noise and rng.random() < 0.7:
        gen = torch.Generator().manual_seed(int(rng.integers(2**31 - 1)))
        n = torch.randn(out.shape, generator=gen)
        out = out + (0.02 * s * amp) * n * out.clamp_min(0.0).sqrt()

    return out.clamp_(0.0, clamp_max)


def flip_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,       # (W, M, 3)
    weak_coords: torch.Tensor,  # (W, N, 3), -1 padded
    masks: torch.Tensor,        # (W, M) bool
    spatial: tuple[int, int, int],
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random axis-aligned flips on Z, Y, X independently.

    ``weak_coords`` is flipped with the same axes as the image. Unlike the GT
    coords it is padded with -1 rather than 0, so the padding is restored
    afterwards instead of being multiplied out by a mask — flipping a -1 row
    would otherwise turn it into a real coordinate at the far edge.
    """
    flip_axes = [ax for ax in range(3) if rng.random() < 0.5]
    if not flip_axes:
        return imgs, coords, weak_coords, masks

    imgs = torch.flip(imgs, dims=[ax + 1 for ax in flip_axes])

    new_coords = coords.clone()
    new_weak = weak_coords.clone()
    valid = (weak_coords >= 0).all(dim=-1, keepdim=True)
    for ax in flip_axes:
        # Coordinates may be fractional after spatial downsampling. The model
        # indexes their containing voxel, so flip that voxel index directly.
        new_coords[..., ax] = (spatial[ax] - 1) - torch.floor(coords[..., ax])
        new_weak[..., ax] = (spatial[ax] - 1) - torch.floor(weak_coords[..., ax])
    new_coords = new_coords * masks.float().unsqueeze(-1)
    new_weak = torch.where(valid, new_weak, torch.full_like(new_weak, -1.0))

    return imgs, new_coords, new_weak, masks
