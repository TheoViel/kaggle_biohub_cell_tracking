"""Shared augmentation transforms for segmentation and tracking datasets."""

from __future__ import annotations

import numpy as np
import torch


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
    masks: torch.Tensor,
    targets: torch.Tensor,
    crop_size: tuple[int, int, int],
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Crop around a GT transition, retaining its cells at both t and t + 1.

    Retained nodes are compacted to the front of each padded node dimension,
    and transition matrices are remapped to match the new node indices.
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

    return imgs, new_coords, new_masks, new_targets


def brightness_augment(
    imgs: torch.Tensor,
    rng: np.random.Generator,
    shift_range: float = 0.1,
) -> torch.Tensor:
    shift = float(rng.uniform(-shift_range, shift_range))
    return (imgs + shift).clamp(0.0)


def flip_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,  # (W, M, 3)
    masks: torch.Tensor,   # (W, M) bool
    spatial: tuple[int, int, int],
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random axis-aligned flips on Z, Y, X independently."""
    flip_axes = [ax for ax in range(3) if rng.random() < 0.5]
    if not flip_axes:
        return imgs, coords, masks

    imgs = torch.flip(imgs, dims=[ax + 1 for ax in flip_axes])

    new_coords = coords.clone()
    for ax in flip_axes:
        # Coordinates may be fractional after spatial downsampling. The model
        # indexes their containing voxel, so flip that voxel index directly.
        new_coords[..., ax] = (spatial[ax] - 1) - torch.floor(coords[..., ax])
    new_coords = new_coords * masks.float().unsqueeze(-1)

    return imgs, new_coords, masks
