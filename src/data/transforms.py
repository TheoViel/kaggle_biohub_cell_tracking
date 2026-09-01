"""Shared augmentation transforms for segmentation and tracking datasets."""

from __future__ import annotations

from collections.abc import Callable, Sequence

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


def clamp_coords(coords: torch.Tensor, spatial) -> torch.Tensor:
    """Pull coordinates onto the last valid index of each axis.

    Coordinates name voxel centres, so the valid range is ``[0, n - 1]`` and
    not ``[0, n)``: ``n - 0.3`` is inside the box but rounds to an index that
    is not there. Every consumer rounds, so producing one is a silent loss of
    whatever the coordinate labelled.
    """
    hi = coords.new_tensor([float(n) - 1.0 for n in spatial])
    return coords.clamp(min=torch.zeros_like(hi), max=hi)


def _compact_nodes(
    coords: torch.Tensor,   # (W, M, 3)
    masks: torch.Tensor,    # (W, M) bool
    targets: torch.Tensor,  # (W-1, M, M)
    keep: torch.Tensor,     # (W, M) bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop the nodes ``keep`` rejects and slide the rest to the front.

    The node axis is padded to a fixed width so batches collate, which means
    index ``i`` carries no meaning beyond "the i-th surviving node of this
    frame". Dropping a node therefore renumbers every node behind it, and the
    transition matrices — indexed by exactly those numbers — have to be
    rebuilt against the new numbering rather than merely masked.
    """
    new_coords = torch.zeros_like(coords)
    new_masks = torch.zeros_like(masks)
    kept_indices: list[torch.Tensor] = []
    for frame in range(coords.shape[0]):
        indices = torch.nonzero(keep[frame], as_tuple=False).squeeze(1)
        kept_indices.append(indices)
        count = indices.numel()
        if count:
            new_coords[frame, :count] = coords[frame, indices]
            new_masks[frame, :count] = True

    new_targets = torch.zeros_like(targets)
    for frame in range(targets.shape[0]):
        source = kept_indices[frame]
        target = kept_indices[frame + 1]
        if source.numel() and target.numel():
            block = targets[frame].index_select(0, source).index_select(1, target)
            new_targets[frame, :source.numel(), :target.numel()] = block

    return new_coords, new_masks, new_targets


def _repad_weak(pts_per_frame: list[torch.Tensor], width: int) -> torch.Tensor:
    """Stack per-frame point lists back into a fixed-width -1-padded tensor."""
    out = torch.full((len(pts_per_frame), width, 3), -1.0, dtype=torch.float32)
    for frame, pts in enumerate(pts_per_frame):
        n = min(pts.shape[0], width)
        if n:
            out[frame, :n] = pts[:n]
    return out


def _map_point_set(
    pts: torch.Tensor,                                   # (W, N, 3), -1 padded
    forward: Callable[[torch.Tensor], torch.Tensor],
    shape_out: tuple[int, int, int],
) -> torch.Tensor:
    """Apply ``forward`` to the real points, drop those leaving ``shape_out``.

    Padding is -1 rather than 0 because a zero row is a valid coordinate, so it
    is stripped before the map and restored after rather than transformed.
    """
    hi = pts.new_tensor(shape_out)
    kept = []
    for frame in range(pts.shape[0]):
        real = pts[frame][(pts[frame] >= 0).all(dim=-1)]
        moved = forward(real) if real.numel() else real
        # Inside the box, then clamped onto a real index -- the same rule the
        # GT coords follow, so a weak positive in the last half-voxel is moved
        # rather than dropped.
        inside = moved[((moved >= 0) & (moved < hi)).all(dim=-1)]
        kept.append(clamp_coords(inside, shape_out))
    return _repad_weak(kept, pts.shape[1])


def random_crop_track(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    point_sets: Sequence[torch.Tensor],
    masks: torch.Tensor,
    targets: torch.Tensor,
    crop_size: tuple[int, int, int],
    rng: np.random.Generator,
    anchor: bool = True,
    halo: Sequence[int] = (0, 0, 0),
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    tuple[int, int, int],
]:
    """
    Crop around a GT transition, retaining its cells at both t and t + 1.

    With ``anchor=False`` the crop is placed uniformly at random instead, and
    may well contain no cells at all. That is the point: anchoring every crop
    on an annotation means the detector is only ever shown tissue, while at
    inference it runs over whole volumes that are largely empty.

    Retained nodes are compacted to the front of each padded node dimension,
    and transition matrices are remapped to match the new node indices.

    ``point_sets`` is a list of -1-padded (W, N, 3) auxiliary point clouds —
    weak positives, ignore regions — each shifted by the same origin, with
    points falling outside the crop dropped. They are defined in the uncropped
    frame and would otherwise silently end up misaligned. The origin is
    returned as well.

    ``halo`` widens the returned crop by up to that many voxels on each side,
    keeping the ``crop_size`` box concentric inside it, so a later centre-crop
    recovers exactly the crop this would otherwise have returned. It is context
    for the resampling transforms to consume and throw away — see
    ``work_crop_size``.

    The halo is opportunistic: it is whatever fits between the chosen crop and
    the edge of the volume, per axis, and it never moves the crop to make room.
    Demanding the full halo instead would push every crop away from the volume
    border and quietly cost the cells living there — a cell 6 voxels from the
    face has no 16-voxel halo, and the answer is to give it the halo it has,
    not to stop training on it.
    """
    # coord-space spatial bounds derived from the (potentially full-scale) image
    coord_spatial = imgs.shape[1:]
    if len(crop_size) != 3 or any(size <= 0 for size in crop_size):
        raise ValueError(f"crop_size must contain three positive values, got {crop_size}")
    if any(crop > size for crop, size in zip(crop_size, coord_spatial)):
        raise ValueError(
            f"crop_size {crop_size} exceeds coord spatial shape {coord_spatial}"
        )

    if anchor:
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

        room = _resolve_halo(
            halo, crop_size, coord_spatial,
            lower_anchor.tolist(), upper_anchor.tolist(),
        )
        # Draw the *widened* crop, so the target box lands concentric in it.
        starts = []
        for lower, upper, image_size, crop, r in zip(
            lower_anchor.tolist(), upper_anchor.tolist(), coord_spatial,
            crop_size, room,
        ):
            low = max(0, upper - r - crop + 1)
            high = min(lower - r, image_size - crop - 2 * r)
            starts.append(int(rng.integers(low, high + 1)))
    else:
        room = _resolve_halo(halo, crop_size, coord_spatial)
        starts = [
            int(rng.integers(0, max(1, image_size - crop - 2 * r + 1)))
            for image_size, crop, r in zip(coord_spatial, crop_size, room)
        ]

    size = tuple(c + 2 * r for c, r in zip(crop_size, room))
    return _crop_at(imgs, coords, point_sets, masks, targets, starts, size)


def _resolve_halo(halo, crop_size, spatial, anchor_lo=None, anchor_hi=None):
    """The halo actually available, per axis.

    The halo is taken in full wherever the volume has room. That is the whole
    point of it: the artefacts it removes are worst exactly at the edge of the
    volume, so a halo that quietly shrank to nothing there would protect only
    the crops that never needed protecting.

    ``anchor_lo``/``anchor_hi`` are for the plain-crop path, which never moves
    the crop to make room. There, a transition closer to the face than the halo
    is wide leaves no legal placement at all, so the halo gives way rather than
    the cell. That path does not rotate, so the halo it loses only feeds the
    blur and the downscale, and losing it costs little.

    Y and X take the smaller of their two halos, because ``rot90_track`` only
    offers quarter turns on a square field.
    """
    room = [min(int(h), (n - c) // 2) for h, c, n in zip(halo, crop_size, spatial)]
    if anchor_lo is not None:
        room = [
            min(r, lo, n - 1 - hi)
            for r, lo, hi, n in zip(room, anchor_lo, anchor_hi, spatial)
        ]
    room = [max(0, int(r)) for r in room]
    room[1] = room[2] = min(room[1], room[2])
    return room


def _crop_at(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    point_sets: Sequence[torch.Tensor],
    masks: torch.Tensor,
    targets: torch.Tensor,
    starts: Sequence[int],
    crop_size: tuple[int, int, int],
):
    """Take the box at ``starts``, shifting every label with it."""
    starts = [int(v) for v in starts]
    z0, y0, x0 = starts
    cz, cy, cx = crop_size
    imgs = imgs[:, z0:(z0 + cz), y0:(y0 + cy), x0:(x0 + cx)]

    offset = coords.new_tensor(starts)
    shifted_coords = coords - offset
    voxels = torch.floor(coords).to(torch.long)
    starts_t = voxels.new_tensor(starts)
    ends_t = starts_t + voxels.new_tensor(list(crop_size))
    in_crop = ((voxels >= starts_t) & (voxels < ends_t)).all(dim=-1)
    keep = masks & in_crop

    # Containment is decided on the voxel a coordinate falls in, so a cell at
    # 63.7 of a 64-wide crop is inside it -- but the coordinate itself then
    # names a voxel that does not exist. Consumers round (see
    # losses._scatter_gaussian) and drop anything that lands past the last
    # index, so left alone this quietly deletes the detection target of every
    # cell in the final half-voxel of a crop. Clamp instead: the cell is
    # genuinely there, and half a voxel is a tenth of its radius.
    shifted_coords = clamp_coords(shifted_coords, crop_size)

    new_coords, new_masks, new_targets = _compact_nodes(
        shifted_coords, masks, targets, keep
    )

    # Same shift for the weak positives, dropping anything outside the crop.
    new_points = [
        _map_point_set(p, lambda q: q - q.new_tensor(starts), crop_size)
        for p in point_sets
    ]

    return imgs, new_coords, new_points, new_masks, new_targets, tuple(starts)


def crop_halo(
    crop_size: tuple[int, int, int] | None,
    spatial: tuple[int, int, int],
    margin: float = 1.5,
) -> tuple[int, int, int]:
    """Voxels of context to keep on each side of a crop, then throw away.

    Every resampling transform corrupts the border of what it touches: a
    rotation pulls zero padding into the corners, ``lowres_augment`` and
    ``blur_augment`` both invent the edge out of replicated voxels. Carrying a
    margin through the geometric stage and centre-cropping it off afterwards
    removes all of that, and training on crops is what makes it free -- the
    context is already in the volume, it just was not being read.

    At ``margin`` 1.5 the guarantee for rotation is exact. The crop has side L
    and is carried at side 1.5 L, whose centre ``affine_track`` keeps at least
    0.75 L from every face; a rotation maps an output offset to an input
    offset of the same length, so everything within 0.75 L of the centre reads
    real data, and the crop's own half-diagonal is 0.71 L. Hence 1.5: it is
    the first factor above sqrt(2). Shrinking by ``scale_range`` stretches the
    read region by 1/k on top, which 1.5 only partly covers.

    Returns zeros when there is no crop -- a window that is not cropped has no
    margin to spend, and shrinking the output instead would break collation.
    The halo is a ceiling, not a promise: each crop takes only what fits
    between it and the edge of the volume.
    """
    if crop_size is None or margin <= 1.0:
        return (0, 0, 0)
    return tuple(
        int(max(0, np.ceil(c * margin) - c) // 2)
        for c, n in zip(crop_size, spatial)
    )


def centre_crop_track(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    point_sets: Sequence[torch.Tensor],
    masks: torch.Tensor,
    targets: torch.Tensor,
    crop_size: tuple[int, int, int],
):
    """Cut the middle ``crop_size`` box out of a window, labels and all.

    The other half of ``work_crop_size``: the margin it added exists to be
    thrown away here, taking the resampling artefacts along the border with it.
    """
    spatial = tuple(imgs.shape[1:])
    crop_size = tuple(min(c, n) for c, n in zip(crop_size, spatial))
    starts = [(n - c) // 2 for n, c in zip(spatial, crop_size)]
    out = _crop_at(imgs, coords, point_sets, masks, targets, starts, crop_size)
    return (*out[:5], crop_size)


# ---------------------------------------------------------------------------
# Geometric transforms
#
# Every one of these applies the *same* transform to all W frames of a window.
# The edge head is trained on where a cell moved between t and t+1, so a
# per-frame transform would inject fake motion and teach it noise.
#
# Rotation and scale are defined in physical space (µm), not voxels. Z is
# sampled 4x coarser than Y/X (1.625 vs 0.40625 µm), so a rotation that is
# rigid in voxel space would shear the cells; conjugating by the voxel size,
# as _affine_grid does, is what keeps a cell spherical under rotation.
# ---------------------------------------------------------------------------

def _rot_matrix(axis: int, angle: float) -> torch.Tensor:
    """Rotation about ``axis`` (0=z, 1=y, 2=x) in (z, y, x) coordinates."""
    c, s = float(np.cos(angle)), float(np.sin(angle))
    m = torch.eye(3, dtype=torch.float64)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    m[i, i], m[j, j] = c, c
    m[i, j], m[j, i] = -s, s
    return m


def _scale_bounds(scale_range) -> tuple[float, float] | None:
    """Normalise ``scale_range`` to multiplicative (low, high) bounds."""
    if scale_range is None:
        return None
    if isinstance(scale_range, (int, float)):
        if scale_range <= 0:
            return None
        return (1.0 - float(scale_range), 1.0 + float(scale_range))
    low, high = (float(v) for v in scale_range)
    if low <= 0 or high <= 0:
        raise ValueError(f"scale_range bounds must be positive factors, got {scale_range}")
    if high < low:
        raise ValueError(f"scale_range must be (low, high), got {scale_range}")
    # A degenerate range is a fixed factor, not "no scaling" — (0.72, 0.72) has
    # to shrink. Only an identity range means there is nothing to do.
    return None if low == high == 1.0 else (low, high)


def sample_affine(
    rng: np.random.Generator,
    rot_z_deg: float = 180.0,
    rot_xy_deg: float = 0.0,
    scale_range=0.0,
    anisotropic_scale: float = 0.0,
) -> torch.Tensor:
    """A random 3x3 physical-space forward map ``p_out = A p_in``.

    ``rot_z_deg`` turns the volume in the imaging plane. Y and X share a voxel
    spacing, so that rotation is exact in physical space and is the one an
    embryo mounted at a different angle actually needs — allow the full 180.
    ``rot_xy_deg`` tilts out of plane, which resamples the coarse Z axis into
    the fine ones and blurs as it goes; keep it to the few degrees of real
    mounting tilt.

    ``scale_range`` is a pair of multiplicative factors, sampled log-uniformly.
    It need not straddle 1: if the training embryo is at an earlier stage than
    the test one its cells are larger and sparser, and only shrinking closes
    that gap. Shrinking does both at once — a crop of fixed voxel size then
    spans more tissue, so the cells in it are smaller *and* more numerous.
    """
    a = torch.eye(3, dtype=torch.float64)
    if rot_z_deg:
        a = a @ _rot_matrix(0, float(rng.uniform(-rot_z_deg, rot_z_deg)) * np.pi / 180.0)
    if rot_xy_deg:
        for axis in (1, 2):
            a = a @ _rot_matrix(
                axis, float(rng.uniform(-rot_xy_deg, rot_xy_deg)) * np.pi / 180.0
            )
    bounds = _scale_bounds(scale_range)
    if bounds or anisotropic_scale:
        k = 1.0
        if bounds:
            k = float(np.exp(rng.uniform(np.log(bounds[0]), np.log(bounds[1]))))
        per_axis = np.exp(rng.uniform(-anisotropic_scale, anisotropic_scale, size=3))
        a = torch.diag(torch.as_tensor(k * per_axis, dtype=torch.float64)) @ a
    return a


def _sample_grid(
    shape_in: tuple[int, int, int],
    shape_out: tuple[int, int, int],
    centre: torch.Tensor,       # (3,) input voxel coords the output centres on
    a_inv: torch.Tensor,        # (3, 3) physical inverse map
    voxel_size: torch.Tensor,   # (3,)
) -> torch.Tensor:
    """grid_sample grid pulling ``shape_out`` voxels out of ``shape_in``."""
    s = voxel_size.to(torch.float64)
    axes = [
        (torch.arange(n, dtype=torch.float64) - (n - 1) / 2.0) * s[i]
        for i, n in enumerate(shape_out)
    ]
    q = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)  # (Cz, Cy, Cx, 3) µm
    v = centre.to(torch.float64) + (q @ a_inv.T) / s               # input voxel coords

    # grid_sample wants normalised coords, last axis ordered (x, y, z).
    lengths = torch.tensor(shape_in, dtype=torch.float64)
    n = (2.0 * v + 1.0) / lengths - 1.0
    return n.flip(-1).unsqueeze(0).float()


def _map_points(
    pts: torch.Tensor,          # (..., 3) input voxel coords
    shape_in: tuple[int, int, int],
    shape_out: tuple[int, int, int],
    centre: torch.Tensor,
    a: torch.Tensor,
    voxel_size: torch.Tensor,
) -> torch.Tensor:
    """The forward map ``_sample_grid`` inverts: input voxels -> output voxels."""
    s = voxel_size.to(torch.float64)
    p = (pts.to(torch.float64) - centre.to(torch.float64)) * s
    q = p @ a.to(torch.float64).T
    half = torch.tensor([(n - 1) / 2.0 for n in shape_out], dtype=torch.float64)
    return (q / s + half).to(pts.dtype)


def affine_track(
    imgs: torch.Tensor,         # (W, Z, Y, X)
    coords: torch.Tensor,       # (W, M, 3)
    point_sets: Sequence[torch.Tensor],  # each (W, N, 3), -1 padded
    masks: torch.Tensor,        # (W, M) bool
    targets: torch.Tensor,      # (W-1, M, M)
    out_size: tuple[int, int, int] | None,
    voxel_size: tuple[float, float, float],
    rng: np.random.Generator,
    rot_z_deg: float = 180.0,
    rot_xy_deg: float = 0.0,
    scale_range: float = 0.0,
    anisotropic_scale: float = 0.0,
    shift_um: float = 0.0,
    centre_jitter: float = 0.25,
    interpolation: str = "bilinear",
    padding_mode: str = "zeros",
    anchor: bool = True,
    halo: Sequence[int] = (0, 0, 0),
    max_tries: int = 4,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    tuple[int, int, int],
] | None:
    """Rotate / scale / shift the window, resampling the crop in one pass.

    When ``out_size`` is given this replaces ``random_crop_track`` rather than
    following it: cropping first and rotating the crop afterwards would leave
    the corners empty, because the content that belongs there was never read.
    Sampling the rotated crop straight out of the full volume costs the same
    one interpolation and keeps the corners filled.

    The output is anchored on a GT transition, as ``random_crop_track`` anchors
    it, so the window still contains a linked pair to learn an edge from.
    Returns ``None`` if no draw kept one — the caller has to fall back to a
    plain crop rather than to the untransformed window, whose shape would not
    collate with the rest of the batch.

    ``halo`` widens the output by up to that many voxels on each side, keeping
    ``out_size`` concentric within it, so a later centre-crop recovers the
    output this would otherwise have produced. That extra ring is where the
    rotation's empty corners go, and it is opportunistic: the anchor is placed
    exactly as it would be without one, and the halo is whatever the volume
    then has room for. See ``work_crop_size``.
    """
    shape_in = tuple(imgs.shape[1:])
    keep_size = tuple(out_size) if out_size is not None else shape_in
    vs = torch.tensor(voxel_size, dtype=torch.float32)

    edges = _feasible_track_edges(coords, masks, targets, keep_size, shape_in)

    for _ in range(max_tries):
        a = sample_affine(rng, rot_z_deg, rot_xy_deg, scale_range, anisotropic_scale)

        # Anchor the output on a GT transition when cropping, on the volume
        # centre when not. Jitter is a fraction of the output extent, small
        # enough that the anchor survives a rotation about that centre.
        centre = torch.tensor([(n - 1) / 2.0 for n in shape_in], dtype=torch.float32)
        if out_size is not None:
            if not anchor:
                # Anywhere in the volume, cells or not.
                centre = torch.as_tensor(
                    rng.uniform(0.0, 1.0, size=3) * np.asarray(shape_in), dtype=torch.float32
                )
            elif edges.shape[0]:
                frame_idx, source_idx, target_idx = edges[
                    int(rng.integers(edges.shape[0]))
                ].tolist()
                centre = 0.5 * (coords[frame_idx, source_idx] + coords[frame_idx + 1, target_idx])
            if anchor:
                centre = centre + torch.tensor(
                    [rng.uniform(-centre_jitter, centre_jitter) * n for n in keep_size],
                    dtype=torch.float32,
                )
            # Keep the axis-aligned output box -- crop plus halo -- inside the
            # volume. That halo is what a rotated corner reads into instead of
            # reading past the face, so it has to be real data and the centre
            # has to stand clear of the face to supply it. The cost is that the
            # crop no longer reaches the outermost `halo` voxels of the volume;
            # the measured gain is 276x less padding, and the anchoring holds
            # (no draw was rejected over 60 windows, and GT edges per sample
            # went up rather than down, 6.2 -> 7.2).
            room = _resolve_halo(halo, keep_size, shape_in)
            lo = torch.tensor(
                [k / 2.0 + r for k, r in zip(keep_size, room)], dtype=torch.float32
            )
            far = torch.tensor(shape_in, dtype=torch.float32) - lo
            centre = torch.minimum(torch.maximum(centre, lo), torch.maximum(lo, far))
        else:
            room = [0, 0, 0]

        if shift_um:
            # Translating the content is translating where we read from.
            centre = centre - torch.as_tensor(
                rng.uniform(-shift_um, shift_um, size=3), dtype=torch.float32
            ) / vs

        shape_out = tuple(k + 2 * r for k, r in zip(keep_size, room))
        hi = torch.tensor(shape_out, dtype=coords.dtype)
        # Corners of the box that outlives the centre-crop, in output voxels.
        keep_lo = torch.tensor([float(r) for r in room], dtype=coords.dtype)
        keep_hi = keep_lo + torch.tensor(keep_size, dtype=coords.dtype) - 1

        new_coords = _map_points(coords, shape_in, shape_out, centre, a, vs)
        inside = ((new_coords >= 0) & (new_coords <= hi - 1)).all(dim=-1)
        keep = masks & inside
        # A window with nodes but no surviving *edge* teaches the detector and
        # starves the edge head, so require a linked pair to have survived —
        # unless the crop was placed at random, where holding no cells at all
        # is the intended outcome. Survival is judged on the box that outlives
        # the centre-crop, while `keep` still spans the whole output: a node in
        # the margin is really there, it just may not be there at the end.
        if not anchor:
            break
        survives = masks & (
            (new_coords >= keep_lo) & (new_coords <= keep_hi)
        ).all(dim=-1)
        if targets.numel():
            ok = (
                (targets > 0)
                & survives[:-1].unsqueeze(2)
                & survives[1:].unsqueeze(1)
            ).any()
        else:
            ok = survives.any()
        if ok:
            break
    else:
        return None

    def forward(pts: torch.Tensor) -> torch.Tensor:
        return _map_points(pts, shape_in, shape_out, centre, a, vs)

    a_inv = torch.linalg.inv(a)

    grid = _sample_grid(shape_in, shape_out, centre, a_inv, vs)
    new_imgs = F.grid_sample(
        imgs.unsqueeze(1).float(),
        grid.expand(imgs.shape[0], -1, -1, -1, -1),
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=False,
    ).squeeze(1)

    new_coords, new_masks, new_targets = _compact_nodes(new_coords, masks, targets, keep)

    new_points = [_map_point_set(p, forward, shape_out) for p in point_sets]

    return new_imgs, new_coords, new_points, new_masks, new_targets, shape_out


def rot90_track(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    point_sets: Sequence[torch.Tensor],
    masks: torch.Tensor,
    spatial: tuple[int, int, int],
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], tuple[int, int, int]]:
    """Quarter turns in the Y-X plane.

    Y and X share a voxel spacing, so this is exact — no interpolation, no
    empty corners, no blur. Together with the existing three flips it hands
    the detector the full eight-fold symmetry of the imaging plane for free,
    where flips alone give four.
    """
    # A quarter turn transposes Y and X. On a non-square field that changes the
    # sample's shape, and samples of two shapes will not collate into a batch —
    # so there, only the half turn is available.
    k = int(rng.integers(4)) if spatial[1] == spatial[2] else 2 * int(rng.integers(2))
    if k == 0:
        return imgs, coords, list(point_sets), spatial

    def quarter_turn(pts, old_x):
        return torch.stack(
            [pts[..., 0], (old_x - 1) - pts[..., 2], pts[..., 1]], dim=-1
        )

    valid = [(p >= 0).all(dim=-1, keepdim=True) for p in point_sets]
    point_sets = list(point_sets)
    z, y, x = spatial
    for _ in range(k):
        # One quarter turn = transpose Y/X, then flip the new Y.
        imgs = imgs.transpose(2, 3).flip(2)
        old_x = x
        coords = quarter_turn(coords, old_x)
        point_sets = [quarter_turn(p, old_x) for p in point_sets]
        y, x = x, y

    coords = coords * masks.float().unsqueeze(-1)
    point_sets = [
        torch.where(v, p, torch.full_like(p, -1.0)) for p, v in zip(point_sets, valid)
    ]
    return imgs.contiguous(), coords, point_sets, (z, y, x)


# ---------------------------------------------------------------------------
# Acquisition-domain transforms
#
# The train and test embryos are imaged on separate sessions, so the point
# spread function, sampling and noise floor are free to differ. These perturb
# exactly those, and unlike the photometric ones they change the *shape* of a
# cell rather than its brightness.
# ---------------------------------------------------------------------------

def blur_augment(
    imgs: torch.Tensor,
    rng: np.random.Generator,
    voxel_size: tuple[float, float, float],
    sigma_um: float = 1.0,
) -> torch.Tensor:
    """Separable Gaussian blur with a physical sigma, widening the PSF."""
    sigma = float(rng.uniform(0.0, sigma_um))
    if sigma <= 0:
        return imgs
    out = imgs.unsqueeze(1).float()
    for axis, vs in enumerate(voxel_size):
        s = sigma / float(vs)
        radius = int(np.ceil(3 * s))
        if radius < 1:
            continue
        t = torch.arange(-radius, radius + 1, dtype=torch.float32)
        k = torch.exp(-0.5 * (t / s) ** 2)
        k = k / k.sum()
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = k.numel()
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - axis)] = pad[2 * (2 - axis) + 1] = radius
        out = F.conv3d(F.pad(out, pad, mode="replicate"), k.view(shape))
    return out.squeeze(1)


def background_level_augment(
    imgs: torch.Tensor,
    rng: np.random.Generator,
    level_range: tuple[float, float] = (0.03, 0.30),
    clamp_max: float = 4.0,
) -> torch.Tensor:
    """Re-level the background to a value drawn from ``level_range``.

    The repo normalises each video by its own 0.001/0.999 quantiles, which
    equalises the *amplitude* of a cell across embryos — measured at 9% apart —
    but not the level underneath it, which sits ~3x apart. That residual is
    what this randomises away.

    It sets the level rather than adding to it, because either embryo can be
    the training one: an additive offset that only ever raises the floor would
    prepare a model trained on the darker embryo for the brighter one and do
    nothing for the reverse fold.

    The map is piecewise linear about the sample's own background ``bg``:
    everything above it is translated, everything below is scaled into
    ``[0, target]``. That is continuous, strictly monotone and bounded below by
    zero, so — unlike subtracting a constant — it cannot drive the background
    through zero and flatten it into a plateau. No peak moves and none is
    created, which is what keeps the GT heatmap and the precomputed weak
    positives valid.
    """
    low, high = level_range
    if high < low:
        raise ValueError(f"level_range must be (low, high), got {level_range}")
    target = float(rng.uniform(low, high))

    flat = imgs.reshape(-1)
    bg = float(flat[:: max(1, flat.numel() // 100_000)].median())
    if bg <= 1e-6:
        return imgs

    out = torch.where(imgs >= bg, imgs - bg + target, imgs * (target / bg))
    return out.clamp(0.0, clamp_max)


def lowres_augment(
    imgs: torch.Tensor,
    rng: np.random.Generator,
    max_factor: float = 2.0,
) -> torch.Tensor:
    """Downsample in Y/X and put it back, as a lower-NA objective would.

    Z is left alone: at 1.625 µm it is already the coarse axis and dropping it
    further would erase the cell rather than soften it.
    """
    factor = float(rng.uniform(1.0, max_factor))
    if factor <= 1.01:
        return imgs
    z, y, x = imgs.shape[-3:]
    small = (z, max(1, int(y / factor)), max(1, int(x / factor)))
    out = imgs.unsqueeze(1).float()
    out = F.interpolate(out, size=small, mode="trilinear", align_corners=False)
    out = F.interpolate(out, size=(z, y, x), mode="trilinear", align_corners=False)
    return out.squeeze(1)


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
    point_sets: Sequence[torch.Tensor],  # each (W, N, 3), -1 padded
    masks: torch.Tensor,        # (W, M) bool
    spatial: tuple[int, int, int],
    rng: np.random.Generator,
    p: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random axis-aligned flips on Z, Y, X independently.

    ``point_sets`` are flipped with the same axes as the image. Unlike the GT
    coords they are padded with -1 rather than 0, so the padding is restored
    afterwards instead of being multiplied out by a mask — flipping a -1 row
    would otherwise turn it into a real coordinate at the far edge.
    """
    flip_axes = [ax for ax in range(3) if rng.random() < p]
    if not flip_axes:
        return imgs, coords, list(point_sets), masks

    imgs = torch.flip(imgs, dims=[ax + 1 for ax in flip_axes])

    new_coords = coords.clone()
    new_points = [p.clone() for p in point_sets]
    valid = [(p >= 0).all(dim=-1, keepdim=True) for p in point_sets]
    for ax in flip_axes:
        # Coordinates may be fractional after spatial downsampling. The model
        # indexes their containing voxel, so flip that voxel index directly.
        new_coords[..., ax] = (spatial[ax] - 1) - torch.floor(coords[..., ax])
        for new_p, p in zip(new_points, point_sets):
            new_p[..., ax] = (spatial[ax] - 1) - torch.floor(p[..., ax])
    new_coords = new_coords * masks.float().unsqueeze(-1)
    new_points = [
        torch.where(v, p, torch.full_like(p, -1.0)) for p, v in zip(new_points, valid)
    ]

    return imgs, new_coords, new_points, masks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _gaussian_profile(
    n: int,
    centre: float,
    sigma: float,
) -> torch.Tensor:
    """1-D Gaussian bump, peak 1 at ``centre``, in voxel units."""
    v = torch.arange(n, dtype=torch.float32)
    return torch.exp(-((v - centre) ** 2) / (2.0 * max(sigma, 1e-3) ** 2))


def _background_field(
    imgs: torch.Tensor,                       # (W, Z, Y, X)
    voxel_size: tuple[float, float, float],
    block_um: float = 16.0,
    smooth_um: float = 48.0,
) -> torch.Tensor:
    """Smooth local background, from block minima rather than block means.

    A single number will not do: a real frame spans a 7x illumination gradient
    (measured over 6bba, octant max/min), so one global background is wrong by
    a large factor at both ends of it.

    The minimum, not the mean, because a mean is *made of* the cells. In a
    crowded region the block mean sits well above the true floor, and worse it
    moves whenever the cells move -- dim them and the "background" follows them
    down, so a contrast measured against it barely changes. Block minima over
    ~16 um (cells are ~9 um across at ~15 um spacing, so a block almost always
    contains some true background) are then averaged up to ``smooth_um`` to
    take the noise off. The result tracks the illumination and ignores the
    cells, which is what makes a contrast ratio against it mean anything.
    """
    W, Z, Y, X = imgs.shape
    k = tuple(
        max(1, min(n, int(round(block_um / v))))
        for n, v in zip((Z, Y, X), voxel_size)
    )
    # min-pool == -max-pool(-x); stride = kernel, so this is a coarse grid.
    coarse = -F.max_pool3d(-imgs.unsqueeze(1), kernel_size=k, stride=k, ceil_mode=True)
    # Average those minima over a few blocks to damp the noise floor.
    blocks = tuple(
        max(1, min(c, int(round(n * v / smooth_um))))
        for c, n, v in zip(coarse.shape[2:], (Z, Y, X), voxel_size)
    )
    coarse = F.adaptive_avg_pool3d(coarse, blocks)
    return F.interpolate(
        coarse, size=(Z, Y, X), mode="trilinear", align_corners=False
    ).squeeze(1)


def dim_augment(
    imgs: torch.Tensor,             # (W, Z, Y, X) normalised
    coords: torch.Tensor,           # (W, M, 3) GT coords, current image space
    masks: torch.Tensor,            # (W, M) bool
    rng: np.random.Generator,
    voxel_size: tuple[float, float, float],
    factor_range: tuple[float, float] = (0.12, 0.85),
    min_peak_ratio: float = 1.3,
    radius_um_range: tuple[float, float] = (25.0, 70.0),
    clamp_max: float = 4.0,
) -> torch.Tensor:
    """Dim a region's cells toward their local background.

    Measured on logs/2026-08-28/17 over 20 stacks, the GT cells the tracker
    misses differ from the ones it finds almost entirely in *contrast*, not in
    absolute brightness and not in where they sit:

        median               matched   missed   ratio
        absolute peak         0.670     0.228    0.34
        local background      0.165     0.144    0.88
        contrast (peak - bg)  0.464     0.064    0.14
        peak / background     3.91      1.44     0.37

    A missed cell rises 44% above its surroundings where a found one rises
    291%. Its background is the same. So the axis to augment along is contrast,
    and the transform is

        out = bg(v) + (x - bg(v)) * f

    with ``bg`` the local background *field*, not a constant: that leaves the
    illumination gradient exactly as it was and scales only the structure on
    top of it. ``f`` runs from ``factor_range`` at the centre of a random
    ellipsoid to 1 outside, so one crop holds cells at both ends of the
    contrast range. ``f = 0.14`` takes a typical 3.9 peak/background down to
    the 1.44 the tracker currently misses.

    Not making cells disappear
    --------------------------
    The floor is a ratio, not a level. A cell dimmed into its background is a
    *wrong* label -- the GT heatmap would ask the detector to fire where there
    is no longer evidence -- and "has evidence" means standing above the local
    background, which an absolute threshold cannot express when that background
    varies 7x across the frame. So:

    * ``factor_range`` is bounded away from 0;
    * the field is then measured at every annotated cell and its depth reduced
      until the faintest still sits at ``min_peak_ratio`` times its own local
      background. Cells already below that ratio are excluded -- scaling cannot
      rescue them and they would otherwise veto the augmentation outright.

    ``min_peak_ratio`` defaults just under the 1.44 of a currently-missed cell,
    so the hard cases are reachable while a cell never sinks to its background.

    The region is smooth over tens of microns (``radius_um_range``), so across
    a ~5 um cell the field is near constant and peaks barely move. That matters
    because the GT heatmap and the precomputed weak positives are derived from
    the *unaugmented* image, so a peak that moved would leave them pointing at
    the wrong voxel. Measured over 40 draws on a real stack: 8.5% of GT peaks
    move at all, by a mean of 0.13 um and a 95th percentile of 1.15 um, against
    a 1.625 um detection grid and a 7 um matching tolerance -- so the label
    stays on the cell. The tail reaches 4.1 um, since dimming tilts a
    flat-topped cell toward its brighter side. Widen ``radius_um_range`` if
    that matters; the shift scales as 1 / radius.

    The same field is applied to every frame of the window. A region that
    flickered frame to frame would make one cell bright then dim, which is a
    linking cue that does not exist in the data.
    """
    out = imgs.float()
    W, Z, Y, X = out.shape
    if Z * Y * X == 0:
        return out

    bg = _background_field(out, voxel_size)

    vs = torch.as_tensor(voxel_size, dtype=torch.float32)
    centre = [float(rng.uniform(0, n - 1)) for n in (Z, Y, X)]
    sigmas = [float(rng.uniform(*radius_um_range)) / float(v) for v in vs]
    mz = _gaussian_profile(Z, centre[0], sigmas[0])
    my = _gaussian_profile(Y, centre[1], sigmas[1])
    mx = _gaussian_profile(X, centre[2], sigmas[2])

    depth = 1.0 - float(rng.uniform(*factor_range))
    if depth <= 0:
        return out.clamp(0.0, clamp_max)

    # --- guard: every annotated cell keeps peak >= min_peak_ratio * its bg ---
    pts, frames = [], []
    for w in range(W):
        m = masks[w].bool()
        if not bool(m.any()):
            continue
        c = coords[w][m].round().long()
        c[:, 0] = c[:, 0].clamp(0, Z - 1)
        c[:, 1] = c[:, 1].clamp(0, Y - 1)
        c[:, 2] = c[:, 2].clamp(0, X - 1)
        pts.append(c)
        frames.append(torch.full((len(c),), w, dtype=torch.long))
    if pts:
        c = torch.cat(pts)
        fr = torch.cat(frames)
        # 3x3x3 max: a GT centroid can sit a voxel off the true peak.
        peak = out[fr, c[:, 0], c[:, 1], c[:, 2]]
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    peak = torch.maximum(peak, out[
                        fr,
                        (c[:, 0] + dz).clamp(0, Z - 1),
                        (c[:, 1] + dy).clamp(0, Y - 1),
                        (c[:, 2] + dx).clamp(0, X - 1),
                    ])
        bgc = bg[fr, c[:, 0], c[:, 1], c[:, 2]]
        mcell = mz[c[:, 0]] * my[c[:, 1]] * mx[c[:, 2]]
        # bgc + (peak - bgc) * (1 - depth * mcell) >= min_peak_ratio * bgc
        above = peak - bgc
        # A 10% margin on the target. The guard is exact against the background
        # field of the *input*, but that field is a non-linear (min-pool)
        # function of the image, so recomputing it on the output lands a few
        # percent higher near the dimmed region's edge and the realised ratio
        # comes out correspondingly low. Aiming slightly high makes the floor
        # hold on the output, which is where it is claimed.
        target = 1.0 + 1.10 * (min_peak_ratio - 1.0)
        ok = (peak > target * bgc) & (above > 1e-6) & (mcell > 1e-6)
        if bool(ok.any()):
            allowed = (
                1.0 - (target - 1.0) * bgc[ok] / above[ok]
            ) / mcell[ok]
            depth = min(depth, float(allowed.min().clamp(min=0.0)))

    if depth <= 0:
        return out.clamp(0.0, clamp_max)

    f = 1.0 - depth * (mz[:, None, None] * my[None, :, None] * mx[None, None, :])
    # Only what rises above the background is scaled. Applying f below it too
    # would lift the dark voxels toward bg, which raises the very background
    # the contrast is measured against -- the cell would lose contrast twice,
    # and the ratio guard above (computed against the original bg) would no
    # longer hold on the output.
    return torch.where(out > bg, bg + (out - bg) * f, out).clamp(0.0, clamp_max)


AUG_DEFAULTS: dict = {
    # --- Sampling ---
    # Fraction of crops centred on a GT transition. The rest land anywhere in
    # the volume, including the ~53% of it that holds no cells — which is what
    # the detector otherwise never sees, and why it over-predicts.
    "crop_anchor_p": 1.0,
    # Frame gap within a window. 1 is consecutive frames; higher values are
    # built at load time, so this is read by build_loader, not per sample.
    "time_stride_max": 1,
    # --- Geometry: same transform for every frame of a window ---
    "flip_p": 0.5,             # per axis, independently
    "rot90_p": 0.0,            # quarter turn in Y/X; exact, free, no empty corners
    "affine_p": 0.0,           # continuous rotate/scale/shift; resamples the crop
    "rot_z_deg": 180.0,        # in-plane, physically exact — the mounting angle
    "rot_xy_deg": 5.0,         # out-of-plane tilt; resamples the coarse Z axis, keep small
    "scale_range": (0.7, 1.45),   # log-uniform isotropic factors, symmetric about 1
    "anisotropic_scale": 0.0,  # extra per-axis scale, i.e. z-spacing miscalibration
    "shift_um": 2.0,           # sub-voxel translation, decorrelates cells from the grid
    "affine_interpolation": "bilinear",
    "affine_padding": "zeros",
    # Work at this multiple of crop_size, then centre-crop back. Every
    # resampling transform corrupts the border of what it touches — a rotation
    # pulls in zero padding at the corners, a downscale-and-back and a blur
    # both invent the edge from replicated voxels — and training on crops means
    # that border does not have to be kept. 1.5 is the smallest factor that
    # covers a quarter-turn: rotating a box by 45 deg needs sqrt(2) = 1.414 of
    # its side to stay filled. Costs 1.5^3 = 3.4x the voxels through the
    # geometric stage; set it to 1.0 to switch the whole thing off.
    "crop_margin": 1.5,
    # --- Acquisition domain: changes a cell's shape, not its brightness ---
    "blur_p": 0.0,             # wider PSF
    "blur_sigma_um": 1.0,
    "lowres_p": 0.0,           # lower sampling in Y/X, then back up
    "lowres_max_factor": 2.0,
    # --- Photometric ---
    "brightness_p": 0.9,         # was hard-coded inside heavy_brightness_augment
    "brightness_strength": 1.0,  # 0 = plain scalar shift instead of the heavy field
    "noise_p": 0.0,              # shot-like noise; voxel-scale, so it can move a peak
    "bg_level_p": 1.0,           # randomise the background level
    "bg_level_range": (0.03, 0.30),  # target level, normalised units
    # Dim a region toward the background, so annotated cells there look like
    # the ones the detector misses (median peak 0.22 in the worst stacks, 0.52
    # in ordinary ones, against 0.63-0.70 for cells it finds).
    "dim_p": 0.0,
    "dim_factor_range": (0.12, 0.85),   # multiplier on contrast (signal - local bg)
    "dim_min_peak_ratio": 1.3,          # no cell falls below this x its local bg
    "dim_radius_um": (25.0, 70.0),      # region radii; >> a cell, so peaks do not move
}


def augment_track(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    point_sets: Sequence[torch.Tensor],
    masks: torch.Tensor,
    targets: torch.Tensor,
    coord_spatial: tuple[int, int, int],
    crop_size: tuple[int, int, int] | None,
    voxel_size: tuple[float, float, float],
    cfg: dict,
    rng: np.random.Generator,
):
    """Apply the configured augmentations to one tracking window.

    Geometry runs first and at most one resampling happens: ``affine_track``
    subsumes the crop when it fires, so a sample is never interpolated twice.
    The exact transforms (quarter turns, flips) come after, and the intensity
    ones last.

    When ``crop_size`` is set, the geometric and resampling stages run on a
    crop ``crop_margin`` times larger and the result is centre-cropped back.
    Everything that resamples corrupts its own border, and cutting the border
    off afterwards is free here in a way it is not at inference: the crop is
    ours to choose. See ``work_crop_size``.
    """
    cfg = {**AUG_DEFAULTS, **(cfg or {})}

    # Anchored on an annotation, or placed anywhere in the volume. Drawn once
    # and shared by whichever crop path runs.
    anchor = rng.random() < cfg["crop_anchor_p"]

    # Context for the resampling stages to consume: the crop is placed exactly
    # as it always was, then widened by up to this many voxels a side.
    halo = crop_halo(crop_size, tuple(imgs.shape[1:]), cfg["crop_margin"])

    done = None
    if rng.random() < cfg["affine_p"]:
        done = affine_track(
            imgs, coords, point_sets, masks, targets,
            out_size=crop_size,
            voxel_size=voxel_size,
            rng=rng,
            rot_z_deg=cfg["rot_z_deg"],
            rot_xy_deg=cfg["rot_xy_deg"],
            scale_range=cfg["scale_range"],
            anisotropic_scale=cfg["anisotropic_scale"],
            shift_um=cfg["shift_um"],
            interpolation=cfg["affine_interpolation"],
            padding_mode=cfg["affine_padding"],
            anchor=anchor,
            halo=halo,
        )

    if done is not None:
        imgs, coords, point_sets, masks, targets, coord_spatial = done
    elif crop_size is not None:
        imgs, coords, point_sets, masks, targets, _ = random_crop_track(
            imgs, coords, point_sets, masks, targets, crop_size, rng,
            anchor=anchor, halo=halo,
        )
        coord_spatial = tuple(imgs.shape[1:])

    if rng.random() < cfg["rot90_p"]:
        imgs, coords, point_sets, coord_spatial = rot90_track(
            imgs, coords, point_sets, masks, coord_spatial, rng
        )

    if cfg["flip_p"] > 0:
        imgs, coords, point_sets, masks = flip_augment(
            imgs, coords, point_sets, masks, coord_spatial, rng, p=cfg["flip_p"]
        )

    if rng.random() < cfg["blur_p"]:
        imgs = blur_augment(imgs, rng, voxel_size, sigma_um=cfg["blur_sigma_um"])
    if rng.random() < cfg["lowres_p"]:
        imgs = lowres_augment(imgs, rng, max_factor=cfg["lowres_max_factor"])

    # Everything that resamples has now run, so cut the margin away and with it
    # every border artefact those transforms left. The photometric transforms
    # come after: they are smooth fields sized to the crop, and applying them
    # to the oversized one would deliver only their middle to the model.
    if crop_size is not None and tuple(coord_spatial) != tuple(crop_size):
        imgs, coords, point_sets, masks, targets, coord_spatial = centre_crop_track(
            imgs, coords, point_sets, masks, targets, crop_size
        )

    if rng.random() < cfg["brightness_p"]:
        if cfg["brightness_strength"] > 0:
            imgs = heavy_brightness_augment(
                imgs,
                rng,
                strength=cfg["brightness_strength"],
                # The draw is made here, like every other augmentation's, so the
                # two are not multiplied together behind the config's back.
                p=1.0,
                noise=rng.random() < cfg["noise_p"],
            )
        else:
            imgs = brightness_augment(imgs, rng)

    if rng.random() < cfg["bg_level_p"]:
        imgs = background_level_augment(imgs, rng, level_range=cfg["bg_level_range"])

    # After the background re-level, so the floor it dims toward is the one the
    # sample actually ships with.
    if rng.random() < cfg["dim_p"]:
        imgs = dim_augment(
            imgs,
            coords,
            masks,
            rng,
            voxel_size,
            factor_range=cfg["dim_factor_range"],
            min_peak_ratio=cfg["dim_min_peak_ratio"],
            radius_um_range=cfg["dim_radius_um"],
        )

    return imgs, coords, point_sets, masks, targets, coord_spatial
