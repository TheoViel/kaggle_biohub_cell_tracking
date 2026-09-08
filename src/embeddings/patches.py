"""Cutting the (z, y, x) = (3, 64, 64) patches NETr reads.

The paper's frame normalisation is kept exactly: intensities are rescaled so
each *frame's* 1st and 99.99th percentiles land on [0, 1]. Per frame, not per
patch -- a patch is mostly one cell, so normalising inside it would erase how
bright that cell is relative to its neighbours, which is a large part of what
distinguishes them.
"""

from __future__ import annotations

import numpy as np

# The paper's patch, in (z, y, x). The three z-slices become the backbone's
# input channels; 64 y/x voxels is 26 um here, so a cell (~8.5 um) sits inside
# it with room for the neighbourhood the transformer attends over.
PATCH_ZYX = (3, 64, 64)


def normalise_frame(volume: np.ndarray, lo_q: float = 1.0,
                    hi_q: float = 99.99) -> np.ndarray:
    """Frame percentile normalisation onto [0, 1], as the paper specifies."""
    v = volume.astype(np.float32)
    lo, hi = np.percentile(v, [lo_q, hi_q])
    return np.clip((v - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def _rotate_scale(dy, dx, angle_deg: float, scale: float):
    """Map patch-local offsets through the view's rotation and scale."""
    if angle_deg == 0.0 and scale == 1.0:
        return dy, dx
    theta = np.deg2rad(angle_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    ry = (cos * dy - sin * dx) / scale
    rx = (sin * dy + cos * dx) / scale
    return ry, rx


def extract_patches(
    volume: np.ndarray,          # (Z, Y, X) normalised frame
    centres: np.ndarray,         # (N, 3) z/y/x in image voxels, may be fractional
    patch_zyx: tuple[int, int, int] = PATCH_ZYX,
    angle_deg: float = 0.0,
    scale: float = 1.0,
    elastic: np.ndarray | None = None,   # (2, Y, X) y/x displacement in voxels
    n_frames: int = 1,
) -> np.ndarray:
    """(N, n_frames, pz, py, px) patches, with the view's geometry applied.

    ``n_frames`` contiguous slabs of ``pz`` slices are cut, centred together on
    the cell, so a patch spans ``n_frames * pz`` slices of z while each slab
    stays the 3-channel image a 2D backbone can read. The slabs are encoded
    independently and averaged after the backbone (see ``NETr.forward``), which
    buys z context without making z a spatial axis the ViT would have to learn
    to interpret, and without a 3D backbone to pretrain.

    The frame axis is always present, so ``n_frames=1`` is the old behaviour
    with a length-1 axis rather than a different shape.

    Sampling is done by building each patch's coordinate grid and reading the
    volume there, rather than transforming the whole frame and cutting squares
    out of it. Same result for a handful of cells, far less work: a frame is
    64x256x256 and the patches cover a small part of it.

    Reads outside the volume come back as zeros, which is what a cell at the
    edge of the field should look like -- there is genuinely nothing there.
    Note that at ``n_frames * pz`` comparable to the stack depth, a cell near
    the top or bottom has whole slabs land outside it, and those all-zero slabs
    still enter the mean.
    """
    pz, py, px = patch_zyx
    n_frames = int(n_frames)
    depth = pz * n_frames        # z slices spanned, before slabbing
    Z, Y, X = volume.shape
    n = len(centres)
    if n == 0:
        return np.zeros((0, n_frames, pz, py, px), np.float32)

    # Patch-local offsets, centred. One contiguous run over the whole depth,
    # split into slabs at the end -- the slabs are adjacent in z, not
    # independently placed.
    oz = np.arange(depth, dtype=np.float32) - (depth - 1) / 2.0
    oy = np.arange(py, dtype=np.float32) - (py - 1) / 2.0
    ox = np.arange(px, dtype=np.float32) - (px - 1) / 2.0
    gy, gx = np.meshgrid(oy, ox, indexing="ij")
    gy, gx = _rotate_scale(gy, gx, angle_deg, scale)

    cz = centres[:, 0][:, None, None, None]
    cy = centres[:, 1][:, None, None, None] + gy[None, None]
    cx = centres[:, 2][:, None, None, None] + gx[None, None]
    zz = cz + oz[None, :, None, None]

    if elastic is not None:
        # Displace by the field sampled at the *undisplaced* location, which is
        # the usual approximation and is exact to first order for a field this
        # smooth.
        iy = np.clip(np.rint(cy).astype(np.int32), 0, Y - 1)
        ix = np.clip(np.rint(cx).astype(np.int32), 0, X - 1)
        cy = cy + elastic[0][iy, ix]
        cx = cx + elastic[1][iy, ix]

    # Nearest-neighbour sampling. Trilinear would be smoother, but these are
    # views of the *same* frame and the network must not be able to tell them
    # apart by interpolation smoothness -- a shortcut that would make the
    # contrastive task easier than the real matching problem.
    iz = np.rint(np.broadcast_to(zz, (n, depth, py, px))).astype(np.int32)
    iy = np.rint(np.broadcast_to(cy, (n, depth, py, px))).astype(np.int32)
    ix = np.rint(np.broadcast_to(cx, (n, depth, py, px))).astype(np.int32)

    inside = (
        (iz >= 0) & (iz < Z) & (iy >= 0) & (iy < Y) & (ix >= 0) & (ix < X)
    )
    out = np.zeros((n, depth, py, px), np.float32)
    out[inside] = volume[
        np.clip(iz, 0, Z - 1)[inside],
        np.clip(iy, 0, Y - 1)[inside],
        np.clip(ix, 0, X - 1)[inside],
    ]
    return out.reshape(n, n_frames, pz, py, px)


def normalise_coords(coords: np.ndarray, shape_zyx) -> np.ndarray:
    """Voxel coordinates -> [-1, 1] per axis, for the positional MLP.

    Normalised by the volume rather than by the cells present, so the same
    physical location maps to the same vector in every frame; a per-frame
    min/max would move the origin whenever the cells did.
    """
    extent = np.asarray(shape_zyx, np.float32) - 1.0
    return (2.0 * coords.astype(np.float32) / np.maximum(extent, 1.0) - 1.0)
