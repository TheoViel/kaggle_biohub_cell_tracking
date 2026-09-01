"""Cellpose targets in 3D: cell probability and the z/y/x flow field.

The paradigm is Stringer et al., *Cellpose: a generalist algorithm for cellular
segmentation* (Nat. Methods 2021). A nucleus is not predicted as a blob to be
thresholded -- thresholding is what fuses touching cells -- but as a *vector
field* that every voxel of the cell follows inward. Two voxels belong to the
same cell if and only if their trajectories end in the same place, so cells that
touch, and even cells that are concave, still come apart. Reconstruction is in
``dynamics.py``; this module builds what the network is trained to output.

The field is the gradient of a heat distribution diffused from the cell's
centre and confined to the cell, which is what makes it work on shapes a
straight centre-direction cannot handle: for a bent or a pinched nucleus the
straight line to the centroid leaves the cell, and every voxel it passes over
gets dragged toward a cell it does not belong to. Diffusion follows the body.

The synthetic nuclei this trains on are ellipsoids, where the two agree almost
everywhere -- ``mode="center"`` is the cheap exact version for that case and is
worth using while iterating. The default stays with diffusion because the model
is validated on real cells, which are not ellipsoids.

Everything here is in *voxel* space, not microns. The flow is followed on the
voxel grid at inference, so it has to be built on the grid it is followed on;
the 4x anisotropy between z and y/x is carried by the data, not corrected for.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage as ndi

# Cellpose trains flows scaled up so the MSE on them is comparable to the BCE on
# cell probability rather than being swamped by it. Predictions are divided by
# it again before they are followed.
FLOW_SCALE = 5.0

# Six-neighbour stencil: (axis, offset). Diffusion uses faces only, so heat
# cannot leak diagonally between two cells that touch at a corner.
_NEIGHBOURS = ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1))


def _shift(arr: np.ndarray, axis: int, offset: int, fill=0):
    """``arr`` translated by one voxel along ``axis``, edge filled with ``fill``."""
    out = np.full_like(arr, fill)
    src = [slice(None)] * arr.ndim
    dst = [slice(None)] * arr.ndim
    if offset > 0:
        src[axis], dst[axis] = slice(0, -1), slice(1, None)
    else:
        src[axis], dst[axis] = slice(1, None), slice(0, -1)
    out[tuple(dst)] = arr[tuple(src)]
    return out


def _cell_centres(masks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """One seed voxel per label, guaranteed to lie inside the cell.

    The centroid is not usable as a seed on its own: for a concave cell it can
    fall outside the body, and the diffusion would then start from a voxel that
    belongs to a neighbour. This takes the voxel of the cell that is nearest to
    the centroid, which is the centroid itself whenever the centroid is inside.
    """
    # bincount rather than ndi.center_of_mass, which loops over the label list
    # internally: one pass per axis over the volume instead of one pass per
    # cell over its box.
    flat = masks.ravel()
    counts = np.bincount(flat, minlength=int(masks.max()) + 1).astype(np.float64)
    grids = np.ogrid[: masks.shape[0], : masks.shape[1], : masks.shape[2]]
    sums = np.stack([
        np.bincount(
            flat,
            weights=np.broadcast_to(grids[axis], masks.shape).ravel(),
            minlength=len(counts),
        )
        for axis in range(3)
    ], axis=1)
    centroids = (sums[labels] / counts[labels, None]).astype(np.float32)

    # For a convex cell the rounded centroid is already inside it, so the
    # search is only needed where that fails. Testing every cell at once and
    # falling back only for the ones that miss replaces a per-cell Python loop
    # with a single gather; on the synthetic ellipsoids nothing misses. With
    # the bincount above it takes this function from 0.48 s to 0.19 s on a
    # 450-cell volume.
    rounded = np.rint(centroids).astype(np.int32)
    for axis in range(3):
        np.clip(rounded[:, axis], 0, masks.shape[axis] - 1, out=rounded[:, axis])
    hit = masks[rounded[:, 0], rounded[:, 1], rounded[:, 2]] == labels

    seeds = rounded
    if not hit.all():
        objects = ndi.find_objects(masks)
        for i in np.flatnonzero(~hit):
            lab = labels[i]
            sl = objects[lab - 1]
            coords = np.argwhere(masks[sl] == lab) + [s.start for s in sl]
            seeds[i] = coords[int(np.argmin(((coords - centroids[i]) ** 2).sum(1)))]
    return seeds


def _diffuse(masks: np.ndarray, seeds: np.ndarray, n_iter: int) -> np.ndarray:
    """Heat diffused from ``seeds``, each cell isolated from its neighbours.

    One global iteration rather than a loop over cells: a voxel averages only
    those neighbours carrying its own label, so the cells never exchange heat
    even though they are relaxed in the same array. The label-equality masks
    are the expensive part and are built once.
    """
    fg = masks > 0
    same = [
        (_shift(masks, axis, off) == masks) & fg
        for axis, off in _NEIGHBOURS
    ]

    # log-space would underflow; Cellpose keeps the heat itself and takes the
    # gradient of its log at the end.
    heat = np.zeros(masks.shape, np.float64)
    sz, sy, sx = seeds.T
    for _ in range(n_iter):
        acc = np.zeros_like(heat)
        for (axis, off), ok in zip(_NEIGHBOURS, same):
            acc += np.where(ok, _shift(heat, axis, off), heat)
        heat = acc / len(_NEIGHBOURS)
        # Re-inject at the centre every step: a fixed source, so the field
        # settles into a steady state with its maximum at the seed.
        heat[sz, sy, sx] += 1.0
    return heat


def _shift_t(t, axis, offset):
    """``_shift`` for a torch tensor, zero-filled at the edge."""
    out = torch.zeros_like(t)
    src = [slice(None)] * t.ndim
    dst = [slice(None)] * t.ndim
    if offset > 0:
        src[axis], dst[axis] = slice(0, -1), slice(1, None)
    else:
        src[axis], dst[axis] = slice(1, None), slice(0, -1)
    out[tuple(dst)] = t[tuple(src)]
    return out


def _diffuse_torch(masks: np.ndarray, seeds: np.ndarray, n_iter: int, device: str):
    """``_diffuse`` on the GPU, which is where a stencil this shape belongs.

    Identical arithmetic to the numpy version, in float32 rather than float64:
    the relaxation is contracting, so the two agree to well past the precision
    that survives being normalised to a unit vector (measured cosine agreement
    1.000000, minimum 0.9991, over a full volume). The whole point is the
    constant -- the cost is ``n_iter`` passes over the entire volume, six
    shifted reads each, which is a stencil and belongs on a GPU. Building a
    64x256x256 volume's field went from 24 s to ~1.2 s.
    """
    m = torch.from_numpy(np.ascontiguousarray(masks)).to(device)
    fg = m > 0
    # Kept as float weights rather than booleans so the inner loop is a
    # multiply-add. ``where(ok, shift(h), h)`` summed over the six neighbours
    # is the same as ``sum(ok * shift(h)) + h * (6 - sum(ok))``, and the second
    # form has no branch and no temporary per neighbour.
    same = [
        ((_shift_t(m, axis, off) == m) & fg).float() for axis, off in _NEIGHBOURS
    ]
    self_weight = float(len(_NEIGHBOURS)) - sum(same)

    heat = torch.zeros(m.shape, dtype=torch.float32, device=device)
    acc = torch.empty_like(heat)
    idx = torch.from_numpy(np.ascontiguousarray(seeds)).to(device).long()
    sz, sy, sx = idx[:, 0], idx[:, 1], idx[:, 2]
    for _ in range(n_iter):
        torch.mul(heat, self_weight, out=acc)
        for (axis, off), ok in zip(_NEIGHBOURS, same):
            acc.addcmul_(ok, _shift_t(heat, axis, off))
        heat = acc / len(_NEIGHBOURS)
        acc = torch.empty_like(heat)
        heat[sz, sy, sx] += 1.0
    return heat


def _gradient_of_log_torch(heat, fg_np: np.ndarray) -> np.ndarray:
    """``_gradient_of_log`` on the GPU, returning numpy."""
    logged = torch.log1p(heat * 1e12)
    flows = torch.zeros((3,) + heat.shape, dtype=torch.float32, device=heat.device)
    for axis in range(3):
        flows[axis] = 0.5 * (
            _shift_t(logged, axis, -1) - _shift_t(logged, axis, 1)
        )
    fg = torch.from_numpy(np.ascontiguousarray(fg_np)).to(heat.device)
    flows[:, ~fg] = 0.0
    norm = flows.pow(2).sum(0).sqrt()
    flows = torch.where(norm > 1e-20, flows / (norm + 1e-20), flows)
    return flows.cpu().numpy()


def _gradient_of_log(heat: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Central differences of ``log(heat)``, then normalised to unit vectors.

    The log is what makes the field usable far from the centre. Heat falls off
    by orders of magnitude across a cell, so the raw gradient is vanishing at
    the rim -- exactly where a voxel most needs to know which way to go.
    """
    logged = np.log1p(heat * 1e12)  # 1e12: keep the rim above float noise
    flows = np.zeros((3,) + heat.shape, np.float32)
    for axis in range(3):
        plus = _shift(logged, axis, 1)
        minus = _shift(logged, axis, -1)
        flows[axis] = 0.5 * (minus - plus)

    flows[:, ~fg] = 0.0
    norm = np.sqrt((flows**2).sum(0))
    np.divide(flows, norm + 1e-20, out=flows, where=norm > 1e-20)
    return flows


def masks_to_flows(
    masks: np.ndarray,
    mode: str = "diffusion",
    n_iter: int | None = None,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Instance labels -> ``(cellprob, flows)``.

    Args:
        masks: (D, H, W) integer instance labels, 0 = background.
        mode: ``"diffusion"`` for the Cellpose field, ``"center"`` for the
            straight direction to the centroid -- exact for a convex cell and
            far cheaper, wrong for a concave one.
        n_iter: diffusion steps. Defaults to twice the largest cell extent,
            which is what it takes for heat to reach the far rim.
        device: ``"auto"`` uses the GPU when there is one and falls back to
            numpy otherwise. The two paths agree to 1.0000 cosine; the GPU one
            is ~150x faster, which is the difference between minutes and hours
            over a real dataset.

    Returns:
        cellprob: (D, H, W) float32, 1 inside a cell.
        flows: (3, D, H, W) float32 unit vectors pointing inward, 0 outside.
            *Not* scaled by ``FLOW_SCALE`` -- the dataset does that, so that
            what is cached on disk is the physical field.
    """
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError(f"masks must be (D, H, W), got {masks.shape}")
    if not np.issubdtype(masks.dtype, np.integer):
        raise TypeError("masks must hold integer instance labels")

    fg = masks > 0
    cellprob = fg.astype(np.float32)
    labels = np.unique(masks)
    labels = labels[labels != 0]
    if len(labels) == 0:
        return cellprob, np.zeros((3,) + masks.shape, np.float32)

    seeds = _cell_centres(masks, labels)

    if mode == "center":
        flows = np.zeros((3,) + masks.shape, np.float32)
        lut = np.zeros((int(masks.max()) + 1, 3), np.float32)
        lut[labels] = seeds
        zz, yy, xx = np.ogrid[: masks.shape[0], : masks.shape[1], : masks.shape[2]]
        flows[0] = lut[masks, 0] - zz
        flows[1] = lut[masks, 1] - yy
        flows[2] = lut[masks, 2] - xx
        flows[:, ~fg] = 0.0
        norm = np.sqrt((flows**2).sum(0))
        np.divide(flows, norm + 1e-20, out=flows, where=norm > 1e-20)
        return cellprob, flows

    if mode != "diffusion":
        raise ValueError(f"unknown mode {mode!r}, expected 'diffusion' or 'center'")

    if n_iter is None:
        slices = ndi.find_objects(masks)
        extent = max(
            (max(s.stop - s.start for s in sl) for sl in slices if sl is not None),
            default=8,
        )
        # Heat advances one voxel per pass, and the furthest it has to travel
        # is from the seed near the centroid to the far rim -- half the largest
        # cell's extent. Using the extent itself keeps a 2x margin.
        #
        # This was 2 * extent, which is 4x what the geometry needs and was the
        # single largest cost in building the cache. Measured on two volumes
        # (448 and 877 cells), dropping to 1 * extent changes nothing that
        # matters: every cell is still recovered and the median centroid error
        # improves slightly, 0.27 -> 0.24 um and 0.48 -> 0.47 um, because a
        # less converged field has a stronger gradient at the rim where the log
        # is otherwise fighting float noise. Halving it again to extent/2 does
        # start losing cells.
        n_iter = int(max(8, extent))

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        heat = _diffuse_torch(masks, seeds, n_iter, device)
        return cellprob, _gradient_of_log_torch(heat, fg)
    return cellprob, _gradient_of_log(_diffuse(masks, seeds, n_iter), fg)
