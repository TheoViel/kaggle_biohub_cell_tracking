"""Rebuilding cells from a flow field: the Cellpose reconstruction heuristic.

This is the half of Cellpose that does the actual separating. The network only
ever sees a vector field; nothing in its output says where one cell stops and
the next begins. That comes from *following* the field: every foreground voxel
is advanced along the flow for a fixed number of Euler steps, and voxels whose
trajectories end at the same attractor are declared the same cell.

Why it beats thresholding a blob map, which is what the rest of this repo's
detector does: two nuclei pressed together share a ridge of high cell
probability and a threshold fuses them into one object, and no threshold
separates them because there is no minimum between them to cut at. Their flows
still point to two different centres, so the trajectories part company even
though the probabilities never do.

The three knobs that decide the result, and what each one trades:

* ``n_iter`` -- how long trajectories run. Too few and voxels at the rim of a
  large cell never arrive, so one cell fragments into several. It has to exceed
  the radius of the largest cell in voxels, which on this data is set by y/x
  (~15 voxels at 6 um) rather than z (~4).
* ``min_count`` -- votes needed for an attractor to be a cell. This is the
  precision/recall dial, and the one that moves the node-count adjustment.
* ``max_dist`` -- how far a trajectory may end from its attractor and still be
  counted. Voxels that stall in a flat region of the field would otherwise be
  handed to whichever cell happens to be nearest.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage as ndi

from segmentation.flows import FLOW_SCALE


def _nearest(points: np.ndarray, centres: np.ndarray, device: str = "cuda",
             chunk: int = 262_144):
    """Nearest ``centres`` row for each of ``points``: (distance, index).

    Brute force on the GPU rather than a KD-tree. There are a few hundred
    centres against ~750k points, so the distance matrix is short and wide and
    a tree's log factor buys nothing against the bandwidth; chunked so the
    matrix never has to exist all at once. Two of these run per threshold and
    on cKDTree they were 0.34 s of the 0.73 s each threshold cost.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    c = torch.from_numpy(np.ascontiguousarray(centres)).to(device).float()
    dists, idxs = [], []
    for start in range(0, len(points), chunk):
        p = torch.from_numpy(
            np.ascontiguousarray(points[start:start + chunk])
        ).to(device).float()
        d = torch.cdist(p, c)
        best, arg = d.min(dim=1)
        dists.append(best.cpu().numpy())
        idxs.append(arg.cpu().numpy())
    return np.concatenate(dists), np.concatenate(idxs)


def follow_flows(
    flows: np.ndarray,
    start: np.ndarray,
    n_iter: int = 200,
    step: float = 1.0,
    device: str = "cuda",
) -> np.ndarray:
    """Advance ``start`` along ``flows`` for ``n_iter`` Euler steps.

    Args:
        flows: (3, D, H, W) float, z/y/x components in voxel units. Unit-length
            inside cells, as ``masks_to_flows`` builds them.
        start: (N, 3) float voxel coordinates to launch, in z/y/x.
        n_iter: number of steps. Each step moves at most ``step`` voxels, so
            this bounds how far a voxel can travel.
        step: Euler step length in voxels.

    Returns:
        (N, 3) float32 final positions, in z/y/x voxel coordinates.

    Interpolation is trilinear, which matters: on a nearest-neighbour lookup a
    trajectory locks onto the voxel lattice and stops short of the true centre,
    and two cells whose centres fall in the same voxel then merge.
    """
    if len(start) == 0:
        return np.zeros((0, 3), np.float32)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    shape = np.asarray(flows.shape[1:], np.float32)

    fl = torch.from_numpy(np.ascontiguousarray(flows)).to(device).float()[None]
    p = torch.from_numpy(np.ascontiguousarray(start)).to(device).float()

    # grid_sample wants coordinates in [-1, 1] and ordered (x, y, z), the
    # reverse of the array's own axis order.
    upper = torch.from_numpy(shape - 1).to(device).clamp(min=1)
    for _ in range(n_iter):
        g = (2.0 * p / upper - 1.0).flip(-1)
        sampled = torch.nn.functional.grid_sample(
            fl, g[None, :, None, None, :], align_corners=True,
            mode="bilinear", padding_mode="border",
        )[0, :, :, 0, 0].T
        p = torch.clamp(p + step * sampled, torch.zeros_like(upper), upper)
    return p.cpu().numpy().astype(np.float32)


def _gaussian_kernel1d(sigma: float, device, dtype):
    radius = max(1, int(round(4.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x**2) / (2.0 * sigma * sigma))
    return k / k.sum()


def _attractors(
    final: np.ndarray,
    shape: tuple[int, int, int],
    min_count: int,
    smooth: float,
    nms_size,
    device: str = "cuda",
) -> np.ndarray:
    """Where the trajectories piled up: one coordinate per cell.

    Trajectories converge to a point but never exactly the same point, so the
    endpoints are histogrammed and the peaks of that histogram are taken as the
    cells. Smoothing first is what stops a single cell's slightly scattered
    endpoints registering as several peaks.

    Histogram, blur and local-maximum search run on the GPU. This is the hot
    spot of the whole reconstruction and it is repeated for every threshold in
    a sweep -- on scipy it was 0.41 s of the 0.73 s each threshold cost, more
    than the model's forward pass.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if isinstance(nms_size, int):
        nms_size = (nms_size,) * 3

    idx = torch.from_numpy(np.ascontiguousarray(final)).to(device).round().long()
    for a in range(3):
        idx[:, a].clamp_(0, shape[a] - 1)

    flat = (idx[:, 0] * shape[1] + idx[:, 1]) * shape[2] + idx[:, 2]
    hist = torch.zeros(int(np.prod(shape)), dtype=torch.float32, device=device)
    hist.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    hist = hist.view(shape)[None, None]

    if smooth > 0:
        # Separable, so three 1-D passes rather than one 3-D kernel.
        k = _gaussian_kernel1d(smooth, device, hist.dtype)
        r = (len(k) - 1) // 2
        for axis in range(3):
            kshape = [1, 1, 1, 1, 1]
            kshape[2 + axis] = len(k)
            pad = [0, 0, 0, 0, 0, 0]
            pad[2 * (2 - axis)] = pad[2 * (2 - axis) + 1] = r
            hist = torch.nn.functional.conv3d(
                torch.nn.functional.pad(hist, pad), k.view(kshape)
            )

    if float(hist.max()) <= 0:
        return np.zeros((0, 3), np.float32)

    pooled = torch.nn.functional.max_pool3d(
        hist, kernel_size=tuple(nms_size), stride=1,
        padding=tuple(s // 2 for s in nms_size),
    )
    peaks = ((hist >= pooled) & (hist >= min_count))[0, 0]

    # Only the peak voxels come back from the GPU -- a few hundred of them,
    # against 4.2M for the whole histogram.
    peak_np = peaks.cpu().numpy()
    comps, n = ndi.label(peak_np)
    if n == 0:
        return np.zeros((0, 3), np.float32)

    pc = np.argwhere(peak_np)
    weights = hist[0, 0][peaks].cpu().numpy().astype(np.float64)
    lab = comps[pc[:, 0], pc[:, 1], pc[:, 2]]

    # One coordinate per plateau, weighted by the votes -- a flat-topped peak
    # would otherwise contribute as many cells as it has voxels. bincount over
    # the peak voxels rather than ndi.center_of_mass, which walks the entire
    # volume once per label and cost more than every GPU step combined.
    total = np.bincount(lab, weights=weights, minlength=n + 1)[1:]
    centres = np.stack([
        np.bincount(lab, weights=weights * pc[:, a], minlength=n + 1)[1:]
        for a in range(3)
    ], axis=1) / np.maximum(total, 1e-12)[:, None]
    return centres.astype(np.float32).reshape(-1, 3)


def _assign(
    coords: np.ndarray,
    final: np.ndarray,
    shape,
    min_count: int,
    min_size: int,
    smooth: float,
    nms_size,
    max_dist: float | None,
    return_masks: bool,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Trajectory endpoints -> cells. The half that does not touch the model.

    Split out from ``masks_from_flows`` so a threshold sweep can reuse one set
    of trajectories: where a voxel *ends up* depends only on the flow field and
    where it started, never on which other voxels were launched alongside it,
    so raising the cell-probability threshold only removes starting points. One
    forward pass and one integration therefore serve every threshold, and this
    -- a histogram, a maximum filter and two KD-tree queries -- is all that has
    to be repeated.
    """
    empty = (np.zeros(shape, np.int32), np.zeros((0, 3), np.float32))
    if len(coords) == 0:
        return empty

    centres = _attractors(final, shape, min_count, smooth, nms_size, device)
    if len(centres) == 0:
        return empty

    dist, assign = _nearest(final, centres, device)
    keep = np.ones(len(final), bool) if max_dist is None else dist <= max_dist
    if not keep.any():
        return empty

    # Drop attractors too weak to be cells, then reassign: a voxel whose
    # attractor has just been removed belongs to whichever cell survives
    # nearest it, not to nothing.
    counts = np.bincount(assign[keep], minlength=len(centres))
    centres = centres[counts >= min_size]
    if len(centres) == 0:
        return empty

    dist, assign = _nearest(final, centres, device)
    keep = np.ones(len(final), bool) if max_dist is None else dist <= max_dist

    counts = np.bincount(assign[keep], minlength=len(centres))
    alive = counts >= min_size
    if not alive.any():
        return empty
    remap = -np.ones(len(centres), np.int64)
    remap[alive] = np.arange(int(alive.sum()))
    keep &= alive[assign]

    labels = remap[assign[keep]]
    member = coords[keep].astype(np.float32)

    # The centroid of the cell's voxels, not the attractor. The attractor sits
    # wherever the trajectories happened to pile up, which is biased toward the
    # side the cell is thickest on; the voxel centroid is the quantity the node
    # metric is scored against.
    n_cells = int(alive.sum())
    centroids = np.zeros((n_cells, 3), np.float32)
    for a in range(3):
        centroids[:, a] = np.bincount(
            labels, weights=member[:, a], minlength=n_cells
        ) / np.maximum(np.bincount(labels, minlength=n_cells), 1)

    instance = np.zeros(shape, np.int32)
    if return_masks:
        ij = member.astype(np.intp)
        instance[ij[:, 0], ij[:, 1], ij[:, 2]] = labels.astype(np.int32) + 1
    return instance, centroids


def masks_from_flows_multi(
    cellprob: np.ndarray,
    flows: np.ndarray,
    thresholds=(0.5,),
    n_iter: int = 200,
    min_count: int = 8,
    min_size: int = 12,
    smooth: float = 1.0,
    nms_size=(3, 5, 5),
    max_dist: float | None = 6.0,
    flow_scale: float = FLOW_SCALE,
    device: str = "cuda",
    return_masks: bool = False,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Reconstruct at several cell-probability thresholds for one integration.

    Voxels are launched from the *lowest* threshold's foreground, which is a
    superset of every higher one's, and each threshold then reuses the subset
    of those trajectories that clears it. The saving is the whole point: the
    integration is 200 interpolated steps over ~800k voxels, next to which the
    clustering is free.

    Returns ``{threshold: (instance_masks, centroids)}``.
    """
    cellprob = np.asarray(cellprob)
    thresholds = sorted(float(t) for t in thresholds)
    shape = cellprob.shape

    fg = cellprob >= thresholds[0]
    if not fg.any():
        return {t: (np.zeros(shape, np.int32), np.zeros((0, 3), np.float32))
                for t in thresholds}

    coords = np.argwhere(fg).astype(np.float32)
    probs = cellprob[fg]
    final = follow_flows(
        np.asarray(flows, np.float32) / flow_scale, coords,
        n_iter=n_iter, device=device,
    )

    out = {}
    for t in thresholds:
        sel = probs >= t
        out[t] = _assign(
            coords[sel], final[sel], shape, min_count, min_size,
            smooth, nms_size, max_dist, return_masks, device,
        )
    return out


def masks_from_flows(
    cellprob: np.ndarray,
    flows: np.ndarray,
    cellprob_threshold: float = 0.5,
    n_iter: int = 200,
    min_count: int = 8,
    min_size: int = 12,
    smooth: float = 1.0,
    nms_size=(3, 5, 5),
    max_dist: float | None = 6.0,
    flow_scale: float = FLOW_SCALE,
    device: str = "cuda",
    return_masks: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Cell probability + flows -> ``(instance_masks, centroids)``.

    Args:
        cellprob: (D, H, W) probability in [0, 1], already through a sigmoid.
        flows: (3, D, H, W) as predicted, i.e. still multiplied by
            ``flow_scale``; it is divided out here.
        cellprob_threshold: which voxels are launched. Only sets what is
            *followed* -- it does not decide where cells split, so it is a far
            less delicate number than a detector's threshold.
        min_count / min_size: an attractor needs ``min_count`` trajectories to
            be a cell, and the cell needs ``min_size`` voxels to survive.
        max_dist: voxels ending farther than this from every attractor are left
            unassigned rather than forced into the nearest cell.
        return_masks: skip painting the label volume when only centroids are
            wanted, which is the case for the node metrics.

    Returns:
        instance_masks: (D, H, W) int32 labels, or an all-zero array of that
            shape when ``return_masks`` is False.
        centroids: (N, 3) float32 in z/y/x voxel coordinates.
    """
    return masks_from_flows_multi(
        cellprob, flows, thresholds=(cellprob_threshold,), n_iter=n_iter,
        min_count=min_count, min_size=min_size, smooth=smooth,
        nms_size=nms_size, max_dist=max_dist, flow_scale=flow_scale,
        device=device, return_masks=return_masks,
    )[float(cellprob_threshold)]
