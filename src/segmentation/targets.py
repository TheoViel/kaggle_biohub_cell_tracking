import numpy as np
from scipy.spatial import cKDTree

import numpy as np


def compute_all_targets(
    masks: np.ndarray,
    sigma: float = 3.0,
    truncate: float = 4.0,
):
    """
    Derive center-heatmap and centroid-offset targets from a 3D instance mask.

    For each instance k:

        centroid:
            c_k = mean({q : masks[q] == k})

        offset for foreground voxel q:
            v(q) = c_k - q

        center heatmap:
            H(q) = exp(-||q - c_k||² / (2 * sigma²))

    Background voxels have zero offsets.

    Parameters
    ----------
    masks
        Integer array of shape (D, H, W). Background must be labeled 0.
    sigma
        Standard deviation of the centroid Gaussian, in voxel units.
    truncate
        Gaussian support radius as a multiple of sigma.

    Returns
    -------
    heatmap
        Float32 array of shape (D, H, W), with values in [0, 1].
    offsets
        Float32 array of shape (3, D, H, W), ordered as
        [delta_z, delta_y, delta_x].
    centroid_map
        Dictionary mapping instance label to its floating-point centroid
        `(cz, cy, cx)`.
    """
    masks = np.asarray(masks)

    if masks.ndim != 3:
        raise ValueError(
            f"`masks` must have shape (D, H, W), got {masks.shape}."
        )
    if not np.issubdtype(masks.dtype, np.integer):
        raise TypeError("`masks` must contain integer instance labels.")
    if np.any(masks < 0):
        raise ValueError("Instance labels must be non-negative.")
    if sigma <= 0:
        raise ValueError("`sigma` must be greater than zero.")
    if truncate <= 0:
        raise ValueError("`truncate` must be greater than zero.")

    depth, height, width = masks.shape

    heatmap = np.zeros((depth, height, width), dtype=np.float32)
    offsets = np.zeros((3, depth, height, width), dtype=np.float32)
    centroid_map: dict[int, tuple[float, float, float]] = {}

    labels = np.unique(masks)
    labels = labels[labels != 0]

    radius = int(np.ceil(truncate * sigma))
    denominator = 2.0 * sigma**2

    for label in labels:
        # Coordinates have shape (N, 3), ordered as z, y, x.
        coords = np.argwhere(masks == label)

        if coords.size == 0:
            continue

        centroid = coords.mean(axis=0, dtype=np.float64)
        cz, cy, cx = centroid

        label_int = int(label)
        centroid_map[label_int] = (float(cz), float(cy), float(cx))

        # Offset targets only exist on voxels belonging to this instance.
        z = coords[:, 0]
        y = coords[:, 1]
        x = coords[:, 2]

        offsets[0, z, y, x] = cz - z
        offsets[1, z, y, x] = cy - y
        offsets[2, z, y, x] = cx - x

        # Compute the Gaussian only in a local bounding box.
        z0 = max(0, int(np.floor(cz)) - radius)
        z1 = min(depth, int(np.floor(cz)) + radius + 1)

        y0 = max(0, int(np.floor(cy)) - radius)
        y1 = min(height, int(np.floor(cy)) + radius + 1)

        x0 = max(0, int(np.floor(cx)) - radius)
        x1 = min(width, int(np.floor(cx)) + radius + 1)

        zz = np.arange(z0, z1, dtype=np.float32)[:, None, None]
        yy = np.arange(y0, y1, dtype=np.float32)[None, :, None]
        xx = np.arange(x0, x1, dtype=np.float32)[None, None, :]

        squared_distance = (
            (zz - cz) ** 2
            + (yy - cy) ** 2
            + (xx - cx) ** 2
        )

        gaussian = np.exp(
            -squared_distance / denominator
        ).astype(np.float32)

        # Prevent nearby centers from creating heatmap values greater than 1.
        current = heatmap[z0:z1, y0:y1, x0:x1]
        np.maximum(current, gaussian, out=current)

    return heatmap, offsets, centroid_map


def compute_all_targets_old(masks, sigma=3.0):
    """
    Derive training targets from instance masks.

    For each cell k, the 3D centroid is:
        c_k = mean({ q : masks[q] == k })

    For each foreground voxel q belonging to cell k:
        offset  v(q) = c_k - q          (3D vector, raw voxel units)
        heatmap H(q) = Gaussian blob centred at c_k

    Returns
    -------
    heatmap     : (D, H, W) float32  in [0, 1]
    offsets     : (3, D, H, W) float32  [Δz, Δy, Δx] in voxel units
    centroid_map: {label (int): (cz, cy, cx) float}
    """
    D, H, W = masks.shape
    heatmap = np.zeros((D, H, W), dtype=np.float32)
    offsets = np.zeros((3, D, H, W), dtype=np.float32)
    centroid_map = {}

    # Coordinate grids — shape (D, H, W) each
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W].astype(np.float32)

    for label in np.unique(masks):
        if label == 0:
            continue

        fg = masks == label
        coords = np.argwhere(fg).astype(np.float32)   # (N, 3) in (z, y, x) order
        cz, cy, cx = coords.mean(axis=0)
        centroid_map[int(label)] = (cz, cy, cx)

        # Heatmap: 3D Gaussian blob at c_k (max-pool so peaks don't sum)
        _add_gaussian(heatmap, cz, cy, cx, sigma)

        # Offset: vector from each foreground voxel to the centroid
        offsets[0][fg] = cz - zz[fg]   # Δz
        offsets[1][fg] = cy - yy[fg]   # Δy
        offsets[2][fg] = cx - xx[fg]   # Δx

    return heatmap, offsets, centroid_map


def _add_gaussian(volume, cz, cy, cx, sigma):
    """In-place max-pool a 3D Gaussian blob into `volume` at (cz, cy, cx)."""
    D, H, W = volume.shape
    r = int(np.ceil(4.0 * sigma))
    z0, z1 = max(0, int(round(cz)) - r), min(D, int(round(cz)) + r + 1)
    y0, y1 = max(0, int(round(cy)) - r), min(H, int(round(cy)) + r + 1)
    x0, x1 = max(0, int(round(cx)) - r), min(W, int(round(cx)) + r + 1)

    zz = (np.arange(z0, z1) - cz)[:, None, None]
    yy = (np.arange(y0, y1) - cy)[None, :, None]
    xx = (np.arange(x0, x1) - cx)[None, None, :]
    blob = np.exp(-(zz**2 + yy**2 + xx**2) / (2.0 * sigma**2)).astype(np.float32)
    volume[z0:z1, y0:y1, x0:x1] = np.maximum(volume[z0:z1, y0:y1, x0:x1], blob)
