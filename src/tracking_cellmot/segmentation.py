"""Unsupervised segmentation of bright, approximately spherical cells."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.morphology import h_maxima

# A Gaussian of this sigma gives the strongest Laplacian response to a sphere
# of radius r, so it is the scale that best matches cells of a known size.
_BLOB_SIGMA_PER_RADIUS = 1.0 / np.sqrt(3.0)

# Classic difference-of-Gaussians ratio approximating a Laplacian of Gaussian.
_DOG_RATIO = 1.6

# A uniform solid ellipsoid has second moment a^2 / 5 along a semi-axis of
# length a, so this converts a measured variance back into a semi-axis.
_UNIFORM_ELLIPSOID_VARIANCE = 5.0

# Default cell size bounds as multiples of ``cell_diameter``.
_MIN_DIAMETER_FRACTION = 0.6
_MAX_DIAMETER_FRACTION = 1.5


def _blob_sigmas(cell_diameter: float, voxel_size: np.ndarray) -> np.ndarray:
    """Per-axis Gaussian sigma, in voxels, matched to cells of ``cell_diameter``."""
    return (cell_diameter / 2.0) * _BLOB_SIGMA_PER_RADIUS / voxel_size


def _empty_result(shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros(shape, dtype=np.uint32)
    seeds = np.empty((0, 3), dtype=np.intp)
    return labels, seeds


def _prominent_maxima(response: np.ndarray, prominence: float) -> np.ndarray:
    """Locate every maximum of ``response`` that rises ``prominence`` above its surroundings.

    Unlike suppressing everything within a fixed radius of the brightest peak,
    this keeps a dim cell that sits beside a bright one, provided the two are
    separated by a dip. Only the relative depth of that dip matters, so faint
    cells are not penalised for being faint.
    """
    spread = float(np.percentile(response, 99) - np.percentile(response, 50))
    if spread <= 0:
        return np.empty((0, 3), dtype=np.intp)

    plateaus = h_maxima(response, prominence * spread)

    # plt.figure(figsize=(10, 10))
    # plt.imshow(plateaus[18], interpolation="none", cmap="gray")
    # plt.show()

    regions, count = ndimage.label(plateaus, structure=np.ones((3, 3, 3)))

    if count == 0:
        return np.empty((0, 3), dtype=np.intp)

    index = np.arange(1, count + 1)
    centers = np.rint(ndimage.center_of_mass(plateaus, regions, index)).astype(np.intp)
    centers = np.clip(centers, 0, np.array(response.shape) - 1)

    # A concave plateau can have its centroid fall outside itself; those few
    # fall back to the plateau's own brightest voxel. Snapping this way also
    # guarantees distinct seeds, since a voxel belongs to at most one region.
    strayed = regions[tuple(centers.T)] != index
    if strayed.any():
        fallback = ndimage.maximum_position(response, regions, index[strayed])
        centers[strayed] = np.atleast_2d(fallback)
    return centers


def _window(
    shape: tuple[int, ...], center: np.ndarray, half: np.ndarray
) -> tuple[tuple[slice, ...], np.ndarray]:
    """Slices of the box of ``2 * half + 1`` voxels around ``center``, and its origin."""
    start = np.maximum(center - half, 0)
    stop = np.minimum(center + half + 1, shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(start, stop)), start


def _moment_ellipsoid(
    offsets: np.ndarray, weight: np.ndarray, min_radius: float, max_radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Semi-axes and axis directions of the ellipsoid matching a weighted point cloud.

    ``offsets`` is ``(3, n)`` in physical units, relative to the centre the
    moment is taken about.
    """
    covariance = (offsets * weight) @ offsets.T / weight.sum()
    variance, directions = np.linalg.eigh(covariance)
    radii = np.sqrt(np.clip(variance, 0.0, None) * _UNIFORM_ELLIPSOID_VARIANCE)
    return np.clip(radii, min_radius, max_radius), directions


def _inside_ellipsoid(offsets: np.ndarray, axes: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Mask of the ``(3, n)`` offsets lying within the ellipsoid centred on the origin."""
    projected = directions.T @ offsets
    return ((projected / axes[:, None]) ** 2).sum(axis=0) <= 1.0


def _fit_ellipsoids(
    flattened: np.ndarray,
    seeds: np.ndarray,
    voxels: np.ndarray,
    *,
    min_radius: float,
    max_radius: float,
    boundary_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one ellipsoid to the bright voxels surrounding each seed.

    The extent of a cell is read from the second moment of the bright blob the
    seed sits in, which is cheap and orientation-aware. Because that blob may
    also contain a touching neighbour, every semi-axis is clamped to
    ``[min_radius, max_radius]``: a cell can be neither smaller nor larger than
    biology allows, so a merged blob yields a plausible cell rather than a
    region spanning both.

    The ellipsoid is then re-centred on the brightness-weighted centre of mass
    of the voxels it contains and refitted about that centre, which pins it to
    the cell's middle instead of the blob-response maximum that seeded it.

    Returns the ``(n_seeds, 3)`` centres in fractional voxel coordinates, the
    semi-axes in physical units, and the ``(n_seeds, 3, 3)`` rotations whose
    columns are the axis directions.
    """
    connectivity = np.ones((3, 3, 3), dtype=bool)
    half = np.maximum(1, np.ceil(max_radius / voxels).astype(int))

    centers = seeds.astype(np.float64)
    semi_axes = np.full((len(seeds), 3), min_radius, dtype=np.float64)
    rotations = np.tile(np.eye(3), (len(seeds), 1, 1))

    for k, seed in enumerate(seeds):
        window, origin = _window(flattened.shape, seed, half)
        local = flattened[window]
        center = seed - origin

        # Thresholding relative to this cell's own peak keeps dim cells from
        # collapsing and bright ones from swallowing their halo.
        peak = float(local[tuple(center)])
        floor = float(np.percentile(local, 10))
        threshold = floor + boundary_fraction * (peak - floor)

        regions, count = ndimage.label(local >= threshold, structure=connectivity)
        own = int(regions[tuple(center)])
        if count == 0 or own == 0:
            continue

        inside = regions == own
        weight = (local[inside] - threshold).astype(np.float64)
        if weight.sum() <= 0.0:
            continue

        offsets = (np.stack(np.nonzero(inside)).astype(np.float64) - center[:, None]) * voxels[:, None]
        axes, directions = _moment_ellipsoid(offsets, weight, min_radius, max_radius)

        # Only voxels already inside that first ellipsoid get a vote on where
        # the centre belongs, so a touching neighbour cannot drag it out of its
        # own cell.
        voting = _inside_ellipsoid(offsets, axes, directions)
        if voting.any() and weight[voting].sum() > 0.0:
            shift = (offsets[:, voting] * weight[voting]).sum(axis=1) / weight[voting].sum()
            centers[k] = seed + shift / voxels

            # Refit ??
            # axes, directions = _moment_ellipsoid(
            #     offsets[:, voting] - shift[:, None], weight[voting], min_radius, max_radius
            # )

        semi_axes[k] = axes
        rotations[k] = directions

    return centers, semi_axes, rotations


def _paint_ellipsoids(
    shape: tuple[int, ...],
    centers: np.ndarray,
    semi_axes: np.ndarray,
    rotations: np.ndarray,
    voxels: np.ndarray,
    labels_of: np.ndarray,
) -> np.ndarray:
    """Rasterise the fitted ellipsoids into a label volume.

    Where two ellipsoids overlap the voxel goes to whichever centre is closer
    in units of that ellipsoid's own radii, which splits neighbours along the
    surface halfway between them and leaves every instance convex. Ellipsoids
    sharing an entry in ``labels_of`` are one cell, so their union is painted
    with that single label.
    """
    labels = np.zeros(shape, dtype=np.uint32)
    closest = np.full(shape, np.inf, dtype=np.float32)

    for label, center, axes, directions in zip(labels_of, centers, semi_axes, rotations):
        # Quadratic form of the ellipsoid: offset @ precision @ offset <= 1.
        precision = (directions / axes**2) @ directions.T
        # One voxel of slack covers the offset between the fractional centre and
        # the voxel the window is anchored on.
        extent = np.sqrt(((directions * axes) ** 2).sum(axis=1))
        half = np.maximum(1, np.ceil(extent / voxels).astype(int)) + 1
        window, _ = _window(shape, np.rint(center).astype(np.intp), half)

        offsets = np.stack(
            np.meshgrid(
                *[
                    (np.arange(s.start, s.stop) - c) * v
                    for s, c, v in zip(window, center, voxels)
                ],
                indexing="ij",
            )
        )
        distance = np.einsum("i...,ij,j...->...", offsets, precision, offsets)

        window_closest = closest[window]
        window_labels = labels[window]
        nearer = (distance <= 1.0) & (distance < window_closest)
        window_closest[nearer] = distance[nearer]
        window_labels[nearer] = label

    return labels


def _merge_groups(centers: np.ndarray, voxels: np.ndarray, min_distance: float) -> np.ndarray:
    """Group the cells whose centres lie within ``min_distance`` of each other.

    ``centers`` is in voxel coordinates but the threshold is physical, so a
    coarse Z sampling cannot make two cells look further apart than they are.
    Grouping is transitive: a chain of cells each within ``min_distance`` of the
    next collapses into one.
    """
    count = len(centers)
    if count == 0 or min_distance <= 0.0:
        return np.arange(count)

    pairs = np.asarray(
        list(cKDTree(centers * voxels).query_pairs(min_distance)), dtype=np.intp
    ).reshape(-1, 2)
    adjacency = coo_matrix(
        (np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])),
        shape=(count, count),
    )
    return connected_components(adjacency, directed=False)[1]


def _group_means(values: np.ndarray, group: np.ndarray, count: int) -> np.ndarray:
    """Mean of the ``values`` rows belonging to each of ``count`` groups."""
    sizes = np.bincount(group, minlength=count)
    sums = np.stack(
        [
            np.bincount(group, weights=values[:, axis], minlength=count)
            for axis in range(values.shape[1])
        ],
        axis=1,
    )
    return sums / sizes[:, None]


def _distinct_centers(
    shape: tuple[int, ...], centers: np.ndarray, group: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Round group centres to voxels, collapsing groups that land on the same one.

    Returns the distinct centre voxels and the group index of every cell,
    renumbered to index them.
    """
    voxel_centers = np.clip(np.rint(centers), 0, np.asarray(shape) - 1).astype(np.intp)
    distinct, inverse = np.unique(voxel_centers, axis=0, return_inverse=True)
    return distinct, inverse.ravel()[group]


def _relabel_and_filter(labels: np.ndarray, seeds: np.ndarray, min_volume: int) -> tuple[np.ndarray, np.ndarray]:
    """Drop instances below ``min_volume``, renumber from 1, and realign seeds."""
    sizes = np.bincount(labels.ravel(), minlength=len(seeds) + 1)
    keep = sizes >= min_volume
    keep[0] = False
    if not keep.any():
        return _empty_result(labels.shape)

    lookup = np.zeros(len(sizes), dtype=np.uint32)
    lookup[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.uint32)

    # ``lookup`` numbers survivors in ascending order of their old label, so
    # indexing the seeds the same way keeps row k paired with label k + 1.
    surviving = np.flatnonzero(keep)
    return lookup[labels], seeds[surviving - 1]


def cell_centers(labels: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Center of mass of every cell in an instance-label volume.

    Parameters
    ----------
    labels
        Three-dimensional integer array ordered as ``(z, y, x)``. Zero is
        background and every other value identifies one cell. The values need
        not be contiguous or start at one, so a volume straight out of any
        segmenter works as is.
    weights
        Optional array shaped like ``labels`` weighting each voxel, typically
        the raw image. The default weights every voxel of a cell equally, which
        gives the centroid of its mask.

    Returns
    -------
    np.ndarray
        ``(n_cells, 3)`` float array of ``(z, y, x)`` centers in voxel
        coordinates, ordered by ascending label value. Centers are fractional,
        so a cell straddling two voxels sits between them.
    """
    if labels.ndim != 3:
        raise ValueError(f"labels must have shape (z, y, x), got {labels.shape}")
    if weights is not None and weights.shape != labels.shape:
        raise ValueError("weights must have the same shape as labels")

    ids = np.unique(labels)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return np.empty((0, 3), dtype=float)

    if weights is None:
        weights = labels > 0
    return np.asarray(ndimage.center_of_mass(weights, labels, ids), dtype=float)


def segment_cells_3d(
    image: np.ndarray,
    *,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cell_diameter: float = 6.0,
    background_diameter: float | None = None,
    seed_prominence: float | None = 0.05,
    boundary_fraction: float = 0.5,
    min_cell_diameter: float | None = None,
    max_cell_diameter: float | None = None,
    min_dist_between_cell: float = 0.0,
    min_cell_volume: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment bright, approximately spherical cells in a 3D image.

    The method needs no training data and no annotations:

    1. Two Gaussian blurs flatten the image, removing out-of-focus haze and
       uneven illumination so that only cell-scale structure survives. This
       also makes the result insensitive to a non-zero background level.
    2. A Laplacian of Gaussian tuned to ``cell_diameter`` responds to roughly
       spherical bright blobs.
    3. Otsu's method on the flattened image separates cells from background.
    4. Prominent maxima of the blob response become one seed per cell.
    5. One ellipsoid is fitted per seed from the second moment of the bright
       voxels around it, with every semi-axis clamped between
       ``min_cell_diameter`` and ``max_cell_diameter``. Each is then re-centred
       on the centre of mass of the voxels it contains, and that centre — not
       the maximum that seeded it — is the reported cell centre.
    6. The ellipsoids are rasterised, overlaps going to the nearer centre in
       units of its own radii.
    7. Cells left closer together than ``min_dist_between_cell`` are merged, on
       the grounds that two centres that close describe one cell twice.

    Fitting bounded ellipsoids rather than growing a watershed keeps each
    instance convex and plausibly cell-sized. A watershed has no notion of how
    large a cell may be, so it fragments into slivers where the intensity is
    noisy and produces regions spanning several cells where they touch.

    All lengths are expressed in the physical units of ``voxel_size``, so the
    defaults assume voxel units unless a real voxel size is supplied. Passing
    the true anisotropic voxel size matters: it keeps cells spherical in
    physical space even when the Z sampling is much coarser than XY.

    Parameters
    ----------
    image
        Three-dimensional ``uint16`` array ordered as ``(z, y, x)``.
    voxel_size
        Physical size of one voxel as ``(z, y, x)``.
    cell_diameter
        Approximate cell diameter in physical units. This is the main tuning
        knob: it sets the blob-detection scale and the minimum seed spacing.
    background_diameter
        Scale of the illumination background to remove, in physical units.
        Defaults to ``6 * cell_diameter``.
    seed_prominence
        How far a maximum must rise above its surroundings to count as a
        separate cell, as a fraction of the blob response's spread. Raise it to
        merge cells that are split in two, lower it to recover dim cells beside
        bright ones. Set to ``None`` to instead suppress every maximum within
        half a cell of a brighter one, which is faster but loses dim cells.
    boundary_fraction
        Where a cell ends, as a fraction of the way from the local background up
        to its own peak brightness. The default of ``0.5`` measures each cell at
        full width at half maximum, which matches how ``cell_diameter`` is
        normally measured. Lower values fit larger ellipsoids.
    min_cell_diameter, max_cell_diameter
        Smallest and largest diameter a cell may have, in physical units. Every
        fitted semi-axis is clamped to this range, so a blob of two touching
        cells cannot become one oversized instance. Default to
        ``0.6 * cell_diameter`` and ``1.5 * cell_diameter``.
    min_dist_between_cell
        Cells whose centres end up closer than this, in physical units, are
        merged into one: the surviving centre is the average of theirs and the
        mask is the union of their ellipsoids. Merging is transitive, so a chain
        of close centres collapses into a single cell. The default of ``0``
        merges nothing.
    min_cell_volume
        Instances smaller than this voxel count are discarded. Defaults to
        5 percent of the voxel volume of a sphere of ``cell_diameter``.

    Returns
    -------
    labels
        ``uint32`` volume shaped like ``image``
    seeds
        ``(n_cells, 3)`` integer array of cell centers as ``(z, y, x)`` voxel
        indices, each the center of that cell's fitted ellipsoid. Row ``k`` is
        the center of label ``k + 1``, so ``labels[tuple(seeds[k])] == k + 1``.
        and ``seeds`` (the ``(z, y, x)`` voxel index of each ellipsoid center).

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ``uint16`` array.
    ValueError
        If the input or a numeric parameter is invalid.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint16:
        raise TypeError("image must be a NumPy array with dtype uint16")
    if image.ndim != 3:
        raise ValueError(f"image must have shape (z, y, x), got {image.shape}")
    if image.size == 0:
        raise ValueError("image must not be empty")

    voxels = np.asarray(voxel_size, dtype=np.float64)
    if voxels.shape != (3,) or np.any(voxels <= 0):
        raise ValueError("voxel_size must be three positive numbers ordered as (z, y, x)")
    if cell_diameter <= 0:
        raise ValueError("cell_diameter must be positive")

    if background_diameter is None:
        background_diameter = 6.0 * cell_diameter
    elif background_diameter <= cell_diameter:
        raise ValueError("background_diameter must be larger than cell_diameter")
    if not 0.0 <= boundary_fraction < 1.0:
        raise ValueError("boundary_fraction must be in [0, 1)")
    if seed_prominence is not None and not 0.0 < seed_prominence < 1.0:
        raise ValueError("seed_prominence must be in (0, 1), or None")

    if min_cell_diameter is None:
        min_cell_diameter = _MIN_DIAMETER_FRACTION * cell_diameter
    if max_cell_diameter is None:
        max_cell_diameter = _MAX_DIAMETER_FRACTION * cell_diameter
    if min_cell_diameter <= 0:
        raise ValueError("min_cell_diameter must be positive")
    if max_cell_diameter < min_cell_diameter:
        raise ValueError("max_cell_diameter must be at least min_cell_diameter")
    if min_dist_between_cell < 0:
        raise ValueError("min_dist_between_cell must not be negative")

    if min_cell_volume is None:
        sphere_volume = (4.0 / 3.0) * np.pi * (cell_diameter / 2.0) ** 3
        min_cell_volume = max(1, int(0.05 * sphere_volume / np.prod(voxels)))
    elif min_cell_volume < 1:
        raise ValueError("min_cell_volume must be at least 1")

    if image.max() == image.min():
        return _empty_result(image.shape)

    intensity = image.astype(np.float32)
    sigma_cell = _blob_sigmas(cell_diameter, voxels)
    sigma_background = (background_diameter / 2.0) / voxels
    
    # Scale-normalised so the response is comparable across cell sizes.
    blob_response = -ndimage.gaussian_laplace(intensity, sigma_cell) * (cell_diameter / 2.0) ** 2
    
    # measured in voxels per axis so anisotropic sampling is respected.
    flattened = ndimage.gaussian_filter(intensity, sigma_cell) - ndimage.gaussian_filter(
        intensity, sigma_background
    )

    if seed_prominence is not None:
        seeds = _prominent_maxima(blob_response, seed_prominence)
    else:
        mask = flattened > threshold_otsu(flattened) / 4
        mask = ndimage.binary_opening(mask, ndimage.generate_binary_structure(rank=3, connectivity=1))
        if not mask.any():
            return _empty_result(image.shape)

        separation = np.maximum(1, np.rint(0.55 * cell_diameter / voxels).astype(int))
        footprint = np.ones(np.minimum(2 * separation + 1, image.shape))
        seeds = peak_local_max(blob_response, footprint=footprint, labels=mask, exclude_border=False)

    if len(seeds) == 0:
        return _empty_result(image.shape)

    # seeds_plot = seeds[np.abs(seeds[:, 0] - 18) <= 3]
    # plt.figure(figsize=(15, 10))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image[18], cmap="gray")
    # plt.axis("off")
    # plt.scatter(
    #     seeds_plot[:, 2],
    #     seeds_plot[:, 1],
    #     s=30,
    #     c='dodgerblue',
    #     edgecolors="white",
    #     linewidths=0.5,
    # )
    # plt.show()

    centers, semi_axes, rotations = _fit_ellipsoids(
        flattened,
        seeds,
        voxels,
        min_radius=min_cell_diameter / 2.0,
        max_radius=max_cell_diameter / 2.0,
        boundary_fraction=boundary_fraction,
    )

    # Cells nearer than min_dist_between_cell describe one cell twice, so they
    # share a label — making their mask the union of their ellipsoids — and are
    # reported at their average centre. Centres rounding onto the same voxel are
    # always merged, whatever the threshold.
    group = _merge_groups(centers, voxels, min_dist_between_cell)
    merged = _group_means(centers, group, int(group.max()) + 1)
    seeds, group = _distinct_centers(image.shape, merged, group)

    labels = _paint_ellipsoids(
        image.shape, centers, semi_axes, rotations, voxels, group + 1
    )
    
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(30, 10))
    # plt.subplot(1, 3, 1)
    # plt.imshow(labels[9])
    # plt.axis(False)
    # plt.subplot(1, 3, 2)
    # plt.imshow(labels[10])
    # plt.axis(False)
    # plt.subplot(1, 3, 3)
    # plt.imshow(labels[11])
    # plt.axis(False)
    # plt.show()

    # A cell always keeps its own centre, so that every surviving label has a
    # seed and the seed/label pairing below stays exact.
    labels[tuple(seeds.T)] = np.arange(1, len(seeds) + 1, dtype=np.uint32)

    labels, seeds = _relabel_and_filter(labels, seeds, min_cell_volume)
    return labels, seeds
