import numpy as np
from cellpose import metrics
from utils.metrics import dist


def custom_iou(masks_true, masks_pred):
    """Calculate the intersection over union of all mask pairs.

    Parameters:
        masks_true (np.ndarray, int): Ground truth masks, where 0=NO masks; 1,2... are mask labels.
        masks_pred (np.ndarray, int): Predicted masks, where 0=NO masks; 1,2... are mask labels.

    Returns:
        iou (np.ndarray, float): Matrix of IOU pairs of size [x.max()+1, y.max()+1].

    How it works:
        The overlap matrix is a lookup table of the area of intersection
        between each set of labels (true and predicted). The true labels
        are taken to be along axis 0, and the predicted labels are taken
        to be along axis 1. The sum of the overlaps along axis 0 is thus
        an array giving the total overlap of the true labels with each of
        the predicted labels, and likewise the sum over axis 1 is the
        total overlap of the predicted labels with each of the true labels.
        Because the label 0 (background) is included, this sum is guaranteed
        to reconstruct the total area of each label. Adding this row and
        column vectors gives a 2D array with the areas of every label pair
        added together. This is equivalent to the union of the label areas
        except for the duplicated overlap area, so the overlap matrix is
        subtracted to find the union matrix.
    """
    if masks_true.size != masks_pred.size:
        raise ValueError("masks_true.size != masks_pred.size")
    overlap = metrics._label_overlap(masks_true, masks_pred)
    overlap = overlap.toarray()
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)

    # iou
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)


    # overlap
    n_pixels_pred = n_pixels_pred.repeat(overlap.shape[0], axis=0)
    n_pixels_true = n_pixels_true.repeat(overlap.shape[1], axis=1)

    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    # iou = overlap / np.min(np.stack([n_pixels_pred, n_pixels_true], axis=0), axis=0)

    iou[np.isnan(iou)] = 0.0
    return iou


def stitch3D(masks, stitch_threshold=0.25):
    """
    Stitch 2D masks into a 3D volume using a stitch_threshold on IOU.

    Args:
        masks (list or ndarray): List of 2D masks.
        stitch_threshold (float, optional): Threshold value for stitching. Defaults to 0.25.

    Returns:
        list: List of stitched 3D masks.
    """
    masks = masks.astype("uint32")

    mmax = masks[0].max()
    empty = 0
    for i in range(len(masks) - 1):

        iou = custom_iou(masks[i + 1], masks[i])[1:, 1:]
        if not iou.size and empty == 0:
            masks[i + 1] = masks[i + 1]
            mmax = masks[i + 1].max()
        elif not iou.size and not empty == 0:
            icount = masks[i + 1].max()
            istitch = np.arange(mmax + 1, mmax + icount + 1, 1, masks.dtype)
            mmax += icount
            istitch = np.append(np.array(0), istitch)
            masks[i + 1] = istitch[masks[i + 1]]
        else:
            iou[iou < stitch_threshold] = 0.0
            iou[iou < iou.max(axis=0)] = 0.0
            istitch = iou.argmax(axis=1) + 1
            ino = np.nonzero(iou.max(axis=1) == 0.0)[0]
            istitch[ino] = np.arange(mmax + 1, mmax + len(ino) + 1, 1, masks.dtype)
            mmax += len(ino)
            istitch = np.append(np.array(0), istitch)
            masks[i + 1] = istitch[masks[i + 1]]
            empty = 1

    return masks


def merge_close_cells(masks, centroids, max_dist, scale=(1.625, 0.40625, 0.40625)):
    """
    Merge cells whose centroids lie within max_dist of each other.

    Stitching by IoU only links cells that overlap in xy, so one cell split
    across the stack survives as several labels whose centroids sit on top of
    each other. Merging is transitive: if a is close to b and b to c, all three
    become one cell, even when a and c are further apart than max_dist.

    Args:
        masks (ndarray): 3D label volume, 0 is background.
        centroids (ndarray): (n_cells, 3) centroids as (z, y, x), ordered by
            ascending label value, as returned by ndimage.center_of_mass over
            np.unique(masks)[1:].
        max_dist (float): Centroids closer than this, in microns, are one cell.
        scale (tuple, optional): Physical voxel size as (z, y, x).

    Returns:
        tuple: The relabelled volume, numbered from 1, and the (n_merged, 3)
            centroids of the merged cells.
    """
    centroids = np.asarray(centroids, dtype=float)
    ids = np.unique(masks)
    ids = ids[ids > 0]
    if len(ids) != len(centroids):
        raise ValueError(f"got {len(centroids)} centroids for {len(ids)} cells")
    if len(ids) == 0:
        return np.zeros_like(masks), np.empty((0, 3))

    # Union-find over every pair of centroids closer than max_dist, so that
    # chains of near-duplicate fragments collapse into a single cell.
    parent = np.arange(len(ids))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(ids) - 1):
        # Transposing lets dist broadcast cell i against every later cell.
        close = np.flatnonzero(
            (dist(centroids[i], centroids[i + 1:].T, scale=scale) <= max_dist) &
            (centroids[i][0] != centroids[i + 1:, 0])
        )
        for j in close + i + 1:
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[max(root_i, root_j)] = min(root_i, root_j)

    _, group = np.unique([find(i) for i in range(len(ids))], return_inverse=True)

    return _apply_groups(masks, centroids, ids, group.ravel())


def _apply_groups(masks, centroids, ids, group):
    """
    Relabel masks so every cell of a group shares one label, and merge centroids.

    Args:
        masks (ndarray): 3D label volume, 0 is background.
        centroids (ndarray): (n_cells, 3) centroids as (z, y, x).
        ids (ndarray): Label values, ascending, aligned with centroids.
        group (ndarray): Group index of every cell, numbered from 0.

    Returns:
        tuple: The relabelled volume, numbered from 1, and the (n_merged, 3)
            centroids of the merged cells.
    """
    lookup = np.zeros(int(ids.max()) + 1, dtype=masks.dtype)
    lookup[ids] = group + 1
    merged = lookup[masks]

    # The centroid of a merged cell is the centre of mass of the union, so the
    # fragments are weighted by how many voxels each contributes.
    sizes = np.bincount(masks.ravel())[ids].astype(float)
    n_groups = int(group.max()) + 1
    weights = np.bincount(group, weights=sizes, minlength=n_groups)
    merged_centroids = np.stack(
        [
            np.bincount(group, weights=centroids[:, axis] * sizes, minlength=n_groups)
            for axis in range(3)
        ],
        axis=1,
    ) / weights[:, None]

    return merged, merged_centroids


def _z_occupancy(masks, ids):
    """
    Return a (n_cells, n_z) boolean of which z planes each label appears on.

    Args:
        masks (ndarray): 3D label volume, 0 is background.
        ids (ndarray): Label values, ascending.

    Returns:
        ndarray: Row k is True on every plane where ids[k] has pixels.
    """
    row_of = np.zeros(int(ids.max()) + 1, dtype=np.intp)
    row_of[ids] = np.arange(len(ids))

    occupancy = np.zeros((len(ids), masks.shape[0]), dtype=bool)
    for z, plane in enumerate(masks):
        present = np.unique(plane)
        occupancy[row_of[present[present > 0]], z] = True
    return occupancy


def merge_close_cells_v2(masks, centroids, max_dist, scale=(1.625, 0.40625, 0.40625)):
    """
    Merge cells whose centroids are close, but never two cells sharing a z plane.

    Same idea as merge_close_cells, with one extra constraint: if two labels both
    have pixels on some z plane, the 2D model looked at that plane and decided
    they were separate instances, so they are never merged. Comparing centroid z
    is not enough to catch this — a cell stitched across several planes has a
    fractional, averaged centroid z, so two cells that overlap on a plane can
    still have different centroid z values.

    The constraint holds for whole groups, not just pairs: a group may only
    absorb a cell that is disjoint in z from every one of its members, so no
    chain of merges can smuggle two same-plane cells into one label. Because the
    result then depends on the order merges happen in, pairs are merged closest
    first.

    Args:
        masks (ndarray): 3D label volume, 0 is background.
        centroids (ndarray): (n_cells, 3) centroids as (z, y, x), ordered by
            ascending label value, as returned by ndimage.center_of_mass over
            np.unique(masks)[1:].
        max_dist (float): Centroids closer than this, in microns, are one cell.
        scale (tuple, optional): Physical voxel size as (z, y, x).

    Returns:
        tuple: The relabelled volume, numbered from 1, and the (n_merged, 3)
            centroids of the merged cells.
    """
    centroids = np.asarray(centroids, dtype=float)
    ids = np.unique(masks)
    ids = ids[ids > 0]
    if len(ids) != len(centroids):
        raise ValueError(f"got {len(centroids)} centroids for {len(ids)} cells")
    if len(ids) == 0:
        return np.zeros_like(masks), np.empty((0, 3))

    # Every candidate pair, closest first: the tightest pair has the best claim
    # to being one cell, and gets to merge before the constraint blocks it.
    candidates = []
    for i in range(len(ids) - 1):
        # Transposing lets dist broadcast cell i against every later cell.
        distances = dist(centroids[i], centroids[i + 1:].T, scale=scale)
        for j in np.flatnonzero(distances <= max_dist):
            candidates.append((distances[j], i, i + 1 + j))
    candidates.sort()

    parent = np.arange(len(ids))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Planes covered by each group, kept on its root so a merge only has to
    # check the two roots rather than every member.
    covers = _z_occupancy(masks, ids)

    for _, i, j in candidates:
        root_i, root_j = find(i), find(j)
        if root_i == root_j:
            continue
        if (covers[root_i] & covers[root_j]).any():
            continue  # they meet on a plane, where the 2D model split them
        root, absorbed = min(root_i, root_j), max(root_i, root_j)
        parent[absorbed] = root
        covers[root] |= covers[absorbed]

    _, group = np.unique([find(i) for i in range(len(ids))], return_inverse=True)

    return _apply_groups(masks, centroids, ids, group.ravel())
