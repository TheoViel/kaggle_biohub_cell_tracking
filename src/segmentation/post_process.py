import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    maximum_filter,
    label as scipy_label,
)
from scipy.spatial import cKDTree


def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))


def postprocess(
    preds,
    fg_thresh=0.5,
    vote_sigma=2.0,
    nms_size=7,
    vote_thresh_abs=1.0,
    vote_thresh_rel=0.05,
    min_support=10,
    max_vote_distance=None,
    fg_prob=None,
):
    """
    Convert foreground and centroid-offset predictions into 3D instances.

    Parameters
    ----------
    preds
        Dictionary containing:

        fg_logits:
            Array of shape (D, H, W).

        offsets:
            Array of shape (3, D, H, W), ordered as
            [delta_z, delta_y, delta_x].

    fg_thresh
        Foreground probability threshold.

    vote_sigma
        Gaussian smoothing sigma for the centroid vote map.
        May be a scalar or a (z, y, x) tuple.

    nms_size
        Local-maximum filter size.
        May be an integer or a (z, y, x) tuple.

    vote_thresh_abs
        Absolute minimum smoothed vote-map value for a peak.

    vote_thresh_rel
        Relative peak threshold with respect to the global maximum.

    min_support
        Minimum number of foreground voxels whose predicted centers are
        assigned to a detected centroid.

    max_vote_distance
        Optional maximum distance between a voxel's predicted center and
        its assigned detected centroid. Votes farther away are rejected.

    Returns
    -------
    instance_masks
        Integer label image of shape (D, H, W).

    centroids
        Detected centroid coordinates with shape (N, 3), in z-y-x order.
    """
    fg_logits = np.asarray(preds["fg_logits"])
    offsets = np.asarray(preds["offsets"])

    if fg_logits.ndim != 3:
        raise ValueError(
            f"fg_logits must have shape (D, H, W), got {fg_logits.shape}"
        )

    if offsets.shape != (3, *fg_logits.shape):
        raise ValueError(
            "offsets must have shape "
            f"(3, D, H, W), got {offsets.shape}"
        )

    if fg_prob is None:
        fg_prob = sigmoid(fg_logits)

    fg_mask = fg_prob >= fg_thresh

    depth, height, width = fg_mask.shape

    empty_mask = np.zeros(fg_mask.shape, dtype=np.int32)
    empty_centroids = np.empty((0, 3), dtype=np.float32)

    if not np.any(fg_mask):
        return empty_mask, empty_centroids

    # ------------------------------------------------------------
    # 1. Each foreground voxel predicts its own cell centroid.
    # ------------------------------------------------------------
    fg_coords = np.argwhere(fg_mask).astype(np.float32)
    voxel_offsets = offsets[:, fg_mask].T.astype(np.float32)

    predicted_centers = fg_coords + voxel_offsets

    predicted_centers[:, 0] = np.clip(
        predicted_centers[:, 0], 0, depth - 1
    )
    predicted_centers[:, 1] = np.clip(
        predicted_centers[:, 1], 0, height - 1
    )
    predicted_centers[:, 2] = np.clip(
        predicted_centers[:, 2], 0, width - 1
    )

    # ------------------------------------------------------------
    # 2. Accumulate votes.
    #
    # This uses nearest-voxel binning. Trilinear splatting can improve
    # subvoxel precision, but this is a reasonable baseline.
    # ------------------------------------------------------------
    vote_indices = np.rint(predicted_centers).astype(np.int32)

    vote_map_raw = np.zeros(fg_mask.shape, dtype=np.float32)

    np.add.at(
        vote_map_raw,
        (
            vote_indices[:, 0],
            vote_indices[:, 1],
            vote_indices[:, 2],
        ),
        1.0,
    )

    vote_map = gaussian_filter(vote_map_raw, sigma=vote_sigma)

    global_peak = float(vote_map.max())

    if global_peak <= 0:
        return empty_mask, empty_centroids

    peak_threshold = max(
        float(vote_thresh_abs),
        float(vote_thresh_rel) * global_peak,
    )

    # ------------------------------------------------------------
    # 3. Detect local maxima.
    #
    # Do not require maxima to lie in thresholded foreground. A centroid
    # may fall into a small foreground-probability hole.
    # ------------------------------------------------------------
    max_map = maximum_filter(
        vote_map,
        size=nms_size,
        mode="constant",
    )

    local_maxima = (
        np.isclose(vote_map, max_map)
        & (vote_map >= peak_threshold)
    )

    peak_components, number_of_peaks = scipy_label(local_maxima)

    if number_of_peaks == 0:
        return empty_mask, empty_centroids

    # Compute one weighted coordinate per connected peak plateau.
    candidate_centroids = []

    for peak_id in range(1, number_of_peaks + 1):
        peak_mask = peak_components == peak_id
        coordinates = np.argwhere(peak_mask)
        weights = vote_map[peak_mask]

        if coordinates.size == 0:
            continue

        centroid = np.average(
            coordinates,
            axis=0,
            weights=weights,
        )

        candidate_centroids.append(centroid)

    if not candidate_centroids:
        return empty_mask, empty_centroids

    candidate_centroids = np.asarray(
        candidate_centroids,
        dtype=np.float32,
    )

    # ------------------------------------------------------------
    # 4. Assign each voxel's predicted center to the nearest candidate.
    #
    # This is the critical distinction:
    # use q + offset(q), not q itself.
    # ------------------------------------------------------------
    center_tree = cKDTree(candidate_centroids)

    vote_distances, candidate_assignments = center_tree.query(
        predicted_centers,
        k=1,
    )

    valid_votes = np.ones(len(predicted_centers), dtype=bool)

    if max_vote_distance is not None:
        valid_votes &= vote_distances <= max_vote_distance

    support_counts = np.bincount(
        candidate_assignments[valid_votes],
        minlength=len(candidate_centroids),
    )

    keep_candidates = support_counts >= min_support

    if not np.any(keep_candidates):
        return empty_mask, empty_centroids

    centroids = candidate_centroids[keep_candidates]

    # ------------------------------------------------------------
    # 5. Reassign votes after removing unsupported candidate centers.
    # ------------------------------------------------------------
    final_tree = cKDTree(centroids)

    final_distances, assignments = final_tree.query(
        predicted_centers,
        k=1,
    )

    valid_assignments = np.ones(len(predicted_centers), dtype=bool)

    if max_vote_distance is not None:
        valid_assignments &= final_distances <= max_vote_distance

    # ------------------------------------------------------------
    # 6. Fill the output label image.
    # ------------------------------------------------------------
    instance_masks = np.zeros(
        fg_mask.shape,
        dtype=np.int32,
    )

    integer_fg_coords = fg_coords.astype(np.intp)

    valid_coords = integer_fg_coords[valid_assignments]
    valid_labels = assignments[valid_assignments].astype(np.int32) + 1

    instance_masks[
        valid_coords[:, 0],
        valid_coords[:, 1],
        valid_coords[:, 2],
    ] = valid_labels

    return instance_masks, centroids