"""Confidence-weighted centroid merging: a free replacement for greedy NMS.

Greedy NMS keeps the strongest detection in each suppression ball and discards
the rest.  But the discarded ones are not noise -- they are the same cell seen
from a neighbouring voxel, and their positions carry information the winner
does not have.  Averaging the ball instead of picking its maximum costs
nothing, produces *exactly the same number of detections*, and lands closer to
the true centre.

Measured on a 3D cell detector's output, 10 movies per embryo, matching
ground truth by optimal bipartite assignment within 7 um:

    embryo  radius   greedy NMS   centroid merge   nodes/frame
    44b6      6 um      0.9846        0.9923        449.6 (same)
    44b6      7 um      0.9421        0.9768        404.0 (same)
    6bba      6 um      0.8831        0.8861        209.8 (same)
    6bba      7 um      0.8731        0.8861        190.5 (same)

The gain grows with the radius, which is the tell: the bigger the ball, the
more information winner-takes-all throws away.

WHY GREEDY AND NOT HIERARCHICAL CLUSTERING.  The obvious implementation is
single-linkage clustering at the merge radius.  Do not do that.  Single linkage
*chains*: in a dense region A-B-C-D where each consecutive pair is within the
radius but A and D are 20 um apart, the whole run collapses to one centroid far
from every member.  Measured on the denser of the two embryos, that dropped the
fraction of ground-truth cells with a detection within 7 um from 98.8% to
92.2%.  The greedy form below bounds every cluster to one radius around a real,
high-confidence seed, which recovered that to 96.1%.

CHOOSING THE RADIUS.  Merging is not free above a certain radius: it destroys
genuinely distinct cells.  In the denser embryo (369 cells/frame, nearest
neighbour 7.9 um) node recall fell 3.6 points at 7 um; in the sparser one
(165 cells/frame, 8.9 um) it fell 0.6.  Set the radius from the measured local
density -- which needs no labels and so works at test time -- rather than
fixing it globally.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def centroid_merge(
    points: np.ndarray,
    scores: np.ndarray,
    radius: float,
    scale: np.ndarray | tuple[float, ...] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Merge detections within *radius* into confidence-weighted centroids.

    A drop-in replacement for greedy NMS: same traversal, same clusters, same
    number of survivors -- it just returns the weighted mean of each cluster
    rather than its strongest member.

    Args:
        points: ``(N, D)`` coordinates, in whatever units you index the image.
        scores: ``(N,)`` detection confidences.
        radius: merge distance, in *physical* units after applying *scale*.
        scale: per-axis size of one coordinate unit, so anisotropic voxels are
            merged in a sphere rather than an ellipsoid.  For 1.625 x 0.40625 x
            0.40625 um voxels, pass that tuple and give *radius* in microns.

    Returns:
        ``(merged_points, merged_scores)``, strongest cluster first.
    """

    points = np.asarray(points, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(points) < 2:
        return points, scores

    physical = points * np.asarray(scale, dtype=np.float64)
    tree = cKDTree(physical)
    claimed = np.zeros(len(points), dtype=bool)

    merged_points, merged_scores = [], []
    for seed in np.argsort(-scores):  # strongest first
        if claimed[seed]:
            continue
        # Every member is within one radius of this seed, so the centroid
        # cannot drift further than that -- the property single linkage loses.
        members = [j for j in tree.query_ball_point(physical[seed], radius)
                   if not claimed[j]]
        weights = scores[members]
        merged_points.append((points[members] * weights[:, None]).sum(0) / weights.sum())
        merged_scores.append(weights.max())
        claimed[members] = True

    return np.asarray(merged_points), np.asarray(merged_scores)


if __name__ == "__main__":
    # Two detections 4 um apart on the same cell: NMS keeps one, this returns
    # the point between them, weighted toward the more confident.
    scale = (1.625, 0.40625, 0.40625)
    points = np.array([[10.0, 100.0, 100.0], [10.0, 110.0, 100.0]])
    scores = np.array([0.9, 0.3])
    merged, kept = centroid_merge(points, scores, radius=7.0, scale=scale)
    print("in :", points.tolist(), scores.tolist())
    print("out:", merged.round(2).tolist(), kept.tolist())
