"""Post-processing: peak detection and tracking graph construction."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import tracksdata as td

from tracking.weak_positives import greedy_nms


def pool_kernel_from_um(
    um: float,
    voxel_size: tuple[float, ...],
) -> tuple[int, ...]:
    """Convert a physical suppression distance (microns) to a per-axis voxel kernel.

    Symmetric padding needs an odd kernel, so each axis gets the odd voxel
    count whose physical width ``k * voxel_size_axis`` is *nearest* to ``um``.

    Rounding up to the next odd instead — which is what taking round() and
    adding one to even results does — costs up to a factor of two on a coarse
    grid. At a 3.25 um grid a 5 um request is 1.54 voxels: rounding up gives a
    9.75 um kernel, which suppresses every cell pair closer than that.

    Ties go to the narrower kernel. The two failure modes are not symmetric —
    an over-wide kernel merges distinct cells into one peak and the extra cells
    cannot be recovered, while a narrow one only leaves duplicate peaks for the
    matching step to sort out.
    """
    kernel = []
    for size in voxel_size:
        # k = 2n + 1 voxels; pick the n nearest to (um / size - 1) / 2.
        half = (um / size - 1.0) / 2.0
        n = math.ceil(half - 0.5)          # ceil(x - 0.5) rounds ties down
        kernel.append(2 * max(0, n) + 1)
    return tuple(kernel)


def detect_peaks(
    probas: torch.Tensor,              # (Z, Y, X) raw detection logits
    pool_kernel_um: float = 3.0,
    voxel_size: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    threshold: float = 0.5,
    pool_kernel: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Detect local-maximum peaks in a detection logit volume.

    Uses max-pool NMS with a kernel sized in physical units, then keeps peaks
    whose probability exceeds ``threshold``.

    Returns
    -------
    torch.Tensor  shape (N, 3) int64 — detected voxel coordinates (z, y, x)
                  in the same (downsampled) grid as the input.
    """
    if pool_kernel is None:
        pool_kernel = pool_kernel_from_um(pool_kernel_um, voxel_size)
    pad = tuple(k // 2 for k in pool_kernel)

    probas_ = probas[None, None]   # (1, 1, Z, Y, X)
    pooled = F.max_pool3d(probas_, pool_kernel, stride=1, padding=pad)
    is_peak = (probas_ == pooled) & (probas_ > threshold)
    return torch.nonzero(is_peak[0, 0])   # (N, 3)


def detect_peaks_greedy(
    probas: torch.Tensor,              # (Z, Y, X) raw detection probabilities
    voxel_size: tuple[float, float, float],
    radius_um: float = 2.5,
    threshold: float = 0.5,
    max_candidates: int = 50_000,
) -> torch.Tensor:
    """Detect peaks by greedy NMS over the above-threshold voxels.

    Alternative to the max-pool path, which has two grid artefacts this avoids:

    * The max-pool kernel is an odd voxel count, so on a coarse grid the
      achievable suppression widths are far apart — at a 3.25 um grid only
      3.25, 9.75, 16.25 um exist, and a 5 um request cannot be expressed.
      ``radius_um`` here is exact and continuous.
    * ``probas == pooled`` marks every voxel of a plateau, so one flat blob
      yields a cluster of detections. Greedy NMS emits one point per
      ``radius_um`` ball instead — but only once the radius exceeds the grid
      spacing. On a 3.25 um grid a 2x2x2 plateau collapses to 8/4/2/1 points
      at radii of 2.5/3.5/5.0/5.7 um, since the block's own diagonal is
      5.63 um. Set the radius from the cell size, not from the voxel size.

    The cost is a Python loop of one round per output peak (see ``greedy_nms``),
    so it is slower than a single max-pool and is meant for inference.
    ``max_candidates`` caps the input by score to bound that.

    Returns
    -------
    torch.Tensor  shape (N, 3) int64 — detected voxel coordinates (z, y, x)
                  in the same (downsampled) grid as the input.
    """
    above = probas > threshold
    points = torch.nonzero(above)                       # (n, 3)
    if points.shape[0] == 0:
        return points
    scores = probas[above]

    if points.shape[0] > max_candidates:
        keep = scores.topk(max_candidates).indices
        points, scores = points[keep], scores[keep]

    vs = torch.as_tensor(voxel_size, dtype=torch.float32, device=probas.device)
    return points[greedy_nms(points, scores, vs, radius_um)]


def _peak_indices(
    probas: torch.Tensor,   # (Z, Y, X) raw probabilities
    threshold: float,
    nms: str = "maxpool",
    pool_kernel: tuple[int, ...] | None = None,
    voxel_size: tuple[float, float, float] | None = None,
    nms_radius_um: float | None = None,
) -> torch.Tensor:
    """Run the configured NMS, returning (N, 3) peak voxel coordinates."""
    if nms == "greedy":
        if voxel_size is None or nms_radius_um is None:
            raise ValueError("nms='greedy' needs voxel_size and nms_radius_um")
        return detect_peaks_greedy(
            probas,
            voxel_size=voxel_size,
            radius_um=nms_radius_um,
            threshold=threshold,
        )
    if nms == "maxpool":
        if pool_kernel is None:
            raise ValueError("nms='maxpool' needs pool_kernel")
        return detect_peaks(probas, threshold=threshold, pool_kernel=pool_kernel)
    raise ValueError(f"unknown nms mode {nms!r}, expected 'maxpool' or 'greedy'")


def _stamp_time(peak_idx: torch.Tensor, t: int) -> np.ndarray:
    """Prefix (N, 3) peak coordinates with their frame index."""
    if peak_idx.shape[0] == 0:
        return np.empty((0, 4), dtype=np.int16)
    coords = peak_idx.float().cpu().numpy()
    t_col = np.full((len(coords), 1), t, dtype=np.float32)
    return np.concatenate([t_col, coords], axis=1).astype(np.int16)


def detect_cells_pooled(
    probas: torch.Tensor,   # (1, Z, Y, X) or (Z, Y, X) raw probabilities
    t: int,
    det_threshold: float = 0.5,
    nms: str = "maxpool",
    pool_kernel: tuple[int, ...] | None = None,
    voxel_size: tuple[float, float, float] | None = None,
    nms_radius_um: float | None = None,
) -> np.ndarray:
    """Extract cell coordinates via local-max suppression, prefixed with time.

    ``nms="maxpool"`` uses the stride-1 max-pool kernel; ``nms="greedy"`` runs
    greedy NMS at an exact ``nms_radius_um`` instead, which needs
    ``voxel_size`` for the detection grid.

    Returns
    -------
    np.ndarray  (N, 4) int16 with columns [t, z, y, x] in downsampled space.
    """
    if probas.dim() == 4:
        probas = probas[0]

    peak_idx = _peak_indices(
        probas, det_threshold, nms=nms, pool_kernel=pool_kernel,
        voxel_size=voxel_size, nms_radius_um=nms_radius_um,
    )
    return _stamp_time(peak_idx, t)


def detect_cells_multi_threshold(
    probas: torch.Tensor,   # (1, Z, Y, X) or (Z, Y, X) raw probabilities
    t: int,
    thresholds: Sequence[float],
    nms: str = "maxpool",
    pool_kernel: tuple[int, ...] | None = None,
    voxel_size: tuple[float, float, float] | None = None,
    nms_radius_um: float | None = None,
    return_scores: bool = False,
) -> dict[float, np.ndarray] | dict[float, tuple[np.ndarray, np.ndarray]]:
    """``detect_cells_pooled`` at several thresholds, for the price of the lowest.

    Both NMS modes nest exactly across thresholds, so one pass at the lowest one
    answers all of them:

    * max-pool keeps ``(probas == pooled) & (probas > threshold)``, and being a
      local maximum does not depend on the threshold at all.
    * greedy NMS walks candidates in descending score, so every peak above a
      higher threshold is visited — and accepted or suppressed identically —
      before any candidate that only the lower threshold admits. Its result at
      the lower threshold therefore contains the higher one's as a prefix.

    Filtering by score is thus exact, not an approximation. That matters most
    for greedy NMS, which costs a GPU round per emitted peak.

    Returns
    -------
    dict  threshold -> (N, 4) int16 array with columns [t, z, y, x], as
    ``detect_cells_pooled`` returns. With ``return_scores``, the value is
    ``(coords, scores)`` instead, where ``scores`` is the (N,) float32
    probability the detection head gave each kept peak, row-aligned with
    ``coords``. It is read off the same map the threshold is applied to, so it
    costs nothing beyond carrying it.
    """
    if probas.dim() == 4:
        probas = probas[0]
    if not len(thresholds):
        return {}

    def out(coords, scores):
        return (coords, scores) if return_scores else coords

    peak_idx = _peak_indices(
        probas, min(thresholds), nms=nms, pool_kernel=pool_kernel,
        voxel_size=voxel_size, nms_radius_um=nms_radius_um,
    )
    if peak_idx.shape[0] == 0:
        empty = np.empty((0, 4), dtype=np.int16), np.empty(0, dtype=np.float32)
        return {th: out(*empty) for th in thresholds}

    scores = probas[peak_idx[:, 0], peak_idx[:, 1], peak_idx[:, 2]]
    scores_np = scores.float().cpu().numpy()
    return {
        th: out(_stamp_time(peak_idx[scores > th], t),
                scores_np[(scores > th).cpu().numpy()])
        for th in thresholds
    }


def build_graph(
    coords: np.ndarray,                              # (N, 4) [t, z, y, x]
    edges: list[tuple[int, int, float, float]],      # (src, tgt, prob, dist)
) -> td.graph.InMemoryGraph:
    """Build a tracksdata graph from detection coords and predicted edges.

    Avoids ``add_node_attr_key`` defaults for coordinates to sidestep a
    tracksdata/Polars issue where the float default is used as a dtype.
    """
    graph = td.graph.InMemoryGraph()

    for key in ["z", "y", "x"]:
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    node_ids = graph.bulk_add_nodes([
        {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
        for t, z, y, x in coords
    ])
    # print(max(node_ids))

    if len(edges):
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.bulk_add_edges([
            {
                "source_id": node_ids[src],
                "target_id": node_ids[tgt],
                "edge_prob": prob,
                "edge_dist": dist,
            }
            for src, tgt, prob, dist in edges
        ])

    return graph
