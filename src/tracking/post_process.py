"""Post-processing: peak detection and tracking graph construction."""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import tracksdata as td


def pool_kernel_from_um(
    um: float,
    voxel_size: tuple[float, ...],
) -> tuple[int, ...]:
    """Convert a physical suppression distance (microns) to a per-axis voxel kernel.

    Each axis gets ``round(um / voxel_size_axis)`` voxels, forced to odd (for
    symmetric padding) and at least 1.
    """
    kernel = []
    for size in voxel_size:
        k = max(1, round(um / size))
        if k % 2 == 0:
            k += 1
        kernel.append(k)
    return tuple(kernel)


def detect_peaks(
    det_logit: torch.Tensor,              # (Z, Y, X) raw detection logits
    pool_kernel_um: float = 3.0,
    voxel_size: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    threshold: float = 0.5,
    pool_kernel: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Detect local-maximum peaks in a detection logit volume.

    Uses max-pool NMS with a kernel sized in physical units, then keeps peaks
    whose sigmoid probability exceeds ``threshold``.

    Returns
    -------
    torch.Tensor  shape (N, 3) int64 — detected voxel coordinates (z, y, x)
                  in the same (downsampled) grid as the input.
    """
    if pool_kernel is None:
        pool_kernel = pool_kernel_from_um(pool_kernel_um, voxel_size)
    pad = tuple(k // 2 for k in pool_kernel)

    logits = det_logit[None, None]   # (1, 1, Z, Y, X)
    pooled = F.max_pool3d(logits, pool_kernel, stride=1, padding=pad)
    is_peak = (logits == pooled) & (torch.sigmoid(logits) > threshold)
    return torch.nonzero(is_peak[0, 0])   # (N, 3)


def detect_cells_pooled(
    det_logit: torch.Tensor,   # (1, Z, Y, X) or (Z, Y, X) raw logits
    t: int,
    det_threshold: float = 0.5,
    pool_kernel: tuple[int, ...] = (3, 3, 3),
) -> np.ndarray:
    """Extract cell coordinates via max-pool local-max, prefixed with time.

    Returns
    -------
    np.ndarray  (N, 4) int16 with columns [t, z, y, x] in downsampled space.
    """
    if det_logit.dim() == 4:
        det_logit = det_logit[0]

    peak_idx = detect_peaks(
        det_logit, threshold=det_threshold, pool_kernel=pool_kernel,
    )
    if peak_idx.shape[0] == 0:
        return np.empty((0, 4), dtype=np.int16)

    coords = peak_idx.float().cpu().numpy()
    t_col = np.full((len(coords), 1), t, dtype=np.float32)
    return np.concatenate([t_col, coords], axis=1).astype(np.int16)


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
