import torch
import numpy as np
import polars as pl
import tracksdata as td

from dataclasses import dataclass
from pathlib import Path

from tracking_cellmot.io import open_dataset, invert_time_graph

from tracking.targets import compute_gt_transition_matrix

# ---------------------------------------------------------------------------
# Data structures (mirrors biohub FrameWindowData / VideoMeta)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameWindowData:
    """Metadata for a window of W consecutive frames (no image data stored)."""

    t_start: int
    n_frames: int
    coords: list[torch.Tensor]  # W × (N_i, 3)  in downsampled space
    node_counts: list[int]
    targets: list[torch.Tensor]  # (W-1) transition matrices


@dataclass(frozen=True)
class VideoMeta:
    """Lightweight per-video metadata; enables on-demand frame loading."""
    zarr_path: Path
    image_shape: tuple[int, ...]  # (T, Z_ds, Y_ds, X_ds)
    voxel_size: tuple[float, ...]  # physical voxel size = scale * downsample
    q_low: float  # 0.1% quantile for normalisation
    q_high: float  # 99.9% quantile for normalisation


# ---------------------------------------------------------------------------
# Window building helpers
# ---------------------------------------------------------------------------


def _get_window_data(
    gt_graph: td.graph.BaseGraph,
    image_shape: tuple[int, ...],
    t_start: int,
    window_size: int = 2,
) -> FrameWindowData | None:
    """Build a FrameWindowData for window_size consecutive frames.

    Returns None if any frame in the window has zero GT nodes.
    """
    gt_attrs = gt_graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    edge_attrs = gt_graph.edge_attrs(attr_keys=["source_id", "target_id"])

    per_frame_ids: list[np.ndarray] = []
    coords_list: list[torch.Tensor] = []
    node_counts: list[int] = []

    for i in range(window_size):
        t = t_start + i
        gt_t = gt_attrs.filter(pl.col("t") == t)
        if len(gt_t) == 0:
            return None

        gt_coords_t = gt_t.select(["z", "y", "x"]).to_numpy().astype(np.float32)
        gt_ids = gt_t["node_id"].to_numpy()

        coords_list.append(torch.from_numpy(gt_coords_t))
        per_frame_ids.append(gt_ids)
        node_counts.append(len(gt_coords_t))

    targets = []
    for i in range(window_size - 1):
        targets.append(
            compute_gt_transition_matrix(
                per_frame_ids[i],
                per_frame_ids[i + 1],
                edge_attrs,
            )
        )

    return FrameWindowData(
        t_start=t_start,
        n_frames=window_size,
        coords=coords_list,
        node_counts=node_counts,
        targets=targets,
    )


def _pad_window(window: FrameWindowData, max_nodes: int) -> dict:
    """Pad GT nodes to max_nodes; return plain dict of tensors."""
    W = window.n_frames
    M = max_nodes

    coords = torch.zeros(W, M, 3, dtype=torch.float32)
    masks = torch.zeros(W, M, dtype=torch.bool)
    node_counts = torch.zeros(W, dtype=torch.long)

    for i in range(W):
        n = window.node_counts[i]
        coords[i, :n] = window.coords[i]
        masks[i, :n] = True
        node_counts[i] = n

    targets = torch.zeros(W - 1, M, M, dtype=torch.float32)
    for i in range(W - 1):
        nt = window.node_counts[i]
        nt1 = window.node_counts[i + 1]
        targets[i, :nt, :nt1] = window.targets[i]

    return {
        "t_start": window.t_start,
        "n_frames": W,
        "coords": coords,
        "masks": masks,
        "targets": targets,
        "node_counts": node_counts,
    }


def load_dataset_windows(
    ds_path: Path,
    window_size: int = 2,
    invert_time: bool = False,
    max_frames: int | None = None,
) -> tuple[VideoMeta, list[FrameWindowData]]:
    """Load per-window metadata and video stats for one dataset.

    Uses open_dataset(load_image=False) to avoid loading the full image.
    """
    ds = open_dataset(
        ds_path,
        normalize=False,
        require_tracks=True,
        load_image=False,
    )
    if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
        raise ValueError(
            f"Zarr attrs missing image_statistics.quantiles for {ds_path}. "
            "Run precompute_quantiles on the dataset first."
        )

    image_shape = ds.image_shape
    tracks = ds.tracks

    if invert_time:
        tracks = invert_time_graph(tracks, max_t=image_shape[0])

    if max_frames is not None:
        image_shape = (max_frames, *image_shape[1:])
        tracks = tracks.filter(td.NodeAttr("t") < max_frames).subgraph()

    video_meta = VideoMeta(
        zarr_path=ds.zarr_path,
        image_shape=image_shape,
        voxel_size=ds.scale,
        q_low=float(ds.quantiles["0.001"]),
        q_high=float(ds.quantiles["0.999"]),
    )

    windows: list[FrameWindowData] = []
    for t in range(image_shape[0] - window_size + 1):
        data = _get_window_data(tracks, image_shape, t, window_size)
        if data is not None:
            windows.append(data)

    return video_meta, windows
