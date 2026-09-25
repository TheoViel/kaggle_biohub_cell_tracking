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
    stride: int = 1  # frame gap; 1 is consecutive


@dataclass(frozen=True)
class VideoMeta:
    """Lightweight per-video metadata; enables on-demand frame loading."""
    zarr_path: Path
    image_shape: tuple[int, ...]  # (T, Z_ds, Y_ds, X_ds)
    voxel_size: tuple[float, ...]  # physical voxel size = scale * downsample
    q_low: float  # 0.1% quantile for normalisation
    q_high: float  # 99.9% quantile for normalisation
    # Whole resolved normalisation config, so a consumer holding only the meta
    # can reproduce the scaling. Under scope="frame" the two quantiles above are
    # the per-video fallback and are not what gets applied.
    norm: dict | None = None


# ---------------------------------------------------------------------------
# Window building helpers
# ---------------------------------------------------------------------------


def _successor_map(edge_attrs: pl.DataFrame) -> dict[int, list[int]]:
    """node_id -> its children one frame later."""
    succ: dict[int, list[int]] = {}
    for source_id, target_id in zip(edge_attrs["source_id"], edge_attrs["target_id"]):
        succ.setdefault(source_id, []).append(target_id)
    return succ


def _k_step_transition(
    ids_a: np.ndarray,
    ids_b: np.ndarray,
    succ: dict[int, list[int]],
    stride: int,
) -> torch.Tensor:
    """Transition matrix over ``stride`` frames, following the track graph.

    At stride 1 this is the direct edge set. Past that it is reachability, so a
    cell that divides somewhere inside the gap reaches both daughters and the
    row carries two ones — which is what "the cells at t + stride that came
    from this one" means, and what the edge head should predict.
    """
    col = {nid: i for i, nid in enumerate(ids_b)}
    matrix = torch.zeros(len(ids_a), len(ids_b), dtype=torch.float32)
    for i, nid in enumerate(ids_a):
        frontier = {nid}
        for _ in range(stride):
            frontier = {c for s in frontier for c in succ.get(s, ())}
            if not frontier:
                break
        for c in frontier:
            j = col.get(c)
            if j is not None:
                matrix[i, j] = 1.0
    return matrix


def _get_window_data(
    gt_graph: td.graph.BaseGraph,
    image_shape: tuple[int, ...],
    t_start: int,
    window_size: int = 2,
    stride: int = 1,
    gt_attrs: pl.DataFrame | None = None,
    succ: dict[int, list[int]] | None = None,
) -> FrameWindowData | None:
    """Build a FrameWindowData for window_size frames spaced ``stride`` apart.

    Returns None if any frame in the window has zero GT nodes.

    ``gt_attrs`` and ``succ`` are hoisted out for the caller to compute once
    per video: they are the same for every window of it, and rebuilding the
    successor map per window is O(E) each time.
    """
    if gt_attrs is None:
        gt_attrs = gt_graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    if succ is None:
        succ = _successor_map(gt_graph.edge_attrs(attr_keys=["source_id", "target_id"]))

    per_frame_ids: list[np.ndarray] = []
    coords_list: list[torch.Tensor] = []
    node_counts: list[int] = []

    for i in range(window_size):
        t = t_start + i * stride
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
            _k_step_transition(per_frame_ids[i], per_frame_ids[i + 1], succ, stride)
        )

    return FrameWindowData(
        t_start=t_start,
        n_frames=window_size,
        stride=stride,
        coords=coords_list,
        node_counts=node_counts,
        targets=targets,
    )


def _pad_window(window: FrameWindowData, max_nodes: int, edges: bool = True) -> dict:
    """Pad GT nodes to max_nodes; return plain dict of tensors.

    ``edges=False`` leaves the transition matrix empty for the detection-only
    stages, which never read it. It is the expensive part of a window by a
    wide margin: at the max_nodes=768 the synthetic mix forces, one is 2.4 MB
    against ~20 KB for the coordinates, so a fold's 12392 windows carry 29 GB
    of zeros that are then cropped, re-indexed and collated for nothing.
    """
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

    # No frames rather than no nodes: numel() is 0 either way, so nothing is
    # allocated, but the frame loops downstream then fall through on their own
    # instead of indexing an empty node axis.
    targets = torch.zeros(W - 1 if edges else 0, M, M, dtype=torch.float32)
    for i in range(targets.shape[0]):
        nt = window.node_counts[i]
        nt1 = window.node_counts[i + 1]
        targets[i, :nt, :nt1] = window.targets[i]

    return {
        "t_start": window.t_start,
        "n_frames": W,
        "stride": window.stride,
        "coords": coords,
        "masks": masks,
        "targets": targets,
        "node_counts": node_counts,
    }


# Subsample stride in y/x for scope="frame" quantiles. Measured against the
# full-volume quantile on 4 stacks: <=0.27% error on q0.5, <=0.08% on q0.99,
# at 12ms/frame instead of 46ms. Nothing next to a forward pass.
QUANTILE_SUBSAMPLE = 2


NORM_DEFAULTS: dict = {
    # Which stored quantile anchors zero. "0.001" sits in the extreme low tail,
    # so whatever pedestal the background carries survives normalisation — the
    # two embryos' floors differ ~3x after it while their cell amplitude
    # matches to 9%. "0.1" sits inside the background instead, which cancels
    # that pedestal at source rather than augmenting around it.
    "low_q": "0.001",
    "high_q": "0.999",
    "clamp_max": 4.0,
    # "video": one anchor pair per stack, read from the zarr attrs (the original
    # behaviour, and the default so existing checkpoints keep their input
    # distribution). "frame": recompute the pair on every frame, which also
    # corrects drift within a stack that per-video anchors cannot see.
    "scope": "video",
    # Floor of the clamp. 0.0 flattens everything below low_q; worth raising in
    # magnitude if low_q is pushed up far enough to swallow real structure.
    "clamp_min": 0.0,
}


def resolve_normalization(cfg: dict | None) -> dict:
    return {**NORM_DEFAULTS, **(cfg or {})}


def frame_anchors(frame: np.ndarray, norm: dict) -> tuple[float, float]:
    """(low, high) anchors computed from one frame, for scope="frame".

    `low_q`/`high_q` are strings because under scope="video" they index the
    quantiles stored in the zarr attrs; here they are just parsed as floats, so
    one config spells both scopes.
    """
    s = QUANTILE_SUBSAMPLE
    sub = np.asarray(frame)[..., ::s, ::s]
    lo, hi = np.quantile(sub, [float(norm["low_q"]), float(norm["high_q"])])
    return float(lo), float(hi)


def normalize_frames(frames, norm: dict, quantiles: dict | None = None) -> np.ndarray:
    """Normalise raw (N, Z, Y, X) frames per a resolved config.

    The one entry point for anything outside the train/predict loops, so a
    consumer cannot silently pin a different anchor pair than the model was
    trained with. `quantiles` is the zarr attr dict, needed only for
    scope="video".
    """
    frames = np.asarray(frames, dtype=np.float32)
    if norm["scope"] == "frame":
        return np.stack(
            [apply_normalization(f, *frame_anchors(f, norm), norm) for f in frames]
        )
    lo = float(quantiles[norm["low_q"]])
    hi = float(quantiles[norm["high_q"]])
    return apply_normalization(frames, lo, hi, norm)


def apply_normalization(arr, lo: float, hi: float, norm: dict):
    """Scale to the anchors and clamp. Works on a numpy array or a torch tensor."""
    out = (arr - lo) / (hi - lo + 1e-6)
    lo_c, hi_c = float(norm["clamp_min"]), float(norm["clamp_max"])
    return out.clamp(lo_c, hi_c) if torch.is_tensor(out) else np.clip(out, lo_c, hi_c)


def load_dataset_windows(
    ds_path: Path,
    window_size: int = 2,
    invert_time: bool = False,
    max_frames: int | None = None,
    strides: tuple[int, ...] = (1,),
    normalization: dict | None = None,
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
    norm = resolve_normalization(normalization)
    # Only scope="video" indexes the stored quantiles; scope="frame" computes
    # its own, so any float is a legal anchor there.
    missing = [
        q for q in (norm["low_q"], norm["high_q"])
        if norm["scope"] == "video" and q not in ds.quantiles
    ]
    if missing:
        raise ValueError(
            f"Zarr attrs missing image_statistics.quantiles {missing} for {ds_path}. "
            f"Stored: {sorted(ds.quantiles)}. Run precompute_quantiles first, or "
            "pick normalization.low_q / high_q from the stored set."
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
        q_low=float(ds.quantiles[norm["low_q"]]) if norm["scope"] == "video" else float("nan"),
        q_high=float(ds.quantiles[norm["high_q"]]) if norm["scope"] == "video" else float("nan"),
        norm=norm,
    )

    # Same for every window of this video; O(E) to build, so build it once.
    gt_attrs = tracks.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    succ = _successor_map(tracks.edge_attrs(attr_keys=["source_id", "target_id"]))

    windows: list[FrameWindowData] = []
    for stride in sorted({int(s) for s in strides if int(s) >= 1}):
        span = (window_size - 1) * stride
        for t in range(image_shape[0] - span):
            data = _get_window_data(
                tracks, image_shape, t, window_size, stride=stride,
                gt_attrs=gt_attrs, succ=succ,
            )
            if data is not None:
                windows.append(data)

    return video_meta, windows
