"""Turn a raw zebrahub embryo into training crops in the synthetic-sequence format.

The public zebrahub release (``scripts/download_zebrahub.py``) ships one
OME-Zarr pyramid per embryo plus a flat ``*_tracks.csv`` lineage table. Two
things differ from the competition data and are corrected here:

* **Voxel scale.** The raw zarr is ``(1.24, 0.439, 0.439)`` um/voxel against the
  competition's :data:`params.SCALE`. Windows are resampled with the same
  trilinear call ``tracking_cellmot.io._process_on_gpu`` uses for
  ``target_scale``.
* **Track coordinates.** The CSV stores ``z, y, x`` in physical micrometers,
  not voxel indices like the competition ``.geff`` files, so they convert to
  the target grid by dividing by the *target* voxel size directly.

Raw intensities are stored unscaled: every loader in this repo re-normalises
against its own quantiles at load time (:func:`tracking.data.normalize_frames`,
:meth:`tracking.synthetic.SyntheticDataset._one`), which is what reconciles
this embryo's ~10x smaller dynamic range with the competition's.

Output matches :mod:`tracking.synthetic` exactly, so a crop directory is a
drop-in ``synthetic_dir``::

    volumes     (W, Z, Y, X) uint16     raw, unnormalised
    nodes       (N, 5) float32          [t, z, y, x, track_id], t local to the window
    edges       (E, 2) int32            index pairs into nodes
    divisions   (D,) int32              indices of dividing nodes
    voxel_um    (3,) float32            == params.SCALE

Note that :class:`~tracking.synthetic.SyntheticDataset` truncates a frame to
its ``max_nodes``, and these crops are annotated far more densely than the
synthetic ones (up to ~1400 cells/frame against ~285). Set ``max_nodes`` at or
above the ``max_cells_per_frame`` the planner reports, or cap it at generation
time with ``max_cells``: cells dropped by truncation are supervised as
background, which is exactly the error this data is meant to avoid.

**Why the frames in a window are not consecutive.** This embryo is imaged
about 3x faster than the competition's, so consecutive raw frames pose an
easier linking problem than the model will ever see. Matched on the
displacement distribution of linked cells (quantiles 0.25-0.95, measured
2026-09-22 over 50 train geffs and this embryo's lineage):

    competition, 1 frame   median 1.86 um   p90 4.34
    zebrahub, lag 2        median 1.58 um   p90 3.34   log-quantile RMSE 0.234
    zebrahub, lag 3        median 2.01 um   p90 4.05   log-quantile RMSE 0.197  <-
    zebrahub, lag 4        median 2.52 um   p90 4.83   log-quantile RMSE 0.338

so :data:`DEFAULT_FRAME_STRIDE` is 3 raw frames per competition frame. The
crop still stores its ``window_size`` frames back to back, which is why
``SyntheticDataset``'s hardcoded ``stride: 1`` stays true — it is 1 in
competition frame units.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import zarr

from params import SCALE

DEFAULT_CROP_SHAPE: tuple[int, int, int] = (64, 256, 256)
DEFAULT_WINDOW_SIZE: int = 2
# Raw frames per competition frame; see the module docstring for the fit.
DEFAULT_FRAME_STRIDE: int = 3


@dataclass(frozen=True)
class Crop:
    """One planned crop: where it is, and the tracks inside it."""

    name: str
    t0: int
    frame_stride: int
    tile: tuple[int, int, int]        # (iz, iy, ix) index into the tiling
    origin_um: tuple[float, float, float]
    nodes: np.ndarray                 # (N, 5) float32
    edges: np.ndarray                 # (E, 2) int32
    divisions: np.ndarray             # (D,) int32

    @property
    def counts(self) -> list[int]:
        """Cells per frame."""
        n_frames = int(self.nodes[:, 0].max()) + 1 if len(self.nodes) else 0
        return [int((self.nodes[:, 0] == w).sum()) for w in range(n_frames)]


# ---------------------------------------------------------------------------
# Source data
# ---------------------------------------------------------------------------


def open_embryo(embryo_dir: Path | str, level: str = "0"):
    """``(zarr array (T,C,Z,Y,X), native voxel size, tracks csv path)`` for an embryo."""
    embryo_dir = Path(embryo_dir)
    name = embryo_dir.name
    group = zarr.open_group(embryo_dir / f"{name}.ome.zarr", mode="r")
    datasets = group.attrs.asdict()["multiscales"][0]["datasets"]
    scale = next(d for d in datasets if d["path"] == level)["coordinateTransformations"][0]["scale"]
    return group[level], tuple(scale[-3:]), embryo_dir / f"{name}_tracks.csv"


# The release is not consistent about this: ZSNS001_tail ships
# ``ParentTrackID`` alongside NodeID/area, ZSNS001 ships ``parent_track_id``
# and nothing else. Only the parent column is actually read.
PARENT_COLUMNS = ("ParentTrackID", "parent_track_id")


def coordinate_units(
    tracks: pl.DataFrame,
    voxel_shape: tuple[int, int, int],
    native_scale: tuple[float, float, float],
) -> str:
    """``"um"`` or ``"voxel"`` — which one this table's ``z, y, x`` are in.

    The release is not consistent about this and nothing in the file says so:
    ZSNS001 and ZSNS001_tail store physical micrometers, ZSNS003/4/5 store
    integer voxel indices. Read the wrong way round the error is a factor of
    the voxel size — about 2.5x in y/x — which silently lands every node
    outside the tissue rather than failing.

    Two independent signals, which agree on all five embryos:

    * a coordinate past the imaged volume in micrometers cannot be micrometers;
    * a table of exact integers over millions of rows is an index, not a
      measurement (the um embryos are 98.6% fractional).
    """
    maxima = np.array([float(tracks[c].max()) for c in ("z", "y", "x")])
    extent_um = np.array(voxel_shape) * np.array(native_scale)
    sample = tracks.select(["z", "y", "x"]).head(100_000).to_numpy()

    outside_volume = bool((maxima > extent_um + 1).any())
    integral = bool(np.all(sample % 1 == 0))
    if not (outside_volume or integral):
        return "um"

    if (maxima > np.array(voxel_shape) + 1).any():
        raise ValueError(
            f"coordinates {maxima} fit neither micrometers (extent {extent_um}) "
            f"nor voxels (grid {voxel_shape})"
        )
    return "voxel"


def load_tracks(
    csv_path: Path | str,
    voxel_shape: tuple[int, int, int] | None = None,
    native_scale: tuple[float, float, float] | None = None,
) -> pl.DataFrame:
    """The lineage table in micrometers, with each track's parent id on every row.

    The parent id is already constant within a track and is only ever set on a
    track born from a division — every value that appears names a track with
    exactly two children, in every embryo — so it doubles as the division
    marker.

    Pass the image geometry to have voxel-indexed tables converted to
    micrometers; see :func:`coordinate_units` for why that is not optional in
    practice. Without it the table is returned exactly as stored, which is only
    safe for the embryos that already ship micrometers.
    """
    tracks = pl.read_csv(csv_path)
    try:
        column = next(c for c in PARENT_COLUMNS if c in tracks.columns)
    except StopIteration:
        raise ValueError(
            f"{csv_path} has none of {PARENT_COLUMNS}; columns are {tracks.columns}"
        ) from None

    tracks = tracks.with_columns(
        pl.col(column).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False).alias("_parent")
    )
    parents = tracks.group_by("track_id").agg(
        pl.col("_parent").first().alias("parent_track_id")
    )
    tracks = tracks.drop("_parent").join(parents, on="track_id", how="left")

    if voxel_shape is not None and native_scale is not None:
        if coordinate_units(tracks, voxel_shape, native_scale) == "voxel":
            tracks = tracks.with_columns(
                [pl.col(c) * s for c, s in zip("zyx", native_scale)]
            )
    return tracks


# ---------------------------------------------------------------------------
# Planning: which crops exist, and what is in them
# ---------------------------------------------------------------------------


def plan_crops(
    tracks: pl.DataFrame,
    extent_um: tuple[float, float, float],
    *,
    crop_shape: tuple[int, int, int] = DEFAULT_CROP_SHAPE,
    window_size: int = DEFAULT_WINDOW_SIZE,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    t_step: int | None = None,
    t_jitter: int | None = None,
    min_cells: int = 50,
    max_cells: int | None = None,
    target_scale: tuple[float, float, float] = SCALE,
    prefix: str = "seq",
    seed: int = 0,
) -> tuple[list[Crop], dict]:
    """Tile the embryo into non-overlapping crops and keep the populated ones.

    The tiling is an axis-aligned grid of ``crop_shape * target_scale`` um
    boxes anchored at the origin, times windows of ``window_size`` frames
    ``frame_stride`` apart, starting every ``t_step`` frames. Only whole tiles
    are emitted — a partial one at the volume edge would have to be stretched
    to ``crop_shape`` on resample, which would put the images and the node
    coordinates on different grids.

    Window starts are jittered by up to ``t_jitter`` frames, drawn per tile so
    neighbouring tiles sample different frames. Without it every tile in the
    embryo is cut at the same instants, and the set is one sequence of whole-
    embryo snapshots rather than an independent sample of each region.
    ``t_jitter`` is clamped so windows within a tile still cannot overlap;
    it defaults to filling whatever slack ``t_step`` leaves.

    A crop is kept when *every* frame of its window holds at least
    ``min_cells`` cells, which is what drops the mostly-empty tiles: the tail
    fills about 3.5% of its volume, so most of the grid is background.

    Returns the crops and a summary dict (tiling shape, coverage, and the
    ``max_cells_per_frame`` that ``max_nodes`` has to clear).
    """
    frame_stride = int(frame_stride)
    span = (window_size - 1) * frame_stride + 1     # frames a window covers
    t_step = int(t_step if t_step is not None else span)
    slack = max(0, t_step - span)
    t_jitter = slack if t_jitter is None else min(int(t_jitter), slack)

    crop_um = tuple(float(s * sc) for s, sc in zip(crop_shape, target_scale))
    n_tiles = tuple(int(e // c) for e, c in zip(extent_um, crop_um))

    binned = tracks.with_columns(
        (pl.col("t") // t_step).cast(pl.Int32).alias("k"),
        (pl.col("z") // crop_um[0]).cast(pl.Int32).alias("iz"),
        (pl.col("y") // crop_um[1]).cast(pl.Int32).alias("iy"),
        (pl.col("x") // crop_um[2]).cast(pl.Int32).alias("ix"),
    )
    # Deterministic per (window, tile), so a row's offset does not depend on
    # which rows happen to be in the frame when it is computed.
    offset = (
        pl.struct(["k", "iz", "iy", "ix"]).hash(seed=seed) % (t_jitter + 1)
        if t_jitter > 0
        else pl.lit(0)
    )
    binned = binned.with_columns(offset.cast(pl.Int32).alias("offset"))
    binned = binned.with_columns(
        (pl.col("t") - (pl.col("k") * t_step + pl.col("offset"))).cast(pl.Int32).alias("rel")
    )
    inside = binned.filter(
        (pl.col("rel") >= 0)
        & (pl.col("rel") % frame_stride == 0)
        & ((pl.col("rel") // frame_stride) < window_size)
        & (pl.col("iz") >= 0) & (pl.col("iz") < n_tiles[0])
        & (pl.col("iy") >= 0) & (pl.col("iy") < n_tiles[1])
        & (pl.col("ix") >= 0) & (pl.col("ix") < n_tiles[2])
    ).with_columns((pl.col("rel") // frame_stride).cast(pl.Int32).alias("w"))

    key = ["k", "iz", "iy", "ix"]
    per_frame = inside.group_by([*key, "w"]).len()
    per_crop = (
        per_frame.group_by(key)
        .agg(
            pl.col("len").min().alias("min_cells"),
            pl.col("len").max().alias("max_cells"),
            pl.len().alias("n_frames"),
        )
        # A window missing a frame entirely is not a window: min_cells above is
        # taken over the frames that are present, so it cannot catch this.
        .filter(pl.col("n_frames") == window_size)
        .filter(pl.col("min_cells") >= min_cells)
    )
    if max_cells is not None:
        per_crop = per_crop.filter(pl.col("max_cells") <= max_cells)

    rows = inside.join(per_crop.select(key), on=key, how="semi").sort([*key, "w", "track_id"])
    crops = _crops_from_rows(rows, crop_um=crop_um, t_step=t_step, frame_stride=frame_stride,
                             window_size=window_size, target_scale=target_scale, prefix=prefix)

    kept_nodes = int(rows.height)
    summary = {
        "n_tiles": list(n_tiles),
        "crop_um": list(crop_um),
        "t_step": t_step,
        "t_jitter": t_jitter,
        "frame_stride": frame_stride,
        "span": span,
        "n_crops": len(crops),
        "n_nodes": kept_nodes,
        # What the loader has to be able to hold; see the module docstring.
        "max_cells_per_frame": int(per_crop["max_cells"].max()) if per_crop.height else 0,
        "min_cells_per_frame": int(per_crop["min_cells"].min()) if per_crop.height else 0,
        # Windowing and the min_cells cut both drop nodes; this is what survived.
        "node_coverage": kept_nodes / tracks.height if tracks.height else 0.0,
    }
    return crops, summary


def _crops_from_rows(
    rows: pl.DataFrame,
    *,
    crop_um: tuple[float, float, float],
    t_step: int,
    frame_stride: int,
    window_size: int,
    target_scale: tuple[float, float, float],
    prefix: str,
) -> list[Crop]:
    """Split the accepted rows (sorted by crop key) into one Crop each."""
    if not rows.height:
        return []

    k = rows["k"].to_numpy()
    offset = rows["offset"].to_numpy()
    tile = np.stack([rows[c].to_numpy() for c in ("iz", "iy", "ix")], axis=1)
    w = rows["w"].to_numpy()
    zyx = np.stack([rows[c].to_numpy() for c in ("z", "y", "x")], axis=1)
    track_id = rows["track_id"].to_numpy()
    parent_id = rows["parent_track_id"].fill_null(-1).to_numpy()

    keys = np.column_stack([k, tile])
    starts = np.flatnonzero(np.r_[True, (keys[1:] != keys[:-1]).any(axis=1)])
    stops = np.r_[starts[1:], len(keys)]

    crops = []
    for a, b in zip(starts, stops):
        iz, iy, ix = (int(v) for v in tile[a])
        t0 = int(k[a]) * t_step + int(offset[a])
        origin = (iz * crop_um[0], iy * crop_um[1], ix * crop_um[2])
        nodes, edges, divisions = _tables(
            w[a:b], zyx[a:b], track_id[a:b], parent_id[a:b],
            origin_um=origin, target_scale=target_scale, window_size=window_size,
        )
        crops.append(Crop(
            name=f"{prefix}_t{t0:04d}_z{iz}_y{iy}_x{ix}",
            t0=t0, frame_stride=frame_stride, tile=(iz, iy, ix), origin_um=origin,
            nodes=nodes, edges=edges, divisions=divisions,
        ))
    return crops


def _tables(w, zyx, track_id, parent_id, *, origin_um, target_scale, window_size):
    """Node/edge/division arrays for one crop, from its rows sorted by frame."""
    nodes = np.empty((len(w), 5), dtype=np.float32)
    nodes[:, 0] = w
    nodes[:, 1:4] = (zyx - np.asarray(origin_um)) / np.asarray(target_scale)
    nodes[:, 4] = track_id

    pairs: list[tuple[int, int]] = []
    for frame in range(window_size - 1):
        src = np.flatnonzero(w == frame)
        dst = np.flatnonzero(w == frame + 1)
        row_of_track = {int(t): int(i) for t, i in zip(track_id[src], src)}
        for j in dst:
            # A cell at t+1 either continues its own track or is one of the two
            # daughters of the track named by its parent id. Never both: a
            # track only has a parent id on the frame it is born.
            i = row_of_track.get(int(track_id[j]))
            if i is not None:
                pairs.append((i, int(j)))
                continue
            i = row_of_track.get(int(parent_id[j]))
            if i is not None:
                pairs.append((i, int(j)))

    edges = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
    out_degree = (
        np.bincount(edges[:, 0], minlength=len(w)) if len(edges) else np.zeros(len(w), dtype=int)
    )
    divisions = np.flatnonzero(out_degree == 2).astype(np.int32)
    return nodes, edges, divisions


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------


def resample_window(
    arr,
    t0: int,
    origin_um: tuple[float, float, float],
    *,
    native_scale: tuple[float, float, float],
    crop_shape: tuple[int, int, int] = DEFAULT_CROP_SHAPE,
    window_size: int = DEFAULT_WINDOW_SIZE,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    target_scale: tuple[float, float, float] = SCALE,
) -> np.ndarray:
    """Read the native-resolution box covering a crop and resample it to ``crop_shape``."""
    crop_um = [s * sc for s, sc in zip(crop_shape, target_scale)]
    starts = [int(round(o / s)) for o, s in zip(origin_um, native_scale)]
    stops = [int(round((o + c) / s)) for o, c, s in zip(origin_um, crop_um, native_scale)]

    volume_shape = arr.shape[2:]
    if any(s < 0 for s in starts) or any(e > v for e, v in zip(stops, volume_shape)):
        # Clipping here would rescale the box by a different factor than the
        # node coordinates, which are placed on the nominal target grid.
        raise ValueError(f"crop at {origin_um} um falls outside the volume")

    # One read per frame: the frames of a window are frame_stride apart, and a
    # strided slice would make zarr fetch the chunks in between as well.
    raw = np.stack([
        np.asarray(
            arr[t0 + i * frame_stride, 0,
                starts[0]:stops[0], starts[1]:stops[1], starts[2]:stops[2]],
            dtype=np.float32,
        )
        for i in range(window_size)
    ])
    tensor = torch.from_numpy(raw)[:, None]
    out = F.interpolate(tensor, size=tuple(crop_shape), mode="trilinear", align_corners=False)
    return out[:, 0].round_().clamp_(0, 65535).to(torch.uint16).numpy()


def save_crop(path: Path, volumes: np.ndarray, crop: Crop, *,
              voxel_um: tuple[float, float, float] = SCALE, compress: bool = True) -> Path:
    """Write one crop in the synthetic-sequence npz layout."""
    arrays = dict(
        volumes=volumes,
        nodes=crop.nodes,
        edges=crop.edges,
        divisions=crop.divisions,
        voxel_um=np.asarray(voxel_um, dtype=np.float32),
    )
    save = np.savez_compressed if compress else np.savez
    save(path, **arrays)
    return path
