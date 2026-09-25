#!/usr/bin/env python
"""Precompute non-overlapping training crops from a raw zebrahub embryo.

Tiles the embryo into ``(window_size, Z, Y, X)`` crops on the competition's
voxel grid and writes each as one npz in the synthetic-sequence format, so the
output directory is a drop-in ``synthetic_dir`` for ``main_track.py``::

    "synthetic_dir": "../input/zebrahub_crops/v1/",
    "synthetic_proportion": 0.3,
    "max_nodes": <the max_cells_per_frame printed below>,

See :mod:`data.zebrahub` for the format and the coordinate/scale conversions.

    # what would be generated, and how much of it
    python scripts/make_zebrahub_crops.py --dry-run

    # the real thing
    python scripts/make_zebrahub_crops.py --workers 32

Re-running skips crops already on disk, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import zarr
from joblib import Parallel, delayed
from tqdm import tqdm

from data.zebrahub import (
    DEFAULT_CROP_SHAPE,
    DEFAULT_FRAME_STRIDE,
    DEFAULT_WINDOW_SIZE,
    coordinate_units,
    load_tracks,
    open_embryo,
    plan_crops,
    resample_window,
    save_crop,
)
from params import SCALE

REPO = Path(__file__).resolve().parent.parent

# One zarr handle per worker process, opened on first use: the array is not
# picklable and re-opening it per crop costs a metadata read each time.
_ARRAYS: dict[str, object] = {}


def _array(zarr_path: str):
    arr = _ARRAYS.get(zarr_path)
    if arr is None:
        # Every worker is already a process; letting each spawn a thread pool
        # on top of that just oversubscribes the machine.
        torch.set_num_threads(1)
        arr = zarr.open_group(zarr_path, mode="r")["0"]
        _ARRAYS[zarr_path] = arr
    return arr


def occupancy(volume: np.ndarray, threshold: float = 0.10) -> float:
    """Share of a crop that holds tissue at all, after the loaders' normalisation.

    The tiling is blind to where the embryo is, so a tile can satisfy
    ``--min-cells`` from cells packed into one corner and be void everywhere
    else. That void is the whole of the remaining gap to the competition crops:
    measured 2026-09-22, the pedestal *inside* tissue already matches (0.26-0.28
    against 0.257) while occupancy does not (13-26% against 46%).
    """
    v = volume.astype(np.float32)
    low, high = np.quantile(v.ravel()[::7], [0.001, 0.999])
    normalised = np.clip((v - low) / (high - low + 1e-6), 0.0, 4.0)
    return float((normalised > threshold).mean())


def _build(crop, zarr_path: str, out_path: Path, *, native_scale, crop_shape,
           window_size, compress: bool, min_occupancy: float = 0.0) -> str | None:
    """Worker: read the crop's images, resample, write. Track tables come precomputed.

    Returns None when the crop is dropped for holding too little tissue; that
    can only be judged once the images are read, so it is decided here rather
    than in the plan.
    """
    volumes = resample_window(
        _array(zarr_path),
        crop.t0,
        crop.origin_um,
        native_scale=native_scale,
        crop_shape=crop_shape,
        window_size=window_size,
        frame_stride=crop.frame_stride,
    )
    if min_occupancy > 0 and occupancy(volumes[0]) < min_occupancy:
        # Rebuilding under a stricter threshold has to remove what a looser one
        # wrote, or the loader keeps reading a crop this run rejected.
        out_path.unlink(missing_ok=True)
        return None
    save_crop(out_path, volumes, crop, compress=compress)
    return crop.name


def _record(crop, path: Path) -> dict:
    """Manifest row for one crop. ``bytes`` is 0 when it is not on disk."""
    return {
        "name": crop.name,
        "t0": crop.t0,
        "iz": crop.tile[0], "iy": crop.tile[1], "ix": crop.tile[2],
        "n_nodes": int(len(crop.nodes)),
        "min_cells": int(min(crop.counts)),
        "max_cells": int(max(crop.counts)),
        "n_edges": int(len(crop.edges)),
        "n_divisions": int(len(crop.divisions)),
        "bytes": int(path.stat().st_size) if path.exists() else 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--embryo-dir", default=str(REPO / "zebrahub_raw" / "ZSNS001_tail"))
    p.add_argument("--out", default=None,
                   help="default: input/zebrahub_crops/<embryo name>, one directory per "
                        "embryo so a training run can name the ones it wants")
    p.add_argument("--crop-shape", type=int, nargs=3, default=list(DEFAULT_CROP_SHAPE),
                   metavar=("Z", "Y", "X"))
    p.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE,
                   help="frames per crop; the tracker's default window")
    p.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE,
                   help="raw frames between the frames of a window. Default 3 matches this "
                        "embryo's cell motion to one competition frame; 1 would pose an "
                        "easier linking problem than the model ever sees")
    p.add_argument("--t-step", type=int, default=None,
                   help="frames between window starts (default: the window's own span, i.e. "
                        "back to back). Raise it to subsample time.")
    p.add_argument("--t-jitter", type=int, default=None,
                   help="max random frames added to a window start, drawn per tile so tiles "
                        "are not all cut at the same instants (default: all the slack "
                        "t-step leaves over the window span)")
    p.add_argument("--seed", type=int, default=0, help="seed for the start jitter")
    p.add_argument("--min-cells", type=int, default=50,
                   help="drop a crop unless every frame holds this many cells")
    p.add_argument("--min-occupancy", type=float, default=0.0,
                   help="drop a crop unless this share of it holds tissue. The blind tiling "
                        "makes void space the last gap to the competition crops (46%% tissue "
                        "there, 13-26%% here); 0.25 closes most of it. Costs the read of the "
                        "crops it then rejects.")
    p.add_argument("--max-cells", type=int, default=None,
                   help="drop a crop if any frame exceeds this, instead of relying on the "
                        "loader's max_nodes to truncate it")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="only build the first N crops")
    p.add_argument("--no-compress", action="store_true",
                   help="store uncompressed, as the synthetic generator does (~4x larger, "
                        "~40ms/crop faster to load)")
    p.add_argument("--overwrite", action="store_true", help="rebuild crops already on disk")
    p.add_argument("--prune", action="store_true",
                   help="delete crops on disk that this plan no longer includes, e.g. after "
                        "tightening --max-cells. Without it they stay and still get trained on.")
    p.add_argument("--dry-run", action="store_true", help="plan and report, write nothing")
    args = p.parse_args()

    embryo_dir = Path(args.embryo_dir)
    out_dir = Path(args.out) if args.out else REPO / "input" / "zebrahub_crops" / embryo_dir.name
    crop_shape = tuple(args.crop_shape)
    zarr_path = str(embryo_dir / f"{embryo_dir.name}.ome.zarr")

    arr, native_scale, csv_path = open_embryo(embryo_dir)
    extent_um = tuple(s * sc for s, sc in zip(arr.shape[2:], native_scale))
    print(f"{embryo_dir.name}: {arr.shape} at {native_scale} um/voxel "
          f"-> {tuple(extent_um)} um, target {tuple(SCALE)} um/voxel")

    t_load = time.time()
    tracks = load_tracks(csv_path, voxel_shape=arr.shape[2:], native_scale=native_scale)
    units = coordinate_units(pl.read_csv(csv_path), arr.shape[2:], native_scale)
    print(f"{tracks.height:,} track rows loaded in {time.time() - t_load:.1f}s "
          f"(stored in {units}{'' if units == 'um' else ', converted to um'})")

    crops, summary = plan_crops(
        tracks, extent_um,
        crop_shape=crop_shape,
        window_size=args.window_size,
        frame_stride=args.frame_stride,
        t_step=args.t_step,
        t_jitter=args.t_jitter,
        min_cells=args.min_cells,
        max_cells=args.max_cells,
        seed=args.seed,
    )
    if args.limit:
        crops = crops[: args.limit]

    print(f"\nplan: {summary['n_crops']:,} crops of {crop_shape} "
          f"({summary['crop_um']} um) on a {summary['n_tiles']} tiling")
    print(f"  windows            : {args.window_size} frames {summary['frame_stride']} apart "
          f"(span {summary['span']}), starting every {summary['t_step']} "
          f"+0-{summary['t_jitter']} frames")
    print(f"  cells/frame        : {summary['min_cells_per_frame']}-{summary['max_cells_per_frame']}")
    print(f"  nodes kept         : {summary['n_nodes']:,} "
          f"({summary['node_coverage']:.1%} of the lineage table)")
    if summary["max_cells_per_frame"] > 768:
        print(f"  ! set max_nodes >= {summary['max_cells_per_frame']} when training on these: "
              "SyntheticDataset truncates a denser frame and the dropped cells then "
              "supervise as background")
    if not crops:
        print("nothing to build")
        return

    seq_dir = out_dir / "sequences"
    suffix = ".npz"
    tasks = []
    skipped = 0
    for crop in crops:
        path = seq_dir / f"{crop.name}{suffix}"
        if path.exists() and not args.overwrite:
            skipped += 1
            continue
        tasks.append((crop, path))

    est_mb = 4.0 if not args.no_compress else 16.8
    print(f"  to build           : {len(tasks):,} crops (~{len(tasks) * est_mb / 1000:.1f} GB), "
          f"{skipped:,} already on disk")

    # Crops left over from a looser plan -- tightening --max-cells or raising
    # --min-cells drops some, and the npz stays on disk where find_sequences
    # will happily keep feeding it to training. --limit is not a plan, so it
    # never prunes.
    stale = []
    if seq_dir.is_dir() and not args.limit:
        planned = {f"{crop.name}{suffix}" for crop in crops}
        stale = [p for p in sorted(seq_dir.glob(f"seq_*{suffix}")) if p.name not in planned]
    if stale:
        if args.prune:
            print(f"  pruning            : {len(stale):,} crops no longer in the plan")
        else:
            print(f"  ! {len(stale):,} crops on disk are NOT in this plan and would still be "
                  "read by the loader; pass --prune to remove them")

    if args.dry_run:
        print("\ndry run, nothing written")
        return

    seq_dir.mkdir(parents=True, exist_ok=True)
    if args.prune:
        for path in stale:
            path.unlink()
    started = time.time()
    results = Parallel(n_jobs=args.workers, return_as="generator_unordered")(
        delayed(_build)(
            crop, zarr_path, path,
            native_scale=native_scale,
            crop_shape=crop_shape,
            window_size=args.window_size,
            compress=not args.no_compress,
            min_occupancy=args.min_occupancy,
        )
        for crop, path in tasks
    )
    outcomes = list(tqdm(results, total=len(tasks), desc="crops"))
    n_built = sum(1 for name in outcomes if name is not None)
    n_sparse = len(outcomes) - n_built
    elapsed = time.time() - started

    # Describe every planned crop that is on disk, not only the ones this run
    # built: a fully resumed run would otherwise write a manifest of zeros.
    records = [r for r in (_record(c, seq_dir / f"{c.name}{suffix}") for c in crops) if r["bytes"]]
    table = pl.DataFrame(records).sort("name")
    table.write_csv(out_dir / "crops.csv")

    total_bytes = int(table["bytes"].sum())
    metadata = {
        "embryo": embryo_dir.name,
        "command": " ".join(sys.argv),
        "source": str(embryo_dir),
        "native_um": list(native_scale),
        "voxel_um": list(SCALE),
        "crop_shape": list(crop_shape),
        "window_size": args.window_size,
        "frame_stride": summary["frame_stride"],
        "t_step": summary["t_step"],
        "t_jitter": summary["t_jitter"],
        "seed": args.seed,
        "min_cells": args.min_cells,
        "max_cells": args.max_cells,
        "min_occupancy": args.min_occupancy,
        "compressed": not args.no_compress,
        "n_sequences": len(list(seq_dir.glob(f"seq_*{suffix}"))),
        "max_cells_per_frame": summary["max_cells_per_frame"],
        "n_divisions": int(table["n_divisions"].sum()) if table.height else 0,
        "total_gb": total_bytes / 1024 ** 3,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if n_sparse:
        print(f"  dropped {n_sparse:,} crops below --min-occupancy {args.min_occupancy}")
    print(f"\nbuilt {n_built:,} crops in {elapsed / 60:.1f} min; "
          f"{table.height:,} on disk ({total_bytes / 1e9:.1f} GB), "
          f"{metadata['n_divisions']:,} divisions")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
