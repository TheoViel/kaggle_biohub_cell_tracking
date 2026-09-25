#!/usr/bin/env python
"""Cut competition-format crops out of the ultrack zebrafish embryo, for pseudo-labelling.

This embryo is already on the competition's voxel grid -- (1.625, 0.40625,
0.40625) um, verified against its own OME metadata -- so unlike the zebrahub
embryos nothing is resampled here. The crops are written as zarr v3 in exactly
the competition's layout, which is what makes them loadable by
``tracking_cellmot.io.open_dataset`` and predictable by the existing pipeline
with no special-casing.

**Normalisation is the thing to get right.** The detector is fitted against
``(x - q[0.001]) / (q[0.999] - q[0.001])`` where those quantiles are read from
``image_statistics.quantiles`` in the zarr attrs -- per dataset, not per frame
and not global (``tracking.data.NORM_DEFAULTS``, scope "video"). A competition
"video" is itself a crop of an embryo, so its quantiles are per-crop; the same
is done here. Get this wrong and the detector sees an intensity mapping it was
never fitted on, and its thresholds stop meaning anything.

That per-crop mapping is also why the crops are filtered on raw dynamic range.
Measured 2026-09-22, competition crops run q0.999 = 941 (p10) to 2910 (p90),
while ultrack crops run 40 to 1355: normalising a crop whose signal spans 40
levels stretches quantisation noise across the detector's whole input range.

    python scripts/make_ultrack_crops.py --dry-run
    python scripts/make_ultrack_crops.py --max-crops 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from params import SCALE  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "input" / "ext" / "ultrack_zebrafish_embryo.ome.zarr"
CROP = (64, 256, 256)
QUANTILES = (0.0, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999, 1.0)


def occupancy(volume: np.ndarray, threshold: float = 0.10) -> float:
    v = volume.astype(np.float32)
    low, high = np.quantile(v.ravel()[::7], [0.001, 0.999])
    return float((np.clip((v - low) / (high - low + 1e-6), 0, 4) > threshold).mean())


def write_crop(out_dir: Path, name: str, volumes: np.ndarray) -> Path:
    """One crop as a competition-format zarr v3 dataset."""
    path = out_dir / f"{name}.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=3)
    group.create_array(
        "0", shape=volumes.shape, chunks=(1, *volumes.shape[1:]), dtype="uint16",
        compressors=[BloscCodec(cname="zstd", clevel=1, shuffle=BloscShuffle.bitshuffle)],
    )[:] = volumes

    flat = volumes.ravel()[::7]
    group.attrs["multiscales"] = [{
        "version": "0.5",
        "axes": [
            {"name": "T", "type": "time", "unit": "second"},
            {"name": "Z", "type": "space", "unit": "micrometer"},
            {"name": "Y", "type": "space", "unit": "micrometer"},
            {"name": "X", "type": "space", "unit": "micrometer"},
        ],
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, *SCALE]}]}],
        "name": "0",
    }]
    # What the detector's normalisation reads. Per crop, as the competition's are.
    group.attrs["image_statistics"] = {
        "quantiles": {str(q): float(v) for q, v in zip(QUANTILES, np.quantile(flat, QUANTILES))}
    }
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(SRC))
    p.add_argument("--out", default=str(REPO / "input" / "ultrack_crops"))
    p.add_argument("--prefix", default="ultr", help="dataset name prefix (the 'embryo id')")
    p.add_argument("--min-occupancy", type=float, default=0.25)
    p.add_argument("--min-q999", type=float, default=200.0,
                   help="raw dynamic range floor; below this, per-crop normalisation is "
                        "stretching quantisation noise (competition p10 is 941)")
    p.add_argument("--max-crops", type=int, default=400)
    p.add_argument("--stride", type=int, nargs=3, default=[64, 256, 256],
                   metavar=("Z", "Y", "X"), help="grid step; equal to the crop = no overlap")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out)
    frames = json.load(open(src / "frames.json"))["frames_present"]
    arr = zarr.open_group(src, mode="r")["0"]
    print(f"source {arr.shape} at {tuple(SCALE)} um/voxel (no resampling needed)")

    # Blocks of consecutive frames -- each becomes one crop "video".
    blocks, run = [], [frames[0]]
    for t in frames[1:]:
        if t == run[-1] + 1:
            run.append(t)
        else:
            blocks.append(run); run = [t]
    blocks.append(run)
    print(f"{len(blocks)} blocks of consecutive frames, sizes {sorted({len(b) for b in blocks})}")

    shape = arr.shape[2:]
    origins = [(z, y, x)
               for z in range(0, shape[0] - CROP[0] + 1, args.stride[0])
               for y in range(0, shape[1] - CROP[1] + 1, args.stride[1])
               for x in range(0, shape[2] - CROP[2] + 1, args.stride[2])]
    print(f"{len(origins)} spatial positions x {len(blocks)} blocks "
          f"= {len(origins) * len(blocks)} candidates")

    rng = np.random.default_rng(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept, seen, rejected = [], 0, {"occupancy": 0, "q999": 0}
    started = time.time()

    for bi, block in enumerate(blocks):
        scout = np.asarray(arr[block[0], 0])          # one frame, to score the grid
        accept = []
        for (z, y, x) in origins:
            sub = scout[z:z + CROP[0], y:y + CROP[1], x:x + CROP[2]]
            seen += 1
            q999 = float(np.quantile(sub.ravel()[::7], 0.999))
            if q999 < args.min_q999:
                rejected["q999"] += 1; continue
            if occupancy(sub) < args.min_occupancy:
                rejected["occupancy"] += 1; continue
            accept.append((z, y, x, q999))
        print(f"  block {bi} (t={block[0]}-{block[-1]}): {len(accept)}/{len(origins)} pass",
              flush=True)
        if args.dry_run:
            kept += [(bi, *a) for a in accept]
            del scout
            continue

        if not accept:
            del scout
            continue
        # A per-block quota, drawn at random within the block: filling the
        # budget in block order would take every crop from the earliest
        # timepoints and none from the rest of the timelapse.
        quota = max(1, -(-args.max_crops // len(blocks)))
        if len(accept) > quota:
            accept = [accept[i] for i in rng.choice(len(accept), quota, replace=False)]

        volumes_by_t = [scout] + [np.asarray(arr[t, 0]) for t in block[1:]]
        for (z, y, x, q999) in accept:
            if len(kept) >= args.max_crops:
                break
            stack = np.stack([v[z:z + CROP[0], y:y + CROP[1], x:x + CROP[2]]
                              for v in volumes_by_t])
            name = f"{args.prefix}_{block[0]:04d}_{z:04d}_{y:04d}_{x:04d}"
            write_crop(out_dir, name, stack)
            kept.append((bi, z, y, x, q999))
        del volumes_by_t, scout
        if len(kept) >= args.max_crops:
            print("  reached --max-crops"); break

    print(f"\ncandidates scored {seen}, rejected: {rejected}")
    print(f"kept {len(kept)} crops in {(time.time() - started)/60:.1f} min")
    if args.dry_run:
        q = np.array([k[4] for k in kept]) if kept else np.zeros(1)
        print(f"raw q0.999 of survivors: p10 {np.quantile(q,.1):.0f} "
              f"p50 {np.median(q):.0f} p90 {np.quantile(q,.9):.0f}")
        print("dry run, nothing written")
        return
    json.dump({"source": str(src), "crop_shape": list(CROP), "voxel_um": list(SCALE),
               "min_occupancy": args.min_occupancy, "min_q999": args.min_q999,
               "n_crops": len(kept), "frames_per_crop": len(blocks[0]),
               "command": " ".join(sys.argv)},
              open(out_dir / "metadata.json", "w"), indent=2)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
