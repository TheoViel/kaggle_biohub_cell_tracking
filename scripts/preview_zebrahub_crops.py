#!/usr/bin/env python
"""Render sample crops with their ground-truth cell centres drawn on top.

One PNG per crop, two panels for the two frames of the window, so a glance
tells you whether the images, the coordinate conversion and the linking all
line up for an embryo -- the failure this catches is a track table stored in
voxels rather than micrometers, which misplaces every node by the voxel size
while every count in the manifest still looks reasonable.

    python scripts/preview_zebrahub_crops.py                    # every embryo
    python scripts/preview_zebrahub_crops.py --embryos ZSNS003 --n 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tracking.synthetic import find_sequences  # noqa: E402
from utils.plot import scatter_by_depth  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def render(path: Path, out_path: Path, embryo: str, frame_stride: int) -> dict:
    """One crop -> one PNG. Returns a row of what it holds."""
    data = np.load(path)
    volumes, nodes, edges = data["volumes"], data["nodes"], data["edges"]
    n_frames = volumes.shape[0]
    # The plane most of the annotation sits on, so a slice shows cells rather
    # than the empty half of a tile.
    first = nodes[nodes[:, 0] == 0]
    z_plane = int(round(float(np.median(first[:, 1])))) if len(first) else volumes.shape[1] // 2
    vmax = float(np.quantile(volumes, 0.999)) or None

    fig, axes = plt.subplots(1, n_frames, figsize=(5.4 * n_frames, 5.6))
    axes = np.atleast_1d(axes)
    for w, ax in enumerate(axes):
        frame_nodes = nodes[nodes[:, 0] == w]
        ax.imshow(volumes[w, z_plane], cmap="gray", vmax=vmax)
        drawn = scatter_by_depth(
            ax, frame_nodes[:, 1:4], z_plane, "#eb6834",
            tol=5, facecolor="none", s=42, linewidths=1.2,
        )
        ax.set_title(
            f"frame {w}  (raw t{'+' if w else ''}{w * frame_stride or ''})   "
            f"{len(frame_nodes)} cells, {int(drawn.sum())} within 5 slices",
            fontsize=9,
        )
        ax.axis(False)

    fig.suptitle(
        f"{embryo}   {path.stem}   z={z_plane}   "
        f"{len(edges)} links between the two frames",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return {"embryo": embryo, "crop": path.stem, "z": z_plane,
            "cells": int((nodes[:, 0] == 0).sum()), "links": int(len(edges))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crops-dir", default=str(REPO / "input" / "zebrahub_crops"))
    p.add_argument("--embryos", nargs="*", default=None,
                   help="default: every embryo directory found")
    p.add_argument("--n", type=int, default=5, help="crops to render per embryo")
    p.add_argument("--out", default=None, help="default: <crops-dir>/preview")
    args = p.parse_args()

    crops_dir = Path(args.crops_dir)
    out_dir = Path(args.out) if args.out else crops_dir / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    embryos = args.embryos
    if not embryos:
        embryos = sorted(
            d.name for d in crops_dir.iterdir()
            if d.is_dir() and d.name != "preview" and (d / "sequences").is_dir()
        )
    if not embryos:
        raise SystemExit(f"no embryo crop directories under {crops_dir}")

    rows = []
    for embryo in embryos:
        directory = crops_dir / embryo
        files = find_sequences(directory)
        meta_path = directory / "metadata.json"
        frame_stride = 1
        if meta_path.exists():
            frame_stride = int(json.loads(meta_path.read_text()).get("frame_stride", 1))

        # Spread the picks over the sorted list rather than taking the head:
        # the names sort by timepoint, so the first N are all the same instant
        # in the same corner of the embryo.
        n = min(args.n, len(files))
        picks = [files[round(i * (len(files) - 1) / max(n - 1, 1))] for i in range(n)]
        for path in picks:
            out_path = out_dir / f"{embryo}_{path.stem}.png"
            rows.append(render(path, out_path, embryo, frame_stride))
            print(f"  {out_path.name}")

    print(f"\n{len(rows)} images -> {out_dir}")
    for embryo in embryos:
        mine = [r for r in rows if r["embryo"] == embryo]
        if mine:
            print(f"  {embryo:<14} cells/frame {min(r['cells'] for r in mine)}-"
                  f"{max(r['cells'] for r in mine)}, links "
                  f"{min(r['links'] for r in mine)}-{max(r['links'] for r in mine)}")


if __name__ == "__main__":
    sys.exit(main())
