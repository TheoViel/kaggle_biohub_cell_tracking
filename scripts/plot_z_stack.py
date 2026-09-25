#!/usr/bin/env python
"""Walk one crop through z, one panel per slice, with the nodes on that slice.

The side-by-side plots draw every node within +-5 slices (+-8.1 um, about one
cell diameter in z) and fade them with depth, which is what makes two cells
stacked in z read as one cell detected twice. Stepping through z with a tight
tolerance separates the two cases: a genuine pair appears on different slices,
a duplicate appears twice on the same one.

    python scripts/plot_z_stack.py --crop ultr_0000_0000_1024_1280 --z 48 58
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from tracking.data import normalize_frames, resolve_normalization  # noqa: E402
from utils.plot import scatter_by_depth  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crop", default="ultr_0000_0000_1024_1280")
    p.add_argument("--crops", default=str(REPO / "input" / "ultrack_crops"))
    p.add_argument("--pl", default=str(REPO / "output" / "pl_ultrack_0.40"))
    p.add_argument("--z", type=int, nargs=2, default=[48, 58], metavar=("Z0", "Z1"))
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--tol", type=float, default=1.0,
                   help="slices either side still drawn; keep small to separate depths")
    p.add_argument("--out", default=str(REPO / "output" / "plots"))
    p.add_argument("--prefix", default="zstack",
                   help="output filename prefix; change it to keep an earlier render")
    args = p.parse_args()

    group = zarr.open_group(Path(args.crops) / f"{args.crop}.zarr", mode="r")
    raw = np.asarray(group["0"][args.frame : args.frame + 1])
    img = normalize_frames(raw, resolve_normalization(None),
                           group.attrs["image_statistics"]["quantiles"])[0]

    coords = np.load(Path(args.pl) / f"{args.crop}.npz")["coords"].astype(float)
    pts = coords[coords[:, 0] == args.frame][:, 1:4]

    n_z = img.shape[0]
    z0, z1 = max(0, args.z[0]), min(n_z - 1, args.z[1])
    if [z0, z1] != list(args.z):
        print(f"note: z clamped to {z0}-{z1}; the crop has {n_z} slices (0-{n_z - 1})")
    planes = list(range(z0, z1 + 1))
    cols = min(4, len(planes))
    rows = -(-len(planes) // cols)
    vmax = float(np.quantile(img, 0.999)) or None

    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.15 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, z in zip(axes, planes):
        ax.imshow(img[z], cmap="gray", vmax=vmax)
        drawn = scatter_by_depth(ax, pts, z, "#35d0ba", tol=args.tol,
                                 facecolor="none", s=52, linewidths=1.4)
        on_plane = int(np.sum(np.abs(pts[:, 0] - z) < 0.5)) if len(pts) else 0
        ax.set_title(f"z={z}   {on_plane} centred here, {int(drawn.sum())} within "
                     f"{args.tol:g}", fontsize=9)
        ax.axis(False)
    for ax in axes[len(planes):]:
        ax.axis(False)

    fig.suptitle(f"{args.crop}   t={args.frame}   z {z0}-{z1}   "
                 f"({len(pts)} nodes in the frame)", fontsize=12)
    fig.tight_layout()
    out = Path(args.out) / f"{args.prefix}_{args.crop}_z{z0}-{z1}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=88, bbox_inches="tight")
    print(f"-> {out}")
    per_z = np.bincount(np.round(pts[:, 0]).astype(int), minlength=img.shape[0])
    print("nodes per slice in range:",
          {z: int(per_z[z]) for z in planes if per_z[z]})


if __name__ == "__main__":
    sys.exit(main())
