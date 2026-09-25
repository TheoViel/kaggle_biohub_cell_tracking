#!/usr/bin/env python
"""Draw pseudo-labelled nodes over the crops they were predicted on.

Sources may be ``.npz`` (this repo's predict output) or ``.geff`` (the public
notebook's), named as ``LABEL=PATH``. Every panel of a figure is the same crop,
frame and z-slice, so two models are compared on identical pixels rather than
on whichever crops each happened to like.

The image is shown through the *same* normalisation the detector saw -- the
per-crop quantiles stored in the zarr attrs -- rather than a display stretch,
so a node that looks unsupported here was unsupported for the model too.

Two thresholds side by side answers the question the counts cannot: whether the
low-threshold extras are real cells the strict pass missed, or noise.

    python scripts/plot_pseudo_labels.py --pl output/pl_ultrack_0.60 output/pl_ultrack_0.30
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

# The crops every comparison is drawn on. Pinned, not picked by density: a set
# that shifts with whichever model happens to be in the run makes two figures
# from different days impossible to lay side by side. Change this only if you
# intend to invalidate every earlier render.
CANONICAL_CROPS = (
    "ultr_0000_0000_0512_0512",    # noise / intensity ramp, no cells
    "ultr_0000_0000_0768_1280",
    "ultr_0000_0000_1024_1280",
    "ultr_0000_0064_1024_0256",    # the z-pair crop
    "ultr_0000_0192_1280_0512",
    "ultr_0000_0256_0000_1024",
)

# Same reason for the z-plane: anchor it on one fixed prediction set (13/1 det
# @0.4) rather than on the densest source in this run.
Z_REFERENCE = REPO / "output" / "pl_ultrack_0.40"


def load_nodes(source: Path, name: str, t: int) -> np.ndarray:
    """(N, 3) z/y/x of one frame, from either prediction format."""
    npz = source / f"{name}.npz"
    if npz.exists():
        c = np.load(npz)["coords"].astype(float)
        return c[c[:, 0] == t][:, 1:4]
    geff = source / f"{name}.geff"
    if not geff.exists():
        return np.zeros((0, 3))
    import tracksdata as td
    g = td.graph.IndexedRXGraph.from_geff(geff)
    g = g[0] if isinstance(g, tuple) else g
    a = g.node_attrs(attr_keys=["t", "z", "y", "x"])
    m = np.asarray(a["t"]) == t
    return np.stack([np.asarray(a[c])[m] for c in ("z", "y", "x")], axis=1)


def load_frame(crop_path: Path, t: int) -> np.ndarray:
    """One frame, normalised exactly as the loader does it (scope='video')."""
    group = zarr.open_group(crop_path, mode="r")
    raw = np.asarray(group["0"][t : t + 1])
    quantiles = group.attrs["image_statistics"]["quantiles"]
    return normalize_frames(raw, resolve_normalization(None), quantiles)[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crops", default=str(REPO / "input" / "ultrack_crops"))
    p.add_argument("--pl", nargs="+", required=True,
                   help="LABEL=PATH per source; .npz and .geff both accepted")
    p.add_argument("--out", default=str(REPO / "output" / "plots"))
    p.add_argument("--names", nargs="+", default=list(CANONICAL_CROPS),
                   help="crops to draw; defaults to the pinned canonical set")
    p.add_argument("--z-ref", default=str(Z_REFERENCE),
                   help="prediction dir the z-plane is taken from, so it is the "
                        "same slice whatever is being compared")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--tol", type=float, default=5, help="slices either side drawn, faded")
    p.add_argument("--prefix", default="pl", help="output filename prefix")
    args = p.parse_args()

    crops_dir = Path(args.crops)
    labelled = [(d.split("=", 1) if "=" in d else (Path(d).name, d)) for d in args.pl]
    labels = [lab for lab, _ in labelled]
    pl_dirs = [Path(path) for _, path in labelled]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Every source must have predicted on every pinned crop; a silently dropped
    # panel is exactly the comparison this script exists to avoid.
    per_source = [{f.stem for f in d.glob("*.npz")} | {f.stem for f in d.glob("*.geff")}
                  for d in pl_dirs]
    have = set.intersection(*per_source)
    picks = list(args.names)
    missing = [n for n in picks if n not in have]
    if missing:
        raise SystemExit(f"missing from at least one source: {missing}")

    z_ref = Path(args.z_ref)

    for name in picks:
        frame = load_frame(crops_dir / f"{name}.zarr", args.frame)
        per_dir = [load_nodes(d, name, args.frame) for d in pl_dirs]
        # Fixed slice from the reference set, so this crop is always shown at
        # the same depth no matter which models are in the figure.
        ref = load_nodes(z_ref, name, args.frame)
        z_plane = int(round(float(np.median(ref[:, 0])))) if len(ref) else frame.shape[0] - 1
        vmax = float(np.quantile(frame, 0.999)) or None

        fig, axes = plt.subplots(1, len(pl_dirs), figsize=(5.6 * len(pl_dirs), 5.8))
        axes = np.atleast_1d(axes)
        for ax, d, pts in zip(axes, labels, per_dir):
            ax.imshow(frame[z_plane], cmap="gray", vmax=vmax)
            drawn = scatter_by_depth(ax, pts, z_plane, "#35d0ba", tol=args.tol,
                                     facecolor="none", s=46, linewidths=1.3)
            ax.set_title(f"{d}   {len(pts)} nodes in frame, "
                         f"{int(drawn.sum())} within {args.tol:g} slices", fontsize=9)
            ax.axis(False)
        fig.suptitle(f"{name}   t={args.frame}  z={z_plane}", fontsize=11)
        fig.tight_layout()
        path = out_dir / f"{args.prefix}_{name}.png"
        fig.savefig(path, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"  {path.name}  ({', '.join(str(len(p)) for p in per_dir)} nodes)")

    print(f"\n{len(picks)} images -> {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
