"""Reading the real data, to calibrate the generator against it.

The generator is only useful if its cell counts and tissue geometry match what
the microscope produces, so both are measured rather than guessed."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import gaussian_filter, maximum_filter

from data_gen.constants import DOWNSAMPLE, EFF_VOXEL, POOL


def find_train_dir():
    for b in [
        Path(
            "../input/competitions/biohub-cell-tracking-during-development/train"
        ),
        Path("../input/biohub-cell-tracking-during-development/train"),
    ]:
        if b.exists() and any(b.glob("*.zarr")):
            return b
    for p in Path("../input").rglob("train"):
        if p.is_dir() and any(p.glob("*.zarr")):
            return p
    return None


def read_frame(zp, t):
    meta = json.loads((zp / "0" / "zarr.json").read_text())
    shape = tuple(int(v) for v in meta["shape"])
    dtype = np.dtype(meta["data_type"])
    fs = shape[1:]
    try:
        import blosc2

        a = np.frombuffer(
            blosc2.decompress((zp / "0" / "c" / str(t) / "0" / "0" / "0").read_bytes()),
            dtype=dtype,
        )
        if a.size == int(np.prod(fs)):
            return a.reshape(fs).copy(), shape
    except Exception:
        pass
    import zarr

    return np.asarray(zarr.open(zp / "0", mode="r")[t]), shape


def norm(f):
    lo, hi = np.percentile(f, 1), np.percentile(f, 99.5)
    return np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def geff_for(td_, name):
    for c in [td_ / f"{name}.geff", td_ / "gt" / f"{name}.geff"]:
        if c.exists():
            return c
    h = list(td_.rglob(f"{name}*.geff"))
    return h[0] if h else None


def read_geff_nodes(gp):
    import tracksdata as td

    g = td.graph.IndexedRXGraph.from_geff(gp)
    g = g[0] if isinstance(g, tuple) else g
    na = g.node_attrs(attr_keys=["t", "z", "y", "x"])
    by_t = {}
    for r in na.iter_rows(named=True):
        by_t.setdefault(int(r["t"]), []).append(
            [float(r["z"]), float(r["y"]), float(r["x"])]
        )
    return {t: np.array(v) for t, v in by_t.items()}


def pool_xy(f):
    return f[:: DOWNSAMPLE[0], :: DOWNSAMPLE[1], :: DOWNSAMPLE[2]].astype(np.float32)


def dog_response(vol, scales=((1.5, 4.0), (2.2, 5.5))):
    dog = None
    for ss, sl in scales:
        r = gaussian_filter(vol, tuple(ss / EFF_VOXEL)) - gaussian_filter(
            vol, tuple(sl / EFF_VOXEL)
        )
        dog = r if dog is None else np.maximum(dog, r)
    return dog


def dog_detect_precise(vol, rel_thr=0.35):
    dog = dog_response(vol)
    size = tuple(np.maximum(1, (2 * np.round(3.0 / EFF_VOXEL) + 1)).astype(int))
    mx = maximum_filter(dog, size=size, mode="nearest")
    return np.argwhere((dog == mx) & (dog >= dog.max() * rel_thr)), dog


def real_stats(train_dir, videos, n_frames=2, max_shell_frames=8):
    """Per-frame cell counts and the tissue shell, measured on real volumes.

    Counts drive how many cells a synthetic field gets; the shell drives where
    they are placed. Both come from a blob detector rather than the ground
    truth, which annotates only a few percent of the cells and would
    under-count by an order of magnitude.
    """
    from data_gen.geometry import fit_shell

    counts, pts = [], []
    for name in videos:
        for t in range(n_frames):
            vol = pool_xy(norm(read_frame(Path(train_dir) / f"{name}.zarr", t)[0]))
            det = dog_detect_precise(vol)[0]
            counts.append(len(det))
            if len(det) and len(pts) < max_shell_frames:
                pts.append(np.atleast_2d(det) * np.array([1, POOL, POOL]))
    shell = fit_shell(np.concatenate(pts)) if pts else None
    return np.array(counts, float), shell
