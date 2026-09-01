"""Measuring the gap between generated and real volumes.

Every number in ``constants.py`` claims to match the real acquisition. This is
what checks the claim: one report, computed the same way on a real frame and on
a generated one, so the two are directly comparable.

    python -m data_gen.calibrate --real 40 --synth ../input/synthetic/v2

The statistics are the ones a detector is sensitive to, not the ones that are
easy to compute:

``contrast``      a cell's peak over the background 7-12 um around it, the
                  quantity that decides whether a blob detector -- or a network
                  -- can separate it from the tissue it sits in. On the real
                  data the median is 2.2 and a quarter of annotated cells sit
                  below 1.0, i.e. they are less than twice their surroundings.
``local_bg``      that background. Real cells sit on a bright pedestal of
                  out-of-focus and scattered light (median 0.25 of full scale),
                  not on black.
``fwhm_z``        axial extent, which the axial PSF and scattering set. Real
                  median 11.4 um against a ~9 um nucleus: cells are smeared.
``octant``        brightest octant over dimmest, i.e. how unevenly the frame is
                  lit. Real median 8.1.
``frac_dark``     share of the frame below 0.05, i.e. how much of it is void.
``hf_dark``       high-frequency noise measured in the dim third of the frame.

All of them are computed on the *normalised* volume (1st to 99.5th percentile),
which is what the training pipeline feeds the network, so a generator that
matches here matches where it counts.

One trap, which cost a calibration round: the real ground truth annotates ~3%
of the cells, and that 3% is not a random sample -- it is the cells somebody
could see and follow, so it skews bright. Comparing it against *every*
generated cell, faint population included, makes any honest generator look too
dim and invites cranking the brightness back up. So the per-cell statistics are
reported twice: over all labelled cells, and over the subset a blob detector
finds (``det_`` prefix). The second applies the same selection to both sides
and is the one to calibrate against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from data_gen.constants import POOL, VOXEL_NATIVE
from data_gen.real_stats import (
    dog_detect_precise, find_train_dir, norm, pool_xy, read_frame,
)

# Percentiles every distribution is summarised by. Three numbers, because a
# median alone hides exactly the thing that is wrong with a generator: too
# narrow a spread.
PCTS = (25, 50, 75)

# The annulus a cell's local background is read from. Inside 7 um is still the
# cell (a nucleus is ~9 um across); beyond 12 um is a different cell.
BG_RING_UM = (7.0, 12.0)


def _q(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [round(float(v), 3) for v in np.percentile(a, PCTS)] if len(a) else None


def cell_appearance(vol, cens):
    """Per-cell peak, local background, contrast and extent, on a normalised volume.

    ``vol`` is native (Z, Y, X) in [0, 1]; ``cens`` is (N, 3) in native voxels.
    Cells within 30 voxels of a y/x border are skipped -- their background ring
    would fall outside the frame and read as spuriously dark.
    """
    Z, Y, X = vol.shape
    yy, xx = np.mgrid[-30:31, -30:31] * VOXEL_NATIVE[1]
    rad = np.hypot(yy, xx)
    ring = (rad >= BG_RING_UM[0]) & (rad <= BG_RING_UM[1])
    core = rad <= 2.0
    rows = []
    for c in np.atleast_2d(np.asarray(cens, float)):
        z, y, x = np.round(c[:3]).astype(int)
        if not (0 <= z < Z and 30 <= y < Y - 30 and 30 <= x < X - 30):
            continue
        sub = vol[z, y - 30:y + 31, x - 30:x + 31]
        peak = float(sub[core].max())
        bg = float(np.median(sub[ring]))
        half = bg + 0.5 * (peak - bg)

        def extent(profile, i):
            """Width of the run of above-half-maximum voxels through the centre."""
            above = profile >= half
            if not above[i]:
                return np.nan
            lo = hi = i
            while lo > 0 and above[lo - 1]:
                lo -= 1
            while hi < len(above) - 1 and above[hi + 1]:
                hi += 1
            return hi - lo + 1

        rows.append([
            peak, bg, (peak - bg) / max(bg, 1e-3),
            extent(vol[z, y, x - 30:x + 31], 30) * VOXEL_NATIVE[2],
            extent(vol[:, y, x], z) * VOXEL_NATIVE[0],
        ])
    return np.array(rows, float).reshape(-1, 5)


def detected(vol, cens, tol_um=4.0):
    """Mask over ``cens``: which of those cells a blob detector finds.

    The same selection on both sides of the comparison: it stands in for
    whatever made somebody annotate one real cell and not its neighbour.
    """
    cens = np.atleast_2d(np.asarray(cens, float)).reshape(-1, 3)
    det = dog_detect_precise(pool_xy(vol))[0]
    if len(cens) == 0 or len(det) == 0:
        return np.zeros(len(cens), bool)
    d_um = det.astype(float) * np.array([1, POOL, POOL]) * VOXEL_NATIVE
    D = np.linalg.norm(cens[:, None] * VOXEL_NATIVE - d_um[None], axis=2)
    return D.min(1) <= tol_um


def frame_appearance(vol):
    """Whole-frame statistics: exposure, unevenness, noise, detected density."""
    Z, Y, X = vol.shape
    octants = [
        vol[a:b, c:d, e:f].mean()
        for a, b in ((0, Z // 2), (Z // 2, Z))
        for c, d in ((0, Y // 2), (Y // 2, Y))
        for e, f in ((0, X // 2), (X // 2, X))
    ]
    # High-frequency noise, read where there is no cell to confuse it with
    # texture: the dim third of the frame.
    hf = vol - gaussian_filter(vol, (0.5, 1.5, 1.5))
    dark = vol < np.percentile(vol, 30)
    lf = gaussian_filter(vol, (2, 20, 20))
    det = dog_detect_precise(pool_xy(vol))[0]
    nn = np.nan
    if len(det) > 3:
        d_um = det.astype(float) * np.array([1, POOL, POOL]) * VOXEL_NATIVE
        D = np.linalg.norm(d_um[:, None] - d_um[None], axis=2)
        nn = float(np.median((D + np.eye(len(d_um)) * 1e9).min(1)))
    return dict(
        img_p50=float(np.percentile(vol, 50)),
        img_p90=float(np.percentile(vol, 90)),
        frac_dark=float((vol < 0.05).mean()),
        lowfreq_cv=float(lf.std() / (lf.mean() + 1e-9)),
        octant=float(max(octants) / max(min(octants), 1e-9)),
        hf_dark=float(hf[dark].std()),
        n_det=int(len(det)),
        nn_det_um=nn,
    )


CELL_KEYS = ("peak", "local_bg", "contrast", "fwhm_xy_um", "fwhm_z_um")


def _summarise(frames, cells, det_cells):
    cells = np.concatenate(cells) if cells else np.zeros((0, 5))
    det_cells = np.concatenate(det_cells) if det_cells else np.zeros((0, 5))
    out = {k: _q([f[k] for f in frames]) for k in frames[0]} if frames else {}
    for i, name in enumerate(CELL_KEYS):
        out[name] = _q(cells[:, i])
        out["det_" + name] = _q(det_cells[:, i])
    out["frac_contrast_below_1"] = (
        round(float((cells[:, 2] < 1.0).mean()), 3) if len(cells) else None
    )
    out["dog_recall"] = (
        round(len(det_cells) / len(cells), 3) if len(cells) else None
    )
    out["n_cells_measured"] = int(len(cells))
    out["n_frames"] = len(frames)
    return out


def report_real(train_dir=None, n_videos=40, frames=(0, 20, 50), n_jobs=8):
    """The report over a sample of real videos, using the sparse ground truth."""
    import tracksdata as td

    train_dir = Path(train_dir or find_train_dir())
    names = sorted(p.stem for p in train_dir.glob("*.zarr"))
    rng = np.random.default_rng(0)
    picks = rng.choice(names, size=min(n_videos, len(names)), replace=False)

    def one(name):
        try:
            g = td.graph.IndexedRXGraph.from_geff(train_dir / f"{name}.geff")
            g = g[0] if isinstance(g, tuple) else g
            gt = g.node_attrs(attr_keys=["t", "z", "y", "x"]).to_numpy().astype(float)
        except Exception:
            gt = np.zeros((0, 4))
        fr, ce, de = [], [], []
        for t in frames:
            try:
                vol = norm(read_frame(train_dir / f"{name}.zarr", t)[0])
            except Exception:
                continue
            cens = gt[gt[:, 0] == t][:, 1:4]
            fr.append(frame_appearance(vol))
            ce.append(cell_appearance(vol, cens))
            de.append(cell_appearance(vol, cens[detected(vol, cens)]))
        return fr, ce, de

    from joblib import Parallel, delayed

    res = Parallel(n_jobs=n_jobs)(delayed(one)(n) for n in picks)
    return _summarise(*[[x for r in res for x in r[i]] for i in range(3)])


def report_synth(root, n_files=8, frames=(0, 3)):
    """The same report on generated sequences, using their complete labels."""
    root = Path(root)
    files = sorted((root / "sequences").glob("seq_*.npz")) or sorted(
        root.glob("seq_*.npz")
    )
    fr, ce, de = [], [], []
    for path in files[:n_files]:
        z = np.load(path, allow_pickle=True)
        nodes = z["nodes"]
        for t in frames:
            if t >= len(z["volumes"]):
                continue
            vol = norm(z["volumes"][t].astype(np.float32) / 65535.0)
            cens = nodes[nodes[:, 0] == t][:, 1:4]
            fr.append(frame_appearance(vol))
            ce.append(cell_appearance(vol, cens))
            de.append(cell_appearance(vol, cens[detected(vol, cens)]))
    return _summarise(fr, ce, de)


def print_table(real, synth):
    keys = [
        "det_contrast", "det_local_bg", "det_peak", "det_fwhm_xy_um",
        "det_fwhm_z_um", "dog_recall",
        "contrast", "local_bg", "peak", "frac_contrast_below_1",
        "fwhm_xy_um", "fwhm_z_um", "img_p50", "img_p90", "frac_dark",
        "octant", "lowfreq_cv", "hf_dark", "n_det", "nn_det_um",
    ]
    w = max(len(k) for k in keys)
    print(f"{'statistic':<{w}}  {'real p25/p50/p75':<26}  synthetic")
    print("-" * (w + 56))
    for k in keys:
        fmt = lambda v: (
            " / ".join(f"{x:g}" for x in v) if isinstance(v, list)
            else ("-" if v is None else f"{v:g}")
        )
        print(f"{k:<{w}}  {fmt(real.get(k)):<26}  {fmt(synth.get(k))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", type=int, default=40, help="real videos to sample")
    ap.add_argument("--synth", type=str, default=None, help="generated dataset dir")
    ap.add_argument("--synth-files", type=int, default=8)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    real = report_real(n_videos=args.real)
    synth = report_synth(args.synth, n_files=args.synth_files) if args.synth else {}
    print_table(real, synth)
    if args.json:
        json.dump({"real": real, "synthetic": synth}, open(args.json, "w"), indent=1)
