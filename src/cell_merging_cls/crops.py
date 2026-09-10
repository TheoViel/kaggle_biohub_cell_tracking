"""Batched crop extraction with a *per-sample* rotation.

`embeddings.patches.extract_patches` takes one `angle_deg` for the whole call,
which is right everywhere it is used: the angle is an augmentation draw shared
by a clip. Here the angle is a property of the sample -- each pair is rotated
onto its own axis -- so one call per pair would be the only option, and there
are ~6.5M pairs to score across the 199 stacks. This does the same
nearest-neighbour sampling with the rotation broadcast over the batch, which is
the difference between minutes and days.

Sampling matches `extract_patches` exactly, nearest-neighbour included: a
training crop and an inference crop of the same place must be the same array.
"""

from __future__ import annotations

import numpy as np


def pair_patches(volume, centres, angles_deg, patch_zyx, chunk=2048):
    """`(N, pz, py, px)` crops, each rotated by its own angle.

    Parameters
    ----------
    volume : (Z, Y, X) float32
        One normalised frame.
    centres : (N, 3) float
        z / y / x in image voxels, fractional allowed.
    angles_deg : (N,) float
        Per-sample in-plane rotation, in the same convention as
        `extract_patches`: patch offset `(gy, gx)` is read at
        `(cos*gy - sin*gx, sin*gy + cos*gx)`.
    """
    pz, py, px = patch_zyx
    centres = np.asarray(centres, np.float32).reshape(-1, 3)
    angles = np.asarray(angles_deg, np.float32).reshape(-1)
    n = len(centres)
    Z, Y, X = volume.shape
    out = np.zeros((n, pz, py, px), np.float32)
    if n == 0:
        return out

    oz = np.arange(pz, dtype=np.float32) - (pz - 1) / 2.0
    oy = np.arange(py, dtype=np.float32) - (py - 1) / 2.0
    ox = np.arange(px, dtype=np.float32) - (px - 1) / 2.0
    gy, gx = np.meshgrid(oy, ox, indexing="ij")

    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        th = np.deg2rad(angles[lo:hi])[:, None, None]
        cos, sin = np.cos(th), np.sin(th)
        ry = cos * gy[None] - sin * gx[None]              # (m, py, px)
        rx = sin * gy[None] + cos * gx[None]

        # z does not depend on (y, x) and (y, x) do not depend on z, so the
        # index arrays stay (m, pz) and (m, py, px) and are broadcast by the
        # gather. Materialising three (m, pz, py, px) int arrays instead -- what
        # the shared-angle reference does -- costs more than the gather itself.
        iz = np.rint(centres[lo:hi, 0][:, None] + oz[None]).astype(np.int32)
        iy = np.rint(centres[lo:hi, 1][:, None, None] + ry).astype(np.int32)
        ix = np.rint(centres[lo:hi, 2][:, None, None] + rx).astype(np.int32)

        z_ok = ((iz >= 0) & (iz < Z))[:, :, None, None]
        yx_ok = ((iy >= 0) & (iy < Y) & (ix >= 0) & (ix < X))[:, None]

        out[lo:hi] = volume[
            np.clip(iz, 0, Z - 1)[:, :, None, None],
            np.clip(iy, 0, Y - 1)[:, None],
            np.clip(ix, 0, X - 1)[:, None],
        ] * (z_ok & yx_ok)
    return out
