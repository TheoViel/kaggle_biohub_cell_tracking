"""Where cells sit: tissue shell geometry and position sampling."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from data_gen.acquisition import reserved_um, sample_aniso
from data_gen.constants import (
    SHELL_DENS_SMOOTH, SHELL_EDGE_FRAC, SHELL_RMIN_FOV, VOXEL_NATIVE,
)


def fit_shell(pts_vox, voxel=None):
    """Fits the shell (sphere OR plane) to points in NATIVE voxels. Returns a dict with the geometry in um.
    `pts_vox` = real detections/centroids (N,3) in (z,y,x) native voxel."""
    voxel = VOXEL_NATIVE if voxel is None else np.asarray(voxel, float)
    P = np.atleast_2d(np.asarray(pts_vox, float)) * voxel  # -> um (isotropic)
    n = len(P)
    if n < 12:
        return dict(kind="none", n=n)
    c0 = P.mean(0)
    Q = P - c0
    # --- SPHERE attempt ---
    A = np.concatenate([2.0 * Q, np.ones((n, 1))], 1)
    b = (Q**2).sum(1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cc = sol[:3]
        k = sol[3]
        R2 = k + float(cc @ cc)
        R = float(np.sqrt(R2)) if R2 > 0 else -1.0
        center = c0 + cc
        rad = np.linalg.norm(P - center, axis=1)
        resid = float(np.std(rad - R)) if R > 0 else np.inf
    except Exception:
        R, resid, center = -1.0, np.inf, c0
    # --- PLANE alternative (R->inf limit) via PCA ---
    U, S, Vt = np.linalg.svd(Q - Q.mean(0), full_matrices=False)
    normal = Vt[-1] / (np.linalg.norm(Vt[-1]) + 1e-12)
    off = (P - c0) @ normal
    resid_plane = float(np.std(off))
    # lateral extent of the tissue (to know whether it covers the whole field or just a band)
    span = np.percentile(P, [2, 98], axis=0)
    # CHOICE: the sphere only wins if it is well conditioned (finite R, not absurd) AND fits better
    fov = float(np.linalg.norm(P.max(0) - P.min(0)) + 1e-9)
    # PHYSICAL RADIUS FLOOR (defect found IN THE FIGURE, invisible in the KS metrics): the fit returned
    # R=50um in a 104um field -- a sphere SMALLER than the field of view, which fits entirely inside the volume.
    # Cells on such a shell form a HOLLOW BALL and the Z projection comes out as a RING with a dark centre.
    # The real data NEVER has a central void: its edge is almost straight, i.e. R >> field. The fit was
    # OVERFITTING local curvature from a few detections. Now the sphere is only accepted if it is
    # physically plausible (radius >= 1.5x the field); below that we use the PLANE (R->inf limit).
    R_MIN = SHELL_RMIN_FOV * fov
    sphere_ok = (
        (R > R_MIN)
        and (R < 60.0 * fov)
        and np.isfinite(resid)
        and (resid < 0.92 * resid_plane)
    )
    if sphere_ok:
        return dict(
            kind="sphere",
            center_um=center,
            R_um=R,
            thick_um=max(resid, 1.0),
            n=n,
            resid=resid,
            resid_plane=resid_plane,
            span_um=span,
        )
    return dict(
        kind="plane",
        point_um=c0,
        normal=normal,
        thick_um=max(resid_plane, 1.0),
        n=n,
        resid=float(resid) if np.isfinite(resid) else -1.0,
        resid_plane=resid_plane,
        span_um=span,
    )


def _density_field(rng, shape_zyx, smooth_vox=14.0, contrast=1.0):
    """Smooth density field in [0,1]: more/less populated regions along the tissue (the real data is not
    uniform within the band). Blurred white noise = low-frequency field."""
    f = rng.normal(0, 1, shape_zyx).astype(np.float32)
    f = gaussian_filter(f, smooth_vox)
    f -= f.min()
    f /= f.max() + 1e-9
    return np.clip(0.5 + contrast * (f - 0.5), 0.0, 1.0)


def sample_shell_positions(
    n_cells,
    shape_zyx,
    rng,
    shell=None,
    r_lo=2.5,
    r_hi=4.5,
    csf=1.0,
    jitter=0.25,
    dens_contrast=1.0,
):
    """Places `n_cells` on a SHELL (sphere/plane) crossing the volume, with a density gradient and a
    collision check. `shell` = output of fit_shell (geometry measured on the real data); None -> draws one.

    Returns ``(pos_voxel (N,3), radii_um (N,), aniso (N,3))``. The anisotropy is
    drawn here rather than left to ``Palette.sample`` so that the collision
    check can reserve the room the cell will actually take -- see
    ``acquisition.reserved_um``. Drawing it twice, independently, is the same
    bug the radii used to have, and it has the same symptom: the shape that was
    packed is not the shape that is rendered.
    """
    Z, Y, X = shape_zyx
    lo = np.array([4.0, 12.0, 12.0])
    hi = np.array([Z - 4.0, Y - 12.0, X - 12.0])
    vox = VOXEL_NATIVE
    ext_um = np.array([Z, Y, X]) * vox  # field in um
    # ---- geometry: uses the measured one OR draws a plausible one (variation across embryos/frames) ----
    if shell is None or shell.get("kind") not in ("sphere", "plane"):
        kind = "sphere" if rng.random() < 0.65 else "plane"
        R = float(rng.uniform(1.2, 4.0) * float(ext_um.max()))  # gentle curvature
        nrm = rng.normal(0, 1, 3)
        nrm /= np.linalg.norm(nrm) + 1e-9
        cen = ext_um * 0.5 + nrm * R * float(
            rng.uniform(0.75, 1.05)
        )  # centre OUTSIDE the field
        shell = dict(
            kind=kind,
            center_um=cen,
            R_um=R,
            point_um=ext_um * rng.uniform(0.35, 0.65, 3),
            normal=nrm,
            thick_um=float(rng.uniform(6.0, 16.0)),
        )
    else:
        shell = dict(shell)
        # PERTURBS the measured geometry -> a NEW shell (generative), not the same one as the real frame
        j = float(jitter)
        if shell["kind"] == "sphere":
            shell["R_um"] = (
                float(shell["R_um"] * rng.normal(1.0, 0.10 * j / 0.25 if j else 0.0))
                if j
                else shell["R_um"]
            )
            shell["center_um"] = np.asarray(shell["center_um"], float) + rng.normal(
                0, j * 12.0, 3
            )
        else:
            nrm = np.asarray(shell["normal"], float) + rng.normal(0, j * 0.18, 3)
            shell["normal"] = nrm / (np.linalg.norm(nrm) + 1e-9)
            shell["point_um"] = np.asarray(shell["point_um"], float) + rng.normal(
                0, j * 10.0, 3
            )
        # EDGE REGIME (defect seen in figure V01): the real data has TWO regimes -- fields INSIDE the
        # tissue, which fill the frame, and fields at the EDGE of the embryo, with a band of cells and a
        # large dark void on the other side. Our shell came from the calibration (interior videos), with
        # the normal along Z -> the slab ends up parallel to the image and ALWAYS covers everything in projection.
        # To get the BAND, the normal has to lie IN THE PLANE of the image (slab crossing edge-on).
        # Here a fraction of the volumes gets that reorientation, reproducing the edge regime.
        if rng.random() < SHELL_EDGE_FRAC:
            ang = rng.uniform(0, 2 * np.pi)
            n_edge = np.array(
                [rng.uniform(-0.25, 0.25), np.cos(ang), np.sin(ang)], float
            )
            shell = dict(
                shell,
                kind="plane",
                normal=n_edge / (np.linalg.norm(n_edge) + 1e-9),
                point_um=np.array([Z, Y, X], float)
                * VOXEL_NATIVE
                * rng.uniform(0.25, 0.75, 3),
                thick_um=float(rng.uniform(14.0, 34.0)),
            )
        shell["thick_um"] = float(max(3.0, shell["thick_um"] * rng.normal(1.0, 0.15)))
    thick = float(shell["thick_um"])
    dens = _density_field(
        rng,
        (Z, Y, X),
        smooth_vox=SHELL_DENS_SMOOTH,
        contrast=dens_contrast,
    )
    # ---- vectorized PROBABILITY MAP (replaces pure rejection, which was fragile and slow):
    # weight(voxel) = gaussian(distance to the surface) * density. Sampling from this map GUARANTEES that the
    # cells land on the shell and covers the whole volume, without blowing through attempts.
    gz, gy, gx = np.meshgrid(
        np.arange(Z, dtype=np.float32),
        np.arange(Y, dtype=np.float32),
        np.arange(X, dtype=np.float32),
        indexing="ij",
    )
    QZ, QY, QX = gz * vox[0], gy * vox[1], gx * vox[2]

    def _dist_map(sh):
        if sh["kind"] == "sphere":
            c = np.asarray(sh["center_um"], float)
            return np.abs(
                np.sqrt((QZ - c[0]) ** 2 + (QY - c[1]) ** 2 + (QX - c[2]) ** 2)
                - float(sh["R_um"])
            )
        p0 = np.asarray(sh["point_um"], float)
        nn = np.asarray(sh["normal"], float)
        return np.abs(
            (QZ - p0[0]) * nn[0] + (QY - p0[1]) * nn[1] + (QX - p0[2]) * nn[2]
        )

    D = _dist_map(shell)
    # RE-ANCHORING: if the shell does not cross the volume (everything far from the surface), it would produce ZERO
    # cells. Shifts the geometry so the surface passes through the volume, preserving the CURVATURE.
    if float(D.min()) > 1.5 * thick:
        ctr_um = np.array([Z, Y, X], float) * vox * 0.5
        if shell["kind"] == "sphere":
            c = np.asarray(shell["center_um"], float)
            u = ctr_um - c
            nu = np.linalg.norm(u) + 1e-9
            shell["center_um"] = ctr_um - (u / nu) * float(
                shell["R_um"]
            )  # the surface passes through the centre
        else:
            shell["point_um"] = ctr_um
        D = _dist_map(shell)
    W = np.exp(-0.5 * (D / (0.5 * thick + 1e-9)) ** 2) * dens
    # zeroes the borders (the cell has to fit entirely inside the volume)
    W[: int(lo[0]), :, :] = 0
    W[int(hi[0]) :, :, :] = 0
    W[:, : int(lo[1]), :] = 0
    W[:, int(hi[1]) :, :] = 0
    W[:, :, : int(lo[2])] = 0
    W[:, :, int(hi[2]) :] = 0
    tot = float(W.sum())
    if tot <= 0:  # degenerate -> uniform
        W = np.zeros_like(W)
        W[int(lo[0]) : int(hi[0]), int(lo[1]) : int(hi[1]), int(lo[2]) : int(hi[2])] = (
            1.0
        )
        tot = float(W.sum())
    p = (W.ravel() / tot).astype(np.float64)
    # OVERSAMPLE: draws more candidates than needed, since collisions will discard some
    ncand = int(min(len(p), max(8 * n_cells, 4000)))
    idx = rng.choice(len(p), size=ncand, replace=True, p=p)
    cz, cyx = np.divmod(idx, Y * X)
    cy, cx = np.divmod(cyx, X)
    cand = np.stack([cz, cy, cx], 1).astype(np.float32)
    cand += rng.uniform(-0.5, 0.5, cand.shape)  # sub-voxel jitter
    cand = np.clip(cand, lo, hi)
    # greedy acceptance with a collision check (same rule as the rest of the generator)
    cand_r = rng.uniform(r_lo, r_hi, len(cand)).astype(np.float32)
    cand_a = sample_aniso(len(cand), rng)
    cand_res = reserved_um(cand_r, cand_a)
    pos, radii, aniso, reserved = [], [], [], []
    for p_i, r, a, res in zip(cand, cand_r, cand_a, cand_res):
        if len(pos) >= n_cells:
            break
        if pos:
            dd = np.linalg.norm((np.array(pos) - p_i) * vox, axis=1)
            if (dd < csf * (res + np.array(reserved))).any():
                continue
        pos.append(p_i)
        radii.append(float(r))
        aniso.append(a)
        reserved.append(float(res))
    return (np.array(pos, np.float32).reshape(-1, 3),
            np.array(radii, np.float32),
            np.array(aniso, np.float32).reshape(-1, 3))
