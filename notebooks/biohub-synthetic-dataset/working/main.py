import os, json
import numpy as np
from scipy.ndimage import gaussian_filter

# NATIVE physical voxel (before pooling)
VOXEL_NATIVE = np.array([1.625, 0.40625, 0.40625])
POOL = 4  # XY downsample of the pipeline (DOWNSAMPLE=(1,4,4))


def _sphere_native(r_um, rng, sharp=7, ellip=0.5, frac=None):
    """PHYSICAL sphere of radius r_um, sampled on the anisotropic NATIVE grid (flattened in z). Super-gaussian
    profile (filled disc, sharp edge) + elliptical variation.
    `frac` = SUB-VOXEL offset (p - round(p)). Without it the cell can only sit centred on a WHOLE
    voxel and the centroid label comes out quantized (a systematic error of up to 0.5 native voxel, which the
    detector learns as noise in the target). With it the profile is sampled offset and the true
    (float) centroid is the label."""
    axes_vox = r_um / VOXEL_NATIVE  # semi-axes in native VOXELS (small z, large xy)
    aniso = rng.uniform(0.82, 1.20, 3)
    if rng.random() < ellip:
        aniso[rng.integers(3)] *= rng.uniform(1.4, 2.0)  # markedly elliptical
    axes = np.maximum(axes_vox * aniso, 0.6)
    size = (np.ceil(axes) * 2 + 3).astype(int)  # bounding box per axis
    zz, yy, xx = np.indices(tuple(size))
    c = size // 2
    fz, fy, fx = (
        (0.0, 0.0, 0.0)
        if frac is None
        else (float(frac[0]), float(frac[1]), float(frac[2]))
    )
    # rotation only in the XY plane (the z anisotropy is physical, it does not rotate along)
    th = rng.uniform(0, np.pi)
    ct, stt = np.cos(th), np.sin(th)
    dy = yy - c[1] - fy
    dx = xx - c[2] - fx
    dz = zz - c[0] - fz
    ry = ct * dy - stt * dx
    rx = stt * dy + ct * dx
    rn = np.sqrt((dz / axes[0]) ** 2 + (ry / axes[1]) ** 2 + (rx / axes[2]) ** 2)
    return np.exp(-(rn**sharp)).astype(np.float32)


# ============================================================================================
# EMBRYO SHELL (NEW algorithm) -- GENERATIVE placement with the spatial structure of the real data.
#
# PROBLEM it solves: the Thomas process (random clusters) scatters clumps through the whole
# volume; the real data has a CURVED BAND of tissue with a DARK VOID on one side (the blastoderm over the
# yolk crossing the field). Copying the real positions (`cens_native`) matches the distribution by
# construction but is NOT generative -- it reuses the few real frames.
# SOLUTION: fit the GEOMETRY (not the positions) of the shell from the real data -- centre/radius of a
# sphere + thickness -- and then SAMPLE new shells. The result is coherent, novel structure.
#
# Algebraic sphere fit (Pratt/Coope): for points p_i, |p_i|^2 = 2 p_i . c + (R^2 - |c|^2),
# linear in [c, k] with k = R^2-|c|^2 -> least squares. R = sqrt(k + |c|^2).
# PLANE LIMIT: a large embryo => R >> field of view and the sphere becomes a PLANE; the fit gets ill-
# conditioned, so we fall back to PCA (normal = eigenvector of smallest variance) -- this is the R->inf case.
# ============================================================================================


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
    R_MIN = float(os.environ.get("SYNTH_SHELL_RMIN_FOV", "1.5")) * fov
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
    Returns (pos_voxel (N,3), radii_um (N,))."""
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
        if rng.random() < float(os.environ.get("SYNTH_EDGE_FRAC", "0.35")):
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
        smooth_vox=float(os.environ.get("SYNTH_SHELL_DENS_SMOOTH", "14")),
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
    pos, radii = [], []
    for p_i, r in zip(cand, cand_r):
        if len(pos) >= n_cells:
            break
        if pos:
            dd = np.linalg.norm((np.array(pos) - p_i) * vox, axis=1)
            if (dd < csf * (r + np.array(radii))).any():
                continue
        pos.append(p_i)
        radii.append(float(r))
    return np.array(pos, np.float32).reshape(-1, 3), np.array(radii, np.float32)


def gen_volume_native(
    zshape=64,
    xyshape=256,
    n_cells=200,
    rng=None,
    cens_native=None,
    medium=None,
    min_sep_um=None,
    shell=None,
):
    """Generates a NATIVE volume (zshape, xyshape, xyshape): dark background + physical spheres + PSF + noise.
    Returns (vol_native, vol_pooled, centroids_native, centroids_pooled, instance_mask_native).
    Instance labels are one-based and correspond to rows in centroids_native. cens_native
    (optional): real positions.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    medium = (
        float(os.environ.get("SYNTH_MEDIUM_LEVEL", "0.04"))
        if medium is None
        else medium
    )
    # RADIUS calibrated by DETECTABILITY (not by eye): with 2.5-4.5 (diam 7um) the DoG found only 0.811
    # of the PLACED cells, whereas on the REAL data it finds 0.91-0.94 -> the synthetic was ~20% HARDER
    # than reality (it teaches the detector to expect smaller cells than exist). With 3.5-5.5
    # (diam 9um) recall goes to 0.913 = matches the real data, AND it matches the EDA (real diameter ~9.5um). Larger radii
    # (diam 10-11um) give recall 0.94-0.96, which EXCEEDS the real data -- a synthetic easier than
    # reality is worse for training. The separation the user wanted is preserved: NN 11.6um vs real ~10-11.
    r_lo = float(os.environ.get("SYNTH_CELL_RUM_LO", "3.5"))  # radius (um)
    r_hi = float(os.environ.get("SYNTH_CELL_RUM_HI", "5.5"))
    sharp = float(
        os.environ.get("SYNTH_CELL_SHARPNESS", "4.5")
    )  # calibrated: real peak/mean relative brightness
    psf_xy = float(
        os.environ.get("SYNTH_PSF_XY_UM", "0.55")
    )  # lateral PSF (um) ~ confocal/light-sheet
    psf_z_ratio = float(
        os.environ.get("SYNTH_PSFZ", "2.5")
    )  # larger axial PSF (real optical anisotropy)
    # COLLISION-AWARE SEPARATION: two cells are only accepted if the centre-to-centre distance
    # (um) >= FACTOR * (radius_i + radius_j). That way they TOUCH but do not INTERPENETRATE. Before, min_sep
    # was FIXED at 7um -> cells ~10um in diameter overlapped by ~3um, and as the size grew the
    # overlap exploded (cells collided, the DoG merged them -> nearest-neighbour ruined).
    # factor 1.0 = tangent (no overlap); <1 = light contact (dense real tissue); >1 = clearance.
    csf = float(os.environ.get("SYNTH_COLLISION_FACTOR", "1.0"))

    Z, Y, X = zshape, xyshape, xyshape
    vol = np.clip(rng.normal(medium, medium * 0.25, (Z, Y, X)), 0, 1).astype(
        np.float32
    )  # dark LIQUID
    instance_mask = np.zeros((Z, Y, X), dtype=np.uint16)

    # positions + RADII of the cells. real cens OR size-aware placement (CLUSTERED or uniform).
    # CLUSTERING (SYNTH_CLUSTER=1, default): the real embryo does NOT have uniformly random cells -- they
    # form GROUPS/clusters (dense and empty regions). Thomas process: draws cluster CENTRES
    # and places most of the cells around them (gaussian), + a uniform background fraction. This gives
    # the spatial structure (clustering patterns) that uniform random placement never reproduces.
    place_mode = os.environ.get(
        "SYNTH_PLACE_MODE", "cluster"
    )  # cluster (Thomas) | shell | uniform
    if cens_native is not None and len(cens_native) >= 5:
        pos = np.atleast_2d(cens_native).astype(np.float32)
        radii = rng.uniform(r_lo, r_hi, len(pos)).astype(np.float32)
    elif place_mode == "shell":
        # SHELL: geometry (curvature/thickness) measured on the real data, perturbed -> coherent and NOVEL layout
        pos, radii = sample_shell_positions(
            n_cells,
            (Z, Y, X),
            rng,
            shell=shell,
            r_lo=r_lo,
            r_hi=r_hi,
            csf=csf,
            jitter=float(os.environ.get("SYNTH_SHELL_JITTER", "0.25")),
            dens_contrast=float(os.environ.get("SYNTH_SHELL_DENS", "1.0")),
        )
    else:
        clustered = os.environ.get("SYNTH_CLUSTER", "1") == "1"
        ncl = max(
            3, int(n_cells * float(os.environ.get("SYNTH_CLUSTER_FRAC_N", "0.10")))
        )  # number of clusters
        cfrac = float(
            os.environ.get("SYNTH_CLUSTER_FRAC", "0.72")
        )  # fraction of cells in clusters
        spread_um = float(
            os.environ.get("SYNTH_CLUSTER_SPREAD_UM", "14.0")
        )  # cluster radius (um)
        centers = (
            rng.uniform([6, 16, 16], [Z - 6, Y - 16, X - 16], size=(ncl, 3)).astype(
                np.float32
            )
            if clustered
            else None
        )
        sp_vox = spread_um / VOXEL_NATIVE  # spread per axis (voxels)
        pos, radii = [], []
        tries = 0
        while len(pos) < n_cells and tries < n_cells * 60:
            tries += 1
            if clustered and rng.random() < cfrac:  # cell INSIDE a cluster
                cc = centers[rng.integers(ncl)]
                p = cc + rng.normal(0, sp_vox, 3).astype(np.float32)
                p = np.clip(p, [4, 12, 12], [Z - 4, Y - 12, X - 12])
            else:  # BACKGROUND cell (uniform)
                p = rng.uniform([4, 12, 12], [Z - 4, Y - 12, X - 12])
            r = float(rng.uniform(r_lo, r_hi))
            if pos:
                d = np.linalg.norm(
                    (np.array(pos) - p) * VOXEL_NATIVE, axis=1
                )  # um, centre-to-centre
                thr = csf * (r + np.array(radii))
                # `min_sep_um` was a DEAD parameter (accepted in the signature and never used -> whoever
                # passed it got a silent no-op). Now it is an absolute separation FLOOR (um).
                if min_sep_um:
                    thr = np.maximum(thr, float(min_sep_um))
                if (d < thr).any():
                    continue  # collides -> reject
            pos.append(p)
            radii.append(r)
        pos = np.array(pos, np.float32)
        radii = np.array(radii, np.float32)

    # EMISSION/SCATTERING: each nucleus emits light that scatters in the medium (PSF tail + scattering in the
    # liquid) -> a dim halo that fills the space between cells with the PURPLE glow of the real data. Modelled as
    # a 2nd layer: the CENTRE of the cell deposits a point source that is then blurred WIDE and summed
    # (accumulates where there are many cells = glow; dark where there are none). The SHARP nucleus goes by maximum.
    halo_amp = float(
        os.environ.get("SYNTH_HALO_AMP", "0.18")
    )  # deliberately LOW: it is the halo (not the brightness)
    #                                                                  that filled the gaps and merged the cells
    halo_um = float(
        os.environ.get("SYNTH_HALO_UM", "2.6")
    )  # 3.5 left the floor far from the
    #                                                     cell at 0.24 vs 0.16 of the real data        # scattering range (um)
    emis = np.zeros((Z, Y, X), np.float32)  # emission layer (summed)
    cens = []
    # per-cell BRIGHTNESS: VARIED and (mostly) BELOW saturation. It used to be uniform(0.72,1.0) and after
    # PSF+emission ALL of them hit 1.0 -> a degenerate peak distribution (a single spike at 1.0), unlike
    # the real data, which has varied brightness (long tail). Log-normal-ish: modest mean, tail upwards.
    # BRIGHTNESS: the REAL data SATURATES A LOT (~41% of cells peak at ~1.0, measured on the gallery histogram).
    # The error was NOT high brightness -- it was the SEPARATION. The "fused white mass" look came from the HALO
    # filling the gaps between cells, not from the brightness. Lowering the brightness to 0.35-0.78 left the synth
    # WASHED OUT (2.9% saturation vs 41% for the real data). The right config = HIGH brightness (matches the real saturation)
    # + LOW halo (keeps the gaps dark, cells distinct): 26.5% sat, recall 0.905.
    b_lo = float(os.environ.get("SYNTH_CELL_BRIGHT_LO", "0.42"))
    b_hi = float(os.environ.get("SYNTH_CELL_BRIGHT_HI", "0.92"))
    for cell_id, (p, r_um) in enumerate(
        zip(pos, radii), start=1
    ):  # radius ALREADY drawn at placement time (consistent)
        bright = float(
            np.clip(rng.normal((b_lo + b_hi) / 2, (b_hi - b_lo) / 3), b_lo, b_hi)
        )  # real variation (does not saturate everything)
        pr = np.round(p)
        frac = np.asarray(p, float) - pr  # SUB-VOXEL offset
        sphere = _sphere_native(float(r_um), rng, sharp=sharp, frac=frac)
        esf = sphere * bright  # SHARP nucleus (sub-voxel)
        rz, ry, rx = np.array(esf.shape) // 2
        z, y, x = pr.astype(int)
        z0, z1 = max(0, z - rz), min(Z, z + rz + 1)
        y0, y1 = max(0, y - ry), min(Y, y + ry + 1)
        x0, x1 = max(0, x - rx), min(X, x + rx + 1)
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            continue
        es = esf[
            z0 - (z - rz) : z0 - (z - rz) + (z1 - z0),
            y0 - (y - ry) : y0 - (y - ry) + (y1 - y0),
            x0 - (x - rx) : x0 - (x - rx) + (x1 - x0),
        ]
        vol[z0:z1, y0:y1, x0:x1] = np.maximum(
            vol[z0:z1, y0:y1, x0:x1], es
        )  # nucleus = point source
        sphere_crop = sphere[
            z0 - (z - rz) : z0 - (z - rz) + (z1 - z0),
            y0 - (y - ry) : y0 - (y - ry) + (y1 - y0),
            x0 - (x - rx) : x0 - (x - rx) + (x1 - x0),
        ]
        mask_crop = instance_mask[z0:z1, y0:y1, x0:x1]
        mask_crop[sphere_crop >= np.exp(-1.0)] = cell_id
        if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
            # deposits the emitted light at the centre. Uses the cell BRIGHTNESS (not `es.max()`, which comes from the patch
            # already CLIPPED at the edge -> edge cells emitted less than identical ones in the interior).
            emis[z, y, x] += bright * halo_amp
        cens.append(
            [float(p[0]), float(p[1]), float(p[2])]
        )  # TRUE centroid (float), not rounded
    # scatters the emission WIDE (anisotropic) and sums it into the volume -> purple glow between cells
    sz2 = halo_um * psf_z_ratio / VOXEL_NATIVE[0]
    sy2 = halo_um / VOXEL_NATIVE[1]
    sx2 = halo_um / VOXEL_NATIVE[2]
    emis = gaussian_filter(emis, (sz2, sy2, sx2))
    # Normalizes by the RESPONSE OF ONE SOURCE (a kernel constant), not by the global maximum. With the global
    # max, each cell's glow ended up depending on the brightest cluster in the volume -> two
    # volumes with different densities got different per-cell brightness. This way the halo ACCUMULATES with
    # density (dense glows, sparse stays dark), which is the desired physical behaviour.
    kern_peak = 1.0 / ((2.0 * np.pi) ** 1.5 * sz2 * sy2 * sx2 + 1e-12)
    emis = np.clip(emis / max(kern_peak, 1e-12), 0.0, 1.0)
    vol = vol + emis  # ADDS the emitted light (accumulates)

    # anisotropic OPTICAL PSF on the native grid (sigma in voxels = sigma_um / voxel_um)
    sz = psf_xy * psf_z_ratio / VOXEL_NATIVE[0]
    sy = psf_xy / VOXEL_NATIVE[1]
    sx = psf_xy / VOXEL_NATIVE[2]
    vol = np.clip(gaussian_filter(vol, (sz, sy, sx)), 0, 1)
    # camera noise (shot + read) at native resolution
    pe = float(os.environ.get("SYNTH_POISSON_PE", "300"))
    rd = float(os.environ.get("SYNTH_READNOISE", "0.006"))
    vol = np.clip(
        rng.poisson(np.clip(vol, 0, 1) * pe) / pe + rng.normal(0, rd, vol.shape), 0, 1
    ).astype(np.float32)

    # POOL 4x in XY. ATTENTION -- REAL INCONSISTENCY FIXED: this generator pooled by MEAN, but the
    # pipeline that SCORES (`dog_max_infer.pool_xy`) pools by STRIDE (`[::1,::4,::4]`). Training/measuring
    # on one pooling and deploying on another: the mean smooths the noise (higher SNR) and the stride does not, so the
    # synthetic data arrived cleaner than what the detector actually sees. Default = STRIDE (= deploy).
    if os.environ.get("SYNTH_POOL_MODE", "stride") == "mean":
        vol_p = (
            vol[:, : (Y // POOL) * POOL, : (X // POOL) * POOL]
            .reshape(Z, Y // POOL, POOL, X // POOL, POOL)
            .mean((2, 4))
            .astype(np.float32)
        )
    else:
        vol_p = vol[::1, ::POOL, ::POOL].astype(np.float32)
    cens_p = (
        (np.array(cens, np.float32) / np.array([1, POOL, POOL]))
        if cens
        else np.zeros((0, 3))
    )
    return vol, vol_p, np.array(cens, np.float32), cens_p, instance_mask


def _cell_radius_um(vol_pooled, dets):
    """APPARENT radius (half-maximum) of each detected cell, in um. Measures the size the DoG sees."""
    EFF = VOXEL_NATIVE[0]  # pooled ~ isotropic 1.625 um/voxel
    out = []
    for c in np.atleast_2d(dets).astype(int):
        z, y, x = c[:3]
        sub = vol_pooled[z, max(0, y - 6) : y + 7, max(0, x - 6) : x + 7]
        if sub.size < 9:
            continue
        pk = float(sub.max())
        if pk <= 1e-3:
            continue
        area = float((sub > 0.5 * pk).sum())  # pixels above half-maximum
        out.append(np.sqrt(area / np.pi) * EFF)  # equivalent radius (um)
    return np.array(out, np.float32)
