"""Rendering a volume: cell shapes, scattered light, optics, detector, pooling.

The pipeline, in the order light actually travels:

    nuclei  ->  scattered light  ->  illumination  ->  depth attenuation
            ->  PSF  ->  depth-dependent defocus  ->  shot + read noise  ->  pool

Every stage is driven by a :class:`~data_gen.acquisition.Regime`, drawn per
volume, rather than by a module-level constant. What that buys is spread: the
old renderer produced one imaging condition and it was the easy one -- cells at
400-1400% contrast over a near-black background, where the real median is 220%
and a quarter of real cells fall below 100%.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from data_gen.acquisition import (
    Palette, Regime, depth_ramp, illumination_field, reserved_um, sample_aniso,
)
from data_gen.geometry import sample_shell_positions
from data_gen.constants import (
    CELL_SHARPNESS, CLUSTER, CLUSTER_FRAC, CLUSTER_FRAC_N, CLUSTER_SPREAD_UM,
    DEBRIS_AMP, DEBRIS_RADIUS_UM, HALO_UM, HAZE_DENSITY_EXP,
    HAZE_NOMINAL_CELLS, MIN_CONTRAST, POOL, POOL_MODE, SHELL_DENS_CONTRAST,
    SHELL_JITTER, VOXEL_NATIVE,
)


def _sphere_native(r_um, rng, sharp=7, aniso=None, theta=None, frac=None,
                   texture_amp=0.0, texture_um=2.0, texture_seed=None):
    """A nucleus of radius ``r_um`` on the anisotropic native grid.

    Returns ``(profile, core)``; ``core`` is the boolean nucleus body, used for
    the instance mask so that the mask traces the nucleus and not its mottling.

    Super-gaussian profile -- a filled ball with a sharp edge -- scaled per axis
    by ``aniso`` and rotated by ``theta`` in the xy plane (the z anisotropy is
    physical and does not rotate with it).

    ``frac`` is the SUB-VOXEL offset (p - round(p)). Without it a cell can only
    sit centred on a whole voxel and the centroid label comes out quantised, a
    systematic error of up to half a native voxel that the detector learns as
    noise in its target.

    ``texture_amp`` mottles the inside. A real nucleus is chromatin, not a
    smooth ball, and that texture is what puts competing local maxima *inside*
    a single cell -- the failure mode a smooth-blob generator never trains for.
    """
    axes_vox = r_um / VOXEL_NATIVE          # semi-axes in native voxels
    if aniso is None:
        aniso = rng.uniform(0.82, 1.20, 3)
    axes = np.maximum(axes_vox * np.asarray(aniso, float), 0.6)
    size = (np.ceil(axes) * 2 + 3).astype(int)
    zz, yy, xx = np.indices(tuple(size))
    c = size // 2
    fz, fy, fx = (0.0, 0.0, 0.0) if frac is None else (
        float(frac[0]), float(frac[1]), float(frac[2])
    )
    th = rng.uniform(0, np.pi) if theta is None else float(theta)
    ct, st = np.cos(th), np.sin(th)
    dy, dx, dz = yy - c[1] - fy, xx - c[2] - fx, zz - c[0] - fz
    ry = ct * dy - st * dx
    rx = st * dy + ct * dx
    rn = np.sqrt((dz / axes[0]) ** 2 + (ry / axes[1]) ** 2 + (rx / axes[2]) ** 2)
    prof = np.exp(-(rn ** sharp))
    # The bounding box is barely wider than the nucleus, so the profile is
    # still at 13% of its peak where the box ends (measured on a 4.5 um cell in
    # y/x) and pasting it dropped straight to zero there -- every nucleus got a
    # square edge, visible in the rendered slices. Subtracting the value at the
    # nearest face and renormalising takes the profile smoothly to zero on the
    # whole boundary instead, at no extra cost and without widening the box.
    edge = float(np.exp(-(min(c / axes) ** sharp)))
    prof = np.clip((prof - edge) / (1.0 - edge), 0.0, 1.0).astype(np.float32)
    core = rn <= 1.0
    if texture_amp <= 0:
        return prof, core
    trng = np.random.default_rng(texture_seed)
    noise = trng.normal(0, 1, prof.shape).astype(np.float32)
    noise = gaussian_filter(noise, tuple(texture_um / VOXEL_NATIVE))
    noise /= noise.std() + 1e-9
    # The upper bound was 1.75, which let mottling alone put a cell 75% above
    # where its own brightness placed it. Real detected cells span 1.77x from
    # the frame median *in total*, across lighting, depth and brightness
    # together, so no single term may spend that much of the budget.
    return prof * np.clip(1.0 + texture_amp * noise, 0.25, 1.45), core


def _scattered_light(emis, n_cells, regime: Regime):
    """The wide haze plus the tight halo, as a field in [0, 1].

    ``emis`` holds one point source per cell, weighted by its brightness. Two
    blurs of it are summed: a tight one (a few um, the near halo of the optics)
    and a wide one (tens of um, light scattered through the tissue). The wide
    term is the one that matters -- it is what lifts a cell's local background
    from 0.06 to the real 0.25 and so takes contrast from 400% down to 220%.

    The wide term is normalised by its own 98th percentile -- that fixes its
    *shape*, bright where the tissue is and dark in the voids -- and its
    *level* is then set by how much tissue the frame holds. The real data ties
    the two: a frame with 51 blobs has a background of 0.13-0.15, one with 404
    has 0.30-0.51, a ratio of ~2.8 for 7.9x the density, which is a square
    root. Normalising by the maximum alone would give a nearly empty frame the
    same pedestal as a packed one.
    """
    vox = VOXEL_NATIVE
    wide = gaussian_filter(emis, (
        regime.haze_um * regime.haze_z_ratio / vox[0],
        regime.haze_um / vox[1],
        regime.haze_um / vox[2],
    ))
    wide /= max(float(np.percentile(wide, 98)), 1e-12)
    # Gamma on the normalised field, which is what keeps the void dark: the
    # haze has to stop where the tissue does.
    wide = np.clip(wide, 0.0, 1.0) ** regime.haze_gamma
    level = regime.pedestal * float(np.clip(
        (max(n_cells, 1) / HAZE_NOMINAL_CELLS) ** HAZE_DENSITY_EXP, 0.4, 1.4
    ))
    near = gaussian_filter(emis, (
        HALO_UM * 2.5 / vox[0], HALO_UM / vox[1], HALO_UM / vox[2],
    ))
    # Normalised by the response of ONE source (a kernel constant), not by the
    # global maximum: with the maximum, each cell's halo depended on the
    # brightest cluster in the volume, so two volumes of different density gave
    # different per-cell brightness.
    kern_peak = 1.0 / (
        (2.0 * np.pi) ** 1.5
        * (HALO_UM * 2.5 / vox[0]) * (HALO_UM / vox[1]) * (HALO_UM / vox[2])
        + 1e-12
    )
    near = np.clip(near / max(kern_peak, 1e-12), 0.0, 1.0)
    return (wide * level + near * regime.halo_amp).astype(np.float32)


def _add_debris(vol, rng, regime: Regime):
    """Diffuse autofluorescent junk: bright, structured, and not a cell.

    Deliberately far above cell scale (7-22 um). An unlabelled *cell-sized*
    blob would be a false negative in the target rather than a hard negative,
    which would teach the detector to suppress real cells.

    Rendered as impulses blurred across the whole volume rather than as
    per-blob patches. A blob 22 um across needs a box wider than the frame, so
    pasting it patch-by-patch cut it off at a third of its amplitude and left
    hard rectangular seams across the image. The cost is that one volume's
    debris shares one size, which for background junk is not worth a second
    convolution to avoid.
    """
    Z, Y, X = vol.shape
    r_um = rng.uniform(*DEBRIS_RADIUS_UM)
    seeds = np.zeros((Z, Y, X), np.float32)
    idx = rng.integers([0, 0, 0], [Z, Y, X], size=(regime.debris, 3))
    amps = rng.uniform(*DEBRIS_AMP, size=regime.debris).astype(np.float32)
    np.add.at(seeds, (idx[:, 0], idx[:, 1], idx[:, 2]), amps)
    sig = np.maximum(r_um / VOXEL_NATIVE, 1.0)
    seeds = gaussian_filter(seeds, tuple(sig))
    # Normalised by the response of one impulse, so a blob's amplitude is the
    # amplitude it was drawn with rather than a function of how many landed.
    kern_peak = 1.0 / ((2.0 * np.pi) ** 1.5 * sig[0] * sig[1] * sig[2] + 1e-12)
    return vol + np.clip(seeds / max(kern_peak, 1e-12), 0.0, 1.0)


def _defocus(vol, regime: Regime, ramp):
    """Blend in a blurrier copy with depth, so deep cells lose their edges."""
    if regime.defocus_um <= 0.05:
        return vol
    vox = VOXEL_NATIVE
    blurred = gaussian_filter(vol, (
        regime.defocus_um * 2.5 / vox[0],
        regime.defocus_um / vox[1],
        regime.defocus_um / vox[2],
    ))
    w = (ramp ** 1.5).astype(np.float32)
    return vol * (1.0 - w) + blurred * w


def gen_volume_native(
    zshape=64,
    xyshape=256,
    n_cells=200,
    rng=None,
    cens_native=None,
    medium=None,
    min_sep_um=None,
    shell=None,
    place_mode="shell",
    regime=None,
    palette=None,
):
    """Generate a native volume ``(zshape, xyshape, xyshape)``.

    Returns ``(vol_native, vol_pooled, centroids_native, centroids_pooled,
    instance_mask_native)``. Instance labels are one-based and index rows of
    ``centroids_native``.

    ``regime``  imaging conditions; drawn from ``rng`` when omitted. Pass the
                same one to every frame of a sequence, or the acquisition
                flickers between t and t+1.
    ``palette`` per-cell appearance aligned with ``cens_native``; drawn when
                omitted. Pass the same one (drifted) across a sequence, or a
                cell's shape and brightness are noise between frames and
                appearance matching becomes unlearnable.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if regime is None:
        regime = Regime.sample(rng)
    if medium is not None:
        regime = regime.with_(medium=float(medium))

    r_lo, r_hi = regime.radius_um
    Z, Y, X = zshape, xyshape, xyshape

    # --- where the cells are ---------------------------------------------
    # Sizes come from the palette when there is one, so the collision check at
    # placement time uses the radii that will actually be rendered. They used
    # to be drawn twice, independently, and the two never agreed.
    if cens_native is not None and len(cens_native) >= 5:
        pos = np.atleast_2d(cens_native).astype(np.float32)
        if palette is None:
            palette = Palette.sample(len(pos), rng, regime)
        radii = palette.radius_um
    elif place_mode == "shell":
        n_want = max(8, int(round(n_cells * regime.density_factor)))
        pos, radii, aniso = sample_shell_positions(
            n_want, (Z, Y, X), rng, shell=shell, r_lo=r_lo, r_hi=r_hi,
            csf=regime.collision_factor, jitter=SHELL_JITTER,
            dens_contrast=SHELL_DENS_CONTRAST,
        )
        palette = Palette.sample(len(pos), rng, regime)
        # Both the size and the shape that placement reserved room for, or the
        # rendered cell is not the cell that was packed.
        palette.radius_um[:] = radii
        palette.aniso[:] = aniso
    else:
        # Thomas process: cluster seeds plus a uniform background fraction. The
        # real embryo is not uniformly random -- it has dense and empty regions.
        n_want = max(8, int(round(n_cells * regime.density_factor)))
        ncl = max(3, int(n_want * CLUSTER_FRAC_N))
        centers = (
            rng.uniform([6, 16, 16], [Z - 6, Y - 16, X - 16], size=(ncl, 3))
            .astype(np.float32) if CLUSTER else None
        )
        sp_vox = CLUSTER_SPREAD_UM / VOXEL_NATIVE
        pos, radii, anisos, reserved, tries = [], [], [], [], 0
        while len(pos) < n_want and tries < n_want * 60:
            tries += 1
            if CLUSTER and rng.random() < CLUSTER_FRAC:
                p = np.clip(
                    centers[rng.integers(ncl)] + rng.normal(0, sp_vox, 3),
                    [4, 12, 12], [Z - 4, Y - 12, X - 12],
                ).astype(np.float32)
            else:
                p = rng.uniform([4, 12, 12], [Z - 4, Y - 12, X - 12])
            r = float(rng.uniform(r_lo, r_hi))
            a = sample_aniso(1, rng)[0]
            res = float(reserved_um(r, a))
            if pos:
                d = np.linalg.norm((np.array(pos) - p) * VOXEL_NATIVE, axis=1)
                # The room the cell takes once elongated, not its nominal
                # radius -- same rule as the shell branch.
                thr = regime.collision_factor * (res + np.array(reserved))
                if min_sep_um:
                    thr = np.maximum(thr, float(min_sep_um))
                if (d < thr).any():
                    continue
            pos.append(p)
            radii.append(r)
            anisos.append(a)
            reserved.append(res)
        pos = np.array(pos, np.float32)
        radii = np.array(radii, np.float32)
        palette = Palette.sample(len(pos), rng, regime)
        palette.radius_um[:] = radii
        if len(pos):
            palette.aniso[:] = np.array(anisos, np.float32).reshape(-1, 3)

    if len(pos) == 0:
        empty = np.zeros((Z, Y, X), np.float32)
        return (empty, empty[:, ::POOL, ::POOL], np.zeros((0, 3), np.float32),
                np.zeros((0, 3), np.float32), np.zeros((Z, Y, X), np.uint16))

    # --- the nuclei -------------------------------------------------------
    # Headroom: the pedestal is added on top, so the cells are rendered into
    # what is left. Without this the sum clips and every bright nucleus loses
    # its shape to a flat saturated cap -- the real data does not clip (its
    # p99.9 sits at 0.65 of the sensor maximum).
    amp = float(np.clip(1.0 - 0.20 * regime.pedestal, 0.6, 1.0))
    signal = np.zeros((Z, Y, X), np.float32)
    instance_mask = np.zeros((Z, Y, X), np.uint16)

    # The scattered light is computed first, from the cell centres alone, so
    # that each cell's own brightness can be checked against the pedestal it
    # will land on. Raising a cell's brightness feeds back into the haze it
    # contributes, but only to second order, so the haze is not recomputed.
    ipos = np.clip(np.round(pos).astype(int), 0, [Z - 1, Y - 1, X - 1])
    emis = np.zeros((Z, Y, X), np.float32)
    np.add.at(emis, (ipos[:, 0], ipos[:, 1], ipos[:, 2]), palette.bright)
    haze = _scattered_light(emis, len(pos), regime)

    # A labelled cell that does not rise above the haze it sits in is not a
    # hard example, it is a wrong label: it teaches the detector to fire where
    # there is nothing to see. The faintest real annotated cell rises 14% above
    # its surroundings (p1 of 836 cells), so that is the floor, applied against
    # the haze at the cell centre.
    #
    # Under the constants as they stand this never binds (measured: 0 cells in
    # 12 regimes). It is here as an invariant, not as a shaper of the faint
    # tail -- it is the guard against the failure this calibration was one bad
    # constant away from, a pedestal raised until the cells underneath it are
    # gone and every one of them still labelled. Note what it does *not* cover:
    # a dim cell beside a bright neighbour reads as negative contrast against a
    # 7-12 um ring, and that is a real hard case, not a bad label -- the cell is
    # there and is still a local maximum. Only the haze is background.
    bright = np.array(palette.bright, np.float32, copy=True)
    floor = MIN_CONTRAST * haze[ipos[:, 0], ipos[:, 1], ipos[:, 2]] / amp
    np.minimum(np.maximum(bright, floor, out=bright), 1.0, out=bright)

    # Strength of the cell that currently owns each mask voxel. The mask used
    # to be written last-wins, which loses two ways: a voxel could be labelled
    # for a cell that is not the one visible there (the signal composites by
    # maximum, so the brightest cell owns the *image*), and a cell drawn early
    # could have its whole body overwritten and end up with no mask at all.
    # Measured over 1930 cells: 5% kept under 62% of their body and the worst
    # kept 16%. Scoring by contribution instead makes the mask agree with the
    # image and makes the result independent of draw order.
    mask_score = np.zeros((Z, Y, X), np.float32)
    cens, drawn = [], []
    for cell_id, p in enumerate(pos, start=1):
        k = cell_id - 1
        pr = np.round(p)
        sphere, core = _sphere_native(
            float(palette.radius_um[k]), rng, sharp=CELL_SHARPNESS,
            aniso=palette.aniso[k], theta=float(palette.theta[k]),
            frac=np.asarray(p, float) - pr,
            texture_amp=regime.texture_amp, texture_um=regime.texture_um,
            texture_seed=int(palette.texture_seed[k]),
        )
        rz, ry, rx = np.array(sphere.shape) // 2
        z, y, x = pr.astype(int)
        z0, z1 = max(0, z - rz), min(Z, z + rz + 1)
        y0, y1 = max(0, y - ry), min(Y, y + ry + 1)
        x0, x1 = max(0, x - rx), min(X, x + rx + 1)
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            continue
        box = (
            slice(z0 - (z - rz), z0 - (z - rz) + (z1 - z0)),
            slice(y0 - (y - ry), y0 - (y - ry) + (y1 - y0)),
            slice(x0 - (x - rx), x0 - (x - rx) + (x1 - x0)),
        )
        contrib = sphere[box] * (bright[k] * amp)
        view = signal[z0:z1, y0:y1, x0:x1]
        np.maximum(view, contrib, out=view)
        # Own the voxel where this cell's body both covers it and outshines
        # whichever cell holds it, which is the same rule the signal above
        # composites by.
        score_crop = mask_score[z0:z1, y0:y1, x0:x1]
        take = core[box] & (contrib > score_crop)
        score_crop[take] = contrib[take]
        instance_mask[z0:z1, y0:y1, x0:x1][take] = cell_id
        cens.append([float(p[0]), float(p[1]), float(p[2])])
        drawn.append(cell_id)

    # A cell can still lose every voxel to brighter neighbours sitting on top
    # of it. It is a real cell at a known place, so it keeps its centre voxel
    # rather than going unlabelled -- an unlabelled cell in an otherwise fully
    # labelled frame is a false negative baked into the ground truth, which is
    # the one error the sparse real annotation already suffers from and the
    # thing this generator exists to avoid.
    held = set(np.unique(instance_mask).tolist())
    for cell_id in drawn:
        if cell_id not in held:
            z, y, x = ipos[cell_id - 1]
            instance_mask[z, y, x] = cell_id

    # --- scattered light, medium, debris ----------------------------------
    vol = signal + haze
    vol += np.clip(
        rng.normal(regime.medium, regime.medium * 0.25, (Z, Y, X)), 0, 1
    ).astype(np.float32)
    if regime.debris:
        vol = _add_debris(vol, rng, regime)

    # --- illumination and depth -------------------------------------------
    vol *= illumination_field((Z, Y, X), regime)
    ramp = depth_ramp((Z, Y, X), regime)
    vol *= (1.0 - regime.atten_frac * ramp)

    # --- optics ------------------------------------------------------------
    vol = gaussian_filter(vol, (
        regime.psf_z_um / VOXEL_NATIVE[0],
        regime.psf_xy_um / VOXEL_NATIVE[1],
        regime.psf_xy_um / VOXEL_NATIVE[2],
    ))
    vol = np.clip(_defocus(vol, regime, ramp), 0, 1)

    # --- detector ----------------------------------------------------------
    vol = np.clip(
        rng.poisson(vol * regime.poisson_pe) / regime.poisson_pe
        + rng.normal(0, regime.read_noise, vol.shape),
        0, 1,
    ).astype(np.float32)

    # --- pooling -----------------------------------------------------------
    # STRIDE, not mean: the pipeline that scores pools by ``[::1, ::4, ::4]``.
    # Training on mean-pooled data and deploying on stride-pooled data hands
    # the detector a cleaner image than it will ever see, because the mean
    # smooths the shot noise away and the stride does not.
    if POOL_MODE == "mean":
        vol_p = (
            vol[:, : (Y // POOL) * POOL, : (X // POOL) * POOL]
            .reshape(Z, Y // POOL, POOL, X // POOL, POOL)
            .mean((2, 4)).astype(np.float32)
        )
    else:
        vol_p = vol[::1, ::POOL, ::POOL].astype(np.float32)
    cens = np.array(cens, np.float32).reshape(-1, 3)
    cens_p = cens / np.array([1, POOL, POOL], np.float32)
    return vol, vol_p, cens, cens_p, instance_mask


def _cell_radius_um(vol_pooled, dets):
    """APPARENT radius (half-maximum) of each detected cell, in um."""
    EFF = VOXEL_NATIVE[0]  # pooled ~ isotropic 1.625 um/voxel
    out = []
    for c in np.atleast_2d(dets).astype(int):
        z, y, x = c[:3]
        sub = vol_pooled[z, max(0, y - 6):y + 7, max(0, x - 6):x + 7]
        if sub.size < 9:
            continue
        pk = float(sub.max())
        if pk <= 1e-3:
            continue
        area = float((sub > 0.5 * pk).sum())
        out.append(np.sqrt(area / np.pi) * EFF)
    return np.array(out, np.float32)
