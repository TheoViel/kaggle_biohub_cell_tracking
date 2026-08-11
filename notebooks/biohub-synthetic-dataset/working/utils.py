import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve as ndconvolve
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.ndimage import zoom as ndzoom


VOXEL_UM = np.array([1.625, 0.40625, 0.40625])
DOWNSAMPLE = (1, 4, 4)
EFF_VOXEL = VOXEL_UM * np.array(DOWNSAMPLE)
# Tolerance RADIUS for extracting templates from the REAL data. GRID SEARCH (biohub-radius-sweep, anti-leakage):
#   9.2 (old default, r_vox=3) mean_KS 0.4096 = WORST | 12.0 (r_vox=4) 0.3632 | 20.0 (r_vox=6) 0.3458
# 12.0 chosen: best cell_mean/cell_peak (0.757->0.647 / 0.772->0.620 = the largest gaps) and it covers the
# 9.2um cell WITHOUT invading neighbours (real NN ~6-10um). 20um wins on KS but would capture whole neighbours
# into the template (artefact) -> the gain is probably spurious. Validate 20 vs 12 with proxy-training if it matters.
CELL_DIAM_UM = float(os.environ.get("SYNTH_CELL_DIAM", "12.0"))
CELL_R_VOX = max(1, int(round((CELL_DIAM_UM / 2) / EFF_VOXEL[0])))
N_EST_PER_FRAME = int(
    os.environ.get("SYNTH_NEST", "240")
)  # real density estimated (EDA)
N_SAMPLES = int(
    os.environ.get("SYNTH_NSAMPLES", "120")
)  # synthetic volumes (per version)
SIM_GATE = float(
    os.environ.get("SYNTH_SIMGATE", "0.70")
)  # each cell >=70% similar to a real one (3d space)
BLUR_SIGMA = float(
    os.environ.get("SYNTH_BLUR", "0.6")
)  # final blur (PSF); tuned by the sweep
PSF_Z_RATIO = float(
    os.environ.get("SYNTH_PSFZ", "1.6")
)  # ANISOTROPIC PSF: light-sheet blurs ~1.6x more in z than in xy
DIV_RATE = float(
    os.environ.get("SYNTH_DIVRATE", "0.0")
)  # rate of DIVIDING cells (measured on the real GT; dumbbells/2 nuclei)
CELL_MEAN_TARGET = float(
    os.environ.get("SYNTH_CELLMEAN", "0.30")
)  # TARGET: mean intensity PER CELL (NORMAL distribution)
CELL_MEAN_SIGMA = float(
    os.environ.get("SYNTH_CELLSIG", "0.10")
)  # std of the normal (recalibrated from the real data at runtime)
SMOKE = os.environ.get("BIOHUB_SMOKE", "0") == "1"


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


SPHERICAL = (
    os.environ.get("SYNTH_SPHERICAL", "1") != "0"
)  # spherical capture (v6) vs cubic


def _sphere_mask(shape, r):
    zz, yy, xx = np.indices(shape)
    c = np.array(shape) // 2
    d = np.sqrt(
        ((zz - c[0]) * EFF_VOXEL[0]) ** 2
        + ((yy - c[1]) * EFF_VOXEL[1]) ** 2
        + ((xx - c[2]) * EFF_VOXEL[2]) ** 2
    )
    return (d <= (r + 0.5) * EFF_VOXEL[0]).astype(np.float32)


def extract_template(vol, zc, yc, xc, r=CELL_R_VOX):
    z0, z1, y0, y1, x0, x1 = zc - r, zc + r + 1, yc - r, yc + r + 1, xc - r, xc + r + 1
    if (
        z0 < 0
        or y0 < 0
        or x0 < 0
        or z1 > vol.shape[0]
        or y1 > vol.shape[1]
        or x1 > vol.shape[2]
    ):
        return None
    t = vol[z0:z1, y0:y1, x0:x1].astype(np.float32).copy()
    if SPHERICAL:  # SPHERICAL mask -> only the cell, no corners
        t = t * _sphere_mask(t.shape, r)
    return t


def build_template_bank(frame_pooled, centroids_pooled, r=CELL_R_VOX, refine=True):
    tb = []
    for zc, yc, xc in np.round(np.atleast_2d(centroids_pooled)).astype(int):
        t = extract_template(frame_pooled, zc, yc, xc, r)
        if t is None:
            continue
        if refine:
            t = np.maximum(t - np.percentile(t, 20), 0.0)
        if t.max() > 0:
            tb.append(t)
    return tb


def liquid_model(frames_pooled, centroids_by_frame, cell_r=CELL_R_VOX):
    # v2: keeps the real FIELDS per frame (tissue/medium structure) -> bank for varying per sample
    bg_vals, fields, field_cens = [], [], []
    for fr, cens in zip(frames_pooled, centroids_by_frame):
        mask = np.ones(fr.shape, bool)
        for zc, yc, xc in np.round(np.atleast_2d(cens)).astype(int):
            z0, z1 = max(0, zc - cell_r), min(fr.shape[0], zc + cell_r + 1)
            y0, y1 = max(0, yc - cell_r), min(fr.shape[1], yc + cell_r + 1)
            x0, x1 = max(0, xc - cell_r), min(fr.shape[2], xc + cell_r + 1)
            mask[z0:z1, y0:y1, x0:x1] = False
        bg_vals.append(fr[mask])
        fields.append(
            gaussian_filter(
                fr, float(os.environ.get("SYNTH_FIELD_SMOOTH", "4"))
            ).astype(np.float32)
        )  # field smoothing (knob for grad_mag)
        # REAL LAYOUT: DENSE centroids (labelled GT union DoG) -> real positions for layout transfer
        det = dog_detect_precise(fr)[0]
        allc = (
            np.vstack([np.atleast_2d(cens), det])
            if len(np.atleast_2d(cens)) and len(det)
            else (det if len(det) else np.atleast_2d(cens))
        )
        field_cens.append(np.round(allc).astype(np.int32))
    bg = np.concatenate(bg_vals) if bg_vals else np.array([0.0])
    # real Z DEPTH PROFILE (microscopy attenuation): mean brightness per Z slice
    zp = (
        np.mean([f.mean(axis=(1, 2)) for f in frames_pooled], axis=0)
        if frames_pooled
        else np.ones(64)
    )
    z_prof = gaussian_filter(zp.astype(np.float32), 2)
    z_prof = z_prof / (z_prof.mean() + 1e-6)  # normalized, smooth
    # NN + COUNT + WHERE the cells are (field/tissue level at the real detections, NOT just at the saturated peak)
    real_nn, dog_counts, gt_field = [], [], []
    _nsub = frames_pooled[:: max(1, len(frames_pooled) // 60)][
        :60
    ]  # up to 60 representative frames (larger base)
    for fr in _nsub:
        det = dog_detect_precise(fr)[0]
        dog_counts.append(len(det))
        d = nn_um(det)
        if len(d):
            real_nn.append(d)
        fld = gaussian_filter(fr, 5)
        fnn = fld / (float(fld.max()) + 1e-6)
        sh = np.array(fr.shape) - 1
        for c in det:
            gt_field.append(float(fnn[tuple(np.clip(np.round(c).astype(int), 0, sh))]))
    real_nn = np.concatenate(real_nn) if real_nn else np.array([CELL_DIAM_UM * 0.55])
    gt_field = np.array(gt_field) if gt_field else np.array([0.6])
    gt_field_med = float(np.median(gt_field))
    gt_field_std = float(np.std(gt_field) + 0.06)
    # EMPIRICAL DISTRIBUTION of LIGHT where the cells REALLY are -> P(cell | field intensity).
    # Replicates the WAY light concentrates in the real data (not just the median) to place the synthetic cells.
    _lh, _le = np.histogram(gt_field, bins=24, range=(0.0, 1.0), density=True)
    light_bins = (0.5 * (_le[:-1] + _le[1:])).astype(np.float32)
    light_pdf = gaussian_filter(_lh.astype(np.float32), 1.0)
    light_pdf = (light_pdf / (light_pdf.max() + 1e-9)).astype(np.float32)
    # MINIMUM SEPARATION. Mode 'p5' (default): uses the 5th percentile of the REAL NN = how close two cells
    # actually get. The old mode ('median') used median*0.72, which FORBADE the close pairs that
    # exist in the real data -> the quality report measured synth NN 12-15um against real 8-9um.
    # DEFAULT = 'median'. The ablation CONDEMNED mode 'p5': it does NOT improve NN (0.779 -> 0.768; what
    # fixed NN was SYNTH_PLACEMENT=real, from 0.479 to 0.774) and it is the SOLE CAUSE of the texture
    # collapse (voxel_int 0.816 -> 0.463, local_std 0.698 -> 0.447, grad_mag 0.739 -> 0.535, mean
    # 0.747 -> 0.706). Mechanism: closer cells MERGE under the PSF -> a brighter connected mass and
    # the DoG detects even FEWER (median 74 -> 56). Turn on only for experiments: SYNTH_MINSEP_MODE=p5.
    _nnp5 = float(np.percentile(real_nn, 5))
    if os.environ.get("SYNTH_MINSEP_MODE", "median") == "p5":
        min_sep_vox = float(np.clip(_nnp5 / EFF_VOXEL[0], 2.0, 6.0))
    else:
        min_sep_vox = float(
            np.clip(
                np.median(real_nn)
                * float(os.environ.get("SYNTH_MINSEP_FACTOR", "0.72"))
                / EFF_VOXEL[0],
                3.0,
                6.0,
            )
        )
    n_cells_real = (
        int(np.median(dog_counts)) if dog_counts else N_EST_PER_FRAME
    )  # real density, not a fixed 240
    # EMPIRICAL DISTRIBUTION of count PAIRED with the tissue extent -> field_n_cells does the
    # quantile mapping (reproduces the volume-to-volume variability, not just the median). The extent
    # is measured on the SMOOTHED field, the same object `fields` stores and that generation consumes.
    _fs = float(os.environ.get("SYNTH_FIELD_SMOOTH", "4"))
    _sf = [gaussian_filter(f, _fs) for f in _nsub]
    _bf_thr = (
        float(np.percentile(np.concatenate([s.ravel()[::7] for s in _sf]), 55))
        if _sf
        else 0.0
    )
    _cnt_bf = [
        field_extent(s, _bf_thr) for s in _sf
    ]  # GLOBAL threshold -> the extent actually varies
    # CANONICAL REFERENCE for HM: POOL of frames by INDEX (no brightness bias) -> distribution = exact real
    # MEAN (same frames as the validation). Stable and decoupled from the structural knobs.
    hm_ref = None
    if frames_pooled:
        idx = np.unique(
            np.linspace(0, len(frames_pooled) - 1, min(16, len(frames_pooled))).astype(
                int
            )
        )
        hm_ref = np.concatenate(
            [np.asarray(frames_pooled[i], np.float32) for i in idx], axis=0
        )
    # MEMORY GUARD: cap on the field/layout bank (the base can have thousands of frames) - representative subsample
    CAP = int(os.environ.get("SYNTH_FIELDCAP", "500"))
    if len(fields) > CAP:
        ci = np.linspace(0, len(fields) - 1, CAP).astype(int)
        fields = [fields[i] for i in ci]
        field_cens = [field_cens[i] for i in ci]
        frames_pooled = [frames_pooled[i] for i in ci]
    # REAL RESIDUAL TEXTURE (frame - smoothed field) = real high-freq to inject into the synthetic (attacks grad_mag)
    residuals = [
        (np.asarray(frames_pooled[i], np.float32) - fields[i]).astype(np.float32)
        for i in range(len(fields))
    ]
    return dict(
        mean=float(bg.mean()),
        std=float(bg.std()),
        p50=float(np.percentile(bg, 50)),
        residuals=residuals,
        p99=float(np.percentile(bg, 99)),
        noise_std=float(np.std(bg) * 0.6),
        fields=fields,
        field_cens=field_cens,  # REAL LAYOUT: dense centroids paired with fields
        ref_frames=[
            np.asarray(f, np.float32) for f in frames_pooled
        ],  # v4: for histogram matching
        count_dist=np.asarray(dog_counts, np.float32),
        bf_dist=np.asarray(_cnt_bf, np.float32),
        bf_thr=_bf_thr,
        hm_ref_canon=hm_ref,
        min_sep_vox=min_sep_vox,
        real_nn_p5_um=_nnp5,
        real_nn_med_um=float(np.median(real_nn)),
        n_cells_real=n_cells_real,
        gt_field_med=gt_field_med,
        gt_field_std=gt_field_std,  # WHERE the cells are (real field level)
        light_bins=light_bins,
        light_pdf=light_pdf,  # EMPIRICAL light->cell distribution (real)
        z_prof=z_prof,  # Z DEPTH PROFILE (real attenuation)
        shape=frames_pooled[0].shape if frames_pooled else None,
    )


def augment_template(t, rng):
    for ax in range(3):
        if rng.random() < 0.5:
            t = np.flip(t, ax)
    if rng.random() < 0.5:
        t = np.rot90(t, rng.integers(1, 4), axes=(1, 2))
    return np.ascontiguousarray(t * rng.uniform(0.7, 1.3))


def field_extent(field, thr=None):
    """Fraction of the volume occupied by tissue (proxy for the developmental stage).

    BUG FIXED: the original version used `field > np.percentile(field, 55)`, which BY DEFINITION always
    gives ~0.45 -- the percentile is of the field itself. In other words, the "time-aware density" was a
    CONSTANT and every synthetic volume got the same count target (the `count` Q-Q saturated).
    With a GLOBAL threshold (the same for every field) the measure discriminates sparse vs extensive
    tissue again."""
    if thr is None:
        return float(
            (field > np.percentile(field, 55)).mean()
        )  # compat (constant ~0.45)
    return float((field > thr).mean())


def field_n_cells(field, base=N_EST_PER_FRAME, bg=None):
    """TIME-AWARE density: n cells ~ tissue extent (developmental stage).

    EMPIRICAL MODE (default, when `bg` carries the distributions): QUANTILE MAPPING -- finds the
    percentile of this field's extent among the real extents and returns the SAME percentile of the
    real count distribution. This reproduces at once the MARGINAL (the real data goes from ~25 to ~330
    cells/volume) and the CORRELATION with the tissue.
    It used to use `int(median(dog_counts))` -- a single number -- still multiplied by 0.60. The
    quality report showed the effect: the `count` Q-Q SATURATED at ~100 while the real data reaches
    330, and nearest-neighbour came out shifted (synth 12-15um vs real 8-9um) because cells were missing.
    """
    # DEFAULT = 'empirical', decided by ABLATION in the quality report (4 rounds, held-out):
    #   A = optimized config                     mean 0.745 | count 0.500 | hi_freq 0.332 | voxel_int 0.769
    #   B = A + count empirical                  mean 0.747 | count 0.542 | hi_freq 0.395 | voxel_int 0.816
    #   C = B + minsep p5                        mean 0.706 | count 0.528 | hi_freq 0.344 | voxel_int 0.463
    # B improves the TWO worst distributions (count +0.042, hi_freq +0.063) and lifts the worst axis from
    # 0.332 to 0.384, with a neutral mean (+0.002, within the noise of 1 seed). C shows the texture
    # damage came from MINSEP, not from here. Go back to the old behaviour with SYNTH_COUNT_MODE=median.
    if (
        bg is not None
        and os.environ.get("SYNTH_COUNT_MODE", "empirical") == "empirical"
    ):
        cd = bg.get("count_dist")
        bd = bg.get("bf_dist")
        thr = bg.get("bf_thr")
        if (
            cd is not None
            and bd is not None
            and thr is not None
            and len(cd) >= 5
            and len(bd) == len(cd)
        ):
            bf = field_extent(field, thr)
            q = float(
                np.searchsorted(np.sort(bd), bf) / max(len(bd), 1)
            )  # percentile of the extent
            return int(np.clip(np.quantile(np.sort(cd), np.clip(q, 0.0, 1.0)), 22, 700))
    return int(
        np.clip(base * field_extent(field) / 0.30 * 0.60, 22, 460)
    )  # old mode (constant ~0.45)


def connected_frac(vol, thr=0.4):
    """Fraction of the bright volume that sits in the LARGEST connected component.
    ~1 = connected mass (like the real tissue); low = a web of scattered blobs."""
    from scipy.ndimage import label

    m = vol > thr
    tot = int(m.sum())
    if tot < 5:
        return 0.0
    lab, n = label(m)
    if n == 0:
        return 0.0
    sizes = np.bincount(lab.ravel())[1:]
    return float(sizes.max() / tot)


def nn_um(coords):
    """Nearest-neighbour distance (one) for a set of centroids."""
    if len(coords) < 2:
        return np.array([])
    P = np.asarray(coords, float) * EFF_VOXEL
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    np.fill_diagonal(D, np.inf)
    return D.min(1)


def clump_index(coords, shape, cell=8):
    """Clustering index: local density variance / Poisson expectation (>1 = aggregated)."""
    if len(coords) < 3:
        return 1.0
    g = np.zeros((shape[0] // cell + 1, shape[1] // cell + 1, shape[2] // cell + 1))
    for z, y, x in np.asarray(coords, int):
        g[z // cell, y // cell, x // cell] += 1
    mu = g.mean()
    return float(g.var() / mu) if mu > 0 else 1.0


def overlap_audit(boxes, thr=0.3):
    """Fraction of boxes that OVERLAP (3D IoU > thr) some neighbour. Should be ~0 (clean data)."""
    b = np.asarray(boxes, float)
    if len(b) < 2:
        return 0.0, 0.0
    vol = np.prod(np.clip(b[:, 3:] - b[:, :3], 0, None), 1)
    bad = 0
    maxiou = 0.0
    for i in range(len(b)):
        lo = np.maximum(b[i, :3], b[:, :3])
        hi = np.minimum(b[i, 3:], b[:, 3:])
        inter = np.prod(np.clip(hi - lo, 0, None), 1)
        inter[i] = 0
        iou = inter / (vol[i] + vol - inter + 1e-9)
        mi = float(iou.max())
        maxiou = max(maxiou, mi)
        if mi > thr:
            bad += 1
    return bad / len(b), maxiou


def place_clustered(
    field,
    n_cells,
    rng,
    margin=6,
    gt_med=None,
    gt_std=None,
    light_bins=None,
    light_pdf=None,
):
    """Placement by the EMPIRICAL DISTRIBUTION of real light: weights each voxel by P(cell | intensity)
    measured at the real detections (the WAY light concentrates). Fallback = gaussian at gt_med.
    """
    fn = field / (float(field.max()) + 1e-6)
    if light_pdf is not None and light_bins is not None:
        emp = np.interp(fn, light_bins, light_pdf).astype(
            np.float32
        )  # real EMPIRICAL P(cell|intensity)
        valid = (fn > np.percentile(fn, 20)).astype(
            np.float32
        )  # tissue region (not background)
        prob = (
            0.7 * emp + 0.3 * valid
        ) * valid  # SPREAD OUT: mixes empirical with uniform (less clumping)
    elif gt_med is not None:
        prob = np.exp(
            -(((fn - gt_med) / (gt_std)) ** 2)
        )  # peak at the REAL field level
        prob = prob * (
            fn > np.percentile(fn, 30)
        )  # excludes the dark medium (background)
    else:
        prob = np.clip(field - np.percentile(field, 45), 0, None) ** 1.1
    prob[:margin] = 0
    prob[-margin:] = 0
    prob[:, :margin] = 0
    prob[:, -margin:] = 0
    prob[:, :, :margin] = 0
    prob[:, :, -margin:] = 0
    s = float(prob.sum())
    if s <= 0:
        return np.zeros((0, 3), int)
    ps = prob.ravel() / s
    n_cand = min(int(n_cells * 3), int((ps > 0).sum()))
    idx = rng.choice(prob.size, size=n_cand, replace=False, p=ps)
    return np.array(np.unravel_index(idx, field.shape)).T


def field_zprof(field):
    """Z profile of the SPECIFIC FIELD used in this volume (real gradient of THAT frame, not the global mean)."""
    fz = gaussian_filter(field.mean(axis=(1, 2)).astype(np.float32), 2)
    return fz / (fz.mean() + 1e-6)


def apply_z_profile(vol, z_prof):
    """Imposes the real Z DEPTH PROFILE (attenuation). RENORMALIZES (does not clip) so the boosted deep
    slices do not saturate -> clipping destroyed the gradient (boosted slices went past 1 and got cut).
    """
    if z_prof is None or len(z_prof) != vol.shape[0]:
        return vol
    cur = vol.mean(axis=(1, 2))
    cur = cur / (cur.mean() + 1e-6)
    gain = np.clip((z_prof / (cur + 1e-6)).astype(np.float32), 0.35, 3.0)
    vol = vol * gain[:, None, None]
    p = float(np.percentile(vol, 99.5))  # renormalizes -> preserves the gradient
    return np.clip(vol / (p + 1e-6), 0, 1) if p > 0 else np.clip(vol, 0, 1)


# per-cell intensity TARGET: DATA-DRIVEN. _CELL_MEANS_RT = REAL sample (the bootstrap matches the SHAPE
# of the distribution, incl. log-normal skew -> fixes cell_mean/cell_peak, which KS flagged as differing).
_CELL_MU_RT = CELL_MEAN_TARGET
_CELL_SIG_RT = CELL_MEAN_SIGMA
_CELL_MEANS_RT = None
_CELL_RATIO_RT = (
    None  # REAL per-cell peak/mean distribution (filled in by measure_cell_stats)
)
_TEMPLATES_RT = None  # bank of REAL cell templates (HYBRID mode: real appearance + parametric placement/global)

_PS_REF = None  # reference MAGNITUDE spectrum (real) for texture matching via Fourier


def match_power_spectrum(vol, bg, rng, w=None):
    """NEW STAGE (attacks grad_mag): matches the POWER SPECTRUM of the volume to that of a REAL frame in the
    Fourier domain (preserves the PHASE=synth structure, adopts the MAGNITUDE=real texture). Blend by weight w.
    """
    w = float(os.environ.get("SYNTH_PSMATCH", "0.0")) if w is None else w
    if w <= 0:
        return vol
    refs = bg.get("ref_frames") or bg.get("residuals")
    if not refs:
        return vol
    ref = np.asarray(refs[int(rng.integers(len(refs)))], np.float32)
    if ref.shape != vol.shape:
        return vol
    Fv = np.fft.fftshift(np.fft.fftn(vol.astype(np.float32)))
    mag_v = np.abs(Fv)
    ph = np.angle(Fv)
    mag_r = np.abs(np.fft.fftshift(np.fft.fftn(ref - ref.mean())))
    # BAND-LIMITED: matches the magnitude ONLY at HIGH-FREQ (texture); preserves LOW-FREQ (structure/intensity)
    shp = vol.shape
    c = [(s - 1) / 2.0 for s in shp]
    zz, yy, xx = np.indices(shp)
    r = np.sqrt(
        ((zz - c[0]) / (shp[0] / 2)) ** 2
        + ((yy - c[1]) / (shp[1] / 2)) ** 2
        + ((xx - c[2]) / (shp[2] / 2)) ** 2
    )
    hicut = float(os.environ.get("SYNTH_PSMATCH_HICUT", "0.35"))
    wr = (1.0 / (1.0 + np.exp(-(r - hicut) * 12))).astype(
        np.float32
    ) * w  # sigmoid: the weight grows at high-freq
    mag = (
        mag_v * (1 - wr) + mag_r * wr
    )  # low-freq intact, high-freq matches the real data (texture)
    out = np.real(np.fft.ifftn(np.fft.ifftshift(mag * np.exp(1j * ph)))).astype(
        np.float32
    )
    p = np.percentile(out, 99.5)
    out = out / (p + 1e-6) if p > 0 else out
    out = np.clip(out, 0, 1)
    if (
        os.environ.get("SYNTH_PSMATCH_HM", "1") == "1"
    ):  # HM after PSMATCH: restores the intensity HISTOGRAM
        try:  # (monotonic -> preserves the texture) -> fixes voxel_int
            from skimage.exposure import match_histograms

            rf = np.clip(ref / (np.percentile(ref, 99.5) + 1e-6), 0, 1)
            out = np.clip(match_histograms(out, rf).astype(np.float32), 0, 1)
        except Exception:
            pass
    return out


def poisson_gaussian_noise(vol, rng):
    """CAMERA NOISE (sCMOS, EMVA1288, from the literature): shot (Poisson, signal-dependent) + read (Gaussian).
    variance = signal/peak_pe + read^2 -> breaks the smoothness of the cells, matches the real GRAIN. Applied last.
    """
    pe = float(
        os.environ.get("SYNTH_POISSON_PE", "220")
    )  # photoelectrons at the peak (higher=less noise)
    rd = float(os.environ.get("SYNTH_READNOISE", "0.004"))
    if pe <= 0:
        return vol
    lam = np.clip(vol, 0, 1) * pe
    noisy = rng.poisson(lam).astype(np.float32) / pe + rng.normal(
        0, rd, vol.shape
    ).astype(np.float32)
    return np.clip(noisy, 0, 1)


def retarget_cell_intensity(vol, boxes, rng, mu=None, sigma=None, targets=None):
    """Matches the per-cell mean intensity to a NORMAL(mu, sigma) measured on the REAL data. Smooth SPHERICAL
    weight (=0 at the box edge -> does NOT leave a dark square in the tissue) + scale clamp (does not punch a hole).

    `targets` (optional, 1 target per box) FIXES the target instead of drawing one. Required in the TEMPORAL
    generator: without it the same tracked cell draws a new brightness every frame and FLICKERS, destroying the
    appearance identity that the edge predictor needs to learn."""
    if boxes is None or len(boxes) == 0:
        return vol
    mu = _CELL_MU_RT if mu is None else mu
    sigma = _CELL_SIG_RT if sigma is None else sigma
    for _bi, b in enumerate(np.atleast_2d(boxes).astype(int)):
        z0, y0, x0, z1, y1, x1 = b[:6]
        z0, y0, x0 = max(z0, 0), max(y0, 0), max(x0, 0)
        reg = vol[z0:z1, y0:y1, x0:x1]
        if reg.size < 8:
            continue
        pk = float(reg.max())
        if pk <= 1e-4:
            continue
        active = (
            reg > 0.3 * pk
        )  # active region (same definition as measure_cell_stats/validate)
        cur = float(reg[active].mean()) if active.any() else float(reg.mean())
        if cur <= 1e-4:
            continue
        if targets is not None and _bi < len(targets):
            tgt = float(
                np.clip(float(targets[_bi]) * rng.uniform(0.98, 1.02), 0.05, 0.95)
            )  # FIXED target for the track (+-2% real flicker)
        elif _CELL_MEANS_RT is not None and len(_CELL_MEANS_RT) > 5:
            tgt = float(
                np.clip(
                    rng.choice(_CELL_MEANS_RT) * rng.uniform(0.95, 1.05), 0.05, 0.95
                )
            )  # BOOTSTRAP from the real data (matches the shape)
        else:
            tgt = float(np.clip(rng.normal(mu, sigma), 0.05, 0.95))
        # PROFILE MATCHING (peak/mean): a scalar gain moves mean and peak together and NEVER fixes the
        # RATIO between them. An exponent on the normalized profile changes the ratio (>1 sharpens the peak, <1
        # flattens it) and the gain then puts the mean back on target. Attacks cell_peak, cell_range and contrast.
        if (
            os.environ.get("SYNTH_MATCH_PEAK", "0") == "1"
            and _CELL_RATIO_RT is not None
            and len(_CELL_RATIO_RT) > 5
        ):
            tgt_ratio = float(rng.choice(_CELL_RATIO_RT))
            base = np.clip(reg / (pk + 1e-6), 0, 1)
            best_g, best_err = 1.0, 1e9
            for g in (0.6, 0.8, 1.0, 1.3, 1.7, 2.2):
                cand = base**g
                cm = float(cand[active].mean()) if active.any() else float(cand.mean())
                if cm <= 1e-6:
                    continue
                err = abs(float(cand.max()) / cm - tgt_ratio)
                if err < best_err:
                    best_g, best_err = g, err
            if abs(best_g - 1.0) > 1e-6:
                reg = (base**best_g) * pk
                pk = float(reg.max())
                cur = float(reg[active].mean()) if active.any() else float(reg.mean())
                if cur <= 1e-4:
                    continue
                vol[z0:z1, y0:y1, x0:x1] = reg
        scale = float(
            np.clip(tgt / cur, 0.6, 1.35)
        )  # CLAMP (iter5: lower ceiling -> less 'poppy' cells)
        # SPHERICAL gaussian weight centred on the box -> 0 at the edge (tissue intact, no square)
        sh = reg.shape
        cc = (np.array(sh) - 1) / 2.0
        zz, yy, xx = np.indices(sh)
        d2 = (
            ((zz - cc[0]) / (sh[0] / 2.0 + 1e-6)) ** 2
            + ((yy - cc[1]) / (sh[1] / 2.0 + 1e-6)) ** 2
            + ((xx - cc[2]) / (sh[2] / 2.0 + 1e-6)) ** 2
        )
        w = np.exp(-2.2 * d2).astype(np.float32)  # ~1 at the centre, ~0 at the edge
        vol[z0:z1, y0:y1, x0:x1] = np.clip(reg * (1 + w * (scale - 1)), 0, 1)
    return vol


def gen_volume(shape, bg, templates, n_cells, rng, min_sep=2):
    """v2: REAL field as the base + cell placement by DENSITY (tissue brightness)
    + cell brightness inherited from the local tissue -> recovers spatial structure and intensity.
    """
    Z, Y, X = shape
    fields = bg.get("fields")
    if fields:
        field = fields[
            rng.integers(len(fields))
        ].copy()  # 1 real field per sample (varies)
    else:
        field = np.full(shape, bg["p50"], np.float32)
    # REAL density x mult (nn_um). SYNTH_NEST only counts if set EXPLICITLY: without that the measured
    # `bg['n_cells_real']` overwrites the default and the knob is DEAD (the biohub-newvars sweep
    # burned 4 SYNTH_NEST configs that gave an identical mean_KS to the 16th decimal place).
    _base = (
        N_EST_PER_FRAME
        if os.environ.get("SYNTH_NEST")
        else bg.get("n_cells_real", N_EST_PER_FRAME)
    )
    n_cells = int(
        field_n_cells(field, _base, bg)
        * float(os.environ.get("SYNTH_COUNT_MULT", "1.0"))
    )
    # noise SCALED by the field -> the medium (dark) stays dark like the real data (black), the tissue has texture
    fn = field / max(float(field.max()), 1e-6)
    _res = bg.get("residuals")
    if os.environ.get("SYNTH_REAL_RESIDUAL", "0") == "1" and _res:
        vol = field + _res[int(rng.integers(len(_res)))] * float(
            os.environ.get("SYNTH_RESIDUAL_W", "1.0")
        )  # REAL TEXTURE (grad_mag)
    else:
        _nz = rng.normal(0, bg.get("noise_std", bg["std"]), shape).astype(np.float32)
        _ncorr = float(
            os.environ.get("SYNTH_NOISE_CORR", "0.0")
        )  # knob for correlating noise; default OFF: the sweep proved it does NOT move hi_freq
        if _ncorr > 0:
            _nz = gaussian_filter(
                _nz, _ncorr
            )  # the real cause of the overshoot = the low-pass of the REAL XY-pooling, which the native 64^3 synth does not replicate
        vol = field + _nz * (0.15 + 0.85 * fn)
    lo, hi = np.percentile(field, 50), np.percentile(
        field, 99
    )  # TISSUE MATRIX -> connected mass
    vol = vol + 0.38 * np.clip((field - lo) / (hi - lo + 1e-6), 0, 1)
    r = templates[0].shape[0] // 2
    min_sep = bg.get("min_sep_vox", min_sep)  # floor = REAL minimum NN (avoids overlap)
    # CLUSTERED placement (clusters like the real tissue) instead of a scattered Poisson
    coords = place_clustered(
        field,
        n_cells,
        rng,
        margin=r,
        gt_med=bg.get("gt_field_med"),
        gt_std=bg.get("gt_field_std"),
        light_bins=bg.get("light_bins"),
        light_pdf=bg.get("light_pdf"),
    )
    centers, boxes = [], []
    for zc, yc, xc in coords:
        if len(centers) >= n_cells:
            break
        msj = min_sep * float(
            rng.uniform(
                float(os.environ.get("SYNTH_MINSEP_JLO", "1.0")),
                float(os.environ.get("SYNTH_MINSEP_JHI", "1.0")),
            )
        )  # JITTER (iter7): broadens NN
        if (
            centers
            and np.linalg.norm(
                (np.array(centers) - [zc, yc, xc]) * EFF_VOXEL, axis=1
            ).min()
            < msj * EFF_VOXEL[0]
        ):
            continue  # rejects an overlapping centre -> distinct boxes
        t = augment_template(templates[rng.integers(len(templates))], rng)
        t = t * (
            0.4 + 1.0 * field[zc, yc, xc]
        )  # inherited brightness (less dominant than the tissue)
        rz, ry, rx = np.array(t.shape) // 2
        vol[zc - rz : zc + rz + 1, yc - ry : yc + ry + 1, xc - rx : xc + rx + 1] = (
            np.maximum(
                vol[
                    zc - rz : zc + rz + 1, yc - ry : yc + ry + 1, xc - rx : xc + rx + 1
                ],
                t,
            )
        )
        centers.append([zc, yc, xc])
        boxes.append([zc - rz, yc - ry, xc - rx, zc + rz + 1, yc + ry + 1, xc + rx + 1])
    vol = np.clip(vol, 0, None)
    if BLUR_SIGMA > 0:
        vol = gaussian_filter(
            vol, (BLUR_SIGMA * PSF_Z_RATIO, BLUR_SIGMA, BLUR_SIGMA)
        )  # final ANISOTROPIC PSF (z>xy)
    m = np.percentile(vol, 99)  # TISSUE level -> nuclei SATURATE at 1.0 (real peak)
    vol = np.clip(vol / m, 0, 1) if m > 0 else vol
    # v4: HISTOGRAM MATCHING -> matches the intensity distribution to that of a REAL frame
    ref = bg.get("hm_ref_canon")  # CANONICAL reference (median brightness) -> stable
    if ref is not None:
        try:
            from skimage.exposure import match_histograms

            hm = match_histograms(vol, ref).astype(np.float32)
            vol = np.clip(
                0.05 * vol + 0.95 * hm, 0, 1
            )  # strong HM to a fixed ref -> intensity ~ real, without coupling to the knobs
        except Exception:
            pass
    vol = apply_z_profile(
        vol, field_zprof(field)
    )  # Z PROFILE of the SPECIFIC FIELD (not the global mean -> no hump)
    vol = retarget_cell_intensity(
        vol, boxes, rng
    )  # per-cell mean -> data-driven Normal (matches the real data)
    vol = poisson_gaussian_noise(
        vol, rng
    )  # realistic camera noise (shot+read) -> real grain
    vol = match_power_spectrum(
        vol, bg, rng
    )  # NEW STAGE: real texture via Fourier (attacks grad_mag)
    return (
        np.clip(vol, 0, 1).astype(np.float32),
        np.array(boxes, np.int32),
        np.array(centers, np.float32),
    )


def rand_rot(rng):
    a, b, c = rng.uniform(0, 2 * np.pi, 3)
    ca, sa, cb, sb, cc, sc = (
        np.cos(a),
        np.sin(a),
        np.cos(b),
        np.sin(b),
        np.cos(c),
        np.sin(c),
    )
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rx = np.array([[1, 0, 0], [0, cc, -sc], [0, sc, cc]])
    return (Rz @ Ry @ Rx).astype(np.float32)


_E3D = np.random.default_rng(1234)
_E3D_FILT = _E3D.standard_normal((16, 3, 3, 3)).astype(
    np.float32
)  # fixed bank of 3d filters


def embed3d(patch, size=8):
    """3D vector-space embedding: normalizes size+brightness -> bank of random 3d filters
    (captures 3d SHAPE/STRUCTURE) -> unit vector. Inner product = shape similarity [-1,1].
    """
    p = np.asarray(patch, np.float32)
    if p.shape != (size, size, size):
        p = ndzoom(p, [size / s for s in p.shape], order=1)
    m = float(p.max())
    if m > 0:
        p = p / m
    p = p - p.mean()
    feats = []
    for f in _E3D_FILT:
        r = ndconvolve(p, f, mode="constant")
        feats += [float(r.mean()), float(r.std()), float(np.abs(r).max())]
    v = np.array(feats, np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def cell_sim(patch, real_embs):
    """Similarity (cosine, 0..1) of the cell to the MOST similar real cell in 3d space."""
    if real_embs is None or len(real_embs) == 0:
        return 1.0
    return float(np.max(real_embs @ embed3d(patch)))


def radius_by_gradient(t, nbins=16):
    """RADIUS by EDGE (steepest radial slope) -> robust to neighbours and to the window size.
    The old criterion (volume>half-maximum WITHIN the window) is biased: a small window clips the cell,
    a large window swallows neighbours (measured in biohub-radius-profile: r_half/r_bg hit the ceiling in dense
    tissue; r_grad does not). Returns the radius in VOXELS (effective isotropic)."""
    c = np.array(t.shape) // 2
    zz, yy, xx = np.indices(t.shape)
    d = np.sqrt(
        ((zz - c[0]) * EFF_VOXEL[0]) ** 2
        + ((yy - c[1]) * EFF_VOXEL[1]) ** 2
        + ((xx - c[2]) * EFF_VOXEL[2]) ** 2
    )
    rmax = float(d.max())
    edges = np.linspace(0, rmax, nbins + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    prof = np.full(nbins, np.nan, np.float32)
    for i in range(nbins):
        m = (d >= edges[i]) & (d < edges[i + 1])
        if m.sum() >= 3:
            prof[i] = float(t[m].mean())
    ok = ~np.isnan(prof)
    if ok.sum() < 4:
        return None
    ctr, prof = ctr[ok], prof[ok]
    r_um = float(ctr[int(np.argmin(np.gradient(prof)))])  # edge = steepest slope
    return max(0.8, r_um / EFF_VOXEL[0])  # um -> effective voxels


def measure_cell_stats(templates):
    """Measures the DISTRIBUTION of the real cells: size, peak, mean brightness, 3d EMBEDDING, elongation.
    SYNTH_RADIUS_MODE=halfmax (DEFAULT): volume>half-maximum in the window. Biased by the window, BUT validated.
    SYNTH_RADIUS_MODE=grad: radius by radial edge. Robust to neighbours on the FULL VOLUME (biohub-radius-profile),
      but VALIDATED LOCALLY AS BIASED on the small template: it underestimates by ~35-50% (the steepest slope of
      a gaussian is at r/sqrt(2), not at the edge) and it saturates with coarse binning (radius 2.0 and 3.0 -> both 1.30).
      Do NOT use as the default without calibrating. Kept as opt-in for experimentation.
    """
    RMODE = os.environ.get("SYNTH_RADIUS_MODE", "halfmax")
    radii, peaks, means, embs, aspect = [], [], [], [], []
    for t in templates:
        pk = float(t.max())
        if pk <= 0:
            continue
        r_g = radius_by_gradient(t) if RMODE == "grad" else None
        if r_g is None:
            mass = float(
                (t > pk * 0.5).sum()
            )  # fallback/old: volume above half-maximum
            r_g = (3 * mass / (4 * np.pi)) ** (1 / 3)
        radii.append(r_g)
        peaks.append(pk)
        means.append(
            float(t[t > pk * 0.3].mean())
        )  # MEAN brightness of the cell (active region)
        embs.append(embed3d(t))
    radii = np.array(radii) if radii else np.array([CELL_R_VOX], float)
    # PEAK/MEAN RATIO per REAL cell. `retarget_cell_intensity` only matched the MEAN -- nothing in the
    # pipeline aimed at the PEAK, and that is why cell_peak/cell_range/contrast were the worst appearance
    # axes (0.605/0.605/0.702 on the scoreboard). Storing it here (the only place that sees peaks AND means together)
    # avoids having to propagate the distribution through the 5 sites that set _CELL_MEANS_RT.
    global _CELL_RATIO_RT
    if peaks and means:
        _r = np.asarray(peaks, float) / np.maximum(np.asarray(means, float), 1e-6)
        _CELL_RATIO_RT = _r[np.isfinite(_r) & (_r > 1.0) & (_r < 8.0)]
    return dict(
        radii=radii,
        r_med=float(np.median(radii)),
        peaks=np.array(peaks) if peaks else np.array([0.5], float),
        means=np.array(means) if means else np.array([0.3], float),
        embs=np.array(embs) if embs else None,
    )


def study_cell_development(stats):
    """STUDY on the REAL data: SIZE distribution + how YOUNG (small) cells differ from OLD ones."""
    r, pk, mn = stats["radii"], stats["peaks"], stats["means"]
    if len(r) < 4:
        return
    med = float(np.median(r))
    yo = r < med
    ol = r >= med
    print("=== ESTUDO DE DESENVOLVIMENTO CELULAR (real) ===")
    print(f"  raio(vox): min {r.min():.2f} | mediana {med:.2f} | max {r.max():.2f}")
    print(
        f"  JOVENS (r<med): n={int(yo.sum())} brilho_med {mn[yo].mean():.3f} pico {pk[yo].mean():.3f}"
    )
    print(
        f"  VELHAS (r>=med): n={int(ol.sum())} brilho_med {mn[ol].mean():.3f} pico {pk[ol].mean():.3f}"
    )
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].hist(r, bins=25, color="purple", alpha=0.8)
    ax[0].axvline(med, ls="--", c="k", label=f"mediana {med:.1f}")
    ax[0].set_title("Distribuicao de TAMANHO (raio vox)")
    ax[0].set_xlabel("raio")
    ax[0].legend()
    ax[1].scatter(r, mn, s=10, alpha=0.5)
    ax[1].set_title("Tamanho vs brilho medio")
    ax[1].set_xlabel("raio")
    ax[1].set_ylabel("brilho")
    ax[2].scatter(r, pk, s=10, alpha=0.5, c="orange")
    ax[2].set_title("Tamanho vs pico de intensidade")
    ax[2].set_xlabel("raio")
    plt.suptitle(
        "Desenvolvimento celular: JOVENS (pequenas) vs VELHAS (grandes) - REAL",
        weight="bold",
    )
    plt.savefig(FIG / "cell_development.png", dpi=180, bbox_inches="tight")
    plt.show()


def fbm_texture_3d(shape, rng, H=None, octaves=None, lacunarity=2.0, base=2):
    """3D fractional-Brownian noise (fBM): PHASE-CORRECT, structured chromatin texture.
    Sum of octaves of value-noise; amplitude ~ gain^o (gain=2^-H), frequency ~ lacunarity^o.
    H=0.7 matches the statistics of real chromatin (P(f) ~ f^-3.4). Applied BEFORE the PSF -> the optical
    blur smears the texture like a real signal. This is the link that PSMATCH (magnitude only) did not capture:
    spectrum matching fixes the magnitude but keeps the synthetic PHASE; fBM injects phase-correlated
    high-freq structure (what was pinning hi_freq at KS~0.5)."""
    H = (
        float(os.environ.get("SYNTH_FBM_H", "0.85")) if H is None else H
    )  # 0.85 tuned in biohub-gs-fbm (0.7 overshoots: hi_freq 0.96)
    octaves = int(os.environ.get("SYNTH_FBM_OCT", "6")) if octaves is None else octaves
    gain = 2.0 ** (-H)  # 0.62 for H=0.7
    field = np.zeros(shape, np.float32)
    amp = 1.0
    tot = 0.0
    for o in range(octaves):
        res = tuple(min(int(base * (lacunarity**o)), d) for d in shape)
        res = tuple(max(2, r) for r in res)
        small = rng.standard_normal(res).astype(np.float32)
        oct_n = (
            small
            if small.shape == tuple(shape)
            else ndzoom(small, [shape[i] / small.shape[i] for i in range(3)], order=1)
        )
        field += amp * oct_n
        tot += amp
        amp *= gain
    field /= tot + 1e-9
    rng_ = float(field.max() - field.min()) + 1e-6
    return ((field - field.min()) / rng_).astype(np.float32)  # [0,1] normalized fBM


def cell_texture(shape, rng, strength=None):
    """INTERNAL chromatin TEXTURE. Now multi-octave fBM (H=0.7, phase-correct) instead of 1 scale.
    Multiplier ~[1-s, 1+s]. Applied inside the nucleus BEFORE the blur/PSF."""
    s = float(os.environ.get("SYNTH_TEXTURE", "0.30")) if strength is None else strength
    if s <= 0:
        return np.ones(shape, np.float32)
    if os.environ.get("SYNTH_FBM", "1") == "1":
        tex = fbm_texture_3d(shape, rng)  # [0,1] phase-correct multi-scale
    else:
        small = rng.standard_normal(tuple(max(2, d // 2) for d in shape)).astype(
            np.float32
        )
        tex = ndzoom(small, [shape[i] / small.shape[i] for i in range(3)], order=1)
        tex = gaussian_filter(tex, 0.7)
        rng_ = float(tex.max() - tex.min()) + 1e-6
        tex = (tex - tex.min()) / rng_
    return (1 + s * (tex - 0.5) * 2).astype(np.float32)  # chromatin granularity


def gen_param_cell(rng, stats, stage=None, shape_kind=None):
    """PARAMETRIC cell: samples COHERENT (size, brightness) from ONE real cell (preserves the
    measured size-brightness relation). Stage (young/old), ELLIPTICAL variation and smooth TRANSITION.
    """
    i = rng.integers(len(stats["radii"]))  # ONE real cell -> coherent size+brightness
    r = float(stats["radii"][i])
    inten = float(stats["means"][i]) * max(
        0.2, rng.normal(1.3, 0.08)
    )  # REAL brightness (~0.65x lower, cell-mean matches 0.29) + narrow normal
    rmed = float(stats.get("r_med", r))
    # explicit STAGE (gallery) OR implicit in the sampled size (young small / old large)
    if stage == "early":
        r *= rng.uniform(0.55, 0.75)
        inten *= rng.uniform(0.7, 0.9)
    elif stage == "dev":
        r *= rng.uniform(1.0, 1.2)
    young = r < rmed
    # ELLIPTICAL SHAPE: real nuclei are rarely perfect spheres. ALL of them have slight anisotropy;
    # a fraction (old/dividing) are clearly elongated. Axes = 3 independent semi-axes.
    p_ellip = 0.30 if young else 0.60  # most are elliptical to some degree
    aniso = rng.uniform(0.82, 1.22, 3)  # base anisotropy (every nucleus)
    if shape_kind == "ellip" or (shape_kind is None and rng.random() < p_ellip):
        aniso[rng.integers(3)] *= rng.uniform(
            1.5, 2.2
        )  # 1 clearly elongated axis (markedly elliptical)
    axes = np.maximum(r * aniso, 0.6)
    size = int(np.ceil(axes.max()) * 2) + 5  # +5: margin for the edge
    zz, yy, xx = np.indices((size, size, size))
    c = size // 2
    P = np.stack([zz - c, yy - c, xx - c], -1).astype(np.float32) @ rand_rot(rng).T
    r_norm = np.sqrt(
        (P[..., 0] / axes[0]) ** 2
        + (P[..., 1] / axes[1]) ** 2
        + (P[..., 2] / axes[2]) ** 2
    )  # =1 at the surface
    # PROFILE = SUPER-GAUSSIAN (FILLED disc, SHARP edge), NOT a gaussian. The probe measured the real
    # nucleus: intensity ~0.63 of the centre at half-distance (flat top) and falling to 0 at the edge -- a
    # gaussian would already be at 0.40. exp(-(r_norm)^n) with high n = plateau in the core + abrupt falloff at the edge.
    n = float(os.environ.get("SYNTH_CELL_SHARPNESS", "7"))
    cell = (np.exp(-(r_norm**n)) * inten).astype(np.float32)
    cell = cell * cell_texture(
        cell.shape, rng
    )  # chromatin TEXTURE (not a smooth sphere)
    _bl = float(
        os.environ.get("SYNTH_CELL_EDGEBLUR", "0.25")
    )  # almost sharp edge (not a wide gaussian)
    return gaussian_filter(cell, _bl) if _bl > 0 else cell


def add_divisions(vol, boxes, centers, field, stats, rng):
    """CELL GENERATION (division): a DIV_RATE fraction of the cells spawns a 2nd nucleus nearby (dumbbell)
    = cells being born, at the real RATE measured on the GT lineage. Models the rare mitotic morphology.
    """
    dr = float(
        os.environ.get("SYNTH_DIVRATE", str(DIV_RATE))
    )  # TUNABLE division rate (optimizer)
    if dr <= 0 or not centers:
        return vol, boxes, centers
    Z, Y, X = vol.shape
    R = CELL_R_VOX
    _slo = float(os.environ.get("SYNTH_SISTER_LO", "1.2"))
    _shi = float(os.environ.get("SYNTH_SISTER_HI", "1.8"))  # sister distance
    for zc, yc, xc in list(centers):
        if rng.random() >= dr:
            continue
        off = rng.normal(0, 1, 3)
        off = off / (np.linalg.norm(off) + 1e-6) * rng.uniform(_slo, _shi) * R
        dz, dy, dx = int(zc + off[0]), int(yc + off[1]), int(xc + off[2])
        # EMERGING CELL (just divided) = YOUNG: smaller, rounder, new-cell brightness (early stage)
        cell = gen_param_cell(rng, stats, stage="early")
        cell = cell * (
            0.96
            + 0.05
            * field[np.clip(dz, 0, Z - 1), np.clip(dy, 0, Y - 1), np.clip(dx, 0, X - 1)]
        )
        rz, ry, rx = np.array(cell.shape) // 2
        if (
            dz - rz < 0
            or dy - ry < 0
            or dx - rx < 0
            or dz + rz + 1 > Z
            or dy + ry + 1 > Y
            or dx + rx + 1 > X
        ):
            continue
        vol[dz - rz : dz + rz + 1, dy - ry : dy + ry + 1, dx - rx : dx + rx + 1] = (
            np.maximum(
                vol[
                    dz - rz : dz + rz + 1, dy - ry : dy + ry + 1, dx - rx : dx + rx + 1
                ],
                cell,
            )
        )
        # also SHRINKS the mother cell slightly (it just divided) -> realistic mother-daughter pair
        centers.append([dz, dy, dx])
        boxes.append([dz - R, dy - R, dx - R, dz + R + 1, dy + R + 1, dx + R + 1])
    return vol, boxes, centers


def gen_volume_param(shape, bg, stats, n_cells, rng, min_sep=2):
    """VERSION D: 100% parametric cells (no template pasting). Real background + placement by density."""
    Z, Y, X = shape
    fields = bg.get("fields")
    _fi = int(rng.integers(len(fields))) if fields else 0
    field = fields[_fi].copy() if fields else np.full(shape, bg["p50"], np.float32)
    # REAL density x mult (nn_um). SYNTH_NEST only counts if set EXPLICITLY: without that the measured
    # `bg['n_cells_real']` overwrites the default and the knob is DEAD (the biohub-newvars sweep
    # burned 4 SYNTH_NEST configs that gave an identical mean_KS to the 16th decimal place).
    _base = (
        N_EST_PER_FRAME
        if os.environ.get("SYNTH_NEST")
        else bg.get("n_cells_real", N_EST_PER_FRAME)
    )
    n_cells = int(
        field_n_cells(field, _base, bg)
        * float(os.environ.get("SYNTH_COUNT_MULT", "1.0"))
    )
    fn = field / max(float(field.max()), 1e-6)
    _res = bg.get("residuals")
    # CORRECT PHYSICAL MODEL (SYNTH_DARK_MEDIUM=1): the base is the DARK LIQUID the embryo is
    # immersed in -- NOT the smoothed real frame (which already contains the mean brightness of the cells -> a bright cloud).
    # The CELLS are the light sources; the ambient brightness EMERGES from the diffusion of their PSF, not from a
    # painted background light. Without this the background takes on the cell colour and the image blurs (defect seen in the
    # showcase: real = sharp dots in the dark, synth = a smear). `field` still decides WHERE the
    # cells go (placement), but it is no longer the visual background.
    DARK = os.environ.get("SYNTH_DARK_MEDIUM", "0") == "1"
    if DARK:
        med = float(
            os.environ.get("SYNTH_MEDIUM_LEVEL", "0.04")
        )  # autofluorescence of the liquid (low)
        _nz = rng.normal(0, bg.get("noise_std", bg["std"]), shape).astype(np.float32)
        vol = np.clip(med + _nz * 0.5, 0, None).astype(
            np.float32
        )  # dark liquid + noise; NO tissue
        tissue = np.zeros(shape, np.float32)
    elif os.environ.get("SYNTH_REAL_RESIDUAL", "0") == "1" and _res:
        vol = field + _res[int(rng.integers(len(_res)))] * float(
            os.environ.get("SYNTH_RESIDUAL_W", "1.0")
        )  # REAL TEXTURE (grad_mag)
    else:
        _nz = rng.normal(0, bg.get("noise_std", bg["std"]), shape).astype(np.float32)
        _ncorr = float(
            os.environ.get("SYNTH_NOISE_CORR", "0.0")
        )  # knob for correlating noise; default OFF: the sweep proved it does NOT move hi_freq
        if _ncorr > 0:
            _nz = gaussian_filter(
                _nz, _ncorr
            )  # the real cause of the overshoot = the low-pass of the REAL XY-pooling, which the native 64^3 synth does not replicate
        vol = field + _nz * (0.15 + 0.85 * fn)
    # bright TISSUE MATRIX: fills the space between cells -> CONNECTED MASS (not dark holes)
    if not DARK:
        lo, hi = np.percentile(field, 50), np.percentile(field, 99)
        tissue = np.clip((field - lo) / (hi - lo + 1e-6), 0, 1)
    vol = (
        vol + float(os.environ.get("SYNTH_TISSUE", "0.55")) * tissue
    )  # brighter TISSUE -> cells embed into it (less contrast)
    # DECOUPLED BLUR: the pipeline applied ONE global filter at the end, blurring background and cells together.
    # That created an artificial TENSION measured in the report: raising the blur fixed `hi_freq` (background
    # texture dominates the high frequencies) but sank `count` (blurred cells merge and the DoG
    # detects fewer). With SYNTH_DECOUPLE_BLUR=1 the cells go to their own layer: the BACKGROUND takes
    # SYNTH_BG_BLUR and the CELLS take SYNTH_BLUR (the real optical PSF), each at its own value.
    _dec = os.environ.get("SYNTH_DECOUPLE_BLUR", "0") == "1"
    cell_layer = np.zeros_like(vol) if _dec else None
    dst = cell_layer if _dec else vol  # where the cells are composited
    min_sep = bg.get("min_sep_vox", min_sep)  # floor = REAL minimum NN (avoids overlap)
    fcs = bg.get("field_cens")
    if (
        os.environ.get("SYNTH_PLACEMENT", "field") == "real"
        and fcs
        and _fi < len(fcs)
        and len(fcs[_fi]) >= 3
    ):
        # REAL PLACEMENT: uses the real DETECTED POSITIONS of this field -> the NN distribution matches the real data by construction
        rc = dedup_cens(np.atleast_2d(fcs[_fi]).astype(np.float32), min_sep)
        hi = np.array(shape, np.float32) - CELL_R_VOX - 1
        rc = np.clip(
            rc, CELL_R_VOX, hi
        )  # within bounds (avoids an IndexError at the edge)
        m = int(
            rng.integers(max(1, int(0.85 * len(rc))), len(rc) + 1)
        )  # light subsample (variation)
        coords = rc[rng.choice(len(rc), size=min(m, len(rc)), replace=False)].astype(
            int
        )
    else:
        coords = place_clustered(
            field,
            n_cells,
            rng,
            margin=6,
            gt_med=bg.get("gt_field_med"),
            gt_std=bg.get("gt_field_std"),
            light_bins=bg.get("light_bins"),
            light_pdf=bg.get("light_pdf"),
        )  # where the REAL cells are
    centers, boxes, centers_sim = [], [], []
    for zc, yc, xc in coords:
        if len(centers) >= n_cells:
            break
        msj = min_sep * float(
            rng.uniform(
                float(os.environ.get("SYNTH_MINSEP_JLO", "1.0")),
                float(os.environ.get("SYNTH_MINSEP_JHI", "1.0")),
            )
        )  # JITTER (iter7): broadens NN
        if (
            centers
            and np.linalg.norm(
                (np.array(centers) - [zc, yc, xc]) * EFF_VOXEL, axis=1
            ).min()
            < msj * EFF_VOXEL[0]
        ):
            continue  # rejects an overlapping centre
        if os.environ.get("SYNTH_CELL_SOURCE", "param") == "template" and _TEMPLATES_RT:
            # HYBRID: cell = augmented REAL TEMPLATE (REAL appearance/size/texture) at the parametric placement
            best = augment_template(
                _TEMPLATES_RT[rng.integers(len(_TEMPLATES_RT))], rng
            )
            centers_sim.append(1.0)
        else:
            # GATE >=70%: resamples until the cell is >=70% similar to a REAL cell (3d space)
            cell = gen_param_cell(rng, stats)
            best = cell
            bs = cell_sim(cell, stats.get("embs"))
            for _t in range(5):
                if bs >= SIM_GATE:
                    break
                cand = gen_param_cell(rng, stats)
                cs = cell_sim(cand, stats.get("embs"))
                if cs > bs:
                    best, bs = cand, cs
            centers_sim.append(bs)
        cell = best * (
            0.96 + 0.05 * field[zc, yc, xc]
        )  # near-constant brightness inherited from the tissue
        rz, ry, rx = np.array(cell.shape) // 2
        if (
            zc - rz < 0
            or yc - ry < 0
            or xc - rx < 0
            or zc + rz + 1 > Z
            or yc + ry + 1 > Y
            or xc + rx + 1 > X
        ):
            continue
        sub = dst[zc - rz : zc + rz + 1, yc - ry : yc + ry + 1, xc - rx : xc + rx + 1]
        _sb = float(
            os.environ.get("SYNTH_SOFT_BLEND", "0.0")
        )  # smooth cell-tissue EDGE (attacks grad_mag): alpha=cell
        if _sb > 0:
            a = (
                np.clip(cell / (float(cell.max()) + 1e-6), 0, 1) * _sb
            )  # alpha proportional to the cell intensity
            dst[zc - rz : zc + rz + 1, yc - ry : yc + ry + 1, xc - rx : xc + rx + 1] = (
                np.maximum(sub, (1 - a) * sub + a * cell)
            )  # smooth transition, no hard edge
        else:
            dst[zc - rz : zc + rz + 1, yc - ry : yc + ry + 1, xc - rx : xc + rx + 1] = (
                np.maximum(sub, cell)
            )
        R = CELL_R_VOX  # TIGHT box ±R (convention of the real GT), not the array padding
        centers.append([zc, yc, xc])
        boxes.append([zc - R, yc - R, xc - R, zc + R + 1, yc + R + 1, xc + R + 1])
    # DIVISIONS (measured real rate): a fraction of the cells SPAWNS a 2nd nucleus nearby (dumbbell) = cells BEING BORN
    dst, boxes, centers = add_divisions(dst, boxes, centers, field, stats, rng)
    if _dec:
        # BACKGROUND and CELLS with independent PSFs, then composited. Breaks the hi_freq vs count tension.
        _bgb = float(os.environ.get("SYNTH_BG_BLUR", str(BLUR_SIGMA)))
        if _bgb > 0:
            vol = gaussian_filter(vol, (_bgb * PSF_Z_RATIO, _bgb, _bgb))
        if BLUR_SIGMA > 0:
            dst = gaussian_filter(
                dst, (BLUR_SIGMA * PSF_Z_RATIO, BLUR_SIGMA, BLUR_SIGMA)
            )
        vol = np.maximum(vol, dst)
    vol = finish_volume(
        vol, boxes, field, bg, rng, skip_blur=_dec
    )  # blur already applied per layer
    return vol, np.array(boxes, np.int32), np.array(centers, np.float32)


def finish_volume(vol, boxes, field, bg, rng, targets=None, skip_blur=False):
    """Optical/camera FINALIZATION (order converged on by the optimizers): PSF -> residual -> normalization
    -> HM -> Z profile -> per-cell intensity retarget -> Poisson-Gauss noise -> power-spectrum match.
    Extracted from gen_volume_param to be a SINGLE SOURCE: the temporal generator (synth_lineage) calls this
    same function, otherwise the frames of a sequence would fall outside the already calibrated distribution.
    """
    vol = np.clip(vol, 0, None)
    if (
        BLUR_SIGMA > 0 and not skip_blur
    ):  # skip: decoupled mode already blurred per layer
        vol = gaussian_filter(
            vol, (BLUR_SIGMA * PSF_Z_RATIO, BLUR_SIGMA, BLUR_SIGMA)
        )  # ANISOTROPIC PSF (z>xy)
    if os.environ.get("SYNTH_RESIDUAL_POST", "0") == "1" and bg.get("residuals"):
        _rp = bg["residuals"][
            int(rng.integers(len(bg["residuals"])))
        ]  # REAL residual after the blur -> texture survives (grad_mag)
        vol = vol + _rp * float(os.environ.get("SYNTH_RESIDUAL_POST_W", "0.7"))
    mm = np.percentile(
        vol, 99.6
    )  # p99.6 -> nuclei do NOT fully saturate (less pure white)
    vol = np.clip(vol / mm, 0, 1) if mm > 0 else vol
    # HISTOGRAM MATCHING: forces the volume histogram to match a REAL frame (bright cloud). In DARK
    # mode this would RE-RAISE the dark background we just created (the real data contains the mean brightness
    # of the cells) -> it would destroy the crispness. That is why it is DISABLED when SYNTH_DARK_MEDIUM=1: in the
    # correct physical model, the intensity distribution EMERGES from the cells+PSF, it is not imposed.
    ref = bg.get("hm_ref_canon")
    if ref is not None and os.environ.get("SYNTH_DARK_MEDIUM", "0") != "1":
        try:
            from skimage.exposure import match_histograms

            vol = np.clip(
                0.05 * vol + 0.95 * match_histograms(vol, ref).astype(np.float32), 0, 1
            )
        except Exception:
            pass
    vol = apply_z_profile(
        vol, field_zprof(field)
    )  # Z PROFILE of the SPECIFIC FIELD (not the global mean -> no hump)
    vol = retarget_cell_intensity(
        vol, boxes, rng, targets=targets
    )  # per-cell mean -> data-driven Normal (matches the real data)
    vol = poisson_gaussian_noise(
        vol, rng
    )  # realistic camera noise (shot+read) -> real grain
    vol = match_power_spectrum(
        vol, bg, rng
    )  # NEW STAGE: real texture via Fourier (attacks grad_mag)
    return np.clip(vol, 0, 1).astype(np.float32)


def dedup_cens(cens, min_sep_vox):
    """Removes duplicate centroids (GT + DoG of the same cell) closer than min_sep."""
    out = []
    for c in np.atleast_2d(cens).astype(np.float32):
        if (
            not out
            or np.linalg.norm((np.array(out) - c) * EFF_VOXEL, axis=1).min()
            >= min_sep_vox * EFF_VOXEL[0]
        ):
            out.append(c.tolist())
    return np.array(out) if out else np.zeros((0, 3))


def gen_volume_layout(shape, bg, stats, rng, real_bg=True):
    """LAYOUT TRANSFER (Fable #1): synthetic cells at the REAL POSITIONS (dense centroids).
    Layout/density/clustering 100% REAL -> ZERO placement error (the killer of sim-to-real).
    Dense+perfect labels = the centroids themselves. real_bg=True: background=real field (max fidelity).
    """
    Z, Y, X = shape
    fcs = bg.get("field_cens")
    fields = bg.get("fields")
    if not fcs or not fields:
        return gen_volume_param(shape, bg, stats, N_EST_PER_FRAME, rng)
    i = rng.integers(len(fcs))
    field = fields[i].copy()
    cens = dedup_cens(
        fcs[i], bg.get("min_sep_vox", 3)
    )  # REAL positions (without duplicating the same cell)
    fn = field / max(float(field.max()), 1e-6)
    _res = bg.get("residuals")
    if os.environ.get("SYNTH_REAL_RESIDUAL", "0") == "1" and _res:
        vol = field + _res[int(rng.integers(len(_res)))] * float(
            os.environ.get("SYNTH_RESIDUAL_W", "1.0")
        )  # REAL TEXTURE (grad_mag)
    else:
        _nz = rng.normal(0, bg.get("noise_std", bg["std"]), shape).astype(np.float32)
        _ncorr = float(
            os.environ.get("SYNTH_NOISE_CORR", "0.0")
        )  # knob for correlating noise; default OFF: the sweep proved it does NOT move hi_freq
        if _ncorr > 0:
            _nz = gaussian_filter(
                _nz, _ncorr
            )  # the real cause of the overshoot = the low-pass of the REAL XY-pooling, which the native 64^3 synth does not replicate
        vol = field + _nz * (0.15 + 0.85 * fn)
    if not real_bg:  # IP-safe: adds a synthetic tissue matrix
        lo, hi = np.percentile(field, 50), np.percentile(field, 99)
        vol = vol + 0.38 * np.clip((field - lo) / (hi - lo + 1e-6), 0, 1)
    R = CELL_R_VOX
    boxes, centers = [], []
    for zc, yc, xc in np.round(cens).astype(int):
        zc, yc, xc = (
            int(np.clip(zc, R, Z - R - 1)),
            int(np.clip(yc, R, Y - R - 1)),
            int(np.clip(xc, R, X - R - 1)),
        )
        cell = gen_param_cell(rng, stats) * (
            0.96 + 0.05 * field[zc, yc, xc]
        )  # near-constant -> preserves the normal
        rz, ry, rx = np.array(cell.shape) // 2
        if (
            zc - rz < 0
            or yc - ry < 0
            or xc - rx < 0
            or zc + rz + 1 > Z
            or yc + ry + 1 > Y
            or xc + rx + 1 > X
        ):
            continue
        vol[zc - rz : zc + rz + 1, yc - ry : yc + ry + 1, xc - rx : xc + rx + 1] = (
            np.maximum(
                vol[
                    zc - rz : zc + rz + 1, yc - ry : yc + ry + 1, xc - rx : xc + rx + 1
                ],
                cell,
            )
        )
        boxes.append([zc - R, yc - R, xc - R, zc + R + 1, yc + R + 1, xc + R + 1])
        centers.append([zc, yc, xc])
    vol = np.clip(vol, 0, None)
    if BLUR_SIGMA > 0:
        vol = gaussian_filter(vol, (BLUR_SIGMA * PSF_Z_RATIO, BLUR_SIGMA, BLUR_SIGMA))
    mm = np.percentile(vol, 99)
    vol = np.clip(vol / mm, 0, 1) if mm > 0 else vol
    ref = bg.get("hm_ref_canon")
    if ref is not None:  # HM always -> intensity matches the real data (tight)
        try:
            from skimage.exposure import match_histograms

            vol = np.clip(
                0.05 * vol + 0.95 * match_histograms(vol, ref).astype(np.float32), 0, 1
            )
        except Exception:
            pass
    vol = apply_z_profile(vol, field_zprof(field))
    vol = retarget_cell_intensity(
        vol, boxes, rng
    )  # per-cell mean -> data-driven Normal (matches the real data)
    vol = poisson_gaussian_noise(
        vol, rng
    )  # realistic camera noise (shot+read) -> real grain
    vol = match_power_spectrum(
        vol, bg, rng
    )  # NEW STAGE: real texture via Fourier (attacks grad_mag)
    return (
        np.clip(vol, 0, 1).astype(np.float32),
        np.array(boxes, np.int32),
        np.array(centers, np.float32),
    )


def render_at(field, positions, stats, bg, rng):
    """Renders synthetic cells AT the given positions + finalizes (blur/HM/z). Basis of the temporal generator."""
    Z, Y, X = field.shape
    R = CELL_R_VOX
    fn = field / max(float(field.max()), 1e-6)
    vol = field + rng.normal(0, bg.get("noise_std", bg["std"]), field.shape).astype(
        np.float32
    ) * (0.15 + 0.85 * fn)
    boxes = []
    for zc, yc, xc in np.round(positions).astype(int):
        zc, yc, xc = (
            int(np.clip(zc, R, Z - R - 1)),
            int(np.clip(yc, R, Y - R - 1)),
            int(np.clip(xc, R, X - R - 1)),
        )
        cell = gen_param_cell(rng, stats) * (0.96 + 0.05 * field[zc, yc, xc])
        rz, ry, rx = np.array(cell.shape) // 2
        if (
            zc - rz < 0
            or yc - ry < 0
            or xc - rx < 0
            or zc + rz + 1 > Z
            or yc + ry + 1 > Y
            or xc + rx + 1 > X
        ):
            continue
        vol[zc - rz : zc + rz + 1, yc - ry : yc + ry + 1, xc - rx : xc + rx + 1] = (
            np.maximum(
                vol[
                    zc - rz : zc + rz + 1, yc - ry : yc + ry + 1, xc - rx : xc + rx + 1
                ],
                cell,
            )
        )
        boxes.append([zc - R, yc - R, xc - R, zc + R + 1, yc + R + 1, xc + R + 1])
    vol = np.clip(vol, 0, None)
    if BLUR_SIGMA > 0:
        vol = gaussian_filter(vol, (BLUR_SIGMA * PSF_Z_RATIO, BLUR_SIGMA, BLUR_SIGMA))
    mm = np.percentile(vol, 99)
    vol = np.clip(vol / mm, 0, 1) if mm > 0 else vol
    ref = bg.get("hm_ref_canon")
    if ref is not None:
        try:
            from skimage.exposure import match_histograms

            vol = np.clip(
                0.05 * vol + 0.95 * match_histograms(vol, ref).astype(np.float32), 0, 1
            )
        except Exception:
            pass
    vol = apply_z_profile(vol, field_zprof(field))
    return np.clip(vol, 0, 1).astype(np.float32), np.array(boxes, np.int32)


def resolve_collisions(pos, minsep_vox, iters=150, shape=None):
    """Robust REPULSION: iterates UNTIL no cell is < min_sep. INTERNAL clip at every step (the edge does not
    push back into a collision). 150 iters converge even at the high real density -> ZERO clashes/contact.
    """
    gate = minsep_vox * EFF_VOXEL[0]
    hi = (np.array(shape) - CELL_R_VOX - 1) if shape is not None else None
    for _ in range(iters):
        D = pos[:, None] - pos[None]
        dm = np.linalg.norm(D * EFF_VOXEL, axis=2)
        np.fill_diagonal(dm, 1e9)
        if float(dm.min()) >= gate:
            break  # resolved -> stop
        ii, jj = np.where(dm < gate)
        for a, b in zip(ii, jj):
            if a < b:
                v = D[a, b].astype(np.float32)
                nv = float(np.linalg.norm(v))
                if nv < 1e-4:
                    v = pos[a] - pos.mean(0)
                    nv = (
                        float(np.linalg.norm(v)) + 1e-4
                    )  # coincident: separate radially
                push = (v / nv) * ((gate - dm[a, b]) / EFF_VOXEL[0]) * 0.55
                pos[a] += push
                pos[b] -= push
        if hi is not None:
            pos = np.clip(
                pos, CELL_R_VOX, hi
            )  # clip INSIDE the loop (the edge does not push back)
    return pos


def gen_sequence(bg, stats, rng, motion, T=8):
    """TEMPORAL SEQUENCE: cells move with directional PERSISTENCE + REAL speed, with anti-collision
    REPULSION. Each cell keeps its ID (trackable). Returns [(vol, boxes, ids, pos)]."""
    fields = bg.get("fields")
    fcs = bg.get("field_cens")
    if not fcs or not fields:
        return []
    i = rng.integers(len(fcs))
    field = fields[i]
    pos = dedup_cens(fcs[i], bg.get("min_sep_vox", 3)).astype(np.float32)
    if len(pos) < 3:
        return []
    ids = np.arange(len(pos))
    shp = np.array(field.shape)
    speed = motion["speed_med"] / EFF_VOXEL[0]  # voxels/frame (real)
    persist = motion["persist"]
    minsep = bg.get("min_sep_vox", 3)
    vel = rng.normal(0, speed, (len(pos), 3)).astype(np.float32)
    frames = []
    for t in range(T):
        vol, boxes = render_at(field, pos, stats, bg, rng)
        frames.append((vol, boxes, ids.copy(), pos.copy()))
        # MOVE: persistent velocity (real direction) + noise, calibrated to the real speed
        nv = persist * vel + (1 - persist) * rng.normal(0, speed, (len(pos), 3)).astype(
            np.float32
        )
        sp = np.linalg.norm(nv, axis=1, keepdims=True)
        nv = (
            nv / (sp + 1e-6) * speed * rng.uniform(0.4, 1.3, (len(pos), 1))
        )  # real speed with variation
        pos = pos + nv
        vel = nv
        pos = np.clip(pos, CELL_R_VOX, shp - CELL_R_VOX - 1)
        pos = resolve_collisions(
            pos, minsep, shape=shp
        )  # AVOIDS CLASHES/CONTACT (last, with internal clip)
    return frames


def visualize_motion(bg, stats, rng, motion):
    """Visualizes the temporal MOTION: trajectories + COLLISION check (minimum distance per frame)."""
    frames = gen_sequence(bg, stats, rng, motion, T=8)
    if not frames:
        print("  [motion] sem sequencia")
        return
    traj = np.array([f[3] for f in frames])  # (T, N, 3) voxels
    contact = CELL_DIAM_UM * 0.5  # 1 radius = contact
    mind = []
    for f in frames:
        p = f[3]
        D = np.linalg.norm((p[:, None] - p[None]) * EFF_VOXEL, axis=2)
        np.fill_diagonal(D, 1e9)
        mind.append(float(D.min()) if len(p) > 1 else 99.0)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    for n in range(traj.shape[1]):
        ax[0].plot(traj[:, n, 2], traj[:, n, 1], "-", lw=0.7, alpha=0.5)
    ax[0].scatter(traj[0, :, 2], traj[0, :, 1], s=10, c="lime", label="inicio")
    ax[0].scatter(traj[-1, :, 2], traj[-1, :, 1], s=10, c="red", label="fim")
    ax[0].set_title("Trajetorias das celulas (movimento realista)")
    ax[0].invert_yaxis()
    ax[0].legend()
    ax[1].plot(range(len(frames)), mind, "-o")
    ax[1].axhline(contact, ls="--", c="r", label=f"contato ({contact:.1f}um)")
    ax[1].set_title("Distancia MINIMA entre celulas/frame (colisao?)")
    ax[1].set_xlabel("frame")
    ax[1].set_ylabel("um")
    ax[1].legend()
    ax[1].set_ylim(0, None)
    ax[2].imshow(frames[-1][0].max(0), cmap="magma")
    ax[2].set_title(f"frame final (MIP-Z)")
    ax[2].axis("off")
    plt.suptitle(
        f"Movimentacao temporal: {motion['speed_med']:.1f}um/frame, persistencia {motion['persist']:.2f} | dist min {min(mind):.1f}um "
        f"({'SEM colisao' if min(mind) >= contact*0.8 else 'ATENCAO'})",
        weight="bold",
    )
    plt.savefig(FIG / "movimentacao.png", dpi=170, bbox_inches="tight")
    plt.show()
    print(
        f"  [motion] dist minima entre celulas ao longo do tempo = {min(mind):.1f}um (contato {contact:.1f}um) -> "
        f"{'SEM colisao/contato' if min(mind) >= contact*0.8 else 'ATENCAO: colisao'}"
    )


def patch_embedding(patch):
    p = patch.astype(np.float32)
    c = np.array(p.shape) // 2
    zz, yy, xx = np.indices(p.shape)
    d = np.sqrt(
        ((zz - c[0]) * EFF_VOXEL[0]) ** 2
        + ((yy - c[1]) * EFF_VOXEL[1]) ** 2
        + ((xx - c[2]) * EFF_VOXEL[2]) ** 2
    )
    bins = np.linspace(0, d.max() + 1e-6, 6)
    radial = [
        (
            p[(d >= bins[i]) & (d < bins[i + 1])].mean()
            if ((d >= bins[i]) & (d < bins[i + 1])).any()
            else 0.0
        )
        for i in range(5)
    ]
    gz, gy, gx = np.gradient(p)
    gmag = np.sqrt(gz**2 + gy**2 + gx**2)
    return np.array(
        radial
        + [
            p.mean(),
            p.std(),
            p.max(),
            np.percentile(p, 90),
            gmag.mean(),
            gmag.std(),
            float((p > p.mean()).mean()),
        ],
        np.float32,
    )


def embed_set(ps):
    return np.stack([patch_embedding(p) for p in ps]) if ps else np.zeros((0, 12))


def frechet_distance(A, B):
    from scipy.linalg import sqrtm

    mu1, mu2 = A.mean(0), B.mean(0)
    s1 = np.cov(A, rowvar=False) + 1e-6 * np.eye(A.shape[1])
    s2 = np.cov(B, rowvar=False) + 1e-6 * np.eye(B.shape[1])
    cm = sqrtm(s1 @ s2)
    cm = cm.real if np.iscomplexobj(cm) else cm
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(s1 + s2 - 2 * cm))


def mmd_rbf(A, B, gamma=None):
    from sklearn.metrics.pairwise import rbf_kernel

    g = gamma or 1.0 / max(A.shape[1], 1)
    return float(
        rbf_kernel(A, A, g).mean()
        + rbf_kernel(B, B, g).mean()
        - 2 * rbf_kernel(A, B, g).mean()
    )


def wass_intensity(a, b):
    from scipy.stats import wasserstein_distance

    return float(wasserstein_distance(a, b))


def ks_stat(a, b):
    from scipy.stats import ks_2samp

    return float(ks_2samp(a, b).statistic)  # 0 = identical distributions


def _radial_power(img):
    F = np.abs(np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))) ** 2
    c = np.array(img.shape) // 2
    yy, xx = np.indices(img.shape)
    r = np.sqrt((yy - c[0]) ** 2 + (xx - c[1]) ** 2).astype(int)
    return np.log(
        np.bincount(r.ravel(), F.ravel()) / np.maximum(np.bincount(r.ravel()), 1) + 1e-9
    )


def spectral_dist(a_img, b_img):
    pa, pb = _radial_power(a_img), _radial_power(b_img)
    n = min(len(pa), len(pb))
    return float(np.abs(pa[:n] - pb[:n]).mean())  # power-spectrum distance (texture)


def ssim_mip(a_vol, b_vol):
    try:
        from skimage.metrics import structural_similarity as ssim

        return float(ssim(a_vol.max(0), b_vol.max(0), data_range=1.0))  # 1 = identical
    except Exception:
        return float("nan")


def grad_wass(a_vol, b_vol):
    ga = np.sqrt(sum(g**2 for g in np.gradient(a_vol)))
    gb = np.sqrt(sum(g**2 for g in np.gradient(b_vol)))
    return wass_intensity(ga.ravel()[:200000], gb.ravel()[:200000])  # edge texture


def embed_set_3d(patches, n=300):
    """Set of 3d embeddings (shape/structure) of cell patches."""
    return np.stack([embed3d(p) for p in patches[:n]]) if patches else np.zeros((0, 48))


def sim3d_analysis(real_patches, synth_patches):
    """Similarity in 3D VECTOR SPACE: Frechet between distributions + % of synth cells >=70%
    similar to some real cell + median similarity."""
    A = embed_set_3d(real_patches)
    B = embed_set_3d(synth_patches)
    if len(A) < 3 or len(B) < 3:
        return {}, np.array([])
    sims = np.array(
        [float(np.max(A @ B[i])) for i in range(len(B))]
    )  # each synth vs the most similar real one
    return (
        dict(
            frechet3d=frechet_distance(A, B),
            mmd3d=mmd_rbf(A, B),
            sim3d_median=float(np.median(sims)),
            sim3d_ge70=float((sims >= 0.70).mean()),
            sim3d_min=float(sims.min()),
        ),
        sims,
    )


def blur_sweep(real_frames, bg, stats, rng, sigmas=(0.0, 0.25, 0.4, 0.55, 0.7, 0.85)):
    """Tests blurs (PSF) and measures TEXTURE similarity (spectral + gradient) -> picks the best one."""
    global BLUR_SIGMA
    rmip = real_frames[0].max(0)
    rows = []
    for s in sigmas:
        BLUR_SIGMA = s
        vols = [
            gen_volume_param(bg["shape"], bg, stats, N_EST_PER_FRAME, rng)[0]
            for _ in range(3)
        ]
        sp = float(
            np.mean([spectral_dist(rmip, v.max(0)) for v in vols])
        )  # texture (freq)
        gr = float(np.mean([grad_wass(real_frames[0], v) for v in vols]))  # edges
        rows.append((float(s), sp, gr, sp + gr))  # combined score
    best = min(rows, key=lambda r: r[3])
    BLUR_SIGMA = best[0]
    print(
        f"[blur sweep] sigma ideal = {best[0]:.2f} (espectral {best[1]:.3f} + grad {best[2]:.3f})"
    )
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(
        [r[0] for r in rows], [r[1] for r in rows], "-o", label="espectral (textura)"
    )
    ax.plot(
        [r[0] for r in rows], [r[2] for r in rows], "-o", label="gradiente (bordas)"
    )
    ax.plot([r[0] for r in rows], [r[3] for r in rows], "-o", lw=2, label="combinado")
    ax.axvline(best[0], ls="--", c="k", lw=1, label=f"ideal sigma={best[0]:.2f}")
    ax.set_title("Sweep de desfoque (PSF): similaridade vs sigma")
    ax.set_xlabel("sigma gaussiano")
    ax.legend()
    plt.savefig(FIG / "blur_sweep.png", dpi=180, bbox_inches="tight")
    plt.show()
    return rows, best[0]


def gallery_stages(stats, rng):
    """Gallery: parametric cell by STAGE (early/dev) x SHAPE (round/elliptical) + 3d similarity."""
    combos = [
        ("early", "round"),
        ("early", "ellip"),
        ("dev", "round"),
        ("dev", "ellip"),
    ]
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    for j, (st, sk) in enumerate(combos):
        cell = gen_param_cell(rng, stats, stage=st, shape_kind=sk)
        cc = cell.shape[0] // 2
        s = cell_sim(cell, stats.get("embs"))
        ax[0, j].imshow(cell.max(0), cmap="magma")
        ax[0, j].set_title(f"{st} / {sk}\nsim3d={s:.2f}")
        ax[0, j].axis("off")
        ax[1, j].imshow(cell[cc], cmap="magma")
        ax[1, j].set_title("corte central")
        ax[1, j].axis("off")
    plt.suptitle(
        "Celulas parametricas: ESTAGIO (iniciando/desenvolvida) x FORMA (round/ELIPTICA)",
        weight="bold",
    )
    plt.savefig(FIG / "gallery_stages.png", dpi=160, bbox_inches="tight")
    plt.show()


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


def geff_for(td_, name):
    for c in [td_ / f"{name}.geff", td_ / "gt" / f"{name}.geff"]:
        if c.exists():
            return c
    h = list(td_.rglob(f"{name}*.geff"))
    return h[0] if h else None


def measure_division_rate(train_dir, videos):
    """CELL GENERATION RATE (divisions): fraction of nodes that divide (2 children) in the GT lineage.
    Feeds the 0.1*division_jaccard term of the metric. Robust to variations in the tracksdata API.
    """
    import tracksdata as td
    from collections import Counter

    div, tot = 0, 0
    for name in videos:
        gp = geff_for(train_dir, name)
        if gp is None:
            continue
        try:
            g = td.graph.IndexedRXGraph.from_geff(gp)
            g = g[0] if isinstance(g, tuple) else g
            srcs = []
            try:
                ea = g.edge_attrs(attr_keys=[])
                for r in ea.iter_rows(named=True):
                    srcs.append(
                        r.get("source_id", r.get("source", r.get("src", r.get("u"))))
                    )
            except Exception:
                for e in g.graph.edge_list():
                    srcs.append(e[0])  # rustworkx fallback
            oc = Counter([s for s in srcs if s is not None])
            div += sum(1 for v in oc.values() if v >= 2)
            tot += (
                g.num_nodes()
                if callable(getattr(g, "num_nodes", None))
                else len(g.node_attrs(attr_keys=["t"]).to_numpy())
            )
        except Exception as e:
            print(f"  [div rate] {name}: {e}")
    rate = div / max(tot, 1)
    print(f"=== TAXA DE GERACAO (divisoes reais no lineage GT) ===")
    print(
        f"  {div} divisoes em {tot} nos -> taxa = {rate*100:.2f}% das celulas se dividem"
    )
    return float(rate)


def measure_motion(train_dir, videos, gate_um=7.0):
    """REAL MOTION: speed (um/frame) + directional PERSISTENCE (cos between consecutive velocities
    of the same cell). Frame-by-frame NN matching of the GT (does not need the edges).
    """
    speeds, coss = [], []
    for name in videos:
        gp = geff_for(train_dir, name)
        if gp is None:
            continue
        try:
            nodes = read_geff_nodes(gp)
        except Exception:
            continue
        ts = sorted(nodes)
        prev_vel = {}
        for a, b in zip(ts, ts[1:]):
            if b != a + 1:
                continue
            Pa = np.atleast_2d(nodes[a]).astype(np.float32) * EFF_VOXEL
            Pb = np.atleast_2d(nodes[b]).astype(np.float32) * EFF_VOXEL
            if not len(Pa) or not len(Pb):
                continue
            cur_vel = {}
            for ia, pa in enumerate(Pa):
                d = np.linalg.norm(Pb - pa, axis=1)
                j = int(d.argmin())
                if d[j] <= gate_um:
                    v = Pb[j] - pa
                    sp = float(np.linalg.norm(v))
                    speeds.append(sp)
                    if sp > 0.3:
                        cur_vel[ia] = v / sp
                    if ia in prev_vel and sp > 0.3:  # directional persistence
                        coss.append(float(np.dot(prev_vel[ia], v / sp)))
            prev_vel = {
                j: cur_vel[ia]
                for ia, (pa) in enumerate(Pa)
                for j in [int(np.linalg.norm(Pb - pa, axis=1).argmin())]
                if ia in cur_vel
            }
    speeds = np.array(speeds) if speeds else np.array([2.3])
    coss = np.array(coss) if coss else np.array([0.4])
    m = dict(
        speed_med=float(np.median(speeds)),
        speed_p90=float(np.percentile(speeds, 90)),
        persist=float(np.clip(np.mean(coss), 0.0, 0.95)),
    )
    print(f"=== MOVIMENTACAO REAL (GT) ===")
    print(
        f"  velocidade: mediana {m['speed_med']:.2f} um/frame (p90 {m['speed_p90']:.2f}) | persistencia direcional {m['persist']:.2f} (1=reto, 0=aleatorio)"
    )
    return m


def collect_real(
    train_dir, videos, frames_per=int(os.environ.get("SYNTH_FRAMESPER", "8"))
):
    """Collects templates (GT), centroids and frames for the liquid. LARGER frames_per = richer real reference."""
    templates, frames, cens_list, gt_boxes = [], [], [], []
    for name in videos:
        zp = train_dir / f"{name}.zarr"
        gp = geff_for(train_dir, name)
        if gp is None:
            continue
        try:
            nodes = read_geff_nodes(gp)
        except Exception:
            continue
        ts = sorted(nodes)[:: max(1, len(nodes) // frames_per)][:frames_per]
        for t in ts:
            fr = norm(read_frame(zp, t)[0])
            frp = pool_xy(fr)
            cens = nodes[t] / np.array(DOWNSAMPLE)  # centroids in pooled space
            templates += build_template_bank(frp, cens)
            frames.append(frp)
            cens_list.append(cens)
    return templates, frames, cens_list


def save_sample(split, ver, idx, vol, boxes):
    d = DSDIR / ver / split
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / f"{idx:04d}.npz",
        volume=vol.astype(np.float32),
        boxes=boxes.astype(np.int32),
        labels=np.zeros(len(boxes), np.int64),
    )  # class 0 = cell (DETR format)


def gen_realbased_variation(frames, cens_list, bg, stats, rng):
    """REAL-BASED DATA: takes a REAL CAPTURE (pooled frame) + its real detections and AGGLOMERATES into variations
    (90 rotation, flips, gamma/intensity jitter, mixing 2 real regions). Motion/capture = real.
    """
    i = rng.integers(len(frames))
    vol = np.asarray(frames[i], np.float32).copy()
    cen = np.round(np.atleast_2d(cens_list[i])).astype(np.float32)
    # AGGLOMERATE: mixes (maximum) with another real region -> more real cells per image (several images in one)
    if rng.random() < 0.6 and len(frames) > 1:
        j = rng.integers(len(frames))
        vol = np.maximum(vol, np.asarray(frames[j], np.float32))
        cen = (
            np.vstack([cen, np.round(np.atleast_2d(cens_list[j])).astype(np.float32)])
            if len(cens_list[j])
            else cen
        )
    Z, Y, X = vol.shape
    # 90 rotation in XY (multiple) + flips: transforms the coords along with it (real capture, just reoriented)
    k = int(rng.integers(4))
    vol = np.rot90(vol, k, axes=(1, 2))
    if k:
        for _ in range(k):
            cen[:, [1, 2]] = np.stack([cen[:, 2], (Y - 1) - cen[:, 1]], 1)
    if rng.random() < 0.5:
        vol = vol[:, ::-1]
        cen[:, 1] = (Y - 1) - cen[:, 1]
    if rng.random() < 0.5:
        vol = vol[:, :, ::-1]
        cen[:, 2] = (X - 1) - cen[:, 2]
    vol = np.clip(
        vol ** float(rng.uniform(0.85, 1.15)) * float(rng.uniform(0.9, 1.1)), 0, 1
    )  # gamma/intensity jitter
    vol = np.ascontiguousarray(vol, np.float32)
    R = CELL_R_VOX
    sh = np.array(vol.shape) - 1
    cen = cen[
        (cen[:, 0] >= R)
        & (cen[:, 1] >= R)
        & (cen[:, 2] >= R)
        & (cen[:, 0] < sh[0] - R)
        & (cen[:, 1] < sh[1] - R)
        & (cen[:, 2] < sh[2] - R)
    ]
    boxes = np.array(
        [
            [c[0] - R, c[1] - R, c[2] - R, c[0] + R + 1, c[1] + R + 1, c[2] + R + 1]
            for c in cen
        ],
        np.int32,
    )
    return vol, boxes


def run_scale(train_dir, tr, rng):
    """SCALE: many UNIQUE volumes without figures (throughput). SYNTH_REALBASED=1 uses real capture+aug.
    SYNTH_CHUNK varies the seed (chunks != data). Stops at SYNTH_MAXGB (Kaggle output limit).
    """
    import time

    realbased = os.environ.get("SYNTH_REALBASED", "0") == "1"
    n = int(os.environ.get("SYNTH_NSCALE", "13000"))
    chunk = int(os.environ.get("SYNTH_CHUNK", "0"))
    maxgb = float(os.environ.get("SYNTH_MAXGB", "15.0"))
    NV = int(os.environ.get("SYNTH_REFVIDEOS", str(len(tr))))
    templates, frames, cens = collect_real(train_dir, tr[:NV])
    bg = liquid_model(frames, cens)
    stats = measure_cell_stats(templates)
    shape = frames[0].shape
    global _CELL_MU_RT, _CELL_SIG_RT, _CELL_MEANS_RT, _TEMPLATES_RT
    _mn = np.asarray(stats["means"], float)
    _CELL_MU_RT = float(np.clip(np.median(_mn), 0.1, 0.9))
    _CELL_SIG_RT = float(np.clip(np.std(_mn), 0.03, 0.20))
    _CELL_MEANS_RT = _mn
    _TEMPLATES_RT = templates  # real bank for HYBRID mode (SYNTH_CELL_SOURCE=template)
    if os.environ.get("SYNTH_CELL_SOURCE", "param") == "template":
        print(
            f"  MODO HIBRIDO: celulas de template REAL ({len(templates)} no banco) + placement param",
            flush=True,
        )
    global DIV_RATE
    if DIV_RATE <= 0:
        DIV_RATE = measure_division_rate(train_dir, tr[: min(NV, len(tr))])
    d = DSDIR / ("R_realbased" if realbased else "D_scale")
    d.mkdir(parents=True, exist_ok=True)
    rs = np.random.default_rng(10007 * (chunk + 1))
    print(
        f"=== ESCALA {'REAL-BASED' if realbased else 'D_param unico'} chunk={chunk} alvo={n} vols max={maxgb}GB "
        f"| intensidade real media={_CELL_MU_RT:.3f} ===",
        flush=True,
    )
    t0 = time.time()
    for i in range(n):
        if realbased:
            vol, boxes = gen_realbased_variation(frames, cens, bg, stats, rs)
        else:
            vol, boxes, _ = gen_volume_param(shape, bg, stats, N_EST_PER_FRAME, rs)
        if len(boxes) == 0:
            continue
        np.savez(
            d / f"{chunk:02d}_{i:05d}.npz",
            volume=vol.astype(np.float32),
            boxes=np.asarray(boxes, np.int32),
            labels=np.zeros(len(boxes), np.int64),
        )
        if i % 500 == 0 and i > 0:
            gb = sum(f.stat().st_size for f in d.glob("*.npz")) / 1e9
            print(
                f"  {i}/{n} | {gb:.1f}GB | {(time.time()-t0)/i:.2f}s/vol | proj {(time.time()-t0)/i*n/3600:.1f}h",
                flush=True,
            )
            if gb >= maxgb:
                print(
                    f"  ATINGIU {gb:.1f}GB (limite) -> parando em {i} vols", flush=True
                )
                break
    gb = sum(f.stat().st_size for f in d.glob("*.npz")) / 1e9
    print(
        f"=== ESCALA CONCLUIDA: {len(list(d.glob('*.npz')))} vols, {gb:.2f}GB em {d} ===",
        flush=True,
    )


def run_split(train_dir, split, videos, rng, n_samples):
    templates, frames, cens = collect_real(train_dir, videos)
    if len(templates) < 5:
        print(f"  [{split}] templates insuficientes ({len(templates)})")
        return None
    bg = liquid_model(frames, cens)
    shape = frames[0].shape
    stats = measure_cell_stats(templates)
    global _CELL_MU_RT, _CELL_SIG_RT, _CELL_MEANS_RT, _TEMPLATES_RT
    _mn = np.asarray(stats["means"], float)
    _CELL_MU_RT = float(np.clip(np.median(_mn), 0.1, 0.9))
    _CELL_SIG_RT = float(np.clip(np.std(_mn), 0.03, 0.20))
    _CELL_MEANS_RT = _mn
    _TEMPLATES_RT = templates  # real bank for HYBRID mode (D_param becomes hybrid if SYNTH_CELL_SOURCE=template)
    _hyb = os.environ.get("SYNTH_CELL_SOURCE", "param") == "template"
    print(
        f"  [{split}] {len(templates)} templates, {len(frames)} frames | liquido mean={bg['mean']:.3f}{' | D_param=HIBRIDO (template real)' if _hyb else ''}"
    )
    # VERSION A: synthetic from pasted real TEMPLATES
    for i in range(n_samples):
        vol, boxes, _ = gen_volume(shape, bg, templates, N_EST_PER_FRAME, rng)
        save_sample(split, "A_synth", i, vol, boxes)
    # VERSION D: parametric OR hybrid (real template cell) depending on SYNTH_CELL_SOURCE
    for i in range(n_samples):
        vol, boxes, _ = gen_volume_param(shape, bg, stats, N_EST_PER_FRAME, rng)
        save_sample(split, "D_param", i, vol, boxes)
    # VERSION E: LAYOUT TRANSFER (Fable) - synthetic appearance at the REAL POSITIONS (zero placement error)
    for i in range(n_samples):
        vol, boxes, _ = gen_volume_layout(shape, bg, stats, rng, real_bg=True)
        if len(boxes):
            save_sample(split, "E_layout", i, vol, boxes)
    # VERSION B: labelled real only (real frame + GT boxes)
    for i, (fr, cs) in enumerate(zip(frames, cens)):
        boxes = np.array(
            [
                [
                    int(z) - CELL_R_VOX,
                    int(y) - CELL_R_VOX,
                    int(x) - CELL_R_VOX,
                    int(z) + CELL_R_VOX + 1,
                    int(y) + CELL_R_VOX + 1,
                    int(x) + CELL_R_VOX + 1,
                ]
                for z, y, x in cs
            ],
            np.int32,
        )
        save_sample(split, "B_real", i, fr, boxes)
    # VERSION C: real + DoG-precision-100% (GT + reliable DoG)
    for i, (fr, cs) in enumerate(zip(frames, cens)):
        det, _ = dog_detect_precise(fr)
        allc = np.vstack([cs, det]) if len(det) else cs
        boxes = np.array(
            [
                [
                    int(z) - CELL_R_VOX,
                    int(y) - CELL_R_VOX,
                    int(x) - CELL_R_VOX,
                    int(z) + CELL_R_VOX + 1,
                    int(y) + CELL_R_VOX + 1,
                    int(x) + CELL_R_VOX + 1,
                ]
                for z, y, x in allc
            ],
            np.int32,
        )
        save_sample(split, "C_real_dog", i, fr, boxes)
    return dict(templates=templates, frames=frames, cens=cens, bg=bg, shape=shape)


def validate(real_frames, real_cens, synth_dir, split="train", n=40, version="A_synth"):
    """Real vs synthetic similarity (Frechet/MMD/Wasserstein/KS/spectral/gradient)."""
    real_patches = []
    for fr, cs in zip(real_frames, real_cens):
        real_patches += build_template_bank(fr, cs)
    synth_patches, synth_vols, synth_cens, synth_boxes = [], [], [], []
    for f in sorted(glob.glob(str(DSDIR / version / split / "*.npz")))[:n]:
        d = np.load(f)
        synth_vols.append(d["volume"])
        cen = (d["boxes"][:, :3] + d["boxes"][:, 3:]) / 2
        synth_patches += build_template_bank(d["volume"], cen)
        synth_cens.append(cen)
        synth_boxes.append(d["boxes"])
    A, B = embed_set(real_patches[:400]), embed_set(synth_patches[:400])
    if len(A) < 5 or len(B) < 5:
        return {}
    # ANALYSIS IN 3D VECTOR SPACE: Frechet + % of synth cells >=70% similar to a real one
    sim3d, sims_arr = sim3d_analysis(real_patches[:400], synth_patches[:400])
    # figure: 3d similarity distribution (each synth cell vs the most similar real one)
    if len(sims_arr):
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        ax[0].hist(sims_arr, bins=30, color="teal", alpha=0.8)
        ax[0].axvline(
            0.70,
            ls="--",
            c="r",
            lw=1.5,
            label=f">=70%: {(sims_arr>=0.70).mean()*100:.0f}%",
        )
        ax[0].set_title(f"Similaridade 3D de cada celula synth a uma REAL ({version})")
        ax[0].set_xlabel("cosseno")
        ax[0].legend()
        # MEAN BRIGHTNESS PER CELL: measures the ACTIVE REGION (>0.3 of the peak) to match the Normal(0.30) target
        rm = np.array(
            [
                float(p[p > 0.3 * p.max()].mean())
                for p in real_patches[:300]
                if p.max() > 0
            ]
        )
        sm = np.array(
            [
                float(p[p > 0.3 * p.max()].mean())
                for p in synth_patches[:300]
                if p.max() > 0
            ]
        )
        try:
            from scipy.stats import shapiro

            sp = float(shapiro(sm[:200]).pvalue) if len(sm) >= 8 else 0.0
        except Exception:
            sp = -1.0
        ax[1].hist(
            rm,
            bins=25,
            alpha=0.5,
            density=True,
            label=f"real (media {rm.mean():.2f})",
            color="steelblue",
        )
        ax[1].hist(
            sm,
            bins=25,
            alpha=0.5,
            density=True,
            label=f"synth (media {sm.mean():.2f})",
            color="darkorange",
        )
        ax[1].axvline(
            CELL_MEAN_TARGET,
            ls="--",
            c="red",
            lw=2,
            label=f"ALVO {CELL_MEAN_TARGET:.2f}",
        )
        xs = np.linspace(0, max(rm.max(), sm.max(), 0.6), 100)  # fitted NORMAL curves
        ax[1].plot(
            xs,
            np.exp(-0.5 * ((xs - rm.mean()) / (rm.std() + 1e-6)) ** 2)
            / ((rm.std() + 1e-6) * np.sqrt(2 * np.pi)),
            "b-",
            lw=2,
        )
        ax[1].plot(
            xs,
            np.exp(-0.5 * ((xs - sm.mean()) / (sm.std() + 1e-6)) ** 2)
            / ((sm.std() + 1e-6) * np.sqrt(2 * np.pi)),
            "-",
            c="darkorange",
            lw=2,
        )
        ax[1].set_title(
            f"Brilho/celula: synth media {sm.mean():.3f} std {sm.std():.3f} | Shapiro-normal p={sp:.3f}"
        )
        ax[1].legend(fontsize=8)
        plt.suptitle(
            f"Espaco vetorial 3D + brilho da celula - {version}", weight="bold"
        )
        plt.savefig(FIG / f"sim3d_{version}.png", dpi=180, bbox_inches="tight")
        plt.show()
    # samples ALL frames representatively (do not truncate at the 1st frame -> correct real mean)
    rcat = np.concatenate([f.ravel()[:: max(1, f.size // 12000)] for f in real_frames])
    scat = np.concatenate([v.ravel()[:: max(1, v.size // 8000)] for v in synth_vols])
    metrics = dict(
        frechet_fid3d=frechet_distance(A, B),
        mmd=mmd_rbf(A, B),
        wasserstein_intensity=wass_intensity(rcat, scat),
        # + METRICS: KS, SSIM, spectral (texture), gradient (edges)
        ks_intensity=ks_stat(rcat, scat),
        spectral_dist=spectral_dist(real_frames[0].max(0), synth_vols[0].max(0)),
        grad_wasserstein=grad_wass(real_frames[0], synth_vols[0]),
        # STRUCTURAL: spatial organization that Frechet ignores
        bright_frac_real=float((rcat > 0.4).mean()),
        bright_frac_synth=float((scat > 0.4).mean()),
        mean_int_real=float(rcat.mean()),
        mean_int_synth=float(scat.mean()),
        n_real_patches=len(real_patches),
        n_synth_patches=len(synth_patches),
    )
    metrics.update(sim3d)
    metrics["blur_sigma"] = float(BLUR_SIGMA)
    # ===== SPATIAL CLUSTERING: how the cells are distributed/agglomerated =====
    real_det = [
        dog_detect_precise(fr)[0] for fr in real_frames[:8]
    ]  # dense real positions (DoG)
    rnn = (
        np.concatenate([nn_um(d) for d in real_det if len(d) > 1])
        if real_det
        else np.array([])
    )
    snn = (
        np.concatenate([nn_um(c) for c in synth_cens if len(c) > 1])
        if synth_cens
        else np.array([])
    )
    rclump = (
        np.mean([clump_index(d, real_frames[0].shape) for d in real_det if len(d) > 2])
        if real_det
        else 1.0
    )
    sclump = (
        np.mean([clump_index(c, synth_vols[0].shape) for c in synth_cens if len(c) > 2])
        if synth_cens
        else 1.0
    )
    # OVERLAP AUDIT (user's alert): fraction of boxes with IoU>0.3
    ov = [overlap_audit(b) for b in synth_boxes if len(b) > 1]
    R = CELL_R_VOX  # REAL overlap: the same ±R boxes at the real detections
    ovr = [
        overlap_audit(
            np.array(
                [
                    [z - R, y - R, x - R, z + R + 1, y + R + 1, x + R + 1]
                    for z, y, x in d
                ]
            )
        )
        for d in real_det
        if len(d) > 1
    ]
    metrics.update(
        overlap_frac_real=float(np.mean([o[0] for o in ovr])) if ovr else 0.0,
        overlap_maxiou_real=float(np.max([o[1] for o in ovr])) if ovr else 0.0,
        nn_real_med_um=float(np.median(rnn)) if len(rnn) else 0.0,
        nn_synth_med_um=float(np.median(snn)) if len(snn) else 0.0,
        nn_real_p5_um=float(np.percentile(rnn, 5)) if len(rnn) else 0.0,
        nn_synth_p5_um=float(np.percentile(snn, 5)) if len(snn) else 0.0,
        clump_real=float(rclump),
        clump_synth=float(sclump),
        overlap_frac_synth=float(np.mean([o[0] for o in ov])) if ov else 0.0,
        overlap_maxiou_synth=float(np.max([o[1] for o in ov])) if ov else 0.0,
    )
    # clustering figure: NN (nearest neighbour) + agglomeration
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    if len(rnn):
        ax[0].hist(
            rnn,
            bins=40,
            alpha=0.6,
            density=True,
            label=f"real (med {np.median(rnn):.1f}um)",
        )
    if len(snn):
        ax[0].hist(
            snn,
            bins=40,
            alpha=0.6,
            density=True,
            label=f"sintetico (med {np.median(snn):.1f}um)",
        )
    ax[0].axvline(
        CELL_DIAM_UM, ls="--", c="k", lw=1, label=f"1 diametro ({CELL_DIAM_UM:.1f}um)"
    )
    ax[0].set_title("Distancia ao vizinho + proximo (agrupamento)")
    ax[0].set_xlabel("um")
    ax[0].legend()
    ax[1].bar([0, 1], [rclump, sclump], color=["steelblue", "orange"])
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["real", "sintetico"])
    ax[1].set_title("Indice de aglomeracao (>1 = agregado)")
    plt.suptitle(f"Agrupamento espacial - {version}", weight="bold")
    plt.savefig(FIG / f"agrupamento_{version}.png", dpi=190, bbox_inches="tight")
    plt.show()
    # ===== NEW POINT OF VIEW: connectivity (mass vs web) + Z depth profile =====
    cfr = float(np.mean([connected_frac(f) for f in real_frames[:8]]))
    cfs = float(np.mean([connected_frac(v) for v in synth_vols[:8]]))
    zr = np.mean(
        [f.mean(axis=(1, 2)) for f in real_frames], axis=0
    )  # ALL frames (same population as the synth)
    zs = np.mean([v.mean(axis=(1, 2)) for v in synth_vols], axis=0)
    metrics.update(connected_frac_real=cfr, connected_frac_synth=cfs)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].bar([0, 1], [cfr, cfs], color=["steelblue", "orange"])
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels([f"real\n{cfr:.2f}", f"sintetico\n{cfs:.2f}"])
    ax[0].set_title("Conectividade: fracao no maior componente (1=massa)")
    ax[0].set_ylim(0, 1)
    ax[1].plot(zr, label="real", lw=2)
    ax[1].plot(zs, label="sintetico", lw=2)
    ax[1].set_title("Perfil de intensidade por profundidade Z")
    ax[1].set_xlabel("slice Z")
    ax[1].legend()
    plt.suptitle(f"Conectividade + profundidade - {version}", weight="bold")
    plt.savefig(FIG / f"conectividade_{version}.png", dpi=190, bbox_inches="tight")
    plt.show()
    # figure: real vs synthetic, multi-angle (MIP z/y/x)
    rv = real_frames[0]
    sv = synth_vols[0]
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    for row, (V, tag) in enumerate([(rv, "REAL"), (sv, "SINTETICO")]):
        for col, (proj, pn) in enumerate(
            [(V.max(0), "MIP-Z"), (V.max(1), "MIP-Y"), (V.max(2), "MIP-X")]
        ):
            ax[row, col].imshow(proj, cmap="magma", aspect="auto")
            ax[row, col].set_title(f"{tag} {pn}")
            ax[row, col].axis("off")
    plt.suptitle("Real vs Sintetico (multi-angulo)", fontsize=15, weight="bold")
    plt.savefig(FIG / "real_vs_synth_mip.png", dpi=200, bbox_inches="tight")
    plt.show()
    # intensity + cell size histograms
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].hist(
        rcat, bins=60, alpha=0.6, density=True, label="real"
    )  # sample of ALL frames (representative)
    ax[0].hist(scat, bins=60, alpha=0.6, density=True, label="sintetico")
    ax[0].set_title("Intensidade (todos os frames)")
    ax[0].legend()
    ax[0].set_yscale("log")
    ax[1].hist(
        [p.mean() for p in real_patches[:300]],
        bins=30,
        alpha=0.6,
        density=True,
        label="real",
    )
    ax[1].hist(
        [p.mean() for p in synth_patches[:300]],
        bins=30,
        alpha=0.6,
        density=True,
        label="sintetico",
    )
    ax[1].set_title("Intensidade media da celula")
    ax[1].legend()
    plt.savefig(FIG / "dist_real_vs_synth.png", dpi=200, bbox_inches="tight")
    plt.show()
    return metrics


def visualize_rich(real_ref, split="train", version="A_synth"):
    """MANY visualizations: individual cell, detected data (boxes), multi-angle."""
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    vtag = "_" + version
    sf = sorted(glob.glob(str(DSDIR / version / split / "*.npz")))[:8]
    synth = [np.load(f) for f in sf]
    rframes, rcens = real_ref["frames"], real_ref["cens"]

    # --- VIZ 1: gallery of individual CELLS (real vs synthetic), central slice + MIP ---
    rp = []
    for fr, cs in zip(rframes, rcens):
        rp += build_template_bank(fr, cs, refine=False)
    spatch = []
    for d in synth:
        ctr = (d["boxes"][:, :3] + d["boxes"][:, 3:]) / 2
        spatch += build_template_bank(d["volume"], ctr, refine=False)
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(4, 10, figsize=(22, 9))
    for col in range(10):
        rc = rp[rng.integers(len(rp))]
        sc = spatch[rng.integers(len(spatch))]
        ax[0, col].imshow(rc[rc.shape[0] // 2], cmap="magma")
        ax[0, col].axis("off")
        ax[1, col].imshow(rc.max(0), cmap="magma")
        ax[1, col].axis("off")
        ax[2, col].imshow(sc[sc.shape[0] // 2], cmap="magma")
        ax[2, col].axis("off")
        ax[3, col].imshow(sc.max(0), cmap="magma")
        ax[3, col].axis("off")
    for r, lab in [
        (0, "REAL corte"),
        (1, "REAL MIP"),
        (2, "SINT corte"),
        (3, "SINT MIP"),
    ]:
        ax[r, 0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=11)
    plt.suptitle(
        f"Galeria de celulas [{version}]: REAL vs SINTETICO (corte central + MIP)",
        fontsize=15,
        weight="bold",
    )
    plt.savefig(FIG / f"viz1_galeria_celulas{vtag}.png", dpi=180, bbox_inches="tight")
    plt.show()

    # --- VIZ 2: DETECTED DATA - boxes over the volume (what the 3D-DETR sees) ---
    fig, ax = plt.subplots(2, 4, figsize=(22, 11))
    ax = ax.ravel()
    for k in range(min(8, len(synth))):
        d = synth[k]
        vol = d["volume"]
        boxes = d["boxes"]
        ax[k].imshow(vol.max(0), cmap="gray")
        for b in boxes:
            ax[k].add_patch(
                mpatches.Rectangle(
                    (b[2], b[1]),
                    b[5] - b[2],
                    b[4] - b[1],
                    fill=False,
                    edgecolor="lime",
                    lw=0.6,
                )
            )
        ax[k].set_title(f"sintetico #{k}: {len(boxes)} celulas rotuladas")
        ax[k].axis("off")
    plt.suptitle(
        "Dados DETECTAVEIS: caixas 3D rotuladas (MIP-Z) - entrada do 3D-DETR",
        fontsize=15,
        weight="bold",
    )
    plt.savefig(FIG / f"viz2_deteccoes_caixas{vtag}.png", dpi=200, bbox_inches="tight")
    plt.show()

    # --- VIZ 3: MULTI-ANGLE (real and synthetic): MIP z/y/x + 3 orthogonal slices ---
    for tag, V, C in [
        ("REAL", rframes[0], rcens[0]),
        (
            "SINTETICO",
            synth[0]["volume"],
            (synth[0]["boxes"][:, :3] + synth[0]["boxes"][:, 3:]) / 2,
        ),
    ]:
        fig, ax = plt.subplots(2, 3, figsize=(17, 11))
        for c, (proj, pn) in enumerate(
            [(V.max(0), "MIP-Z"), (V.max(1), "MIP-Y"), (V.max(2), "MIP-X")]
        ):
            ax[0, c].imshow(proj, cmap="magma", aspect="auto")
            ax[0, c].set_title(f"{tag} {pn}")
            ax[0, c].axis("off")
        zc, yc, xc = np.array(V.shape) // 2
        for c, (sl, pn) in enumerate(
            [
                (V[zc], f"corte Z={zc}"),
                (V[:, yc], f"corte Y={yc}"),
                (V[:, :, xc], f"corte X={xc}"),
            ]
        ):
            ax[1, c].imshow(sl, cmap="magma", aspect="auto")
            ax[1, c].set_title(pn)
            ax[1, c].axis("off")
        plt.suptitle(
            f"{tag} - multi-angulo (projecoes + cortes ortogonais)",
            fontsize=15,
            weight="bold",
        )
        plt.savefig(
            FIG / f"viz3_multiangulo_{tag}{vtag}.png", dpi=190, bbox_inches="tight"
        )
        plt.show()

    # --- VIZ 4: 3D cloud of the detectable centroids (several camera angles) ---
    d = synth[0]
    ctr = (d["boxes"][:, :3] + d["boxes"][:, 3:]) / 2
    fig = plt.figure(figsize=(20, 6))
    for i, (az, el) in enumerate([(30, 20), (120, 30), (210, 15), (300, 45)]):
        a = fig.add_subplot(1, 4, i + 1, projection="3d")
        a.scatter(
            ctr[:, 2] * EFF_VOXEL[2],
            ctr[:, 1] * EFF_VOXEL[1],
            ctr[:, 0] * EFF_VOXEL[0],
            s=14,
            c=ctr[:, 0],
            cmap="viridis",
        )
        a.view_init(elev=el, azim=az)
        a.set_title(f"azim={az} elev={el}")
        a.set_xlabel("X(um)")
        a.set_ylabel("Y(um)")
    plt.suptitle(
        "Centroides 3D detectaveis (varios angulos de camera)",
        fontsize=15,
        weight="bold",
    )
    plt.savefig(FIG / f"viz4_3d_angulos{vtag}.png", dpi=170, bbox_inches="tight")
    plt.show()

    # --- VIZ 5: sweep of Z slices with boxes (detectable data by depth) ---
    d = synth[1] if len(synth) > 1 else synth[0]
    vol = d["volume"]
    boxes = d["boxes"]
    zs = np.linspace(vol.shape[0] * 0.2, vol.shape[0] * 0.8, 6).astype(int)
    fig, ax = plt.subplots(1, 6, figsize=(24, 4.2))
    for c, z in enumerate(zs):
        ax[c].imshow(vol[z], cmap="gray")
        ax[c].set_title(f"Z={z}")
        ax[c].axis("off")
        for b in boxes:
            if b[0] <= z < b[3]:
                ax[c].add_patch(
                    mpatches.Rectangle(
                        (b[2], b[1]),
                        b[5] - b[2],
                        b[4] - b[1],
                        fill=False,
                        edgecolor="cyan",
                        lw=0.7,
                    )
                )
    plt.suptitle(
        "Varredura em Z com caixas (celulas detectaveis por profundidade)",
        fontsize=14,
        weight="bold",
    )
    plt.savefig(FIG / f"viz5_cortes_z_caixas{vtag}.png", dpi=190, bbox_inches="tight")
    plt.show()
    print("visualizacoes ricas salvas:", len(list(FIG.glob("viz*.png"))), "figuras")


def sweep_capture(train_dir, videos):
    """STUDY: capture radius (2..5) x spherical/cubic -> which is more similar to the real data.
    Grounded in the EDA (cell diameter ~9.2um -> radius ~2.8 pooled voxels)."""
    global CELL_R_VOX, SPHERICAL
    frames, cens = [], []
    for name in videos:
        zp = train_dir / f"{name}.zarr"
        gp = geff_for(train_dir, name)
        if gp is None:
            continue
        try:
            nodes = read_geff_nodes(gp)
        except Exception:
            continue
        for t in sorted(nodes)[:: max(1, len(nodes) // 3)][:3]:
            frames.append(pool_xy(norm(read_frame(zp, t)[0])))
            cens.append(nodes[t] / np.array(DOWNSAMPLE))
    rcat = np.concatenate([f.ravel() for f in frames])[:200000]
    r_real = float((rcat > 0.4).mean())
    rng = np.random.default_rng(0)
    rows = []
    r0, s0 = CELL_R_VOX, SPHERICAL
    print(f"EDA: raio esperado ~2.8 vox (diam 9.2um). real bright_frac={r_real:.3f}")
    for r in [2, 3, 4, 5]:
        for sph in [True, False]:
            CELL_R_VOX, SPHERICAL = r, sph
            templates = []
            for fr, cs in zip(frames, cens):
                templates += build_template_bank(fr, cs)
            if len(templates) < 5:
                continue
            bg = liquid_model(frames, cens, cell_r=r)
            vols, rp, sp = [], [], []
            for _ in range(10):
                v, _, ctr = gen_volume(
                    frames[0].shape, bg, templates, N_EST_PER_FRAME, rng
                )
                vols.append(v)
                sp += build_template_bank(v, ctr, refine=False)
            for fr, cs in zip(frames, cens):
                rp += build_template_bank(fr, cs, refine=False)
            scat = np.concatenate([v.ravel() for v in vols])[:200000]
            A, B = embed_set(rp[:300]), embed_set(sp[:300])
            fd = frechet_distance(A, B) if len(A) > 4 and len(B) > 4 else 9
            ks = ks_stat(rcat, scat)
            bf = abs(r_real - float((scat > 0.4).mean()))
            spec = spectral_dist(frames[0].max(0), vols[0].max(0))
            score = (
                fd + ks + 0.3 * spec + bf
            )  # rebalanced: Frechet(cell)+KS+spectral, unbiased
            rows.append((r, "esf" if sph else "cubo", fd, ks, bf, spec, score))
    CELL_R_VOX, SPHERICAL = r0, s0
    rows.sort(key=lambda x: x[-1])
    print("\n  raio forma | Frechet   KS    d-bright spectral | SCORE (menor=melhor)")
    for r, sh, fd, ks, bf, sc_, score in rows:
        print(
            f"   {r:>2}  {sh:4s} | {fd:6.4f}  {ks:5.3f}  {bf:6.4f}   {sc_:6.3f}  | {score:.4f}"
        )
    best = rows[0]
    print(
        f"\n  >>> MELHOR CAPTURA: raio={best[0]} forma={best[1]} (score {best[-1]:.4f})"
    )
    return best[0], best[1] == "esf"


def temporal_study(train_dir, videos):
    """How the cells change OVER TIME (start/middle/end of development)."""
    binkeys = ["inicio", "meio", "fim"]
    bins = {k: [] for k in binkeys}
    for name in videos:
        zp = train_dir / f"{name}.zarr"
        gp = geff_for(train_dir, name)
        if gp is None:
            continue
        try:
            nodes = read_geff_nodes(gp)
        except Exception:
            continue
        ts = sorted(nodes)
        T = (ts[-1] + 1) if ts else 100
        for t in ts[:: max(1, len(ts) // 9)][:9]:
            b = 0 if t < T / 3 else (1 if t < 2 * T / 3 else 2)
            fr = pool_xy(norm(read_frame(zp, t)[0]))
            cens = nodes[t] / np.array(DOWNSAMPLE)
            sh = np.array(fr.shape) - 1
            ci = (
                np.mean([fr[tuple(np.round(c).astype(int).clip(0, sh))] for c in cens])
                if len(cens)
                else 0.0
            )
            bins[binkeys[b]].append((len(cens), float((fr > 0.4).mean()), float(ci)))
    print("=== ESTUDO TEMPORAL (celulas ao longo do desenvolvimento) ===")
    print(f"  {'periodo':8s} {'cels/frame':>11s} {'tecido%':>9s} {'intensidade':>12s}")
    rows = []
    for k in binkeys:
        v = np.array(bins[k]) if bins[k] else np.zeros((1, 3))
        print(
            f"  {k:8s} {v[:,0].mean():11.1f} {v[:,1].mean()*100:8.1f}% {v[:,2].mean():12.3f}"
        )
        rows.append(
            (k, float(v[:, 0].mean()), float(v[:, 1].mean()), float(v[:, 2].mean()))
        )
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    x = range(len(rows))
    for a, idx, tit, col in [
        (ax[0], 1, "Celulas/frame (rotuladas)", "steelblue"),
        (ax[1], 2, "Extensao do tecido", "orange"),
        (ax[2], 3, "Intensidade da celula", "green"),
    ]:
        a.bar(x, [r[idx] for r in rows], color=col)
        a.set_xticks(list(x))
        a.set_xticklabels([r[0] for r in rows])
        a.set_title(tit)
    plt.suptitle(
        "Evolucao das celulas ao longo do desenvolvimento (REAL)", weight="bold"
    )
    plt.savefig(FIG / "temporal_estudo.png", dpi=180, bbox_inches="tight")
    plt.show()
    json.dump(rows, open(OUT / "temporal.json", "w"), default=float)
    return rows
