"""What varies between acquisitions, and what has to stay fixed within one.

The real set is not one imaging condition, it is a spread of them. Measured
over 210 frames of 70 videos: the median frame intensity spans 0.04 to 0.40 of
full scale, the octant illumination ratio 2.7 to 23, and a cell's axial extent
4.9 to 26 um. A generator built on one fixed set of optics constants produces a
single point in that space, and the point the old constants produced was the
easy corner of it -- cells rising 400-1400% above a near-black background,
where the real median is 220% and a quarter of annotated cells rise less
than 100%.

Two objects carry that:

``Regime``  one draw from the space of acquisitions: optics, illumination,
            scattering, detector. Drawn once per volume -- or once per
            *sequence*, which matters, see below -- and then held fixed.

``Palette`` per-cell appearance: size, elongation, orientation, brightness.
            Drawn once per *track*, not per frame. The generator used to redraw
            a cell's shape and brightness at every frame, which hands the
            tracker a cell whose appearance between t and t+1 is pure noise:
            appearance matching is then not merely hard to learn, it is
            unlearnable, and the only signal left is position.

Everything drawn here is a distribution fitted to the measurements in
``calibrate.py``, not a guess. Run ``python -m data_gen.calibrate`` to check a
generated set back against the real one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from data_gen.constants import (
    ATTEN_FRAC, BRIGHT_SPAN, CELL_RADIUS_UM, COLLISION_FACTOR, DEBRIS_PER_VOLUME,
    DEFOCUS_UM, DENSITY_FACTOR, ELLIPTICAL_P, FAINT_FRAC, FAINT_MEDIAN, FAINT_MIN,
    HALO_AMP, HAZE_GAMMA, HAZE_UM, HAZE_Z_RATIO, ILLUM_CONTRAST, MEDIUM_LEVEL,
    PEDESTAL,
    POISSON_PE, PSF_XY_UM, PSF_Z_UM, READ_NOISE, TEXTURE_AMP, TEXTURE_UM,
    VOXEL_NATIVE,
)


def _loguniform(rng, lo, hi):
    """Uniform in log space: spans an order of magnitude without favouring its top."""
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _lognormal(rng, median, sigma, lo, hi):
    return float(np.clip(rng.lognormal(np.log(median), sigma), lo, hi))


@dataclass
class Regime:
    """One acquisition: the imaging conditions a whole volume is rendered under.

    Fields are absolute physical quantities (um, fractions of full scale) so
    that a value read here means the same thing as a value measured on a real
    frame.
    """

    # --- Scattered light -------------------------------------------------
    # The single largest gap against the real data. A real cell does not sit on
    # black: it sits on a pedestal of out-of-focus and scattered light from the
    # tissue around it, median 0.25 of full scale against 0.06 for the old
    # generator. That pedestal is what crushes contrast from 400% down to the
    # real 220%, and low contrast is the regime the detector actually fails in.
    pedestal: float = 0.22
    haze_um: float = 14.0          # lateral scattering length
    haze_z_ratio: float = 1.6      # axial spread, larger (the PSF is axially long)
    haze_gamma: float = 2.0        # how sharply the pedestal stops at the void
    halo_amp: float = HALO_AMP     # tight per-cell halo, on top of the wide haze

    # --- Optics ----------------------------------------------------------
    psf_xy_um: float = 0.55
    psf_z_um: float = 2.2          # absolute sigma, not a ratio of psf_xy
    # Extra blur applied to the far side of the volume: deeper light has
    # travelled through more tissue and arrives scattered, not just dimmer.
    defocus_um: float = 1.0

    # --- Illumination ----------------------------------------------------
    illum_contrast: float = 7.0    # end-to-end ratio of the smooth lighting field
    illum_ctrl: np.ndarray = field(
        default_factory=lambda: np.full((3, 4), 0.5)
    )                              # control points, so the field is reproducible
    atten_frac: float = 0.5        # signal lost across the depth of the volume
    atten_axis: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )

    # --- Cell appearance -------------------------------------------------
    radius_um: tuple[float, float] = CELL_RADIUS_UM
    bright_lo: float = 0.42
    bright_hi: float = 0.92
    faint_frac: float = 0.30       # a share, not the range FAINT_FRAC
    texture_amp: float = 0.25      # mottling inside the nucleus
    texture_um: float = 2.0

    # --- Packing ---------------------------------------------------------
    collision_factor: float = 0.9  # a factor, not the range COLLISION_FACTOR
    density_factor: float = 1.0

    # --- Detector and medium ---------------------------------------------
    medium: float = MEDIUM_LEVEL
    poisson_pe: float = 900.0
    read_noise: float = 0.004
    debris: int = 0

    @staticmethod
    def sample(rng) -> "Regime":
        """Draw an acquisition from the distribution fitted to the real set."""
        # Pedestal: fitted to the local background of 836 annotated real cells
        # (p5 0.09, p50 0.25, p95 0.57), which is close to log-normal.
        pedestal = _lognormal(rng, *PEDESTAL)
        # Axial extent: real fwhm_z runs 4.9 (p5) to 26 um (p95) against a ~9 um
        # nucleus, so the axial sigma has to reach ~6 um in the worst stacks.
        psf_z = _loguniform(rng, *PSF_Z_UM)
        # Rescaled to span [0, 1] on every axis, so the field actually
        # realises the contrast it was drawn with. Left as raw uniforms, four
        # similar control points flattened the profile and the frame came out
        # at an octant ratio of 3.4 against a real median of 8.1.
        ctrl = rng.uniform(0.0, 1.0, (3, 4))
        ctrl -= ctrl.min(axis=1, keepdims=True)
        ctrl /= ctrl.max(axis=1, keepdims=True) + 1e-9
        axis = rng.normal(0, 1, 3)
        # Depth attenuation is mostly axial -- the detection objective looks
        # along z -- but the light sheet enters laterally, so not purely.
        axis[0] = abs(axis[0]) * 2.0 + 0.4
        axis /= np.linalg.norm(axis)
        b_lo, b_hi = BRIGHT_SPAN
        span = rng.uniform(0.22, 0.52)
        top = rng.uniform(0.80, 1.0)
        return Regime(
            pedestal=pedestal,
            haze_um=_loguniform(rng, *HAZE_UM),
            haze_z_ratio=float(rng.uniform(*HAZE_Z_RATIO)),
            haze_gamma=float(rng.uniform(*HAZE_GAMMA)),
            halo_amp=float(HALO_AMP * rng.uniform(0.5, 1.6)),
            psf_xy_um=_loguniform(rng, *PSF_XY_UM),
            psf_z_um=psf_z,
            defocus_um=float(rng.uniform(*DEFOCUS_UM)),
            illum_contrast=_lognormal(rng, *ILLUM_CONTRAST),
            illum_ctrl=ctrl,
            atten_frac=float(rng.uniform(*ATTEN_FRAC)),
            atten_axis=axis,
            radius_um=(
                float(CELL_RADIUS_UM[0] * rng.uniform(0.85, 1.12)),
                float(CELL_RADIUS_UM[1] * rng.uniform(0.88, 1.18)),
            ),
            bright_lo=float(np.clip(top - span, b_lo, b_hi)),
            bright_hi=float(np.clip(top, b_lo, b_hi)),
            faint_frac=float(rng.uniform(*FAINT_FRAC)),
            texture_amp=float(rng.uniform(*TEXTURE_AMP)),
            texture_um=float(rng.uniform(*TEXTURE_UM)),
            # Below 1.0 the nuclei interpenetrate and merge into each other
            # after the PSF, which is what dense real tissue looks like and
            # what a tangent-spheres rule can never produce.
            collision_factor=float(rng.uniform(*COLLISION_FACTOR)),
            density_factor=_lognormal(rng, *DENSITY_FACTOR),
            medium=float(MEDIUM_LEVEL * rng.uniform(0.5, 2.0)),
            # Real dark-region noise is 0.006 (p50); the old fixed 300
            # photo-electrons produced 0.010, i.e. the generator was noisier
            # than the microscope while also being higher contrast.
            poisson_pe=_loguniform(rng, *POISSON_PE),
            read_noise=float(rng.uniform(*READ_NOISE)),
            debris=int(rng.integers(*DEBRIS_PER_VOLUME)),
        )

    def with_(self, **kw) -> "Regime":
        return replace(self, **kw)


# --------------------------------------------------------------------------
# Per-cell appearance, held across a track's lifetime
# --------------------------------------------------------------------------


def sample_aniso(n, rng) -> np.ndarray:
    """(N, 3) per-axis scaling of the nucleus, z/y/x.

    A share of nuclei are markedly elongated -- real ones are, especially while
    dividing -- with the long axis anywhere. At ``ELLIPTICAL_P = 0.5`` that is
    half of them, which is why packing cannot ignore it: see ``reserved_um``.
    """
    aniso = rng.uniform(0.82, 1.20, (n, 3))
    el = rng.random(n) < ELLIPTICAL_P
    which = rng.integers(0, 3, n)
    aniso[el, which[el]] *= rng.uniform(1.4, 2.0, int(el.sum()))
    return aniso


def reserved_um(radius_um, aniso) -> np.ndarray:
    """Radius of the sphere that encloses a nucleus, in microns.

    ``_sphere_native`` renders the body out to ``radius_um * aniso`` along each
    axis, so a cell reaches ``radius_um * max(aniso)`` at its longest -- up to
    2.4x its nominal radius for an elongated one. Packing used to reserve only
    ``radius_um``, so half the population was placed as if it were a third
    smaller than it renders, and the cores ran into each other: measured over
    1930 cells, 19.7% of a core was covered by another core on average and 9.2%
    of cells were more than half swallowed.

    The enclosing sphere ignores the rotation ``theta``, which is the point --
    it is rotation-invariant, so it holds whichever way the cell is turned. It
    over-reserves for a long thin nucleus, which costs some density and is the
    conservative direction to err in.
    """
    return np.asarray(radius_um, float) * np.asarray(aniso, float).max(-1)


@dataclass
class Palette:
    """Appearance of N cells: one row per cell, aligned with a centroid array.

    Held constant across frames (with a slow drift, see ``drift``) so that a
    cell looks like itself from one frame to the next.
    """

    radius_um: np.ndarray
    aniso: np.ndarray        # (N, 3) axis scaling, z/y/x
    theta: np.ndarray        # (N,) rotation in the xy plane
    bright: np.ndarray
    texture_seed: np.ndarray

    def __len__(self):
        return len(self.radius_um)

    @staticmethod
    def sample(n, rng, regime: Regime) -> "Palette":
        r_lo, r_hi = regime.radius_um
        radius = rng.uniform(r_lo, r_hi, n)
        aniso = sample_aniso(n, rng)
        bright = np.clip(
            rng.normal(
                (regime.bright_lo + regime.bright_hi) / 2,
                (regime.bright_hi - regime.bright_lo) / 3,
                n,
            ),
            regime.bright_lo,
            regime.bright_hi,
        )
        # The faint population. The real data holds cells that rise only ~44%
        # above their local background where a typical one rises ~220%, and
        # those are precisely the ones a detector misses. A symmetric spread
        # around a bright mean never produces them; a log-normal knock-down on
        # a share of cells does.
        faint = rng.random(n) < regime.faint_frac
        bright[faint] *= np.clip(
            rng.lognormal(np.log(FAINT_MEDIAN), 0.45, int(faint.sum())),
            FAINT_MIN, 1.0,
        )
        return Palette(
            radius_um=radius.astype(np.float32),
            aniso=aniso.astype(np.float32),
            theta=rng.uniform(0, np.pi, n).astype(np.float32),
            bright=bright.astype(np.float32),
            texture_seed=rng.integers(0, 2**31 - 1, n).astype(np.int64),
        )

    def take(self, idx) -> "Palette":
        idx = np.asarray(idx, int)
        return Palette(
            self.radius_um[idx], self.aniso[idx], self.theta[idx],
            self.bright[idx], self.texture_seed[idx],
        )

    def drift(self, rng, regime: Regime) -> "Palette":
        """One frame of slow appearance change: bleaching, growth, rotation.

        Small on purpose. A cell has to remain recognisable across a window --
        that is the signal the tracker is meant to use -- while not being
        pixel-identical, which would let it match on a fingerprint that the
        real data does not provide.
        """
        n = len(self)
        r_lo, r_hi = regime.radius_um
        return Palette(
            radius_um=np.clip(
                self.radius_um * rng.normal(1.0, 0.02, n), r_lo * 0.8, r_hi * 1.2
            ).astype(np.float32),
            aniso=np.clip(
                self.aniso * rng.normal(1.0, 0.02, (n, 3)), 0.6, 2.6
            ).astype(np.float32),
            theta=(self.theta + rng.normal(0, 0.08, n)).astype(np.float32),
            bright=np.clip(
                self.bright * rng.normal(0.995, 0.05, n), 0.02, 1.0
            ).astype(np.float32),
            texture_seed=self.texture_seed,
        )

    def divide(self, k, rng, regime: Regime) -> tuple["Palette", "Palette"]:
        """Two daughters from cell ``k``: smaller, dimmer, freshly textured."""
        out = []
        for _ in range(2):
            out.append(Palette(
                radius_um=np.array(
                    [self.radius_um[k] * rng.uniform(0.72, 0.88)], np.float32
                ),
                aniso=(self.aniso[k] * rng.normal(1.0, 0.08, 3))[None].astype(
                    np.float32
                ),
                theta=np.array([rng.uniform(0, np.pi)], np.float32),
                bright=np.array(
                    [min(1.0, self.bright[k] * rng.uniform(0.9, 1.25))], np.float32
                ),
                texture_seed=rng.integers(0, 2**31 - 1, 1).astype(np.int64),
            ))
        return out[0], out[1]


def concat(palettes: list[Palette]) -> Palette:
    return Palette(*[np.concatenate([getattr(p, f) for p in palettes])
                     for f in ("radius_um", "aniso", "theta", "bright",
                               "texture_seed")])


# --------------------------------------------------------------------------
# Fields derived from a regime
# --------------------------------------------------------------------------


def illumination_field(shape, regime: Regime) -> np.ndarray:
    """Smooth multiplicative lighting, spanning ``illum_contrast`` end to end.

    Separable -- one smooth profile per axis, combined as a product -- which
    covers what a light sheet shows (a directional ramp and depth attenuation)
    without materialising an intermediate volume. Every profile varies over the
    whole field, so across a ~9 um cell it is flat: cells keep their shape and
    no new maxima appear, only their absolute brightness changes.

    Driven by ``regime.illum_ctrl`` rather than by an rng, so every frame of a
    sequence is lit identically. It used to be redrawn per frame, which made
    the background of a tracking window flicker between t and t+1.
    """
    if regime.illum_contrast <= 1.0:
        return np.ones((1, 1, 1), np.float32)
    span = float(regime.illum_contrast) ** (1.0 / 3.0)   # each axis a cube root
    prof = []
    for ax, n in enumerate(shape):
        ctrl = regime.illum_ctrl[ax]
        g = np.interp(np.linspace(0, len(ctrl) - 1, n), np.arange(len(ctrl)), ctrl)
        prof.append((span ** (2.0 * g - 1.0)).astype(np.float32))
    fieldv = prof[0][:, None, None] * prof[1][None, :, None] * prof[2][None, None, :]
    return fieldv / fieldv.max()


def depth_ramp(shape, regime: Regime) -> np.ndarray:
    """0 at the near face of the volume, 1 at the far one, along ``atten_axis``.

    Drives both the attenuation and the extra blur: light from deep in the
    tissue arrives dimmer *and* scattered. Modelling only the dimming produces
    deep cells that are faint but still crisp, which is not what a real stack
    shows and is far easier to detect than what it does show.
    """
    n = np.asarray(regime.atten_axis, float)
    coords = [np.arange(s, dtype=np.float32) * v for s, v in zip(shape, VOXEL_NATIVE)]
    s = (n[0] * coords[0][:, None, None]
         + n[1] * coords[1][None, :, None]
         + n[2] * coords[2][None, None, :])
    s -= s.min()
    return (s / (s.max() + 1e-9)).astype(np.float32)
