"""Acquisition constants the generator matches.

Taken from the real data: a voxel is 1.625 um in z and 0.40625 um in y/x, and
the training pipeline pools 4x in y/x. Everything downstream is expressed in
microns so the two never drift apart.

Most of what follows is a *range*, not a value. The real set is a spread of
imaging conditions, not one condition (see ``acquisition.Regime``), and a
generator pinned to the middle of that spread produces the easy corner of it.
Each range is fitted to a measurement over 210 real frames from 70 videos and
836 annotated cells; ``calibrate.py`` recomputes those measurements and is what
should be re-run after touching anything here.
"""

from __future__ import annotations

import numpy as np

VOXEL_NATIVE = np.array([1.625, 0.40625, 0.40625])
POOL = 4                                   # y/x pooling of the training pipeline
DOWNSAMPLE = (1, POOL, POOL)
EFF_VOXEL = VOXEL_NATIVE * np.array(DOWNSAMPLE)


# --- Tissue shell and placement -------------------------------------------

SHELL_RMIN_FOV = 1.5        # smallest sphere radius fitted, in field-of-view units
SHELL_EDGE_FRAC = 0.35      # share of cells pushed onto the shell surface
SHELL_DENS_SMOOTH = 14.0    # voxels, smoothing of the density field
SHELL_JITTER = 0.25         # radial jitter about the shell
SHELL_DENS_CONTRAST = 1.0

# Clustering: a fraction of cells drawn into tight groups, as real tissue shows
CLUSTER = True
CLUSTER_FRAC_N = 0.10       # share of cells acting as cluster seeds
CLUSTER_FRAC = 0.72         # share of the rest assigned to a seed
CLUSTER_SPREAD_UM = 14.0

# Centre-to-centre separation as a multiple of the two radii: 1.0 = tangent,
# below 1.0 = interpenetrating. Held at exactly 1.0 the tissue was a lattice of
# touching-but-separate spheres; real nuclei crowd each other and merge into
# one blob after the PSF, so the low end of this range is where the hard,
# ambiguous detections come from.
COLLISION_FACTOR = (0.68, 1.05)
# Per-volume multiplier on the cell count, so density varies as it does between
# real videos (blob counts run 37 to 323 per frame, p5 to p95). The upper tail
# matters: the real set reaches tissue packed shoulder to shoulder, where the
# blobs merge and detection is hardest, and a generator whose densest frame is
# its median frame never renders that.
DENSITY_FACTOR = (1.3, 0.40, 0.5, 2.8)       # median, sigma, lo, hi (log-normal)


# --- Cell appearance ------------------------------------------------------

# Real cells measure 8.5 um across at half maximum (p50 of the blob-detected
# annotated cells, 6.5 to 11.4 across the quartiles). 3.5-5.5 rendered them at
# 6.9 um, i.e. visibly smaller than the tissue they were meant to imitate.
CELL_RADIUS_UM = (3.8, 6.0)
CELL_SHARPNESS = 4.5
ELLIPTICAL_P = 0.5          # share of nuclei with one markedly longer axis
# Brightness is drawn as a per-acquisition window inside this envelope rather
# than a single fixed (0.42, 0.92): some real stacks are exposed so that most
# nuclei saturate, others so that none do.
BRIGHT_SPAN = (0.34, 1.0)
# Mottling inside the nucleus. A real nucleus is not a smooth filled ball --
# chromatin is textured, and that texture is what puts spurious local maxima
# inside a cell. Rendering perfectly smooth blobs means local-max suppression
# never has to cope with them.
TEXTURE_AMP = (0.04, 0.32)
TEXTURE_UM = (1.2, 3.0)

# A share of cells is dimmed by a log-normal factor, to reproduce the faint
# population the real data has: a quarter of annotated real cells rise less
# than 100% above their local background.
FAINT_FRAC = (0.18, 0.45)
FAINT_MEDIAN = 0.45
FAINT_MIN = 0.15


# --- Scattered light ------------------------------------------------------

# The pedestal of out-of-focus and scattered light a cell sits on, as a
# fraction of full scale: log-normal (median, sigma, lo, hi) fitted to the
# local background of 836 annotated real cells (p5 0.09, p50 0.25, p95 0.57).
# This is the single largest correction against the old constants, which put
# a cell on a background of 0.06 and so gave it 400-1400% contrast where the
# real median is 220%.
PEDESTAL = (0.15, 0.55, 0.04, 0.58)
HAZE_UM = (7.0, 22.0)       # lateral scattering length (log-uniform)
HAZE_Z_RATIO = (1.1, 2.4)   # axial spread of the scattering, relative to lateral
# Falloff of the haze away from the tissue. At 1.0 the pedestal leaked across
# the whole frame and only 9% of voxels stayed below 0.05, against 31% for the
# real data: a real frame has genuinely empty space next to its tissue, and a
# generator that fills it removes both the dark void and the low-frequency
# structure (cv 0.59 against a real 0.81).
HAZE_GAMMA = (1.2, 3.0)
# The pedestal scales as the square root of cell density about this count --
# the relation the real frames show (51 blobs -> background 0.13-0.15, 404
# blobs -> 0.30-0.51: 7.9x the density for 2.8x the background).
HAZE_NOMINAL_CELLS = 250.0
HAZE_DENSITY_EXP = 0.5
# Floor on how far a labelled cell must rise above the pedestal at its centre.
# The faintest of 836 real annotated cells rises 14% above its surroundings, so
# anything below that is not a hard example but a label on nothing, which
# teaches the detector to fire where there is no cell. See render.
MIN_CONTRAST = 0.14
HALO_AMP = 0.18             # tight per-cell halo, on top of the wide haze
HALO_UM = 2.6

# Debris and autofluorescent junk: diffuse blobs, well above cell scale, which
# a real embryo is full of and which this generator produced none of. They are
# deliberately *not* cell-sized -- an unlabelled cell-sized blob would be a
# false negative in the target rather than a hard negative.
DEBRIS_PER_VOLUME = (0, 9)
DEBRIS_RADIUS_UM = (7.0, 22.0)
DEBRIS_AMP = (0.04, 0.22)


# --- Optics and detector --------------------------------------------------

MEDIUM_LEVEL = 0.04         # background of the mounting medium
PSF_XY_UM = (0.45, 1.15)    # lateral sigma (log-uniform)
# Axial sigma, absolute (this used to be a *ratio* of the lateral one, pinned
# at 2.5, giving a 1.4 um axial sigma). Real cells measure 11.4 um across in z
# at half maximum (p50), up to 26 um (p95), against a ~9 um nucleus: the axial
# smearing has to reach ~6 um of sigma for the blurriest stacks.
PSF_Z_UM = (1.2, 6.5)
# Extra blur on the far side of the volume: deep light arrives scattered, not
# merely dimmer.
DEFOCUS_UM = (0.0, 2.5)
# Signal lost from the near face to the far one.
ATTEN_FRAC = (0.10, 0.55)
# End-to-end ratio of the smooth lighting field, log-normal. Real frames
# measure an octant max/min of 8.1 (p50) and 23 (p95); the old fixed 5.0 gave
# 2.5-6.9, i.e. the generator was more evenly lit than any real stack.
#
# Most of that real unevenness is *tissue*, though, not lighting: a real frame
# holds a bright band and an empty void, and the void is dark because nothing
# is there. Driving the whole octant ratio from the lighting instead dims the
# cells themselves -- at a contrast of 8 the detected-cell peak fell to 0.42
# against a real 0.90 and the blob count halved. So the lighting stays modest
# and the haze gamma, which darkens the void, carries the rest.
ILLUM_CONTRAST = (3.2, 0.45, 1.6, 12.0)

# Photo-electrons at full scale, which sets shot noise (log-uniform). The old
# fixed 300 gave 0.010 of high-frequency noise in dark regions against a real
# median of 0.006 -- the generator was noisier than the microscope *and*
# higher contrast, which is the wrong trade in both directions. The real
# frames imply far more photons than that: at the 0.05 intensity where the
# noise is measured, 0.006 of shot noise is ~1400 electrons, and the pedestal
# now puts most voxels well above 0.05.
POISSON_PE = (3000.0, 25000.0)
READ_NOISE = (0.001, 0.004)

# "stride" matches the official evaluator, which subsamples rather than averages.
POOL_MODE = "stride"
