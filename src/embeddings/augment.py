"""The six augmentations ASCENT builds its positive pairs from.

Two views of the same frame are drawn independently; a cell's identity is
whatever survives both. That is the whole supervision signal for a synthetic
pair, so what these transforms do and do not preserve *is* the task definition.

A *real* pair -- frame t against frame t+1, paired by the ground-truth graph --
uses the same transforms but draws the frame-wide half **once** for both
frames, so the only difference left between them is the one the annotation
describes: the cells actually moving. Hence the split below into
``sample_geometry`` (shared) and ``sample_cells`` (per frame).

The paper names six -- rotation, elastic deformation, crop/resize, position
jitter, intensity jitter, object dropout -- and reports an ablation for one of
them: position jitter takes accuracy from 37.3% to 95.1%. It does not give
parameter ranges for any of them, so the magnitudes below are chosen for this
data and flagged as such; they are the part most worth re-tuning.

Why position jitter matters so much is worth stating, because it decides how
the jitter has to be applied. A cell is identified by a patch *and* a
coordinate. If the coordinate were exact in both views, the network could match
cells by coordinate alone and never look at the patch -- the task would be
trivial and the embeddings worthless. Jittering **independently per view**, and
moving the crop centre and the reported coordinate together within a view,
makes the coordinate unreliable at exactly the scale a detector's own error
lives at, so appearance has to carry the rest. That is what "robust to small
positional perturbations" buys.
"""

from __future__ import annotations

import numpy as np

# Magnitudes are ours, not the paper's. Voxels are anisotropic here -- 1.625 um
# in z against 0.40625 in y/x -- so anything spatial is specified in microns and
# converted, or the same number would mean four different things per axis.
AUG_DEFAULTS: dict = {
    # --- position jitter: the one the ablation singles out ---
    # Drawn per view, per cell, uniform in +/- this per axis -- *bounded*, not
    # Gaussian. A normal draw's tail reaches several sigma, and cells here sit
    # ~8 um apart, so a tail event would recentre the patch on the neighbouring
    # cell and hand the loss two different cells as a positive pair. 2 um is
    # about a quarter of that spacing and comparable to the detector's own
    # centring error, so the patch always still shows the cell it names.
    "jitter_um": 2.0,
    # --- rotation, in the y/x plane only ---
    # z is the coarse axis at 4x the spacing; rotating it into y/x would claim
    # a resolution the microscope never had.
    "rot_p": 0.5,
    "rot_deg": 180.0,
    # --- elastic deformation ---
    "elastic_p": 0.5,
    "elastic_um": 1.5,      # displacement amplitude
    "elastic_sigma_um": 8.0,  # smoothness of the displacement field
    # --- crop / resize ---
    "scale_p": 0.5,
    "scale_range": (0.85, 1.18),
    # --- intensity jitter ---
    "intensity_p": 0.8,
    "brightness": 0.25,     # additive, in normalised units
    "contrast": 0.25,       # multiplicative, 1 +/- this
    "gamma": 0.25,          # exponent, 1 +/- this
    "noise_std": 0.02,
    # --- object dropout ---
    # Cells missing from one view. It is what teaches the matcher that a track
    # can go unmatched, which a detector's own misses will produce anyway.
    "dropout_p": 0.1,
}


def resolve_aug_cfg(cfg: dict | None) -> dict:
    """``AUG_DEFAULTS`` overridden by ``cfg``, rejecting names it does not know.

    Same reasoning as the tracking pipeline's: a mistyped key that silently
    keeps the default is a setting that looks applied and is not.
    """
    cfg = dict(cfg or {})
    unknown = sorted(set(cfg) - set(AUG_DEFAULTS))
    if unknown:
        import difflib

        hints = []
        for key in unknown:
            close = difflib.get_close_matches(key, AUG_DEFAULTS, n=1, cutoff=0.6)
            hints.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise KeyError(
            "unknown augmentation setting(s): " + ", ".join(hints)
            + f". Known keys: {sorted(AUG_DEFAULTS)}"
        )
    return {**AUG_DEFAULTS, **cfg}


def sample_cells(
    coords: np.ndarray,
    cfg: dict,
    rng: np.random.Generator,
    voxel_size: tuple[float, float, float],
) -> dict:
    """The per-cell half of a view: position jitter and object dropout.

    Returns the jittered ``coords`` (image voxels) and the boolean ``keep``
    mask. Split from the geometry because a *real* pair -- frame t and frame
    t+1 -- has to share the geometry while drawing these independently: the two
    frames hold different cells, and the jitter is per cell by construction.
    """
    n = len(coords)
    vs = np.asarray(voxel_size, np.float32)

    # --- position jitter, per cell, in microns -> voxels ---
    jitter = rng.uniform(
        -cfg["jitter_um"], cfg["jitter_um"], size=(n, 3)
    ).astype(np.float32) / vs
    jittered = coords.astype(np.float32) + jitter

    # --- object dropout, per cell ---
    keep = rng.random(n) >= cfg["dropout_p"]
    if n and not keep.any():    # never hand back an empty view
        keep[rng.integers(n)] = True

    return {"coords": jittered, "keep": keep}


def sample_geometry(
    cfg: dict,
    rng: np.random.Generator,
    voxel_size: tuple[float, float, float],
    shape_yx: tuple[int, int],
) -> dict:
    """The frame-wide half of a view: rotation, scale, elastic field, photometry.

    Decided once per view and shared by every cell in it -- one rotation, one
    scale, one displacement field for the whole frame -- because a view is meant
    to be the same scene imaged differently, not a bag of cells each transformed
    on its own. For a real pair the same draw is applied to both frames, so the
    ground-truth motion between them survives the augmentation unchanged.
    """
    angle = (
        float(rng.uniform(-cfg["rot_deg"], cfg["rot_deg"]))
        if rng.random() < cfg["rot_p"] else 0.0
    )
    scale = (
        float(rng.uniform(*cfg["scale_range"]))
        if rng.random() < cfg["scale_p"] else 1.0
    )

    elastic = None
    if rng.random() < cfg["elastic_p"]:
        elastic = _elastic_field(
            shape_yx, cfg, rng, np.asarray(voxel_size, np.float32)[1:]
        )

    # --- photometry, shared by the whole view ---
    photo = {"brightness": 0.0, "contrast": 1.0, "gamma": 1.0, "noise": 0.0}
    if rng.random() < cfg["intensity_p"]:
        photo = {
            "brightness": float(rng.uniform(-cfg["brightness"], cfg["brightness"])),
            "contrast": float(1.0 + rng.uniform(-cfg["contrast"], cfg["contrast"])),
            "gamma": float(1.0 + rng.uniform(-cfg["gamma"], cfg["gamma"])),
            "noise": float(cfg["noise_std"]),
        }

    return {
        "angle": angle,
        "scale": scale,
        "elastic": elastic,
        "photo": photo,
    }


def sample_view(
    coords: np.ndarray,
    cfg: dict,
    rng: np.random.Generator,
    voxel_size: tuple[float, float, float],
    shape_yx: tuple[int, int],
) -> dict:
    """One whole view: ``sample_geometry`` and ``sample_cells`` together.

    Returns a dict with the jittered ``coords`` (image voxels), the boolean
    ``keep`` mask, and the per-view photometric and geometric parameters the
    patch extractor needs.
    """
    return {
        **sample_geometry(cfg, rng, voxel_size, shape_yx),
        **sample_cells(coords, cfg, rng, voxel_size),
    }


def _elastic_field(shape_yx, cfg, rng, vs_yx):
    """A smooth random displacement field over the frame, in voxels.

    Coarse noise upsampled rather than per-voxel noise smoothed: the field only
    has to be smooth on the scale of a cell, and building it at full resolution
    for every sample costs more than the augmentation is worth.
    """
    from scipy.ndimage import zoom

    sigma_vox = np.asarray(cfg["elastic_sigma_um"], np.float32) / vs_yx
    coarse = np.maximum(2, (np.asarray(shape_yx) / np.maximum(sigma_vox, 1)).astype(int))
    amp = np.asarray(cfg["elastic_um"], np.float32) / vs_yx
    field = []
    for axis in range(2):
        low = rng.normal(0.0, 1.0, size=tuple(coarse)).astype(np.float32)
        up = zoom(low, np.asarray(shape_yx) / coarse, order=3)
        field.append(up * amp[axis])
    return np.stack(field)          # (2, Y, X) displacement in y and x


def apply_photometric(patch: np.ndarray, photo: dict, rng: np.random.Generator):
    """Brightness, contrast, gamma and noise, on a patch already in [0, 1]."""
    out = patch * photo["contrast"] + photo["brightness"]
    out = np.clip(out, 0.0, None)
    if photo["gamma"] != 1.0:
        out = out ** photo["gamma"]
    if photo["noise"] > 0:
        out = out + rng.normal(0.0, photo["noise"], size=out.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
