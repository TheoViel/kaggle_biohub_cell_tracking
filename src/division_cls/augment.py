"""Augmentations for the division clip, on top of the contrastive pipeline's.

`embeddings/augment.py` supplies rotation, elastic deformation, scale, position
jitter and photometry, all drawn **once per clip** so the motion between frames
survives. Those are the right defaults and `dataset.py` still uses them. What is
here is the part that is specific to this task, and it exists because of what
the task is short of: 151 positives across two embryos, 26 of them in fold 1.
Synthetic divisions were tried three times and degraded every configuration
(see README.md), so the labels are not going to be more numerous -- the
only lever left is making each one count for more.

Two rules decide what may go in here.

**A division is a temporal event.** Measured on the annotation, a parent at its
own division frame is indistinguishable from an ordinary cell (elongation 1.51
against 1.51, bimodality 0.244 against 0.253); the signature is at t+1, where
bimodality doubles. So anything that scrambles time, blanks the frame after the
division, or re-rolls the geometry per frame destroys the label rather than
augmenting it. Everything below either applies one draw to the whole clip, or
applies a *small* per-frame perturbation that stays under the real inter-frame
motion (1.86 um median).

**The group of the image, not of the object.** Reflection is not reachable by
rotation, so `flip_p` genuinely doubles the symmetry group the continuous
rotation already covers. `rot90_p` is worth having separately even though
`rot_deg=180` subsumes it, because a quarter turn on a 32x32 grid is exact
while a 37-degree one interpolates -- the model sees both a clean and a
resampled version of the same crop.

`time_reverse_p` is the one that changes the label, and it is off by default.
A reversed division is two cells becoming one, which is not a division, so it
makes a *hard negative* that no other augmentation can: it shares the positive's
appearance exactly and differs only in direction. The reason for caution is that
cells do not merge in real data, so these negatives are off-distribution and the
model may learn to detect reversal by some other artefact. Worth a sweep, not
worth assuming.
"""

from __future__ import annotations

import numpy as np

from embeddings.augment import AUG_DEFAULTS, resolve_aug_cfg

DIV_AUG_DEFAULTS: dict = {
    # --- exact symmetries of the y/x plane -------------------------------
    # Free and label-preserving: a division has no canonical orientation, and
    # these are the transforms that cost no interpolation.
    "flip_p": 0.5,          # mirror, per axis -- not reachable by rotation
    "rot90_p": 0.5,         # exact quarter turn, unlike the continuous one

    # --- the crop's own placement ----------------------------------------
    # z is the coarse axis (1.625 um a slice) and the detector's z estimate is
    # the least reliable coordinate, so the crop should not assume it is right.
    # One draw for the clip: a per-frame z shift would fake motion in z.
    "z_jitter": 1,          # slices, integer, uniform in +-this

    # Residual per-frame translation: registration and centroid error, which
    # are per frame in the real data. Kept far below the 1.86 um median
    # inter-frame motion -- at that scale it would erase the displacement the
    # model reads rather than perturb it.
    "frame_shift_um": 0.4,

    # --- acquisition, per frame ------------------------------------------
    # On top of the clip-wide photometric draw. Bleaching and illumination
    # flicker are genuinely per frame, so a model that keys on absolute frame
    # brightness is keying on an artefact. Small, for the same reason as above.
    "frame_brightness": 0.06,
    "frame_contrast": 0.06,

    # Signal-dependent noise, which is what a photon-limited fluorescence image
    # actually has; the base config's Gaussian `noise_std` is added on top.
    "shot_noise": 0.04,

    # Focus varies through the stack and between acquisitions.
    "blur_p": 0.25,
    "blur_sigma": (0.4, 1.1),   # voxels, y/x

    # --- occlusion --------------------------------------------------------
    # Random erasing, one box per clip at the same place in every frame. Same
    # place, or the box itself becomes a moving object and invents a division.
    "cutout_p": 0.25,
    "cutout_frac": (0.1, 0.3),  # of the crop's side

    # --- label-changing; see the module docstring -------------------------
    "time_reverse_p": 0.0,
    # Blank the frame *before* the division only. It carries the least signal
    # of the three (bimodality 0.188, the lowest row of the measured table), so
    # dropping it tests whether the model leans on it. Never t or t+1: those
    # carry the label.
    "drop_context_p": 0.0,
}

# Keys owned here versus keys owned by `embeddings/augment.py`. Split rather
# than merged so a typo is still caught: `resolve_aug_cfg` rejects names it does
# not know, and silently keeping a default is a setting that looks applied and
# is not.
_DIV_KEYS = frozenset(DIV_AUG_DEFAULTS)
_BASE_KEYS = frozenset(AUG_DEFAULTS)


def resolve_div_aug(cfg: dict | None) -> tuple[dict, dict]:
    """Split one flat config into `(base_cfg, div_cfg)`, both fully resolved."""
    cfg = dict(cfg or {})
    unknown = sorted(set(cfg) - _DIV_KEYS - _BASE_KEYS)
    if unknown:
        import difflib

        known = sorted(_DIV_KEYS | _BASE_KEYS)
        hints = []
        for key in unknown:
            close = difflib.get_close_matches(key, known, n=1, cutoff=0.6)
            hints.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise KeyError("unknown augmentation setting(s): " + ", ".join(hints)
                       + f". Known keys: {known}")
    base = resolve_aug_cfg({k: v for k, v in cfg.items() if k in _BASE_KEYS})
    div = {**DIV_AUG_DEFAULTS, **{k: v for k, v in cfg.items() if k in _DIV_KEYS}}
    return base, div


def sample_clip_aug(cfg: dict, rng: np.random.Generator, n_frames: int,
                    voxel_yx) -> dict:
    """Draw everything that has to be decided before the pixels are read.

    Returned as a plain dict so the caller can apply the parts that belong at
    different stages: `z` and `shift` change *which* pixels are cut, the rest
    are applied to the assembled clip.
    """
    vy, vx = float(voxel_yx[0]), float(voxel_yx[1])
    zj = int(cfg["z_jitter"])
    s = float(cfg["frame_shift_um"])
    return {
        "z": int(rng.integers(-zj, zj + 1)) if zj else 0,
        # per frame, in voxels
        "shift": (rng.normal(0.0, s, (n_frames, 2)) / np.array([vy, vx])
                  if s else np.zeros((n_frames, 2))),
        "flip_y": bool(rng.random() < cfg["flip_p"]),
        "flip_x": bool(rng.random() < cfg["flip_p"]),
        "k90": int(rng.integers(4)) if rng.random() < cfg["rot90_p"] else 0,
        "reverse": bool(rng.random() < cfg["time_reverse_p"]),
        "drop_context": bool(rng.random() < cfg["drop_context_p"]),
        "blur": (float(rng.uniform(*cfg["blur_sigma"]))
                 if rng.random() < cfg["blur_p"] else 0.0),
        "cutout": (float(rng.uniform(*cfg["cutout_frac"])),
                   float(rng.random()), float(rng.random()))
        if rng.random() < cfg["cutout_p"] else None,
        "frame_photo": np.stack([
            1.0 + rng.uniform(-cfg["frame_contrast"], cfg["frame_contrast"], n_frames),
            rng.uniform(-cfg["frame_brightness"], cfg["frame_brightness"], n_frames),
        ], axis=1),
    }


def _blur1d(a, sigma, axis):
    """Separable Gaussian blur without pulling in scipy for one call."""
    r = max(1, int(round(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    b = np.pad(a, pad, mode="edge")
    out = np.zeros_like(a)
    for i, w in enumerate(k):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(i, i + a.shape[axis])
        out += w * b[tuple(sl)]
    return out


def apply_clip_aug(clip: np.ndarray, draw: dict, cfg: dict,
                   rng: np.random.Generator, label: float) -> tuple[np.ndarray, float]:
    """Everything that acts on the assembled `(T, pz, H, W)` clip.

    Returns the clip and its label, since `time_reverse` changes the answer: a
    division played backwards is two cells becoming one, which is a negative.
    Order is deliberate -- geometry, then occlusion, then the sensor. Blurring
    before adding noise would make the noise sharper than the signal, which no
    microscope does.
    """
    # Copy up front. The steps below write in place, and `ascontiguousarray` is
    # a no-op on an array that is already contiguous float32 -- so without this
    # the cutout blanks the *caller's* clip. Harmless for a freshly cut crop,
    # silent corruption for a cached one, which `SyntheticClipDataset` serves.
    out = np.array(clip, np.float32)
    if draw["reverse"]:
        out = out[::-1]
        # Only a positive changes answer. A negative reversed is still a
        # negative, so it costs nothing and is left in as extra variety.
        label = 0.0 if label else label
    if draw["drop_context"] and len(out) >= 3:
        out[0] = 0.0
    if draw["flip_y"]:
        out = out[:, :, ::-1]
    if draw["flip_x"]:
        out = out[:, :, :, ::-1]
    if draw["k90"]:
        out = np.rot90(out, draw["k90"], axes=(2, 3))
    out = np.ascontiguousarray(out, np.float32)

    if draw["cutout"] is not None:
        frac, fy, fx = draw["cutout"]
        h = max(1, int(round(frac * out.shape[2])))
        w = max(1, int(round(frac * out.shape[3])))
        y0 = int(fy * (out.shape[2] - h))
        x0 = int(fx * (out.shape[3] - w))
        # Same box in every frame: a box that moved would itself look like an
        # object appearing, which is the event being classified.
        out[:, :, y0:y0 + h, x0:x0 + w] = 0.0

    photo = draw["frame_photo"]
    out = out * photo[:, 0, None, None, None] + photo[:, 1, None, None, None]

    if draw["blur"]:
        out = _blur1d(_blur1d(out, draw["blur"], 2), draw["blur"], 3)

    shot = float(cfg["shot_noise"])
    if shot:
        # Variance proportional to signal, as a photon count is -- the bright
        # nucleus is noisier than the background, which additive Gaussian noise
        # gets backwards.
        out = out + rng.normal(0.0, 1.0, out.shape).astype(np.float32) * (
            shot * np.sqrt(np.clip(out, 0.0, None)))

    return np.clip(out, 0.0, 1.0).astype(np.float32), label
