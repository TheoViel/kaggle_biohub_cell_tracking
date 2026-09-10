"""Oriented crops around a pair of detections, labelled "one cell" or "two".

One sample is a pair of detections in the same frame, rendered as a short clip
of crops the way `division_cls.dataset` renders one cell. Three things differ,
and each is forced by a measurement rather than chosen.

**The crop spans z, not just y/x.** 61% of duplicate pairs are separated more in
z than in the plane, at a median of 3 z-slices, because the voxel is 4x coarser
in z (1.625 um against 0.40625) and the detector resolves one cell into two
peaks on different planes. The division classifier's `patch_z=3` reaches +-1
slice and would simply not contain the second detection. `patch_z=9` reaches
+-4, which covers the 90th percentile.

**The crop is oriented along the pair.** The pair's axis is rotated onto the
crop's +x, so "the two detections" always sit at the same two places and the
model spends no capacity learning that the question is rotation-invariant. It
also makes the diagnostic feature legible: along that axis, one cell is a single
lobe and two cells are two lobes with a valley between them.

**Rotation augmentation is therefore off by default.** `rot90_p` and `rot_p`
would undo the canonicalisation. Flips stay on: flipping across the axis is a
mirror, and flipping along it swaps the two members, which the label does not
distinguish because a pair is unordered.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from cell_merging_cls.pairs import (
    DUP_RADIUS_UM,
    MAX_PAIR_UM,
    SCALE_ARR,
    division_probe,
    labelled_pairs,
)
from division_cls.augment import apply_clip_aug, resolve_div_aug, sample_clip_aug
from embeddings.augment import apply_photometric, sample_geometry
from embeddings.patches import extract_patches
from params import SCALE

warnings.filterwarnings("ignore", category=UserWarning)


def pair_geometry(a_zyx, b_zyx):
    """`(angle_deg, centre_zyx, features)` for a pair given in image voxels.

    The angle is the one that puts the pair on the crop's +x axis. Patch offset
    `(gy, gx)` is sampled at `(cos*gy - sin*gx, sin*gy + cos*gx)`, so the +x
    offset `(0, 1)` lands at `(-sin, cos)`; matching that to the unit pair
    direction `(uy, ux)` gives `theta = atan2(-uy, ux)`.

    `features` are the separations the pixels cannot show once the crop is cut:
    total, in-plane and along z, all in microns.
    """
    a = np.asarray(a_zyx, float)
    b = np.asarray(b_zyx, float)
    dz, dy, dx = (b - a) * SCALE_ARR
    dxy = float(np.hypot(dy, dx))
    angle = float(np.degrees(np.arctan2(-(b[1] - a[1]), (b[2] - a[2]))))
    centre = (a + b) / 2.0
    sep = float(np.sqrt(dxy ** 2 + dz ** 2))
    return angle, centre, np.array([sep, dxy, abs(dz)], np.float32)


class MergePairDataset(Dataset):
    """Pairs of detections, labelled by the annotation where it can speak.

    Parameters
    ----------
    data_dir : str
        Folder of `<sample>.zarr` and `<sample>.geff` -- the images and the
        annotation.
    preds_dir : str
        Folder of `<sample>.npz` written by `main_edge_predict.py`. The pairs
        are built from *detections*, not from annotated nodes, because that is
        what the model sees at inference.
    embryos : list of str
        The split, by embryo, for the reason everything else in this repo splits
        that way.
    neg_per_pos : float
        Ordinary "two distinct cells" negatives kept per duplicate. Sister pairs
        are exempt and always kept in full.
    sister_multiplier : int
        Times each sister negative is repeated in the epoch. They are the
        negative the whole model is judged on and there are few of them.
    """

    def __init__(
        self,
        data_dir,
        preds_dir,
        embryos=None,
        patch_yx: int = 32,
        patch_z: int = 9,
        n_frames: int = 3,
        max_pair_um: float = MAX_PAIR_UM,
        dup_radius: float = DUP_RADIUS_UM,
        neg_per_pos: float = 8.0,
        unlabelled_per_pos: float = 20.0,
        positive_multiplier: int = 1,
        sister_multiplier: int = 4,
        augment: bool = False,
        aug_cfg: dict | None = None,
        max_stacks: int | None = None,
        probe: bool = False,
        seed: int = 0,
    ):
        if n_frames % 2 == 0:
            raise ValueError(f"n_frames must be odd, got {n_frames}")

        self.data_dir = Path(data_dir)
        self.preds_dir = Path(preds_dir)
        self.patch_yx = int(patch_yx)
        self.patch_z = int(patch_z)
        self.n_frames = int(n_frames)
        self.augment = bool(augment)
        self.aug_cfg, self.div_aug = resolve_div_aug(aug_cfg)
        self.seed = int(seed)
        self._vol = {}

        rng = np.random.default_rng(seed)
        self.samples = []        # (stem, t, a_zyx, b_zyx, label, kind)
        self.stacks = []
        for npz in sorted(self.preds_dir.glob("*.npz")):
            stem = npz.stem
            if embryos is not None and not any(stem.startswith(e) for e in embryos):
                continue
            geff = self.data_dir / f"{stem}.geff"
            if not geff.exists():
                continue
            with np.load(npz) as d:
                coords = d["coords"]
            if probe:
                found = division_probe(coords, geff, dup_radius, max_pair_um)
            else:
                found = labelled_pairs(coords, geff, max_pair_um, dup_radius,
                                       rng=rng, neg_per_pos=neg_per_pos,
                                       unlabelled_per_pos=unlabelled_per_pos)
            if not found:
                continue
            for a, b, label, kind in found:
                self.samples.append((stem, int(coords[a, 0]),
                                     coords[a, 1:].astype(np.float32),
                                     coords[b, 1:].astype(np.float32),
                                     int(label), kind))
            self.stacks.append(stem)
            if max_stacks and len(self.stacks) >= max_stacks:
                break

        self.labels = np.array([s[4] for s in self.samples], np.int8)
        self.kinds = np.array([s[5] for s in self.samples])

        # Oversampling is an index list so `samples` stays the honest set.
        self.index = np.arange(len(self.samples))
        if augment:
            extra = []
            mult = max(1, int(positive_multiplier))
            if mult > 1:
                extra.append(np.repeat(np.flatnonzero(self.labels == 1), mult - 1))
            smult = max(1, int(sister_multiplier))
            if smult > 1:
                extra.append(np.repeat(np.flatnonzero(self.kinds == "sister"), smult - 1))
            if extra:
                self.index = np.concatenate([self.index, *extra])
        self.positive_multiplier = max(1, int(positive_multiplier))
        self.sister_multiplier = max(1, int(sister_multiplier))

    def __len__(self):
        return len(self.index)

    @property
    def summary(self) -> str:
        n_pos = int((self.labels == 1).sum())
        n_sis = int((self.kinds == "sister").sum())
        n_unl = int((self.kinds == "unlabelled").sum())
        n_neg = int((self.labels == 0).sum())
        s = (f"{len(self.stacks)} stacks, {n_pos} duplicate pairs, {n_neg} negatives "
             f"({n_sis} sisters, {n_unl} unlabelled), 1:{n_neg / max(n_pos, 1):.1f}")
        if len(self.index) != len(self.samples):
            s += (f", epoch {len(self.index)} after {self.positive_multiplier}x pos "
                  f"/ {self.sister_multiplier}x sister oversampling")
        return s

    def _volume(self, stem):
        if stem not in self._vol:
            self._vol[stem] = zarr.open_group(
                str(self.data_dir / f"{stem}.zarr"), mode="r")["0"]
        return self._vol[stem]

    def _slab(self, vol, t, z):
        """`patch_z` slices of frame `t` centred on `z`, as (pz, Y, X) float32."""
        pz = self.patch_z
        z0 = int(round(z)) - pz // 2
        lo, hi = max(0, z0), min(vol.shape[1], z0 + pz)
        slab = np.zeros((pz, vol.shape[2], vol.shape[3]), np.float32)
        if lo < hi:
            slab[lo - z0: hi - z0] = np.asarray(vol[t, lo:hi], np.float32)
        return slab

    def __getitem__(self, i):
        stem, t, a_zyx, b_zyx, label, kind = self.samples[int(self.index[i])]
        vol = self._volume(stem)
        n_t, n_z, n_y, n_x = vol.shape

        angle, centre, feats = pair_geometry(a_zyx, b_zyx)

        rng = np.random.default_rng((self.seed, i, int(torch.initial_seed()) % (1 << 31)))
        geom = {"angle": 0.0, "scale": 1.0, "elastic": None,
                "photo": {"brightness": 0.0, "contrast": 1.0, "gamma": 1.0, "noise": 0.0}}
        zc, cy, cx = float(centre[0]), float(centre[1]), float(centre[2])
        draw = None
        if self.augment:
            geom = sample_geometry(self.aug_cfg, rng, SCALE, (n_y, n_x))
            jit = float(self.aug_cfg.get("jitter_um", 0.0))
            if jit:
                cy += float(rng.normal(0, jit / SCALE[1]))
                cx += float(rng.normal(0, jit / SCALE[2]))
            draw = sample_clip_aug(self.div_aug, rng, self.n_frames, SCALE[1:])
            zc += draw["z"]

        # The canonical rotation puts the pair on +x; any augmentation angle is
        # added on top, which is only non-zero if someone deliberately turned
        # rotation back on to ablate the canonicalisation.
        angle_total = angle + float(geom["angle"])

        half = self.n_frames // 2
        clip = np.zeros((self.n_frames, self.patch_z, self.patch_yx, self.patch_yx),
                        np.float32)
        for k, dt in enumerate(range(-half, half + 1)):
            tt = t + dt
            if tt < 0 or tt >= n_t:
                continue            # off the ends of the video: genuinely nothing
            slab = self._slab(vol, tt, zc)
            lo, hi = np.percentile(slab, [1.0, 99.99])
            slab = np.clip((slab - lo) / (hi - lo + 1e-6), 0.0, 1.0)

            dy, dx = (draw["shift"][k] if draw is not None else (0.0, 0.0))
            clip[k] = extract_patches(
                slab, np.array([[self.patch_z // 2, cy + dy, cx + dx]], np.float32),
                patch_zyx=(self.patch_z, self.patch_yx, self.patch_yx),
                angle_deg=angle_total, scale=geom["scale"],
                elastic=geom["elastic"], n_frames=1,
            )[0, 0]
            if self.augment:
                clip[k] = apply_photometric(clip[k], geom["photo"], rng)

        label = float(label)
        if draw is not None:
            # `time_reverse` and `drop_context` are division-only label surgery
            # and default to 0 here; the label comes back unchanged.
            clip, label = apply_clip_aug(clip, draw, self.div_aug, rng, label)

        return {
            "clip": torch.from_numpy(clip),            # (T, pz, H, W)
            "geom": torch.from_numpy(feats),           # (3,) um
            "label": torch.tensor(float(label)),
            "is_sister": torch.tensor(float(kind == "sister")),
            "sample": stem,
            "t": int(t),
        }
