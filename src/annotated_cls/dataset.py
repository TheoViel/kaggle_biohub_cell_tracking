"""Crops around detections, labelled "the annotation covers this one, tracked right".

One sample is a **single detection at one frame**: a crop centred on it, cut as
`patch_z` z-slices so a 2D backbone can read them as channels, labelled by
`labels.node_labels`. `n_frames > 1` widens it to a short clip in time, which is
off by default -- the question is about the cell, not about a change in it.

The negatives are the whole difficulty. There are 214 of them per positive and
almost every one is a real cell, so the model is not being asked "is this a
cell" but "is this one of the cells the annotators followed". `neg_per_pos`
subsamples them to something trainable; the sampling is uniform over the stack
rather than nearest-neighbour to the positives, because the positives sit in one
region and drawing negatives from their neighbourhood would hand the model the
very shortcut the out-of-fold split exists to catch.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from annotated_cls.labels import ground_truth, load_graph, node_labels
from embeddings.augment import apply_photometric, resolve_aug_cfg, sample_geometry
from embeddings.patches import extract_patches
from params import SCALE
from tracking_cellmot.io import list_datasets

warnings.filterwarnings("ignore", category=UserWarning)


class AnnotatedCellDataset(Dataset):
    """Detections of a set of embryos, each a crop and a 0/1 label.

    Parameters
    ----------
    data_dir : str
        Folder of `<sample>.zarr` and `<sample>.geff`.
    preds_dir : str
        Folder of post-processed `<sample>.npz` -- the graph whose nodes are
        being judged. This is deliberately the *final* output rather than the
        raw detections: the label is "carries a correct link", which only exists
        once something has linked them.
    neg_per_pos : float
        Negatives kept per positive. ``0`` keeps all of them, which is 214:1 and
        not trainable in any reasonable epoch.
    """

    def __init__(
        self,
        data_dir,
        preds_dir,
        embryos=None,
        patch_yx: int = 32,
        patch_z: int = 9,
        n_frames: int = 1,
        neg_per_pos: float = 30.0,
        positive_multiplier: int = 1,
        augment: bool = False,
        aug_cfg: dict | None = None,
        max_stacks: int | None = None,
        seed: int = 0,
    ):
        if n_frames % 2 == 0:
            raise ValueError(f"n_frames must be odd, got {n_frames}")
        self.data_dir = Path(data_dir)
        self.preds_dir = Path(preds_dir)
        self.patch_yx, self.patch_z, self.n_frames = int(patch_yx), int(patch_z), int(n_frames)
        self.augment = bool(augment)
        self.aug_cfg = resolve_aug_cfg(aug_cfg)
        self.positive_multiplier = int(positive_multiplier)
        self.seed = int(seed)
        self._vol: dict = {}

        paths = [p for p in list_datasets(self.data_dir, require_geff=True)
                 if embryos is None or any(p.name.startswith(e) for e in embryos)]
        paths = [p for p in paths if (self.preds_dir / f"{p.name}.npz").exists()]
        if max_stacks:
            paths = paths[:max_stacks]

        rng = np.random.default_rng(self.seed)
        rows, self.stacks = [], []
        for p in paths:
            coords, edges = load_graph(self.preds_dir, p.name)
            y, _ = node_labels(coords, edges, ground_truth(p.name, self.data_dir))
            pos = np.flatnonzero(y)
            neg = np.flatnonzero(~y)
            if neg_per_pos and len(pos):
                keep = min(len(neg), int(round(neg_per_pos * len(pos))))
                neg = rng.choice(neg, size=keep, replace=False)
            elif not len(pos):
                # A stack with no positive is all negatives and teaches the
                # model only the prior; keep a token few so the embryo is
                # represented without swamping the epoch.
                neg = rng.choice(neg, size=min(len(neg), 64), replace=False)
            for r in np.concatenate([pos, neg]) if len(pos) else neg:
                t, z, yy, xx = coords[int(r)]
                rows.append((p.name, int(t), float(z), float(yy), float(xx),
                             int(y[int(r)])))
            self.stacks.append(p.name)

        self.samples = rows
        self.labels = np.array([r[5] for r in rows], np.int64)
        index = np.arange(len(rows))
        if self.positive_multiplier > 1:
            extra = np.repeat(np.flatnonzero(self.labels == 1),
                              self.positive_multiplier - 1)
            index = np.concatenate([index, extra])
        self.index = index

    @property
    def summary(self):
        n_pos = int((self.labels == 1).sum())
        n_neg = int((self.labels == 0).sum())
        return (f"{len(self.stacks)} stacks, {n_pos} positive, {n_neg} negative "
                f"(1:{n_neg / max(n_pos, 1):.0f})"
                + (f", epoch {len(self.index)} after {self.positive_multiplier}x"
                   if len(self.index) != len(self.samples) else ""))

    def _volume(self, stem):
        if stem not in self._vol:
            self._vol[stem] = zarr.open_group(
                str(self.data_dir / f"{stem}.zarr"), mode="r")["0"]
        return self._vol[stem]

    def _slab(self, vol, t, z):
        """`patch_z` slices of frame `t` centred on `z`, as (pz, Y, X)."""
        pz = self.patch_z
        z0 = int(round(z)) - pz // 2
        lo, hi = max(0, z0), min(vol.shape[1], z0 + pz)
        slab = np.zeros((pz, vol.shape[2], vol.shape[3]), np.float32)
        if lo < hi:
            slab[lo - z0: hi - z0] = np.asarray(vol[t, lo:hi], np.float32)
        return slab

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        stem, t, z, y, x, label = self.samples[int(self.index[i])]
        vol = self._volume(stem)
        n_t, _, n_y, n_x = vol.shape

        rng = np.random.default_rng((self.seed, i, int(torch.initial_seed()) % (1 << 31)))
        geom = {"angle": 0.0, "scale": 1.0, "elastic": None,
                "photo": {"brightness": 0.0, "contrast": 1.0, "gamma": 1.0, "noise": 0.0}}
        cy, cx = float(y), float(x)
        if self.augment:
            geom = sample_geometry(self.aug_cfg, rng, SCALE, (n_y, n_x))
            jit = float(self.aug_cfg.get("jitter_um", 0.0))
            if jit:
                cy += float(rng.normal(0, jit / SCALE[1]))
                cx += float(rng.normal(0, jit / SCALE[2]))

        half = self.n_frames // 2
        clip = np.zeros((self.n_frames, self.patch_z, self.patch_yx, self.patch_yx),
                        np.float32)
        for k, dt in enumerate(range(-half, half + 1)):
            tt = int(t) + dt
            if tt < 0 or tt >= n_t:
                continue
            slab = self._slab(vol, tt, z)
            # Per-frame percentiles, as everywhere else here: a per-crop range
            # would erase how bright this cell is beside its neighbours.
            lo, hi = np.percentile(slab, [1.0, 99.99])
            slab = np.clip((slab - lo) / (hi - lo + 1e-6), 0.0, 1.0)
            patch = extract_patches(
                slab, np.array([[self.patch_z // 2, cy, cx]], np.float32),
                patch_zyx=(self.patch_z, self.patch_yx, self.patch_yx),
                angle_deg=geom["angle"], scale=geom["scale"],
                elastic=geom["elastic"], n_frames=1,
            )[0, 0]
            if self.augment:
                patch = apply_photometric(patch, geom["photo"], rng)
            clip[k] = patch

        return {
            "clip": torch.from_numpy(clip),
            "target": torch.tensor(float(label), dtype=torch.float32),
        }
