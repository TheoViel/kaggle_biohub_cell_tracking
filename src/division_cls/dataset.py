"""Crops around annotated cells, labelled "divides here" or not.

The unit is one **annotated cell at one frame**: a short stack of crops centred
on it, spanning `n_frames` frames of time, labelled 1 when the ground truth has
that cell dividing at the centre frame. `notebooks/Divisions.ipynb` draws
exactly this picture, one panel per frame.

Two choices decide whether this is learnable at all, and both are about the
negatives rather than the positives.

**Only stacks that contain a division.** 87 of the 199 do. The other 112 add
nothing but easy negatives and would triple the epoch for it.

**Negatives from the dividing cells' own lineages, by default.** A dividing
cell exists at every frame of its track and divides at exactly one of them, so
its other frames are the negatives that matter: same cell, same neighbourhood,
same brightness, different answer. That forces the model onto *when* rather than
*which*, and it collapses the imbalance from 1:880 (every annotated node) to
roughly 1:12. `negatives="stack"` widens it to any annotated cell in a stack
that has a division, `negatives="all"` to every stack.

The frames immediately around a division are the ambiguous ones -- a cell one
frame from dividing looks much like a cell dividing -- so `exclude_window`
drops them from the negatives instead of teaching the model to split a hair the
annotation does not actually resolve. They are dropped, not relabelled: the
annotation says the division is at `t`, and that is the only thing it says.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from embeddings.augment import apply_photometric, resolve_aug_cfg, sample_geometry
from embeddings.patches import extract_patches
from params import SCALE
from tracking_cellmot.io import list_datasets

warnings.filterwarnings("ignore", category=UserWarning)


def gt_lineage(geff_path):
    """`(nodes, divides, track_of)` for one annotated stack.

    `nodes` is (N, 4) as t/z/y/x, `divides` a bool over them, and `track_of` a
    connected-component id over the *undirected* lineage -- so a dividing cell
    and both its daughters share one id, which is what "the trajectories that
    have divisions" has to mean once a track forks.
    """
    import tracksdata as td

    graph = td.graph.IndexedRXGraph.from_geff(Path(geff_path))
    graph = graph[0] if isinstance(graph, tuple) else graph

    rows = list(graph.node_attrs().iter_rows(named=True))
    index = {int(r["node_id"]): i for i, r in enumerate(rows)}
    nodes = np.array([[r["t"], r["z"], r["y"], r["x"]] for r in rows], np.float32)

    parent = np.arange(len(rows))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    n_out = np.zeros(len(rows), np.int32)
    for e in graph.edge_attrs().iter_rows(named=True):
        s, t = index[int(e["source_id"])], index[int(e["target_id"])]
        n_out[s] += 1
        ra, rb = find(s), find(t)
        if ra != rb:
            parent[ra] = rb

    track_of = np.array([find(i) for i in range(len(rows))], np.int64)
    return nodes, n_out >= 2, track_of


class DivisionDataset(Dataset):
    """Labelled crops for the division classifier.

    Parameters
    ----------
    data_dir : str
        Folder of `<sample>.zarr` and `<sample>.geff`.
    embryos : list of str
        Prefixes to keep. This is the split, and it is by embryo for the same
        reason everything else in this repo is: two of them exist, and a random
        split would put frames of the same cell on both sides.
    patch_yx : int
        Crop side in voxels. 32 at 0.40625 um/voxel is 13 um, so a cell (~8.5
        um) and the sliver of neighbourhood that says whether it just split.
    patch_z : int
        z slices per frame, taken as the backbone's input channels.
    n_frames : int
        Frames of *time*, centred on the labelled one. Must be odd.
    negatives : {"track", "stack", "all"}
        Where the label-0 crops come from; see the module docstring.
    neg_per_pos : float
        Cap on negatives, as a multiple of the positives in the same stack.
        `0` or None keeps every candidate.
    exclude_window : int
        Frames either side of a division that are dropped from the negatives.
    positive_multiplier : int
        Times each positive is repeated in the epoch. Oversampling rather than
        loss weighting so that the positives are seen under *different*
        augmentation draws, which is the point of having augmentation at all.
    """

    def __init__(
        self,
        data_dir,
        embryos=None,
        patch_yx: int = 32,
        patch_z: int = 3,
        n_frames: int = 3,
        negatives: str = "track",
        neg_per_pos: float = 12.0,
        exclude_window: int = 1,
        positive_multiplier: int = 1,
        augment: bool = False,
        aug_cfg: dict | None = None,
        max_stacks: int | None = None,
        seed: int = 0,
    ):
        if n_frames % 2 == 0:
            raise ValueError(f"n_frames must be odd to centre on the division, got {n_frames}")
        if negatives not in ("track", "stack", "all"):
            raise ValueError(f"unknown negatives {negatives!r}")

        self.data_dir = Path(data_dir)
        self.patch_yx = int(patch_yx)
        self.patch_z = int(patch_z)
        self.n_frames = int(n_frames)
        self.augment = bool(augment)
        self.aug_cfg = resolve_aug_cfg(aug_cfg)
        self.seed = int(seed)
        self._vol = {}

        paths = [
            p for p in list_datasets(self.data_dir, require_geff=True)
            if embryos is None or any(p.name.startswith(e) for e in embryos)
        ]

        rng = np.random.default_rng(seed)
        self.samples = []            # (stem, t, z, y, x, label)
        self.stacks = []
        for path in paths:
            nodes, divides, track_of = gt_lineage(self.data_dir / f"{path.name}.geff")
            n_div = int(divides.sum())
            if n_div == 0 and negatives != "all":
                continue

            if negatives == "track":
                div_tracks = set(track_of[divides].tolist())
                pool = np.array([i for i in range(len(nodes))
                                 if track_of[i] in div_tracks], np.int64)
            else:
                pool = np.arange(len(nodes), dtype=np.int64)

            # Drop the frames flanking a division from the negatives: they are
            # the cases the annotation does not resolve, not free labels.
            near = np.zeros(len(nodes), bool)
            if exclude_window > 0 and n_div:
                for tr, t in zip(track_of[divides], nodes[divides, 0]):
                    same = (track_of == tr) & (np.abs(nodes[:, 0] - t) <= exclude_window)
                    near |= same
            near &= ~divides

            neg = pool[~divides[pool] & ~near[pool]]
            if neg_per_pos and n_div:
                keep = min(len(neg), int(round(neg_per_pos * n_div)))
                neg = rng.choice(neg, size=keep, replace=False) if keep < len(neg) else neg

            for i in np.flatnonzero(divides):
                self.samples.append((path.name, *nodes[i], 1))
            for i in neg:
                self.samples.append((path.name, *nodes[i], 0))
            if n_div:
                self.stacks.append(path.name)
            if max_stacks and len(self.stacks) >= max_stacks:
                break

        self.labels = np.array([s[-1] for s in self.samples], np.int8)
        # Oversampling is an index list, so `samples` stays the honest set and
        # anything counting positives is not fooled by the multiplier.
        self.index = np.arange(len(self.samples))
        mult = max(1, int(positive_multiplier))
        if mult > 1 and augment:
            extra = np.repeat(np.flatnonzero(self.labels == 1), mult - 1)
            self.index = np.concatenate([self.index, extra])
        self.positive_multiplier = mult

    def __len__(self):
        return len(self.index)

    @property
    def summary(self) -> str:
        n_pos = int((self.labels == 1).sum())
        n_neg = int((self.labels == 0).sum())
        return (f"{len(self.stacks)} stacks, {n_pos} divisions, {n_neg} negatives "
                f"(1:{n_neg / max(n_pos, 1):.0f})"
                + (f", epoch {len(self.index)} after {self.positive_multiplier}x "
                   "oversampling" if len(self.index) != len(self.samples) else ""))

    def _volume(self, stem):
        if stem not in self._vol:
            self._vol[stem] = zarr.open_group(
                str(self.data_dir / f"{stem}.zarr"), mode="r")["0"]
        return self._vol[stem]

    def _slab(self, vol, t, z):
        """`patch_z` slices of frame `t` centred on `z`, as (pz, Y, X) float32.

        Read as slices rather than whole frames: the crop needs three sections
        of a 64x256x256 volume, and normalisation only needs their statistics.
        """
        pz = self.patch_z
        z0 = int(round(z)) - pz // 2
        lo, hi = max(0, z0), min(vol.shape[1], z0 + pz)
        slab = np.zeros((pz, vol.shape[2], vol.shape[3]), np.float32)
        if lo < hi:
            slab[lo - z0: hi - z0] = np.asarray(vol[t, lo:hi], np.float32)
        return slab

    def __getitem__(self, i):
        stem, t, z, y, x, label = self.samples[int(self.index[i])]
        t = int(t)
        vol = self._volume(stem)
        n_t, n_z, n_y, n_x = vol.shape

        rng = np.random.default_rng((self.seed, i, int(torch.initial_seed()) % (1 << 31)))
        geom = {"angle": 0.0, "scale": 1.0, "elastic": None,
                "photo": {"brightness": 0.0, "contrast": 1.0, "gamma": 1.0, "noise": 0.0}}
        cy, cx = float(y), float(x)
        if self.augment:
            # One draw for the whole clip, not one per frame: the frames are the
            # same scene a moment apart, and re-rolling per frame would destroy
            # the change between them -- which is the only thing being asked
            # about.
            geom = sample_geometry(self.aug_cfg, rng, SCALE, (n_y, n_x))
            jit = float(self.aug_cfg.get("jitter_um", 0.0))
            if jit:
                cy += float(rng.normal(0, jit / SCALE[1]))
                cx += float(rng.normal(0, jit / SCALE[2]))

        half = self.n_frames // 2
        clip = np.zeros((self.n_frames, self.patch_z, self.patch_yx, self.patch_yx),
                        np.float32)
        for k, dt in enumerate(range(-half, half + 1)):
            tt = t + dt
            if tt < 0 or tt >= n_t:
                continue            # off the ends of the video: genuinely nothing
            slab = self._slab(vol, tt, z)
            # Percentiles over the slab, following the paper's per-frame rule;
            # a per-crop range would erase how bright this cell is next to its
            # neighbours, which is part of what marks a mitotic one.
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
            "clip": torch.from_numpy(clip),                       # (T, pz, H, W)
            "label": torch.tensor(float(label)),
            "sample": stem,
            "t": t,
        }
