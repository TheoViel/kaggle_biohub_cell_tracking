"""Synthetic division crops, pre-extracted, and the real/synthetic mixture.

Fold 1 trains on 44b6's **26** divisions and reaches AP 0.449 where fold 0, with
6bba's 125, reaches 0.705. The gap is labels, and the generator has thousands:
`input/synthetic/v5` runs about one division per 790 nodes against the
annotation's one per 880, over far more nodes.

Mixing `v5` in **hurt at every proportion tried** (0.577 -> 0.361 / 0.342 /
0.273 / 0.248). Its sister separation is calibrated -- median 11.8 um against
10.6 -- but `sequences._lobe` renders the parent as two visible kernels at the
frame the label names, and measuring the 151 real divisions says that is false:
a parent at its own division frame is indistinguishable from an ordinary cell,
and the signature is entirely at t+1. So `v5` teaches a rule real images do not
follow, which for a classifier whose whole premise is mitotic appearance is the
one kind of wrong input that cannot be outvoted by 26 real positives.

`data_gen/division_samples.py` is the replacement: purpose-built division clips
whose appearance is fitted to that measurement. Build its cache with
`python -m division_cls.gen_divisions`; it writes the same `{clips, labels}`
npz, so everything below works unchanged with either.

**Crops are extracted once, not read per sample.** A sequence is a 50 MB npz and
reading `volumes` decompresses all six frames, so serving a random sample from a
random sequence would cost that every time. `build_cache` walks each sequence
once and writes the clips to a single array; training then reads slices of it.
The cost is that frame-wide augmentation (rotation, the elastic field) cannot be
applied afterwards -- what remains is flips, 90-degree rotations and photometry,
which for an event with no canonical orientation is most of the value.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from embeddings.patches import extract_patches


def sequence_samples(path, exclude_window=1, neg_per_pos=12.0, rng=None):
    """`(rows, labels)` for one synthetic sequence, with the real dataset's rule.

    Positives are the generator's own `divisions`; negatives come from the same
    *lineages*, which is what makes the question "when does this cell divide"
    rather than "which cell looks odd". Synthetic tracks carry the parent's id
    through a division, so `track_id` already is the lineage.
    """
    rng = rng or np.random.default_rng(0)
    with np.load(path) as z:
        nodes, edges = z["nodes"], z["edges"]
        divisions = z["divisions"]
    if not len(divisions):
        return np.zeros(0, int), np.zeros(0, np.int8)

    divides = np.zeros(len(nodes), bool)
    divides[np.asarray(divisions, int)] = True
    track = nodes[:, 4].astype(np.int64)
    div_tracks = set(track[divides].tolist())
    pool = np.array([i for i in range(len(nodes)) if track[i] in div_tracks], np.int64)

    # The frames flanking a division are the ones the label does not resolve.
    near = np.zeros(len(nodes), bool)
    if exclude_window > 0:
        for tr, t in zip(track[divides], nodes[divides, 0]):
            near |= (track == tr) & (np.abs(nodes[:, 0] - t) <= exclude_window)
    near &= ~divides

    pos = np.flatnonzero(divides)
    neg = pool[~divides[pool] & ~near[pool]]
    keep = min(len(neg), int(round(neg_per_pos * len(pos))))
    if keep < len(neg):
        neg = rng.choice(neg, size=keep, replace=False)
    rows = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos), np.int8), np.zeros(len(neg), np.int8)])
    return rows, labels


def build_cache(synthetic_dir, out_path, patch_yx=32, patch_z=3, n_frames=3,
                exclude_window=1, neg_per_pos=12.0, max_sequences=None, seed=0,
                verbose=True):
    """Extract every sequence's division crops into one array on disk."""
    files = sorted(Path(synthetic_dir).glob("sequences/*.npz"))[:max_sequences]
    if not files:
        raise FileNotFoundError(f"no sequences under {synthetic_dir}")
    half = n_frames // 2
    rng = np.random.default_rng(seed)

    clips, labels = [], []
    for n, path in enumerate(files):
        rows, lab = sequence_samples(path, exclude_window, neg_per_pos, rng)
        if not len(rows):
            continue
        with np.load(path) as z:
            nodes, vols = z["nodes"], z["volumes"]
            # Percentiles per frame, as the real loader does; a per-crop range
            # would erase how bright a cell is next to its neighbours.
            norm = []
            for t in range(vols.shape[0]):
                v = vols[t].astype(np.float32)
                lo, hi = np.percentile(v, [1.0, 99.99])
                norm.append(np.clip((v - lo) / (hi - lo + 1e-6), 0.0, 1.0))

        for row in rows:
            t = int(nodes[row, 0])
            zc, y, x = (float(q) for q in nodes[row, 1:4])
            clip = np.zeros((n_frames, patch_z, patch_yx, patch_yx), np.float32)
            for k, dt in enumerate(range(-half, half + 1)):
                tt = t + dt
                if tt < 0 or tt >= len(norm):
                    continue
                v = norm[tt]
                z0 = int(round(zc)) - patch_z // 2
                lo, hi = max(0, z0), min(v.shape[0], z0 + patch_z)
                slab = np.zeros((patch_z, v.shape[1], v.shape[2]), np.float32)
                if lo < hi:
                    slab[lo - z0:hi - z0] = v[lo:hi]
                clip[k] = extract_patches(
                    slab, np.array([[patch_z // 2, y, x]], np.float32),
                    patch_zyx=(patch_z, patch_yx, patch_yx), n_frames=1)[0, 0]
            clips.append(clip)
        labels.append(lab)
        if verbose and (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(files)} sequences, {len(clips)} crops", flush=True)

    X = np.asarray(clips, np.float32)
    y = np.concatenate(labels) if labels else np.zeros(0, np.int8)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, clips=X, labels=y)
    if verbose:
        print(f"-> {len(X)} crops ({int(y.sum())} divisions, "
              f"1:{len(y) / max(int(y.sum()), 1):.0f}) to {out_path}")
    return out_path


class SyntheticClipDataset(Dataset):
    """The cached crops, with the augmentation a pre-extracted crop still allows."""

    def __init__(self, cache_path, augment=True, positive_multiplier=1, seed=0):
        with np.load(cache_path) as d:
            self.clips, self.labels = d["clips"], d["labels"].astype(np.int8)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.index = np.arange(len(self.labels))
        mult = max(1, int(positive_multiplier))
        if mult > 1 and augment:
            extra = np.repeat(np.flatnonzero(self.labels == 1), mult - 1)
            self.index = np.concatenate([self.index, extra])

    def __len__(self):
        return len(self.index)

    @property
    def summary(self):
        n_pos = int((self.labels == 1).sum())
        return (f"{len(self.labels)} synthetic crops, {n_pos} divisions "
                f"(1:{(len(self.labels) - n_pos) / max(n_pos, 1):.0f})")

    def __getitem__(self, i):
        j = int(self.index[i])
        clip = self.clips[j].copy()
        if self.augment:
            rng = np.random.default_rng((self.seed, i, int(torch.initial_seed()) % (1 << 31)))
            # A division has no canonical orientation, so the eight flips and
            # quarter-turns of the y/x plane are all label-preserving.
            if rng.random() < 0.5:
                clip = clip[:, :, ::-1]
            if rng.random() < 0.5:
                clip = clip[:, :, :, ::-1]
            k = int(rng.integers(4))
            if k:
                clip = np.rot90(clip, k, axes=(2, 3))
            clip = np.ascontiguousarray(clip)
            clip = np.clip(clip * rng.normal(1.0, 0.12) + rng.normal(0.0, 0.06), 0.0, None)
        return {"clip": torch.from_numpy(np.ascontiguousarray(clip)),
                "label": torch.tensor(float(self.labels[j])),
                "sample": "synthetic", "t": 0}


class MixedDivisionDataset(Dataset):
    """Real crops, a `proportion` of them replaced by synthetic ones.

    Length and index space are the real dataset's, so DistributedSampler and the
    rest keep working unchanged, and the draw is per access -- the proportion
    holds in expectation over an epoch rather than exactly within it. Same
    construction as `tracking.synthetic.MixedDataset`, for the same reason.
    """

    def __init__(self, real, synthetic, proportion: float, seed: int = 0):
        if not 0.0 <= proportion <= 1.0:
            raise ValueError(f"proportion must be in [0, 1], got {proportion}")
        self.real, self.synthetic = real, synthetic
        self.proportion = float(proportion)
        self.seed = int(seed)

    def __len__(self):
        return len(self.real)

    @property
    def summary(self):
        return (f"{self.real.summary}  +  {self.proportion:.0%} of batches from "
                f"[{self.synthetic.summary}]")

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, int(torch.initial_seed()) % (1 << 31)))
        if self.proportion and rng.random() < self.proportion and len(self.synthetic):
            return self.synthetic[int(rng.integers(len(self.synthetic)))]
        return self.real[i]
