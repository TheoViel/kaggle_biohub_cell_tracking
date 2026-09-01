"""Training data for the 3D Cellpose model: the synthetic static volumes.

Only the *static* volumes carry instance masks -- ``generate_sequence`` saves
nodes and edges but no per-voxel labels -- so they are what a flow model can be
trained on at all. There are few of them and each is large, so an epoch is
defined as a number of random crops rather than a pass over the files.

The one thing worth care here is that flows are a *vector* field. Flipping the
image and forgetting to negate the matching flow component trains the model
against itself: the picture says the centre is to the left and the target says
it is to the right. Geometric augmentation is therefore restricted to the
axis-aligned operations whose effect on a vector is exact -- flips, and
rotations in the y/x plane. An arbitrary affine would have to rotate the
vectors as well as resample them, and interpolating a rotated unit field
quietly shortens it wherever neighbouring vectors disagree.

Rotations stay inside y/x on purpose: a voxel is 1.625 um in z against 0.40625
in y/x, so turning z into y would claim the microscope sampled four times finer
than it did.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.transforms import (
    background_level_augment,
    blur_augment,
    brightness_augment,
    lowres_augment,
)
from segmentation.flows import FLOW_SCALE, masks_to_flows
from tracking.data import resolve_normalization
from tqdm import tqdm

VOXEL_UM = (1.625, 0.40625, 0.40625)

AUG_DEFAULTS = {
    "flip_p": 0.5,
    "rot90_p": 0.5,
    "blur_p": 0.2,
    "blur_sigma_um": 1.0,
    "lowres_p": 0.2,
    "lowres_max_factor": 2.0,
    "brightness_p": 0.3,
    "brightness_strength": 0.1,
    "bg_level_p": 0.3,
    "bg_level_range": (0.03, 0.30),
}


def flow_cache_path(volume_path: str | Path) -> Path:
    p = Path(volume_path)
    return p.with_name(p.stem + "_flows.npz")


def build_flow_cache(volume_path, mode="diffusion", overwrite=False,
                     device="auto") -> Path:
    """Compute and store the flow field for one static volume.

    Done once per volume rather than per crop: the diffusion is ``n_iter``
    passes over the whole volume and there is no point repeating it every
    epoch. Flows are stored as float16 -- they are unit vectors, so the range
    is known and the precision costs about a thousandth of a voxel in
    direction. ``savez`` rather than ``savez_compressed``: the field is
    high-entropy noise to a compressor, so zipping it buys a few percent and
    costs more than the diffusion did.
    """
    out = flow_cache_path(volume_path)
    if out.exists() and not overwrite:
        return out
    with np.load(volume_path) as d:
        masks = d["mask"].astype(np.int32)
    cellprob, flows = masks_to_flows(masks, mode=mode, device=device)
    np.savez(out, cellprob=cellprob.astype(np.uint8), flows=flows.astype(np.float16))
    return out


def _rot90_yx(img, cellprob, flows, k):
    """Rotate by ``k`` right angles in y/x, turning the flow vectors with it.

    ``torch.rot90`` sends the point (y, x) to (N-1-x, y), so a displacement
    (dy, dx) becomes (-dx, dy). Applying that k times is the whole correction.
    """
    k %= 4
    if k == 0:
        return img, cellprob, flows
    img = torch.rot90(img, k, dims=(-2, -1))
    cellprob = torch.rot90(cellprob, k, dims=(-2, -1))
    flows = torch.rot90(flows, k, dims=(-2, -1))
    for _ in range(k):
        fy, fx = flows[1].clone(), flows[2].clone()
        flows[1], flows[2] = -fx, fy
    return img, cellprob, flows


def _flip(img, cellprob, flows, axis):
    """Mirror along one axis; the flow component along it changes sign."""
    img = torch.flip(img, dims=(axis,))
    cellprob = torch.flip(cellprob, dims=(axis,))
    flows = torch.flip(flows, dims=(axis + 1,))
    flows[axis] = -flows[axis]
    return img, cellprob, flows


class SyntheticStaticDataset(Dataset):
    """Random crops of the synthetic static volumes, with Cellpose targets.

    Args:
        data_dir: dataset root, e.g. ``../input/synthetic/v3``. Reads
            ``static/vol_*.npz``.
        crop_size: (D, H, W) crop, or None for the whole volume.
        epoch_size: crops per epoch. With only a handful of volumes on disk,
            a pass over the files is far too short to be an epoch.
        flow_mode: passed to ``masks_to_flows`` when building the cache.
        normalization: the tracking pipeline's quantile normalisation, so the
            model sees the same intensity convention it will meet on real data.
    """

    def __init__(
        self,
        data_dir,
        crop_size=(32, 128, 128),
        epoch_size=2000,
        augment=True,
        flow_mode="diffusion",
        normalization: dict | None = None,
        aug_cfg: dict | None = None,
    ):
        root = Path(data_dir)
        # The flow caches live beside the volumes and match "vol_*.npz" too.
        self.files = [
            f for f in sorted((root / "static").glob("vol_*.npz"))
            if not f.stem.endswith("_flows")
        ]
        if not self.files:
            raise FileNotFoundError(f"no static/vol_*.npz under {root}")
        self.crop_size = tuple(crop_size) if crop_size else None
        self.epoch_size = int(epoch_size)
        self.augment = augment
        self.flow_mode = flow_mode
        self.aug = {**AUG_DEFAULTS, **(aug_cfg or {})}

        norm = resolve_normalization(normalization)
        self.low_q = float(norm["low_q"])
        self.high_q = float(norm["high_q"])
        self.clamp_max = float(norm["clamp_max"])

    def __len__(self) -> int:
        return self.epoch_size

    def prepare(self, overwrite=False, n_jobs=8, device="auto"):
        """Build every flow cache up front.

        On a GPU this runs in one process: the diffusion is already ~50x faster
        there than on a core, and eight workers would only queue up on the same
        device while each holding its own CUDA context. Processes are for the
        CPU path, where they do help.
        """
        todo = [f for f in self.files if overwrite or not flow_cache_path(f).exists()]
        if not todo:
            return []

        use_gpu = device.startswith("cuda") or (
            device == "auto" and torch.cuda.is_available()
        )
        if use_gpu:
            return [
                build_flow_cache(f, self.flow_mode, overwrite, device)
                for f in tqdm(todo, desc="flow cache (gpu)")
            ]

        from joblib import Parallel, delayed

        return Parallel(n_jobs=n_jobs)(
            delayed(build_flow_cache)(f, self.flow_mode, overwrite, "cpu")
            for f in tqdm(todo, desc="flow cache (cpu)")
        )

    def _load(self, idx):
        path = self.files[idx]
        cache = flow_cache_path(path)
        if not cache.exists():
            build_flow_cache(path, self.flow_mode)
        with np.load(path) as d:
            volume = d["volume"]
        with np.load(cache) as d:
            return volume, d["cellprob"], d["flows"]

    def __getitem__(self, _idx):
        rng = np.random.default_rng()
        volume, cellprob, flows = self._load(int(rng.integers(len(self.files))))

        if self.crop_size is not None:
            shape = volume.shape
            lo = [int(rng.integers(0, max(1, s - c + 1)))
                  for s, c in zip(shape, self.crop_size)]
            sl = tuple(slice(l, l + c) for l, c in zip(lo, self.crop_size))
            volume = volume[sl]
            cellprob = cellprob[sl]
            flows = flows[(slice(None),) + sl]

        raw = volume.astype(np.float32)
        # Quantiles over the crop, matching what the real loader does per video.
        lo_v = float(np.quantile(raw, self.low_q))
        hi_v = float(np.quantile(raw, self.high_q))
        img = torch.from_numpy((raw - lo_v) / (hi_v - lo_v + 1e-6)).clamp(0, self.clamp_max)
        cellprob = torch.from_numpy(np.ascontiguousarray(cellprob)).float()
        flows = torch.from_numpy(np.ascontiguousarray(flows)).float()

        if self.augment:
            img, cellprob, flows = self._augment(img, cellprob, flows, rng)

        return {
            "image": img[None],                     # (1, D, H, W)
            "cellprob": cellprob,                   # (D, H, W) in {0, 1}
            "flows": flows * FLOW_SCALE,            # (3, D, H, W)
        }

    def _augment(self, img, cellprob, flows, rng):
        a = self.aug
        for axis in range(3):
            if rng.random() < a["flip_p"]:
                img, cellprob, flows = _flip(img, cellprob, flows, axis)
        if rng.random() < a["rot90_p"]:
            img, cellprob, flows = _rot90_yx(
                img, cellprob, flows, int(rng.integers(1, 4))
            )

        # Photometric only from here: these never move a voxel, so the flow
        # field is untouched and stays consistent with the image.
        w = img[None]
        if rng.random() < a["blur_p"]:
            w = blur_augment(w, rng, VOXEL_UM, a["blur_sigma_um"])
        if rng.random() < a["lowres_p"]:
            w = lowres_augment(w, rng, a["lowres_max_factor"])
        if rng.random() < a["bg_level_p"]:
            w = background_level_augment(w, rng, a["bg_level_range"], self.clamp_max)
        if rng.random() < a["brightness_p"]:
            w = brightness_augment(w, rng, a["brightness_strength"])
        return w[0].clamp(0, self.clamp_max), cellprob, flows
