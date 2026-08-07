import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from .targets import compute_all_targets


class CellSegDataset(Dataset):
    """
    Loads npz files produced by the cellpose preprocessing pipeline.

    Each npz must contain:
        img     : (D, H, W) uint16
        masks   : (D, H, W) uint16  instance labels
        probas  : (D, H, W) float32 cellpose foreground logits

    Training targets are derived on-the-fly:
        fg       : sigmoid(probas) — foreground probability in [0, 1]
        heatmap  : Gaussian blobs at mask-derived cell centers
        offsets  : per-voxel normalized offset to nearest cell centroid
        fg_mask  : bool mask where masks > 0 (for offset loss)
    """

    def __init__(self, data_dir, sigma=3.0, augment=True, patch_size=None, use_cellpose=False):
        self.files = sorted(Path(data_dir).glob('*.npz'))
        if not self.files:
            raise FileNotFoundError(f"No *.npz files found in {data_dir}")
        self.sigma = sigma
        self.augment = augment
        self.patch_size = patch_size  # (pd, ph, pw) or None for full volume
        self.use_cellpose = use_cellpose

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        img = data['img'].astype(np.float32) / 65535.0
        masks = data['masks']
        
        if self.use_cellpose:
            probas = data['probas']
            flows = data['flows'].transpose(3, 0, 1, 2)
            flows = (flows / 255).astype(np.float32)

        fg_target = (masks > 0).astype(np.float32)

        heatmap, offsets, _ = compute_all_targets(
            masks, sigma=self.sigma,
        )
        fg_mask = (masks > 0)

        if self.use_cellpose:
            img = np.stack([img, probas], axis=0)
        else:
            img = img[None]

        if self.patch_size is not None:
            img, fg_target, heatmap, offsets, fg_mask = self._random_crop(
                img, fg_target, heatmap, offsets, fg_mask
            )

        if self.augment:
            img, fg_target, heatmap, offsets, fg_mask = self._augment(
                img, fg_target, heatmap, offsets, fg_mask
            )

        return {
            'image': torch.from_numpy(img),                 # (1, D, H, W)
            'fg': torch.from_numpy(fg_target),              # (D, H, W)
            'heatmap': torch.from_numpy(heatmap),           # (D, H, W)
            'offsets': torch.from_numpy(offsets),           # (3, D, H, W)
            'fg_mask': torch.from_numpy(fg_mask),           # (D, H, W) bool
        }

    def _random_crop(self, img, fg, heatmap, offsets, fg_mask):
        D, H, W = img.shape[-3:]
        pd, ph, pw = self.patch_size
        z0 = np.random.randint(0, max(1, D - pd))
        y0 = np.random.randint(0, max(1, H - ph))
        x0 = np.random.randint(0, max(1, W - pw))
        s = np.s_[z0:z0+pd, y0:y0+ph, x0:x0+pw]
        return (
            img[(slice(None),) + s], fg[s], heatmap[s],
            offsets[:, z0:z0+pd, y0:y0+ph, x0:x0+pw],
            fg_mask[s],
        )

    def _augment(self, img, fg, heatmap, offsets, fg_mask):
        # Random flips along each spatial axis
        for axis in range(3):
            if np.random.random() > 0.5:
                img = np.flip(img, axis=axis + 1).copy()
                fg = np.flip(fg, axis=axis).copy()
                heatmap = np.flip(heatmap, axis=axis).copy()
                offsets = np.flip(offsets, axis=axis + 1).copy()
                fg_mask = np.flip(fg_mask, axis=axis).copy()
                # Flipping spatial axis flips that offset component's sign
                offsets[axis] = -offsets[axis]

        # Intensity jitter: multiplicative scale + additive shift
        scale = 1.0 + 0.15 * np.random.randn()
        shift = 0.05 * np.random.randn()
        img = np.clip(img * scale + shift, 0.0, 1.0)

        return img, fg, heatmap, offsets, fg_mask
