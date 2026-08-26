from pathlib import Path

import zarr
import torch
import numpy as np
from torch.utils.data import Dataset

from data.transforms import (
    brightness_augment,
    heavy_brightness_augment,
    flip_augment,
    has_feasible_track_crop,
    random_crop_track,
)
from tracking.data import (
    VideoMeta,
    FrameWindowData,
    load_dataset_windows,
    _pad_window,
)
from tracking.weak_positives import compute_weak_positives


class TrackDataset(Dataset):
    """Fixed-size dataset of temporal tracking windows.

    Accepts a list of (VideoMeta, windows) tuples — one per source video.
    No image data is held in RAM; each __getitem__ opens the zarr file on demand.
    """

    def __init__(
        self,
        video_data: list[tuple[VideoMeta, list[FrameWindowData]]],
        max_nodes: int | None = None,
        augment: bool = True,
        crop_size: tuple[int, int, int] | None = None,
        downscale_factor: int = 1,
        brightness_strength: float = 0.0,
        weak_positives_cfg: dict | None = None,
        load_images: bool = True,
    ):
        all_windows = [w for _, windows in video_data for w in windows]
        if not all_windows:
            raise ValueError("No windows found — check data_dir and splits.")
        if max_nodes is None:
            max_nodes = max(max(w.node_counts) for w in all_windows)

        self.downscale_factor = downscale_factor
        self.load_images = load_images
        self.weak_positives_cfg = weak_positives_cfg
        self._weak_cache: dict[str, torch.Tensor] = {}
        # One npz per video, a few hundred KB each — cheaper to hold than to
        # re-read for every window of that video.
        self._pl_cache: dict[str, np.ndarray] = {}
        self.brightness_strength = brightness_strength
        self.max_nodes = max_nodes
        self.augment = augment
        self.crop_size = crop_size

        self._data: list[tuple[dict, VideoMeta]] = []
        for video_meta, windows in video_data:
            spatial = tuple(s // downscale_factor for s in video_meta.image_shape[1:])
            for window in windows:
                meta = _pad_window(window, max_nodes)
                if (
                    self.augment
                    and self.crop_size is not None
                    and not has_feasible_track_crop(
                        meta["coords"] / downscale_factor,
                        meta["masks"],
                        meta["targets"],
                        self.crop_size,
                        spatial,
                    )
                ):
                    continue
                self._data.append((meta, video_meta))
        if not self._data:
            raise ValueError(
                "No windows contain linked GT cells that fit inside the configured crop"
            )

    def __len__(self) -> int:
        return len(self._data)

    def _weak_to_coords(self, weak: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense weak-positive mask -> padded (W, N, 3) coords + (W, N) mask.

        Padding is -1, not 0: with no companion mask a zero row is
        indistinguishable from a real peak at the origin, and the loss would
        splat a weak positive there on every sample. gaussian_heatmap already
        rejects negative coordinates, so -1 rows drop out on their own.
        """
        W = weak.shape[0]
        N = self.weak_positives_cfg['max_nodes']
        coords = torch.full((W, N, 3), -1.0, dtype=torch.float32)
        mask = torch.zeros(W, N, dtype=torch.bool)
        for f in range(W):
            pts = torch.nonzero(weak[f])  # (n, 3)
            n = min(pts.shape[0], N)
            if n:
                coords[f, :n] = pts[:n].float()
                mask[f, :n] = True
        return coords, mask

    def _pl_positives(
        self,
        sample_name: str,                 # video stem, e.g. 44b6_0113de3b
        t_start: int,
        n_frames: int,
        gt_coords: torch.Tensor,          # (W, M, 3) downscaled voxels
        gt_masks: torch.Tensor,           # (W, M) bool
        voxel_size: tuple[float, float, float],
    ) -> torch.Tensor:                    # (W, N, 3), -1 padded
        """Weak positives read from a pseudo-label run instead of the image.

        The predictions are a tracker's own detections, so they cover the cells
        the annotation misses — which is exactly what the weak-positive term is
        for. Anything within ``pl_min_gt_dist_um`` of a GT point is dropped: a
        prediction there is that same cell, already supervised at full weight,
        and keeping it would fight the GT target with a down-weighted duplicate.
        """
        folder = Path(self.weak_positives_cfg["pl_folder"])
        coords = self._pl_cache.get(sample_name)
        if coords is None:
            path = folder / f"{sample_name}.npz"
            if not path.exists():
                raise FileNotFoundError(f"No pseudo-labels for {sample_name} at {path}")
            # (N, 4) [t, z, y, x] at original resolution, as predict.py writes.
            coords = np.load(path)["coords"].astype(np.float32)
            self._pl_cache[sample_name] = coords

        radius = self.weak_positives_cfg.get("pl_min_gt_dist_um", 7.0)
        n_max = self.weak_positives_cfg["max_nodes"]
        vs = torch.tensor(voxel_size, dtype=torch.float32)

        out = torch.full((n_frames, n_max, 3), -1.0, dtype=torch.float32)
        for f in range(n_frames):
            pts = coords[coords[:, 0] == t_start + f][:, 1:]
            if not len(pts):
                continue
            pts = torch.from_numpy(pts) / self.downscale_factor

            gt = gt_coords[f][gt_masks[f]]
            if len(gt):
                d = torch.cdist(pts * vs, gt * vs)
                pts = pts[d.min(dim=1).values >= radius]

            n = min(len(pts), n_max)
            if n:
                out[f, :n] = pts[:n]
        return out

    def _get_weak_positives(
        self,
        imgs: torch.Tensor,
        coords: torch.Tensor,
        masks: torch.Tensor,
        spatial: tuple[int, int, int],
        voxel_size: tuple[float, float, float],
        sample_name: str,
        frame: int,
    ) -> torch.Tensor:
        """(W, N, 3) -1-padded weak-positive coords for this window.

        Computing the peaks costs ~180 ms/frame, so each window is done once
        and cached.
        """
        name = f"{sample_name}_t{frame}"
        cached = self._weak_cache.get(name)
        if cached is None:
            weak_positives = compute_weak_positives(
                imgs,
                coords,
                masks,
                voxel_size,
                nms_um=self.weak_positives_cfg["nms_um"],
                threshold=self.weak_positives_cfg["threshold"],
            )
            weak_coords, _ = self._weak_to_coords(weak_positives)
            # Cache the coordinates, not the dense mask: the mask is ~8 MB per
            # window at full resolution, so caching it across ~2e4 window keys
            # would need >100 GB. Trim the padding too — a frame holds ~1e2
            # peaks against max_weak_nodes of ~2e3, so keeping the full pad
            # width would cost 20x more for nothing.
            n_max = int((weak_coords >= 0).all(dim=-1).sum(dim=1).max())
            cached = weak_coords[:, :n_max].clone()
            self._weak_cache[name] = cached

        # Re-pad to the fixed width so samples still collate.
        W, n = cached.shape[0], cached.shape[1]
        out = torch.full((W, self.weak_positives_cfg["max_nodes"], 3), -1.0, dtype=torch.float32)
        out[:, :n] = cached
        return out

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Load metadata
        meta, vm = self._data[idx]
        t_start = meta["t_start"]
        W = meta["n_frames"]

        f = self.downscale_factor
        spatial = tuple(-(-s // f) for s in vm.image_shape[1:])  # ceil, as strides give

        if self.load_images:
            z_arr = zarr.open_group(str(vm.zarr_path), mode="r")["0"]
            if f > 1:
                raw = z_arr[t_start : t_start + W, ::f, ::f, ::f]
            else:
                raw = z_arr[t_start : t_start + W]

            imgs = torch.from_numpy(
                (raw.astype(np.float32) - vm.q_low) / (vm.q_high - vm.q_low + 1e-6)
            ).clamp(0.0, 4.0)  # (W, Z, Y, X)
        else:
            # Transformer-only training after the first epoch: the node features
            # are cached, so decompressing the volume is pure waste. The shape is
            # kept so image_shape and the collate stay valid.
            imgs = torch.zeros((W, 1, 1, 1))

        # Targets
        coords = meta["coords"].clone()  # (W, M, 3) in 1/4-scale coord space
        masks = meta["masks"]  # (W, M) bool
        targets = meta["targets"]

        if f > 1:
            coords /= f

        coord_spatial = tuple(imgs.shape[1:]) if self.load_images else spatial
        voxel_size = tuple(v * self.downscale_factor for v in vm.voxel_size)

        # Derived from the raw normalised image, before any augmentation.
        weak_coords = None
        if self.weak_positives_cfg.get("use", False):
            if self.weak_positives_cfg.get("pl_folder"):
                # A previous run's detections beat image peaks: they already
                # know what a cell looks like.
                weak_coords = self._pl_positives(
                    sample_name=vm.zarr_path.stem,
                    t_start=t_start,
                    n_frames=W,
                    gt_coords=coords,
                    gt_masks=masks,
                    voxel_size=voxel_size,
                )
            else:
                weak_coords = self._get_weak_positives(
                    imgs=imgs,
                    coords=coords,
                    masks=masks,
                    spatial=coord_spatial,
                    voxel_size=voxel_size,
                    sample_name=vm.zarr_path.name,
                    frame=t_start,
                )
        else:
            # -1, not 0: a zero row is a valid coordinate, and the loss would
            # splat a phantom weak positive at the origin of every sample.
            weak_coords = torch.full((W, 1, 3), -1.0, dtype=torch.float32)

        # print(weak_coords)

        if self.augment:
            rng = np.random.default_rng()
            if self.crop_size is not None:
                imgs, coords, weak_coords, masks, targets, origin = random_crop_track(
                    imgs,
                    coords,
                    weak_coords,
                    masks,
                    targets,
                    self.crop_size,
                    rng,
                )
                coord_spatial = self.crop_size

            if self.brightness_strength > 0:
                imgs = heavy_brightness_augment(
                    imgs,
                    rng,
                    strength=self.brightness_strength,
                )
            else:
                imgs = brightness_augment(imgs, rng)

            imgs, coords, weak_coords, masks = flip_augment(
                imgs, coords, weak_coords, masks, coord_spatial, rng
            )

        image_shape = (vm.image_shape[0], *coord_spatial)
        return {
            **meta,
            "coords": coords,
            "masks": masks,
            "targets": targets,
            "node_counts": masks.sum(dim=1),
            "imgs": imgs.half(),  # (W, Z, Y, X) full-scale float16
            "weak_positives": weak_coords,  # (W, N, 3) image-resolution voxels
            "image_shape": torch.tensor(image_shape, dtype=torch.long),
            "voxel_size": torch.tensor(voxel_size, dtype=torch.float32),
            "name": f'{vm.zarr_path.name}_t{t_start}',
        }
