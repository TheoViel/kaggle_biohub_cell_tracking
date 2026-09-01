from pathlib import Path

import zarr
import torch
import numpy as np
from torch.utils.data import Dataset

from data.transforms import (
    AUG_DEFAULTS,
    augment_track,
    has_feasible_track_crop,
)
from tracking.data import (
    VideoMeta,
    FrameWindowData,
    load_dataset_windows,
    _pad_window,
)
from tracking.weak_positives import compute_weak_positives
from tracking.weak_positives_new import compute_weak_positives as compute_weak_positives_v2


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
        aug_cfg: dict | None = None,
        weak_positives_cfg: dict | None = None,
        load_images: bool = True,
        clamp_max: float = 4.0,
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
        self.aug_cfg = {**AUG_DEFAULTS, **(aug_cfg or {})}
        self.clamp_max = clamp_max
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

    def _load_pl(self, folder: str, sample_name: str) -> np.ndarray:
        """(N, 4) [t, z, y, x] detections at original resolution, cached."""
        key = f"{folder}::{sample_name}"
        coords = self._pl_cache.get(key)
        if coords is None:
            path = Path(folder) / f"{sample_name}.npz"
            if not path.exists():
                raise FileNotFoundError(f"No pseudo-labels for {sample_name} at {path}")
            coords = np.load(path)["coords"].astype(np.float32)
            self._pl_cache[key] = coords
        return coords

    def _pl_points(
        self,
        folder: str,
        sample_name: str,
        frames: list[int],                # absolute frame indices, stride applied
        exclude: list[torch.Tensor],      # per frame, (n, 3) points to stay clear of
        radius: float,
        n_max: int,
        voxel_size: tuple[float, float, float],
    ) -> torch.Tensor:                    # (W, N, 3), -1 padded
        """Points from a prediction folder, minus anything near ``exclude``.

        The predictions are a tracker's own detections, so they cover the cells
        the annotation misses — which is exactly what the weak-positive term is
        for. Anything within ``radius`` of an excluded point is dropped: for the
        weak set that is the GT, already supervised at full weight, and keeping
        a duplicate there would fight the GT target with a down-weighted copy.
        """
        coords = self._load_pl(folder, sample_name)
        vs = torch.tensor(voxel_size, dtype=torch.float32)

        out = torch.full((len(frames), n_max, 3), -1.0, dtype=torch.float32)
        for f, t in enumerate(frames):
            pts = coords[coords[:, 0] == t][:, 1:]
            if not len(pts):
                continue
            pts = torch.from_numpy(pts) / self.downscale_factor

            keep_out = exclude[f]
            if len(keep_out):
                d = torch.cdist(pts * vs, keep_out * vs)
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(W, N, 3) -1-padded (weak_coords, ignore_coords) for this window.

        ``ignore_coords`` is None under the "legacy" strategy, which derives
        weak positives only. Computing the peaks costs ~180 ms/frame, so each
        window is done once and cached.
        """
        cfg = self.weak_positives_cfg
        strategy = cfg.get("strategy", "legacy")
        # Strategy is part of the key: the two produce different points, so a
        # cache built under one must not be served to the other.
        name = f"{sample_name}_t{frame}_{strategy}"
        cached = self._weak_cache.get(name)
        if cached is None:
            if strategy == "v2":
                weak_positives, ignore_positives = compute_weak_positives_v2(
                    imgs,
                    coords,
                    masks,
                    voxel_size,
                    nms_um=cfg["nms_um"],
                    threshold=cfg["threshold"],
                    ignore_threshold=cfg.get("ignore_threshold", 0.0),
                    ignore_nms_um=cfg.get("ignore_nms_um"),
                    ignore_floor=cfg.get("ignore_floor", 0.0),
                )
            else:
                weak_positives = compute_weak_positives(
                    imgs,
                    coords,
                    masks,
                    voxel_size,
                    nms_um=cfg["nms_um"],
                    threshold=cfg["threshold"],
                )
                ignore_positives = None

            weak_coords, _ = self._weak_to_coords(weak_positives)
            # Cache the coordinates, not the dense mask: the mask is ~8 MB per
            # window at full resolution, so caching it across ~2e4 window keys
            # would need >100 GB. Trim the padding too — a frame holds ~1e2
            # peaks against max_weak_nodes of ~2e3, so keeping the full pad
            # width would cost 20x more for nothing.
            n_max = int((weak_coords >= 0).all(dim=-1).sum(dim=1).max())
            weak_cached = weak_coords[:, :n_max].clone()

            ignore_cached = None
            if ignore_positives is not None:
                ignore_coords, _ = self._weak_to_coords(ignore_positives)
                n_max = int((ignore_coords >= 0).all(dim=-1).sum(dim=1).max())
                ignore_cached = ignore_coords[:, :n_max].clone()

            cached = (weak_cached, ignore_cached)
            self._weak_cache[name] = cached

        weak_cached, ignore_cached = cached

        def _repad(pts, width):
            """Back to the fixed width so samples still collate."""
            out = torch.full((pts.shape[0], width, 3), -1.0, dtype=torch.float32)
            n = min(pts.shape[1], width)
            out[:, :n] = pts[:, :n]
            return out

        weak_out = _repad(weak_cached, cfg["max_nodes"])
        if ignore_cached is None:
            return weak_out, None
        return weak_out, _repad(ignore_cached, cfg.get("ignore_max_nodes", cfg["max_nodes"]))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Load metadata
        meta, vm = self._data[idx]
        t_start = meta["t_start"]
        W = meta["n_frames"]

        f = self.downscale_factor
        spatial = tuple(-(-s // f) for s in vm.image_shape[1:])  # ceil, as strides give

        stride = int(meta.get("stride", 1))
        if self.load_images:
            z_arr = zarr.open_group(str(vm.zarr_path), mode="r")["0"]
            t_slice = slice(t_start, t_start + W * stride, stride)
            if f > 1:
                raw = z_arr[t_slice, ::f, ::f, ::f]
            else:
                raw = z_arr[t_slice]

            imgs = torch.from_numpy(
                (raw.astype(np.float32) - vm.q_low) / (vm.q_high - vm.q_low + 1e-6)
            ).clamp(0.0, self.clamp_max)  # (W, Z, Y, X)
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

        frames = [t_start + i * stride for i in range(W)]
        gt_per_frame = [coords[i][masks[i]] for i in range(W)]
        wp_cfg = self.weak_positives_cfg
        # -1, not 0: a zero row is a valid coordinate, and the loss would splat
        # a phantom point at the origin of every sample.
        empty = torch.full((W, 1, 3), -1.0, dtype=torch.float32)

        # Derived from the raw normalised image, before any augmentation.
        weak_coords = empty
        ignore_coords = empty
        if wp_cfg.get("use", False):
            radius = wp_cfg.get("pl_min_gt_dist_um", 7.0)
            n_max = wp_cfg["max_nodes"]
            if wp_cfg.get("pl_folder"):
                # A previous run's detections beat image peaks: they already
                # know what a cell looks like.
                weak_coords = self._pl_points(
                    wp_cfg["pl_folder"], vm.zarr_path.stem, frames,
                    exclude=gt_per_frame, radius=radius, n_max=n_max,
                    voxel_size=voxel_size,
                )
                if wp_cfg.get("pl_ignore_folder"):
                    # A lower-threshold sweep of the same run: too weak to
                    # supervise as a cell, too suspicious to call background.
                    # Anything already covered by GT or the weak set is not an
                    # ignore region — it is a positive.
                    covered = [
                        torch.cat([gt, w[(w >= 0).all(dim=-1)]])
                        for gt, w in zip(gt_per_frame, weak_coords)
                    ]
                    ignore_coords = self._pl_points(
                        wp_cfg["pl_ignore_folder"], vm.zarr_path.stem, frames,
                        exclude=covered,
                        radius=wp_cfg.get("ignore_min_dist_um", radius),
                        n_max=wp_cfg.get("ignore_max_nodes", n_max),
                        voxel_size=voxel_size,
                    )
            elif self.load_images:
                # Image-derived peaks need the image. With load_images off
                # (transformer-only, features cached) ``imgs`` is a (W,1,1,1)
                # placeholder, and detection is frozen anyway — det_logits is
                # empty, so the loss never reads these.
                weak_coords, derived_ignore = self._get_weak_positives(
                    imgs=imgs,
                    coords=coords,
                    masks=masks,
                    spatial=coord_spatial,
                    voxel_size=voxel_size,
                    sample_name=vm.zarr_path.name,
                    frame=t_start,
                )
                # None under the legacy strategy, which has no ignore set.
                if derived_ignore is not None:
                    ignore_coords = derived_ignore

        point_sets = [weak_coords, ignore_coords]
        if self.augment:
            rng = np.random.default_rng()
            imgs, coords, point_sets, masks, targets, coord_spatial = augment_track(
                imgs,
                coords,
                point_sets,
                masks,
                targets,
                coord_spatial=coord_spatial,
                crop_size=self.crop_size,
                voxel_size=voxel_size,
                cfg=self.aug_cfg,
                rng=rng,
            )
        weak_coords, ignore_coords = point_sets

        image_shape = (vm.image_shape[0], *coord_spatial)
        return {
            **meta,
            "coords": coords,
            "masks": masks,
            "targets": targets,
            "node_counts": masks.sum(dim=1),
            "imgs": imgs.half(),  # (W, Z, Y, X) full-scale float16
            "weak_positives": weak_coords,  # (W, N, 3) image-resolution voxels
            "ignore_positives": ignore_coords,  # (W, N, 3), excluded from the negative term
            "image_shape": torch.tensor(image_shape, dtype=torch.long),
            "voxel_size": torch.tensor(voxel_size, dtype=torch.float32),
            "name": f'{vm.zarr_path.name}_t{t_start}',
        }
