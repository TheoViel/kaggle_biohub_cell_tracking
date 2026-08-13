import zarr
import torch
import numpy as np
from torch.utils.data import Dataset

from data.transforms import (
    brightness_augment,
    flip_augment,
    has_feasible_track_crop,
    random_crop_track,
)
from tracking.data import (
    VideoMeta, FrameWindowData, load_dataset_windows, _pad_window,
)


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
    ):
        all_windows = [w for _, windows in video_data for w in windows]
        if not all_windows:
            raise ValueError("No windows found — check data_dir and splits.")
        if max_nodes is None:
            max_nodes = max(max(w.node_counts) for w in all_windows)

        self.max_nodes = max_nodes
        self.augment = augment
        self.crop_size = crop_size

        self._data: list[tuple[dict, VideoMeta]] = []
        for video_meta, windows in video_data:
            # spatial bounds in coord space (1/4-scale) for feasibility checking
            spatial = tuple(video_meta.image_shape[1:])
            if self.augment and self.crop_size is not None and any(
                crop > size for crop, size in zip(self.crop_size, spatial)
            ):
                raise ValueError(
                    f"crop_size {self.crop_size} exceeds coord spatial shape {spatial}"
                )
            for window in windows:
                meta = _pad_window(window, max_nodes)
                if (
                    self.augment
                    and self.crop_size is not None
                    and not has_feasible_track_crop(
                        meta["coords"],
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Load metadata
        meta, vm = self._data[idx]
        t_start = meta["t_start"]
        W = meta["n_frames"]

        # Load image at full scale; spatial downsampling is handled by the model
        z_arr = zarr.open_group(str(vm.zarr_path), mode="r")["0"]
        raw = z_arr[t_start : t_start + W].astype(np.float32)
        imgs = torch.from_numpy(
            (raw - vm.q_low) / (vm.q_high - vm.q_low + 1e-6)
        ).clamp(0.0, 4.0)  # (W, Z, Y, X)

        # Targets
        coords = meta["coords"].clone()  # (W, M, 3) in 1/4-scale coord space
        masks = meta["masks"]            # (W, M) bool
        targets = meta["targets"]

        # coord-space spatial (1/4-scale), returned as image_shape for pos embeddings
        coord_spatial = tuple(vm.image_shape[1:])

        if self.augment:
            rng = np.random.default_rng()
            if self.crop_size is not None:
                # image_scale bridges full-scale pixels to coord-space voxels
                imgs, coords, masks, targets = random_crop_track(
                    imgs,
                    coords,
                    masks,
                    targets,
                    self.crop_size,
                    rng,
                )
                coord_spatial = self.crop_size
            imgs = brightness_augment(imgs, rng)
            imgs, coords, masks = flip_augment(
                imgs, coords, masks, coord_spatial, rng
            )

        image_shape = (vm.image_shape[0], *coord_spatial)
        return {
            **meta,
            "coords": coords,
            "masks": masks,
            "targets": targets,
            "node_counts": masks.sum(dim=1),
            "imgs": imgs.half(),  # (W, Z, Y, X) full-scale float16
            "image_shape": torch.tensor(image_shape, dtype=torch.long),
            "voxel_size": torch.tensor(vm.voxel_size, dtype=torch.float32),
        }
