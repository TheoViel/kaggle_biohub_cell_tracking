import json
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
    apply_normalization,
    frame_anchors,
    load_dataset_windows,
    _pad_window,
)
from tracking.weak_positives import compute_weak_positives


def _valid(points: torch.Tensor) -> torch.Tensor:
    """The real rows of a -1-padded (N, 3) point set."""
    return points[(points >= 0).all(dim=-1)]


def _merge_points(sets: list[torch.Tensor], n_max: int) -> torch.Tensor:
    """Concatenate -1-padded (W, N, 3) point sets back into one (W, n_max, 3).

    Earlier sets win the budget: the caller has already deduplicated the later
    ones against them, so what gets cut when a frame overflows ``n_max`` is the
    lower-priority source rather than an arbitrary slice of both.
    """
    W = sets[0].shape[0]
    out = torch.full((W, n_max, 3), -1.0, dtype=torch.float32)
    for f in range(W):
        pts = torch.cat([_valid(s[f]) for s in sets])
        n = min(len(pts), n_max)
        if n:
            out[f, :n] = pts[:n]
    return out


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
        nodes_cfg: dict | None = None,
        load_edges: bool = True,
    ):
        all_windows = [w for _, windows in video_data for w in windows]
        if not all_windows:
            raise ValueError("No windows found — check data_dir and splits.")
        if max_nodes is None:
            max_nodes = max(max(w.node_counts) for w in all_windows)

        self.downscale_factor = downscale_factor
        self.load_images = load_images
        # The detection-only stages never read the transition matrix. Skipping
        # it drops the padded window from ~2.4 MB to ~20 KB and takes the
        # matrix out of the crop and augmentation path with it.
        self.load_edges = load_edges
        self.weak_positives_cfg = weak_positives_cfg
        self._weak_cache: dict[str, torch.Tensor] = {}
        # One npz per video, a few hundred KB each — cheaper to hold than to
        # re-read for every window of that video.
        self._pl_cache: dict[str, np.ndarray] = {}
        # FOCUS-3D pseudo-labels: one json per stack, and None for the stacks
        # that have no file or that the recall/ratio gate rejected.
        self._focus3d_cache: dict[str, np.ndarray | None] = {}
        # Stage 3 can take its nodes from a predictions folder instead of the
        # U-Net's own peaks, with the NETr embeddings that go with them.
        self.nodes_cfg = dict(nodes_cfg or {})
        self._emb_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.aug_cfg = {**AUG_DEFAULTS, **(aug_cfg or {})}
        self.clamp_max = clamp_max
        self.max_nodes = max_nodes
        self.augment = augment
        self.crop_size = crop_size

        self._data: list[tuple[dict, VideoMeta]] = []
        for video_meta, windows in video_data:
            spatial = tuple(s // downscale_factor for s in video_meta.image_shape[1:])
            for window in windows:
                meta = _pad_window(window, max_nodes, edges=load_edges)
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

    def _load_focus3d(self, sample_name: str) -> np.ndarray | None:
        """FOCUS-3D pseudo-labels as (N, 4) [t, z, y, x], or None if unusable.

        ``pl_focus_3D.py`` writes one json per stack holding ``nodes`` keyed by
        frame index, plus the two sanity numbers this gate reads: ``recall``
        against the sparse annotation and ``ratio``, predicted cells over the
        organiser's estimate. A stack that misses annotated cells or that
        hallucinates far more cells than the metadata expects is not a
        pseudo-label source worth training on, so it is dropped whole rather
        than per-frame — the failure is a segmentation-wide miscalibration, not
        a handful of bad frames.

        A stack with no json is dropped the same way: the folder covers only
        part of the training set, so a missing file is normal and the caller
        falls back to image peaks.
        """
        cfg = self.weak_positives_cfg
        if sample_name in self._focus3d_cache:
            return self._focus3d_cache[sample_name]

        folder = Path(cfg.get("focus_3d_dir") or "../output/focus3d/")
        if not folder.is_dir():
            raise FileNotFoundError(
                f"use_focus_3d is set but {folder} does not exist"
            )

        path = folder / f"predicted_nodes_{sample_name}.json"
        coords = None
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            recall = float(data.get("recall", float("nan")))
            ratio = float(data.get("ratio", float("nan")))
            # A NaN recall (an unannotated stack scores nothing rather than
            # zero) fails both comparisons, so it drops out here too.
            if (
                recall >= float(cfg.get("focus_3d_min_recall", 0.8))
                and ratio <= float(cfg.get("focus_3d_max_ratio", 1.5))
            ):
                rows = [
                    np.concatenate(
                        [np.full((len(pts), 1), int(t), dtype=np.float32),
                         np.asarray(pts, dtype=np.float32).reshape(-1, 3)],
                        axis=1,
                    )
                    for t, pts in data["nodes"].items()
                    if len(pts)
                ]
                coords = (
                    np.concatenate(rows) if rows
                    else np.zeros((0, 4), dtype=np.float32)
                )

        self._focus3d_cache[sample_name] = coords
        return coords

    def _load_emb(self, folder: str, sample_name: str) -> tuple[np.ndarray, np.ndarray]:
        """``(coords (N, 4), embeddings (N, D))`` from inference_contrastive."""
        key = f"{folder}::{sample_name}"
        hit = self._emb_cache.get(key)
        if hit is None:
            path = Path(folder) / f"{sample_name}.npz"
            if not path.exists():
                raise FileNotFoundError(f"No embeddings for {sample_name} at {path}")
            with np.load(path) as d:
                if "embeddings" not in d:
                    raise KeyError(
                        f"{path} has no 'embeddings' array -- contrastive_folder "
                        "should point at inference_contrastive.py's output, not "
                        "at the tracking predictions folder"
                    )
                hit = (d["coords"].astype(np.float32),
                       d["embeddings"].astype(np.float32))
            self._emb_cache[key] = hit
        return hit

    def _precomputed_nodes(
        self,
        sample_name: str,
        frames: list[int],
        n_max: int,
        emb_tol: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The window's precomputed nodes and their embeddings.

        Returns ``(coords (W, n_max, 3), emb (W, n_max, D))``, coords -1-padded
        in the same convention the point-set machinery uses so they can ride
        through augmentation with everything else, and the embeddings padded
        with zero rows -- which is what marks a node as carrying none.

        The two folders are matched by coordinate rather than by row order.
        ``inference_contrastive`` copies the coords through unchanged, so the
        rows normally line up exactly; matching on the values means a folder
        built from a *different* detection run degrades to "these nodes have no
        embedding" instead of silently pairing each node with a stranger's.
        """
        cfg = self.nodes_cfg
        coords_all = self._load_pl(cfg["coordinates_folder"], sample_name)

        emb_coords = emb_feats = None
        if cfg.get("contrastive_folder"):
            emb_coords, emb_feats = self._load_emb(
                cfg["contrastive_folder"], sample_name
            )
        dim = 0 if emb_feats is None else emb_feats.shape[1]

        out = torch.full((len(frames), n_max, 3), -1.0, dtype=torch.float32)
        emb = torch.zeros(len(frames), n_max, dim, dtype=torch.float32)

        for f, t in enumerate(frames):
            pts = coords_all[coords_all[:, 0] == t][:, 1:]
            if not len(pts):
                continue
            n = min(len(pts), n_max)
            pts = pts[:n]
            out[f, :n] = torch.from_numpy(pts) / self.downscale_factor

            if dim:
                ref = emb_coords[emb_coords[:, 0] == t]
                if len(ref):
                    rows = np.flatnonzero(emb_coords[:, 0] == t)
                    d = np.linalg.norm(pts[:, None] - ref[None, :, 1:], axis=2)
                    j = d.argmin(axis=1)
                    ok = d[np.arange(n), j] <= emb_tol
                    emb[f, :n][torch.from_numpy(ok)] = torch.from_numpy(
                        emb_feats[rows[j[ok]]]
                    )
        return out, emb

    def _pl_points(
        self,
        coords: np.ndarray,               # (N, 4) [t, z, y, x] at original resolution
        frames: list[int],                # absolute frame indices, stride applied
        exclude: list[torch.Tensor],      # per frame, (n, 3) points to stay clear of
        radius: float,
        n_max: int,
        voxel_size: tuple[float, float, float],
    ) -> torch.Tensor:                    # (W, N, 3), -1 padded
        """Predicted points for this window, minus anything near ``exclude``.

        The predictions are a tracker's own detections (or FOCUS-3D's mask
        centroids), so they cover the cells the annotation misses — which is
        exactly what the weak-positive term is for. Anything within ``radius``
        of an excluded point is dropped: for the weak set that is the GT,
        already supervised at full weight, and keeping a duplicate there would
        fight the GT target with a down-weighted copy. The same pass deduplicates
        one prediction source against another when both are configured.
        """
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

            raw = raw.astype(np.float32)
            # self.clamp_max is the dataset's own knob and stays authoritative,
            # so the two branches cannot clamp differently.
            norm = {**(vm.norm or {"scope": "video", "clamp_min": 0.0}),
                    "clamp_max": self.clamp_max}
            if norm.get("scope") == "frame":
                # Anchors per frame, so illumination drift across a stack does
                # not shift the detector's operating point mid-video.
                imgs = torch.from_numpy(
                    np.stack([
                        apply_normalization(fr, *frame_anchors(fr, norm), norm)
                        for fr in raw
                    ])
                )
            else:
                imgs = torch.from_numpy(
                    (raw - vm.q_low) / (vm.q_high - vm.q_low + 1e-6)
                ).clamp(float(norm.get("clamp_min", 0.0)), self.clamp_max)  # (W, Z, Y, X)
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
            # None unless FOCUS-3D is on and this stack passed the gate.
            focus3d = (
                self._load_focus3d(vm.zarr_path.stem)
                if wp_cfg.get("use_focus_3d")
                else None
            )
            has_pl = bool(wp_cfg.get("pl_folder"))
            if has_pl:
                # A previous run's detections beat image peaks: they already
                # know what a cell looks like.
                weak_coords = self._pl_points(
                    self._load_pl(wp_cfg["pl_folder"], vm.zarr_path.stem), frames,
                    exclude=gt_per_frame, radius=radius, n_max=n_max,
                    voxel_size=voxel_size,
                )
            if focus3d is not None:
                # The pseudo-labels win ties: where both sources fire, the
                # tracker's own detection is the one to keep, and FOCUS-3D is
                # left to fill in the cells it missed. Same order in the merge,
                # so a frame that overflows n_max drops FOCUS-3D points first.
                f3_coords = self._pl_points(
                    focus3d, frames,
                    exclude=[
                        torch.cat([gt, _valid(w)])
                        for gt, w in zip(gt_per_frame, weak_coords)
                    ] if has_pl else gt_per_frame,
                    radius=radius, n_max=n_max, voxel_size=voxel_size,
                )
                weak_coords = (
                    _merge_points([weak_coords, f3_coords], n_max) if has_pl
                    else f3_coords
                )
            if has_pl and wp_cfg.get("pl_ignore_folder"):
                # A lower-threshold sweep of the same run: too weak to
                # supervise as a cell, too suspicious to call background.
                # Anything already covered by GT or the weak set is not an
                # ignore region — it is a positive.
                covered = [
                    torch.cat([gt, _valid(w)])
                    for gt, w in zip(gt_per_frame, weak_coords)
                ]
                ignore_coords = self._pl_points(
                    self._load_pl(
                        wp_cfg["pl_ignore_folder"], vm.zarr_path.stem
                    ), frames,
                    exclude=covered,
                    radius=wp_cfg.get("ignore_min_dist_um", radius),
                    n_max=wp_cfg.get("ignore_max_nodes", n_max),
                    voxel_size=voxel_size,
                )
            elif not has_pl and focus3d is None and self.load_images:
                # Image peaks are the last resort: FOCUS-3D overrides them when
                # it has this stack, so they only run for a stack the folder
                # does not cover or that the recall/ratio gate rejected.
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

        # Precomputed nodes ride through augmentation with the other point sets,
        # so a crop or a flip moves them exactly as it moves the image. Their
        # embeddings do not follow: NETr saw the un-augmented appearance and
        # there is no way to recompute it here, which is one reason stage 3
        # runs unaugmented.
        pre_coords = torch.zeros(W, 0, 3)
        pre_emb = torch.zeros(W, 0, 0)
        if self.nodes_cfg.get("coordinates_folder"):
            pre_coords, pre_emb = self._precomputed_nodes(
                vm.zarr_path.stem, frames,
                int(self.nodes_cfg.get("max_nodes", 768)),
            )

        point_sets = [weak_coords, ignore_coords, pre_coords]
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
        weak_coords, ignore_coords, pre_coords = point_sets

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
            # (W, P, 3) image-voxel coords, -1 padded, and their (W, P, D)
            # embeddings; both empty unless a coordinates_folder is configured.
            "pre_coords": pre_coords,
            "pre_emb": pre_emb,
            "image_shape": torch.tensor(image_shape, dtype=torch.long),
            "voxel_size": torch.tensor(voxel_size, dtype=torch.float32),
            "name": f'{vm.zarr_path.name}_t{t_start}',
        }
