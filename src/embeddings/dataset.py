"""Frames of detections, delivered as pairs the contrastive loss can score.

A sample is one *frame*, not one cell: NT-Xent needs every cell in the frame at
once, because the other cells are the negatives, and the transformer needs them
because an embedding is contextual. Batch size is therefore counted in frames,
as the paper counts it (4).

Detections come from ``coords_log_folder`` -- the ``coords`` array a tracking
run wrote, ``(N, 4)`` as ``t, z, y, x``. Training on the pipeline's own
detections rather than on ground truth is the point: these are the nodes the
matcher will actually be asked to link, including the spurious ones.

Two kinds of pair are produced, mixed at ``real_prop``:

**Synthetic** (ASCENT's own, ``real_prop = 0``): two augmented views of the
same frame, cell i pairing with cell i. No annotation at all -- the supervision
is entirely the augmentation pipeline's, which means the embeddings are only
ever asked to be invariant to *transforms*, never to a cell actually moving,
dimming or changing shape between two frames.

**Real**: frame t against frame t+1, paired by the ground-truth graph. This is
the task at inference, so it is worth some of the budget even though the
annotation is thin. Two things make it work:

- The frame-wide augmentation is drawn **once** and applied to both frames, so
  the pair still looks like a real pair of consecutive frames. Rotating t and
  t+1 by different angles would leave the network a correspondence problem the
  ground truth no longer describes.
- The annotation is **partial** -- roughly one link a frame pair against ~500
  detections -- so only the annotated rows are scored. Every other detection
  stays in the batch as a *negative*, which is all that is needed: for an
  annotated cell we know which cell at t+1 is its positive, and every other
  candidate is by construction not it. Unannotated cells simply contribute no
  term of their own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from embeddings.augment import (
    apply_photometric,
    resolve_aug_cfg,
    sample_cells,
    sample_geometry,
)
from embeddings.links import MATCH_UM, detection_links
from embeddings.patches import (
    PATCH_ZYX,
    extract_patches,
    normalise_coords,
    normalise_frame,
)
from params import SCALE


def find_coord_files(coords_log_folder: str | Path) -> list[Path]:
    """The npz files of a predictions folder, sorted."""
    folder = Path(coords_log_folder)
    files = sorted(p for p in folder.glob("*.npz") if not p.stem.endswith("_flows"))
    if not files:
        raise FileNotFoundError(f"no *.npz under {folder}")
    return files


class EmbeddingDataset(Dataset):
    """One frame -> a pair of views, synthetic or real, for NT-Xent.

    Args:
        coords_log_folder: predictions folder holding ``<sample>.npz`` with a
            ``coords`` array of ``(t, z, y, x)`` rows.
        data_dir: where the matching ``<sample>.zarr`` volumes live -- and, for
            ``real_prop > 0``, the ``<sample>.geff`` ground-truth graphs.
        embryos: restrict to samples whose name starts with one of these.
        max_nodes: cap per frame. Frames here hold ~180-790 detections and the
            transformer is quadratic in that, so the cap is a real budget, not
            a formality. Cells are dropped at random rather than by score, so
            the survivors stay a fair sample of the frame -- except that an
            annotated endpoint of a real pair is always kept, since dropping it
            would throw away the only supervised row the pair has.
        min_nodes: frames with fewer are skipped -- a handful of cells gives
            NT-Xent almost no negatives, so the loss is close to free and the
            gradient close to noise.
        real_prop: probability that a frame is served as a real (t, t+1) pair
            rather than two views of itself. Frames whose ground truth yields
            no usable link fall back to the synthetic pair, so the realised
            proportion is ``real_prop`` times ``self.real_coverage``.
    """

    def __init__(
        self,
        coords_log_folder,
        data_dir,
        embryos: list[str] | None = None,
        max_nodes: int = 256,
        min_nodes: int = 8,
        patch_zyx: tuple[int, int, int] = PATCH_ZYX,
        n_frames: int = 1,
        aug_cfg: dict | None = None,
        augment: bool = True,
        max_frames: int | None = None,
        real_prop: float = 0.0,
        match_um: float = MATCH_UM,
    ):
        self.data_dir = Path(data_dir)
        self.patch_zyx = tuple(patch_zyx)
        self.n_frames = int(n_frames)
        self.aug = resolve_aug_cfg(aug_cfg)
        self.augment = augment
        self.max_nodes = int(max_nodes)
        self.real_prop = float(real_prop or 0.0)

        files = find_coord_files(coords_log_folder)
        if embryos:
            files = [f for f in files if any(f.stem.startswith(e) for e in embryos)]
            if not files:
                raise ValueError(f"no samples for embryos {embryos} in {coords_log_folder}")

        # One index entry per (sample, frame). The coords are small, so they
        # are held in memory; the volumes are not and are opened on demand.
        self.coords: dict[str, np.ndarray] = {}
        self.index: list[tuple[str, int]] = []
        counts: dict[str, dict[int, int]] = {}
        for path in files:
            with np.load(path) as d:
                coords = np.asarray(d["coords"])
            if coords.size == 0:
                continue
            self.coords[path.stem] = coords
            frames = np.unique(coords[:, 0].astype(int))
            if max_frames is not None:
                frames = frames[:max_frames]
            counts[path.stem] = {
                int(t): int((coords[:, 0] == t).sum()) for t in frames
            }
            self.index += [
                (path.stem, int(t)) for t in frames
                if counts[path.stem][int(t)] >= min_nodes
            ]
        if not self.index:
            raise ValueError("no frame held enough detections to train on")

        # Ground-truth links, as detection indices. Read once here rather than
        # per item: it is ~0.1 s a video, and the workers inherit the result
        # by fork instead of each paying for it.
        self.links: dict[str, dict[int, np.ndarray]] = {}
        if self.real_prop > 0:
            for sample in self.coords:
                path = self.data_dir / sample
                if not path.with_suffix(".geff").exists():
                    continue
                pairs = detection_links(path, self.coords[sample], match_um)
                # A real pair needs both frames to be trainable frames in their
                # own right -- enough detections for the negatives to mean
                # something -- so it is held to the same min_nodes bar.
                self.links[sample] = {
                    t: rows for t, rows in pairs.items()
                    if counts[sample].get(t, 0) >= min_nodes
                    and counts[sample].get(t + 1, 0) >= min_nodes
                }
        n_real = sum(1 for s, t in self.index if t in self.links.get(s, {}))
        self.real_coverage = n_real / len(self.index)

        self._zarr: dict[str, zarr.Array] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _volume(self, sample: str, t: int) -> np.ndarray:
        # Opened lazily and cached per worker: zarr handles are not picklable
        # across processes, so this must not happen in __init__.
        if sample not in self._zarr:
            self._zarr[sample] = zarr.open_group(
                str(self.data_dir / f"{sample}.zarr"), mode="r"
            )["0"]
        return np.asarray(self._zarr[sample][t])

    def _frame_coords(self, sample: str, t: int) -> np.ndarray:
        rows = self.coords[sample]
        return rows[rows[:, 0].astype(int) == t][:, 1:4].astype(np.float32)

    def _subsample(self, n: int, keep: np.ndarray, rng) -> np.ndarray:
        """Which of ``n`` cells to use, always including the rows in ``keep``.

        Returns indices into the frame's detections, in order, or None when the
        frame already fits.
        """
        if n <= self.max_nodes:
            return None
        keep = np.unique(np.asarray(keep, int))[: self.max_nodes]
        rest = np.setdiff1d(np.arange(n), keep, assume_unique=False)
        extra = rng.choice(rest, self.max_nodes - len(keep), replace=False)
        return np.sort(np.concatenate([keep, extra]))

    def _render(self, volume, coords, geom, cells, rng, shape_zyx):
        """One view's patches and normalised coordinates.

        ``geom`` is the frame-wide transform (shared across a real pair) and
        ``cells`` the per-cell jitter and dropout. With ``augment`` off both are
        None and the frame is read as it is.
        """
        if geom is None:
            patches = extract_patches(volume, coords, self.patch_zyx,
                                      n_frames=self.n_frames)
            return patches, normalise_coords(coords, shape_zyx)

        patches = extract_patches(
            volume, cells["coords"], self.patch_zyx,
            angle_deg=geom["angle"], scale=geom["scale"], elastic=geom["elastic"],
            n_frames=self.n_frames,
        )
        patches = apply_photometric(patches, geom["photo"], rng)
        # The coordinate the network sees carries the same jitter the patch
        # was cut with, so the two stay consistent within a view while
        # differing between views -- see augment.sample_cells.
        return patches, normalise_coords(cells["coords"], shape_zyx)

    def _views(self, sample, t, rng):
        """A pair of views: ``(view_1, view_2, pos_1to2, pos_2to1, is_real)``.

        Each view is ``(patches, coords, keep)``; the two ``pos`` arrays hold,
        for every cell of one view, the row of its positive in the other, or -1
        where there is none to score against.
        """
        pairs = self.links.get(sample, {}).get(t) if self.real_prop > 0 else None
        is_real = pairs is not None and rng.random() < self.real_prop

        volume = normalise_frame(self._volume(sample, t))
        shape_zyx = volume.shape
        geom = (
            sample_geometry(self.aug, rng, SCALE, shape_zyx[1:])
            if self.augment else None
        )

        def view(vol, coords):
            cells = sample_cells(coords, self.aug, rng, SCALE) if self.augment else None
            keep = cells["keep"] if cells else np.ones(len(coords), bool)
            patches, coords_n = self._render(vol, coords, geom, cells, rng, shape_zyx)
            return (patches, coords_n, keep)

        if not is_real:
            coords = self._frame_coords(sample, t)
            sel = self._subsample(len(coords), np.empty(0, int), rng)
            if sel is not None:
                coords = coords[sel]
            v1, v2 = view(volume, coords), view(volume, coords)
            # Cell i pairs with cell i, wherever both views kept it.
            both = v1[2] & v2[2]
            pos = np.where(both, np.arange(len(coords)), -1)
            return v1, v2, pos, pos.copy(), False

        # --- real pair: frame t against frame t+1 ---
        src, tgt, src_unique = pairs[:, 0], pairs[:, 1], pairs[:, 2].astype(bool)
        coords_1 = self._frame_coords(sample, t)
        coords_2 = self._frame_coords(sample, t + 1)

        sel_1 = self._subsample(len(coords_1), src, rng)
        sel_2 = self._subsample(len(coords_2), tgt, rng)
        row_1 = self._remap(len(coords_1), sel_1)
        row_2 = self._remap(len(coords_2), sel_2)
        if sel_1 is not None:
            coords_1 = coords_1[sel_1]
        if sel_2 is not None:
            coords_2 = coords_2[sel_2]

        v1 = view(volume, coords_1)
        v2 = view(normalise_frame(self._volume(sample, t + 1)), coords_2)

        pos_1to2 = np.full(len(coords_1), -1, np.int64)
        pos_2to1 = np.full(len(coords_2), -1, np.int64)
        for a, b, unique in zip(row_1[src], row_2[tgt], src_unique):
            # A cell dropped by the augmentation is not in the batch at all, so
            # the edge that pointed at it has no target left to score.
            if a < 0 or b < 0 or not v1[2][a] or not v2[2][b]:
                continue
            if unique:      # a dividing cell has two right answers: see links.py
                pos_1to2[a] = b
            pos_2to1[b] = a
        return v1, v2, pos_1to2, pos_2to1, True

    @staticmethod
    def _remap(n: int, sel: np.ndarray | None) -> np.ndarray:
        """old detection row -> new row after subsampling, -1 where dropped."""
        if sel is None:
            return np.arange(n)
        out = np.full(n, -1, np.int64)
        out[sel] = np.arange(len(sel))
        return out

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample, t = self.index[idx]
        rng = np.random.default_rng()

        v1, v2, pos_1to2, pos_2to1, is_real = self._views(sample, t, rng)

        pz, py, px = self.patch_zyx
        M = self.max_nodes

        # Patches ride to the GPU as fp16. They are a batch's whole bulk --
        # max_nodes * n_frames * pz * py * px * 2 views, ~1.1 GB a step in
        # float32 -- and autocast casts them to fp16 on arrival anyway, so
        # float32 buys nothing but PCIe traffic and collate time. The FP32
        # training path casts back on the GPU, where it is nearly free.
        def pad(view, pos):
            patches, coords_n, keep = view
            n = len(coords_n)
            P = np.zeros((M, self.n_frames, pz, py, px), np.float16)
            C = np.zeros((M, 3), np.float32)
            K = np.zeros(M, bool)
            L = np.full(M, -1, np.int64)
            P[:n], C[:n], K[:n], L[:n] = patches, coords_n, keep, pos
            # A cell the augmentation dropped is not embedded, so it can be
            # neither scored nor named as anyone's positive.
            L[~K] = -1
            return P, C, K, L

        P1, C1, K1, L1 = pad(v1, pos_1to2)
        P2, C2, K2, L2 = pad(v2, pos_2to1)

        return {
            "patches_1": torch.from_numpy(P1),
            "patches_2": torch.from_numpy(P2),
            "coords_1": torch.from_numpy(C1),
            "coords_2": torch.from_numpy(C2),
            "mask_1": torch.from_numpy(K1),
            "mask_2": torch.from_numpy(K2),
            "pos_1to2": torch.from_numpy(L1),
            "pos_2to1": torch.from_numpy(L2),
            "is_real": torch.tensor(bool(is_real)),
            "name": f"{sample}_t{t}",
        }
