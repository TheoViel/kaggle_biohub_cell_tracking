"""Synthetic sequences as extra supervision for the detection stage.

The generator writes one npz per sequence under ``<dir>/sequences``:

    volumes           (T, Z, Y, X) uint16   raw, unnormalised
    nodes             (N, 5) float32        [t, z, y, x, track_id]
    edges             (E, 2) int32          index pairs into ``nodes``
    divisions         (D,) int32            indices of dividing nodes
    voxel_um_pooled   (3,) float32          -- see VOXEL_SIZE below, do not use

What makes these worth mixing in is not the images, it is the labels: every
cell is annotated. The real data annotates ~3% of them (0.7% in 44b6), so its
detection loss trains almost every true cell as background and the whole
weak-positive machinery exists to paper over that. Here a voxel with no node
really is background, so the negative term is correct for once.

Verified against the real data before use (2026-08-29, 1000 sequences):
cells/frame 285 (real 141-352), apparent cell radius 3.7-4.1 um isotropic
(real 3.4-4.7), normalised background 0.049 (real 0.053 in 6bba), normalised
cell peak 0.76 (real matched cells 0.63-0.70). Coordinates are in bounds.

Two things in the shipped metadata are wrong, hence the constants below:

* ``voxel_um_pooled`` says (1.625, 1.625, 1.625). It is not. Under that
  reading a cell measures 4.1 um in z against 14.6 um in y/x -- a 3.6:1 disc,
  which is not a cell and not what the images show. Under the real voxel size
  it is 4.06 / 3.66 / 3.86 um, a sphere, and the nearest-neighbour spacing
  (8.4 um) then matches 44b6's. The volumes are at the real voxel size.
* ``metadata.json`` reports ``n_sequences: 9`` for 1000 files on disk.

Divisions are over-sampled ~15x against the real rate (0.040 vs 0.0026), which
the generator's own note flags. It does not affect the detection stage, which
never looks at edges, but it would bias a division prior trained on this.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.transforms import AUG_DEFAULTS, augment_track, clamp_coords
from params import SCALE
from tracking.data import resolve_normalization

# The real acquisition's voxel size, which is what these volumes are rendered
# at -- not the npz's own voxel_um_pooled. See the module docstring.
VOXEL_SIZE: tuple[float, float, float] = tuple(SCALE)


def find_sequences(root: str | Path) -> list[Path]:
    """The npz files under ``root``, accepting the dir or its ``sequences`` child."""
    root = Path(root)
    for cand in (root / "sequences", root):
        files = sorted(cand.glob("seq_*.npz"))
        if files:
            return files
    raise FileNotFoundError(f"No seq_*.npz under {root} or {root / 'sequences'}")


class SyntheticDataset(Dataset):
    """Windows of ``window_size`` consecutive frames from the synthetic set.

    Emits the same keys as ``TrackDataset``, so the two collate together.
    ``weak_positives`` and ``ignore_positives`` come back as empty padding on
    purpose: the annotation is complete, so there is no unlabelled cell to
    recover and any weak positive here would be a false one.
    """

    def __init__(
        self,
        files: list[Path],
        window_size: int = 2,
        max_nodes: int = 768,
        crop_size: tuple[int, int, int] | None = None,
        augment: bool = True,
        aug_cfg: dict | None = None,
        normalization: dict | None = None,
        voxel_size: tuple[float, float, float] = VOXEL_SIZE,
        mosaic_p: float = 0.0,
        mosaic_jitter: float = 0.2,
        load_edges: bool = True,
    ):
        self.files = list(files)
        if not self.files:
            raise ValueError("No synthetic sequences given")
        self.window_size = int(window_size)
        self.max_nodes = int(max_nodes)
        self.crop_size = crop_size
        self.augment = augment
        self.aug_cfg = {**AUG_DEFAULTS, **(aug_cfg or {})}
        norm = resolve_normalization(normalization)
        self.low_q = float(norm["low_q"])
        self.high_q = float(norm["high_q"])
        self.clamp_max = float(norm["clamp_max"])
        self.voxel_size = tuple(voxel_size)
        self.mosaic_p = float(mosaic_p)
        self.mosaic_jitter = float(mosaic_jitter)
        # Matches TrackDataset.load_edges: the two collate into one batch, so
        # the transition matrix is either present on both or on neither.
        self.load_edges = bool(load_edges)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng()
        if self.mosaic_p > 0 and len(self.files) >= 4 and rng.random() < self.mosaic_p:
            # The other three are drawn anywhere in the set; each goes through
            # _one, so each carries its own crop and its own photometry.
            others = rng.choice(len(self.files), size=3, replace=False).tolist()
            samples = [self._one(idx)] + [self._one(int(j)) for j in others]
            return mosaic_samples(samples, rng, jitter=self.mosaic_jitter)
        return self._one(idx)

    def _one(self, idx: int) -> dict[str, torch.Tensor]:
        """One augmented window, before any mosaicking."""
        path = self.files[idx]
        d = np.load(path, allow_pickle=True)
        vol = d["volumes"]
        nodes = d["nodes"].astype(np.float32)
        edges = d["edges"].astype(np.int64)

        T = vol.shape[0]
        W = min(self.window_size, T)
        rng = np.random.default_rng()
        t0 = int(rng.integers(0, T - W + 1)) if T > W else 0
        frames = list(range(t0, t0 + W))

        raw = vol[t0 : t0 + W].astype(np.float32)
        # Per-sequence quantiles, the same reference the real loader uses. The
        # generator ships no stored quantiles, so they are computed here.
        sub = raw.reshape(-1)[:: max(1, raw.size // 200_000)]
        lo = float(np.quantile(sub, self.low_q))
        hi = float(np.quantile(sub, self.high_q))
        imgs = torch.from_numpy((raw - lo) / (hi - lo + 1e-6)).clamp(0.0, self.clamp_max)

        # --- nodes -> padded coords/masks, and the global index of each slot,
        # so the edges can be mapped onto slot pairs below.
        M = self.max_nodes
        coords = torch.zeros(W, M, 3, dtype=torch.float32)
        masks = torch.zeros(W, M, dtype=torch.bool)
        slot_of = {}
        for w, t in enumerate(frames):
            sel = np.nonzero(nodes[:, 0] == t)[0]
            n = min(len(sel), M)
            if n:
                coords[w, :n] = torch.from_numpy(nodes[sel[:n], 1:4])
                masks[w, :n] = True
            for s, gi in enumerate(sel[:n]):
                slot_of[int(gi)] = (w, s)

        # No frames when the stage does not supervise edges: a synthetic
        # frame holds ~285 cells against max_nodes 768, so this matrix is the
        # bulk of a sample and none of it would be read.
        targets = torch.zeros(
            max(W - 1, 1) if self.load_edges else 0, M, M, dtype=torch.float32
        )
        if self.load_edges:
            for src, tgt in edges:
                a, b = slot_of.get(int(src)), slot_of.get(int(tgt))
                if a is None or b is None or b[0] != a[0] + 1:
                    continue
                targets[a[0], a[1], b[1]] = 1.0

        spatial = tuple(imgs.shape[1:])
        empty = torch.full((W, 1, 3), -1.0, dtype=torch.float32)
        point_sets = [empty, empty]

        if self.augment:
            imgs, coords, point_sets, masks, targets, spatial = augment_track(
                imgs,
                coords,
                point_sets,
                masks,
                targets,
                coord_spatial=spatial,
                crop_size=self.crop_size,
                voxel_size=self.voxel_size,
                cfg=self.aug_cfg,
                rng=rng,
            )
        weak_coords, ignore_coords = point_sets

        return {
            "t_start": t0,
            "n_frames": W,
            # Consecutive frames; the real loader's windows carry the same key
            # and the two are collated together.
            "stride": 1,
            "coords": coords,
            "masks": masks,
            "targets": targets,
            "node_counts": masks.sum(dim=1),
            "imgs": imgs.half(),
            "weak_positives": weak_coords,
            "ignore_positives": ignore_coords,
            # Empty, exactly as the real loader emits them when no
            # coordinates_folder is configured -- which is the only way a
            # synthetic mix is ever built. MixedDataset collates the two, so a
            # key the real half carries has to be here as well.
            "pre_coords": torch.zeros(W, 0, 3, dtype=torch.float32),
            "pre_emb": torch.zeros(W, 0, 0, dtype=torch.float32),
            "image_shape": torch.tensor((T, *spatial), dtype=torch.long),
            "voxel_size": torch.tensor(self.voxel_size, dtype=torch.float32),
            "name": f"synth_{path.stem}_t{t0}",
        }


def mosaic_samples(
    samples: list[dict],
    rng: np.random.Generator,
    jitter: float = 0.25,
) -> dict:
    """Combine four augmented windows into one, 2x2 in Y/X (YOLO-style mosaic).

    Each source arrives already augmented and already at the output size, so
    the four quarters differ in crop, orientation and photometry before they
    are ever tiled -- which is the point of the transform: one crop then shows
    four unrelated pieces of tissue, with four different appearances, meeting
    at edges the model has to not hallucinate cells across.

    Geometry follows the 2D original. A join point ``(by, bx)`` is drawn near
    the middle of the output ``Y x X`` frame; the four quadrants meet there,
    so their sizes are ``(by, bx)``, ``(by, X-bx)``, ``(Y-by, bx)`` and
    ``(Y-by, X-bx)``. Each source is resized straight onto its own quadrant --
    which is where the differing stretches and aspect ratios come from -- and
    placed at the matching corner, so all four always contribute and the join
    sits exactly at ``(by, bx)``.

    Z is left alone, as asked: the tiling is purely in Y/X, and every source
    contributes its full depth. Anisotropy is the reason -- a z step is 4x a
    y/x step, so a volume that is 64 deep and 256 wide is far thinner in
    microns than in voxels, and quartering it would leave slabs a couple of
    cells thick.

    ``jitter`` is the fraction of the image width/height the join may wander
    off centre, 0 being a plain equal-quarters tiling and 0.2 (the default)
    letting it drift +/-20% of Y/X. It is clamped to leave each quadrant at
    least a tenth of the frame.
    """
    if len(samples) != 4:
        raise ValueError(f"mosaic needs exactly 4 samples, got {len(samples)}")

    ref = samples[0]["imgs"]
    W, Z, Y, X = ref.shape
    if Y < 2 or X < 2:
        return samples[0]

    # Join point near the middle of the output frame, in output coordinates
    # (0.2 -> wanders +/-20% of Y/X off centre), clamped to leave every
    # quadrant at least a tenth of the frame.
    j = float(np.clip(jitter, 0.0, 0.4))
    by = int(Y / 2 + Y * rng.uniform(-j, j))
    bx = int(X / 2 + X * rng.uniform(-j, j))
    by = int(np.clip(by, Y // 10, Y - Y // 10))
    bx = int(np.clip(bx, X // 10, X - X // 10))

    # quadrant: (origin_y, origin_x, size_y, size_x) in the output frame.
    quads = [
        (0, 0, by, bx),                    # top-left
        (0, bx, by, X - bx),               # top-right
        (by, 0, Y - by, bx),               # bottom-left
        (by, bx, Y - by, X - bx),          # bottom-right
    ]

    out_imgs = torch.zeros((W, Z, Y, X), dtype=ref.dtype)
    max_nodes = samples[0]["coords"].shape[1]
    coords = torch.zeros(W, max_nodes, 3, dtype=torch.float32)
    masks = torch.zeros(W, max_nodes, dtype=torch.bool)
    n_used = [0] * W
    # (source, frame, old slot) -> new slot, to rebuild the transition matrices.
    remap: dict[tuple[int, int, int], int] = {}

    for i, (smp, (qy, qx, qh, qw)) in enumerate(zip(samples, quads)):
        img = smp["imgs"].float()
        # Resize Y/X onto the quadrant; Z is untouched. Folding W and Z into
        # the batch axis keeps this a plain 2-D interpolate.
        res = F.interpolate(
            img.reshape(W * Z, 1, img.shape[-2], img.shape[-1]),
            size=(qh, qw), mode="bilinear", align_corners=False,
        ).reshape(W, Z, qh, qw)
        out_imgs[:, :, qy : qy + qh, qx : qx + qw] = res.to(ref.dtype)

        sy, sx = qh / img.shape[-2], qw / img.shape[-1]
        for w in range(W):
            sel = torch.nonzero(smp["masks"][w]).squeeze(1)
            if not len(sel):
                continue
            c = smp["coords"][w][sel].float()
            # source voxels -> quadrant -> output frame
            gy = c[:, 1] * sy + qy
            gx = c[:, 2] * sx + qx
            keep = (gy >= 0) & (gy < Y) & (gx >= 0) & (gx < X) & (c[:, 0] >= 0) & (c[:, 0] < Z)
            for k in torch.nonzero(keep).squeeze(1).tolist():
                if n_used[w] >= max_nodes:
                    break
                slot = n_used[w]
                # Clamped onto a real index: the quadrant rescale can push a
                # cell into the last half-voxel of the canvas, which is inside
                # the crop but names a voxel that is not, and consumers round.
                coords[w, slot] = clamp_coords(
                    torch.tensor([[c[k, 0], gy[k], gx[k]]]), (Z, Y, X)
                )[0]
                masks[w, slot] = True
                remap[(i, w, int(sel[k]))] = slot
                n_used[w] += 1

    # Rebuild the transition matrices from each source's own links: a mosaic
    # never links across sources, and the slots have all moved.
    n_target_frames = max(W - 1, 1) if samples[0]["targets"].numel() else 0
    targets = torch.zeros(n_target_frames, max_nodes, max_nodes, dtype=torch.float32)
    for i, smp in enumerate(samples):
        t = smp["targets"]
        for w in range(min(t.shape[0], max(W - 1, 1))):
            for a, b in torch.nonzero(t[w] > 0).tolist():
                na, nb = remap.get((i, w, a)), remap.get((i, w + 1, b))
                if na is not None and nb is not None:
                    targets[w, na, nb] = 1.0

    out = dict(samples[0])
    out.update(
        imgs=out_imgs,
        coords=coords,
        masks=masks,
        targets=targets,
        node_counts=masks.sum(dim=1),
        name="mosaic_" + "_".join(s["name"].split("_t")[0][-4:] for s in samples),
    )
    return out


class MixedDataset(Dataset):
    """Real windows, with a ``proportion`` of them replaced by synthetic ones.

    Length and index space are the real dataset's, so the samplers around it --
    DistributedSampler, the stable-shard SubsetRandomSampler -- keep working
    unchanged and every rank still gets an equal share. The draw is per access,
    like the augmentations', so the proportion holds in expectation over an
    epoch rather than exactly within it.
    """

    def __init__(self, real: Dataset, synthetic: Dataset, proportion: float):
        if not 0.0 <= proportion <= 1.0:
            raise ValueError(f"proportion must be in [0, 1], got {proportion}")
        self.real = real
        self.synthetic = synthetic
        self.proportion = float(proportion)
        # The two halves must collate together, and the real path's padding
        # widths depend on the weak-positive config in ways not worth
        # reproducing here (which strategy, whether a pl_ignore folder is set).
        # Read them off one real sample instead, once, in the parent process.
        self._widths: dict[str, int] = {}
        if proportion > 0 and len(real):
            probe = real[0]
            for key in ("weak_positives", "ignore_positives"):
                if key in probe:
                    self._widths[key] = int(probe[key].shape[1])
            self._keys = set(probe)

    def _match_widths(self, sample: dict) -> dict:
        """Pad the synthetic point sets out to the real ones' width.

        Also checks the key sets agree. default_collate reports a mismatch as
        a bare ``KeyError`` from inside a worker, which says nothing about
        which dataset is short of what.
        """
        missing = self._keys - set(sample)
        extra = set(sample) - self._keys
        if missing or extra:
            raise KeyError(
                "synthetic and real samples must carry the same keys to collate; "
                f"synthetic is missing {sorted(missing)} and adds {sorted(extra)}"
            )
        for key, width in self._widths.items():
            pts = sample.get(key)
            if pts is None or pts.shape[1] == width:
                continue
            out = torch.full((pts.shape[0], width, 3), -1.0, dtype=pts.dtype)
            n = min(pts.shape[1], width)
            out[:, :n] = pts[:, :n]
            sample[key] = out
        return sample

    def __len__(self) -> int:
        return len(self.real)

    # Transformer-only training flips this on the dataset to stop reading
    # volumes; forward it so a MixedDataset is a drop-in there too.
    @property
    def load_images(self) -> bool:
        return getattr(self.real, "load_images", True)

    @load_images.setter
    def load_images(self, value: bool) -> None:
        self.real.load_images = value

    @property
    def crop_size(self):
        return self.real.crop_size

    def __getitem__(self, idx: int):
        if self.proportion > 0 and np.random.default_rng().random() < self.proportion:
            j = int(np.random.default_rng().integers(0, len(self.synthetic)))
            return self._match_widths(self.synthetic[j])
        return self.real[idx]
