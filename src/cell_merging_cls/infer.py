"""Score every close pair of detections in a stack with "these two are one cell".

Frames are walked in order and held in a small rolling window, so each frame is
read and normalised once and every pair in it is cut from what is already in
memory -- the same amortisation `division_cls.infer` does, and it matters more
here because there are ~0.65 pairs per detection to ask about.

Normalisation follows the training loader exactly: percentiles over the
`patch_z` slab the crop is cut from, not over the whole frame. The slab depends
only on the integer z it starts at, so the percentiles are computed once per
`(frame, z0)` and reused, which turns a per-pair cost into a per-plane one.

**Which model scores which embryo matters.** Fold `i` held out `EMBRYOS[i]`, so
a stack must be scored by the fold that did *not* train on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import zarr
from scipy.spatial import cKDTree

from cell_merging_cls.crops import pair_patches
from cell_merging_cls.dataset import pair_geometry
from cell_merging_cls.model import MergeNet
from cell_merging_cls.pairs import SCALE_ARR
from params import EMBRYOS


def load_model(ckpt_path, device="cuda"):
    """`(model, cfg)` from a checkpoint written by `cell_merging_cls.train.fit`."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(state.get("model_config") or {})
    if not cfg:
        raise ValueError(f"{ckpt_path} carries no model_config")
    model = MergeNet(
        backbone=cfg["backbone"], pretrained=False,
        patch_yx=cfg["patch_yx"], patch_z=cfg["patch_z"], n_frames=cfg["n_frames"],
        d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"],
        min_tokens=cfg["min_tokens"], geom_dim=cfg.get("geom_dim", 32),
    )
    model.load_state_dict(state["model_state_dict"])
    return model.eval().to(device), cfg


def load_fold_models(run_dir, device="cuda"):
    """`{embryo: (model, cfg)}` -- each embryo mapped to the fold that held it out."""
    run_dir = Path(run_dir)
    out = {}
    for fold, embryo in enumerate(EMBRYOS):
        ckpt = run_dir / f"fold_{fold}" / "ckpt_final.pt"
        if ckpt.exists():
            out[embryo] = load_model(ckpt, device)
    if not out:
        raise FileNotFoundError(f"no fold_*/ckpt_final.pt under {run_dir}")
    return out


def candidate_pairs(coords, max_pair_um):
    """`(K, 2)` rows of `coords` that are close enough to be the same cell.

    The radius is what could plausibly be one cell found twice, not what could
    be a division: measured duplicate pairs top out at 8.1 um, so a 9 um default
    covers them with margin while asking about a quarter as many pairs as the
    14 um the training negatives are drawn from.
    """
    coords = np.asarray(coords)
    p_um = coords[:, 1:].astype(float) * SCALE_ARR
    frame_of = coords[:, 0].astype(int)
    out = []
    for t in np.unique(frame_of):
        rows = np.flatnonzero(frame_of == t)
        if len(rows) < 2:
            continue
        pairs = cKDTree(p_um[rows]).query_pairs(max_pair_um, output_type="ndarray")
        if len(pairs):
            out.append(rows[pairs])
    return np.vstack(out) if out else np.zeros((0, 2), int)


@torch.no_grad()
def score_pairs(coords, zarr_path, model, cfg, device="cuda", batch_size=512,
                fp16=True, max_pair_um=9.0, pairs=None):
    """`(pairs, probs)` -- one merge probability per close pair."""
    coords = np.asarray(coords)
    if pairs is None:
        pairs = candidate_pairs(coords, max_pair_um)
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    if len(pairs) == 0:
        return pairs, np.zeros(0, np.float32)

    n_frames, patch_z, patch_yx = cfg["n_frames"], cfg["patch_z"], cfg["patch_yx"]
    half = n_frames // 2
    vol = zarr.open_group(str(zarr_path), mode="r")["0"]
    n_t = vol.shape[0]

    frame_of = coords[pairs[:, 0], 0].astype(int)
    probs = np.zeros(len(pairs), np.float32)

    raw: dict[int, np.ndarray] = {}
    norm: dict[tuple[int, int], tuple[float, float]] = {}

    def frame(t):
        if t not in raw:
            raw[t] = np.asarray(vol[t], np.float32)
            for old in [k for k in raw if k < t - 2 * half - 1]:
                del raw[old]
                for key in [k for k in norm if k[0] == old]:
                    del norm[key]
        return raw[t]

    def slab_norm(t, z0):
        """`(lo, hi)` percentiles of the slab starting at `z0`, as in training."""
        key = (t, z0)
        if key not in norm:
            v = frame(t)
            lo_z, hi_z = max(0, z0), min(v.shape[0], z0 + patch_z)
            slab = np.zeros((patch_z, v.shape[1], v.shape[2]), np.float32)
            if lo_z < hi_z:
                slab[lo_z - z0:hi_z - z0] = v[lo_z:hi_z]
            norm[key] = tuple(np.percentile(slab, [1.0, 99.99]))
        return norm[key]

    buf_clips, buf_geom, buf_rows = [], [], []

    def flush():
        if not buf_clips:
            return
        clips = torch.from_numpy(np.stack(buf_clips)).to(device, non_blocking=True)
        geom = torch.from_numpy(np.stack(buf_geom)).to(device, non_blocking=True)
        with torch.autocast("cuda", torch.float16,
                            enabled=fp16 and str(device).startswith("cuda")):
            logits = model(clips, geom)
        probs[np.asarray(buf_rows)] = torch.sigmoid(logits.float()).cpu().numpy()
        buf_clips.clear()
        buf_geom.clear()
        buf_rows.clear()

    for t in np.unique(frame_of):
        here = np.flatnonzero(frame_of == t)
        a = coords[pairs[here, 0], 1:].astype(float)
        b = coords[pairs[here, 1], 1:].astype(float)
        geo = [pair_geometry(a[i], b[i]) for i in range(len(here))]
        angles = np.array([g[0] for g in geo], np.float32)
        centres = np.array([g[1] for g in geo], np.float32)
        feats = np.array([g[2] for g in geo], np.float32)

        clip = np.zeros((len(here), n_frames, patch_z, patch_yx, patch_yx), np.float32)
        for k, dt in enumerate(range(-half, half + 1)):
            tt = int(t) + dt
            if tt < 0 or tt >= n_t:
                continue        # off the ends of the video: genuinely nothing
            v = frame(tt)
            # Normalise per slab, as training does. Group the pairs by the slab
            # they land in so each (frame, z0) is measured once.
            z0s = np.rint(centres[:, 0]).astype(int) - patch_z // 2
            cut = pair_patches(v, centres, angles, (patch_z, patch_yx, patch_yx))
            for z0 in np.unique(z0s):
                sel = z0s == z0
                lo, hi = slab_norm(tt, int(z0))
                clip[sel, k] = np.clip((cut[sel] - lo) / (hi - lo + 1e-6), 0.0, 1.0)

        for i in range(len(here)):
            buf_clips.append(clip[i])
            buf_geom.append(feats[i])
            buf_rows.append(int(here[i]))
            if len(buf_clips) >= batch_size:
                flush()
    flush()
    return pairs, probs
