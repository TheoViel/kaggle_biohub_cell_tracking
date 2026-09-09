"""Score every detected node with "does this cell divide here", frame by frame.

The training loader reads three frames per *sample*, which costs ~69 ms and is
fine when a sample is a training example. At inference the same volume is asked
about for every cell in it, so the read is amortised instead: frames are walked
in order, each is loaded once into a rolling three-frame window, and every node
in the centre frame is cropped out of what is already in memory.

The output is one probability per node, aligned to the prediction npz's `coords`
rows, which is what `edge_pp.propose_divisions` gates on.

**Which model scores which embryo matters.** Fold `i` held out `EMBRYOS[i]`, so
a stack must be scored by the fold that did *not* train on it, exactly as the
edge transformer is applied. `load_fold_models` returns the mapping and
`score_stack` refuses a stack it has no out-of-fold model for, rather than
quietly using a model that has seen it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import zarr

from division_cls.model import DivisionNet
from embeddings.patches import extract_patches
from params import EMBRYOS


def load_model(ckpt_path, device="cuda"):
    """`(model, cfg)` from a checkpoint written by `division_cls.train.fit`."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(state.get("model_config") or {})
    if not cfg:
        raise ValueError(f"{ckpt_path} carries no model_config")
    model = DivisionNet(
        backbone=cfg["backbone"], pretrained=False,
        patch_yx=cfg["patch_yx"], patch_z=cfg["patch_z"], n_frames=cfg["n_frames"],
        d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"],
        min_tokens=cfg["min_tokens"],
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


@torch.no_grad()
def score_stack(coords, zarr_path, model, cfg, device="cuda", batch_size=512,
                fp16=True, rows=None):
    """`(N,)` division probabilities for `coords`, or for the subset `rows`.

    `rows` restricts the work to the nodes a caller actually needs -- typically
    the parents `propose_divisions` would consider -- and the returned array is
    still full length, with `nan` where nothing was scored. Scoring everything
    is affordable but pointless when the post-processing only ever asks about a
    few percent of the nodes.
    """
    coords = np.asarray(coords)
    n_frames, patch_z, patch_yx = cfg["n_frames"], cfg["patch_z"], cfg["patch_yx"]
    half = n_frames // 2

    vol = zarr.open_group(str(zarr_path), mode="r")["0"]
    n_t = vol.shape[0]

    want = np.zeros(len(coords), bool)
    want[np.arange(len(coords)) if rows is None else np.asarray(rows, int)] = True
    frame_of = coords[:, 0].astype(int)

    out = np.full(len(coords), np.nan, np.float32)
    cache: dict[int, np.ndarray] = {}

    def frame(t):
        """One normalised frame, from a rolling window of the ones in use."""
        if t not in cache:
            v = np.asarray(vol[t], np.float32)
            lo, hi = np.percentile(v, [1.0, 99.99])
            cache[t] = np.clip((v - lo) / (hi - lo + 1e-6), 0.0, 1.0)
            for old in [k for k in cache if k < t - half - 1]:
                del cache[old]
        return cache[t]

    buf_clips, buf_rows = [], []

    def flush():
        if not buf_clips:
            return
        clips = torch.from_numpy(np.stack(buf_clips)).to(device, non_blocking=True)
        with torch.autocast("cuda", torch.float16,
                            enabled=fp16 and str(device).startswith("cuda")):
            logits = model(clips)
        out[np.asarray(buf_rows)] = torch.sigmoid(logits.float()).cpu().numpy()
        buf_clips.clear()
        buf_rows.clear()

    for t in range(n_t):
        here = np.flatnonzero((frame_of == t) & want)
        if not len(here):
            continue
        slabs = {}
        for dt in range(-half, half + 1):
            tt = t + dt
            slabs[dt] = frame(tt) if 0 <= tt < n_t else None

        for row in here:
            z, y, x = coords[row, 1:4].astype(float)
            clip = np.zeros((n_frames, patch_z, patch_yx, patch_yx), np.float32)
            for k, dt in enumerate(range(-half, half + 1)):
                v = slabs[dt]
                if v is None:
                    continue        # off the ends of the video: genuinely nothing
                z0 = int(round(z)) - patch_z // 2
                lo, hi = max(0, z0), min(v.shape[0], z0 + patch_z)
                slab = np.zeros((patch_z, v.shape[1], v.shape[2]), np.float32)
                if lo < hi:
                    slab[lo - z0:hi - z0] = v[lo:hi]
                clip[k] = extract_patches(
                    slab, np.array([[patch_z // 2, y, x]], np.float32),
                    patch_zyx=(patch_z, patch_yx, patch_yx), n_frames=1,
                )[0, 0]
            buf_clips.append(clip)
            buf_rows.append(row)
            if len(buf_clips) >= batch_size:
                flush()
    flush()
    return out


def candidate_parents(coords, edges, max_parent_um=9.0, scale=None):
    """The nodes `propose_divisions` could ever ask about.

    A parent needs exactly one committed child in the next frame and at least
    one unclaimed node near it -- everything else is unscoreable by
    construction, and scoring it is wasted GPU.
    """
    from scipy.spatial import cKDTree

    from params import SCALE

    scale = np.asarray(SCALE if scale is None else scale)
    coords = np.asarray(coords)
    edges = np.asarray(edges, float).reshape(-1, 4)
    if not len(edges):
        return np.zeros(0, int)

    p_um = coords[:, 1:] * scale
    frame_of = coords[:, 0].astype(int)
    src, tgt = edges[:, 0].astype(int), edges[:, 1].astype(int)
    n_children = np.bincount(src, minlength=len(coords))
    only = np.full(len(coords), -1, np.int64)
    single = n_children[src] == 1
    only[src[single]] = tgt[single]
    has_parent = np.zeros(len(coords), bool)
    has_parent[tgt] = True

    out = []
    for t in np.unique(frame_of):
        rows_next = np.flatnonzero(frame_of == t + 1)
        free = rows_next[~has_parent[rows_next]]
        if not len(free):
            continue
        parents = [int(r) for r in np.flatnonzero(frame_of == t)
                   if only[r] >= 0 and frame_of[only[r]] == t + 1]
        if not parents:
            continue
        tree = cKDTree(p_um[free])
        for parent in parents:
            if tree.query_ball_point(p_um[parent], max_parent_um):
                out.append(parent)
    return np.asarray(sorted(out), int)
