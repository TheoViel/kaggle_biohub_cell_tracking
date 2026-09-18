"""Rebuilding a trained NETr from a checkpoint, and embedding detections with it.

The geometry lives in the checkpoint's ``model_config``, written by
``main_contrastive`` for exactly this purpose, so nothing here needs the run's
``args.json`` or the flags it was launched with.

One rule governs how the work is cut up. NETr is *contextual*: with the frame
transformer on, a cell's embedding depends on every other cell in the frame
through attention. So a frame is embedded in a single forward pass and never
split into chunks -- chunking would silently change the answer, giving each cell
a different neighbourhood than the one it had at training time. That is why
there is no batch-size knob here: the batch is the frame, and the frame is
however many detections it holds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from embeddings.model import NETr
from embeddings.patches import (
    extract_patches,
    normalise_and_extract_torch,
    normalise_coords,
    normalise_frame,
)


def _opt_int(value):
    """A flag that defaults to None reaches argparse untyped, so it may be str."""
    if value is None or value == "" or str(value).lower() == "none":
        return None
    return int(value)


def load_netr(ckpt_path, device="cpu", verbose: bool = True):
    """Rebuild the NETr a checkpoint was trained as and load its weights.

    Returns ``(model, model_config)``. The model is in eval mode on ``device``.

    Defaults for absent keys reproduce the behaviour of the run that wrote the
    checkpoint, not today's: a checkpoint from before ``min_tokens`` existed was
    trained with no backbone truncation, so it must be rebuilt with 0 rather
    than the current default of 3, or the weights would not fit the model.
    """
    ckpt_path = Path(ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(state.get("model_config") or {})
    sd = state.get("model_state_dict", state)

    patch_zyx = tuple(cfg.get("patch_zyx") or (3, 64, 64))
    model = NETr(
        d_model=int(cfg.get("d_model", 256)),
        n_layers=int(cfg.get("n_layers", 4)),
        n_heads=int(cfg.get("n_heads", 4)),
        dim_feedforward=int(cfg.get("dim_feedforward", 512)),
        dropout=float(cfg.get("dropout", 0.1)),
        patch_z=int(patch_zyx[0]),
        patch_size=int(patch_zyx[1]),
        embed_dim=_opt_int(cfg.get("embed_dim")),
        backbone=cfg.get("backbone", "vit_small_patch16_224.dino"),
        # The weights come from the checkpoint, so fetching pretrained ones
        # first would only be a slow way to overwrite them -- and it would need
        # network access that inference should not require.
        pretrained=False,
        use_visual=bool(cfg.get("use_visual", True)),
        use_position=bool(cfg.get("use_position", True)),
        use_transformer=bool(cfg.get("use_transformer", True)),
        frame_aggregation=cfg.get("frame_aggregation", "mean"),
        min_tokens=int(cfg.get("min_tokens", 0) or 0),
        verbose=0,
    )
    # strict, so a geometry the config failed to describe is a loud failure
    # here rather than silently random weights and plausible-looking embeddings.
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    if getattr(model, "channels_last", False):
        model = model.to(memory_format=torch.channels_last)

    if verbose:
        branches = "+".join(
            b for b, on in (("visual", model.use_visual),
                            ("position", model.use_position)) if on
        )
        if not model.use_transformer:
            branches += ", no frame attention"
        print(f"-> {ckpt_path}")
        print(f"   NETr [{branches}]  backbone={cfg.get('backbone')}  "
              f"patch={patch_zyx}x{cfg.get('n_frames', 1)}  "
              f"({model.count_parameters() / 1e6:.1f}M params)")
    return model, cfg


@torch.no_grad()
def embed_frame(
    model,
    volume: np.ndarray,      # (Z, Y, X) raw frame
    coords: np.ndarray,      # (N, 3) z/y/x in image voxels
    patch_zyx,
    device,
    n_frames: int = 1,
    fp16: bool = True,
) -> np.ndarray:
    """(N, D) L2-normalised embeddings for every detection in one frame.

    The whole frame goes through in one pass -- see the module docstring.
    """
    if len(coords) == 0:
        return np.zeros((0, 0), np.float32)

    dev = torch.device(device)
    if dev.type == "cuda":
        # Normalisation and the patch gather on the device: 204 ms of CPU a
        # frame against an 11 ms forward, on data that was going there anyway.
        # See `normalise_and_extract_torch`.
        patch_t = normalise_and_extract_torch(
            volume, coords, patch_zyx, n_frames=n_frames, device=dev
        )
    else:
        patch_t = torch.from_numpy(extract_patches(
            normalise_frame(volume), coords, patch_zyx, n_frames=n_frames
        )).to(dev)
    norm = normalise_coords(coords, volume.shape)
    use_fp16 = bool(fp16) and dev.type == "cuda"
    with torch.amp.autocast(
        device_type=dev.type, dtype=torch.float16, enabled=use_fp16
    ):
        z = model(
            patch_t[None],
            torch.from_numpy(norm)[None].to(device),
            torch.ones(1, len(coords), dtype=torch.bool, device=device),
        )
    return z[0].float().cpu().numpy()


@torch.no_grad()
def embed_sample(
    model,
    volume,                  # zarr array or ndarray, indexed by t
    coords: np.ndarray,      # (N, 4) t/z/y/x, the tracking pipeline's own rows
    patch_zyx,
    device,
    n_frames: int = 1,
    fp16: bool = True,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed one sample's detections.

    Returns ``(coords_kept, embeddings)`` with the embeddings **row-aligned to
    the coords**, which is the contract the rest of the pipeline reads: row i of
    the embeddings is the detection on row i of the coords. Frames are visited
    in order but the rows are scattered back to their original positions, so the
    caller never has to re-derive the ordering.
    """
    times = coords[:, 0].astype(int)
    frames = np.unique(times)
    if max_frames is not None:
        frames = frames[:max_frames]

    keep = np.isin(times, frames)
    rows_of = {int(t): np.flatnonzero(times == t) for t in frames}

    out: np.ndarray | None = None
    for t in frames:
        rows = rows_of[int(t)]
        emb = embed_frame(
            model, np.asarray(volume[int(t)]), coords[rows, 1:4].astype(np.float32),
            patch_zyx, device, n_frames=n_frames, fp16=fp16,
        )
        if out is None:                      # width is known only once embedded
            out = np.zeros((len(coords), emb.shape[1]), np.float32)
        out[rows] = emb

    if out is None:                          # no frame survived max_frames
        return coords[keep], np.zeros((int(keep.sum()), 0), np.float32)
    return coords[keep], out[keep]
