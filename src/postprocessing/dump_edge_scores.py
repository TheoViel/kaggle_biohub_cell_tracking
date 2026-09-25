"""Re-score every plausible parent for every detection, not just the winner.

The saved npz keeps one parent per child — the tracker's argmax. That throws
away exactly what post-processing needs: what the model thought of the *other*
candidates. Where the tracker is wrong the true parent is almost always the
runner-up, so the runner-up's score is the whole question.

This re-runs the encoder over the same nodes (coordinates are taken from the
npz, not re-detected, so everything stays comparable) and saves the top-K
parents per child with their softmax probabilities.

Run:  python src/postprocessing/dump_edge_scores.py
"""

import json

import numpy as np
import torch
import zarr

from common import EXPERIMENT, PRED_NPZ, RESULTS, SAMPLE, load_prediction
from main_predict import _reconcile_arch
from params import TRAIN_DIR
from tracking.data import normalize_frames, resolve_normalization
from tracking.main import build_model
from tracking.targets import pos_embed_numpy
from tracking_cellmot.io import open_dataset
from utils.logger import Config

LOG_FOLDER = "../logs/" + EXPERIMENT.replace("_", "/") + "/"
TOP_K = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    cfg = Config(json.load(open(LOG_FOLDER + "args.json")))
    state = torch.load(LOG_FOLDER + "fold_0/ckpt_final.pt", map_location="cpu")
    state = state.get("model_state_dict", state)
    state = {k.replace("unet.module.", "unet.", 1): v for k, v in state.items()}
    cfg = _reconcile_arch(cfg, state, verbose=True)
    model = build_model(cfg, None)
    model.load_state_dict(state, strict=True)
    return model.eval().to(DEVICE), cfg


@torch.no_grad()
def main():
    model, cfg = load_model()
    coords, _ = load_prediction()

    ds = open_dataset(f"{TRAIN_DIR}/{SAMPLE}.zarr", normalize=False, load_image=False)
    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    # Follow the model's own config: pinning quantiles here would score under a
    # different normalisation than the one it was trained with.
    norm = resolve_normalization(getattr(cfg, "normalization", None))

    frames = sorted(np.unique(coords[:, 0]).tolist())
    rows_by_t = {t: np.flatnonzero(coords[:, 0] == t) for t in frames}

    scaling = None
    out = []  # (t, child row, parent row, prob)
    for t in frames[:-1]:
        if t + 1 not in rows_by_t:
            continue
        imgs = torch.from_numpy(
            normalize_frames(np.stack([zarr_arr[t], zarr_arr[t + 1]]), norm, ds.quantiles)
        )
        imgs = imgs.unsqueeze(0).to(DEVICE)

        unet_out, _, _, _ = model.encode(imgs)
        if scaling is None:
            scaling = tuple(s // o for s, o in zip(imgs.shape[-3:], unet_out.shape[-3:]))
            print(f"-> unet scaling {scaling}, out shape {tuple(unet_out.shape[-3:])}")
        scale_arr = np.asarray(scaling, dtype=np.float32)

        here, there = rows_by_t[t], rows_by_t[t + 1]
        # The npz stores coordinates in original resolution; the edge head was
        # trained on detection-grid coordinates, the ones detect_and_match makes.
        grid_src = coords[here, 1:].astype(np.float32) / scale_arr
        grid_tgt = coords[there, 1:].astype(np.float32) / scale_arr

        pos_shape = (2,) + tuple(unet_out.shape[-3:])

        def prep(grid, frame_index):
            rel = np.concatenate(
                [np.full((len(grid), 1), frame_index, np.float32), grid], axis=1
            )
            return (
                torch.from_numpy(grid).unsqueeze(0).to(DEVICE),
                torch.from_numpy(pos_embed_numpy(rel, pos_shape)).unsqueeze(0).to(DEVICE),
            )

        p_src, pos_src = prep(grid_src, 0)
        p_tgt, pos_tgt = prep(grid_tgt, 1)
        m_src = torch.ones(1, len(here), dtype=torch.bool, device=DEVICE)
        m_tgt = torch.ones(1, len(there), dtype=torch.bool, device=DEVICE)

        maps = [
            unet_out if source == "detection" else model.encode_embed(imgs)
            for source in model.unet_features
        ]
        feat_src = (
            torch.cat([model._index_features(m[:, 0], p_src, m_src) for m in maps], -1)
            if maps else p_src.new_zeros((1, len(here), 0))
        )
        feat_tgt = (
            torch.cat([model._index_features(m[:, 1], p_tgt, m_tgt) for m in maps], -1)
            if maps else p_tgt.new_zeros((1, len(there), 0))
        )

        logits = model.predict_edges(
            feat_src, feat_tgt, p_src, p_tgt, pos_src, pos_tgt, m_src, m_tgt
        )[0]
        # Softmax over parents, matching PredictConfig.edge_activation.
        probs = torch.softmax(logits, dim=0)
        k = min(TOP_K, probs.shape[0])
        top_p, top_i = probs.topk(k, dim=0)              # (k, n_tgt)
        top_p = top_p.cpu().numpy()
        top_i = top_i.cpu().numpy()

        for j in range(probs.shape[1]):
            for r in range(k):
                out.append((t, int(there[j]), int(here[top_i[r, j]]), float(top_p[r, j])))

        del unet_out, maps, imgs
        if t % 20 == 0:
            print(f"   frame pair {t} -> {t+1}   ({len(here)} x {len(there)})", flush=True)

    arr = np.asarray(out, dtype=np.float64)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{SAMPLE}_edge_candidates.npz"
    np.savez_compressed(path, candidates=arr, scaling=np.asarray(scaling))
    print(f"\n-> {len(arr)} candidate edges over {len(frames)-1} pairs")
    print(f"-> wrote {path}")


if __name__ == "__main__":
    main()
