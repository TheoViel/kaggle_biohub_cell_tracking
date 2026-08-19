from pathlib import Path

import numpy as np
import torch
import zarr
from tqdm import tqdm

from model_zoo.models_track import UNetNodeTransformer, define_track_model
from tracking.post_process import detect_cells_pooled, pool_kernel_from_um
from tracking.targets import pos_embed_numpy
from tracking_cellmot.io import open_dataset



class PredictConfig:
    """Inference hyper-parameters.

    Node selection should mirror what training and validation used — the model
    was fitted against the node sets those settings produce.
    """

    det_tta = False

    # --- Nodes ---
    nms = "greedy"           # "greedy" (exact radius) or "maxpool" (grid-quantised)
    nms_radius_um = 7.0      # greedy only
    pool_kernel_um = 5.0     # maxpool only
    det_threshold = 0.25

    # --- Edges ---
    edge_activation = "softmax"
    edge_selection = "argmax"
    edge_threshold = 0.0

    max_parents_per_node = 1
    max_children_per_node = 2


def _select_edges(
    probs: np.ndarray,        # (n_src, n_tgt) parent x child probabilities
    cfg,
) -> list[tuple[float, int, int]]:
    """Pick (prob, src_idx, tgt_idx) links subject to the capacity limits.

    ``argmax`` walks children in descending confidence and gives each one its
    best parent that still has capacity, falling back down that child's parent
    ranking rather than dropping the child entirely.
    """
    n_src, n_tgt = probs.shape
    if n_src == 0 or n_tgt == 0:
        return []

    children_count: dict[int, int] = {}
    parents_count: dict[int, int] = {}
    edges: list[tuple[float, int, int]] = []

    def _take(i: int, j: int, p: float) -> bool:
        if (
            cfg.max_children_per_node is not None
            and children_count.get(i, 0) >= cfg.max_children_per_node
        ):
            return False
        if (
            cfg.max_parents_per_node is not None
            and parents_count.get(j, 0) >= cfg.max_parents_per_node
        ):
            return False
        edges.append((p, i, j))
        children_count[i] = children_count.get(i, 0) + 1
        parents_count[j] = parents_count.get(j, 0) + 1
        return True

    if cfg.edge_selection == "argmax":
        ranked = np.argsort(-probs, axis=0)                  # parents, best first per child
        best = probs[ranked[0], np.arange(n_tgt)]            # (n_tgt,)
        for j in np.argsort(-best):                          # most confident child first
            j = int(j)
            for rank in range(n_src):
                if (
                    cfg.max_parents_per_node is not None
                    and parents_count.get(j, 0) >= cfg.max_parents_per_node
                ):
                    break
                i = int(ranked[rank, j])
                p = float(probs[i, j])
                # p == 0 means kNN-gated (softmax of -inf): never link these,
                # even if every plausible parent is already at capacity.
                if p <= 0.0 or p < cfg.edge_threshold:
                    break                                    # rest of the column is worse
                _take(i, j, p)
        return edges

    candidates = sorted(
        (
            (float(probs[i, j]), i, j)
            for i in range(n_src)
            for j in range(n_tgt)
            if probs[i, j] > cfg.edge_threshold
        ),
        reverse=True,
    )
    for p, i, j in candidates:
        _take(i, j, p)
    return edges


def load_track_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
) -> UNetNodeTransformer:
    """Load a UNetNodeTransformer from a checkpoint.

    The checkpoint must contain a 'model_config' key (saved by train.py).
    """
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    cfg = ckpt.get("model_config", {})
    unet_pool_stride = cfg.get("unet_pool_stride")
    # unet_features decides whether the contrastive branch is built at all, so
    # it has to come from the checkpoint — building it differently here would
    # give the transformer a different input width than it was trained with.
    model = define_track_model(
        unet_out_channels=cfg.get("unet_out_channels", 32),
        unet_layers=cfg.get("unet_layers", [32, 64, 128]),
        unet_pool_stride=tuple(unet_pool_stride) if unet_pool_stride else 2,
        unet_n_decode_steps=cfg.get("unet_n_decode_steps"),
        transformer_hidden=cfg.get("transformer_hidden", 128),
        transformer_heads=cfg.get("transformer_heads", 4),
        transformer_blocks=cfg.get("transformer_blocks", 4),
        pos_feat_dim=cfg.get("pos_feat_dim", 0),
        edge_knn_k=cfg.get("edge_knn_k"),
        edge_geom_feats=cfg.get("edge_geom_feats", True),
        unet_features=tuple(cfg.get("unet_features", ("detection",)) or ()),
        contrastive=cfg.get("contrastive"),
        gradient_checkpointing=False,
    )

    state = ckpt.get("model_state_dict", ckpt)
    # Strip any DataParallel "unet.module." prefix
    state = {k.replace("unet.module.", "unet.", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def _load_frame(zarr_arr, t: int) -> torch.Tensor:
    """Load one full-resolution frame from zarr (no normalisation)."""
    raw = zarr_arr[t].astype(np.float32)
    return torch.from_numpy(raw)


@torch.no_grad()
def predict_video(
    model: UNetNodeTransformer,
    ds_path: str | Path,
    device: str | torch.device = "cuda",
    cfg: dict | None = None,
    window_size: int = 2,
    max_frames: int | None = None,
    progress: bool = True,
    downscale_factor: int = 1,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    """Run inference on a single video using sliding windows of W frames.

    Windows slide with stride ``W - 1`` so every consecutive pair is covered
    exactly once. UNet features from each window are reused for edge
    prediction on all ``W - 1`` consecutive pairs within the window.

    Returns
    -------
    coords : np.ndarray
        Shape (N, 4) — columns [t, z, y, x] in original resolution.
    edges : list of (src_idx, tgt_idx, prob, distance) tuples, indexing rows
        of ``coords`` (distance in detection-grid voxel units).
    """
    cfg = cfg or PredictConfig()
    device = torch.device(device)

    ds = open_dataset(ds_path, normalize=False, load_image=False)
    if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
        raise ValueError(f"Zarr attrs missing image_statistics.quantiles for {ds_path}")
    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low = float(ds.quantiles["0.001"])
    q_high = float(ds.quantiles["0.999"])

    T = ds.image_shape[0] if max_frames is None else min(ds.image_shape[0], max_frames)
    image_shape = (T,) + ds.image_shape[1:]
    W = window_size

    # Measured from the model's own output on the first window: how much
    # the encoder downsamples (Z, Y, X). Constant across windows since
    # every frame in the video has the same spatial shape.
    unet_scaling: tuple[int, int, int] | None = None
    out_shape: tuple[int, int, int] | None = None
    pool_k: tuple[int, int, int] | None = None

    # Running node registry — each entry records the frame-t detections.
    # coord_offset[t] = (start, end) half-open range into the stacked array.
    seen_frames: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    coord_lists: list[np.ndarray] = []
    coord_offset: dict[int, tuple[int, int]] = {}
    global_node_count = 0
    all_edges: list[tuple[int, int, float, float]] = []

    # Sliding windows with stride W-1 cover every consecutive pair exactly once.
    stride = max(W - 1, 1)
    window_starts = list(range(0, T - W + 1, stride))
    # Ensure the very last pair (T-2 -> T-1) is covered.
    if not window_starts or window_starts[-1] + W < T:
        last = max(T - W, 0)
        if not window_starts or last != window_starts[-1]:
            window_starts.append(last)

    det_logits_list: list[torch.Tensor] = []
    for ws in tqdm(window_starts, desc="windows", leave=False, disable=not progress):
        frame_indices = list(range(ws, ws + W))

        # --- UNet encode (single window, batch_size=1), full resolution ---
        imgs = torch.stack(
            [_load_frame(zarr_arr, t) for t in frame_indices]
        )  # (W, Z, Y, X)
        # Quantile normalisation (0.1%-99.9%), matching tracking/dataset.py.
        imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0, 4.0)

        if downscale_factor > 1:
            imgs = imgs[:, ::downscale_factor, ::downscale_factor, ::downscale_factor]
            image_shape = (T,) + tuple(imgs.shape[1:])

        imgs = imgs.unsqueeze(0).to(device)  # (1, W, Z, Y, X)

        unet_out, det_logits = model.encode(imgs)
        det_logits_list.append(torch.stack(det_logits, 0).detach().cpu())

        # The maps the edge transformer reads its node features from, in the
        # order the model was trained with. Empty when unet_features is empty,
        # i.e. a geometry-only tracker.
        node_feat_maps = [
            unet_out if source == "detection" else model.encode_embed(imgs)
            for source in model.unet_features
        ]

        if unet_scaling is None:
            d, h, w = imgs.shape[-3:]
            d_out, h_out, w_out = unet_out.shape[-3:]
            unet_scaling = (d // d_out, h // h_out, w // w_out)
            out_shape = tuple(s // u for s, u in zip(image_shape[-3:], unet_scaling))
            voxel_size_out = tuple(
                s * downscale_factor * u for s, u in zip(ds.scale, unet_scaling)
            )
            pool_k = pool_kernel_from_um(cfg.pool_kernel_um, voxel_size_out)

        # Detection TTA: average logits over Y/X flips.
        if cfg.det_tta:
            for dims in [(-1,), (-2,), (-2, -1)]:
                imgs_flip = imgs.flip(dims)
                _, det_flip = model.encode(imgs_flip)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                del imgs_flip, det_flip
            for f in range(W):
                det_logits[f] = det_logits[f] / 4

        # imgs is needed until the edge pass below when a contrastive branch
        # exists, so it is freed at the end of the window instead of here.

        # --- Detect cells in each frame (dedup across windows) ---
        # Coordinates come out in detection-grid space (same grid as
        # det_logits / unet_out), matching tracking/targets.py.
        for f_idx, t in enumerate(frame_indices):
            if t in seen_frames:
                continue
            arr = detect_cells_pooled(
                det_logits[f_idx][0],
                t,
                cfg.det_threshold,
                nms=cfg.nms,
                pool_kernel=pool_k,
                voxel_size=voxel_size_out,
                nms_radius_um=cfg.nms_radius_um,
            )
            coord_offset[t] = (global_node_count, global_node_count + len(arr))
            global_node_count += len(arr)
            coord_lists.append(arr)
            seen_frames.add(t)

        coords_so_far = (
            np.concatenate(coord_lists)
            if coord_lists
            else np.empty((0, 4), dtype=np.int16)
        )

        pos_shape = (W,) + out_shape

        # --- Edge prediction for each consecutive pair in the window ---
        for f_idx in range(W - 1):
            t_src, t_tgt = frame_indices[f_idx], frame_indices[f_idx + 1]
            if (t_src, t_tgt) in seen_pairs:
                continue
            seen_pairs.add((t_src, t_tgt))

            if t_src not in coord_offset or t_tgt not in coord_offset:
                continue
            s_src, e_src = coord_offset[t_src]
            s_tgt, e_tgt = coord_offset[t_tgt]
            if e_src == s_src or e_tgt == s_tgt:
                continue

            c_src = coords_so_far[s_src:e_src]
            c_tgt = coords_so_far[s_tgt:e_tgt]
            n_src, n_tgt = len(c_src), len(c_tgt)
            idx_src = np.arange(s_src, e_src, dtype=np.int64)
            idx_tgt = np.arange(s_tgt, e_tgt, dtype=np.int64)

            # Detection-grid coords — predict_edges expects the same space
            # detect_and_match produces during training, no rescaling.
            p_coords_src = (
                torch.from_numpy(c_src[:, 1:].astype(np.float32))
                .unsqueeze(0)
                .to(device)
            )
            p_coords_tgt = (
                torch.from_numpy(c_tgt[:, 1:].astype(np.float32))
                .unsqueeze(0)
                .to(device)
            )

            # Window-relative time (f_idx, f_idx+1) normalised by W, matching training.
            c_src_rel = c_src.copy()
            c_src_rel[:, 0] = f_idx
            c_tgt_rel = c_tgt.copy()
            c_tgt_rel[:, 0] = f_idx + 1
            p_pos_src = (
                torch.from_numpy(pos_embed_numpy(c_src_rel, pos_shape))
                .unsqueeze(0)
                .to(device)
            )
            p_pos_tgt = (
                torch.from_numpy(pos_embed_numpy(c_tgt_rel, pos_shape))
                .unsqueeze(0)
                .to(device)
            )

            p_mask_src = torch.ones(1, n_src, dtype=torch.bool, device=device)
            p_mask_tgt = torch.ones(1, n_tgt, dtype=torch.bool, device=device)

            # Same node-feature stack as training: every map named in
            # unet_features, concatenated in that order.
            unet_feat_src = torch.cat(
                [
                    model._index_features(maps[:, f_idx], p_coords_src, p_mask_src)
                    for maps in node_feat_maps
                ],
                dim=-1,
            ) if node_feat_maps else p_coords_src.new_zeros((1, n_src, 0))
            unet_feat_tgt = torch.cat(
                [
                    model._index_features(maps[:, f_idx + 1], p_coords_tgt, p_mask_tgt)
                    for maps in node_feat_maps
                ],
                dim=-1,
            ) if node_feat_maps else p_coords_tgt.new_zeros((1, n_tgt, 0))
            edge_logits_pair = model.predict_edges(
                unet_feat_src,
                unet_feat_tgt,
                p_coords_src,
                p_coords_tgt,
                p_pos_src,
                p_pos_tgt,
                p_mask_src,
                p_mask_tgt,
            )  # (1, n_src, n_tgt)

            raw = edge_logits_pair[0]
            if cfg.edge_activation == "softmax":
                probs = torch.softmax(raw, dim=0).cpu().numpy()
            else:
                probs = torch.sigmoid(raw).cpu().numpy()

            for prob, i, j in _select_edges(probs, cfg):
                gi, gj = int(idx_src[i]), int(idx_tgt[j])
                dist = float(
                    np.linalg.norm(
                        coords_so_far[gi, 1:].astype(np.float32)
                        - coords_so_far[gj, 1:].astype(np.float32)
                    )
                )
                all_edges.append((gi, gj, prob, dist))

        del unet_out

    coords = (
        np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
    )
    # Scale detection-grid coords back to original resolution.
    coords = coords.astype(np.float32)
    if unet_scaling is not None:
        coords[:, 1:] *= np.array(unet_scaling, dtype=np.float32)
    if downscale_factor > 1:
        coords[:, 1:] *= downscale_factor

    det_logits = torch.stack(det_logits_list)
    return coords.astype(np.int16), all_edges, det_logits
