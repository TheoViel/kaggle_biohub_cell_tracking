from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import zarr
from tqdm import tqdm

from model_zoo.models_track import UNetNodeTransformer, define_track_model
from tracking.post_process import detect_cells_multi_threshold, pool_kernel_from_um
from tracking.targets import pos_embed_numpy
from tracking.data import resolve_normalization
from tracking_cellmot.io import open_dataset



class PredictConfig:
    """Inference hyper-parameters.

    Node selection should mirror what training and validation used — the model
    was fitted against the node sets those settings produce.
    """

    det_tta = False

    # --- Normalisation ---
    # Which stored quantiles map the raw volume into the range the detector was
    # trained on. Must match the training run's `normalization` config; the
    # checkpoint's model_config carries it so main_predict can restore it.
    normalization = None

    # --- Nodes ---
    nms = "greedy"           # "greedy" (exact radius) or "maxpool" (grid-quantised)
    nms_radius_um = 7.0      # greedy only
    pool_kernel_um = 5.0     # maxpool only
    det_threshold = 0.25
    # Move each detection onto the centre its own voxel votes for, when the
    # model was trained with the head. Ignored otherwise.
    use_offset = True

    # --- Edges ---
    # "transformer" scores every pair with the edge head. "nearest" links each
    # child to its closest parent instead and never touches the transformer —
    # which is what makes detection quality measurable while the edge head is
    # still at init, i.e. during detection pretraining.
    edge_mode = "transformer"
    max_link_um = 15.0       # nearest only: never link further than this

    edge_activation = "softmax"
    edge_selection = "argmax"
    edge_threshold = 0.0

    max_parents_per_node = 1
    max_children_per_node = 2


def _nearest_probs(
    c_src: np.ndarray,        # (n_src, 3) detection-grid coords
    c_tgt: np.ndarray,        # (n_tgt, 3)
    voxel_size: tuple[float, float, float],
    max_link_um: float,
) -> np.ndarray:              # (n_src, n_tgt)
    """Proximity scores standing in for transformer edge probabilities.

    1 at zero separation, falling linearly to 0 at ``max_link_um``. That gives
    ``_select_edges`` exactly what it needs: ranking a child's parents by this
    score is ranking them nearest-first, and anything past the cutoff lands at
    0, which it already treats as unlinkable (it stops the column at p <= 0,
    the same way it handles a kNN-gated pair).

    Distances are physical, so the cutoff means the same thing whatever the
    detection grid's stride is.
    """
    vs = np.asarray(voxel_size, dtype=np.float32)
    d = np.linalg.norm(
        c_src[:, None, :].astype(np.float32) * vs
        - c_tgt[None, :, :].astype(np.float32) * vs,
        axis=-1,
    )
    return np.clip(1.0 - d / float(max_link_um), 0.0, 1.0)


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


def _load_frame(zarr_arr, t: int) -> torch.Tensor:
    """Load one full-resolution frame from zarr (no normalisation)."""
    raw = zarr_arr[t].astype(np.float32)
    return torch.from_numpy(raw)


class _NodeRegistry:
    """The nodes and edges accumulated for one video at one detection threshold.

    Windows overlap by a frame, so both frames and consecutive pairs are
    recorded once and skipped on the next window that covers them.
    """

    def __init__(self) -> None:
        self.seen_frames: set[int] = set()
        self.seen_pairs: set[tuple[int, int]] = set()
        self.coord_lists: list[np.ndarray] = []
        self.offset_lists: list[np.ndarray] = []
        self.score_lists: list[np.ndarray] = []
        # coord_offset[t] = (start, end), a half-open range into the stacked array.
        self.coord_offset: dict[int, tuple[int, int]] = {}
        self.n_nodes = 0
        self.edges: list[tuple[int, int, float, float]] = []

    def add_frame(
        self,
        t: int,
        arr: np.ndarray,
        offsets: np.ndarray | None = None,
        scores: np.ndarray | None = None,
    ) -> None:
        """Register frame t's detections, in detection-grid space.

        ``offsets`` is the model's centroid vote at each detection, in grid
        voxels; it is carried alongside rather than folded in, because the edge
        scorer must keep reading the integer coordinates it was trained on.

        ``scores`` is the detection probability at each peak. A node that only
        just cleared the threshold is a different thing from one the head was
        certain of, and nothing downstream can recover the difference once the
        coordinates are all that is kept.
        """
        self.coord_offset[t] = (self.n_nodes, self.n_nodes + len(arr))
        self.n_nodes += len(arr)
        self.coord_lists.append(arr)
        self.offset_lists.append(
            np.zeros((len(arr), 3), dtype=np.float32) if offsets is None else offsets
        )
        self.score_lists.append(
            np.full(len(arr), np.nan, dtype=np.float32) if scores is None
            else np.asarray(scores, dtype=np.float32)
        )
        self.seen_frames.add(t)

    def coords(self) -> np.ndarray:
        """Every detection so far, as (N, 4) [t, z, y, x]."""
        if not self.coord_lists:
            return np.empty((0, 4), dtype=np.int16)
        return np.concatenate(self.coord_lists)

    def scores(self) -> np.ndarray:
        """The detection probability at every node, as (N,) float32."""
        if not self.score_lists:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self.score_lists)

    def offsets(self) -> np.ndarray:
        """The centroid vote at every detection, as (N, 3) in grid voxels."""
        if not self.offset_lists:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(self.offset_lists)


def _predict_window_edges(
    model: UNetNodeTransformer,
    reg: _NodeRegistry,
    frame_indices: list[int],
    node_feat_maps: list[torch.Tensor],
    out_shape: tuple[int, int, int],
    cfg,
    device: torch.device,
    voxel_size: tuple[float, float, float] | None = None,
) -> None:
    """Score every consecutive pair in one window, appending links to ``reg``."""
    nearest = getattr(cfg, "edge_mode", "transformer") == "nearest"
    if nearest and voxel_size is None:
        raise ValueError("edge_mode='nearest' needs voxel_size for physical distances")
    coords_so_far = reg.coords()
    W = len(frame_indices)
    pos_shape = (W,) + out_shape

    for f_idx in range(W - 1):
        t_src, t_tgt = frame_indices[f_idx], frame_indices[f_idx + 1]
        if (t_src, t_tgt) in reg.seen_pairs:
            continue
        reg.seen_pairs.add((t_src, t_tgt))

        if t_src not in reg.coord_offset or t_tgt not in reg.coord_offset:
            continue
        s_src, e_src = reg.coord_offset[t_src]
        s_tgt, e_tgt = reg.coord_offset[t_tgt]
        if e_src == s_src or e_tgt == s_tgt:
            continue

        c_src = coords_so_far[s_src:e_src]
        c_tgt = coords_so_far[s_tgt:e_tgt]
        n_src, n_tgt = len(c_src), len(c_tgt)
        idx_src = np.arange(s_src, e_src, dtype=np.int64)
        idx_tgt = np.arange(s_tgt, e_tgt, dtype=np.int64)

        if nearest:
            # No transformer, no feature indexing, no positional embeddings —
            # geometry alone. Skips straight to selection.
            probs = _nearest_probs(
                c_src[:, 1:], c_tgt[:, 1:], voxel_size, cfg.max_link_um
            )
            for prob, i, j in _select_edges(probs, cfg):
                gi, gj = int(idx_src[i]), int(idx_tgt[j])
                dist = float(
                    np.linalg.norm(
                        coords_so_far[gi, 1:].astype(np.float32)
                        - coords_so_far[gj, 1:].astype(np.float32)
                    )
                )
                reg.edges.append((gi, gj, prob, dist))
            continue

        # Detection-grid coords — predict_edges expects the same space
        # detect_and_match produces during training, no rescaling.
        p_coords_src = (
            torch.from_numpy(c_src[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
        )
        p_coords_tgt = (
            torch.from_numpy(c_tgt[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
        )

        # Window-relative time (f_idx, f_idx+1) normalised by W, matching training.
        c_src_rel = c_src.copy()
        c_src_rel[:, 0] = f_idx
        c_tgt_rel = c_tgt.copy()
        c_tgt_rel[:, 0] = f_idx + 1
        p_pos_src = (
            torch.from_numpy(pos_embed_numpy(c_src_rel, pos_shape)).unsqueeze(0).to(device)
        )
        p_pos_tgt = (
            torch.from_numpy(pos_embed_numpy(c_tgt_rel, pos_shape)).unsqueeze(0).to(device)
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
            reg.edges.append((gi, gj, prob, dist))


def _gather_offsets(
    offset_maps: list[torch.Tensor] | None,
    f_idx: int,
    arr: np.ndarray,          # (N, 4) [t, z, y, x] in detection-grid space
) -> np.ndarray | None:
    """The centroid vote at each detection, as (N, 3) in grid voxels."""
    if offset_maps is None or len(arr) == 0:
        return None
    vol = offset_maps[f_idx][0]          # (3, Z, Y, X)
    idx = torch.from_numpy(arr[:, 1:].astype(np.int64)).to(vol.device)
    for ax in range(3):
        idx[:, ax] = idx[:, ax].clamp(0, vol.shape[ax + 1] - 1)
    return vol[:, idx[:, 0], idx[:, 1], idx[:, 2]].T.float().cpu().numpy()


def _rescale_coords(
    coords: np.ndarray,
    unet_scaling: tuple[int, int, int] | None,
    downscale_factor: int,
    offsets: np.ndarray | None = None,
    det_stride: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Scale detection-grid coords back to original resolution.

    A detection is a grid *cell*, not a point. The cell with index g spans
    image voxels [s*g, s*g + s - 1], and tracking/targets.py assigns a
    ground-truth cell to it with floor(gt / s) — so the best estimate of where
    the cell actually is, is the middle of that span, not its corner.
    Multiplying by s alone leaves every detection (s - 1) / 2 voxels short: at
    the configured (1, 4, 4) grid that is a fixed -1.5 voxel (-0.61 um) error
    in y and x, which measures out at -0.59 and -0.54 um against the
    annotation (postprocessing/check_centring.py). Correcting it moves the
    run-level adjusted Jaccard from 0.8065 to 0.8130.

    With an offset head the guess stops being the middle of the span: the
    position becomes ``(g + o) * s`` for the voted offset o. The two agree when
    the head is absent — the centre of the span is o = (s - 1) / (2s), which is
    exactly the mean offset of a cell placed uniformly in it — so this is the
    same formula with a measured o in place of an assumed one.

    ``det_stride`` is the span, in image voxels, of the cell a detection
    actually names. That is the node-grid stride only while the heatmap shares
    the node grid; with an extra decoder stage the peak comes off a finer map
    and names a smaller cell, so the correction is half of *that* span.
    Assuming ``unet_scaling`` there would over-shift every detection by the
    difference — a whole image voxel in y and x at the configured geometry,
    which is most of what the finer head was added to recover.
    """
    coords = coords.astype(np.float32)
    if unet_scaling is not None:
        scaling = np.array(unet_scaling, dtype=np.float32)
        if offsets is not None:
            # The vote is measured from the corner of the node voxel it was
            # read at, so it replaces the within-voxel position rather than
            # adding to it. A no-op back when every coordinate was an integer.
            coords[:, 1:] = (np.floor(coords[:, 1:]) + offsets) * scaling
        else:
            span = (
                scaling if det_stride is None
                else np.array(det_stride, dtype=np.float32)
            )
            coords[:, 1:] = coords[:, 1:] * scaling + (span - 1.0) / 2.0
    if downscale_factor > 1:
        coords[:, 1:] = coords[:, 1:] * downscale_factor + (downscale_factor - 1.0) / 2.0
    # round, not truncate: the centres are half-integers now, and int16() would
    # throw away that half a voxel again.
    return np.round(coords).astype(np.int16)


@torch.no_grad()
def predict_video_thresholds(
    model: UNetNodeTransformer,
    ds_path: str | Path,
    thresholds: Sequence[float],
    device: str | torch.device = "cuda",
    cfg: dict | None = None,
    window_size: int = 2,
    max_frames: int | None = None,
    progress: bool = True,
    downscale_factor: int = 1,
    return_det_logits: bool = True,
    nms_radii: Sequence[float] | None = None,
) -> tuple[dict[tuple[float, float], tuple[np.ndarray, list]], torch.Tensor | None]:
    """Run inference on a single video over a grid of NMS radii x thresholds.

    Windows slide with stride ``W - 1`` so every consecutive pair is covered
    exactly once. UNet features from each window are reused for edge
    prediction on all ``W - 1`` consecutive pairs within the window.

    Neither the threshold nor the NMS radius changes the UNet forward — the
    cost of this function — so it is shared across the whole grid. Beyond that
    the two sweep differently:

    * Thresholds share the NMS pass as well: one run at the lowest one answers
      all of them (see ``detect_cells_multi_threshold``).
    * Radii do not. Greedy NMS results do not nest across radius — a peak
      suppressed at a small radius can free a neighbour that a larger radius
      then keeps — so each radius gets its own pass over the same probabilities.

    Only the NMS and the edge pass repeat, and both are small beside the
    forward, so an extra radius still costs a fraction of a full run.

    ``cfg.det_threshold`` and ``cfg.nms_radius_um`` are ignored;  ``thresholds``
    and ``nms_radii`` replace them. ``nms_radii`` defaults to
    ``[cfg.nms_radius_um]``.

    Returns
    -------
    per_key : dict
        ``(nms_radius_um, threshold)`` -> ``(coords, edges, node_probs)``.
        ``node_probs`` is the (N,) detection probability at each node,
        row-aligned with ``coords``.
    det_logits : torch.Tensor or None
        Stacked raw detection logits per window, or None if not requested. They
        hold a full-resolution volume per window, so ask for them deliberately.
    """
    cfg = cfg or PredictConfig()
    device = torch.device(device)

    thresholds = sorted({float(th) for th in thresholds})
    if not thresholds:
        raise ValueError("thresholds must not be empty")

    radii = sorted({float(r) for r in (nms_radii or [cfg.nms_radius_um])})
    if not radii:
        raise ValueError("nms_radii must not be empty")
    if len(radii) > 1 and cfg.nms != "greedy":
        # maxpool takes its neighbourhood from pool_kernel_um and never reads
        # the radius, so the sweep would run N identical passes.
        raise ValueError(
            f"nms_radii sweeps the greedy NMS radius, but nms={cfg.nms!r}; "
            "sweep pool_kernel_um instead, or pass --nms greedy"
        )

    ds = open_dataset(ds_path, normalize=False, load_image=False)
    # Must match what training normalised with, or every intensity the detector
    # learned a threshold against is shifted underneath it.
    norm = resolve_normalization(getattr(cfg, "normalization", None))
    missing = [q for q in (norm["low_q"], norm["high_q"]) if q not in ds.quantiles]
    if missing:
        raise ValueError(
            f"Zarr attrs missing image_statistics.quantiles {missing} for {ds_path}"
        )
    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low = float(ds.quantiles[norm["low_q"]])
    q_high = float(ds.quantiles[norm["high_q"]])
    clamp_max = float(norm["clamp_max"])

    T = ds.image_shape[0] if max_frames is None else min(ds.image_shape[0], max_frames)
    image_shape = (T,) + ds.image_shape[1:]
    W = window_size

    # Measured from the model's own output on the first window: how much
    # the encoder downsamples (Z, Y, X). Constant across windows since
    # every frame in the video has the same spatial shape.
    unet_scaling: tuple[int, int, int] | None = None
    det_stride: tuple[float, float, float] | None = None
    out_shape: tuple[int, int, int] | None = None
    pool_k: tuple[int, int, int] | None = None

    # One node registry per (radius, threshold): the node sets differ, so the
    # indices the edges refer to do too.
    registries = {(r, th): _NodeRegistry() for r in radii for th in thresholds}

    # Sliding windows with stride W-1 cover every consecutive pair exactly once.
    stride = max(W - 1, 1)
    window_starts = list(range(0, T - W + 1, stride))
    # Ensure the very last pair (T-2 -> T-1) is covered.
    if not window_starts or window_starts[-1] + W < T:
        last = max(T - W, 0)
        if not window_starts or last != window_starts[-1]:
            window_starts.append(last)

    # Decided once, from the model rather than from the numbers it produces: a
    # trained head is free to vote ~0, and that must still mean "the cell is at
    # the corner of its voxel", not "there is no head".
    use_offset = bool(getattr(cfg, "use_offset", True)) and (
        getattr(model, "offset_head", None) is not None
    )

    det_logits_list: list[torch.Tensor] = []
    for ws in tqdm(window_starts, desc="windows", leave=False, disable=not progress):
        frame_indices = list(range(ws, ws + W))

        # --- UNet encode (single window, batch_size=1), full resolution ---
        imgs = torch.stack(
            [_load_frame(zarr_arr, t) for t in frame_indices]
        )  # (W, Z, Y, X)
        # Quantile normalisation, matching tracking/dataset.py.
        imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0, clamp_max)

        if downscale_factor > 1:
            imgs = imgs[:, ::downscale_factor, ::downscale_factor, ::downscale_factor]
            image_shape = (T,) + tuple(imgs.shape[1:])

        imgs = imgs.unsqueeze(0).to(device)  # (1, W, Z, Y, X)

        unet_out, det_logits, _ = model.encode(imgs)
        offset_maps = model.encode_offsets(unet_out) if use_offset else None
        if return_det_logits:
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
            # The heatmap may sit on a finer grid than the node features do, in
            # which case NMS runs there and the peaks are divided back onto the
            # node grid — exactly what detect_and_match does in training.
            d_det, h_det, w_det = det_logits[0].shape[-3:]
            det_scaling = np.array(
                [d_det // d_out, h_det // h_out, w_det // w_out], dtype=np.float32
            )
            voxel_size_det = tuple(
                v / float(s) for v, s in zip(voxel_size_out, det_scaling)
            )
            # In image voxels, for the half-cell centring _rescale_coords applies.
            det_stride = tuple(
                float(u) / float(s) for u, s in zip(unet_scaling, det_scaling)
            )
            pool_k = pool_kernel_from_um(cfg.pool_kernel_um, voxel_size_det)

            # print("UNET SCALING: ", unet_scaling)
            # print("OUT SHAPE: ", out_shape)
            # print("VOXEL SIZE OUT: ", voxel_size_out)
            # print("POOL K: ", pool_k)

        # Detection TTA: average logits over Y/X flips.
        if cfg.det_tta:
            for dims in [(-1,), (-2,), (-2, -1)]:
                imgs_flip = imgs.flip(dims)
                _, det_flip, _ = model.encode(imgs_flip)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                del imgs_flip, det_flip
            for f in range(W):
                det_logits[f] = det_logits[f] / 4

        # imgs is needed until the edge pass below when a contrastive branch
        # exists, so it is freed at the end of the window instead of here.

        # --- Detect cells in each frame (dedup across windows) ---
        # Coordinates come out in detection-grid space (same grid as
        # det_logits / unet_out), matching tracking/targets.py. One NMS pass
        # answers every threshold at once.
        for f_idx, t in enumerate(frame_indices):
            probas = None
            for r in radii:
                pending = [
                    th for th in thresholds
                    if t not in registries[(r, th)].seen_frames
                ]
                if not pending:
                    continue
                if probas is None:  # one sigmoid for every radius
                    probas = det_logits[f_idx][0].sigmoid()
                per_threshold = detect_cells_multi_threshold(
                    probas,
                    t,
                    pending,
                    nms=cfg.nms,
                    pool_kernel=pool_k,
                    voxel_size=voxel_size_det,
                    nms_radius_um=r,
                    return_scores=True,
                )
                for th, (arr, node_scores) in per_threshold.items():
                    # Onto the node grid, which is what the registry, the offset
                    # head and the edge scorer all read.
                    arr = arr.astype(np.float32)
                    arr[:, 1:] /= det_scaling
                    registries[(r, th)].add_frame(
                        t, arr, _gather_offsets(offset_maps, f_idx, arr), node_scores
                    )

        # --- Edge prediction for each consecutive pair in the window ---
        # Repeated per (radius, threshold): each one has its own nodes, so its
        # own pairs.
        for reg in registries.values():
            _predict_window_edges(
                model, reg, frame_indices, node_feat_maps, out_shape, cfg, device,
                voxel_size=voxel_size_out,
            )

        del unet_out

    results = {
        key: (
            _rescale_coords(
                reg.coords(), unet_scaling, downscale_factor,
                offsets=reg.offsets() if use_offset else None,
                det_stride=det_stride,
            ),
            reg.edges,
            reg.scores(),
        )
        for key, reg in registries.items()
    }
    det_logits = torch.stack(det_logits_list) if return_det_logits else None
    return results, det_logits


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
) -> tuple[np.ndarray, list[tuple[int, int, float, float]], torch.Tensor]:
    """Run inference on a single video at ``cfg.det_threshold``.

    Thin wrapper over ``predict_video_thresholds``; use that one directly to
    sweep thresholds, which costs far less than one call per value.

    Returns
    -------
    coords : np.ndarray
        Shape (N, 4) — columns [t, z, y, x] in original resolution.
    edges : list of (src_idx, tgt_idx, prob, distance) tuples, indexing rows
        of ``coords`` (distance in detection-grid voxel units).
    det_logits : torch.Tensor
        Stacked raw detection logits, one entry per window.
    """
    cfg = cfg or PredictConfig()
    results, det_logits = predict_video_thresholds(
        model,
        ds_path,
        [cfg.det_threshold],
        device=device,
        cfg=cfg,
        window_size=window_size,
        max_frames=max_frames,
        progress=progress,
        downscale_factor=downscale_factor,
    )
    coords, edges, _ = results[(float(cfg.nms_radius_um), float(cfg.det_threshold))]
    return coords, edges, det_logits



@torch.no_grad()
def predict_video_precomputed(
    model: UNetNodeTransformer,
    ds_path: str | Path,
    coordinates_folder: str | Path,
    contrastive_folder: str | Path | None = None,
    device: str | torch.device = "cuda",
    cfg=None,
    downscale_factor: int = 1,
    max_frames: int | None = None,
    emb_tol: float = 1e-3,
) -> tuple[np.ndarray, list]:
    """``predict_video`` for a model whose nodes come from a folder.

    Same contract -- returns ``(coords, edges)`` ready for ``build_graph`` --
    and the same edge machinery, ``_select_edges`` and all, so the two paths
    are scored by identical rules. What it drops is everything the U-Net was
    for: no forward pass, no NMS, no detection threshold. The node set is
    given, so the only question left is which pairs to link.

    Two consequences worth stating.

    *There is no threshold sweep.* Every threshold would see the same nodes, so
    the caller gets one result rather than a dict.

    *The coordinates come back exactly as the folder wrote them.* The U-Net path
    has to undo its own downsampling in ``_rescale_coords``, which costs it half
    a grid cell of accuracy; here the original-resolution coordinate was never
    lost, so it is returned untouched.
    """
    from tracking.fit_utils import node_scaling_from_voxel

    cfg = cfg or PredictConfig()
    device = torch.device(device)
    ds_path = Path(ds_path)
    sample = ds_path.name

    ds = open_dataset(ds_path, normalize=False, load_image=False)
    T = ds.image_shape[0] if max_frames is None else min(ds.image_shape[0], max_frames)

    with np.load(Path(coordinates_folder) / f"{sample}.npz") as d:
        all_coords = np.asarray(d["coords"]).astype(np.float32)   # (N, 4) t,z,y,x

    emb_coords = emb_feats = None
    if contrastive_folder:
        with np.load(Path(contrastive_folder) / f"{sample}.npz") as d:
            emb_coords = np.asarray(d["coords"]).astype(np.float32)
            emb_feats = np.asarray(d["embeddings"]).astype(np.float32)
    dim = 0 if emb_feats is None else emb_feats.shape[1]

    # The isotropic node grid the edge scorer was trained to measure in --
    # derived exactly as encode_nodes_precomputed derives it.
    scaling = np.asarray(
        node_scaling_from_voxel(tuple(ds.scale)), dtype=np.float32
    )
    spatial = tuple(int(s // downscale_factor) for s in ds.image_shape[1:])
    out_shape = tuple(int(s // sc) for s, sc in zip(spatial, scaling))
    # Window-relative, matching training: a pair is frames 0 and 1 of a
    # window_size=2 window, whatever its absolute index in the video.
    pos_shape = (2,) + out_shape

    times = all_coords[:, 0].astype(int)
    reg = _NodeRegistry()
    per_frame_emb: dict[int, np.ndarray] = {}
    # Accumulated frame by frame rather than sliced out of all_coords at the
    # end: the registry stacks in this order, and the edges index into it, so
    # an npz that was not sorted by t would otherwise silently mislabel every
    # edge endpoint.
    orig_lists: list[np.ndarray] = []

    for t in range(T):
        rows = np.flatnonzero(times == t)
        pts = all_coords[rows]
        orig_lists.append(pts)
        grid = pts.copy()
        grid[:, 1:] = pts[:, 1:] / downscale_factor / scaling
        reg.add_frame(t, grid)

        e = np.zeros((len(rows), dim), np.float32)
        if dim and len(rows):
            ref_rows = np.flatnonzero(emb_coords[:, 0].astype(int) == t)
            if len(ref_rows):
                # Matched on the coordinate, not the row index, for the reason
                # given in TrackDataset._precomputed_nodes.
                dmat = np.linalg.norm(
                    pts[:, None, 1:] - emb_coords[None, ref_rows, 1:], axis=2
                )
                j = dmat.argmin(axis=1)
                ok = dmat[np.arange(len(rows)), j] <= emb_tol
                e[ok] = emb_feats[ref_rows[j[ok]]]
        per_frame_emb[t] = e

    if getattr(model, "unet_features", ()):
        # Those features are read off U-Net maps at the node coordinates, and
        # there is no U-Net here. Failing now beats scoring a model on node
        # tokens that are silently zero where it expects encoder features.
        raise ValueError(
            f"precomputed-node inference cannot supply unet_features "
            f"{list(model.unet_features)}; set unet_features=[] for this path"
        )
    needs_emb = int(getattr(model, "contrastive_feat_dim", 0) or 0) > 0
    nearest = getattr(cfg, "edge_mode", "transformer") == "nearest"
    voxel_size_out = tuple(float(s) * downscale_factor * float(u)
                           for s, u in zip(ds.scale, scaling))

    coords_grid = reg.coords()
    for t in range(T - 1):
        s_src, e_src = reg.coord_offset[t]
        s_tgt, e_tgt = reg.coord_offset[t + 1]
        if e_src == s_src or e_tgt == s_tgt:
            continue
        c_src, c_tgt = coords_grid[s_src:e_src], coords_grid[s_tgt:e_tgt]
        idx_src = np.arange(s_src, e_src, dtype=np.int64)
        idx_tgt = np.arange(s_tgt, e_tgt, dtype=np.int64)

        if nearest:
            probs = _nearest_probs(c_src[:, 1:], c_tgt[:, 1:],
                                   voxel_size_out, cfg.max_link_um)
        else:
            p_c_src = torch.from_numpy(c_src[:, 1:]).float().unsqueeze(0).to(device)
            p_c_tgt = torch.from_numpy(c_tgt[:, 1:]).float().unsqueeze(0).to(device)
            src_rel, tgt_rel = c_src.copy(), c_tgt.copy()
            src_rel[:, 0], tgt_rel[:, 0] = 0, 1
            p_p_src = torch.from_numpy(
                pos_embed_numpy(src_rel, pos_shape)).float().unsqueeze(0).to(device)
            p_p_tgt = torch.from_numpy(
                pos_embed_numpy(tgt_rel, pos_shape)).float().unsqueeze(0).to(device)
            m_src = torch.ones(1, len(c_src), dtype=torch.bool, device=device)
            m_tgt = torch.ones(1, len(c_tgt), dtype=torch.bool, device=device)

            emb_src = emb_tgt = v_src = v_tgt = None
            if needs_emb:
                emb_src = torch.from_numpy(per_frame_emb[t]).unsqueeze(0).to(device)
                emb_tgt = torch.from_numpy(per_frame_emb[t + 1]).unsqueeze(0).to(device)
                v_src = emb_src.abs().sum(-1) > 0
                v_tgt = emb_tgt.abs().sum(-1) > 0

            logits = model.predict_edges(
                p_c_src.new_zeros((1, len(c_src), 0)),
                p_c_tgt.new_zeros((1, len(c_tgt), 0)),
                p_c_src, p_c_tgt, p_p_src, p_p_tgt, m_src, m_tgt,
                emb_src=emb_src, emb_tgt=emb_tgt,
                emb_valid_src=v_src, emb_valid_tgt=v_tgt,
            )[0]
            probs = (
                torch.softmax(logits, dim=0) if cfg.edge_activation == "softmax"
                else torch.sigmoid(logits)
            ).float().cpu().numpy()

        for prob, i, j in _select_edges(probs, cfg):
            gi, gj = int(idx_src[i]), int(idx_tgt[j])
            dist = float(np.linalg.norm(
                coords_grid[gi, 1:] - coords_grid[gj, 1:]))
            reg.edges.append((gi, gj, prob, dist))

    # The folder's own coordinates, at original resolution, in registry order.
    stacked = (
        np.concatenate(orig_lists) if orig_lists
        else np.empty((0, 4), np.float32)
    )
    return np.round(stacked).astype(np.int16), reg.edges
