"""GT target generation for cell tracking.

Provides:
- sinusoidal positional embeddings for node coordinates
- GT transition matrix construction from tracksdata graphs
- detection target masks (binary voxel-level GT)
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch

from tracking.post_process import pool_kernel_from_um
from tracking.weak_positives import greedy_nms

_POS_EMBED_DIM = 8  # per axis; total = 4 axes × _POS_EMBED_DIM = 32


def pos_embed_numpy(
    coords: np.ndarray,
    image_shape: tuple[int, ...],
    pos_embed_dim: int = _POS_EMBED_DIM,
) -> np.ndarray:
    """Sinusoidal positional embeddings for node coordinates (numpy, CPU).

    Parameters
    ----------
    coords : (N, 4) with columns [t, z, y, x].
    image_shape : (T, Z, Y, X) for normalisation.

    Returns
    -------
    np.ndarray  shape (N, 4 * pos_embed_dim), float32.
    """
    t, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    norms = [c / max(s, 1) for c, s in zip([t, z, y, x], image_shape)]

    def _embed(vals: np.ndarray) -> np.ndarray:
        freqs = 2 ** np.arange(pos_embed_dim // 2)
        angles = vals[:, None] * freqs * np.pi
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)

    return np.concatenate([_embed(n) for n in norms], axis=1).astype(np.float32)


def pos_embed_torch(
    coords: torch.Tensor,
    image_shape: tuple[int, ...],
    pos_embed_dim: int = _POS_EMBED_DIM,
) -> torch.Tensor:
    """Sinusoidal positional embeddings (pure torch, stays on device).

    Parameters
    ----------
    coords : (*, 4) with columns [t, z, y, x].
    image_shape : (T, Z, Y, X) for normalisation.

    Returns
    -------
    torch.Tensor  shape (*, 4 * pos_embed_dim).
    """
    shape_t = torch.tensor(image_shape, dtype=torch.float32, device=coords.device)
    norms = coords / shape_t.clamp(min=1)  # (*, 4)
    freqs = (
        2.0 ** torch.arange(pos_embed_dim // 2, device=coords.device, dtype=torch.float32)
        * torch.pi
    )
    parts = []
    for ax in range(4):
        angles = norms[..., ax].unsqueeze(-1) * freqs  # (*, D//2)
        parts.extend([angles.sin(), angles.cos()])
    return torch.cat(parts, dim=-1)


def compute_gt_transition_matrix(
    gt_ids_t: np.ndarray,
    gt_ids_t1: np.ndarray,
    edge_attrs: pl.DataFrame,
) -> torch.Tensor:
    """Build GT transition matrix as a float32 torch tensor.

    Returns
    -------
    torch.Tensor  shape (N_t, N_t1), float32 binary.
    """
    t_to_row = {nid: i for i, nid in enumerate(gt_ids_t)}
    t1_to_col = {nid: i for i, nid in enumerate(gt_ids_t1)}

    matrix = torch.zeros(len(gt_ids_t), len(gt_ids_t1), dtype=torch.float32)
    for source_id, target_id in zip(edge_attrs["source_id"], edge_attrs["target_id"]):
        if source_id in t_to_row and target_id in t1_to_col:
            matrix[t_to_row[source_id], t1_to_col[target_id]] = 1.0
    return matrix


def build_detection_target(
    coords: torch.Tensor,   # (B, max_nodes, 3) downsampled space
    mask: torch.Tensor,     # (B, max_nodes) bool
    spatial: tuple[int, int, int],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build binary detection target by marking GT node voxels.

    Returns
    -------
    torch.Tensor  shape (B, Z, Y, X), float32 — 1 at GT voxels, 0 elsewhere.
    """
    if device is None:
        device = coords.device
    B = coords.shape[0]
    target = torch.zeros(B, *spatial, device=device)
    nt = mask.sum(dim=1).long()
    for b in range(B):
        n = int(nt[b].item())
        if n == 0:
            continue
        zi = coords[b, :n, 0].long().clamp(0, spatial[0] - 1)
        yi = coords[b, :n, 1].long().clamp(0, spatial[1] - 1)
        xi = coords[b, :n, 2].long().clamp(0, spatial[2] - 1)
        target[b, zi, yi, xi] = 1.0
    return target


def _top_k_peaks(
    probas: torch.Tensor,     # (Z, Y, X)
    peak_mask: torch.Tensor,  # (Z, Y, X) bool — local maxima, unthresholded
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Highest-scoring local maxima, ignoring the detection threshold."""
    _, Y, X = probas.shape
    masked = torch.where(peak_mask, probas, torch.full_like(probas, -1.0))
    vals, flat = masked.reshape(-1).topk(min(k, masked.numel()))
    keep = vals > 0
    vals, flat = vals[keep], flat[keep]
    z = flat // (Y * X)
    rem = flat % (Y * X)
    coords = torch.stack([z, rem // X, rem % X], dim=1).float()
    return coords, vals


def detect_and_match(
    det_probas: torch.Tensor,       # (B, 1, Z, Y, X)
    gt_coords: torch.Tensor,        # (B, max_nodes, 3)
    mask: torch.Tensor,             # (B, max_nodes) bool
    image_shape: tuple,
    det_threshold: float = 0.3,
    pool_kernel_um: float = 5.0,
    max_match_distance: float = 7.0,
    voxel_size: tuple | None = None,
    unet_scaling: tuple | None = None,
    frame_index: int = 0,
    window_size: int | None = None,
    max_dets_per_sample: int | None = None,
    top_k_nodes: int | None = None,
    teacher_forcing: bool = False,
    nms: str = "maxpool",
    nms_radius_um: float = 7.0,
    max_nms_candidates: int = 50_000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Detect local-max peaks in logits, match to GT, return padded edge-prediction inputs.

    ``nms`` selects how peaks are picked out of the above-threshold voxels:
    "maxpool" uses a stride-1 max-pool over an odd voxel kernel derived from
    ``pool_kernel_um``, "greedy" suppresses at an exact ``nms_radius_um``.
    The max-pool kernel quantises hard on a coarse grid (at 3.25 um only
    3.25/9.75/16.25 um are reachable) and reports every voxel of a plateau;
    greedy has neither problem, at the cost of a loop over output peaks.
    ``max_nms_candidates`` caps its input by score to bound that.

    If max_dets_per_sample is given, each sample keeps only its highest-scoring
    peaks up to that count, which bounds the size of the edge-scoring matrix.

    If top_k_nodes is given, a sample yielding fewer than that many peaks above
    det_threshold instead falls back to its top-k highest-scoring local maxima.
    This keeps the node set non-empty while the detector is still weak — with
    no nodes the edge branch receives an identically zero gradient.

    If teacher_forcing is True the GT points are used directly as the node set
    (each trivially matched to itself), decoupling edge training from detector
    quality entirely. Intended for the start of training only.

    Returns
    -------
    coords  : (B, M, 3)              detected coords in downsampled space
    pos     : (B, M, 4*D)            sinusoidal positional embeddings
    det_mask: (B, M)  bool
    matches : list of B tensors each (n_det,) — GT index matched or -1
    """
    import torch.nn.functional as F

    B = det_probas.shape[0]
    device = det_probas.device

    if voxel_size is not None:
        vs = torch.tensor(voxel_size, dtype=torch.float32, device=device)

        if unet_scaling is not None:
            unet_vs = vs * torch.tensor(unet_scaling, dtype=torch.float32, device=device)
        else:  # assume (1, 1, 1)
            unet_vs = vs
    else:
        vs = torch.ones(3, dtype=torch.float32, device=device)
        unet_vs = vs

    # Shared with predict.py, so training and inference suppress over the same
    # physical distance.
    pool_kernel = pool_kernel_from_um(pool_kernel_um, tuple(unet_vs.tolist()))
    # print(pool_kernel)

    pad = tuple(k // 2 for k in pool_kernel)
    spatial = det_probas.shape[2:]

    use_greedy = nms == "greedy"
    if nms not in ("greedy", "maxpool"):
        raise ValueError(f"unknown nms mode {nms!r}, expected 'maxpool' or 'greedy'")

    with torch.no_grad():
        # The max-pool is still needed for the top_k fallback, which ranks raw
        # local maxima, but not otherwise in greedy mode.
        if use_greedy and top_k_nodes is None:
            is_peak_raw = None
        else:
            pooled = F.max_pool3d(det_probas, pool_kernel, stride=1, padding=pad)
            is_peak_raw = det_probas == pooled

        above = det_probas > det_threshold
        # Greedy suppresses among all above-threshold voxels; max-pool has
        # already reduced them to local maxima.
        is_peak = above if use_greedy else (is_peak_raw & above)
        peak_idx = torch.nonzero(is_peak[:, 0])   # (N_total, 4): [b, z, y, x]

    batch_ids = peak_idx[:, 0]
    peak_coords = peak_idx[:, 1:].float()
    peak_scores = det_probas[:, 0][
        peak_idx[:, 0], peak_idx[:, 1], peak_idx[:, 2], peak_idx[:, 3]
    ]  # (N_total,)

    nt_per_sample = mask.sum(dim=1).long()
    sample_matches: list[torch.Tensor] = []
    sample_coords: list[torch.Tensor] = []
    max_det = 0

    if unet_scaling is not None:
        scale_t = torch.tensor(unet_scaling, dtype=torch.float32, device=device)
    else:
        scale_t = torch.ones(3, dtype=torch.float32, device=device)

    for b in range(B):
        nt = int(nt_per_sample[b].item())

        if teacher_forcing:
            # GT points *are* the node set; each matches itself.
            det_b = (gt_coords[b, :nt].float() / scale_t).floor()
            for a in range(3):
                det_b[:, a].clamp_(0, spatial[a] - 1)
            sample_matches.append(torch.arange(nt, dtype=torch.long, device=device))
            sample_coords.append(det_b)
            max_det = max(max_det, nt)
            continue

        sel = batch_ids == b
        det_b = peak_coords[sel]
        scores_b = peak_scores[sel]

        if use_greedy and det_b.shape[0]:
            if det_b.shape[0] > max_nms_candidates:
                top = scores_b.topk(max_nms_candidates).indices
                det_b, scores_b = det_b[top], scores_b[top]
            keep = greedy_nms(det_b, scores_b, unet_vs, nms_radius_um)
            det_b, scores_b = det_b[keep], scores_b[keep]

        # Too few peaks cleared the threshold — fall back to the top-k local
        # maxima so the edge branch still has nodes to learn from.
        if top_k_nodes is not None and det_b.shape[0] < top_k_nodes:
            det_b, scores_b = _top_k_peaks(
                det_probas[b, 0], is_peak_raw[b, 0], top_k_nodes,
            )

        if max_dets_per_sample is not None and det_b.shape[0] > max_dets_per_sample:
            det_b = det_b[scores_b.topk(max_dets_per_sample).indices]

        n_det = det_b.shape[0]
        gt_b = gt_coords[b, :nt]

        matched = torch.full((n_det,), -1, dtype=torch.long, device=device)
        if n_det > 0 and nt > 0:
            dists = torch.cdist(det_b * unet_vs, gt_b * vs)
            min_d, min_i = dists.min(dim=1)
            # print('Min distance to GT: ', min_d.min())
            order = min_d.argsort()
            gt_taken = torch.zeros(nt, dtype=torch.bool, device=device)
            for idx in order:
                if min_d[idx] > max_match_distance:
                    # print('break')
                    break
                gi = min_i[idx]
                if not gt_taken[gi]:
                    matched[idx] = gi
                    gt_taken[gi] = True
                    # print('ok ( gi: ', gi, ' idx: ', idx, ' min_d: ', min_d[idx], ' max_match_distance: ', max_match_distance, ')')

        sample_matches.append(matched)
        sample_coords.append(det_b)
        if n_det > max_det:
            max_det = n_det

        # print(matched.max())
        # print(det_b)

    max_det = max(max_det, 1)

    # Pad the coordinates and mask
    padded_coords = torch.zeros(B, max_det, 3, device=device)
    padded_mask = torch.zeros(B, max_det, dtype=torch.bool, device=device)
    for b in range(B):
        n = sample_coords[b].shape[0]
        if n == 0:
            continue
        padded_coords[b, :n] = sample_coords[b]
        padded_mask[b, :n] = True

    # Positional embeddings (to check)
    t_col = torch.full((B, max_det, 1), frame_index, device=device, dtype=torch.float32)
    full_coords = torch.cat([t_col, padded_coords], dim=-1)

    out_shape = tuple([shape // s for shape, s in zip(image_shape[-3:], unet_scaling)])
    pos_shape = (window_size,) + out_shape
    # print(pos_shape)
    padded_pos = pos_embed_torch(full_coords, pos_shape)

    return padded_coords, padded_pos, padded_mask, sample_matches


def build_matched_edge_targets(
    match_t: list[torch.Tensor],
    match_t1: list[torch.Tensor],
    gt_target: torch.Tensor,   # (B, max_nodes, max_nodes) GT transition matrix
    max_det_t: int,
    max_det_t1: int,
) -> torch.Tensor:
    """Build (B, max_det_t, max_det_t1) edge targets via detection→GT matching."""
    B = gt_target.shape[0]
    device = gt_target.device
    target = torch.zeros(B, max_det_t, max_det_t1, device=device)

    for b in range(B):
        mt = match_t[b]
        mt1 = match_t1[b]
        gt_trans = gt_target[b]
        n_t, n_t1 = mt.shape[0], mt1.shape[0]
        if n_t == 0 or n_t1 == 0:
            continue

        valid_t = mt >= 0
        valid_t1 = mt1 >= 0
        valid_mask = valid_t.unsqueeze(1) & valid_t1.unsqueeze(0)
        safe_t = mt.clamp(min=0)
        safe_t1 = mt1.clamp(min=0)
        block = gt_trans[safe_t][:, safe_t1] * valid_mask.float()
        target[b, :n_t, :n_t1] = block

    return target
