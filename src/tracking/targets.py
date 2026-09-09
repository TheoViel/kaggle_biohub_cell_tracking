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


def build_offset_target(
    coords: torch.Tensor,          # (B, M, 3) GT coords, image-voxel space
    mask: torch.Tensor,            # (B, M) bool
    spatial: tuple[int, int, int],  # detection-grid shape
    unet_scaling: tuple | None = None,
    radius: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-voxel vote for the cell centre, in detection-grid voxels.

    Every grid voxel within ``radius`` of a ground-truth cell is asked for the
    vector from itself to that cell's centre; a voxel in range of two cells
    votes for the nearer one. This is the segmentation branch's centroid voting
    (``segmentation/targets.py``), moved onto the detection grid.

    A vote, rather than a sub-voxel residual, is what this problem needs. The
    grid is 1.625 um isotropic, so a CenterNet-style offset confined to the
    peak's own cell can move a detection by at most half of that — and the peak
    is in the *wrong* cell 89% of the time on the hardest stacks, 39% of them by
    two cells or more. Confined to its own cell such a head recovers 0.4 um of a
    2.7 um error; reaching over a couple of cells is what makes it worth having.

    Returns
    -------
    target : (B, 3, Z, Y, X)  the vector to the nearest cell centre, grid voxels
    valid  : (B, Z, Y, X)     bool, True where a cell was in range to vote for
    """
    B, M = coords.shape[:2]
    device = coords.device
    Z, Y, X = spatial
    n_vox = B * Z * Y * X

    p = coords.float()
    if unet_scaling is not None:
        p = p / torch.tensor(unet_scaling, dtype=p.dtype, device=device)

    r = max(int(radius), 0)
    off = torch.arange(-r, r + 1, device=device)
    oz, oy, ox = torch.meshgrid(off, off, off, indexing="ij")
    oz, oy, ox = oz.reshape(-1), oy.reshape(-1), ox.reshape(-1)   # (K,)

    base = p.floor().long()                                        # (B, M, 3)
    zc = base[..., 0:1] + oz                                       # (B, M, K)
    yc = base[..., 1:2] + oy
    xc = base[..., 2:3] + ox

    in_range = (
        mask.unsqueeze(-1)
        & (zc >= 0) & (zc < Z)
        & (yc >= 0) & (yc < Y)
        & (xc >= 0) & (xc < X)
    )
    bidx = torch.arange(B, device=device).view(B, 1, 1).expand_as(zc)
    flat = ((bidx * Z + zc.clamp(0, Z - 1)) * Y + yc.clamp(0, Y - 1)) * X + xc.clamp(0, X - 1)
    # One spare slot absorbs the out-of-range votes, then is dropped.
    flat = torch.where(in_range, flat, torch.full_like(flat, n_vox))

    vec = torch.stack(
        [p[..., 0:1] - zc, p[..., 1:2] - yc, p[..., 2:3] - xc], dim=-1
    )                                                              # (B, M, K, 3)
    dist = vec.norm(dim=-1)
    dist = torch.where(in_range, dist, torch.full_like(dist, float("inf")))

    # Nearest cell wins the voxel, so a cell never votes over a closer one.
    best = torch.full((n_vox + 1,), float("inf"), device=device)
    best.scatter_reduce_(0, flat.reshape(-1), dist.reshape(-1), reduce="amin")
    win = in_range & (dist <= best[flat] + 1e-6)

    target = torch.zeros(3, n_vox + 1, device=device)
    for c in range(3):
        target[c].scatter_(0, flat[win], vec[..., c][win])
    valid = torch.zeros(n_vox + 1, dtype=torch.bool, device=device)
    valid.scatter_(0, flat[win], torch.ones_like(flat[win], dtype=torch.bool))

    return (
        target[:, :n_vox].view(3, B, Z, Y, X).movedim(0, 1).contiguous(),
        valid[:n_vox].view(B, Z, Y, X),
    )


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
    det_scaling: tuple | None = None,
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

    ``det_scaling`` is how many voxels of ``det_probas``'s grid make up one node
    -grid voxel — (1, 2, 2) when the model runs an extra decoder stage for the
    heatmap alone. NMS then happens at the heatmap's own resolution, and the
    peaks are divided back down, so a node lands on a half-voxel of the node
    grid rather than being rounded onto it. Everything downstream keeps
    measuring in node-grid units, which is the only space that is isotropic.

    Returns
    -------
    coords  : (B, M, 3)              detected coords in node-grid space
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

    # The heatmap may sit on a finer grid than the nodes do; peak finding runs
    # on the grid the heatmap is actually on.
    det_scale_t = torch.tensor(
        det_scaling if det_scaling is not None else (1, 1, 1),
        dtype=torch.float32, device=device,
    )
    det_vs = unet_vs / det_scale_t

    # Shared with predict.py, so training and inference suppress over the same
    # physical distance.
    pool_kernel = pool_kernel_from_um(pool_kernel_um, tuple(det_vs.tolist()))
    # print(pool_kernel)

    pad = tuple(k // 2 for k in pool_kernel)
    det_spatial = det_probas.shape[2:]
    # The node grid, which is what the returned coordinates are expressed in.
    spatial = tuple(int(s // d) for s, d in zip(det_spatial, det_scale_t.tolist()))

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
            keep = greedy_nms(det_b, scores_b, det_vs, nms_radius_um)
            det_b, scores_b = det_b[keep], scores_b[keep]

        # Too few peaks cleared the threshold — fall back to the top-k local
        # maxima so the edge branch still has nodes to learn from.
        if top_k_nodes is not None and det_b.shape[0] < top_k_nodes:
            det_b, scores_b = _top_k_peaks(
                det_probas[b, 0], is_peak_raw[b, 0], top_k_nodes,
            )

        if max_dets_per_sample is not None and det_b.shape[0] > max_dets_per_sample:
            det_b = det_b[scores_b.topk(max_dets_per_sample).indices]

        # Out of the heatmap's grid and into the node grid. Exact: a fine index
        # divided by an integer factor, so multiplying back recovers it.
        det_b = det_b / det_scale_t

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


def nodes_from_precomputed(
    pre_coords: torch.Tensor,          # (B, P, 3) image-voxel space, -1 padded
    pre_emb: torch.Tensor | None,      # (B, P, D) L2-normalised, or None
    gt_coords: torch.Tensor,           # (B, max_nodes, 3) image-voxel space
    mask: torch.Tensor,                # (B, max_nodes) bool
    image_shape: tuple,
    voxel_size: tuple | None = None,
    node_scaling: tuple | None = None,
    frame_index: int = 0,
    window_size: int | None = None,
    max_match_distance: float = 7.0,
    max_dets_per_sample: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor],
           torch.Tensor, torch.Tensor]:
    """``detect_and_match`` for nodes that are given rather than detected.

    Stage 3 can take its nodes from a predictions folder instead of running the
    U-Net, in which case there is no heatmap to find peaks in and no NMS to run
    -- the node set is fixed and the only thing still to compute is which
    ground-truth cell each node stands for, so the edge targets can be built.
    Everything else about the output is deliberately identical to
    ``detect_and_match``, so the training step cannot tell the two apart.

    ``node_scaling`` is how many image voxels one node-grid unit spans. The
    U-Net path gets this for free from its own downsampling, which is chosen so
    the detection grid comes out isotropic; without a U-Net there is nothing to
    measure, so it is passed in. Isotropy is not cosmetic here -- ``_geom_feats``
    and the kNN gate both read plain Euclidean distance in this space, and at
    the raw 1.625/0.40625 um anisotropy a z-step would count four times a y-step.

    Returns ``(coords, pos, mask, matches, emb, emb_valid)``, the first four
    exactly as ``detect_and_match`` does, plus the per-node embeddings aligned
    to them and a flag for which nodes actually carry one.
    """
    B = pre_coords.shape[0]
    device = pre_coords.device
    D = 0 if pre_emb is None else pre_emb.shape[-1]

    if voxel_size is not None:
        vs = torch.tensor(voxel_size, dtype=torch.float32, device=device)
    else:
        vs = torch.ones(3, dtype=torch.float32, device=device)
    scaling = torch.tensor(
        node_scaling if node_scaling is not None else (1.0, 1.0, 1.0),
        dtype=torch.float32, device=device,
    )
    node_vs = vs * scaling

    sample_coords, sample_emb, sample_matches = [], [], []
    max_det = 0
    nt_all = mask.sum(dim=1).long()

    for b in range(B):
        # -1 is the padding the point-set machinery uses; a real coordinate is
        # never negative.
        valid = (pre_coords[b] >= 0).all(dim=-1)
        idx = torch.nonzero(valid, as_tuple=True)[0]
        if max_dets_per_sample is not None and idx.numel() > max_dets_per_sample:
            idx = idx[:max_dets_per_sample]

        # Into the isotropic node grid the edge scorer measures distance in.
        det_b = pre_coords[b][idx] / scaling
        emb_b = (
            pre_emb[b][idx] if pre_emb is not None
            else pre_coords.new_zeros(idx.numel(), 0)
        )

        n_det = int(det_b.shape[0])
        nt = int(nt_all[b].item())
        gt_b = gt_coords[b, :nt]

        # Greedy one-to-one nearest matching in microns, nearest pair first --
        # the same rule detect_and_match uses, so the edge targets built from
        # these matches mean the same thing.
        matched = torch.full((n_det,), -1, dtype=torch.long, device=device)
        if n_det > 0 and nt > 0:
            dists = torch.cdist(det_b * node_vs, gt_b * vs)
            min_d, min_i = dists.min(dim=1)
            order = min_d.argsort()
            gt_taken = torch.zeros(nt, dtype=torch.bool, device=device)
            for j in order:
                if min_d[j] > max_match_distance:
                    break
                gi = min_i[j]
                if not gt_taken[gi]:
                    matched[j] = gi
                    gt_taken[gi] = True

        sample_coords.append(det_b)
        sample_emb.append(emb_b)
        sample_matches.append(matched)
        max_det = max(max_det, n_det)

    max_det = max(max_det, 1)

    padded_coords = torch.zeros(B, max_det, 3, device=device)
    padded_mask = torch.zeros(B, max_det, dtype=torch.bool, device=device)
    padded_emb = torch.zeros(B, max_det, D, device=device)
    emb_valid = torch.zeros(B, max_det, dtype=torch.bool, device=device)
    for b in range(B):
        n = sample_coords[b].shape[0]
        if n == 0:
            continue
        padded_coords[b, :n] = sample_coords[b]
        padded_mask[b, :n] = True
        if D:
            padded_emb[b, :n] = sample_emb[b]
            # A zero row is what the loader writes for a node the embedding
            # folder had nothing for; it is not a legitimate unit vector, so it
            # is exactly what marks the node as abstaining.
            emb_valid[b, :n] = sample_emb[b].abs().sum(dim=-1) > 0

    t_col = torch.full((B, max_det, 1), frame_index, device=device, dtype=torch.float32)
    full_coords = torch.cat([t_col, padded_coords], dim=-1)
    out_shape = tuple(
        int(s // sc) for s, sc in zip(image_shape[-3:], scaling.tolist())
    )
    padded_pos = pos_embed_torch(full_coords, (window_size,) + out_shape)

    return padded_coords, padded_pos, padded_mask, sample_matches, padded_emb, emb_valid
