"""Evaluation utilities for the cell tracking model."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from params import TRAIN_DIR

from tracking.targets import build_matched_edge_targets
from tracking.fit_utils import (
    collate_frame_det,
    encode_nodes,
    encode_nodes_precomputed,
    filter_frame_det,
    split_frame_det,
)
from training.distributed import unwrap_model


def _evaluate_pair(
    logits: torch.Tensor,  # (N_t, N_t1)
    target: torch.Tensor,  # (N_t, N_t1)
) -> tuple[float, int, int, int]:
    """Returns (focal_bce_loss, tp, fp, fn) for one unpadded pair.

    Links are picked by argmax over the parent axis — each child takes its
    best-scoring parent — matching ``predict.py``'s default
    ``edge_selection="argmax"``. A fixed probability cutoff is useless here:
    the softmax is normalised across all N candidate parents, so scores scale
    like 1/N and never approach 0.5 once a frame holds a few hundred
    detections, which pins the jaccard at 0 however well the model ranks.
    """
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    if not active_rows.any():
        return 0.0, 0, 0, 0

    from tracking.losses import edge_loss

    loss_val = edge_loss(logits, target).item()

    probs = torch.softmax(logits, dim=0)
    preds = torch.zeros_like(probs, dtype=torch.bool)
    preds[probs.argmax(dim=0), torch.arange(probs.shape[1], device=probs.device)] = True

    target_bool = target > 0.5
    mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)

    tp = (preds[mask] & target_bool[mask]).sum().item()
    fp = (preds[mask] & ~target_bool[mask]).sum().item()
    fn = (~preds[mask] & target_bool[mask]).sum().item()
    return loss_val, tp, fp, fn


_ESTIMATED_NODES: dict[str, float] | None = None


def _estimated_nodes() -> dict[str, float]:
    """sample_name -> estimated_number_of_nodes, from the dataset metadata."""
    global _ESTIMATED_NODES
    if _ESTIMATED_NODES is None:
        import pandas as pd

        path = Path(TRAIN_DIR).parent / "metadata.csv"
        df = pd.read_csv(path)
        _ESTIMATED_NODES = dict(
            zip(df["sample_name"], df["estimated_number_of_nodes"].astype(float))
        )
    return _ESTIMATED_NODES


def _sample_name(window_name: str) -> str:
    """Video name from a window key like ``44b6_0113de3b.zarr_t7``."""
    return window_name.split("_t")[0].removesuffix(".zarr")


def _adjustment_factor(frame_nodes: dict, n_frames: dict) -> float:
    """Mean of 1 - 0.1 * (T_pred - T_true) / T_true over the validated samples.

    Validation windows overlap and cover only part of a video — 50 windows of
    52 distinct frames out of 100 on this data — so summing detections over
    windows would both double-count frames and miss the rest. ``T_pred`` is
    instead the mean count over the *distinct* frames seen, scaled to the
    video's full length.
    """
    estimated = _estimated_nodes()
    per_sample: dict[str, list[int]] = {}
    for (sample, _t), n_det in frame_nodes.items():
        per_sample.setdefault(sample, []).append(n_det)

    factors = []
    for sample, counts in per_sample.items():
        t_true = estimated.get(sample)
        if not t_true:
            continue
        t_pred = sum(counts) / len(counts) * n_frames[sample]
        factors.append(1.0 - 0.1 * (t_pred - t_true) / t_true)
    return sum(factors) / len(factors) if factors else 0.0


def _output_stride(model) -> tuple[int, int, int] | None:
    """How much coarser the UNet's grid is than the image, or None if unknown."""
    unet = getattr(unwrap_model(model), "unet", None)
    unet = getattr(unet, "module", unet)  # DataParallel
    stride = getattr(unet, "output_stride", None)
    return tuple(stride) if stride is not None else None


def _matched_within(
    det_coords: torch.Tensor,   # (N, 3) in UNet output space
    det_mask: torch.Tensor,     # (N,) bool
    matches: torch.Tensor,      # (N,) GT index per detection, -1 for unmatched
    keep: torch.Tensor,         # (N,) bool, this threshold's subset
    gt_coords: torch.Tensor,    # (M, 3) in image voxel space
    scaling: tuple[int, int, int],
    voxel_size: tuple[float, float, float],
    tol_um: float,
) -> int:
    """Of the matches this threshold keeps, how many are within ``tol_um``.

    detect_and_match assigns greedily in ascending distance under a 7 um gate,
    so filtering its output by a smaller distance gives exactly the assignment
    a run gated at that distance would have produced — no rematching needed.
    """
    sel = keep[: len(matches)] & (matches >= 0)
    if not bool(sel.any()):
        return 0
    idx = torch.nonzero(sel, as_tuple=False).squeeze(1)
    gt_idx = matches[idx]
    vs = torch.tensor(voxel_size, device=det_coords.device, dtype=torch.float32)
    scale = torch.tensor(scaling, device=det_coords.device, dtype=torch.float32)
    # Detections live on the coarse grid; put them back in image voxels first.
    d = ((det_coords[idx] * scale - gt_coords[gt_idx]) * vs).norm(dim=-1)
    return int((d <= tol_um).sum())


def _gather_dicts(local: dict, world_size: int) -> dict:
    """Merge one dict per rank; later ranks win on a key, which is fine here
    because a (sample, frame) count is the same whichever rank produced it."""
    if world_size == 1 or not (dist.is_available() and dist.is_initialized()):
        return local
    parts: list[dict | None] = [None] * world_size
    dist.all_gather_object(parts, local)
    merged: dict = {}
    for part in parts:
        merged.update(part or {})
    return merged


def _all_reduce_sums(values: list[float], device: torch.device) -> list[float]:
    if not (dist.is_available() and dist.is_initialized()):
        return values
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    args,
    device: torch.device | None = None,
    eval_edges: bool = True,
    transformer_only: bool = False,
    fts_cache: dict[str, list] = None,
) -> tuple[float, dict, dict, dict, dict]:
    """Evaluate on a sharded validation loader across all ranks.

    Returns ``(avg_loss, edge_jaccard, node_recall, adjustment, edge_recall,
    node_recall_tight)``
    where every metric but the loss is keyed by detection threshold; the loss is
    reported at the configured one. ``adjustment`` is the competition's
    node-count penalty, 1 - 0.1 * (T_pred - T_true) / T_true, averaged over
    samples.

    Detection runs once, at ``args.detect_and_match_threshold``; every higher
    threshold is a subset of those peaks, so the sweep filters on each node's
    stored score instead of re-detecting. (Not bit-identical to re-running:
    greedy NMS on a smaller candidate set can keep a peak that a
    just-above-threshold neighbour would otherwise have suppressed.)
    """
    if device is None:
        device = next(model.parameters()).device

    if loader is None:
        return 0., {}, {}, {}, {}

    base = args.detect_and_match_threshold
    thresholds = sorted(
        {base} | {t for t in getattr(args, "eval_thresholds", ()) or () if t >= base}
    )

    # A second, tighter matching radius. The competition's 7 um tolerance is
    # wider than a cell, so node_rec at it saturates — a detector that never
    # trained already scores ~0.91 — and cannot rank two models. Re-scoring the
    # same matches at a few um is what still discriminates.
    tight_um = float(getattr(args, "eval_node_match_um", 0.0) or 0.0)
    scaling = _output_stride(model)
    if tight_um and scaling is None:
        tight_um = 0.0

    model.eval()
    gt_total = 0
    gt_matched = {t: 0 for t in thresholds}
    gt_matched_tight = {t: 0 for t in thresholds}
    total_loss = {t: 0.0 for t in thresholds}
    tp = {t: 0 for t in thresholds}
    fp = {t: 0 for t in thresholds}
    fn = {t: 0 for t in thresholds}
    n_pairs = {t: 0 for t in thresholds}
    # (sample, frame) -> detections, deduplicated: validation windows overlap.
    frame_nodes: dict[float, dict[tuple[str, int], int]] = {t: {} for t in thresholds}
    n_frames: dict[str, int] = {}
    # Precomputed nodes carry no detection score, so the sweep has nothing to
    # sweep. Remember it and report only the threshold that was really scored
    # -- leaving the others at their initialised zero would read as a failure
    # rather than as "not evaluated".
    collapsed_sweep = False

    for batch in loader:
        imgs = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        image_shape = tuple(batch["image_shape"][0].tolist())
        voxel_size = tuple(batch["voxel_size"][0].tolist())

        B, W, d, h, w = imgs.shape

        # Nodes off disk: the same path stage 3 trains on. Detection scores are
        # all 1 there, so the threshold sweep has nothing to sweep and is
        # collapsed to the base value rather than repeating one forward per
        # threshold over an identical node set.
        pre_coords = batch.get("pre_coords")
        precomputed_nodes = pre_coords is not None and pre_coords.shape[-2] > 0
        collapsed_sweep = collapsed_sweep or precomputed_nodes
        if precomputed_nodes:
            frame_det = encode_nodes_precomputed(
                model, pre_coords.to(device, non_blocking=True),
                batch["pre_emb"].to(device, non_blocking=True),
                coords, masks, image_shape, voxel_size, args,
            )
        elif transformer_only and fts_cache is not None:
            names = batch["name"]
            if any(name not in fts_cache for name in names):
                fresh = encode_nodes(
                    model, imgs, coords, masks, image_shape, voxel_size, args, nms=args.nms, nms_radius_um=args.nms_radius_um,
                )[0]
                for b, name in enumerate(names):
                    fts_cache[name] = split_frame_det(fresh, b)
            frame_det = collate_frame_det([fts_cache[n] for n in names])
        else:
            frame_det = encode_nodes(
                model, imgs, coords, masks, image_shape, voxel_size, args, nms=args.nms, nms_radius_um=args.nms_radius_um,
            )[0]

        names = [_sample_name(n) for n in batch["name"]]
        t_starts = batch["t_start"].tolist()
        video_len = batch["image_shape"][:, 0].tolist()
        for i in range(W):
            matches, det_mask, scores = frame_det[i][3], frame_det[i][2], frame_det[i][5]
            for b in range(B):
                gt_total += int(masks[b, i].sum().item())
                n_frames[names[b]] = int(video_len[b])
                for t in thresholds:
                    # Subset of the peaks already detected, by their own score.
                    keep = det_mask[b] & (scores[b] > t)
                    sel = matches[b][keep[: len(matches[b])]]
                    gt_matched[t] += int((sel >= 0).sum())
                    frame_nodes[t][(names[b], t_starts[b] + i)] = int(keep.sum())
                    if tight_um:
                        gt_matched_tight[t] += _matched_within(
                            frame_det[i][0][b], det_mask[b], matches[b], keep,
                            coords[b, i], scaling, voxel_size, tight_um,
                        )

        if not eval_edges:
            continue

        # One edge forward per threshold
        for t in ([base] if precomputed_nodes else thresholds):
            fd = frame_det if t == base else filter_frame_det(frame_det, t)
            for i in range(W - 1):
                pair_target = build_matched_edge_targets(
                    fd[i][3], fd[i + 1][3], targets[:, i],
                    fd[i][0].shape[1], fd[i + 1][0].shape[1],
                )
                pair_logits = model.predict_edges(
                    fd[i][4], fd[i + 1][4],
                    fd[i][0], fd[i + 1][0],
                    fd[i][1], fd[i + 1][1],
                    fd[i][2], fd[i + 1][2],
                    emb_src=fd[i][6] if len(fd[i]) > 6 else None,
                    emb_tgt=fd[i + 1][6] if len(fd[i + 1]) > 6 else None,
                    emb_valid_src=fd[i][7] if len(fd[i]) > 7 else None,
                    emb_valid_tgt=fd[i + 1][7] if len(fd[i + 1]) > 7 else None,
                )
                for b in range(B):
                    ns_b = int(fd[i][2][b].sum().item())
                    nt_b = int(fd[i + 1][2][b].sum().item())
                    pair_loss, pair_tp, pair_fp, pair_fn = _evaluate_pair(
                        pair_logits[b, :ns_b, :nt_b],
                        pair_target[b, :ns_b, :nt_b],
                    )
                    total_loss[t] += pair_loss
                    tp[t] += pair_tp
                    fp[t] += pair_fp
                    fn[t] += pair_fn
                    n_pairs[t] += 1

    if collapsed_sweep:
        thresholds = [base]

    # Aggregate: everything per threshold, in one reduction.
    n = len(thresholds)
    flat = _all_reduce_sums(
        [total_loss[t] for t in thresholds]
        + [tp[t] for t in thresholds]
        + [fp[t] for t in thresholds]
        + [fn[t] for t in thresholds]
        + [n_pairs[t] for t in thresholds]
        + [gt_matched[t] for t in thresholds]
        + [gt_matched_tight[t] for t in thresholds]
        + [gt_total],
        device,
    )
    loss_a, tp_a, fp_a, fn_a, pairs_a, matched_a, tight_a = (
        flat[i * n : (i + 1) * n] for i in range(7)
    )
    gt_total_all = flat[-1]

    edge_jaccard = {
        t: tp_a[i] / max(tp_a[i] + fp_a[i] + fn_a[i], 1.0) for i, t in enumerate(thresholds)
    }
    edge_recall = {
        t: tp_a[i] / max(tp_a[i] + fn_a[i], 1.0) for i, t in enumerate(thresholds)
    }
    node_recall = {
        t: matched_a[i] / max(gt_total_all, 1.0) for i, t in enumerate(thresholds)
    }
    if tight_um:
        node_recall_tight = {
            t: tight_a[i] / max(gt_total_all, 1.0) for i, t in enumerate(thresholds)
        }
    else:
        node_recall_tight = {}
    avg_loss = loss_a[thresholds.index(base)] / max(pairs_a[thresholds.index(base)], 1.0)

    world = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    n_frames_all = _gather_dicts(n_frames, world)
    adjustment = {
        t: _adjustment_factor(_gather_dicts(frame_nodes[t], world), n_frames_all)
        for t in thresholds
    }
    return avg_loss, edge_jaccard, node_recall, adjustment, edge_recall, node_recall_tight


def evaluate_videos(
    model,
    paths: list[Path],
    args,
    device: torch.device | None = None,
    thresholds: tuple[float, ...] = (0.2, 0.3, 0.4),
) -> dict[float, dict[str, float]]:
    """Competition metrics from full-video inference, per detection threshold.

    ``evaluate`` above scores detections against the GT points inside a
    validation window. For a stage that only trains the detector that says
    very little: node recall at the competition's 7 um tolerance saturates —
    an untrained peak finder already scores ~0.91 — so it cannot rank two
    detectors. This instead runs the real inference path over whole videos and
    scores the resulting tracks the way the competition does, so
    ``adj_edge_jaccard`` actually moves when detection improves.

    Edges come from ``cfg.edge_mode``. During detection pretraining the edge
    transformer is still at init and would only add noise, so "nearest" links
    each child to its closest parent instead — good enough to turn a node set
    into a track graph, and it keeps the number a function of detection alone.

    Videos are sharded across ranks and the per-sample rows gathered, so every
    rank returns the same summary.

    Whole videos only, deliberately: node recall counts matched GT nodes over
    *all* GT nodes, so predicting a prefix of the frames would divide by the
    full total and report a fraction of the truth (20 of 100 frames reads as
    0.12 where the real number is 1.00). Budget the cost with how many videos
    are passed in, not with how much of each one is scored.
    """
    import contextlib
    import io

    import pandas as pd
    import tracksdata as td

    from params import SCALE
    from tracking.post_process import build_graph
    from tracking.predict import (
        PredictConfig,
        predict_video_precomputed,
        predict_video_thresholds,
    )
    from tracking_cellmot.metrics import (
        evaluate as evaluate_graph,
        nan_metrics_row,
        node_recall,
        per_sample_metrics,
        summarise,
    )

    if device is None:
        device = next(model.parameters()).device
    thresholds = sorted({float(t) for t in thresholds})
    if not paths:
        return {}

    data_dir = Path(args.data_dir)
    meta_csv = data_dir.parent / "metadata.csv"
    meta = pd.read_csv(meta_csv) if meta_csv.exists() else None

    # Mirror what training detected with, or the thresholds mean something else.
    cfg = PredictConfig()
    cfg.normalization = getattr(args, "normalization", None)
    cfg.nms = getattr(args, "nms", "greedy")
    cfg.nms_radius_um = getattr(args, "nms_radius_um", 7.0)
    cfg.pool_kernel_um = getattr(args, "pool_kernel_um", 5.0)
    cfg.use_offset = bool(getattr(args, "offset_head", False))
    cfg.edge_mode = getattr(args, "eval_edge_mode", "nearest")
    cfg.max_link_um = float(getattr(args, "eval_max_link_um", 15.0))

    world = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    rank = dist.get_rank() if world > 1 else 0

    was_training = model.training
    model.eval()
    core = unwrap_model(model)

    # Nodes off disk: no U-Net, and every detection threshold would see the
    # same node set, so the sweep collapses to a single result reported at the
    # base threshold.
    nodes_cfg = dict(getattr(args, "_nodes_cfg", None) or {})
    precomputed = bool(nodes_cfg.get("coordinates_folder"))
    if precomputed:
        thresholds = [thresholds[0]]

    rows: dict[float, list[dict]] = {t: [] for t in thresholds}
    for path in paths[rank::world]:
        if precomputed:
            per_threshold = {
                thresholds[0]: predict_video_precomputed(
                    core,
                    path,
                    nodes_cfg["coordinates_folder"],
                    nodes_cfg.get("contrastive_folder"),
                    device=device,
                    cfg=cfg,
                    downscale_factor=getattr(args, "downscale_factor", 1),
                )
            }
        else:
            # Training-time eval sweeps thresholds only, at the run's own NMS
            # radius, so the grid collapses back to one key per threshold.
            per_key, _ = predict_video_thresholds(
                core,
                path,
                thresholds,
                device=device,
                cfg=cfg,
                window_size=args.window_size,
                downscale_factor=getattr(args, "downscale_factor", 1),
                progress=False,
                return_det_logits=False,
            )
            # [:2] drops the node probabilities: the precomputed branch above
            # has none, and the metrics below read neither, so both branches
            # stay the same (coords, edges) shape.
            per_threshold = {th: per_key[("det", float(cfg.nms_radius_um), th)][:2]
                             for th in thresholds}

        gt = td.graph.IndexedRXGraph.from_geff(data_dir / f"{path.name}.geff")
        gt_graph = gt[0] if isinstance(gt, tuple) else gt

        n_total = float("nan")
        if meta is not None:
            hit = meta.loc[meta["sample_name"] == path.name, "estimated_number_of_nodes"]
            if len(hit):
                n_total = float(hit.values[0])

        for th in thresholds:
            coords, edges = per_threshold[th]
            # An empty prediction is a real state early in training, and one
            # the metric does not survive: evaluate_graph warns and returns a
            # zero score, but node_recall then reads a match column that was
            # never written and raises KeyError. It is a genuine zero, not
            # missing data, so it is scored rather than skipped -- NaN here
            # would quietly drop the worst samples out of the average.
            if coords.shape[0] == 0 or len(edges) == 0:
                # Every ground-truth edge is a false negative, which is what
                # makes the Jaccard 0 rather than 0/0. Leaving the counts at
                # zero would micro-average to NaN and take the sample out of
                # the run summary -- the one sample that most deserves to be
                # in it.
                row = {**nan_metrics_row(), "num_pred_nodes": int(coords.shape[0])}
                row.update(
                    edge_tp=0, edge_fp=0, edge_fn=int(gt_graph.num_edges()),
                    division_tp=0, division_fp=0, division_fn=0,
                    node_recall=0.0, edge_jaccard=0.0, adj_edge_jaccard=0.0,
                )
                row["n_nodes_pred"] = int(coords.shape[0])
                rows[th].append(row)
                continue
            # build_graph is chatty on some builds; keep the log readable.
            with contextlib.redirect_stdout(io.StringIO()):
                graph = build_graph(coords, edges)
                res = evaluate_graph(graph, gt_graph, scale=SCALE)
                rec = node_recall(graph, gt_graph)
            row = per_sample_metrics(res, n_total, rec)
            row["n_nodes_pred"] = int(coords.shape[0])
            rows[th].append(row)

    if was_training:
        model.train()

    if world > 1:
        parts: list[dict | None] = [None] * world
        dist.all_gather_object(parts, rows)
        rows = {
            t: [r for part in parts if part for r in part[t]] for t in thresholds
        }

    out: dict[float, dict[str, float]] = {}
    for th in thresholds:
        if not rows[th]:
            continue
        summary = summarise(rows[th])
        out[th] = {
            "node_recall": float(summary.get("node_recall", float("nan"))),
            "edge_jaccard": float(summary.get("edge_jaccard", float("nan"))),
            "adj_edge_jaccard": float(summary.get("adj_edge_jaccard", float("nan"))),
            "score": float(summary.get("score", float("nan"))),
        }
    return out
