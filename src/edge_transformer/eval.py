"""Two evaluations, answering two different questions.

``evaluate`` scores the model where it is supervised: over the candidate edges
the ground truth can judge, how well does the sigmoid separate real links from
the alternatives? It is cheap, it runs on the same windows training uses, and
average precision makes it insensitive to where the threshold sits -- which
matters early, when the scores are not yet calibrated.

``evaluate_videos`` scores the thing that is actually being built: whole videos
through the real inference path, linked, and measured with the competition's
own metrics. The window-level numbers can look excellent while the linked graph
does not, because a per-edge score says nothing about what happens when two
children want the same parent. This is the number to move.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from edge_transformer.losses import edge_loss
from edge_transformer.predict import predict_video
from params import SCALE


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve, by the step-wise definition."""
    if not len(scores) or labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    hits = labels[order] > 0.5
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, len(hits) + 1)
    return float(precision[hits].sum() / hits.sum())


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
    threshold: float = 0.5,
    fp16: bool = True,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Loss, precision/recall/F1 at ``threshold``, and AP over supervised edges."""
    was_training = model.training
    model.eval()

    losses, all_scores, all_labels = [], [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=fp16 and device.type == "cuda",
        ):
            out = model(
                batch["node_feats"].to(device),
                batch["node_pos"].to(device),
                batch["edge_pos"].to(device),
                batch["edge_index"].to(device),
                batch["node_mask"].to(device),
                batch["edge_mask"].to(device),
                edge_feats=batch["edge_feats"].to(device),
            )
        target = batch["edge_target"].to(device)
        supervised = batch["edge_supervised"].to(device)
        logits = out.edge_logits.float().squeeze(-1)
        losses.append(float(edge_loss(logits, target, supervised).loss))

        sel = supervised.cpu().numpy()
        all_scores.append(torch.sigmoid(logits).cpu().numpy()[sel])
        all_labels.append(target.cpu().numpy()[sel])

    if was_training:
        model.train()

    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    labels = np.concatenate(all_labels) if all_labels else np.zeros(0)
    pred = scores >= threshold
    tp = float((pred & (labels > 0.5)).sum())
    fp = float((pred & (labels <= 0.5)).sum())
    fn = float((~pred & (labels > 0.5)).sum())

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "ap": _average_precision(scores, labels),
        "n_edges": float(len(scores)),
    }


@torch.no_grad()
def evaluate_videos(
    model,
    dataset,
    data_dir: str | Path,
    device: torch.device,
    threshold: float = 0.5,
    n_videos: int | None = None,
    fp16: bool = True,
) -> dict[str, float]:
    """The competition metrics, from full-video inference on ``dataset``'s videos.

    Videos are sharded across ranks and the per-sample rows gathered, so every
    rank returns the same summary -- the caller can log it from rank 0 without
    a second collective.

    Whole videos only: node recall counts matched GT nodes over *all* the GT
    nodes of the video, so scoring a prefix of the frames would divide by the
    full total and report a fraction of the truth. Budget the cost with
    ``n_videos``, not with how much of each video is scored.
    """
    import pandas as pd
    import tracksdata as td

    from tracking.post_process import build_graph
    from tracking_cellmot.metrics import (
        evaluate as evaluate_graph,
        nan_metrics_row,
        node_recall,
        per_sample_metrics,
        summarise,
    )

    data_dir = Path(data_dir)
    meta_csv = data_dir.parent / "metadata.csv"
    meta = pd.read_csv(meta_csv) if meta_csv.exists() else None

    world = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    rank = dist.get_rank() if world > 1 else 0

    indices = list(range(len(dataset.videos)))[: n_videos or len(dataset.videos)]
    rows: list[dict] = []
    for vi in indices[rank::world]:
        video = dataset.videos[vi]
        coords, edges = predict_video(
            model, dataset, vi, device, threshold=threshold, fp16=fp16,
        )
        gt = td.graph.IndexedRXGraph.from_geff(data_dir / f"{video.name}.geff")
        gt_graph = gt[0] if isinstance(gt, tuple) else gt

        n_total = float("nan")
        if meta is not None:
            hit = meta.loc[meta["sample_name"] == video.name, "estimated_number_of_nodes"]
            if len(hit):
                n_total = float(hit.values[0])

        # An empty link set is a real state early in training and one the
        # metric does not survive: node_recall reads a match column
        # evaluate_graph never writes. It is a genuine zero rather than missing
        # data, so it is scored as one -- NaN here would quietly drop the worst
        # videos out of the average.
        if len(edges) == 0:
            row = {**nan_metrics_row(), "num_pred_nodes": int(coords.shape[0])}
            row.update(
                edge_tp=0, edge_fp=0, edge_fn=int(gt_graph.num_edges()),
                division_tp=0, division_fp=0, division_fn=0,
                node_recall=0.0, edge_jaccard=0.0, adj_edge_jaccard=0.0,
            )
        else:
            # build_graph and evaluate are chatty on some builds.
            with contextlib.redirect_stdout(io.StringIO()):
                graph = build_graph(coords, edges)
                res = evaluate_graph(graph, gt_graph, scale=SCALE)
                rec = node_recall(graph, gt_graph)
            row = per_sample_metrics(res, n_total, rec)
        row["n_edges_pred"] = len(edges)
        rows.append(row)

    if world > 1:
        parts: list[list[dict] | None] = [None] * world
        dist.all_gather_object(parts, rows)
        rows = [r for part in parts if part for r in part]

    if not rows:
        return {}
    summary = summarise(rows)
    return {
        "node_recall": float(summary.get("node_recall", float("nan"))),
        "edge_jaccard": float(summary.get("edge_jaccard", float("nan"))),
        "adj_edge_jaccard": float(summary.get("adj_edge_jaccard", float("nan"))),
        "score": float(summary.get("score", float("nan"))),
        "n_videos": float(len(rows)),
    }
