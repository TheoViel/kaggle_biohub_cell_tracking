"""A scorer fast enough to sweep with, validated against the real one.

tracking_cellmot.metrics.evaluate rebuilds a tracksdata graph and re-runs the
node matching every call, which is seconds per sample — fine for reporting, far
too slow for a grid over 71 stacks. But the node matching depends only on the
node set, and every edge-side count follows from it by lookup. So match once
per node set and count directly.

The counting rule is the metric's own: a predicted edge is *seen* by the score
only when its source matched a ground-truth node that has a child, or its
target matched one that has a parent. Everything else is invisible, which is
why the ground truth being sparse makes precision cheap and recall expensive.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from params import SCALE
from tracking_cellmot.metrics import ADJUSTMENT_ALPHA

SCALE_ARR = np.asarray(SCALE)
MAX_MATCH_UM = 7.0


class GroundTruth:
    """The parts of a ground-truth graph the score depends on."""

    def __init__(self, gt_graph):
        nodes = gt_graph.node_attrs().to_pandas()
        self.ids = nodes["node_id"].to_numpy()
        self.t = nodes["t"].to_numpy().astype(int)
        self.xyz = nodes[["z", "y", "x"]].to_numpy(dtype=float) * SCALE_ARR
        edges = gt_graph.edge_attrs().to_pandas()
        self.pairs = set(zip(edges["source_id"].to_numpy(), edges["target_id"].to_numpy()))
        self.n_edges = len(self.pairs)
        self.has_child = set(edges["source_id"].to_numpy())
        self.has_parent = set(edges["target_id"].to_numpy())
        self.n_nodes = len(self.ids)
        self.by_t = {int(tt): np.flatnonzero(self.t == tt) for tt in np.unique(self.t)}


def match_nodes(coords, gt: GroundTruth):
    """pred row -> gt node id, by optimal distance assignment within each frame."""
    pos_um = coords[:, 1:].astype(float) * SCALE_ARR
    match = {}
    for t, gt_rows in gt.by_t.items():
        pred_rows = np.flatnonzero(coords[:, 0] == t)
        if not len(pred_rows):
            continue
        d = np.linalg.norm(
            pos_um[pred_rows][:, None, :] - gt.xyz[gt_rows][None, :, :], axis=-1
        )
        big = np.where(d <= MAX_MATCH_UM, d, 1e6)
        rows, cols = linear_sum_assignment(big)
        for r, c in zip(rows, cols):
            if big[r, c] < 1e6:
                match[int(pred_rows[r])] = int(gt.ids[gt_rows[c]])
    return match


def edge_flags(edges, match, gt: GroundTruth):
    """Per edge: is the metric looking at it, and is it a true positive.

    Computed once for a node set so a sweep over edge filters is a sum over
    boolean masks rather than a re-walk of the graph.
    """
    edges = np.asarray(edges).reshape(-1, edges.shape[1] if np.ndim(edges) > 1 else 1)
    src = edges[:, 0].astype(int)
    tgt = edges[:, 1].astype(int)
    ga = np.array([match.get(int(a), -1) for a in src])
    gb = np.array([match.get(int(b), -1) for b in tgt])
    has_child = np.fromiter((g in gt.has_child for g in ga), bool, len(ga))
    has_parent = np.fromiter((g in gt.has_parent for g in gb), bool, len(gb))
    valid = has_child | has_parent
    is_tp = np.fromiter(
        ((a, b) in gt.pairs for a, b in zip(ga, gb)), bool, len(ga)
    )
    return valid, is_tp & valid


def counts_from_flags(keep, valid, is_tp, gt: GroundTruth):
    tp = int((is_tp & keep).sum())
    fp = int((valid & keep).sum()) - tp
    return tp, fp, gt.n_edges - tp


def row_from_counts(tp, fp, fn, n_nodes, n_total, node_recall):
    denom = tp + fp + fn
    jaccard = tp / denom if denom else float("nan")
    ratio = (n_nodes - n_total) / n_total if n_total > 0 else float("nan")
    adj = max(0.0, jaccard * (1 - ADJUSTMENT_ALPHA * ratio)) if denom else float("nan")
    return {
        "edge_tp": tp, "edge_fp": fp, "edge_fn": fn,
        "edge_jaccard": jaccard, "adj_edge_jaccard": adj,
        "total_node_ratio": ratio, "n_nodes": n_nodes, "node_recall": node_recall,
    }


def counts(edges, match, gt: GroundTruth):
    """(tp, fp, fn) for an edge set, given a node matching."""
    tp = valid = 0
    for a, b in np.asarray(edges)[:, :2].astype(int):
        ga, gb = match.get(int(a)), match.get(int(b))
        if ga is None and gb is None:
            continue
        if (ga in gt.has_child) or (gb in gt.has_parent):
            valid += 1
            tp += (ga, gb) in gt.pairs
    return tp, valid - tp, gt.n_edges - tp


def metrics(coords, edges, gt: GroundTruth, n_total, match=None):
    """The per-sample row the notebook prints, without building a graph."""
    if match is None:
        match = match_nodes(coords, gt)
    tp, fp, fn = counts(edges, match, gt)
    denom = tp + fp + fn
    jaccard = tp / denom if denom else float("nan")
    ratio = (len(coords) - n_total) / n_total if n_total > 0 else float("nan")
    adj = max(0.0, jaccard * (1 - ADJUSTMENT_ALPHA * ratio)) if denom else float("nan")
    return {
        "edge_tp": tp, "edge_fp": fp, "edge_fn": fn,
        "edge_jaccard": jaccard, "adj_edge_jaccard": adj,
        "total_node_ratio": ratio, "n_nodes": len(coords),
        "node_recall": len(set(match.values())) / gt.n_nodes if gt.n_nodes else float("nan"),
    }


def roll_up(rows):
    """summarise()'s adjusted Jaccard: per sample, weighted by sample size."""
    valid = [r for r in rows if r["edge_tp"] == r["edge_tp"]]
    weights = np.array([r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in valid], float)
    adj = np.array([r["adj_edge_jaccard"] for r in valid], float)
    ok = np.isfinite(adj) & (weights > 0)
    tp = sum(r["edge_tp"] for r in valid)
    fp = sum(r["edge_fp"] for r in valid)
    fn = sum(r["edge_fn"] for r in valid)
    return {
        "n": len(valid),
        "edge_tp": tp, "edge_fp": fp, "edge_fn": fn,
        "edge_jaccard": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
        "adj_edge_jaccard": float((adj[ok] * weights[ok]).sum() / weights[ok].sum()),
        "node_recall": float(np.mean([r["node_recall"] for r in valid])),
    }
