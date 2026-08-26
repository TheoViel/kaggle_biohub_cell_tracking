"""Can motion tell a right link from a wrong one, and is the right partner reachable?

Two questions decide which post-processing can work at all:

1. Do the metric's FP edges disagree with the local flow more than its TP edges?
   If so, a motion-residual filter removes FPs without costing TPs.
2. For the GT links we got wrong, is the correct parent among the child's
   nearest candidates once the frame's drift is removed? If it is, re-linking
   can recover them; if not, only a better scorer can.

Run:  python src/postprocessing/analysis_flow.py
"""

import contextlib
import io
from collections import defaultdict

import numpy as np
import tracksdata as td
from scipy.spatial import cKDTree

from common import load_gt, load_prediction, to_graph
from params import SCALE
from tracking_cellmot.metrics import _evaluate_matched_graph, evaluate

K_MASK = td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK
K_MATCH = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
SCALE_ARR = np.asarray(SCALE)


def local_flow(src_um, tgt_um, links, query_um, k=30):
    """Median displacement of the k nearest already-linked cells, per query point.

    The whole field slides between frames and cells drift with their
    neighbours, so "where did this cell go" is answered far better by its
    neighbours' motion than by its own nearest match.
    """
    if len(links) < k:
        if len(links) == 0:
            return np.zeros_like(query_um)
        k = len(links)
    anchor = src_um[links[:, 0]]
    disp = tgt_um[links[:, 1]] - anchor
    _, near = cKDTree(anchor).query(query_um, k=k)
    return np.median(disp[near.reshape(len(query_um), -1)], axis=1)


def main():
    gt_graph, n_total = load_gt()
    coords, edges = load_prediction()
    graph = to_graph(coords, edges)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluate(graph, gt_graph, scale=SCALE)

    ea = _evaluate_matched_graph(graph, gt_graph)
    ea = ea.filter(ea["pred_valid"])
    in_play = {
        (s, t): m
        for s, t, m in zip(ea["source_id"].to_list(), ea["target_id"].to_list(), ea[K_MASK].to_list())
    }

    # node id -> row in coords (build_graph adds nodes in coords order)
    node_ids = graph.node_ids()
    row_of = {int(n): i for i, n in enumerate(sorted(node_ids))}
    rows_by_t = {int(t): np.flatnonzero(coords[:, 0] == t) for t in np.unique(coords[:, 0])}
    id_by_row = {i: n for n, i in row_of.items()}

    pe = graph.edge_attrs()
    e_src = np.array([row_of[s] for s in pe["source_id"].to_list()])
    e_tgt = np.array([row_of[t] for t in pe["target_id"].to_list()])
    e_prob = np.array(pe["edge_prob"].to_list())
    e_key = list(zip(pe["source_id"].to_list(), pe["target_id"].to_list()))

    pos_um = coords[:, 1:].astype(float) * SCALE_ARR

    # --- residual of every edge against its frame pair's local flow ---
    residual = np.full(len(e_src), np.nan)
    for t in sorted(rows_by_t):
        if t + 1 not in rows_by_t:
            continue
        here, there = rows_by_t[t], rows_by_t[t + 1]
        local_src = {r: i for i, r in enumerate(here)}
        local_tgt = {r: i for i, r in enumerate(there)}
        mask = np.isin(e_src, here) & np.isin(e_tgt, there)
        if not mask.any():
            continue
        idx = np.flatnonzero(mask)
        links = np.stack([[local_src[e_src[i]] for i in idx],
                          [local_tgt[e_tgt[i]] for i in idx]], axis=1)
        src_um, tgt_um = pos_um[here], pos_um[there]
        flow = local_flow(src_um, tgt_um, links, src_um)
        moved = tgt_um[links[:, 1]] - src_um[links[:, 0]]
        residual[idx] = np.linalg.norm(moved - flow[links[:, 0]], axis=1)

    tp_r = np.array([residual[i] for i, k in enumerate(e_key) if in_play.get(k) is True])
    fp_r = np.array([residual[i] for i, k in enumerate(e_key) if in_play.get(k) is False])
    all_r = residual[~np.isnan(residual)]
    print("--- flow residual (um), how far a link departs from its neighbours' motion ---")
    for name, arr in (("all edges", all_r), (f"TP ({len(tp_r)})", tp_r), (f"FP ({len(fp_r)})", fp_r)):
        print(f"    {name:<12s} med={np.median(arr):.2f}  p75={np.percentile(arr,75):.2f}  "
              f"p90={np.percentile(arr,90):.2f}  max={arr.max():.2f}")
    print("\n    a residual cut would keep:")
    for cut in (2, 3, 4, 5, 6, 8, 10):
        keep_tp, keep_fp = int((tp_r <= cut).sum()), int((fp_r <= cut).sum())
        j = keep_tp / (gt_graph.num_edges() + keep_fp)
        print(f"      <= {cut:>2d} um : {keep_tp:>3d}/{len(tp_r)} TP  {keep_fp:>3d}/{len(fp_r)} FP"
              f"   -> J = {j:.4f}")

    # --- for the GT links we missed, where does the correct parent rank? ---
    na = graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, K_MATCH])
    na = na.filter(na[K_MATCH] != -1)
    pred_of_gt = dict(zip(na[K_MATCH].to_list(), na[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list()))

    ge = gt_graph.edge_attrs(attr_keys=[K_MASK])
    ranks, absent = [], 0
    for gs, gt_id, ok in zip(ge["source_id"].to_list(), ge["target_id"].to_list(), ge[K_MASK].to_list()):
        if ok:
            continue
        p, q = pred_of_gt.get(gs), pred_of_gt.get(gt_id)
        if p is None or q is None:
            absent += 1
            continue
        rp, rq = row_of[p], row_of[q]
        t = int(coords[rp, 0])
        here, there = rows_by_t[t], rows_by_t[t + 1]
        # drift-corrected candidate ranking, exactly what a re-linker would see
        local_src = {r: i for i, r in enumerate(here)}
        local_tgt = {r: i for i, r in enumerate(there)}
        mask = np.isin(e_src, here) & np.isin(e_tgt, there)
        idx = np.flatnonzero(mask)
        links = np.stack([[local_src[e_src[i]] for i in idx],
                          [local_tgt[e_tgt[i]] for i in idx]], axis=1)
        src_um, tgt_um = pos_um[here], pos_um[there]
        flow = local_flow(src_um, tgt_um, links, src_um)
        d = np.linalg.norm((src_um + flow)[:, None, :] - tgt_um[None, :, :], axis=-1)
        col = d[:, local_tgt[rq]]
        ranks.append(int((col < col[local_src[rp]]).sum()))
    ranks = np.array(ranks)
    print(f"\n--- the {len(ranks)} missed GT links: rank of the true parent after flow correction ---")
    print(f"    (both endpoints undetected: {absent})")
    for r in (0, 1, 2, 4, 9):
        print(f"    within top-{r+1:<2d}: {int((ranks <= r).sum()):>3d}/{len(ranks)}")
    print(f"    ranks: {sorted(ranks.tolist())}")


if __name__ == "__main__":
    main()
