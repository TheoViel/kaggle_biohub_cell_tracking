"""Facts needed before choosing a strategy: GT divisions, prob separability, spacing.

Run:  python src/postprocessing/explore.py
"""

from collections import Counter, defaultdict

import numpy as np
import tracksdata as td
from scipy.spatial import cKDTree

from common import load_gt, load_prediction, to_graph
from params import SCALE
from tracking_cellmot.metrics import evaluate

K_MASK = td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK
K_MATCH = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID


def main():
    gt_graph, n_total = load_gt()
    coords, edges = load_prediction()

    # --- how many divisions does the GT actually contain? ---
    ids = gt_graph.node_ids()
    out_deg = Counter(gt_graph.out_degree(ids))
    in_deg = Counter(gt_graph.in_degree(ids))
    print("GT out-degree:", dict(sorted(out_deg.items())))
    print("GT in-degree :", dict(sorted(in_deg.items())))
    print(f"GT divisions (out-degree 2): {out_deg.get(2, 0)}  of {gt_graph.num_nodes()} nodes")

    # --- are TP edges more confident than FP edges? ---
    graph = to_graph(coords, edges)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        evaluate(graph, gt_graph, scale=SCALE)
    from tracking_cellmot.metrics import _evaluate_matched_graph
    ea = _evaluate_matched_graph(graph, gt_graph)
    ea = ea.filter(ea["pred_valid"])
    pe = graph.edge_attrs()
    prob_of = dict(zip(zip(pe["source_id"].to_list(), pe["target_id"].to_list()),
                       pe["edge_prob"].to_list()))
    tp_p, fp_p = [], []
    for s, t, m in zip(ea["source_id"].to_list(), ea["target_id"].to_list(), ea[K_MASK].to_list()):
        (tp_p if m else fp_p).append(prob_of[(s, t)])
    tp_p, fp_p = np.array(tp_p), np.array(fp_p)
    print(f"\nedge_prob of the {len(tp_p)} TP: "
          f"min={tp_p.min():.3f} q25={np.quantile(tp_p,.25):.3f} med={np.median(tp_p):.3f}")
    print(f"edge_prob of the {len(fp_p)} FP: "
          f"min={fp_p.min():.3f} q25={np.quantile(fp_p,.25):.3f} med={np.median(fp_p):.3f}")
    for tau in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        print(f"    threshold {tau:.1f}: keeps {int((tp_p>=tau).sum()):>3d}/{len(tp_p)} TP, "
              f"{int((fp_p>=tau).sum()):>3d}/{len(fp_p)} FP")

    # --- second children: are they ever right? ---
    children = defaultdict(list)
    for s, t, p in zip(pe["source_id"].to_list(), pe["target_id"].to_list(), pe["edge_prob"].to_list()):
        children[s].append((t, p))
    matched_edges = {(s, t) for s, t, m in zip(ea["source_id"].to_list(),
                                               ea["target_id"].to_list(), ea[K_MASK].to_list()) if m}
    best_tp = worse_tp = best_fp = worse_fp = 0
    for s, kids in children.items():
        if len(kids) < 2:
            continue
        order = sorted(kids, key=lambda kp: -kp[1])
        for rank, (t, _) in enumerate(order):
            if (s, t) in matched_edges:
                best_tp += rank == 0
                worse_tp += rank > 0
            elif any((s, tt) in matched_edges or (ss, t) in matched_edges
                     for tt, _ in kids for ss in [s]):
                pass
    print(f"\nat predicted divisions, GT edges that were the most confident child: {best_tp}")
    print(f"                        GT edges that were the *less* confident child: {worse_tp}")

    # --- node spacing: how much room is there to de-duplicate? ---
    scale = np.asarray(SCALE)
    per_frame_nn = []
    for t in np.unique(coords[:, 0]):
        pts = coords[coords[:, 0] == t][:, 1:] * scale
        if len(pts) < 2:
            continue
        d, _ = cKDTree(pts).query(pts, k=2)
        per_frame_nn.append(d[:, 1])
    nn = np.concatenate(per_frame_nn)
    print(f"\nnearest-neighbour distance within a frame (um), n={len(nn)}:")
    for q in (1, 5, 10, 25, 50):
        print(f"    p{q:<2d} = {np.percentile(nn, q):.2f}")
    for r in (7, 8, 9, 10, 11, 12):
        print(f"    nodes with a neighbour closer than {r:>2d} um: "
              f"{int((nn < r).sum()):>6d}  ({100*(nn<r).mean():.1f}%)")
    print(f"\nnodes {len(coords)}, estimated {n_total:.0f}, "
          f"excess {len(coords)-n_total:.0f} ({100*(len(coords)-n_total)/n_total:.0f}%)")


if __name__ == "__main__":
    main()
