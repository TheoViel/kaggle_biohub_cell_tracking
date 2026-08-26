"""Does the model rank the true parent highly when its argmax is wrong?

If the correct parent is usually the model's runner-up, a global assignment can
recover it. If the model buries it, no amount of post-processing will.

Run:  python src/postprocessing/analysis_candidates.py
"""

import contextlib
import io
from collections import defaultdict

import numpy as np
import tracksdata as td

from common import RESULTS, SAMPLE, load_gt, load_prediction, score, to_graph
from params import SCALE
from tracking_cellmot.metrics import evaluate

K_MASK = td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK
K_MATCH = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID


def load_candidates():
    d = np.load(RESULTS / f"{SAMPLE}_edge_candidates.npz")
    cand = d["candidates"]
    by_child = defaultdict(list)
    for t, child, parent, prob in cand:
        by_child[int(child)].append((int(parent), float(prob)))
    for child in by_child:
        by_child[child].sort(key=lambda pp: -pp[1])
    return by_child


def main():
    gt_graph, n_total = load_gt()
    coords, edges = load_prediction()
    graph = to_graph(coords, edges)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluate(graph, gt_graph, scale=SCALE)

    node_ids = sorted(graph.node_ids())
    row_of = {int(n): i for i, n in enumerate(node_ids)}

    na = graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, K_MATCH])
    na = na.filter(na[K_MATCH] != -1)
    pred_of_gt = dict(zip(na[K_MATCH].to_list(), na[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list()))

    by_child = load_candidates()

    ge = gt_graph.edge_attrs(attr_keys=[K_MASK])
    ranks_all, ranks_missed, probs_true, probs_top = [], [], [], []
    for gs, gt_id, ok in zip(ge["source_id"].to_list(), ge["target_id"].to_list(),
                             ge[K_MASK].to_list()):
        p, q = pred_of_gt.get(gs), pred_of_gt.get(gt_id)
        if p is None or q is None:
            continue
        rp, rq = row_of[p], row_of[q]
        cands = by_child.get(rq, [])
        rank = next((i for i, (par, _) in enumerate(cands) if par == rp), None)
        ranks_all.append(-1 if rank is None else rank)
        if not ok:
            ranks_missed.append(-1 if rank is None else rank)
        if rank is not None:
            probs_true.append(cands[rank][1])
            probs_top.append(cands[0][1])

    ranks_all = np.array(ranks_all)
    ranks_missed = np.array(ranks_missed)
    print(f"--- rank of the true parent in the model's own ranking ({len(ranks_all)} GT links) ---")
    for r in range(6):
        print(f"    rank {r}: {int((ranks_all == r).sum()):>3d}")
    print(f"    outside top-12: {int((ranks_all == -1).sum())}")

    print(f"\n--- the {len(ranks_missed)} links the tracker got wrong ---")
    for r in range(6):
        n = int((ranks_missed == r).sum())
        if n:
            print(f"    true parent was the model's rank {r}: {n}")
    print(f"    outside top-12: {int((ranks_missed == -1).sum())}")
    reachable = int((ranks_missed >= 0).sum())
    print(f"    => {reachable}/{len(ranks_missed)} are recoverable from these candidates")
    print(f"    ceiling on TP: {int((ranks_all >= 0).sum())} of {gt_graph.num_edges()}")

    probs_true, probs_top = np.array(probs_true), np.array(probs_top)
    ratio = probs_true / np.maximum(probs_top, 1e-9)
    print(f"\n--- confidence ---")
    print(f"    p(true parent):   med={np.median(probs_true):.3f}")
    print(f"    p(model's best):  med={np.median(probs_top):.3f}")
    print(f"    ratio true/best:  med={np.median(ratio):.3f}, "
          f"{int((ratio > 0.5).sum())}/{len(ratio)} above 0.5")


if __name__ == "__main__":
    main()
