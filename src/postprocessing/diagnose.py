"""Decompose the metric: exactly which predicted edges it sees, and why they miss.

The ground truth here is ~89 nodes out of an estimated 24k, so the score is
decided by a tiny neighbourhood of the graph. This prints that neighbourhood.

Run:  python src/postprocessing/diagnose.py
"""

import contextlib
import io
from collections import Counter, defaultdict

import numpy as np
import tracksdata as td

from common import load_gt, load_prediction, score, to_graph
from params import SCALE
from tracking_cellmot.metrics import evaluate

K_MASK = td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK
K_MATCH = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID


def matched_maps(graph, gt_graph):
    """gt node id -> pred node id, and back."""
    na = graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, K_MATCH])
    na = na.filter(na[K_MATCH] != -1)
    pred_of_gt = dict(zip(na[K_MATCH].to_list(), na[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list()))
    return pred_of_gt, {v: k for k, v in pred_of_gt.items()}


def main():
    gt_graph, n_total = load_gt()
    coords, edges = load_prediction()
    graph = to_graph(coords, edges)
    row = score(graph, gt_graph, n_total, "raw")

    pred_of_gt, gt_of_pred = matched_maps(graph, gt_graph)
    print(f"\nGT nodes {gt_graph.num_nodes()}, matched {len(pred_of_gt)}")

    # --- predicted adjacency, by node id ---
    pe = graph.edge_attrs()
    children = defaultdict(list)
    parents = defaultdict(list)
    for s, t, p in zip(pe["source_id"].to_list(), pe["target_id"].to_list(),
                       pe["edge_prob"].to_list()):
        children[s].append((t, p))
        parents[t].append((s, p))

    ge = gt_graph.edge_attrs(attr_keys=[K_MASK])
    gt_edges = list(zip(ge["source_id"].to_list(), ge["target_id"].to_list(), ge[K_MASK].to_list()))

    # --- why each GT edge was missed ---
    why = Counter()
    detail = []
    for src, tgt, ok in gt_edges:
        if ok:
            continue
        p, q = pred_of_gt.get(src), pred_of_gt.get(tgt)
        if p is None or q is None:
            why["endpoint undetected"] += 1
            continue
        got = [c for c, _ in children[p]]
        if q in got:
            why["edge exists but unmatched (?)"] += 1
            continue
        if not got and not parents[q]:
            why["parent childless AND child parentless"] += 1
        elif not got:
            why["parent given no child"] += 1
        elif not parents[q]:
            why["child given no parent"] += 1
        else:
            why["both linked, to each other's wrong partner"] += 1
        detail.append((src, tgt, p, q, children[p], parents[q]))
    print("\n--- why the 17 FN happened ---")
    for k, v in why.most_common():
        print(f"    {v:>3d}  {k}")

    # --- what the FPs are ---
    from tracking_cellmot.metrics import _evaluate_matched_graph
    ea = _evaluate_matched_graph(graph, gt_graph)
    valid = ea.filter(ea["pred_valid"] & ~ea[K_MASK])
    kinds = Counter()
    for s, t in zip(valid["source_id"].to_list(), valid["target_id"].to_list()):
        gs, gt_ = gt_of_pred.get(s), gt_of_pred.get(t)
        n_ch = len(children[s])
        if gs is not None and n_ch > 1:
            kinds[f"extra child of a matched node (deg={n_ch})"] += 1
        elif gs is not None:
            kinds["matched parent linked to a non-GT child"] += 1
        elif gt_ is not None:
            kinds["matched child linked from a non-GT parent"] += 1
        else:
            kinds["neither endpoint matched (?)"] += 1
    print(f"\n--- what the {len(valid)} FP edges are ---")
    for k, v in kinds.most_common():
        print(f"    {v:>3d}  {k}")

    # --- degree structure of the whole predicted graph ---
    out_deg = Counter(len(children[n]) for n in graph.node_ids())
    in_deg = Counter(len(parents[n]) for n in graph.node_ids())
    print("\n--- predicted graph degrees (all nodes) ---")
    print("    out:", dict(sorted(out_deg.items())))
    print("    in :", dict(sorted(in_deg.items())))
    n_div = sum(v for k, v in out_deg.items() if k > 1)
    print(f"    nodes with >1 child (predicted divisions): {n_div}")

    # --- the node-count penalty ---
    n_pred = graph.num_nodes()
    ratio = (n_pred - n_total) / n_total
    print("\n--- node-count penalty ---")
    print(f"    predicted {n_pred}, estimated {n_total:.0f} -> ratio {ratio:+.3f}, "
          f"factor {1 - 0.1 * ratio:.4f}")
    print(f"    J={row['edge_jaccard']:.4f} -> adjJ={row['adj_edge_jaccard']:.4f}")
    print(f"    at ratio 0 the same J would score {row['edge_jaccard']:.4f} "
          f"(+{row['edge_jaccard'] - row['adj_edge_jaccard']:.4f})")

    # --- how many nodes are structurally worthless ---
    isolated = [n for n in graph.node_ids() if not children[n] and not parents[n]]
    print(f"\n    isolated nodes (no parent, no child): {len(isolated)}")
    # track components by length
    seen, comp_sizes = set(), []
    adj = defaultdict(set)
    for s, t, _ in zip(pe["source_id"].to_list(), pe["target_id"].to_list(), pe["edge_prob"].to_list()):
        adj[s].add(t)
        adj[t].add(s)
    for n in graph.node_ids():
        if n in seen:
            continue
        stack, comp = [n], []
        seen.add(n)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comp_sizes.append(len(comp))
    comp_sizes = np.array(comp_sizes)
    print(f"    components: {len(comp_sizes)}, sizes "
          f"min={comp_sizes.min()} med={np.median(comp_sizes):.0f} max={comp_sizes.max()}")
    for cut in (2, 3, 5, 10, 20):
        small = comp_sizes[comp_sizes < cut]
        print(f"      components with <{cut:>2d} nodes: {len(small):>5d}  "
              f"holding {small.sum():>6d} nodes")


if __name__ == "__main__":
    main()
