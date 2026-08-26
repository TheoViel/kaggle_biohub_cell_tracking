"""Reproduce the notebook's three scores, and break down where the metric goes.

Run:  python src/postprocessing/baseline.py
"""

import contextlib
import io

import numpy as np
import tracksdata as td

from common import (
    SAMPLE,
    load_gt,
    load_prediction,
    save,
    score,
    to_graph,
)

ILP_EDGE_W = -1.0
ILP_APPEARANCE_W = 0.1
ILP_DISAPPEARANCE_W = 0.1
ILP_DIVISION_W = 1.0


def run_ilp(graph, edge_w=ILP_EDGE_W, appear=ILP_APPEARANCE_W,
            disappear=ILP_DISAPPEARANCE_W, division=ILP_DIVISION_W):
    solver = td.solvers.ILPSolver(
        edge_weight=edge_w * td.EdgeAttr("edge_prob"),
        appearance_weight=appear,
        disappearance_weight=disappear,
        division_weight=division,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return solver.solve(graph).detach()


def run_pp_public(graph):
    """tracking.pp_public.postprocess_graph, back into a scoreable graph."""
    from tracking.pp_public import postprocess_graph

    with contextlib.redirect_stdout(io.StringIO()):
        nodes_pp, edges_pp, stats = postprocess_graph(graph, dataset=SAMPLE)
    id_map = {n["node_id"]: i for i, n in enumerate(nodes_pp)}
    coords = np.array([[n["t"], n["z"], n["y"], n["x"]] for n in nodes_pp])
    edges = [
        (id_map[e["source_id"]], id_map[e["target_id"]], e["edge_prob"], e["distance_um"])
        for e in edges_pp
    ]
    return to_graph(coords, edges), stats


def breakdown(graph, gt_graph):
    """Where the Jaccard actually comes from — only edges touching GT matter.

    The GT is sparse: a predicted edge is counted (as TP or FP) only when one
    of its endpoints matched a GT node that itself has a GT edge. Everything
    else is invisible to the score. Knowing which predicted edges are *in play*
    is what tells you whether to chase recall or precision.
    """
    from tracking_cellmot.metrics import _evaluate_matched_graph

    ea = _evaluate_matched_graph(graph, gt_graph)
    valid = ea.filter(ea["pred_valid"])
    tp = int(valid[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK].sum())
    print(f"    GT edges                     : {gt_graph.num_edges()}")
    print(f"    predicted edges              : {graph.num_edges()}")
    print(f"    ...of which the metric sees  : {len(valid)}  (touch a linked GT node)")
    print(f"    ...of which matched (TP)     : {tp}")
    print(f"    => FP = {len(valid) - tp}, FN = {gt_graph.num_edges() - tp}")


def main():
    gt_graph, n_total = load_gt()
    coords, edges = load_prediction()
    print(f"raw prediction: {len(coords)} nodes, {len(edges)} edges "
          f"(estimated nodes {n_total:.0f})\n")

    rows = []

    graph = to_graph(coords, edges)
    rows.append(score(graph, gt_graph, n_total, "raw"))
    breakdown(graph, gt_graph)

    graph_ilp = run_ilp(graph)
    rows.append(score(graph_ilp, gt_graph, n_total, "raw + ILP"))
    breakdown(graph_ilp, gt_graph)

    graph_pp, stats = run_pp_public(graph_ilp)
    rows.append(score(graph_pp, gt_graph, n_total, "raw + ILP + pp_public"))
    breakdown(graph_pp, gt_graph)
    print(f"    pp_public filter stats: {stats}")

    save(rows, "baseline")


if __name__ == "__main__":
    main()
