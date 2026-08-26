"""Score the pipeline over every validation stack of the run, not just one.

A single stack carries ~85 annotated links, so one sample's Jaccard moves by
0.01 for every link. The run's own predictions cover 71 stacks; this scores
baseline and pipeline on all of them and reports the summarised score the
notebook reports, so a gain here is not one sample's luck.

Run:  python src/postprocessing/run_all.py [--limit N] [--jobs N]
"""

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import tracksdata as td

from common import ROOT, save, to_graph
from params import SCALE, TRAIN_DIR
from pipeline import postprocess
from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise

PRED_DIR = ROOT / "logs" / "2026-08-20" / "20" / "preds"


def score_one(name, coords, edges, gt_graph, n_total):
    with contextlib.redirect_stdout(io.StringIO()):
        graph = to_graph(coords, edges)
        res = evaluate(graph, gt_graph, scale=SCALE)
        recall = node_recall(graph, gt_graph)
    row = per_sample_metrics(res, n_total, recall)
    row["sample_name"] = name
    row["n_nodes"] = len(coords)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    meta = pd.read_csv("../input/metadata.csv").set_index("sample_name")
    paths = sorted(PRED_DIR.glob("*.npz"))[: args.limit]
    print(f"{len(paths)} stacks\n")

    base_rows, pp_rows = [], []
    for i, path in enumerate(paths):
        name = path.stem
        gt_path = Path(TRAIN_DIR) / f"{name}.geff"
        if not gt_path.exists():
            continue
        result = td.graph.IndexedRXGraph.from_geff(gt_path)
        gt_graph = result[0] if isinstance(result, tuple) else result
        n_total = float(meta.loc[name, "estimated_number_of_nodes"])

        d = np.load(path, allow_pickle=True)
        coords = d["coords"].astype(np.int64)
        edges = d["edges"].astype(np.float64).reshape(-1, 4)

        base = score_one(name, coords, edges, gt_graph, n_total)

        # evaluate() writes matching attributes onto gt_graph; take a clean
        # copy so the second scoring is not judged against the first's marks.
        result = td.graph.IndexedRXGraph.from_geff(gt_path)
        gt_graph2 = result[0] if isinstance(result, tuple) else result
        c2, e2 = postprocess(coords, edges)
        pp = score_one(name, c2, e2, gt_graph2, n_total)

        base_rows.append(base)
        pp_rows.append(pp)
        print(f"[{i+1:>3d}/{len(paths)}] {name}  "
              f"adjJ {base['adj_edge_jaccard']:.3f} -> {pp['adj_edge_jaccard']:.3f}   "
              f"tp {base['edge_tp']:.0f}->{pp['edge_tp']:.0f}  "
              f"fp {base['edge_fp']:.0f}->{pp['edge_fp']:.0f}  "
              f"nodes {base['n_nodes']}->{pp['n_nodes']}", flush=True)

    for label, rows in (("baseline", base_rows), ("pipeline", pp_rows)):
        s = summarise(rows)
        print(f"\n=== {label} over {s['n']} stacks ===")
        for k in ("edge_tp", "edge_fp", "edge_fn"):
            print(f"    {k}: {s[k]:.0f}")
        for k in ("edge_jaccard", "adj_edge_jaccard", "node_recall", "score"):
            print(f"    {k}: {s[k]:.4f}")

    save({"baseline": base_rows, "pipeline": pp_rows}, "run_all")


if __name__ == "__main__":
    main()
