"""Shared harness for the post-processing experiments.

Every script here scores a (nodes, edges) candidate the same way the notebook
does, so numbers are comparable across strategies. Run any of them from
anywhere; ``params`` uses paths relative to ``src/``, so we chdir there.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT / "src")
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
import tracksdata as td  # noqa: E402

from params import INPUT_FOLDER, SCALE, TRAIN_DIR  # noqa: E402
from tracking.post_process import build_graph  # noqa: E402
from tracking_cellmot.metrics import (  # noqa: E402
    evaluate,
    node_recall,
    per_sample_metrics,
    summarise,
)

SAMPLE = "44b6_2f31fc2f"
EXPERIMENT = "2026-08-20_20"
PRED_NPZ = ROOT / "output" / "predictions" / EXPERIMENT / f"{SAMPLE}.npz"

# Where every strategy writes its scored result, so runs can be compared
# without re-running the expensive ones.
RESULTS = ROOT / "output" / "pp_experiments"


def load_prediction(path: Path | None = None):
    """The raw tracker output: (N, 4) [t, z, y, x] coords and (M, 4) edges.

    Edges are (source row, target row, probability, distance), indexing coords.
    """
    data = np.load(path or PRED_NPZ, allow_pickle=True)
    coords = data["coords"].astype(np.int64)
    edges = data["edges"].astype(np.float64).reshape(-1, 4)
    return coords, edges


def load_gt_graph(sample: str):
    """The ground-truth graph and the sample's estimated total node count.

    Matching mutates the graph, so load a fresh one per scored candidate.
    """
    result = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{sample}.geff")
    gt_graph = result[0] if isinstance(result, tuple) else result

    meta = pd.read_csv(INPUT_FOLDER + "metadata.csv")
    n_total = float(meta.loc[meta["sample_name"] == sample, "estimated_number_of_nodes"].iloc[0])
    return gt_graph, n_total


def load_gt():
    """``load_gt_graph`` for the sample under study."""
    return load_gt_graph(SAMPLE)


def to_graph(coords, edges):
    """build_graph, with the tuple/int coercion it expects."""
    # pp_public invents edges with no probability of their own; they are links
    # it is confident enough to add, so score them as such.
    edge_tuples = [
        (int(s), int(t), 1.0 if p is None else float(p), 0.0 if d is None else float(d))
        for s, t, p, d in np.asarray(edges, dtype=object).reshape(-1, 4)
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        return build_graph(np.asarray(coords), edge_tuples)


def score(graph, gt_graph, n_total, label="", verbose=True):
    """The notebook's metric block, as a dict."""
    with contextlib.redirect_stdout(io.StringIO()):
        res = evaluate(graph, gt_graph, scale=SCALE)
        recall = node_recall(graph, gt_graph)
    row = summarise([per_sample_metrics(res, n_total, recall)])
    row["label"] = label
    row["n_nodes"] = graph.num_nodes()
    row["n_edges"] = graph.num_edges()
    row["node_recall"] = recall
    if verbose:
        print(fmt(row))
    return row


def fmt(row):
    return (
        f"{row['label']:<34s} "
        f"nodes={row['n_nodes']:>6d} edges={row['n_edges']:>6d}  "
        f"tp={row['edge_tp']:>4.0f} fp={row['edge_fp']:>4.0f} fn={row['edge_fn']:>4.0f}  "
        f"recall={row['node_recall']:.3f}  "
        f"J={row['edge_jaccard']:.4f}  adjJ={row['adj_edge_jaccard']:.4f}"
    )


def save(rows, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\n-> wrote {path}")
