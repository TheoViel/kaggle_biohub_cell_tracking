"""Check the fast scorer against tracking_cellmot.metrics.evaluate.

Run:  python src/postprocessing/validate_fast.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tracksdata as td

from common import ROOT, score, to_graph
from fast_score import GroundTruth, metrics
from params import TRAIN_DIR
from pipeline import postprocess

NAMES = ["44b6_2f31fc2f", "44b6_0113de3b", "44b6_0c582fdc", "44b6_0db75fae", "44b6_a2bb48bb"]


def gt_of(name):
    r = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{name}.geff")
    return r[0] if isinstance(r, tuple) else r


def main():
    meta = pd.read_csv("../input/metadata.csv").set_index("sample_name")
    worst = 0.0
    for name in NAMES:
        n_total = float(meta.loc[name, "estimated_number_of_nodes"])
        d = np.load(ROOT / "logs/2026-08-20/20/preds" / f"{name}.npz", allow_pickle=True)
        coords = d["coords"].astype(np.int64)
        edges = d["edges"].astype(float).reshape(-1, 4)
        for label, (c, e) in (("raw", (coords, edges)), ("pp", postprocess(coords, edges))):
            fast = metrics(c, e, GroundTruth(gt_of(name)), n_total)
            real = score(to_graph(c, e), gt_of(name), n_total, "", verbose=False)
            gap = abs(fast["adj_edge_jaccard"] - real["adj_edge_jaccard"])
            worst = max(worst, gap)
            print(f"{name:<16s} {label:<3s}  fast {fast['edge_tp']:>4d}/{fast['edge_fp']:>3d}/"
                  f"{fast['edge_fn']:>3d} adjJ={fast['adj_edge_jaccard']:.4f}   "
                  f"real {real['edge_tp']:>4.0f}/{real['edge_fp']:>3.0f}/{real['edge_fn']:>3.0f} "
                  f"adjJ={real['adj_edge_jaccard']:.4f}   delta={gap:.4f}")
    print(f"\nlargest adjJ disagreement: {worst:.4f}")


if __name__ == "__main__":
    main()
