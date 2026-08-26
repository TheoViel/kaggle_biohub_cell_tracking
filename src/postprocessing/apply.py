"""Run the pipeline on one npz, score it, and optionally write the geff.

    python src/postprocessing/apply.py                                  # the sample under study
    python src/postprocessing/apply.py --npz path/to/x.npz --sample x   # any other
    python src/postprocessing/apply.py --save                           # also write a geff
                                                                # for the viewer
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tracksdata as td

from common import EXPERIMENT, PRED_NPZ, ROOT, SAMPLE, score, to_graph
from params import TRAIN_DIR
from pipeline import DEFAULTS, postprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(PRED_NPZ))
    ap.add_argument("--sample", default=SAMPLE)
    ap.add_argument("--save", action="store_true",
                    help="write output/predictions/<exp>_pp/<sample>.geff")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v) if not isinstance(v, float) else float,
                        default=v)
    args = ap.parse_args()

    cfg = {k: getattr(args, k) for k in DEFAULTS}
    print(f"sample {args.sample}\nsettings {cfg}\n")

    d = np.load(args.npz, allow_pickle=True)
    coords = d["coords"].astype(np.int64)
    edges = d["edges"].astype(np.float64).reshape(-1, 4)

    result = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{args.sample}.geff")
    gt_graph = result[0] if isinstance(result, tuple) else result
    meta = pd.read_csv("../input/metadata.csv").set_index("sample_name")
    n_total = float(meta.loc[args.sample, "estimated_number_of_nodes"])

    score(to_graph(coords, edges), gt_graph, n_total, "before")

    c2, e2 = postprocess(coords, edges, **cfg)
    result = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{args.sample}.geff")
    gt2 = result[0] if isinstance(result, tuple) else result
    graph = to_graph(c2, e2)
    score(graph, gt2, n_total, "after")

    if args.save:
        from tracking_cellmot.io import save_graph
        out = ROOT / "output" / "predictions" / f"{EXPERIMENT}_pp"
        out.mkdir(parents=True, exist_ok=True)
        save_graph(graph, str(out / f"{args.sample}.geff"))
        print(f"\n-> wrote {out / f'{args.sample}.geff'}")


if __name__ == "__main__":
    main()
