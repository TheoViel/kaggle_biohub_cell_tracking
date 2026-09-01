"""Does the classifier gate still pay once tracking.pp_public runs after it?

    python edge_classifier/final_eval.py --jobs 24

pp_public re-links everything itself, so a gate placed before it may simply be
overwritten. This measures the full chains against the best one found without a
classifier (pipeline -> pp_public, 0.8197), out of fold throughout.
"""

import argparse
import json
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "postprocessing"))
os.chdir(SRC)

from multiprocessing import Pool  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tracksdata as td  # noqa: E402

from edge_classifier.gate import classifier_postprocess  # noqa: E402
from fast_score import GroundTruth, metrics, roll_up  # noqa: E402
from params import TRAIN_DIR  # noqa: E402
from postprocessing.pipeline import postprocess  # noqa: E402
from stages import run_pp_public  # noqa: E402

PRED_DIR = SRC.parent / "logs" / "2026-08-20" / "20" / "preds"
OUT = SRC.parent / "output" / "edge_classifier"
PP_BEST = dict(filter_short_tracks=1, min_track_len=4, relink_tight_um=6.5)
TAUS = (0.30, 0.45)

_BUNDLE = None
_META = None


def _model_for(sample):
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(OUT / "model.joblib")
    fold = _BUNDLE["stack_fold"].get(sample)
    return _BUNDLE["full"] if fold is None else _BUNDLE["folds"][fold]


def work(name):
    global _META
    if _META is None:
        _META = pd.read_csv("../input/metadata.csv").set_index("sample_name")
    n_total = float(_META.loc[name, "estimated_number_of_nodes"])

    data = np.load(PRED_DIR / f"{name}.npz", allow_pickle=True)
    coords = data["coords"].astype(np.int64)
    edges = data["edges"].astype(np.float64).reshape(-1, 4)
    result = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{name}.geff")
    gt = GroundTruth(result[0] if isinstance(result, tuple) else result)

    model = _model_for(name)
    rows = {}

    c, e = postprocess(coords, edges)
    rows["pipeline"] = metrics(c, e, gt, n_total)
    c2, e2 = run_pp_public(c, e, sample=name, **PP_BEST)
    rows["pipeline -> pp_public"] = metrics(c2, e2, gt, n_total)

    for tau in TAUS:
        c, e = classifier_postprocess(coords, edges, model, threshold=tau)
        rows[f"classifier({tau:.2f})"] = metrics(c, e, gt, n_total)
        c2, e2 = run_pp_public(c, e, sample=name, **PP_BEST)
        rows[f"classifier({tau:.2f}) -> pp_public"] = metrics(c2, e2, gt, n_total)

    return name, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    names = [p.stem for p in sorted(PRED_DIR.glob("*.npz"))
             if (Path(TRAIN_DIR) / f"{p.stem}.geff").exists()][: args.limit]
    print(f"{len(names)} stacks, out-of-fold\n")

    with Pool(args.jobs) as pool:
        results = dict(pool.imap_unordered(work, names))

    labels = list(next(iter(results.values())).keys())
    table = []
    print(f"{'chain':<34s} {'adjJ':>7s} {'J':>7s} {'tp':>7s} {'fp':>6s} {'fn':>6s}")
    print("-" * 72)
    for label in labels:
        rows = [r[label] for r in results.values() if label in r]
        s = roll_up(rows)
        s["config"] = label
        table.append(s)
    for s in sorted(table, key=lambda r: -r["adj_edge_jaccard"]):
        print(f"{s['config']:<34s} {s['adj_edge_jaccard']:>7.4f} {s['edge_jaccard']:>7.4f} "
              f"{s['edge_tp']:>7d} {s['edge_fp']:>6d} {s['edge_fn']:>6d}")

    (OUT / "final_eval.json").write_text(json.dumps(
        {"table": table, "per_sample": results}, indent=1, default=float))
    print(f"\n-> wrote {OUT / 'final_eval.json'}")


if __name__ == "__main__":
    main()
