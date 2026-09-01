"""Score the classifier gate against the gates it replaces, out of fold.

    python edge_classifier/evaluate.py --jobs 24

Every stack is judged by the fold model that never saw it, so the numbers are
not the model grading its own homework. The margin and residual gates are
recomputed on the identical prepared links, so the comparison is only ever
about which links each rule chooses to drop.
"""

import argparse
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

from edge_classifier.features import link_features  # noqa: E402
from edge_classifier.gate import BREAK_EVEN  # noqa: E402
from edge_classifier.prepare import prepare  # noqa: E402
from fast_score import GroundTruth, metrics, roll_up  # noqa: E402
from params import TRAIN_DIR  # noqa: E402

PRED_DIR = SRC.parent / "logs" / "2026-08-20" / "20" / "preds"
OUT = SRC.parent / "output" / "edge_classifier"

THRESHOLDS = (0.0, 0.30, 0.45, BREAK_EVEN, 0.70, 0.75, 0.80, 0.85)
MARGINS = (1.0, 2.0, 3.0)

_BUNDLE = None
_META = None


def _model_for(sample):
    """The fold model that was not trained on this stack."""
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
    raw_coords = data["coords"].astype(np.int64)
    raw_edges = data["edges"].astype(np.float64).reshape(-1, 4)

    result = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{name}.geff")
    gt = GroundTruth(result[0] if isinstance(result, tuple) else result)

    rows = {"raw": metrics(raw_coords, raw_edges, gt, n_total)}

    coords, linked, npz_edges = prepare(raw_coords, raw_edges)
    if len(linked) == 0:
        return name, rows

    rows["pipeline (no gate)"] = metrics(coords, linked[:, :4], gt, n_total)
    for m in MARGINS:
        kept = linked[linked[:, 4] >= m]
        rows[f"margin>={m:g}um"] = metrics(coords, kept[:, :4], gt, n_total)

    X = link_features(coords, npz_edges, linked)
    p = _model_for(name).predict_proba(X)[:, 1]
    for tau in THRESHOLDS:
        kept = linked[p >= tau]
        rows[f"classifier p>={tau:.2f}"] = metrics(coords, kept[:, :4], gt, n_total)

    return name, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    names = [p.stem for p in sorted(PRED_DIR.glob("*.npz"))
             if (Path(TRAIN_DIR) / f"{p.stem}.geff").exists()][: args.limit]
    print(f"{len(names)} stacks, out-of-fold predictions\n")

    with Pool(args.jobs) as pool:
        results = dict(pool.imap_unordered(work, names))

    labels = list(next(iter(results.values())).keys())
    base = roll_up([r["pipeline (no gate)"] for r in results.values()
                    if "pipeline (no gate)" in r])

    print(f"{'gate':<26s} {'adjJ':>7s} {'J':>7s} {'tp':>7s} {'fp':>6s} "
          f"{'dTP':>6s} {'dFP':>6s} {'FP per TP':>10s}")
    print("-" * 82)
    table = []
    for label in labels:
        rows = [r[label] for r in results.values() if label in r]
        if len(rows) < len(names):
            continue
        s = roll_up(rows)
        s["config"] = label
        dtp = s["edge_tp"] - base["edge_tp"]
        dfp = s["edge_fp"] - base["edge_fp"]
        ratio = f"{dfp / dtp:.2f}" if dtp else "-"
        print(f"{label:<26s} {s['adj_edge_jaccard']:>7.4f} {s['edge_jaccard']:>7.4f} "
              f"{s['edge_tp']:>7d} {s['edge_fp']:>6d} {dtp:>6d} {dfp:>6d} {ratio:>10s}")
        table.append(s)

    # The gate must remove this many false positives per true positive to pay.
    need = (base["edge_tp"] + base["edge_fn"] + base["edge_fp"]) / base["edge_tp"]
    print(f"\nbreak-even: a gate must remove {need:.2f} FP per TP it costs")

    import json
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluate.json").write_text(json.dumps(
        {"table": table, "per_sample": results}, indent=1, default=float))
    print(f"-> wrote {OUT / 'evaluate.json'}")


if __name__ == "__main__":
    main()
