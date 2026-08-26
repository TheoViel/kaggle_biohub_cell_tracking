"""Tune the pipeline on every validation stack, not on the one hard sample.

The stack the investigation started from is the worst of the 71 this run
predicts — 0.496 against a run-level 0.759. Settings chosen on it alone move it
a long way and move the rest the wrong way, so the grid is scored the way the
run is scored: per-sample adjusted Jaccard, weighted by sample size.

Run:  python src/postprocessing/tune.py [--jobs 8] [--limit N]
"""

import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import tracksdata as td

from common import ROOT, RESULTS
from fast_score import (
    GroundTruth,
    counts_from_flags,
    edge_flags,
    match_nodes,
    roll_up,
    row_from_counts,
)
from params import TRAIN_DIR
from pipeline import _flow_residual, _prune_short
from relink import relink

PRED_DIR = ROOT / "logs" / "2026-08-20" / "20" / "preds"

DISTANCES = (7.0, 7.5, 8.0, 8.5)
MIN_TRACKS = (1, 3, 5, 8)
MARGINS = (0.0, 1.0, 2.0, 3.0)
RESIDUALS = (np.inf, 5.0, 4.0, 3.0)


def key(d, mt, mm, mr):
    return f"d={d:g}|track={mt}|margin={mm:g}|res={mr:g}"


def work(name):
    """Every grid cell's counts for one stack."""
    gt_path = Path(TRAIN_DIR) / f"{name}.geff"
    result = td.graph.IndexedRXGraph.from_geff(gt_path)
    gt = GroundTruth(result[0] if isinstance(result, tuple) else result)

    meta = pd.read_csv("../input/metadata.csv").set_index("sample_name")
    n_total = float(meta.loc[name, "estimated_number_of_nodes"])

    d = np.load(PRED_DIR / f"{name}.npz", allow_pickle=True)
    coords = d["coords"].astype(np.int64)
    edges = d["edges"].astype(np.float64).reshape(-1, 4)

    out = {}
    # The tracker's own output, for reference.
    match = match_nodes(coords, gt)
    valid, is_tp = edge_flags(edges, match, gt)
    tp, fp, fn = counts_from_flags(np.ones(len(edges), bool), valid, is_tp, gt)
    out["baseline"] = row_from_counts(
        tp, fp, fn, len(coords), n_total, len(set(match.values())) / gt.n_nodes
    )

    for dist in DISTANCES:
        linked = relink(coords, edges, max_distance=dist)
        if len(linked) == 0:
            continue
        residual_full = _flow_residual(coords, linked, 30)
        for min_track in MIN_TRACKS:
            # Carry the residual through the pruning by riding it along as a column.
            rider = np.column_stack([linked, residual_full])
            c2, e2 = _prune_short(coords, rider, min_track)
            if len(e2) == 0:
                continue
            margin, residual = e2[:, 4], e2[:, 5]
            m2 = match_nodes(c2, gt)
            recall = len(set(m2.values())) / gt.n_nodes
            valid, is_tp = edge_flags(e2, m2, gt)
            for mm, mr in itertools.product(MARGINS, RESIDUALS):
                keep = (margin >= mm) & (residual <= mr)
                tp, fp, fn = counts_from_flags(keep, valid, is_tp, gt)
                out[key(dist, min_track, mm, mr)] = row_from_counts(
                    tp, fp, fn, len(c2), n_total, recall
                )
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    names = [p.stem for p in sorted(PRED_DIR.glob("*.npz"))][: args.limit]
    names = [n for n in names if (Path(TRAIN_DIR) / f"{n}.geff").exists()]
    print(f"{len(names)} stacks, "
          f"{len(DISTANCES)*len(MIN_TRACKS)*len(MARGINS)*len(RESIDUALS)} settings each\n")

    with Pool(args.jobs) as pool:
        results = dict(pool.imap_unordered(work, names))

    configs = sorted({k for r in results.values() for k in r})
    table = []
    for cfg in configs:
        rows = [r[cfg] for r in results.values() if cfg in r]
        if len(rows) < len(names):
            continue
        summary = roll_up(rows)
        summary["config"] = cfg
        hard = results.get("44b6_2f31fc2f", {}).get(cfg)
        summary["hard_sample"] = hard["adj_edge_jaccard"] if hard else float("nan")
        table.append(summary)

    table.sort(key=lambda r: -r["adj_edge_jaccard"])
    base = next(r for r in table if r["config"] == "baseline")
    print(f"{'config':<38s} {'adjJ':>7s} {'J':>7s} {'tp':>7s} {'fp':>6s} "
          f"{'fn':>6s} {'recall':>7s} {'2f31fc2f':>9s}")
    print(f"{'baseline':<38s} {base['adj_edge_jaccard']:>7.4f} {base['edge_jaccard']:>7.4f} "
          f"{base['edge_tp']:>7d} {base['edge_fp']:>6d} {base['edge_fn']:>6d} "
          f"{base['node_recall']:>7.3f} {base['hard_sample']:>9.3f}")
    print("-" * 92)
    for r in table[:20]:
        if r["config"] == "baseline":
            continue
        print(f"{r['config']:<38s} {r['adj_edge_jaccard']:>7.4f} {r['edge_jaccard']:>7.4f} "
              f"{r['edge_tp']:>7d} {r['edge_fp']:>6d} {r['edge_fn']:>6d} "
              f"{r['node_recall']:>7.3f} {r['hard_sample']:>9.3f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "tune.json").write_text(json.dumps(
        {"table": table, "per_sample": results}, indent=1, default=float))
    print(f"\n-> wrote {RESULTS / 'tune.json'}")


if __name__ == "__main__":
    main()
