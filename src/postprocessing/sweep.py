"""Run named post-processing chains over many stacks and roll up the score.

Every chain is scored the way the run is scored — per-sample adjusted Jaccard,
weighted by sample size — using the exact fast reimplementation validated in
validate_fast.py.

Chains are declared in RECIPES as a list of (stage function, kwargs). Two
starting points are compared throughout, because a stage that helps the raw
graph may only be undoing something the pipeline already did:

    raw   -> stage
    raw   -> pipeline -> stage
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import tracksdata as td

from common import ROOT, RESULTS
from fast_score import GroundTruth, metrics, roll_up
from params import TRAIN_DIR

PRED_DIR = ROOT / "logs" / "2026-08-20" / "20" / "preds"
_META = None


def stack_names(limit=None, spread=False):
    names = [p.stem for p in sorted(PRED_DIR.glob("*.npz"))
             if (Path(TRAIN_DIR) / f"{p.stem}.geff").exists()]
    if spread and limit:
        # A subset that spans the difficulty range rather than the alphabet,
        # so a setting tuned on it is not tuned on the easy stacks.
        scores = json.load(open(ROOT / "logs/2026-08-20/20/scores_0.25.json"))["per_sample"]
        names.sort(key=lambda n: scores.get(n, {}).get("adj_edge_jaccard", 0.5))
        step = max(len(names) // limit, 1)
        return sorted(names[::step][:limit])
    return names[:limit]


def load(name):
    global _META
    if _META is None:
        _META = pd.read_csv("../input/metadata.csv").set_index("sample_name")
    d = np.load(PRED_DIR / f"{name}.npz", allow_pickle=True)
    coords = d["coords"].astype(np.int64)
    edges = d["edges"].astype(np.float64).reshape(-1, 4)
    r = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{name}.geff")
    gt = GroundTruth(r[0] if isinstance(r, tuple) else r)
    n_total = float(_META.loc[name, "estimated_number_of_nodes"])
    return coords, edges, gt, n_total


def _run_one(job):
    name, recipes = job
    coords, edges, gt, n_total = load(name)
    out = {}
    for label, chain in recipes.items():
        c, e = coords, edges
        t0 = time.time()
        try:
            for fn, kw in chain:
                c, e = fn(c, e, **({**kw, "sample": name} if "sample" in
                                   getattr(fn, "__code__").co_varnames else kw))
            row = metrics(c, e, gt, n_total)
        except Exception as exc:                       # a chain that blows up
            row = {"edge_tp": float("nan"), "error": repr(exc)[:120]}
        row["seconds"] = time.time() - t0
        out[label] = row
    return name, out


def run(recipes, names, jobs=12, tag="sweep", quiet=False):
    with Pool(jobs) as pool:
        results = dict(pool.imap_unordered(
            _run_one, [(n, recipes) for n in names], chunksize=1))

    table = []
    for label in recipes:
        rows = [r[label] for r in results.values() if label in r]
        good = [r for r in rows if r["edge_tp"] == r["edge_tp"]]
        if not good:
            err = next((r.get("error") for r in rows if r.get("error")), "all failed")
            print(f"{label:<44s} FAILED: {err}")
            continue
        s = roll_up(good)
        s["config"] = label
        s["n_failed"] = len(rows) - len(good)
        s["seconds"] = float(np.mean([r["seconds"] for r in rows]))
        s["hard"] = results.get("44b6_2f31fc2f", {}).get(label, {}).get(
            "adj_edge_jaccard", float("nan"))
        table.append(s)

    table.sort(key=lambda r: -r["adj_edge_jaccard"])
    if not quiet:
        print(f"\n{'chain':<44s} {'adjJ':>7s} {'J':>7s} {'tp':>7s} {'fp':>6s} "
              f"{'fn':>6s} {'recall':>7s} {'2f31fc2f':>9s} {'s/stack':>8s}")
        print("-" * 106)
        for r in table:
            print(f"{r['config']:<44s} {r['adj_edge_jaccard']:>7.4f} "
                  f"{r['edge_jaccard']:>7.4f} {r['edge_tp']:>7d} {r['edge_fp']:>6d} "
                  f"{r['edge_fn']:>6d} {r['node_recall']:>7.3f} {r['hard']:>9.3f} "
                  f"{r['seconds']:>8.1f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(json.dumps(
        {"table": table, "per_sample": results}, indent=1, default=float))
    return table, results


def add_common_args(ap):
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--spread", action="store_true",
                    help="with --limit, take a subset spanning the difficulty range")
    return ap
