"""Tune the division stage of `main_postprocess.py` on a tracker run.

    python postprocessing/exp_divisions_pp.py --jobs 24

Reads the same preds folder `main_postprocess.py` would, runs the base chain
(pipeline -> pp_public) once per stack, then scores every grid point of the
division stage against that cached result. The base chain is ~90% of the work
and it does not depend on the division knobs, so a 20-point sweep costs barely
more than one full pass.

`fast_score` is no use here -- it computes the edge Jaccard alone and is blind
to the division term, which is the whole point of this stage. This scores with
`tracking_cellmot.metrics.evaluate`, the same scorer main_postprocess reports.

`--stage pp_public` sweeps pp_public's own `safe_divisions` instead, so the
tuned stage can be compared against the thing it replaces rather than only
against nothing.
"""

from __future__ import annotations

import os

# One worker per stack already saturates the machine; BLAS's own pool on top of
# that oversubscribes by an order of magnitude and the sweep runs ~10x slower
# than it should. Set before numpy is imported, which is the only point the
# thread counts are read.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import tracksdata as td

# torch's intra-op pool ignores OMP_NUM_THREADS, and `recover`'s drift estimate
# goes through it -- same reason main_edge_postprocess.py sets this.
torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from params import EMBRYOS, LOG_FOLDER, SCALE, TRAIN_DIR  # noqa: E402
from stages import run_divisions, run_pipeline, run_pp_public  # noqa: E402
from tracking.post_process import build_graph  # noqa: E402
from tracking_cellmot.metrics import (  # noqa: E402
    evaluate,
    macro_summarise,
    node_recall,
    per_sample_metrics,
    summarise,
)
from utils.logger import get_last_log_folder  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

_META = None
_GT = {}


def _n_total(sample):
    global _META
    import pandas as pd
    if _META is None:
        meta_path = Path(TRAIN_DIR).parent / "metadata.csv"
        _META = (pd.read_csv(meta_path).set_index("sample_name")
                 if meta_path.exists() else pd.DataFrame())
    if sample in getattr(_META, "index", []):
        return float(_META.loc[sample, "estimated_number_of_nodes"])
    return float("nan")


def _gt_graph(sample):
    """The annotation, read once per worker -- geff parsing dominates otherwise."""
    if sample not in _GT:
        result = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{sample}.geff")
        _GT[sample] = result[0] if isinstance(result, tuple) else result
    return _GT[sample]


def _score(coords, edges, sample):
    graph = build_graph(np.round(np.asarray(coords)).astype(np.int64), [
        (int(s), int(t), float(p), float(d))
        for s, t, p, d in np.asarray(edges, float).reshape(-1, 4)
    ])
    gt = _gt_graph(sample)
    row = per_sample_metrics(evaluate(graph, gt, scale=SCALE),
                             _n_total(sample), node_recall(graph, gt))
    one = summarise([row])
    row["division_jaccard"] = one["division_jaccard"]
    row["score"] = one["score"]
    row["sample_name"] = sample
    row["embryo"] = sample.split("_")[0]
    row["fold"] = EMBRYOS.index(row["embryo"]) if row["embryo"] in EMBRYOS else None
    return row


def process(job):
    """One stack, every grid point: (label -> metrics row).

    ``--stage divisions`` runs the base chain once and varies only what comes
    after it, which is most of the saving. ``--stage base`` varies the chain
    itself, so each point costs a full pass; the division stage then runs on top
    with whatever ``div`` overrides the point carries, because a base setting
    that changes the node set changes what there is to fork from.
    """
    path, grid, base_cfg, stage = job
    sample = Path(path).stem
    if not (Path(TRAIN_DIR) / f"{sample}.geff").exists():
        return sample, {}

    data = np.load(path, allow_pickle=True)
    coords0 = data["coords"].astype(np.int64)
    edges0 = data["edges"].astype(np.float64).reshape(-1, 4)

    out = {}
    if stage == "base":
        for label, knobs in grid:
            knobs = knobs or {}
            c, e = run_pipeline(coords0, edges0,
                                **{**base_cfg["pipeline"], **knobs.get("pipeline", {})})
            c, e = run_pp_public(c, e, sample=sample,
                                 **{**base_cfg["pp_public"], **knobs.get("pp_public", {})})
            div = knobs.get("div", base_cfg.get("div"))
            if div is not None:
                c, e = run_divisions(c, e, **div)
            out[label] = _score(c, e, sample)
        return sample, out

    c1, e1 = run_pipeline(coords0, edges0, **base_cfg["pipeline"])
    coords, edges = run_pp_public(c1, e1, sample=sample, **base_cfg["pp_public"])
    for label, knobs in grid:
        if knobs is None:
            out[label] = _score(coords, edges, sample)
        elif "pp_public" in knobs:
            # The arm this stage replaces: pp_public's own safe_divisions,
            # which lives inside it and so needs its own pass.
            c, e = run_pp_public(c1, e1, sample=sample,
                                 **{**base_cfg["pp_public"], **knobs["pp_public"]})
            out[label] = _score(c, e, sample)
        else:
            c, e = run_divisions(coords, edges, **knobs)
            out[label] = _score(c, e, sample)
    return sample, out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log_folder", default=None)
    p.add_argument("--preds_dir", default=None)
    p.add_argument("--stage", default="divisions",
                   choices=("divisions", "pp_public", "base"))
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--grid", default=None,
                   help="json list of [label, knobs] pairs; default = the built-in grid")
    p.add_argument("--base", default=None,
                   help="json dict overriding the base chain's pipeline/pp_public knobs")
    p.add_argument("--out", default=None,
                   help="append each stack's rows here as jsonl, so a long sweep "
                        "survives being killed")
    p.add_argument("--report", default=None,
                   help="print the table from an --out file and exit")
    return p.parse_args()


# The chain everything is measured on top of. Mirrors main_postprocess.py's
# defaults; pp_public's own safe_divisions is off so the two division stages
# are never stacked.
BASE = {
    "pipeline": dict(recentre="auto", recover=1, recover_max_gap=3,
                     recover_passes=1, recover_max_residual=5.0),
    "pp_public": dict(filter_short_tracks=1, min_track_len=4,
                      relink_tight_um=6.5, safe_divisions=0),
}

DIV_BASE = dict(
    min_prob=0.0, max_parent_um=9.0, max_existing_child_um=10.0,
    max_sister_um=14.0, symmetry_tau=0.6, diverge_um=3.0,
    require_divergence=True, require_mutual_nn=True,
    frame_frac_cap=0.0076, global_frac_cap=0.00375,
)


def default_grid():
    grid = [
        ("off (no divisions at all)", None),
        ("pp_public safe_divisions", {"pp_public": {"safe_divisions": 1}}),
    ]
    for diverge, cap in itertools.product((2.25, 3.0, 4.0), (0.00375, 0.0075)):
        grid.append((f"div(diverge={diverge:g}, cap={cap:g})",
                     {**DIV_BASE, "diverge_um": diverge, "global_frac_cap": cap}))
    grid.append(("div(no divergence)",
                 {**DIV_BASE, "require_divergence": False}))
    grid.append(("div(no mutual_nn)", {**DIV_BASE, "require_mutual_nn": False}))
    grid.append(("div(symmetry off)", {**DIV_BASE, "symmetry_tau": 0.0}))
    return grid


def main():
    args = parse_args()

    if args.report:
        rows, order = {}, []
        for line in Path(args.report).read_text().splitlines():
            if not line.strip():
                continue
            for label, row in json.loads(line)["rows"].items():
                if label not in rows:
                    rows[label] = []
                    order.append(label)
                rows[label].append(row)
        n = {len(v) for v in rows.values()}
        print(f"-> {args.report}: {max(n) if n else 0} stacks, {len(order)} configs")
        _report([(label, None) for label in order], rows)
        return

    log_folder = Path(args.log_folder or get_last_log_folder(LOG_FOLDER))
    if args.preds_dir:
        preds_dir = Path(args.preds_dir)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from main_postprocess import _best_threshold_preds_dir
        preds_dir = _best_threshold_preds_dir(log_folder)

    base = dict(BASE)
    if args.base:
        for key, value in json.loads(args.base).items():
            base[key] = {**base.get(key, {}), **value}

    grid = ([(lbl, knobs) for lbl, knobs in json.loads(args.grid)] if args.grid
            else default_grid())
    if args.stage == "pp_public":
        grid = [(lbl, knobs or {}) for lbl, knobs in grid]

    paths = sorted(preds_dir.glob("*.npz"))[: args.limit]
    if not paths:
        raise SystemExit(f"No npz files under {preds_dir}")

    print(f"-> reading : {preds_dir}  ({len(paths)} stacks)")
    print(f"-> base    : {base}")
    print(f"-> stage   : {args.stage}   ({len(grid)} grid points)\n")

    rows = {label: [] for label, _ in grid}
    jobs = [(str(p), grid, base, args.stage) for p in paths]
    # A base-chain sweep is ~40 minutes and the table only exists at the end, so
    # a kill anywhere in it used to lose every stack. Appended per result
    # instead, and `--report` re-prints from the file without recomputing.
    dump = Path(args.out) if args.out else None
    if dump:
        dump.parent.mkdir(parents=True, exist_ok=True)
        dump.write_text("")
    with Pool(args.jobs) as pool:
        for i, (sample, per_label) in enumerate(
            pool.imap_unordered(process, jobs, chunksize=1)
        ):
            for label, row in per_label.items():
                rows[label].append(row)
            if dump:
                with dump.open("a") as f:
                    f.write(json.dumps({"sample": sample, "rows": _jsonable(per_label)}) + "\n")
            print(f"[{i + 1:>3d}/{len(paths)}] {sample}", flush=True)

    _report(grid, rows)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _report(grid, rows):
    print(f"\n{'=' * 104}")
    print(f"{'config':<34}{'cv':>9}{'Δ':>9}{'adj_ej':>9}{'div_j':>9}"
          f"{'div tp/fp/fn':>16}{'node_rec':>10}{'edges':>10}")
    print(f"{'=' * 104}")
    ref = None
    for label, _ in grid:
        rs = rows[label]
        if not rs:
            continue
        summary = summarise(rs)
        _, macro = macro_summarise(rs)
        cv = macro["score"]
        if ref is None:
            ref = cv
        div = (f"{int(summary.get('division_tp', 0))}/"
               f"{int(summary.get('division_fp', 0))}/"
               f"{int(summary.get('division_fn', 0))}")
        print(f"{label:<34}{cv:>9.4f}{cv - ref:>+9.4f}"
              f"{summary['adj_edge_jaccard']:>9.4f}"
              f"{summary['division_jaccard']:>9.4f}{div:>16}"
              f"{summary['node_recall']:>10.4f}"
              f"{int(summary['edge_tp'] + summary['edge_fp']):>10d}")


if __name__ == "__main__":
    main()
