"""
Post-process every stack of a prediction run, and score the result.

Reads  <log_folder>/preds/*.npz     — the tracker's own output
Writes <log_folder>/preds_pp/*.npz  — the same format, post-processed
       <log_folder>/geff_pp/*.geff  — the same graphs, for the viewer
       <log_folder>/preds_pp/metrics.csv, summary.json

    python main_postprocess.py --log_folder ../logs/2026-08-20/20/

Three stages, each switchable — see src/postprocessing/README.md for how they were
chosen and what each is worth. Measured over the 71 validation stacks of
logs/2026-08-20/20/, adjusted edge Jaccard:

    tracker as-is                       0.7592
    --no_pp_public   (pipeline only)    0.8065   improves 68 of 71 stacks
    (default)                           0.8197   improves 59, regresses 10
    --ilp                               0.8203   +0.0006 for ~40 s a stack

Stacks are shared across worker processes; nothing here needs a GPU or the
image data.
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import tracksdata as td

# postprocessing/ holds the stages next to the experiments that chose them, and
# they import each other by bare name; putting the directory on the path lets
# that work whether this is run from src/ or from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent / "postprocessing"))

from params import SCALE, TRAIN_DIR  # noqa: E402
from stages import run_ilp, run_pipeline, run_pp_public  # noqa: E402
from tracking.post_process import build_graph  # noqa: E402
from tracking_cellmot.io import save_graph  # noqa: E402
from tracking_cellmot.metrics import (  # noqa: E402
    evaluate,
    node_recall,
    per_sample_metrics,
    summarise,
)

# Tuned on a 10-stack subset spanning the difficulty range; src/postprocessing/exp_pp_public.py.
PP_PUBLIC_TUNED = dict(filter_short_tracks=1, min_track_len=4, relink_tight_um=6.5)
ILP_TUNED = dict(appear=0.02, disappear=0.02, division=1.0, timeout=900.0)

_META = None


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--log_folder", required=True,
                   help="run folder holding preds/*.npz")
    p.add_argument("--preds_dir", default=None,
                   help="override the input directory (default <log_folder>/preds)")
    p.add_argument("--out_dir", default=None,
                   help="override the npz output (default <log_folder>/preds_pp)")
    p.add_argument("--geff_dir", default=None,
                   help="override the geff output (default <log_folder>/geff_pp)")

    p.add_argument("--no_pipeline", action="store_true",
                   help="skip the re-linking and short-track pruning")
    p.add_argument("--no_pp_public", action="store_true",
                   help="skip tracking.pp_public.postprocess_graph")
    p.add_argument("--ilp", action="store_true",
                   help="run the ILP solver as a final pass (slow, worth ~+0.0006)")

    p.add_argument("--jobs", type=int, default=8, help="worker processes")
    p.add_argument("--limit", type=int, default=None, help="cap the stacks (debugging)")
    p.add_argument("--skip_existing", action="store_true",
                   help="leave stacks that already have an output npz alone")
    p.add_argument("--no_score", action="store_true",
                   help="post-process only; do not score against ground truth")
    return p.parse_args()


def _to_graph(coords, edges):
    """build_graph, with the tuple/int coercion it expects."""
    return build_graph(np.asarray(coords), [
        (int(s), int(t), float(p), float(d))
        for s, t, p, d in np.asarray(edges, dtype=float).reshape(-1, 4)
    ])


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


def _n_total(sample):
    """The stack's estimated node count, for the node-count adjustment."""
    global _META
    if _META is None:
        meta_path = Path(TRAIN_DIR).parent / "metadata.csv"
        _META = pd.read_csv(meta_path).set_index("sample_name") if meta_path.exists() \
            else pd.DataFrame()
    if sample in getattr(_META, "index", []):
        return float(_META.loc[sample, "estimated_number_of_nodes"])
    return float("nan")


def process(job):
    """One stack: post-process, save, score. Returns a metrics row or None."""
    path, cfg = job
    sample = Path(path).stem
    out_npz = Path(cfg["out_dir"]) / f"{sample}.npz"
    if cfg["skip_existing"] and out_npz.exists():
        return sample, None, "skipped (output exists)"

    data = np.load(path, allow_pickle=True)
    coords = data["coords"].astype(np.int64)
    edges = data["edges"].astype(np.float64).reshape(-1, 4)
    n_in = (len(coords), len(edges))

    if not cfg["no_pipeline"]:
        coords, edges = run_pipeline(coords, edges)
    if not cfg["no_pp_public"]:
        coords, edges = run_pp_public(coords, edges, sample=sample, **PP_PUBLIC_TUNED)
    if cfg["ilp"]:
        coords, edges = run_ilp(coords, edges, **ILP_TUNED)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        coords=np.asarray(coords, dtype=np.int16),
        edges=np.asarray(edges, dtype=np.float64).reshape(-1, 4),
    )

    graph = _to_graph(coords, edges)
    Path(cfg["geff_dir"]).mkdir(parents=True, exist_ok=True)
    save_graph(graph, str(Path(cfg["geff_dir"]) / f"{sample}.geff"))

    note = f"{n_in[0]}n/{n_in[1]}e -> {len(coords)}n/{len(edges)}e"
    if cfg["no_score"]:
        return sample, None, note

    gt_path = Path(TRAIN_DIR) / f"{sample}.geff"
    if not gt_path.exists():
        return sample, None, note + "  (no ground truth)"

    result = td.graph.IndexedRXGraph.from_geff(gt_path)
    gt_graph = result[0] if isinstance(result, tuple) else result
    res = evaluate(graph, gt_graph, scale=SCALE)
    recall = node_recall(graph, gt_graph)
    row = per_sample_metrics(res, _n_total(sample), recall)
    row["sample_name"] = sample
    row["n_nodes_pred"] = len(coords)
    row["n_edges_pred"] = len(edges)
    row["n_nodes_estimated"] = _n_total(sample)
    return sample, row, note


def main():
    args = parse_args()
    log_folder = Path(args.log_folder)
    preds_dir = Path(args.preds_dir) if args.preds_dir else log_folder / "preds"
    out_dir = Path(args.out_dir) if args.out_dir else log_folder / "preds_pp"
    geff_dir = Path(args.geff_dir) if args.geff_dir else log_folder / "geff_pp"

    paths = sorted(preds_dir.glob("*.npz"))[: args.limit]
    if not paths:
        raise SystemExit(f"No npz files under {preds_dir}")

    stages = [name for name, on in (
        ("pipeline", not args.no_pipeline),
        ("pp_public(tuned)", not args.no_pp_public),
        ("ILP(tuned)", args.ilp),
    ) if on]

    print(f"-> reading   : {preds_dir}  ({len(paths)} stacks)")
    print(f"-> npz out   : {out_dir}")
    print(f"-> geff out  : {geff_dir}")
    print(f"-> stages    : {' -> '.join(stages) or 'none (pass-through)'}")
    print(f"-> workers   : {args.jobs}\n")

    cfg = dict(
        out_dir=str(out_dir), geff_dir=str(geff_dir),
        no_pipeline=args.no_pipeline, no_pp_public=args.no_pp_public, ilp=args.ilp,
        no_score=args.no_score, skip_existing=args.skip_existing,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    geff_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with Pool(args.jobs) as pool:
        for i, (sample, row, note) in enumerate(
            pool.imap_unordered(process, [(str(p), cfg) for p in paths], chunksize=1)
        ):
            score = f"  adjJ={row['adj_edge_jaccard']:.4f}" if row else ""
            print(f"[{i + 1:>3d}/{len(paths)}] {sample:<16s} {note}{score}", flush=True)
            if row:
                rows.append(row)

    if not rows:
        print(f"\n-> wrote {len(paths)} stacks to {out_dir} and {geff_dir} (not scored)")
        return

    df = pd.DataFrame(rows).sort_values("sample_name")
    df.to_csv(out_dir / "metrics.csv", index=False)
    summary = summarise(rows)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(_jsonable(summary), f, indent=4, default=str)

    print(f"\n{'=' * 60}")
    print(f"{len(rows)} stacks  ->  {out_dir}")
    print(f"{'=' * 60}")
    for k in ("edge_tp", "edge_fp", "edge_fn"):
        if k in summary:
            print(f"- {k}: {summary[k]:.0f}")
    for k in ("node_recall", "edge_jaccard", "adj_edge_jaccard",
              "division_jaccard", "score"):
        if k in summary:
            print(f"- {k}: {summary[k]:.4f}")
    print(f"\n-> metrics : {out_dir / 'metrics.csv'}")
    print(f"-> summary : {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
