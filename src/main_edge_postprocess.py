"""Post-process the edge transformer's predictions, and score the result.

Reads  <preds_dir>/*.npz   — written by main_edge_predict.py --dump_candidates
Writes <out_dir>/*.npz     — the same format, post-processed
       <geff_dir>/*.geff   — the same graphs, for the viewer
       <preds_dir>/../scores_<out_dir name>.json

    python main_edge_postprocess.py --preds_dir ../logs/edge_run/2026-09-04/3/preds_edge

The stages are in `postprocessing/edge_pp.py` and the reasoning for each is
there; the chain is

    assign (+ per-track momentum) -> recover -> prune -> linefit

and every stage is switchable, so an ablation is one flag. What separates this
from `main_postprocess.py` is the input: that one post-processes a graph whose
edges are already committed and re-decides them on geometry alone, because the
detector's probabilities are not worth much. Here the model scored every
candidate and the npz carries all of them, so the assignment is re-solved from
the probabilities themselves and the geometric stages only fill what the model
could not see -- the nodes that are missing, and the chains it left in pieces.

Nothing here needs a GPU or the image data.
"""

import warnings
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import tracksdata as td

sys.path.insert(0, str(Path(__file__).resolve().parent / "postprocessing"))

import edge_pp  # noqa: E402
import torch  # noqa: E402

from params import EMBRYOS, LOG_FOLDER, SCALE, TRAIN_DIR  # noqa: E402
from tracking.post_process import build_graph  # noqa: E402
from tracking_cellmot.io import save_graph  # noqa: E402
from tracking_cellmot.metrics import (  # noqa: E402
    evaluate,
    macro_summarise,
    node_recall,
    per_sample_metrics,
    summarise,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Tuned on 16 of the 71 fold-0 stacks spanning the node-excess range, then
# checked on all 71 -- see postprocessing/README_edge.md, including the two
# settings that won on the subset and lost on the fold.
#
# `min_track` and `recover_min_dup` were re-tuned again for
# `logs/edge_run/2026-09-07/2/preds_0.45` -- a newer, more over-detected
# checkpoint (node ratio 1.157 raw against the 1.081 the block above was tuned
# on) -- see README_edge.md's "Retuned again for preds_0.45". min_track went
# 2 -> 5 and recover_min_dup 4.0 -> 3.0; everything else held.
DEFAULTS = dict(
    threshold=0.4,
    recover=1,
    recover_max_gap=5,
    recover_max_residual=5.0,
    # 3.0, not the 4.0 this was tuned to. On a more over-detected graph the
    # nodes recover would skip as "too close to an existing one" are more
    # often real duplicates than genuine misses, so letting it insert a little
    # closer and leaving `min_track` to clean up the result pays again -- the
    # opposite direction from "Retuned for preds_pp input" below, on a graph
    # that has moved further along the same axis.
    recover_min_dup=3.0,
    through=0.0,          # measured, finds nothing -- see the README
    through_max_gap=2,
    # 5, not the 2 this was tuned to on preds_pp input. That retune reasoned
    # from "little left to recover, so pruning deletes real short tracks" --
    # true there, false here: preds_0.45 over-detects hard enough (ratio 1.157
    # raw, 1.196 through the old min_track=2 chain) that pruning at 2 is again
    # cheap unwanted fragments, not real tracks. Worth +0.0031 over min_track=2 on the full
    # 71, positive on both halves of the fold and on 57 of 71 stacks
    # individually -- see README_edge.md.
    min_track=5,
    linefit=0.6,
    linefit_window=3,
    vel_weight=0.02,
    vel_momentum=0.7,
)

# torch is only here for `recover`'s drift estimate, and its intra-op pool
# ignores OMP_NUM_THREADS; one worker per core would oversubscribe 3x.
torch.set_num_threads(1)

_META = None


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--preds_dir", default=None,
                   help="the npz main_edge_predict.py wrote; default = the "
                        "best-scoring preds_* of the latest run under "
                        f"{LOG_FOLDER}edge_run/, which is the one a threshold "
                        "sweep meant you to use")
    p.add_argument("--out_dir", default=None,
                   help="default: <preds_dir>_pp")
    p.add_argument("--geff_dir", default=None,
                   help="default: <out_dir> with 'preds' replaced by 'geff'")
    p.add_argument("--threshold", type=float, default=DEFAULTS["threshold"],
                   help="probability a link must beat to be committed "
                        "(default %(default)s)")
    p.add_argument("--no_recover", action="store_true",
                   help="skip putting back the detections missed for a frame or two")
    p.add_argument("--recover_max_gap", type=int, default=DEFAULTS["recover_max_gap"],
                   help="longest run of missing frames to fill (default %(default)s)")
    p.add_argument("--recover_max_residual", type=float,
                   default=DEFAULTS["recover_max_residual"],
                   help="um of drift-corrected motion allowed across a one-frame "
                        "gap, scaled by span for longer ones (default %(default)s)")
    p.add_argument("--recover_min_dup", type=float,
                   default=DEFAULTS["recover_min_dup"],
                   help="um; do not insert a node beside an existing detection "
                        "(default %(default)s)")
    p.add_argument("--through", type=float, default=DEFAULTS["through"],
                   help="um; join two chains through an unlinked detection that "
                        "already sits in the gap. 0 turns it off")
    p.add_argument("--through_max_gap", type=int, default=DEFAULTS["through_max_gap"])
    p.add_argument("--min_track", type=int, default=DEFAULTS["min_track"],
                   help="drop chains shorter than this (default %(default)s)")
    p.add_argument("--linefit", type=float, default=DEFAULTS["linefit"],
                   help="blend each node this far toward a line fitted along its "
                        "own track. 0 turns it off (default %(default)s)")
    p.add_argument("--linefit_window", type=int, default=DEFAULTS["linefit_window"])
    p.add_argument("--vel_weight", type=float, default=DEFAULTS["vel_weight"],
                   help="log-odds charged per um a link departs from where the "
                        "parent track's own momentum says it should go. 0 turns "
                        "it off (default %(default)s)")
    p.add_argument("--vel_momentum", type=float, default=DEFAULTS["vel_momentum"],
                   help="EMA on that velocity (default %(default)s)")
    p.add_argument("--jobs", type=int, default=16, help="worker processes")
    p.add_argument("--limit", type=int, default=None, help="cap the stacks (debugging)")
    p.add_argument("--no_score", action="store_true",
                   help="post-process only; do not score against ground truth")
    return p.parse_args()


def _latest_run(base_dir: Path) -> Path:
    """The most recently written run under ``base_dir/<date>/<exp_id>/``."""
    candidates = sorted(base_dir.glob("*/*/args.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No run with an args.json found under {base_dir}")
    return candidates[-1].parent


def _folder_score(preds_dir: Path) -> float | None:
    """A preds folder's headline score, from the json main_edge_predict wrote.

    ``score`` is the competition number; ``edge_jaccard`` stands in for a split
    where it could not be formed (no divisions present, say). A NaN is treated
    as no score at all rather than compared, since NaN loses every comparison
    and would quietly rank last -- or first, depending which way it is written.
    """
    path = preds_dir.parent / f"scores_{preds_dir.name}.json"
    if not path.exists():
        return None
    overall = (json.load(open(path)).get("overall") or {})
    for key in ("score", "edge_jaccard"):
        value = overall.get(key)
        if value is not None and value == value:
            return float(value)
    return None


def _best_preds_dir(base_dir: Path) -> tuple[Path, str]:
    """The best-scoring ``preds_*`` folder of the latest run, and why it won.

    A threshold sweep leaves several folders written within seconds of each
    other, so "the most recent" picks between them essentially at random. The
    scores are sitting right there beside them, so rank by those instead and
    fall back to mtime only when nothing was scored (a ``--no_score`` run).

    Skips folders already suffixed ``_pp``: those are this script's own output,
    not raw candidates to post-process.
    """
    run_dir = _latest_run(base_dir)
    # Sorted so a tie resolves the same way twice: the metric only judges edges
    # touching an annotated node, a few dozen of the ~25k committed, so adjacent
    # thresholds routinely score identically and glob order would decide.
    candidates = sorted(
        d for d in run_dir.glob("preds_*") if d.is_dir() and not d.name.endswith("_pp")
    )
    if not candidates:
        raise FileNotFoundError(f"No preds_* folder found under {run_dir}")

    scored = [(d, s) for d in candidates if (s := _folder_score(d)) is not None]
    if scored:
        best, score = max(scored, key=lambda kv: kv[1])
        others = ", ".join(f"{d.name}={s:.4f}" for d, s in sorted(scored, key=lambda kv: -kv[1])[1:])
        return best, (f"best score {score:.4f}" + (f"; also {others}" if others else ""))
    return max(candidates, key=lambda d: d.stat().st_mtime), "most recent (none scored)"


def postprocess(coords, edges, cand, cfg):
    """The chain. `(coords, edges, candidates)` -> `(coords, edges)`."""
    assign_kw = dict(vel_weight=cfg["vel_weight"],
                     vel_momentum=cfg["vel_momentum"])
    edges = edge_pp.assign(coords, cand, threshold=cfg["threshold"], **assign_kw)
    if cfg["recover"]:
        coords, edges = edge_pp.recover_nodes(
            coords, edges, cand,
            max_gap=cfg["recover_max_gap"],
            max_residual=cfg["recover_max_residual"],
            min_dup=cfg["recover_min_dup"],
            threshold=cfg["threshold"],
            assign_kw=assign_kw,
        )
    if cfg["through"]:
        # Before pruning: the isolated nodes it wires in are exactly the
        # one-node "tracks" pruning is about to delete.
        edges = edge_pp.link_through_isolated(
            coords, edges, max_gap=cfg["through_max_gap"], max_um=cfg["through"])
    if cfg["min_track"]:
        coords, edges = edge_pp.prune_short(coords, edges, cfg["min_track"])
    if cfg["linefit"]:
        coords = edge_pp.linefit_smooth(
            coords, edges, window=cfg["linefit_window"], weight=cfg["linefit"])
    return coords, edges


def _n_total(sample):
    global _META
    if _META is None:
        meta_path = Path(TRAIN_DIR).parent / "metadata.csv"
        _META = (pd.read_csv(meta_path).set_index("sample_name")
                 if meta_path.exists() else pd.DataFrame())
    if sample in getattr(_META, "index", []):
        return float(_META.loc[sample, "estimated_number_of_nodes"])
    return float("nan")


def process(job):
    path, cfg = job
    sample = Path(path).stem
    coords, edges, cand = edge_pp.load_pred(path)
    if len(cand) == 0:
        raise SystemExit(
            f"{path} carries no candidates; re-run main_edge_predict.py "
            "with --dump_candidates"
        )
    coords, edges = postprocess(coords, edges, cand, cfg)

    out_npz = Path(cfg["out_dir"]) / f"{sample}.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        # round, not truncate: linefit smoothing leaves sub-voxel positions and
        # astype() would drag every one of them toward zero.
        coords=np.round(np.asarray(coords, np.float64)).astype(np.int16),
        edges=np.asarray(edges, np.float64).reshape(-1, 4),
    )

    graph = build_graph(np.round(np.asarray(coords)).astype(np.int64), [
        (int(s), int(t), float(p), float(d))
        for s, t, p, d in np.asarray(edges, float).reshape(-1, 4)
    ])
    Path(cfg["geff_dir"]).mkdir(parents=True, exist_ok=True)
    save_graph(graph, str(Path(cfg["geff_dir"]) / f"{sample}.geff"))

    gt_path = Path(TRAIN_DIR) / f"{sample}.geff"
    if cfg["no_score"] or not gt_path.exists():
        return sample, None

    result = td.graph.IndexedRXGraph.from_geff(gt_path)
    gt_graph = result[0] if isinstance(result, tuple) else result
    res = evaluate(graph, gt_graph, scale=SCALE)
    row = per_sample_metrics(res, _n_total(sample), node_recall(graph, gt_graph))
    one = summarise([row])
    row["division_jaccard"] = one["division_jaccard"]
    row["score"] = one["score"]
    row["sample_name"] = sample
    row["embryo"] = sample.split("_")[0]
    row["fold"] = EMBRYOS.index(row["embryo"]) if row["embryo"] in EMBRYOS else None
    row["n_nodes_pred"] = len(coords)
    row["n_edges_pred"] = len(edges)
    return sample, row


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


def main():
    args = parse_args()
    if args.preds_dir:
        preds_dir = Path(args.preds_dir)
    else:
        preds_dir, why = _best_preds_dir(Path(LOG_FOLDER) / "edge_run")
        print(f"-> no --preds_dir given, using {preds_dir}  ({why})")
    out_dir = Path(args.out_dir) if args.out_dir else preds_dir.with_name(
        preds_dir.name + "_pp")
    geff_dir = Path(args.geff_dir) if args.geff_dir else out_dir.with_name(
        out_dir.name.replace("preds", "geff", 1))

    cfg = dict(
        out_dir=str(out_dir), geff_dir=str(geff_dir), no_score=args.no_score,
        threshold=args.threshold,
        recover=0 if args.no_recover else 1,
        recover_max_gap=args.recover_max_gap,
        recover_max_residual=args.recover_max_residual,
        recover_min_dup=args.recover_min_dup,
        through=args.through, through_max_gap=args.through_max_gap,
        min_track=args.min_track,
        linefit=args.linefit, linefit_window=args.linefit_window,
        vel_weight=args.vel_weight, vel_momentum=args.vel_momentum,
    )
    stages = [name for name, on in (
        (f"assign(th={args.threshold:g}"
         + (f", momentum w={args.vel_weight:g} m={args.vel_momentum:g}"
            if args.vel_weight else "") + ")", True),
        (f"recover(gap<={args.recover_max_gap}, r={args.recover_max_residual:g})",
         not args.no_recover),
        (f"through(gap<={args.through_max_gap}, {args.through:g} um)", args.through),
        (f"prune(min_track={args.min_track})", args.min_track),
        (f"linefit(w={args.linefit:g}, +-{args.linefit_window})", args.linefit),
    ) if on]

    paths = sorted(preds_dir.glob("*.npz"))[: args.limit]
    if not paths:
        raise SystemExit(f"No npz files under {preds_dir}")

    print(f"-> reading   : {preds_dir}  ({len(paths)} stacks)")
    print(f"-> npz out   : {out_dir}")
    print(f"-> geff out  : {geff_dir}")
    print(f"-> stages    : {' -> '.join(stages)}")
    print(f"-> workers   : {args.jobs}\n")
    out_dir.mkdir(parents=True, exist_ok=True)
    geff_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with Pool(args.jobs) as pool:
        for i, (sample, row) in enumerate(
            pool.imap_unordered(process, [(str(p), cfg) for p in paths], chunksize=1)
        ):
            score = (f"node_recall={row['node_recall']:.3f}  "
                     f"adj_edge_jaccard={row['adj_edge_jaccard']:.3f}") if row else ""
            print(f"[{i + 1:>3d}/{len(paths)}] {sample:<16s} - {score}", flush=True)
            if row:
                rows.append(row)

    if not rows:
        print(f"\n-> wrote {len(paths)} stacks to {out_dir} and {geff_dir} (not scored)")
        return

    summary = summarise(rows)
    group_summaries, macro = macro_summarise(rows)
    by_embryo = {}
    for row in rows:
        by_embryo.setdefault(row.get("embryo", "?"), []).append(row)
    report_keys = ("score", "edge_jaccard", "adj_edge_jaccard", "node_recall",
                   "edge_tp", "edge_fp", "edge_fn", "n_nodes_pred", "n_edges_pred")

    scores = {
        "preds_dir": str(preds_dir),
        "stages": stages,
        "settings": {k: v for k, v in cfg.items()
                     if k not in ("out_dir", "geff_dir", "no_score")},
        "n_samples": len(rows),
        "cv_score": _jsonable(macro["score"]),
        "macro": _jsonable(macro),
        "overall": _jsonable(summary),
        "per_embryo": {
            emb: {"n_samples": len(by_embryo[emb]), **_jsonable(s)}
            for emb, s in group_summaries.items()
        },
        "per_sample": {
            r["sample_name"]: _jsonable({k: r[k] for k in report_keys if k in r})
            for r in sorted(rows, key=lambda r: r["sample_name"])
        },
        "worst_10": [
            {"sample_name": r["sample_name"], "score": _jsonable(r["score"])}
            for r in sorted(rows, key=lambda r: r["score"])[:10]
        ],
    }
    scores_file = out_dir.parent / f"scores_{out_dir.name}.json"
    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=4)

    print(f"\n{'=' * 60}")
    print(f"{len(rows)} stacks  ->  {out_dir}")
    print(f"{'=' * 60}")
    for k in ("edge_tp", "edge_fp", "edge_fn"):
        print(f"- {k}: {summary[k]:.0f}")
    for k in ("node_recall", "edge_jaccard", "adj_edge_jaccard", "score"):
        print(f"- {k}: {summary[k]:.4f}")
    if macro["n_groups"] > 1:
        print(f"- cv_score: {macro['score']:.4f}   (mean over "
              f"{macro['n_groups']} embryos, not pooled)")
    print(f"\n-> scores  : {scores_file}")


if __name__ == "__main__":
    main()
