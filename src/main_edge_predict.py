"""
Write the edge transformer's tracks in the npz format the rest of the repo reads.

    python main_edge_predict.py --log_folder ../logs/.../

Multi-GPU (one process per GPU, videos sharded across ranks):
    torchrun --standalone --nproc_per_node=8 main_edge_predict.py \
        --log_folder ../logs/.../

Several link thresholds can be swept in one pass. Scoring the candidates is the
GPU cost and does not depend on the threshold -- only the Hungarian linking
below it does, and that is cheap CPU -- so the extra values come nearly free:

    python main_edge_predict.py --threshold 0.3,0.4,0.5

Each value writes its own ``preds_<threshold>/`` and ``scores_preds_<threshold>.json``,
and the run ends with a table ranking them.

``--gt`` answers a different question: how would this model link if the detector
had missed nothing? It appends the annotated nodes no detection reached (from
the ``gt_coords`` / ``<key>_gt`` arrays ``inference_contrastive.py --gt``
writes), so node recall is 1 and the edge Jaccard is the linking alone, with the
detector's misses taken out of it. The scores are an upper bound on a node set
no submission can have, so they go to ``preds_gt_<threshold>/`` and carry a
``gt_augmented`` flag -- they never overwrite or stand in for the real ones.

Mirrors ``main_predict.py``: every fold in the run's ``selected_folds`` is
predicted in turn, each against its own held-out embryo, using the fold's
``fold_<n>/ckpt_final.pt``. Since the folds' val embryos are disjoint, one pass
covers every stack once.

One ``<sample>.npz`` per video, holding ``coords`` (N, 4) as t/z/y/x and
``edges`` (M, 4) as ``source row, target row, probability, distance um`` --
exactly what ``main_predict.py`` writes and what ``main_postprocess.py``,
``tracking.post_process.build_graph`` and the submission code consume. Writing
that format rather than a bespoke one is the whole point: it lets the tuned
post-processing chain run over this model's graph unchanged,

    python main_postprocess.py --log_folder <run> --preds_dir <run>/preds_0.40 \
                               --out_dir <run>/preds_edge_pp --geff_dir <run>/geff_edge_pp

and the two stages there treat a learned graph differently, which is worth
knowing before reading the numbers. ``--no_pipeline`` keeps this model's links
as the base and lets ``pp_public`` repair them, using its probabilities as a
re-linking bonus. The default instead runs ``postprocessing.relink`` first,
which re-decides every link from flow-corrected geometry alone (its
``prob_weight`` is 0), so it keeps this model's *nodes* and discards its edges.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from edge_transformer.data import load_videos
from edge_transformer.infer import dataset_for, load_hoct
from edge_transformer.predict import link_video, score_video
from params import EMBRYOS, LOG_FOLDER, SCALE, TRAIN_DIR
from training.distributed import cleanup_distributed, init_distributed
from utils.logger import Config

warnings.filterwarnings("ignore", category=UserWarning)


def _thresholds(value: str) -> list[float]:
    """Parse "0.3,0.4,0.5" into a sorted list of link thresholds."""
    parsed = sorted({float(v) for v in str(value).split(",") if v.strip()})
    if not parsed:
        raise argparse.ArgumentTypeError(f"no thresholds in {value!r}")
    return parsed


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--log_folder", default=None,
                   help="run folder holding args.json and fold_<n>/ckpt_final.pt; "
                        f"default = the latest run under {LOG_FOLDER}edge_run/")
    p.add_argument("--fold", type=int, default=None,
                   help="a single fold; default = every fold in the run's "
                        "selected_folds, whose val embryos are disjoint, so "
                        "one pass covers them all")
    p.add_argument("--ckpt", default=None,
                   help="explicit checkpoint; default = fold_<n>/ckpt_final.pt")
    p.add_argument("--nodes_folder", default=None,
                   help="the precomputed nodes to predict over; default = the "
                        "run's own nodes_folder from args.json, i.e. the ones "
                        "it was trained against. Another folder is a different "
                        "detector, and the numbers then mix its node recall "
                        "with this model's linking")
    p.add_argument("--embeddings_key", default=None,
                   help="npz key holding the appearance vectors; default = the "
                        "checkpoint's own, as recorded by main_edge.py")
    p.add_argument("--data_dir", default=TRAIN_DIR)
    p.add_argument("--out_dir", default=None,
                   help="default <log_folder>/preds_<threshold>, or "
                        "preds_gt_<threshold> under --gt")
    p.add_argument("--embryos", default=None,
                   help="comma-separated override; default = the fold's val embryo")
    p.add_argument("--threshold", type=_thresholds, default=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
                   help="probability a link must reach to be committed, or a "
                        "comma-separated list to sweep several in one pass. "
                        "Scoring the video is the GPU cost and does not depend "
                        "on the threshold, so the extra values are nearly free")
    p.add_argument("--max_nodes_per_frame", type=int, default=None,
                   help="per-frame node cap; default = none. The training cap is "
                        "a memory bound, and a node it drops here gets no "
                        "candidate edges at all, so it cannot be linked")
    p.add_argument("--gt", action="store_true",
                   help="append the annotated nodes no detection reached, so "
                        "node recall is 1 and the scores isolate this model's "
                        "linking. Needs inference_contrastive.py --gt to have "
                        "run over the nodes folder. Diagnostic only -- the "
                        "scores are an upper bound, and they are written to "
                        "preds_gt_<threshold>/ so they cannot be mistaken for, "
                        "or overwrite, the honest ones")
    p.add_argument("--limit", type=int, default=None, help="first N videos only")
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--no_score", action="store_true",
                   help="skip the metrics, e.g. when there is no ground truth")
    return p.parse_args()


def _latest_run(base_dir: Path) -> Path:
    """The most recently written run under ``base_dir/<date>/<exp_id>/``."""
    candidates = sorted(base_dir.glob("*/*/args.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No run with an args.json found under {base_dir}")
    return candidates[-1].parent


def _print_sweep(summaries: dict) -> None:
    """Rank the swept thresholds, best first, so the choice is one glance.

    On ``score``, the competition metric, and not on ``edge_jaccard``: the
    latter ignores both the node-count adjustment and the division term, and
    the adjustment is exactly what a threshold sweep moves -- a lower threshold
    commits more links and detects more nodes, and the raw Jaccard sees only
    the first half of that trade. Ranking on it picks a threshold the metric
    does not agree with. ``edge_jaccard`` stays as the fallback for a split
    where ``score`` could not be formed, which is what ``_folder_score`` in
    main_edge_postprocess.py does with the same numbers.
    """
    columns = ("edge_jaccard", "adj_edge_jaccard", "node_recall", "score")

    def ranked_by(th):
        """``(metric name, value)`` this threshold is judged on."""
        # NaN compares False against everything, so a failed threshold would
        # otherwise win by being first. Sink it instead.
        for key in ("score", "edge_jaccard"):
            value = summaries[th].get(key, float("nan"))
            if value == value:
                return key, float(value)
        return None, float("-inf")

    def rank(th):
        return ranked_by(th)[1]

    best = max(summaries, key=rank)
    metric, best_value = ranked_by(best)
    ranked = metric is not None

    print(f"\n{'=' * 78}")
    print("THRESHOLD SWEEP")
    print(f"{'=' * 78}")
    print("  thr  " + "".join(f"{c:>18s}" for c in columns))
    for th in sorted(summaries):
        cells = "".join(f"{summaries[th].get(c, float('nan')):>18.4f}" for c in columns)
        print(f"{th:>5g}  {cells}{'   <- best' if ranked and th == best else ''}")
    if ranked:
        print(f"\n-> best {metric} {best_value:.4f} at threshold {best:g}")
    else:
        print("\n-> every threshold scored NaN; nothing to rank")


def _gather(rows: list, world_size: int) -> list:
    """all_gather_object a rank's rows and flatten; identity when not distributed."""
    if world_size == 1 or not (dist.is_available() and dist.is_initialized()):
        return rows
    out = [None] * world_size
    dist.all_gather_object(out, rows)
    return [r for part in out for r in part]


def main():
    args = parse_args()
    ctx = init_distributed(None)
    device = ctx.device

    if args.log_folder:
        log_folder = args.log_folder.rstrip("/") + "/"
    else:
        log_folder = str(_latest_run(Path(LOG_FOLDER) / "edge_run")) + "/"
        if ctx.is_main:
            print(f"-> no --log_folder given, using latest run {log_folder}")

    thresholds = args.threshold
    sweeping = len(thresholds) > 1
    if args.out_dir and sweeping:
        raise ValueError(
            f"--out_dir names one folder but {len(thresholds)} thresholds are "
            "swept; each needs its own or they would overwrite each other"
        )
    # Each threshold keeps its own folder, which is what the plain default
    # already encodes -- so a sweep is just several of the usual runs.
    #
    # --gt names its folders apart rather than sharing them. Its scores are an
    # upper bound taken on a node set no submission can have, and the whole
    # point of running it is to compare against the honest numbers, so
    # overwriting them with it would destroy the comparison it exists to make.
    stem = "preds_gt" if args.gt else "preds"
    out_dirs = {
        th: Path(args.out_dir or f"{log_folder}{stem}_{th:.2f}") for th in thresholds
    }
    # An explicit --out_dir escapes the naming above, so check it cannot land on
    # a folder of honest predictions. Refusing a non-empty one is blunt but it
    # is the only rule that does not depend on how the folder was named.
    if args.gt and args.out_dir:
        clashes = [d for d in out_dirs.values() if any(d.glob("*.npz"))]
        if clashes:
            raise ValueError(
                f"--gt would write into {clashes[0]}, which already holds "
                "predictions. Its scores are an upper bound and must not "
                "replace a real run's; drop --out_dir to use the "
                "preds_gt_<threshold>/ folders, or name an empty one."
            )

    if ctx.is_main:
        for d in out_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
    if ctx.distributed:
        dist.barrier()

    run_cfg = Config(json.load(open(log_folder + "args.json")))

    # The nodes come from the run's own args.json unless overridden. A standing
    # default here would be a different detector's folder, and since the npz
    # format is the same and the embeddings key is often present in both, the
    # mismatch does not raise -- it just silently reports that detector's node
    # recall, on a node distribution the checkpoint never saw.
    nodes_folder = args.nodes_folder or getattr(run_cfg, "nodes_folder", None)
    if not nodes_folder:
        raise ValueError(
            f"{log_folder}args.json records no nodes_folder, so there is no "
            "default to inherit; pass --nodes_folder"
        )

    # Which folds to run. Each fold holds out a different embryo, so their val
    # splits are disjoint and running them all writes one out-of-fold
    # prediction per stack into the same folder -- no collisions, full coverage.
    if args.fold is not None:
        folds = [args.fold]
    else:
        folds = [int(f) for f in (getattr(run_cfg, "selected_folds", None) or [0])]

    # Both of these name one fold's worth of work, so they cannot stand in for
    # several: every fold would predict the same stacks and overwrite the
    # previous fold's npz files.
    if len(folds) > 1:
        if args.embryos:
            raise ValueError(
                f"--embryos overrides the split for a single fold, but {folds} "
                "are selected; pass --fold to pick one"
            )
        if args.ckpt:
            raise ValueError(
                f"--ckpt is one checkpoint, but folds {folds} are selected; "
                "pass --fold to pick one"
            )

    # Check every fold's checkpoint before running any of them, so a fold whose
    # training never happened is caught before another fold's npz files land.
    if not args.ckpt:
        missing = [
            f for f in folds
            if not os.path.exists(os.path.join(log_folder, f"fold_{f}", "ckpt_final.pt"))
        ]
        if missing:
            raise FileNotFoundError(
                f"No ckpt_final.pt for fold(s) {missing} under {log_folder}. "
                f"selected_folds is {folds}; pass --fold to run only a trained one."
            )

    if ctx.is_main:
        print(f"-> log_folder     : {log_folder}")
        print(f"-> nodes_folder   : {nodes_folder}"
              + ("" if args.nodes_folder else "   (from the run's args.json)"))
        print(f"-> folds          : {folds}")
        print(f"-> writing to     : "
              + ", ".join(str(d) for d in out_dirs.values()))
        if ctx.distributed:
            print(f"-> distributed    : {ctx.world_size} processes")
        if args.gt:
            print("\n!! --gt: the unmatched ground-truth nodes are appended to "
                  "the detections.\n"
                  "!! Node recall is 1 by construction and every score below is "
                  "an upper bound\n"
                  "!! on a node set no submission can have. Diagnostic only.")

    rows = {th: [] for th in thresholds}
    n_gt_added = 0
    n_videos = 0
    max_nodes_per_frame = None
    for fold in folds:
        embryos = (
            [e.strip() for e in args.embryos.split(",") if e.strip()]
            if args.embryos else [EMBRYOS[fold]]
        )
        ckpt_path = args.ckpt or os.path.join(log_folder, f"fold_{fold}", "ckpt_final.pt")
        if ctx.is_main:
            print(f"\n-> fold {fold}: {ckpt_path}")
            print(f"   val embryos : {embryos}")

        model, cfg = load_hoct(ckpt_path, device=device, verbose=ctx.is_main)
        videos = load_videos(
            nodes_folder, args.data_dir, embryos=embryos,
            n_videos=args.limit, use_embeddings=bool(cfg.get("use_embeddings", True)),
            # The run's own embeddings key: another run's vectors are a different
            # feature space and the checkpoint would be reading noise.
            embeddings_key=(
                args.embeddings_key or cfg.get("embeddings_key") or "embeddings"
            ),
            # Every video is predicted, including any the annotation cannot reach.
            require_links=False, verbose=ctx.is_main, with_gt=args.gt,
        )
        n_gt_added += sum(v.n_gt_added for v in videos)
        dataset = dataset_for(cfg, videos, max_nodes_per_frame=args.max_nodes_per_frame)
        max_nodes_per_frame = dataset.max_nodes_per_frame
        if ctx.is_main:
            cap = f"{max_nodes_per_frame}/frame" if max_nodes_per_frame else "none"
            print(f"- Nodes {nodes_folder}  cap {cap}, "
                  f"window {dataset.window_size}, threshold(s) "
                  f"{', '.join(f'{t:g}' for t in thresholds)}\n")

        # Shard videos across ranks; each writes its own npz files.
        mine = list(range(len(videos)))[ctx.rank :: ctx.world_size]
        for i, vi in enumerate(mine):
            video = videos[vi]
            # One pass over the video answers every threshold: the model scores
            # the candidates, and only the linking below reads the cutoff. The
            # candidate array is likewise threshold-free, so it is built once.
            scored = score_video(model, dataset, vi, device, fp16=not args.no_fp16)
            candidates = _candidate_array(video, scored)

            reported = []
            for th in thresholds:
                coords, edges = link_video(video, scored, threshold=th)
                np.savez_compressed(
                    out_dirs[th] / f"{video.name}.npz",
                    coords=np.round(np.asarray(coords, np.float64)).astype(np.int16),
                    edges=np.asarray(edges, np.float64).reshape(-1, 4),
                    candidates=candidates,
                )
                part = f"[thr {th:g}] {len(edges):05d} edges"
                if not args.no_score:
                    row = _score(video, coords, edges, Path(args.data_dir))
                    if row is not None:
                        rows[th].append(row)
                        part += (f" edge_jac={row['edge_jaccard']:.4f}"
                                 f" adj={row['adj_edge_jaccard']:.4f}")
                reported.append(part)

            print(f"[rank {ctx.rank}]  [{i + 1:>3}/{len(mine)}] {video.name}  "
                  + "   ".join(reported), flush=True)

        n_videos += len(videos)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # One gather for the whole sweep rather than one per threshold.
    gathered = _gather([rows], ctx.world_size)
    all_rows = {th: [r for part in gathered for r in part[th]] for th in thresholds}

    if not ctx.is_main:
        return

    summaries = {}
    for th in thresholds:
        th_rows = all_rows[th]
        out_dir = out_dirs[th]
        print(f"\n-> Wrote {n_videos} npz to {out_dir}")
        if not th_rows:
            continue

        summary = _summarise(th_rows)
        summaries[th] = summary
        scores_file = out_dir.parent / f"scores_{out_dir.name}.json"
        with open(scores_file, "w") as f:
            json.dump({"preds_dir": str(out_dir), "threshold": th,
                       "max_nodes_per_frame": max_nodes_per_frame or None,
                       "nodes_folder": str(nodes_folder),
                       # Travels with the numbers, so a file read months later
                       # still says these are an upper bound.
                       "gt_augmented": bool(args.gt),
                       "n_gt_nodes_appended": int(n_gt_added) if args.gt else 0,
                       "overall": summary,
                       "per_sample": {r["sample_name"]: {
                           k: v for k, v in r.items() if k != "sample_name"
                       } for r in th_rows}},
                      f, indent=4, default=float)

        print(f"{'=' * 60}")
        print(f"threshold {th:g}  -  {len(th_rows)} stacks  ->  {out_dir}")
        if args.gt:
            print(f"UPPER BOUND: {n_gt_added} unmatched GT nodes appended")
        print(f"{'=' * 60}")
        for k in ("edge_tp", "edge_fp", "edge_fn"):
            print(f"- {k}: {summary[k]:.0f}")
        for k in ("node_recall", "edge_jaccard", "adj_edge_jaccard", "score"):
            print(f"- {k}: {summary[k]:.4f}")
        print(f"-> scores  : {scores_file}")

    if sweeping and summaries:
        _print_sweep(summaries)


def _candidate_array(video, scored):
    """Every scored candidate as (src row, tgt row, prob, distance um, dt).

    ``link_video`` throws all of this away except the assignment it commits to,
    which makes the threshold and the linking rule a GPU decision. Saved beside
    the committed edges they become a post-processing decision instead: the same
    graph can be re-linked at any threshold, and the gap-closing candidates
    (``dt > 1``), which ``link_video`` never commits, are there for a stage that
    wants them.
    """
    frame_of = np.asarray(video.coords)[:, 0].astype(int)
    if not scored:
        return np.empty((0, 5))
    rows = [
        (a, b, p, d, frame_of[b] - frame_of[a])
        for (a, b), (p, d) in scored.items()
    ]
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


def _score(video, coords, edges, data_dir: Path):
    """One video's competition metrics, or None when it has no ground truth."""
    import contextlib
    import io

    import pandas as pd
    import tracksdata as td

    from tracking.post_process import build_graph
    from tracking_cellmot.metrics import (
        evaluate as evaluate_graph, node_recall, per_sample_metrics,
    )

    gt_path = data_dir / f"{video.name}.geff"
    if not gt_path.exists() or len(edges) == 0:
        return None
    result = td.graph.IndexedRXGraph.from_geff(gt_path)
    gt_graph = result[0] if isinstance(result, tuple) else result

    meta_path = data_dir.parent / "metadata.csv"
    n_total = float("nan")
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        hit = meta.loc[meta["sample_name"] == video.name, "estimated_number_of_nodes"]
        if len(hit):
            n_total = float(hit.values[0])

    with contextlib.redirect_stdout(io.StringIO()):
        graph = build_graph(coords, edges)
        res = evaluate_graph(graph, gt_graph, scale=SCALE)
        rec = node_recall(graph, gt_graph)
    row = per_sample_metrics(res, n_total, rec)
    row["sample_name"] = video.name
    row["n_nodes_pred"] = int(len(coords))
    row["n_edges_pred"] = int(len(edges))
    return row


def _summarise(rows):
    from tracking_cellmot.metrics import summarise

    out = dict(summarise(rows))
    for key in ("edge_tp", "edge_fp", "edge_fn"):
        out[key] = float(sum(r[key] for r in rows))
    return out


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
