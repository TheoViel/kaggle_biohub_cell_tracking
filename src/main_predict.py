"""
Run tracking inference over every validation stack of a fold, in parallel.

Mirrors notebooks/TrackingInf.ipynb up to (not including) the "ILP PP"
section: predict_video -> build_graph -> evaluate, per sample.

Predictions land in <log_folder>/preds/<sample>.npz (coords + edges), with
per-sample metrics in <log_folder>/preds/metrics.csv and a run-level summary
in summary.json.

Single GPU:
    python main_predict.py --log_folder ../logs/2026-08-14/17/

Multi-GPU (one process per GPU, videos sharded across ranks):
    torchrun --standalone --nproc_per_node=8 main_predict.py \
        --log_folder ../logs/2026-08-14/17/
"""

import argparse
import contextlib
import glob
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tracksdata as td

from params import EMBRYOS, SCALE, TRAIN_DIR
from tracking.main import _paths_for_embryos, build_model
from tracking.post_process import build_graph
from tracking.predict import PredictConfig, predict_video
from tracking_cellmot.metrics import (
    evaluate,
    node_recall,
    per_sample_metrics,
    summarise,
)
from training.distributed import cleanup_distributed, init_distributed
from utils.logger import Config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log_folder", required=True,
                   help="run folder holding args.json and fold_<n>/ckpt_*.pt")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--ckpt", default=None,
                   help="explicit checkpoint; default = last ckpt_*.pt of the fold")
    p.add_argument("--out_dir", default=None, help="default <log_folder>/preds")
    p.add_argument("--embryos", default=None,
                   help="comma-separated override; default = the fold's val embryo")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--n_videos", type=int, default=None,
                   help="cap the number of stacks (debugging)")

    p.add_argument("--pl", action="store_true",
                   help="pseudo-labelling: predict on the fold's TRAIN embryos "
                        "instead of its val one, writing to <log_folder>/preds_pl")

    p.add_argument("--det_tta", action=argparse.BooleanOptionalAction,
                   default=PredictConfig.det_tta)
    p.add_argument("--det_threshold", type=float, default=PredictConfig.det_threshold)
    p.add_argument("--pool_kernel_um", type=float, default=PredictConfig.pool_kernel_um)
    p.add_argument("--nms", choices=("maxpool", "greedy"), default=PredictConfig.nms)
    p.add_argument("--nms_radius_um", type=float, default=PredictConfig.nms_radius_um,
                   help="greedy NMS radius; default pool_kernel_um / 2")
    p.add_argument("--edge_activation", default=PredictConfig.edge_activation)
    p.add_argument("--edge_selection", default=PredictConfig.edge_selection)
    p.add_argument("--edge_threshold", type=float, default=PredictConfig.edge_threshold)
    p.add_argument("--max_parents_per_node", type=int,
                   default=PredictConfig.max_parents_per_node)
    p.add_argument("--max_children_per_node", type=int,
                   default=PredictConfig.max_children_per_node)
    return p.parse_args()


def _predict_config(args) -> PredictConfig:
    cfg = PredictConfig()
    for k in (
        "det_tta", "det_threshold", "pool_kernel_um", "nms", "nms_radius_um",
        "edge_activation",
        "edge_selection", "edge_threshold", "max_parents_per_node",
        "max_children_per_node",
    ):
        setattr(cfg, k, getattr(args, k))
    return cfg


def _jsonable(obj):
    """Coerce numpy scalars to plain Python so json.dump keeps them numeric."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _reconcile_arch(cfg, state, verbose=False):
    """Infer architecture switches from the weights themselves.

    args.json only records the flags that existed when the run was launched,
    so a checkpoint trained before a flag was added would be rebuilt with
    today's default and fail to load. The layer shapes are unambiguous, so
    read them instead of trusting the defaults.
    """
    hidden = getattr(cfg, "transformer_hidden", 128)

    w = state.get("transformer.pair_mlp.0.weight")
    if w is not None:
        n_extra = w.shape[1] - (2 * hidden + 3)
        geom = n_extra >= 4
        if geom != getattr(cfg, "edge_geom_feats", True):
            if verbose:
                print(f"-> edge_geom_feats={geom} (inferred from pair_mlp "
                      f"in_features={w.shape[1]})")
            cfg.edge_geom_feats = geom

    # Whether the contrastive branch exists is unambiguous in the weights, and
    # it decides how wide the node features are — so reconcile it before using
    # that width to infer pos_feat_dim.
    has_embed = any(k.startswith("embed_unet.") for k in state)
    features = list(getattr(cfg, "unet_features", ["detection"]) or [])
    if has_embed and "contrastive" not in features:
        features.append("contrastive")
    elif not has_embed and "contrastive" in features:
        features.remove("contrastive")
    if features != list(getattr(cfg, "unet_features", ["detection"]) or []):
        if verbose:
            print(f"-> unet_features={features} (embed_unet weights "
                  f"{'present' if has_embed else 'absent'})")
        cfg.unet_features = features

    contrastive = getattr(cfg, "contrastive", None) or {}
    widths = {
        "detection": getattr(cfg, "unet_out_channels", 32),
        "contrastive": contrastive.get("embed_dim", 32),
    }
    node_feat_dim = sum(widths[f] for f in features)

    w = state.get("transformer.proj.weight")
    if w is not None:
        pos_dim = w.shape[1] - node_feat_dim
        if pos_dim < 0:
            raise ValueError(
                f"transformer.proj expects {w.shape[1]} inputs but unet_features "
                f"{features} account for {node_feat_dim}; the checkpoint was "
                "trained with a different node-feature composition"
            )
        if pos_dim != getattr(cfg, "pos_feat_dim", 0):
            if verbose:
                print(f"-> pos_feat_dim={pos_dim} (inferred from proj "
                      f"in_features={w.shape[1]} minus node features {node_feat_dim})")
            cfg.pos_feat_dim = pos_dim
    return cfg


def _gather(obj, world_size):
    """all_gather a picklable object; identity when not distributed."""
    if world_size == 1 or not (dist.is_available() and dist.is_initialized()):
        return [obj]
    out = [None] * world_size
    dist.all_gather_object(out, obj)
    return out


def main():
    args = parse_args()
    ctx = init_distributed(None)
    device = ctx.device

    log_folder = args.log_folder.rstrip("/") + "/"
    out_dir = Path(args.out_dir or (log_folder + ("preds_pl" if args.pl else "preds")))
    if ctx.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if ctx.distributed:
        dist.barrier()

    cfg = Config(json.load(open(log_folder + "args.json")))
    window_size = getattr(cfg, "window_size", 2)
    downscale_factor = getattr(cfg, "downscale_factor", 1)

    # Same split logic as tracking.main.k_fold. --pl flips it: the fold's
    # training embryos are the ones the detector has never been scored on, and
    # they are what a pseudo-labelling pass wants to label.
    if args.embryos:
        embryos = [e.strip() for e in args.embryos.split(",") if e.strip()]
    elif args.pl:
        embryos = [e for e in EMBRYOS if e != EMBRYOS[args.fold]]
    else:
        embryos = [EMBRYOS[args.fold]]

    train_dir = Path(TRAIN_DIR)
    paths = sorted(_paths_for_embryos(str(train_dir), embryos))
    if args.n_videos is not None:
        paths = paths[: args.n_videos]
    if not paths:
        raise ValueError(f"No datasets found for embryos {embryos}")

    ckpt_path = args.ckpt or os.path.join(log_folder, f"fold_{args.fold}", "ckpt_final.pt")

    if ctx.is_main:
        print(f"-> log_folder     : {log_folder}")
        print(f"-> checkpoint     : {ckpt_path}")
        split = "train (pseudo-label)" if args.pl else "val"
        print(f"-> {split} embryos : {embryos}  ({len(paths)} stacks)")
        print(f"-> threshold       : {args.det_threshold}")
        print(f"-> writing to      : {out_dir}")
        if ctx.distributed:
            print(f"-> distributed    : {ctx.world_size} processes\n")

    state = torch.load(ckpt_path, map_location="cpu")
    state = state.get("model_state_dict", state)
    state = {k.replace("unet.module.", "unet.", 1): v for k, v in state.items()}
    # cfg = _reconcile_arch(cfg, state, verbose=ctx.is_main)

    model = build_model(cfg, None)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    meta_csv = train_dir.parent / "metadata.csv"
    meta = pd.read_csv(meta_csv) if meta_csv.exists() else None

    pred_cfg = _predict_config(args)

    # Shard stacks across ranks; each writes its own npz files.
    mine = paths[ctx.rank :: ctx.world_size]
    rows = []

    for i, path in enumerate(mine):
        name = path.name
        # predict_video also returns the raw detection logits; only the graph
        # is needed here, and keeping them would hold a full-resolution volume
        # per window in memory.
        coords, edges, _ = predict_video(
            model,
            path,
            device=device,
            cfg=pred_cfg,
            window_size=window_size,
            downscale_factor=downscale_factor,
            max_frames=args.max_frames,
            progress=False,
        )

        np.savez_compressed(
            out_dir / f"{name}.npz",
            coords=coords,
            edges=np.asarray(edges, dtype=np.float64).reshape(-1, 4),
        )

        gt = td.graph.IndexedRXGraph.from_geff(train_dir / f"{name}.geff")
        gt_graph = gt[0] if isinstance(gt, tuple) else gt

        n_total = float("nan")
        if meta is not None:
            hit = meta.loc[meta["sample_name"] == name, "estimated_number_of_nodes"]
            if len(hit):
                n_total = float(hit.values[0])

        # build_graph is chatty on some builds; keep the log readable.
        with contextlib.redirect_stdout(io.StringIO()):
            graph = build_graph(coords, edges)
            res = evaluate(graph, gt_graph, scale=SCALE)
            rec = node_recall(graph, gt_graph)

        row = per_sample_metrics(res, n_total, rec)
        # per_sample_metrics stops at the raw columns; 'score' and
        # 'division_jaccard' come out of summarise, so roll this one sample up
        # on its own — same thing the notebook does per stack.
        one = summarise([row])
        row["division_jaccard"] = one["division_jaccard"]
        row["score"] = one["score"]

        row["sample_name"] = name
        row["n_nodes_pred"] = int(coords.shape[0])
        row["n_edges_pred"] = int(len(edges))
        row["n_nodes_estimated"] = n_total
        rows.append(row)

        print(
            f"[rank {ctx.rank}] {i + 1}/{len(mine)}  -  {name}  -  "
            f"node_recall={rec:.3f}  edge_jaccard={row['edge_jaccard']:.3f}  score={row['score']:.4f}",
            flush=True,
        )

    all_rows = [r for part in _gather(rows, ctx.world_size) for r in part]

    if ctx.is_main:
        df = pd.DataFrame(all_rows).sort_values("sample_name")
        df.to_csv(out_dir / "metrics.csv", index=False)
        summary = summarise(all_rows)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=4, default=str)

        # Per-sample scores alongside the run-level roll-up.
        report_keys = (
            "score", "edge_jaccard", "adj_edge_jaccard", "division_jaccard",
            "node_recall", "edge_tp", "edge_fp", "edge_fn",
            "n_nodes_pred", "n_edges_pred", "n_nodes_estimated",
        )
        scores = {
            "checkpoint": ckpt_path,
            "fold": args.fold,
            "split": "train" if args.pl else "val",
            "embryos": embryos,
            "n_samples": len(all_rows),
            "overall": _jsonable(summary),
            "threshold": args.det_threshold,
            "per_sample": {
                r["sample_name"]: _jsonable({k: r[k] for k in report_keys if k in r})
                for r in sorted(all_rows, key=lambda r: r["sample_name"])
            },
        }
        scores_path = Path(log_folder) / ("scores_pl.json" if args.pl else f"scores_{args.det_threshold:.2f}.json")
        with open(scores_path, "w") as f:
            json.dump(scores, f, indent=4)
        print(f"\n-> Saving scores to {scores_path}")

        print(f"\n{'=' * 60}")
        print(f"{len(all_rows)} stacks  ->  {out_dir}")
        print(f"{'=' * 60}")
        for k in ("edge_tp", "edge_fp", "edge_fn"):
            if k in summary:
                print(f"- {k}: {summary[k]:.0f}")
        for k in ("node_recall", "edge_jaccard", "adj_edge_jaccard",
                  "division_jaccard", "score"):
            if k in summary:
                print(f"- {k}: {summary[k]:.4f}")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
