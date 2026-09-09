"""
Run tracking inference over every validation stack of a fold, in parallel.

Mirrors notebooks/TrackingInf.ipynb up to (not including) the "ILP PP"
section: predict_video -> build_graph -> evaluate, per sample.

Predictions land in <log_folder>/preds/<sample>.npz (coords + edges), with
per-sample metrics in <log_folder>/preds/metrics.csv and a run-level summary
in summary.json.

Several detection thresholds and greedy-NMS radii can be swept in one pass, as
their full cross product. The UNet forward is what this costs and it depends on
neither, so the extra values come nearly free; each grid point gets its own
<log_folder>/preds_<...>/ folder and the run ends with a table ranking them.

Single GPU, on the most recent run under ../logs/:
    python main_predict.py

An explicit run:
    python main_predict.py --log_folder ../logs/2026-08-14/17/

Threshold sweep (one forward pass, four sets of predictions):
    python main_predict.py --log_folder ../logs/2026-08-14/17/ \
        --det_threshold 0.2,0.3,0.4,0.5

Threshold x NMS radius (one forward pass, 4 x 4 = 16 sets of predictions in
preds_r4_0.20/ ... preds_r7_0.50/):
    python main_predict.py --log_folder ../logs/2026-08-14/17/ \
        --det_threshold 0.2,0.3,0.4,0.5 --nms_radius_um 4,5,6,7

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

from params import EMBRYOS, LOG_FOLDER, SCALE, TRAIN_DIR
from tracking.main import _paths_for_embryos, build_model
from tracking.post_process import build_graph
from tracking.predict import PredictConfig, predict_video_thresholds
from tracking_cellmot.metrics import (
    evaluate,
    node_recall,
    macro_summarise,
    per_sample_metrics,
    summarise,
)
from training.distributed import cleanup_distributed, init_distributed
from utils.logger import Config, get_last_log_folder


def _floats(value: str) -> list[float]:
    """Parse "0.2,0.3,0.4" into a sorted list of sweep values."""
    parsed = sorted({float(v) for v in str(value).split(",") if v.strip()})
    if not parsed:
        raise argparse.ArgumentTypeError(f"no values in {value!r}")
    return parsed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log_folder", default=None,
                   help="run folder holding args.json and fold_<n>/ckpt_*.pt; "
                        f"default = the most recent run under {LOG_FOLDER}")
    p.add_argument("--fold", type=int, default=None,
                   help="a single fold; default = every fold in the run's "
                        "selected_folds, whose val embryos are disjoint, so "
                        "one pass covers them all")
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
    p.add_argument("--det_threshold", type=_floats, default=[PredictConfig.det_threshold],
                   help="detection threshold, or a comma-separated list to sweep several "
                        "in one forward pass")
    p.add_argument("--pool_kernel_um", type=float, default=PredictConfig.pool_kernel_um)
    p.add_argument("--nms", choices=("maxpool", "greedy"), default=PredictConfig.nms)
    p.add_argument("--nms_radius_um", type=_floats, default=[PredictConfig.nms_radius_um],
                   help="greedy NMS radius, or a comma-separated list to sweep several "
                        "in the same forward pass; crossed with --det_threshold")
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
        "det_tta", "pool_kernel_um", "nms",
        "edge_activation",
        "edge_selection", "edge_threshold", "max_parents_per_node",
        "max_children_per_node",
    ):
        setattr(cfg, k, getattr(args, k))
    # Only meaningful on a model built with the head; a model without one
    # produces no vote and the flag is a no-op.
    cfg.use_offset = bool(getattr(args, "use_offset", True))
    # predict_video_thresholds takes both sweeps as its own arguments and
    # ignores these, but a single-valued cfg should still read back what it ran
    # at.
    cfg.det_threshold = args.det_threshold[0]
    cfg.nms_radius_um = args.nms_radius_um[0]
    return cfg


def _out_dir_for(
    base: Path, radius: float, threshold: float, sweep_th: bool, sweep_r: bool
) -> Path:
    """Where one grid point's predictions go.

    A lone setting keeps writing to the plain preds/ folder, which the
    pseudo-labelling and post-processing paths already point at; a sweep gives
    each value its own suffixed folder instead. The radius is only named when
    it is swept, so a threshold-only sweep keeps the preds_<thr>/ layout the
    downstream scripts were written against.
    """
    suffix = _key_suffix(radius, threshold, sweep_th, sweep_r)
    return base.with_name(base.name + suffix) if suffix else base


def _key_suffix(
    radius: float, threshold: float, sweep_th: bool, sweep_r: bool
) -> str:
    """The ``_r6_0.40``-style tag naming one point of the sweep grid."""
    return (f"_r{radius:g}" if sweep_r else "") + (
        f"_{threshold:.2f}" if sweep_th else ""
    )


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

    # The offset head is a later addition, so a checkpoint predating it has no
    # such weights and must be rebuilt without one. Its radius is not in the
    # weights — it only bounds the output — so args.json stays the source for
    # that, defaulting to what the head was introduced with.
    has_offset = any(k.startswith("offset_head.") for k in state)
    if has_offset != bool(getattr(cfg, "offset_head", False)):
        if verbose:
            print(f"-> offset_head={has_offset} (offset_head weights "
                  f"{'present' if has_offset else 'absent'})")
        cfg.offset_head = has_offset

    # Likewise for the extra decoder branch feeding the fine detection head. The
    # count is the number of blocks that were built, which the weights name
    # directly (unet.fine_decoder_blocks.<i>...), so it is read rather than
    # trusted from args.json.
    n_extra = len({
        k.split(".")[2] for k in state if k.startswith("unet.fine_decoder_blocks.")
    })
    if n_extra != int(getattr(cfg, "unet_n_extra_decode_steps", 0) or 0):
        if verbose:
            print(f"-> unet_n_extra_decode_steps={n_extra} (inferred from "
                  f"unet.fine_decoder_blocks weights)")
        cfg.unet_n_extra_decode_steps = n_extra
    return cfg


def _print_sweep(summaries: dict, per_embryo: dict | None = None):
    """Rank the swept grid points, best score first, so the choice is one glance.

    ``summaries`` and ``per_embryo`` are keyed by ``(nms_radius_um, threshold)``.
    ``per_embryo`` then maps to embryo -> summary. When several embryos were
    predicted they get their own table each, because their best grid point need
    not be the pooled one — the embryos differ in cell density, so the detection
    cutoff and the suppression radius that balance recall against false
    positives differ too.
    """
    base_columns = ("score", "edge_jaccard", "adj_edge_jaccard",
                    "division_jaccard", "node_recall")
    # A column for a dimension that was not swept would repeat one value down
    # the whole table.
    sweep_r = len({r for r, _ in summaries}) > 1
    sweep_th = len({th for _, th in summaries}) > 1
    labels = (("radius",) if sweep_r else ()) + (("thr",) if sweep_th else ())

    def _label(key):
        r, th = key
        return "".join(
            f"{v:>7g}" for v in
            ((r,) if sweep_r else ()) + ((th,) if sweep_th else ())
        )

    def _table(title, table: dict, rank_key="score"):
        # cv_score only exists on the pooled table; a per-embryo one would
        # print a column of NaN.
        has_cv = any("cv_score" in row for row in table.values())
        columns = (("score", "cv_score") if has_cv else ("score",)) + base_columns[1:]

        def rank(key):
            # NaN compares False against everything, so a failed grid point
            # would otherwise win by being first. Sink it instead.
            score = table[key].get(rank_key, float("nan"))
            return float("-inf") if score != score else score

        best = max(table, key=rank)
        ranked = rank(best) != float("-inf")

        print(f"\n{'=' * 78}")
        print(title)
        print(f"{'=' * 78}")
        print("".join(f"{lbl:>7s}" for lbl in labels) + "  "
              + "".join(f"{c:>18s}" for c in columns))
        for key in sorted(table):
            cells = "".join(f"{table[key].get(c, float('nan')):>18.4f}" for c in columns)
            print(f"{_label(key)}  {cells}"
                  f"{'   <- best' if ranked and key == best else ''}")

        if ranked:
            label = "cv_score" if rank_key == "cv_score" else "score"
            r, th = best
            at = f"threshold {th:g}" + (f", nms_radius_um {r:g}" if sweep_r else "")
            print(f"\n-> best {label} {rank(best):.4f} at {at}")
        else:
            print("\n-> every grid point scored NaN; nothing to rank")

    title = "SWEEP" if sweep_r and sweep_th else (
        "NMS RADIUS SWEEP" if sweep_r else "THRESHOLD SWEEP"
    )
    embryos = sorted({e for pe in (per_embryo or {}).values() for e in pe})
    # Rank by cv_score once there is more than one embryo: the pooled score
    # would pick whichever grid point suits the embryo with more stacks.
    multi = len(embryos) > 1
    _table(
        title + ("  (all embryos)" if multi else ""),
        summaries,
        rank_key="cv_score" if multi else "score",
    )

    if len(embryos) > 1:
        for emb in embryos:
            table = {
                key: pe[emb] for key, pe in per_embryo.items() if emb in pe
            }
            if table:
                folds = sorted({f for s in table.values() for f in s.get("fold", [])})
                suffix = f"  (fold {','.join(str(f) for f in folds)})" if folds else ""
                _table(f"{title}  -  {emb}{suffix}", table)


def _gather(obj, world_size):
    """all_gather a picklable object; identity when not distributed."""
    if world_size == 1 or not (dist.is_available() and dist.is_initialized()):
        return [obj]
    out = [None] * world_size
    dist.all_gather_object(out, obj)
    return out


def _predict_fold(
    model, paths, keys, thresholds, radii, out_dirs, rows, meta, pred_cfg,
    window_size, downscale_factor, train_dir, args, ctx, device, fold,
):
    """Infer one fold's stacks, writing npz files and appending metric rows.

    ``rows`` is accumulated across folds by the caller: each fold holds out a
    different embryo, so the rows stay disjoint and summarise to a single
    out-of-fold score over every stack.
    """
    # Shard stacks across ranks; each writes its own npz files.
    mine = paths[ctx.rank :: ctx.world_size]

    for i, path in enumerate(mine):
        name = path.name
        # One UNet pass over the video answers the whole threshold x radius
        # grid. The raw detection logits are not needed here, and keeping them
        # would hold a full-resolution volume per window in memory.
        per_key, _ = predict_video_thresholds(
            model,
            path,
            thresholds,
            device=device,
            cfg=pred_cfg,
            window_size=window_size,
            downscale_factor=downscale_factor,
            max_frames=args.max_frames,
            progress=False,
            return_det_logits=False,
            nms_radii=radii,
        )

        gt = td.graph.IndexedRXGraph.from_geff(train_dir / f"{name}.geff")
        gt_graph = gt[0] if isinstance(gt, tuple) else gt

        n_total = float("nan")
        if meta is not None:
            hit = meta.loc[meta["sample_name"] == name, "estimated_number_of_nodes"]
            if len(hit):
                n_total = float(hit.values[0])

        reported = []
        for key in keys:
            radius, th = key
            coords, edges, node_probs = per_key[key]

            np.savez_compressed(
                out_dirs[key] / f"{name}.npz",
                coords=coords,
                edges=np.asarray(edges, dtype=np.float64).reshape(-1, 4),
                # Row-aligned with coords: the detection head's probability at
                # each peak. The threshold above only says a node cleared it,
                # and by how much is exactly what a consumer wanting to
                # discount a shaky detection needs.
                node_probs=np.asarray(node_probs, dtype=np.float32).reshape(-1),
            )

            # build_graph is chatty on some builds; keep the log readable.
            with contextlib.redirect_stdout(io.StringIO()):
                graph = build_graph(coords, edges)
                res = evaluate(graph, gt_graph, scale=SCALE)
                rec = node_recall(graph, gt_graph)

            row = per_sample_metrics(res, n_total, rec)
            # per_sample_metrics stops at the raw columns; 'score' and
            # 'division_jaccard' come out of summarise, so roll this one sample
            # up on its own — same thing the notebook does per stack.
            one = summarise([row])
            row["division_jaccard"] = one["division_jaccard"]
            row["score"] = one["score"]

            row["sample_name"] = name
            # Each fold holds out a different embryo, so these two tags are the
            # same partition seen from either side. Both are carried so the
            # roll-up can group without re-deriving one from the other.
            row["embryo"] = name.split("_")[0]
            row["fold"] = fold
            row["n_nodes_pred"] = int(coords.shape[0])
            row["n_edges_pred"] = int(len(edges))
            row["n_nodes_estimated"] = n_total
            row["det_threshold"] = th
            row["nms_radius_um"] = radius
            rows[key].append(row)

            tag = f"thr {th:g}" + (f" r {radius:g}" if len(radii) > 1 else "")
            reported.append(
                f"[{tag}] node_recall={rec:.3f} edge_jaccard={row['edge_jaccard']:.3f} "
                f"score={row['score']:.4f}"
            )

        print(
            f"[rank {ctx.rank}] {i + 1:02d}/{len(mine):02d}  -  {name}  -  " + "  ".join(reported),
            flush=True,
        )


def main():
    args = parse_args()
    ctx = init_distributed(None)
    device = ctx.device

    # No --log_folder: the run that just finished is nearly always the one to
    # predict with, and naming it means reading today's date off the filesystem.
    log_folder = (args.log_folder or get_last_log_folder(LOG_FOLDER)).rstrip("/") + "/"
    thresholds = args.det_threshold
    radii = args.nms_radius_um
    sweep_th = len(thresholds) > 1
    sweep_r = len(radii) > 1
    sweeping = sweep_th or sweep_r
    # The grid, in the order everything downstream iterates it.
    keys = [(r, th) for r in radii for th in thresholds]

    base_dir = Path(args.out_dir or (log_folder + ("preds_pl" if args.pl else "preds")))
    out_dirs = {
        key: _out_dir_for(base_dir, *key, sweep_th, sweep_r) for key in keys
    }
    if ctx.is_main:
        for d in out_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
    if ctx.distributed:
        dist.barrier()

    cfg = Config(json.load(open(log_folder + "args.json")))
    window_size = getattr(cfg, "window_size", 2)
    downscale_factor = getattr(cfg, "downscale_factor", 1)

    # Which folds to run. Each fold holds out a different embryo, so their val
    # splits are disjoint and running them all writes one out-of-fold
    # prediction per stack into the same folder — no collisions, full coverage.
    if args.fold is not None:
        folds = [args.fold]
    else:
        folds = [int(f) for f in (getattr(cfg, "selected_folds", None) or [0])]

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

    # Check every fold's checkpoint before running any of them: a fold whose
    # training never happened is an easy mistake once selected_folds drives
    # this, and finding out on fold 1 would leave fold 0's npz files behind.
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

    train_dir = Path(TRAIN_DIR)
    meta_csv = train_dir.parent / "metadata.csv"
    meta = pd.read_csv(meta_csv) if meta_csv.exists() else None

    pred_cfg = _predict_config(args)
    # The training run's normalisation, not the CLI's: the detector's thresholds
    # only mean anything against the intensity mapping it was fitted on.
    pred_cfg.normalization = getattr(cfg, "normalization", None)

    if ctx.is_main:
        print(f"-> log_folder     : {log_folder}")
        print(f"-> folds          : {folds}")
        print(f"-> thresholds     : {', '.join(f'{t:g}' for t in thresholds)}")
        print(f"-> nms radii (um) : {', '.join(f'{r:g}' for r in radii)}"
              f"{'  (one forward pass for the whole grid)' if sweeping else ''}")
        print(f"-> writing to      : {', '.join(str(d) for d in out_dirs.values())}")
        if ctx.distributed:
            print(f"-> distributed    : {ctx.world_size} processes\n")

    rows = {key: [] for key in keys}
    all_embryos: list[str] = []
    ckpt_paths: list[str] = []

    for fold in folds:
        # Same split logic as tracking.main.k_fold. --pl flips it: the fold's
        # training embryos are the ones the detector has never been scored on,
        # and they are what a pseudo-labelling pass wants to label.
        if args.embryos:
            embryos = [e.strip() for e in args.embryos.split(",") if e.strip()]
        elif args.pl:
            embryos = [e for e in EMBRYOS if e != EMBRYOS[fold]]
        else:
            embryos = [EMBRYOS[fold]]

        paths = sorted(_paths_for_embryos(str(train_dir), embryos))
        if args.n_videos is not None:
            paths = paths[: args.n_videos]
        if not paths:
            raise ValueError(f"No datasets found for embryos {embryos}")

        ckpt_path = args.ckpt or os.path.join(
            log_folder, f"fold_{fold}", "ckpt_final.pt"
        )
        all_embryos.extend(embryos)
        ckpt_paths.append(ckpt_path)

        if ctx.is_main:
            split = "train (pseudo-label)" if args.pl else "val"
            print(f"-> fold {fold}: {ckpt_path}")
            print(f"   {split} embryos : {embryos}  ({len(paths)} stacks)")

        state = torch.load(ckpt_path, map_location="cpu")
        state = state.get("model_state_dict", state)
        state = {k.replace("unet.module.", "unet.", 1): v for k, v in state.items()}
        # cfg = _reconcile_arch(cfg, state, verbose=ctx.is_main)

        model = build_model(cfg, None)
        model.load_state_dict(state, strict=True)
        model.eval().to(device)

        _predict_fold(
            model, paths, keys, thresholds, radii, out_dirs, rows, meta, pred_cfg,
            window_size, downscale_factor, train_dir, args, ctx, device, fold,
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    embryos = all_embryos
    ckpt_path = ckpt_paths[0] if len(ckpt_paths) == 1 else ckpt_paths

    # One gather for the whole sweep rather than one per grid point.
    gathered = _gather(rows, ctx.world_size)
    all_rows = {key: [r for part in gathered for r in part[key]] for key in keys}

    if not ctx.is_main:
        return

    report_keys = (
        "score", "edge_jaccard", "adj_edge_jaccard", "division_jaccard",
        "node_recall", "edge_tp", "edge_fp", "edge_fn",
        "n_nodes_pred", "n_edges_pred", "n_nodes_estimated",
    )
    summaries = {}
    per_embryo_all = {}

    for key in keys:
        radius, th = key
        th_rows = all_rows[key]
        out_dir = out_dirs[key]

        df = pd.DataFrame(th_rows).sort_values("sample_name")
        df.to_csv(out_dir / "metrics.csv", index=False)
        summary = summarise(th_rows)
        summaries[key] = summary
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=4, default=str)

        # One summary per embryo. Folds hold out different embryos, so this is
        # also the per-fold breakdown. Worth keeping separate: the embryos are
        # annotated at very different densities, and edge FP is only counted
        # against annotated nodes, so pooling them averages two metrics that
        # are not on the same scale.
        by_embryo = {}
        for row in th_rows:
            by_embryo.setdefault(row.get("embryo", "?"), []).append(row)
        group_summaries, macro = macro_summarise(th_rows)
        per_embryo = {
            emb: {
                "fold": sorted(
                    {r.get("fold") for r in by_embryo[emb] if r.get("fold") is not None}
                ),
                "n_samples": len(by_embryo[emb]),
                **_jsonable(summary_emb),
            }
            for emb, summary_emb in group_summaries.items()
        }
        per_embryo_all[key] = per_embryo
        summaries[key]["cv_score"] = macro["score"]

        # Per-sample scores alongside the run-level roll-up.
        # The headline number. The pooled "overall" score weights each embryo
        # by its sample count -- 128 stacks of 6bba against 71 of 44b6 -- so it
        # is close to 6bba's score alone. cv_score gives each embryo one vote.
        worst = sorted(
            (r for r in th_rows if r.get("score") == r.get("score")),
            key=lambda r: r["score"],
        )[:10]
        scores = {
            "checkpoint": ckpt_path,
            "fold": folds[0] if len(folds) == 1 else folds,
            "split": "train" if args.pl else "val",
            "embryos": embryos,
            "n_samples": len(th_rows),
            "cv_score": _jsonable(macro["score"]),
            "macro": _jsonable(macro),
            "overall": _jsonable(summary),
            "per_embryo": per_embryo,
            "threshold": th,
            "nms_radius_um": radius,
            # Named outright rather than left to be rebuilt from the two above:
            # the folder's name depends on which dimensions were swept, which a
            # single scores file cannot tell.
            "preds_dir": str(out_dir),
            "per_sample": {
                r["sample_name"]: _jsonable({k: r[k] for k in report_keys if k in r})
                for r in sorted(th_rows, key=lambda r: r["sample_name"])
            },
            "worst_10": [
                {
                    "sample_name": r["sample_name"],
                    "embryo": r.get("embryo"),
                    **_jsonable({k: r[k] for k in report_keys if k in r}),
                }
                for r in worst
            ],
        }
        stem = "scores_pl" if args.pl else "scores"
        # A lone grid point keeps the historical scores_pl.json /
        # scores_0.25.json names.
        suffix = _key_suffix(radius, th, sweep_th or not args.pl, sweep_r)
        # Beside the predictions they score, which is the log folder unless --out_dir
        # sent those elsewhere. A run writing to a scratch directory should not reach
        # back into the log folder and overwrite the scores of a real one.
        scores_path = base_dir.parent / f"{stem}{suffix}.json"
        with open(scores_path, "w") as f:
            json.dump(scores, f, indent=4)

        print(f"\n{'=' * 60}")
        head = f"threshold {th:g}" + (f"  nms_radius_um {radius:g}" if sweep_r else "")
        print(f"{head}  -  {len(th_rows)} stacks  ->  {out_dir}")
        print(f"-> scores: {scores_path}")
        print(f"{'=' * 60}")
        for k in ("edge_tp", "edge_fp", "edge_fn"):
            if k in summary:
                print(f"- {k}: {summary[k]:.0f}")
        for k in ("node_recall", "edge_jaccard", "adj_edge_jaccard",
                  "division_jaccard", "score"):
            if k in summary:
                print(f"- {k}: {summary[k]:.4f}")

        # Only when there is more than one, otherwise it just repeats the above.
        if len(per_embryo) > 1:
            cols = ("node_recall", "edge_jaccard", "adj_edge_jaccard",
                    "division_jaccard", "score")
            print(f"\n  {'embryo':<10}{'fold':<6}{'n':<5}"
                  + "".join(f"{c:>18s}" for c in cols))
            for emb, s in per_embryo.items():
                fold_str = ",".join(str(f) for f in s["fold"]) or "-"
                cells = "".join(f"{s.get(c, float('nan')):>18.4f}" for c in cols)
                print(f"  {emb:<10}{fold_str:<6}{s['n_samples']:<5}{cells}")

    if sweeping:
        _print_sweep(summaries, per_embryo_all)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
