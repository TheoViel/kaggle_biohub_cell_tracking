"""
Run tracking inference with several detectors' detection logits blended.

`main_predict.py` with one extra idea and the same everything else: N runs are
forwarded over each video, their detection logits combined into one heatmap, and
peaks found in that. Only *detection* is ensembled -- the edge pass still reads
the first run's features -- so what changes is which cells are found, not how
they are linked. That is the public pipeline's split (`PublicChanges.md`, "A
adds a dual-seed ensemble") and it is what keeps the output a drop-in for
everything downstream.

Predictions land in <out_dir>/<sample>.npz (coords + edges + node_probs), with
per-sample metrics in metrics.csv and a run-level summary in scores_<...>.json,
exactly as main_predict writes them.

The first --experiments entry is the primary: the run whose scale the others are
aligned onto, whose features link the nodes, and whose config sets the window
size and normalisation. Order matters.

    python main_inference_ensemble.py --experiments ../logs/2026-09-11/1 ../logs/2026-09-15/0

Threshold x radius sweep, one forward pass per model for the whole grid:

    python main_inference_ensemble.py --experiments ../logs/2026-09-11/1 ../logs/2026-09-15/0 \
        --det_threshold 0.25,0.30,0.35,0.40 --nms_radius_um 5,6,7

Multi-GPU (one process per GPU, videos sharded across ranks):

    torchrun --standalone --nproc_per_node=8 main_inference_ensemble.py \
        --experiments ../logs/2026-09-11/1 ../logs/2026-09-15/0

**Sweep the threshold.** Blending shifts the logit distribution -- `aligned`
matches the secondary's mean to the primary's, which moves every peak relative
to a fixed cut -- so a blended run's best threshold is not the single model's.
On six fold-0 stacks the same blend reads as -0.0507 at the primary's 0.40 and
+0.0050 at its own 0.30. Inheriting the threshold is the one way to make this
look broken; see `tracking/blend.py` for the table.

Cost. Each secondary is a full extra forward over the video, so N models cost
about N times the detection. Nothing else changes: the threshold sweep still
rides on one forward per model because neither the cutoff nor the NMS radius
touches it, and the radii still share that forward while each taking its own NMS
pass.
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tracksdata as td

import main_predict as MP
from params import EMBRYOS, LOG_FOLDER, SCALE, TRAIN_DIR
from tracking.blend import MODES, WEIGHT, blend_logits
from tracking.main import _paths_for_embryos, build_model
from tracking.post_process import build_graph
from tracking.predict import PredictConfig, predict_video_thresholds
from tracking_cellmot.metrics import (
    evaluate,
    macro_summarise,
    node_recall,
    per_sample_metrics,
    pooled_counts,
    summarise,
)
from training.distributed import cleanup_distributed, init_distributed
from utils.logger import Config


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", nargs="+", required=True,
                   help="run folders to blend, each holding args.json and "
                        "fold_<n>/ckpt_final.pt. The FIRST is the primary: the "
                        "others are aligned onto its logit scale, its features "
                        "link the nodes, and its config sets the window size "
                        "and normalisation")
    p.add_argument("--blend", default="aligned", choices=MODES,
                   help="how the logits are combined (default %(default)s, the "
                        "public pipeline's rule)")
    p.add_argument("--blend_weight", type=float, default=WEIGHT,
                   help="total mass on the secondaries, split equally between "
                        "them (default %(default)s)")

    # --- everything below mirrors main_predict.py ---
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--out_dir", default=None,
                   help="default <primary>/preds_ens")
    p.add_argument("--embryos", default=None)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--n_videos", type=int, default=None)
    p.add_argument("--pl", action="store_true")
    p.add_argument("--no_window_merge", action="store_true")
    p.add_argument("--n_tta", type=int, choices=(1, 4, 8), default=PredictConfig.n_tta)
    p.add_argument("--det_threshold", type=MP._floats,
                   default=[PredictConfig.det_threshold],
                   help="detection threshold, or a comma-separated list swept "
                        "in one forward pass. Blending moves the distribution, "
                        "so sweep it rather than inheriting the single model's")
    p.add_argument("--pool_kernel_um", type=float, default=PredictConfig.pool_kernel_um)
    p.add_argument("--nms", choices=("maxpool", "greedy"), default=PredictConfig.nms)
    p.add_argument("--nms_radius_um", type=MP._floats,
                   default=[PredictConfig.nms_radius_um])
    p.add_argument("--edge_activation", default=PredictConfig.edge_activation)
    p.add_argument("--edge_selection", default=PredictConfig.edge_selection)
    p.add_argument("--edge_threshold", type=float, default=PredictConfig.edge_threshold)
    p.add_argument("--max_parents_per_node", type=int,
                   default=PredictConfig.max_parents_per_node)
    p.add_argument("--max_children_per_node", type=int,
                   default=PredictConfig.max_children_per_node)
    return p.parse_args()


def _load(run_dir, fold, device):
    """One run's fold checkpoint, loaded the way main_predict loads it."""
    cfg = Config(json.load(open(Path(run_dir) / "args.json")))
    state = torch.load(Path(run_dir) / f"fold_{fold}" / "ckpt_final.pt",
                       map_location="cpu")
    state = state.get("model_state_dict", state)
    state = {k.replace("unet.module.", "unet.", 1): v for k, v in state.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(cfg, None)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device), cfg


def _check_compatible(cfgs, runs):
    """The blend is elementwise, so the models must agree on the input.

    Window size and downscaling decide how many windows a video has and what
    shape each logit volume is; normalisation decides what the models are shown.
    A mismatch in the first two is a crash later on a shape, which is fine but
    unhelpful; a mismatch in the third is silent and worse -- the secondary
    would be reading a differently scaled image and its logits would mean
    something else.
    """
    def norm(c):
        mc = getattr(c, "model_config", None) or {}
        return (mc.get("normalization") if isinstance(mc, dict) else None) \
            or getattr(c, "normalization", None)

    base = cfgs[0]
    for cfg, run in zip(cfgs[1:], runs[1:]):
        for attr, default in (("window_size", 2), ("downscale_factor", 1)):
            if getattr(cfg, attr, default) != getattr(base, attr, default):
                raise SystemExit(
                    f"{run} has {attr}={getattr(cfg, attr, default)} against the "
                    f"primary's {getattr(base, attr, default)}; the two produce "
                    "different window counts and cannot be blended")
        if norm(cfg) != norm(base):
            print(f"   WARNING: {run} normalises differently from the primary "
                  f"({norm(cfg)} vs {norm(base)}). Both are forwarded on their "
                  "own normalisation, so the logits describe differently scaled "
                  "images -- align may not rescue that.")


def _predict_fold(models, cfgs, paths, keys, thresholds, radii, out_dirs, rows,
                  meta, pred_cfgs, window_size, downscale_factor, train_dir,
                  args, ctx, device, fold):
    """One fold's stacks, blended. Mirrors `main_predict._predict_fold`."""
    heads = tuple(dict.fromkeys(h for h, _, _ in keys))
    mine = paths[ctx.rank:: ctx.world_size]
    primary, secondaries = models[0], models[1:]

    for i, path in enumerate(mine):
        name = path.name

        # Each secondary is forwarded once and its per-window logits kept. The
        # grid is deliberately minimal here: the secondary's own peaks are never
        # used, only its heatmap, so its NMS and edge passes are pure waste and
        # a one-point grid is the cheapest way to skip them.
        caches = []
        for model, cfg in zip(secondaries, pred_cfgs[1:]):
            with torch.no_grad():
                _, logits = predict_video_thresholds(
                    model, path, [thresholds[0]], device=device, cfg=cfg,
                    window_size=window_size, downscale_factor=downscale_factor,
                    max_frames=args.max_frames, progress=False,
                    return_det_logits=True, nms_radii=[radii[0]], heads=("det",))
            caches.append(logits)

        def fuse(window_i, det_logits, caches=caches):
            out = []
            for f, pri in enumerate(det_logits):
                secs = []
                for cache in caches:
                    if window_i < len(cache) and f < len(cache[window_i]):
                        s = cache[window_i][f].to(pri.device, pri.dtype)
                        if s.shape != pri.shape:
                            raise SystemExit(
                                f"{name}: detection grids differ, "
                                f"{tuple(s.shape)} against {tuple(pri.shape)}; "
                                "the runs do not share a detection grid and "
                                "their logits cannot be blended elementwise")
                        secs.append(s)
                out.append(blend_logits(pri, secs, mode=args.blend,
                                        weight=args.blend_weight))
            return out

        with torch.no_grad():
            per_key, _ = predict_video_thresholds(
                primary, path, thresholds, device=device, cfg=pred_cfgs[0],
                window_size=window_size, downscale_factor=downscale_factor,
                max_frames=args.max_frames, progress=False,
                return_det_logits=False, nms_radii=radii, heads=heads,
                det_logits_fn=None if args.blend == "primary" else fuse)
        del caches

        gt = td.graph.IndexedRXGraph.from_geff(train_dir / f"{name}.geff")
        gt_graph = gt[0] if isinstance(gt, tuple) else gt

        n_total = float("nan")
        if meta is not None:
            hit = meta.loc[meta["sample_name"] == name, "estimated_number_of_nodes"]
            if len(hit):
                n_total = float(hit.values[0])

        reported = []
        for key in keys:
            head, radius, th = key
            coords, edges, node_probs, annot_probs, coords_f = per_key[key]
            np.savez_compressed(
                out_dirs[key] / f"{name}.npz",
                coords=coords,
                edges=np.asarray(edges, dtype=np.float64).reshape(-1, 4),
                node_probs=np.asarray(node_probs, dtype=np.float32).reshape(-1),
                annot_probs=np.asarray(annot_probs, dtype=np.float32).reshape(-1),
                coords_f=np.asarray(coords_f, dtype=np.float32).reshape(-1, 4),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                graph = build_graph(coords, edges)
                res = evaluate(graph, gt_graph, scale=SCALE)
                try:
                    rec = node_recall(graph, gt_graph)
                except KeyError:
                    # `evaluate` writes the matching as a side effect and skips
                    # it on an empty graph, which a high threshold on a blend
                    # that shifts the volume down can produce.
                    rec = 0.0

            row = per_sample_metrics(res, n_total, rec)
            one = summarise([row])
            row["division_jaccard"] = one["division_jaccard"]
            row["score"] = one["score"]
            row["sample_name"] = name
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
                f"[{tag}] node_recall={rec:.3f} "
                f"edge_jaccard={row['edge_jaccard']:.3f} score={row['score']:.4f}")

        print(f"[rank {ctx.rank}] {i + 1:02d}/{len(mine):02d}  -  {name}  -  "
              + "  ".join(reported), flush=True)


def main():
    args = parse_args()
    ctx = init_distributed(None)
    device = ctx.device

    runs = [r.rstrip("/") + "/" for r in args.experiments]
    thresholds, radii = args.det_threshold, args.nms_radius_um
    sweep_th, sweep_r = len(thresholds) > 1, len(radii) > 1

    cfgs = [Config(json.load(open(r + "args.json"))) for r in runs]
    if ctx.is_main:
        print(f"-> blend   : {args.blend} (weight {args.blend_weight} on "
              f"{len(runs) - 1} secondary/ies)")
        print(f"-> primary : {runs[0]}")
        for r in runs[1:]:
            print(f"   secondary {r}")
        _check_compatible(cfgs, runs)
        if not sweep_th and args.blend != "primary":
            print("   NOTE: one threshold only. Blending moves the logit "
                  "distribution, so the single model's threshold is not this "
                  "run's -- pass --det_threshold with a list.")

    cfg = cfgs[0]
    heads = ("det", "annot") if bool(getattr(cfg, "annot_head", False)) else ("det",)
    keys = [(h, r, th) for h in heads for r in radii for th in thresholds]

    base_dir = Path(args.out_dir or (runs[0] + ("preds_ens_pl" if args.pl
                                                else "preds_ens")))
    out_dirs = {k: MP._out_dir_for(base_dir, *k, sweep_th, sweep_r, args.n_tta)
                for k in keys}
    if ctx.is_main:
        for d in out_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
    if ctx.distributed:
        dist.barrier()

    window_size = getattr(cfg, "window_size", 2)
    downscale_factor = getattr(cfg, "downscale_factor", 1)
    folds = ([args.fold] if args.fold is not None
             else [int(f) for f in (getattr(cfg, "selected_folds", None) or [0])])
    if len(folds) > 1 and args.embryos:
        raise ValueError("--embryos names one fold's split; pass --fold to pick one")

    train_dir = Path(TRAIN_DIR)
    meta_path = train_dir.parent / "metadata.csv"
    meta = pd.read_csv(meta_path) if meta_path.exists() else None
    rows = {k: [] for k in keys}
    all_embryos = []

    for fold in folds:
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
        all_embryos.extend(embryos)

        loaded = [_load(r, fold, device) for r in runs]
        models = [m for m, _ in loaded]
        pred_cfgs = []
        for _, c in loaded:
            pc = MP._predict_config(args)
            mc = getattr(c, "model_config", None) or {}
            pc.normalization = (mc.get("normalization") if isinstance(mc, dict)
                                else None) or getattr(c, "normalization", None)
            pred_cfgs.append(pc)

        if ctx.is_main:
            print(f"-> fold {fold}: {len(paths)} stacks, embryos {embryos}")

        _predict_fold(models, cfgs, paths, keys, thresholds, radii, out_dirs,
                      rows, meta, pred_cfgs, window_size, downscale_factor,
                      train_dir, args, ctx, device, fold)
        del models, loaded
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gathered = MP._gather(rows, ctx.world_size)
    all_rows = {k: [r for part in gathered for r in part[k]] for k in keys}
    if not ctx.is_main:
        return

    report_keys = ("score", "edge_jaccard", "adj_edge_jaccard", "division_jaccard",
                   "node_recall", "edge_tp", "edge_fp", "edge_fn",
                   "n_nodes_pred", "n_edges_pred", "n_nodes_estimated")
    best = None
    for key in keys:
        head, radius, th = key
        th_rows = all_rows[key]
        out_dir = out_dirs[key]
        pd.DataFrame(th_rows).sort_values("sample_name").to_csv(
            out_dir / "metrics.csv", index=False)
        group_summaries, macro = macro_summarise(th_rows)
        by_embryo = {}
        for row in th_rows:
            by_embryo.setdefault(row.get("embryo", "?"), []).append(row)
        per_embryo = {
            emb: {"fold": sorted({r.get("fold") for r in by_embryo[emb]
                                  if r.get("fold") is not None}),
                  "n_samples": len(by_embryo[emb]), **MP._jsonable(s)}
            for emb, s in group_summaries.items()
        }
        worst = sorted((r for r in th_rows if r.get("score") == r.get("score")),
                       key=lambda r: r["score"])[:10]
        scores = {
            "experiments": runs,
            "blend": args.blend,
            "blend_weight": args.blend_weight,
            "fold": folds[0] if len(folds) == 1 else folds,
            "split": "train" if args.pl else "val",
            "embryos": all_embryos,
            "n_samples": len(th_rows),
            "cv_score": MP._jsonable(macro["score"]),
            "macro": MP._jsonable(macro),
            "counts": MP._jsonable(pooled_counts(th_rows)),
            "per_embryo": per_embryo,
            "threshold": th,
            "nms_radius_um": radius,
            "preds_dir": str(out_dir),
            "per_sample": {r["sample_name"]: MP._jsonable(
                {k: r[k] for k in report_keys if k in r})
                for r in sorted(th_rows, key=lambda r: r["sample_name"])},
            "worst_10": [{"sample_name": r["sample_name"], "embryo": r.get("embryo"),
                          **MP._jsonable({k: r[k] for k in report_keys if k in r})}
                         for r in worst],
        }
        suffix = MP._key_suffix(head, radius, th, sweep_th or not args.pl, sweep_r,
                                args.n_tta)
        stem = "scores_ens_pl" if args.pl else "scores_ens"
        with open(base_dir.parent / f"{stem}{suffix}.json", "w") as f:
            json.dump(scores, f, indent=4)
        if best is None or macro["score"] > best[0]:
            best = (macro["score"], key)

    if len(keys) > 1:
        print(f"\n{'head':<6}{'radius':>8}{'thr':>7}{'node rec':>10}{'adj J':>9}"
              f"{'div J':>8}{'cv':>9}")
        for key in keys:
            _, macro = macro_summarise(all_rows[key])
            h, r, t = key
            print(f"{h:<6}{r:>8g}{t:>7.2f}{macro['node_recall']:>10.4f}"
                  f"{macro['adj_edge_jaccard']:>9.4f}"
                  f"{macro['division_jaccard']:>8.4f}{macro['score']:>9.4f}")
        print(f"\n-> best: {best[1]} at cv {best[0]:.4f}")


if __name__ == "__main__":
    main()
