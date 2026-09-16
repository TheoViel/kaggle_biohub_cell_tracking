"""Blend two detectors' detection logits, and ask what the node set is worth.

    cd src/postprocessing
    python exp_logit_blend.py --primary ../logs/2026-09-15/0 \
        --secondary ../logs/2026-09-11/1 --limit 5

Paths are relative to ``src/``: ``common`` chdirs there, as the other scripts
here do.

The public pipeline's headline ensemble is a second, independently seeded
detector whose *detection logits* are mixed into the primary's before peaks are
found (`PublicChanges.md`, "A adds a dual-seed ensemble"). The secondary is
aligned to the primary by mean and standard deviation, with the scale ratio
clamped to `[0.5, 2.0]`, and mixed at weight 0.475 -- very nearly balanced. Only
the detection stage is blended here; the edge pass still reads the primary's own
features, so what changes is *which cells are found*, not how they are linked.

Why alignment is not optional. Two networks trained separately put their logits
on different scales -- one may be confident at +6 where the other saturates at
+3 -- and averaging raw logits then silently weights the louder one. Matching
the second's mean and standard deviation to the first's puts them on a common
footing before the mix. The clamp is what stops that correction running away on
a frame where the secondary is nearly flat and its standard deviation is close
to zero, which would otherwise multiply its noise by a huge factor.

The arms
--------
``primary`` / ``secondary``   each model alone, for the baseline.
``mean_logit``                the unaligned average, which is the thing
                              alignment is supposed to beat.
``mean_prob``                 average after the sigmoid. Averaging
                              probabilities is not averaging logits: it pulls
                              toward whichever model is *less* certain, so a
                              confident single vote survives less well.
``max_logit``                 the optimistic rule -- a peak either finds is
                              kept, which should buy node recall and pay in
                              node count.
``aligned``                   the public rule: mean/std-align the secondary to
                              the primary, clamp the scale ratio, mix at
                              ``--weight``.

Scored with node recall and adjusted Jaccard through `main_predict`'s own path
-- the detector's edge head, no edge transformer -- so the number is comparable
to a `scores_r<radius>_<threshold>.json` written by `main_predict.py`.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: F401,E402  (chdir to src/ and set the path)

import pandas as pd  # noqa: E402
import tracksdata as td  # noqa: E402

from params import SCALE, TRAIN_DIR  # noqa: E402
from tracking.main import build_model  # noqa: E402
from tracking.post_process import build_graph  # noqa: E402
from tracking.predict import PredictConfig, predict_video_thresholds  # noqa: E402
from tracking_cellmot.metrics import (  # noqa: E402
    evaluate,
    node_recall,
    per_sample_metrics,
)
from utils.logger import Config  # noqa: E402

warnings.filterwarnings("ignore")

CLAMP = (0.5, 2.0)          # the public pipeline's scale-ratio clamp


# --------------------------------------------------------------------------
# the blends
# --------------------------------------------------------------------------

def align(sec: torch.Tensor, pri: torch.Tensor, clamp=CLAMP) -> torch.Tensor:
    """``sec`` put on ``pri``'s scale: match the mean, match the spread.

    The ratio of standard deviations is clamped because a frame where the
    secondary is nearly flat has a standard deviation near zero, and dividing by
    it turns its noise into the loudest thing in the volume.
    """
    ps, ss = pri.std(), sec.std()
    ratio = torch.clamp(ps / (ss + 1e-6), clamp[0], clamp[1])
    return (sec - sec.mean()) * ratio + pri.mean()


def blend(pri: torch.Tensor, sec: torch.Tensor, mode: str, weight: float):
    if mode == "primary":
        return pri
    if mode == "secondary":
        return sec
    if mode == "mean_logit":
        return (1 - weight) * pri + weight * sec
    if mode == "max_logit":
        return torch.maximum(pri, sec)
    if mode == "mean_prob":
        # Back to a logit afterwards: everything downstream -- the threshold,
        # the NMS, the node probabilities -- expects the primary's logit scale.
        p = (1 - weight) * torch.sigmoid(pri) + weight * torch.sigmoid(sec)
        return torch.logit(p.clamp(1e-6, 1 - 1e-6))
    if mode == "aligned":
        return (1 - weight) * pri + weight * align(sec, pri)
    raise ValueError(f"unknown blend {mode!r}")


# --------------------------------------------------------------------------

def load_detector(run_dir, fold, device):
    """The fold's checkpoint, loaded exactly as `main_predict.py` loads it."""
    config = Config(json.load(open(Path(run_dir) / "args.json")))
    state = torch.load(Path(run_dir) / f"fold_{fold}" / "ckpt_final.pt",
                       map_location="cpu")
    state = state.get("model_state_dict", state)
    state = {k.replace("unet.module.", "unet.", 1): v for k, v in state.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(config, None)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device), config


def predict_cfg(config, radius, threshold):
    cfg = PredictConfig()
    mc = getattr(config, "model_config", None) or {}
    cfg.normalization = (mc.get("normalization") if isinstance(mc, dict)
                         else None) or getattr(config, "normalization", None)
    cfg.nms_radius_um = float(radius)
    cfg.det_threshold = float(threshold)
    return cfg


def score(coords, edges, sample, n_total):
    """One stack's metrics, the way `main_predict.py` computes them.

    A high threshold on a deflating blend can leave a stack with no detections
    at all. `evaluate` writes the node matching onto the graph as a side effect
    and skips it when there is nothing to match, so `node_recall` then raises on
    the missing attribute -- an empty arm has recall 0, not an exception.
    """
    g = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{sample}.geff")
    gt = g[0] if isinstance(g, tuple) else g
    with contextlib.redirect_stdout(io.StringIO()):
        graph = build_graph(np.asarray(coords), [
            (int(s), int(t), float(p), float(d)) for s, t, p, d in edges])
        res = evaluate(graph, gt, scale=SCALE)
        try:
            rec = node_recall(graph, gt)
        except KeyError:
            rec = 0.0
    return per_sample_metrics(res, n_total, rec)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--primary", default="../logs/2026-09-15/0")
    p.add_argument("--secondary", default="../logs/2026-09-11/1")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--embryo", default="44b6")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--radius", type=float, default=6.0)
    p.add_argument("--thresholds", default="0.40",
                   help="comma-separated. One forward answers all of them, so "
                        "the grid is nearly free -- and it is the only fair way "
                        "to compare `aligned`, which shifts the logit volume's "
                        "mean and so moves every peak relative to a fixed cut.")
    p.add_argument("--weight", type=float, default=0.475,
                   help="the public pipeline's mix weight on the secondary")
    p.add_argument("--modes", default="primary,secondary,mean_logit,mean_prob,"
                                      "max_logit,aligned")
    a = p.parse_args()
    THS = sorted(float(x) for x in a.thresholds.split(","))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pri_model, pri_cfg = load_detector(a.primary, a.fold, device)
    sec_model, sec_cfg = load_detector(a.secondary, a.fold, device)
    meta = pd.read_csv("../input/metadata.csv").set_index("sample_name")

    samples = sorted(p.stem for p in Path(TRAIN_DIR).glob(f"{a.embryo}_*.geff")
                     )[: a.limit]
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    print(f"-> {len(samples)} stacks, r={a.radius:g}, th {a.thresholds}, "
          f"weight {a.weight}")
    print(f"   primary   {a.primary}")
    print(f"   secondary {a.secondary}\n")

    rows = {(m, t): [] for m in modes for t in THS}
    for i, s in enumerate(samples):
        ds = Path(TRAIN_DIR) / f"{s}.zarr"
        n_total = float(meta.loc[s, "estimated_number_of_nodes"])

        # The secondary runs once; its logits are kept per window and replayed
        # into the primary's pass by the hook.
        with torch.no_grad():
            _, sec_logits = predict_video_thresholds(
                sec_model, ds, THS, device=device,
                cfg=predict_cfg(sec_cfg, a.radius, THS[0]),
                window_size=2, progress=False, return_det_logits=True,
                nms_radii=[a.radius])
        cache = [w for w in sec_logits]          # (n_windows, W, 1, 1, Z, Y, X)

        for mode in modes:
            if mode == "secondary":
                model, cfg, fn = sec_model, sec_cfg, None
            else:
                model, cfg = pri_model, pri_cfg

                def fn(wi, det_logits, mode=mode):
                    out = []
                    for f, pri in enumerate(det_logits):
                        if wi >= len(cache) or f >= len(cache[wi]):
                            out.append(pri)
                            continue
                        sec = cache[wi][f].to(pri.device).to(pri.dtype)
                        if sec.shape != pri.shape:
                            raise SystemExit(
                                f"grid mismatch {tuple(sec.shape)} vs "
                                f"{tuple(pri.shape)}: the two detectors do not "
                                "share a detection grid, so their logits cannot "
                                "be blended without resampling")
                        out.append(blend(pri, sec, mode, a.weight))
                    return out

                if mode == "primary":
                    fn = None
            with torch.no_grad():
                per_key, _ = predict_video_thresholds(
                    model, ds, THS, device=device,
                    cfg=predict_cfg(cfg, a.radius, THS[0]),
                    window_size=2, progress=False, return_det_logits=False,
                    nms_radii=[a.radius], det_logits_fn=fn)
            for th in THS:
                coords, edges = per_key[("det", float(a.radius), th)][:2]
                rows[(mode, th)].append(score(coords, edges, s, n_total))
        print(f"[{i + 1}/{len(samples)}] {s}", flush=True)

    print(f"\n{'mode':<14}{'th':>6}{'nodes':>9}{'ratio':>8}{'node rec':>10}"
          f"{'edge J':>9}{'adj J':>9}")
    best = {}
    for m in modes:
        for th in THS:
            r = rows[(m, th)]
            adj = float(np.mean([x["adj_edge_jaccard"] for x in r]))
            print(f"{m:<14}{th:>6.2f}"
                  f"{np.mean([x['num_pred_nodes'] for x in r]):>9.0f}"
                  f"{np.mean([x['total_node_ratio'] for x in r]):>8.3f}"
                  f"{np.mean([x['node_recall'] for x in r]):>10.4f}"
                  f"{np.mean([x['edge_jaccard'] for x in r]):>9.4f}{adj:>9.4f}")
            if m not in best or adj > best[m][1]:
                best[m] = (th, adj)
        print()
    print(f"{'mode':<14}{'best th':>9}{'best adj J':>12}{'vs primary':>12}")
    ref = best[modes[0]][1]
    for m in modes:
        th, adj = best[m]
        print(f"{m:<14}{th:>9.2f}{adj:>12.4f}{adj - ref:>+12.4f}")
    print("\nMeans over stacks, not the pooled metric: a few stacks cannot be "
          "read as a fold score.")


if __name__ == "__main__":
    main()
