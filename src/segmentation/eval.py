"""Validation on real frames: node recall and the node-count adjustment.

These are the two numbers the segmenter is judged on, and they pull in opposite
directions, which is the point of reporting them together.

**Node recall** is the share of annotated cells that have a prediction within
the competition's 7 um matching radius. It is the only honest detection number
available here: the real ground truth annotates ~3% of the cells, so a
prediction with no annotation next to it is almost always a real cell that
nobody labelled, not a false positive. Precision computed against this ground
truth would be meaningless and is deliberately not reported.

**Adjustment** is the metric's own penalty on over-prediction,
``1 - 0.1 * (T_pred - T_true) / T_true``, where ``T_true`` is the dataset's
``estimated_number_of_nodes`` -- an estimate of *all* the cells, annotated or
not. This is what stops recall being gamed: predicting every voxel would score
perfect recall and a catastrophic adjustment.

Read them as a pair. Recall alone says the detector finds cells; recall at an
adjustment near 1.0 says it finds them without inventing a second copy of each.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from training.distributed import all_gather_lists

from segmentation.dynamics import masks_from_flows_multi
from segmentation.predict import (
    NODES_FOLDER,
    PROJECT_ROOT,
    SCALE,
    gt_nodes,
    load_frame,
    predict_volume,
)

MATCH_UM = 7.0
N_FRAMES_PER_VIDEO = 100


def _estimated_nodes() -> dict[str, float]:
    df = pd.read_csv(PROJECT_ROOT / "input" / "metadata.csv")
    return dict(zip(df["sample_name"], df["estimated_number_of_nodes"].astype(float)))


def match_nodes(pred: np.ndarray, gt: np.ndarray, tol_um: float = MATCH_UM) -> int:
    """How many ground-truth nodes are matched, one prediction each.

    A greedy nearest-neighbour count would let one prediction stand in for two
    annotations sitting close together and overstate recall, so this is a true
    one-to-one assignment.
    """
    if len(pred) == 0 or len(gt) == 0:
        return 0
    d = np.linalg.norm((gt[:, None] - pred[None]) * SCALE, axis=2)
    d = np.where(d <= tol_um, d, 1e6)
    rows, cols = linear_sum_assignment(d)
    return int((d[rows, cols] <= tol_um).sum())


def annotated_frames(sample: str, max_frames: int = 8) -> list[int]:
    """The timepoints of a sample that actually carry annotations, best first.

    The real ground truth is sparse in *time* as well as in space -- one video
    has 52 annotated nodes spread over 76 frames -- so evaluating on frames 0..n
    measures recall against one or two cells and reports a number that swings
    between 0 and 1 on a single match. This takes the most densely annotated
    frames instead, so the recall denominator is large enough to mean something.
    """
    import pandas as pd

    df = pd.read_csv(NODES_FOLDER / f"{sample}.csv")
    counts = df.groupby("t").size().sort_values(ascending=False)
    return [int(t) for t in counts.index[:max_frames]]


def select_eval_samples(n_samples: int, embryos=None) -> list[str]:
    """``n_samples`` real stacks, split evenly between the embryos.

    The two embryos are not interchangeable -- ``44b6`` runs at a median 327
    estimated cells per frame against ``6bba``'s 97, a 3.4x difference in
    density -- so a set drawn without regard to which is which measures
    whichever one it happened to over-sample. Splitting evenly is what makes
    the number comparable between runs.

    Selection is an even spread over the sorted names rather than the first N
    or a random draw: deterministic, so every rank under DDP picks the same
    stacks and two runs are comparable, and unbiased, where taking the most
    densely annotated stacks would quietly evaluate on the least typical ones.
    """
    available = sorted(p.stem for p in NODES_FOLDER.glob("*.csv"))
    by_embryo: dict[str, list[str]] = {}
    for name in available:
        by_embryo.setdefault(name.split("_")[0], []).append(name)
    if embryos:
        by_embryo = {e: by_embryo[e] for e in embryos if e in by_embryo}
    if not by_embryo:
        raise FileNotFoundError(f"no annotated samples under {NODES_FOLDER}")

    keys = sorted(by_embryo)
    # The remainder goes to the first embryos in name order, so an odd count is
    # still reproducible rather than depending on dict ordering.
    quota = {e: n_samples // len(keys) for e in keys}
    for e in keys[: n_samples % len(keys)]:
        quota[e] += 1

    chosen = []
    for embryo in keys:
        pool, k = by_embryo[embryo], quota[embryo]
        if k <= 0:
            continue
        k = min(k, len(pool))
        idx = np.linspace(0, len(pool) - 1, k).round().astype(int)
        chosen += [pool[i] for i in dict.fromkeys(idx.tolist())]
    return sorted(chosen)


def evaluate(
    model,
    samples,
    frames=None,
    max_frames=8,
    thresholds=(0.5,),
    normalization=None,
    device="cuda",
    verbose=True,
    ctx=None,
    **dynamics_kwargs,
):
    """Node recall and adjustment over a set of real samples, per threshold.

    Args:
        samples: sample names, e.g. ``["44b6_0113de3b"]``.
        frames: which timepoints to run per sample. ``None`` picks the most
            densely annotated ones, which is what makes the recall denominator
            large enough to read. ``T_pred`` is the mean
            detection count over the frames actually run, scaled to the video's
            full length -- the same convention ``tracking.eval`` uses, so a
            partial sweep is comparable with a full one.
        ctx: the ``DistContext`` from ``init_distributed``. Under DDP the
            (sample, frame) pairs are dealt out across ranks and the per-frame
            records gathered back, so every rank returns the same numbers. It
            has to be called on *every* rank -- the gather is a collective, and
            putting it behind ``ctx.is_main`` deadlocks the run.

        thresholds: cell-probability thresholds to score. All of them share a
            single forward pass per frame, so a sweep costs barely more than a
            single operating point -- read the curve, not one row.

    Returns:
        ``(summary, per_sample)``, both DataFrames. ``summary`` has one row per
        threshold with ``node_recall``, ``adjustment`` and ``n_pred_per_frame``;
        ``per_sample`` breaks that down by sample.
    """
    estimated = _estimated_nodes()
    thresholds = sorted(float(t) for t in thresholds)
    plan = {
        s: (annotated_frames(s, max_frames) if frames is None else list(frames))
        for s in samples
    }
    # One work item per frame rather than per sample: sharding whole samples
    # leaves ranks idle whenever there are fewer samples than ranks, which is
    # the normal case here.
    items = [(s, t) for s, ts in plan.items() for t in ts]
    rank, world = (ctx.rank, ctx.world_size) if ctx is not None and ctx.distributed else (0, 1)

    was_training = model.training
    model.eval()
    try:
        records = []
        for sample, t in items[rank::world]:
            # The forward is per frame, not per threshold: cell probability and
            # flows do not depend on where the threshold is put, so scoring N
            # thresholds costs one pass over the network and N reconstructions
            # rather than N of each.
            cellprob, flows = predict_volume(
                model, load_frame(sample, t), normalization, device
            )
            gt = gt_nodes(sample, t)
            per_threshold = masks_from_flows_multi(
                cellprob, flows, thresholds=thresholds, device=device,
                return_masks=False, **dynamics_kwargs,
            )
            for th, (_, centroids) in per_threshold.items():
                records.append({
                    "sample": sample,
                    "cellprob_threshold": th,
                    "n_pred": len(centroids),
                    "n_matched": match_nodes(centroids, gt),
                    "n_gt": len(gt),
                })
    finally:
        model.train(was_training)

    records = all_gather_lists(records)

    rows = []
    for th in thresholds:
        for sample in samples:
            mine = [
                r for r in records
                if r["sample"] == sample and r["cellprob_threshold"] == th
            ]
            counts = [r["n_pred"] for r in mine]
            matched = sum(r["n_matched"] for r in mine)
            n_gt = sum(r["n_gt"] for r in mine)

            t_true = estimated.get(sample)
            t_pred = float(np.mean(counts)) * N_FRAMES_PER_VIDEO if counts else 0.0
            rows.append({
                "cellprob_threshold": th,
                "sample": sample,
                "n_frames_scored": len(mine),
                "n_pred_per_frame": float(np.mean(counts)) if counts else 0.0,
                "n_gt": n_gt,
                "n_matched": matched,
                "node_recall": matched / n_gt if n_gt else float("nan"),
                "t_pred": t_pred,
                "t_true": t_true,
                "adjustment": (
                    1.0 - 0.1 * (t_pred - t_true) / t_true if t_true else float("nan")
                ),
            })

    per_sample = pd.DataFrame(rows)
    summary = (
        per_sample.groupby("cellprob_threshold")
        .apply(
            lambda g: pd.Series({
                "node_recall": g["n_matched"].sum() / g["n_gt"].sum()
                if g["n_gt"].sum() else 0.0,
                "adjustment": g["adjustment"].mean(),
                "n_pred_per_frame": g["n_pred_per_frame"].mean(),
                "n_gt": int(g["n_gt"].sum()),
                "n_matched": int(g["n_matched"].sum()),
                "n_samples": len(g),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    if verbose and (ctx is None or ctx.is_main):
        for _, r in summary.iterrows():
            print(
                f"  th {r['cellprob_threshold']:.2f} | "
                f"recall {r['node_recall']:.4f} "
                f"({int(r['n_matched'])}/{int(r['n_gt'])}) | "
                f"nodes/frame {r['n_pred_per_frame']:.0f} | "
                f"adj {r['adjustment']:.4f}"
            )
    return summary, per_sample


def sweep_threshold(model, samples, thresholds=(0.3, 0.5, 0.7), **kwargs):
    """Deprecated: ``evaluate`` takes ``thresholds`` and shares the forward pass."""
    summary, _ = evaluate(model, samples, thresholds=thresholds, **kwargs)
    return summary
