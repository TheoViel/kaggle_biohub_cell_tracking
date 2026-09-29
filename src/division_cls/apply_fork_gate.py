"""Drop the forks the division classifier disbelieves, from a finished chain.

    python division_cls/apply_fork_gate.py \
        --preds_dir /tmp/preds_pp/ --data_dir /kaggle/.../test/ \
        --run_dir  /kaggle/.../division_2026-09-16_0/ \
        --out_dir /tmp/preds_pp_gated/ --geff_dir /tmp/preds_pp_gated_geff/

`tune_fork_threshold.py` does this inside a sweep -- it rebuilds the graph per
threshold and scores it against the annotation. This is the production form of
the same step: read a post-processed folder, delete the second-child edge of
every fork the classifier scores below `--tau`, and write the npz and geff back
out in `main_edge_postprocess.py`'s format so `submission.py` reads them
unchanged.

Measured on the 199 out-of-fold stacks, over `main_edge_postprocess.py --ilp`
on `edge_run/2026-09-13/0/preds_0.45`: **cv 0.8735 -> 0.8791**, from
`division_jaccard` 0.1619 -> 0.2150 with `adj_edge_jaccard` unmoved
(0.8574 -> 0.8576). It removes 121 of 138 division false positives and 9 of 46
true ones.

Two things differ from that measurement and both are inherent to running on a
test split rather than on the folds:

* **No fold mapping.** Out of fold, each embryo is scored by the model that
  held it out. A test stack belongs to neither, so `--fold mean` averages the
  probabilities of every `fold_*/ckpt_final.pt` in `--run_dir`. The two folds
  differ in strength (AP 0.627 against 0.426), so this is not the same
  estimator that produced the number above -- it is the closest thing
  available when the embryo is unknown.
* **No re-pruning.** The measurement's best arm leaves `prune_short` where it
  was, so this does not re-run it. `--reprune` re-runs it for the case where
  the pool came from the ILP's fork harvest, which strands fragments when the
  forks holding them are deleted; without those forks it is worth ~0.

`--tau` is deliberately not a tuned parameter: the classifier saturates
(positives median 1.000, negatives 0.000) and the result is unchanged from 0.1
to 0.7.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "postprocessing"))

from torch.utils.data import DataLoader  # noqa: E402

import edge_pp  # noqa: E402
from division_cls.dataset import DivisionDataset  # noqa: E402
from division_cls.gen_fork_candidates import COLUMNS, forks_table  # noqa: E402
from division_cls.infer import load_model  # noqa: E402
from tracking.post_process import build_graph  # noqa: E402
from tracking_cellmot.io import save_graph  # noqa: E402


def _key(t, z, y, x, prob):
    return (int(t), round(float(z), 3), round(float(y), 3), round(float(x), 3),
            round(float(prob), 6))


def build_candidates(preds_dir, out_path):
    """Every fork in a post-processed folder, in `gen_fork_candidates` format.

    The label column stays 0: there is no annotation on a test split, and
    nothing downstream of here reads it (`candidates_labelled_only=False`).
    """
    stems = sorted(q.stem for q in Path(preds_dir).glob("*.npz"))
    stem_col, tables = [], []
    for stem in stems:
        z = np.load(Path(preds_dir) / f"{stem}.npz")
        # `coords` is rounded to the voxel grid on the way to disk, and the
        # features below (parent_dist_um, sister_dist_um) plus the crop centre
        # are derived from these positions -- so the rounded copy makes this
        # gate score something the chain never proposed. Measured: reading
        # `coords` instead of `coords_exact` costs 0.0140 cv on
        # edge_run/2026-09-24/4 (0.8715 against the sweep's 0.8855), because
        # the shifted features move the classifier off its saturated regime
        # and it then rejects 2.5x as many forks.
        coords = z["coords_exact"] if "coords_exact" in z.files else z["coords"]
        table = forks_table(np.asarray(coords, float),
                            np.asarray(z["edges"], float).reshape(-1, 4))
        if len(table):
            stem_col += [stem] * len(table)
            tables.append(table)
    table = (np.concatenate(tables) if tables
             else np.zeros((0, len(COLUMNS)), np.float32))
    np.savez_compressed(out_path, stem=np.asarray(stem_col), table=table,
                        columns=np.asarray(COLUMNS))
    return stems, len(table)


def rank_per_stack(stems, score):
    """`score` replaced by its within-stack rank in [0, 1], NaN preserved.

    Identical to `merge_fork_scores.py --mode rank`, and for its reasons: the
    models are calibrated differently -- one saturates at 0/1, another does not
    -- so averaging probabilities lets the confident-but-wrong model decide.
    Ranks carry only the ordering, which is all the threshold uses. Per stack,
    because one threshold is applied everywhere and a stack with many forks
    would otherwise set the scale for all of them.
    """
    stems = np.asarray([str(s) for s in stems])
    s = np.asarray(score, float)
    out = np.full(len(s), np.nan)
    for stem in set(stems.tolist()):
        m = stems == stem
        v = s[m]
        ok = np.isfinite(v)
        if not ok.any():
            continue
        rr = np.full(len(v), np.nan)
        # argsort(argsort(v)) IS the rank vector; scattering it instead returns
        # the sorting indices, a different permutation that destroys the order.
        rr[ok] = np.argsort(np.argsort(v[ok])) / max(ok.sum() - 1, 1)
        out[m] = rr
    return out


TTA_FLIPS = [(), (-1,), (-2,), (-1, -2)]


def tta_probs(model, clip, feat, n_tta):
    """Mean sigmoid over `n_tta` flips of the Y/X plane.

    The clip is ``(B, T, pz, H, W)``, so the flips act on the last two axes
    only -- flipping z would reorder the slab, which is a depth axis and not a
    symmetry of the data.

    Averaging probabilities rather than logits: the flips are the *same* model
    on the same crop, so they share a calibration and the mean is the estimator
    we want. (The rank-vs-probability argument in `merge_fork_scores` is about
    combining *different* models, which is the opposite case.)

    Nearly free here. Cutting the crops dominates -- 8.4 s a stack for two
    models against 5.2 s for one -- and TTA reuses tensors already in memory,
    so four views cost four cheap forwards, not four passes.

    Training already applies `flip_p 0.5` and `rot90_p 0.5`, so these views are
    inside the distribution the model was fitted on rather than novel ones.
    """
    flips = TTA_FLIPS[:n_tta] if n_tta > 1 else [()]
    out = None
    for d in flips:
        x = clip.flip(dims=d) if d else clip
        p = torch.sigmoid(model(x, feat)).float()
        out = p if out is None else out + p
    return out / len(flips)


def _ckpts_for(run_dir, fold):
    """The checkpoints of one run that `fold` selects."""
    if fold == "fullfit":
        # One model trained on every embryo. Half the crops-and-forward cost of
        # averaging the folds, and trained on both embryos' divisions rather
        # than one -- but it has seen every annotated stack, so it is only
        # honest on a test split.
        return [Path(run_dir) / "fold_fullfit" / "ckpt_final.pt"]
    if fold == "mean":
        # `fold_fullfit` matches `fold_*`, and averaging it in here would
        # quietly turn an out-of-fold estimator into a leaked one -- the
        # failure would look like an improvement. Excluded by name; ask for it.
        return sorted(q for q in Path(run_dir).glob("fold_*/ckpt_final.pt")
                      if q.parent.name != "fold_fullfit")
    return [Path(run_dir) / f"fold_{int(fold)}" / "ckpt_final.pt"]


def score_candidates(cand_path, data_dir, run_dirs, device, batch_size, workers,
                     fold="mean", devices=None, n_tta=1):
    """`{run_dir: (n,) probabilities}` for every model, in as few passes as possible.

    Two costs to beat, and the profile is lopsided: the network forward on the
    ~80 forks of a stack is nothing, while opening the zarr and cutting the
    crops is nearly all of it -- 8.4 s a stack for two fold models against
    5.2 s for one, so the *second* model costs 3.2 s and the crop pass ~2 s.

    **One crop pass per crop geometry, not per model.** Any models that agree on
    ``(patch_yx, patch_z, n_frames)`` read the same tensors, so they are scored
    inside one batch loop. The five-model ensemble has three geometries --
    32/3/3 (three models), 32/3/5, 48/3/5 -- so this is three passes rather
    than five, and the two extra models on the shared geometry are nearly free.
    Mismatched geometry is grouped apart rather than silently shared: scoring a
    model on another's crops is wrong, not slow.

    **Both GPUs, sharded by row.** The forks are independent, so each device
    takes a contiguous half of the candidate table and runs the whole group --
    dataset, loader and models -- on its own. Sharding rows rather than
    assigning whole groups to devices keeps the split even; with three groups
    on two devices, per-group assignment would leave one card doing two thirds
    of the work. A thread per device is enough: the GIL is released across the
    CUDA calls and the loader's workers are separate processes anyway.
    """
    devices = devices or [device]
    blob = np.load(cand_path, allow_pickle=True)
    n = len(blob["table"])
    if not n:
        return {rd: np.zeros(0, np.float32) for rd in run_dirs}

    # Load each run's checkpoints once, on the first device, only to read the
    # crop geometry out of the config; the per-device copies are loaded below.
    groups: dict[tuple, list] = {}
    for rd in run_dirs:
        ck = _ckpts_for(rd, fold)
        if not ck:
            raise FileNotFoundError(f"no fold_*/ckpt_final.pt under {rd}")
        _, cfg = load_model(ck[0], "cpu")
        geom = (cfg["patch_yx"], cfg["patch_z"], cfg["n_frames"])
        for c in ck[1:]:
            _, c2 = load_model(c, "cpu")
            if (c2["patch_yx"], c2["patch_z"], c2["n_frames"]) != geom:
                raise RuntimeError(
                    f"{rd}: fold models disagree on crop geometry; they cannot "
                    "share one pass. Score them separately with --fold <n>.")
        groups.setdefault(geom, []).append((rd, ck, cfg))

    print(f"-> {len(run_dirs)} model(s) in {len(groups)} crop geometr"
          f"{'y' if len(groups) == 1 else 'ies'}, {len(devices)} device(s)",
          flush=True)

    out = {rd: np.zeros(n, np.float64) for rd in run_dirs}
    counts = {rd: 0 for rd in run_dirs}

    for geom, members in groups.items():
        cfg = members[0][2]
        ds = DivisionDataset(
            data_dir, embryos=None, candidates=str(cand_path),
            candidates_labelled_only=False, augment=False,
            patch_yx=cfg["patch_yx"], patch_z=cfg["patch_z"],
            n_frames=cfg["n_frames"],
        )
        if len(ds.samples) != n:
            raise RuntimeError(
                f"dataset has {len(ds.samples)} rows against {n} candidates; "
                "they must be the same rows in the same order")
        shards = np.array_split(np.arange(n), len(devices))

        def work(dev, rows):
            if not len(rows):
                return
            loaded = [(rd, [load_model(c, dev)[0] for c in ck])
                      for rd, ck, _ in members]
            loader = DataLoader(
                torch.utils.data.Subset(ds, rows.tolist()),
                batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True)
            acc = {rd: [] for rd, _, _ in members}
            with torch.no_grad():
                for batch in loader:
                    clip = batch["clip"].to(dev, non_blocking=True)
                    feat = batch.get("feat")
                    feat = (feat.to(dev, non_blocking=True)
                            if feat is not None and feat.numel() else None)
                    with torch.autocast("cuda", torch.float16,
                                        enabled=str(dev).startswith("cuda")):
                        for rd, models in loaded:
                            p = sum(tta_probs(mo, clip, feat, n_tta)
                                    for mo in models) / len(models)
                            acc[rd].append(p.cpu())
            for rd, parts in acc.items():
                out[rd][rows] = torch.cat(parts).numpy().astype(np.float64)

        threads = [threading.Thread(target=work, args=(d, r))
                   for d, r in zip(devices, shards)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for rd, ck, _ in members:
            counts[rd] = len(ck)
            print(f"   {Path(rd).name}: {len(ck)} ckpt(s) on geometry "
                  f"{geom[0]}/{geom[1]}/{geom[2]}", flush=True)

    return {rd: out[rd].astype(np.float32) for rd in run_dirs}


def gate_stack(coords, edges, lookup, tau, min_track=None):
    """Delete the second-child edge of every fork scoring below `tau`."""
    edges = np.asarray(edges, float).reshape(-1, 4)
    src = edges[:, 0].astype(int)
    children: dict[int, list[int]] = {}
    for k in np.argsort(-edges[:, 2], kind="stable"):
        children.setdefault(int(src[k]), []).append(int(k))
    drop = []
    for parent, ks in children.items():
        if len(ks) < 2:
            continue
        c = coords[parent]
        for k in ks[1:]:
            # An unscored fork is kept: absence of a score is not disbelief.
            s = lookup.get(_key(c[0], c[1], c[2], c[3], edges[k, 2]), np.nan)
            if np.isfinite(s) and s < tau:
                drop.append(k)
    if drop:
        edges = edges[np.setdiff1d(np.arange(len(edges)), np.asarray(drop, int))]
    if min_track:
        coords, edges = edge_pp.prune_short(coords, edges, min_track)
    return coords, edges, len(drop)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preds_dir", required=True, help="main_edge_postprocess output")
    p.add_argument("--data_dir", required=True, help="the zarr folder the crops come from")
    p.add_argument("--run_dir", required=True, nargs="+",
                   help="one or more division_cls runs. With several, each "
                        "scores every fork and the scores are rank-averaged "
                        "per stack -- so --tau becomes a QUANTILE, not a "
                        "probability. Measured on the v4 harvest: one model at "
                        "prob 0.5 gives cv 0.8860 / div_J 0.2398; the four-model "
                        "set (c48nf, f5, c48f5, base) at rank 0.90 gives 0.8883 "
                        "/ 0.2622, by cutting division FPs 77 -> 45")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--geff_dir", required=True)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--fold", default="mean",
                   help='"mean" (default, out-of-fold models only), '
                        '"fullfit" for the all-embryo model, or a fold index')
    p.add_argument("--reprune", type=int, default=0,
                   help="re-run prune_short at this min_track after gating; "
                        "0 = off, which is what the measured arm does")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--devices", default=None,
                   help="comma-separated devices to shard the forks over, e.g. "
                        "'cuda:0,cuda:1'. Default: every visible CUDA device. "
                        "The forks are independent, so each device takes a "
                        "contiguous slice of the table and runs the whole "
                        "model group on it")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--merge", choices=("prob", "rank"), default="prob",
                   help="how several --run_dir are combined. 'prob' averages "
                        "the probabilities and --tau is a probability; 'rank' "
                        "averages per-stack ranks and --tau is a quantile. "
                        "prob measured better on every grouping tried on the "
                        "28/0 chain (0.2884 vs 0.2095 div_J on three models)")
    p.add_argument("--tta", type=int, choices=(1, 2, 4), default=1,
                   help="average the probabilities over this many Y/X flips "
                        "(see tta_probs). Nearly free: the crops are reused")
    a = p.parse_args()

    if a.devices:
        devices = [d.strip() for d in a.devices.split(",") if d.strip()]
    elif str(a.device).startswith("cuda") and torch.cuda.device_count() > 1:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        devices = [a.device]

    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    Path(a.geff_dir).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        cand_path = Path(tmp) / "forks.npz"
        stems, n = build_candidates(a.preds_dir, cand_path)
        print(f"-> {len(stems)} stacks, {n} forks", flush=True)
        blob = np.load(cand_path, allow_pickle=True)
        cand_stems = [str(s) for s in blob["stem"]]
        table = blob["table"]
        # One call for every model: it groups them by crop geometry and
        # shards the rows across the devices, so the crops are cut once per
        # geometry rather than once per model.
        scored = score_candidates(cand_path, a.data_dir, a.run_dir, a.device,
                                  a.batch_size, a.num_workers, a.fold,
                                  devices=devices, n_tta=a.tta)
        per_model = [scored[rd] for rd in a.run_dir]
        for rd, sc in zip(a.run_dir, per_model):
            print(f"  {Path(rd).name}: median {np.nanmedian(sc):.3f}", flush=True)
        if len(per_model) == 1:
            scores = per_model[0]
        elif a.merge == "prob":
            # The mean of the probabilities. Measured on the 28/0 harvest, three
            # models: mean 0.2884 mean-div_J against rank's 0.2095 over the same
            # tau range. `merge_fork_scores` prefers rank because members
            # calibrate differently, and that was true of the 13/0-chain pool
            # where one model saturated at 0/1 and another did not -- on pools
            # built from the current chain the calibrations are close, so
            # ranking discards the magnitude for nothing.
            stack = np.vstack(per_model)
            scores = np.nanmean(stack, axis=0)
            scores[np.all(~np.isfinite(stack), axis=0)] = np.nan
            print(f"-> probability-averaged {len(per_model)} models; --tau "
                  f"{a.tau} is a probability", flush=True)
        else:
            ranked = np.vstack([rank_per_stack(cand_stems, s) for s in per_model])
            scores = np.nansum(ranked, axis=0) / len(per_model)
            scores[np.all(~np.isfinite(ranked), axis=0)] = np.nan
            print(f"-> rank-averaged {len(per_model)} models; --tau {a.tau} is a "
                  f"quantile", flush=True)

    lookups: dict[str, dict] = {}
    ip = list(COLUMNS).index("child_prob")
    for stem, row, s in zip(cand_stems, table, scores):
        lookups.setdefault(stem, {})[
            _key(row[0], row[1], row[2], row[3], row[ip])] = float(s)

    total_drop = 0
    for stem in stems:
        z = np.load(Path(a.preds_dir) / f"{stem}.npz")
        # Must be the array `build_candidates` keyed the lookup on, or every
        # `_key` misses and the gate silently keeps every fork.
        coords = np.asarray(
            z["coords_exact"] if "coords_exact" in z.files else z["coords"],
            float)
        coords, edges, n_drop = gate_stack(
            coords, z["edges"], lookups.get(stem, {}), a.tau, a.reprune)
        total_drop += n_drop
        np.savez_compressed(
            Path(a.out_dir) / f"{stem}.npz",
            coords=np.round(np.asarray(coords, np.float64)).astype(np.int16),
            edges=np.asarray(edges, np.float64).reshape(-1, 4))
        graph = build_graph(np.round(np.asarray(coords)).astype(np.int64), [
            (int(s), int(t), float(p), float(d))
            for s, t, p, d in np.asarray(edges, float).reshape(-1, 4)])
        save_graph(graph, str(Path(a.geff_dir) / f"{stem}.geff"))

    print(f"-> dropped {total_drop} of {n} forks at tau={a.tau}")
    print(f"-> npz  : {a.out_dir}")
    print(f"-> geff : {a.geff_dir}")


if __name__ == "__main__":
    main()
