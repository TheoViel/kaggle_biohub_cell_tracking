"""Tune the fork candidate pool the division classifier sees, on fold 0.

    cd src
    python division_cls/sweep_candidates.py --pool          # counts only, no GPU
    python division_cls/sweep_candidates.py --eval ARM ...  # score + gate + evaluate

Two stages, because they cost very different amounts and answer different
questions.

`--pool` runs the chain per arm and counts the forks it produces, split by
label. It needs no model and no GPU, and it bounds everything: a GT division
whose fork is not in the pool cannot be retrieved by any classifier at any
threshold, so `pos` is the ceiling on division recall and `n` is what the
classifier has to reject to get there. Arms that do not move `pos` are not
worth scoring.

`--eval` takes the arms worth scoring, runs the existing out-of-fold model over
their pools, gates at each tau and evaluates the graph. No training: the model
is whatever `--run_dir` holds.

The gate order is a parameter, because where `prune_short` and `linefit` sit
relative to the gate is itself one of the things being tested:

    in_chain   postprocess (prune, linefit inside) -> gate
    reprune    postprocess (prune, linefit inside) -> gate -> prune
    late       postprocess (no prune, no linefit)  -> gate -> prune -> linefit

`late` is the order the chain arguably should have had all along -- a fork that
is about to be deleted should not have rescued a fragment from pruning, and
`linefit` should smooth the graph that survives rather than the one that does
not. It needs its own candidate pool: pruning removes forks whose parent went
with it, and linefit moves the coordinates the fork key is built from, so a
pool generated under one order does not match a graph built under another.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "postprocessing"))

import edge_pp  # noqa: E402
import main_edge_postprocess as MEP  # noqa: E402
import multi_hypothesis as M  # noqa: E402
from division_cls import gen_fork_candidates as G  # noqa: E402
from tracking_cellmot.metrics import macro_summarise  # noqa: E402

# Set from the CLI before any Pool is created, so the workers inherit them.
PREDS = "../logs/edge_run/2026-09-13/0/preds_0.45"
RUN_DIR = "../logs/division/2026-09-16/0"
WORK = Path("../logs/fork_cls/sweep")
EMBRYO = "44b6"          # "" = every stack, which is what a macro needs

# Every arm keeps the ILP *node filter* (production) and drops its fork harvest,
# which the 3-stack count said contributes ~1 fork in 2 213 once
# `propose_divisions` is ungated.
BASE = dict(ilp=1, ilp_disappearance=0.75, ilp_threshold=0.4,
            ilp_div_forks=0, divisions=1)

# Loosening `propose_divisions`, which is what actually fills the pool. Each
# arm is BASE plus these. `order` picks where prune/linefit sit.
ARMS: dict[str, dict] = {
    # -- reference points -------------------------------------------------
    # The production gates: divergence required, caps binding.
    "prod": dict(),
    # What FORK_CFG does to the proposer, minus the ILP forks.
    "ungated": dict(div_require_divergence=0,
                    div_global_frac_cap=1.0, div_frame_frac_cap=1.0),
    # -- one knob at a time, from `prod` ----------------------------------
    "caps_off": dict(div_global_frac_cap=1.0, div_frame_frac_cap=1.0),
    "diverge_off": dict(div_require_divergence=0),
    "mutual_off": dict(div_require_mutual_nn=0),
    "prob_010": dict(div_min_prob=0.10),
    "prob_002": dict(div_min_prob=0.02),
    "parent_15": dict(div_max_parent_um=15.0),
    "sister_20": dict(div_max_sister_um=20.0),
    "child_15": dict(div_max_existing_child_um=15.0),
    "diverge_15": dict(div_diverge_um=1.5),
    # -- combinations -----------------------------------------------------
    "wide": dict(div_require_divergence=0, div_global_frac_cap=1.0,
                 div_frame_frac_cap=1.0, div_min_prob=0.10,
                 div_max_parent_um=15.0, div_max_sister_um=20.0),
    "wider": dict(div_require_divergence=0, div_global_frac_cap=1.0,
                  div_frame_frac_cap=1.0, div_min_prob=0.02,
                  div_max_parent_um=20.0, div_max_sister_um=26.0,
                  div_max_existing_child_um=15.0, div_require_mutual_nn=0,
                  div_diverge_um=1.5),
    # -- stealing: a claimed node as second child, priced against the edge it
    # breaks. The only arm here that can raise the 11-of-25 pool ceiling.
    "steal00": dict(div_steal_margin=0.0),
    "steal01": dict(div_steal_margin=0.1),
    "steal_neg": dict(div_steal_margin=-0.1),
    "steal_ungated": dict(div_steal_margin=0.0, div_require_divergence=0,
                          div_global_frac_cap=1.0, div_frame_frac_cap=1.0),
    # Directed by the gate diagnostic: on the 12 divisions whose second
    # daughter is claimed, the model scores that link a median 0.139 against
    # `div_min_prob` 0.3 (9/12 rejected) and the assignment's competing edge
    # scores 0.713 (so the price rejects 9/12 too). Opening one without the
    # other reaches nothing, which is what `steal_ungated` measured.
    "steal_all": dict(div_steal_margin=-1.0, div_min_prob=0.02,
                      div_require_divergence=0, div_global_frac_cap=1.0,
                      div_frame_frac_cap=1.0, div_max_parent_um=15.0),
    "steal_all_nn": dict(div_steal_margin=-1.0, div_min_prob=0.02,
                         div_require_divergence=0, div_global_frac_cap=1.0,
                         div_frame_frac_cap=1.0, div_max_parent_um=15.0,
                         div_require_mutual_nn=0),
}

# Arms whose pool is built without prune/linefit, for the `late` order. Filled
# from the winners of the pool stage by `--late_of`.
ORDERS = ("in_chain", "reprune", "late")


def _stems():
    """The stacks this run covers. `EMBRYO=""` means both, which is what a
    macro against Experiments.md needs -- the classifier is leave-one-embryo-out
    so each embryo is still scored by a model that never saw it."""
    return sorted(q.stem for q in Path(PREDS).glob("*.npz")
                  if q.stem.startswith(EMBRYO))


def arm_cfg(name: str, order: str) -> dict:
    cfg = dict(MEP.DEFAULTS)
    cfg.update(out_dir="", geff_dir="", no_score=True, assign=1)
    cfg.update(BASE)
    cfg.update(ARMS[name])
    if order == "late":
        # The pool and the graph it is gated on must come from the same chain,
        # so both are built with these off and both stages run afterwards.
        cfg.update(min_track=0, linefit=0.0)
    return cfg


# --------------------------------------------------------------------------
# pool stage
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# the shared prefix
#
# Every arm here varies only `div_*`, `min_track` and `linefit` -- all of them
# in `postprocess_suffix`. The prefix above them (ILP -> assign -> recover) is
# identical across the whole sweep, and the ILP alone is ~30 s a stack against
# ~1 s for the suffix. Solved once, cached to disk, and each arm runs the
# suffix over it: 13 arms cost one prefix pass rather than thirteen.
# --------------------------------------------------------------------------

def _prefix_one(stem):
    path = WORK / "prefix" / f"{stem}.npz"
    if path.exists():
        return stem, True
    z = np.load(Path(PREDS) / f"{stem}.npz")
    cfg = dict(MEP.DEFAULTS)
    cfg.update(out_dir="", geff_dir="", no_score=True, assign=1)
    cfg.update(BASE)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        coords, edges, cand = MEP.postprocess_prefix(
            z["coords"].astype(np.float64), np.empty((0, 4)),
            np.asarray(z["candidates"], float).reshape(-1, 5), cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, coords=np.asarray(coords, np.float32),
                        edges=np.asarray(edges, float).reshape(-1, 4).astype(np.float32),
                        cand=np.asarray(cand, np.float32))
    return stem, False


def build_prefix(stems, workers):
    hits = 0
    with Pool(workers) as pool:
        for i, (stem, cached) in enumerate(pool.imap_unordered(_prefix_one, stems), 1):
            hits += cached
            if i % 20 == 0 or i == len(stems):
                print(f"  prefix {i}/{len(stems)} ({hits} cached)", flush=True)


def load_prefix(stem):
    z = np.load(WORK / "prefix" / f"{stem}.npz")
    return (z["coords"].astype(np.float64),
            z["edges"].astype(float).reshape(-1, 4),
            z["cand"].astype(float).reshape(-1, 5))


def _pool_one(job):
    stem, cfg = job
    coords, edges, cand = load_prefix(stem)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        coords, edges = MEP.postprocess_suffix(coords, edges, cand, cfg)
    table = G.forks_table(coords, edges)
    if len(table):
        table[:, 4] = G.label_table(table, stem)
    return stem, table


def run_pool(names, order, workers):
    out = {}
    for name in names:
        cfg = arm_cfg(name, order)
        stems = _stems()
        stem_col, tables = [], []
        with Pool(workers) as pool:
            for stem, table in pool.imap_unordered(
                    _pool_one, [(s, cfg) for s in stems]):
                if len(table):
                    stem_col += [stem] * len(table)
                    tables.append(table)
        table = (np.concatenate(tables) if tables
                 else np.zeros((0, len(G.COLUMNS)), np.float32))
        lab = table[:, 4].astype(np.int8)
        # Unique dividing parents, not rows: a three-way split contributes
        # several rows for one GT division and the ceiling is per division.
        pos_parents = len({tuple(r) for r in table[lab == 1, :4].tolist()})
        rec = {"arm": name, "order": order, "n": len(table),
               "pos": int((lab == 1).sum()), "pos_parents": pos_parents,
               "neg": int((lab == 0).sum()), "unl": int((lab == -1).sum())}
        path = WORK / f"forks_{name}_{order}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, stem=np.asarray(stem_col), table=table,
                            columns=np.asarray(G.COLUMNS))
        rec["path"] = str(path)
        out[name] = rec
        print(f"  {name:12s} {order:9s} n={rec['n']:7d}  pos={rec['pos']:4d}  "
              f"div_parents={pos_parents:3d}/26  neg={rec['neg']:5d}", flush=True)
    return out


# --------------------------------------------------------------------------
# eval stage
# --------------------------------------------------------------------------

def _key(t, z, y, x, prob):
    return (int(t), round(float(z), 3), round(float(y), 3), round(float(x), 3),
            round(float(prob), 6))


def _eval_one(job):
    stem, cfg, order, taus, lookup, min_track, linefit, linefit_window = job
    coords, edges, cand = load_prefix(stem)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        coords, edges = MEP.postprocess_suffix(coords, edges, cand, cfg)
    edges = np.asarray(edges, float).reshape(-1, 4)
    src = edges[:, 0].astype(int)
    children: dict[int, list[int]] = {}
    for k in np.argsort(-edges[:, 2], kind="stable"):
        children.setdefault(int(src[k]), []).append(int(k))
    scored = []
    for parent, ks in children.items():
        if len(ks) < 2:
            continue
        c = coords[parent]
        for k in ks[1:]:
            scored.append((int(k), lookup.get(
                _key(c[0], c[1], c[2], c[3], edges[k, 2]), np.nan)))

    rows = []
    for tau in taus:
        drop = [k for k, s in scored if np.isfinite(s) and s < tau] if scored else []
        e = (edges[np.setdiff1d(np.arange(len(edges)), np.asarray(drop, int))]
             if drop else edges)
        c = coords
        if order in ("reprune", "late"):
            c, e = edge_pp.prune_short(c, e, min_track)
        if order == "late" and linefit:
            c = edge_pp.linefit_smooth(c, e, window=linefit_window, weight=linefit)
        row = M.score_one(stem, c, e)
        row.update(_tau=tau, _dropped=len(drop), _forks=len(scored),
                   _matched=sum(1 for _, s in scored if np.isfinite(s)))
        rows.append(row)
    return rows


def load_lookup(path):
    """`{stem: {fork key: score}}` -- `tune_fork_threshold.load_lookups`.

    Imported rather than reimplemented: a scores file stores `coords` and
    `features`, not the candidate table, and `features[2]` is `child_prob`,
    which is half the key. Getting that wrong does not raise -- every lookup
    misses, every fork counts as unscored, and unscored forks are kept, so the
    sweep reports a gate that drops nothing at every threshold.
    """
    from division_cls.tune_fork_threshold import load_lookups
    return load_lookups(path)


def run_eval(name, order, scores_path, taus, workers):
    cfg = arm_cfg(name, order)
    lookups = load_lookup(scores_path)
    stems = _stems()
    jobs = [(s, cfg, order, taus, lookups.get(s, {}),
             MEP.DEFAULTS["min_track"], MEP.DEFAULTS["linefit"],
             MEP.DEFAULTS["linefit_window"]) for s in stems]
    per_tau: dict[float, list] = {t: [] for t in taus}
    with Pool(workers) as pool:
        for rows in pool.imap_unordered(_eval_one, jobs):
            for r in rows:
                per_tau[r["_tau"]].append(r)
    out = []
    for tau in taus:
        rows = per_tau[tau]
        _, macro = macro_summarise(rows)
        out.append(dict(arm=name, order=order, tau=tau,
                        cv=macro["score"], adj=macro["adj_edge_jaccard"],
                        divJ=macro["division_jaccard"],
                        tp=sum(r["division_tp"] for r in rows),
                        fp=sum(r["division_fp"] for r in rows),
                        fn=sum(r["division_fn"] for r in rows),
                        dropped=sum(r["_dropped"] for r in rows),
                        forks=sum(r["_forks"] for r in rows),
                        matched=sum(r["_matched"] for r in rows)))
    return out


def score_pool(path, out_path, device="cuda"):
    cmd = [sys.executable, "division_cls/score_forks.py",
           "--candidates", str(path), "--run_dir", RUN_DIR,
           "--out", str(out_path), "--device", device]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", action="store_true")
    p.add_argument("--eval", nargs="*", default=None)
    p.add_argument("--order", default="reprune", choices=ORDERS)
    p.add_argument("--arms", default=None, help="comma-separated subset")
    p.add_argument("--taus", default="0,0.5")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--out", default=None, help="append results as jsonl")
    p.add_argument("--preds_dir", default=None, help="override PREDS")
    p.add_argument("--embryo", default=None, help='"" for both embryos')
    p.add_argument("--work", default=None, help="override the cache/output dir")
    a = p.parse_args()

    global PREDS, EMBRYO, WORK
    if a.preds_dir:
        PREDS = a.preds_dir
    if a.embryo is not None:
        EMBRYO = a.embryo
    if a.work:
        WORK = Path(a.work)

    names = ([n.strip() for n in a.arms.split(",")] if a.arms
             else list(ARMS))
    taus = [float(t) for t in a.taus.split(",")]

    stems = _stems()
    build_prefix(stems, a.workers)

    if a.pool:
        print(f"pool stage, order={a.order}, {len(names)} arms\n", flush=True)
        recs = run_pool(names, a.order, a.workers)
        if a.out:
            with open(a.out, "a") as f:
                for r in recs.values():
                    f.write(json.dumps({"stage": "pool", **r}) + "\n")

    if a.eval is not None:
        for name in (a.eval or names):
            pool_path = WORK / f"forks_{name}_{a.order}.npz"
            score_path = WORK / f"scores_{name}_{a.order}.npz"
            if not score_path.exists():
                score_pool(pool_path, score_path)
            for r in run_eval(name, a.order, score_path, taus, a.workers):
                print(f"  {r['arm']:12s} {r['order']:9s} tau={r['tau']:<4} "
                      f"cv={r['cv']:.4f} adj={r['adj']:.4f} divJ={r['divJ']:.4f} "
                      f"tp={r['tp']:3d} fp={r['fp']:4d} fn={r['fn']:3d} "
                      f"dropped={r['dropped']}/{r['forks']}", flush=True)
                if a.out:
                    with open(a.out, "a") as f:
                        f.write(json.dumps({"stage": "eval", **r}) + "\n")


if __name__ == "__main__":
    main()
