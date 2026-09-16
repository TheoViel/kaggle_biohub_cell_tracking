"""Drop the forks the classifier disbelieves, and see whether the score moves.

AUC on held-out forks says the model ranks them. It does not say the pipeline
gains, because the metric does not score forks -- it scores
``adj_edge_jaccard + 0.1 * division_jaccard`` over a graph, and a dropped fork
also removes an edge. This is the step that settles it.

For each threshold: rebuild the graph with the same configuration that produced
the candidates, delete the second-child edge of every fork whose parent scores
below the threshold, and evaluate. ``tau=0`` keeps everything and must reproduce
the "ILP forks + PP divisions" arm exactly; ``tau=1`` drops every fork and must
land near the no-divisions chain. Both ends are printed as checks, because a
sweep whose endpoints are wrong is measuring something other than the threshold.

Forks are matched to their scores by value -- parent ``(t, z, y, x)`` and the
child's edge probability -- rather than by row order, so the sweep does not
depend on the solver enumerating the graph the same way twice.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "postprocessing"))

import main_edge_postprocess as MEP  # noqa: E402
import multi_hypothesis as M  # noqa: E402
from division_cls.gen_fork_candidates import FORK_CFG  # noqa: E402
from tracking_cellmot.metrics import macro_summarise  # noqa: E402

torch.set_num_threads(1)

def _key(t, z, y, x, prob):
    """Identity of one fork: its parent, plus the probability of the edge to the
    second child, which is what distinguishes the rows of a multi-way split."""
    return (int(t), round(float(z), 3), round(float(y), 3), round(float(x), 3),
            round(float(prob), 6))


def load_lookups(score_path):
    """`{stem: {fork key: score}}`.

    Passed to the workers inside each job rather than held in a module global
    populated by a `Pool` initializer. The initializer version silently left
    every worker's table empty when this file was run as a script -- while
    working when imported -- and a lookup that misses is indistinguishable from
    a fork the classifier likes, so the sweep reported "0 dropped" at every
    threshold instead of failing. No shared state, no way for that to recur.
    """
    blob = np.load(score_path, allow_pickle=True)
    stems = np.asarray([str(s) for s in blob["stem"]])
    coords, feats, score = blob["coords"], blob["features"], blob["score"]
    out: dict[str, dict] = {}
    for stem in sorted(set(stems.tolist())):
        m = stems == stem
        out[stem] = {_key(c[0], c[1], c[2], c[3], f[2]): float(s)
                     for c, f, s in zip(coords[m], feats[m], score[m])}
    return out


def _job(arg):
    stem, pred_dir, taus, lookup = arg
    z = np.load(Path(pred_dir) / f"{stem}.npz")
    cfg = dict(MEP.DEFAULTS)
    cfg.update(out_dir="", geff_dir="", no_score=True, assign=1)
    cfg.update(FORK_CFG)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        coords, edges = MEP.postprocess(
            z["coords"].astype(np.float64), np.empty((0, 4)),
            np.asarray(z["candidates"], float).reshape(-1, 5), cfg)
    edges = np.asarray(edges, float).reshape(-1, 4)
    src = edges[:, 0].astype(int)
    children: dict[int, list[int]] = {}
    for k in np.argsort(-edges[:, 2], kind="stable"):
        children.setdefault(int(src[k]), []).append(int(k))

    # (edge index, classifier score) for every second-child edge.
    scored = []
    for parent, ks in children.items():
        if len(ks) < 2:
            continue
        c = coords[parent]
        for k in ks[1:]:
            scored.append((int(k), lookup.get(
                _key(c[0], c[1], c[2], c[3], edges[k, 2]), np.nan)))

    out = []
    for tau in taus:
        if scored:
            # An unscored fork is kept: absence of a score is not disbelief,
            # and dropping them would confound the threshold with coverage.
            drop = [k for k, s in scored if np.isfinite(s) and s < tau]
        else:
            drop = []
        e = edges[np.setdiff1d(np.arange(len(edges)), np.asarray(drop, int))] if drop else edges
        row = M.score_one(stem, coords, e)
        row["_tau"] = tau
        row["_dropped"] = len(drop)
        row["_forks"] = len(scored)
        row["_matched"] = sum(1 for _, s in scored if np.isfinite(s))
        out.append(row)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preds_dir", required=True)
    p.add_argument("--scores", required=True, help="score_forks .npz")
    p.add_argument("--embryo", default=None)
    p.add_argument("--workers", type=int, default=28)
    p.add_argument("--taus", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    args = p.parse_args()

    taus = [float(t) for t in args.taus.split(",")]
    pred_dir = Path(args.preds_dir)
    stems = sorted(q.stem for q in pred_dir.glob("*.npz")
                   if args.embryo is None or q.stem.startswith(args.embryo))
    print(f"{len(stems)} stacks, {len(taus)} thresholds\n", flush=True)

    lookups = load_lookups(args.scores)
    missing = [s for s in stems if not lookups.get(s)]
    if missing:
        print(f"WARNING: no scores for {len(missing)} stacks, e.g. {missing[:3]}",
              flush=True)
    jobs = [(s, str(pred_dir), taus, lookups.get(s, {})) for s in stems]
    per_tau: dict[float, list] = {t: [] for t in taus}
    with Pool(args.workers) as pool:
        for rows in pool.imap_unordered(_job, jobs):
            for r in rows:
                per_tau[r["_tau"]].append(r)

    # A sweep that drops nothing is the failure mode this file had; say so
    # rather than printing a flat table that looks like a real null result.
    matched = sum(r.get("_matched", 0) for r in per_tau[taus[0]])
    total = sum(r["_forks"] for r in per_tau[taus[0]])
    print(f"{matched}/{total} forks matched a score\n", flush=True)
    if total and not matched:
        raise SystemExit("no fork matched a score -- the sweep would be vacuous")

    print(f"{'tau':>6}{'cv':>9}{'delta':>9}{'adj':>9}{'div_jac':>9}"
          f"{'dropped':>9}{'tp':>5}{'fp':>5}{'fn':>5}")
    ref = None
    for tau in taus:
        rows = per_tau[tau]
        _, m = macro_summarise(rows)
        cv = m["score"]
        if ref is None:
            ref = cv
        print(f"{tau:>6.2f}{cv:>9.4f}{cv-ref:>+9.4f}{m['adj_edge_jaccard']:>9.4f}"
              f"{m['division_jaccard']:>9.4f}"
              f"{sum(r['_dropped'] for r in rows):>9}"
              f"{sum(r.get('division_tp', 0) for r in rows):>5}"
              f"{sum(r.get('division_fp', 0) for r in rows):>5}"
              f"{sum(r.get('division_fn', 0) for r in rows):>5}", flush=True)


if __name__ == "__main__":
    main()
