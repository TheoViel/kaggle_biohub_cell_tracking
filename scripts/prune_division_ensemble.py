#!/usr/bin/env python
"""Greedy forward selection over division classifiers, scored on the real graph.

    python scripts/prune_division_ensemble.py --preds_dir /tmp/hA --max_size 5

Ten models rank-averaged reach div_J 0.2968 on 6bba, against 0.2332 for one.
That is not deployable -- it is ten forward passes over the fork crops at
inference -- so this finds the smallest subset that keeps the gain.

**Selection is on 6bba, not on the macro.** 6bba carries 125 of the 151
annotated divisions and 44b6 carries 26, so a macro optimum is decided by the
embryo where one division moves div_J by 0.03. Measured: the macro optimum of
the 6-model ensemble sits at tau 0.90 purely because of 44b6, while 6bba's own
curve peaks at 0.85 and is *falling* by 0.90. Selecting on macro here would fit
ten events. The macro is still reported, and a subset that wins on 6bba while
collapsing on 44b6 should be read as suspect rather than as a winner.

Why greedy rather than exhaustive: 2^10 subsets against ~40 s an evaluation is
a day of compute for a decision worth minutes. Greedy forward selection is
n(n+1)/2 evaluations and is the standard answer when the members are
individually weak and jointly useful, which is exactly the regime the
`merge_fork_scores` docstring describes.

What makes it affordable at all: every subset is evaluated against the *same*
per-stack fork enumeration and the same per-model ranks, both computed once per
stack. Only the mean-rank, the threshold and `prune_short` are redone per
candidate, so a step over ten candidates costs barely more than one.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "postprocessing"))

import edge_pp  # noqa: E402
import multi_hypothesis as M  # noqa: E402
from tracking_cellmot.metrics import summarise  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("sfg", REPO / "scripts" / "sweep_fork_geometry.py")
_sfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sfg)
forks_of, load_lookups, _key = _sfg.forks_of, _sfg.load_lookups, _sfg._key


def rank_within(v):
    """`v` replaced by its rank in [0, 1]; NaN preserved. Matches
    `merge_fork_scores.py --mode rank`, which ranks per stack because one
    threshold is applied everywhere."""
    out = np.full(len(v), np.nan)
    ok = np.isfinite(v)
    if ok.any():
        out[ok] = np.argsort(np.argsort(v[ok])) / max(ok.sum() - 1, 1)
    return out


def run_stack(arg):
    stem, pred_dir, per_model, subsets, taus, min_track = arg
    z = np.load(Path(pred_dir) / f"{stem}.npz")
    coords = np.asarray(
        z["coords_exact"] if "coords_exact" in z.files else z["coords"], float)
    edges = np.asarray(z["edges"], float).reshape(-1, 4)
    forks = forks_of(coords, edges)
    if not forks:
        return []
    keys = [f["key"] for f in forks]
    ks = np.array([f["k"] for f in forks])
    # One rank vector per model, computed once and reused by every subset.
    ranks = {name: rank_within(np.array([lk.get(k, np.nan) for k in keys]))
             for name, lk in per_model.items()}

    rows = []
    for subset in subsets:
        stack = np.vstack([ranks[n] for n in subset])
        score = np.nanmean(stack, axis=0)
        score[np.all(~np.isfinite(stack), axis=0)] = np.nan
        for tau in taus:
            drop = ks[np.isfinite(score) & (score < tau)]
            e = (edges[np.setdiff1d(np.arange(len(edges)), drop)]
                 if len(drop) else edges)
            c, e = edge_pp.prune_short(coords, e, min_track)
            r = M.score_one(stem, c, e)
            r["_subset"] = "+".join(subset)
            r["_tau"] = tau
            rows.append(r)
    return rows


def evaluate(subsets, taus, stems, pred_dir, per_model, min_track, jobs):
    # Per stack: `load_lookups` returns {stem: {key: score}}, so the worker
    # must get this stack's inner dict. Passing the outer one makes every
    # `.get(fork_key)` miss, every score NaN, and nothing is ever dropped --
    # which reads as a working sweep whose every subset scores identically.
    got = Parallel(n_jobs=jobs, backend="loky")(
        delayed(run_stack)((s, pred_dir,
                            {n: lk.get(s, {}) for n, lk in per_model.items()},
                            subsets, taus, min_track))
        for s in stems)
    by = {}
    for rows in got:
        for r in rows:
            by.setdefault((r["_subset"], r["_tau"]), []).append(r)
    out = {}
    if not by:
        raise SystemExit("no rows scored -- check --preds_dir and the score files")
    for key, rows in by.items():
        per = {e: summarise([r for r in rows if r["embryo"] == e])
               for e in sorted({r["embryo"] for r in rows})}
        out[key] = {
            "macro": float(np.mean([per[e]["division_jaccard"] for e in per])),
            **{e: per[e]["division_jaccard"] for e in per},
            **{f"{e}_tp": per[e]["division_tp"] for e in per},
            **{f"{e}_fp": per[e]["division_fp"] for e in per},
        }
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds_dir", default="/tmp/hA")
    p.add_argument("--scores", nargs="+", default=None,
                   help="name=path pairs; default is the ten scored today")
    p.add_argument("--taus", type=float, nargs="+", default=(0.85, 0.88, 0.90))
    p.add_argument("--max_size", type=int, default=5)
    p.add_argument("--select_on", default="macro",
                   help="what drives selection: 'macro' (the competition "
                        "metric, the default) or an embryo name. Both embryos "
                        "are reported either way -- 44b6 has 26 of the 151 "
                        "divisions and 6bba 125, which is a reason to distrust "
                        "a narrow macro win, not a reason to score on half the "
                        "data")
    p.add_argument("--min_track", type=int, default=5)
    p.add_argument("--jobs", type=int, default=24)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    R = str(REPO)
    if a.scores:
        pairs = [s.split("=", 1) for s in a.scores]
    else:
        pairs = [
            ("m_c48nf", f"{R}/logs/division_merged/c48nf"),
            ("m_f5", f"{R}/logs/division_merged/f5"),
            ("m_c48f5", f"{R}/logs/division_merged/c48f5"),
            ("d_16_0", f"{R}/logs/division/2026-09-16/0"),
            ("d_27_0", f"{R}/logs/division/2026-09-27/0"),
            ("d_27_1", f"{R}/logs/division/2026-09-27/1"),
            ("v4_base", f"{R}/logs/division_v4pool/base"),
            ("v4_c48nf", f"{R}/logs/division_v4pool/c48nf"),
            ("v4_f5", f"{R}/logs/division_v4pool/f5"),
            ("v4_c48f5", f"{R}/logs/division_v4pool/c48f5"),
        ]
        pairs = [(n, f"{d}/scores_forks_v4_harvest.npz") for n, d in pairs]

    per_model = {n: load_lookups(p_) for n, p_ in pairs}
    names = [n for n, _ in pairs]
    stems = sorted(q.stem for q in Path(a.preds_dir).glob("*.npz"))
    print(f"{len(names)} models, {len(stems)} stacks, taus {list(a.taus)}, "
          f"selecting on {a.select_on}\n", flush=True)

    chosen, history = [], []
    best_overall = None
    while len(chosen) < a.max_size:
        cands = [tuple(chosen + [n]) for n in names if n not in chosen]
        res = evaluate([list(c) for c in cands], a.taus, stems, a.preds_dir,
                       per_model, a.min_track, a.jobs)
        # Best (subset, tau) of this round, by the selection embryo.
        best = max(res.items(), key=lambda kv: kv[1][a.select_on])
        (subset, tau), v = best
        chosen = subset.split("+")
        history.append({"size": len(chosen), "subset": chosen, "tau": tau, **v})
        print(f"size {len(chosen)}: {'+'.join(chosen):<44} tau {tau}  "
              f"macro {v['macro']:.4f}   "
              f"44b6 {v.get('44b6', float('nan')):.4f} "
              f"({v.get('44b6_tp')}/{v.get('44b6_fp')})   "
              f"6bba {v.get('6bba', float('nan')):.4f} "
              f"({v.get('6bba_tp')}/{v.get('6bba_fp')})", flush=True)
        if best_overall is None or v[a.select_on] > best_overall[a.select_on]:
            best_overall = {"subset": chosen, "tau": tau, **v}

    b = best_overall
    print(f"\nbest: {'+'.join(b['subset'])} at tau {b['tau']}   macro {b['macro']:.4f}"
          f"   44b6 {b.get('44b6', float('nan')):.4f}   6bba {b.get('6bba', float('nan')):.4f}")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"history": history, "best": best_overall}, indent=2, default=float))
        print(f"-> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
