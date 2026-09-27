#!/usr/bin/env python
"""Filter the classifier-gated forks on geometry, and sweep how hard to filter.

    python scripts/sweep_fork_geometry.py \
        --preds_dir /tmp/hA --scores logs/fork_cls/scores_hA.npz --jobs 24

The fork-gate arm leaves 6 division false positives on 44b6 and 71 on 6bba at
`div_J 0.2398`. `division_jaccard` is `tp / (tp + fp + fn)`, so an FP removed
without a TP is worth as much as a TP found, and the geometry the harvest threw
away is the obvious place to look for them: `FORK_CFG` opens both caps and turns
`require_divergence` off precisely so the classifier sees the proposals rather
than a pre-filter's survivors.

**This applies the gates *after* the classifier, not before.** That distinction
is the whole point. Running them before -- which is what the tuned
`div_min_prob 0.325` / `div_symmetry_tau 0.525` do -- prunes the pool the
classifier is meant to judge and was measured at 0.8842 against 0.8860, taking
6bba from 45 true divisions to 27. After the gate they can only remove forks the
classifier already kept, so a relaxed setting is free to reject nothing.

Why this is cheap enough to sweep. The harvest graph is read from
`--preds_dir` rather than recomputed, and every gate here is a function of the
candidates table, so one pass over the stacks serves every configuration: per
stack the forks are enumerated once, their geometry computed once, and each
configuration is then a set difference plus `prune_short` plus the scorer.
Parallel over stacks with the configurations looped inside, for the reason
`consensus_sweep.py` documents at length -- one joblib call per configuration
re-reads every ground-truth `.geff` per worker per configuration.

The gates, all replicating `edge_pp.propose_divisions` exactly:

* ``symmetry_tau`` -- ``|d1 - d2| / mean(d1, d2)``, the parent sitting between
  its daughters rather than beside one. 0 disables it, as in the chain.
* ``max_parent_um`` -- ``max(d1, d2)``, matching the ILP fork path's parent
  test rather than the second-child distance.
* ``max_sister_um`` / ``min_sister_um`` -- how far apart the daughters are. The
  minimum has no counterpart in the chain and is here on suspicion: a "division"
  whose daughters sit on top of each other is the shape a duplicate detection
  makes, and duplicates are what this detector over-produces.

`tau=0.5` reproduces the ungated-geometry arm and is printed first as the
reference every other row is a delta against.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "postprocessing"))

import edge_pp  # noqa: E402
import main_edge_postprocess as MEP  # noqa: E402
import multi_hypothesis as M  # noqa: E402
from division_cls.tune_fork_threshold import _key  # noqa: E402
from params import SCALE  # noqa: E402
from tracking_cellmot.metrics import macro_summarise, summarise  # noqa: E402


def load_lookups(score_path):
    """`{stem: {fork key: score}}` -- the same keying `apply_fork_gate` uses."""
    blob = np.load(score_path, allow_pickle=True)
    stems = np.asarray([str(s) for s in blob["stem"]])
    coords, feats, score = blob["coords"], blob["features"], blob["score"]
    out: dict[str, dict] = {}
    for stem in sorted(set(stems.tolist())):
        m = stems == stem
        out[stem] = {_key(c[0], c[1], c[2], c[3], f[2]): float(s)
                     for c, f, s in zip(coords[m], feats[m], score[m])}
    return out


def forks_of(coords, edges):
    """Every second-child edge, with the geometry the gates test.

    Enumerated exactly as `apply_fork_gate.gate_stack` does -- children sorted
    by descending edge probability, `ks[1:]` are the second children -- so a
    fork here is the same object the classifier scored.
    """
    src = edges[:, 0].astype(int)
    children: dict[int, list[int]] = {}
    for k in np.argsort(-edges[:, 2], kind="stable"):
        children.setdefault(int(src[k]), []).append(int(k))
    scale = np.asarray(SCALE, float)
    out = []
    for parent, ks in children.items():
        if len(ks) < 2:
            continue
        p = coords[parent]
        first = int(edges[ks[0], 1])
        for k in ks[1:]:
            second = int(edges[k, 1])
            # zyx only: t is the frame index and is not a distance.
            d1 = float(np.linalg.norm((coords[first][1:] - p[1:]) * scale))
            d2 = float(np.linalg.norm((coords[second][1:] - p[1:]) * scale))
            sis = float(np.linalg.norm(
                (coords[first][1:] - coords[second][1:]) * scale))
            denom = max((d1 + d2) / 2.0, 1e-6)
            out.append({
                "k": int(k),
                "key": _key(p[0], p[1], p[2], p[3], edges[k, 2]),
                "sym": abs(d1 - d2) / denom,
                "parent_um": max(d1, d2),
                "sister_um": sis,
            })
    return out


def run_stack(arg):
    stem, pred_dir, lookup, cfgs, min_track = arg
    z = np.load(Path(pred_dir) / f"{stem}.npz")
    coords = np.asarray(
        z["coords_exact"] if "coords_exact" in z.files else z["coords"], float)
    edges = np.asarray(z["edges"], float).reshape(-1, 4)
    forks = forks_of(coords, edges)
    for f in forks:
        s = lookup.get(f["key"], np.nan)
        f["score"] = float(s)

    rows = []
    for cfg in cfgs:
        drop = []
        for f in forks:
            # An unscored fork is kept: absence of a score is not disbelief.
            if np.isfinite(f["score"]) and f["score"] < cfg["tau"]:
                drop.append(f["k"]); continue
            if cfg["symmetry_tau"] > 0 and f["sym"] > cfg["symmetry_tau"]:
                drop.append(f["k"]); continue
            if f["parent_um"] > cfg["max_parent_um"]:
                drop.append(f["k"]); continue
            if not (cfg["min_sister_um"] <= f["sister_um"] <= cfg["max_sister_um"]):
                drop.append(f["k"]); continue
        e = (edges[np.setdiff1d(np.arange(len(edges)), np.asarray(drop, int))]
             if drop else edges)
        c, e = edge_pp.prune_short(coords, e, min_track)
        r = M.score_one(stem, c, e)
        r["_cfg"] = cfg["label"]
        r["_dropped"] = len(drop)
        rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds_dir", required=True,
                   help="the harvest post-processing output (pre-gate)")
    p.add_argument("--scores", required=True, help="score_forks .npz")
    p.add_argument("--tau", type=float, nargs="+", default=(0.5,))
    p.add_argument("--symmetry_tau", type=float, nargs="+",
                   default=(0.0, 0.7, 0.8, 0.9))
    p.add_argument("--max_parent_um", type=float, nargs="+", default=(11.0, 14.0, 99.0))
    p.add_argument("--max_sister_um", type=float, nargs="+", default=(14.0, 18.0, 99.0))
    p.add_argument("--min_sister_um", type=float, nargs="+", default=(0.0, 2.0, 3.0))
    p.add_argument("--min_track", type=int, default=MEP.DEFAULTS["min_track"])
    p.add_argument("--jobs", type=int, default=24)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None, help="write the ranked table here")
    a = p.parse_args()

    stems = sorted(q.stem for q in Path(a.preds_dir).glob("*.npz"))[: a.limit]
    lookups = load_lookups(a.scores)

    cfgs = []
    for tau, sym, mp, xs, ns in itertools.product(
            a.tau, a.symmetry_tau, a.max_parent_um, a.max_sister_um, a.min_sister_um):
        if ns >= xs:
            continue
        cfgs.append({
            "tau": tau, "symmetry_tau": sym, "max_parent_um": mp,
            "max_sister_um": xs, "min_sister_um": ns,
            "label": f"tau{tau:g} sym{sym:g} par{mp:g} sis{ns:g}-{xs:g}",
        })
    print(f"{len(stems)} stacks x {len(cfgs)} configs, min_track={a.min_track}\n",
          flush=True)

    got = Parallel(n_jobs=a.jobs, backend="loky")(
        delayed(run_stack)((s, a.preds_dir, lookups.get(s, {}), cfgs, a.min_track))
        for s in stems)

    by = {}
    for rows in got:
        for r in rows:
            by.setdefault(r["_cfg"], []).append(r)

    res = {}
    for label, rows in by.items():
        _, macro = macro_summarise(rows)
        per = {e: summarise([r for r in rows if r["embryo"] == e])
               for e in sorted({r["embryo"] for r in rows})}
        res[label] = (macro, per, sum(r["_dropped"] for r in rows))

    # The identity row, matched whole. `startswith` picked up "par99 sis0-18"
    # and quietly made every delta a comparison against a filtered arm.
    base = next((c["label"] for c in cfgs
                 if c["symmetry_tau"] == 0 and c["max_parent_um"] >= 99
                 and c["max_sister_um"] >= 99 and c["min_sister_um"] == 0
                 and c["tau"] == a.tau[0]), None)
    ref = res[base][0]["score"] if base else None
    hdr = (f"{'config':<34}{'cv':>9}{'delta':>9}{'adj':>9}{'div_J':>8}"
           f"{'tp':>5}{'fp':>5}{'fn':>5}{'dropped':>9}")
    lines = [hdr, "-" * len(hdr)]
    for label in sorted(res, key=lambda l: -res[l][0]["score"]):
        m, per, dropped = res[label]
        tp = sum(per[e]["division_tp"] for e in per)
        fp = sum(per[e]["division_fp"] for e in per)
        fn = sum(per[e]["division_fn"] for e in per)
        d = f"{m['score'] - ref:+.4f}" if ref is not None else ""
        lines.append(f"{label:<34}{m['score']:>9.4f}{d:>9}{m['adj_edge_jaccard']:>9.4f}"
                     f"{m['division_jaccard']:>8.4f}{tp:>5}{fp:>5}{fn:>5}{dropped:>9}")
    out = "\n".join(lines)
    print(out)
    if a.out:
        Path(a.out).write_text(out + "\n")
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
