"""Full accounting of what division retrieval adds to the graph, and what it costs.

Three arms, all on the same nodes and the same ILP settings, so every
difference is the division mechanism and nothing else:

  no divisions    the linking graph alone -- `propose_divisions` off, no forks.
                  This is the "before retrieving divisions" reference for the
                  node and edge deltas.
  production      + `propose_divisions`, the tuned geometric proposer. The
                  current chain.
  gated forks     + the ILP's forks at `ilp_fork_division=0.3`, kept only where
                  the division classifier scores them >= tau.

Reported per embryo and macro. The columns that answer "what did it add":

  nodes / d_nodes   total predicted nodes, and the change from `no divisions`.
                    Divisions run *before* `prune_short`, so attaching a free
                    detection to a parent puts it in that parent's component
                    and a fragment that would have been pruned survives. That
                    is the only way this stage changes the node count, and it
                    feeds the metric's node-ratio penalty.
  edges / d_edges   total edges, and the change. A fork adds exactly one edge,
                    so d_edges is (forks kept) + (edges riding along on rescued
                    fragments).
  forks             parents with out-degree >= 2 in the final graph.

and the columns that say what it cost:

  J                 edge Jaccard, tp / (tp + fp + fn).
  ratio             (n_pred - n_est) / n_est, signed and uncapped.
  adj               J * (1 - 0.1 * ratio), which is 90% of the score.

Only forks landing on an annotated lineage are visible to the edge metric at
all, so `edge_fp` moves by far less than `d_edges` -- that gap is the point of
the table.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "postprocessing"))

import edge_pp  # noqa: E402
import main_edge_postprocess as MEP  # noqa: E402
import multi_hypothesis as M  # noqa: E402
from division_cls.gen_fork_candidates import FORK_CFG  # noqa: E402
from division_cls.tune_fork_threshold import _key, load_lookups  # noqa: E402
from tracking_cellmot.metrics import macro_summarise  # noqa: E402

torch.set_num_threads(1)

ILP = dict(ilp=1, ilp_disappearance=0.75, ilp_threshold=0.4)
TAUS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _postprocess(stem, pred_dir, over):
    z = np.load(Path(pred_dir) / f"{stem}.npz")
    cfg = dict(MEP.DEFAULTS)
    cfg.update(out_dir="", geff_dir="", no_score=True, assign=1)
    cfg.update(over)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        coords, edges = MEP.postprocess(
            z["coords"].astype(np.float64), np.empty((0, 4)),
            np.asarray(z["candidates"], float).reshape(-1, 5), cfg)
    return coords, np.asarray(edges, float).reshape(-1, 4)


def _row(stem, coords, edges):
    r = M.score_one(stem, coords, edges)
    outdeg = (np.bincount(edges[:, 0].astype(int), minlength=len(coords))
              if len(edges) else np.zeros(1))
    r["_nodes"] = int(len(coords))
    r["_edges"] = int(len(edges))
    r["_forks"] = int((outdeg >= 2).sum())
    return r


def job(arg):
    stem, pred_dir, lookup = arg
    out = {"_stem": stem}

    coords, edges = _postprocess(stem, pred_dir, {**ILP, "divisions": 0})
    out["no divisions"] = _row(stem, coords, edges)

    coords, edges = _postprocess(stem, pred_dir, ILP)
    out["production"] = _row(stem, coords, edges)

    # One fork solve, then each tau is a cheap edge deletion on the same graph.
    coords, edges = _postprocess(stem, pred_dir, FORK_CFG)
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
    for tau in TAUS:
        drop = [k for k, s in scored if np.isfinite(s) and s < tau]
        e = (edges[np.setdiff1d(np.arange(len(edges)), np.asarray(drop, int))]
             if drop else edges)
        out[f"gated forks tau={tau}"] = _row(stem, coords, e)

        # The same graph with the length filter re-run. `prune_short` fires
        # while *every* fork is still in the graph -- that ordering is
        # deliberate, so a real second daughter is not pruned as a fragment --
        # and gating then deletes edges afterwards, leaving behind every
        # fragment a rejected fork rescued. Hence gated and ungated report
        # identical node counts. Re-pruning is the missing half of the gate.
        if tau == REPRUNE_TAU:
            c2, e2 = edge_pp.prune_short(coords, e, MEP.DEFAULTS["min_track"])
            out[f"gated tau={tau} + reprune"] = _row(stem, c2, e2)

    out["ungated forks"] = _row(stem, coords, edges)
    return out


REPRUNE_TAU = 0.5
ARMS = (["no divisions", "production"]
        + [f"gated forks tau={t}" for t in TAUS]
        + [f"gated tau={REPRUNE_TAU} + reprune", "ungated forks"])


def table(rows_by_arm, ref="no divisions"):
    lines = []
    head = (f"| {'arm':<22} | {'cv':>7} | {'adj':>7} | {'J':>7} | {'ratio':>7} | "
            f"{'nodes':>9} | {'d_nodes':>8} | {'edges':>9} | {'d_edges':>8} | "
            f"{'forks':>7} | {'e_fp':>7} | {'d_e_fp':>7} | {'div_J':>6} | "
            f"{'tp':>4} | {'fp':>5} | {'fn':>4} |")
    lines.append(head)
    lines.append("|" + "|".join("-" * (len(c) + 2) for c in head.split("|")[1:-1]) + "|")
    r0 = rows_by_arm[ref]
    n0 = sum(r["_nodes"] for r in r0)
    e0 = sum(r["_edges"] for r in r0)
    fp0 = sum(r["edge_fp"] for r in r0)
    for arm in ARMS:
        rs = rows_by_arm.get(arm)
        if not rs:
            continue
        _, m = macro_summarise(rs)
        w = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in rs]
        tw = sum(w) or 1
        ratio = sum(wi * r["total_node_ratio"] for wi, r in zip(w, rs)) / tw
        n = sum(r["_nodes"] for r in rs)
        e = sum(r["_edges"] for r in rs)
        fp = sum(r["edge_fp"] for r in rs)
        lines.append(
            f"| {arm:<22} | {m['score']:>7.4f} | {m['adj_edge_jaccard']:>7.4f} | "
            f"{m['edge_jaccard']:>7.4f} | {ratio:>7.4f} | {n:>9d} | {n - n0:>+8d} | "
            f"{e:>9d} | {e - e0:>+8d} | {sum(r['_forks'] for r in rs):>7d} | "
            f"{fp:>7d} | {fp - fp0:>+7d} | {m['division_jaccard']:>6.4f} | "
            f"{sum(r.get('division_tp', 0) for r in rs):>4d} | "
            f"{sum(r.get('division_fp', 0) for r in rs):>5d} | "
            f"{sum(r.get('division_fn', 0) for r in rs):>4d} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preds_dir", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--workers", type=int, default=26)
    p.add_argument("--out", default=None, help="write the markdown report here")
    args = p.parse_args()

    pred_dir = Path(args.preds_dir)
    stems = sorted(q.stem for q in pred_dir.glob("*.npz"))
    lookups = load_lookups(args.scores)
    print(f"{len(stems)} stacks, {len(ARMS)} arms\n", flush=True)

    results = []
    with Pool(args.workers) as pool:
        jobs = [(s, str(pred_dir), lookups.get(s, {})) for s in stems]
        for i, out in enumerate(pool.imap_unordered(job, jobs), 1):
            results.append(out)
            if i % 20 == 0 or i == len(stems):
                print(f"  {i}/{len(stems)} stacks", flush=True)

    chunks = []
    for label, keep in (("macro (both embryos)", lambda s: True),
                        ("44b6 (fold 0)", lambda s: s.startswith("44b6")),
                        ("6bba (fold 1)", lambda s: s.startswith("6bba"))):
        sel = [o for o in results if keep(o["_stem"])]
        by_arm = {a: [o[a] for o in sel if a in o] for a in ARMS}
        chunks.append(f"\n### {label}  ({len(sel)} stacks)\n\n" + table(by_arm))
    report = ("# Division retrieval: before and after\n\n"
              "`no divisions` is the reference for `d_nodes` / `d_edges` / `d_e_fp`.\n"
              + "\n".join(chunks) + "\n")
    print(report, flush=True)
    if args.out:
        Path(args.out).write_text(report)
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
