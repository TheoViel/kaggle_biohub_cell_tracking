#!/usr/bin/env python
"""Pair every trained run with a fixed partner under the consensus fusion.

Consensus beat every append recipe on 13/1 + 24/5 (0.7764 against 0.7738 at
matched recall, 199 OOF samples). This asks whether some *other* run pairs with
24/5 better than 13/1 does.

Cluster both models' detections within ``--radius`` um, take the
confidence-weighted centroid, and score each cluster by the mean of the two
probabilities with a zero filled in for the model that did not fire. In
practice that is close to a strict intersection -- measured on 13/1 + 24/5,
exactly 1 of 5.56M nodes at th 0.40 came from a single model -- so the score
threshold is really "how confidently did both models have to agree".

WHY IT IS STRUCTURED THIS WAY.  The obvious loop is `for config: for sample:`,
one joblib call per configuration.  Do not do that.  Each joblib call respawns
workers, and every worker then re-reads the 199 ground-truth .geff files, so
the GT loading is repeated once per configuration and dominates everything --
that is what made the earlier sweeps take tens of minutes per grid point.  This
parallelises over *samples* and loops the configurations inside each task, so a
GT graph is read once per (worker, sample) no matter how many configurations
run.  The clustering is also done once per candidate and thresholded N times,
since the threshold applies to the fused score and not to the clustering.

Results are appended to a TSV as each batch finishes, so an interrupted run
still leaves usable numbers.

    python scripts/consensus_sweep.py --partner logs/2026-09-24/5 --jobs 48
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import ensemble_detectors as E  # noqa: E402

# Every run named in the Experiments tables that has predictions on disk.
# preds_r6_0.30 is preferred: the fusion clusters at 0.30, so a folder cut at
# 0.40 silently withholds that run's weaker detections and its pairing is not
# comparable to the rest. The two r6_0.40-only runs are still included, flagged,
# because 2026-09-15/1 tops the Experiments_2 table and excluding it on a
# technicality would be worse than reporting it with a caveat.
CANDIDATES = [
    "2026-09-13/1", "2026-09-13/0", "2026-09-13/2", "2026-09-11/1", "2026-09-11/3",
    "2026-09-12/0", "2026-09-12/1", "2026-09-14/0", "2026-09-14/4", "2026-09-14/5",
    "2026-09-14/7", "2026-09-16/0", "2026-09-17/0", "2026-09-17/3", "2026-09-17/4",
    "2026-09-17/6", "2026-09-18/0", "2026-09-22/10", "2026-09-23/6",
    "2026-09-15/0", "2026-09-15/1",
]


def tag_for(run: Path, preferred: str, fallback: str) -> str | None:
    if (run / f"preds_{preferred}").is_dir() and any(
            (run / f"preds_{preferred}").glob("*.npz")):
        return preferred
    if (run / f"preds_{fallback}").is_dir() and any(
            (run / f"preds_{fallback}").glob("*.npz")):
        return fallback
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--partner", default="logs/2026-09-24/5",
                   help="the run every candidate is fused with")
    p.add_argument("--radius", type=float, default=5.0)
    p.add_argument("--th", type=float, nargs="+", default=(0.40, 0.45))
    p.add_argument("--cluster-th", type=float, default=0.30)
    p.add_argument("--mode", choices=("consensus", "append"), default="consensus")
    p.add_argument("--th-base", type=float, nargs="+", default=(0.40, 0.45),
                   help="append mode: threshold on whichever run is the base")
    p.add_argument("--d-add", type=float, default=7.0,
                   help="append mode: keep an added node only this far from "
                        "every kept base node; must clear the NMS radius")
    p.add_argument("--both-directions", action="store_true",
                   help="append mode: also try the partner as the base")
    p.add_argument("--tag", default="r6_0.30")
    p.add_argument("--fallback-tag", default="r6_0.40")
    p.add_argument("--jobs", type=int, default=48)
    p.add_argument("--batch", type=int, default=6,
                   help="candidates per batch; a ranked table prints after each")
    p.add_argument("--max-disp", type=float, default=10.0)
    p.add_argument("--out", default=str(REPO / "output" / "consensus_sweep.tsv"))
    p.add_argument("--limit", type=int, default=None, help="samples (debugging)")
    args = p.parse_args()

    partner = Path(args.partner)
    partner = partner if partner.is_absolute() else REPO / partner
    p_tag = tag_for(partner, args.tag, args.fallback_tag)
    if p_tag is None:
        raise SystemExit(f"no predictions under {partner}")

    cands = []
    for c in CANDIDATES:
        run = REPO / "logs" / c
        if run == partner:
            continue
        t = tag_for(run, args.tag, args.fallback_tag)
        if t is None:
            print(f"!  {c}: no predictions, skipped")
            continue
        cands.append((c, run, t))

    names = sorted(q.stem for q in (partner / f"preds_{p_tag}").glob("*.npz"))
    if args.limit:
        names = names[: args.limit]

    print(f"partner   : {partner.relative_to(REPO)}  ({p_tag})")
    print(f"candidates: {len(cands)}")
    print(f"samples   : {len(names)}")
    print(f"fusion    : radius {args.radius} um, cluster_th {args.cluster_th}, "
          f"th {list(args.th)}")
    print(f"tsv       : {args.out}\n", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.write_text("when\tcandidate\ttag\tmode\tconfig\tnode_recall\tscore\t"
                       "edge_jaccard\tn_pred\tratio\n")

    results = {}
    for start in range(0, len(cands), args.batch):
        batch = cands[start : start + args.batch]
        print(f"--- batch {start // args.batch + 1}: "
              f"{', '.join(c for c, _, _ in batch)}", flush=True)

        def one(name):
            """Every configuration for one sample; GT is touched once here."""
            pz = np.load(partner / f"preds_{p_tag}" / f"{name}.npz")
            pc = pz["coords"].astype(float)
            pp = pz["node_probs"].astype(float)
            rows = {}
            for cname, run, t in batch:
                f = run / f"preds_{t}" / f"{name}.npz"
                if not f.exists():
                    continue
                cz = np.load(f)
                cc_, cp_ = (cz["coords"].astype(float),
                            cz["node_probs"].astype(float))
                if args.mode == "consensus":
                    # Cluster once; the threshold acts on the fused score, so
                    # the geometry does not change with it.
                    c, s, _, _, _ = E.consensus_clusters(
                        cc_, cp_, pc, pp, args.radius, args.cluster_th)
                    for th in args.th:
                        cc = c[s >= th]
                        if not len(cc):
                            continue
                        cc = cc[np.argsort(cc[:, 0], kind="stable")]
                        rows[(cname, f"th{th:.2f}")] = E.score_one(
                            name, cc, E.nn_edges(cc, args.max_disp, False))
                else:
                    # E.merge's signature is (add_coords, add_probs,
                    # base_coords, base_probs, th_add, th_base, d_add), so the
                    # direction is expressed by which pair goes first.
                    dirs = [("cand-base", cc_, cp_, pc, pp)]
                    if args.both_directions:
                        dirs.append(("part-base", pc, pp, cc_, cp_))
                    for dlabel, bc, bp, ac, ap in dirs:
                        for tb in args.th_base:
                            for ta in args.th:
                                cc = E.merge(ac, ap, bc, bp, ta, tb, args.d_add)
                                if not len(cc):
                                    continue
                                rows[(cname, f"{dlabel} b{tb:.2f} a{ta:.2f}")] = \
                                    E.score_one(name, cc,
                                                E.nn_edges(cc, args.max_disp, False))
            return name, rows

        got = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(one)(n) for n in names)

        by_cfg = defaultdict(dict)
        for name, rows in got:
            for k, r in rows.items():
                by_cfg[k][name] = r

        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        tags = {c: t for c, _, t in batch}
        with open(out, "a") as fh:
            for (cname, cfg), rows in sorted(by_cfg.items()):
                m, _ = E.macro(rows)
                results[(cname, cfg)] = m
                fh.write(f"{stamp}\t{cname}\t{tags[cname]}\t{args.mode}\t"
                         f"{cfg}\t{m['node_recall']:.4f}\t{m['score']:.4f}\t"
                         f"{m['edge_jaccard']:.4f}\t{m['n_pred']:.0f}\t"
                         f"{m['ratio']:.3f}\n")
        report(results, args)

    print("\nDONE")
    report(results, args, full=True)


def report(results, args, full=False):
    """Ranked by node recall, not by `score`.

    `score` here carries a node-count penalty and uses nearest-neighbour edges,
    and both of those reward a sparse node set. The real pipeline hands these
    nodes to a trained edge model that can discard a spurious node but can
    never recover a missing one, so recall is the property it actually
    consumes. Measured end to end, this surrogate ranked consensus above append
    twice while the pipeline preferred append -- read the score column as a
    sanity check on over-detection, not as the objective.
    """
    if not results:
        return
    print(f"\n{'candidate':<15}{'config':<22}{'node_rec':>10}{'score':>9}"
          f"{'edge_jac':>10}{'ratio':>8}")
    print("-" * 74)
    ranked = sorted(results.items(), key=lambda kv: -kv[1]["node_recall"])
    for (c, cfg), m in ranked[: None if full else 12]:
        print(f"{c:<15}{str(cfg):<22}{m['node_recall']:>10.4f}{m['score']:>9.4f}"
              f"{m['edge_jaccard']:>10.4f}{m['ratio']:>8.3f}")
    b = ranked[0]
    print(f"\nbest node_recall: {b[0][0]} {b[0][1]}  {b[1]['node_recall']:.4f}"
          f"   (v1 append reference: node_rec 0.9687, score 0.7738)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
