"""What happens to every ground-truth link, for the edge transformer's graph.

    cd src/postprocessing
    python diag_edge.py --preds_dir /abs/path/preds_edge [--chain assign]
    python diag_edge.py --preds_dir /abs/path/preds_edge \
           --chain '[recover=1,recover_max_gap=5,min_track=4,linefit=0.6,linefit_window=3]'

Every ground-truth edge lands in exactly one bucket, and the bucket names the
stage that could win it back:

  tp        reproduced.
  lost      one of its endpoints has no detection within the 7 um matching
            radius -- nothing that only decides links can recover it, only
            `recover_nodes` (or a better detector) can.
  declined  both endpoints exist and are matched, and the linker left one of
            them free -- in practice a downstream symptom of a missing node,
            not a threshold that is set too high; see README_edge.md.
  stolen    both endpoints exist and the linker gave one of them to somebody
            else -- a ranking error, which is what the assignment is for.

This is `analysis_worst.py`'s decomposition, run against the candidate graph so
it can also report the *ceiling*: whether the model scored the right pair at
all, and at what rank. A link whose true parent is not among the candidates is
out of post-processing's reach entirely. That half is only reported for a chain
that leaves the node indices alone, since the candidate rows index them.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: F401,E402
import edge_pp  # noqa: E402
import fast_score  # noqa: E402
from exp_edge_pp import _gt, resolve  # noqa: E402


def _one(job):
    path, chain = job
    sample = Path(path).stem
    coords0, edges, cand = edge_pp.load_pred(path)
    gt = _gt(sample)
    coords, edges = resolve(chain)(coords0.copy(), edges, cand)
    # Round like `submission.py` does, so the node matching here is the one the
    # score actually sees -- see README_edge.md, "Scoring caveat".
    coords = np.round(np.asarray(coords, float))
    match = fast_score.match_nodes(coords, gt)
    inv = {}
    for row, node in match.items():
        inv[node] = row

    pred = set(map(tuple, np.asarray(edges)[:, :2].astype(int))) if len(edges) else set()
    has_child = {int(a) for a, _ in pred}
    has_parent = {int(b) for _, b in pred}

    # Candidate lookup, for the ceiling: does the model even propose the pair,
    # and where does it rank among the candidates leaving that parent?
    #
    # Candidate rows index the *original* coords, so this is only meaningful
    # for a chain that leaves those indices alone. `recover` (which appends)
    # and `linefit` (which moves nodes) do; `prune_short` renumbers, and asking
    # anyway would report every true pair as absent.
    keeps_index = (len(coords) >= len(coords0)
                   and np.array_equal(coords[:len(coords0), 0], coords0[:, 0]))
    cand = np.asarray(cand).reshape(-1, 5)
    by_src = {}
    if keeps_index:
        for a, b, p, _d, _dt in cand:
            by_src.setdefault(int(a), []).append((p, int(b)))
        for a in by_src:
            by_src[a].sort(reverse=True)

    counts = Counter()
    ranks = Counter()
    for ga, gb in gt.pairs:
        ra, rb = inv.get(ga), inv.get(gb)
        if ra is None or rb is None:
            counts["lost"] += 1
            continue
        if (ra, rb) in pred:
            counts["tp"] += 1
        elif ra not in has_child or rb not in has_parent:
            counts["declined"] += 1
        else:
            counts["stolen"] += 1
        # the ceiling, for every link whose endpoints both exist
        if keeps_index:
            cands = by_src.get(ra, [])
            rank = next((i for i, (_p, b) in enumerate(cands) if b == rb), None)
            ranks["absent" if rank is None else f"rank{min(rank, 4)}"] += 1
    return sample, counts, ranks, len(gt.pairs)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds_dir", required=True)
    p.add_argument("--chain", default="assign",
                   help="a name from exp_edge_pp.CHAINS, or an inline [k=v,...] spec")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--jobs", type=int, default=24)
    args = p.parse_args()

    paths = sorted(Path(args.preds_dir).glob("*.npz"))[: args.limit]
    total, rank_total, n_gt = Counter(), Counter(), 0
    with Pool(args.jobs) as pool:
        for _sample, counts, ranks, n in pool.imap_unordered(
            _one, [(str(q), args.chain) for q in paths]
        ):
            total.update(counts)
            rank_total.update(ranks)
            n_gt += n

    print(f"\n{len(paths)} samples, chain '{args.chain}', {n_gt} ground-truth links\n")
    for key in ("tp", "stolen", "declined", "lost"):
        print(f"  {key:<10}{total[key]:>7}{total[key] / max(n_gt, 1):>9.1%}")
    seen = sum(rank_total.values())
    if not seen:
        print("\n  (candidate ranks not reported: this chain renumbers the "
              "nodes, so the candidate rows no longer index them)")
        return
    print("\n  where the true pair sits among the parent's candidates "
          "(links whose endpoints both exist):")
    for key in ("rank0", "rank1", "rank2", "rank3", "rank4", "absent"):
        if rank_total[key]:
            print(f"  {key:<10}{rank_total[key]:>7}{rank_total[key] / max(seen, 1):>9.1%}")


if __name__ == "__main__":
    main()
