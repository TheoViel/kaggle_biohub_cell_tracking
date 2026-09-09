"""Why each ground-truth division is not proposed: the first gate that rejects it.

    cd src/postprocessing
    python diag_divisions.py --cache_dir ../output/pp_experiments/div_cache/preds_0.50

`exp_divisions.py` says how a gate setting scores. This says *where the recall
goes*: for every annotated division it locates the predicted parent and both
daughters, then walks `propose_divisions`' gates in their real order against the
pair that ought to have been proposed and reports the first one that says no.

Reads the same cached base chain the sweep does -- the graph after `assign` and
`recover`, which is exactly what the stage is handed -- so a node `recover` put
back counts as available here, as it does there.

The buckets are ordered, so a division is attributed to the *earliest* reason it
failed and the counts sum to the total. Anything reaching `passes_all_gates` is
recall the caps or the ranking then threw away.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: F401,E402
import edge_pp  # noqa: E402
import tracksdata as td  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from params import SCALE, TRAIN_DIR  # noqa: E402

warnings.filterwarnings("ignore")
SC = np.asarray(SCALE)

# In the order propose_divisions applies them.
ORDER = [
    "parent not detected",
    "a daughter not detected",
    "parent has no single committed child",
    "committed child is neither daughter",
    "second daughter already has a parent",
    "pair absent from the candidate set",
    "prob < min_prob",
    "parent distance > max_parent_um",
    "sister distance > max_sister_um",
    "not the mutual nearest neighbour",
    "divergence: a grandchild is missing",
    "divergence: sisters do not separate enough",
    "symmetry",
    "passes all gates",
]


def _match(gt_pos, rows, pos, radius=7.0):
    """Assign GT points to distinct prediction rows within `radius`, or None."""
    if len(rows) == 0:
        return [None] * len(gt_pos)
    tree = cKDTree(pos[rows])
    near = sorted({j for g in gt_pos for j in tree.query_ball_point(g, radius)})
    if not near:
        return [None] * len(gt_pos)
    cost = np.array([[np.linalg.norm(g - pos[rows[j]]) for j in near] for g in gt_pos])
    cost = np.where(cost <= radius, cost, 1e6)
    ri, ci = linear_sum_assignment(cost)
    out = [None] * len(gt_pos)
    for r, c in zip(ri, ci):
        if cost[r, c] < 1e5:
            out[r] = int(rows[near[c]])
    return out


def diagnose(job):
    path, cfg = job
    stem = Path(path).stem
    with np.load(path) as d:
        coords, edges, cand = d["coords"], d["edges"], d["candidates"]

    gt = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{stem}.geff")
    gt = gt[0] if isinstance(gt, tuple) else gt
    nodes = {int(r["node_id"]): r for r in gt.node_attrs().iter_rows(named=True)}
    kids: dict[int, list[int]] = {}
    for e in gt.edge_attrs().iter_rows(named=True):
        kids.setdefault(int(e["source_id"]), []).append(int(e["target_id"]))
    divs = {k: v for k, v in kids.items() if len(v) >= 2}
    if not divs:
        return Counter(), []

    pos = coords[:, 1:] * SC
    frame_of = coords[:, 0].astype(int)
    by_frame = {t: np.flatnonzero(frame_of == t) for t in np.unique(frame_of)}
    src, tgt = edges[:, 0].astype(int), edges[:, 1].astype(int)
    n_children = np.bincount(src, minlength=len(coords))
    only_child = np.full(len(coords), -1, np.int64)
    single = n_children[src] == 1
    only_child[src[single]] = tgt[single]
    has_parent = np.zeros(len(coords), bool)
    has_parent[tgt] = True
    prob_of = {}
    c1s = cand[cand[:, 4] == 1] if len(cand) else cand
    for s, t_, pr in zip(c1s[:, 0].astype(int), c1s[:, 1].astype(int), c1s[:, 2]):
        prob_of[(s, t_)] = pr

    gpos = lambda n: np.array([nodes[n]["z"], nodes[n]["y"], nodes[n]["x"]], float) * SC

    counts, margins = Counter(), []
    for par, ch in divs.items():
        t = int(nodes[par]["t"])
        rp = _match([gpos(par)], by_frame.get(t, np.array([], int)), pos)[0]
        if rp is None:
            counts["parent not detected"] += 1
            continue
        d1, d2 = ch[0], ch[1]
        rows_next = by_frame.get(t + 1, np.array([], int))
        m = _match([gpos(d1), gpos(d2)], rows_next, pos)
        if m[0] is None or m[1] is None:
            counts["a daughter not detected"] += 1
            continue
        committed = int(only_child[rp])
        if committed < 0:
            counts["parent has no single committed child"] += 1
            continue
        if committed == m[0]:
            child, other = m[0], m[1]
        elif committed == m[1]:
            child, other = m[1], m[0]
        else:
            counts["committed child is neither daughter"] += 1
            continue
        if has_parent[other]:
            counts["second daughter already has a parent"] += 1
            continue
        if (rp, other) not in prob_of:
            counts["pair absent from the candidate set"] += 1
            continue
        prob = prob_of[(rp, other)]
        if prob < cfg["min_prob"]:
            counts["prob < min_prob"] += 1
            continue
        parent_dist = float(np.linalg.norm(pos[rp] - pos[other]))
        sister_dist = float(np.linalg.norm(pos[child] - pos[other]))
        child_dist = float(np.linalg.norm(pos[rp] - pos[child]))
        if parent_dist > cfg["max_parent_um"]:
            counts["parent distance > max_parent_um"] += 1
            continue
        if sister_dist > cfg["max_sister_um"]:
            counts["sister distance > max_sister_um"] += 1
            continue
        free = [int(r) for r in rows_next if not has_parent[r]]
        if free:
            _, idx = cKDTree(pos[free]).query(pos[child])
            if free[int(idx)] != other:
                counts["not the mutual nearest neighbour"] += 1
                continue
        g1, g2 = only_child[child], only_child[other]
        if g1 < 0 or g2 < 0 or frame_of[g1] != t + 2 or frame_of[g2] != t + 2:
            counts["divergence: a grandchild is missing"] += 1
            continue
        gap = float(np.linalg.norm(pos[g1] - pos[g2])) - sister_dist
        margins.append((gap, sister_dist, float(np.linalg.norm(pos[g1] - pos[g2]))))
        if gap < cfg["diverge_um"]:
            counts["divergence: sisters do not separate enough"] += 1
            continue
        denom = max((child_dist + parent_dist) / 2.0, 1e-6)
        if abs(child_dist - parent_dist) / denom > cfg["symmetry_tau"]:
            counts["symmetry"] += 1
            continue
        counts["passes all gates"] += 1
    return counts, margins


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_dir", default="../output/pp_experiments/div_cache/preds_0.50")
    p.add_argument("--min_prob", type=float, default=0.40)
    p.add_argument("--max_parent_um", type=float, default=9.0)
    p.add_argument("--max_sister_um", type=float, default=14.0)
    p.add_argument("--diverge_um", type=float, default=3.0)
    p.add_argument("--symmetry_tau", type=float, default=0.6)
    p.add_argument("--jobs", type=int, default=24)
    args = p.parse_args()

    cfg = dict(min_prob=args.min_prob, max_parent_um=args.max_parent_um,
               max_sister_um=args.max_sister_um, diverge_um=args.diverge_um,
               symmetry_tau=args.symmetry_tau)
    paths = sorted(Path(args.cache_dir).glob("*.npz"))
    if not paths:
        raise SystemExit(f"no cache under {args.cache_dir}; run exp_divisions.py first")

    total, all_margins = Counter(), []
    per_embryo: dict[str, Counter] = {}
    with Pool(args.jobs) as pool:
        for path, (counts, margins) in zip(paths, pool.map(
                diagnose, [(str(q), cfg) for q in paths], chunksize=1)):
            emb = path.stem.split("_")[0]
            per_embryo.setdefault(emb, Counter()).update(counts)
            total.update(counts)
            all_margins.extend(margins)

    n = sum(total.values())
    print(f"\n{'=' * 74}\nWHERE THE {n} ANNOTATED DIVISIONS ARE LOST"
          f"   (min_prob={cfg['min_prob']:g}, diverge={cfg['diverge_um']:g})\n{'=' * 74}")
    embs = sorted(per_embryo)
    print(f"{'first gate that rejects it':<46s}" + "".join(f"{e:>9s}" for e in embs) + f"{'all':>8s}")
    for key in ORDER:
        if not total.get(key):
            continue
        cells = "".join(f"{per_embryo[e].get(key, 0):>9d}" for e in embs)
        print(f"{key:<46s}{cells}{total[key]:>8d}")

    if all_margins:
        gaps = np.array([m[0] for m in all_margins])
        print(f"\nOf the {len(gaps)} that reach the divergence test, the separation gained\n"
              f"by t+2 (grandchild distance - sister distance), in um:")
        for q in (5, 10, 25, 50, 75, 90):
            print(f"    p{q:<3d} {np.percentile(gaps, q):7.2f}")
        for thr in (0.0, 1.0, 2.25, 3.0, 4.0, 5.0):
            print(f"    would pass at diverge_um={thr:<5g}: {(gaps >= thr).sum():3d}"
                  f" of {len(gaps)}  ({(gaps >= thr).mean():.0%})")


if __name__ == "__main__":
    main()
