"""Build a labelled candidate set for division proposals, for a classifier.

    cd src/postprocessing
    python div_dataset.py --cache_dir ../output/pp_experiments/div_cache/preds_0.50

`propose_divisions` decides with a conjunction of hand-set thresholds, and
`diag_divisions.py` shows why that is the wrong shape: no single gate is doing
the work, several reject a handful of true divisions each, and tightening any
one of them trades true positives for false ones at a poor rate. The signal is a
*combination* of weak features, which is a classifier's job.

This writes the training data for one: every pair the loose proposer would
consider, with the features that might separate them and a label from the
annotation.

**Candidates.** A parent holding exactly one committed child, and a *free* node
in the next frame. The free requirement is not a tunable -- a second parent is
an invalid lineage -- and it is also this set's recall ceiling, since a division
whose second daughter the assignment gave to somebody else cannot appear here at
all. `--report` prints that ceiling; it is the number to look at before training
anything.

Everything else the current stage gates on -- divergence, symmetry, mutual
nearest neighbour, distances, probability -- is recorded as a *feature* rather
than applied as a filter, so the classifier can weigh them jointly.

**Labels.** A candidate is positive when its parent matches an annotated
dividing cell and its two children match that cell's two annotated daughters,
one each, all within the metric's 7 um. Matching is the distinct assignment
`diag_divisions.py` uses, so a pair collapsed by NMS scores as one match, not
two.
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: F401,E402
import tracksdata as td  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from diag_divisions import _match  # noqa: E402
from params import SCALE, TRAIN_DIR  # noqa: E402

SC = np.asarray(SCALE)

FEATURES = [
    "prob_q",           # model probability, parent -> candidate second child
    "prob_c1",          # ... parent -> the child it committed to
    "prob_ratio",       # prob_q / prob_c1
    "prob_rank",        # rank of the candidate among the parent's candidates
    "parent_dist",      # um, parent -> candidate
    "child_dist",       # um, parent -> committed child
    "sister_dist",      # um, the two would-be sisters
    "symmetry",         # |child_dist - parent_dist| / mean
    "has_grandkids",    # both sisters have a committed child at t+2
    "diverge_gap",      # grandchild distance - sister distance (0 if unknown)
    "gc_dist",          # grandchild distance (0 if unknown)
    "best_rival_prob",  # the best any other parent scores for this candidate
    "prob_margin",      # prob_q - best_rival_prob
    "is_mutual_nn",     # candidate is the free node nearest the committed child
    "nn_dist_parent",   # um to the parent's own nearest neighbour in its frame
    "free_frac",        # free nodes / nodes in the child frame
    "parent_track_len", # nodes in the parent's component
    "cand_track_len",   # nodes in the candidate's component
    "flow_residual",    # um, parent->candidate displacement minus the frame's median
    "cos_angle",        # angle between parent->child and parent->candidate
    "t_frac",           # frame index / n frames
]

# Loose enough to keep the recall ceiling, tight enough not to emit the whole
# frame: these are not decisions, they only bound the set.
MAX_PARENT_UM = 14.0
MAX_SISTER_UM = 20.0
MIN_PROB = 0.02


def build(job):
    path, = job
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

    p_um = coords[:, 1:] * SC
    frame_of = coords[:, 0].astype(int)
    by_frame = {t: np.flatnonzero(frame_of == t) for t in np.unique(frame_of)}
    n_frames = max(by_frame) + 1 if by_frame else 1
    src, tgt = edges[:, 0].astype(int), edges[:, 1].astype(int)
    n_children = np.bincount(src, minlength=len(coords))
    only_child = np.full(len(coords), -1, np.int64)
    single = n_children[src] == 1
    only_child[src[single]] = tgt[single]
    has_parent = np.zeros(len(coords), bool)
    has_parent[tgt] = True

    # component sizes, for track-length features
    parent_uf = np.arange(len(coords))

    def find(a):
        while parent_uf[a] != a:
            parent_uf[a] = parent_uf[parent_uf[a]]
            a = parent_uf[a]
        return a

    for a, b in zip(src, tgt):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent_uf[ra] = rb
    label_of = np.array([find(i) for i in range(len(coords))])
    comp_size = np.bincount(label_of, minlength=len(coords))[label_of]

    c1s = cand[cand[:, 4] == 1] if len(cand) else cand
    prob_of: dict[tuple[int, int], float] = {}
    by_parent: dict[int, list[tuple[float, int]]] = {}
    best_for_child: dict[int, float] = {}
    for a, b, pr in zip(c1s[:, 0].astype(int), c1s[:, 1].astype(int), c1s[:, 2]):
        prob_of[(int(a), int(b))] = float(pr)
        by_parent.setdefault(int(a), []).append((float(pr), int(b)))
        if pr > best_for_child.get(int(b), 0.0):
            best_for_child[int(b)] = float(pr)
    for v in by_parent.values():
        v.sort(reverse=True)
    rank_of = {(a, b): i for a, v in by_parent.items() for i, (_, b) in enumerate(v)}

    # which prediction rows realise each annotated division
    truth: set[tuple[int, int, int]] = set()
    for par, ch in divs.items():
        t = int(nodes[par]["t"])
        gp = lambda n: np.array([nodes[n]["z"], nodes[n]["y"], nodes[n]["x"]], float) * SC
        rp = _match([gp(par)], by_frame.get(t, np.array([], int)), p_um)[0]
        if rp is None:
            continue
        m = _match([gp(ch[0]), gp(ch[1])], by_frame.get(t + 1, np.array([], int)), p_um)
        if m[0] is None or m[1] is None:
            continue
        truth.add((int(rp), int(m[0]), int(m[1])))
        truth.add((int(rp), int(m[1]), int(m[0])))

    rows, labels, meta = [], [], []
    for t in sorted(by_frame):
        rows_next = by_frame.get(t + 1)
        if rows_next is None:
            continue
        parents = [int(r) for r in by_frame[t]
                   if only_child[r] >= 0 and frame_of[only_child[r]] == t + 1]
        free = [int(r) for r in rows_next if not has_parent[r]]
        if not parents or not free:
            continue
        free_pos = p_um[free]
        free_tree = cKDTree(free_pos)
        here = by_frame[t]
        nn_tree = cKDTree(p_um[here])
        nn_d, _ = nn_tree.query(p_um[here], k=min(2, len(here)))
        nn_of = {int(here[i]): float(nn_d[i, 1] if nn_d.ndim > 1 else 0.0)
                 for i in range(len(here))}
        # the frame's median displacement, for the flow residual
        disp = p_um[tgt[frame_of[src] == t]] - p_um[src[frame_of[src] == t]]
        flow = np.median(disp, axis=0) if len(disp) else np.zeros(3)
        free_frac = len(free) / max(len(rows_next), 1)

        for parent in parents:
            child = int(only_child[parent])
            child_vec = p_um[child] - p_um[parent]
            child_dist = float(np.linalg.norm(child_vec))
            _, idx = free_tree.query(p_um[child])
            mutual = free[int(idx)]
            for q in free_tree.query_ball_point(p_um[parent], MAX_PARENT_UM):
                q = free[int(q)]
                if q == child:
                    continue
                prob_q = prob_of.get((parent, q), 0.0)
                if prob_q < MIN_PROB:
                    continue
                q_vec = p_um[q] - p_um[parent]
                parent_dist = float(np.linalg.norm(q_vec))
                sister_dist = float(np.linalg.norm(p_um[child] - p_um[q]))
                if sister_dist > MAX_SISTER_UM:
                    continue
                g1, g2 = only_child[child], only_child[q]
                ok_g = (g1 >= 0 and g2 >= 0
                        and frame_of[g1] == t + 2 and frame_of[g2] == t + 2)
                gc = float(np.linalg.norm(p_um[g1] - p_um[g2])) if ok_g else 0.0
                prob_c1 = prob_of.get((parent, child), 0.0)
                denom = max((child_dist + parent_dist) / 2.0, 1e-6)
                rival = max(0.0, best_for_child.get(q, 0.0) - prob_q)
                nrm = np.linalg.norm(child_vec) * np.linalg.norm(q_vec)
                rows.append([
                    prob_q, prob_c1, prob_q / max(prob_c1, 1e-6),
                    float(rank_of.get((parent, q), 99)),
                    parent_dist, child_dist, sister_dist,
                    abs(child_dist - parent_dist) / denom,
                    float(ok_g), (gc - sister_dist) if ok_g else 0.0, gc,
                    rival, prob_q - rival, float(q == mutual),
                    nn_of.get(parent, 0.0), free_frac,
                    float(comp_size[parent]), float(comp_size[q]),
                    float(np.linalg.norm(q_vec - flow)),
                    float(np.dot(child_vec, q_vec) / nrm) if nrm > 0 else 0.0,
                    t / max(n_frames, 1),
                ])
                labels.append(1 if (parent, child, q) in truth else 0)
                meta.append((t, parent, child, q))

    X = np.asarray(rows, np.float32).reshape(-1, len(FEATURES))
    y = np.asarray(labels, np.int8)
    m = np.asarray(meta, np.int64).reshape(-1, 4)
    return stem, X, y, m, len(divs)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_dir", default="../output/pp_experiments/div_cache/preds_0.50")
    p.add_argument("--out", default="../output/pp_experiments/div_dataset.npz")
    p.add_argument("--jobs", type=int, default=24)
    args = p.parse_args()

    paths = sorted(Path(args.cache_dir).glob("*.npz"))
    if not paths:
        raise SystemExit(f"no cache under {args.cache_dir}; run exp_divisions.py first")

    Xs, ys, ms, stems, n_div, n_hit = [], [], [], [], 0, 0
    with Pool(args.jobs) as pool:
        for stem, X, y, m, ndiv in pool.imap_unordered(
                build, [(str(q),) for q in paths], chunksize=1):
            if len(X):
                Xs.append(X); ys.append(y); ms.append(m)
                stems.extend([stem] * len(X))
            n_div += ndiv
            n_hit += int(y.sum() > 0) * 0  # counted properly below

    X = np.vstack(Xs); y = np.concatenate(ys); m = np.vstack(ms)
    stems = np.asarray(stems)
    emb = np.asarray([s.split("_")[0] for s in stems])

    # one positive candidate per recovered division (both orderings are stored
    # in `truth`, but only one of the two is ever a legal candidate here since
    # the other daughter is the committed child)
    print(f"\ncandidates {len(X):,}   positives {int(y.sum())}   "
          f"ratio 1:{len(X) / max(int(y.sum()), 1):.0f}")
    print(f"annotated divisions in these stacks: {n_div}")
    print(f"CANDIDATE-SET RECALL CEILING: {int(y.sum())} / {n_div} "
          f"= {y.sum() / max(n_div, 1):.0%}")
    for e in sorted(set(emb)):
        sel = emb == e
        print(f"  {e}: {int(y[sel].sum()):4d} positives of {int(sel.sum()):,} candidates")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, meta=m, stems=stems, embryo=emb,
                        features=np.asarray(FEATURES), n_divisions=n_div)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
