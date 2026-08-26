"""Solve the whole video at once, over the model's candidates rather than its picks.

The notebook's ILP runs on the graph the tracker already committed to — one
parent per child, chosen greedily. The solver can then only delete edges, never
choose a different parent, so it cannot reach the runner-up that is the true
parent whenever the tracker errs.

Feeding it the top-K candidates instead gives it the choice, and the objective
becomes a global one: a link is taken only if it beats paying for the cell to
disappear here and another to appear there, with that decision propagating
along the whole track rather than being made frame by frame.

Edge weights blend the model's probability with how well the link agrees with
the local flow, since the model scores appearance and knows nothing about the
field sliding between frames.

Run:  python src/postprocessing/exp_ilp.py
"""

import contextlib
import io

import numpy as np
import tracksdata as td
from scipy.spatial import cKDTree

from common import RESULTS, SAMPLE, fmt, load_gt, load_prediction, save, score, to_graph
from exp_selective import prune_short_tracks
from model_assign import load_candidates
from params import SCALE
from relink import relink

SCALE_ARR = np.asarray(SCALE)


def candidate_edges(coords, by_pair, seed_edges, top_k=4, residual_sigma=3.0):
    """Top-K parents per child, weighted by model probability x flow agreement."""
    coords = np.asarray(coords)
    pos_um = coords[:, 1:].astype(float) * SCALE_ARR
    rows_by_t = {int(t): np.flatnonzero(coords[:, 0] == t) for t in np.unique(coords[:, 0])}

    seed = np.asarray(seed_edges, dtype=float).reshape(-1, seed_edges.shape[1])
    seed_t = coords[seed[:, 0].astype(int), 0].astype(int)

    out = []
    for t, per_child in by_pair.items():
        if t not in rows_by_t or t + 1 not in rows_by_t:
            continue
        # Flow from the links we already trust, evaluated at every parent.
        idx = np.flatnonzero(seed_t == t)
        here = rows_by_t[t]
        if len(idx) >= 10:
            anchor = pos_um[seed[idx, 0].astype(int)]
            disp = pos_um[seed[idx, 1].astype(int)] - anchor
            k = min(30, len(idx))
            _, near = cKDTree(anchor).query(pos_um[here], k=k)
            flow = np.median(disp[near.reshape(len(here), -1)], axis=1)
        else:
            flow = np.zeros((len(here), 3))
        flow_of = {int(r): flow[i] for i, r in enumerate(here)}

        for child, cands in per_child.items():
            for parent, prob in sorted(cands, key=lambda pp: -pp[1])[:top_k]:
                p, c = int(parent), int(child)
                residual = np.linalg.norm(
                    pos_um[c] - pos_um[p] - flow_of.get(p, 0.0)
                )
                weight = prob * float(np.exp(-(residual / residual_sigma) ** 2))
                out.append((p, c, weight, float(np.linalg.norm(pos_um[c] - pos_um[p]))))
    return np.asarray(out, dtype=float).reshape(-1, 4)


def solve(graph, edge_w, appear, disappear, division, timeout=900.0):
    solver = td.solvers.ILPSolver(
        edge_weight=edge_w * td.EdgeAttr("edge_prob"),
        appearance_weight=appear,
        disappearance_weight=disappear,
        division_weight=division,
        num_threads=8,
        timeout=timeout,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return solver.solve(graph).detach()


def main():
    gt_graph, n_total = load_gt()
    coords, edges = load_prediction()
    by_pair = load_candidates(RESULTS / f"{SAMPLE}_edge_candidates.npz")

    rows = [score(to_graph(coords, edges), gt_graph, n_total, "baseline (raw npz)")]

    seed = relink(coords, edges, max_distance=7.5)
    cand = candidate_edges(coords, by_pair, seed, top_k=4)
    print(f"\ncandidate graph: {len(coords)} nodes, {len(cand)} edges "
          f"(weight = p_model x flow agreement)")
    cand_graph = to_graph(coords, cand)

    out = []
    for division in (1.0, 50.0):
        for appear in (0.05, 0.15, 0.3):
            g = solve(cand_graph, -1.0, appear, appear, division)
            r = score(g, gt_graph, n_total,
                      f"ILP(cand) appear={appear:g} div={division:g}")
            out.append(r)
            # Track pruning on the solved graph.
            e = np.array([
                [s, t, p, 0.0, 0.0] for s, t, p in zip(
                    g.edge_attrs()["source_id"].to_list(),
                    g.edge_attrs()["target_id"].to_list(),
                    g.edge_attrs()["edge_prob"].to_list(),
                )
            ]).reshape(-1, 5)
            node_index = {int(n): i for i, n in enumerate(sorted(g.node_ids()))}
            e[:, 0] = [node_index[int(s)] for s in e[:, 0]]
            e[:, 1] = [node_index[int(t)] for t in e[:, 1]]
            for cut in (5, 8, 12):
                c2, e2 = prune_short_tracks(coords, e, cut)
                out.append(score(to_graph(c2, e2[:, :4]), gt_graph, n_total,
                                 f"ILP(cand) appear={appear:g} div={division:g} track>={cut}",
                                 verbose=False))

    out.sort(key=lambda r: -r["adj_edge_jaccard"])
    print("\n--- top 10 ---")
    for r in out[:10]:
        print(fmt(r))
    save(rows + out, "ilp_candidates")
    best = max(rows + out, key=lambda r: r["adj_edge_jaccard"])
    print(f"\nbest: {best['label']}  adjJ={best['adj_edge_jaccard']:.4f}")


if __name__ == "__main__":
    main()
