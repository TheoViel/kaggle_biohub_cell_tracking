"""Why the worst stacks are the worst: a per-edge post-mortem, not a score.

Takes the scored output of a run (``<log_folder>/scores_pp.json`` plus the
``preds_pp/*.npz`` it describes) and, for every ground-truth link, asks what
actually happened to it:

* **tp**       — reproduced.
* **stolen**   — the child was linked, to the wrong parent. Costs two false
                 positives on top of the false negative, so it is the expensive
                 failure and the one worth attacking.
* **declined** — the child was left with no parent at all. One false negative,
                 no false positive; the pipeline's gates do this deliberately.
* **lost**     — one of the two ground-truth nodes was never detected. Nothing
                 downstream can fix it.

For every stolen link it also reports where the true parent ranked among the
child's candidates once the frame's flow is removed, which says whether the
answer was reachable and just mis-ranked, or genuinely not there.

Run:  python src/postprocessing/analysis_worst.py --log_folder ../logs/2026-08-20/20/
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import tracksdata as td
from scipy.spatial import cKDTree

from common import ROOT, to_graph
from params import SCALE, TRAIN_DIR
from relink import local_flow
from tracking_cellmot.metrics import evaluate

SCALE_ARR = np.asarray(SCALE)
MATCH_UM = 7.0


def load_stack(npz_path, sample):
    d = np.load(npz_path, allow_pickle=True)
    coords = d["coords"].astype(np.float64)
    edges = d["edges"].astype(np.float64).reshape(-1, 4)
    result = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{sample}.geff")
    gt = result[0] if isinstance(result, tuple) else result
    return coords, edges, gt


def match_nodes(coords, edges, gt_graph):
    """gt node id -> row of ``coords``, using the metric's own matching."""
    graph = to_graph(coords, edges)
    node_ids = np.asarray(graph.node_ids())
    # build_graph adds nodes in coords order; the analysis indexes by row.
    assert np.array_equal(node_ids, np.arange(len(coords))), "node ids are not row indices"
    with contextlib.redirect_stdout(io.StringIO()):
        evaluate(graph, gt_graph, scale=SCALE)
    attrs = graph.node_attrs(attr_keys=[
        td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    matched = attrs.filter(
        pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID).is_not_null()
        & (pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID) != -1)
    )
    return {
        int(g): int(p) for p, g in zip(
            matched[td.DEFAULT_ATTR_KEYS.NODE_ID],
            matched[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID],
        )
    }


def gt_edges(gt_graph):
    df = gt_graph.edge_attrs(attr_keys=[])
    return list(zip(
        df[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list(),
        df[td.DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list(),
    ))


def flow_by_frame(coords, edges, flow_k=30):
    """Per-node displacement its neighbourhood took, frame by frame."""
    pos_um = coords[:, 1:] * SCALE_ARR
    t = coords[:, 0].astype(int)
    rows_by_t = {int(tt): np.flatnonzero(t == tt) for tt in np.unique(t)}
    src = edges[:, 0].astype(int)
    tgt = edges[:, 1].astype(int)
    out = {}
    for u in sorted(rows_by_t):
        here = rows_by_t.get(u)
        there = rows_by_t.get(u + 1)
        if there is None or len(here) == 0 or len(there) == 0:
            continue
        local_s = {int(r): i for i, r in enumerate(here)}
        local_t = {int(r): i for i, r in enumerate(there)}
        mask = np.isin(src, here) & np.isin(tgt, there)
        links = np.array(
            [[local_s[src[i]], local_t[tgt[i]]] for i in np.flatnonzero(mask)], dtype=int
        ).reshape(-1, 2)
        out[u] = (here, there, local_flow(pos_um[here], pos_um[there], links, k=flow_k))
    return out, pos_um, rows_by_t


def diagnose(sample, npz_path):
    coords, edges, gt_graph = load_stack(npz_path, sample)
    g2p = match_nodes(coords, edges, gt_graph)
    parent_of = {int(t): int(s) for s, t, *_ in edges}

    flows, pos_um, rows_by_t = flow_by_frame(coords, edges)
    gt_um = {
        int(i): np.array([z, y, x]) * SCALE_ARR
        for i, z, y, x in zip(*[
            gt_graph.node_attrs(attr_keys=["z", "y", "x"])[c].to_numpy()
            if c != "id" else gt_graph.node_ids()
            for c in ("id", "z", "y", "x")
        ])
    }

    counts = Counter()
    ranks, steal_gain, thief_um = [], [], []
    for gs, gt_ in gt_edges(gt_graph):
        ps, pt = g2p.get(gs), g2p.get(gt_)
        if ps is None or pt is None:
            counts["lost"] += 1
            continue
        got = parent_of.get(pt)
        if got == ps:
            counts["tp"] += 1
            continue
        counts["declined" if got is None else "stolen"] += 1
        if got is None:
            continue

        # Is the thief a different cell, or another detection of the same one?
        # The metric matches one detection per ground-truth node; a second one
        # sitting on the same cell is a valid parent physically and a false
        # positive by the score.
        thief_um.append(float(np.linalg.norm(pos_um[got] - gt_um[int(gs)])))

        # Where did the true parent rank, once the frame's flow is removed?
        u = int(coords[ps, 0])
        if u not in flows:
            continue
        here, _there, flow = flows[u]
        d = np.linalg.norm((pos_um[here] + flow) - pos_um[pt], axis=1)
        order = here[np.argsort(d)]
        rank = int(np.flatnonzero(order == ps)[0]) + 1 if ps in set(order.tolist()) else -1
        ranks.append(rank)
        d_true = float(d[np.flatnonzero(here == ps)[0]])
        d_got = float(d[np.flatnonzero(here == got)[0]]) if got in set(here.tolist()) else np.nan
        steal_gain.append(d_true - d_got)

    # --- stack-level features -------------------------------------------------
    nn, motion = [], []
    for u, (here, there, flow) in flows.items():
        if len(here) > 1:
            d, _ = cKDTree(pos_um[here]).query(pos_um[here], k=2)
            nn.append(np.median(d[:, 1]))
        motion.append(np.median(np.linalg.norm(flow, axis=1)))
    loc = []
    gt_pos = gt_graph.node_attrs(attr_keys=["t", "z", "y", "x"])
    gt_arr = np.column_stack([gt_pos[c].to_numpy() for c in ("t", "z", "y", "x")]).astype(float)
    rivals = 0
    for row in gt_arr:
        here = rows_by_t.get(int(row[0]))
        if here is None or len(here) == 0:
            continue
        d = np.linalg.norm(pos_um[here] - row[1:] * SCALE_ARR, axis=1)
        loc.append(d.min())
        rivals += int((d <= MATCH_UM).sum() > 1)

    return dict(
        sample=sample,
        n_gt_edges=sum(counts.values()),
        **{k: counts.get(k, 0) for k in ("tp", "stolen", "declined", "lost")},
        rank1=sum(1 for r in ranks if r == 1),
        rank2_3=sum(1 for r in ranks if 2 <= r <= 3),
        rank_gt3=sum(1 for r in ranks if r > 3),
        # A thief within the matching radius of the true parent is the same
        # cell, detected twice; outside it, a genuine mis-assignment.
        dup_steal=sum(1 for d in thief_um if d <= MATCH_UM),
        far_steal=sum(1 for d in thief_um if d > MATCH_UM),
        median_thief_um=float(np.median(thief_um)) if thief_um else float("nan"),
        median_steal_gain=float(np.nanmedian(steal_gain)) if steal_gain else float("nan"),
        nn_um=float(np.median(nn)) if nn else float("nan"),
        motion_um=float(np.median(motion)) if motion else float("nan"),
        loc_um=float(np.median(loc)) if loc else float("nan"),
        rival_frac=rivals / max(len(gt_arr), 1),
        n_nodes=len(coords),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_folder", default=str(ROOT / "logs" / "2026-08-20" / "20"))
    ap.add_argument("--scores", default=None)
    ap.add_argument("--preds_pp", default=None)
    ap.add_argument("--n", type=int, default=10, help="how many worst / best stacks")
    args = ap.parse_args()

    log_folder = Path(args.log_folder)
    scores_path = Path(args.scores) if args.scores else log_folder / "scores_pp.json"
    if not scores_path.exists():
        scores_path = log_folder / "preds_pp" / "scores_pp.json"
    scores = json.loads(scores_path.read_text())
    preds_pp = Path(args.preds_pp) if args.preds_pp else log_folder / "preds_pp"

    ranked = sorted(scores["per_sample"].items(), key=lambda kv: kv[1]["score"])
    groups = {"worst": ranked[: args.n], "best": ranked[-args.n:]}

    hdr = ("sample", "score", "tp", "stolen", "dup", "far", "declin", "lost",
           "rank1", "rk2-3", "rk>3", "thief", "nn_um", "motion", "loc_um", "rival")
    tally = ("tp", "stolen", "dup_steal", "far_steal", "declined", "lost",
             "rank1", "rank2_3", "rank_gt3")
    for label, items in groups.items():
        print(f"\n=== {label} {len(items)} ===")
        print(f"{hdr[0]:<16s}" + "".join(f"{h:>8s}" for h in hdr[1:]))
        acc = []
        for name, row in items:
            d = diagnose(name, preds_pp / f"{name}.npz")
            d["score"] = row["score"]
            acc.append(d)
            print(f"{name:<16s}"
                  f"{d['score']:>8.3f}{d['tp']:>8d}{d['stolen']:>8d}"
                  f"{d['dup_steal']:>8d}{d['far_steal']:>8d}{d['declined']:>8d}"
                  f"{d['lost']:>8d}{d['rank1']:>8d}{d['rank2_3']:>8d}{d['rank_gt3']:>8d}"
                  f"{d['median_thief_um']:>8.2f}{d['nn_um']:>8.2f}{d['motion_um']:>8.2f}"
                  f"{d['loc_um']:>8.2f}{d['rival_frac']:>8.2f}")
        tot = defaultdict(float)
        for d in acc:
            for k in tally:
                tot[k] += d[k]
        n_gt = sum(d["n_gt_edges"] for d in acc)
        print(f"{'TOTAL':<16s}{'':>8s}" + "".join(f"{tot[k]:>8.0f}" for k in tally))
        print(f"  of {n_gt} GT links: "
              f"tp {tot['tp'] / n_gt:.1%}, stolen {tot['stolen'] / n_gt:.1%} "
              f"(dup {tot['dup_steal'] / n_gt:.1%} + far {tot['far_steal'] / n_gt:.1%}), "
              f"declined {tot['declined'] / n_gt:.1%}, lost {tot['lost'] / n_gt:.1%}")
        for k in ("nn_um", "motion_um", "loc_um", "rival_frac"):
            print(f"  median {k:<10s} {np.median([d[k] for d in acc]):.3f}")


if __name__ == "__main__":
    main()
