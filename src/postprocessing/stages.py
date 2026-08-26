"""Post-processing stages, each (coords, edges) -> (coords, edges).

Written as a common signature so they can be chained in any order and compared
on equal terms: raw -> X, and best-pipeline -> X.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import os

import numpy as np
import tracksdata as td
from scipy.optimize import linear_sum_assignment

from common import to_graph
from params import SCALE
from pipeline import postprocess

SCALE_ARR = np.asarray(SCALE)


# --------------------------------------------------------------------------
# zero-division prior
# --------------------------------------------------------------------------

def cap_children(coords, edges, mode="best"):
    """Leave every node at most one child.

    The annotation contains 26 dividing nodes in 19 826 edges across the whole
    validation set, so forbidding divisions can cost at most 0.13% of the
    recoverable links — while 7.7% of predicted edges are second children, and
    every one of those that touches an annotated cell is a false positive.

    mode="best"   : keep each parent's most confident child.
    mode="assign" : solve each frame pair as a one-to-one assignment on
                    probability, so a child whose only good parent is taken can
                    still keep it and the loser falls to its own second choice.
    """
    edges = np.asarray(edges, dtype=float).reshape(-1, 4)
    if len(edges) == 0:
        return coords, edges

    if mode == "best":
        order = np.argsort(-edges[:, 2], kind="stable")
        seen = set()
        keep = np.zeros(len(edges), bool)
        for i in order:
            src = int(edges[i, 0])
            if src not in seen:
                seen.add(src)
                keep[i] = True
        return coords, edges[keep]

    coords = np.asarray(coords)
    src_t = coords[edges[:, 0].astype(int), 0].astype(int)
    keep = np.zeros(len(edges), bool)
    for t in np.unique(src_t):
        idx = np.flatnonzero(src_t == t)
        parents = np.unique(edges[idx, 0].astype(int))
        children = np.unique(edges[idx, 1].astype(int))
        pi = {p: i for i, p in enumerate(parents)}
        ci = {c: i for i, c in enumerate(children)}
        # Maximise total probability; absent pairs must never be chosen.
        cost = np.full((len(parents), len(children)), 1.0)
        where = {}
        for i in idx:
            a, b = pi[int(edges[i, 0])], ci[int(edges[i, 1])]
            cost[a, b] = -edges[i, 2]
            where[(a, b)] = i
        rows, cols = linear_sum_assignment(cost)
        for a, b in zip(rows, cols):
            if (a, b) in where:
                keep[where[(a, b)]] = True
    return coords, edges[keep]


# --------------------------------------------------------------------------
# ILP
# --------------------------------------------------------------------------

def run_ilp(coords, edges, edge_w=-1.0, appear=0.1, disappear=0.1, division=1.0,
            timeout=600.0, threads=1):
    """The notebook's ILP, returned as (coords, edges) rather than a graph."""
    edges = np.asarray(edges, dtype=float).reshape(-1, 4)
    if len(edges) == 0:
        return coords, edges
    graph = to_graph(coords, edges)
    solver = td.solvers.ILPSolver(
        edge_weight=edge_w * td.EdgeAttr("edge_prob"),
        appearance_weight=appear,
        disappearance_weight=disappear,
        division_weight=division,
        num_threads=threads,
        timeout=timeout,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solved = solver.solve(graph).detach()

    # build_graph adds nodes in coords order, so node ids are row indices; the
    # solver keeps ids, it only removes edges.
    ea = solved.edge_attrs()
    if solved.num_edges() == 0:
        return coords, np.empty((0, 4))
    out = np.column_stack([
        np.asarray(ea["source_id"].to_list(), float),
        np.asarray(ea["target_id"].to_list(), float),
        np.asarray(ea["edge_prob"].to_list(), float),
        np.asarray(ea["edge_dist"].to_list(), float),
    ])
    return coords, out


# --------------------------------------------------------------------------
# tracking.pp_public
# --------------------------------------------------------------------------

PP_PUBLIC_ENV = {
    "motion_relink": "BIOHUB_OUTPUT_MOTION_RELINK",
    "single_child_repair": "BIOHUB_OUTPUT_SINGLE_CHILD_REPAIR",
    "single_parent_repair": "BIOHUB_OUTPUT_SINGLE_PARENT_REPAIR",
    "gap_close": "BIOHUB_OUTPUT_GAP_CLOSE",
    "gap2": "BIOHUB_OUTPUT_GAP2_RECOVERY",
    "safe_divisions": "BIOHUB_OUTPUT_SAFE_DIVISIONS",
    "division_geometry_filter": "BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER",
    "filter_short_tracks": "BIOHUB_OUTPUT_FILTER_SHORT_TRACKS",
    "min_track_len": "BIOHUB_OUTPUT_MIN_TRACK_LEN",
    "linefit": "BIOHUB_OUTPUT_LINEFIT_SMOOTH",
    "linefit_weight": "BIOHUB_OUTPUT_LINEFIT_WEIGHT",
    "edge_max_um": "BIOHUB_OUTPUT_EDGE_MAX_UM",
    "gap_refine": "BIOHUB_GAP_REFINE_SYNTHETIC",
    "prune_isolated": "BIOHUB_OUTPUT_PRUNE_ISOLATED",
    "relink_tight_um": "BIOHUB_MOTION_RELINK_TIGHT_UM",
    "relink_relaxed_um": "BIOHUB_MOTION_RELINK_RELAXED_UM",
    "relink_learned_bonus": "BIOHUB_MOTION_RELINK_LEARNED_BONUS",
}


# The module's own defaults, restated so every call starts from the same place.
# Without this a worker process would carry the previous chain's environment
# into the next one, and an ablation would silently measure the wrong thing.
PP_PUBLIC_DEFAULTS = {
    "motion_relink": 1, "single_child_repair": 0, "single_parent_repair": 1,
    "gap_close": 1, "gap2": 1, "safe_divisions": 1,
    "division_geometry_filter": 0, "filter_short_tracks": 0, "min_track_len": 4,
    "linefit": 1, "linefit_weight": 0.74, "edge_max_um": 14.2,
    "gap_refine": 0, "prune_isolated": 1,
    "relink_tight_um": 5.95, "relink_relaxed_um": 10.0, "relink_learned_bonus": 0.76,
}


def run_pp_public(coords, edges, sample="", **knobs):
    """postprocess_graph with its module-level knobs overridden.

    The module reads every setting from the environment at import time, so the
    only way to change one is to set it and reload. Image-based gap refinement
    is off: it wants the test frames and fails on training stacks.
    """
    for name, value in {**PP_PUBLIC_DEFAULTS, **knobs}.items():
        os.environ[PP_PUBLIC_ENV[name]] = str(int(value) if isinstance(value, bool) else value)

    import tracking.pp_public as pp
    importlib.reload(pp)

    graph = to_graph(coords, np.asarray(edges, dtype=float).reshape(-1, 4))
    with contextlib.redirect_stdout(io.StringIO()):
        nodes_pp, edges_pp, stats = pp.postprocess_graph(graph, dataset=sample)

    index = {n["node_id"]: i for i, n in enumerate(nodes_pp)}
    new_coords = np.array([[n["t"], n["z"], n["y"], n["x"]] for n in nodes_pp], dtype=float)
    new_edges = np.array([
        [index[e["source_id"]], index[e["target_id"]],
         1.0 if e["edge_prob"] is None else e["edge_prob"],
         0.0 if e.get("distance_um") is None else e["distance_um"]]
        for e in edges_pp
    ], dtype=float).reshape(-1, 4)
    return new_coords.astype(np.int64), new_edges


def run_pipeline(coords, edges, **kw):
    return postprocess(coords, edges, **kw)
