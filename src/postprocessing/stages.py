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

from common import to_graph
from params import SCALE
from pipeline import postprocess

SCALE_ARR = np.asarray(SCALE)


# --------------------------------------------------------------------------
# zero-division prior
# --------------------------------------------------------------------------


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
    "prune_isolated": "BIOHUB_OUTPUT_PRUNE_ISOLATED",
    "relink_tight_um": "BIOHUB_MOTION_RELINK_TIGHT_UM",
    "relink_relaxed_um": "BIOHUB_MOTION_RELINK_RELAXED_UM",
    "relink_learned_bonus": "BIOHUB_MOTION_RELINK_LEARNED_BONUS",
    # Wired in so the 946 post-process sweep is reachable. `PP_SWEEP_KEYS` in
    # that notebook sweeps ten knobs; six of them are implemented in
    # tracking/pp_public.py but had no entry here, so three of its seven named
    # candidates -- gap45 (GAP_CLOSE_UM 4.5), gap2step40 (GAP2_MAX_STEP_UM 4.0)
    # and reuse28 (GAP_CLOSE_REUSE_UM 2.8) -- were live code that could not be
    # set. The values below restate pp_public's own defaults, so adding them
    # changes nothing until one is passed.
    "gap_close_um": "BIOHUB_GAP_CLOSE_UM",
    "gap_close_reuse_um": "BIOHUB_GAP_CLOSE_REUSE_UM",
    "gap_close_max_gap": "BIOHUB_GAP_CLOSE_MAX_GAP",
    "gap2_max_step_um": "BIOHUB_GAP2_MAX_STEP_UM",
    "gap2_max_total_um": "BIOHUB_GAP2_MAX_TOTAL_UM",
    "gap2_require_context": "BIOHUB_GAP2_REQUIRE_CONTEXT",
    "relink_velocity_weight": "BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT",
    "keep_division_components": "BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS",
    "enforce_next_frame": "BIOHUB_OUTPUT_ENFORCE_NEXT_FRAME",
}


# The module's own defaults, restated so every call starts from the same place.
# Without this a worker process would carry the previous chain's environment
# into the next one, and an ablation would silently measure the wrong thing.
PP_PUBLIC_DEFAULTS = {
    "motion_relink": 1, "single_child_repair": 0, "single_parent_repair": 1,
    "gap_close": 1, "gap2": 1, "safe_divisions": 1,
    "division_geometry_filter": 0, "filter_short_tracks": 0, "min_track_len": 4,
    "linefit": 1, "linefit_weight": 0.74, "edge_max_um": 14.2,
    "prune_isolated": 1,
    "relink_tight_um": 5.95, "relink_relaxed_um": 10.0, "relink_learned_bonus": 0.76,
    "gap_close_um": 5.9, "gap_close_reuse_um": 3.1, "gap_close_max_gap": 1,
    "gap2_max_step_um": 4.0, "gap2_max_total_um": 9.5, "gap2_require_context": 1,
    "relink_velocity_weight": 0.50, "keep_division_components": 1,
    "enforce_next_frame": 1,
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
    # round, not truncate: pp_public's linefit smoothing leaves sub-voxel
    # positions, and astype() would drag every one of them toward zero — the
    # same half-voxel bias the grid-centring correction exists to remove.
    return np.round(new_coords).astype(np.int64), new_edges


def run_pipeline(coords, edges, **kw):
    return postprocess(coords, edges, **kw)


# --------------------------------------------------------------------------
# divisions
# --------------------------------------------------------------------------

def run_divisions(coords, edges, cand=None, **kw):
    """``edge_pp.propose_divisions`` on a graph with no candidate array.

    The tracker's npz carries committed edges only, so the one gate that stage
    was built around -- the model's own probability for the parent/second-child
    pair -- has nothing to read. ``min_prob=0`` is what turns it off: an absent
    pair scores 0.0, and ``0.0 < 0.0`` is false, so every pair passes and the
    geometry decides alone.

    What is left is still more than ``pp_public``'s own
    ``add_safe_divisions_postlink``, which is the same 946 repair with only the
    three distance gates. ``require_divergence`` is the one that matters -- it
    asks the two sisters' own children to have separated by ``t + 2``, which is
    what tells a division from two detections of one cell -- and it is why
    pp_public's version ran at 1 true positive against 51 false ones on
    2026-09-09/2 while this one runs at 4 against 16. The two are alternatives,
    not a stack; see ``PP_PUBLIC_TUNED`` in main_postprocess.py.

    ``symmetry_tau`` also lives here but the tuned config leaves it off; the
    reasoning is with the constant.

    Only edges are added, never nodes, so this can run last.
    """
    from edge_pp import propose_divisions

    edges = np.asarray(edges, dtype=float).reshape(-1, 4)
    if cand is None:
        cand = np.empty((0, 5))
    return coords, propose_divisions(coords, edges, cand, **kw)


