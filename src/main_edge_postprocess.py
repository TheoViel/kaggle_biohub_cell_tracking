"""Post-process the edge transformer's predictions, and score the result.

Reads  <preds_dir>/*.npz   — written by main_edge_predict.py --dump_candidates
Writes <out_dir>/*.npz     — the same format, post-processed
       <geff_dir>/*.geff   — the same graphs, for the viewer
       <preds_dir>/../scores_<out_dir name>.json

    python main_edge_postprocess.py --preds_dir ../logs/edge_run/2026-09-04/3/preds_edge

The stages are in `postprocessing/edge_pp.py` and the reasoning for each is
there; the chain is

    assign (+ per-track momentum) -> recover -> prune -> linefit

and every stage is switchable, so an ablation is one flag. What separates this
from `main_postprocess.py` is the input: that one post-processes a graph whose
edges are already committed and re-decides them on geometry alone, because the
detector's probabilities are not worth much. Here the model scored every
candidate and the npz carries all of them, so the assignment is re-solved from
the probabilities themselves and the geometric stages only fill what the model
could not see -- the nodes that are missing, and the chains it left in pieces.

Nothing here needs a GPU or the image data.
"""

import warnings
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import tracksdata as td

sys.path.insert(0, str(Path(__file__).resolve().parent / "postprocessing"))

import edge_pp  # noqa: E402
import torch  # noqa: E402

from params import EMBRYOS, LOG_FOLDER, SCALE, TRAIN_DIR  # noqa: E402
from tracking.post_process import build_graph  # noqa: E402
from tracking_cellmot.io import save_graph  # noqa: E402
from tracking_cellmot.metrics import (  # noqa: E402
    MACRO_METRICS,
    evaluate,
    macro_summarise,
    pooled_counts,
    node_recall,
    per_sample_metrics,
    summarise,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Tuned on 16 of the 71 fold-0 stacks spanning the node-excess range, then
# checked on all 71 -- see postprocessing/README_edge.md, including the two
# settings that won on the subset and lost on the fold.
#
# `min_track` and `recover_min_dup` were re-tuned again for
# `logs/edge_run/2026-09-07/2/preds_0.45` -- a newer, more over-detected
# checkpoint (node ratio 1.157 raw against the 1.081 the block above was tuned
# on) -- see README_edge.md's "Retuned again for preds_0.45". min_track went
# 2 -> 5 and recover_min_dup 4.0 -> 3.0; everything else held.
#
# Retuned once more on the 199 out-of-fold stacks of
# `logs/edge_run/2026-09-09/0/preds_0.45`, the first checkpoint with the fine
# detection head. The whole `recover` block tightened and `min_track` rose:
# together +0.0131 (0.8374 -> 0.8506 without the ILP), gaining on both embryos
# (+0.0111 on 44b6, +0.0152 on 6bba). Every one of the four was positive on both
# embryos on its own as well, which is the bar README_edge.md sets. The four sit
# on a plateau -- gap 2 scores 0.8507 to gap 3's 0.8506, min_track 6/7/8 span
# 0.0011 -- so read the values as "somewhere in here", not as a peak.
DEFAULTS = dict(
    threshold=0.4,
    recover=1,
    # 3, down from 5. The finer detection head leaves less to recover, so the
    # long gaps that used to pay now mostly interpolate nodes into holes that
    # are real. 5 -> 3 is worth +0.0080 on its own; 7 costs -0.0085, 4 recovers
    # only +0.0039, and 1-2 are level with 3. Turning recover off entirely still
    # beats gap 5 (+0.0037) but loses on 44b6 (-0.0051), so the stage earns its
    # place -- at a shorter reach.
    recover_max_gap=3,
    recover_max_residual=4.0,
    # Back to 4.0, undoing the preds_0.45 retune below. Same reasoning read the
    # other way: this checkpoint over-detects less, so a node recover would skip
    # as "too close to an existing one" is once again more often a real
    # duplicate than a genuine miss. +0.0037 on its own, both embryos.
    recover_min_dup=4.0,
    through=0.0,          # measured, finds nothing -- see the README
    through_max_gap=2,
    # 7, up from 5, and the direction is the same as every retune before it:
    # whatever the chain leaves as short fragments, pruning them pays. +0.0027
    # on its own; 6 and 8 are within 0.0011 of 7 once the recover block is
    # tightened, and 9 gives a third of it back.
    min_track=7,
    linefit=0.6,
    linefit_window=3,
    vel_weight=0.02,
    vel_momentum=0.7,
    # The ILP node filter -- see `ilp_filter_nodes`. Off: one solve is ~9-16 s a
    # stack against ~1 s for the rest of the chain, so it is a final-run switch
    # rather than something to sweep behind.
    #
    # As of the 2026-09-09/0 retune it is not worth turning on. That has to be
    # measured with the divisions refitted *to the ILP graph*, because the
    # filter removes nodes and so changes what there is to fork from -- refit
    # both sides or the comparison is rigged:
    #
    #   base chain   ILP   divisions tuned for   cv
    #   old          off   off graph             0.8375
    #   old          on    on  graph             0.8466   <- ILP worth +0.0091
    #   new          off   off graph             0.8506
    #   new          on    on  graph             0.8508   <- ILP worth +0.0002
    #
    # So the filter was doing real work, and the tightened `recover` block and
    # `min_track=7` now do the same work for ~1 s a stack instead of 9-16. The
    # two were always substitutes; the retune made the cheap one sufficient.
    # Consistently, the ILP buys macro adj_edge_jaccard (0.8391 against 0.8370)
    # and sells division_jaccard (0.114 against 0.136): it drops spurious
    # detections and dividing nodes alike. Re-measure -- both sides -- before
    # trusting this on a checkpoint that over-detects again.
    ilp=0,
    ilp_threshold=0.50,
    ilp_edge_weight=-1.0,
    ilp_appearance=0.0,
    ilp_disappearance=1.5,
    # Must sit below the probability of a second child or a fork can never pay
    # for itself: with edge_weight -1 an extra edge is worth at most -1 against
    # a cost of +ilp_division, so anything >= 1.0 provably never divides. It
    # matters little here -- the ILP's edges are discarded and only its node set
    # is kept -- but it still moves which nodes end up linked.
    ilp_division=1.0,
    # Divisions. The score is `adj_edge_jaccard + 0.1 * division_jaccard` and
    # this chain scored a flat 0 on the second term -- `division_fp` was 0 too,
    # so it was not proposing them and getting them wrong, it was not proposing
    # them at all: the one-to-one assignment cannot express a fork. The gates
    # are the public "safe division" repair (PublicChanges.md, 946 Harmonic
    # Fusion) plus a `div_min_prob` gate that notebook has no way to apply,
    # since it works from geometry with no scored candidate graph.
    # See README_edge.md, "Divisions".
    divisions=1,
    # 0.40 and 0.35 are within 0.0001 of each other on the 199 out-of-fold
    # stacks of 2026-09-09/0 (0.8374 against 0.8375) and both gain on both
    # embryos; 0.40 is the conservative end of that tie, proposing 6/50 false
    # divisions against 0.35's 10/73 for the same score. 0.45 costs 0.0014.
    div_min_prob=0.40,
    div_max_parent_um=9.0,
    div_max_existing_child_um=10.0,
    div_max_sister_um=14.0,
    # 0 -- off -- retuned on the 199 out-of-fold stacks of
    # `logs/edge_run/2026-09-09/0/preds_0.45`. Worth +0.0018 over 0.6 at
    # diverge 3.0 and +0.0025 at 3.5, gaining on *both* embryos. The gate is
    # effectively binary rather than a knob: 1.5 scores identically to 0 (the
    # ratio is bounded by 2, so it almost never trips) and the whole cost sits
    # in the 0.3-0.6 middle, where 0.3 is -0.0060. It asks the parent to sit
    # between its daughters, and on this data it often does not.
    #
    # The same retune on the tracking chain, where there is no `min_prob` to do
    # the rejecting, moved it much further -- see DIVISIONS_TUNED in
    # main_postprocess.py.
    div_symmetry_tau=0.0,
    # 3.5 with the symmetry gate off, where the old 3.0 was tuned with it on.
    # Read 3.0-4.5 as one plateau at ~0.8365 +- 0.001: the curve is not monotone
    # inside it (3.5 and 4.5 both 0.8374, 3.75 0.8361), so that spread is noise
    # on 151 divisions and only the ends mean anything -- 2.25 is -0.0048 and
    # 5.0 is -0.0031.
    div_diverge_um=3.5,
    div_require_divergence=1,
    div_require_mutual_nn=1,
    div_frame_frac_cap=0.0076,
    div_global_frac_cap=0.00375,
)

# torch is only here for `recover`'s drift estimate, and its intra-op pool
# ignores OMP_NUM_THREADS; one worker per core would oversubscribe 3x.
torch.set_num_threads(1)

_META = None


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--preds_dir", default=None,
                   help="the npz main_edge_predict.py wrote; default = the "
                        "best-scoring preds_* of the latest run under "
                        f"{LOG_FOLDER}edge_run/, which is the one a threshold "
                        "sweep meant you to use")
    p.add_argument("--out_dir", default=None,
                   help="default: <preds_dir>_pp")
    p.add_argument("--geff_dir", default=None,
                   help="default: <out_dir> with 'preds' replaced by 'geff'")
    p.add_argument("--threshold", type=float, default=DEFAULTS["threshold"],
                   help="probability a link must beat to be committed "
                        "(default %(default)s)")
    p.add_argument("--no_recover", action="store_true",
                   help="skip putting back the detections missed for a frame or two")
    p.add_argument("--recover_max_gap", type=int, default=DEFAULTS["recover_max_gap"],
                   help="longest run of missing frames to fill (default %(default)s)")
    p.add_argument("--recover_max_residual", type=float,
                   default=DEFAULTS["recover_max_residual"],
                   help="um of drift-corrected motion allowed across a one-frame "
                        "gap, scaled by span for longer ones (default %(default)s)")
    p.add_argument("--recover_min_dup", type=float,
                   default=DEFAULTS["recover_min_dup"],
                   help="um; do not insert a node beside an existing detection "
                        "(default %(default)s)")
    p.add_argument("--through", type=float, default=DEFAULTS["through"],
                   help="um; join two chains through an unlinked detection that "
                        "already sits in the gap. 0 turns it off")
    p.add_argument("--through_max_gap", type=int, default=DEFAULTS["through_max_gap"])
    p.add_argument("--min_track", type=int, default=DEFAULTS["min_track"],
                   help="drop chains shorter than this (default %(default)s)")
    p.add_argument("--linefit", type=float, default=DEFAULTS["linefit"],
                   help="blend each node this far toward a line fitted along its "
                        "own track. 0 turns it off (default %(default)s)")
    p.add_argument("--linefit_window", type=int, default=DEFAULTS["linefit_window"])
    p.add_argument("--vel_weight", type=float, default=DEFAULTS["vel_weight"],
                   help="log-odds charged per um a link departs from where the "
                        "parent track's own momentum says it should go. 0 turns "
                        "it off (default %(default)s)")
    p.add_argument("--vel_momentum", type=float, default=DEFAULTS["vel_momentum"],
                   help="EMA on that velocity (default %(default)s)")
    p.add_argument("--ilp", action="store_true",
                   help="filter the nodes with a global ILP before linking: "
                        "keep only the detections it places in a track, then "
                        "run the usual chain over what is left. Worth +0.0087 "
                        "(t=7.0, 149/199 stacks) but ~9-16 s a stack, so it is "
                        "off by default -- turn it on for a final run")
    p.add_argument("--ilp_threshold", type=float, default=DEFAULTS["ilp_threshold"],
                   help="lowest candidate probability the ILP may use "
                        "(default %(default)s)")
    p.add_argument("--ilp_edge_weight", type=float, default=DEFAULTS["ilp_edge_weight"])
    p.add_argument("--ilp_appearance", type=float, default=DEFAULTS["ilp_appearance"],
                   help="ILP cost of a track starting (default %(default)s)")
    p.add_argument("--ilp_disappearance", type=float,
                   default=DEFAULTS["ilp_disappearance"],
                   help="ILP cost of a track ending (default %(default)s)")
    p.add_argument("--ilp_division", type=float, default=DEFAULTS["ilp_division"],
                   help="ILP cost of a fork; must be below a second child's "
                        "probability or none is ever taken (default %(default)s)")
    p.add_argument("--no_assign", action="store_true",
                   help="keep the committed edges already in the npz instead of "
                        "re-solving each frame pair from the candidates. Use it "
                        "to score a different matcher through the same "
                        "downstream chain; the default replaces its links")
    p.add_argument("--no_divisions", action="store_true",
                   help="skip the division proposal stage; the chain then scores "
                        "a flat 0 on the division term, which is 0.1 of the score")
    p.add_argument("--div_min_prob", type=float, default=DEFAULTS["div_min_prob"],
                   help="the model's probability a parent/second-child pair must "
                        "reach before it can become a division (default %(default)s)")
    p.add_argument("--div_max_parent_um", type=float,
                   default=DEFAULTS["div_max_parent_um"])
    p.add_argument("--div_max_existing_child_um", type=float,
                   default=DEFAULTS["div_max_existing_child_um"])
    p.add_argument("--div_max_sister_um", type=float,
                   default=DEFAULTS["div_max_sister_um"])
    p.add_argument("--div_symmetry_tau", type=float,
                   default=DEFAULTS["div_symmetry_tau"],
                   help="reject when the parent sits beside one daughter rather "
                        "than between the two (default %(default)s)")
    p.add_argument("--div_diverge_um", type=float, default=DEFAULTS["div_diverge_um"],
                   help="um the grandchildren must have separated beyond the "
                        "sisters by t+2 (default %(default)s)")
    p.add_argument("--div_no_divergence", action="store_true",
                   help="drop the divergence requirement (measured: 10x the "
                        "false positives -- see README_edge.md)")
    p.add_argument("--div_no_mutual_nn", action="store_true")
    p.add_argument("--div_frame_frac_cap", type=float,
                   default=DEFAULTS["div_frame_frac_cap"])
    p.add_argument("--div_global_frac_cap", type=float,
                   default=DEFAULTS["div_global_frac_cap"])
    p.add_argument("--jobs", type=int, default=16, help="worker processes")
    p.add_argument("--limit", type=int, default=None, help="cap the stacks (debugging)")
    p.add_argument("--no_score", action="store_true",
                   help="post-process only; do not score against ground truth")
    return p.parse_args()


def _latest_run(base_dir: Path) -> Path:
    """The most recently written run under ``base_dir/<date>/<exp_id>/``."""
    candidates = sorted(base_dir.glob("*/*/args.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No run with an args.json found under {base_dir}")
    return candidates[-1].parent


def _folder_score(preds_dir: Path) -> float | None:
    """A preds folder's headline score, from the json main_edge_predict wrote.

    ``cv_score`` is the competition number, one vote per embryo; macro
    ``edge_jaccard`` stands in for a split where it could not be formed (no
    divisions present, say). A NaN is treated as no score at all rather than
    compared, since NaN loses every comparison and would quietly rank last --
    or first, depending which way it is written.

    Files written before the pooled ``overall`` block was dropped are still read
    from it, last, so an old run remains rankable. That is a fallback and not a
    preference: on two embryos of very different sample counts the two numbers
    disagree, and mixing them across a sweep would rank the folders by which
    format they happen to be in.
    """
    path = preds_dir.parent / f"scores_{preds_dir.name}.json"
    if not path.exists():
        return None
    blob = json.load(open(path))
    macro = blob.get("macro") or {}
    for value in (blob.get("cv_score"), macro.get("score"), macro.get("edge_jaccard")):
        if value is not None and value == value:
            return float(value)
    legacy = blob.get("overall") or {}
    for key in ("score", "edge_jaccard"):
        value = legacy.get(key)
        if value is not None and value == value:
            return float(value)
    return None


def _best_preds_dir(base_dir: Path) -> tuple[Path, str]:
    """The best-scoring ``preds_*`` folder of the latest run, and why it won.

    A threshold sweep leaves several folders written within seconds of each
    other, so "the most recent" picks between them essentially at random. The
    scores are sitting right there beside them, so rank by those instead and
    fall back to mtime only when nothing was scored (a ``--no_score`` run).

    Skips folders already suffixed ``_pp``: those are this script's own output,
    not raw candidates to post-process.
    """
    run_dir = _latest_run(base_dir)
    # Sorted so a tie resolves the same way twice: the metric only judges edges
    # touching an annotated node, a few dozen of the ~25k committed, so adjacent
    # thresholds routinely score identically and glob order would decide.
    candidates = sorted(
        d for d in run_dir.glob("preds_*") if d.is_dir() and not d.name.endswith("_pp")
    )
    if not candidates:
        raise FileNotFoundError(f"No preds_* folder found under {run_dir}")

    scored = [(d, s) for d in candidates if (s := _folder_score(d)) is not None]
    if scored:
        best, score = max(scored, key=lambda kv: kv[1])
        others = ", ".join(f"{d.name}={s:.4f}" for d, s in sorted(scored, key=lambda kv: -kv[1])[1:])
        return best, (f"best score {score:.4f}" + (f"; also {others}" if others else ""))
    return max(candidates, key=lambda d: d.stat().st_mtime), "most recent (none scored)"


def ilp_filter_nodes(coords, edges, cand, cfg):
    """Keep only the detections a global ILP places in a track.

    `tracksdata`'s ILPSolver over the whole candidate graph, subject to
    in-degree <= 1 and out-degree <= 2 -- the matcher the public notebooks run
    before their own repair chain. What is kept from it is deliberately *only
    the node set*: measured over 199 out-of-fold stacks, keeping its edges as
    well scored +0.0027 against the plain chain while keeping just its nodes
    and letting `assign` re-link scored +0.0087 (t=7.0, better on 149 of 199).
    Its edge Jaccard is in fact the worst of the four arms tried -- the ILP is
    a better judge of which detections are real than of which links are.

    That is also what the notebooks do, for all that they run the solver: their
    `motion_relink` discards the ILP edge set and re-solves from geometry,
    inheriting only `nodes_by_id`.

    Returns `(coords, edges, cand)` renumbered onto the surviving nodes.
    """
    import contextlib
    import io

    import tracksdata as td

    from tracking.post_process import build_graph

    keep_prob = cand[:, 2] >= cfg["ilp_threshold"]
    c = cand[(cand[:, 4] == 1) & keep_prob]
    if not len(c):
        return coords, edges, cand

    with contextlib.redirect_stdout(io.StringIO()):
        graph = build_graph(
            coords,
            [(int(a), int(b), float(p), float(d)) for a, b, p, d, _ in c],
        )
    if graph.num_edges() == 0:
        return coords, edges, cand

    solver = td.solvers.ILPSolver(
        edge_weight=cfg["ilp_edge_weight"] * td.EdgeAttr("edge_prob"),
        appearance_weight=cfg["ilp_appearance"],
        disappearance_weight=cfg["ilp_disappearance"],
        division_weight=cfg["ilp_division"],
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        solved = solver.solve(graph)

    ea = solved.edge_attrs()
    if "solution" in ea.columns:
        ea = ea.filter(ea["solution"])
    if not len(ea):
        return coords, edges, cand

    # `build_graph` numbers nodes in coords order, so a solution node id is a
    # coords row. A node the solver linked to nothing is one it judged not to
    # belong to any track, which is exactly what this stage is here to drop.
    n = len(coords)
    deg = np.zeros(n, int)
    np.add.at(deg, ea["source_id"].to_numpy().astype(int), 1)
    np.add.at(deg, ea["target_id"].to_numpy().astype(int), 1)
    keep = np.flatnonzero(deg > 0)
    remap = -np.ones(n, int)
    remap[keep] = np.arange(len(keep))

    # Renumber the candidates with the coords, dropping any that touch a node
    # the ILP declined -- those are the links it ruled out, and `assign` should
    # not get to re-propose them.
    cand = cand.reshape(-1, 5)
    if len(cand):
        ok = (remap[cand[:, 0].astype(int)] >= 0) & (remap[cand[:, 1].astype(int)] >= 0)
        cand = cand[ok].copy()
        cand[:, 0] = remap[cand[:, 0].astype(int)]
        cand[:, 1] = remap[cand[:, 1].astype(int)]
    edges = np.asarray(edges, float).reshape(-1, 4)
    if len(edges):
        ok = (remap[edges[:, 0].astype(int)] >= 0) & (remap[edges[:, 1].astype(int)] >= 0)
        edges = edges[ok].copy()
        edges[:, 0] = remap[edges[:, 0].astype(int)]
        edges[:, 1] = remap[edges[:, 1].astype(int)]
    return coords[keep], edges, cand


def postprocess(coords, edges, cand, cfg):
    """The chain. `(coords, edges, candidates)` -> `(coords, edges)`."""
    if cfg.get("ilp"):
        # Before everything: it removes nodes, so every later stage should see
        # the reduced graph rather than re-deriving it.
        coords, edges, cand = ilp_filter_nodes(coords, edges, cand, cfg)

    assign_kw = dict(vel_weight=cfg["vel_weight"],
                     vel_momentum=cfg["vel_momentum"])
    if cfg.get("assign", 1):
        edges = edge_pp.assign(coords, cand, threshold=cfg["threshold"], **assign_kw)
    else:
        # Keep whatever matcher wrote the file. `assign` re-solves the frame
        # pairs from the candidate array and never reads the committed edges,
        # so an alternative matcher upstream -- the ILP, say -- would otherwise
        # be silently discarded here and the two inputs would score identically.
        edges = np.asarray(edges, dtype=float).reshape(-1, 4)
    if cfg["recover"]:
        coords, edges = edge_pp.recover_nodes(
            coords, edges, cand,
            max_gap=cfg["recover_max_gap"],
            max_residual=cfg["recover_max_residual"],
            min_dup=cfg["recover_min_dup"],
            threshold=cfg["threshold"],
            assign_kw=assign_kw,
        )
    if cfg["through"]:
        # Before pruning: the isolated nodes it wires in are exactly the
        # one-node "tracks" pruning is about to delete.
        edges = edge_pp.link_through_isolated(
            coords, edges, max_gap=cfg["through_max_gap"], max_um=cfg["through"])
    if cfg.get("divisions"):
        # Before pruning: attaching a free detection to a parent puts it in that
        # parent's component, so a second daughter that would have been pruned
        # as a fragment survives on the strength of the link.
        edges = edge_pp.propose_divisions(
            coords, edges, cand,
            min_prob=cfg["div_min_prob"],
            max_parent_um=cfg["div_max_parent_um"],
            max_existing_child_um=cfg["div_max_existing_child_um"],
            max_sister_um=cfg["div_max_sister_um"],
            symmetry_tau=cfg["div_symmetry_tau"],
            diverge_um=cfg["div_diverge_um"],
            require_divergence=bool(cfg["div_require_divergence"]),
            require_mutual_nn=bool(cfg["div_require_mutual_nn"]),
            frame_frac_cap=cfg["div_frame_frac_cap"],
            global_frac_cap=cfg["div_global_frac_cap"],
        )
    if cfg["min_track"]:
        coords, edges = edge_pp.prune_short(coords, edges, cfg["min_track"])
    if cfg["linefit"]:
        coords = edge_pp.linefit_smooth(
            coords, edges, window=cfg["linefit_window"], weight=cfg["linefit"])
    return coords, edges


def _n_total(sample):
    global _META
    if _META is None:
        meta_path = Path(TRAIN_DIR).parent / "metadata.csv"
        _META = (pd.read_csv(meta_path).set_index("sample_name")
                 if meta_path.exists() else pd.DataFrame())
    if sample in getattr(_META, "index", []):
        return float(_META.loc[sample, "estimated_number_of_nodes"])
    return float("nan")


def process(job):
    path, cfg = job
    sample = Path(path).stem
    coords, edges, cand = edge_pp.load_pred(path)
    if len(cand) == 0:
        raise SystemExit(
            f"{path} carries no candidates; re-run main_edge_predict.py "
            "with --dump_candidates"
        )
    coords, edges = postprocess(coords, edges, cand, cfg)

    out_npz = Path(cfg["out_dir"]) / f"{sample}.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        # round, not truncate: linefit smoothing leaves sub-voxel positions and
        # astype() would drag every one of them toward zero.
        coords=np.round(np.asarray(coords, np.float64)).astype(np.int16),
        edges=np.asarray(edges, np.float64).reshape(-1, 4),
    )

    graph = build_graph(np.round(np.asarray(coords)).astype(np.int64), [
        (int(s), int(t), float(p), float(d))
        for s, t, p, d in np.asarray(edges, float).reshape(-1, 4)
    ])
    Path(cfg["geff_dir"]).mkdir(parents=True, exist_ok=True)
    save_graph(graph, str(Path(cfg["geff_dir"]) / f"{sample}.geff"))

    gt_path = Path(TRAIN_DIR) / f"{sample}.geff"
    if cfg["no_score"] or not gt_path.exists():
        return sample, None

    result = td.graph.IndexedRXGraph.from_geff(gt_path)
    gt_graph = result[0] if isinstance(result, tuple) else result
    res = evaluate(graph, gt_graph, scale=SCALE)
    row = per_sample_metrics(res, _n_total(sample), node_recall(graph, gt_graph))
    one = summarise([row])
    row["division_jaccard"] = one["division_jaccard"]
    row["score"] = one["score"]
    row["sample_name"] = sample
    row["embryo"] = sample.split("_")[0]
    row["fold"] = EMBRYOS.index(row["embryo"]) if row["embryo"] in EMBRYOS else None
    row["n_nodes_pred"] = len(coords)
    row["n_edges_pred"] = len(edges)
    return sample, row


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def main():
    args = parse_args()
    if args.preds_dir:
        preds_dir = Path(args.preds_dir)
    else:
        preds_dir, why = _best_preds_dir(Path(LOG_FOLDER) / "edge_run")
        print(f"-> no --preds_dir given, using {preds_dir}  ({why})")
    out_dir = Path(args.out_dir) if args.out_dir else preds_dir.with_name(
        preds_dir.name + "_pp")
    geff_dir = Path(args.geff_dir) if args.geff_dir else out_dir.with_name(
        out_dir.name.replace("preds", "geff", 1))

    cfg = dict(
        out_dir=str(out_dir), geff_dir=str(geff_dir), no_score=args.no_score,
        threshold=args.threshold,
        recover=0 if args.no_recover else 1,
        recover_max_gap=args.recover_max_gap,
        recover_max_residual=args.recover_max_residual,
        recover_min_dup=args.recover_min_dup,
        through=args.through, through_max_gap=args.through_max_gap,
        min_track=args.min_track,
        linefit=args.linefit, linefit_window=args.linefit_window,
        vel_weight=args.vel_weight, vel_momentum=args.vel_momentum,
        ilp=1 if args.ilp else 0,
        ilp_threshold=args.ilp_threshold,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance=args.ilp_appearance,
        ilp_disappearance=args.ilp_disappearance,
        ilp_division=args.ilp_division,
        assign=0 if args.no_assign else 1,
        divisions=0 if args.no_divisions else 1,
        div_min_prob=args.div_min_prob,
        div_max_parent_um=args.div_max_parent_um,
        div_max_existing_child_um=args.div_max_existing_child_um,
        div_max_sister_um=args.div_max_sister_um,
        div_symmetry_tau=args.div_symmetry_tau,
        div_diverge_um=args.div_diverge_um,
        div_require_divergence=0 if args.div_no_divergence else 1,
        div_require_mutual_nn=0 if args.div_no_mutual_nn else 1,
        div_frame_frac_cap=args.div_frame_frac_cap,
        div_global_frac_cap=args.div_global_frac_cap,
    )
    stages = [name for name, on in (
        (f"ilp-node-filter(p>={args.ilp_threshold:g}, dis={args.ilp_disappearance:g}, "
         f"div={args.ilp_division:g})", args.ilp),
        ("keep-input-edges(no assign)", args.no_assign),
        (f"assign(th={args.threshold:g}"
         + (f", momentum w={args.vel_weight:g} m={args.vel_momentum:g}"
            if args.vel_weight else "") + ")", not args.no_assign),
        (f"recover(gap<={args.recover_max_gap}, r={args.recover_max_residual:g})",
         not args.no_recover),
        (f"through(gap<={args.through_max_gap}, {args.through:g} um)", args.through),
        (f"divisions(p>={args.div_min_prob:g}, diverge={args.div_diverge_um:g}, "
         f"cap={args.div_global_frac_cap:g})", not args.no_divisions),
        (f"prune(min_track={args.min_track})", args.min_track),
        (f"linefit(w={args.linefit:g}, +-{args.linefit_window})", args.linefit),
    ) if on]

    paths = sorted(preds_dir.glob("*.npz"))[: args.limit]
    if not paths:
        raise SystemExit(f"No npz files under {preds_dir}")

    print(f"-> reading   : {preds_dir}  ({len(paths)} stacks)")
    print(f"-> npz out   : {out_dir}")
    print(f"-> geff out  : {geff_dir}")
    print(f"-> stages    : {' -> '.join(stages)}")
    print(f"-> workers   : {args.jobs}\n")
    out_dir.mkdir(parents=True, exist_ok=True)
    geff_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with Pool(args.jobs) as pool:
        for i, (sample, row) in enumerate(
            pool.imap_unordered(process, [(str(p), cfg) for p in paths], chunksize=1)
        ):
            score = (f"node_recall={row['node_recall']:.3f}  "
                     f"adj_edge_jaccard={row['adj_edge_jaccard']:.3f}") if row else ""
            print(f"[{i + 1:>3d}/{len(paths)}] {sample:<16s} - {score}", flush=True)
            if row:
                rows.append(row)

    if not rows:
        print(f"\n-> wrote {len(paths)} stacks to {out_dir} and {geff_dir} (not scored)")
        return

    counts = pooled_counts(rows)
    group_summaries, macro = macro_summarise(rows)
    by_embryo = {}
    for row in rows:
        by_embryo.setdefault(row.get("embryo", "?"), []).append(row)
    report_keys = ("score", "edge_jaccard", "adj_edge_jaccard", "division_jaccard",
                   "node_recall", "edge_tp", "edge_fp", "edge_fn",
                   "division_tp", "division_fp", "division_fn",
                   "n_nodes_pred", "n_edges_pred")

    scores = {
        "preds_dir": str(preds_dir),
        "stages": stages,
        "settings": {k: v for k, v in cfg.items()
                     if k not in ("out_dir", "geff_dir", "no_score")},
        "n_samples": len(rows),
        # The only score. `overall` -- the sample-count-weighted pooling -- used
        # to sit here beside it and is gone: see metrics.pooled_counts.
        "cv_score": _jsonable(macro["score"]),
        "macro": _jsonable(macro),
        "counts": _jsonable(counts),
        "per_embryo": {
            emb: {
                "fold": sorted({r["fold"] for r in by_embryo[emb]
                                if r.get("fold") is not None}),
                "n_samples": len(by_embryo[emb]),
                **_jsonable(s),
            }
            for emb, s in group_summaries.items()
        },
        "per_sample": {
            r["sample_name"]: _jsonable({k: r[k] for k in report_keys if k in r})
            for r in sorted(rows, key=lambda r: r["sample_name"])
        },
        "worst_10": [
            {"sample_name": r["sample_name"], "score": _jsonable(r["score"])}
            for r in sorted(rows, key=lambda r: r["score"])[:10]
        ],
    }
    scores_file = out_dir.parent / f"scores_{out_dir.name}.json"
    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=4)

    print(f"\n{'=' * 78}")
    print(f"{len(rows)} stacks  ->  {out_dir}")
    print(f"{'=' * 78}")

    # Pooled over every stack: counts only, where summing is the meaningful
    # thing. The pooled *rates* are not reported at all -- see
    # metrics.pooled_counts for why.
    for k in ("edge_tp", "edge_fp", "edge_fn",
              "division_tp", "division_fp", "division_fn"):
        if k in counts:
            print(f"- {k}: {counts[k]:.0f}")

    # The number a fold is judged on: one vote per embryo.
    print(f"\nmacro  (mean over {macro['n_groups']} "
          f"embryo{'s' if macro['n_groups'] != 1 else ''}, one vote each)")
    for k in MACRO_METRICS:
        if k in macro:
            label = "cv_score" if k == "score" else k
            print(f"- {label:<17s}: {macro[k]:.4f}")

    per_embryo = scores["per_embryo"]
    if len(per_embryo) > 1:
        cols = ("node_recall", "edge_jaccard", "adj_edge_jaccard",
                "division_jaccard", "score")
        print(f"\nper embryo\n  {'embryo':<8}{'fold':<6}{'n':<6}"
              + "".join(f"{c:>18s}" for c in cols)
              + f"{'div tp/fp/fn':>16s}")
        for emb, s in per_embryo.items():
            folds = sorted({r["fold"] for r in by_embryo[emb]
                            if r.get("fold") is not None})
            cells = "".join(f"{s.get(c, float('nan')):>18.4f}" for c in cols)
            div = (f"{int(s.get('division_tp', 0))}/{int(s.get('division_fp', 0))}"
                   f"/{int(s.get('division_fn', 0))}")
            print(f"  {emb:<8}{','.join(str(f) for f in folds) or '-':<6}"
                  f"{s['n_samples']:<6}{cells}{div:>16s}")

    print(f"\n-> scores  : {scores_file}")


if __name__ == "__main__":
    main()
