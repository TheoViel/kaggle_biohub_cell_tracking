"""
Post-process every stack of a prediction run, and score the result.

Reads  <log_folder>/preds/*.npz     — the tracker's own output
Writes <log_folder>/preds_pp/*.npz  — the same format, post-processed
       <log_folder>/geff_pp/*.geff  — the same graphs, for the viewer
       <log_folder>/preds_pp/metrics.csv, scores_pp.json

    python main_postprocess.py --log_folder ../logs/2026-08-20/20/

With no --log_folder it takes the most recent run under ../logs/, which is the
one main_predict.py just wrote into:

    python main_postprocess.py

--pl post-processes a pseudo-labelling pass instead — the predictions
main_predict.py --pl made on the fold's *training* embryos. It reads preds_pl
and writes preds_pl_pp / geff_pl_pp, so the two never touch each other's
output, and it picks its threshold from scores_pl<_th>.json rather than from
the val scores, which were measured on different embryos:

    python main_postprocess.py --log_folder ../logs/2026-08-20/20/ --pl

Four stages, each switchable — see src/postprocessing/README.md for how they were
chosen and what each is worth. Measured over the 71 validation stacks of
logs/2026-08-20/20/, adjusted edge Jaccard:

    tracker as-is                       0.7592
    --no_pp_public   (pipeline only)    0.8130
    (default)                           0.8251
    --ilp                               +0.0006 for ~40 s a stack, not worth it

and over the 71 of logs/2026-08-26/4/preds_0.25, which is where the gap
recovery below was measured:

    tracker as-is                       0.7603
    --no_pp_public --no_recover         0.8076
    --no_pp_public                      0.8238
    --no_recover                        0.8217
    (default)                           0.8280

Gap recovery (--no_recover to turn it off) puts back the detections the
detector dropped for a run of one to three frames, by interpolating between the
track that ends before the hole and the one that starts after it, with the
sample's own drift removed first. It is worth +0.0161 on its own and +0.0063
under pp_public, and costs a second re-linking pass (5.5 -> 11.1 s a stack).
--recover_passes 2 iterates it; that is worth +0.0002 for 50% more time.

Divisions (--no_divisions to turn it off) give a parent a second child where the
geometry says a cell divided, ported from main_edge_postprocess.py. Everything
before it re-decides links one-to-one and so cannot express a fork at all, which
left the division term — 0.1 of the competition score — sitting near zero. It is
the last stage because every earlier one re-solves the links from scratch, and
because it only ever adds edges. Over the 71 validation stacks of
logs/2026-09-09/2/preds_r6_0.40, cv score:

    no divisions                        0.8489   div tp/fp/fn   0/0/26
    pp_public's own safe_divisions      0.8502                  1/51/25
    (default)                           0.8580                  4/16/22

pp_public's safe_divisions is the same public repair without the divergence,
mutual-nearest-neighbour and symmetry gates, so the two are alternatives rather
than a stack; PP_PUBLIC_TUNED turns it off. The one gate that does not carry
over is the model's own probability for the pair, which needs a scored candidate
graph that this pipeline's npz does not hold — see DIVISIONS_TUNED.

Caveat worth keeping in view: fold 0 holds out 44b6, which has 26 annotated
divisions in 71 stacks, and the tuning above moves 3 of them. The settings sit
on a broad plateau rather than a spike, but they are chosen on small counts.

The pipeline also corrects a grid-centring error in the saved coordinates: the
detection grid is 4x coarser than the image in y and x and a cell was reported
at its corner rather than its centre, a fixed -0.61 um shift. It is applied only
to npz files that still carry it, detected from the coordinates themselves, so
running this over predictions made after the fix in tracking/predict.py is safe.
Worth +0.0065 to the pipeline alone, +0.0015 after pp_public (--no_recentre).

Stacks are shared across worker processes; nothing here needs a GPU or the
image data.
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import tracksdata as td

# postprocessing/ holds the stages next to the experiments that chose them, and
# they import each other by bare name; putting the directory on the path lets
# that work whether this is run from src/ or from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent / "postprocessing"))

from params import EMBRYOS, LOG_FOLDER, SCALE, TRAIN_DIR  # noqa: E402
from pipeline import DEFAULTS as PIPELINE_DEFAULTS  # noqa: E402
from stages import run_divisions, run_ilp, run_pipeline, run_pp_public  # noqa: E402
from tracking.post_process import build_graph  # noqa: E402
from tracking_cellmot.io import save_graph  # noqa: E402
from utils.logger import get_last_log_folder  # noqa: E402
from tracking_cellmot.metrics import (  # noqa: E402
    MACRO_METRICS,
    evaluate,
    macro_summarise,
    node_recall,
    per_sample_metrics,
    pooled_counts,
    summarise,
)

# Tuned on a 10-stack subset spanning the difficulty range; src/postprocessing/exp_pp_public.py.
#
# safe_divisions is pp_public's own division repair, and it is off because the
# `divisions` stage below is the same 946 repair with the gates it lacks. Left
# on it ran 1 true positive against 51 false ones over the 71 stacks of
# 2026-09-09/2 -- see src/postprocessing/exp_divisions_pp.py.
PP_PUBLIC_TUNED = dict(
    filter_short_tracks=1, min_track_len=4, relink_tight_um=6.5, safe_divisions=0,
)
ILP_TUNED = dict(appear=0.02, disappear=0.02, division=1.0, timeout=900.0)

# The division stage, ported from main_edge_postprocess.py. Every gate but one
# carries over unchanged; `min_prob` cannot, because it reads the model's
# probability for the parent/second-child pair and this pipeline's npz holds
# committed edges only, with no scored candidate graph. 0.0 disables that gate
# -- see stages.run_divisions.
DIVISIONS_TUNED = dict(
    min_prob=0.0,
    max_parent_um=9.0,
    max_existing_child_um=10.0,
    max_sister_um=14.0,
    # 0 -- off -- not the edge chain's 0.6. Measured on the 71 stacks of
    # 2026-09-09/2: the gate rejects 3 of the 4 true divisions this stage can
    # otherwise find (tp 4 -> 1), for 13 fewer false positives, and that trade
    # loses. It is a plateau rather than a spike: tau=0 and tau=1.5 (high enough
    # that the ratio, bounded by 2, almost never trips) both score 0.858 and the
    # dip is the 0.5-1.0 middle. The gate asks the parent to sit between its
    # daughters, and on this data it often does not. It survives in the edge
    # chain because `min_prob` is doing the rejecting there.
    symmetry_tau=0.0,
    # 2.25 and 3.0 are within 0.0001 of each other (6/35/20 against 4/16/22);
    # 3.0 is the conservative end of that tie, and 3.5 gives a third of it back.
    diverge_um=3.0,
    # Not optional. Without it: 2 true against 90 false, and the stage is worth
    # +0.0009 instead of +0.0091.
    require_divergence=True,
    require_mutual_nn=True,
    # Neither cap binds at these settings -- doubling global_frac_cap changes
    # nothing -- so the tighter value stands.
    frame_frac_cap=0.0076,
    global_frac_cap=0.00375,
)

_META = None


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--log_folder", default=None,
                   help="run folder holding preds/*.npz; default = the most "
                        f"recent run under {LOG_FOLDER}")
    p.add_argument("--pl", action="store_true",
                   help="post-process a pseudo-labelling run: read preds_pl "
                        "instead of preds, and write preds_pl_pp / geff_pl_pp")
    p.add_argument("--preds_dir", default=None,
                   help="override the input directory (default: preds_<th> of the "
                        "best-scoring threshold found in <log_folder>/scores_<th>.json, "
                        "or the preds_pl_<th> / scores_pl_<th> pair under --pl)")
    p.add_argument("--th", type=float, default=None,
                   help="detection threshold to post-process, naming the "
                        "preds_<th> folder; default = the best-scoring one")
    p.add_argument("--out_dir", default=None,
                   help="override the npz output (default <log_folder>/preds_pp)")
    p.add_argument("--geff_dir", default=None,
                   help="override the geff output (default <log_folder>/geff_pp)")

    p.add_argument("--no_recentre", action="store_true",
                   help="skip the grid-centring correction (applied only to npz "
                        "files that still need it, detected from the coordinates)")
    p.add_argument("--no_pipeline", action="store_true",
                   help="skip the re-linking and short-track pruning")
    p.add_argument("--no_recover", action="store_true",
                   help="skip putting back the detections missed for a frame or two")
    p.add_argument("--recover_max_gap", type=int,
                   default=PIPELINE_DEFAULTS["recover_max_gap"],
                   help="longest run of missing frames to fill (default %(default)s)")
    p.add_argument("--recover_passes", type=int,
                   default=PIPELINE_DEFAULTS["recover_passes"],
                   help="rounds of recover + relink (default %(default)s)")
    p.add_argument("--recover_max_residual", type=float,
                   default=PIPELINE_DEFAULTS["recover_max_residual"],
                   help="um of drift-corrected motion allowed across a one-frame "
                        "gap, scaled by span for longer ones (default %(default)s)")
    p.add_argument("--no_pp_public", action="store_true",
                   help="skip tracking.pp_public.postprocess_graph")
    p.add_argument("--ilp", action="store_true",
                   help="run the ILP solver as a final pass (slow, worth ~+0.0006)")

    p.add_argument("--no_divisions", action="store_true",
                   help="skip the division proposal stage; the chain then scores "
                        "near 0 on the division term, which is 0.1 of the score")
    p.add_argument("--div_max_parent_um", type=float,
                   default=DIVISIONS_TUNED["max_parent_um"])
    p.add_argument("--div_max_existing_child_um", type=float,
                   default=DIVISIONS_TUNED["max_existing_child_um"])
    p.add_argument("--div_max_sister_um", type=float,
                   default=DIVISIONS_TUNED["max_sister_um"])
    p.add_argument("--div_symmetry_tau", type=float,
                   default=DIVISIONS_TUNED["symmetry_tau"],
                   help="reject when the parent sits beside one daughter rather "
                        "than between the two (default %(default)s)")
    p.add_argument("--div_diverge_um", type=float,
                   default=DIVISIONS_TUNED["diverge_um"],
                   help="um the grandchildren must have separated beyond the "
                        "sisters by t+2 (default %(default)s)")
    p.add_argument("--div_no_divergence", action="store_true",
                   help="drop the divergence requirement")
    p.add_argument("--div_no_mutual_nn", action="store_true")
    p.add_argument("--div_frame_frac_cap", type=float,
                   default=DIVISIONS_TUNED["frame_frac_cap"])
    p.add_argument("--div_global_frac_cap", type=float,
                   default=DIVISIONS_TUNED["global_frac_cap"])

    p.add_argument("--jobs", type=int, default=8, help="worker processes")
    p.add_argument("--limit", type=int, default=None, help="cap the stacks (debugging)")
    p.add_argument("--skip_existing", action="store_true",
                   help="leave stacks that already have an output npz alone")
    p.add_argument("--no_score", action="store_true",
                   help="post-process only; do not score against ground truth")
    return p.parse_args()


def _preds_dir_from_scores(scores, log_folder, base, th):
    """The predictions folder one scores file scored.

    Newer runs name it outright, which is the only thing that survives a sweep
    over both the threshold and the NMS radius — ``preds_r5_0.40`` is not
    derivable from the threshold. Only the basename is taken from it: a run
    predicted into a scratch --out_dir and later moved should still resolve
    against the folder actually being post-processed.

    Older files carry no such field, so fall back to rebuilding the name, still
    honouring the recorded radius when there is one, and then to the
    unsuffixed folder a lone-threshold run writes.
    """
    named = scores.get("preds_dir")
    if named:
        preds_dir = log_folder / Path(named).name
        if preds_dir.is_dir():
            return preds_dir

    radius = scores.get("nms_radius_um")
    candidates = [log_folder / f"{base}_{th:.2f}", log_folder / base]
    if radius is not None:
        candidates.insert(0, log_folder / f"{base}_r{radius:g}_{th:.2f}")
    for preds_dir in candidates:
        if preds_dir.is_dir():
            return preds_dir
    return candidates[-1]


def _best_threshold_preds_dir(log_folder, pl=False, th=None):
    """The predictions folder of the best-scoring point of main_predict's sweep.

    main_predict.py writes one scores file per grid point next to the folder it
    scored: scores_<th>.json by preds_<th>/, or scores_r<radius>_<th>.json by
    preds_r<radius>_<th>/ once the NMS radius is swept too. A --pl run names
    both halves after the pseudo-label split (scores_pl…/preds_pl…), and the
    two families are kept apart in both directions — a val run must not pick a
    pseudo-label threshold, and vice versa — since they are scored on different
    embryos.

    ``th`` picks a threshold explicitly instead. It is matched against what the
    scores files record rather than against their names, and a threshold that
    several radii were swept at is ambiguous — pass --preds_dir there.
    """
    stem, base = ("scores_pl", "preds_pl") if pl else ("scores", "preds")

    if th is None:
        # Only files that name a threshold. The glob also matches this
        # script's own scores_pp.json, which carries no threshold and -- being
        # post-processed -- outscores every raw one, so it would win the
        # comparison below and then fail on the lookup. That made a second
        # post-processing pass over the same folder crash.
        score_files = [
            f for f in log_folder.glob(f"{stem}*.json")
            if (pl or not f.stem.startswith("scores_pl"))
            and json.loads(f.read_text()).get("threshold") is not None
        ]
        if not score_files:
            raise SystemExit(
                f"No {stem}<_th>.json under {log_folder}; "
                "pass --th or --preds_dir explicitly"
            )
        # cv_score, one vote per embryo. The pooled score this used to fall
        # back on would pick whichever threshold suits the embryo contributing
        # the most stacks; it is no longer written, and the fallback stays only
        # so files from before that change remain rankable.
        best_file, best_score, best, best_key = None, float("-inf"), None, "score"
        for f in score_files:
            scores = json.loads(f.read_text())
            key = "cv_score" if scores.get("cv_score") is not None else "score"
            score = (scores[key] if key == "cv_score"
                     else (scores.get("overall") or {}).get("score"))
            if score is not None and score == score and score > best_score:
                best_file, best_score, best, best_key = f, score, scores, key
        if best is None:
            raise SystemExit(
                f"Every {stem}*.json under {log_folder} scored NaN; pass --th"
            )
        th = best["threshold"]
        radius = best.get("nms_radius_um")
        print(
            f"-> best threshold: {th:g}"
            + (f"  nms_radius_um {radius:g}" if radius is not None else "")
            + f"  ({best_key}={best_score:.4f}, from {best_file.name})"
        )
        return _preds_dir_from_scores(best, log_folder, base, th)

    print(f"-> using threshold: {th:g}")
    # An NMS-radius sweep has one folder per radius at this threshold, so the
    # name cannot be rebuilt from the threshold alone; ask the scores files,
    # which know the folder they scored. Same val/pseudo-label split as above.
    matches = set()
    for f in log_folder.glob(f"{stem}*.json"):
        if not pl and f.stem.startswith("scores_pl"):
            continue
        scores = json.loads(f.read_text())
        got = scores.get("threshold")
        if got is None or abs(float(got) - th) >= 1e-9:
            continue
        candidate = _preds_dir_from_scores(scores, log_folder, base, th)
        if candidate.is_dir():
            matches.add(candidate)
    if len(matches) > 1:
        raise SystemExit(
            f"{len(matches)} folders under {log_folder} were scored at --th "
            f"{th:g} ({', '.join(sorted(d.name for d in matches))}); "
            "pass --preds_dir to pick one"
        )
    if matches:
        return matches.pop()

    preds_dir = log_folder / f"{base}_{th:.2f}"
    if not preds_dir.is_dir():
        # Same lone-threshold fallback, but only when the plain folder is
        # genuinely the one asked for: silently post-processing a different
        # threshold than --th requested would be scored as if it were this one.
        plain = log_folder / base
        scores_plain = log_folder / f"{stem}.json"
        if plain.is_dir() and scores_plain.exists():
            got = json.loads(scores_plain.read_text()).get("threshold")
            if got is not None and abs(float(got) - th) < 1e-9:
                return plain
        raise SystemExit(
            f"No {preds_dir.name}/ under {log_folder} for --th {th:g}. "
            f"Available: {', '.join(sorted(p.name for p in log_folder.glob(f'{base}*') if p.is_dir())) or 'none'}"
        )
    return preds_dir


def _to_graph(coords, edges):
    """build_graph, with the tuple/int coercion it expects."""
    return build_graph(np.asarray(coords), [
        (int(s), int(t), float(p), float(d))
        for s, t, p, d in np.asarray(edges, dtype=float).reshape(-1, 4)
    ])


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


def _n_total(sample):
    """The stack's estimated node count, for the node-count adjustment."""
    global _META
    if _META is None:
        meta_path = Path(TRAIN_DIR).parent / "metadata.csv"
        _META = pd.read_csv(meta_path).set_index("sample_name") if meta_path.exists() \
            else pd.DataFrame()
    if sample in getattr(_META, "index", []):
        return float(_META.loc[sample, "estimated_number_of_nodes"])
    return float("nan")


def process(job):
    """One stack: post-process, save, score. Returns a metrics row or None."""
    path, cfg = job
    sample = Path(path).stem
    out_npz = Path(cfg["out_dir"]) / f"{sample}.npz"
    if cfg["skip_existing"] and out_npz.exists():
        return sample, None, "skipped (output exists)"

    data = np.load(path, allow_pickle=True)
    coords = data["coords"].astype(np.int64)
    edges = data["edges"].astype(np.float64).reshape(-1, 4)
    n_in = (len(coords), len(edges))

    if not cfg["no_pipeline"]:
        coords, edges = run_pipeline(
            coords, edges,
            recentre=None if cfg["no_recentre"] else "auto",
            recover=0 if cfg["no_recover"] else 1,
            recover_max_gap=cfg["recover_max_gap"],
            recover_passes=cfg["recover_passes"],
            recover_max_residual=cfg["recover_max_residual"],
        )
    if not cfg["no_pp_public"]:
        coords, edges = run_pp_public(coords, edges, sample=sample, **PP_PUBLIC_TUNED)
    if cfg["ilp"]:
        coords, edges = run_ilp(coords, edges, **ILP_TUNED)
    # Last: it only ever adds edges, so nothing downstream needs to see it, and
    # every earlier stage re-decides links from scratch -- pp_public's
    # motion_relink most of all -- which would throw the fork away.
    if cfg["divisions"]:
        coords, edges = run_divisions(coords, edges, **cfg["div"])

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        # round, not truncate — see stages.run_pp_public.
        coords=np.round(np.asarray(coords, dtype=np.float64)).astype(np.int16),
        edges=np.asarray(edges, dtype=np.float64).reshape(-1, 4),
    )

    graph = _to_graph(coords, edges)
    Path(cfg["geff_dir"]).mkdir(parents=True, exist_ok=True)
    save_graph(graph, str(Path(cfg["geff_dir"]) / f"{sample}.geff"))

    if cfg["no_score"]:
        return sample, None

    gt_path = Path(TRAIN_DIR) / f"{sample}.geff"
    if not gt_path.exists():
        return sample, None

    result = td.graph.IndexedRXGraph.from_geff(gt_path)
    gt_graph = result[0] if isinstance(result, tuple) else result
    res = evaluate(graph, gt_graph, scale=SCALE)
    recall = node_recall(graph, gt_graph)
    row = per_sample_metrics(res, _n_total(sample), recall)
    # per_sample_metrics stops at the raw columns; 'score' and 'division_jaccard'
    # come out of summarise, so roll this one sample up on its own — same as
    # main_predict.py does per stack.
    one = summarise([row])
    row["division_jaccard"] = one["division_jaccard"]
    row["score"] = one["score"]
    row["sample_name"] = sample
    # A preds folder can hold several embryos at once, since main_predict runs
    # every fold into one. Fold i holds out EMBRYOS[i], so the embryo names the
    # fold that produced it; None when it is not one of the known embryos.
    row["embryo"] = sample.split("_")[0]
    row["fold"] = (
        EMBRYOS.index(row["embryo"]) if row["embryo"] in EMBRYOS else None
    )
    row["n_nodes_pred"] = len(coords)
    row["n_edges_pred"] = len(edges)
    row["n_nodes_estimated"] = _n_total(sample)
    return sample, row


def main():
    args = parse_args()
    # No --log_folder: the run main_predict just wrote into is nearly always the
    # one to post-process, and naming it means reading today's date off disk.
    log_folder = Path(args.log_folder or get_last_log_folder(LOG_FOLDER))
    print(f"-> log folder: {log_folder}")
    # Both name the input folder, so honouring one silently ignores the other.
    if args.preds_dir and args.th is not None:
        raise SystemExit("--th and --preds_dir both name the input; pass one")
    # A pseudo-label pass keeps its own set of folders end to end, so it never
    # overwrites the val predictions the run is actually scored on.
    suffix = "_pl" if args.pl else ""
    preds_dir = (Path(args.preds_dir) if args.preds_dir
                 else _best_threshold_preds_dir(log_folder, pl=args.pl, th=args.th))
    out_dir = Path(args.out_dir) if args.out_dir else log_folder / f"preds{suffix}_pp"
    geff_dir = Path(args.geff_dir) if args.geff_dir else log_folder / f"geff{suffix}_pp"

    paths = sorted(preds_dir.glob("*.npz"))[: args.limit]
    if not paths:
        raise SystemExit(f"No npz files under {preds_dir}")

    recover = not args.no_pipeline and not args.no_recover
    stages = [name for name, on in (
        ("pipeline", not args.no_pipeline),
        (f"recover(gap<={args.recover_max_gap}, x{args.recover_passes}, "
         f"r={args.recover_max_residual:g})", recover),
        ("pp_public(tuned)", not args.no_pp_public),
        ("ILP(tuned)", args.ilp),
        (f"divisions(diverge={args.div_diverge_um:g}, "
         f"cap={args.div_global_frac_cap:g})", not args.no_divisions),
    ) if on]

    print(f"-> split     : {'train (pseudo-label)' if args.pl else 'val'}")
    print(f"-> reading   : {preds_dir}  ({len(paths)} stacks)")
    print(f"-> npz out   : {out_dir}")
    print(f"-> geff out  : {geff_dir}")
    print(f"-> stages    : {' -> '.join(stages) or 'none (pass-through)'}")
    print(f"-> workers   : {args.jobs}\n")

    cfg = dict(
        out_dir=str(out_dir), geff_dir=str(geff_dir),
        no_pipeline=args.no_pipeline, no_pp_public=args.no_pp_public, ilp=args.ilp,
        no_recentre=args.no_recentre,
        no_recover=args.no_recover,
        recover_max_gap=args.recover_max_gap,
        recover_passes=args.recover_passes,
        recover_max_residual=args.recover_max_residual,
        divisions=not args.no_divisions,
        div=dict(
            DIVISIONS_TUNED,
            max_parent_um=args.div_max_parent_um,
            max_existing_child_um=args.div_max_existing_child_um,
            max_sister_um=args.div_max_sister_um,
            symmetry_tau=args.div_symmetry_tau,
            diverge_um=args.div_diverge_um,
            require_divergence=not args.div_no_divergence,
            require_mutual_nn=not args.div_no_mutual_nn,
            frame_frac_cap=args.div_frame_frac_cap,
            global_frac_cap=args.div_global_frac_cap,
        ),
        no_score=args.no_score, skip_existing=args.skip_existing,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    geff_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with Pool(args.jobs) as pool:
        for i, (sample, row) in enumerate(
            pool.imap_unordered(process, [(str(p), cfg) for p in paths], chunksize=1)
        ):
            score = f"node_recall={row['node_recall']:.3f}  adj_edge_jaccard={row['adj_edge_jaccard']:.3f}" if row else ""
            print(f"[{i + 1:>3d}/{len(paths)}] {sample:<16s} - {score}", flush=True)
            if row:
                rows.append(row)

    if not rows:
        print(f"\n-> wrote {len(paths)} stacks to {out_dir} and {geff_dir} (not scored)")
        return

    metrics_file = out_dir.parent / "metrics.csv" if not args.pl else out_dir.parent / "metrics_pl.csv"
    df = pd.DataFrame(rows).sort_values("sample_name")
    df.to_csv(metrics_file, index=False)
    counts = pooled_counts(rows)

    # One summary per embryo, i.e. per fold. Kept separate rather than pooled:
    # the embryos are annotated at very different densities and edge FP is only
    # counted against annotated nodes, so a pooled figure averages two numbers
    # that are not on the same scale.
    by_embryo = {}
    for row in rows:
        by_embryo.setdefault(row.get("embryo", "?"), []).append(row)
    group_summaries, macro = macro_summarise(rows)
    per_embryo = {
        emb: {
            "fold": sorted(
                {r["fold"] for r in by_embryo[emb] if r.get("fold") is not None}
            ),
            "n_samples": len(by_embryo[emb]),
            **_jsonable(summary_emb),
        }
        for emb, summary_emb in group_summaries.items()
    }

    report_keys = (
        "score", "edge_jaccard", "adj_edge_jaccard", "division_jaccard",
        "node_recall", "edge_tp", "edge_fp", "edge_fn",
        "n_nodes_pred", "n_edges_pred", "n_nodes_estimated",
    )
    worst = sorted(rows, key=lambda r: r["score"])[:10]
    scores = {
        "log_folder": str(log_folder),
        "preds_dir": str(preds_dir),
        "split": "train" if args.pl else "val",
        "stages": stages,
        "n_samples": len(rows),
        # The only score: one vote per embryo. The pooled `overall` block that
        # used to sit here is gone -- see metrics.pooled_counts.
        "cv_score": _jsonable(macro["score"]),
        "macro": _jsonable(macro),
        "counts": _jsonable(counts),
        "per_embryo": per_embryo,
        "per_sample": {
            r["sample_name"]: _jsonable({k: r[k] for k in report_keys if k in r})
            for r in sorted(rows, key=lambda r: r["sample_name"])
        },
        "worst_10": [
            {"sample_name": r["sample_name"], "score": _jsonable(r["score"])}
            for r in worst
        ],
    }

    scores_file = out_dir.parent / ("scores_pp_pl.json" if args.pl else "scores_pp.json")
    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=4)

    print(f"\n{'=' * 60}")
    print(f"{len(rows)} stacks  ->  {out_dir}")
    print(f"{'=' * 60}")
    # Counts pool; rates do not -- see metrics.pooled_counts.
    for k in ("edge_tp", "edge_fp", "edge_fn"):
        if k in counts:
            print(f"- {k}: {counts[k]:.0f}")
    print(f"\nmacro  (mean over {macro['n_groups']} "
          f"embryo{'s' if macro['n_groups'] != 1 else ''}, one vote each)")
    for k in MACRO_METRICS:
        if k in macro:
            label = "cv_score" if k == "score" else k
            print(f"- {label:<17s}: {macro[k]:.4f}")

    # Only when there is more than one, otherwise it repeats the block above.
    if len(per_embryo) > 1:
        cols = ("node_recall", "edge_jaccard", "adj_edge_jaccard",
                "division_jaccard", "score")
        print(f"\n  {'embryo':<10}{'fold':<6}{'n':<5}"
              + "".join(f"{c:>18s}" for c in cols))
        for emb, s in per_embryo.items():
            fold_str = ",".join(str(f) for f in s["fold"]) or "-"
            cells = "".join(f"{s.get(c, float('nan')):>18.4f}" for c in cols)
            print(f"  {emb:<10}{fold_str:<6}{s['n_samples']:<5}{cells}")

    print(f"\n-> metrics : {metrics_file}")
    print(f"-> scores  : {scores_file}")


if __name__ == "__main__":
    main()
