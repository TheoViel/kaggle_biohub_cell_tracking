"""Post-process a saved tracker npz — the best chain found, from the npz alone.

    python src/postprocessing/postprocess_npz.py PRED.npz [-o OUT.npz] [--geff DIR]

Needs nothing but the npz: no model, no checkpoint, no images. Three stages,
each independently switchable:

  1. pipeline    (on)   re-link every frame pair as a one-to-one assignment on
                        flow-corrected distance, then drop the short tracks.
                        Also where the zero-division prior lives, since a
                        one-to-one assignment cannot produce a second child.
  2. pp_public   (on)   tracking.pp_public.postprocess_graph, with the knobs
                        tuned in exp_pp_public.py --tune.
  3. ILP         (off)  tracksdata's ILPSolver. Off because on top of the other
                        two it is worth +0.0006 for ~40 s a stack — it only
                        restates the degree constraint stage 1 already imposes.

Measured over the 71 validation stacks of logs/2026-08-20/20/ (exp_final.py),
adjusted edge Jaccard:

    raw                                   0.7592
    --no-pp-public                        0.8065     (pipeline only)
    default                               0.8197     (pipeline + pp_public)
    --ilp                                 0.8203     (all three)
    --no-pipeline                         0.8101     (pp_public only)

Writes an npz in the same format it reads — `coords` (N, 4) [t, z, y, x] and
`edges` (M, 4) [source row, target row, probability, distance] — so the output
is a drop-in for anything that consumes the tracker's own output.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# common.py chdirs into src/ so that params' relative paths resolve, which would
# otherwise silently re-root any relative path given on the command line.
INVOKED_FROM = Path.cwd()
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from common import to_graph  # noqa: E402  (imports chdir into src/)
from stages import run_ilp, run_pipeline, run_pp_public  # noqa: E402

# Tuned on a 10-stack subset spanning the difficulty range; see the README.
PP_PUBLIC_TUNED = dict(filter_short_tracks=1, min_track_len=4, relink_tight_um=6.5)
ILP_TUNED = dict(appear=0.02, disappear=0.02, division=1.0, timeout=900.0)


def resolve(path):
    """A path as the user meant it, despite the chdir above."""
    path = Path(path)
    return path if path.is_absolute() else INVOKED_FROM / path


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("npz", help="tracker output: coords (N,4) + edges (M,4)")
    p.add_argument("-o", "--out", default=None,
                   help="output npz (default: <input>_pp.npz)")
    p.add_argument("--geff", default=None,
                   help="also write a geff into this directory, for the viewer")

    p.add_argument("--no-recentre", action="store_true",
                   help="skip the grid-centring correction. It is applied only "
                        "to npz files that still need it, detected from the "
                        "coordinates, so this is rarely what you want")
    p.add_argument("--no-pipeline", action="store_true",
                   help="skip the re-linking and short-track pruning")
    p.add_argument("--no-pp-public", action="store_true",
                   help="skip tracking.pp_public.postprocess_graph")
    p.add_argument("--ilp", action="store_true",
                   help="run the ILP solver as a final pass (slow, ~+0.0006)")

    p.add_argument("--sample", default=None,
                   help="sample name; defaults to the npz stem. Only used to "
                        "label the output and to find ground truth for scoring")
    p.add_argument("--no-score", action="store_true",
                   help="do not score against ground truth even if it is present")
    return p.parse_args()


def score_if_possible(sample, coords, edges, label):
    """Print the notebook's metrics, when this sample has annotation to score on."""
    import pandas as pd
    import tracksdata as td

    from common import score
    from params import TRAIN_DIR

    gt_path = Path(TRAIN_DIR) / f"{sample}.geff"
    meta_path = Path("../input/metadata.csv")
    if not gt_path.exists() or not meta_path.exists():
        return False
    meta = pd.read_csv(meta_path).set_index("sample_name")
    if sample not in meta.index:
        return False

    result = td.graph.IndexedRXGraph.from_geff(gt_path)
    gt_graph = result[0] if isinstance(result, tuple) else result
    n_total = float(meta.loc[sample, "estimated_number_of_nodes"])
    score(to_graph(coords, edges), gt_graph, n_total, label)
    return True


def main():
    args = parse_args()
    npz_path = resolve(args.npz)
    sample = args.sample or npz_path.stem

    data = np.load(npz_path, allow_pickle=True)
    coords = data["coords"].astype(np.int64)
    edges = data["edges"].astype(np.float64).reshape(-1, 4)
    print(f"-> {npz_path}")
    print(f"   {len(coords)} nodes, {len(edges)} edges\n")

    stages = []
    if not args.no_pipeline:
        recentre = None if args.no_recentre else "auto"
        stages.append(("pipeline", lambda c, e: run_pipeline(c, e, recentre=recentre)))
    if not args.no_pp_public:
        stages.append(("pp_public(tuned)",
                       lambda c, e: run_pp_public(c, e, sample=sample, **PP_PUBLIC_TUNED)))
    if args.ilp:
        stages.append(("ILP(tuned)", lambda c, e: run_ilp(c, e, **ILP_TUNED)))
    if not stages:
        print("   nothing to do — every stage is disabled")

    scored = not args.no_score and score_if_possible(sample, coords, edges, "raw")
    for name, fn in stages:
        coords, edges = fn(coords, edges)
        print(f"   after {name:<18s} {len(coords)} nodes, {len(edges)} edges")
        if scored:
            score_if_possible(sample, coords, edges, f"after {name}")

    out_path = resolve(args.out) if args.out else npz_path.with_name(f"{npz_path.stem}_pp.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        # round, not truncate — see stages.run_pp_public.
        coords=np.round(np.asarray(coords, dtype=np.float64)).astype(np.int16),
        edges=np.asarray(edges, dtype=np.float64).reshape(-1, 4),
    )
    print(f"\n-> wrote {out_path}")

    if args.geff:
        from tracking_cellmot.io import save_graph
        geff_dir = resolve(args.geff)
        geff_dir.mkdir(parents=True, exist_ok=True)
        save_graph(to_graph(coords, edges), str(geff_dir / f"{sample}.geff"))
        print(f"-> wrote {geff_dir / f'{sample}.geff'}")


if __name__ == "__main__":
    main()
