"""An oracle that answers one question: is localisation the whole story?

The lost links have both endpoints about 4 um from the annotation, with nothing
closer in the detection set, while the cell itself moved 2.3 um. So the
apparent displacement is mostly detector error, and it is larger than the
spacing between cells. That *suggests* no linker could recover them — but
suggests is not shows, because the neighbourhood flow and the one-to-one
constraint might still carry it.

This settles it. It moves the detection matched to each ground-truth node onto
the annotation — perfect localisation, but only for the ~1% of cells that are
annotated, everything else untouched — re-runs the linking unchanged, and
rescores.

    if the score goes to ~1.0   the linker is fine and localisation is the
                                entire remaining error
    if it barely moves          the linker is also wrong, and a better detector
                                would not be enough

This LEAKS THE GROUND TRUTH. It is a diagnostic, never a stage.

Run:  python src/postprocessing/analysis_oracle.py --log_folder ../logs/2026-08-20/20/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_worst import SCALE_ARR, load_stack, match_nodes
from common import ROOT, load_gt_graph, score, to_graph
from pipeline import DEFAULTS, postprocess


def oracle_coords(coords, edges, gt_graph, jitter=0.0, rng=None):
    """coords with every matched detection moved onto its annotation."""
    g2p = match_nodes(coords, edges, gt_graph)
    attrs = gt_graph.node_attrs(attr_keys=["z", "y", "x"])
    gt_zyx = {
        int(i): np.array(p, dtype=float)
        for i, p in zip(gt_graph.node_ids(),
                        np.column_stack([attrs[c].to_numpy() for c in "zyx"]))
    }
    out = coords.copy()
    for g, p in g2p.items():
        target = gt_zyx[int(g)]
        if jitter:
            # An imperfect detector, to read the curve rather than one point.
            target = target + rng.normal(0.0, jitter, 3) / SCALE_ARR
        out[p, 1:] = target
    return out, len(g2p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_folder", default=str(ROOT / "logs" / "2026-08-20" / "20"))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--jitter", type=float, nargs="*", default=[0.0, 1.0, 2.0, 3.0],
                    help="um of gaussian error left on the oracle positions")
    args = ap.parse_args()

    log_folder = Path(args.log_folder)
    scores = json.loads((log_folder / "scores_pp.json").read_text())
    preds_dir = Path(scores["preds_dir"])
    if not preds_dir.is_dir():
        preds_dir = log_folder / Path(scores["preds_dir"]).name
    ranked = sorted(scores["per_sample"].items(), key=lambda kv: kv[1]["score"])

    rng = np.random.default_rng(0)
    for name, row in ranked[: args.n]:
        print(f"\n--- {name}  (scored {row['score']:.3f})")
        coords, edges, gt_graph = load_stack(preds_dir / f"{name}.npz", name)

        c, e = postprocess(coords, edges, **DEFAULTS)
        score(to_graph(c, e), *load_gt_graph(name), "pipeline")

        for j in args.jitter:
            oc, n_moved = oracle_coords(coords, edges, gt_graph, jitter=j, rng=rng)
            c, e = postprocess(oc, edges, **DEFAULTS)
            score(to_graph(c, e), *load_gt_graph(name),
                  f"oracle loc (+{j:g} um) on {n_moved} nodes")


if __name__ == "__main__":
    main()
