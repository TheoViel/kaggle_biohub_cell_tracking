"""
Turn a folder of .geff graphs into the Kaggle submission CSV.

    python submission.py ../logs/2026-08-20/20/geff_pp -o ../output/submission.csv

One row per node and one row per edge, both carrying the dataset (the geff's
stem) and a running ``id``. Node rows put the coordinates in and set the edge
columns to -1; edge rows do the reverse. Coordinates are written as integer
voxels — the graphs may hold sub-voxel positions after smoothing, and the format
does not.

Conversion only: whatever is in the geffs is what lands in the CSV. Run the
post-processing first if you want it (``main_postprocess.py`` writes its graphs
to ``<log_folder>/geff_pp``).
"""

import argparse
import csv
import sys
from pathlib import Path

import tracksdata as td

SUBMISSION_COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x",
                      "source_id", "target_id"]
CSV_COLUMNS = ["id", *SUBMISSION_COLUMNS]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("geff_dir", help="folder holding <dataset>.geff graphs")
    p.add_argument("-o", "--out", default="submission.csv", help="output CSV")
    p.add_argument("--expect", default=None,
                   help="folder of the datasets that must be present, e.g. "
                        "../input/test/ — fails if any graph is missing or extra")
    p.add_argument("--allow-empty", action="store_true",
                   help="write a dataset that has no nodes instead of failing")
    return p.parse_args()


def graph_from_geff(path):
    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def read_graph(path):
    """(nodes, edges) as plain dicts, in the columns the submission needs."""
    graph = graph_from_geff(path)
    nodes = {}
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        nodes[node_id] = {
            "node_id": node_id,
            "t": int(row["t"]),
            # The graphs carry floats; the format wants voxels.
            "z": int(round(float(row["z"]))),
            "y": int(round(float(row["y"]))),
            "x": int(round(float(row["x"]))),
        }
    edges = [
        (int(row["source_id"]), int(row["target_id"]))
        for row in graph.edge_attrs().iter_rows(named=True)
    ]
    return nodes, edges


def write_submission(geff_paths, out_path, allow_empty=False):
    """Write every graph to one CSV. Returns per-dataset (nodes, edges) counts."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {}
    row_id = 0
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for path in geff_paths:
            dataset = Path(path).stem
            nodes, edges = read_graph(path)
            if not nodes and not allow_empty:
                raise SystemExit(f"{dataset}: graph has no nodes "
                                 f"(pass --allow-empty to write it anyway)")

            for node_id in sorted(nodes):
                node = nodes[node_id]
                writer.writerow({
                    "id": row_id, "dataset": dataset, "row_type": "node",
                    "node_id": node["node_id"], "t": node["t"],
                    "z": node["z"], "y": node["y"], "x": node["x"],
                    "source_id": -1, "target_id": -1,
                })
                row_id += 1

            for source_id, target_id in edges:
                # An edge naming a node that is not in the file would be
                # unscoreable, and is a sign the graph was filtered by
                # something that forgot to drop its edges.
                if source_id not in nodes or target_id not in nodes:
                    raise SystemExit(
                        f"{dataset}: edge {source_id} -> {target_id} refers to a "
                        f"node the graph does not contain"
                    )
                writer.writerow({
                    "id": row_id, "dataset": dataset, "row_type": "edge",
                    "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1,
                    "source_id": source_id, "target_id": target_id,
                })
                row_id += 1

            counts[dataset] = (len(nodes), len(edges))
            print(f"  {dataset:<20s} {len(nodes):>8,d} nodes  {len(edges):>8,d} edges",
                  flush=True)

    return counts, row_id


def main():
    args = parse_args()
    geff_dir = Path(args.geff_dir)
    paths = sorted(geff_dir.glob("*.geff"))
    if not paths:
        raise SystemExit(f"No .geff graphs under {geff_dir}")

    print(f"-> reading {len(paths)} graphs from {geff_dir}\n")

    if args.expect:
        expected = {p.stem for p in Path(args.expect).glob("*.zarr")}
        found = {p.stem for p in paths}
        missing, extra = sorted(expected - found), sorted(found - expected)
        if missing or extra:
            raise SystemExit(
                f"dataset mismatch against {args.expect}\n"
                f"  missing: {missing[:10]}\n  extra:   {extra[:10]}"
            )

    counts, n_rows = write_submission(paths, args.out, allow_empty=args.allow_empty)

    total_nodes = sum(n for n, _ in counts.values())
    total_edges = sum(e for _, e in counts.values())
    assert n_rows == total_nodes + total_edges, "row counter disagrees with the counts"

    header = Path(args.out).open().readline().strip().split(",")
    assert header == CSV_COLUMNS, f"bad CSV header: {header}"

    print(f"\n-> wrote {args.out}")
    print(f"   {len(counts)} datasets, {n_rows:,} rows "
          f"({total_nodes:,} nodes + {total_edges:,} edges)")


if __name__ == "__main__":
    main()
