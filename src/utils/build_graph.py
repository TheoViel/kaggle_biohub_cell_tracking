import json
import numpy as np
import polars as pl
import pandas as pd
import tracksdata as td
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


from params import OUTPUT_FOLDER, TRAIN_DIR, SCALE, INPUT_FOLDER
from tracking_cellmot.metrics import evaluate, per_sample_metrics, node_recall, summarise


def _mutual_nearest(distances, max_distance):
    """
    Keep the pairs that choose each other.

    A cell is linked to a cell in the next frame only when each is the other's
    closest option. That leaves a cell unlinked rather than handing it a partner
    that some other cell has a better claim on, which matters here because the
    centroids wander by about as much as the cells actually move, so a link made
    on a small margin is close to a coin toss.

    Args:
        distances (ndarray): (n_source, n_target) physical distances in microns.
        max_distance (float): Pairs further apart than this are not linked.

    Returns:
        tuple: Source indices and target indices of the surviving pairs.
    """
    if distances.size == 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)

    sources = np.arange(distances.shape[0])
    closest_target = np.argmin(distances, axis=1)
    closest_source = np.argmin(distances, axis=0)
    agreed = (closest_source[closest_target] == sources) & (
        distances[sources, closest_target] <= max_distance
    )
    return sources[agreed], closest_target[agreed]


def _local_flow(source, target, parents, children, neighbours):
    """
    Estimate where each cell went from where its neighbours went.

    Cells sit in tissue and drift together, so the links already found in a
    neighbourhood say more about a cell's motion than its own centroid, which
    jitters by a z slice or two between frames. Taking the median over
    neighbours keeps a handful of wrong links from dragging the estimate.

    Args:
        source (ndarray): (n_source, 3) centres in microns.
        target (ndarray): (n_target, 3) centres in microns.
        parents (ndarray): Source indices of the links found so far.
        children (ndarray): Their target indices.
        neighbours (int): How many neighbouring links to average over.

    Returns:
        ndarray: (n_source, 3) displacement to add to each source centre.
    """
    if len(parents) < neighbours:
        return np.zeros_like(source)

    linked = source[parents]
    displacement = target[children] - linked
    _, nearby = cKDTree(linked).query(source, k=neighbours)
    return np.median(displacement[nearby.reshape(len(source), -1)], axis=1)


def link_frames(
    source,
    target,
    max_distance,
    flow_neighbours=40,
    passes=8,
    scale=(1.625, 0.40625, 0.40625),
):
    """
    Link the cells of one frame to the cells of a later one.

    Linking is repeated a few times. The first round pairs cells that are
    mutually nearest where they stand; every round after that first shifts each
    cell by the motion its neighbours were seen to make, and pairs cells on
    what is left over once that shared motion is accounted for. Cells with no
    mutually nearest partner within max_distance are left unlinked, which is
    what a false negative in the next frame looks like.

    Args:
        source (ndarray): (n_source, 3) cell centres as (z, y, x) voxel coordinates.
        target (ndarray): (n_target, 3) cell centres in the later frame.
        max_distance (float): Cells further apart than this, in microns, are
            never linked.
        flow_neighbours (int, optional): How many neighbouring links the motion
            estimate averages over.
        passes (int, optional): Rounds of linking. One skips the motion estimate
            altogether.
        scale (tuple, optional): Physical voxel size as (z, y, x).

    Returns:
        tuple: An (n_links, 2) array of (source index, target index) pairs, and
            the physical distance of each link in microns.
    """
    scale = np.asarray(scale, dtype=float)
    source = np.asarray(source, dtype=float).reshape(-1, 3) * scale
    target = np.asarray(target, dtype=float).reshape(-1, 3) * scale
    if len(source) == 0 or len(target) == 0:
        return np.empty((0, 2), dtype=np.intp), np.empty(0)

    shift = np.zeros_like(source)
    for step in range(max(passes, 1)):
        distances = cdist(source + shift, target)
        parents, children = _mutual_nearest(distances, max_distance)
        if step < passes - 1:
            shift = _local_flow(source, target, parents, children, flow_neighbours)

    travelled = np.linalg.norm(target[children] - source[parents], axis=1)
    return np.stack([parents, children], axis=1), travelled


def build_graph(
    nodes_by_time,
    max_distance=7.0,
    flow_neighbours=40,
    passes=8,
    max_gap=1,
    scale=(1.625, 0.40625, 0.40625),
):
    """
    Build a tracking graph from per-frame cell centres.

    Frames are linked in order, each cell to the next frame's cells, by mutual
    nearest neighbour on centres corrected for the motion of the surrounding
    tissue. A cell the segmentation missed leaves a hole, which shows up as a
    cell with no child and, one frame later, a cell with no parent; if max_gap
    allows, those two are then offered each other.

    Gap closing runs only after every consecutive frame has been linked, and
    only between cells that came out of that childless and parentless, so a
    bridge is always a fallback and never competes with a direct link.

    Args:
        nodes_by_time (dict): {t: (n_cells, 3) centres as (z, y, x)}. Times need
            not be contiguous.
        max_distance (float, optional): Linking gate for consecutive frames, in
            microns. Widened in proportion to the gap when bridging.
        flow_neighbours (int, optional): How many neighbouring links the motion
            estimate averages over.
        passes (int, optional): Rounds of linking per frame pair.
        max_gap (int, optional): How many frames a link may span. The default of
            1 links consecutive frames only; 2 also bridges a single frame in
            which a cell was missed. Worth raising only when cells are being
            missed often enough that the holes cost more than the wrong links a
            bridge invents.
        scale (tuple, optional): Physical voxel size as (z, y, x).

    Returns:
        tracksdata.graph.InMemoryGraph: Nodes carry t, z, y, x; edges run from a
            cell to its child and carry the link distance in microns.
    """
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1")

    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    # Registered up front so that reading the attribute works even on a graph
    # that ends up with no edges at all.
    graph.add_edge_attr_key("distance", pl.Float64, 0.0)

    times = sorted(nodes_by_time)
    coords, node_ids = {}, {}
    for t in times:
        coords[t] = np.asarray(nodes_by_time[t], dtype=float).reshape(-1, 3)
        node_ids[t] = np.asarray(
            graph.bulk_add_nodes(
                [
                    {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
                    for z, y, x in coords[t]
                ]
            ),
            dtype=np.intp,
        )

    n_children = {t: np.zeros(len(coords[t]), dtype=int) for t in times}
    has_parent = {t: np.zeros(len(coords[t]), dtype=bool) for t in times}

    edges = []
    for gap in range(1, max_gap + 1):
        for t in times:
            later = t + gap
            if later not in coords:
                continue

            # A bridge is for cells the previous passes left with nothing, so
            # only childless parents and parentless children take part.
            free_source = np.flatnonzero(n_children[t] == 0) if gap > 1 else np.arange(len(coords[t]))
            free_target = np.flatnonzero(~has_parent[later])
            if len(free_source) == 0 or len(free_target) == 0:
                continue

            links, distances = link_frames(
                coords[t][free_source],
                coords[later][free_target],
                max_distance=max_distance * gap,
                flow_neighbours=flow_neighbours,
                passes=passes,
                scale=scale,
            )

            for (i, j), distance in zip(links, distances):
                parent, child = free_source[i], free_target[j]
                n_children[t][parent] += 1
                has_parent[later][child] = True
                edges.append(
                    {
                        "source_id": int(node_ids[t][parent]),
                        "target_id": int(node_ids[later][child]),
                        "distance": float(distance),
                    }
                )

    if edges:
        graph.bulk_add_edges(edges)

    return graph


def main():
    sample_name = "6bba_09961292"

    df = pd.read_csv(f"{str(INPUT_FOLDER)}/metadata.csv")

    predicted_nodes = json.load(open(OUTPUT_FOLDER + f"predicted_nodes_{sample_name}.json"))
    predicted_nodes = {int(k): np.asarray(v) for k, v in predicted_nodes.items()}
    graph = build_graph(predicted_nodes, max_distance=7.0)

    gt_graph, _ = td.graph.IndexedRXGraph.from_geff(f"{str(TRAIN_DIR)}/{sample_name}.geff")

    res = evaluate(graph, gt_graph, scale=SCALE)

    node_recall_ = node_recall(graph, gt_graph)
    print(f"- Node recall: {node_recall_:.3f}")

    n_total = df[df["sample_name"] == sample_name]["estimated_number_of_nodes"].values[0]
    print(f"- Estimated number of nodes: {n_total}")

    metrics = per_sample_metrics(res, n_total, node_recall_)
    metrics = summarise([metrics])

    edge_jaccard = metrics["edge_jaccard"]

    print(f"- Edge Jaccard: {edge_jaccard:.3f}")

    return metrics


if __name__ == "__main__":
    main()
