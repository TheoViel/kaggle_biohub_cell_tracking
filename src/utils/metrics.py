import numpy as np
import polars as pl
import tracksdata as td
import logging
import warnings
from tracksdata.metrics import DistanceMatching
from tracksdata.options import get_options, set_options


def dist(a, b, scale=(1.625, 0.40625, 0.40625)):
    return np.sqrt(scale[0] ** 2 * (a[0] - b[0]) ** 2 + scale[1] ** 2 * (a[1] - b[1]) ** 2 + scale[2] ** 2 * (a[2] - b[2]) ** 2)


def _build_frame_graph(centers: np.ndarray, t: int) -> td.graph.InMemoryGraph:
    """Build a single-frame node graph from ``(n, 3)`` ``(z, y, x)`` centers."""
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    if len(centers):
        graph.bulk_add_nodes([
            {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
            for z, y, x in centers
        ])
    return graph


def match_frame_nodes(
    seeds: np.ndarray,
    truth: np.ndarray,
    *,
    t: int,
    scale: tuple[float, float, float],
    max_distance: float,
) -> pl.DataFrame:
    """Match predicted seeds to sparse GT nodes and return paired attributes.

    Returns a DataFrame with one row per matched pair and columns:
    ``pred_idx``, ``gt_z``, ``gt_y``, ``gt_x``, ``distance``.
    """
    pred_graph = _build_frame_graph(seeds.astype(float), t)
    gt_graph = _build_frame_graph(truth, t)

    if pred_graph.num_nodes() == 0 or gt_graph.num_nodes() == 0:
        return [], [], truth

    # print(pred_graph.num_nodes(), gt_graph.num_nodes())

    matching = DistanceMatching(max_distance=max_distance, scale=scale)

    # Node-only graphs have no edges, so tracksdata's "No matching edges found"
    # warning is expected noise here.
    td_log = logging.getLogger("tracksdata.utils._logging")
    prev_level = td_log.level
    prev_show_progress = get_options().show_progress
    set_options(show_progress=False)
    td_log.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            from scipy.sparse import SparseEfficiencyWarning

            warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
            pred_graph.match(gt_graph, matching=matching)
    finally:
        td_log.setLevel(prev_level)
        set_options(show_progress=prev_show_progress)

    pred_attrs = pred_graph.node_attrs(
        attr_keys=[
            td.DEFAULT_ATTR_KEYS.NODE_ID,
            td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID,
            td.DEFAULT_ATTR_KEYS.MATCH_SCORE,
            "z",
            "y",
            "x",
        ]
    )
    gt_attrs = gt_graph.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "z", "y", "x"]
    ).rename(
        {
            "z": "gt_z",
            "y": "gt_y",
            "x": "gt_x",
            td.DEFAULT_ATTR_KEYS.NODE_ID: "gt_node_id",
        }
    )

    matched = pred_attrs.filter(
        pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID).is_not_null()
        & (pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID) != -1)
    ).with_columns(
        pl.col(td.DEFAULT_ATTR_KEYS.NODE_ID).alias("pred_idx"),
    ).join(
        gt_attrs,
        left_on=td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID,
        right_on="gt_node_id",
        how="inner",
    )

    scale_arr = np.asarray(scale, dtype=float)
    if len(matched) == 0:
        return [], [], truth

    pred_xyz = matched.select(["z", "y", "x"]).to_numpy() * scale_arr
    gt_xyz = matched.select(["gt_z", "gt_y", "gt_x"]).to_numpy() * scale_arr
    distances = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
    pairs = matched.with_columns(pl.Series("distance", distances))

    matched = pairs.to_pandas()[["gt_z", "gt_y", "gt_x"]].values.astype(int)
    matched = [tuple(node) for node in matched]

    matched_gt = []
    unmatched_gt = []
    for node in truth:
        if tuple(node) in matched:
            matched_gt.append(node)
        else:
            unmatched_gt.append(node)
    matched_gt = np.array(matched_gt)
    unmatched_gt = np.array(unmatched_gt)

    return pairs, matched_gt, unmatched_gt

