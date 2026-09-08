"""
Browse a sample's 3D stack with the ground truth, the predictions and the tracks on top.

Run it from the repository root:

    streamlit run webapp/trajectory_viewer.py

The view is one z slice of one timepoint. Cells are drawn when they sit within
the z tolerance of that slice, faded by how far off it they are, so scrolling z
walks through the stack rather than jumping between disjoint sets of cells.
Tracks are drawn backwards from the current timepoint, which is what makes a
wrong link visible: a trail that jumps sideways came off the wrong cell.

The stack is a plotly chart that is updated in place, so it holds its zoom and
pan while t and z move: zoom into a handful of cells and step through time, and
the view stays on them.
"""

import base64
import io
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tracksdata as td
import zarr
from matplotlib.colors import to_rgb
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from params import SCALE  # noqa: E402
from utils.metrics import match_frame_nodes  # noqa: E402

TRAIN_DIR = ROOT / "input" / "train"
OUTPUT_DIR = ROOT / "output"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

# Radius, in microns, within which a prediction counts as having found a
# ground-truth cell. The same value tracking_cellmot.metrics.evaluate matches
# at, so the rings drawn here mean what the score means.
MAX_MATCH_DISTANCE = 7.0

# How big the view is, in pixels, and how the page splits between it and the
# controls beside it. Resizing is this one number: the markers follow it, so
# that they keep ringing the cells rather than dotting them.
VIEW_PIXELS = 1240
DISPLAY_SHARE = (3, 1)
# Room to the right of the stack for the legend to stand in.
LEGEND_PIXELS = 190

_OF_460_PIXELS = VIEW_PIXELS / 460
PRED_MARKER = 7 * _OF_460_PIXELS
# Same size as a matched prediction, so the ring sits right on top of its dot.
GT_MARKER = PRED_MARKER
FAINT_MARKER = 4 * _OF_460_PIXELS
RING_WIDTH = 1.2 * _OF_460_PIXELS
DOT_WIDTH = 0.8 * _OF_460_PIXELS
TRAIL_WIDTH = 1.0 * _OF_460_PIXELS

# Same colours as scatter_plot_center_preds, so the app and the notebook figures
# read the same way.
GT_MATCHED = "limegreen"
GT_MATCHED_EDGE = "darkgreen"
GT_UNMATCHED = "orangered"
GT_UNMATCHED_EDGE = "darkred"
PRED_MATCHED = "skyblue"
PRED_MATCHED_EDGE = "dodgerblue"
PRED_UNMATCHED = "orangered"

# What became of a ground-truth edge, in the colours utils.plot uses for the
# same verdicts, so a colour means the same thing here as in the notebook.
VERDICT_COLORS = {
    "correct": "limegreen",
    "wrong partner": "orangered",
    "no link": "goldenrod",
    "undetected": "orangered",
    "unjudged": "limegreen",
}
WRONG_LINK = "darkorange"


@st.cache_data(show_spinner=False)
def list_all_samples():
    """Sample names that have a stack, whether or not anything is predicted for them."""
    return sorted(path.stem for path in TRAIN_DIR.glob("*.zarr"))


@st.cache_data(show_spinner=False)
def list_experiments():
    """Experiment names, the subfolders of output/predictions holding *.geff predictions."""
    return sorted(
        path.name for path in PREDICTIONS_DIR.glob("*")
        if path.is_dir() and any(path.glob("*.geff"))
    )


@st.cache_data(show_spinner=False)
def list_samples(experiment):
    """Sample names an experiment has predictions for, that also have a stack."""
    predicted = {path.stem for path in (PREDICTIONS_DIR / experiment).glob("*.geff")}
    return sorted(predicted & set(list_all_samples()))


@st.cache_data(show_spinner=False)
def list_frames(sample):
    """Timepoints available for a sample."""
    return list(range(load_stack(sample).shape[0]))


@st.cache_data(show_spinner=False)
def load_stack(sample):
    """The whole (t, z, y, x) stack for a sample, loaded into memory once."""
    group = zarr.open_group(TRAIN_DIR / f"{sample}.zarr", mode="r")
    return np.asarray(group["0"])


def load_volume(sample, t):
    """The (z, y, x) stack of one timepoint, sliced out of the in-memory stack."""
    return load_stack(sample)[t]


@st.cache_data(show_spinner=False)
def contrast_limits(sample, low=0.01, high=0.999):
    """Display range from quantiles, since the raw range is set by a few bright voxels."""
    quantiles = zarr.open_group(TRAIN_DIR / f"{sample}.zarr", mode="r").attrs.get(
        "image_statistics", {}
    ).get("quantiles", {})
    limits = [next((float(v) for k, v in quantiles.items() if abs(float(k) - q) < 1e-6), None)
              for q in (low, high)]
    if all(limit is not None for limit in limits):
        return tuple(limits)
    flat = load_stack(sample).ravel()[::50]
    return float(np.quantile(flat, low)), float(np.quantile(flat, high))


def _load_geff(path):
    """A geff's graph, or None if there is nothing at that path."""
    if not path.exists():
        return None
    result = td.graph.IndexedRXGraph.from_geff(path)
    return result[0] if isinstance(result, tuple) else result


@st.cache_data(show_spinner=False)
def load_prediction_graph(experiment, sample):
    """An experiment's predicted graph: where its cells are, and how it linked them.

    Returns the centres keyed by timepoint, the (timepoint, row) each node id
    occupies in them, and the edges as (source id, target id, probability).

    The edges are read off the geff rather than re-derived from the centres.
    They are what the metric scores, so linking the centres afresh here would
    draw a different tracker than the one being measured — which is precisely
    how the view came to disagree with the numbers.
    """
    if not experiment:
        return {}, {}, np.empty((0, 3))
    graph = _load_geff(PREDICTIONS_DIR / experiment / f"{sample}.geff")
    if graph is None:
        return {}, {}, np.empty((0, 3))

    centres, row_of = {}, {}
    for (t,), frame in graph.node_attrs().group_by("t"):
        centres[int(t)] = frame.select("z", "y", "x").to_numpy()
        for row, node in enumerate(frame["node_id"].to_list()):
            row_of[int(node)] = (int(t), row)

    if graph.num_edges() == 0:
        return centres, row_of, np.empty((0, 3))

    edges = graph.edge_attrs()
    # A geff written by a linker that keeps no score still has edges worth
    # drawing; they are simply all equally confident.
    probability = (
        edges["edge_prob"].to_numpy().astype(float)
        if "edge_prob" in edges.columns
        else np.ones(len(edges))
    )
    return centres, row_of, np.stack(
        [
            edges["source_id"].to_numpy().astype(float),
            edges["target_id"].to_numpy().astype(float),
            probability,
        ],
        axis=1,
    )


@st.cache_data(show_spinner=False)
def load_predictions(experiment, sample):
    """{t: (n, 3) centres as (z, y, x)} from an experiment's geff file."""
    return load_prediction_graph(experiment, sample)[0]


@st.cache_data(show_spinner=False)
def load_ground_truth(sample):
    """Ground-truth centres per timepoint, plus what is needed to walk a lineage back.

    Returns the centres keyed by timepoint, every node's (z, y, x) keyed by node
    id, the node ids present at each timepoint in the same order as the centres,
    and each node's parent.
    """
    graph = _load_geff(TRAIN_DIR / f"{sample}.geff")
    if graph is None:
        # Test-split samples have a stack but nothing annotated on it.
        return {}, {}, {}, {}

    nodes = graph.node_attrs()
    centres, ids = {}, {}
    for (t,), frame in nodes.group_by("t"):
        centres[int(t)] = frame.select("z", "y", "x").to_numpy()
        ids[int(t)] = frame["node_id"].to_numpy()

    position = {
        int(row["node_id"]): np.array([row["z"], row["y"], row["x"]], dtype=float)
        for row in nodes.iter_rows(named=True)
    }

    edges = graph.edge_attrs()
    parent = dict(zip(edges["target_id"].to_list(), edges["source_id"].to_list()))

    return centres, position, ids, parent


@st.cache_data(show_spinner=False)
def _estimated_nodes():
    """sample_name -> estimated_number_of_nodes, from the dataset metadata."""
    import pandas as pd

    df = pd.read_csv(ROOT / "input" / "metadata.csv")
    return dict(zip(df["sample_name"], df["estimated_number_of_nodes"].astype(float)))


@st.cache_data(show_spinner=False)
def sample_metrics(experiment, sample):
    """This sample's competition metrics for an experiment, or None if it can't be scored.

    Computed straight off the geffs with the same metric predict_track.sh
    scores with, rather than read from a scores_*.json, so every experiment
    can be scored here, not just the one that last had a json exported for it.

    Returns node_recall, edge_tp/fp/fn, edge_jaccard and adj_edge_jaccard (the
    node-count-penalised score); adj_edge_jaccard is NaN where the node-count
    metadata is missing, in which case edge_jaccard is the score to read.
    """
    gt_graph = _load_geff(TRAIN_DIR / f"{sample}.geff")
    pred_graph = _load_geff(PREDICTIONS_DIR / experiment / f"{sample}.geff")
    if gt_graph is None or pred_graph is None or gt_graph.num_nodes() == 0:
        return None

    from tracking_cellmot.metrics import evaluate as evaluate_graph, node_recall, per_sample_metrics

    n_total = _estimated_nodes().get(sample, float("nan"))
    result = evaluate_graph(pred_graph, gt_graph, scale=SCALE, max_distance=MAX_MATCH_DISTANCE)
    return per_sample_metrics(result, n_total, node_recall(pred_graph, gt_graph))


@st.cache_data(show_spinner=False)
def truth_frames(sample):
    """Timepoints that have at least one annotated ground-truth cell."""
    return sorted(t for t, centres in load_ground_truth(sample)[0].items() if len(centres))


@st.cache_data(show_spinner=False)
def truth_zs(sample, t):
    """z slices holding an annotated cell at t."""
    centres = load_ground_truth(sample)[0].get(t, np.empty((0, 3)))
    return sorted({int(round(z)) for z in centres[:, 0]})


@st.cache_data(show_spinner=False)
def split_by_match(experiment, sample, t):
    """Ground-truth and predicted centres of one timepoint, split into matched and not.

    The ground truth is sparse, so most predictions are unmatched simply because
    nothing was annotated there; it is the unmatched *ground truth* that means a
    cell was missed.
    """
    truth = load_ground_truth(sample)[0].get(t, np.empty((0, 3)))
    seeds = load_predictions(experiment, sample).get(t, np.empty((0, 3)))
    if len(truth) == 0 or len(seeds) == 0:
        return np.empty((0, 3)), truth, np.empty((0, 3)), seeds

    _, matched_gt, unmatched_gt = match_frame_nodes(
        seeds, truth, t=t, scale=SCALE, max_distance=MAX_MATCH_DISTANCE
    )
    matched_gt = np.asarray(matched_gt, dtype=float).reshape(-1, 3)
    unmatched_gt = np.asarray(unmatched_gt, dtype=float).reshape(-1, 3)

    # One prediction per matched ground-truth cell, the nearest one, rather than
    # every prediction that happens to fall within the matching distance: a
    # ground-truth cell was found once, not five times.
    scale = np.asarray(SCALE)
    matched = np.zeros(len(seeds), dtype=bool)
    if len(matched_gt):
        distances, nearest = cKDTree(seeds * scale).query(matched_gt * scale)
        matched[nearest[distances <= MAX_MATCH_DISTANCE]] = True
    return matched_gt, unmatched_gt, seeds[matched], seeds[~matched]


@st.cache_data(show_spinner=False)
def predicted_links(experiment, sample):
    """The predicted edges as, per timepoint, the parent of each cell in it.

    Keyed by the *child* timepoint: ``parents[t][i]`` is the row at t - 1 that
    cell i of frame t was linked to, or -1 where nothing linked to it, and
    ``probability[t][i]`` is that link's score.

    Built in one pass over the edge list rather than per frame pair: the whole
    graph is already in memory, and walking it once per timepoint would make
    scrubbing t quadratic. A cell given more than one parent keeps the most
    confident of them, which is the one a single trail can honestly draw.
    """
    centres, row_of, edges = load_prediction_graph(experiment, sample)
    parents = {t: np.full(len(c), -1, dtype=int) for t, c in centres.items()}
    probability = {t: np.zeros(len(c)) for t, c in centres.items()}

    for source, target, prob in edges:
        parent_at = row_of.get(int(source))
        child_at = row_of.get(int(target))
        if parent_at is None or child_at is None:
            continue
        t, row = child_at
        # Only consecutive frames: a gap-closing edge has no frame pair to be
        # drawn on, and stepping over it would draw a trail through a cell
        # that was never there.
        if parent_at[0] != t - 1:
            continue
        if parents[t][row] < 0 or prob > probability[t][row]:
            parents[t][row] = parent_at[1]
            probability[t][row] = prob

    return parents, probability


def parents_of_frame(experiment, sample, t):
    """For each cell at t + 1, the index of the cell at t it came from, or -1."""
    parents = predicted_links(experiment, sample)[0]
    return parents.get(t + 1, np.empty(0, dtype=int))


def predicted_trails(experiment, sample, t, length, keep):
    """Where each of the kept cells at t was over the previous frames."""
    predictions = load_predictions(experiment, sample)
    centres = predictions.get(t)
    if centres is None or length < 1:
        return []

    paths = {int(i): [centres[i]] for i in keep}
    alive = {int(i): int(i) for i in keep}
    for step in range(length):
        earlier = t - step - 1
        if earlier < 0 or earlier not in predictions or not alive:
            break
        parent = parents_of_frame(experiment, sample, earlier)
        moved = {}
        for start, current in alive.items():
            if current < len(parent) and parent[current] >= 0:
                moved[start] = int(parent[current])
                paths[start].append(predictions[earlier][moved[start]])
        alive = moved

    return [np.asarray(path) for path in paths.values() if len(path) > 1]


@st.cache_data(show_spinner=False)
def detections_of_truth(experiment, sample, t):
    """For each ground-truth cell at t, the prediction that found it, or -1.

    Which cells count as found is decided by the same assignment split_by_match
    uses, so the rings drawn on the slice and the verdicts drawn along the
    lineages cannot disagree.
    """
    truth = load_ground_truth(sample)[0].get(t, np.empty((0, 3)))
    found = np.full(len(truth), -1, dtype=int)
    matched_gt = split_by_match(experiment, sample, t)[0]
    seeds = load_predictions(experiment, sample).get(t, np.empty((0, 3)))
    if len(found) == 0 or len(matched_gt) == 0 or len(seeds) == 0:
        return found

    scale = np.asarray(SCALE)
    distances, nearest = cKDTree(seeds * scale).query(truth * scale)
    was_matched = {tuple(centre) for centre in matched_gt}
    for i, centre in enumerate(truth):
        if tuple(centre) in was_matched and distances[i] <= MAX_MATCH_DISTANCE:
            found[i] = nearest[i]
    return found


def _verdict(experiment, sample, t, parent_row, child_row):
    """What became of one ground-truth edge, in the words of utils.plot.link_errors.

    Returns the verdict, and where we linked the parent instead when we linked
    it somewhere: an edge is only wrong in a way worth drawing if there is
    another edge to draw in its place.
    """
    found_parent = detections_of_truth(experiment, sample, t)[parent_row]
    found_child = detections_of_truth(experiment, sample, t + 1)[child_row]
    if found_parent < 0 or found_child < 0:
        return "undetected", None

    children = np.flatnonzero(parents_of_frame(experiment, sample, t) == found_parent)
    if found_child in children:
        return "correct", None
    if len(children) == 0:
        return "no link", None

    # Where a division was predicted, draw the arm we were most sure of.
    probability = predicted_links(experiment, sample)[1].get(t + 1)
    instead = children[int(np.argmax(probability[children]))]

    predictions = load_predictions(experiment, sample)
    return "wrong partner", np.stack(
        [predictions[t][found_parent], predictions[t + 1][instead]]
    )


def truth_trails(experiment, sample, t, length, keep):
    """The ground-truth lineages behind the kept cells, every step of them judged.

    A step of a lineage is a ground-truth edge, and its colour says what we made
    of it: green where we reproduced it, and otherwise the reason we did not.
    Returned grouped by verdict, along with the links we made in place of the
    ones we missed, which is where a trajectory went wrong rather than merely
    that it did.
    """
    _, position, ids, parent = load_ground_truth(sample)
    segments = {kind: [] for kind in VERDICT_COLORS}
    wrong = []
    if t not in ids or length < 1:
        return segments, wrong

    row_of = {
        int(node): (frame, i)
        for frame, nodes in ids.items()
        for i, node in enumerate(nodes)
    }

    for i in keep:
        child = int(ids[t][i])
        for _ in range(length):
            source = parent.get(child)
            if source not in position:
                break

            step = np.stack([position[source], position[child]])
            (source_t, source_row) = row_of.get(source, (None, None))
            (child_t, _) = row_of.get(child, (None, None))
            if experiment is None:
                segments["unjudged"].append(step)
            elif source_t is None or child_t != source_t + 1:
                # An edge spanning a gap in the annotation was never ours to get.
                segments["undetected"].append(step)
            else:
                kind, instead = _verdict(experiment, sample, source_t, source_row, row_of[child][1])
                segments[kind].append(step)
                if instead is not None:
                    wrong.append(instead)
            child = source

    return segments, wrong


@st.cache_data(show_spinner=False)
def edge_mistake_frames(experiment, sample):
    """Timepoints t whose GT edge into t + 1 was not reproduced, or where a
    wrong link was drawn in its place -- an edge FN or FP, judged the same way
    truth_trails colours its lineages, just walked over every GT edge instead
    of only the ones kept for drawing.
    """
    _, _, ids, parent = load_ground_truth(sample)
    row_of = {
        int(node): (frame, i)
        for frame, nodes in ids.items()
        for i, node in enumerate(nodes)
    }
    mistakes = set()
    for child, source in parent.items():
        child_t, child_row = row_of.get(int(child), (None, None))
        source_t, source_row = row_of.get(int(source), (None, None))
        if child_t is None or source_t is None or child_t != source_t + 1:
            continue  # a gap in the annotation, not ours to score
        kind, instead = _verdict(experiment, sample, source_t, source_row, child_row)
        if kind != "correct":
            mistakes.add(child_t)  # edge FN
        if instead is not None:
            mistakes.add(child_t)  # edge FP: linked here instead
    return sorted(mistakes)


def _near(centres, z, tol):
    """Indices of the centres within tol slices of z, and how faded each should be."""
    if len(centres) == 0:
        return np.empty(0, dtype=int), np.empty(0)
    offset = np.abs(centres[:, 0] - z)
    keep = np.flatnonzero(offset <= tol)
    fade = 1.0 - 0.6 * (offset[keep] / max(tol, 1))
    return keep, fade


def _rgba(colour, alpha):
    """A named colour as a plotly rgba string, or one string per alpha given."""
    red, green, blue = (int(255 * channel) for channel in to_rgb(colour))
    if np.ndim(alpha) == 0:
        return f"rgba({red},{green},{blue},{alpha:.3f})"
    return [f"rgba({red},{green},{blue},{a:.3f})" for a in np.clip(alpha, 0.0, 1.0)]


@st.cache_data(show_spinner=False)
def slice_png(sample, t, z):
    """One slice as a data URI, contrast-stretched, for plotly to lay under the cells.

    Sent as a PNG rather than as a heatmap of raw values: forty kilobytes go
    down the wire on every move of a slider instead of a quarter of a megabyte,
    which is the difference between a view that keeps up and one that stutters.
    """
    low, high = contrast_limits(sample)
    plane = load_volume(sample, t)[z].astype(np.float32)
    grey = np.clip((plane - low) / max(high - low, 1e-6), 0.0, 1.0)
    buffer = io.BytesIO()
    Image.fromarray((grey * 255).astype(np.uint8)).save(buffer, format="png")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _points(figure, centres, keep, fade, label, face, edge, size, width=DOT_WIDTH):
    """Add the kept centres, each faded by how far off the slice it is."""
    if len(keep) == 0:
        return
    figure.add_trace(
        go.Scatter(
            x=centres[keep, 2],
            y=centres[keep, 1],
            mode="markers",
            name=f"{label} ({len(keep)})",
            customdata=centres[keep, 0],
            marker=dict(
                size=size,
                color="rgba(0,0,0,0)" if face == "none" else _rgba(face, fade),
                line=dict(width=width, color=_rgba(edge, fade)),
            ),
            hovertemplate="x %{x:.0f}   y %{y:.0f}   z %{customdata:.0f}<extra></extra>",
        )
    )


def _trails(figure, paths, colour, name=None, alpha=0.45, width=TRAIL_WIDTH):
    """Add the paths as one trace, separated by gaps, named only if worth a legend."""
    if not paths:
        return
    x, y = [], []
    for path in paths:
        x.extend([*path[:, 2].tolist(), None])
        y.extend([*path[:, 1].tolist(), None])
    figure.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines", hoverinfo="skip",
            name=name or "", showlegend=name is not None,
            line=dict(color=_rgba(colour, alpha), width=width),
        )
    )


def busiest_slice(experiment, sample, t, depth):
    """A z worth opening on: the middle of wherever the cells are, not of the stack."""
    if experiment is not None:
        centres = load_predictions(experiment, sample).get(t)
        if centres is not None and len(centres):
            return int(np.clip(np.median(centres[:, 0]), 0, depth - 1))
    centres = load_ground_truth(sample)[0].get(t)
    if centres is not None and len(centres):
        return int(np.clip(np.median(centres[:, 0]), 0, depth - 1))
    return depth // 2


def render(sample, experiment, t, z, tol, trail, show_truth, show_trails):
    """Draw one slice with everything asked for on top of it."""
    height, width = load_volume(sample, t).shape[1:]
    figure = go.Figure()

    counts = {}

    if show_truth and experiment is not None:
        matched_gt, unmatched_gt, matched_pred, unmatched_pred = split_by_match(
            experiment, sample, t
        )
        counts["gt matched"] = len(matched_gt)
        counts["gt missed"] = len(unmatched_gt)

        for centres, colour, edge, label in (
            (unmatched_gt, GT_UNMATCHED, GT_UNMATCHED_EDGE, "GT unmatched"),
            (matched_gt, GT_MATCHED, GT_MATCHED_EDGE, "GT matched"),
        ):
            keep, fade = _near(centres, z, tol)
            _points(figure, centres, keep, fade, label, face=colour, edge=edge,
                    size=GT_MARKER, width=RING_WIDTH)

        keep, fade = _near(unmatched_pred, z, tol)
        _points(figure, unmatched_pred, keep, fade * 0.35, f"{experiment} unmatched",
                face=PRED_UNMATCHED, edge="red", size=FAINT_MARKER)
        keep, fade = _near(matched_pred, z, tol)
        _points(figure, matched_pred, keep, fade, f"{experiment} matched",
                face=PRED_MATCHED, edge=PRED_MATCHED_EDGE, size=PRED_MARKER)
    else:
        # Nothing to match against, or nothing to match: draw what there is.
        if show_truth:
            centres = load_ground_truth(sample)[0].get(t, np.empty((0, 3)))
            keep, fade = _near(centres, z, tol)
            _points(figure, centres, keep, fade, "GT", face=GT_MATCHED,
                    edge=GT_MATCHED_EDGE, size=GT_MARKER, width=RING_WIDTH)
            counts["gt"] = len(centres)
        if experiment is not None:
            centres = load_predictions(experiment, sample).get(t, np.empty((0, 3)))
            keep, fade = _near(centres, z, tol)
            _points(figure, centres, keep, fade, experiment, face=PRED_MATCHED,
                    edge=PRED_MATCHED_EDGE, size=PRED_MARKER)

    if show_trails:
        if experiment is not None:
            annotated = bool(load_ground_truth(sample)[0])
            centres = load_predictions(experiment, sample).get(t, np.empty((0, 3)))
            keep, _ = _near(centres, z, tol)
            if annotated:
                # Only the cells that found an annotated one: the rest are so
                # many that their tracks bury the slice.
                found = detections_of_truth(experiment, sample, t)
                keep = np.intersect1d(keep, found[found >= 0])
            _trails(figure, predicted_trails(experiment, sample, t, trail, keep), PRED_MATCHED_EDGE,
                    name=f"{experiment} track")

        if show_truth:
            centres = load_ground_truth(sample)[0].get(t, np.empty((0, 3)))
            keep, _ = _near(centres, z, tol)
            judged, instead = truth_trails(experiment, sample, t, trail, keep)
            for kind, segments in judged.items():
                _trails(
                    figure, segments, VERDICT_COLORS[kind],
                    name="GT lineage" if kind == "unjudged" else f"GT link: {kind}",
                    alpha=0.9, width=TRAIL_WIDTH * 1.3,
                )
            _trails(
                figure, instead, WRONG_LINK, name="linked here instead",
                alpha=0.9, width=TRAIL_WIDTH * 1.3,
            )

    # The slice goes under the traces as a background image rather than as data,
    # anchored to the pixel corners so a cell at (y, x) lands on pixel (y, x).
    figure.add_layout_image(
        dict(
            source=slice_png(sample, t, z), xref="x", yref="y",
            x=-0.5, y=-0.5, sizex=width, sizey=height,
            xanchor="left", yanchor="top", sizing="stretch", layer="below",
        )
    )
    figure.update_xaxes(range=[-0.5, width - 0.5], visible=False, constrain="domain")
    figure.update_yaxes(
        range=[height - 0.5, -0.5], visible=False, scaleanchor="x", scaleratio=1,
        constrain="domain",
    )
    # Height alone, so that a window too narrow for the full size shrinks the
    # view to fit instead of pushing the controls off the side of the page.
    figure.update_layout(
        height=VIEW_PIXELS,
        margin=dict(l=0, r=LEGEND_PIXELS, t=0, b=0),
        paper_bgcolor="black",
        plot_bgcolor="black",
        dragmode="pan",
        hovermode="closest",
        legend=dict(
            x=1.0, xanchor="left", y=1.0, yanchor="top",
            bgcolor="rgba(0,0,0,0)", font=dict(size=12, color="white"),
            # Otherwise the legend draws the markers at their size on the image,
            # which at this scale fills it with circles.
            itemsizing="constant",
        ),
        # Keeps the zoom and pan across reruns, so scrubbing t or z while zoomed
        # into a cluster stays where you left it, and only a new sample resets it.
        uirevision=sample,
    )
    return figure, counts


def _step(key, options, delta):
    """Move a slider one option along, from the button beside it."""
    index = options.index(st.session_state[key]) + delta
    st.session_state[key] = options[int(np.clip(index, 0, len(options) - 1))]


def _stepper(label, options, default, key, back, forward):
    """A slider with a button either side of it, each moving it exactly one step.

    The buttons take the arrow keys as well, so a stack can be walked through
    from the keyboard, one frame or one slice per press.
    """
    # The starting position goes into the session rather than being passed as a
    # default, since the buttons write there too and streamlit objects to a
    # widget being given its value both ways.
    st.session_state.setdefault(key, default)

    minus, track, plus = st.columns([1, 7, 1], vertical_alignment="bottom")
    with track:
        chosen = st.select_slider(label, options=options, key=key)

    index = options.index(chosen)
    minus.button(
        "−", key=f"{key} back", width="stretch", disabled=index == 0,
        shortcut=back, help=f"{label} − 1   ({back} arrow)",
        on_click=_step, args=(key, options, -1),
    )
    plus.button(
        "+", key=f"{key} forward", width="stretch", disabled=index == len(options) - 1,
        shortcut=forward, help=f"{label} + 1   ({forward} arrow)",
        on_click=_step, args=(key, options, 1),
    )
    return chosen


@st.fragment
def view(sample, experiment, frames, depth, tol, trail, show_truth, show_trails):
    """The stack and the two sliders that move through it.

    A fragment, so that moving a slider reruns this and nothing else: the page
    around it is left standing and the chart is handed new data in place rather
    than being torn down and rebuilt, which is what made it blink before.
    """
    display, controls = st.columns(DISPLAY_SHARE)

    with controls:
        if experiment is not None:
            metrics = sample_metrics(experiment, sample)
            score = metrics.get("adj_edge_jaccard") if metrics else None
            # Node-count metadata is missing for some samples; the un-adjusted
            # Jaccard is still a real score, just without the penalty applied.
            if score is None or score != score:
                score = metrics.get("edge_jaccard") if metrics else None
            st.markdown(" ")
            st.markdown(f"#### Score : {np.round(score, 3) if score is not None else '—'} ")
            if metrics:
                st.caption(
                    f"node recall: {metrics['node_recall']:.3f}  ·  "
                    f"edge jaccard: {metrics['edge_jaccard']:.3f}"
                )
                st.caption(
                    f"edge tp: {metrics['edge_tp']}  ·  "
                    f"edge fp: {metrics['edge_fp']}  ·  "
                    f"edge fn: {metrics['edge_fn']}"
                )

        st.markdown("#### Navigate")
        # Keyed by sample, so that changing sample starts afresh at a sensible
        # frame and slice instead of holding a position that meant something in
        # the stack before it.
        t = _stepper("t", frames, frames[0], f"t {sample}", "Left", "Right")

        # Follow t to the nearest annotated slice, so scrubbing time keeps GT
        # in view instead of leaving z wherever it was for the last frame.
        # Only on an actual change of t, so dragging z doesn't fight back.
        z_key = f"z {sample}"
        synced_key = f"_z_synced_t {sample}"
        if st.session_state.get(synced_key) != t:
            st.session_state[synced_key] = t
            gt_zs = truth_zs(sample, t)
            if gt_zs:
                current_z = st.session_state.get(
                    z_key, busiest_slice(experiment, sample, t, depth)
                )
                st.session_state[z_key] = min(gt_zs, key=lambda candidate: abs(candidate - current_z))

        z = _stepper(
            "z", list(range(depth)), busiest_slice(experiment, sample, frames[0], depth),
            z_key, "Down", "Up",
        )

        st.markdown("#### Ground truth")
        gt_frames = truth_frames(sample)
        if gt_frames:
            if experiment is not None:
                bad_frames = edge_mistake_frames(experiment, sample)
                st.caption(
                    f"t with edge FP/FN ({len(bad_frames)}/{len(gt_frames)}): "
                    + (", ".join(map(str, bad_frames)) if bad_frames else "none")
                )
            else:
                st.caption(
                    f"t with GT ({len(gt_frames)}/{len(frames)}): "
                    + ", ".join(map(str, gt_frames))
                )
            gt_zs = truth_zs(sample, t)
            st.caption(
                f"z with GT at t={t}: " + ", ".join(map(str, gt_zs))
                if gt_zs else f"No GT at t={t}."
            )
        else:
            st.caption("No ground truth for this sample.")

    figure, counts = render(sample, experiment, t, z, tol, trail, show_truth, show_trails)

    with display:
        st.plotly_chart(
            figure, width="stretch", key="stack", theme=None,
            config={"scrollZoom": True, "displaylogo": False,
                    "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
        )

    with controls:
        if counts:
            st.caption("   ".join(f"{name}: {value}" for name, value in counts.items()))
        st.caption("Scroll to zoom, drag to pan, double-click to reset. "
                   "Click a legend entry to hide it.")


def main():
    st.set_page_config(page_title="Cell tracking viewer", layout="wide")
    # Streamlit leaves several centimetres above the first element, which here
    # is the view itself and wants all the height it can get.
    st.markdown(
        "<style>.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True
    )

    with st.sidebar:
        st.header("Predictions")
        if st.button("🔄 Refresh sample list",
                     help="Rescan output/predictions for new or changed .geff files."):
            list_experiments.clear()
            list_samples.clear()
            list_all_samples.clear()
            st.rerun()
        experiments = list_experiments()
        experiment = None
        if experiments:
            experiment = st.selectbox(
                "experiment", experiments,
                help="Predictions come from output/predictions/<experiment>/<sample>.geff. "
                     "The ground truth is matched against it and its cells split into "
                     "matched and unmatched.",
            )
        else:
            st.warning(f"No experiments found under {PREDICTIONS_DIR}.")

        st.header("Sample")
        samples = list_samples(experiment) if experiment else list_all_samples()
        if not samples:
            st.error(
                f"No samples with predictions in {experiment}." if experiment
                else f"No stacks found under {TRAIN_DIR}."
            )
            return
        sample = st.selectbox("sample_name", samples)

        st.header("Display")
        tol = st.slider("z tolerance (slices)", 0, 10, 5,
                        help="How far off the slice a cell can be and still be drawn, faded.")
        show_truth = st.checkbox("ground truth", value=True)
        show_trails = st.checkbox("tracks", value=True)
        trail = st.slider("track length (frames)", 1, 100, 100, disabled=not show_trails,
                          help="100 covers the full track: predicted_trails stops early on its own once a lineage runs out of parents.")

    frames = list_frames(sample)
    depth = load_volume(sample, frames[0]).shape[0]
    view(sample, experiment, frames, depth, tol, trail, show_truth, show_trails)


if __name__ == "__main__":
    main()
