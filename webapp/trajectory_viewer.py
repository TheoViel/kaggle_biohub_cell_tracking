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
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import to_rgb
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from params import SCALE  # noqa: E402
from utils.build_graph import link_frames  # noqa: E402
from utils.metrics import match_frame_nodes  # noqa: E402

IMAGE_DIR = ROOT / "input" / "processed" / "images"
NODE_DIR = ROOT / "input" / "processed" / "nodes"
EDGE_DIR = ROOT / "input" / "processed" / "edges"

MAX_MATCH_DISTANCE = 7.0

# How big the view is, in pixels, and how the page splits between it and the
# controls beside it. Resizing is this one number: the markers follow it, so
# that they keep ringing the cells rather than dotting them.
VIEW_PIXELS = 1240
DISPLAY_SHARE = (3, 1)
# Room to the right of the stack for the legend to stand in.
LEGEND_PIXELS = 190

_OF_460_PIXELS = VIEW_PIXELS / 460
# Wider than a prediction, so that the two stay apart when one sits on the other.
GT_MARKER = 11 * _OF_460_PIXELS
PRED_MARKER = 7 * _OF_460_PIXELS
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
OTHER_FOLDERS = ("gold", "violet", "darkorange", "aqua")

# What became of a ground-truth edge, in the colours utils.plot uses for the
# same verdicts, so a colour means the same thing here as in the notebook.
VERDICT_COLORS = {
    "correct": "limegreen",
    "wrong partner": "orangered",
    "no link": "goldenrod",
    "undetected": "slategray",
    "unjudged": "limegreen",
}
WRONG_LINK = "darkorange"


@st.cache_data(show_spinner=False)
def list_samples():
    """Sample names that have both a stack and ground-truth nodes."""
    stacks = {path.name.rsplit("_", 1)[0] for path in IMAGE_DIR.glob("*.npy")}
    labelled = {path.stem for path in NODE_DIR.glob("*.csv")}
    return sorted(stacks & labelled) or sorted(stacks)


@st.cache_data(show_spinner=False)
def list_frames(sample):
    """Timepoints available for a sample."""
    return sorted(int(path.stem.rsplit("_", 1)[1]) for path in IMAGE_DIR.glob(f"{sample}_*.npy"))


@st.cache_data(show_spinner=False)
def list_prediction_folders():
    """Folders holding predicted_nodes_*.json, as paths relative to the repository."""
    folders = {
        path.parent.relative_to(ROOT).as_posix()
        for path in ROOT.glob("*/predicted_nodes_*.json")
    }
    folders |= {
        path.parent.relative_to(ROOT).as_posix()
        for path in ROOT.glob("*/*/predicted_nodes_*.json")
    }
    return sorted(folders | {"output"})


@st.cache_data(show_spinner=False)
def load_volume(sample, t):
    """The (z, y, x) stack of one timepoint."""
    return np.load(IMAGE_DIR / f"{sample}_{t}.npy")


@st.cache_data(show_spinner=False)
def contrast_limits(sample, t, low=0.01, high=0.999):
    """Display range from quantiles, since the raw range is set by a few bright voxels."""
    flat = load_volume(sample, t).ravel()[::10]
    return float(np.quantile(flat, low)), float(np.quantile(flat, high))


@st.cache_data(show_spinner=False)
def load_predictions(folder, sample):
    """{t: (n, 3) centres as (z, y, x)} from a folder's predicted_nodes file."""
    path = ROOT / folder / f"predicted_nodes_{sample}.json"
    if not path.exists():
        return {}
    raw = json.load(open(path))
    return {int(t): np.asarray(centres, dtype=float).reshape(-1, 3) for t, centres in raw.items()}


@st.cache_data(show_spinner=False)
def load_ground_truth(sample):
    """Ground-truth centres per timepoint, plus what is needed to walk a lineage back.

    Returns the centres keyed by timepoint, every node's (z, y, x) keyed by node
    id, the node ids present at each timepoint in the same order as the centres,
    and each node's parent.
    """
    node_path = NODE_DIR / f"{sample}.csv"
    if not node_path.exists():
        # Test-split samples have a stack but nothing annotated on it.
        return {}, {}, {}, {}

    nodes = pd.read_csv(node_path)
    centres, ids = {}, {}
    for t, frame in nodes.groupby("t"):
        centres[int(t)] = frame[["z", "y", "x"]].to_numpy(dtype=float)
        ids[int(t)] = frame["node_id"].to_numpy()

    position = {
        int(row.node_id): np.array([row.z, row.y, row.x], dtype=float)
        for row in nodes.itertuples()
    }

    parent = {}
    edge_path = EDGE_DIR / f"{sample}.csv"
    if edge_path.exists():
        edges = pd.read_csv(edge_path)
        parent = dict(zip(edges["target_id"].astype(int), edges["source_id"].astype(int)))

    return centres, position, ids, parent


@st.cache_data(show_spinner=False)
def split_by_match(folder, sample, t):
    """Ground-truth and predicted centres of one timepoint, split into matched and not.

    The ground truth is sparse, so most predictions are unmatched simply because
    nothing was annotated there; it is the unmatched *ground truth* that means a
    cell was missed.
    """
    truth = load_ground_truth(sample)[0].get(t, np.empty((0, 3)))
    seeds = load_predictions(folder, sample).get(t, np.empty((0, 3)))
    if len(truth) == 0 or len(seeds) == 0:
        return np.empty((0, 3)), truth, seeds, np.empty((0, 3))

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
def parents_of_frame(folder, sample, t):
    """For each cell at t + 1, the index of the cell at t it came from, or -1.

    This runs the same linker the graph is built with, one frame pair at a time
    so that scrubbing through t only ever pays for the pair it just reached.
    """
    predictions = load_predictions(folder, sample)
    here, there = predictions.get(t), predictions.get(t + 1)
    if here is None or there is None or len(here) == 0 or len(there) == 0:
        return np.full(0 if there is None else len(there), -1, dtype=int)

    links, _ = link_frames(here, there, max_distance=MAX_MATCH_DISTANCE, scale=SCALE)
    parent = np.full(len(there), -1, dtype=int)
    for source, target in links:
        parent[target] = source
    return parent


def predicted_trails(folder, sample, t, length, keep):
    """Where each of the kept cells at t was over the previous frames."""
    predictions = load_predictions(folder, sample)
    centres = predictions.get(t)
    if centres is None or length < 1:
        return []

    paths = {int(i): [centres[i]] for i in keep}
    alive = {int(i): int(i) for i in keep}
    for step in range(length):
        earlier = t - step - 1
        if earlier < 0 or earlier not in predictions or not alive:
            break
        parent = parents_of_frame(folder, sample, earlier)
        moved = {}
        for start, current in alive.items():
            if current < len(parent) and parent[current] >= 0:
                moved[start] = int(parent[current])
                paths[start].append(predictions[earlier][moved[start]])
        alive = moved

    return [np.asarray(path) for path in paths.values() if len(path) > 1]


@st.cache_data(show_spinner=False)
def detections_of_truth(folder, sample, t):
    """For each ground-truth cell at t, the prediction that found it, or -1.

    Which cells count as found is decided by the same assignment split_by_match
    uses, so the rings drawn on the slice and the verdicts drawn along the
    lineages cannot disagree.
    """
    truth = load_ground_truth(sample)[0].get(t, np.empty((0, 3)))
    found = np.full(len(truth), -1, dtype=int)
    matched_gt = split_by_match(folder, sample, t)[0]
    seeds = load_predictions(folder, sample).get(t, np.empty((0, 3)))
    if len(found) == 0 or len(matched_gt) == 0 or len(seeds) == 0:
        return found

    scale = np.asarray(SCALE)
    distances, nearest = cKDTree(seeds * scale).query(truth * scale)
    was_matched = {tuple(centre) for centre in matched_gt}
    for i, centre in enumerate(truth):
        if tuple(centre) in was_matched and distances[i] <= MAX_MATCH_DISTANCE:
            found[i] = nearest[i]
    return found


def _verdict(folder, sample, t, parent_row, child_row):
    """What became of one ground-truth edge, in the words of utils.plot.link_errors.

    Returns the verdict, and where we linked the parent instead when we linked
    it somewhere: an edge is only wrong in a way worth drawing if there is
    another edge to draw in its place.
    """
    found_parent = detections_of_truth(folder, sample, t)[parent_row]
    found_child = detections_of_truth(folder, sample, t + 1)[child_row]
    if found_parent < 0 or found_child < 0:
        return "undetected", None

    children = np.flatnonzero(parents_of_frame(folder, sample, t) == found_parent)
    if found_child in children:
        return "correct", None
    if len(children) == 0:
        return "no link", None

    predictions = load_predictions(folder, sample)
    return "wrong partner", np.stack(
        [predictions[t][found_parent], predictions[t + 1][children[0]]]
    )


def truth_trails(folder, sample, t, length, keep):
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
            if folder is None:
                segments["unjudged"].append(step)
            elif source_t is None or child_t != source_t + 1:
                # An edge spanning a gap in the annotation was never ours to get.
                segments["undetected"].append(step)
            else:
                kind, instead = _verdict(folder, sample, source_t, source_row, row_of[child][1])
                segments[kind].append(step)
                if instead is not None:
                    wrong.append(instead)
            child = source

    return segments, wrong


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
    low, high = contrast_limits(sample, t)
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


def busiest_slice(folders, sample, t, depth):
    """A z worth opening on: the middle of wherever the cells are, not of the stack."""
    for folder in folders:
        centres = load_predictions(folder, sample).get(t)
        if centres is not None and len(centres):
            return int(np.clip(np.median(centres[:, 0]), 0, depth - 1))
    centres = load_ground_truth(sample)[0].get(t)
    if centres is not None and len(centres):
        return int(np.clip(np.median(centres[:, 0]), 0, depth - 1))
    return depth // 2


def render(sample, folders, t, z, tol, trail, show_truth, show_trails):
    """Draw one slice with everything asked for on top of it."""
    height, width = load_volume(sample, t).shape[1:]
    figure = go.Figure()

    reference = folders[0] if folders else None
    counts = {}

    if show_truth and reference is not None:
        matched_gt, unmatched_gt, matched_pred, unmatched_pred = split_by_match(
            reference, sample, t
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
        _points(figure, unmatched_pred, keep, fade * 0.35, f"{reference} unmatched",
                face=PRED_UNMATCHED, edge="red", size=FAINT_MARKER)
        keep, fade = _near(matched_pred, z, tol)
        _points(figure, matched_pred, keep, fade, f"{reference} matched",
                face=PRED_MATCHED, edge=PRED_MATCHED_EDGE, size=PRED_MARKER)
    else:
        # Nothing to match against, or nothing to match: draw what there is.
        if show_truth:
            centres = load_ground_truth(sample)[0].get(t, np.empty((0, 3)))
            keep, fade = _near(centres, z, tol)
            _points(figure, centres, keep, fade, "GT", face=GT_MATCHED,
                    edge=GT_MATCHED_EDGE, size=GT_MARKER, width=RING_WIDTH)
            counts["gt"] = len(centres)
        if reference is not None:
            centres = load_predictions(reference, sample).get(t, np.empty((0, 3)))
            keep, fade = _near(centres, z, tol)
            _points(figure, centres, keep, fade, reference, face=PRED_MATCHED,
                    edge=PRED_MATCHED_EDGE, size=PRED_MARKER)

    # Any further folder is drawn plainly, in its own colour, for comparison.
    for folder, colour in zip(folders[1:], OTHER_FOLDERS):
        centres = load_predictions(folder, sample).get(t, np.empty((0, 3)))
        keep, fade = _near(centres, z, tol)
        _points(figure, centres, keep, fade, folder, face=colour, edge="white",
                size=PRED_MARKER)
        counts[folder] = len(keep)

    if show_trails:
        annotated = bool(load_ground_truth(sample)[0])
        for folder, colour in zip(folders, (PRED_MATCHED_EDGE, *OTHER_FOLDERS)):
            centres = load_predictions(folder, sample).get(t, np.empty((0, 3)))
            keep, _ = _near(centres, z, tol)
            if annotated:
                # Only the cells that found an annotated one: the rest are so
                # many that their tracks bury the slice.
                found = detections_of_truth(folder, sample, t)
                keep = np.intersect1d(keep, found[found >= 0])
            _trails(figure, predicted_trails(folder, sample, t, trail, keep), colour,
                    name=f"{folder} track")

        if show_truth:
            centres = load_ground_truth(sample)[0].get(t, np.empty((0, 3)))
            keep, _ = _near(centres, z, tol)
            judged, instead = truth_trails(reference, sample, t, trail, keep)
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
def view(sample, folders, frames, depth, tol, trail, show_truth, show_trails):
    """The stack and the two sliders that move through it.

    A fragment, so that moving a slider reruns this and nothing else: the page
    around it is left standing and the chart is handed new data in place rather
    than being torn down and rebuilt, which is what made it blink before.
    """
    display, controls = st.columns(DISPLAY_SHARE)

    with controls:
        st.markdown("#### Navigate")
        # Keyed by sample, so that changing sample starts afresh at a sensible
        # frame and slice instead of holding a position that meant something in
        # the stack before it.
        t = _stepper("t", frames, frames[0], f"t {sample}", "Left", "Right")
        z = _stepper(
            "z", list(range(depth)), busiest_slice(folders, sample, frames[0], depth),
            f"z {sample}", "Down", "Up",
        )

    figure, counts = render(sample, folders, t, z, tol, trail, show_truth, show_trails)

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

    samples = list_samples()
    if not samples:
        st.error(f"No stacks found under {IMAGE_DIR}.")
        return

    with st.sidebar:
        st.header("Sample")
        with_predictions = [s for s in samples if (ROOT / "output" / f"predicted_nodes_{s}.json").exists()]
        sample = st.selectbox(
            "sample_name", samples,
            index=samples.index(with_predictions[0]) if with_predictions else 0,
        )

        st.header("Predictions")
        folders = st.multiselect(
            "folders", list_prediction_folders(), default=["output"],
            help="The first one is the reference: the ground truth is matched against it "
                 "and its cells are split into matched and unmatched. The rest are drawn plainly.",
        )
        missing = [f for f in folders if not (ROOT / f / f"predicted_nodes_{sample}.json").exists()]
        if missing:
            st.warning(f"No predictions for {sample} in: {', '.join(missing)}")

        st.header("Display")
        tol = st.slider("z tolerance (slices)", 0, 10, 3,
                        help="How far off the slice a cell can be and still be drawn, faded.")
        show_truth = st.checkbox("ground truth", value=True)
        show_trails = st.checkbox("tracks", value=True)
        trail = st.slider("track length (frames)", 1, 40, 10, disabled=not show_trails)

    frames = list_frames(sample)
    depth = load_volume(sample, frames[0]).shape[0]
    view(sample, folders, frames, depth, tol, trail, show_truth, show_trails)


if __name__ == "__main__":
    main()
