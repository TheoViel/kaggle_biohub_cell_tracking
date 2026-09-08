import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tracksdata as td
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle

from cellpose.utils import masks_to_outlines
from utils.metrics import dist


SCALE = (1.625, 0.40625, 0.40625)

# Per-slice opacity multiplier for cells drawn off the plane being shown.
# A cell is ~9 um across and a z voxel is 1.625 um, so a nucleus spans about
# five slices: at 0.75 a cell two slices out is still clearly visible (0.56)
# and one six slices out has all but gone (0.18), which is the distinction the
# eye needs to make.
DEPTH_FALLOFF = 0.75

# One colour per verdict, reused by every error plot so a colour always means
# the same thing.
KINDS = ("correct", "wrong partner", "no link", "undetected")
KIND_COLORS = {
    "correct": "limegreen",
    "wrong partner": "orangered",
    "no link": "goldenrod",
    "undetected": "slategray",
}


def depth_rgba(dz, color, falloff=DEPTH_FALLOFF, base_alpha=1.0):
    """``color`` as one RGBA row per point, faded by distance from the plane.

    Args:
        dz (array): Signed or unsigned slice offset of each point.
        color: Any matplotlib colour.
        falloff (float, optional): Opacity multiplier per slice.
        base_alpha (float, optional): Opacity of a point sitting on the plane.

    Returns:
        np.ndarray: (N, 4) RGBA, alpha = base_alpha * falloff ** |dz|.
    """
    dz = np.abs(np.asarray(dz, dtype=float)).ravel()
    alpha = np.clip(base_alpha * falloff ** dz, 0.0, 1.0)
    red, green, blue = to_rgba(color)[:3]
    return np.column_stack([
        np.full(len(alpha), red),
        np.full(len(alpha), green),
        np.full(len(alpha), blue),
        alpha,
    ])


def scatter_by_depth(
    ax,
    points,
    z,
    color,
    tol=10,
    falloff=DEPTH_FALLOFF,
    base_alpha=1.0,
    facecolor="none",
    s=40,
    linewidths=1.2,
    **kwargs,
):
    """Draw cell centres over one z-slice, fading them out with depth.

    Drawing every cell within a window of the plane at full opacity makes a
    dense frame unreadable: a cell six slices away is painted as boldly as the
    one you are looking at, so the picture stops saying which cells are
    actually in this section. Fading geometrically with |dz| keeps the
    out-of-plane context visible without letting it compete.

    Points are drawn nearest first. That is for the legend, which takes its
    swatch from the first point: ordered the other way round, every legend
    entry comes out in the faintest colour in the set. Far points then land on
    top of near ones, which is invisible in practice because they are the
    transparent ones.

    Args:
        ax: Axes to draw on. The ``pyplot`` module itself also works.
        points (array): (N, 3) cell centres as (z, y, x), in voxels.
        z (float): The plane being displayed.
        color: Edge colour, faded per point.
        tol (float, optional): Slices either side of ``z`` to draw at all.
        falloff (float, optional): Opacity multiplier per slice.
        base_alpha (float, optional): Opacity of a point on the plane. Set it
            here rather than passing ``alpha``, which matplotlib applies on top
            of the per-point colours and so flattens the fade away.
        facecolor (optional): Fill colour, faded the same way. ``"none"``
            leaves the markers hollow, which is what keeps the image visible
            underneath them.
        s, linewidths, **kwargs: Passed to ``scatter``.

    Returns:
        np.ndarray: (N,) bool, which of ``points`` were drawn.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(points):
        return np.zeros(0, dtype=bool)

    dz = points[:, 0] - float(z)
    near = np.abs(dz) <= tol
    if not near.any():
        return near

    order = np.argsort(np.abs(dz[near]))
    drawn = points[near][order]
    dz = dz[near][order]

    edge = depth_rgba(dz, color, falloff, base_alpha)
    if not MarkerStyle(kwargs.get("marker", "o")).is_filled():
        # An unfilled marker ("x", "+", ...) is drawn from its face colour and
        # matplotlib warns if an edge colour is supplied as well.
        colors = dict(facecolors=edge)
    else:
        colors = dict(
            edgecolors=edge,
            facecolors=(
                "none" if facecolor is None or facecolor == "none"
                else depth_rgba(dz, facecolor, falloff, base_alpha)
            ),
        )

    ax.scatter(
        drawn[:, 2],
        drawn[:, 1],
        s=s,
        linewidths=linewidths,
        **colors,
        **kwargs,
    )
    return near


def plot_frame(img, nodes, t=None, z=None, mask=None, tol=5):
    """
    Plot a single frame of an image with nodes and mask.

    Args:
        img (np.ndarray): The image to plot.
        nodes (pd.DataFrame): The nodes to plot.
        t (int, Optional): The frame index. Defaults to None.
        z (int, Optional): The z-plane index. Defaults to None.
        mask (np.ndarray, Optional): The mask to plot. Defaults to None.
        tol (int, Optional): The tolerance for the z-plane index. Defaults to 5.
    """
    # Restrict to exact frame
    if t is not None:
        nodes = nodes[nodes["t"] == t]
        if img.ndim == 4:
            img = img[t]
    
    # Restrict to exact z-plane
    if z is not None:
        nodes = nodes[abs(nodes["z"] - z) <= tol]
        if img.ndim == 3:
            img = img[z]
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[z]

    plt.imshow(img, cmap="gray")

    if mask is not None:
        mask = np.where(mask == 0, np.nan, mask)
        plt.imshow(mask, cmap="hot", alpha=0.5)

    if z is None:
        # A projection over every plane: there is no depth to fade against.
        plt.scatter(
            nodes["x"], nodes["y"], s=20, c="#eb6834",
            edgecolors="white", linewidths=0.5, zorder=5,
        )
    else:
        scatter_by_depth(
            plt,
            nodes[["z", "y", "x"]].to_numpy(dtype=float),
            z,
            "#eb6834",
            tol=tol,
            facecolor="#eb6834",
            s=20,
            linewidths=0.5,
            zorder=5,
        )
    plt.title(f"t = {t}, z = {z}")
    plt.grid(False)
    plt.axis(False)



def plot_contours(img, mask, color=(255, 0, 0)):
    """Show an image with its mask outlines drawn in red on top.

    Args:
        img (np.ndarray): (H, W) or (H, W, 1) image, gray or uint16.
        mask (np.ndarray): (H, W) label mask, one integer id per cell.
        color (tuple, optional): RGB colour for the outlines.
    """
    img = np.asarray(img).squeeze().astype(np.float32)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    img = (img * 255).astype(np.uint8)
    img = np.stack([img] * 3, axis=-1)

    outlines = masks_to_outlines(mask)
    img[outlines] = color

    plt.imshow(img)


def show_cellpose_segmentation(img, maski, flowi, nodes=None, z=None):
    """Image, predicted outlines and flow side by side, with cell centres.

    ``nodes`` is (N, 3) as (z, y, x). Pass ``z`` -- the plane ``img`` is a slice
    of -- to fade the centres by how far they sit from it; without it they are
    all drawn solid.
    """
    img0 = img.copy().squeeze()
    img0 = img0[..., None].repeat(3, -1)

    plt.subplot(1, 3, 1)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title("original image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plot_contours(img, maski)
    plt.title("predicted outlines")
    plt.axis("off")

    def _nodes():
        """The centres, faded by depth when the plane they sit on is known."""
        if nodes is None:
            return
        if z is None:
            plt.scatter(
                nodes[:, 2], nodes[:, 1], s=20, c="dodgerblue",
                edgecolors="white", linewidths=0.5, zorder=5,
            )
        else:
            scatter_by_depth(
                plt, nodes, z, "white", facecolor="dodgerblue",
                s=20, linewidths=0.5, zorder=5,
            )

    _nodes()

    plt.subplot(1, 3, 3)
    plt.imshow(flowi)

    _nodes()
    plt.title("predicted cell pose")
    plt.axis("off")


# def lineage(graph, root):
#     """
#     Collect everything reachable from a node by following edges forward in time.

#     A trajectory is a tree rather than a chain, since a dividing cell has two
#     children, so this returns the whole subtree below root.

#     Args:
#         graph (tracksdata graph): Graph whose nodes carry t, z, y, x.
#         root (int): Node id to start from.

#     Returns:
#         tuple: The node table as a DataFrame indexed by node id and ordered
#             breadth-first from root, and the list of (parent, child) node id
#             pairs joining them.
#     """
#     children = {}
#     for source, target in graph.edge_attrs(
#         attr_keys=["source_id", "target_id"]
#     ).select(["source_id", "target_id"]).iter_rows():
#         children.setdefault(int(source), []).append(int(target))

#     nodes = graph.node_attrs(
#         attr_keys=["node_id", "t", "z", "y", "x"]
#     ).to_pandas().set_index("node_id")

#     order, segments = [int(root)], []
#     seen = {int(root)}
#     queue = [int(root)]
#     while queue:
#         node = queue.pop(0)
#         for child in children.get(node, []):
#             segments.append((node, child))
#             # A cell has one parent, so this only guards against a malformed graph.
#             if child not in seen:
#                 seen.add(child)
#                 order.append(child)
#                 queue.append(child)

#     return nodes.loc[order], segments


# def plot_trajectory(graph, root=None, cmap="viridis", show_z=True):
#     """
#     Plot one trajectory, from a starting cell down through all of its children.

#     Both panels draw one line per edge rather than a single polyline, so a
#     division shows up as a fork instead of a segment darting back and forth
#     between the two daughters. Colour runs along time, giving the path a
#     direction, and the xy panel is drawn in image orientation so it lines up
#     with plot_frame.

#     Args:
#         graph (tracksdata graph): Graph whose nodes carry t, z, y, x.
#         root (int, optional): Node id to start from. Defaults to the node in the
#             first frame with the most descendants, i.e. the longest trajectory.
#         cmap (str, optional): Colormap used to shade time.
#         show_z (bool, optional): Add a second panel of z against t, which the xy
#             view cannot show. Defaults to True.

#     Returns:
#         np.ndarray: The node ids making up the trajectory.
#     """
#     nodes = graph.node_attrs(attr_keys=["node_id", "t"]).to_pandas()
#     if len(nodes) == 0:
#         raise ValueError("graph has no nodes")

#     if root is None:
#         first_frame = nodes[nodes["t"] == nodes["t"].min()]["node_id"]
#         root = max(first_frame, key=lambda node: len(lineage(graph, node)[0]))

#     track, segments = lineage(graph, root)
#     times = track["t"].to_numpy()
#     index = {node: i for i, node in enumerate(track.index)}
#     child_times = np.array([track["t"].iloc[index[child]] for _, child in segments])

#     n_children = np.zeros(len(track), dtype=int)
#     for parent, _ in segments:
#         n_children[index[parent]] += 1
#     divisions = n_children > 1

#     norm = plt.Normalize(times.min(), times.max())
#     axes = plt.subplots(1, 2, figsize=(14, 6))[1] if show_z else [plt.gca()]

#     for ax, (horizontal, vertical) in zip(axes, [("x", "y"), ("t", "z")]):
#         points = track[[horizontal, vertical]].to_numpy(dtype=float)
#         if len(segments):
#             lines = LineCollection(
#                 [[points[index[p]], points[index[c]]] for p, c in segments],
#                 cmap=cmap,
#                 norm=norm,
#                 linewidths=2,
#                 zorder=2,
#             )
#             lines.set_array(child_times)
#             ax.add_collection(lines)

#         ax.scatter(points[:, 0], points[:, 1], s=14, c=times, cmap=cmap, norm=norm, zorder=3)
#         ax.scatter(
#             *points[0],
#             s=140,
#             marker="o",
#             color=(0, 0, 0, 0),
#             edgecolors="limegreen",
#             linewidths=2,
#             zorder=4,
#             label="Start",
#         )
#         if divisions.any():
#             ax.scatter(
#                 points[divisions, 0],
#                 points[divisions, 1],
#                 s=120,
#                 marker="X",
#                 c="orangered",
#                 edgecolors="white",
#                 linewidths=0.5,
#                 zorder=5,
#                 label="Division",
#             )

#         ax.set_xlabel(horizontal)
#         ax.set_ylabel(vertical)
#         ax.autoscale_view()
#         ax.legend(loc="best")

#     axes[0].set_aspect("equal")
#     axes[0].invert_yaxis()  # image orientation, so it matches plot_frame
#     axes[0].set_title(
#         f"node {int(root)}: {len(track)} cells, t = {times.min()} to {times.max()}, "
#         f"{int(divisions.sum())} division(s)"
#     )
#     axes[0].set_xlim(0, 256)
#     axes[0].set_ylim(0, 256)
#     if show_z:
#         axes[1].set_title("z over time")
#         # axes[1].set_xlim(0, 100)
#         axes[1].set_ylim(0, 64)
#         plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[1], label="t")

#     return track.index.to_numpy()


def link_errors(graph, gt_graph, scale=SCALE):
    """
    Say what became of every ground-truth edge, one row each.

    The graph must have been matched already, by evaluate or graph.match, since
    the verdict is reached by following the ground-truth edge into the predicted
    graph: take the detection that matched the parent, and see where we sent it.
    Every plot below is a view of this table, so anything you want to select on
    is easier done here, with the usual dataframe filtering, than through an
    argument.

    Coordinates are voxels, to sit alongside an image; distances are microns.

    Args:
        graph (tracksdata graph): Matched predicted graph.
        gt_graph (tracksdata graph): Ground-truth graph.
        scale (tuple, optional): Physical voxel size as (z, y, x).

    Returns:
        pd.DataFrame: One row per ground-truth edge, with

            kind
                correct, wrong partner (we linked the parent to someone else),
                no link (we left the parent childless), or undetected (an end of
                the edge has no detection at all, so linking never had a chance).
            stolen, thief
                Whether the child we should have taken went to another cell, and
                which cell that was. This is the difference between missing a
                link and losing it, and it is most of what goes wrong.
            margin
                How much further the right child was than the one we took, in
                microns. Negative means a closer detection outbid it, so no
                threshold would have saved the link; near zero means it was a
                coin toss.
            true_distance, chosen_distance
                Distance from the parent detection to the right child and to the
                one we took.
            gt_step
                How far the cell really moved, by the ground truth's own
                coordinates. Compare with true_distance to see how much of the
                apparent motion is our centroids wandering.
    """
    matched_key = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
    if matched_key not in graph.node_attr_keys():
        raise ValueError(
            "graph has not been matched to gt_graph; "
            "run evaluate(graph, gt_graph, scale=scale) first"
        )

    scale = np.asarray(scale, dtype=float)
    pred = graph.node_attrs(
        attr_keys=["node_id", matched_key, "t", "z", "y", "x"]
    ).to_pandas()
    voxel = dict(zip(pred["node_id"], pred[["z", "y", "x"]].to_numpy(dtype=float)))
    matched = pred[pred[matched_key] > -1]
    pred_of_gt = dict(zip(matched[matched_key], matched["node_id"]))

    children, parent_of = {}, {}
    if graph.num_edges():
        edges = graph.edge_attrs(attr_keys=["source_id", "target_id"]).to_pandas()
        for source, target in zip(edges["source_id"], edges["target_id"]):
            children.setdefault(int(source), []).append(int(target))
            parent_of[int(target)] = int(source)

    gt_nodes = (
        gt_graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        .to_pandas()
        .set_index("node_id")
    )
    gt_edges = gt_graph.edge_attrs(attr_keys=["source_id", "target_id"]).to_pandas()

    def microns(a, b):
        return float(np.linalg.norm((voxel[a] - voxel[b]) * scale))

    rows = []
    for gt_source, gt_target in zip(gt_edges["source_id"], gt_edges["target_id"]):
        gt_source, gt_target = int(gt_source), int(gt_target)
        row = {
            "t": int(gt_nodes.loc[gt_source, "t"]),
            "gt_source": gt_source,
            "gt_target": gt_target,
            "gt_step": float(
                np.linalg.norm(
                    (
                        gt_nodes.loc[gt_target, ["z", "y", "x"]].to_numpy(dtype=float)
                        - gt_nodes.loc[gt_source, ["z", "y", "x"]].to_numpy(dtype=float)
                    )
                    * scale
                )
            ),
            "parent": -1,
            "true_child": -1,
            "chosen_child": -1,
            "stolen": False,
            "thief": -1,
            "true_distance": np.nan,
            "chosen_distance": np.nan,
            "margin": np.nan,
        }

        if gt_source not in pred_of_gt or gt_target not in pred_of_gt:
            rows.append({**row, "kind": "undetected"})
            continue

        parent, true_child = int(pred_of_gt[gt_source]), int(pred_of_gt[gt_target])
        taken = children.get(parent, [])
        thief = parent_of.get(true_child, parent)
        row |= {
            "parent": parent,
            "true_child": true_child,
            "true_distance": microns(parent, true_child),
            "stolen": thief != parent,
            "thief": thief if thief != parent else -1,
        }

        if true_child in taken:
            rows.append({**row, "kind": "correct", "chosen_child": true_child,
                         "chosen_distance": row["true_distance"], "margin": 0.0})
        elif not taken:
            rows.append({**row, "kind": "no link"})
        else:
            chosen = min(taken, key=lambda node: microns(parent, node))
            chosen_distance = microns(parent, chosen)
            rows.append({
                **row,
                "kind": "wrong partner",
                "chosen_child": chosen,
                "chosen_distance": chosen_distance,
                "margin": chosen_distance - row["true_distance"],
            })

    columns = ["t", "kind", "stolen", "gt_source", "gt_target", "parent", "true_child",
               "chosen_child", "thief", "true_distance", "chosen_distance", "margin", "gt_step"]
    return pd.DataFrame(rows, columns=columns)


def error_report(errors):
    """
    Count the verdicts in an error table, as counts and as shares.

    Args:
        errors (pd.DataFrame): Table from link_errors.

    Returns:
        pd.DataFrame: One row per kind, plus how many of them were cases where
            the right child was taken by another cell.
    """
    counts = errors["kind"].value_counts()
    report = pd.DataFrame(
        {
            "edges": [int(counts.get(kind, 0)) for kind in KINDS],
            "share": [counts.get(kind, 0) / max(len(errors), 1) for kind in KINDS],
            "stolen": [
                int(errors[errors["kind"] == kind]["stolen"].sum()) for kind in KINDS
            ],
        },
        index=list(KINDS),
    )
    report.index.name = "kind"
    return report


def plot_error_summary(errors, bins=40):
    """
    Four views of an error table: what went wrong, when, by how little, and to whom.

    The margin panel is the one to read first. Errors piled up just left of zero
    were near-ties that a better cost might still win, while a long negative tail
    means another detection was decisively closer and no amount of tuning the
    linker will recover those.

    Args:
        errors (pd.DataFrame): Table from link_errors.
        bins (int, optional): Bins for the two histograms.

    Returns:
        np.ndarray: The four axes.
    """
    axes = plt.subplots(2, 2, figsize=(14, 9))[1]
    report = error_report(errors)

    top = axes[0, 0]
    top.barh(
        list(KINDS),
        report["edges"],
        color=[KIND_COLORS[kind] for kind in KINDS],
        edgecolor="white",
    )
    for kind, count, share in zip(KINDS, report["edges"], report["share"]):
        top.text(count, kind, f"  {count} ({share:.1%})", va="center", fontsize=9)
    top.invert_yaxis()
    top.set_xlim(0, report["edges"].max() * 1.25)
    top.set_xlabel("ground-truth edges")
    top.set_title(f"{len(errors)} ground-truth edges")

    per_frame = (
        errors.pivot_table(index="t", columns="kind", aggfunc="size", fill_value=0)
        .reindex(columns=list(KINDS), fill_value=0)
    )
    bottom = np.zeros(len(per_frame))
    for kind in KINDS:
        axes[0, 1].bar(
            per_frame.index, per_frame[kind], bottom=bottom,
            color=KIND_COLORS[kind], width=1.0, label=kind,
        )
        bottom += per_frame[kind].to_numpy()
    axes[0, 1].set_ylim(0, bottom.max() * 1.3)
    axes[0, 1].set_xlabel("t")
    axes[0, 1].set_ylabel("edges")
    axes[0, 1].set_title("verdicts over time")
    axes[0, 1].legend(loc="upper center", fontsize=8, ncol=4)

    margins = errors[errors["kind"] == "wrong partner"]["margin"].dropna()
    if len(margins):
        axes[1, 0].hist(margins, bins=bins, color=KIND_COLORS["wrong partner"], edgecolor="white")
    axes[1, 0].axvline(0, color="black", linewidth=1)
    axes[1, 0].set_xlabel("chosen − true distance (um)")
    axes[1, 0].set_ylabel("edges")
    axes[1, 0].set_title("how close the right child came to winning")

    failed = errors["kind"].isin(["wrong partner", "no link"])
    span = errors["true_distance"].dropna()
    edges = np.histogram_bin_edges(span, bins=bins) if len(span) else bins
    for label, mask, color in (
        ("correct", errors["kind"] == "correct", KIND_COLORS["correct"]),
        ("failed", failed, KIND_COLORS["wrong partner"]),
    ):
        values = errors[mask]["true_distance"].dropna()
        if len(values):
            axes[1, 1].hist(values, bins=edges, color=color, alpha=0.6, label=label)
    axes[1, 1].set_xlabel("parent to true child (um)")
    axes[1, 1].set_ylabel("edges")
    axes[1, 1].set_title("how far the cell appeared to move")
    axes[1, 1].legend()

    plt.tight_layout()
    return axes


def plot_trajectory_errors(graph, gt_graph, root=None, errors=None, show_z=True, scale=SCALE):
    """
    Follow one ground-truth trajectory and show what we made of it.

    The trajectory itself is drawn from the ground truth, each of its edges
    solid where we reproduced it and dashed where we missed it. Around that go
    the predicted edges that touch the trajectory and should not: the cell we
    linked to instead of the right one, and the cell that reached in and took a
    child that was ours. Those are the two ways a predicted edge lands on this
    trajectory as a false positive, so the three colours account for exactly the
    true positives, false negatives and false positives the score sees here.

    Ground-truth cells sit at their own coordinates and the wrong ends sit at
    the coordinates of the detection involved, so an arrow leaving the
    trajectory shows where we actually went.

    Args:
        graph (tracksdata graph): Matched predicted graph.
        gt_graph (tracksdata graph): Ground-truth graph.
        root (int, optional): Ground-truth node id to start from. Defaults to
            the node in the first frame with the longest trajectory below it.
        errors (pd.DataFrame, optional): Table from link_errors, to save
            recomputing it when looking through several trajectories.
        show_z (bool, optional): Add a second panel of z against t.
        scale (tuple, optional): Physical voxel size as (z, y, x).

    Returns:
        pd.DataFrame: The rows of the error table belonging to this trajectory.
    """
    if errors is None:
        errors = link_errors(graph, gt_graph, scale=scale)

    gt_nodes = (
        gt_graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        .to_pandas()
        .set_index("node_id")
    )
    if root is None:
        first_frame = gt_nodes.index[gt_nodes["t"] == gt_nodes["t"].min()]
        root = max(first_frame, key=lambda node: len(lineage(gt_graph, node)[0]))

    track = lineage(gt_graph, int(root))[0]
    # A lineage holds every descendant, so the ground-truth edges leaving it are
    # exactly the rows whose parent is on it.
    rows = errors[errors["gt_source"].isin(set(track.index))]
    pred = (
        graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        .to_pandas()
        .set_index("node_id")
    )

    wrong = rows[(rows["kind"] != "correct") & (rows["chosen_child"] > -1)]
    stolen = rows[rows["thief"] > -1]
    false_positives = {(int(r.parent), int(r.chosen_child)) for r in wrong.itertuples()}
    false_positives |= {(int(r.thief), int(r.true_child)) for r in stolen.itertuples()}

    axes = plt.subplots(1, 2, figsize=(14, 6))[1] if show_z else [plt.gca()]
    for ax, (horizontal, vertical) in zip(axes, [("x", "y"), ("t", "z")]):
        labelled = set()

        def point(table, node):
            row = table.loc[int(node)]
            return float(row[horizontal]), float(row[vertical])

        def once(label):
            """Label only the first of each kind, so the legend has one of each."""
            if label in labelled:
                return None
            labelled.add(label)
            return label

        for row in rows.itertuples():
            start, end = point(gt_nodes, row.gt_source), point(gt_nodes, row.gt_target)
            if row.kind == "correct":
                ax.plot(
                    *zip(start, end), color=KIND_COLORS["correct"], linewidth=2.4,
                    zorder=3, label=once("true positive"),
                )
            else:
                ax.plot(
                    *zip(start, end), color="crimson", linewidth=1.8, linestyle="--",
                    zorder=3, label=once("false negative"),
                )

            if row.kind != "correct" and row.chosen_child > -1:
                landed = point(pred, row.chosen_child)
                ax.annotate(
                    "", xy=landed, xytext=start,
                    arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.8), zorder=2,
                )
                ax.scatter(
                    *landed, s=90, marker="X", c="darkorange", edgecolors="white",
                    linewidths=0.5, zorder=4, label=once("false positive: we linked here"),
                )

            if row.thief > -1:
                took = point(pred, row.thief)
                ax.annotate(
                    "", xy=end, xytext=took,
                    arrowprops=dict(arrowstyle="->", color="darkviolet", lw=1.6, linestyle="--"),
                    zorder=2,
                )
                ax.scatter(
                    *took, s=90, marker="o", color=(0, 0, 0, 0), edgecolors="darkviolet",
                    linewidths=1.8, zorder=4,
                    label=once("false positive: another cell took it"),
                )

        points = track[[horizontal, vertical]].to_numpy(dtype=float)
        ax.scatter(points[:, 0], points[:, 1], s=16, c="black", zorder=5,
                   label=once("ground-truth cell"))
        ax.scatter(*points[0], s=150, marker="o", color=(0, 0, 0, 0), edgecolors="black",
                   linewidths=2, zorder=6, label=once("start"))

        ax.set_xlabel(horizontal)
        ax.set_ylabel(vertical)
        ax.autoscale_view()

    # One legend for the pair, on the panel with room for it.
    axes[-1].legend(loc="best", fontsize=8, framealpha=0.9)

    true_positives = int((rows["kind"] == "correct").sum())
    axes[0].set_aspect("equal")
    axes[0].invert_yaxis()  # image orientation, so it matches plot_frame
    axes[0].set_title(
        f"gt node {int(root)}: {len(track)} cells over t = {track['t'].min()} to "
        f"{track['t'].max()}\n{true_positives} TP, {len(rows) - true_positives} FN, "
        f"{len(false_positives)} FP"
    )
    if show_z:
        axes[1].set_title("z over time")

    return rows


def plot_error_map(graph, errors, t, ax=None, scale=SCALE):
    """
    Map one frame's links from above, so you can see where the errors sit.

    Each ground-truth cell is drawn where it was detected, with an arrow to the
    cell we linked it to and, when that was the wrong one, a dashed line to
    where it should have gone. Every other detection in the frame is a grey dot,
    which is what shows whether a mistake happened in a crowd or in open space.

    Args:
        graph (tracksdata graph): Matched predicted graph.
        errors (pd.DataFrame): Table from link_errors.
        t (int): Frame to draw, meaning links from t to t + 1.
        ax (matplotlib axes, optional): Axes to draw on.
        scale (tuple, optional): Physical voxel size as (z, y, x).

    Returns:
        matplotlib axes: The axes drawn on.
    """
    ax = ax or plt.gca()
    nodes = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"]).to_pandas()
    position = dict(zip(nodes["node_id"], nodes[["z", "y", "x"]].to_numpy(dtype=float)))

    crowd = nodes[nodes["t"].isin([t, t + 1])]
    ax.scatter(crowd["x"], crowd["y"], s=6, c="lightgrey", zorder=1, label="detections")

    frame = errors[errors["t"] == t]
    for kind in KINDS:
        rows = frame[frame["kind"] == kind]
        if not len(rows):
            continue
        known = rows[rows["parent"] > -1]
        start = np.array([position[p] for p in known["parent"]]).reshape(-1, 3)
        ax.scatter(
            start[:, 2], start[:, 1], s=45, color=(0, 0, 0, 0),
            edgecolors=KIND_COLORS[kind], linewidths=1.8, zorder=4,
            label=f"{kind} ({len(rows)})",
        )
        for _, row in known.iterrows():
            origin = position[row["parent"]]
            if row["chosen_child"] > -1:
                landed = position[row["chosen_child"]]
                ax.annotate(
                    "", xy=(landed[2], landed[1]), xytext=(origin[2], origin[1]),
                    arrowprops=dict(arrowstyle="->", color=KIND_COLORS[kind], lw=1.4),
                    zorder=3,
                )
            if kind in ("wrong partner", "no link"):
                missed = position[row["true_child"]]
                ax.plot(
                    [origin[2], missed[2]], [origin[1], missed[1]],
                    color=KIND_COLORS["correct"], linestyle=":", linewidth=1.2, zorder=2,
                )

    ax.set_aspect("equal")
    ax.invert_yaxis()  # image orientation, so it matches plot_frame
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"links from t = {t} to {t + 1}   (dotted green = where it should have gone)")
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_error_neighbourhood(graph, error, radius=10.0, img=None, scale=SCALE):
    """
    Zoom in on a single bad link and its neighbourhood, from the side as well as above.

    The second panel is x against z, because z is where the centroids wander
    most and where two cells that look well separated from above turn out to be
    on top of each other.

    Args:
        graph (tracksdata graph): Matched predicted graph.
        error (pd.Series): One row of the table from link_errors, e.g.
            errors.query("kind == 'wrong partner'").iloc[0].
        radius (float, optional): How much to show around the parent, in microns.
        img (ndarray, optional): (z, y, x) volume of the parent's frame. The
            slice nearest the parent is drawn behind the xy panel.
        scale (tuple, optional): Physical voxel size as (z, y, x).

    Returns:
        np.ndarray: The two axes.
    """
    if error["parent"] < 0:
        raise ValueError("this ground-truth edge has no detected parent, so there is nothing to draw")

    scale = np.asarray(scale, dtype=float)
    nodes = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"]).to_pandas()
    position = dict(zip(nodes["node_id"], nodes[["z", "y", "x"]].to_numpy(dtype=float)))

    parent = position[error["parent"]]
    t = int(error["t"])
    nearby = nodes[
        nodes["t"].isin([t, t + 1])
        & (np.linalg.norm((nodes[["z", "y", "x"]].to_numpy(dtype=float) - parent) * scale, axis=1) <= radius)
    ]

    thief = int(error.get("thief", -1))
    drawn = [parent] + [
        position[node]
        for node in (error["true_child"], error["chosen_child"], thief)
        if node > -1
    ]
    drawn = np.vstack([np.asarray(drawn), nearby[["z", "y", "x"]].to_numpy(dtype=float)])

    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (horizontal, vertical) in zip(axes, [("x", "y"), ("x", "z")]):
        columns = {"z": 0, "y": 1, "x": 2}
        across, down = columns[horizontal], columns[vertical]

        if img is not None and horizontal == "x" and vertical == "y":
            ax.imshow(img[int(round(parent[0]))], cmap="gray", zorder=0)

        for frame, colour in ((t, "dodgerblue"), (t + 1, "dimgrey")):
            cells = nearby[nearby["t"] == frame]
            if vertical == "y":
                # Looking down on a plane, so fade the cells that are not in it.
                scatter_by_depth(
                    ax, cells[["z", "y", "x"]].to_numpy(dtype=float),
                    parent[0], colour, tol=radius, s=40, linewidths=1.4,
                    zorder=2, label=f"t = {frame}",
                )
            else:
                # The side view shows z on an axis, so depth is already drawn.
                ax.scatter(
                    cells[horizontal], cells[vertical], s=40, linewidths=1.4,
                    zorder=2, marker="o", facecolors="none", edgecolors=colour,
                    label=f"t = {frame}",
                )

        ax.scatter(
            parent[across], parent[down], s=180, marker="o", color=(0, 0, 0, 0),
            edgecolors=KIND_COLORS["correct"], linewidths=2.2, zorder=5, label="parent",
        )
        for node, color, label in (
            (error["true_child"], KIND_COLORS["correct"], "true child"),
            (error["chosen_child"], KIND_COLORS[error["kind"]], "we linked to"),
        ):
            if node < 0:
                continue
            point = position[node]
            ax.annotate(
                "", xy=(point[across], point[down]), xytext=(parent[across], parent[down]),
                arrowprops=dict(arrowstyle="->", color=color, lw=2), zorder=4,
            )
            ax.scatter(
                point[across], point[down], s=140, marker="X", c=color,
                edgecolors="white", linewidths=0.6, zorder=6, label=label,
            )

        if thief > -1:
            took = position[thief]
            landed = position[error["true_child"]]
            ax.annotate(
                "", xy=(landed[across], landed[down]), xytext=(took[across], took[down]),
                arrowprops=dict(arrowstyle="->", color="darkviolet", lw=1.6, linestyle="--"),
                zorder=4,
            )
            ax.scatter(
                took[across], took[down], s=150, marker="o", color=(0, 0, 0, 0),
                edgecolors="darkviolet", linewidths=1.8, zorder=5, label="took it instead",
            )

        # Keep every drawn cell in view, including a true child that lies well
        # outside the neighbourhood, which is exactly the case worth seeing.
        for limit, axis in ((ax.set_xlim, across), (ax.set_ylim, down)):
            low, high = drawn[:, axis].min(), drawn[:, axis].max()
            pad = max(0.05 * (high - low), 1.0)
            limit(low - pad, high + pad)

        ax.set_xlabel(horizontal)
        ax.set_ylabel(vertical)

    axes[0].set_aspect("equal")
    axes[0].invert_yaxis()  # image orientation, so it matches plot_frame
    axes[0].legend(loc="best", fontsize=8)
    axes[0].set_title("from above")
    # A z voxel is several times deeper than an x voxel, so the side view is
    # stretched to physical proportions or it lies about how close cells are.
    axes[1].set_aspect(scale[0] / scale[2])
    axes[1].set_title("from the side")

    detail = f"true child at {error['true_distance']:.2f} um"
    if error["chosen_child"] > -1:
        detail += f", took one at {error['chosen_distance']:.2f} um"
    if error["stolen"]:
        detail += ", true child taken by another cell"
    figure.suptitle(f"{error['kind']} at t = {t}: {detail}")
    plt.tight_layout()
    return axes


def scatter_plot_center_preds(seeds, matched_gt, unmatched_gt, frame, tol=3):
    unmatched_gt_centers = unmatched_gt[unmatched_gt[:, 0] == frame] if len(unmatched_gt) > 0 else []
    matched_gt_centers = matched_gt[matched_gt[:, 0] == frame] if len(matched_gt) > 0 else []
    centers = seeds[np.abs(seeds[:, 0].astype(int) - frame) <= tol]

    # Ground truth is selected on its exact plane, so it never fades; it goes
    # through the same call anyway, so widening that filter would just work.
    if len(unmatched_gt_centers) > 0:
        scatter_by_depth(
            plt, unmatched_gt_centers[:, :3], frame, "orangered",
            tol=tol, s=100, linewidths=2, label="Unmatched",
        )

    if len(matched_gt_centers) > 0:
        scatter_by_depth(
            plt, matched_gt_centers[:, :3], frame, "limegreen",
            tol=tol, s=100, linewidths=2, label="Matched",
        )

    # A z-plane can hold ground truth but no prediction within tol, and taking a
    # minimum over an empty distance list would raise.
    if len(matched_gt_centers) > 0 and len(centers) > 0:
        dists_to_gt = []
        for gtc in matched_gt_centers:
            dists_to_gt.append([dist(gtc, c) for c in centers])

        dists_to_gt = np.min(dists_to_gt, axis=0)
        matched = np.argwhere(dists_to_gt <= 7).flatten()
        unmatched = np.argwhere(dists_to_gt > 7).flatten()
    else:
        unmatched = np.arange(len(centers))
        matched = []
    
    if len(matched) > 0:
        scatter_by_depth(
            plt, centers[matched, :3], frame, "dodgerblue",
            tol=tol, facecolor="skyblue", s=20, linewidths=1,
            label="Predicted Match",
        )

    if len(unmatched) > 0:
        scatter_by_depth(
            plt, centers[unmatched, :3], frame, "red",
            tol=tol, facecolor="orangered", s=10, linewidths=1,
            base_alpha=0.5, label="Predicted Unmatch",
        )

    # A plane can be empty, and an empty legend only warns.
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend()


def interactive_center_preds(
    img,
    seeds,
    matched_gt,
    unmatched_gt,
    mask=None,
    tol=3,
    t=None,
    figsize=(8, 8),
):
    """
    Step through z with a slider, drawing scatter_plot_center_preds on each plane.

    Ground truth is drawn only on its own plane, while predictions are drawn if
    they lie within tol planes, so a prediction can appear beside a cell whose
    centre is in a neighbouring plane. That is deliberate: it is what stops a
    correct detection in an adjacent plane from looking like a miss. Raising tol
    widens that window.

    Opens on the plane holding the most unmatched ground truth, since that is
    where the misses are.

    Args:
        img (np.ndarray): Volume for one timepoint, as (z, y, x). A 4D (t, z, y, x)
            array is accepted if t is given.
        seeds (np.ndarray): Predicted centres, (n, 3) as (z, y, x).
        matched_gt (np.ndarray): Ground-truth centres that got a prediction,
            (n, 3) as (z, y, x). May be empty.
        unmatched_gt (np.ndarray): Ground-truth centres that did not, same shape.
        mask (np.ndarray, optional): Label volume (z, y, x) drawn behind the
            points, with a checkbox to hide it.
        tol (int, optional): Initial plane window for predictions.
        t (int, optional): Timepoint to slice out of a 4D img, and to show in the
            title.
        figsize (tuple, optional): Figure size.

    Returns:
        ipywidgets.Widget: The assembled control panel and plot.
    """
    import ipywidgets as widgets

    img = np.asarray(img).squeeze()
    if img.ndim == 4:
        if t is None:
            raise ValueError("img is 4D (t, z, y, x), so pass t to choose a timepoint")
        img = img[t]
    if img.ndim != 3:
        raise ValueError(f"img must be (z, y, x), got shape {img.shape}")

    if img.dtype == np.uint16:
        img = np.clip(img, np.percentile(img, 1), np.percentile(img, 99)).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255).astype(np.uint8)

    n_z = img.shape[0]
    seeds = np.asarray(seeds)

    def planes(points):
        """Plane index of each point, or an empty array when there are none."""
        points = np.asarray(points)
        if points.size == 0:
            return np.empty(0, dtype=int)
        return points[:, 0].astype(int)

    unmatched_planes = planes(unmatched_gt)
    if unmatched_planes.size:
        counts = np.bincount(np.clip(unmatched_planes, 0, n_z - 1), minlength=n_z)
        start_z = int(counts.argmax())
    else:
        start_z = n_z // 2

    z_slider = widgets.IntSlider(
        value=start_z, min=0, max=n_z - 1, description="z",
        continuous_update=False, layout=widgets.Layout(width="70%"),
    )
    tol_slider = widgets.IntSlider(
        value=tol, min=0, max=max(10, 2 * tol), description="tol",
        continuous_update=False, layout=widgets.Layout(width="70%"),
    )
    show_mask = widgets.Checkbox(
        value=mask is not None, description="mask", disabled=mask is None
    )

    if mask is not None:
        img_outlines = img.copy()[..., None].repeat(3, -1)

        for i in range(len(mask)):
            outlines = masks_to_outlines(mask[i])
            outX, outY = np.nonzero(outlines)
            img_outlines[i][outX, outY] = np.array([255, 0, 255], dtype=img.dtype)  # pure red

    def draw(z, tol, show_mask):
        plt.figure(figsize=figsize)

        if mask is not None and show_mask:
            plt.imshow(img_outlines[z])
        else:
            plt.imshow(img[z], cmap="gray")

        scatter_plot_center_preds(seeds, matched_gt, unmatched_gt, frame=z, tol=tol)

        plt.axis("off")
        plt.show()

    output = widgets.interactive_output(
        draw, {"z": z_slider, "tol": tol_slider, "show_mask": show_mask}
    )

    return widgets.VBox([output, z_slider, tol_slider, show_mask])
