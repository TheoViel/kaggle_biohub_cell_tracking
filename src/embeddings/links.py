"""Ground-truth links, expressed as pairs of *detection* indices.

The annotation and the detections are two different things: ground-truth nodes
are sparse (~1 a frame here, against ~500 detections) and sit at their own
coordinates, while the model only ever embeds detections. So a GT link is only
usable once each of its two endpoints has been attached to the detection that
stands for it -- the nearest one within ``MATCH_UM``, the competition's own
node-matching radius and the tolerance the rest of the repo judges "same cell"
by. An annotation with no detection near it is dropped rather than counted
against anything: that is a detection failure, and it belongs to the detector.

Both the training dataset (real consecutive-frame positives) and the validation
metric need exactly this, on the same terms, so it lives here rather than in
either of them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from params import SCALE

# The competition's node-matching radius, in microns.
MATCH_UM = 7.0


def gt_frames(ds_path: Path):
    """Ground-truth nodes per frame and the links between consecutive frames.

    Returns ``(nodes, links)`` where ``nodes[t]`` is ``(M, 3)`` z/y/x and
    ``links[t]`` is a list of ``(i, j)`` index pairs into ``nodes[t]`` and
    ``nodes[t + 1]``.
    """
    from tracking_cellmot.io import open_dataset

    ds = open_dataset(ds_path, normalize=False, require_tracks=True, load_image=False)
    attrs = ds.tracks.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    node_id = np.asarray(attrs["node_id"])
    t = np.asarray(attrs["t"]).astype(int)
    coords = np.stack([np.asarray(attrs[k]) for k in ("z", "y", "x")], 1).astype(np.float32)

    nodes: dict[int, np.ndarray] = {}
    row_of: dict[int, tuple[int, int]] = {}     # node_id -> (frame, row)
    for frame in np.unique(t):
        sel = np.flatnonzero(t == frame)
        nodes[int(frame)] = coords[sel]
        for row, idx in enumerate(sel):
            row_of[int(node_id[idx])] = (int(frame), row)

    edges = ds.tracks.edge_attrs(attr_keys=["source_id", "target_id"])
    links: dict[int, list[tuple[int, int]]] = {}
    for src, tgt in zip(edges["source_id"], edges["target_id"]):
        a, b = row_of.get(int(src)), row_of.get(int(tgt))
        if a is None or b is None or b[0] != a[0] + 1:
            continue                             # only consecutive-frame links
        links.setdefault(a[0], []).append((a[1], b[1]))
    return nodes, links


def gt_node_coords(ds_path: Path) -> np.ndarray:
    """Every annotated cell as one ``(M, 4)`` t/z/y/x array, in voxels.

    The same space and column order as a predictions ``coords``, so the two can
    be concatenated or distance-matched without further conversion. Rows are
    grouped by frame in the order :func:`gt_frames` returns them, which is the
    order the ``(i, j)`` pairs of its ``links`` index into.
    """
    nodes, _ = gt_frames(ds_path)
    blocks = [np.column_stack([np.full(len(zyx), t, np.float32), zyx])
              for t, zyx in sorted(nodes.items()) if len(zyx)]
    if not blocks:
        return np.zeros((0, 4), np.float32)
    return np.concatenate(blocks, axis=0).astype(np.float32)


def attach(gt: np.ndarray, det: np.ndarray, match_um: float = MATCH_UM) -> np.ndarray:
    """For each GT node, the nearest detection within ``match_um``, else -1."""
    if len(gt) == 0 or len(det) == 0:
        return np.full(len(gt), -1, int)
    scale = np.asarray(SCALE, np.float32)
    d = np.linalg.norm((gt[:, None] - det[None]) * scale, axis=2)
    best = d.argmin(axis=1)
    return np.where(d[np.arange(len(gt)), best] <= match_um, best, -1)


def detection_links(
    ds_path: Path,
    coords: np.ndarray,
    match_um: float = MATCH_UM,
) -> dict[int, np.ndarray]:
    """GT links rewritten as detection indices, per frame.

    ``coords`` is a predictions ``(N, 4)`` array of ``t, z, y, x`` rows. The
    result maps ``t`` to a ``(K, 3)`` int array of ``[a, b, src_unique]``: row
    ``a`` of frame ``t``'s detections links to row ``b`` of frame ``t + 1``'s,
    counting rows the way the dataset slices them (``coords[coords[:, 0] == t]``,
    in file order).

    Two kinds of ambiguity are resolved here rather than in the loss, since both
    would otherwise present a genuinely correct match as a negative:

    ``src_unique`` is 0 when the source detection has more than one target --
    a **division**. Its two daughters are both right, so the parent cannot be
    scored in the t -> t+1 direction (whichever daughter were named the
    positive, the other would become a hard negative). The reverse direction is
    unaffected: each daughter still has exactly one parent, so those rows stay
    usable and are kept.

    A target detection claimed by two annotations is dropped outright. That
    means ``attach`` collapsed two ground-truth cells onto one detection -- a
    merged or missing detection -- and neither link can be trusted.
    """
    nodes, links = gt_frames(ds_path)
    if not links:
        return {}

    frame_of = coords[:, 0].astype(int)
    out: dict[int, np.ndarray] = {}
    for t, pairs in links.items():
        if t not in nodes or (t + 1) not in nodes:
            continue
        det_t = coords[frame_of == t][:, 1:4].astype(np.float32)
        det_n = coords[frame_of == t + 1][:, 1:4].astype(np.float32)
        if len(det_t) == 0 or len(det_n) == 0:
            continue

        src = attach(nodes[t], det_t, match_um)
        tgt = attach(nodes[t + 1], det_n, match_um)
        rows = [(int(src[i]), int(tgt[j])) for i, j in pairs
                if src[i] >= 0 and tgt[j] >= 0]
        if not rows:
            continue
        rows = np.unique(np.asarray(rows, int), axis=0)      # (K, 2)

        _, tgt_counts = np.unique(rows[:, 1], return_counts=True)
        keep = tgt_counts[np.searchsorted(np.unique(rows[:, 1]), rows[:, 1])] == 1
        rows = rows[keep]
        if not len(rows):
            continue

        uniq_src, src_counts = np.unique(rows[:, 0], return_counts=True)
        src_unique = src_counts[np.searchsorted(uniq_src, rows[:, 0])] == 1
        out[int(t)] = np.concatenate([rows, src_unique[:, None].astype(int)], axis=1)
    return out
