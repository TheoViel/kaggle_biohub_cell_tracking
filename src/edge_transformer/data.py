"""Videos, their precomputed nodes, and the windows training draws from.

Mirrors ``tracking/data.py``, with one structural difference that shapes the
whole pipeline: **nothing is detected here**. The nodes and their appearance
features were produced by earlier stages and sit in ``nodes_folder`` as one
``<sample>.npz`` per video, holding

    coords      (N, 4) int/float  -- t, z, y, x in image voxels
    embeddings  (N, D) float32    -- L2-normalised, one row per node
    node_probs  (N,)   float32    -- the detector's confidence, optional

which is exactly what ``inference_contrastive.py`` writes. So a video is a
fixed node set, and the only thing left to learn is which of the candidate links
between those nodes are real.

Ground truth is attached the way the rest of the repo attaches it: each
annotated cell is matched to the nearest node within ``MATCH_UM``, and a GT link
becomes a pair of *node* indices (``embeddings.links.detection_links``). The
annotation is sparse -- roughly one cell a frame against ~260 detections -- so
most nodes carry no label at all, and the supervision mask built in
``targets.py`` is what keeps that from being read as "no link here".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embeddings.links import MATCH_UM, detection_links
from params import SCALE
from tracking_cellmot.io import list_datasets


@dataclass
class VideoData:
    """One video's nodes, features and ground-truth links.

    ``rows_by_frame[t]`` indexes ``coords``/``emb``, and the *order* of that
    index array is the per-frame row order every other module assumes -- it is
    what ``detection_links`` returns pairs in, and what ``node_row`` in a batch
    refers back to.
    """

    name: str
    path: Path                                  # the dataset stem, for the geff
    coords: np.ndarray                          # (N, 4) float32, t/z/y/x voxels
    emb: np.ndarray | None                      # (N, D) float16
    conf: np.ndarray | None                     # (N,) float32 detection probability
    rows_by_frame: dict[int, np.ndarray]
    links: dict[int, np.ndarray]                # t -> (K, 3) [src_row, tgt_row, _]
    frames: np.ndarray
    n_gt_added: int = 0                         # unmatched GT nodes appended

    @property
    def embed_dim(self) -> int:
        return 0 if self.emb is None else int(self.emb.shape[1])


def find_node_files(nodes_folder: str | Path) -> dict[str, Path]:
    """``sample name -> npz`` for a precomputed-node folder."""
    folder = Path(nodes_folder)
    files = {p.stem: p for p in sorted(folder.glob("*.npz"))
             if not p.stem.endswith("_flows")}
    if not files:
        raise FileNotFoundError(f"no *.npz under {folder}")
    return files


def unmatched_gt_rows(
    coords: np.ndarray,
    gt_coords: np.ndarray,
    match_um: float = MATCH_UM,
) -> np.ndarray:
    """Rows of ``gt_coords`` that no detection stands for, as an index array.

    The pairing is the optimal one-to-one assignment per frame, within
    ``match_um``, which is what the competition metric does -- and not the
    independent nearest-neighbour of :func:`embeddings.links.attach`. The
    difference matters here: when two annotated cells share one nearest
    detection, ``attach`` calls both matched while the metric can only credit
    one, so going by ``attach`` would leave the other unmatched and node recall
    short of 1. Pairing them the metric's way instead means appending one
    detection per row this returns is exactly enough: every GT node then either
    keeps the detection it already had or takes its own copy, which is a
    complete assignment, so an optimal matcher must find one.
    """
    from scipy.optimize import linear_sum_assignment

    if len(gt_coords) == 0:
        return np.zeros(0, int)

    scale = np.asarray(SCALE, np.float32)
    frame_of = coords[:, 0].astype(int)
    gt_frame = gt_coords[:, 0].astype(int)

    out: list[np.ndarray] = []
    for t in np.unique(gt_frame):
        gt_rows = np.flatnonzero(gt_frame == t)
        det_rows = np.flatnonzero(frame_of == t)
        if len(det_rows) == 0:
            out.append(gt_rows)
            continue

        d = np.linalg.norm(
            (gt_coords[gt_rows, None, 1:4] - coords[None, det_rows, 1:4]) * scale,
            axis=2,
        )
        near = d <= match_um
        # Pairs beyond the radius are not matches, so they must never be chosen
        # over leaving a row unassigned -- but linear_sum_assignment fills
        # min(n, m) rows regardless, so they are priced out of the way and the
        # ones it still picked are dropped afterwards by `near`.
        cost = np.where(near, d, 1e6)
        rows, cols = linear_sum_assignment(cost)
        matched = {int(r) for r, c in zip(rows, cols) if near[r, c]}
        out.append(gt_rows[[i for i in range(len(gt_rows)) if i not in matched]])

    return np.concatenate(out) if out else np.zeros(0, int)


def load_video(
    name: str,
    node_path: Path,
    ds_path: Path,
    use_embeddings: bool = True,
    match_um: float = MATCH_UM,
    embeddings_key: str = "embeddings",
    with_gt: bool = False,
) -> VideoData:
    """Read one video's nodes, features and GT links.

    ``embeddings_key`` names which run's embeddings to read.
    ``inference_contrastive.py`` writes them back into the detections npz under
    ``embeddings_<run>``, so one detections folder can carry several NETr runs
    side by side and this picks the one to train against. The default is the
    flat ``embeddings`` the older, one-folder-per-run layout used.

    ``with_gt`` appends the annotated nodes no detection reached, from the
    ``gt_coords`` / ``<embeddings_key>_gt`` arrays ``inference_contrastive.py
    --gt`` writes. That makes node recall 1 by construction, which is a
    diagnostic and not a result: it hands the model the cells the detector
    missed, so anything scored on it is an upper bound. It is never what a
    submission runs.
    """
    with np.load(node_path) as d:
        coords = np.asarray(d["coords"]).astype(np.float32)
        emb = None
        if use_embeddings:
            if embeddings_key not in d:
                available = [k for k in d.files if k.startswith("embeddings")]
                raise KeyError(
                    f"{node_path} has no '{embeddings_key}' array. It holds "
                    f"{available or 'no embeddings at all'} -- run "
                    "inference_contrastive.py against this folder, point "
                    "embeddings_key at one of those, or set use_embeddings=false "
                    "to train on geometry alone."
                )
            # fp16 halves both the read and the resident set (~27 MB a video in
            # fp32, and a fold holds ~130 of them). The features are cast to
            # fp32 per item, well before anything numerically delicate.
            emb = np.asarray(d[embeddings_key]).astype(np.float16)

        # Always read, never required: it is N floats, and a folder written
        # before main_predict.py carried it simply has no column. The dataset
        # decides whether the run needs one.
        conf = (
            np.asarray(d["node_probs"]).astype(np.float32).reshape(-1)
            if "node_probs" in d.files else None
        )

        gt_coords = gt_emb = None
        if with_gt:
            gt_key = f"{embeddings_key}_gt"
            missing = [k for k in ("gt_coords", *( (gt_key,) if use_embeddings else () ))
                       if k not in d.files]
            if missing:
                raise KeyError(
                    f"{node_path} has no {', '.join(missing)} array, so there is "
                    "nothing to append. Run inference_contrastive.py --gt against "
                    "this folder first."
                )
            gt_coords = np.asarray(d["gt_coords"]).astype(np.float32)
            if use_embeddings:
                gt_emb = np.asarray(d[gt_key]).astype(np.float16)

    n_gt_added = 0
    if gt_coords is not None and len(gt_coords):
        add = unmatched_gt_rows(coords, gt_coords, match_um)
        n_gt_added = int(len(add))
        if n_gt_added:
            # Appended, not merged in frame order: every existing row index --
            # in the embeddings, and in anything already computed against them
            # -- has to keep meaning the same node. ``rows_by_frame`` below is
            # rebuilt from the coords either way, and it and ``detection_links``
            # both read a frame's rows in file order, so the new nodes simply
            # come last within their frame.
            coords = np.concatenate([coords, gt_coords[add]], axis=0)
            if emb is not None:
                emb = np.concatenate([emb, gt_emb[add]], axis=0)
            if conf is not None:
                # An annotated cell is a cell. 1.0 is the honest confidence for
                # a node that came from the ground truth rather than the
                # detector -- and it keeps --gt from being quietly penalised by
                # a confidence feature it has no detection to fill in.
                conf = np.concatenate(
                    [conf, np.ones(n_gt_added, np.float32)], axis=0
                )

    frame_of = coords[:, 0].astype(int)
    frames = np.unique(frame_of)
    rows_by_frame = {int(t): np.flatnonzero(frame_of == t) for t in frames}

    # ``ds_path`` is the dataset *stem* -- open_dataset appends the suffixes --
    # so the graph to test for is the .geff beside it, not the stem itself.
    # Attached after the append, so the GT links of a node that was missing land
    # on the copy just added rather than being dropped as unreachable.
    links = (
        detection_links(ds_path, coords, match_um)
        if ds_path.with_suffix(".geff").exists() else {}
    )
    return VideoData(
        name=name, path=ds_path, coords=coords, emb=emb, conf=conf,
        rows_by_frame=rows_by_frame, links=links, frames=frames,
        n_gt_added=n_gt_added,
    )


def load_videos(
    nodes_folder: str | Path,
    data_dir: str | Path,
    embryos: list[str] | None = None,
    n_videos: int | None = None,
    use_embeddings: bool = True,
    match_um: float = MATCH_UM,
    require_links: bool = True,
    verbose: bool = False,
    embeddings_key: str = "embeddings",
    with_gt: bool = False,
) -> list[VideoData]:
    """Every video of ``embryos`` that has both nodes and a ground-truth graph.

    ``with_gt`` is the diagnostic node set -- see :func:`load_video`.
    """
    node_files = find_node_files(nodes_folder)
    paths = [p for p in list_datasets(Path(data_dir), require_geff=True)
             if embryos is None or any(p.name.startswith(e) for e in embryos)]

    videos: list[VideoData] = []
    for path in paths:
        if path.name not in node_files:
            continue
        video = load_video(
            path.name, node_files[path.name], path,
            use_embeddings=use_embeddings, match_um=match_um,
            embeddings_key=embeddings_key, with_gt=with_gt,
        )
        if require_links and not video.links:
            continue
        videos.append(video)
        if n_videos is not None and len(videos) >= n_videos:
            break

    if not videos:
        raise ValueError(
            f"no videos for embryos {embryos} with nodes in {nodes_folder} "
            f"and ground truth in {data_dir}"
        )
    if verbose:
        n_nodes = sum(len(v.coords) for v in videos)
        n_links = sum(sum(len(r) for r in v.links.values()) for v in videos)
        note = ""
        if with_gt:
            n_added = sum(v.n_gt_added for v in videos)
            note = (f", including {n_added} unmatched GT nodes appended "
                    "(node recall is 1 by construction)")
        print(f"- Loaded {len(videos)} videos: {n_nodes} nodes, "
              f"{n_links} ground-truth links attached to them{note}")
    return videos


def build_windows(
    videos: list[VideoData],
    window_size: int = 2,
    require_links: bool = True,
) -> list[tuple[int, int]]:
    """``(video index, t_start)`` for every trainable window.

    A window with no usable ground-truth link inside it is dropped when
    ``require_links``: with per-edge supervision it would contribute an exactly
    zero loss, so it is pure cost. Prediction keeps them -- there the point is
    to score every frame pair of the video.
    """
    out: list[tuple[int, int]] = []
    for vi, video in enumerate(videos):
        t_max = int(video.frames.max())
        for t in video.frames.astype(int):
            span = [t + k for k in range(window_size)]
            if span[-1] > t_max or any(s not in video.rows_by_frame for s in span):
                continue
            if require_links and not any(
                len(video.links.get(s, ())) for s in span[:-1]
            ):
                continue
            out.append((vi, int(t)))
    if not out:
        raise ValueError("no window held a ground-truth link")
    return out


def to_microns(coords_zyx: np.ndarray) -> np.ndarray:
    """Image voxels -> microns, the space every distance in this pipeline uses.

    The voxels are 4x anisotropic (1.625 um in z against 0.40625 in y/x), so a
    Euclidean distance taken on raw voxel coordinates would count a z-step as a
    quarter of what it is. Everything downstream -- the candidate radius, the
    attention cutoff, the rotary wavelengths, the segment geometry -- is
    specified in microns for that reason.
    """
    return coords_zyx.astype(np.float32) * np.asarray(SCALE, np.float32)
