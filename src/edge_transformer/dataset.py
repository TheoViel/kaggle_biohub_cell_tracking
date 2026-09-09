"""Windows of precomputed nodes, delivered as candidate graphs.

A sample is a window of ``window_size`` consecutive frames of one video: every
node in those frames, the candidate links between them, and the ground truth's
verdict on the ones it can speak about. Two frames is the default because the
competition scores ``t -> t+1`` edges; a longer window costs quadratically in
the edge attention and buys context rather than new edges.

Padding is to the batch's maximum rather than to a fixed cap (see
``collate_graphs``), because the two axes vary a lot between windows -- node
counts by video, edge counts by how crowded the frame is -- and a fixed cap
would either clip the crowded windows or pay for empty slots on every other one.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from edge_transformer.data import VideoData, build_windows, to_microns
from edge_transformer.graph import (
    build_candidates,
    drift_between_frames,
    edge_feature_names,
    edge_features,
    node_feature_dim,
    node_features,
)
from edge_transformer.targets import edge_targets, window_links


class EdgeDataset(Dataset):
    """One window -> one candidate graph, with per-edge targets.

    Args:
        videos: as loaded by ``data.load_videos``.
        window_size: frames per window.
        max_nodes_per_frame: cap, applied per frame; ``None`` or 0 for none.
            Frames here hold ~260 detections; the cap exists to bound the edge
            attention, which is quadratic in the edge count. Nodes are dropped at
            random so the survivors stay a fair sample -- except that an
            annotated endpoint is always kept, since dropping it throws away the
            only supervision the window has.

            A capped *inference* pass is strictly worse than an uncapped one: a
            node that loses the draw is not merely seen in less context, it gets
            no candidate edges at all and so cannot be linked, which the metric
            reads as missing recall. The cap is a training-memory knob, and
            ``infer.dataset_for`` therefore defaults to no cap.
        drift_correction: remove the whole-field translation between frames
            before anything measures a distance. Kept from the tracking
            pipeline, where it is worth having: a video that slides -1.6 um in z
            every frame otherwise biases every distance and every rank feature.
        require_links: keep only windows the ground truth can supervise. Off for
            prediction, where every frame pair has to be scored.
        time_flip_p: probability of running a window backwards -- the frames
            relabelled in reverse, so what was the parent becomes the child.
            Unlike the spatial flips this is *not* an exact symmetry, and the
            asymmetry is worth stating: reversed, a dividing parent becomes two
            cells merging into one, which forward time never contains and which
            is the one structural fact this ground truth obeys without exception
            (6358/6358 children have exactly one parent). Divisions are 0.09% of
            parents here, so the corruption is small, but it is a corruption --
            the augmentation trades a little of that prior for a doubling of the
            motion patterns each window shows. Off by default.
        division_weight: what each link of a *dividing* parent is worth. Divisions
            are 0.13% of the annotation, so at weight 1 the model barely learns
            them -- see edge_transformer/targets.py.
        pl_weight: what a pseudo-labelled edge is worth against an annotated
            one, 0 to ignore them. The pseudo-labels themselves come from
            ``video.pl_links`` (``data.pseudo_links``), already vetoed against
            the annotation; this is only the price. They are the model's own
            output from a previous round, so they carry its mistakes as well as
            its knowledge -- the weight is how much of that risk is accepted in
            exchange for reaching the ~99% of the graph the annotation cannot.
            The per-frame cap still protects only *annotated* endpoints, so a
            capped frame samples pseudo-labelled ones like any other node.
        use_conf_score: feed the detector's per-node confidence -- the
            ``node_probs`` array ``main_predict.py`` writes -- to the model, as
            a node feature and as three edge features. The detection threshold
            only records that a node cleared it; by how much is a real signal
            about whether the node is a cell at all, and therefore about whether
            an edge touching it should be believed.
    """

    def __init__(
        self,
        videos: list[VideoData],
        window_size: int = 2,
        max_nodes_per_frame: int | None = 512,
        n_neighbors: int = 5,
        max_distance_um: float = 20.0,
        max_dt: int = 1,
        pos_scale: float = 20.0,
        drift_correction: bool = True,
        require_links: bool = True,
        use_conf_score: bool = False,
        augment: bool = False,
        flip_p: float = 0.5,
        rot_p: float = 0.5,
        jitter_um: float = 0.3,
        time_flip_p: float = 0.0,
        pl_weight: float = 0.0,
        division_weight: float = 1.0,
        seed: int = 0,
    ):
        self.videos = videos
        self.window_size = int(window_size)
        # 0 is the "no cap" spelling everywhere below, so None collapses onto it.
        self.max_nodes_per_frame = int(max_nodes_per_frame or 0)
        self.n_neighbors = int(n_neighbors)
        self.max_distance_um = float(max_distance_um)
        self.max_dt = int(max_dt)
        self.pos_scale = float(pos_scale)
        self.drift_correction = bool(drift_correction)
        self.use_conf_score = bool(use_conf_score)
        self.augment = bool(augment)
        self.flip_p = float(flip_p)
        self.rot_p = float(rot_p)
        self.jitter_um = float(jitter_um)
        self.time_flip_p = float(time_flip_p)
        self.pl_weight = float(pl_weight)
        self.division_weight = float(division_weight)
        self.seed = int(seed)

        self.windows = build_windows(videos, self.window_size, require_links)
        dims = {v.embed_dim for v in videos}
        if len(dims) > 1:
            raise ValueError(f"videos disagree on the embedding width: {sorted(dims)}")
        self.embed_dim = dims.pop()

        if self.use_conf_score:
            # Checked here rather than per window: a missing column would
            # otherwise surface as a width mismatch thousands of steps in.
            without = [v.name for v in videos if v.conf is None]
            if without:
                raise ValueError(
                    f"use_conf_score is on but {len(without)} video(s) carry no "
                    f"'node_probs' array, e.g. {', '.join(without[:3])}. That "
                    "folder was written before main_predict.py saved the "
                    "detection confidences -- re-run it, or turn the flag off."
                )

    @property
    def node_feat_dim(self) -> int:
        return node_feature_dim(self.embed_dim, self.use_conf_score)

    @property
    def edge_feat_dim(self) -> int:
        return len(edge_feature_names(self.use_conf_score))

    def __len__(self) -> int:
        return len(self.windows)

    def _select(self, video: VideoData, t: int, keep: set[int], rng) -> np.ndarray:
        """Which rows of frame ``t`` to use, always including ``keep``."""
        rows = video.rows_by_frame[t]
        n = len(rows)
        if not self.max_nodes_per_frame or n <= self.max_nodes_per_frame:
            return np.arange(n)
        forced = np.array(sorted(keep), int)[: self.max_nodes_per_frame]
        rest = np.setdiff1d(np.arange(n), forced)
        extra = rng.choice(rest, self.max_nodes_per_frame - len(forced), replace=False)
        return np.sort(np.concatenate([forced, extra]))

    def build(
        self, video_idx: int, t_start: int, rng=None, keep_annotated: bool = True,
        time_flip: bool | None = None,
    ) -> dict:
        """The candidate graph for one window, as plain tensors.

        Exposed as a method rather than hidden in ``__getitem__`` because
        prediction walks a video window by window and wants the same graph
        without the Dataset's indexing.

        ``keep_annotated`` decides whether an annotated node is protected from
        the per-frame cap. Training wants that -- dropping the only labelled
        endpoint in a window wastes its supervision -- but real inference never
        has the annotation to consult, so scoring a video the same way training
        validates it would report a recall the deployed model cannot reach.
        ``predict.score_video`` always passes ``False`` for exactly that reason;
        this default only serves ``__getitem__``, i.e. training and its
        window-level (not video-level) validation.

        ``time_flip`` runs the window backwards -- see ``time_flip_p``. ``None``
        draws it, which is what training wants; prediction passes it explicitly
        because TTA needs both directions of the *same* window, not a coin toss.
        """
        if rng is None:
            # Prediction has to be reproducible: when the per-frame cap bites it
            # picks a random subset, and an unseeded draw would make the same
            # video score differently on every run -- ~0.02 of edge Jaccard on a
            # dense one. Keyed on the window so two videos still get different
            # subsets, and left random while augmenting, where fresh draws are
            # the point.
            rng = (
                np.random.default_rng()
                if self.augment
                else np.random.default_rng((self.seed, video_idx, t_start))
            )
        video = self.videos[video_idx]
        frames = [t_start + k for k in range(self.window_size)]

        # Annotated rows are kept through the per-frame cap -- training only;
        # see keep_annotated's docstring above.
        forced: dict[int, set[int]] = {k: set() for k in range(len(frames))}
        if keep_annotated:
            for k, t in enumerate(frames[:-1]):
                rows = video.links.get(int(t))
                if rows is None:
                    continue
                forced[k].update(int(r) for r in rows[:, 0])
                forced[k + 1].update(int(r) for r in rows[:, 1])

        sel_rows, frame_idx, node_row, node_global = [], [], [], []
        node_of: dict[tuple[int, int], int] = {}
        base = 0
        for k, t in enumerate(frames):
            sel = self._select(video, t, forced[k], rng)
            for local, row in enumerate(sel):
                node_of[(k, int(row))] = base + local
            base += len(sel)
            sel_rows.append(video.rows_by_frame[t][sel])
            frame_idx.append(np.full(len(sel), k, np.int64))
            node_row.append(sel.astype(np.int64))
            node_global.append(video.rows_by_frame[t][sel].astype(np.int64))

        rows = np.concatenate(sel_rows)
        frame_idx = np.concatenate(frame_idx)
        node_row = np.concatenate(node_row)
        node_global = np.concatenate(node_global)

        # The window is assembled forwards above -- `forced` and `node_of` both
        # read video.links[t], which is indexed by real frame -- and reversed
        # here, once, by relabelling the frame each node sits in. Everything
        # after this point derives direction from `frame_idx` alone: the drift
        # is estimated the new way round, `build_candidates` proposes parent ->
        # child in the new ordering, and delta_t stays +1. Only the ground truth
        # has to be told, which is the swap below.
        if time_flip is None:
            time_flip = self.augment and rng.random() < self.time_flip_p
        if time_flip:
            frame_idx = (len(frames) - 1) - frame_idx

        pos_um = to_microns(video.coords[rows, 1:4])
        emb = None if video.emb is None else video.emb[rows]
        # Left out of _augment_positions on purpose: the augmentations are rigid
        # transforms of the window, and how sure the detector was does not move
        # with the cell.
        conf = video.conf[rows] if self.use_conf_score else None

        if self.augment:
            pos_um = self._augment_positions(pos_um, rng)

        if self.drift_correction:
            drift = drift_between_frames(pos_um, frame_idx, len(frames))
            pos_um = pos_um - drift[frame_idx]

        edges = build_candidates(
            pos_um, frame_idx, len(frames),
            n_neighbors=self.n_neighbors,
            max_distance_um=self.max_distance_um,
            max_dt=self.max_dt,
        )
        n_nodes = len(pos_um)

        links = window_links(video, frames, node_of)
        pl = (
            window_links(video, frames, node_of, video.pl_links)
            if self.pl_weight > 0 and video.pl_links else []
        )
        # window_links speaks real time, so under a flip each link points the
        # other way through the window. A dividing parent becomes two links into
        # one node, which forward time never contains -- see time_flip_p.
        if time_flip:
            links = [(b, a) for a, b in links]
            pl = [(b, a) for a, b in pl]
        target, supervised, weight, n_missed = edge_targets(
            edges, links, n_nodes, pl_links=pl, pl_weight=self.pl_weight,
            division_weight=self.division_weight,
        )

        return {
            "node_feats": torch.from_numpy(
                node_features(pos_um, frame_idx, emb, len(frames), self.pos_scale,
                              conf=conf)
            ),
            "node_pos": torch.from_numpy(pos_um),
            "node_t": torch.from_numpy(frame_idx),
            "node_row": torch.from_numpy(node_row),
            "node_global": torch.from_numpy(node_global),
            "edge_index": torch.from_numpy(edges),
            "edge_pos": torch.from_numpy(
                0.5 * (pos_um[edges[:, 0]] + pos_um[edges[:, 1]])
                if len(edges) else np.zeros((0, 3), np.float32)
            ),
            "edge_feats": torch.from_numpy(
                edge_features(edges, pos_um, frame_idx, emb, n_nodes,
                              self.max_distance_um, self.max_dt, conf=conf)
                if len(edges) else np.zeros((0, self.edge_feat_dim), np.float32)
            ),
            "edge_target": torch.from_numpy(target),
            "edge_supervised": torch.from_numpy(supervised),
            "edge_weight": torch.from_numpy(weight),
            "n_missed_links": int(n_missed),
            "n_links": len(links),
            "n_pl_links": len(pl),
            "video_idx": int(video_idx),
            "t_start": int(t_start),
            # Read by predict.score_video to put a flipped window's edges back
            # into real time before averaging them with the forward pass.
            "time_flip": bool(time_flip),
            "name": f"{video.name}_t{t_start}",
        }

    def _augment_positions(self, pos_um: np.ndarray, rng) -> np.ndarray:
        """Flips, an in-plane rotation, and sub-voxel jitter.

        The features are precomputed, so nothing photometric is reachable from
        here, and anything that moved cells *relative to each other* by much
        would invent motion the ground truth does not describe. That leaves
        rigid transforms of the whole window, applied identically to every frame
        in it so the links they imply are unchanged.

        The first two are exact. Flips and rotations preserve every pairwise
        distance, so they change no distance or rank feature at all -- what they
        do change is the *direction* of every offset, which is what the rotary
        encoding and the segment geometry read. The rotation is in the y/x plane
        only: z is the coarse axis at 4x the spacing, and rotating it into y/x
        would claim a resolution the microscope never had.

        The jitter is not exact, and is the point of it. At ``jitter_um`` well
        below the ~8 um spacing it never reorders a true nearest neighbour, but
        it does perturb the distances the rank features are computed from -- so
        the model cannot learn to trust a rank margin finer than the detector's
        own centring error.
        """
        out = pos_um.copy()
        for axis in range(3):
            if rng.random() < self.flip_p:
                out[:, axis] = out[:, axis].max() + out[:, axis].min() - out[:, axis]

        if self.rot_p and rng.random() < self.rot_p:
            theta = rng.uniform(-np.pi, np.pi)
            cos, sin = np.cos(theta), np.sin(theta)
            centre = out[:, 1:3].mean(axis=0, keepdims=True)
            d = out[:, 1:3] - centre
            out[:, 1:3] = centre + np.stack(
                [cos * d[:, 0] - sin * d[:, 1], sin * d[:, 0] + cos * d[:, 1]], axis=1
            )

        if self.jitter_um:
            out = out + rng.uniform(
                -self.jitter_um, self.jitter_um, size=out.shape
            ).astype(np.float32)
        return out.astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        video_idx, t_start = self.windows[idx]
        return self.build(video_idx, t_start)


def collate_graphs(items: list[dict]) -> dict:
    """Pad a list of candidate graphs to the batch's own maxima.

    ``edge_index`` pads with 0 rather than -1: it is consumed by ``gather``,
    which would raise on a negative index, and the padded rows are masked out of
    both the attention and the loss.
    """
    items = [b for b in items if b is not None]
    n_max = max(int(b["node_feats"].shape[0]) for b in items)
    e_max = max(max(int(b["edge_index"].shape[0]) for b in items), 1)

    def pad(tensors, size, fill=0):
        out = []
        for t in tensors:
            shape = (size - t.shape[0], *t.shape[1:])
            if shape[0] == 0:
                out.append(t)
                continue
            out.append(torch.cat([t, torch.full(shape, fill, dtype=t.dtype)]))
        return torch.stack(out)

    node_keys = ("node_feats", "node_pos", "node_t", "node_row", "node_global")
    edge_keys = ("edge_index", "edge_pos", "edge_feats", "edge_target",
                 "edge_weight")

    batch = {k: pad([b[k] for b in items], n_max) for k in node_keys}
    batch.update({k: pad([b[k] for b in items], e_max) for k in edge_keys})
    batch["node_mask"] = pad(
        [torch.ones(b["node_feats"].shape[0], dtype=torch.bool) for b in items], n_max
    )
    batch["edge_mask"] = pad(
        [torch.ones(b["edge_index"].shape[0], dtype=torch.bool) for b in items], e_max
    )
    batch["edge_supervised"] = pad(
        [b["edge_supervised"] for b in items], e_max
    ) & batch["edge_mask"]

    for key in ("n_missed_links", "n_links", "n_pl_links", "video_idx", "t_start"):
        batch[key] = torch.tensor([b[key] for b in items])
    batch["name"] = [b["name"] for b in items]
    return batch
