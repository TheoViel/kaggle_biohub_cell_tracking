import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model_zoo.models_track import UNetNodeTransformer
from tracking.targets import detect_and_match, nodes_from_precomputed


# Everything that changes what a cached node feature *is*. Stored beside the
# cache and checked on load, so a config edit invalidates it instead of
# silently training on features from a different detector.
_CACHE_SIGNATURE_KEYS = (
    "window_size",
    "downscale_factor",
    "unet_n_extra_decode_steps",
    "detect_and_match_threshold",
    "max_dets_per_sample",
    "top_k_nodes",
    "nms",
    "nms_radius_um",
    "pool_kernel_um",
)


def resolve_run_path(path: str, fold: int, run_name: str | None = None) -> str:
    """Point a checkpoint path at the directory of the run being trained.

    Runs write to ``<output_dir>/<run_name>``, i.e. ``fold_0``, ``fold_1``, or
    ``fold_fullfit``. A config that reuses a previous run's checkpoints wants
    the matching one, so it writes ``fold_*`` once and this substitutes the
    current run's own name:

        .../fold_*/ckpt.pt   ->   .../fold_1/ckpt.pt   (on fold 1)
                             ->   .../fold_fullfit/ckpt.pt  (on fullfit)

    The older ``{fold}`` placeholder still works, but it can only ever name an
    integer, so it maps fullfit onto fold 0's checkpoint rather than its own.
    """
    if "{fold}" in path:
        path = path.format(fold=fold)
    return path.replace("fold_*", run_name or f"fold_{fold}")


def fts_cache_dir(args, fold: int) -> Path:
    """Where the node-feature caches live.

    Beside the contrastive weights that produced them when those are reused, so
    a run pointing at a pretrained embedding UNet also inherits its cache;
    otherwise inside this run's own fold folder.
    """
    # None when the run has no reusable upstream weights to key the cache on —
    # a fresh in-run pretraining, or a feature set neither branch covers.
    weights = None
    if getattr(args, "transformer_only", False):
        features = getattr(args, "unet_features", None) or ()
        if "contrastive" in features:
            weights = (getattr(args, "contrastive", None) or {}).get("weights")
        else:
            weights = (getattr(args, "pretraining", None) or {}).get("weights")

    if weights:
        # Same resolution as load_weights, so the cache sits beside the
        # checkpoint actually loaded. args.output_dir is this run's own
        # directory, whose name is the run name.
        run_name = Path(getattr(args, "output_dir", "") or "").name or None
        weights = resolve_run_path(weights, fold, run_name)
        return Path(weights).parent / "fts_cache" / (run_name or f"fold_{fold}")
    return Path(args.output_dir) / "fts_cache"


def _cache_signature(args) -> dict:
    signature = {k: getattr(args, k, None) for k in _CACHE_SIGNATURE_KEYS}
    contrastive = getattr(args, "contrastive", None) or {}
    signature["contrastive"] = {
        k: contrastive.get(k)
        for k in ("weights", "embed_dim", "unet_out_channels")
    }
    signature["unet_features"] = list(getattr(args, "unet_features", []) or [])
    signature["format"] = "v4-fine-det"
    signature["pretraining_weights"] = (getattr(args, "pretraining", None) or {}).get("weights")
    return {k: str(v) for k, v in signature.items()}


def _cache_path(folder, split: str, ctx) -> Path:
    # Per rank: with stable shards a rank always encodes the same windows, and
    # the world size is in the name so a different layout misses rather than
    # loading someone else's shard.
    return Path(folder) / f"{split}_rank{ctx.rank}of{ctx.world_size}.pt"


def load_fts_cache(folder, split: str, args, ctx, device) -> dict:
    """Cached node features from a previous run, or {} if unusable."""
    path = _cache_path(folder, split, ctx)
    if not path.exists():
        return {}
    blob = torch.load(path, map_location=device, weights_only=True)
    if blob.get("signature") != _cache_signature(args):
        return {}
    return blob["cache"]


def save_fts_cache(cache: dict, folder, split: str, args, ctx) -> Path:
    path = _cache_path(folder, split, ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"signature": _cache_signature(args), "cache": cache}, path)
    return path


def set_frozen_eval(core_model, trainable: tuple[str, ...]) -> None:
    """Put every top-level module outside ``trainable`` into eval mode.

    ``requires_grad_(False)`` stops gradients but not BatchNorm's running
    statistics: those are buffers, updated on the *forward* pass whenever the
    module is in train mode. A "frozen" detector therefore keeps drifting for
    as long as it is still being called — and the node features cached from it
    drift with it, so windows encoded early in an epoch disagree with those
    encoded late.

    train() recurses into children, so this has to be re-applied after every
    model.train() call, not just once at setup.
    """
    keep = tuple(t.rstrip(".") for t in trainable)
    for name, module in core_model.named_children():
        if name not in keep:
            module.eval()


def loader_without_images(loader: DataLoader) -> DataLoader:
    """Same loader, but its dataset stops reading the image volumes.

    Once every window's node features are cached, decompressing the volume
    again costs ~17 MB per window for nothing. The sampler is carried over as
    is, so the epoch order (shuffled or distributed) is unchanged; new workers
    pick up the flag because they re-pickle the dataset.
    """
    dataset = copy.copy(loader.dataset)
    dataset.load_images = False
    return DataLoader(
        dataset,
        batch_size=loader.batch_size,
        sampler=loader.sampler,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        persistent_workers=loader.persistent_workers,
        worker_init_fn=loader.worker_init_fn,
        prefetch_factor=loader.prefetch_factor,
        drop_last=loader.drop_last,
    )


def split_frame_det(frame_det: list, b: int) -> list:
    """One sample's node features, with the batch padding trimmed off.

    ``detect_and_match`` pads every sample to the widest node count *in its
    batch*, so caching the padded rows would tie a sample to the batch it was
    first seen in. Storing only its own nodes lets it be re-batched with any
    other samples later.
    """
    return [
        (coords[b, :n], pos[b, :n], matches[b], feat[b, :n], scores[b, :n])
        for coords, pos, mask, matches, feat, scores in frame_det
        for n in [int(mask[b].sum())]
    ]


def collate_frame_det(samples: list) -> list:
    """Inverse of ``split_frame_det``: re-pad per-sample nodes into a batch.

    Returns the same ``(coords, pos_embed, mask, matches, unet_feat, scores)``
    per-frame layout ``encode_nodes`` produces, padded to the widest sample of
    this batch.
    """
    frame_det = []
    for w in range(len(samples[0])):
        n_max = max(max(s[w][0].shape[0] for s in samples), 1)
        ref = samples[0][w]
        shape = (len(samples), n_max)
        coords = ref[0].new_zeros(shape + (3,))
        pos = ref[1].new_zeros(shape + (ref[1].shape[-1],))
        feat = ref[3].new_zeros(shape + (ref[3].shape[-1],))
        scores = ref[4].new_zeros(shape)
        mask = torch.zeros(shape, dtype=torch.bool, device=coords.device)
        matches = []
        for b, sample in enumerate(samples):
            c, p, m, f, sc = sample[w]
            n = c.shape[0]
            coords[b, :n], pos[b, :n], feat[b, :n], scores[b, :n] = c, p, f, sc
            mask[b, :n] = True
            matches.append(m)
        frame_det.append((coords, pos, mask, matches, feat, scores))
    return frame_det


def filter_frame_det(frame_det: list, threshold: float) -> list:
    """Same nodes, minus the ones scoring at or below ``threshold``.

    Detection ran once at the lowest threshold, so every higher one is a subset
    of those peaks. Re-padding the survivors gives the node set that threshold
    would have produced, ready for another edge forward — no second UNet pass.
    """
    samples = []
    for b in range(frame_det[0][0].shape[0]):
        per_frame = []
        for coords, pos, mask, matches, feat, scores in frame_det:
            n = int(mask[b].sum())
            keep = scores[b, :n] > threshold
            per_frame.append((
                coords[b, :n][keep],
                pos[b, :n][keep],
                matches[b][keep],
                feat[b, :n][keep],
                scores[b, :n][keep],
            ))
        samples.append(per_frame)
    return collate_frame_det(samples)


def encode_nodes(
    core_model: UNetNodeTransformer,
    imgs: torch.Tensor,          # (B, W, Z, Y, X)
    coords: torch.Tensor,        # (B, W, M, 3)
    masks: torch.Tensor,         # (B, W, M) bool
    image_shape: tuple,
    voxel_size: tuple,
    args,
    teacher_forcing: bool = False,
    detect: bool = True,
    nms: str = "maxpool",
    nms_radius_um: float = 7.0,
    node_features: bool = True,
) -> tuple[list, list, tuple, list | None, list | None, tuple]:
    """UNet encode, then peak detection and GT matching on every frame.

    Returns ``(frame_det, det_logits, unet_scaling, offset_maps, det_logits_ds,
    det_stride)`` where ``frame_det`` holds one ``(coords, pos_embed, mask,
    matches, unet_feat, scores)`` tuple per frame, empty when ``detect`` is
    False (the detection-only warmup never reads it). ``scores`` is each node's
    detection probability, which lets a higher threshold be applied later by
    filtering instead of re-detecting. ``offset_maps`` is the per-frame centroid
    vote, or None on a model built without the head.

    ``unet_scaling`` is the node grid — image voxels per node-grid unit — and is
    what every coordinate coming out of here is expressed in. ``det_stride`` is
    the same measure for the heatmap, which sits on a finer grid when the model
    runs extra decoder stages; the two are equal otherwise. ``det_logits_ds`` is
    the coarse head's heatmap in that case, on the node grid, and None when
    there is no second head to supervise.

    Node coordinates stay on the integer grid even when the vote exists: it
    refines where a detection is *reported*, not which voxel the edge scorer
    reads its features from, and training and inference must agree on that.

    ``nms`` defaults to the cheap max-pool path: greedy NMS costs one GPU round
    per emitted peak, which is fine once per evaluation but not per training
    step.

    Node features come from the contrastive UNet when the model has one — the
    detection UNet only says where cells are, not which is which. Set
    ``node_features=False`` to skip indexing them altogether, which is what the
    contrastive stage wants: it needs the coordinates but computes its own
    embeddings with gradient attached.
    """
    _, window_size, d, h, w = imgs.shape
    unet_out, det_logits, det_logits_ds = core_model.encode(imgs)
    offset_maps = core_model.encode_offsets(unet_out)

    d_out, h_out, w_out = unet_out.shape[-3:]
    unet_scaling = (d // d_out, h // h_out, w // w_out)

    # Voxels of the heatmap's grid per node-grid voxel: (1, 1, 1) unless the
    # model decodes further for the detection head alone.
    d_det, h_det, w_det = det_logits[0].shape[-3:] if det_logits else (d_out, h_out, w_out)
    det_scaling = (d_det // d_out, h_det // h_out, w_det // w_out)
    det_stride = tuple(u // s for u, s in zip(unet_scaling, det_scaling))

    # Which feature maps the nodes are read from, in the order the model's
    # unet_features lists them. Concatenation happens per node, not per voxel —
    # the maps are full-resolution volumes, the node features are a few
    # hundred vectors.
    source_maps = []
    if node_features and detect:
        for source in core_model.unet_features:
            source_maps.append(
                unet_out if source == "detection" else core_model.encode_embed(imgs)
            )

    probas = [logit.sigmoid() for logit in det_logits] if detect else []
    frame_det = []
    for i in range(window_size if detect else 0):
        det_coords, det_pos_embed, det_mask, matches = detect_and_match(
            probas[i],
            coords[:, i],
            masks[:, i],
            image_shape,
            voxel_size=voxel_size,
            unet_scaling=unet_scaling,
            det_scaling=det_scaling,
            pool_kernel_um=args.pool_kernel_um,
            frame_index=i,
            window_size=window_size,
            det_threshold=args.detect_and_match_threshold,
            max_dets_per_sample=args.max_dets_per_sample,
            top_k_nodes=args.top_k_nodes,
            teacher_forcing=teacher_forcing,
            nms=args.nms,
            nms_radius_um=args.nms_radius_um,
        )
        unet_feat = torch.cat(
            [
                core_model._index_features(maps[:, i], det_coords, det_mask)
                for maps in source_maps
            ],
            dim=-1,
        ) if source_maps else det_coords.new_zeros(det_coords.shape[:2] + (0,))
        # The peak's own probability, read back off the same grid it came from —
        # the heatmap's, so the node coordinates go back through det_scaling.
        det_scores = core_model._index_features(
            probas[i],
            det_coords * torch.tensor(det_scaling, dtype=det_coords.dtype, device=det_coords.device),
            det_mask,
        ).squeeze(-1)
        frame_det.append(
            (det_coords, det_pos_embed, det_mask, matches, unet_feat, det_scores)
        )

    return frame_det, det_logits, unet_scaling, offset_maps, det_logits_ds, det_stride


def node_scaling_from_voxel(voxel_size) -> tuple[float, float, float]:
    """Image voxels per isotropic node-grid unit, from the voxel size alone.

    The U-Net path gets an isotropic detection grid as a side effect of its
    downsampling -- ``pool_stride (1, 2, 2)`` over two stages contracts Y/X by
    4, which exactly cancels the 4x Z anisotropy of this data. With no U-Net
    there is nothing to measure, so the same grid is derived from the physics:
    one unit is the coarsest axis's voxel, and the finer axes are divided down
    to match. For (1.625, 0.40625, 0.40625) this returns (1, 4, 4) -- the very
    scaling the configured U-Net would have produced.
    """
    vs = [float(v) for v in voxel_size]
    coarsest = max(vs)
    return tuple(coarsest / v for v in vs)


def encode_nodes_precomputed(
    core_model: UNetNodeTransformer,
    pre_coords: torch.Tensor,   # (B, W, P, 3) image voxels, -1 padded
    pre_emb: torch.Tensor,      # (B, W, P, D)
    coords: torch.Tensor,       # (B, W, M, 3) GT
    masks: torch.Tensor,        # (B, W, M) bool
    image_shape: tuple,
    voxel_size: tuple,
    args,
) -> list:
    """``encode_nodes`` for a stage that reads its nodes off disk.

    The U-Net is never called: there is no heatmap to detect in, nothing to
    supervise on the detection side, and the node features are the NETr
    embeddings that arrived with the coordinates. The returned ``frame_det``
    has the same six-tuple shape the U-Net path produces plus the embeddings
    and their validity, so the training step reads both identically.

    ``unet_feat`` comes back empty rather than absent. The embeddings are node
    features too, but they are passed separately to ``predict_edges`` because
    the pair scorer needs the raw vectors, not just extra token width.
    """
    window_size = pre_coords.shape[1]
    scaling = node_scaling_from_voxel(voxel_size)

    frame_det = []
    for i in range(window_size):
        det_coords, det_pos, det_mask, matches, emb, emb_valid = nodes_from_precomputed(
            pre_coords[:, i],
            pre_emb[:, i] if pre_emb.shape[-1] else None,
            coords[:, i],
            masks[:, i],
            image_shape,
            voxel_size=voxel_size,
            node_scaling=scaling,
            frame_index=i,
            window_size=window_size,
            max_match_distance=getattr(args, "nms_radius_um", 7.0),
            max_dets_per_sample=getattr(args, "max_dets_per_sample", None),
        )
        unet_feat = det_coords.new_zeros(det_coords.shape[:2] + (0,))
        det_scores = det_coords.new_ones(det_coords.shape[:2])
        frame_det.append(
            (det_coords, det_pos, det_mask, matches, unet_feat, det_scores,
             emb, emb_valid)
        )
    return frame_det
