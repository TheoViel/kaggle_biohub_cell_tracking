import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model_zoo.models_track import UNetNodeTransformer
from tracking.targets import detect_and_match


# Everything that changes what a cached node feature *is*. Stored beside the
# cache and checked on load, so a config edit invalidates it instead of
# silently training on features from a different detector.
_CACHE_SIGNATURE_KEYS = (
    "window_size",
    "downscale_factor",
    "detect_and_match_threshold",
    "max_dets_per_sample",
    "top_k_nodes",
    "nms",
    "nms_radius_um",
    "pool_kernel_um",
)


def fts_cache_dir(args, fold: int) -> Path:
    """Where the node-feature caches live.

    Beside the contrastive weights that produced them when those are reused, so
    a run pointing at a pretrained embedding UNet also inherits its cache;
    otherwise inside this run's own fold folder.
    """
    weights = (getattr(args, "contrastive", None) or {}).get("weights")
    if weights:
        weights = weights.format(fold=fold) if "{fold}" in weights else weights
        return Path(weights).parent / "fts_cache"
    return Path(args.output_dir) / "fts_cache"


def _cache_signature(args) -> dict:
    signature = {k: getattr(args, k, None) for k in _CACHE_SIGNATURE_KEYS}
    contrastive = getattr(args, "contrastive", None) or {}
    signature["contrastive"] = {
        k: contrastive.get(k)
        for k in ("weights", "embed_dim", "unet_out_channels")
    }
    signature["unet_features"] = list(getattr(args, "unet_features", []) or [])
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
        (coords[b, :n], pos[b, :n], matches[b], feat[b, :n])
        for coords, pos, mask, matches, feat in frame_det
        for n in [int(mask[b].sum())]
    ]


def collate_frame_det(samples: list) -> list:
    """Inverse of ``split_frame_det``: re-pad per-sample nodes into a batch.

    Returns the same ``(coords, pos_embed, mask, matches, unet_feat)`` per-frame
    layout ``encode_nodes`` produces, padded to the widest sample of this batch.
    """
    frame_det = []
    for w in range(len(samples[0])):
        n_max = max(max(s[w][0].shape[0] for s in samples), 1)
        ref = samples[0][w]
        shape = (len(samples), n_max)
        coords = ref[0].new_zeros(shape + (3,))
        pos = ref[1].new_zeros(shape + (ref[1].shape[-1],))
        feat = ref[3].new_zeros(shape + (ref[3].shape[-1],))
        mask = torch.zeros(shape, dtype=torch.bool, device=coords.device)
        matches = []
        for b, sample in enumerate(samples):
            c, p, m, f = sample[w]
            n = c.shape[0]
            coords[b, :n], pos[b, :n], feat[b, :n] = c, p, f
            mask[b, :n] = True
            matches.append(m)
        frame_det.append((coords, pos, mask, matches, feat))
    return frame_det


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
) -> tuple[list, list, tuple]:
    """UNet encode, then peak detection and GT matching on every frame.

    Returns ``(frame_det, det_logits, unet_scaling)`` where ``frame_det`` holds
    one ``(coords, pos_embed, mask, matches, unet_feat)`` tuple per frame, empty
    when ``detect`` is False (the detection-only warmup never reads it).

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
    unet_out, det_logits = core_model.encode(imgs)

    d_out, h_out, w_out = unet_out.shape[-3:]
    unet_scaling = (d // d_out, h // h_out, w // w_out)

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

    frame_det = []
    for i in range(window_size if detect else 0):
        det_coords, det_pos_embed, det_mask, matches = detect_and_match(
            det_logits[i].sigmoid(),
            coords[:, i],
            masks[:, i],
            image_shape,
            voxel_size=voxel_size,
            unet_scaling=unet_scaling,
            pool_kernel_um=args.pool_kernel_um,
            frame_index=i,
            window_size=window_size,
            det_threshold=args.detect_and_match_threshold,
            max_dets_per_sample=args.max_dets_per_sample,
            top_k_nodes=args.top_k_nodes,
            teacher_forcing=teacher_forcing,
            nms=nms,
            nms_radius_um=nms_radius_um,
        )
        unet_feat = torch.cat(
            [
                core_model._index_features(maps[:, i], det_coords, det_mask)
                for maps in source_maps
            ],
            dim=-1,
        ) if source_maps else det_coords.new_zeros(det_coords.shape[:2] + (0,))
        frame_det.append((det_coords, det_pos_embed, det_mask, matches, unet_feat))

    return frame_det, det_logits, unet_scaling
