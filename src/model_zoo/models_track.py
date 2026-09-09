"""Cell tracking model: temporal U-Net + detection head + SimpleNodeTransformer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_zoo.simple_node_transformer import SimpleNodeTransformer
from model_zoo.temporal_resenc_unet import TemporalResEncUNet3D
from model_zoo.temporal_unet import TemporalUNet3D


# Backbones the tracking model can be built on, by name. "temporal" is the plain two-conv
# U-Net; "resenc" is the nnU-Net residual encoder at the same geometry, which is deeper at
# equal widths. Both take the same geometry arguments and return (B, T, C, Z', Y', X').
UNET_BACKBONES = {"temporal": TemporalUNet3D, "resenc": TemporalResEncUNet3D}


class UNetNodeTransformer(nn.Module):
    """Temporal U-Net encoder + detection head + SimpleNodeTransformer edge predictor.

    Forward path during training:
      1. imgs (B, W, *spatial) → UNet → (B, W, C_feat, *spatial) + W×(B,1,*spatial) det_logits
      2. Integer-index feature maps at detected node coords (round + clamp; differentiable)
      3. Concatenate with sinusoidal positional embeddings
      4. Cross-attention transformer → (B, N_src, N_tgt) edge logits per pair
    """

    def __init__(
        self,
        unet: TemporalUNet3D | TemporalResEncUNet3D,
        unet_out_channels: int,
        pos_feat_dim: int = 0,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.3,
        edge_knn_k: int | None = None,
        edge_geom_feats: bool = True,
        edge_contrastive_feats: bool = False,
        contrastive_feat_dim: int = 0,
        edge_attention: bool = True,
        edge_drift_correction: bool = False,
        embed_unet: TemporalUNet3D | TemporalResEncUNet3D | None = None,
        embed_unet_out_channels: int = 32,
        embed_dim: int = 32,
        unet_features: tuple[str, ...] = ("detection",),
        offset_head: bool = False,
        offset_radius: int = 2,
        verbose: int = 0,
    ):
        super().__init__()
        self.unet = unet
        self.unet_out_channels = unet_out_channels

        self.detect_head = nn.Sequential(
            nn.Conv3d(unet_out_channels, unet_out_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(unet_out_channels, 1, kernel_size=1),
        )

        # A backbone with extra decoder stages hands back a second, finer map,
        # and the heatmap is what wants that resolution: a peak on the coarse
        # grid is only ever located to within its own 1.625 um cell. The head
        # above then moves from being *the* detector to being a first pass over
        # it — still supervised, on the target it has always had, and its answer
        # is handed forward rather than thrown away. `detect_head_fine` sees the
        # fine features, the coarse head's hidden activations and the coarse
        # logit itself, so it starts from a decision it only has to refine.
        self.deep_supervision = bool(getattr(unet, "n_extra_decode_steps", 0))
        self.detect_head_fine = None
        if self.deep_supervision:
            self.detect_head_fine = nn.Sequential(
                nn.Conv3d(2 * unet_out_channels + 1, unet_out_channels, kernel_size=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv3d(unet_out_channels, 1, kernel_size=1),
            )

        # Where the cell actually is, as a vector from each voxel to its centre
        # — the segmentation branch's centroid vote on the detection grid. The
        # detect head answers "a cell is in this 1.625 um cell" and the grid
        # centre is then the best guess available, which is worth about 0.8 um
        # of error on its own and rather more when the peak lands one cell off.
        self.offset_radius = int(offset_radius)
        self.offset_head = None
        if offset_head:
            self.offset_head = nn.Sequential(
                nn.Conv3d(unet_out_channels, unet_out_channels, kernel_size=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv3d(unet_out_channels, 3, kernel_size=1),
            )

        # Second, independent UNet trained purely to tell cells apart. The
        # detection UNet cannot do that job: its loss asks only "is a cell
        # here", so two look-alike neighbours *should* map to the same feature.
        self.embed_unet = embed_unet
        self.embed_dim = embed_dim
        if embed_unet is not None:
            self.embed_head = nn.Sequential(
                nn.Conv3d(embed_unet_out_channels, embed_unet_out_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv3d(embed_unet_out_channels, embed_dim, kernel_size=1),
            )

        # Which feature maps the node vectors are read from, in this order. An
        # empty list leaves the transformer with geometry alone — the pair MLP
        # still gets the relative offset and edge_geom_feats, and every node
        # token becomes the same learned constant.
        self.unet_features = tuple(unet_features)
        widths = {"detection": unet_out_channels, "contrastive": embed_dim}
        unknown = set(self.unet_features) - set(widths)
        if unknown:
            raise ValueError(
                f"unknown unet_features {sorted(unknown)}, expected 'detection' / 'contrastive'"
            )
        if "contrastive" in self.unet_features and embed_unet is None:
            raise ValueError("unet_features asks for 'contrastive' but no embed_unet was built")
        node_feat_dim = sum(widths[name] for name in self.unet_features)

        # NETr embeddings, precomputed by inference_contrastive.py and carried
        # in with the nodes rather than read off a feature map. They enter twice
        # and the two uses are independent: as extra node-token width here, and
        # as the pair statistics of _contrastive_feats when
        # edge_contrastive_feats is on. Width 0 means no embeddings at all.
        self.contrastive_feat_dim = int(contrastive_feat_dim)
        self.edge_contrastive_feats = bool(edge_contrastive_feats)
        if self.edge_contrastive_feats and self.contrastive_feat_dim == 0:
            raise ValueError(
                "edge_contrastive_feats needs contrastive_feat_dim > 0; set it "
                "to the embedding width the contrastive_folder holds"
            )

        self.use_pos_embed = pos_feat_dim > 0
        self.transformer = SimpleNodeTransformer(
            feat_dim=node_feat_dim + pos_feat_dim + self.contrastive_feat_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            dropout=dropout,
            knn_k=edge_knn_k,
            use_geom_feats=edge_geom_feats,
            use_contrastive_feats=self.edge_contrastive_feats,
            use_attention=edge_attention,
            drift_correction=edge_drift_correction,
        )

        self.verbose = verbose

    def _index_features(
        self,
        feat_maps: torch.Tensor,  # (B, C, *spatial)
        coords: torch.Tensor,     # (B, max_nodes, 3) — downsampled space
        mask: torch.Tensor,       # (B, max_nodes) bool
    ) -> torch.Tensor:
        """Integer-index feat_maps at node positions; padded slots → zeros."""
        B, C = feat_maps.shape[:2]
        spatial = feat_maps.shape[2:]
        max_nodes = coords.shape[1]

        out = torch.zeros(B, max_nodes, C, device=feat_maps.device, dtype=feat_maps.dtype)
        for b in range(B):
            nt = int(mask[b].sum().item())
            if nt == 0:
                continue
            z = coords[b, :nt, 0].long().clamp(0, spatial[0] - 1)
            y = coords[b, :nt, 1].long().clamp(0, spatial[1] - 1)
            x = coords[b, :nt, 2].long().clamp(0, spatial[2] - 1)
            out[b, :nt] = feat_maps[b, :, z, y, x].T
        return out

    def encode(
        self,
        imgs: torch.Tensor,  # (B, W, *spatial) — already downsampled, single channel
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor] | None]:
        """Run the temporal U-Net on W frames.

        ``unet_out`` is the node-feature map and never moves: it stays on the
        grid ``unet.output_stride`` names, which is the isotropic one the edge
        scorer measures distances in. With deep supervision the heatmap alone
        moves onto the finer grid, and ``det_logits_ds`` is the coarse head's
        answer on the old grid, kept so it can still be supervised.

        Returns
        -------
        unet_out : (B, W, C_feat, *spatial)
        det_logits : list of W tensors, each (B, 1, *fine_spatial)
        det_logits_ds : list of W tensors on ``unet_out``'s grid, or None when
            the model has no extra decoder branch (then det_logits is on that
            grid too).
        """
        if self.verbose >= 1:
            print(f"UNetNodeTransformer - Encode - Input shape: {imgs.shape}")

        window = imgs.unsqueeze(2)          # (B, W, 1, *spatial)
        out = self.unet(window)             # (B, W, C_feat, *spatial)
        unet_out, fine_out = out if self.deep_supervision else (out, None)

        if self.verbose >= 1:
            print(f"UNetNodeTransformer - Encode - UNet output shape: {unet_out.shape}")

        W = unet_out.shape[1]
        if fine_out is None:
            det_logits, det_logits_ds = [
                self.detect_head(unet_out[:, i]) for i in range(W)
            ], None
        else:
            det_logits, det_logits_ds = [], []
            for i in range(W):
                # Split at the head's own hidden layer, so the coarse head keeps
                # the exact parameters (and names) it has always had.
                hidden = self.detect_head[:3](unet_out[:, i])
                ds_logit = self.detect_head[3](hidden)
                carry = F.interpolate(
                    torch.cat([hidden, ds_logit], dim=1),
                    size=fine_out.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
                det_logits.append(
                    self.detect_head_fine(torch.cat([fine_out[:, i], carry], dim=1))
                )
                det_logits_ds.append(ds_logit)

        if self.verbose >= 1:
            print(f"UNetNodeTransformer - Encode - Detect logits shape: {[det.shape for det in det_logits]}")

        return unet_out, det_logits, det_logits_ds

    def encode_offsets(
        self,
        unet_out: torch.Tensor,  # (B, W, C, *spatial) — from encode()
    ) -> list[torch.Tensor] | None:
        """Centroid votes per frame, in detection-grid voxels, or None.

        Bounded to ``offset_radius + 1`` voxels by a tanh. Nothing in the target
        ever asks for more — a voxel only votes for a cell inside the radius —
        and at inference an unbounded head could throw a detection clear across
        the field on a voxel it has no opinion about.

        Takes ``unet_out`` rather than the images so the encoder runs once;
        callers already have it from ``encode``.

        Returns
        -------
        list of W tensors, each (B, 3, *spatial), or None if the head is off.
        """
        if self.offset_head is None:
            return None
        limit = float(self.offset_radius + 1)
        return [
            limit * torch.tanh(self.offset_head(unet_out[:, i]))
            for i in range(unet_out.shape[1])
        ]

    def encode_embed(
        self,
        imgs: torch.Tensor,  # (B, W, *spatial)
    ) -> torch.Tensor:
        """Run the contrastive UNet, returning L2-normalised node embeddings.

        Normalising here means every downstream similarity is a cosine, so the
        contrastive temperature is the only thing setting the logit scale.

        Returns
        -------
        (B, W, embed_dim, *spatial)
        """
        if self.embed_unet is None:
            raise RuntimeError("encode_embed called on a model with no embed_unet")

        out = self.embed_unet(imgs.unsqueeze(2))     # (B, W, C, *spatial)
        W = out.shape[1]
        embed = torch.stack([self.embed_head(out[:, i]) for i in range(W)], dim=1)
        return torch.nn.functional.normalize(embed, dim=2)

    def predict_edges(
        self,
        unet_feat_src: torch.Tensor,  # (B, N_src, C_feat) pre-indexed
        unet_feat_tgt: torch.Tensor,  # (B, N_tgt, C_feat) pre-indexed
        coords_src: torch.Tensor,     # (B, N_src, 3) physical coords
        coords_tgt: torch.Tensor,     # (B, N_tgt, 3) physical coords
        pos_feat_src: torch.Tensor,   # (B, N_src, pos_feat_dim)
        pos_feat_tgt: torch.Tensor,   # (B, N_tgt, pos_feat_dim)
        mask_src: torch.Tensor,       # (B, N_src) bool
        mask_tgt: torch.Tensor,       # (B, N_tgt) bool
        emb_src: torch.Tensor | None = None,   # (B, N_src, D) L2-normalised
        emb_tgt: torch.Tensor | None = None,   # (B, N_tgt, D)
        emb_valid_src: torch.Tensor | None = None,  # (B, N_src) has an embedding
        emb_valid_tgt: torch.Tensor | None = None,  # (B, N_tgt)
    ) -> torch.Tensor:
        """Run cross-attention edge predictor on pre-indexed node features."""
        parts_src = [unet_feat_src]
        parts_tgt = [unet_feat_tgt]
        if self.use_pos_embed:
            parts_src.append(pos_feat_src)
            parts_tgt.append(pos_feat_tgt)
        if self.contrastive_feat_dim:
            if emb_src is None:
                raise ValueError(
                    "the model was built with contrastive_feat_dim="
                    f"{self.contrastive_feat_dim} but no embeddings were passed"
                )
            parts_src.append(emb_src.to(unet_feat_src.dtype))
            parts_tgt.append(emb_tgt.to(unet_feat_tgt.dtype))

        feat_src = torch.cat(parts_src, dim=-1) if len(parts_src) > 1 else parts_src[0]
        feat_tgt = torch.cat(parts_tgt, dim=-1) if len(parts_tgt) > 1 else parts_tgt[0]

        return self.transformer(
            feat_src, feat_tgt, coords_src, coords_tgt, mask_src, mask_tgt,
            emb_t=emb_src, emb_t1=emb_tgt,
            emb_valid_t=emb_valid_src, emb_valid_t1=emb_valid_tgt,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def define_track_model(
    unet_out_channels: int = 32,
    unet_layers: list[int] | None = None,
    unet_pool_stride: int | tuple[int, int, int] = 2,
    unet_n_decode_steps: int | None = None,
    unet_n_extra_decode_steps: int = 0,
    unet_arch: str = "temporal",
    unet_n_anisotropic: int = 2,
    unet_blocks_per_stage: list[int] | None = None,
    unet_temporal_attention: bool = True,
    unet_spatial_attention: str = "none",
    unet_spatial_attention_stages: int = 1,
    unet_spatial_attention_heads: int = 4,
    unet_spatial_attention_reduction: int | tuple[int, int, int] = 4,
    unet_spatial_attention_window: tuple[int, int, int] = (4, 8, 8),
    unet_stem: str = "stride",
    unet_norm: str = "batch",
    unet_hires_fusion: bool = False,
    transformer_hidden: int = 128,
    transformer_heads: int = 4,
    transformer_blocks: int = 4,
    edge_knn_k: int | None = None,
    edge_geom_feats: bool = True,
    edge_contrastive_feats: bool = False,
    contrastive_feat_dim: int = 0,
    edge_attention: bool = True,
    edge_drift_correction: bool = False,
    pos_feat_dim: int = 0,
    transformer_dropout: float = 0.3,
    gradient_checkpointing: bool = False,
    contrastive: dict | None = None,
    unet_features: tuple[str, ...] = ("detection",),
    offset_head: bool = False,
    offset_radius: int = 2,
    verbose: int = 0,
) -> UNetNodeTransformer:
    """Build the full tracking model (UNet + detect head + transformer).

    Parameters
    ----------
    unet_out_channels : int
        Feature channels produced by the UNet per frame.
    unet_layers : list of int
        Encoder channel widths, shallow to deep (default: [32, 64, 128]).
    unet_pool_stride : int or (int, int, int)
        Pool/upsample stride per encoder level.  Use ``(1, 2, 2)`` to
        pool only in Y/X when passing full-scale 3D images.
    unet_n_decode_steps : int or None
        Number of decoder stages.  None = fully symmetric.  0 = no
        decoder (output at bottleneck resolution).  This sets the grid the
        *node features* live on, and with them everything the edge scorer
        measures, so it is the one to leave alone.
    unet_n_extra_decode_steps : int
        Decoder stages built past that grid, feeding the detection head a finer
        heatmap while the node features stay put.  The head on the coarse grid
        is kept and supervised on the target it already had, and its output is
        fed forward into the fine head.  Temporal backbone only.
    unet_arch : str
        Which backbone to build, see ``UNET_BACKBONES``.  ``"temporal"``
        is the plain two-conv U-Net, ``"resenc"`` the nnU-Net residual
        encoder at the same geometry.
    unet_n_anisotropic : int
        How many downsampling steps use ``unet_pool_stride`` before the
        rest go isotropic.  The stem's own step counts as the first.
    unet_blocks_per_stage : list of int or None
        Residual blocks per encoder stage.  None takes nnU-Net's
        ``(1, 3, 4, 6, 6, ...)`` on ResEnc and the plain two-conv block on
        ``TemporalUNet3D``; give it e.g. ``[1, 3, 4]`` to put residual depth
        behind the temporal backbone as well.
    unet_temporal_attention : bool
        If False, the backbone drops its temporal attention blocks and the
        frames of a window are encoded independently.
    unet_spatial_attention : {"none", "pooled", "window", "cbam"}
        Convolutional attention on the spatial axes, inside the backbone. The
        temporal blocks only ever mix a voxel with itself across frames, so
        without this nothing in the encoder lets two *places* interact beyond
        the reach of the convolution stack. ``"pooled"`` attends globally over
        convolutionally reduced keys, ``"window"`` locally inside fixed windows,
        ``"cbam"`` re-weights without any pairwise interaction. See
        ``temporal_unet.spatial_attention_block``.
    unet_spatial_attention_stages : int
        How many of the deepest encoder stages carry a block. 1 is the
        bottleneck alone, 0 turns it off.
    unet_spatial_attention_heads : int
        Attention heads, reduced to a divisor of the stage width if it is not
        one already.
    unet_spatial_attention_reduction : int or (int, int, int)
        ``"pooled"`` only: the stride of the convolution the keys and values
        are read through.
    unet_spatial_attention_window : (int, int, int)
        ``"window"`` only: the window each voxel attends inside.
    unet_stem : str
        Temporal backbone only: ``"unshuffle"`` folds the stem's downsampling
        block into the channel axis so no Y/X detail is dropped before the
        first convolution; ``"stride"`` is the strided convolution it replaces.
    unet_norm : str
        Temporal backbone only: ``"batch"`` / ``"instance"`` / ``"group"``.
    unet_hires_fusion : bool
        Temporal backbone only: pool the encoder skips finer than the head's
        grid back into the head instead of discarding them.
    """
    if unet_layers is None:
        unet_layers = [32, 64, 128]
    if unet_arch not in UNET_BACKBONES:
        raise ValueError(f"unknown unet_arch {unet_arch!r}, expected one of {sorted(UNET_BACKBONES)}")

    backbone = UNET_BACKBONES[unet_arch]
    # Stem layout, normalisation and skip fusion are TemporalUNet3D's; the ResEnc
    # encoder fixes all three by construction.
    extra = (
        {}
        if unet_arch == "resenc"
        else {"stem": unet_stem, "norm": unet_norm, "hires_skip_fusion": unet_hires_fusion}
    )
    # Both backbones take these, so they go in the shared block rather than in
    # `extra`; "none" leaves every stage an Identity and the model unchanged.
    extra["spatial_attention"] = unet_spatial_attention
    extra["spatial_attention_stages"] = unet_spatial_attention_stages
    extra["spatial_attention_heads"] = unet_spatial_attention_heads
    extra["spatial_attention_reduction"] = unet_spatial_attention_reduction
    extra["spatial_attention_window"] = tuple(unet_spatial_attention_window)

    unet = backbone(
        in_channels=1,
        out_channels=unet_out_channels,
        layers=unet_layers,
        blocks_per_stage=unet_blocks_per_stage,
        gradient_checkpointing=gradient_checkpointing,
        pool_stride=unet_pool_stride,
        n_anisotropic=unet_n_anisotropic,
        n_decode_steps=unet_n_decode_steps,
        n_extra_decode_steps=unet_n_extra_decode_steps,
        temporal_attention=unet_temporal_attention,
        verbose=verbose,
        **extra,
    )

    # The embedding UNet keeps the detection UNet's output stride, so node
    # coordinates index both feature maps identically. It exists only if the
    # transformer is asked for its features — leaving "contrastive" out of
    # unet_features is how the whole branch, and its training stage, is turned
    # off.
    contrastive = contrastive or {}
    embed_unet = None
    if "contrastive" in unet_features:
        embed_layers = list(contrastive.get("unet_layers", unet_layers))
        embed_unet = backbone(
            in_channels=1,
            out_channels=contrastive.get("unet_out_channels", 32),
            layers=embed_layers,
            # The contrastive branch may run a shallower encoder, in which case
            # the depth schedule is not its to reuse.
            blocks_per_stage=(
                unet_blocks_per_stage
                if unet_blocks_per_stage and len(unet_blocks_per_stage) == len(embed_layers)
                else None
            ),
            gradient_checkpointing=gradient_checkpointing,
            pool_stride=unet_pool_stride,
            n_anisotropic=unet_n_anisotropic,
            n_decode_steps=unet_n_decode_steps,
            temporal_attention=unet_temporal_attention,
            verbose=verbose,
            **extra,
        )

    return UNetNodeTransformer(
        unet=unet,
        unet_out_channels=unet_out_channels,
        pos_feat_dim=pos_feat_dim,
        hidden_dim=transformer_hidden,
        n_heads=transformer_heads,
        n_blocks=transformer_blocks,
        edge_knn_k=edge_knn_k,
        edge_geom_feats=edge_geom_feats,
        edge_contrastive_feats=edge_contrastive_feats,
        contrastive_feat_dim=contrastive_feat_dim,
        edge_attention=edge_attention,
        edge_drift_correction=edge_drift_correction,
        embed_unet=embed_unet,
        embed_unet_out_channels=contrastive.get("unet_out_channels", 32),
        embed_dim=contrastive.get("embed_dim", 32),
        unet_features=unet_features,
        offset_head=offset_head,
        offset_radius=offset_radius,
        dropout=transformer_dropout,
        verbose=verbose,
    )
