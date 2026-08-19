"""Cell tracking model: TemporalUNet3D + detection head + SimpleNodeTransformer."""

from __future__ import annotations

import torch
import torch.nn as nn

from model_zoo.simple_node_transformer import SimpleNodeTransformer
from model_zoo.temporal_unet import TemporalUNet3D


class UNetNodeTransformer(nn.Module):
    """TemporalUNet3D encoder + detection head + SimpleNodeTransformer edge predictor.

    Forward path during training:
      1. imgs (B, W, *spatial) → UNet → (B, W, C_feat, *spatial) + W×(B,1,*spatial) det_logits
      2. Integer-index feature maps at detected node coords (round + clamp; differentiable)
      3. Concatenate with sinusoidal positional embeddings
      4. Cross-attention transformer → (B, N_src, N_tgt) edge logits per pair
    """

    def __init__(
        self,
        unet: TemporalUNet3D,
        unet_out_channels: int,
        pos_feat_dim: int = 0,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.3,
        edge_knn_k: int | None = None,
        edge_geom_feats: bool = True,
        embed_unet: TemporalUNet3D | None = None,
        embed_unet_out_channels: int = 32,
        embed_dim: int = 32,
        unet_features: tuple[str, ...] = ("detection",),
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

        self.use_pos_embed = pos_feat_dim > 0
        self.transformer = SimpleNodeTransformer(
            feat_dim=node_feat_dim + pos_feat_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            dropout=dropout,
            knn_k=edge_knn_k,
            use_geom_feats=edge_geom_feats,
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
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run TemporalUNet3D on W frames.

        Returns
        -------
        unet_out : (B, W, C_feat, *spatial)
        det_logits : list of W tensors, each (B, 1, *spatial)
        """
        if self.verbose >= 1:
            print(f"UNetNodeTransformer - Encode - Input shape: {imgs.shape}")

        window = imgs.unsqueeze(2)          # (B, W, 1, *spatial)
        unet_out = self.unet(window)        # (B, W, C_feat, *spatial)

        if self.verbose >= 1:
            print(f"UNetNodeTransformer - Encode - UNet output shape: {unet_out.shape}")

        W = unet_out.shape[1]
        det_logits = [self.detect_head(unet_out[:, i]) for i in range(W)]

        if self.verbose >= 1:
            print(f"UNetNodeTransformer - Encode - Detect logits shape: {[det.shape for det in det_logits]}")

        return unet_out, det_logits

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
    ) -> torch.Tensor:
        """Run cross-attention edge predictor on pre-indexed UNet features."""
        if self.use_pos_embed:
            feat_src = torch.cat([unet_feat_src, pos_feat_src], dim=-1)
            feat_tgt = torch.cat([unet_feat_tgt, pos_feat_tgt], dim=-1)
        else:
            feat_src = unet_feat_src
            feat_tgt = unet_feat_tgt

        return self.transformer(feat_src, feat_tgt, coords_src, coords_tgt, mask_src, mask_tgt)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def define_track_model(
    unet_out_channels: int = 32,
    unet_layers: list[int] | None = None,
    unet_pool_stride: int | tuple[int, int, int] = 2,
    unet_n_decode_steps: int | None = None,
    transformer_hidden: int = 128,
    transformer_heads: int = 4,
    transformer_blocks: int = 4,
    edge_knn_k: int | None = None,
    edge_geom_feats: bool = True,
    pos_feat_dim: int = 0,
    transformer_dropout: float = 0.3,
    gradient_checkpointing: bool = False,
    contrastive: dict | None = None,
    unet_features: tuple[str, ...] = ("detection",),
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
        decoder (output at bottleneck resolution).
    """
    if unet_layers is None:
        unet_layers = [32, 64, 128]

    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=unet_out_channels,
        layers=unet_layers,
        gradient_checkpointing=gradient_checkpointing,
        pool_stride=unet_pool_stride,
        n_decode_steps=unet_n_decode_steps,
        verbose=verbose,
    )

    # The embedding UNet keeps the detection UNet's output stride, so node
    # coordinates index both feature maps identically. It exists only if the
    # transformer is asked for its features — leaving "contrastive" out of
    # unet_features is how the whole branch, and its training stage, is turned
    # off.
    contrastive = contrastive or {}
    embed_unet = None
    if "contrastive" in unet_features:
        embed_unet = TemporalUNet3D(
            in_channels=1,
            out_channels=contrastive.get("unet_out_channels", 32),
            layers=list(contrastive.get("unet_layers", unet_layers)),
            gradient_checkpointing=gradient_checkpointing,
            pool_stride=unet_pool_stride,
            n_decode_steps=unet_n_decode_steps,
            verbose=verbose,
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
        embed_unet=embed_unet,
        embed_unet_out_channels=contrastive.get("unet_out_channels", 32),
        embed_dim=contrastive.get("embed_dim", 32),
        unet_features=unet_features,
        dropout=transformer_dropout,
        verbose=verbose,
    )
