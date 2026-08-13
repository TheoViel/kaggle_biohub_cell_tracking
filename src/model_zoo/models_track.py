"""Cell tracking model: TemporalUNet3D + detection head + SimpleNodeTransformer."""

from __future__ import annotations

import torch
import torch.nn as nn

from model_zoo.simple_node_transformer import SimpleNodeTransformer
from model_zoo.temporal_unet import TemporalUNet3D

_POS_EMBED_DIM = 8  # per axis; total = 4 axes × _POS_EMBED_DIM = 32


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
        pos_feat_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.3,
        verbose: int = 0,
    ):
        super().__init__()
        self.unet = unet
        self.unet_out_channels = unet_out_channels

        self.detect_head = nn.Conv3d(unet_out_channels, 1, kernel_size=1)

        self.transformer = SimpleNodeTransformer(
            feat_dim=unet_out_channels + pos_feat_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            dropout=dropout,
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
        feat_src = torch.cat([unet_feat_src, pos_feat_src], dim=-1)
        feat_tgt = torch.cat([unet_feat_tgt, pos_feat_tgt], dim=-1)
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
    transformer_dropout: float = 0.3,
    gradient_checkpointing: bool = True,
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

    pos_feat_dim = 4 * _POS_EMBED_DIM  # 32

    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=unet_out_channels,
        layers=unet_layers,
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
        dropout=transformer_dropout,
        verbose=verbose,
    )
