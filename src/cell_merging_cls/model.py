"""`MergeNet`: an oriented pair clip -> one logit, "these two detections are one cell".

The backbone and the transformer are `division_cls.model`'s, unchanged and
imported rather than copied -- the question has the same shape (a short clip
whose *spatial* arrangement is the evidence, so the frame must not be pooled
before the frames are compared) and the sweep that chose those defaults applies
here too.

One thing is added. The crop is cut around the pair's midpoint and rotated onto
its axis, which throws away the two facts that survive no crop: how far apart
the detections are, and how much of that is in z rather than in the plane. Both
matter -- 61% of duplicate pairs are separated mostly in z -- so they are fed
back in as three scalars beside the CLS token. They enter *after* the
transformer, not as tokens, so the model cannot use the separation as a shortcut
for reading the pixels: a linear term on three numbers can shift the decision
boundary but cannot substitute for the image, which is the failure the
distance-only merge already demonstrated at 10 508 firings for 34 opportunities.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from division_cls.model import FrameEncoder

N_GEOM = 3          # separation, in-plane separation, z separation (um)


class MergeNet(nn.Module):
    """`(B, T, pz, H, W)` clip plus `(B, 3)` geometry -> one logit."""

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        patch_yx: int = 32,
        patch_z: int = 9,
        n_frames: int = 3,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        min_tokens: int = 4,
        geom_dim: int = 32,
        grad_checkpointing: bool = False,
        verbose: int = 0,
    ):
        super().__init__()
        self.n_frames = int(n_frames)
        self.encoder = FrameEncoder(
            backbone, pretrained=pretrained, in_chans=patch_z,
            img_size=patch_yx, min_tokens=min_tokens, verbose=verbose,
        )
        grid = self.encoder.grid
        self.n_tokens = grid * grid

        self.proj = nn.Linear(self.encoder.feat_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.frame_embed = nn.Embedding(self.n_frames, d_model)
        self.space_embed = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.space_embed, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.geom = nn.Sequential(
            nn.Linear(N_GEOM, geom_dim), nn.GELU(), nn.Linear(geom_dim, geom_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model + geom_dim), nn.Dropout(dropout),
            nn.Linear(d_model + geom_dim, 1),
        )

        if grad_checkpointing and hasattr(self.encoder.backbone, "set_grad_checkpointing"):
            self.encoder.backbone.set_grad_checkpointing(True)
        if verbose:
            print(f"  - pair clip {n_frames}x{patch_z}x{patch_yx}x{patch_yx} -> "
                  f"{n_frames} x {self.n_tokens} tokens + CLS, d_model {d_model}"
                  f" (+{N_GEOM} geometry scalars)")

    def forward(self, clip: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        b, t = clip.shape[:2]
        if t != self.n_frames:
            raise ValueError(f"model built for {self.n_frames} frames, got {t}")

        tokens = self.encoder(clip.flatten(0, 1))
        tokens = self.norm(self.proj(tokens))
        tokens = tokens + self.space_embed
        tokens = tokens.reshape(b, t, self.n_tokens, -1)
        tokens = tokens + self.frame_embed.weight[None, :, None, :]
        tokens = tokens.reshape(b, t * self.n_tokens, -1)

        x = torch.cat([self.cls.expand(b, -1, -1), tokens], dim=1)
        pooled = self.transformer(x)[:, 0]
        return self.head(torch.cat([pooled, self.geom(geom)], dim=-1)).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
