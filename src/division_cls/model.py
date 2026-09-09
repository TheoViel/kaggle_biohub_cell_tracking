"""A 2D CNN per frame, then one transformer over every frame's feature map.

The alternative would have been to pool each frame to a vector and aggregate
those, which is what `embeddings/model.py` does. That cannot work here. A
division is a *spatial* event -- one blob becomes two, and which way they part
matters -- so pooling the frame away before the frames are compared throws out
the thing being classified.

So the CNN keeps its map. A `(H, W, d)` map per frame becomes `H*W` tokens, the
`n_frames` maps are concatenated into one sequence, and a transformer attends
over all of it at once. A token in the last frame can therefore look directly at
the position it occupied in the first, which is the comparison a human makes
when they say a cell has divided.

Two embeddings carry what concatenation destroys:

* a **frame embedding**, learned per time index, so the sequence knows which
  frame a token came from -- without it the clip is a bag and "one blob became
  two" is indistinguishable from "two blobs became one";
* a **spatial embedding**, learned per `(H, W)` cell and shared across frames,
  so the same position in different frames gets the same code and the model can
  tell "moved" from "appeared elsewhere".

Shared, not per-frame: the point is that position `p` at `t-1` and position `p`
at `t+1` are the same place.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    """A timm backbone cut at the deepest stage still wider than `min_tokens`.

    At a 32-voxel crop most pyramids have collapsed to 1x1 by their last stage,
    which is the pooled vector this module exists to avoid. The stage is chosen
    by measuring rather than by assuming, so the backbone stays a config string.
    """

    def __init__(self, name="convnext_tiny", pretrained=True, in_chans=3,
                 img_size=32, min_tokens=4, verbose=0):
        super().__init__()
        import timm

        def create(**kw):
            try:
                return timm.create_model(name, pretrained=pretrained,
                                         img_size=img_size, in_chans=in_chans, **kw)
            except TypeError:       # fully convolutional: no img_size kwarg
                return timm.create_model(name, pretrained=pretrained,
                                         in_chans=in_chans, **kw)

        probe = create(features_only=True)
        # eval(), and not just no_grad: the probe is a single sample, and a
        # BatchNorm backbone in train mode raises on a batch of one. convnext
        # normalises over the layer and survives either way, which is exactly
        # how this went unnoticed until a resnet was tried.
        probe.eval()
        with torch.no_grad():
            maps = probe(torch.zeros(1, in_chans, img_size, img_size))
            sizes = [int(min(m.shape[-2:])) for m in maps]
        channels = probe.feature_info.channels()
        del probe

        clears = [i for i, s in enumerate(sizes) if s >= min_tokens]
        idx = max(clears) if clears else 0
        self.backbone = create(features_only=True, out_indices=(idx,))
        self.feat_dim = int(channels[idx])
        self.grid = int(sizes[idx])
        if verbose:
            print(f"  - {name}: stages {sizes} at {img_size}px -> stage {idx}, "
                  f"{self.grid}x{self.grid} x {self.feat_dim}ch")

    def forward(self, x):
        """(M, C, H, W) -> (M, tokens, feat_dim)."""
        m = self.backbone(x)[-1]
        if m.shape[1] != self.feat_dim:     # timm emits NHWC for some nets
            m = m.permute(0, 3, 1, 2)
        return m.flatten(2).transpose(1, 2)


class DivisionNet(nn.Module):
    """`(B, T, C, H, W)` clip -> one logit, "this cell divides at the centre frame"."""

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        patch_yx: int = 32,
        patch_z: int = 3,
        n_frames: int = 3,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        min_tokens: int = 4,
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
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1),
        )

        if grad_checkpointing and hasattr(self.encoder.backbone, "set_grad_checkpointing"):
            self.encoder.backbone.set_grad_checkpointing(True)
        if verbose:
            print(f"  - clip {n_frames}x{patch_z}x{patch_yx}x{patch_yx} -> "
                  f"{n_frames} x {self.n_tokens} tokens + CLS, d_model {d_model}")

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        b, t = clip.shape[:2]
        if t != self.n_frames:
            raise ValueError(f"model built for {self.n_frames} frames, got {t}")

        tokens = self.encoder(clip.flatten(0, 1))            # (B*T, N, feat)
        tokens = self.norm(self.proj(tokens))
        tokens = tokens + self.space_embed                   # same code per place
        tokens = tokens.reshape(b, t, self.n_tokens, -1)
        tokens = tokens + self.frame_embed.weight[None, :, None, :]
        tokens = tokens.reshape(b, t * self.n_tokens, -1)

        x = torch.cat([self.cls.expand(b, -1, -1), tokens], dim=1)
        return self.head(self.transformer(x)[:, 0]).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
