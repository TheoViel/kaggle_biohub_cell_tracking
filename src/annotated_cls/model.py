"""`AnnotatedNet`: a crop -> one logit, "the annotation covers this detection".

A 2D CNN, deliberately. The question is about a single cell at a single moment,
so there is no temporal arrangement to preserve and nothing for a transformer
over frame tokens to do -- `division_cls` needs one because "divides *here*" is
a statement about change between frames, and `cell_merging_cls` needs one
because its evidence is the spatial arrangement of two detections. Here the
backbone's own pooled vector is the whole representation.

z is carried as channels rather than as a third spatial axis, which is the same
trick the rest of the repo uses: the voxel is 4x coarser in z (1.625 um against
0.40625), so a 9-slice crop is ~15 um of depth against ~13 um across, and a 2D
backbone reading those slices as channels keeps ImageNet weights usable.

`n_frames > 1` averages the encoder over a short clip in time. It is off by
default and is a knob rather than a design: if it helps, the label is being
predicted from motion rather than appearance, which is worth knowing.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AnnotatedNet(nn.Module):
    """`(B, T, pz, H, W)` -> `(B,)` logits."""

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        patch_yx: int = 32,
        patch_z: int = 9,
        n_frames: int = 1,
        dropout: float = 0.1,
        grad_checkpointing: bool = False,
        verbose: int = 0,
    ):
        super().__init__()
        import timm

        self.n_frames = int(n_frames)
        try:
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained, in_chans=int(patch_z),
                num_classes=0, img_size=int(patch_yx))
        except TypeError:           # fully convolutional: no img_size kwarg
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained, in_chans=int(patch_z),
                num_classes=0)
        with torch.no_grad():
            feat_dim = int(self.backbone(
                torch.zeros(1, int(patch_z), int(patch_yx), int(patch_yx))).shape[-1])
        self.feat_dim = feat_dim
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Dropout(dropout), nn.Linear(feat_dim, 1))

        if grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
        if verbose:
            print(f"  - {backbone}: {n_frames}x{patch_z}x{patch_yx}x{patch_yx} "
                  f"-> {feat_dim}d -> 1 logit")

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        b, t = clip.shape[:2]
        if t != self.n_frames:
            raise ValueError(f"model built for {self.n_frames} frames, got {t}")
        feats = self.backbone(clip.flatten(0, 1))        # (B*T, D)
        feats = feats.reshape(b, t, -1).mean(dim=1)      # frames are views, not a sequence
        return self.head(feats).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
