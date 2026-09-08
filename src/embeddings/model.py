"""NETr, the Neuron Embedding Transformer of ASCENT.

Han, Pritchard & Lu, *ASCENT: Annotation-free Self-supervised Contrastive
Embeddings for 3D Neuron Tracking in Fluorescence Microscopy*, bioRxiv 2025
(10.1101/2025.07.23.666425).

One embedding per detected cell, built from two things the paper keeps separate
until the last moment:

* what the cell **looks like** -- a (3, 64, 64) patch around its centroid, the
  three z-slices fed to a 2-D backbone as channels, which is what "ChannelViT"
  means here;
* where it **is** -- normalised z/y/x through a 3-layer MLP.

The two are combined by element-wise addition rather than concatenation, so the
transformer sees one 256-d token per cell, and then attends over every cell in
the frame. That attention is the point: an embedding is not a property of the
cell alone but of the cell *in its neighbourhood*, which is what makes two
cells that look alike separable when their surroundings differ.

Nothing here is supervised. Identity comes from the augmentation pipeline --
two views of the same frame -- so the embeddings never see a ground-truth link.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEncoder(nn.Module):
    """The appearance branch: a patch of image -> a d_model vector.

    The paper uses a DINO-pretrained ChannelViT. ``timm``'s DINO ViT-S is the
    closest thing that is one line to build, taken at ``img_size`` 64 with the
    patch's z-slices as input channels. ``pretrained`` needs network access the
    first time; it falls back to a small convolutional encoder rather than
    failing, since the alternative is no model at all on an offline box.

    ``min_tokens`` guards against a failure mode that only shows up because the
    patches are small. A hierarchical backbone -- ConvNeXt, CAFormer, FastViT,
    anything with a 4x stem and three 2x downsamples -- reduces its input 32x,
    so a 48-voxel patch leaves the last stage a 1x1 or 2x2 feature map. That
    stage's 7x7 depthwise convs and attention windows are then almost entirely
    padding, and what comes out is a global descriptor with no spatial structure
    for the frame transformer to work with. It is also where most of the
    parameters live. So when the pyramid ends below ``min_tokens``, the backbone
    is rebuilt truncated to the deepest stage that still clears it. timm's
    ``out_indices`` prunes the dropped stages outright rather than leaving them
    unused, which matters under DDP: a registered parameter that never receives
    a gradient aborts the backward pass.

    ViTs are unaffected -- every block holds the same resolution, so the last
    stage is no smaller than the first and nothing is truncated.
    """

    def __init__(
        self,
        in_chans: int = 3,
        d_model: int = 256,
        name: str = "vit_small_patch16_224.dino",
        img_size: int = 64,
        pretrained: bool = True,
        min_tokens: int = 3,
        grad_checkpointing: bool = False,
        verbose: int = 0,
    ):
        super().__init__()
        self.backbone, self.feat_dim, self.kind, self.truncated = self._build(
            name, in_chans, img_size, pretrained, int(min_tokens or 0), verbose
        )
        # Recompute the backbone's activations in the backward instead of
        # keeping them. This branch is where essentially all the memory goes --
        # a step encodes batch_size * max_nodes * n_frames * 2 views patches,
        # ~25k of them at the configured sizes -- so the saving is the
        # difference between fitting and not: 71 GB to 13 GB at full frames,
        # for about 27% more time. Only worth turning on when memory is the
        # binding constraint, which with max_nodes=256 it is.
        if grad_checkpointing:
            enable = getattr(self.backbone, "set_grad_checkpointing", None)
            if enable is not None:
                enable(True)
            elif verbose:
                print(f"  ! {name} has no set_grad_checkpointing; ignored")
        # A truncated stage is raw output, with none of the head's final norm,
        # so put one back before the projection. The untruncated path already
        # got its norm inside the backbone and takes an Identity here.
        self.norm = nn.LayerNorm(self.feat_dim) if self.truncated else nn.Identity()
        self.proj = nn.Linear(self.feat_dim, d_model)

    @staticmethod
    def _build(name, in_chans, img_size, pretrained, min_tokens, verbose):
        import timm

        def create(**kw):
            try:
                return timm.create_model(
                    name, pretrained=pretrained, img_size=img_size,
                    in_chans=in_chans, **kw,
                )
            except TypeError:   # fully-convolutional: img_size is not a kwarg
                return timm.create_model(
                    name, pretrained=pretrained, in_chans=in_chans, **kw
                )

        # Where does this backbone's pyramid actually end at this patch size?
        idx = sizes = chans = None
        if min_tokens > 1:
            try:
                probe = create(features_only=True)
                with torch.no_grad():
                    sizes = [
                        int(min(m.shape[-2:]))
                        for m in probe(torch.zeros(1, in_chans, img_size, img_size))
                    ]
                chans = probe.feature_info.channels()
                del probe
                if sizes and sizes[-1] < min_tokens:
                    clears = [i for i, s in enumerate(sizes) if s >= min_tokens]
                    idx = max(clears) if clears else None
            except Exception:
                # features_only is unsupported (most ViTs) or the architecture
                # rejects this input size. Neither is fatal -- fall through to
                # the plain pooled model, which is the previous behaviour.
                idx = None

        if idx is not None:
            model = create(features_only=True, out_indices=(idx,))
            if verbose:
                print(f"  - {name}: collapses to {sizes[-1]}x{sizes[-1]} at "
                      f"{img_size}px, truncated to stage {idx} "
                      f"({sizes[idx]}x{sizes[idx]}, {chans[idx]}ch)")
            return model, int(chans[idx]), name, True

        try:
            model = create(num_classes=0)
            return model, model.num_features, name, False
        except Exception as exc:  # offline, or an unknown model name
            print(f"  ! {name} unavailable ({type(exc).__name__})")
            raise exc

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """(M, C, H, W) patches -> (M, d_model)."""
        x = self.backbone(patches)
        if self.truncated:
            x = x[-1]           # features_only hands back a list of one stage
            # Pool the stage's map to a vector. Which axis holds the channels
            # is read off feat_dim rather than assumed, since timm emits NHWC
            # for some architectures and NCHW for most.
            dims = (2, 3) if x.shape[1] == self.feat_dim else (1, 2)
            x = x.mean(dim=dims)
        return self.proj(self.norm(x))


class PositionEncoder(nn.Module):
    """The spatial branch: normalised z/y/x -> a d_model vector."""

    def __init__(self, d_model: int = 256, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(coords)


class FrameAggregator(nn.Module):
    """Collapse a variable-length sequence of frame embeddings into one token.

    mode="mean": plain average (the paper's default).
    mode="bigru": bidirectional GRU; forward and backward final hidden states
        are concatenated, matching the d_model width (each direction gets
        d_model // 2 hidden units).
    """

    def __init__(self, mode: str = "mean", d_model: int = 256):
        super().__init__()
        self.mode = mode
        if mode == "bigru":
            if d_model % 2 != 0:
                raise ValueError(f"d_model must be even for BiGRU, got {d_model}")
            self.gru = nn.GRU(
                input_size=d_model, hidden_size=d_model // 2,
                num_layers=1, batch_first=True, bidirectional=True,
            )
        elif mode != "mean":
            raise ValueError(f"Unknown frame_aggregation mode: {mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(M, F, d_model) -> (M, d_model)."""
        if self.mode == "mean":
            return x.mean(dim=1)
        # bigru: h has shape (2, M, d_model//2); concat directions -> (M, d_model)
        _, h = self.gru(x)
        return torch.cat([h[0], h[1]], dim=-1)


class NETr(nn.Module):
    """Neuron Embedding Transformer: one L2-normalised embedding per cell.

    Args:
        d_model: token width; the paper's 256.
        n_layers / n_heads / dim_feedforward / dropout: the paper's 4 / 4 /
            512 / 0.1.
        use_visual / use_position: the paper's two ablations. Dropping the
            positional MLP leaves "visual features only"; dropping the
            ChannelViT leaves "position features only". Both are still run
            through the transformer, so what is being ablated is the input to
            the context model, not the context itself.
        use_transformer: ablates the context itself, which the two above cannot.
            With it off a cell's embedding is a function of that cell's own
            patch and coordinate alone -- no attention over the frame, so two
            cells that look alike stay alike however different their
            neighbourhoods are. That is the paper's central claim stated as an
            experiment, and it is also the cheaper model, so it is worth knowing
            what the attention is actually buying.
        frame_aggregation: how to merge per-frame embeddings before the
            transformer. ``"mean"`` (default, paper's behaviour) or
            ``"bigru"`` (bidirectional GRU).
        patch_z: z-slices per patch, the backbone's input channels.
        embed_dim: output width. Defaults to ``d_model``; a separate head is
            only worth it if the matching stage wants something narrower.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        patch_z: int = 3,
        patch_size: int = 64,
        embed_dim: int | None = None,
        backbone: str = "vit_small_patch16_224.dino",
        pretrained: bool = True,
        use_visual: bool = True,
        use_position: bool = True,
        use_transformer: bool = True,
        frame_aggregation: str = "mean",
        channels_last: bool = True,
        min_tokens: int = 3,
        grad_checkpointing: bool = False,
        verbose: int = 0,
    ):
        super().__init__()
        if not (use_visual or use_position):
            raise ValueError("use_visual and use_position cannot both be False")
        self.d_model = d_model
        self.use_visual = bool(use_visual)
        self.use_position = bool(use_position)
        self.use_transformer = bool(use_transformer)
        # A layout choice, not geometry: it changes neither the weights nor the
        # arithmetic, so it stays out of model_config and old checkpoints load
        # unchanged. ConvNeXt is the reason it defaults on -- its depthwise 7x7
        # convs are badly served by contiguous layout, and under DDP the
        # mismatch between the grad's strides and the bucket's also earns a
        # "Grad strides do not match bucket view strides" warning every step.
        self.channels_last = bool(channels_last)

        # The omitted branch is not built at all, rather than built and ignored.
        # An unused module still registers parameters, and under DDP a parameter
        # that receives no gradient aborts the backward pass -- so an ablation
        # left in place would fail on multi-GPU while working on one.
        self.patch_encoder = (
            PatchEncoder(
                in_chans=patch_z,
                d_model=d_model,
                name=backbone,
                img_size=patch_size,
                pretrained=pretrained,
                min_tokens=min_tokens,
                grad_checkpointing=grad_checkpointing,
                verbose=verbose,
            )
            if self.use_visual else None
        )
        self.pos_encoder = PositionEncoder(d_model=d_model) if self.use_position else None
        self.frame_agg = FrameAggregator(mode=frame_aggregation, d_model=d_model)

        # Not built when ablated, for the same DDP reason as the branches above.
        self.transformer = None
        if self.use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            # Stated rather than left to default. The nested-tensor fast path is
            # incompatible with norm_first, so PyTorch turns it off and warns on
            # every construction; saying so here is the same behaviour without
            # the noise, and without a warnings filter that would also hide the
            # next warning this class earns.
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=n_layers, enable_nested_tensor=False
            )
        self.head = (
            nn.Identity() if embed_dim in (None, d_model)
            else nn.Linear(d_model, embed_dim)
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        patches: torch.Tensor,   # (B, N, F, C, H, W)
        coords: torch.Tensor,    # (B, N, 3) normalised
        mask: torch.Tensor,      # (B, N) bool, True where a cell is present
    ) -> torch.Tensor:
        """-> (B, N, embed_dim), L2-normalised."""
        B, N = patches.shape[:2]
        tokens = None
        if self.use_visual:
            # The F z-slabs of one cell go through the backbone as separate
            # images and are merged into a single token (see FrameAggregator).
            # Merging here rather than concatenating keeps the token width
            # independent of how much z a patch spans, so F can change without
            # touching the transformer or the checkpoint's shapes.
            #
            # Only the occupied slots are encoded. `mask` is False both for
            # padding up to max_nodes and for cells object-dropout removed, and
            # those tokens are excluded from attention below, so their gradient
            # is exactly zero -- encoding them is pure waste. Frames here hold a
            # median of ~199 cells against max_nodes=256, so this is about a
            # third of the backbone's work and a third of its activations.
            F = patches.shape[2]
            sel = mask.reshape(B * N)
            sub = patches.reshape(B * N, F, *patches.shape[3:])[sel]
            sub = sub.reshape(-1, *sub.shape[2:])
            if self.channels_last:
                sub = sub.contiguous(memory_format=torch.channels_last)
            enc = self.patch_encoder(sub)
            enc = self.frame_agg(enc.reshape(-1, F, self.d_model))
            # new_zeros off `enc` so dtype follows autocast rather than input.
            tokens = enc.new_zeros(B * N, self.d_model)
            tokens[sel] = enc
            tokens = tokens.reshape(B, N, self.d_model)
        if self.use_position:
            pos = self.pos_encoder(coords)
            # Element-wise addition, as the paper fuses them; with one branch
            # ablated away the sum is simply that branch.
            tokens = pos if tokens is None else tokens + pos

        if self.use_transformer:
            # Padded slots must not be attended to. A row that is entirely
            # padding would make softmax over an all-masked row NaN, so those
            # are left unmasked and discarded by the caller's own mask instead.
            pad = ~mask
            pad = torch.where(pad.all(dim=1, keepdim=True), torch.zeros_like(pad), pad)
            tokens = self.transformer(tokens, src_key_padding_mask=pad)
        # Ablated: tokens go to the head as the per-cell branches produced them,
        # so the embedding never sees another cell. No mask is needed -- with no
        # attention there is nothing for a padded slot to leak into.

        out = self.head(tokens)
        return nn.functional.normalize(out, dim=-1)
