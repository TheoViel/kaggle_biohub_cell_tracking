"""Fast temporal U-Net v4: cacheable spatial trunk + coarse learned flow.

V2 and v3 run their temporal matcher on a relatively large decoder feature
map.  V4 keeps the complete spatial trunk frame-independent, estimates motion
on the encoder bottleneck, then transports low-rank features between frames on
the existing coarse and fine output grids.  The expensive spatial trunk can
therefore be cached per frame during overlapping-window inference.

The temporal residual has separate recovery and modulation paths.  Recovery is
gated by evidence warped from the neighbouring frame, not by the target frame,
so a missed target can still receive a proposal.  Both paths are zero-scaled
at initialisation, making the module exactly the per-frame U-Net before it has
learned useful temporal behaviour.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_zoo.temporal_unet_v2 import (
    P99_DISPLACEMENT_VOXELS,
    TemporalUNet3Dv2,
    _warp,
)


PreparedFrames = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]


class _GridFusion(nn.Module):
    """Warp a low-rank neighbour representation and fuse it into one grid."""

    def __init__(self, channels: int, value_channels: int) -> None:
        super().__init__()
        self.value = nn.Conv3d(channels, value_channels, kernel_size=1, bias=False)
        self.objectness = nn.Conv3d(channels, 1, kernel_size=1)

        self.trunk = nn.Sequential(
            nn.Conv3d(
                channels + value_channels + 1,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GELU(),
        )
        self.recover = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.modulate = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

        # ReZero: at initialisation the temporal module is exactly an identity.
        self.recover_scale = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.modulate_scale = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

    def contribution(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        flow: torch.Tensor,
        agreement: torch.Tensor,
        support_confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        payload = torch.cat(
            [self.value(source), torch.sigmoid(self.objectness(source))], dim=1
        )
        warped = _warp(payload, flow)
        proposal, source_support = warped[:, :-1], warped[:, -1:]

        hidden = self.trunk(torch.cat([target, proposal, source_support], dim=1))
        support = source_support * support_confidence
        recover = self.recover_scale * support * self.recover(hidden)
        modulate = (
            self.modulate_scale
            * (2.0 * agreement - 1.0)
            * self.modulate(hidden)
        )
        return recover, modulate


class _FastCoarseFlowFusion(nn.Module):
    """Estimate motion at the bottleneck and apply it on output feature grids."""

    def __init__(
        self,
        bottleneck_channels: int,
        output_channels: int,
        bottleneck_stride: Sequence[int],
        flow_channels: int = 8,
        hidden_channels: int = 32,
        value_channels: int = 8,
        max_neighbours: int | None = 2,
    ) -> None:
        super().__init__()
        flow_channels = max(4, int(flow_channels))
        hidden_channels = max(8, int(hidden_channels))
        value_channels = max(4, int(value_channels))

        self.project = nn.Conv3d(
            bottleneck_channels, flow_channels, kernel_size=1, bias=False
        )
        self.motion = nn.Sequential(
            nn.Conv3d(4 * flow_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GELU(),
            # A second, dilated depthwise layer gives a seven-voxel receptive
            # field on the cheap bottleneck grid.
            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GELU(),
            # dz, dy, dx, agreement logit, source-support logit.
            nn.Conv3d(hidden_channels, 5, kernel_size=1),
        )
        # Start with zero displacement and neutral gates.
        nn.init.zeros_(self.motion[-1].weight)
        nn.init.zeros_(self.motion[-1].bias)

        stride = tuple(max(1, int(v)) for v in bottleneck_stride)
        max_flow = [
            max(1.0, float(-(-p99 // s)))
            for p99, s in zip(P99_DISPLACEMENT_VOXELS, stride)
        ]
        self.register_buffer(
            "max_flow",
            torch.tensor(max_flow, dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.max_neighbours = max_neighbours
        self.grid_fusions = nn.ModuleList(
            [
                _GridFusion(output_channels, value_channels),
                _GridFusion(output_channels, value_channels),
            ]
        )

    def _neighbours(self, t: int, n_frames: int) -> list[int]:
        neighbours = sorted(
            (s for s in range(n_frames) if s != t),
            key=lambda s: (abs(s - t), s),
        )
        if self.max_neighbours is not None:
            neighbours = neighbours[: self.max_neighbours]
        return neighbours

    def _predict_flow(
        self, target: torch.Tensor, source: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = torch.cat(
            [target, source, target - source, target * source], dim=1
        )
        pred = self.motion(pair)
        flow = torch.tanh(pred[:, :3]) * self.max_flow.to(pred.dtype)
        agreement = torch.sigmoid(pred[:, 3:4])
        support = torch.sigmoid(pred[:, 4:5])
        return flow, agreement, support

    @staticmethod
    def _to_grid(
        tensor: torch.Tensor,
        shape: Sequence[int],
        motion_shape: Sequence[int],
        is_flow: bool,
    ) -> torch.Tensor:
        out = F.interpolate(
            tensor, size=tuple(shape), mode="trilinear", align_corners=True
        )
        if is_flow:
            scale = tensor.new_tensor(
                [
                    (o - 1) / (m - 1) if m > 1 else 1.0
                    for o, m in zip(shape, motion_shape)
                ]
            ).view(1, 3, 1, 1, 1)
            out = out * scale
        return out

    def forward(
        self,
        bottleneck: torch.Tensor,
        grids: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        # bottleneck/grids: (B, T, C, Z, Y, X)
        batch, n_frames = bottleneck.shape[:2]
        if n_frames < 2:
            return list(grids)

        motion_shape = bottleneck.shape[-3:]
        projected = self.project(
            bottleneck.reshape(batch * n_frames, *bottleneck.shape[2:])
        ).reshape(batch, n_frames, -1, *motion_shape)

        outputs: list[list[torch.Tensor]] = [[] for _ in grids]
        for t in range(n_frames):
            recoveries: list[list[torch.Tensor]] = [[] for _ in grids]
            modulations: list[list[torch.Tensor]] = [[] for _ in grids]
            neighbours = self._neighbours(t, n_frames)
            for s in neighbours:
                flow, agreement, support = self._predict_flow(
                    projected[:, t], projected[:, s]
                )
                for i, grid in enumerate(grids):
                    shape = grid.shape[-3:]
                    flow_grid = self._to_grid(flow, shape, motion_shape, True)
                    agreement_grid = self._to_grid(
                        agreement, shape, motion_shape, False
                    )
                    support_grid = self._to_grid(
                        support, shape, motion_shape, False
                    )
                    recover, modulate = self.grid_fusions[i].contribution(
                        grid[:, t],
                        grid[:, s],
                        flow_grid,
                        agreement_grid,
                        support_grid,
                    )
                    recoveries[i].append(recover)
                    modulations[i].append(modulate)

            for i, grid in enumerate(grids):
                recovery = torch.stack(recoveries[i]).mean(dim=0)
                modulation = torch.stack(modulations[i]).mean(dim=0)
                outputs[i].append(grid[:, t] + recovery + modulation)

        return [torch.stack(per_frame, dim=1) for per_frame in outputs]


class TemporalUNet3Dv4(TemporalUNet3Dv2):
    """Frame-independent U-Net trunk with fast coarse-flow temporal fusion."""

    def __init__(
        self,
        *args,
        temporal_fusion_at: Sequence[str] = ("head",),
        temporal_flow_channels: int = 8,
        temporal_flow_hidden_channels: int = 32,
        temporal_value_channels: int = 8,
        temporal_max_neighbours: int | None = 2,
        **kwargs,
    ) -> None:
        sites = tuple(temporal_fusion_at)
        if sites not in (("head",), ("fine",), ("head", "fine"), ("fine", "head")):
            raise ValueError(
                "TemporalUNet3Dv4 fuses the completed coarse and fine trunks; "
                "temporal_fusion_at must contain 'head', 'fine', or both"
            )

        # Build v2's cacheable spatial geometry without any of its expensive
        # correlation blocks. V4 owns the sole temporal operation below.
        super().__init__(
            *args,
            temporal_fusion_at=(),
            temporal_max_neighbours=temporal_max_neighbours,
            **kwargs,
        )
        widths = self._stage_widths()
        bottleneck_stride = self._stride_at(len(widths) - 1)
        self.fast_temporal_fusion = _FastCoarseFlowFusion(
            bottleneck_channels=widths[-1],
            output_channels=self.out_channels,
            bottleneck_stride=bottleneck_stride,
            flow_channels=temporal_flow_channels,
            hidden_channels=temporal_flow_hidden_channels,
            value_channels=temporal_value_channels,
            max_neighbours=temporal_max_neighbours,
        )

    def prepare_frames(self, x: torch.Tensor) -> PreparedFrames:
        """Run the complete frame-independent spatial trunk.

        ``x`` is ``(N, C, Z, Y, X)``. Returned tensors retain that flat frame
        axis and can safely be cached and assembled into arbitrary windows.
        """
        x = self.stem(x)
        skips: list[torch.Tensor] = []
        for i, (pool, block, spatial) in enumerate(
            zip(self.pools, self.encoder_blocks, self.spatial_blocks)
        ):
            x = pool(x)
            x = self._run(block, x)
            if not isinstance(spatial, nn.Identity):
                x = self._run(spatial, x)
            if i < len(self.encoder_blocks) - 1:
                skips.append(x)
        bottleneck = x

        for up, block, skip in zip(
            self.upsamples, self.decoder_blocks, skips[::-1]
        ):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="trilinear", align_corners=False
                )
            x = self._run(block, torch.cat([x, skip], dim=1))

        fine = None
        if self.fine_head is not None:
            fine = x
            for j, (up, block) in enumerate(
                zip(self.fine_upsamples, self.fine_decoder_blocks)
            ):
                skip = skips[self.head_stage - 1 - j]
                fine = up(fine)
                if fine.shape[2:] != skip.shape[2:]:
                    fine = F.interpolate(
                        fine,
                        size=skip.shape[2:],
                        mode="trilinear",
                        align_corners=False,
                    )
                fine = self._run(block, torch.cat([fine, skip], dim=1))
            fine = self.fine_head(fine)

        if self.hires_projs:
            detail = []
            for proj, pool, skip in zip(
                self.hires_projs, self.hires_pools, skips
            ):
                d = proj(pool(skip))
                if d.shape[2:] != x.shape[2:]:
                    d = F.interpolate(
                        d, size=x.shape[2:], mode="trilinear", align_corners=False
                    )
                detail.append(d)
            x = torch.cat([x, *detail], dim=1)

        coarse = self.head(x)
        return bottleneck, coarse, fine

    def fuse_window(
        self, prepared: PreparedFrames
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply temporal fusion to prepared ``(B,T,...)`` frame tensors."""
        bottleneck, coarse, fine = prepared
        grids = [coarse] + ([] if fine is None else [fine])
        fused = self.fast_temporal_fusion(bottleneck, grids)
        if fine is None:
            return fused[0]
        return fused[0], fused[1]

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, n_frames = x.shape[:2]
        prepared = self.prepare_frames(
            x.reshape(batch * n_frames, *x.shape[2:])
        )
        window = tuple(
            None
            if tensor is None
            else tensor.reshape(batch, n_frames, *tensor.shape[1:])
            for tensor in prepared
        )
        return self.fuse_window(window)  # type: ignore[arg-type]
