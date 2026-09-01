# Adapted from https://github.com/MIC-DKFZ/dynamic-network-architectures (Apache-2.0),
# the architecture library nnU-Net instantiates for its ResEnc presets. See
# https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md
"""
nnU-Net Residual Encoder (ResEnc) U-Nets.

Module and parameter names match the reference implementation, so state dicts are
interchangeable with nnU-Net checkpoints.

The M / L / XL presets share the same channel widths and block counts (32 base features,
capped at 320 in 3D, (1, 3, 4, 6, 6, 6, ...) residual blocks per encoder stage). They only
differ in the GPU memory they budget for, which nnU-Net spends on a larger patch size --
and hence on more resolution stages. `plan_topology` reproduces the planner's
`get_pool_and_conv_props`, so the stage count, strides and kernel sizes follow from the
training patch size exactly as nnU-Net would derive them.
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


CONV_DIMS = {nn.Conv1d: 1, nn.Conv2d: 2, nn.Conv3d: 3}
NORM_OPS = {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}
TRANSP_OPS = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
AVG_POOL_OPS = {1: nn.AvgPool1d, 2: nn.AvgPool2d, 3: nn.AvgPool3d}

# nnU-Net's ResEncUNetPlanner. The presets inherit these and only override the VRAM budget.
BASE_NUM_FEATURES = 32
MAX_FEATURES_2D = 512
MAX_FEATURES_3D = 320
BLOCKS_PER_STAGE_ENCODER = (1, 3, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)
BLOCKS_PER_STAGE_DECODER = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
FEATUREMAP_MIN_EDGE_LENGTH = 4

# Reference feature map sizes each preset budgets for, at its VRAM target and at nnU-Net's
# reference batch size (2 in 3D, 12 in 2D). Used to suggest a batch size for a patch size.
RESENC_PRESETS = {
    "resenc_m": {"vram_target_gb": 8, "reference_val_3d": 680000000, "reference_val_2d": 135000000},
    "resenc_l": {"vram_target_gb": 24, "reference_val_3d": 2100000000, "reference_val_2d": 380000000},
    "resenc_xl": {"vram_target_gb": 40, "reference_val_3d": 3600000000, "reference_val_2d": 560000000},
}
REFERENCE_BATCH_SIZE_2D = 12
REFERENCE_BATCH_SIZE_3D = 2


def to_list(conv_op, scalar):
    """
    Broadcast a scalar kernel size / stride to one entry per spatial axis.

    Args:
        conv_op (type): Convolution class, which fixes the number of spatial axes.
        scalar (int or list or tuple): Value to broadcast, or an already expanded sequence.

    Returns:
        list: One entry per spatial axis.
    """
    if not isinstance(scalar, (tuple, list, np.ndarray)):
        return [scalar] * CONV_DIMS[conv_op]
    return list(scalar)


def cumulative_stride(*stride_groups):
    """
    Per-axis product of a set of strides.

    Args:
        *stride_groups: Each argument is either one stride (a sequence, one entry per axis) or
            a sequence of them.

    Returns:
        tuple: The combined downsampling factor per axis.
    """
    strides = []
    for group in stride_groups:
        if len(group) and isinstance(group[0], (tuple, list, np.ndarray)):
            strides.extend(group)
        else:
            strides.append(group)
    return tuple(int(np.prod(axis)) for axis in zip(*strides))


class ConvDropoutNormReLU(nn.Module):
    """
    Convolution followed by optional dropout, normalization and non-linearity.

    Attributes:
        conv (nn.Module): The convolution. Also the first entry of `all_modules`.
        all_modules (nn.Sequential): The full op sequence, which is what `forward` runs.
    """

    def __init__(
        self,
        conv_op,
        input_channels,
        output_channels,
        kernel_size,
        stride,
        conv_bias=False,
        norm_op=None,
        norm_op_kwargs=None,
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=None,
        nonlin_kwargs=None,
    ):
        """
        Constructor.

        Args:
            conv_op (type): Convolution class, e.g. nn.Conv3d.
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            kernel_size (int or tuple): Convolution kernel size. Padding is always "same".
            stride (int or tuple): Convolution stride.
            conv_bias (bool, optional): Whether the convolution has a bias. Default is False.
            norm_op (type, optional): Normalization class. Default is None.
            norm_op_kwargs (dict, optional): Kwargs for the normalization. Default is None.
            dropout_op (type, optional): Dropout class. Default is None.
            dropout_op_kwargs (dict, optional): Kwargs for the dropout. Default is None.
            nonlin (type, optional): Non-linearity class. Default is None.
            nonlin_kwargs (dict, optional): Kwargs for the non-linearity. Default is None.
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = to_list(conv_op, stride)

        kernel_size = to_list(conv_op, kernel_size)
        norm_op_kwargs = {} if norm_op_kwargs is None else norm_op_kwargs
        nonlin_kwargs = {} if nonlin_kwargs is None else nonlin_kwargs

        # `conv`, `norm` and `nonlin` are registered both directly and inside `all_modules`, as
        # in the reference implementation. Both sets of keys appear in the state dict, sharing
        # storage, and nnU-Net checkpoints carry them too.
        self.conv = conv_op(
            input_channels,
            output_channels,
            kernel_size,
            self.stride,
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            bias=conv_bias,
        )
        ops = [self.conv]

        if dropout_op is not None:
            self.dropout = dropout_op(**(dropout_op_kwargs or {}))
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Convolved, normalized and activated tensor.
        """
        return self.all_modules(x)

    def compute_conv_feature_map_size(self, input_size):
        """
        Number of activation elements this block produces, for VRAM estimation.

        Args:
            input_size (tuple): Spatial size of the input, without batch or channel axes.

        Returns:
            np.int64: Number of elements in the output feature map.
        """
        output_size = [i // j for i, j in zip(input_size, self.stride)]
        return np.prod([self.output_channels, *output_size], dtype=np.int64)


class StackedConvBlocks(nn.Module):
    """
    A run of `ConvDropoutNormReLU` blocks, only the first of which is strided.

    Attributes:
        convs (nn.Sequential): The stacked blocks.
    """

    def __init__(
        self,
        num_convs,
        conv_op,
        input_channels,
        output_channels,
        kernel_size,
        initial_stride,
        conv_bias=False,
        norm_op=None,
        norm_op_kwargs=None,
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=None,
        nonlin_kwargs=None,
    ):
        """
        Constructor.

        Args:
            num_convs (int): Number of stacked convolution blocks.
            conv_op (type): Convolution class, e.g. nn.Conv3d.
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels of every block.
            kernel_size (int or tuple): Convolution kernel size.
            initial_stride (int or tuple): Stride of the first block. The rest use stride 1.
            conv_bias (bool, optional): Whether the convolutions have a bias. Default is False.
            norm_op (type, optional): Normalization class. Default is None.
            norm_op_kwargs (dict, optional): Kwargs for the normalization. Default is None.
            dropout_op (type, optional): Dropout class. Default is None.
            dropout_op_kwargs (dict, optional): Kwargs for the dropout. Default is None.
            nonlin (type, optional): Non-linearity class. Default is None.
            nonlin_kwargs (dict, optional): Kwargs for the non-linearity. Default is None.
        """
        super().__init__()
        args = (conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(conv_op, input_channels, output_channels, kernel_size, initial_stride, *args),
            *[
                ConvDropoutNormReLU(conv_op, output_channels, output_channels, kernel_size, 1, *args)
                for _ in range(1, num_convs)
            ],
        )
        self.output_channels = output_channels
        self.initial_stride = to_list(conv_op, initial_stride)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the last block.
        """
        return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        """
        Number of activation elements this stack produces, for VRAM estimation.

        Args:
            input_size (tuple): Spatial size of the input, without batch or channel axes.

        Returns:
            np.int64: Number of elements across all block outputs.
        """
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for block in self.convs[1:]:
            output += block.compute_conv_feature_map_size(size_after_stride)
        return output


class BasicBlockD(nn.Module):
    """
    Residual block following ResNet-D: the skip is an avg-pool followed by a 1x1 projection.

    He, Tong, et al. "Bag of tricks for image classification with convolutional neural networks."
    CVPR 2019.

    Attributes:
        conv1 (ConvDropoutNormReLU): First convolution, which carries the stride.
        conv2 (ConvDropoutNormReLU): Second convolution, always stride 1 and without non-linearity.
        skip (nn.Module): Identity, or downsampling and projection of the residual.
    """

    def __init__(
        self,
        conv_op,
        input_channels,
        output_channels,
        kernel_size,
        stride,
        conv_bias=False,
        norm_op=None,
        norm_op_kwargs=None,
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=None,
        nonlin_kwargs=None,
    ):
        """
        Constructor.

        Args:
            conv_op (type): Convolution class, e.g. nn.Conv3d.
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            kernel_size (int or tuple): Kernel size of the two convolutions, not of the skip.
            stride (int or tuple): Stride of the first convolution and of the skip.
            conv_bias (bool, optional): Whether the convolutions have a bias. Default is False.
            norm_op (type, optional): Normalization class. Default is None.
            norm_op_kwargs (dict, optional): Kwargs for the normalization. Default is None.
            dropout_op (type, optional): Dropout class, applied in the first conv only. Default is None.
            dropout_op_kwargs (dict, optional): Kwargs for the dropout. Default is None.
            nonlin (type, optional): Non-linearity class. Default is None.
            nonlin_kwargs (dict, optional): Kwargs for the non-linearity. Default is None.
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = to_list(conv_op, stride)
        kernel_size = to_list(conv_op, kernel_size)
        nonlin_kwargs = {} if nonlin_kwargs is None else nonlin_kwargs

        self.conv1 = ConvDropoutNormReLU(
            conv_op, input_channels, output_channels, kernel_size, self.stride, conv_bias,
            norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
        )
        self.conv2 = ConvDropoutNormReLU(
            conv_op, output_channels, output_channels, kernel_size, 1, conv_bias,
            norm_op, norm_op_kwargs, None, None, None, None,
        )
        self.nonlin2 = nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity()

        has_stride = any(s != 1 for s in self.stride)
        requires_projection = input_channels != output_channels

        if has_stride or requires_projection:
            ops = []
            if has_stride:
                ops.append(AVG_POOL_OPS[CONV_DIMS[conv_op]](self.stride, self.stride))
            if requires_projection:
                ops.append(
                    ConvDropoutNormReLU(
                        conv_op, input_channels, output_channels, 1, 1, False,
                        norm_op, norm_op_kwargs, None, None, None, None,
                    )
                )
            self.skip = nn.Sequential(*ops)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Residual sum, activated.
        """
        residual = self.skip(x)
        out = self.conv2(self.conv1(x))
        out = out + residual
        return self.nonlin2(out)

    def compute_conv_feature_map_size(self, input_size):
        """
        Number of activation elements this block produces, for VRAM estimation.

        Args:
            input_size (tuple): Spatial size of the input, without batch or channel axes.

        Returns:
            np.int64: Number of elements across the two convolutions and the skip.
        """
        size_after_stride = [i // j for i, j in zip(input_size, self.stride)]
        output = 2 * np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        if isinstance(self.skip, nn.Sequential):
            output += np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        return output


class StackedResidualBlocks(nn.Module):
    """
    A run of residual blocks, only the first of which is strided.

    Attributes:
        blocks (nn.Sequential): The stacked residual blocks.
    """

    def __init__(
        self,
        n_blocks,
        conv_op,
        input_channels,
        output_channels,
        kernel_size,
        initial_stride,
        conv_bias=False,
        norm_op=None,
        norm_op_kwargs=None,
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=None,
        nonlin_kwargs=None,
    ):
        """
        Constructor.

        Args:
            n_blocks (int): Number of residual blocks. Must be > 0.
            conv_op (type): Convolution class, e.g. nn.Conv3d.
            input_channels (int): Number of channels entering the first block.
            output_channels (int): Number of channels of every block output.
            kernel_size (int or tuple): Kernel size of the convolutions.
            initial_stride (int or tuple): Stride of the first block. The rest use stride 1.
            conv_bias (bool, optional): Whether the convolutions have a bias. Default is False.
            norm_op (type, optional): Normalization class. Default is None.
            norm_op_kwargs (dict, optional): Kwargs for the normalization. Default is None.
            dropout_op (type, optional): Dropout class. Default is None.
            dropout_op_kwargs (dict, optional): Kwargs for the dropout. Default is None.
            nonlin (type, optional): Non-linearity class. Default is None.
            nonlin_kwargs (dict, optional): Kwargs for the non-linearity. Default is None.
        """
        super().__init__()
        assert n_blocks > 0, "n_blocks must be > 0"
        args = (conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)

        self.blocks = nn.Sequential(
            BasicBlockD(conv_op, input_channels, output_channels, kernel_size, initial_stride, *args),
            *[
                BasicBlockD(conv_op, output_channels, output_channels, kernel_size, 1, *args)
                for _ in range(1, n_blocks)
            ],
        )
        self.output_channels = output_channels
        self.initial_stride = to_list(conv_op, initial_stride)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the last block.
        """
        return self.blocks(x)

    def compute_conv_feature_map_size(self, input_size):
        """
        Number of activation elements this stack produces, for VRAM estimation.

        Args:
            input_size (tuple): Spatial size of the input, without batch or channel axes.

        Returns:
            np.int64: Number of elements across all block outputs.
        """
        output = self.blocks[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for block in self.blocks[1:]:
            output += block.compute_conv_feature_map_size(size_after_stride)
        return output


class ResidualEncoder(nn.Module):
    """
    Residual encoder: a stem followed by one strided residual stage per resolution.

    Attributes:
        stem (StackedConvBlocks): Stem, which does not downsample unless `stem_stride` says so.
        stages (nn.Sequential): One `StackedResidualBlocks` per resolution stage.
        output_channels (tuple): Number of channels each stage outputs.
        strides (list): Per-axis stride of each stage.
        stem_stride (list): Per-axis stride of the stem.
    """

    def __init__(
        self,
        input_channels,
        n_stages,
        features_per_stage,
        conv_op,
        kernel_sizes,
        strides,
        n_blocks_per_stage,
        conv_bias=False,
        norm_op=None,
        norm_op_kwargs=None,
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=None,
        nonlin_kwargs=None,
        return_skips=False,
        stem_channels=None,
        stem_stride=1,
    ):
        """
        Constructor.

        Args:
            input_channels (int): Number of input channels.
            n_stages (int): Number of resolution stages.
            features_per_stage (list): Number of channels per stage.
            conv_op (type): Convolution class, e.g. nn.Conv3d.
            kernel_sizes (list): Kernel size per stage.
            strides (list): Stride per stage. The first is usually 1, else the input is strided.
            n_blocks_per_stage (list): Number of residual blocks per stage.
            conv_bias (bool, optional): Whether the convolutions have a bias. Default is False.
            norm_op (type, optional): Normalization class. Default is None.
            norm_op_kwargs (dict, optional): Kwargs for the normalization. Default is None.
            dropout_op (type, optional): Dropout class. Default is None.
            dropout_op_kwargs (dict, optional): Kwargs for the dropout. Default is None.
            nonlin (type, optional): Non-linearity class. Default is None.
            nonlin_kwargs (dict, optional): Kwargs for the non-linearity. Default is None.
            return_skips (bool, optional): Return every stage output instead of the last.
                Default is False.
            stem_channels (int, optional): Stem width. Defaults to `features_per_stage[0]`.
            stem_stride (int or tuple, optional): Stride of the stem. nnU-Net leaves this at 1,
                so the first stage sees the input at full resolution; striding it instead keeps
                whole volumes affordable, since only the stem's single convolution ever reads
                them. Default is 1.
        """
        super().__init__()
        for name, value in [
            ("kernel_sizes", kernel_sizes),
            ("features_per_stage", features_per_stage),
            ("n_blocks_per_stage", n_blocks_per_stage),
            ("strides", strides),
        ]:
            assert len(value) == n_stages, f"{name} must have one entry per resolution stage ({n_stages})"

        stem_channels = features_per_stage[0] if stem_channels is None else stem_channels
        self.stem = StackedConvBlocks(
            1, conv_op, input_channels, stem_channels, kernel_sizes[0], stem_stride, conv_bias,
            norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
        )
        self.stem_stride = to_list(conv_op, stem_stride)
        input_channels = stem_channels

        stages = []
        for s in range(n_stages):
            stages.append(
                StackedResidualBlocks(
                    n_blocks_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s],
                    strides[s], conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                    nonlin, nonlin_kwargs,
                )
            )
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # Stashed for the decoder, which mirrors the encoder's configuration.
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            list: Per-stage outputs, if `return_skips`.
            torch.Tensor: Bottleneck features, otherwise.
        """
        x = self.stem(x)
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips if self.return_skips else skips[-1]

    def compute_conv_feature_map_size(self, input_size):
        """
        Number of activation elements the encoder produces, for VRAM estimation.

        Args:
            input_size (tuple): Spatial size of the input, without batch or channel axes.

        Returns:
            np.int64: Number of elements across the stem and every stage.
        """
        output = self.stem.compute_conv_feature_map_size(input_size)
        input_size = [i // j for i, j in zip(input_size, self.stem_stride)]
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class UNetDecoder(nn.Module):
    """
    U-Net decoder: transposed convolution, concatenation with the skip, then plain conv blocks.

    Attributes:
        stages (nn.ModuleList): Conv blocks merging each upsampled map with its skip.
        transpconvs (nn.ModuleList): Transposed convolutions, one per stage.
        seg_layers (nn.ModuleList): 1x1 segmentation heads, always built so that deep
            supervision checkpoints load at inference time.
    """

    def __init__(self, encoder, num_classes, n_conv_per_stage, deep_supervision, n_decode_steps=None):
        """
        Constructor.

        Args:
            encoder (ResidualEncoder): Encoder whose skips this decoder consumes.
            num_classes (int): Number of output channels of the segmentation heads.
            n_conv_per_stage (int or list): Convolutions per decoder stage. One entry fewer
                than the encoder has stages.
            deep_supervision (bool): Return the logits of every decoder stage, not just the last.
            n_decode_steps (int, optional): How many of those stages to build, counting from the
                bottleneck. None decodes all the way back to the encoder's input resolution;
                a smaller number stops early and leaves the output at a coarser stride, which
                is what the tracking pipeline reads its node features from. Default is None.
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes

        n_stages_encoder = len(encoder.output_channels)
        n_full_decode = n_stages_encoder - 1
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_full_decode
        if n_decode_steps is None:
            n_decode_steps = n_full_decode
        n_decode_steps = max(0, min(n_decode_steps, n_full_decode))
        self.n_decode_steps = n_decode_steps

        assert len(n_conv_per_stage) >= n_decode_steps, (
            f"n_conv_per_stage must cover the {n_decode_steps} decoder stages being built"
        )

        transpconv_op = TRANSP_OPS[CONV_DIMS[encoder.conv_op]]

        stages, transpconvs, seg_layers = [], [], []
        for s in range(1, n_decode_steps + 1):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]

            transpconvs.append(
                transpconv_op(
                    input_features_below, input_features_skip, stride_for_transpconv,
                    stride_for_transpconv, bias=encoder.conv_bias,
                )
            )
            # Input is 2x input_features_skip: the upsampled map concatenated with the skip.
            stages.append(
                StackedConvBlocks(
                    n_conv_per_stage[s - 1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                    encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op,
                    encoder.norm_op_kwargs, encoder.dropout_op, encoder.dropout_op_kwargs,
                    encoder.nonlin, encoder.nonlin_kwargs,
                )
            )
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        if n_decode_steps == 0:  # no decoder at all: the head reads the bottleneck
            seg_layers.append(encoder.conv_op(encoder.output_channels[-1], num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

        # Without deep supervision only the finest head runs, but the others stay registered so
        # that deep supervision checkpoints -- nnU-Net's included -- still load. Freezing them
        # keeps them out of the autograd graph, which DDP requires: it rejects parameters that
        # never receive a gradient.
        if not deep_supervision:
            for layer in self.seg_layers[:-1]:
                layer.requires_grad_(False)

    def forward(self, skips):
        """
        Forward pass.

        Args:
            skips (list): Encoder outputs in the order they were computed, bottleneck last.

        Returns:
            list: Logits from every stage, finest first, if `deep_supervision`.
            torch.Tensor: Logits at the finest decoded resolution, otherwise.
        """
        lres_input = skips[-1]

        if self.n_decode_steps == 0:
            bottleneck = self.seg_layers[-1](lres_input)
            return [bottleneck] if self.deep_supervision else bottleneck

        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]  # finest resolution first
        return seg_outputs if self.deep_supervision else seg_outputs[0]

    def compute_conv_feature_map_size(self, input_size):
        """
        Number of activation elements the decoder produces, for VRAM estimation.

        Args:
            input_size (tuple): Spatial size of the *encoder* input, without batch or channel axes.

        Returns:
            np.int64: Number of elements across the conv blocks, transposed convs and heads.
        """
        input_size = [i // j for i, j in zip(input_size, self.encoder.stem_stride)]
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        output = np.int64(0)
        if self.n_decode_steps == 0:  # the head sits on the bottleneck
            bottleneck = [i // j for i, j in zip(skip_sizes[-1], self.encoder.strides[-1])]
            return np.prod([self.num_classes, *bottleneck], dtype=np.int64)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output


class ResidualEncoderUNet(nn.Module):
    """
    The U-Net nnU-Net builds for its ResEnc presets: residual encoder, plain conv decoder.

    Attributes:
        encoder (ResidualEncoder): The residual encoder.
        decoder (UNetDecoder): The decoder, which also holds a reference to the encoder.
    """

    def __init__(
        self,
        input_channels,
        n_stages,
        features_per_stage,
        conv_op,
        kernel_sizes,
        strides,
        n_blocks_per_stage,
        num_classes,
        n_conv_per_stage_decoder,
        conv_bias=False,
        norm_op=None,
        norm_op_kwargs=None,
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=None,
        nonlin_kwargs=None,
        deep_supervision=False,
        stem_channels=None,
        stem_stride=1,
        n_decode_steps=None,
    ):
        """
        Constructor.

        Args:
            input_channels (int): Number of input channels.
            n_stages (int): Number of resolution stages.
            features_per_stage (int or list): Number of channels per stage.
            conv_op (type): Convolution class, e.g. nn.Conv3d.
            kernel_sizes (int or list): Kernel size per stage.
            strides (int or list): Stride per stage. The first is usually 1.
            n_blocks_per_stage (int or list): Number of residual blocks per encoder stage.
            num_classes (int): Number of output channels of the segmentation heads.
            n_conv_per_stage_decoder (int or list): Convolutions per decoder stage.
            conv_bias (bool, optional): Whether the convolutions have a bias. Default is False.
            norm_op (type, optional): Normalization class. Default is None.
            norm_op_kwargs (dict, optional): Kwargs for the normalization. Default is None.
            dropout_op (type, optional): Dropout class. Default is None.
            dropout_op_kwargs (dict, optional): Kwargs for the dropout. Default is None.
            nonlin (type, optional): Non-linearity class. Default is None.
            nonlin_kwargs (dict, optional): Kwargs for the non-linearity. Default is None.
            deep_supervision (bool, optional): Return the logits of every decoder stage.
                Default is False.
            stem_channels (int, optional): Stem width. Defaults to `features_per_stage[0]`.
            stem_stride (int or tuple, optional): Stride of the stem. Default is 1.
            n_decode_steps (int, optional): Number of decoder stages to build, counting from the
                bottleneck. None decodes all the way back. Default is None.
        """
        super().__init__()

        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        self.encoder = ResidualEncoder(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_blocks_per_stage,
            conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
            return_skips=True, stem_channels=stem_channels, stem_stride=stem_stride,
        )
        self.decoder = UNetDecoder(
            self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, n_decode_steps=n_decode_steps
        )

        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.n_decode_steps = self.decoder.n_decode_steps
        # Spatial sizes must be divisible by this for the stride chain to line up. The whole
        # encoder runs whatever the decoder does, so every stride counts, decoded back or not.
        self.divisible_by = cumulative_stride(self.encoder.stem_stride, self.encoder.strides)
        # How much coarser the output is than the input: the decoder stops at encoder stage
        # `n_stages - 1 - n_decode_steps`, so only the strides up to it have been undone.
        self.output_stride = cumulative_stride(
            self.encoder.stem_stride, self.encoder.strides[: n_stages - self.n_decode_steps]
        )

        self.apply(init_weights_he)
        self.apply(init_last_norm_before_add_to_zero)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, *spatial).

        Returns:
            list: Logits from every decoder stage, finest first, if `deep_supervision`.
            torch.Tensor: Full-resolution logits, otherwise.
        """
        return self.decoder(self.encoder(x))

    def compute_conv_feature_map_size(self, input_size):
        """
        Number of activation elements a forward pass produces, for VRAM estimation.

        Args:
            input_size (tuple): Spatial size of the input, without batch or channel axes.

        Returns:
            np.int64: Number of elements across the encoder and the decoder.
        """
        return (
            self.encoder.compute_conv_feature_map_size(input_size)
            + self.decoder.compute_conv_feature_map_size(input_size)
        )


def init_weights_he(module, neg_slope=1e-2):
    """
    He-initialize convolutions, matching nnU-Net's `InitWeights_He`.

    Args:
        module (nn.Module): Module to initialize. Anything but a convolution is left alone.
        neg_slope (float, optional): Negative slope of the leaky ReLU. Default is 1e-2.
    """
    if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight, a=neg_slope)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_last_norm_before_add_to_zero(module):
    """
    Zero the norm before each residual addition, so blocks start as the identity.

    Args:
        module (nn.Module): Module to initialize. Anything but a `BasicBlockD` is left alone.
    """
    if isinstance(module, BasicBlockD) and getattr(module.conv2, "norm", None) is not None:
        if module.conv2.norm.weight is not None:
            nn.init.constant_(module.conv2.norm.weight, 0)
            nn.init.constant_(module.conv2.norm.bias, 0)


def plan_topology(patch_size, spacing=None, min_feature_map_size=FEATUREMAP_MIN_EDGE_LENGTH, max_numpool=999999):
    """
    Derive the stage count, strides and kernel sizes from a patch size.

    Port of nnU-Net's `get_pool_and_conv_props`. Axes are pooled while they stay at least
    twice `min_feature_map_size`, anisotropic axes only once their spacing is within a
    factor of two of the finest one. Kernels along an axis switch from 1 to 3 at that
    same point, so heavily anisotropic axes stay 2D-ish near full resolution.

    Args:
        patch_size (tuple): Training patch size, one entry per spatial axis.
        spacing (tuple, optional): Voxel spacing. Defaults to isotropic.
        min_feature_map_size (int, optional): Minimum bottleneck edge length. Default is 4.
        max_numpool (int, optional): Maximum number of poolings per axis. Default is 999999.

    Returns:
        tuple: Per-stage strides, one entry per axis each.
        tuple: Per-stage kernel sizes, one entry per axis each.
        list: Number of poolings per axis.
        tuple: `patch_size` padded up to a valid multiple.
    """
    dim = len(patch_size)
    spacing = [1.0] * dim if spacing is None else list(spacing)
    assert len(spacing) == dim, "spacing must have one entry per spatial axis"

    current_spacing = deepcopy(spacing)
    current_size = list(patch_size)

    pool_op_kernel_sizes = [[1] * dim]
    conv_kernel_sizes = []
    num_pool_per_axis = [0] * dim
    kernel_size = [1] * dim

    while True:
        valid_axes = [i for i in range(dim) if current_size[i] >= 2 * min_feature_map_size]
        if len(valid_axes) < 1:
            break

        # Only pool axes whose spacing is within a factor of two of the finest one.
        min_spacing_of_valid = min(current_spacing[i] for i in valid_axes)
        valid_axes = [i for i in valid_axes if current_spacing[i] / min_spacing_of_valid < 2]
        valid_axes = [i for i in valid_axes if num_pool_per_axis[i] < max_numpool]

        if len(valid_axes) < 1:
            break
        if len(valid_axes) == 1 and current_size[valid_axes[0]] < 3 * min_feature_map_size:
            break

        # Kernels go from 1 to 3 once their axis catches up with the finest spacing, and stay 3.
        for d in range(dim):
            if kernel_size[d] != 3 and current_spacing[d] / min(current_spacing) < 2:
                kernel_size[d] = 3

        pool_kernel_sizes = [2 if d in valid_axes else 1 for d in range(dim)]
        for d in valid_axes:
            num_pool_per_axis[d] += 1
            current_spacing[d] *= 2
            current_size[d] = int(np.ceil(current_size[d] / 2))

        pool_op_kernel_sizes.append(pool_kernel_sizes)
        conv_kernel_sizes.append(deepcopy(kernel_size))

    conv_kernel_sizes.append([3] * dim)  # the bottleneck always convolves in 3D

    must_be_divisible_by = [2 ** n for n in num_pool_per_axis]
    padded_patch_size = tuple(
        s if s % m == 0 else s + m - s % m for s, m in zip(patch_size, must_be_divisible_by)
    )

    def as_tuple(nested):
        return tuple(as_tuple(i) if isinstance(i, list) else i for i in nested)

    return as_tuple(pool_op_kernel_sizes), as_tuple(conv_kernel_sizes), num_pool_per_axis, padded_patch_size


def anisotropic_strides(n_stages, pool_stride=(1, 2, 2), n_anisotropic=2, stem_strides=True):
    """
    Build a stride schedule that downsamples anisotropically for the first few steps.

    Volumes whose Z spacing is coarser than Y/X should not lose Z resolution until Y and X have
    caught up, which is what `pool_stride=(1, 2, 2)` for the first `n_anisotropic` downsamplings
    gives; everything past that is isotropic. `plan_topology` derives the same schedule from
    voxel spacing, so prefer it when the spacing is known and use this when the schedule is a
    choice rather than a measurement.

    Args:
        n_stages (int): Number of encoder stages.
        pool_stride (tuple, optional): The anisotropic stride. Default is (1, 2, 2).
        n_anisotropic (int, optional): How many downsampling steps use it. Default is 2.
        stem_strides (bool, optional): Let the stem take the first downsampling step, so nothing
            but its single convolution ever runs at full resolution. When False the stem keeps
            nnU-Net's stride of 1 and the first stage downsamples instead. Default is True.

    Returns:
        tuple: Stem stride, one entry per axis.
        tuple: Per-stage strides, one entry per axis each.
    """
    isotropic = tuple([max(pool_stride)] * len(pool_stride))
    ones = tuple([1] * len(pool_stride))

    stem_stride = pool_stride if stem_strides else ones
    # The stem's step already counts towards n_anisotropic, and a non-striding stem means the
    # first stage has to take that step instead.
    taken = 1 if stem_strides else 0
    strides = [ones] if stem_strides else []
    while len(strides) < n_stages:
        strides.append(pool_stride if taken < n_anisotropic else isotropic)
        taken += 1

    return stem_stride, tuple(strides[:n_stages])


def define_resenc_unet(
    preset="resenc_m",
    in_channels=1,
    num_classes=1,
    patch_size=(64, 64, 64),
    spacing=None,
    deep_supervision=False,
    n_stages=None,
    features_per_stage=None,
    strides=None,
    kernel_sizes=None,
    n_blocks_per_stage=None,
    n_conv_per_stage_decoder=None,
    stem_stride=1,
    n_decode_steps=None,
    verbose=0,
):
    """
    Build an nnU-Net ResEnc U-Net for a given patch size.

    The topology is planned from `patch_size` the way nnU-Net's `ResEncUNetPlanner` would.
    Every planned value can be overridden individually.

    Args:
        preset (str, optional): One of `RESENC_PRESETS`. Only sets the VRAM budget used by
            `suggest_batch_size`, not the widths. Default is "resenc_m".
        in_channels (int, optional): Number of input channels. Default is 1.
        num_classes (int, optional): Number of output channels. Default is 1.
        patch_size (tuple, optional): Training patch size, 2 or 3 entries. Default is (64, 64, 64).
        spacing (tuple, optional): Voxel spacing, used to plan anisotropic topologies.
            Defaults to isotropic.
        deep_supervision (bool, optional): Return the logits of every decoder stage.
            Default is False.
        n_stages (int, optional): Override the planned number of resolution stages. Default is None.
        features_per_stage (list, optional): Override the planned channel widths. Default is None.
        strides (list, optional): Override the planned strides. Default is None.
        kernel_sizes (list, optional): Override the planned kernel sizes. Default is None.
        n_blocks_per_stage (list, optional): Override the encoder block counts. Default is None.
        n_conv_per_stage_decoder (list, optional): Override the decoder conv counts. Default is None.
        verbose (int, optional): Print the planned configuration. Default is 0.

    Returns:
        ResidualEncoderUNet: The configured network.

    Raises:
        AssertionError: If the preset is unknown or the patch size is not 2D or 3D.
    """
    assert preset in RESENC_PRESETS, f"Unknown ResEnc preset {preset}, expected one of {list(RESENC_PRESETS)}"
    assert len(patch_size) in (2, 3), "patch_size must have 2 or 3 entries"

    dim = len(patch_size)
    conv_op = nn.Conv2d if dim == 2 else nn.Conv3d
    max_features = MAX_FEATURES_2D if dim == 2 else MAX_FEATURES_3D

    planned_strides, planned_kernels, _, padded_patch_size = plan_topology(patch_size, spacing)

    n_stages = len(planned_strides) if n_stages is None else n_stages
    assert n_stages <= len(BLOCKS_PER_STAGE_ENCODER), (
        f"{n_stages} stages exceeds the {len(BLOCKS_PER_STAGE_ENCODER)} nnU-Net specifies block counts for"
    )

    strides = planned_strides[:n_stages] if strides is None else strides
    kernel_sizes = planned_kernels[:n_stages] if kernel_sizes is None else kernel_sizes
    if features_per_stage is None:
        features_per_stage = tuple(min(max_features, BASE_NUM_FEATURES * 2**i) for i in range(n_stages))
    if n_blocks_per_stage is None:
        n_blocks_per_stage = BLOCKS_PER_STAGE_ENCODER[:n_stages]
    if n_conv_per_stage_decoder is None:
        n_conv_per_stage_decoder = BLOCKS_PER_STAGE_DECODER[: n_stages - 1]

    model = ResidualEncoderUNet(
        input_channels=in_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_blocks_per_stage=n_blocks_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=NORM_OPS[dim],
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision,
    )
    model.preset = preset
    model.patch_size = tuple(padded_patch_size)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n-> nnU-Net {preset} on patch {tuple(padded_patch_size)}  ({n_params / 1e6:.1f}M parameters)")
        print(f"   stages={n_stages}  features={tuple(features_per_stage)}  blocks={tuple(n_blocks_per_stage)}")
        print(f"   strides={tuple(strides)}")
        print(f"   suggested batch size at the {preset} VRAM target: {suggest_batch_size(model, preset)}\n")

    return model


def suggest_batch_size(model, preset="resenc_m"):
    """
    Largest batch size the preset's VRAM target allows for this network and patch size.

    Mirrors the planner: it scales nnU-Net's reference batch size by how much room the
    preset's feature map budget leaves, and never goes below 2.

    Args:
        model (ResidualEncoderUNet): Network to size, built with a `patch_size`.
        preset (str, optional): One of `RESENC_PRESETS`. Default is "resenc_m".

    Returns:
        int: Suggested batch size.
    """
    dim = len(model.patch_size)
    reference = RESENC_PRESETS[preset]["reference_val_2d" if dim == 2 else "reference_val_3d"]
    reference_bs = REFERENCE_BATCH_SIZE_2D if dim == 2 else REFERENCE_BATCH_SIZE_3D

    estimate = model.compute_conv_feature_map_size(model.patch_size)
    return max(2, round((reference / estimate) * reference_bs))
