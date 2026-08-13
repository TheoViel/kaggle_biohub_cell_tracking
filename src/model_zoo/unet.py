# Adapted from https://github.com/qubvel/segmentation_models.pytorch

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable

import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading


class UnetDecoderBlock(nn.Module):
    """A decoder block in the U-Net architecture that performs upsampling and feature fusion."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        attention_type: Optional[str] = None,
        interpolation_mode: str = "nearest",
    ):
        """
        Initialize U-Net decoder block.

        Args:
            in_channels (int): Number of input channels.
            skip_channels (int): Number of skip connection channels.
            out_channels (int): Number of output channels.
            use_norm (Union[bool, str, Dict[str, Any]], optional): Normalization
                type. Defaults to "batchnorm".
            attention_type (Optional[str], optional): Attention mechanism type.
                Defaults to None.
            interpolation_mode (str, optional): Interpolation mode for
                upsampling. Defaults to "nearest".
        """
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
        )
        self.attention1 = md.Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(
        self,
        feature_map: torch.Tensor,
        target_size: Tuple[int, int],
        skip_connection: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through decoder block.

        Args:
            feature_map (torch.Tensor): Input feature map to upsample.
            target_size (Tuple[int, int]): Target spatial size for upsampling.
            skip_connection (Optional[torch.Tensor], optional): Skip connection
                from encoder. Defaults to None.

        Returns:
            torch.Tensor: Decoded feature map.
        """
        # print("feature_map", feature_map.size())
        # print("target_size", target_size)
        # if skip_connection is not None:
        #     print("skip_connection", skip_connection.size())
        # print("skip_connection", skip_connection.size())
        # print("before", feature_map.size())
        feature_map = F.interpolate(
            feature_map,
            size=target_size,
            mode=self.interpolation_mode,
        )
        # print("after", feature_map.size())
        if skip_connection is not None:
            # print("skip_connection", skip_connection.size())
            feature_map = torch.cat([feature_map, skip_connection], dim=1)
            feature_map = self.attention1(feature_map)
        feature_map = self.conv1(feature_map)
        feature_map = self.conv2(feature_map)
        feature_map = self.attention2(feature_map)
        return feature_map


class UnetCenterBlock(nn.Sequential):
    """Center block of the Unet decoder. Applied to the last feature map of the encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
    ):
        """
        Initialize U-Net center block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_norm (Union[bool, str, Dict[str, Any]], optional): Normalization
                type. Defaults to "batchnorm".
        """
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    """The decoder part of the U-Net architecture.

    Takes encoded features from different stages of the encoder and progressively upsamples
    them while combining with skip connections.
    This helps preserve fine-grained details in the final segmentation.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        n_blocks: int = 5,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        attention_type: Optional[str] = None,
        add_center_block: bool = False,
        interpolation_mode: str = "nearest",
    ):
        """
        Initialize U-Net decoder.

        Args:
            encoder_channels (Sequence[int]): Number of channels from each
                encoder stage.
            decoder_channels (Sequence[int]): Number of channels for each
                decoder block.
            n_blocks (int, optional): Number of decoder blocks. Defaults to 5.
            use_norm (Union[bool, str, Dict[str, Any]], optional): Normalization
                type. Defaults to "batchnorm".
            attention_type (Optional[str], optional): Attention mechanism type.
                Defaults to None.
            add_center_block (bool, optional): Whether to add center block.
                Defaults to False.
            interpolation_mode (str, optional): Interpolation mode for
                upsampling. Defaults to "nearest".
        """
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if add_center_block:
            self.center = UnetCenterBlock(
                head_channels,
                head_channels,
                use_norm=use_norm,
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        self.blocks = nn.ModuleList()
        for block_in_channels, block_skip_channels, block_out_channels in zip(
            in_channels, skip_channels, out_channels
        ):
            block = UnetDecoderBlock(
                block_in_channels,
                block_skip_channels,
                block_out_channels,
                use_norm=use_norm,
                attention_type=attention_type,
                interpolation_mode=interpolation_mode,
            )
            self.blocks.append(block)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            features (List[torch.Tensor]): List of feature maps from encoder stages.

        Returns:
            torch.Tensor: Decoded feature map.
        """
        # spatial shapes of features: [hw, hw/2, hw/4, hw/8, ...]
        spatial_shapes = [feature.shape[2:] for feature in features]
        spatial_shapes = spatial_shapes[::-1]

        # print(spatial_shapes)

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skip_connections = features[1:]

        x = self.center(head)

        for i, decoder_block in enumerate(self.blocks):
            # upsample to the next spatial shape
            skip_connection = skip_connections[i] if i < len(skip_connections) else None
            x = decoder_block(x, spatial_shapes[i + 1], skip_connection=skip_connection)

            # print(f"block {i+1}: {x.shape}")

        return x


class Unet(SegmentationModel):
    """
    U-Net model for image segmentation.
    """

    requires_divisible_input_shape = False

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        decoder_interpolation: str = "nearest",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        """
        Initialize U-Net model.

        Args:
            encoder_name (str, optional): Name of encoder backbone.
                Defaults to "resnet34".
            encoder_depth (int, optional): Depth of the encoder. Defaults to 5.
            encoder_weights (Optional[str], optional): Pretrained weights for
                encoder. Defaults to "imagenet".
            decoder_use_norm (Union[bool, str, Dict[str, Any]], optional):
                Normalization type. Defaults to "batchnorm".
            decoder_channels (Sequence[int], optional): Number of channels in
                decoder blocks. Defaults to (256, 128, 64, 32, 16).
            decoder_attention_type (Optional[str], optional): Attention
                mechanism type. Defaults to None.
            decoder_interpolation (str, optional): Interpolation mode for
                upsampling. Defaults to "nearest".
            in_channels (int, optional): Number of input channels.
                Defaults to 3.
            classes (int, optional): Number of output classes. Defaults to 1.
            activation (Optional[Union[str, Callable]], optional): Activation
                function. Defaults to None.
            aux_params (Optional[dict], optional): Parameters for auxiliary
                classifier. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()

        decoder_use_batchnorm = kwargs.pop("decoder_use_batchnorm", None)
        if decoder_use_batchnorm is not None:
            warnings.warn(
                "The usage of decoder_use_batchnorm is deprecated. Please modify your code for decoder_use_norm",  # noqa
                DeprecationWarning,
                stacklevel=2,
            )
            decoder_use_norm = decoder_use_batchnorm

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        add_center_block = encoder_name.startswith("vgg")

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_norm=decoder_use_norm,
            add_center_block=add_center_block,
            attention_type=decoder_attention_type,
            interpolation_mode=decoder_interpolation,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()
