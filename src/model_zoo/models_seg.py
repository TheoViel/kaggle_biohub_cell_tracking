import timm
import torch
import torch.nn as nn
import segmentation_models_pytorch

from timm.layers import Conv2dSame
# from model_zoo.conv3d_same import Conv3dSame
from model_zoo.cellpose_net import CPNET_PRESETS, define_cpnet
from model_zoo.unet import Unet
from model_zoo.resenc_unet import RESENC_PRESETS, define_resenc_unet, suggest_batch_size


DECODERS = {"Unet": Unet}

# nnU-Net ResEnc U-Nets are whole architectures, not timm encoders, so they get their own
# branch in define_model: `decoder_name="ResEncUNet"`, `encoder_name` picks the preset.
RESENC_DECODER = "ResEncUNet"

# Cellpose's own CPnet, ported to 3D: decoder_name="CPnet", encoder_name picks
# the channel widths. See model_zoo/cellpose_net.py.
CPNET_DECODER = "CPnet"

# Patch sizes used to plan the ResEnc topology when the caller trains on full volumes.
DEFAULT_RESENC_PATCH = {2: (256, 256), 3: (64, 64, 64)}

# Output layout of the 3D cell segmentation pipeline, as name -> channel count.
CELL_SEG_HEADS = {"fg_logits": 1, "heatmap_logits": 1, "offsets": 3}
# Cellpose in 3D: one interior-probability channel and the z/y/x flow field.
# See segmentation/flows.py for what the flows are and dynamics.py for how the
# cells are rebuilt from them.
CELLPOSE_HEADS = {"cellprob_logits": 1, "flows": 3}


def define_model(
    decoder_name,
    encoder_name,
    num_classes=1,
    num_classes_aux=1,
    increase_stride=False,
    encoder_weights="imagenet",
    pretrained=True,
    use_cls=False,
    n_channels=3,
    pretrained_weights=None,
    use_3d=False,
    heads=None,
    patch_size=None,
    spacing=None,
    deep_supervision=False,
    verbose=0,
):
    """
    Define a segmentation model.

    Two families are available:
      - a U-Net decoder over a timm encoder, e.g. decoder_name="Unet", encoder_name="resnet18",
        built in 2D and converted to 3D when use_3d is set;
      - an nnU-Net ResEnc U-Net, decoder_name="ResEncUNet" with encoder_name one of
        RESENC_PRESETS ("resenc_m", "resenc_l", "resenc_xl"), which is natively 2D or 3D and
        whose topology is planned from patch_size.

    Args:
        decoder_name (str): Name of the decoder architecture.
        encoder_name (str): Name of the encoder architecture, or of the ResEnc preset.
        num_classes (int, optional): Number of primary classes. Default is 1.
        num_classes_aux (int, optional): Number of auxiliary classes. Default is 1.
        increase_stride (bool, optional): Flag to increase the stride. Default is False.
        encoder_weights (str, optional): Encoder weights source. Default is "imagenet".
        pretrained (bool, optional): Flag to use pretrained weights. Ignored by ResEnc U-Nets,
            which have no pretrained checkpoints. Default is True.
        use_cls (bool, optional): Flag to use auxiliary classifiers. Default is False.
        n_channels (int, optional): Number of input channels. Default is 3.
        pretrained_weights (str, optional): Path to pretrained model weights. Default is None.
        use_3d (bool, optional): Flag to use a 3D model. Default is False.
        heads (dict, optional): name -> channel count. When given, the model returns a
            dict of logits split along the channel axis instead of (masks, labels),
            and num_classes is set to the total. Default is None.
        patch_size (tuple, optional): Training patch size, which ResEnc U-Nets plan their
            stage count, strides and kernel sizes from. Falls back to DEFAULT_RESENC_PATCH
            when training on full volumes. Ignored by the timm family. Default is None.
        spacing (tuple, optional): Voxel spacing, used to plan anisotropic ResEnc topologies.
            Defaults to isotropic. Default is None.
        deep_supervision (bool, optional): For ResEnc U-Nets, also return the logits of the
            coarser decoder stages under the "deep_supervision" key. The loss has to consume
            them: under DDP, heads that receive no gradient abort the backward pass.
            Default is False.
        verbose (int, optional): Verbosity level. Default is 0.

    Returns:
        SegWrapper: Segmentation model with the specified configurations.

    Raises:
        AssertionError: If the decoder name is not supported.
    """
    assert (
        decoder_name in DECODERS
        or decoder_name in (RESENC_DECODER, CPNET_DECODER)
    ), "Decoder name not supported"

    if heads is not None:
        num_classes = sum(heads.values())

    if decoder_name == CPNET_DECODER:
        model = define_cpnet(
            preset=encoder_name,
            nout=num_classes,
            n_channels=n_channels,
            heads=heads,
            patch_size=patch_size,
            spacing=spacing,
            verbose=verbose,
        )
    elif decoder_name == RESENC_DECODER:
        model = define_resenc_seg_model(
            preset=encoder_name,
            num_classes=num_classes,
            num_classes_aux=num_classes_aux if use_cls else None,
            n_channels=n_channels,
            use_3d=use_3d,
            heads=heads,
            patch_size=patch_size,
            spacing=spacing,
            deep_supervision=deep_supervision,
            increase_stride=increase_stride,
            pretrained=pretrained,
            verbose=verbose,
        )
    else:
        decoder = DECODERS[decoder_name]
        model = decoder(
            encoder_name="tu-" + encoder_name,
            encoder_weights=encoder_weights if pretrained else None,
            in_channels=n_channels,
            classes=num_classes,
            aux_params={"dropout": 0.0, "classes": num_classes_aux} if use_cls else None,
        )
        model.n_channels = n_channels
        model.name = encoder_name
        model.decoder_channels = (256, 128, 64, 32, 16)
        model.num_classes = num_classes

        model = SegWrapper(model, heads=heads)

        if increase_stride:
            model.increase_stride()

        if use_3d:
            model = convert_3d(model)

    if pretrained_weights is not None:
        if verbose:
            print(f'\n-> Loading weights from "{pretrained_weights}"\n')
        state_dict = torch.load(pretrained_weights)
        model.load_state_dict(state_dict, strict=True)

    return model


def define_resenc_seg_model(
    preset="resenc_m",
    num_classes=1,
    num_classes_aux=None,
    n_channels=1,
    use_3d=True,
    heads=None,
    patch_size=None,
    spacing=None,
    deep_supervision=False,
    increase_stride=False,
    pretrained=False,
    verbose=0,
):
    """
    Build an nnU-Net ResEnc U-Net wrapped to return the pipeline's head layout.

    Args:
        preset (str, optional): One of RESENC_PRESETS. Default is "resenc_m".
        num_classes (int, optional): Total number of output channels. Default is 1.
        num_classes_aux (int, optional): Number of auxiliary classes. None disables the
            classification head. Default is None.
        n_channels (int, optional): Number of input channels. Default is 1.
        use_3d (bool, optional): Build a 3D model rather than a 2D one. Default is True.
        heads (dict, optional): name -> channel count used to split the output. Default is None.
        patch_size (tuple, optional): Training patch size the topology is planned from.
            Falls back to DEFAULT_RESENC_PATCH. Default is None.
        spacing (tuple, optional): Voxel spacing. Defaults to isotropic. Default is None.
        deep_supervision (bool, optional): Also return the coarser decoder stages, which the
            loss then has to consume. Default is False.
        increase_stride (bool, optional): Unsupported, kept for signature parity. Default is False.
        pretrained (bool, optional): Ignored, there are no pretrained ResEnc checkpoints.
            Default is False.
        verbose (int, optional): Verbosity level. Default is 0.

    Returns:
        ResEncSegWrapper: The wrapped network.

    Raises:
        AssertionError: If the preset is unknown, increase_stride is set, or patch_size does
            not match use_3d.
    """
    assert preset in RESENC_PRESETS, (
        f'Unknown ResEnc preset "{preset}", expected one of {list(RESENC_PRESETS)}'
    )
    assert not increase_stride, "increase_stride is not supported by ResEnc U-Nets"

    dim = 3 if use_3d else 2
    patch_size = DEFAULT_RESENC_PATCH[dim] if patch_size is None else tuple(patch_size)
    assert len(patch_size) == dim, f"patch_size {patch_size} does not match use_3d={use_3d}"

    if verbose and pretrained:
        print(f"\n-> No pretrained weights for {preset}, training the ResEnc U-Net from scratch")

    model = define_resenc_unet(
        preset=preset,
        in_channels=n_channels,
        num_classes=num_classes,
        patch_size=patch_size,
        spacing=spacing,
        deep_supervision=deep_supervision,
        verbose=verbose,
    )
    return ResEncSegWrapper(model, heads=heads, num_classes_aux=num_classes_aux)


def split_heads(masks, heads):
    """
    Split the channel axis of a logits tensor into the named heads.

    Args:
        masks (torch.Tensor): Logits of shape (B, sum(heads.values()), *spatial).
        heads (dict, optional): name -> channel count. None returns the tensor as "masks".

    Returns:
        dict: Logits per head name.
    """
    if heads is None:
        return {"masks": masks}

    split, start = {}, 0
    for name, channels in heads.items():
        split[name] = masks[:, start : start + channels]
        start += channels
    return split


class GlobalPoolClassificationHead(nn.Sequential):
    """
    Auxiliary classifier over pooled bottleneck features, for 2D or 3D inputs.
    """

    def __init__(self, in_channels, classes, dropout=0.0, dim=3):
        """
        Constructor.

        Args:
            in_channels (int): Number of bottleneck channels.
            classes (int): Number of auxiliary classes.
            dropout (float, optional): Dropout applied before the linear layer. Default is 0.0.
            dim (int, optional): Number of spatial dimensions, 2 or 3. Default is 3.
        """
        pool = nn.AdaptiveAvgPool3d(1) if dim == 3 else nn.AdaptiveAvgPool2d(1)
        super().__init__(
            pool,
            nn.Flatten(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_channels, classes, bias=True),
        )


class ResEncSegWrapper(nn.Module):
    """
    Wraps an nnU-Net ResEnc U-Net so that it returns the pipeline's dict of head logits.

    Attributes:
        model (ResidualEncoderUNet): The underlying network.
        heads (dict): Optional name -> channel count used to split the output.
        divisible_by (tuple): Per-axis multiple the input spatial size must be padded to.
    """

    def __init__(self, model, heads=None, num_classes_aux=None):
        """
        Constructor.

        Args:
            model (ResidualEncoderUNet): The underlying network.
            heads (dict, optional): name -> channel count. When given, forward returns a dict
                of logits instead of a single "masks" entry. Default is None.
            num_classes_aux (int, optional): Number of auxiliary classes. None disables the
                classification head. Default is None.
        """
        super().__init__()

        self.model = model
        self.name = model.preset
        self.heads = heads
        self.num_classes = model.num_classes
        self.n_channels = model.encoder.stem.convs[0].input_channels
        self.divisible_by = model.divisible_by
        self.patch_size = model.patch_size

        self.classification_head = None
        if num_classes_aux is not None:
            self.classification_head = GlobalPoolClassificationHead(
                model.encoder.output_channels[-1],
                num_classes_aux,
                dim=len(model.patch_size),
            )

    def increase_stride(self):
        """
        Unsupported: a ResEnc U-Net's stride chain is fixed by its planned topology.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("increase_stride is not supported by ResEnc U-Nets")

    def forward(self, x):
        """
        Forward pass of the segmentation model.

        Args:
            x (torch.Tensor): Input data as a tensor of shape (B, C, *spatial).

        Returns:
            dict: Logits per head, plus "labels" if a classification head was built, plus
                "deep_supervision" -- a list of per-head dicts, coarsest last -- if the
                underlying network was built with deep supervision.
        """
        skips = self.model.encoder(x)
        out = self.model.decoder(skips)

        if self.model.deep_supervision:
            masks, coarser = out[0], out[1:]
        else:
            masks, coarser = out, []

        heads = split_heads(masks, self.heads)
        if coarser:
            heads["deep_supervision"] = [split_heads(m, self.heads) for m in coarser]

        if self.classification_head is not None:
            heads["labels"] = self.classification_head(skips[-1])

        return heads

    def count_parameters(self):
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def suggest_batch_size(self, preset=None):
        """
        Largest batch size the preset's VRAM target allows for this network and patch size.

        Args:
            preset (str, optional): One of RESENC_PRESETS. Defaults to the model's own preset.

        Returns:
            int: Suggested batch size.
        """
        return suggest_batch_size(self.model, preset or self.name)


class SegWrapper(nn.Module):
    """
    A wrapper class for segmentation models.

    Attributes:
        model (nn.Module): The underlying segmentation model.
        num_classes (int): The number of classes in the model.
        heads (dict): Optional name -> channel count used to split the output.
    """

    def __init__(self, model, heads=None):
        """
        Constructor.

        Args:
            model (nn.Module): The underlying segmentation model.
            heads (dict, optional): name -> channel count. When given, forward returns
                a dict of logits instead of (masks, labels). Default is None.
        """
        super().__init__()

        self.model = model
        self.name = model.name
        self.num_classes = model.num_classes
        self.heads = heads

        self._is_transformer_style = model.encoder._is_transformer_style
        self.model.encoder._is_transformer_style = False

    def increase_stride(self):
        """
        Increase the stride of the first layer of the encoder.
        """
        if "efficientnet" in self.name:
            self.model.encoder.model.conv_stem.stride = (4, 4)
        else:
            try:
                self.model.encoder.model.conv1[3].stride = (2, 2)
            except Exception:
                self.model.encoder.model.conv1.stride = (4, 4)

        self.model.segmentation_head = (
            segmentation_models_pytorch.base.SegmentationHead(
                in_channels=self.model.decoder_channels[-1],
                upsampling=2,
                out_channels=self.model.num_classes,
                activation=None,
                kernel_size=3,
            )
        )

    def forward(self, x):
        """
        Forward pass of the segmentation model.

        Args:
            x (torch.Tensor): Input data as a tensor.

        Returns:
            dict: Logits per head, if the model was built with `heads`.
            torch.Tensor: Segmentation masks, otherwise.
            torch.Tensor: Labels, otherwise.
        """

        features = self.model.encoder(x)

        if self._is_transformer_style:  # convnextv2
            B, _, D, H, W = x.shape
            dummy_1 = torch.empty(
                [B, 0, D // 2, H // 2, W // 2], dtype=x.dtype, device=x.device
            )
            dummy_2 = torch.empty([B, 0, D, H, W], dtype=x.dtype, device=x.device)
            features = [dummy_2, dummy_1] + features[1:]

        # for f in features:
        #     print(f.shape)

        decoder_output = self.model.decoder(features)

        masks = self.model.segmentation_head(decoder_output)

        heads = split_heads(masks, self.heads)

        if self.model.classification_head is not None:
            heads["labels"] = self.model.classification_head(features[-1])

        return heads

    def count_parameters(self):
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def convert_3d(module):
    """
    Convert a 2D module to its 3D counterpart.
    Adapted from: https://www.kaggle.com/code/haqishen/rsna-2022-1st-place-solution-train-stage1

    Args:
        module (torch.nn.Module): The 2D module to be converted to 3D.

    Returns:
        torch.nn.Module: The 3D equivalent of the input 2D module.
    """
    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = torch.nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    elif isinstance(module, Conv2dSame):
        raise NotImplementedError("Conv3dSame is not implemented")
        # module_output = Conv3dSame(
        #     in_channels=module.in_channels,
        #     out_channels=module.out_channels,
        #     kernel_size=module.kernel_size[0],
        #     stride=module.stride[0],
        #     padding=module.padding[0],
        #     dilation=module.dilation[0],
        #     groups=module.groups,
        #     bias=module.bias is not None,
        # )
        # module_output.weight = torch.nn.Parameter(
        #     module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0])
        # )

    elif isinstance(module, torch.nn.Conv2d):
        module_output = torch.nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
        )
        module_output.weight = torch.nn.Parameter(
            module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0])
        )

    elif isinstance(module, torch.nn.MaxPool2d):
        module_output = torch.nn.MaxPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        module_output = torch.nn.AvgPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.UpsamplingBilinear2d):
        module_output = torch.nn.Upsample(
            scale_factor=module.scale_factor, mode="trilinear"
        )

    elif isinstance(module, timm.layers.norm.LayerNorm2d):
        module_output = LayerNorm3d(module.normalized_shape)
        module_output.weight = module.weight
        module_output.bias = module.bias
        module_output.eps = module.eps

    elif isinstance(module, timm.models.convnext.ConvNeXtBlock):

        def forward_3d(self, x):
            """Forward pass."""
            shortcut = x
            x = self.conv_dw(x)
            if self.use_conv_mlp:
                x = self.norm(x)
                x = self.mlp(x)
            else:
                x = x.permute(0, 2, 3, 4, 1)
                x = self.norm(x)
                x = self.mlp(x)
                x = x.permute(0, 4, 1, 2, 3)
            if self.gamma is not None:
                x = x.mul(self.gamma.reshape(1, -1, 1, 1, 1))

            x = self.drop_path(x) + self.shortcut(shortcut)
            return x

        module_output = module
        module_output.forward = lambda x: forward_3d(module_output, x)

    for name, child in module.named_children():
        module_output.add_module(name, convert_3d(child))
    del module

    return module_output


class LayerNorm3d(nn.LayerNorm):
    """
    LayerNorm for channels of '3D' spatial BCDHW tensors.

    Args:
        num_channels (int): Number of channels to normalize.
    """

    def __init__(self, num_channels):
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to 3D tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: Normalized tensor of same shape.
        """
        return nn.functional.layer_norm(
            x.permute(0, 2, 3, 4, 1),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        ).permute(0, 4, 1, 2, 3)
