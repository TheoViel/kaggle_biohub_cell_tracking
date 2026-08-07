import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """nnUNet-style double conv: Conv → InstanceNorm → LeakyReLU (×2)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """Strided conv for downsampling, then double conv."""

    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.conv = ConvBlock(out_ch, out_ch)

    def forward(self, x):
        return self.conv(self.down(x))


class UpBlock(nn.Module):
    """Transposed conv upsample + skip concat + double conv."""

    def __init__(self, in_ch, skip_ch, out_ch, stride=2):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, stride, stride=stride)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        return self.conv(torch.cat([x, skip], dim=1))


class CellSegNet(nn.Module):
    """
    3D U-Net with nnUNet-style blocks.

    Input  : (B, 1, D, H, W) — normalised to [0, 1]
    Outputs (dict):
        fg_logits       (B, 1, D, H, W) — foreground probability logits
        heatmap_logits  (B, 1, D, H, W) — cell-centre heatmap logits
        offsets         (B, 3, D, H, W) — normalised offset vectors to centroid
    """

    def __init__(self, in_channels=1, features=(32, 64, 128, 256, 256)):
        super().__init__()
        f = features

        # Encoder
        self.enc0 = ConvBlock(in_channels, f[0])
        self.enc1 = DownBlock(f[0], f[1])
        self.enc2 = DownBlock(f[1], f[2])
        self.enc3 = DownBlock(f[2], f[3])

        # Bottleneck
        self.bottleneck = DownBlock(f[3], f[4])

        # Decoder
        self.dec3 = UpBlock(f[4], f[3], f[3])
        self.dec2 = UpBlock(f[3], f[2], f[2])
        self.dec1 = UpBlock(f[2], f[1], f[1])
        self.dec0 = UpBlock(f[1], f[0], f[0])

        # Output heads (1×1×1 conv)
        self.head_fg = nn.Conv3d(f[0], 1, 1)
        self.head_heatmap = nn.Conv3d(f[0], 1, 1)
        self.head_offsets = nn.Conv3d(f[0], 3, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)

        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)

        return {
            'fg_logits': self.head_fg(d0),
            'heatmap_logits': self.head_heatmap(d0),
            'offsets': self.head_offsets(d0),
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
