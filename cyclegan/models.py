"""
models.py
=========
Network architectures for the CBAM-CycleGAN image-enhancement module used in
"Breaking reCAPTCHAv2" pipeline.

This module defines:
    - ChannelAttention, SpatialAttention, CBAM   : attention block (Woo et al., 2018)
    - ResidualBlock                              : spectral-normalized residual
                                                    block with CBAM, used inside
                                                    the generator's bottleneck
    - CycleGenerator                             : ResNet-style generator
                                                    (Johnson et al., 2016 /
                                                    Zhu et al., 2017, CycleGAN)
    - PatchDiscriminator                         : 70x70 PatchGAN discriminator
                                                    with spectral normalization
                                                    (Miyato et al., 2018)
    - PerceptualLoss                             : VGG16-based feature
                                                    reconstruction loss
                                                    (Johnson et al., 2016)

Design notes for reproducibility
---------------------------------
* All normalization layers are InstanceNorm2d, following the original
  CycleGAN design choice for style-transfer tasks (batch statistics are not
  meaningful when translating individual images).
* Spectral normalization is applied to every convolution in the generator's
  residual blocks and to every convolution in the discriminator. This
  stabilizes adversarial training by constraining the Lipschitz constant of
  each layer, which we found necessary given the small, domain-specific
  reCAPTCHA image corpus used for training (small-sample instability is a
  well-known failure mode of vanilla CycleGAN).
* CBAM (Convolutional Block Attention Module) is inserted inside each
  residual block so the generator can focus on task-relevant regions
  (e.g. object boundaries, small foreground elements typical of reCAPTCHA
  tiles) rather than uniformly transforming the whole image.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


# ---------------------------------------------------------------------------
# Attention modules (CBAM)
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    """Channel-attention sub-module of CBAM (Woo et al., ECCV 2018)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.avg_pool(x).view(x.size(0), -1)
        maxp = self.max_pool(x).view(x.size(0), -1)
        attn = self.sigmoid(self.fc(avg) + self.fc(maxp))
        return attn.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """Spatial-attention sub-module of CBAM (Woo et al., ECCV 2018)."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        maxp, _ = x.max(dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, maxp], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention followed by
    spatial attention, applied multiplicatively to the input feature map."""

    def __init__(self, channels: int):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


def spectral_norm(module: nn.Module) -> nn.Module:
    """Thin wrapper around torch.nn.utils.spectral_norm for readability."""
    return nn.utils.spectral_norm(module)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Spectral-normalized residual block with a CBAM attention gate.

    Structure: [ReflectionPad -> SN-Conv3x3 -> InstanceNorm -> ReLU] x2,
    followed by CBAM, added back to the block input (identity skip
    connection), as in ResNet (He et al., 2016).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(channels, channels, kernel_size=3)),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(channels, channels, kernel_size=3)),
            nn.InstanceNorm2d(channels),
        )
        self.cbam = CBAM(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cbam(self.block(x))


class CycleGenerator(nn.Module):
    """ResNet-style CycleGAN generator (9 residual blocks), following
    Johnson et al. (2016) / Zhu et al. (2017), extended with CBAM attention
    inside every residual block.

    Pipeline: c7s1-64 -> d128 -> d256 -> R256 x 9 -> u128 -> u64 -> c7s1-3
    (naming convention from the original CycleGAN paper), operating on
    256x256 inputs normalized to [-1, 1] and producing outputs in the same
    range via a final Tanh activation.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64, n_res_blocks: int = 9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, kernel_size=7),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.down = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels * 4) for _ in range(n_res_blocks)]
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 4, base_channels * 2, kernel_size=3, stride=2,
                padding=1, output_padding=1,
            ),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                base_channels * 2, base_channels, kernel_size=3, stride=2,
                padding=1, output_padding=1,
            ),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, in_channels, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.down(x)
        x = self.res_blocks(x)
        x = self.up(x)
        return self.final(x)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------
class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator (Isola et al., 2017 / Zhu et al., 2017)
    with spectral normalization applied to every convolution for training
    stability (Miyato et al., 2018).

    Classifies overlapping image patches as real/fake rather than the image
    as a whole, which encourages the generator to produce locally realistic
    high-frequency detail.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        ]
        layers = [spectral_norm(l) if isinstance(l, nn.Conv2d) else l for l in layers]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Perceptual loss
# ---------------------------------------------------------------------------
class PerceptualLoss(nn.Module):
    """VGG16 feature-reconstruction loss (Johnson et al., 2016).

    Compares multi-scale feature activations of a frozen, ImageNet-pretrained
    VGG16 between the generated and the reference image, encouraging
    perceptual/structural similarity beyond raw pixel matching. The backbone
    is kept frozen (requires_grad=False) so gradients only flow into the
    generator.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for PerceptualLoss: pip install timm")
        self.vgg = timm.create_model(
            "vgg16.tv_in1k", pretrained=True, features_only=True
        ).to(device).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _extract(self, x: torch.Tensor):
        return self.vgg(x)

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        fake_feats = self.vgg(fake)
        with torch.no_grad():
            real_feats = self.vgg(real)
        loss = 0.0
        for f, r in zip(fake_feats, real_feats):
            loss = loss + F.l1_loss(f, r)
        return loss
