"""Spatial-Channel Attention Autoencoder (scAE).

This is the backbone of the TS-SSL framework. During self-supervised
pretraining the full autoencoder is trained to reconstruct input patches.
Afterwards the encoder is frozen and used as a task-specific feature
extractor for downstream patch- and slide-level classification.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ChannelAttention, SpatialAttention


class ConvBlock(nn.Module):
    """3x3 conv -> BatchNorm -> ReLU, as described in the paper."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """scAE encoder: two conv blocks, a spatial + channel attention branch,
    and a fusion conv block that produces the task-specific embedding map.
    """

    def __init__(self, in_channels: int = 3, base: int = 32, embed_dim: int = 128) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_channels, base)
        self.conv2 = ConvBlock(base, base * 2)
        self.pool = nn.MaxPool2d(2)  # halve spatial size after each conv block

        self.spatial = SpatialAttention(base * 2)
        self.channel = ChannelAttention(base * 2)

        # Concatenate [original, spatial-attended, channel-attended] -> fuse.
        self.fuse = ConvBlock(base * 2 * 3, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        s = self.spatial(x)
        c = self.channel(x)
        x = torch.cat([x, s, c], dim=1)
        return self.fuse(x)  # (B, embed_dim, H/4, W/4)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return a single feature vector per input via global average pooling.

        This is what downstream heads and the feature extractor consume.
        """
        feat = self.forward(x)
        return F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, embed_dim)


class Decoder(nn.Module):
    """Symmetric decoder that upsamples the embedding back to the input."""

    def __init__(self, out_channels: int = 3, base: int = 32, embed_dim: int = 128) -> None:
        super().__init__()
        self.up1 = ConvBlock(embed_dim, base * 2)
        self.spatial = SpatialAttention(base * 2)
        self.channel = ChannelAttention(base * 2)
        self.fuse = ConvBlock(base * 2 * 3, base)
        self.up2 = ConvBlock(base, base)
        self.head = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, z: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)
        s = self.spatial(x)
        c = self.channel(x)
        x = torch.cat([x, s, c], dim=1)
        x = self.fuse(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)
        x = self.head(x)
        # Ensure the reconstruction matches the exact input resolution.
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(x)


class SCAE(nn.Module):
    """Full Spatial-Channel Attention Autoencoder used for pretraining."""

    def __init__(self, in_channels: int = 3, base: int = 32, embed_dim: int = 128) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, base, embed_dim)
        self.decoder = Decoder(in_channels, base, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z, out_size=x.shape[-2:])

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """L2 reconstruction loss, Eq. (1) in the paper."""
        recon = self.forward(x)
        return F.mse_loss(recon, x)
