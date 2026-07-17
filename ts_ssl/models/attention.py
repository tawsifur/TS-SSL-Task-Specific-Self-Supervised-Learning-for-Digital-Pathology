"""Attention blocks used inside the Spatial-Channel Attention Autoencoder (scAE).

These implement the two attention mechanisms described in the TS-SSL paper
(Rahman, Baras & Chellappa, *Modern Pathology* 2025):

* ``SpatialAttention`` -- a non-local (embedded-Gaussian) self-attention block
  that captures long-range context between local regions of the feature map.
  It follows the formulation ``alpha = Softmax(K Q^T) V`` used in the paper,
  building on Non-Local Neural Networks (Wang et al., CVPR 2018).

* ``ChannelAttention`` -- a squeeze-and-excitation style block that reweights
  channels using two fully connected layers, following
  ``F' = M (x) sigma(Phi(M))`` from the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """Non-local self-attention over spatial positions.

    Args:
        in_channels: Number of input feature-map channels.
        reduction: Channel reduction factor for the embedded representation.
    """

    def __init__(self, in_channels: int, reduction: int = 2) -> None:
        super().__init__()
        self.inter_channels = max(in_channels // reduction, 1)

        # Three parallel 1x1 convolutions produce the K, Q and V embeddings.
        self.key = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.query = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)

        # Max pooling reduces the number of key/value positions (efficiency).
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Project the attended features back to the input dimensionality.
        self.project = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape

        # Query keeps full resolution; key/value are pooled for efficiency.
        query = self.query(x).view(b, self.inter_channels, h * w)
        query = query.permute(0, 2, 1)                       # (B, HW, C')
        key = self.pool(self.key(x)).view(b, self.inter_channels, -1)      # (B, C', HW'')
        value = self.pool(self.value(x)).view(b, self.inter_channels, -1)  # (B, C', HW'')
        value = value.permute(0, 2, 1)                       # (B, HW'', C')

        attn = torch.softmax(torch.bmm(query, key), dim=-1)  # (B, HW, HW'')
        out = torch.bmm(attn, value)                         # (B, HW, C')
        out = out.permute(0, 2, 1).contiguous().view(b, self.inter_channels, h, w)

        out = self.project(out)
        # Residual connection keeps the original signal while adding context.
        return self.relu(out + x)


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel re-weighting.

    Args:
        in_channels: Number of input feature-map channels.
        reduction: Bottleneck reduction factor for the excitation MLP.
    """

    def __init__(self, in_channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(in_channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.squeeze(x).view(b, c)
        weights = self.excite(weights).view(b, c, 1, 1)
        return x * weights
