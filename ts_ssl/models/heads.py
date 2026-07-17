"""Downstream classification heads.

* ``MLPHead`` -- two dense layers with dropout for patch-level classification
  (Eq. 4 in the paper).
* ``AttentionMIL`` / ``GatedAttentionMIL`` -- attention-based multiple-instance
  learning pooling for weakly-supervised whole-slide classification (Eq. 5,
  after Ilse, Tomczak & Welling, ICML 2018).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """MLP classification head for patch-level tasks."""

    def __init__(self, in_dim: int = 128, hidden: int = 256, n_classes: int = 2,
                 dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionMIL(nn.Module):
    """Attention-based MIL pooling + classifier for a single bag of instances.

    A forward pass consumes one bag of shape ``(K, in_dim)`` (K instances) and
    returns class logits of shape ``(1, n_classes)`` plus the attention weights.
    """

    def __init__(self, in_dim: int = 128, hidden: int = 128, n_classes: int = 2,
                 gated: bool = False) -> None:
        super().__init__()
        self.gated = gated
        self.attention_V = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())
        if gated:
            self.attention_U = nn.Sequential(nn.Linear(in_dim, hidden), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden, 1)
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # bag may arrive as (1, K, D) from a DataLoader; squeeze the batch dim.
        if bag.dim() == 3:
            bag = bag.squeeze(0)

        a = self.attention_V(bag)                 # (K, hidden)
        if self.gated:
            a = a * self.attention_U(bag)
        a = self.attention_w(a)                   # (K, 1)
        a = torch.softmax(a, dim=0)               # attention weights over instances

        pooled = torch.sum(a * bag, dim=0, keepdim=True)  # (1, in_dim)  -- Eq. (5)
        logits = self.classifier(pooled)          # (1, n_classes)
        return logits, a.squeeze(-1)


class GatedAttentionMIL(AttentionMIL):
    """Convenience subclass that enables the gated attention variant."""

    def __init__(self, in_dim: int = 128, hidden: int = 128, n_classes: int = 2) -> None:
        super().__init__(in_dim=in_dim, hidden=hidden, n_classes=n_classes, gated=True)
