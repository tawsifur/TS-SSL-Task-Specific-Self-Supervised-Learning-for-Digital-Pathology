"""Stage 2a: patch-level classification with a frozen scAE encoder + MLP head."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models import Encoder, MLPHead
from ..utils import classification_metrics, save_checkpoint


@torch.no_grad()
def _evaluate(encoder, head, loader, device, freeze_encoder):
    head.eval()
    if freeze_encoder:
        encoder.eval()
    all_logits, all_targets = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device).float()
        feats = encoder.embed(imgs) if freeze_encoder else \
            nn.functional.adaptive_avg_pool2d(encoder(imgs), 1).flatten(1)
        logits = head(feats)
        all_logits.append(logits.cpu().numpy())
        all_targets.append(np.asarray(labels))
    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)
    return classification_metrics(logits, targets)


def train_patch_classifier(
    encoder: Encoder,
    train_set,
    val_set,
    out_path: str,
    n_classes: int,
    freeze_encoder: bool = True,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
) -> MLPHead:
    """Train an MLP head on top of scAE features. Returns the trained head."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    head = MLPHead(in_dim=encoder.embed_dim, n_classes=n_classes).to(device)

    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad_(False)
        params = head.parameters()
    else:
        params = list(encoder.parameters()) + list(head.parameters())

    optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=len(train_set) > batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    print(f"[patch] frozen_encoder={freeze_encoder} | classes={n_classes} | device={device}")

    best_acc, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        head.train()
        if not freeze_encoder:
            encoder.train()
        running = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device).float()
            labels = labels.to(device).long()

            if freeze_encoder:
                with torch.no_grad():
                    feats = encoder.embed(imgs)
            else:
                feats = nn.functional.adaptive_avg_pool2d(encoder(imgs), 1).flatten(1)

            logits = head(feats)
            loss = criterion(logits, labels)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += loss.item()

        metrics = _evaluate(encoder, head, val_loader, device, freeze_encoder)
        print(f"[patch] epoch {epoch:03d} | loss {running / len(train_loader):.4f} "
              f"| val_acc {metrics['accuracy']:.4f} | val_auc {metrics['auc']:.4f}")

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
    save_checkpoint(out_path, head, meta={"stage": "patch_head", "n_classes": n_classes,
                                          "best_val_acc": best_acc})
    print(f"[patch] best val acc {best_acc:.4f} | saved head -> {os.path.abspath(out_path)}")
    return head
