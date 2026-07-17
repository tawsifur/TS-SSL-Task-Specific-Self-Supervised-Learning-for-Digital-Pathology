"""Stage 2b: weakly-supervised WSI classification with attention-MIL.

Consumes per-slide feature bags (produced by ``ts-ssl extract``) and trains an
attention-MIL classifier. Each slide is one bag; a batch size of 1 bag is used
because bags have variable numbers of patches.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models import AttentionMIL, GatedAttentionMIL
from ..utils import classification_metrics, save_checkpoint


def _collate_single_bag(batch):
    # DataLoader batch_size=1 -> list with one (features, label) tuple.
    feats, label = batch[0]
    return feats, label


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    for feats, label in loader:
        feats = feats.to(device).float()
        logits, _ = model(feats)
        all_logits.append(logits.cpu().numpy())
        all_targets.append(int(label))
    logits = np.concatenate(all_logits)
    targets = np.asarray(all_targets)
    return classification_metrics(logits, targets)


def train_wsi_classifier(
    train_set,
    val_set,
    out_path: str,
    feature_dim: int,
    n_classes: int,
    gated: bool = False,
    epochs: int = 50,
    lr: float = 5e-4,
    weight_decay: float = 1e-2,
    hidden: int = 128,
    device: Optional[torch.device] = None,
):
    """Train an attention-MIL classifier over slide-level feature bags."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Model = GatedAttentionMIL if gated else AttentionMIL
    model = (Model(in_dim=feature_dim, hidden=hidden, n_classes=n_classes)
             if gated else
             Model(in_dim=feature_dim, hidden=hidden, n_classes=n_classes, gated=False)).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                             weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              collate_fn=_collate_single_bag)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            collate_fn=_collate_single_bag)

    print(f"[wsi] gated={gated} | feature_dim={feature_dim} | classes={n_classes} "
          f"| device={device}")

    best_acc, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for feats, label in train_loader:
            feats = feats.to(device).float()
            label = torch.tensor([int(label)], device=device)

            logits, _ = model(feats)
            loss = criterion(logits, label)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += loss.item()

        metrics = _evaluate(model, val_loader, device)
        print(f"[wsi] epoch {epoch:03d} | loss {running / len(train_loader):.4f} "
              f"| val_acc {metrics['accuracy']:.4f} | val_auc {metrics['auc']:.4f}")

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(out_path, model, meta={"stage": "wsi_mil", "n_classes": n_classes,
                                           "feature_dim": feature_dim, "gated": gated,
                                           "best_val_acc": best_acc})
    print(f"[wsi] best val acc {best_acc:.4f} | saved model -> {os.path.abspath(out_path)}")
    return model
