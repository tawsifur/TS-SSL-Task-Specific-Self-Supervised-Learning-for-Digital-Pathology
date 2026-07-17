"""Stage 1: self-supervised pretraining of the scAE.

The autoencoder is trained to reconstruct a *subsample* of the downstream
patches (the paper uses 1-50%). Only the reconstruction loss is used; no
labels are required.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset

from ..models import SCAE
from ..utils import save_checkpoint


def _subsample(dataset, fraction: float, seed: int):
    if fraction >= 1.0:
        return dataset
    n = max(1, int(len(dataset) * fraction))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
    return Subset(dataset, idx)


def pretrain_scae(
    dataset,
    out_path: str,
    embed_dim: int = 128,
    base: int = 32,
    in_channels: int = 3,
    subsample: float = 0.1,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 3e-3,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    seed: int = 42,
    log_every: int = 10,
) -> SCAE:
    """Pretrain and checkpoint an ``SCAE``. Returns the trained model."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = _subsample(dataset, subsample, seed)
    # Only drop the last partial batch when we have more than one full batch;
    # otherwise a small dataset would yield an empty loader.
    drop_last = len(train_set) > batch_size
    loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=drop_last, pin_memory=(device.type == "cuda"),
    )

    model = SCAE(in_channels=in_channels, base=base, embed_dim=embed_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    print(f"[pretrain] {len(train_set)}/{len(dataset)} patches "
          f"({subsample:.0%} subsample) | device={device}")

    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for step, batch in enumerate(loader, 1):
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = imgs.to(device, non_blocking=True).float()

            optim.zero_grad(set_to_none=True)
            loss = model.reconstruction_loss(imgs)
            loss.backward()
            optim.step()

            running += loss.item()
            if step % log_every == 0:
                print(f"  epoch {epoch:03d} | step {step:04d} | "
                      f"recon_loss {running / step:.4f}")
        print(f"[pretrain] epoch {epoch:03d} done | "
              f"avg_loss {running / max(1, len(loader)):.4f}")

    save_checkpoint(out_path, model, meta={
        "embed_dim": embed_dim, "base": base, "in_channels": in_channels,
        "stage": "scae_pretrain",
    })
    print(f"[pretrain] saved scAE -> {os.path.abspath(out_path)}")
    return model
