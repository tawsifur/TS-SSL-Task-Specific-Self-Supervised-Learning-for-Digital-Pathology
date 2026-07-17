"""Feature extraction with a frozen scAE encoder.

Given a trained scAE checkpoint, extract task-specific embeddings for a set of
patches and save them as ``.h5`` files (one ``features`` matrix per input
source) ready for the WSI-MIL stage.
"""

from __future__ import annotations

import os
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models import Encoder


@torch.no_grad()
def extract_features(
    encoder: Encoder,
    dataset,
    out_h5: str,
    batch_size: int = 128,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    store_coords: bool = True,
) -> str:
    """Run the frozen encoder over a dataset and dump features to ``out_h5``."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device).eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    feats_all, coords_all = [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            imgs, coords = batch
        else:
            imgs, coords = batch, None
        imgs = imgs.to(device).float()
        emb = encoder.embed(imgs).cpu().numpy()
        feats_all.append(emb)
        if store_coords and coords is not None:
            coords_all.append(np.asarray(coords))

    features = np.concatenate(feats_all, axis=0).astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(out_h5)), exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("features", data=features)
        if coords_all:
            f.create_dataset("coords", data=np.concatenate(coords_all, axis=0))
    print(f"[extract] {features.shape[0]} vectors x {features.shape[1]}-d "
          f"-> {os.path.abspath(out_h5)}")
    return out_h5
