#!/usr/bin/env python
"""Extract TS-SSL features for many slides at once.

Given a directory where each slide is a subfolder of patches, e.g.

    slides_root/
    ├── slide_000/  patch0.png  patch1.png ...
    ├── slide_001/  ...
    └── ...

this writes one ``<slide_id>.h5`` (with a ``features`` dataset) per slide into
``--out``, ready for ``ts-ssl train-wsi``.

Usage:
    python scripts/extract_wsi_features.py \
        --checkpoint checkpoints/scae.pt \
        --slides-root slides_root/ \
        --out wsi_features/ \
        --image-size 96
"""

from __future__ import annotations

import argparse
import os

import torch

from ts_ssl.data import PatchFolder, build_transform
from ts_ssl.engine import extract_features
from ts_ssl.models import SCAE
from ts_ssl.utils import get_device, load_checkpoint


def _load_scae(ckpt_path: str, device) -> SCAE:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    model = SCAE(in_channels=meta.get("in_channels", 3),
                 base=meta.get("base", 32),
                 embed_dim=meta.get("embed_dim", 128))
    load_checkpoint(ckpt_path, model)
    return model.to(device)


class _FlatFolder(torch.utils.data.Dataset):
    """Dataset over a flat folder of images (no class subdirectories)."""

    def __init__(self, folder, transform):
        from PIL import Image
        self._Image = Image
        self.transform = transform
        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        self.files = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(exts)
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = self._Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--slides-root", required=True, dest="slides_root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-size", type=int, default=96, dest="image_size")
    ap.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    scae = _load_scae(args.checkpoint, device)
    tf = build_transform(args.image_size, train=False)
    os.makedirs(args.out, exist_ok=True)

    slides = sorted(
        d for d in os.listdir(args.slides_root)
        if os.path.isdir(os.path.join(args.slides_root, d))
    )
    print(f"Found {len(slides)} slides under {args.slides_root}")

    for slide_id in slides:
        folder = os.path.join(args.slides_root, slide_id)
        ds = _FlatFolder(folder, tf)
        if len(ds) == 0:
            print(f"  [skip] {slide_id}: no images")
            continue
        out_h5 = os.path.join(args.out, f"{slide_id}.h5")
        extract_features(scae.encoder, ds, out_h5=out_h5,
                         batch_size=args.batch_size, num_workers=args.workers,
                         device=device, store_coords=False)


if __name__ == "__main__":
    main()
