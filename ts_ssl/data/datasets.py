"""Dataset utilities for TS-SSL.

Supported layouts
-----------------
* **Patch classification / pretraining** -- a directory of images arranged as
  ``root/<class_name>/<image>.png`` (standard ``ImageFolder`` layout). For
  self-supervised pretraining the labels are ignored.

* **HDF5 tiles** -- ``.h5`` files containing a ``tiles`` (or ``imgs``) dataset
  and optional ``coords``. Useful when patches were pre-extracted from WSIs.

* **WSI classification** -- one feature file per slide (produced by
  ``ts-ssl extract``) plus a CSV mapping ``slide_id`` -> ``label``. Each slide
  is a "bag" of patch feature vectors.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# --------------------------------------------------------------------------- #
# Patch-level datasets
# --------------------------------------------------------------------------- #
class PatchFolder(Dataset):
    """Loads labelled patches from an ``ImageFolder``-style directory.

    Set ``return_label=False`` to use it for self-supervised pretraining.
    """

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 return_label: bool = True) -> None:
        from PIL import Image  # local import keeps import-time light

        self._Image = Image
        self.root = root
        self.transform = transform
        self.return_label = return_label

        self.classes = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[str, int]] = []
        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(IMG_EXTENSIONS):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))
        if not self.samples:
            raise RuntimeError(f"No images found under {root!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = self._Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.return_label:
            return img, label
        return img


class HDF5TileDataset(Dataset):
    """Reads image tiles (and optional coords) from a single ``.h5`` file."""

    def __init__(self, h5_path: str, transform: Optional[Callable] = None,
                 tiles_key: str = "tiles", coords_key: str = "coords") -> None:
        self.h5_path = h5_path
        self.transform = transform
        self.tiles_key = tiles_key
        self.coords_key = coords_key
        with h5py.File(h5_path, "r") as f:
            key = tiles_key if tiles_key in f else "imgs"
            self.tiles_key = key
            self.length = f[key].shape[0]
            self.has_coords = coords_key in f

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        with h5py.File(self.h5_path, "r") as f:
            tile = np.asarray(f[self.tiles_key][idx])
            coord = np.asarray(f[self.coords_key][idx]) if self.has_coords else np.array([0, 0])
        tile = torch.from_numpy(tile).float()
        if tile.ndim == 3 and tile.shape[-1] in (1, 3):  # HWC -> CHW
            tile = tile.permute(2, 0, 1)
        if tile.max() > 1.5:  # assume 0-255
            tile = tile / 255.0
        if self.transform is not None:
            tile = self.transform(tile)
        return tile, torch.from_numpy(coord)


# --------------------------------------------------------------------------- #
# Slide-level (bag) dataset
# --------------------------------------------------------------------------- #
class WSIFeatureBags(Dataset):
    """One bag of patch features per slide for attention-MIL training.

    Args:
        feature_dir: Directory of ``<slide_id>.h5`` feature files, each with a
            ``features`` dataset of shape ``(n_patches, feature_dim)``.
        label_csv: CSV with columns ``slide_id,label`` (label may be a string
            or an integer index).
    """

    def __init__(self, feature_dir: str, label_csv: str,
                 features_key: str = "features") -> None:
        import csv

        self.feature_dir = feature_dir
        self.features_key = features_key

        rows: List[Tuple[str, str]] = []
        with open(label_csv, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append((row["slide_id"], row["label"]))

        raw_labels = sorted({lbl for _, lbl in rows})
        # If labels already look like integers, keep them as-is.
        if all(lbl.lstrip("-").isdigit() for lbl in raw_labels):
            self.classes = [str(c) for c in sorted(int(l) for l in raw_labels)]
            self.label_to_idx = {c: int(c) for c in self.classes}
        else:
            self.classes = raw_labels
            self.label_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[str, int]] = []
        for slide_id, lbl in rows:
            path = self._resolve(slide_id)
            if path is not None:
                self.samples.append((path, self.label_to_idx[lbl]))
        if not self.samples:
            raise RuntimeError("No feature files matched entries in the label CSV.")

    def _resolve(self, slide_id: str) -> Optional[str]:
        for cand in (slide_id, f"{slide_id}.h5"):
            path = os.path.join(self.feature_dir, cand)
            if os.path.isfile(path):
                return path
        return None

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with h5py.File(path, "r") as f:
            feats = np.asarray(f[self.features_key])
        return torch.from_numpy(feats).float(), torch.tensor(label, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Synthetic demo data (lets anyone run the full pipeline with no downloads)
# --------------------------------------------------------------------------- #
def make_synthetic_patches(root: str, n_per_class: int = 200, n_classes: int = 2,
                           size: int = 64, seed: int = 0) -> str:
    """Create a small ``ImageFolder`` of coloured-texture patches for demos."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    for c in range(n_classes):
        cls_dir = os.path.join(root, f"class_{c}")
        os.makedirs(cls_dir, exist_ok=True)
        base = rng.uniform(0.2, 0.8, size=3) * (c + 1) / n_classes
        for i in range(n_per_class):
            noise = rng.normal(0, 0.12, size=(size, size, 3))
            freq = 2 + 3 * c
            yy, xx = np.mgrid[0:size, 0:size]
            wave = 0.15 * np.sin(2 * np.pi * freq * xx / size)[..., None]
            img = np.clip(base[None, None, :] + wave + noise, 0, 1)
            Image.fromarray((img * 255).astype(np.uint8)).save(
                os.path.join(cls_dir, f"patch_{i:04d}.png")
            )
    return root


def make_synthetic_wsi_features(feature_dir: str, label_csv: str, n_slides: int = 40,
                                feature_dim: int = 128, n_classes: int = 2,
                                patches_range: Tuple[int, int] = (30, 80),
                                seed: int = 0) -> Tuple[str, str]:
    """Create per-slide feature bags + a label CSV for demoing WSI MIL training."""
    import csv

    rng = np.random.default_rng(seed)
    os.makedirs(feature_dir, exist_ok=True)
    rows = []
    for s in range(n_slides):
        label = s % n_classes
        n_patches = int(rng.integers(*patches_range))
        # Class signal lives in a subset of "tumour" patches; the rest is noise.
        feats = rng.normal(0, 1, size=(n_patches, feature_dim)).astype(np.float32)
        n_signal = max(1, n_patches // 5)
        feats[:n_signal, : feature_dim // 4] += (label + 1) * 1.5
        rng.shuffle(feats)
        slide_id = f"slide_{s:03d}"
        with h5py.File(os.path.join(feature_dir, f"{slide_id}.h5"), "w") as f:
            f.create_dataset("features", data=feats)
        rows.append((slide_id, label))

    with open(label_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["slide_id", "label"])
        writer.writerows(rows)
    return feature_dir, label_csv
