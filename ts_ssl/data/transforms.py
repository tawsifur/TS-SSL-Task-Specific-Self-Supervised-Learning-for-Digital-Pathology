"""Lightweight image transforms.

Kept dependency-free where possible so the repo runs even if torchvision is
absent; if torchvision is installed the standard transforms are used.
"""

from __future__ import annotations

from typing import Callable


def build_transform(image_size: int = 64, train: bool = True) -> Callable:
    """Return a transform mapping a PIL image to a normalised CHW tensor."""
    try:
        from torchvision import transforms
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "torchvision is required for image transforms. Install it with "
            "`pip install torchvision`."
        ) from exc

    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
