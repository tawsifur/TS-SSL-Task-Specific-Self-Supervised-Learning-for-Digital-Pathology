"""Miscellaneous helpers: seeding, devices, checkpoints, config loading."""

from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    """Resolve the compute device.

    ``prefer`` may be ``"auto"``, ``"cuda"``, ``"mps"`` or ``"cpu"``.
    """
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(prefer)


def save_checkpoint(path: str, model: torch.nn.Module, meta: Dict[str, Any] | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)


def load_checkpoint(path: str, model: torch.nn.Module, strict: bool = True) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=strict)
    return ckpt.get("meta", {})


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh) or {}
