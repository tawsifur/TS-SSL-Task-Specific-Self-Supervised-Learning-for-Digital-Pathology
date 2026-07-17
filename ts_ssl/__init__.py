"""TS-SSL: Task-Specific Self-Supervised Learning for digital pathology.

Reference implementation of

    Rahman, Baras & Chellappa,
    "Evaluation of a Task-Specific Self-Supervised Learning Framework in
     Digital Pathology Relative to Transfer Learning Approaches and Existing
     Foundation Models", Modern Pathology 38 (2025) 100636.
"""

__version__ = "0.1.0"

from . import data, engine, models, utils  # noqa: F401

__all__ = ["models", "data", "engine", "utils", "__version__"]
