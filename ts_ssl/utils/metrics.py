"""Evaluation metrics used across TS-SSL tasks."""

from __future__ import annotations

from typing import Dict

import numpy as np


def top1_accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Top-1 classification accuracy."""
    preds = logits.argmax(axis=1)
    return float((preds == targets).mean())


def auc_score(logits: np.ndarray, targets: np.ndarray) -> float:
    """Area under the ROC curve (macro one-vs-rest for multi-class)."""
    from scipy.special import softmax
    from sklearn.metrics import roc_auc_score

    probs = softmax(logits, axis=1)
    n_classes = probs.shape[1]
    try:
        if n_classes == 2:
            return float(roc_auc_score(targets, probs[:, 1]))
        return float(roc_auc_score(targets, probs, multi_class="ovr", average="macro"))
    except ValueError:
        # Happens when a class is missing from a small eval split.
        return float("nan")


def classification_metrics(logits: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    return {"accuracy": top1_accuracy(logits, targets), "auc": auc_score(logits, targets)}


def clustering_metrics(features: np.ndarray, n_clusters: int, n_runs: int = 10,
                       seed: int = 0) -> Dict[str, float]:
    """K-means clustering quality via Silhouette (SC) and Davies-Bouldin (DBI).

    Averaged over ``n_runs`` random initialisations, matching the paper's
    protocol. Higher SC and lower DBI indicate better-separated clusters.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    scs, dbis = [], []
    for run in range(n_runs):
        km = KMeans(n_clusters=n_clusters, n_init=5, random_state=seed + run)
        labels = km.fit_predict(features)
        if len(np.unique(labels)) < 2:
            continue
        scs.append(silhouette_score(features, labels))
        dbis.append(davies_bouldin_score(features, labels))

    return {
        "silhouette": float(np.mean(scs)) if scs else float("nan"),
        "silhouette_std": float(np.std(scs)) if scs else float("nan"),
        "davies_bouldin": float(np.mean(dbis)) if dbis else float("nan"),
        "davies_bouldin_std": float(np.std(dbis)) if dbis else float("nan"),
    }
