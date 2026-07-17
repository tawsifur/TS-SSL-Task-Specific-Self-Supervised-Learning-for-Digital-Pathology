"""Visualisation utilities: t-SNE embeddings, reconstructions, clustering scores.

All plotting uses a non-interactive Matplotlib backend so it works on headless
machines / clusters.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ..models import SCAE  # noqa: E402
from ..utils import clustering_metrics  # noqa: E402


def plot_tsne(features: np.ndarray, labels: np.ndarray, out_path: str,
              title: str = "TS-SSL features (t-SNE)", perplexity: float = 30.0,
              seed: int = 0) -> str:
    """Project features to 2-D with t-SNE and colour points by label."""
    from sklearn.manifold import TSNE

    perplexity = min(perplexity, max(5, (len(features) - 1) / 3))
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca",
               random_state=seed).fit_transform(features)

    fig, ax = plt.subplots(figsize=(7, 6))
    classes = np.unique(labels)
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(classes):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.7,
                   color=cmap(i % 10), label=f"class {c}")

    metrics = clustering_metrics(features, n_clusters=len(classes))
    ax.set_title(f"{title}\nSC={metrics['silhouette']:.3f}  "
                 f"DBI={metrics['davies_bouldin']:.3f}")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(markerscale=2, fontsize=8, loc="best")
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] t-SNE -> {os.path.abspath(out_path)}  "
          f"(SC={metrics['silhouette']:.3f}, DBI={metrics['davies_bouldin']:.3f})")
    return out_path


@torch.no_grad()
def plot_reconstructions(scae: SCAE, images: torch.Tensor, out_path: str,
                         device: Optional[torch.device] = None, n: int = 8) -> str:
    """Save a grid comparing input patches to their scAE reconstructions."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scae = scae.to(device).eval()
    images = images[:n].to(device).float()
    recon = scae(images).cpu()
    images = images.cpu()

    n = images.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.4))
    for i in range(n):
        for row, data, name in ((0, images, "input"), (1, recon, "recon")):
            img = data[i].permute(1, 2, 0).clamp(0, 1).numpy()
            ax = axes[row, i] if n > 1 else axes[row]
            ax.imshow(img)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(name, rotation=0, labelpad=25, fontsize=11)
    fig.suptitle("Top: ground truth   Bottom: scAE reconstruction")
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] reconstructions -> {os.path.abspath(out_path)}")
    return out_path
