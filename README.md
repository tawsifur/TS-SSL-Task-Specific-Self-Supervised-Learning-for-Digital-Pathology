# TS-SSL: Task-Specific Self-Supervised Learning for Digital Pathology

[![CI](https://github.com/tawsifur/TS-SSL/actions/workflows/ci.yml/badge.svg)](https://github.com/tawsifur/TS-SSL/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Modern%20Pathology-2025-b31b1b.svg)](https://doi.org/10.1016/j.modpat.2024.100636)

Reference implementation of **TS-SSL**, a task-specific self-supervised learning framework
for whole-slide image (WSI) analysis. Instead of relying on encoders pretrained *outside*
the histopathology domain (ImageNet) or on generic pathology corpora, TS-SSL pretrains a
lightweight **Spatial-Channel Attention Autoencoder (scAE)** on a *subsample of the very
data* it will later analyze — closing both the domain gap and the task-specific knowledge gap.

> Rahman T., Baras A. S., Chellappa R.
> *Evaluation of a Task-Specific Self-Supervised Learning Framework in Digital Pathology
> Relative to Transfer Learning Approaches and Existing Foundation Models.*
> **Modern Pathology** 38 (2025) 100636. https://doi.org/10.1016/j.modpat.2024.100636

---

## Why TS-SSL?

| Pipeline | Domain gap | Task gap | Params |
| --- | --- | --- | --- |
| Frozen ImageNet encoder | ❌ present | ❌ present | large |
| Frozen pathology foundation model | ✅ closed | ⚠️ often unclear | very large |
| **TS-SSL (this repo)** | ✅ closed | ✅ closed | **up to 3.2× fewer than ResNet-50** |

The scAE is trained to reconstruct patches, then **frozen** and used as a feature extractor.
Because pretraining uses a subsample (as little as **1–10%**) of the target dataset, the
learned features are tailored to the exact downstream task while remaining cheap to compute
(**~4× faster inference** than a ResNet-50 patch encoder in the paper's benchmarks).

## Highlights

- 🧠 **scAE backbone** — non-local *spatial attention* + squeeze-and-excitation *channel
  attention* inside a symmetric convolutional autoencoder.
- 🔬 **Two downstream tasks** — patch classification (MLP head) and weakly-supervised WSI
  classification (attention-based Multiple-Instance Learning).
- ⚡ **One unified CLI** — `ts-ssl pretrain | extract | train-patch | train-wsi | visualize`.
- 🚀 **Runs out of the box** — `ts-ssl demo` executes the entire pipeline on synthetic data,
  no downloads required.
- 📊 **Built-in analysis** — t-SNE embeddings plus Silhouette (SC) and Davies–Bouldin (DBI)
  clustering metrics, and reconstruction grids.

---

## Method at a glance

```
                     ┌─────────────── scAE (self-supervised) ───────────────┐
  patch subsample →  │  ConvBlock×2 → [Spatial Attn ⊕ Channel Attn] → Fuse   │ → embedding → Decoder → reconstruction
  (1–50% of data)    └───────────────────────┬──────────────────────────────┘
                                  freeze encoder │  (knowledge transfer)
                                                 ▼
  full dataset  →  frozen Encoder  →  embeddings ─┬─► MLP head              → patch label
                                                  └─► Attention-MIL pooling → slide label
```

- **Spatial attention** — embedded-Gaussian non-local block, `α = Softmax(K·Qᵀ)·V`, capturing
  long-range context between regions (`ts_ssl/models/attention.py`).
- **Channel attention** — global-average-pool → 2 FC layers → sigmoid gate,
  `F' = M ⊙ σ(Φ(M))`.
- **Reconstruction objective** — `L = ‖X − D(E(X))‖²` (`SCAE.reconstruction_loss`).
- **Attention-MIL** — `M = Σ aₖ·hₖ` with softmax attention over instances; standard and
  gated variants (`ts_ssl/models/heads.py`).

---

## Installation

```bash
git clone https://github.com/tawsifur/TS-SSL.git
cd TS-SSL

# (recommended) create an environment
python -m venv .venv && source .venv/bin/activate

# install the package + the `ts-ssl` CLI
pip install -e .
```

For a GPU build of PyTorch, install `torch`/`torchvision` from the
[official selector](https://pytorch.org/get-started/locally/) first, then `pip install -e .`.

---

## Quickstart — run the whole pipeline in one command

No data needed. This generates synthetic patches + feature bags and runs all five stages:

```bash
ts-ssl demo --workdir demo_run
```

You'll get, under `demo_run/`:

```
checkpoints/scae.pt         # pretrained autoencoder
checkpoints/patch_head.pt   # patch classifier
checkpoints/wsi_mil.pt      # slide-level MIL classifier
viz/tsne.png                # feature embedding + SC/DBI scores
viz/reconstructions.png     # input vs. scAE reconstruction
```

Force CPU / pick a device with the global flag: `ts-ssl --device cpu demo`.

---

## Full workflow on your own data

### 1. Data layout

**Patches** (for pretraining + patch classification) use the standard `ImageFolder` layout:

```
patches/
├── class_0/  img000.png  img001.png  ...
└── class_1/  img000.png  img001.png  ...
```

**WSI classification** uses one feature file per slide plus a label CSV:

```
wsi_features/  slide_000.h5  slide_001.h5   ...   # each has a `features` (N_patches × D) dataset
labels.csv                                        # columns: slide_id,label
```

### 2. Pretrain the scAE (self-supervised)

```bash
ts-ssl pretrain \
  --data patches/ \
  --out checkpoints/scae.pt \
  --config configs/pretrain.yaml     # subsample=0.1, lr=0.003, bs=256 (paper defaults)
```

### 3. Extract features with the frozen encoder

```bash
ts-ssl extract \
  --checkpoint checkpoints/scae.pt \
  --data patches/ \
  --out features/patches.h5
```

For WSIs, run `extract` once per slide (loop over your slide patch folders) so you end up
with one `.h5` per slide in `wsi_features/`.

### 4a. Patch classification (frozen encoder + MLP)

```bash
ts-ssl train-patch \
  --checkpoint checkpoints/scae.pt \
  --data patches/ \
  --out checkpoints/patch_head.pt \
  --config configs/patch_classification.yaml
# add --finetune to unfreeze the encoder and train end-to-end
```

### 4b. Weakly-supervised WSI classification (attention-MIL)

```bash
ts-ssl train-wsi \
  --features wsi_features/ \
  --labels labels.csv \
  --out checkpoints/wsi_mil.pt \
  --config configs/wsi_classification.yaml
# add --gated for the gated-attention variant
```

### 5. Visualize features & reconstructions

```bash
ts-ssl visualize \
  --checkpoint checkpoints/scae.pt \
  --data patches/ \
  --out viz/
```

<p align="center">
  <img src="assets/demo_tsne.png" width="46%" alt="t-SNE of TS-SSL features with SC/DBI">
  <img src="assets/demo_reconstructions.png" width="52%" alt="scAE reconstructions">
</p>

---

## Configuration

Hyperparameters can be passed as flags or via YAML files in [`configs/`](configs). Explicit
CLI flags always take precedence over a config value. The shipped defaults match the paper:

| Stage | epochs / steps | batch | lr | subsample |
| --- | --- | --- | --- | --- |
| scAE pretraining | ~1500 steps | 256 | 0.003 | 1–50% |
| Patch classification | ~5000 steps | 128 | 0.001 | — |
| WSI classification | ~2500 steps | (1 bag) | 0.0005 | — |

---

## Reported results (from the paper)

**Patch classification — Top-1 accuracy (%)**

| Encoder | PANDA | CRC | PCam | CIFAR-10 |
| --- | --- | --- | --- | --- |
| ResNet-50 (ImageNet) | 83.56 | 60.50 | 76.35 | 64.97 |
| CONCH (foundation) | 90.88 | 82.56 | 88.56 | — |
| **TS-SSL (ours)** | **92.11** | **86.60** | **89.51** | **68.22** |

**Weakly-supervised WSI classification — Top-1 accuracy (%)**

| Encoder | TCGA-NSCLC | TCGA-BRCA | PANDA |
| --- | --- | --- | --- |
| ResNet-50 (ImageNet) | 81.09 | 78.89 | 94.15 |
| CONCH (foundation) | 85.76 | 84.32 | 96.65 |
| **TS-SSL (ours)** | **86.56** | **85.51** | **97.77** |

TS-SSL remains competitive using only **1–10%** of patches for pretraining, and uses up to
**3.2× fewer parameters** and **7.5× fewer FLOPs** than an end-to-end ResNet-50.

> The public datasets used are PANDA, CRC (Kather), PatchCamelyon, CIFAR-10, and TCGA
> (NSCLC / BRCA via the [GDC portal](https://gdc.cancer.gov)). This repo does not
> redistribute any data.

---

## Repository structure

```
TS-SSL/
├── ts_ssl/
│   ├── models/        # attention blocks, scAE, MLP + attention-MIL heads
│   ├── data/          # ImageFolder / HDF5 / WSI-bag datasets + synthetic generators
│   ├── engine/        # pretrain, patch/WSI training, feature extraction, visualization
│   ├── utils/         # metrics (acc/AUC/SC/DBI), seeding, checkpoints, config
│   └── cli.py         # `ts-ssl` entry point
├── configs/           # YAML presets matching paper hyperparameters
├── tests/             # fast CPU smoke tests
└── .github/workflows/ # CI
```

---

## Development

```bash
pip install -e ".[dev]"
pytest -q          # run smoke tests
ruff check .       # lint
```

## Citation

```bibtex
@article{rahman2025tsssl,
  title   = {Evaluation of a Task-Specific Self-Supervised Learning Framework in
             Digital Pathology Relative to Transfer Learning Approaches and Existing
             Foundation Models},
  author  = {Rahman, Tawsifur and Baras, Alexander S. and Chellappa, Rama},
  journal = {Modern Pathology},
  volume  = {38},
  pages   = {100636},
  year    = {2025},
  doi     = {10.1016/j.modpat.2024.100636}
}
```

## License

Released under the [MIT License](LICENSE). Not for clinical use.
