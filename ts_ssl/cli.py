"""Command-line interface for the TS-SSL framework.

Run ``ts-ssl --help`` (or ``python -m ts_ssl.cli --help``) for the full list of
subcommands. The ``demo`` subcommand runs the entire pipeline on synthetic
data and needs no external downloads.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import torch

from .data import (
    PatchFolder,
    WSIFeatureBags,
    build_transform,
    make_synthetic_patches,
    make_synthetic_wsi_features,
)
from .engine import (
    extract_features,
    plot_reconstructions,
    plot_tsne,
    pretrain_scae,
    train_patch_classifier,
    train_wsi_classifier,
)
from .models import SCAE
from .utils import get_device, load_checkpoint, load_config, set_seed


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _split_dataset(dataset, val_frac: float, seed: int):
    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    return torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )


def _load_scae(ckpt_path: str, device) -> SCAE:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    model = SCAE(
        in_channels=meta.get("in_channels", 3),
        base=meta.get("base", 32),
        embed_dim=meta.get("embed_dim", 128),
    )
    load_checkpoint(ckpt_path, model)
    return model.to(device)


def _merge_config(args: argparse.Namespace) -> argparse.Namespace:
    """Override defaults with a YAML config, keeping explicit CLI flags."""
    if getattr(args, "config", None):
        cfg = load_config(args.config)
        for key, value in cfg.items():
            if hasattr(args, key) and getattr(args, key) == PARSER_DEFAULTS.get(key):
                setattr(args, key, value)
    return args


PARSER_DEFAULTS: dict = {}


# --------------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------------- #
def cmd_pretrain(args) -> None:
    set_seed(args.seed)
    device = get_device(args.device)
    tf = build_transform(args.image_size, train=True)
    dataset = PatchFolder(args.data, transform=tf, return_label=False)
    pretrain_scae(
        dataset, out_path=args.out, embed_dim=args.embed_dim, base=args.base,
        subsample=args.subsample, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, num_workers=args.workers, device=device, seed=args.seed,
    )


def cmd_extract(args) -> None:
    device = get_device(args.device)
    scae = _load_scae(args.checkpoint, device)
    tf = build_transform(args.image_size, train=False)
    dataset = PatchFolder(args.data, transform=tf, return_label=False)
    # PatchFolder returns just an image when return_label=False; wrap for coords.
    extract_features(scae.encoder, dataset, out_h5=args.out,
                     batch_size=args.batch_size, num_workers=args.workers,
                     device=device, store_coords=False)


def cmd_train_patch(args) -> None:
    set_seed(args.seed)
    device = get_device(args.device)
    scae = _load_scae(args.checkpoint, device)

    train_tf = build_transform(args.image_size, train=True)
    dataset = PatchFolder(args.data, transform=train_tf, return_label=True)
    train_set, val_set = _split_dataset(dataset, args.val_frac, args.seed)
    train_patch_classifier(
        scae.encoder, train_set, val_set, out_path=args.out,
        n_classes=len(dataset.classes), freeze_encoder=not args.finetune,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        num_workers=args.workers, device=device,
    )


def cmd_train_wsi(args) -> None:
    set_seed(args.seed)
    device = get_device(args.device)
    dataset = WSIFeatureBags(args.features, args.labels)
    train_set, val_set = _split_dataset(dataset, args.val_frac, args.seed)
    feats0, _ = dataset[0]
    train_wsi_classifier(
        train_set, val_set, out_path=args.out,
        feature_dim=feats0.shape[-1], n_classes=dataset.n_classes,
        gated=args.gated, epochs=args.epochs, lr=args.lr, device=device,
    )


def cmd_visualize(args) -> None:
    import h5py
    import numpy as np

    device = get_device(args.device)
    scae = _load_scae(args.checkpoint, device)
    tf = build_transform(args.image_size, train=False)
    dataset = PatchFolder(args.data, transform=tf, return_label=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    feats, labels, sample_imgs = [], [], None
    with torch.no_grad():
        for imgs, lbls in loader:
            if sample_imgs is None:
                sample_imgs = imgs[:8].clone()
            feats.append(scae.encoder.embed(imgs.to(device).float()).cpu().numpy())
            labels.append(np.asarray(lbls))
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)

    os.makedirs(args.out, exist_ok=True)
    plot_tsne(feats, labels, os.path.join(args.out, "tsne.png"))
    if sample_imgs is not None:
        plot_reconstructions(scae, sample_imgs, os.path.join(args.out, "reconstructions.png"),
                             device=device)


def cmd_demo(args) -> None:
    """End-to-end pipeline on synthetic data -- no downloads required."""
    set_seed(args.seed)
    device = get_device(args.device)
    root = args.workdir
    os.makedirs(root, exist_ok=True)
    patches = os.path.join(root, "patches")
    ckpt_dir = os.path.join(root, "checkpoints")
    viz_dir = os.path.join(root, "viz")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("\n=== [1/5] Generating synthetic patches ===")
    make_synthetic_patches(patches, n_per_class=args.n_per_class, n_classes=2,
                           size=args.image_size, seed=args.seed)

    print("\n=== [2/5] Pretraining scAE (self-supervised) ===")
    tf_train = build_transform(args.image_size, train=True)
    pre_set = PatchFolder(patches, transform=tf_train, return_label=False)
    scae = pretrain_scae(pre_set, out_path=os.path.join(ckpt_dir, "scae.pt"),
                         embed_dim=64, base=16, subsample=0.5, epochs=args.epochs,
                         batch_size=64, device=device, seed=args.seed, log_every=50,
                         num_workers=0)

    print("\n=== [3/5] Patch classification (frozen encoder + MLP) ===")
    cls_set = PatchFolder(patches, transform=tf_train, return_label=True)
    train_set, val_set = _split_dataset(cls_set, 0.2, args.seed)
    train_patch_classifier(scae.encoder, train_set, val_set,
                           out_path=os.path.join(ckpt_dir, "patch_head.pt"),
                           n_classes=2, epochs=args.epochs, batch_size=64,
                           device=device, num_workers=0)

    print("\n=== [4/5] WSI attention-MIL (synthetic feature bags) ===")
    feat_dir = os.path.join(root, "wsi_features")
    label_csv = os.path.join(root, "wsi_labels.csv")
    make_synthetic_wsi_features(feat_dir, label_csv, n_slides=40,
                                feature_dim=64, n_classes=2, seed=args.seed)
    bags = WSIFeatureBags(feat_dir, label_csv)
    tr, va = _split_dataset(bags, 0.25, args.seed)
    train_wsi_classifier(tr, va, out_path=os.path.join(ckpt_dir, "wsi_mil.pt"),
                         feature_dim=64, n_classes=2, epochs=max(20, args.epochs),
                         device=device)

    print("\n=== [5/5] Visualisation (t-SNE + reconstructions) ===")
    import numpy as np
    loader = torch.utils.data.DataLoader(cls_set, batch_size=128, shuffle=False)
    feats, labels, sample = [], [], None
    with torch.no_grad():
        for imgs, lbls in loader:
            if sample is None:
                sample = imgs[:8].clone()
            feats.append(scae.encoder.embed(imgs.to(device).float()).cpu().numpy())
            labels.append(np.asarray(lbls))
    plot_tsne(np.concatenate(feats), np.concatenate(labels),
              os.path.join(viz_dir, "tsne.png"))
    plot_reconstructions(scae, sample, os.path.join(viz_dir, "reconstructions.png"),
                         device=device)

    print(f"\nDemo complete. Artifacts written under: {os.path.abspath(root)}")


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ts-ssl",
        description="Task-Specific Self-Supervised Learning for digital pathology.",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    sub = p.add_subparsers(dest="command", required=True)

    # pretrain
    sp = sub.add_parser("pretrain", help="Self-supervised pretraining of the scAE.")
    sp.add_argument("--data", required=True, help="ImageFolder of patches.")
    sp.add_argument("--out", default="checkpoints/scae.pt")
    sp.add_argument("--config", default=None)
    sp.add_argument("--embed-dim", type=int, default=128, dest="embed_dim")
    sp.add_argument("--base", type=int, default=32)
    sp.add_argument("--subsample", type=float, default=0.1,
                    help="Fraction of patches used for pretraining (paper: 0.01-0.5).")
    sp.add_argument("--epochs", type=int, default=20)
    sp.add_argument("--batch-size", type=int, default=256, dest="batch_size")
    sp.add_argument("--lr", type=float, default=3e-3)
    sp.add_argument("--image-size", type=int, default=64, dest="image_size")
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_pretrain)

    # extract
    se = sub.add_parser("extract", help="Extract features with a frozen scAE encoder.")
    se.add_argument("--checkpoint", required=True)
    se.add_argument("--data", required=True, help="ImageFolder of patches.")
    se.add_argument("--out", default="features/features.h5")
    se.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    se.add_argument("--image-size", type=int, default=64, dest="image_size")
    se.add_argument("--workers", type=int, default=4)
    se.set_defaults(func=cmd_extract)

    # train-patch
    stp = sub.add_parser("train-patch", help="Patch classification (scAE encoder + MLP).")
    stp.add_argument("--checkpoint", required=True, help="Pretrained scAE checkpoint.")
    stp.add_argument("--data", required=True, help="Labelled ImageFolder.")
    stp.add_argument("--out", default="checkpoints/patch_head.pt")
    stp.add_argument("--finetune", action="store_true", help="Unfreeze the encoder.")
    stp.add_argument("--epochs", type=int, default=30)
    stp.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    stp.add_argument("--lr", type=float, default=1e-3)
    stp.add_argument("--val-frac", type=float, default=0.2, dest="val_frac")
    stp.add_argument("--image-size", type=int, default=64, dest="image_size")
    stp.add_argument("--workers", type=int, default=4)
    stp.add_argument("--seed", type=int, default=42)
    stp.set_defaults(func=cmd_train_patch)

    # train-wsi
    stw = sub.add_parser("train-wsi", help="Weakly-supervised WSI classification (attention-MIL).")
    stw.add_argument("--features", required=True, help="Dir of per-slide feature .h5 files.")
    stw.add_argument("--labels", required=True, help="CSV: slide_id,label.")
    stw.add_argument("--out", default="checkpoints/wsi_mil.pt")
    stw.add_argument("--gated", action="store_true", help="Use gated attention.")
    stw.add_argument("--epochs", type=int, default=50)
    stw.add_argument("--lr", type=float, default=5e-4)
    stw.add_argument("--val-frac", type=float, default=0.2, dest="val_frac")
    stw.add_argument("--seed", type=int, default=42)
    stw.set_defaults(func=cmd_train_wsi)

    # visualize
    sv = sub.add_parser("visualize", help="t-SNE + reconstruction visualisations.")
    sv.add_argument("--checkpoint", required=True)
    sv.add_argument("--data", required=True, help="Labelled ImageFolder.")
    sv.add_argument("--out", default="viz")
    sv.add_argument("--image-size", type=int, default=64, dest="image_size")
    sv.set_defaults(func=cmd_visualize)

    # demo
    sd = sub.add_parser("demo", help="Run the whole pipeline on synthetic data.")
    sd.add_argument("--workdir", default="demo_run")
    sd.add_argument("--epochs", type=int, default=5)
    sd.add_argument("--n-per-class", type=int, default=150, dest="n_per_class")
    sd.add_argument("--image-size", type=int, default=48, dest="image_size")
    sd.add_argument("--seed", type=int, default=42)
    sd.set_defaults(func=cmd_demo)

    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args = _merge_config(args)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
