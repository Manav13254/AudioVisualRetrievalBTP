"""
t-SNE visualization of the learned embedding space, for diagnosing *why*
a config under/over-performs, not just by how much.

Plots the raw per-modality embeddings (VisionEncoder/audio-encoder output,
BEFORE CrossLearning) -- this is the actual shared 512-d space the model's
cosine similarity is computed from once fused, and the natural thing to
t-SNE: CrossLearning's own output is pairwise (Bx, By, D), relational to a
specific image-audio combination rather than a fixed per-item vector, so it
doesn't have a single embedding per sample to plot.

Two views, both saved to the same output directory:
  1. modality view: all sampled image points vs all sampled audio points,
     colored by modality. A well-aligned cross-modal space should show the
     two colors intermixed, not separated into two distinct blobs -- a
     visible "modality gap" here is a common failure mode in cross-modal
     retrieval and would explain weak R@K even with reasonable per-modality
     features.
  2. paired view: a small subset of image/audio pairs, each pair given its
     own color (image = circle, audio = triangle) with a connecting line.
     Short lines = the model places matched pairs close together; long/
     crossed lines = specific failures worth looking at individually.

Usage:
    python -m sarci.scripts.tsne_plot --dataset_mode paired \
        --manifest_dir data/UCM/pairs --image_root data/UCM/images \
        --audio_root data/UCM/audio --audio_backbone voice_feature \
        --sample_rate 16000 --max_audio_len 96000 \
        --checkpoint checkpoints/sarci_ucm_enhanced_v2.pth --use_enhancements true \
        --split test --out_dir tsne_plots/v2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")  # headless -- no display on a remote server
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from sarci.configs import DEFAULTS
from sarci.models import SARCIModel
from sarci.data import PairedAudioVisualDataset, ImageListDataset, AudioListDataset, build_vision_transform
from sarci.utils.audioset import load_cnn14_class


def build_args():
    parser = argparse.ArgumentParser()
    for key, value in DEFAULTS.items():
        arg_type = type(value) if value is not None else str
        if arg_type is bool:
            parser.add_argument(f"--{key}", type=lambda s: s.lower() in ("1", "true", "yes"), default=value)
        else:
            parser.add_argument(f"--{key}", type=arg_type, default=value)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_points", type=int, default=500,
                         help="cap on images sampled for the modality view (audio matches 1:1 via audio_per_image)")
    parser.add_argument("--num_pairs", type=int, default=15, help="number of pairs shown in the paired view")
    parser.add_argument("--out_dir", default="tsne_plots")
    return parser.parse_args()


@torch.no_grad()
def extract_features(model, ds, device, max_images):
    n_images = min(max_images, len(ds.images))
    idx = np.random.RandomState(123).choice(len(ds.images), size=n_images, replace=False)

    v_loader = DataLoader(ImageListDataset(ds.image_root, [ds.images[i] for i in idx], ds.image_transform),
                           batch_size=32)
    v_feats = []
    for x, _ in v_loader:
        v_feats.append(model.encode_vision(x.to(device)).cpu())
    v_feats = torch.cat(v_feats, dim=0).numpy()

    # one audio clip per sampled image (the first caption), for a clean 1:1 pairing
    audio_files = [ds.voices[i * ds.audio_per_image] for i in idx]
    a_loader = DataLoader(AudioListDataset(ds.audio_root, audio_files, ds.sample_rate, ds.max_audio_len),
                           batch_size=32)
    a_feats = []
    for x, _ in a_loader:
        a_feats.append(model.encode_audio(x.to(device)).cpu())
    a_feats = torch.cat(a_feats, dim=0).numpy()

    return v_feats, a_feats


def plot_modality_view(v_feats, a_feats, out_path, title):
    combined = np.concatenate([v_feats, a_feats], axis=0)
    coords = TSNE(n_components=2, init="pca", random_state=123, perplexity=min(30, len(combined) // 4)).fit_transform(combined)
    n = v_feats.shape[0]

    plt.figure(figsize=(8, 8))
    plt.scatter(coords[:n, 0], coords[:n, 1], c="#3b82f6", label="image", alpha=0.6, s=20)
    plt.scatter(coords[n:, 0], coords[n:, 1], c="#f97316", label="audio", alpha=0.6, s=20)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_paired_view(v_feats, a_feats, num_pairs, out_path, title):
    n = min(num_pairs, v_feats.shape[0])
    v_sub, a_sub = v_feats[:n], a_feats[:n]
    combined = np.concatenate([v_sub, a_sub], axis=0)
    coords = TSNE(n_components=2, init="pca", random_state=123, perplexity=min(15, len(combined) // 4)).fit_transform(combined)

    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(n)
    plt.figure(figsize=(8, 8))
    for i in range(n):
        vi, ai = coords[i], coords[n + i]
        color = cmap(i)
        plt.plot([vi[0], ai[0]], [vi[1], ai[1]], color=color, alpha=0.4, linewidth=1)
        plt.scatter(*vi, color=color, marker="o", s=80, edgecolors="black", linewidths=0.5)
        plt.scatter(*ai, color=color, marker="^", s=80, edgecolors="black", linewidths=0.5)
    plt.title(title + "\n(circle=image, triangle=audio, same color=matched pair)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = build_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    cnn14_cls = load_cnn14_class() if args.audio_backbone == "cnn14" else None
    model = SARCIModel(cnn14_cls, embed_dim=args.embed_dim, audio_backbone=args.audio_backbone,
                        with_decoder=args.use_enhancements).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    ds = PairedAudioVisualDataset(args.manifest_dir, args.image_root, args.audio_root, split=args.split,
                                   image_transform=build_vision_transform("eval"), audio_per_image=args.audio_per_image,
                                   sample_rate=args.sample_rate, max_audio_len=args.max_audio_len)

    print(f"Extracting features for {min(args.max_points, len(ds.images))} images ({args.split} split)...")
    v_feats, a_feats = extract_features(model, ds, device, args.max_points)

    tag = os.path.splitext(os.path.basename(args.checkpoint))[0]
    plot_modality_view(v_feats, a_feats, os.path.join(args.out_dir, f"{tag}_modality.png"),
                        f"t-SNE: image vs audio embeddings ({tag}, {args.split})")
    plot_paired_view(v_feats, a_feats, args.num_pairs, os.path.join(args.out_dir, f"{tag}_pairs.png"),
                      f"t-SNE: matched pairs ({tag}, {args.split})")
    print(f"Saved plots to {args.out_dir}/{tag}_modality.png and {args.out_dir}/{tag}_pairs.png")


if __name__ == "__main__":
    main()
