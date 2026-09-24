"""
Standalone final evaluation on the held-out "test" split, given a trained
checkpoint. Kept separate from scripts/train.py's per-epoch "val" evaluation
so the reported paper-comparison numbers come from a split the model never
influenced early-stopping decisions on.

Usage:
    python -m sarci.scripts.evaluate --dataset_mode paired \
        --manifest_dir data/UCM/manifests --image_root data/UCM/images \
        --audio_root data/UCM/audio --audio_backbone voice_feature \
        --sample_rate 16000 --checkpoint checkpoints/best_sarci.pth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from sarci.configs import DEFAULTS
from sarci.models import SARCIModel
from sarci.data import (
    build_vision_transform,
    PairedAudioVisualDataset,
    ImageListDataset,
    AudioListDataset,
    ValVisionDataset,
    ValAudioDataset,
)
from sarci.engine import shard_similarity, retrieval_metrics_paired, retrieval_metrics_by_label
from sarci.utils.audioset import load_cnn14_class
from torch.utils.data import DataLoader


def build_args():
    parser = argparse.ArgumentParser()
    for key, value in DEFAULTS.items():
        arg_type = type(value) if value is not None else str
        if arg_type is bool:
            parser.add_argument(f"--{key}", type=lambda s: s.lower() in ("1", "true", "yes"), default=value)
        else:
            parser.add_argument(f"--{key}", type=arg_type, default=value)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main():
    args = build_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn14_cls = load_cnn14_class() if args.audio_backbone == "cnn14" else None
    model = SARCIModel(
        cnn14_cls, embed_dim=args.embed_dim, audio_backbone=args.audio_backbone,
        with_decoder=args.use_enhancements,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    if args.dataset_mode == "paired":
        test_ds = PairedAudioVisualDataset(
            args.manifest_dir, args.image_root, args.audio_root, split="test",
            image_transform=build_vision_transform("eval"), audio_per_image=args.audio_per_image,
            sample_rate=args.sample_rate, max_audio_len=args.max_audio_len,
        )
        v_loader = DataLoader(
            ImageListDataset(test_ds.image_root, test_ds.images, test_ds.image_transform),
            batch_size=args.eval_batch_size, num_workers=2,
        )
        a_loader = DataLoader(
            AudioListDataset(test_ds.audio_root, test_ds.voices, args.sample_rate, args.max_audio_len),
            batch_size=args.eval_batch_size, num_workers=2,
        )
        sim, _, _ = shard_similarity(model, v_loader, a_loader, device, shard_size=args.shard_size)

        i2a, a2i, mr = retrieval_metrics_paired(sim, audio_per_image=args.audio_per_image)
        print("Image -> Audio:", i2a)
        print("Audio -> Image:", a2i)
        print(f"mR: {mr:.2f}")

    elif args.dataset_mode == "class":
        data_root = os.path.expanduser(args.data_root)
        val_v = os.path.join(data_root, "test", "vision")
        val_a = os.path.join(data_root, "test", "sound")
        v_loader = DataLoader(ValVisionDataset(val_v, transform=build_vision_transform("eval")),
                               batch_size=args.eval_batch_size)
        a_loader = DataLoader(ValAudioDataset(val_a, sample_rate=args.sample_rate, max_audio_len=args.max_audio_len),
                               batch_size=args.eval_batch_size)
        sim, v_labels, a_labels = shard_similarity(model, v_loader, a_loader, device, shard_size=args.shard_size)
        print("Image -> Audio:", retrieval_metrics_by_label(sim, v_labels, a_labels))
        print("Audio -> Image:", retrieval_metrics_by_label(sim.T, a_labels, v_labels))

    else:
        raise ValueError(f"unknown dataset_mode: {args.dataset_mode}")


if __name__ == "__main__":
    main()
