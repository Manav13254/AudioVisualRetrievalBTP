"""
One-off data-prep script: converts dataset.json (UCM-Captions, Karpathy-style
JSON: 2100 images x 5 text sentences, 80/10/10 split) into the paired-manifest
format sarci/data/paired_dataset.py expects, with:
  - a fresh 60/20/20 train/val/test re-split (by image, fixed seed), matching
    SARCI's paper split ratio -- NOT the same split as dataset.json's own
    field, and NOT guaranteed to match the paper's actual (unpublished) split.
  - audio synthesized via offline Piper TTS from each caption's `raw` text,
    since the paper's real recorded UCM audio-visual dataset (Baidu Netdisk,
    see sarci/README.md) is not accessible to us. This makes any resulting
    numbers an approximation of the paper's Table I UCM row, not a true
    reproduction -- see sarci/README.md "Known limitations".

Usage:
    python -m sarci.scripts.prepare_ucm_paired \
        --dataset_json dataset.json --imgs_rar imgs.rar \
        --out_dir data/UCM --piper_voice tts_voices/en_US-lessac-medium.onnx
"""

import argparse
import json
import os
import random
import subprocess
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tqdm import tqdm


def extract_images(imgs_rar, out_dir):
    """Extract imgs.rar via the WinRAR console tool (UnRAR.exe)."""
    unrar_candidates = [
        r"C:\Program Files\WinRAR\UnRAR.exe",
        r"C:\Program Files (x86)\WinRAR\UnRAR.exe",
    ]
    unrar = next((p for p in unrar_candidates if os.path.exists(p)), None)
    if unrar is None:
        raise RuntimeError("UnRAR.exe not found; extract imgs.rar manually into out_dir/images/")

    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([unrar, "x", "-y", imgs_rar, out_dir + os.sep], check=True)


def resplit(images, train_frac=0.6, val_frac=0.2, seed=123):
    rng = random.Random(seed)
    shuffled = list(images)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def synthesize_split(voice, split_images, audio_dir, split_name):
    from piper import PiperVoice

    os.makedirs(audio_dir, exist_ok=True)
    image_lines, voice_lines = [], []

    for img in tqdm(split_images, desc=f"synthesizing {split_name}"):
        image_lines.append(img["filename"])
        sentences = sorted(img["sentences"], key=lambda s: s["sentid"])[:5]
        for i, sent in enumerate(sentences):
            wav_name = f"{img['imgid']}_{i}.wav"
            wav_path = os.path.join(audio_dir, wav_name)
            if not os.path.exists(wav_path):
                with wave.open(wav_path, "wb") as wf:
                    voice.synthesize_wav(sent["raw"], wf)
            voice_lines.append(wav_name)

    return image_lines, voice_lines


def write_manifest(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json", default="dataset.json")
    parser.add_argument("--imgs_rar", default="imgs.rar")
    parser.add_argument("--out_dir", default="data/UCM")
    parser.add_argument("--piper_voice", default="tts_voices/en_US-lessac-medium.onnx")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip_image_extract", action="store_true")
    args = parser.parse_args()

    with open(args.dataset_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = data["images"]
    print(f"Loaded {len(images)} images from {args.dataset_json} (dataset={data.get('dataset')})")

    image_dir = os.path.join(args.out_dir, "images")
    audio_dir = os.path.join(args.out_dir, "audio")
    manifest_dir = os.path.join(args.out_dir, "pairs")
    os.makedirs(manifest_dir, exist_ok=True)

    if not args.skip_image_extract:
        print("Extracting images...")
        extract_images(args.imgs_rar, image_dir)
        # UnRAR preserves the "imgs/" folder inside the archive; flatten it.
        nested = os.path.join(image_dir, "imgs")
        if os.path.isdir(nested):
            for fname in os.listdir(nested):
                os.replace(os.path.join(nested, fname), os.path.join(image_dir, fname))
            os.rmdir(nested)

    splits = resplit(images, seed=args.seed)
    for name, imgs in splits.items():
        print(f"{name}: {len(imgs)} images")

    from piper import PiperVoice
    voice = PiperVoice.load(args.piper_voice)

    for split_name, split_images in splits.items():
        image_lines, voice_lines = synthesize_split(voice, split_images, audio_dir, split_name)
        write_manifest(image_lines, os.path.join(manifest_dir, f"{split_name}_images.txt"))
        write_manifest(voice_lines, os.path.join(manifest_dir, f"{split_name}_voices.txt"))

    print(f"Done. manifests={manifest_dir} images={image_dir} audio={audio_dir}")


if __name__ == "__main__":
    main()
