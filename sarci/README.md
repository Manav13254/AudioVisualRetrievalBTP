# sarci/ -- modular SARCI implementation

This package is a from-scratch, modular rebuild of the SARCI network, checked
line-by-line against the **official repository**, `WUTCM-Lab/SARCI` on GitHub
(paper: Chen et al., "Scale-Aware Adaptive Refinement and Cross-Interaction
for Remote Sensing Audio-Visual Cross-Modal Retrieval," IEEE TGRS 2024).

Important: **the paper's prose (eq. 19-28, describing an "ICLM" with
Vlocal/Vglobal/Afine/Aiteration) does not match what the released code
actually does.** Where the two disagree, this package follows the *code*
(`layers/Modules.py`, `layers/MODEL_MAIN.py`, `utils.py`, `engine.py`,
`data.py`, `train.py`), not the paper text -- see `models/cross_learning.py`
for the specifics.

## Folder guide

```
sarci/
  models/
    attention.py        SQA and CoordinateAttention -- verified identical to
                         the official QUATER_ATTENTION / CA_Block
    qdmvr.py             multiscale visual fusion (paper eq. 13-18), verified
                         identical to the official ExtractFeature's fusion logic
    vision_encoder.py    ResNet18 + QDMVR -> raw 512-d pooled feature
                         (no projection, no L2-norm -- matches official)
    audio_encoder.py     Cnn14 (NOT the paper's E-ECAPA-TDNN, see below) +
                         coordinate-attention rescale + attentive stats pooling
    voice_feature.py     the paper's REAL audio encoder (VoiceFeature:
                         MelSpectrogram@16kHz -> Bottle2neck x3 -> coordinate
                         attention -> attentive stats pooling), ported verbatim
                         from the official repo. Use this for paper reproduction
                         runs on real paired speech-caption audio.
    cross_learning.py    the real cross-modal module (CrossLearning /
                         Cross_Attention), ported verbatim from the official
                         repo -- NOT the paper's ICLM prose
    sarci.py             SARCIModel: wires the three pieces together,
                         forward(img, audio) -> full similarity matrix
  data/
    transforms.py        image preprocessing (matches official data.py)
    paired_dataset.py    paper-style dataset: explicit image<->audio manifests
    class_dataset.py     class-folder dataset (current ADVANCE_DATA_split setup)
  losses/
    hinge.py              calcul_loss, ported verbatim from official utils.py
                          (this IS the paper's eq. 29, in code form)
  engine/
    trainer.py            one training epoch
    evaluator.py           similarity-matrix construction + R@K/mR metrics,
                          both paper-style (fixed 5-per-image gallery layout)
                          and class-based (label-match) variants
  configs/
    default.py            single source of truth for all hyperparameter defaults
  scripts/
    train.py               CLI entrypoint
  utils/
    audioset.py            locates/imports Cnn14 from audioset_tagging_cnn/
    seed.py                 matches the official repo's seeding (123)
```

## Why Cnn14 instead of E-ECAPA-TDNN

The paper's audio backbone (E-ECAPA-TDNN, and the official repo's
`VoiceFeature`) is a **speaker-verification** architecture, built for the
paper's datasets where audio = spoken-word captions describing an RS image.
This project's audio is environmental/scene sound, not speech, so a
speaker-verification prior is the wrong fit. `audio_encoder.py` uses **Cnn14**
from the AudioSet tagging CNN instead (general-purpose environmental sound
classifier), while keeping everything else from the paper's audio-branch
design that still makes sense: a coordinate-attention rescale layer
(verified identical to `VoiceFeature`'s `CA_Block`) and attentive statistics
pooling. This is a deliberate, dataset-motivated substitution, not an
oversight -- kept exactly as it was in "initial experiments/rn18_cnn14_iclm.py".

## Verified-vs-paper-prose summary

| Piece | Paper prose | Official code | This package |
|---|---|---|---|
| Vision backbone | ResNet18, 5-scale QDMVR+SQA | matches prose almost exactly | ported verbatim, verified identical math |
| Vision output | implied embedding | raw 512-d, no projection/norm | matches code |
| Audio backbone | E-ECAPA-TDNN (speech) | full ECAPA-TDNN (`VoiceFeature`) | **Cnn14** for this project's data; **`VoiceFeature`** (ported verbatim, `voice_feature.py`) available for paper-reproduction runs |
| Cross-modal module | "ICLM": Vlocal/Vglobal/Afine/Aiteration | `CrossLearning`: batched cross-attention over the whole (Bx,By) grid in one call | ported verbatim from code |
| Similarity | cosine(V, A) per pair | `cosine_similarity` over the full (Bx,By) grid, one forward call | matches code |
| Loss | hinge sum over in-batch negatives (eq. 29) | `calcul_loss`: same, plus optional hardest-negative mode | ported verbatim (`losses/hinge.py`) |

## Dataset modes (switchable via `--dataset_mode`)

- **`class`** (default): current ADVANCE_DATA_split layout --
  `vision/<class>/*`, `sound/<class>/*.wav`. Positive = same class folder.
  No explicit image<->audio pairing exists, so evaluation uses class-label
  recall (`retrieval_metrics_by_label`), not the paper's fixed 5-per-image
  layout.
- **`paired`**: paper-style explicit pairs, for when the new database gives
  each image a fixed number of paired audio clips (`audio_per_image`, 5 in
  the paper's own datasets). Point `--manifest_dir` at a directory containing
  `train_images.txt` / `train_voices.txt` / `test_images.txt` /
  `test_voices.txt` (one path per line; voices file has `audio_per_image`
  lines per image, in image order -- same convention as the official
  `data.py`). Evaluation uses `retrieval_metrics_paired`, matching the
  official `acc_i2t2`/`acc_t2i2`.

When you get the new database, the fastest path is: write a small script
that walks its files and emits those four manifest .txt files, then run
with `--dataset_mode paired`. No model code needs to change.

## Running

```bash
# class-folder (current setup)
python -m sarci.scripts.train --dataset_mode class --data_root ~/ADVANCE_DATA_split

# paired (new database, once manifests exist)
python -m sarci.scripts.train --dataset_mode paired \
    --manifest_dir /path/to/manifests --image_root /path/to/images --audio_root /path/to/audio
```

All hyperparameters in `configs/default.py` are exposed as `--flag` CLI
overrides.

## Known environment gap (not a code issue)

In this environment, `torchaudio` fails to import (`OSError: [WinError 127]`,
a broken native DLL) and `torchlibrosa` (required by Cnn14) isn't installed.
Both pre-date this refactor -- the old flat scripts had the same
dependencies. The vision path and the cross-modal/loss path were smoke-tested
directly (forward + backward pass, correct shapes) and work correctly; the
audio path is architecturally identical to before and just needs those two
packages fixed/installed to run.

## What did NOT get touched

Per scope agreed with the user: `baselines/` (AMFMN, PVSE, VSE++, CLIP+CNN14)
and `initial experiments/` (the prior flat SARCI scripts, including the
image-masking/MIM variant and the unfrozen-backbone variant -- both real
experiment steps tracked in the results spreadsheet) were relocated but not
rewritten. They still run standalone exactly as before.
