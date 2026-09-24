#!/bin/bash
# Evaluation Script for DGX Cluster

# ==========================================
# CONFIGURATION - Update these paths to match your DGX setup
# ==========================================
DATASET_MODE="paired"
MANIFEST_DIR="data/UCM/manifests"
IMAGE_ROOT="data/UCM/images"
AUDIO_ROOT="data/UCM/audio"

# Audio config
AUDIO_BACKBONE="voice_feature" # or "cnn14" depending on what was used
SAMPLE_RATE=16000 # 16000 for voice_feature, 32000 for cnn14

# Checkpoints config - update these with the actual paths to your .pth files on DGX
CKPT_BASELINE="checkpoints/baseline_corrected.pth"
CKPT_ABLATION1="checkpoints/ablation1_masking.pth"
CKPT_ABLATION2="checkpoints/ablation2_triplet.pth"
CKPT_ABLATION3="checkpoints/ablation3_trip_cont.pth"
CKPT_ABLATION4="checkpoints/ablation4_all_enhanced.pth"
# ==========================================

# Base arguments shared across all evaluations
BASE_ARGS="--dataset_mode $DATASET_MODE \
           --manifest_dir $MANIFEST_DIR \
           --image_root $IMAGE_ROOT \
           --audio_root $AUDIO_ROOT \
           --audio_backbone $AUDIO_BACKBONE \
           --sample_rate $SAMPLE_RATE"

echo "--------------------------------------------------------"
echo "1. Evaluating Baseline (Corrected)"
echo "--------------------------------------------------------"
if [ -f "$CKPT_BASELINE" ]; then
    python -m sarci.scripts.evaluate $BASE_ARGS --checkpoint "$CKPT_BASELINE"
else
    echo "Checkpoint not found: $CKPT_BASELINE"
fi
echo ""

# For all ablations, --use_enhancements 1 is required to match the model 
# architecture (which includes the decoder) during the `train_enhanced_epoch` training.

echo "--------------------------------------------------------"
echo "2. Evaluating Ablation 1: Masking"
echo "--------------------------------------------------------"
if [ -f "$CKPT_ABLATION1" ]; then
    python -m sarci.scripts.evaluate $BASE_ARGS --use_enhancements 1 --checkpoint "$CKPT_ABLATION1"
else
    echo "Checkpoint not found: $CKPT_ABLATION1"
fi
echo ""

echo "--------------------------------------------------------"
echo "3. Evaluating Ablation 2: Triplet"
echo "--------------------------------------------------------"
if [ -f "$CKPT_ABLATION2" ]; then
    python -m sarci.scripts.evaluate $BASE_ARGS --use_enhancements 1 --checkpoint "$CKPT_ABLATION2"
else
    echo "Checkpoint not found: $CKPT_ABLATION2"
fi
echo ""

echo "--------------------------------------------------------"
echo "4. Evaluating Ablation 3: Trip+Cont"
echo "--------------------------------------------------------"
if [ -f "$CKPT_ABLATION3" ]; then
    python -m sarci.scripts.evaluate $BASE_ARGS --use_enhancements 1 --checkpoint "$CKPT_ABLATION3"
else
    echo "Checkpoint not found: $CKPT_ABLATION3"
fi
echo ""

echo "--------------------------------------------------------"
echo "5. Evaluating Ablation 4: All Enhanced"
echo "--------------------------------------------------------"
if [ -f "$CKPT_ABLATION4" ]; then
    python -m sarci.scripts.evaluate $BASE_ARGS --use_enhancements 1 --checkpoint "$CKPT_ABLATION4"
else
    echo "Checkpoint not found: $CKPT_ABLATION4"
fi
echo ""

echo "All evaluations finished!"
