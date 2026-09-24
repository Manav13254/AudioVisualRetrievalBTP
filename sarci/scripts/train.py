"""
CLI training entrypoint. Replaces the main() in the old flat scripts
(rn18_cnn14_iclm.py, sarci_masked.py, etc. -- now under "initial experiments/").

Examples:
    # class-folder dataset (current ADVANCE_DATA_split setup)
    python -m sarci.scripts.train --dataset_mode class --data_root ~/ADVANCE_DATA_split

    # paper-style paired dataset (new database, once it has image<->audio manifests)
    python -m sarci.scripts.train --dataset_mode paired \\
        --manifest_dir /path/to/manifests --image_root /path/to/images --audio_root /path/to/audio
"""

import argparse
import os
import sys

# allow `python sarci/scripts/train.py` as well as `python -m sarci.scripts.train`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader

from sarci.configs import DEFAULTS
from sarci.models import SARCIModel
from sarci.data import (
    build_vision_transform,
    PairedAudioVisualDataset,
    ImageListDataset,
    AudioListDataset,
    BidirectionalDataset,
    ValVisionDataset,
    ValAudioDataset,
)
from sarci.engine import (
    train_one_epoch,
    train_enhanced_epoch,
    shard_similarity,
    retrieval_metrics_paired,
    retrieval_metrics_by_label,
)
from sarci.utils.audioset import load_cnn14_class
from sarci.utils.seed import set_seed
from sarci.utils.wandb_logger import WandbLogger


def build_args():
    parser = argparse.ArgumentParser()
    for key, value in DEFAULTS.items():
        arg_type = type(value) if value is not None else str
        if arg_type is bool:
            parser.add_argument(f"--{key}", type=lambda s: s.lower() in ("1", "true", "yes"), default=value)
        else:
            parser.add_argument(f"--{key}", type=arg_type, default=value)
    return parser.parse_args()


def main():
    args = build_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn14_cls = load_cnn14_class() if args.audio_backbone == "cnn14" else None
    model = SARCIModel(
        cnn14_cls,
        embed_dim=args.embed_dim,
        freeze_vision_backbone=args.freeze_vision_backbone,
        freeze_audio_backbone=args.freeze_audio_backbone,
        audio_backbone=args.audio_backbone,
        with_decoder=args.use_enhancements,
    ).to(device)

    if args.audio_backbone == "cnn14" and os.path.exists(args.audio_weights):
        print(f"Loading AudioSet Cnn14 weights: {args.audio_weights}")
        ckpt = torch.load(args.audio_weights, map_location=device, weights_only=True)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.audio_encoder.base_model.load_state_dict(state_dict, strict=False)
    elif args.audio_backbone == "voice_feature" and args.sample_rate != model.audio_encoder.SAMPLE_RATE:
        print(f"WARNING: audio_backbone='voice_feature' requires {model.audio_encoder.SAMPLE_RATE}Hz audio, "
              f"but --sample_rate={args.sample_rate}. Pass --sample_rate {model.audio_encoder.SAMPLE_RATE}.")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # Paper Section IV-B: lr starts at 2e-4 and decays ×0.7 every 20 epochs.
    # StepLR.step() is called once per epoch (after the optimizer step), so
    # epoch 1-20 use the initial lr, epoch 21-40 use lr*0.7, etc.
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )

    if args.dataset_mode == "class":
        data_root = os.path.expanduser(args.data_root)
        train_v, train_a = os.path.join(data_root, "train", "vision"), os.path.join(data_root, "train", "sound")
        val_v, val_a = os.path.join(data_root, "test", "vision"), os.path.join(data_root, "test", "sound")

        train_ds = BidirectionalDataset(
            train_v, train_a, transform=build_vision_transform("train"),
            sample_rate=args.sample_rate, max_audio_len=args.max_audio_len,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

        def evaluate():
            v_loader = DataLoader(ValVisionDataset(val_v, transform=build_vision_transform("eval")),
                                   batch_size=args.eval_batch_size, num_workers=4)
            a_loader = DataLoader(ValAudioDataset(val_a, sample_rate=args.sample_rate,
                                                   max_audio_len=args.max_audio_len),
                                   batch_size=args.eval_batch_size, num_workers=4)
            sim, v_labels, a_labels = shard_similarity(model, v_loader, a_loader, device, shard_size=args.shard_size)
            i2a = retrieval_metrics_by_label(sim, v_labels, a_labels)
            a2i = retrieval_metrics_by_label(sim.T, a_labels, v_labels)
            score = (i2a["R@5"] + a2i["R@5"]) / 2
            print(f"  I->A: {i2a}")
            print(f"  A->I: {a2i}")
            return i2a, a2i, score

    elif args.dataset_mode == "paired":
        assert args.manifest_dir and args.image_root and args.audio_root, (
            "paired mode requires --manifest_dir --image_root --audio_root"
        )
        train_ds = PairedAudioVisualDataset(
            args.manifest_dir, args.image_root, args.audio_root, split="train",
            image_transform=build_vision_transform("train"), audio_per_image=args.audio_per_image,
            sample_rate=args.sample_rate, max_audio_len=args.max_audio_len,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # "val" for model-selection during training; "test" is held out for a
        # final, single evaluation after training (see scripts/evaluate.py).
        val_ds = PairedAudioVisualDataset(
            args.manifest_dir, args.image_root, args.audio_root, split="val",
            image_transform=build_vision_transform("eval"), audio_per_image=args.audio_per_image,
            sample_rate=args.sample_rate, max_audio_len=args.max_audio_len,
        )

        def evaluate():
            # Chunked feature extraction (via shard_similarity) -- stacking the
            # whole split into one batch OOMs on small GPUs (VoiceFeature/Cnn14
            # are memory-heavy backbones).
            v_loader = DataLoader(
                ImageListDataset(val_ds.image_root, val_ds.images, val_ds.image_transform),
                batch_size=args.eval_batch_size, num_workers=2,
            )
            a_loader = DataLoader(
                AudioListDataset(val_ds.audio_root, val_ds.voices, args.sample_rate, args.max_audio_len),
                batch_size=args.eval_batch_size, num_workers=2,
            )
            sim, _, _ = shard_similarity(model, v_loader, a_loader, device, shard_size=args.shard_size)

            i2a, a2i, mr = retrieval_metrics_paired(sim, audio_per_image=args.audio_per_image)
            print(f"  I->A: {i2a}")
            print(f"  A->I: {a2i}")
            return i2a, a2i, mr

    else:
        raise ValueError(f"unknown dataset_mode: {args.dataset_mode}")

    run_name = args.wandb_run_name or os.path.splitext(os.path.basename(args.save_path))[0]
    wandb_logger = WandbLogger(enabled=args.use_wandb, project=args.wandb_project,
                                run_name=run_name, config=vars(args))

    best_score = 0.0
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    try:
        for epoch in range(args.max_epochs):
            if args.use_enhancements:
                train_metrics = train_enhanced_epoch(
                    model, train_loader, optimizer, device, epoch=epoch + 1,
                    mask_ratio=args.mask_ratio, lambda_recon=args.lambda_recon,
                    margin=args.margin, lambda_triplet=args.lambda_triplet,
                    contrastive_temperature=args.contrastive_temperature,
                    lambda_contrastive=args.lambda_contrastive,
                    lambda_hinge=args.lambda_hinge,
                    audio_augment=args.audio_augment,
                )
            else:
                train_metrics = train_one_epoch(model, train_loader, optimizer, device,
                                                 margin=args.margin, max_violation=args.max_violation,
                                                 epoch=epoch + 1)
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch + 1}: loss {train_metrics['loss']:.4f}  lr {current_lr:.2e}")

            print("Validating...")
            i2a, a2i, score = evaluate()
            print(f"  score: {score:.2f}")

            wandb_logger.log(
                {**{f"train/{k}": v for k, v in train_metrics.items()},
                 **{f"val/i2a_{k}": v for k, v in i2a.items()},
                 **{f"val/a2i_{k}": v for k, v in a2i.items()},
                 "val/score": score,
                 "train/lr": current_lr},
                step=epoch + 1,
            )

            scheduler.step()  # decay lr at end of epoch (paper: ×0.7 every 20 epochs)

            if score > best_score:
                best_score = score
                epochs_no_improve = 0
                torch.save(model.state_dict(), args.save_path)
                wandb_logger.watch_checkpoint(args.save_path)
                print(f"  Best model saved to {args.save_path}")
            else:
                epochs_no_improve += 1
                print(f"  No improvement for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= args.patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

        if args.use_wandb and args.tsne_at_end and args.dataset_mode == "paired":
            _log_final_tsne(model, args, device, wandb_logger, run_name)

    finally:
        wandb_logger.finish()


def _log_final_tsne(model, args, device, wandb_logger, run_name):
    """Reloads the best checkpoint (not the last epoch's weights, which may
    be worse under early stopping) and logs t-SNE plots -- see
    scripts/tsne_plot.py for what these show and why."""
    import tempfile
    from sarci.scripts.tsne_plot import extract_features, plot_modality_view, plot_paired_view

    model.load_state_dict(torch.load(args.save_path, map_location=device, weights_only=True))
    model.eval()

    test_ds = PairedAudioVisualDataset(
        args.manifest_dir, args.image_root, args.audio_root, split="test",
        image_transform=build_vision_transform("eval"), audio_per_image=args.audio_per_image,
        sample_rate=args.sample_rate, max_audio_len=args.max_audio_len,
    )
    print("Generating final t-SNE plots...")
    v_feats, a_feats = extract_features(model, test_ds, device, max_images=500)

    with tempfile.TemporaryDirectory() as tmp:
        modality_path = os.path.join(tmp, "modality.png")
        pairs_path = os.path.join(tmp, "pairs.png")
        plot_modality_view(v_feats, a_feats, modality_path, f"t-SNE: image vs audio ({run_name}, test)")
        plot_paired_view(v_feats, a_feats, 15, pairs_path, f"t-SNE: matched pairs ({run_name}, test)")
        wandb_logger.log_image("tsne/modality", modality_path)
        wandb_logger.log_image("tsne/pairs", pairs_path)


if __name__ == "__main__":
    main()
