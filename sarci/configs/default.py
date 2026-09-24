"""Default hyperparameters, documented in one place. scripts/train.py exposes
all of these as CLI flags; this module is just the single source of truth
for their defaults so the CLI help text and this file never drift apart.
"""

DEFAULTS = dict(
    # dataset selection: "paired" (paper-style, PairedAudioVisualDataset) or
    # "class" (ADVANCE-style folders, ClassPairedDataset)
    dataset_mode="class",

    # -- class-mode paths (ADVANCE_DATA_split layout) --
    data_root="~/ADVANCE_DATA_split",

    # -- paired-mode paths (paper-style manifests) --
    manifest_dir=None,
    image_root=None,
    audio_root=None,
    audio_per_image=5,

    # audio
    audio_backbone="cnn14",  # "cnn14" (this project's environmental audio) or
                              # "voice_feature" (paper's real E-ECAPA-TDNN, for
                              # reproducing paper numbers on real speech-caption audio)
    audio_weights="audioset_tagging_cnn/Cnn14_mAP=0.431.pth",  # only used when audio_backbone="cnn14"
    sample_rate=32000,       # cnn14 expects 32kHz; voice_feature requires 16kHz
                              # (VoiceFeature.SAMPLE_RATE) -- override this when
                              # switching audio_backbone to "voice_feature"
    max_audio_len=320000,  # 10s @ 32kHz

    # model
    embed_dim=512,  # must equal VisionEncoder.OUTPUT_DIM; see models/sarci.py
    freeze_vision_backbone=True,
    freeze_audio_backbone=True,

    # optimization
    lr=2e-4,          # paper Section IV-B: "initially set to 2e-4"
    batch_size=32,
    eval_batch_size=16,  # backbone forward-pass batch size during evaluation;
                          # kept small since VoiceFeature/Cnn14 are memory-heavy
                          # on small GPUs (this caused a 6GB-VRAM OOM at the
                          # default that evaluate() used to stack everything into
                          # one batch instead of chunking -- now fixed, but keep
                          # this conservative on constrained hardware)
    shard_size=32,        # cross_learning batch size when building the full
                          # similarity matrix (cheap; backbone features are
                          # already extracted by then)
    max_epochs=40,
    patience=10,
    margin=0.2,
    lr_step_size=20,   # StepLR: decay every N epochs (paper: 20)
    lr_gamma=0.7,      # StepLR: multiply lr by this factor (paper: 0.7)
    max_violation=False,  # False = sum over all in-batch negatives (paper eq. 29);
                          # True = hardest-negative variant

    # "improved" recipe, ported from Retrieval1/sarci_contrastive_triplet.py
    # (masking + reconstruction + mixed triplet/contrastive loss). Backbones
    # stay frozen as normal -- unfreezing/discriminative LR was intentionally
    # left out of this port, not yet enabled.
    use_enhancements=False,
    mask_ratio=0.20,
    lambda_recon=0.5,
    # lambda_triplet/lambda_contrastive/contrastive_temperature were 1.0/1.0/0.07
    # (the ADVANCE-tuned values) on the first UCM attempt and overfit fast --
    # train loss smoothly -> ~0 while val R@K peaked at epoch 16 then degraded.
    # Softened here (lower weights, higher/softer temperature) as part of the
    # overfitting fix, alongside audio_augment and a smaller batch_size.
    lambda_triplet=0.5,
    lambda_contrastive=0.5,
    contrastive_temperature=0.15,
    lambda_hinge=0.0,  # the plain baseline loss (train_one_epoch's calcul_loss,
                        # sum-mode); 0 by default here since use_enhancements'
                        # combo doesn't include it, but set >0 for ablation runs
                        # that isolate one component -- see engine/trainer.py
    audio_augment=True,  # mild waveform noise/gain/shift; see data/audio_augment.py

    # misc
    save_path="checkpoints/best_sarci.pth",
    seed=123,

    # experiment tracking -- logs every epoch's train/val metrics + final
    # t-SNE plots automatically, so results don't need to be copy-pasted out
    # of the terminal by hand. No-ops entirely (including the import) when
    # use_wandb=False. run_name defaults to the checkpoint's basename
    # (from save_path) when left unset, so runs are identifiable at a glance.
    use_wandb=False,
    wandb_project="sarci-ucm",
    wandb_run_name=None,
    tsne_at_end=True,  # log final t-SNE plots (see scripts/tsne_plot.py) to
                        # wandb when training finishes; only takes effect if
                        # use_wandb=True and dataset_mode="paired"
)
