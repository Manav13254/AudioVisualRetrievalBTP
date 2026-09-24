"""
Training loop -- mirrors the official repo's engine.train (engine.py):
for each batch, model(img, audio) returns the full similarity matrix
directly, and losses.hinge.calcul_loss (eq. 29 in code form) is applied to
it. Requires the image batch and audio batch to be aligned by index (image
i and audio i are a positive pair) -- both PairedAudioVisualDataset and
ClassPairedDataset satisfy this.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..losses import calcul_loss, InfoNCELoss
from ..models import apply_random_mask
from ..data import apply_audio_augment


def train_one_epoch(model, loader, optimizer, device, margin=0.2, max_violation=False, epoch=None):
    """Returns {"loss": avg_loss} -- a dict (not a bare float) so callers can
    treat this and train_enhanced_epoch's return value uniformly, e.g. when
    logging to wandb (see scripts/train.py)."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}" if epoch is not None else "Training")
    for images, audio, _ids in pbar:
        images = images.to(device)
        audio = audio.to(device)

        optimizer.zero_grad()
        scores = model(images, audio)  # (B, B) similarity matrix
        loss = calcul_loss(scores, images.size(0), margin, max_violation)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return {"loss": total_loss / max(len(loader), 1)}


def train_enhanced_epoch(model, loader, optimizer, device, epoch=None,
                          mask_ratio=0.0, lambda_recon=0.0,
                          margin=0.2, lambda_triplet=0.0,
                          contrastive_temperature=0.07, lambda_contrastive=0.0,
                          lambda_hinge=0.0, audio_augment=False):
    """Flexible combination of retrieval-loss terms, for ablating the
    "improved" recipe ported from Retrieval1/sarci_contrastive_triplet.py
    (image masking + reconstruction, a hardest-negative triplet, a
    symmetric InfoNCE contrastive term) against each other and against the
    plain baseline loss:

        loss = lambda_hinge * hinge (same eq. as train_one_epoch, sum-mode)
             + lambda_triplet * triplet (hinge, max_violation=True)
             + lambda_contrastive * contrastive (InfoNCE)
             + lambda_recon * recon_mse (only computed if mask_ratio > 0)

    All four terms are independently toggleable via their lambda (each
    defaults to 0, i.e. off) so you can isolate one component at a time --
    e.g. lambda_hinge=1 with everything else 0 reproduces train_one_epoch's
    loss exactly, useful as a sanity check that this function's plumbing
    doesn't itself change results when "nothing extra" is turned on.

    Requires a model built with SARCIModel(with_decoder=True) whenever
    lambda_recon > 0 (mask_ratio only matters together with lambda_recon --
    masking a batch you never reconstruct from is a no-op). Backbones are
    left however they were constructed (frozen by default) -- this function
    does not touch requires_grad or use discriminative LR groups.

    audio_augment=True applies mild waveform noise/gain/shift to the batch
    (see data/audio_augment.py) -- added after the first UCM run of the
    full recipe overfit fast (train loss to near-zero while val R@K peaked
    at epoch 16 then degraded), traced to the same TTS clips being reused
    byte-identical every epoch with zero variation, unlike the vision side
    which already gets random crop/rotation per epoch.
    """
    model.train()
    contrastive_loss_fn = InfoNCELoss(temperature=contrastive_temperature) if lambda_contrastive > 0 else None
    do_recon = lambda_recon > 0
    totals = {"loss": 0.0, "hinge": 0.0, "trip": 0.0, "nce": 0.0, "rec": 0.0}

    pbar = tqdm(loader, desc=f"Epoch {epoch}" if epoch is not None else "Training")
    for images, audio, _ids in pbar:
        images = images.to(device)
        audio = audio.to(device)
        if audio_augment:
            audio = apply_audio_augment(audio)
        batch_size = images.size(0)

        optimizer.zero_grad()

        if do_recon:
            masked_images = apply_random_mask(images, mask_ratio=mask_ratio) if mask_ratio > 0 else images
            v_feat, recon = model.encode_vision(masked_images, reconstruct=True)
            recon_loss = F.mse_loss(recon, images)
        else:
            v_feat = model.encode_vision(images)
            recon_loss = torch.zeros((), device=device)

        a_feat = model.encode_audio(audio)
        scores = model.similarity_from_features(v_feat, a_feat)

        # NOT divided by batch_size -- matches train_one_epoch's calcul_loss call
        # exactly (raw sum-mode, O(B^2) terms), so lambda_hinge=1 with everything
        # else 0 reproduces the plain baseline's actual training signal bit for
        # bit. (A batch_size division here previously made the gradient ~batch_size
        # times weaker than the real baseline uses at the same lr, and collapsed
        # a "masking-only" ablation run to near-random -- see conversation.)
        hinge = calcul_loss(scores, batch_size, margin, max_violation=False) if lambda_hinge > 0 \
            else torch.zeros((), device=device)
        # hardest-negative-vs-diagonal-positive triplet, mean-normalized so its
        # scale matches the other (already per-sample-mean) loss terms -- this
        # is intentionally different from hinge above: max_violation mode only
        # ever has O(B) nonzero terms (one hardest negative per row/column), so
        # a /batch_size normalization here is a reasonable per-row average, not
        # the same O(B) vs O(B^2) mismatch that affected the sum-mode hinge term.
        triplet = calcul_loss(scores, batch_size, margin, max_violation=True) / batch_size if lambda_triplet > 0 \
            else torch.zeros((), device=device)
        contrastive = contrastive_loss_fn(scores) if contrastive_loss_fn is not None \
            else torch.zeros((), device=device)

        loss = (lambda_hinge * hinge + lambda_triplet * triplet
                + lambda_contrastive * contrastive + lambda_recon * recon_loss)
        loss.backward()
        optimizer.step()

        totals["loss"] += loss.item()
        totals["hinge"] += hinge.item()
        totals["trip"] += triplet.item()
        totals["nce"] += contrastive.item()
        totals["rec"] += recon_loss.item()
        pbar.set_postfix(loss=f"{loss.item():.3f}", hinge=f"{hinge.item():.3f}", trip=f"{triplet.item():.3f}",
                          nce=f"{contrastive.item():.3f}", rec=f"{recon_loss.item():.3f}")

    n = max(len(loader), 1)
    return {k: v / n for k, v in totals.items()}
