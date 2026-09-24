"""
Symmetric InfoNCE / NT-Xent contrastive loss over a (B, B) similarity
matrix, diagonal = positive pairs.

This is the diagonal-adapted equivalent of Retrieval1/sarci_contrastive_triplet.py's
SupervisedContrastiveLoss: that version grouped positives by ADVANCE class
label (multiple positives per row possible); our UCM data has explicit
per-image pairing instead of class labels, giving exactly one positive per
row -- the special case where SupCon reduces to standard InfoNCE.

The matching hardest-negative triplet term from that same file
(CrossModalTripletLoss) doesn't need a new implementation: it's already
what losses.hinge.calcul_loss computes when max_violation=True (see that
module's docstring) -- just in cosine-similarity space rather than the
original's Euclidean cdist space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        logits = scores / self.temperature
        labels = torch.arange(scores.size(0), device=scores.device)
        loss_i2a = F.cross_entropy(logits, labels)
        loss_a2i = F.cross_entropy(logits.t(), labels)
        return (loss_i2a + loss_a2i) / 2
