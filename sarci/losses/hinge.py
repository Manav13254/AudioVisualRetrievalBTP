"""
Batch hinge ranking loss -- faithful port of utils.calcul_loss from the
official SARCI repository (this is the eq. 29 loss in code form).

Expects `scores`, a square (B, B) similarity matrix from SARCIModel.forward
for a batch where image i and audio i are the positive pair (i.e. the
image batch and audio batch must be the same size and aligned by index --
see data/paired_dataset.py). The diagonal holds positive-pair scores;
every off-diagonal entry in a row/column is a negative.
"""

import torch
import torch.nn as nn


def calcul_loss(scores: torch.Tensor, size: int, margin: float, max_violation: bool = False) -> torch.Tensor:
    diagonal = scores.diag().view(size, 1)
    d1 = diagonal.expand_as(scores)        # positive score, broadcast across each row
    d2 = diagonal.t().expand_as(scores)    # positive score, broadcast across each column

    # eq. 29, audio-retrieval term: compare each row's positive to every score in that row
    cost_s = (margin + scores - d1).clamp(min=0)
    # eq. 29, image-retrieval term: compare each column's positive to every score in that column
    cost_im = (margin + scores - d2).clamp(min=0)

    mask = torch.eye(scores.size(0), device=scores.device) > 0.5
    cost_s = cost_s.masked_fill(mask, 0)
    cost_im = cost_im.masked_fill(mask, 0)

    if max_violation:
        # hardest-negative variant (not the paper's default: sum over all in-batch negatives)
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

    return cost_s.sum() + cost_im.sum()


class HingeLoss(nn.Module):
    def __init__(self, margin: float = 0.2, max_violation: bool = False):
        super().__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return calcul_loss(scores, scores.size(0), self.margin, self.max_violation)
