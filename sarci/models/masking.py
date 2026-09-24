"""
Masked-image-modeling (MIM) side-objective: randomly zero out 20% of the
image in fixed patches, and train a decoder to reconstruct the original
from QDMVR's fused feature map. Ported from Retrieval1/sarci_contrastive_triplet.py
(the enhanced PixelDecoder there, with an extra Conv2d refinement layer per
upsample stage vs. the simpler decoder in initial experiments/sarci_masked.py).

Not part of the paper -- this is one of this project's own experimental
additions (see the results spreadsheet: "SARCI+(Image mask20%)...").
"""

import torch
import torch.nn as nn


def apply_random_mask(imgs: torch.Tensor, mask_ratio: float = 0.20, patch_size: int = 16) -> torch.Tensor:
    """Zeroes out mask_ratio of the image in patch_size x patch_size blocks, per-sample."""
    if mask_ratio <= 0.0:
        return imgs

    b, c, h, w = imgs.shape
    num_patches_h, num_patches_w = h // patch_size, w // patch_size
    total_patches = num_patches_h * num_patches_w
    num_masked = int(total_patches * mask_ratio)

    masked = imgs.clone()
    for i in range(b):
        mask_indices = torch.randperm(total_patches)[:num_masked]
        for idx in mask_indices:
            row = (idx // num_patches_w) * patch_size
            col = (idx % num_patches_w) * patch_size
            masked[i, :, row:row + patch_size, col:col + patch_size] = 0.0
    return masked


class PixelDecoder(nn.Module):
    """Upscales QDMVR's 512x7x7 fused feature map back to a 224x224x3 image.

    Enhanced version (matches Retrieval1/sarci_contrastive_triplet.py): each
    upsample stage gets an extra 3x3 refinement conv after the transposed
    conv, vs. the plainer decoder in the original sarci_masked.py experiment.
    """

    def __init__(self):
        super().__init__()

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.up1 = up_block(512, 256)
        self.up2 = up_block(256, 128)
        self.up3 = up_block(128, 64)
        self.up4 = up_block(64, 32)
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return x
