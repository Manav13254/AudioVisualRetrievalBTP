"""
Visual encoder: ImageNet-pretrained ResNet18 + QDMVR (Section III-B1, III-C).

Verified against the official repo's ExtractFeature (layers/Modules.py):
the pooled 512-d feature is returned raw -- no projection layer, no
L2-normalization. This means the "embedding dim" on the vision side is
fixed at 512 (ResNet18's layer4 width) rather than a configurable value;
the audio encoder must match that width for CrossLearning/cosine_similarity
to work (see audio_encoder.py, sarci.py).

Official default is `finetune=True` (backbone trainable). We keep the
backbone frozen by default (`freeze_backbone=True`) to match this
project's prior experiments (see "initial experiments/"), where an
unfrozen-backbone run was tried as a deliberate variant, not the default.
Set freeze_backbone=False to mirror the official repo's default exactly.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from .qdmvr import QDMVR
from .masking import PixelDecoder


class VisionEncoder(nn.Module):
    OUTPUT_DIM = 512  # fixed: matches ResNet18 layer4 width, no projection layer

    def __init__(self, freeze_backbone: bool = True, with_decoder: bool = False):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False

        # F1 is taken pre-maxpool (112x112) so five distinct spatial scales exist;
        # see docstring in qdmvr.py for the full F1..F5 shape table.
        self.stem_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool  # AdaptiveAvgPool2d((1, 1)), reused from the pretrained resnet

        self.qdmvr = QDMVR()

        # Optional MIM reconstruction head -- not part of the paper, see masking.py.
        self.decoder = PixelDecoder() if with_decoder else None

    def extract_features(self, x):
        """Returns (F1, F2, F3, F4, F5) as described in the paper (eq. 3)."""
        f1 = self.stem_conv(x)
        f2 = self.layer1(self.maxpool(f1))
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        return f1, f2, f3, f4, f5

    def forward(self, x, reconstruct: bool = False):
        f1, f2, f3, f4, f5 = self.extract_features(x)
        vg = self.qdmvr(f1, f2, f3, f4, f5)  # eq. 13-18
        pooled = torch.flatten(self.avgpool(vg), 1)  # (B, 512), raw -- no projection/normalize

        if reconstruct:
            assert self.decoder is not None, "VisionEncoder(with_decoder=True) required for reconstruct=True"
            return pooled, self.decoder(vg)
        return pooled
