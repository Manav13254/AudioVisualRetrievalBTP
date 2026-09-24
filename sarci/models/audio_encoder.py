"""
Audio encoder.

The paper uses E-ECAPA-TDNN, an enhanced ECAPA-TDNN (a *speaker-verification*
network) fed with MFCCs, because its RS datasets (Sydney/UCM/RSICD) pair
images with spoken-word audio captions.

Our data is environmental/scene audio, not speech, so a speaker-verification
backbone is the wrong prior. We instead use Cnn14 from the AudioSet tagging
CNN (Kong et al.), pretrained for general environmental sound tagging on
AudioSet, as the audio backbone. This is a deliberate, dataset-motivated
substitution -- everything else in the paper's audio-branch design is kept:
    - a coordinate-attention "rescale" layer (Section III-B2) applied to the
      backbone's feature map before pooling, playing the same role as the
      paper's coordinate rescale layer inside E-ECAPA-TDNN / the official
      repo's VoiceFeature (which uses the same CA_Block, verified identical
      to sarci/models/attention.py:CoordinateAttention);
    - attentive statistics pooling (mean + weighted std) instead of plain
      average pooling, matching VoiceFeature's attentive statistics pooling.

Verified against the official VoiceFeature (layers/Modules.py): the final
projection is followed by BatchNorm1d, not L2-normalization -- fc6+bn6
there, mirrored here as project+bn. `embedding_dim` defaults to 512 to
match VisionEncoder.OUTPUT_DIM (fixed, unprojected) since CrossLearning
requires both modalities at the same width.
"""

import torch
import torch.nn as nn

from .attention import CoordinateAttention


class AudioEncoder(nn.Module):
    def __init__(self, cnn14_cls, embedding_dim: int = 512, freeze_backbone: bool = True,
                 sample_rate: int = 32000, window_size: int = 1024, hop_size: int = 320,
                 mel_bins: int = 64, fmin: int = 50, fmax: int = 14000, classes_num: int = 527):
        super().__init__()
        self.base_model = cnn14_cls(
            sample_rate=sample_rate, window_size=window_size, hop_size=hop_size,
            mel_bins=mel_bins, fmin=fmin, fmax=fmax, classes_num=classes_num,
        )

        if freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.coord_attn = CoordinateAttention(2048)
        self.attn_pool = nn.Sequential(
            nn.Linear(2048, 128), nn.Tanh(), nn.Linear(128, 1), nn.Softmax(dim=1)
        )
        self.project = nn.Linear(4096, embedding_dim)  # 4096 = mean (2048) concat std (2048)
        self.bn = nn.BatchNorm1d(embedding_dim)  # mirrors VoiceFeature's bn6

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)

        x = self.base_model.spectrogram_extractor(x)
        x = self.base_model.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.base_model.bn0(x)
        x = x.transpose(1, 3)

        x = self.base_model.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block6(x, pool_size=(1, 1), pool_type='avg')

        x = self.coord_attn(x)

        x = torch.mean(x, dim=3).transpose(1, 2)  # (B, T, C)
        w = self.attn_pool(x)                      # (B, T, 1) attention weights over time
        mu = torch.sum(x * w, dim=1)
        var = torch.sum(w * (x - mu.unsqueeze(1)) ** 2, dim=1)
        std = torch.sqrt(var.clamp(min=1e-5))

        out = torch.cat([mu, std], dim=1)
        return self.bn(self.project(out))  # raw (BatchNorm'd) -- no L2 normalize, matches VoiceFeature
