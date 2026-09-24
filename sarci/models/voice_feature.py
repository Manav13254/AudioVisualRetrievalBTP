"""
VoiceFeature: the paper's real audio encoder (their E-ECAPA-TDNN), ported
verbatim from the official repo (layers/Modules.py: SEModule, Bottle2neck,
VoiceFeature). Use this -- not audio_encoder.py's Cnn14 -- when reproducing
the paper's own reported numbers on real paired speech-caption audio
(Sydney/UCM/RSICD image-sound datasets). Cnn14 remains the encoder for this
project's own environmental-audio data; the two are picked via config, not
merged, since they expect different sample rates and audio content.

Verbatim details worth flagging (not "fixed", kept exactly as shipped):
  - torchfbank runs at 16 kHz (MelSpectrogram sample_rate=16000) -- audio fed
    into this module must be resampled to 16 kHz, unlike Cnn14's 32 kHz path.
  - The frontend (torchfbank + conv1 + bn1) is wrapped in `with torch.no_grad()`
    in the original code, which means conv1/bn1 never receive gradients even
    though they're declared as ordinary trainable layers -- this looks like
    an artifact of adapting a pretrained-frontend recipe, but we replicate it
    exactly since the brief is to mirror the shipped code.
  - CA_Block here is the same coordinate-attention block already implemented
    as CoordinateAttention in attention.py (verified identical); reused
    directly rather than duplicated.
"""

import math

import torch
import torch.nn as nn
import torchaudio

from .attention import CoordinateAttention


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super().__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        num_pad = math.floor(kernel_size / 2) * dilation
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(self.nums)])
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.bn1(self.relu(self.conv1(x)))
        spx = torch.split(out, self.width, 1)
        sp = None
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.bns[i](self.relu(self.convs[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        out = self.bn3(self.relu(self.conv3(out)))
        out = self.se(out)
        return out + residual


class VoiceFeature(nn.Module):
    SAMPLE_RATE = 16000  # torchfbank is fixed at this rate; resample audio before feeding in

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.torchfbank = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160,
                f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80,
            ),
        )
        self.conv1 = nn.Conv1d(80, 1024, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(1024)

        c = 1024
        self.layer1 = Bottle2neck(c, c, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(c, c, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(c, c, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3 * c, 1536, kernel_size=1)

        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, embed_dim)
        self.bn6 = nn.BatchNorm1d(embed_dim)
        self.ca_att = CoordinateAttention(1536)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)

        with torch.no_grad():  # verbatim from the official repo -- see module docstring
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            x = self.conv1(x)
            x = self.relu(x)
            x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        x = self.relu(self.layer4(torch.cat((x1, x2, x3), dim=1)))

        x = x.unsqueeze(-1)          # (B, C, T, 1) for the 2D coordinate-attention block
        x = self.ca_att(x)
        x = x.squeeze(-1)            # back to (B, C, T)

        t = x.size(-1)
        global_x = torch.cat((
            x,
            torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
            torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t),
        ), dim=1)
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sigma = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat((mu, sigma), dim=1)
        x = self.bn5(x)
        x = self.fc6(x)
        return self.bn6(x)  # raw (BatchNorm'd), no L2-normalize -- matches official
