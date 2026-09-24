"""
Attention primitives used inside SARCI.

- SQA: Symmetric Quaternion Attention, paper Section III-C1, Fig. 3, eq. (4)-(12).
  Used twice inside QDMVR to refine the low-level and high-level visual features.
- CoordinateAttention: the "coordinate rescale layer" of the paper's audio branch
  (Section III-B2), built on Coordinate Attention [Hou et al., CVPR 2021]. In the
  paper it sits inside E-ECAPA-TDNN; here it plays the same role on top of the
  Cnn14 audio backbone (see audio_encoder.py for why Cnn14 replaces ECAPA-TDNN).
"""

import torch
import torch.nn as nn


class SQA(nn.Module):
    """Symmetric Quaternion Attention (paper eq. 4-12).

    Four symmetric branches pool the input along height/width with both
    average and max pooling, squeeze each pair with a *shared*-weight 1x1
    conv (eq. 7-8), fuse each direction with a 7x7 conv (eq. 9-10), combine
    the two directions into a full spatial attention map (eq. 11), and
    finally gate the original input with a sigmoid (eq. 12).

    Unlike CBAM, channel and spatial cues are fused in parallel rather than
    sequentially, and both avg- and max-pooling are used in both directions.
    """

    def __init__(self, in_planes: int, ratio: int = 8):
        super().__init__()
        reduced = in_planes // ratio

        # eq. 7: shared squeeze conv for the height-direction (avg/max) branches
        self.fc_h = nn.Conv2d(in_planes, reduced, kernel_size=1, bias=False)
        self.bn_h = nn.BatchNorm2d(reduced)
        self.relu_h = nn.ReLU()
        self.conv_h_spatial = nn.Conv2d(2 * reduced, reduced, kernel_size=7, padding=3, bias=False)  # eq. 9

        # eq. 8: shared squeeze conv for the width-direction (avg/max) branches
        self.fc_w = nn.Conv2d(in_planes, reduced, kernel_size=1, bias=False)
        self.bn_w = nn.BatchNorm2d(reduced)
        self.relu_w = nn.ReLU()
        self.conv_w_spatial = nn.Conv2d(2 * reduced, reduced, kernel_size=7, padding=3, bias=False)  # eq. 10

        self.expand = nn.Conv2d(reduced, in_planes, kernel_size=1, bias=False)  # tau_1 in eq. 11
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # eq. 5: average/max pooling along width -> height profile (C, H, 1)
        f_h_avg = torch.mean(x, dim=3, keepdim=True)
        f_h_max, _ = torch.max(x, dim=3, keepdim=True)
        # eq. 6: average/max pooling along height -> width profile (C, 1, W)
        f_w_avg = torch.mean(x, dim=2, keepdim=True)
        f_w_max, _ = torch.max(x, dim=2, keepdim=True)

        # eq. 7
        f_h_avg = self.relu_h(self.bn_h(self.fc_h(f_h_avg)))
        f_h_max = self.relu_h(self.bn_h(self.fc_h(f_h_max)))
        # eq. 8
        f_w_avg = self.relu_w(self.bn_w(self.fc_w(f_w_avg)))
        f_w_max = self.relu_w(self.bn_w(self.fc_w(f_w_max)))

        # eq. 9-10
        f_h = self.conv_h_spatial(torch.cat([f_h_avg, f_h_max], dim=1))  # (B, C/r, H, 1)
        f_w = self.conv_w_spatial(torch.cat([f_w_avg, f_w_max], dim=1))  # (B, C/r, 1, W)

        # eq. 11: broadcast-multiply the two directions into a full (C/r, H, W) map, expand channels
        f_q = self.expand(f_w * f_h)
        # eq. 12
        return x * self.sigmoid(f_q)


class CoordinateAttention(nn.Module):
    """Coordinate-attention rescale layer used on the audio branch (Section III-B2).

    Reduces spatial(/temporal) information loss when squeezing along one axis
    by keeping height and width statistics separate before recombining them.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = channels // reduction
        self.conv_1x1 = nn.Conv2d(channels, reduced, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(reduced)
        self.relu = nn.ReLU()
        self.f_h = nn.Conv2d(reduced, channels, kernel_size=1, bias=False)
        self.f_w = nn.Conv2d(reduced, channels, kernel_size=1, bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)  # (B, C, 1, H)
        x_w = torch.mean(x, dim=2, keepdim=True)  # (B, C, 1, W)
        cat = self.relu(self.bn(self.conv_1x1(torch.cat([x_h, x_w], dim=3))))
        split_h, split_w = cat.split([h, w], dim=3)
        s_h = self.sigmoid_h(self.f_h(split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.f_w(split_w))
        return x * s_h.expand_as(x) * s_w.expand_as(x)
