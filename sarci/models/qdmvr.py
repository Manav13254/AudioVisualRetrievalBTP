"""
QDMVR: Quaternion-attention Dominated Multiscale Visual Refinement.

Paper Section III-C2, eq. (13)-(18). Takes the five ResNet feature maps
F1..F5 (increasingly coarse, F5 = highest level / smallest spatial size),
fuses F1..F4 into a low-level feature Fl, refines Fl and the high-level
feature Fh = F5 with SQA, and symmetrically gates them together into the
final multiscale visual feature Vg.

Feature map shapes for a ResNet18 backbone with a 224x224 input:
    F1 (post conv1+bn+relu, pre-maxpool):  (B,  64, 112, 112)
    F2 (layer1 output):                    (B,  64,  56,  56)
    F3 (layer2 output):                    (B, 128,  28,  28)
    F4 (layer3 output):                    (B, 256,  14,  14)
    F5 (layer4 output, "Fh"):              (B, 512,   7,   7)

tau_3 (3x3, stride 2, pad 1) and tau_7 (7x7, stride 4, pad 3) downsample a
feature map to line its spatial size up with the next one before
channel-wise concatenation, exactly as eq. (13)-(15) describe:

    F12 = [tau_3(F1), F2]                          # both become 56x56
    F34 = [tau_3(F3), F4]                           # both become 14x14
    Fl  = tau_3([tau_7(F12), F34])                  # both become 14x14,
                                                     # final tau_3 drops to 7x7
"""

import torch
import torch.nn as nn

from .attention import SQA


class QDMVR(nn.Module):
    def __init__(self, channels=(64, 64, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4, c5 = channels

        # eq. 13: F12 = [tau_3(F1), F2]
        self.tau3_f1 = nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1)
        # eq. 14: F34 = [tau_3(F3), F4]
        self.tau3_f3 = nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1)
        # eq. 15: Fl = tau_3([tau_7(F12), F34])
        self.tau7_f12 = nn.Conv2d(c1 + c2, c1 + c2, kernel_size=7, stride=4, padding=3)
        self.tau3_fuse = nn.Conv2d((c1 + c2) + (c3 + c4), c5, kernel_size=3, stride=2, padding=1)

        # eq. 17: SQA applied separately to the low-level and high-level features
        self.sqa_low = SQA(c5)
        self.sqa_high = SQA(c5)

    def forward(self, f1, f2, f3, f4, f5):
        f12 = torch.cat([self.tau3_f1(f1), f2], dim=1)
        f34 = torch.cat([self.tau3_f3(f3), f4], dim=1)
        fl = self.tau3_fuse(torch.cat([self.tau7_f12(f12), f34], dim=1))  # eq. 15
        fh = f5  # eq. 16

        fl_prime = self.sqa_low(fl)   # eq. 17
        fh_prime = self.sqa_high(fh)  # eq. 17

        # eq. 18: Fg = (sigmoid(Fl') * Fh') + (sigmoid(Fh') * Fl')
        fg = torch.sigmoid(fl_prime) * fh_prime + torch.sigmoid(fh_prime) * fl_prime
        # residual connection with the high-level feature (Section III-C2, after eq. 18)
        vg = fg + fh_prime
        return vg
