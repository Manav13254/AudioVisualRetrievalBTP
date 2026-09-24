"""
Cross-modal interaction module.

This is a faithful port of `CrossLearning` / `Cross_Attention` from the
official SARCI repository (WUTCM-Lab/SARCI, layers/Modules.py), verified
against the live source on GitHub. It deliberately does NOT follow the
paper's prose (eq. 19-28, "ICLM" with Vlocal/Vglobal/Afine/Aiteration) --
that description does not match what the released code actually does, so
we mirror the code, not the prose, per instruction.

Key behavioural difference from a normal dual encoder: given a batch of
Bx images and By audio clips, CrossLearning broadcasts both to a (Bx, By)
grid and returns cross-learned features for *every* image-audio pair in
one call. cosine_similarity() then reduces that to a (Bx, By) similarity
matrix directly -- the model's forward pass IS the similarity matrix, not
a pair of independent embeddings compared afterwards.

Note on Cross_Attention's own broadcast construction (kept intact from the
original code, not "fixed"): because x_1 is expanded (repeated, not varied)
along the sequence dimension used for keys/values, and y_1 likewise, the
attention weights inside each Cross_Attention call end up uniform (all
keys/values for a given row are identical), so cross_attention1 reduces to
a per-image linear transform and cross_attention2 to a per-audio linear
transform. The genuine cross-modal coupling happens afterwards, through the
sigmoid gates in x_final/y_final, which do mix both indices. This is an
accurate description of the shipped behaviour, not a bug we're introducing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Verbatim port of Cross_Attention from layers/Modules.py.

    Note self.scale = dim ** -0.5 (scaled by the full embedding dim, not
    dim_head) -- this matches the original source exactly.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, m, mask=None):
        b, n, _ = x.shape
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        def split_heads(t):
            return t.view(b, n, h, -1).transpose(1, 2)  # b h n d

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim=-1) if self.softmax else dots
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class CrossLearning(nn.Module):
    """Verbatim port of CrossLearning from layers/Modules.py."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attention1 = CrossAttention(dim=embed_dim, heads=8, dim_head=32)
        self.cross_attention2 = CrossAttention(dim=embed_dim, heads=8, dim_head=32)

    def forward(self, x, y):
        """x: (Bx, D) image features, y: (By, D) audio features.

        Returns x_final, y_final, each of shape (Bx, By, D).
        """
        batch_x = x.size(0)
        batch_y = y.size(0)

        x_1 = x.unsqueeze(1).expand(-1, batch_y, -1)
        y_1 = y.unsqueeze(0).expand(batch_x, -1, -1)
        x_2 = x.unsqueeze(0).expand(batch_y, -1, -1)
        y_2 = y.unsqueeze(1).expand(-1, batch_x, -1)

        x_cross = self.cross_attention1(y_1, x_1)
        y_cross = self.cross_attention2(x_2, y_2).transpose(0, 1)

        x_final = x_1 + torch.sigmoid(y_cross) * x_cross + torch.sigmoid(y_1) * x_cross
        y_final = y_1 + torch.sigmoid(y_cross) * y_1 + torch.sigmoid(x_cross) * y_1
        return x_final, y_final


def cosine_similarity(x1, x2, dim=-1, eps=1e-8):
    """Verbatim port of the module-level cosine_similarity in Modules.py."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
