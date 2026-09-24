"""
SARCIModel: the full network, mirroring the official repo's BaseModel
(layers/MODEL_MAIN.py) exactly.

    image_feature = vision_encoder(img)          # (Bx, 512) raw
    audio_feature = audio_encoder(audio)         # (By, 512) raw
    v, a = cross_learning(image_feature, audio_feature)   # (Bx, By, 512) each
    scores = cosine_similarity(v, a)             # (Bx, By)

forward() returns the full similarity matrix directly -- see
cross_learning.py for why this is a single fused call, not two
independent embeddings compared afterward.

Two audio encoders are available, picked via `audio_backbone`:
    "cnn14"         -- audio_encoder.py, deliberate substitution for this
                       project's environmental audio (32 kHz input).
    "voice_feature" -- voice_feature.py, the paper's own E-ECAPA-TDNN,
                       for reproducing the paper's reported numbers on real
                       paired speech-caption audio (16 kHz input -- see
                       VoiceFeature.SAMPLE_RATE).
"""

import torch.nn as nn

from .vision_encoder import VisionEncoder
from .audio_encoder import AudioEncoder
from .voice_feature import VoiceFeature
from .cross_learning import CrossLearning, cosine_similarity


class SARCIModel(nn.Module):
    def __init__(self, cnn14_cls=None, embed_dim: int = 512,
                 freeze_vision_backbone: bool = True, freeze_audio_backbone: bool = True,
                 audio_backbone: str = "cnn14", with_decoder: bool = False):
        super().__init__()
        assert embed_dim == VisionEncoder.OUTPUT_DIM, (
            f"embed_dim must equal VisionEncoder.OUTPUT_DIM ({VisionEncoder.OUTPUT_DIM}); "
            "the vision branch has no projection layer (see vision_encoder.py)."
        )
        self.vision_encoder = VisionEncoder(freeze_backbone=freeze_vision_backbone, with_decoder=with_decoder)

        self.audio_backbone = audio_backbone
        if audio_backbone == "cnn14":
            assert cnn14_cls is not None, "cnn14_cls is required when audio_backbone='cnn14'"
            self.audio_encoder = AudioEncoder(cnn14_cls, embedding_dim=embed_dim,
                                               freeze_backbone=freeze_audio_backbone)
        elif audio_backbone == "voice_feature":
            self.audio_encoder = VoiceFeature(embed_dim=embed_dim)
        else:
            raise ValueError(f"unknown audio_backbone: {audio_backbone!r}")

        self.cross_learning = CrossLearning(embed_dim=embed_dim)
        self.Eiters = 0  # kept for parity with the official BaseModel (used by its training logger)

    def forward(self, img, audio):
        image_feature = self.vision_encoder(img)
        audio_feature = self.audio_encoder(audio)
        return self.similarity_from_features(image_feature, audio_feature)

    def encode_vision(self, img, reconstruct: bool = False):
        """Backbone-only pass. Useful to cache features once and reuse them
        across many cross_learning calls (see engine/evaluator.py) -- the
        official repo's own shard_dis re-runs the whole model per shard,
        which is correct but recomputes the frozen backbones needlessly."""
        return self.vision_encoder(img, reconstruct=reconstruct)

    def encode_audio(self, audio):
        return self.audio_encoder(audio)

    def similarity_from_features(self, image_feature, audio_feature):
        v, a = self.cross_learning(image_feature, audio_feature)
        return cosine_similarity(v, a)
