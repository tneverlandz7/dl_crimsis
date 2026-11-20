"""Image branch on cached ResNet18 features and the image-only classifier.

The encoder maps frozen 512-d features [B, 512] to [B, IMAGE_OUT_DIM]; the fusion model can reuse it unchanged:

    encoder = ImageEncoder()
    encoder.out_dim  # 256
"""

import torch
from torch import nn

from src import config

IMAGE_OUT_DIM = 256


class ImageEncoder(nn.Module):
    """One hidden layer over the frozen features: Linear -> ReLU -> Dropout."""

    def __init__(self, in_dim: int = config.IMAGE_FEATURE_DIM, out_dim: int = IMAGE_OUT_DIM, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))
        self.out_dim = out_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ImageClassifier(nn.Module):
    """Image-only model: encoder -> linear. Takes a batch dict with an "image" tensor."""

    def __init__(self, encoder: nn.Module, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.out_dim, num_classes)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.classifier(self.encoder(batch["image"]))
