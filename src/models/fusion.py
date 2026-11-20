"""Feature-level fusion of the shared text and image encoders.

The coursework comparison swaps only the text encoder (TextCNN or BiLSTM).
The image branch and fusion classifier therefore have one shared definition.
"""

import torch
from torch import nn

from src.models.image import ImageEncoder


class FusionClassifier(nn.Module):
    """Concatenate text/image representations, then classify with a small MLP."""

    def __init__(
        self,
        text_encoder: nn.Module,
        num_classes: int,
        image_encoder: nn.Module | None = None,
        hidden_dim: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder or ImageEncoder()
        self.hidden_dim = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.text_encoder.out_dim + self.image_encoder.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        text_features = self.text_encoder(batch["text"])
        image_features = self.image_encoder(batch["image"])
        fused_features = torch.cat([text_features, image_features], dim=1)
        return self.classifier(fused_features)
