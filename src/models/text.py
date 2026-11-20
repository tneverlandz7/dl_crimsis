"""Text encoders (TextCNN, one-layer BiLSTM) and the text-only classifier.

Both encoders map token IDs [B, 64] to a fixed vector [B, TEXT_OUT_DIM], so they are interchangeable
inside any model (e.g. the fusion model's TextCNN <-> BiLSTM swap):

    encoder = build_text_encoder("textcnn", vocab_size=len(vocab), pad_id=vocab.pad_id)
    encoder.out_dim  # 300
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from src import config

TEXT_OUT_DIM = 300


class TextCNNEncoder(nn.Module):
    """Kim (2014)-style TextCNN: parallel 1-D convolutions + ReLU + max-over-time pooling."""

    def __init__(self, vocab_size: int, pad_id: int = 0, embed_dim: int = config.EMBED_DIM,
                 filter_sizes: tuple[int, ...] = (3, 4, 5), num_filters: int = TEXT_OUT_DIM // 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList(nn.Conv1d(embed_dim, num_filters, k) for k in filter_sizes)
        self.out_dim = num_filters * len(filter_sizes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens).transpose(1, 2)  # [B, E, L]
        return torch.cat([torch.relu(conv(x)).amax(dim=2) for conv in self.convs], dim=1)


class BiLSTMEncoder(nn.Module):
    """One-layer BiLSTM over the unpadded tokens; output = [last forward state; last backward state]."""

    def __init__(self, vocab_size: int, pad_id: int = 0, embed_dim: int = config.EMBED_DIM,
                 hidden_dim: int = TEXT_OUT_DIM // 2):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.out_dim = 2 * hidden_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        lengths = (tokens != self.pad_id).sum(dim=1).clamp(min=1).cpu()  # empty tweets keep one pad step
        packed = pack_padded_sequence(self.embedding(tokens), lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)  # h: [2, B, H]
        return torch.cat([h[0], h[1]], dim=1)


ENCODERS = {"textcnn": TextCNNEncoder, "bilstm": BiLSTMEncoder}


def build_text_encoder(name: str, vocab_size: int, pad_id: int = 0) -> nn.Module:
    if name not in ENCODERS:
        raise ValueError(f"Unknown text encoder '{name}'; choose from {sorted(ENCODERS)}")
    return ENCODERS[name](vocab_size, pad_id)


class TextClassifier(nn.Module):
    """Text-only model: encoder -> dropout -> linear. Takes a batch dict with a "text" tensor."""

    def __init__(self, encoder: nn.Module, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.out_dim, num_classes)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.classifier(self.dropout(self.encoder(batch["text"])))
