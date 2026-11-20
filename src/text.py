import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import torch

from src import config

PAD, UNK = "<pad>", "<unk>"
_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("&amp;", "&")
    text = _URL.sub(" xxurl ", text)
    text = _MENTION.sub(" xxuser ", text)
    return _TOKEN.findall(text)


class Vocab:
    """Token vocabulary. Build from training texts only."""

    def __init__(self, itos: list[str]):
        self.itos = list(itos)
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]

    @classmethod
    def build(cls, train_texts: Iterable[str], min_freq: int = 1, max_size: int | None = None) -> "Vocab":
        counts = Counter(tok for text in train_texts for tok in tokenize(text))
        tokens = sorted((t for t, c in counts.items() if c >= min_freq), key=lambda t: (-counts[t], t))
        if max_size is not None:
            tokens = tokens[:max_size]
        return cls([PAD, UNK, *tokens])

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int = config.MAX_TOKENS) -> list[int]:
        ids = [self.stoi.get(tok, self.unk_id) for tok in tokenize(text)][:max_len]
        return ids + [self.pad_id] * (max_len - len(ids))

    def encode_batch(self, texts: Iterable[str], max_len: int = config.MAX_TOKENS) -> torch.Tensor:
        return torch.tensor([self.encode(t, max_len) for t in texts], dtype=torch.long)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text()))
