import csv
from pathlib import Path

import pandas as pd
import torch

from src import config


def split_path(task: str, split: str, root: Path | None = None) -> Path:
    root = Path(root) if root else config.DATA_ROOT
    return root / config.SPLIT_DIRNAME / f"task_{task}_text_img_agreed_lab_{split}.tsv"


def load_split(task: str, split: str, root: Path | None = None) -> pd.DataFrame:
    """Load an official agreed-label TSV. All columns are read as strings so IDs are never coerced."""
    if task not in config.TASKS or split not in config.SPLITS:
        raise ValueError(f"Unknown task/split: {task}/{split}")
    df = pd.read_csv(split_path(task, split, root), sep="\t", dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE)
    unknown = set(df["label"]) - set(config.LABELS[task])
    if unknown:
        raise ValueError(f"Unexpected labels in {task}/{split}: {sorted(unknown)}")
    return df


def encode_labels(df: pd.DataFrame, task: str) -> torch.Tensor:
    index = {label: i for i, label in enumerate(config.LABELS[task])}
    return torch.tensor([index[label] for label in df["label"]], dtype=torch.long)


def load_image_features(path: Path = config.IMAGE_FEATURES):
    """Return (features [N, 512], {image_id: row}, invalid image_ids)."""
    cache = torch.load(path)
    index = {image_id: i for i, image_id in enumerate(cache["image_id"])}
    return cache["features"], index, set(cache["invalid"])


def align_image_features(df: pd.DataFrame, features: torch.Tensor, index: dict[str, int]):
    """Gather cached features in DataFrame order. Returns (features, mask of rows with a valid image)."""
    rows = [index.get(image_id, -1) for image_id in df["image_id"]]
    mask = torch.tensor([r >= 0 for r in rows], dtype=torch.bool)
    out = torch.zeros(len(rows), features.shape[1])
    out[mask] = features[torch.tensor([r for r in rows if r >= 0], dtype=torch.long)]
    return out, mask
