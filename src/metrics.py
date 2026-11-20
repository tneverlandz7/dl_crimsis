"""Shared evaluation: metrics, per-example predictions and confusion-matrix plots.

Every model should report through these functions so Part A/B results share one format:
    m = compute_metrics(y_true, y_pred, task)
    save_predictions(df, y_pred, probs, task, path)
    plot_confusion_matrix(m, path, title)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support  # noqa: E402

from src import config  # noqa: E402

POSITIVE_CLASS = {"informative": "informative"}  # reported separately for Part A


def _to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def compute_metrics(y_true, y_pred, task: str) -> dict:
    """Metrics over all classes of `task`, including classes that are never predicted (F1 = 0)."""
    y_true, y_pred = _to_numpy(y_true), _to_numpy(y_pred)
    labels = config.LABELS[task]
    ids = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=ids, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=ids)
    metrics = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=ids, average="weighted", zero_division=0)),
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
                    "support": int(support[i]), "correct": int(cm[i, i])}
            for i, label in enumerate(labels)
        },
        "labels": labels,
        "confusion_matrix": cm.tolist(),  # rows = true label, columns = predicted label
    }
    if task in POSITIVE_CLASS:
        pos = metrics["per_class"][POSITIVE_CLASS[task]]
        metrics[f"{POSITIVE_CLASS[task]}_precision"] = pos["precision"]
        metrics[f"{POSITIVE_CLASS[task]}_recall"] = pos["recall"]
    return metrics


def save_predictions(df: pd.DataFrame, y_pred, probs, task: str, path: Path) -> None:
    """Per-example predictions with class probabilities, keyed by the string tweet/image IDs."""
    labels = config.LABELS[task]
    y_pred, probs = _to_numpy(y_pred), _to_numpy(probs)
    out = pd.DataFrame({"tweet_id": df["tweet_id"].values, "image_id": df["image_id"].values,
                        "label": df["label"].values, "pred": [labels[i] for i in y_pred]})
    for i, label in enumerate(labels):
        out[f"prob_{label}"] = probs[:, i]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def save_json(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def plot_confusion_matrix(metrics: dict, path: Path, title: str = "") -> None:
    cm = np.array(metrics["confusion_matrix"])
    labels = metrics["labels"]
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 3.5, 1.0 * len(labels) + 2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def result_dir(part: str, task: str, model: str) -> Path:
    """Standard results location: results/<part>/<task>/<model>/"""
    return config.RESULTS_DIR / part / task / model
