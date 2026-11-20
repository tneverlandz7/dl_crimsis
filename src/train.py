"""Shared training loop for every learned model (text-only, image-only, fusion).

Models take a batch dict and return logits, e.g. {"text": [B, 64], "image": [B, 512], "label": [B]}.
Training follows the brief: Adam, batch size 64, at most 20 epochs, checkpoint selected by validation
macro-F1 with early-stopping patience 3.

    train_loader = make_loader({"text": x, "label": y}, shuffle=True, seed=seed)
    result = train_model(model, train_loader, dev_loader, task, device)
    preds, probs = predict(model, test_loader, device)
"""

import time
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src import config  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402


class DictDataset(Dataset):
    """Aligned tensors indexed together; each item is a dict of per-example tensors."""

    def __init__(self, tensors: dict[str, torch.Tensor]):
        lengths = {len(t) for t in tensors.values()}
        if len(lengths) != 1:
            raise ValueError(f"Tensors have different lengths: {lengths}")
        self.tensors = tensors
        self.n = lengths.pop()

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        return {k: v[i] for k, v in self.tensors.items()}


def make_loader(tensors: dict[str, torch.Tensor], shuffle: bool = False, seed: int = 0,
                batch_size: int = config.BATCH_SIZE, sampler=None) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(DictDataset(tensors), batch_size=batch_size, shuffle=shuffle and sampler is None,
                      sampler=sampler, generator=generator)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return (predicted class [N], probabilities [N, C], logits [N, C]) on CPU."""
    model.eval()
    logits = torch.cat([model(_to_device(batch, device)).float().cpu() for batch in loader])
    probs = logits.softmax(dim=1)
    return probs.argmax(dim=1), probs, logits


@dataclass
class TrainResult:
    selected_epoch: int
    best_val_macro_f1: float
    epochs_run: int
    early_stopped: bool
    train_time_s: float
    history: list[dict] = field(default_factory=list)


def train_model(model: nn.Module, train_loader: DataLoader, dev_loader: DataLoader, task: str,
                device: torch.device, lr: float = 1e-3, max_epochs: int = config.MAX_EPOCHS,
                patience: int = config.PATIENCE, criterion: nn.Module | None = None) -> TrainResult:
    """Train, keep the epoch with the best dev macro-F1, and restore those weights into `model`.

    `criterion` defaults to unweighted cross-entropy; pass a weighted loss for Part B imbalance runs.
    The logged dev loss is always unweighted cross-entropy so curves stay comparable across runs.
    """
    criterion = (criterion or nn.CrossEntropyLoss()).to(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    y_dev = dev_loader.dataset.tensors["label"]

    best_f1, best_epoch, best_state, bad_epochs, history = -1.0, 0, None, 0, []
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum, seen = 0.0, 0
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch["label"])
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch["label"])
            seen += len(batch["label"])

        preds, _, logits = predict(model, dev_loader, device)
        dev = compute_metrics(y_dev, preds, task)
        history.append({"epoch": epoch, "train_loss": loss_sum / seen, "dev_loss": F.cross_entropy(logits, y_dev).item(),
                        "dev_macro_f1": dev["macro_f1"], "dev_accuracy": dev["accuracy"]})

        if dev["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = dev["macro_f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    train_time_s = time.perf_counter() - start

    model.load_state_dict(best_state)
    return TrainResult(selected_epoch=best_epoch, best_val_macro_f1=best_f1, epochs_run=len(history),
                       early_stopped=bad_epochs >= patience, train_time_s=train_time_s, history=history)


def plot_learning_curves(history: list[dict], path, title: str = "", selected_epoch: int | None = None) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax_loss.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax_loss.plot(epochs, [h["dev_loss"] for h in history], marker="o", label="validation")
    ax_loss.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    ax_f1.plot(epochs, [h["dev_macro_f1"] for h in history], marker="o", color="tab:green")
    ax_f1.set(xlabel="Epoch", ylabel="Macro-F1", title="Validation macro-F1")
    for ax in (ax_loss, ax_f1):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if selected_epoch:
            ax.axvline(selected_epoch, color="grey", linestyle="--", linewidth=1)
        ax.grid(alpha=0.3)
    ax_loss.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
