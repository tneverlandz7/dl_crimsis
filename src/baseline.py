"""Majority-class baseline: always predict the most frequent training label.

The majority class and the class prior (stored as the "probabilities") come from the training split only.
The baseline is deterministic, so it is run once rather than over seeds.

    python -m src.baseline --task informative --exp-id A-inf-majority-s0-final-01
"""

import argparse
import time

import numpy as np

from src import config
from src.data import encode_labels, load_split
from src.env_logger import EnvRun
from src.metrics import compute_metrics, plot_confusion_matrix, result_dir, save_json, save_predictions

PART = {"informative": "A", "humanitarian": "B"}
MODEL = "majority"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", choices=config.TASKS, default="informative")
    p.add_argument("--exp-id", required=True, help="e.g. A-inf-majority-s0-final-01 (must be unique)")
    p.add_argument("--run-type", choices=["debug", "exploratory", "final"], default="final")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    task, part = args.task, PART[args.task]
    out_dir = result_dir(part, task, MODEL)

    with EnvRun(args.exp_id, part=part, task=task, run_type=args.run_type, model=MODEL,
                seed=0, batch_size="", device="cpu", cache_checkpoint_used="none") as run:
        train = load_split(task, "train")
        start = time.perf_counter()
        counts = np.bincount(encode_labels(train, task).numpy(), minlength=len(config.LABELS[task]))
        majority = int(counts.argmax())
        prior = counts / counts.sum()
        train_time_s = time.perf_counter() - start

        summary = {
            "model": MODEL, "task": task, "part": part, "experiment_id": args.exp_id, "seed": None,
            "majority_class": config.LABELS[task][majority],
            "train_class_counts": dict(zip(config.LABELS[task], counts.tolist())),
            "trainable_params": 0, "selected_epoch": None, "train_time_s": train_time_s,
        }
        for split in ("dev", "test"):
            df = load_split(task, split)
            y_pred = np.full(len(df), majority)
            probs = np.tile(prior, (len(df), 1))
            metrics = compute_metrics(encode_labels(df, task), y_pred, task)
            summary[split] = metrics
            save_predictions(df, y_pred, probs, task, out_dir / f"predictions_{split}.csv")
            plot_confusion_matrix(metrics, config.FIGURES_DIR / f"{part}_{task}_{MODEL}_cm_{split}.png",
                                  title=f"Majority baseline – {task} ({split})")

        save_json(summary, out_dir / "metrics.json")
        run.update(examples_processed=len(train), epochs=0, val_macro_f1=f"{summary['dev']['macro_f1']:.4f}",
                   run_status="completed")

    for split in ("dev", "test"):
        m = summary[split]
        print(f"{split:4s} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} weighted_f1={m['weighted_f1']:.4f}")
    print(f"Majority class: {summary['majority_class']}  ->  {out_dir}")


if __name__ == "__main__":
    main()
