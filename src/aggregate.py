"""Aggregate final runs across seeds: mean ± std per model, plus the confusion matrix of the median run.

For each model under results/<part>/<task>/<model>/, the confusion-matrix run is chosen by median
validation macro-F1 (ties -> lowest seed). The choice uses validation scores only, as the brief requires.
Models with a single run (e.g. the majority baseline) are reported with std = 0.

    python -m src.aggregate --part A --task informative

Writes results/<part>/<task>/summary.csv (one row per model), runs.csv (one row per run) and
figures/<part>_<task>_<model>_cm_test_median.png.
"""

import argparse
import json

import numpy as np
import pandas as pd

from src import config
from src.metrics import plot_confusion_matrix

METRICS = ["accuracy", "macro_f1", "weighted_f1"]


def load_runs(model_dir) -> list[dict]:
    files = sorted(model_dir.glob("seed*/metrics.json")) or sorted(model_dir.glob("metrics.json"))
    return [json.loads(f.read_text()) for f in files]


def run_row(run: dict, task: str) -> dict:
    test = run["test"]
    row = {"model": run["model"], "seed": run.get("seed"), "experiment_id": run.get("experiment_id"),
           "val_macro_f1": run.get("val_macro_f1", run["dev"]["macro_f1"]),
           "selected_epoch": run.get("selected_epoch"), "trainable_params": run["trainable_params"],
           "train_time_s": run["train_time_s"]}
    row.update({f"test_{m}": test[m] for m in METRICS})
    if task == "informative":
        row.update(test_informative_precision=test["informative_precision"],
                   test_informative_recall=test["informative_recall"])
    for label, scores in test["per_class"].items():
        row[f"test_f1_{label}"] = scores["f1"]
        row[f"test_correct_{label}"] = scores["correct"]
    return row


def median_run(runs: list[dict]) -> dict:
    """Run whose validation macro-F1 is the median (lower median for even counts; ties -> lowest seed)."""
    ordered = sorted(runs, key=lambda r: (r.get("val_macro_f1", r["dev"]["macro_f1"]), r.get("seed") or 0))
    return ordered[(len(ordered) - 1) // 2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--part", default="A")
    p.add_argument("--task", choices=config.TASKS, default="informative")
    args = p.parse_args()

    task_dir = config.RESULTS_DIR / args.part / args.task
    rows, summary = [], []
    for model_dir in sorted(d for d in task_dir.iterdir() if d.is_dir()):
        runs = [r for r in load_runs(model_dir) if "test" in r]
        if not runs:
            continue
        model_rows = [run_row(r, args.task) for r in runs]
        rows += model_rows
        df = pd.DataFrame(model_rows).drop(columns=["model", "seed", "experiment_id"])
        chosen = median_run(runs)
        entry = {"model": model_dir.name, "n_seeds": len(runs), "cm_seed": chosen.get("seed")}
        for col in df.columns:
            values = df[col].astype(float)
            entry[f"{col}_mean"] = values.mean()
            entry[f"{col}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        summary.append(entry)
        if len(runs) > 1:  # single-run models (baseline) already have their own test confusion matrix
            plot_confusion_matrix(chosen["test"], config.FIGURES_DIR / f"{args.part}_{args.task}_{model_dir.name}_cm_test_median.png",
                              title=f"{model_dir.name} – {args.task} test (seed {chosen.get('seed')}, median val)")

    pd.DataFrame(rows).to_csv(task_dir / "runs.csv", index=False)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(task_dir / "summary.csv", index=False)

    cols = ["val_macro_f1"] + [f"test_{m}" for m in METRICS]
    if args.task == "informative":
        cols += ["test_informative_precision", "test_informative_recall"]
    cols += ["trainable_params", "train_time_s"]
    print(f"Part {args.part} - {args.task} (mean +/- sample std over seeds)")
    for entry in summary:
        cells = "  ".join(f"{c.removeprefix('test_')}={entry[c + '_mean']:.4f}+/-{entry[c + '_std']:.4f}"
                          if c not in ("trainable_params", "train_time_s") else f"{c}={entry[c + '_mean']:,.1f}"
                          for c in cols)
        print(f"  {entry['model']:10s} n={entry['n_seeds']} cm_seed={entry['cm_seed']}  {cells}")
    print(f"-> {task_dir / 'summary.csv'}, {task_dir / 'runs.csv'}")


if __name__ == "__main__":
    main()
