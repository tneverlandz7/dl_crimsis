"""Train and evaluate a text-only model (TextCNN or BiLSTM) for one task and seed.

    python -m src.run_text --task informative --encoder textcnn --seed 0 \
        --run-type final --exp-id A-inf-textcnn-s0-final-01

Debug/exploratory runs evaluate on validation only and write to logs/<exp-id>/.
Final runs also evaluate on test and write to results/<part>/<task>/<encoder>/seed<k>/.
The vocabulary is built from the training split only. Checkpoints go to checkpoints/<exp-id>.pt.
"""

import argparse

import torch

from src import config
from src.data import encode_labels, load_split
from src.env_logger import EnvRun
from src.metrics import compute_metrics, result_dir, save_json, save_predictions
from src.models.text import ENCODERS, TextClassifier, build_text_encoder
from src.text import Vocab
from src.train import make_loader, plot_learning_curves, predict, train_model
from src.utils import count_trainable_params, get_device, set_seed

PART = {"informative": "A", "humanitarian": "B"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", choices=config.TASKS, default="informative")
    p.add_argument("--encoder", choices=sorted(ENCODERS), default="textcnn")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exp-id", required=True, help="<part>-<task>-<model>-s<seed>-<dbg|exp|final>-<nn>")
    p.add_argument("--run-type", choices=["debug", "exploratory", "final"], default="debug")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--min-freq", type=int, default=2, help="Minimum training-split token count for the vocabulary")
    p.add_argument("--max-epochs", type=int, default=config.MAX_EPOCHS)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    task, part, final = args.task, PART[args.task], args.run_type == "final"
    device = get_device(args.device)
    out_dir = (result_dir(part, task, args.encoder) / f"seed{args.seed}") if final else config.LOGS_DIR / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with EnvRun(args.exp_id, part=part, task=task, run_type=args.run_type, model=args.encoder,
                seed=args.seed, batch_size=config.BATCH_SIZE, device=device,
                cache_checkpoint_used=f"saves {args.exp_id}.pt") as run:
        set_seed(args.seed)
        splits = {s: load_split(task, s) for s in config.SPLITS}
        vocab = Vocab.build(splits["train"]["tweet_text"], min_freq=args.min_freq)

        def tensors(df):
            return {"text": vocab.encode_batch(df["tweet_text"]), "label": encode_labels(df, task)}

        train_loader = make_loader(tensors(splits["train"]), shuffle=True, seed=args.seed)
        dev_loader = make_loader(tensors(splits["dev"]))

        encoder = build_text_encoder(args.encoder, len(vocab), vocab.pad_id)
        model = TextClassifier(encoder, len(config.LABELS[task]), dropout=args.dropout)
        result = train_model(model, train_loader, dev_loader, task, device, lr=args.lr,
                             max_epochs=args.max_epochs)

        summary = {
            "model": args.encoder, "task": task, "part": part, "experiment_id": args.exp_id,
            "run_type": args.run_type, "seed": args.seed,
            "config": {"lr": args.lr, "dropout": args.dropout, "min_freq": args.min_freq, "vocab_size": len(vocab),
                       "max_tokens": config.MAX_TOKENS, "embed_dim": config.EMBED_DIM,
                       "text_out_dim": encoder.out_dim, "batch_size": config.BATCH_SIZE,
                       "max_epochs": args.max_epochs, "patience": config.PATIENCE, "optimizer": "Adam"},
            "device": str(device), "trainable_params": count_trainable_params(model),
            "selected_epoch": result.selected_epoch, "epochs_run": result.epochs_run,
            "early_stopped": result.early_stopped, "train_time_s": result.train_time_s,
            "val_macro_f1": result.best_val_macro_f1, "history": result.history,
        }

        eval_splits = ("dev", "test") if final else ("dev",)
        for split in eval_splits:
            df = splits[split]
            loader = dev_loader if split == "dev" else make_loader(tensors(df))
            preds, probs, _ = predict(model, loader, device)
            summary[split] = compute_metrics(encode_labels(df, task), preds, task)
            save_predictions(df, preds, probs, task, out_dir / f"predictions_{split}.csv")

        save_json(summary, out_dir / "metrics.json")
        plot_learning_curves(result.history, out_dir / "learning_curve.png",
                             title=f"{args.encoder} – {task} – seed {args.seed}", selected_epoch=result.selected_epoch)
        config.CHECKPOINTS_DIR.mkdir(exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "vocab": vocab.itos, "encoder": args.encoder, "task": task,
                    "dropout": args.dropout, "selected_epoch": result.selected_epoch},
                   config.CHECKPOINTS_DIR / f"{args.exp_id}.pt")

        run.update(examples_processed=len(splits["train"]) * result.epochs_run, epochs=result.epochs_run,
                   val_macro_f1=f"{result.best_val_macro_f1:.4f}",
                   run_status="early_stopped" if result.early_stopped else "completed")

    print(f"{args.exp_id}: params={summary['trainable_params']:,} vocab={len(vocab):,} "
          f"selected_epoch={result.selected_epoch}/{result.epochs_run} time={result.train_time_s:.1f}s")
    for split in eval_splits:
        m = summary[split]
        print(f"  {split:4s} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
              f"inf_P={m['informative_precision'] if task == 'informative' else float('nan'):.3f} "
              f"inf_R={m['informative_recall'] if task == 'informative' else float('nan'):.3f}")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
