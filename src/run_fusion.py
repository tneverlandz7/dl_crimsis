"""Train and evaluate Part A/B feature-fusion models for one encoder and seed.

Examples (each command writes a permanent row to env_tracker.csv):

    python -m src.run_fusion --task informative --encoder textcnn --seed 0 \
        --run-type debug --exp-id A-inf-fusion-textcnn-s0-dbg-01

    python -m src.run_fusion --task informative --encoder bilstm --seed 0 \
        --run-type final --exp-id A-inf-fusion-bilstm-s0-final-01

Debug/exploratory runs evaluate on validation only and write to logs/<exp-id>/.
Final runs also evaluate on test and write to results/<part>/<task>/fusion_<encoder>/seed<k>/.
"""

import argparse

import torch

from src import config
from src.data import align_image_features, encode_labels, load_image_features, load_split
from src.env_logger import EnvRun
from src.metrics import compute_metrics, result_dir, save_json, save_predictions
from src.models.fusion import FusionClassifier
from src.models.image import ImageEncoder
from src.models.text import ENCODERS, build_text_encoder
from src.text import Vocab
from src.train import make_loader, plot_learning_curves, predict, train_model
from src.utils import count_trainable_params, get_device, set_seed


PART = {"informative": "A", "humanitarian": "B"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", choices=config.TASKS, default="informative")
    p.add_argument("--encoder", choices=sorted(ENCODERS), default="textcnn")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exp-id", required=True, help="<part>-<task>-fusion-<encoder>-s<seed>-<dbg|exp|final>-<nn>")
    p.add_argument("--run-type", choices=["debug", "exploratory", "final"], default="debug")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-dropout", type=float, default=0.5)
    p.add_argument("--fusion-dropout", type=float, default=0.5)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--max-epochs", type=int, default=config.MAX_EPOCHS)
    p.add_argument("--features", default=str(config.IMAGE_FEATURES))
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    task, part, final = args.task, PART[args.task], args.run_type == "final"
    model_name = f"fusion_{args.encoder}"
    device = get_device(args.device)
    out_dir = (result_dir(part, task, model_name) / f"seed{args.seed}") if final else config.LOGS_DIR / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = config.CHECKPOINTS_DIR / f"{args.exp_id}.pt"

    with EnvRun(
        args.exp_id,
        part=part,
        task=task,
        run_type=args.run_type,
        model=model_name,
        seed=args.seed,
        batch_size=config.BATCH_SIZE,
        device=device,
        cache_checkpoint_used=f"feature cache {config.IMAGE_FEATURES.name}; saves {checkpoint.name}",
    ) as run:
        set_seed(args.seed)
        splits = {split: load_split(task, split) for split in config.SPLITS}
        vocab = Vocab.build(splits["train"]["tweet_text"], min_freq=args.min_freq)
        image_features, image_index, _ = load_image_features(args.features)

        def tensors(df):
            images, has_image = align_image_features(df, image_features, image_index)
            if not has_image.all():
                raise ValueError(f"{int((~has_image).sum())} examples have no cached image feature")
            return {
                "text": vocab.encode_batch(df["tweet_text"]),
                "image": images,
                "label": encode_labels(df, task),
            }

        train_loader = make_loader(tensors(splits["train"]), shuffle=True, seed=args.seed)
        dev_loader = make_loader(tensors(splits["dev"]))

        text_encoder = build_text_encoder(args.encoder, len(vocab), vocab.pad_id)
        image_encoder = ImageEncoder(dropout=args.image_dropout)
        model = FusionClassifier(
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            num_classes=len(config.LABELS[task]),
            hidden_dim=args.hidden_dim,
            dropout=args.fusion_dropout,
        )
        result = train_model(
            model,
            train_loader,
            dev_loader,
            task,
            device,
            lr=args.lr,
            max_epochs=args.max_epochs,
        )

        summary = {
            "model": model_name,
            "task": task,
            "part": part,
            "experiment_id": args.exp_id,
            "run_type": args.run_type,
            "seed": args.seed,
            "config": {
                "text_encoder": args.encoder,
                "vocab_size": len(vocab),
                "min_freq": args.min_freq,
                "max_tokens": config.MAX_TOKENS,
                "embed_dim": config.EMBED_DIM,
                "text_out_dim": text_encoder.out_dim,
                "image_feature_dim": config.IMAGE_FEATURE_DIM,
                "image_out_dim": image_encoder.out_dim,
                "image_encoder": "resnet18 IMAGENET1K_V1 (frozen, cached) -> shared ImageEncoder",
                "image_dropout": args.image_dropout,
                "fusion_hidden_dim": args.hidden_dim,
                "fusion_dropout": args.fusion_dropout,
                "lr": args.lr,
                "batch_size": config.BATCH_SIZE,
                "max_epochs": args.max_epochs,
                "patience": config.PATIENCE,
                "optimizer": "Adam",
            },
            "device": str(device),
            "trainable_params": count_trainable_params(model),
            "selected_epoch": result.selected_epoch,
            "epochs_run": result.epochs_run,
            "early_stopped": result.early_stopped,
            "train_time_s": result.train_time_s,
            "val_macro_f1": result.best_val_macro_f1,
            "history": result.history,
        }

        eval_splits = ("dev", "test") if final else ("dev",)
        for split in eval_splits:
            df = splits[split]
            loader = dev_loader if split == "dev" else make_loader(tensors(df))
            preds, probs, _ = predict(model, loader, device)
            summary[split] = compute_metrics(encode_labels(df, task), preds, task)
            save_predictions(df, preds, probs, task, out_dir / f"predictions_{split}.csv")

        save_json(summary, out_dir / "metrics.json")
        plot_learning_curves(
            result.history,
            out_dir / "learning_curve.png",
            title=f"{model_name} - {task} - seed {args.seed}",
            selected_epoch=result.selected_epoch,
        )
        config.CHECKPOINTS_DIR.mkdir(exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "vocab": vocab.itos,
                "encoder": args.encoder,
                "task": task,
                "image_dropout": args.image_dropout,
                "fusion_dropout": args.fusion_dropout,
                "hidden_dim": args.hidden_dim,
                "selected_epoch": result.selected_epoch,
            },
            checkpoint,
        )

        run.update(
            examples_processed=len(splits["train"]) * result.epochs_run,
            epochs=result.epochs_run,
            val_macro_f1=f"{result.best_val_macro_f1:.4f}",
            run_status="early_stopped" if result.early_stopped else "completed",
        )

    print(
        f"{args.exp_id}: model={model_name} params={summary['trainable_params']:,} "
        f"vocab={len(vocab):,} selected_epoch={result.selected_epoch}/{result.epochs_run} "
        f"time={result.train_time_s:.1f}s"
    )
    for split in eval_splits:
        metrics = summary[split]
        extra = (
            f" inf_P={metrics['informative_precision']:.3f} inf_R={metrics['informative_recall']:.3f}"
            if task == "informative"
            else ""
        )
        print(
            f"  {split:4s} acc={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}{extra}"
        )
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
