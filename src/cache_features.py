"""Cache 512-d features from a frozen ImageNet-pretrained ResNet18 for CrisisMMD v2.0 images.

Images shared across tasks/splits are extracted once. Re-running only processes images not yet cached.
Output: features/resnet18_imagenet1k_v1.pt = {"image_id": list[str], "features": FloatTensor[N, 512],
"invalid": list[str], "weights": str}.
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from src import config
from src.data import load_split
from src.env_logger import EnvRun
from src.utils import get_device

WEIGHTS = ResNet18_Weights.IMAGENET1K_V1


class ImageDataset(Dataset):
    def __init__(self, items: list[tuple[str, str]], root: Path):
        self.items = items
        self.root = root
        self.transform = WEIGHTS.transforms()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        image_id, rel_path = self.items[i]
        try:
            with Image.open(self.root / rel_path) as img:
                return image_id, self.transform(img.convert("RGB"))
        except (OSError, ValueError):
            return image_id, None


def collate(batch):
    valid = [(i, x) for i, x in batch if x is not None]
    invalid = [i for i, x in batch if x is None]
    images = torch.stack([x for _, x in valid]) if valid else None
    return [i for i, _ in valid], images, invalid


def build_encoder(device: torch.device) -> torch.nn.Module:
    model = resnet18(weights=WEIGHTS)
    model.fc = torch.nn.Identity()
    model.eval().requires_grad_(False)
    return model.to(device)


def collect_images(root: Path, tasks, splits) -> list[tuple[str, str]]:
    items: dict[str, str] = {}
    for task in tasks:
        for split in splits:
            df = load_split(task, split, root)
            for image_id, rel_path in zip(df["image_id"], df["image"]):
                items.setdefault(image_id, rel_path)
    return sorted(items.items())


def collect_all_images(root: Path) -> list[tuple[str, str]]:
    """Every image under data_image/; image_id is the file stem."""
    return sorted((p.stem, p.relative_to(root).as_posix()) for p in (root / "data_image").rglob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def load_cache(path: Path) -> dict:
    if path.exists():
        return torch.load(path)
    return {"image_id": [], "features": torch.empty(0, config.IMAGE_FEATURE_DIM), "invalid": [],
            "weights": "IMAGENET1K_V1"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", type=Path, default=config.DATA_ROOT)
    p.add_argument("--out", type=Path, default=config.IMAGE_FEATURES)
    p.add_argument("--tasks", nargs="+", default=list(config.TASKS), choices=config.TASKS)
    p.add_argument("--splits", nargs="+", default=list(config.SPLITS), choices=config.SPLITS)
    p.add_argument("--all-images", action="store_true", help="Cache every image in data_image/ (ignores split files)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None, help="Process at most N uncached images (debug runs)")
    p.add_argument("--exp-id", default=None)
    p.add_argument("--run-type", choices=["debug", "exploratory", "final"], default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    run_type = args.run_type or ("debug" if args.limit else "final")
    exp_id = args.exp_id or f"FEAT-resnet18-{run_type}-{datetime.now():%Y%m%d-%H%M%S}"

    cache = load_cache(args.out)
    done = set(cache["image_id"]) | set(cache["invalid"])
    items = (collect_all_images(args.data_root) if args.all_images
             else collect_images(args.data_root, args.tasks, args.splits))
    todo = [item for item in items if item[0] not in done]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print(f"Cache up to date: {args.out} ({len(cache['image_id'])} images)")
        return

    task = "all-images" if args.all_images else "+".join(args.tasks)
    with EnvRun(exp_id, part="shared", task=task, run_type=run_type,
                model="resnet18-imagenet1k_v1-frozen", batch_size=args.batch_size,
                cache_checkpoint_used=f"resumed from {len(done)} cached" if done else "none",
                device=device) as run:
        loader = DataLoader(ImageDataset(todo, args.data_root), batch_size=args.batch_size,
                            num_workers=args.num_workers, collate_fn=collate)
        encoder = build_encoder(device)
        ids, feats, invalid = [], [], []
        with torch.inference_mode():
            for batch_ids, images, bad in tqdm(loader, desc="ResNet18 features"):
                invalid.extend(bad)
                if images is None:
                    continue
                feats.append(encoder(images.to(device)).float().cpu())
                ids.extend(batch_ids)

        cache["image_id"] += ids
        cache["features"] = torch.cat([cache["features"], *feats])
        cache["invalid"] += invalid
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, args.out)
        run.update(examples_processed=len(todo), epochs=0)

    print(f"Cached {len(ids)} new features ({len(invalid)} invalid) -> {args.out}")


if __name__ == "__main__":
    main()
