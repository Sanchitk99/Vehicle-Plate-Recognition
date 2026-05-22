from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Optional, Tuple

import torch
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete YOLO11n license plate training pipeline.")
    parser.add_argument("--datasets-dir", type=str, default="datasets", help="Dataset root containing images/ and labels/.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=512, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--name", type=str, default="plate_train", help="Run name in runs/detect/.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio if auto-splitting.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for inference test.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.yaml",
        help="Model config (.yaml) or checkpoint (.pt).",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights (only if model is a .pt checkpoint).",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Train from scratch without pretrained weights.",
    )
    parser.set_defaults(pretrained=False)
    return parser.parse_args()


def iter_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def resolve_datasets_dir(root: Path, datasets_dir_arg: str) -> Path:
    datasets_dir = Path(datasets_dir_arg)
    if not datasets_dir.is_absolute():
        datasets_dir = root / datasets_dir
    if datasets_dir.exists():
        return datasets_dir

    fallback = root / "dataset"
    if fallback.exists():
        print(f"[INFO] '{datasets_dir}' not found. Falling back to '{fallback}'.")
        return fallback

    raise FileNotFoundError(f"Dataset directory not found: {datasets_dir}")


def has_train_val_split(images_root: Path) -> bool:
    train_images = iter_images(images_root / "train")
    val_images = iter_images(images_root / "val")
    return len(train_images) > 0 and len(val_images) > 0


def find_label_for_image(image_path: Path, images_root: Path, labels_root: Path) -> Optional[Path]:
    rel = image_path.relative_to(images_root)
    by_relative = labels_root / rel.with_suffix(".txt")
    by_stem = labels_root / f"{image_path.stem}.txt"
    if by_relative.exists():
        return by_relative
    if by_stem.exists():
        return by_stem
    return None


def clear_split_folders(datasets_dir: Path) -> None:
    split_dirs = [
        datasets_dir / "images" / "train",
        datasets_dir / "images" / "val",
        datasets_dir / "labels" / "train",
        datasets_dir / "labels" / "val",
    ]
    for split_dir in split_dirs:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def auto_split_dataset(datasets_dir: Path, val_ratio: float, seed: int) -> Tuple[int, int, int]:
    images_root = datasets_dir / "images"
    labels_root = datasets_dir / "labels"

    if has_train_val_split(images_root):
        train_count = len(iter_images(images_root / "train"))
        val_count = len(iter_images(images_root / "val"))
        print("[INFO] Train/val split already exists. Skipping auto-split.")
        return train_count, val_count, 0

    if not images_root.exists() or not labels_root.exists():
        raise FileNotFoundError("Expected datasets/images and datasets/labels to exist.")

    source_images = [
        p
        for p in iter_images(images_root)
        if "train" not in p.parts and "val" not in p.parts
    ]
    if not source_images:
        raise RuntimeError("No source images found for splitting.")

    paired: list[tuple[Path, Path, Path]] = []
    missing_labels = 0
    for img_path in source_images:
        label_path = find_label_for_image(img_path, images_root, labels_root)
        if label_path is None:
            missing_labels += 1
            continue
        rel = img_path.relative_to(images_root)
        paired.append((img_path, label_path, rel))

    if not paired:
        raise RuntimeError("No image/label pairs found in YOLO format.")

    rng = random.Random(seed)
    rng.shuffle(paired)

    val_count = max(1, int(len(paired) * val_ratio))
    val_split = paired[:val_count]
    train_split = paired[val_count:]

    clear_split_folders(datasets_dir)

    for split_name, split_items in (("train", train_split), ("val", val_split)):
        for src_img, src_lbl, rel in split_items:
            dst_img = images_root / split_name / rel
            dst_lbl = labels_root / split_name / rel.with_suffix(".txt")
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)

    print(f"[INFO] Auto-split complete. Missing labels skipped: {missing_labels}")
    return len(train_split), len(val_split), missing_labels


def as_yaml_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def write_dataset_yaml(root: Path, datasets_dir: Path) -> Path:
    train_path = datasets_dir / "images" / "train"
    val_path = datasets_dir / "images" / "val"
    yaml_path = root / "dataset.yaml"

    yaml_text = "\n".join(
        [
            f"train: {as_yaml_path(train_path, root)}",
            f"val: {as_yaml_path(val_path, root)}",
            "",
            "nc: 1",
            'names: ["license_plate"]',
            "",
        ]
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def resolve_model_weights(root: Path, model_arg: str) -> str:
    requested = Path(model_arg)
    if requested.exists():
        return str(requested)
    return model_arg


def read_last_metrics(results_csv: Path) -> dict[str, float]:
    if not results_csv.exists():
        return {}

    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {}

    row = rows[-1]
    keys = [
        "metrics/mAP50(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
    ]

    out: dict[str, float] = {}
    for key in keys:
        try:
            out[key] = float(row[key])
        except (KeyError, ValueError, TypeError):
            continue
    return out


def print_metrics(metrics: dict[str, float]) -> None:
    if not metrics:
        print("[WARN] Could not parse metrics summary from results.csv")
        return

    print("\n=== Final Training Metrics (last epoch) ===")
    if "metrics/mAP50(B)" in metrics:
        print(f"mAP@0.5: {metrics['metrics/mAP50(B)']:.5f}")
    if "metrics/precision(B)" in metrics:
        print(f"precision: {metrics['metrics/precision(B)']:.5f}")
    if "metrics/recall(B)" in metrics:
        print(f"recall: {metrics['metrics/recall(B)']:.5f}")
    if "train/box_loss" in metrics:
        print(f"box_loss: {metrics['train/box_loss']:.5f}")
    if "train/cls_loss" in metrics:
        print(f"cls_loss: {metrics['train/cls_loss']:.5f}")
    if "train/dfl_loss" in metrics:
        print(f"dfl_loss: {metrics['train/dfl_loss']:.5f}")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    datasets_dir = resolve_datasets_dir(root, args.datasets_dir)
    runs_dir = root / "runs" / "detect"
    runs_dir.mkdir(parents=True, exist_ok=True)

    train_count, val_count, missing_labels = auto_split_dataset(datasets_dir, args.val_ratio, args.seed)
    dataset_yaml = write_dataset_yaml(root, datasets_dir)

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = "cuda:0" if device == 0 else "cpu"
    print("\n=== Pipeline Configuration ===")
    print(f"dataset root: {datasets_dir}")
    print(f"dataset yaml: {dataset_yaml}")
    print(f"train images: {train_count}")
    print(f"val images: {val_count}")
    print(f"missing labels skipped: {missing_labels}")
    print(f"device: {device_name}")

    model_weights = resolve_model_weights(root, args.model)
    print(f"base model: {model_weights}")
    print(f"pretrained: {args.pretrained}")

    if not args.pretrained and Path(model_weights).suffix.lower() == ".pt":
        raise ValueError(
            "Training from scratch requires a .yaml model config. "
            "Use --model yolo11n.yaml (or another .yaml) with --no-pretrained."
        )

    model = YOLO(model_weights)
    train_results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(runs_dir),
        name=args.name,
        pretrained=args.pretrained,
        save=True,
        exist_ok=True,
    )

    save_dir = Path(getattr(train_results, "save_dir", runs_dir / args.name))
    if not save_dir.exists():
        save_dir = runs_dir / args.name

    metrics = read_last_metrics(save_dir / "results.csv")
    print_metrics(metrics)

    print("\n=== Validation ===")
    val_results = model.val(
        data=str(dataset_yaml),
        imgsz=args.imgsz,
        batch=1,
        device=device,
        workers=0,
        split="val",
        project=str(runs_dir),
        name=args.name,
        exist_ok=True,
    )
    if hasattr(val_results, "results_dict") and isinstance(val_results.results_dict, dict):
        maybe_map = val_results.results_dict.get("metrics/mAP50(B)")
        if maybe_map is not None:
            print(f"validation mAP@0.5: {float(maybe_map):.5f}")

    weights_dir = save_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    best_for_infer = best_pt if best_pt.exists() else last_pt
    if not best_for_infer.exists():
        raise FileNotFoundError("No trained checkpoint found (best.pt or last.pt).")

    print("\n=== Inference Test ===")
    print(f"best checkpoint: {best_pt}")
    print(f"last checkpoint: {last_pt}")
    infer_model = YOLO(str(best_for_infer))
    infer_model.predict(
        source=str(datasets_dir / "images" / "val"),
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        batch=1,
        augment=False,
        save=True,
        project=str(runs_dir),
        name=f"{args.name}_infer",
        exist_ok=True,
    )
    print(f"inference outputs saved to: {runs_dir / f'{args.name}_infer'}")


if __name__ == "__main__":
    main()
