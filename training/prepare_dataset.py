from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class Box:
    class_id: int
    x: float
    y: float
    width: float
    height: float


def load_boxes(label_path: Path) -> List[Box]:
    items = json.loads(label_path.read_text(encoding="utf-8"))
    if isinstance(items, dict):
        items = [items]

    boxes: List[Box] = []
    for item in items:
        boxes.append(
            Box(
                class_id=int(item["class_id"]),
                x=float(item["x"]),
                y=float(item["y"]),
                width=float(item["width"]),
                height=float(item["height"]),
            )
        )
    return boxes


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def to_yolo_line(box: Box, img_w: int, img_h: int, class_map: Dict[int, int]) -> str:
    # Dataset stores absolute top-left + width/height.
    cx = (box.x + box.width / 2.0) / img_w
    cy = (box.y + box.height / 2.0) / img_h
    bw = box.width / img_w
    bh = box.height / img_h

    cls = class_map[box.class_id]
    return f"{cls} {clamp01(cx):.6f} {clamp01(cy):.6f} {clamp01(bw):.6f} {clamp01(bh):.6f}"


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        dst.write_bytes(src.read_bytes())


def write_data_yaml(out_dir: Path, names: List[str]) -> None:
    yaml_text = "\n".join(
        [
            f"path: {out_dir.as_posix()}",
            "train: train/images",
            "val: val/images",
            f"nc: {len(names)}",
            f"names: {names}",
            "",
        ]
    )
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def pair_files(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    image_paths = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    pairs: List[Tuple[Path, Path]] = []
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.json"
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def collect_class_ids(label_paths: Iterable[Path]) -> List[int]:
    class_ids = set()
    for label_path in label_paths:
        for box in load_boxes(label_path):
            class_ids.add(box.class_id)
    return sorted(class_ids)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    images_dir = root / "dataset" / "images-20260424T142546Z-3-001" / "images"
    labels_dir = root / "dataset" / "labels-20260424T143217Z-3-001" / "labels"
    out_dir = root / "dataset" / "yolo_prepared"

    val_ratio = 0.2
    seed = 42

    pairs = pair_files(images_dir, labels_dir)
    if not pairs:
        raise RuntimeError("No image/label pairs found.")

    class_ids = collect_class_ids([lp for _, lp in pairs])
    class_map = {cid: idx for idx, cid in enumerate(class_ids)}
    names = [f"class_{cid}" for cid in class_ids]

    random.Random(seed).shuffle(pairs)
    val_count = int(len(pairs) * val_ratio)
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    for split, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        for image_path, label_path in split_pairs:
            target_image = out_dir / split / "images" / image_path.name
            target_label = out_dir / split / "labels" / f"{image_path.stem}.txt"

            hardlink_or_copy(image_path, target_image)

            import cv2

            img = cv2.imread(str(image_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            lines = [to_yolo_line(box, img_w, img_h, class_map) for box in load_boxes(label_path)]
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    write_data_yaml(out_dir, names)
    (out_dir / "class_map.json").write_text(json.dumps(class_map, indent=2), encoding="utf-8")

    print(f"Prepared dataset at: {out_dir}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Classes: {class_ids} -> {names}")


if __name__ == "__main__":
    main()

