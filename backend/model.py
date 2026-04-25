import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

VEHICLE_CLASS_MAP: Dict[int, str] = {
    2: "car",
    3: "bike",
    5: "bus",
    7: "truck",
}

PLATE_CLASS_MAP: Dict[int, str] = {
    0: "plate",
    1: "plate",
}


class YoloDetector:
    def __init__(
        self,
        weights: str,
        imgsz: int = 512,
        class_map: Optional[Dict[int, str]] = None,
        restrict_class_ids: Optional[List[int]] = None,
        default_label: str = "object",
    ) -> None:
        resolved_weights = self._resolve_weights_name(weights)
        self.weights_name = resolved_weights
        self.imgsz = imgsz
        self.class_map = class_map or {}
        self.restrict_class_ids = restrict_class_ids
        self.default_label = default_label

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(resolved_weights)
        self.model.to(self.device)

        self.weights_path = self._resolve_weights_path(resolved_weights)
        self.model_size_bytes = (
            os.path.getsize(self.weights_path) if self.weights_path and self.weights_path.exists() else None
        )

    def _resolve_weights_name(self, weights: str) -> str:
        if Path(weights).exists():
            return weights

        aliases = {
            "yolov11n.pt": "yolo11n.pt",
            "yolov11s.pt": "yolo11s.pt",
            "yolov11m.pt": "yolo11m.pt",
            "yolov11l.pt": "yolo11l.pt",
            "yolov11x.pt": "yolo11x.pt",
        }
        return aliases.get(weights, weights)

    def _resolve_weights_path(self, weights: str) -> Optional[Path]:
        direct = Path(weights)
        if direct.exists():
            return direct.resolve()

        ckpt_path = getattr(self.model, "ckpt_path", None)
        if ckpt_path:
            ckpt = Path(str(ckpt_path))
            if ckpt.exists():
                return ckpt.resolve()

        return None

    @property
    def model_size_text(self) -> str:
        if self.model_size_bytes is None:
            return "Unknown (weights in cache)"
        return f"{self.model_size_bytes / (1024 * 1024):.2f} MB"

    def _resolve_label(self, class_id: int) -> str:
        if class_id in self.class_map:
            return self.class_map[class_id]

        model_names = getattr(self.model, "names", {})
        if isinstance(model_names, dict) and class_id in model_names:
            return str(model_names[class_id])
        if isinstance(model_names, list) and 0 <= class_id < len(model_names):
            return str(model_names[class_id])
        return self.default_label

    def detect(self, image_bgr: np.ndarray, conf_threshold: float = 0.25) -> Tuple[List[dict], float]:
        start_time = time.time()
        predict_kwargs = {
            "source": image_bgr,
            "imgsz": self.imgsz,
            "conf": conf_threshold,
            "device": self.device,
            "augment": False,
            "agnostic_nms": False,
            "max_det": 50,
            "batch": 1,
            "verbose": False,
        }
        if self.restrict_class_ids is not None:
            predict_kwargs["classes"] = self.restrict_class_ids

        results = self.model.predict(**predict_kwargs)
        inference_ms = (time.time() - start_time) * 1000.0

        detections: List[dict] = []
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections, inference_ms

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        for box, conf, class_id in zip(xyxy, confs, class_ids):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            detections.append(
                {
                    "label": self._resolve_label(class_id),
                    "confidence": round(float(conf), 4),
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return detections, inference_ms


_base_dir = Path(__file__).resolve().parents[1]
_plate_weights = _base_dir / "runs" / "detect" / "plate_train" / "weights" / "best.pt"
plate_weights_path = str(_plate_weights) if _plate_weights.exists() else "yolo11n.pt"

vehicle_detector = YoloDetector(
    weights="yolov11n.pt",
    imgsz=512,
    class_map=VEHICLE_CLASS_MAP,
    restrict_class_ids=list(VEHICLE_CLASS_MAP.keys()),
    default_label="vehicle",
)

plate_detector = YoloDetector(
    weights=plate_weights_path,
    imgsz=512,
    class_map=PLATE_CLASS_MAP,
    restrict_class_ids=None,
    default_label="plate",
)

