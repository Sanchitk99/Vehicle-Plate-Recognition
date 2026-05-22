import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from inference_sdk import InferenceHTTPClient

    _ROBOFLOW_AVAILABLE = True
except Exception:
    InferenceHTTPClient = None
    _ROBOFLOW_AVAILABLE = False

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
        self.backend_name = "yolo"

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

    def detect(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float = 0.25,
        imgsz: Optional[int] = None,
        augment: bool = False,
        max_det: int = 50,
    ) -> Tuple[List[dict], float]:
        start_time = time.time()
        predict_kwargs = {
            "source": image_bgr,
            "imgsz": imgsz or self.imgsz,
            "conf": conf_threshold,
            "device": self.device,
            "augment": augment,
            "agnostic_nms": False,
            "max_det": max_det,
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


class RoboflowDetector:
    def __init__(self, api_url: str, api_key: str, model_id: str, default_label: str = "plate") -> None:
        if not _ROBOFLOW_AVAILABLE:
            raise RuntimeError("inference-sdk is not installed. Run: pip install inference-sdk")
        if not api_key:
            raise RuntimeError("ROBOFLOW_API_KEY is not set.")

        self.client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self.model_id = model_id
        self.default_label = default_label
        self.device = "remote"
        self.weights_name = f"roboflow:{model_id}"
        self.model_size_bytes = None
        self.backend_name = "roboflow"

    @property
    def model_size_text(self) -> str:
        return "Hosted"

    def _encode_image(self, image_bgr: np.ndarray) -> bytes:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        success, encoded = cv2.imencode(".jpg", image_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            raise ValueError("Failed to encode image for Roboflow.")
        return encoded.tobytes()

    def _extract_bbox(self, prediction: dict) -> Optional[list[int]]:
        if not isinstance(prediction, dict):
            return None
        if {"x", "y", "width", "height"}.issubset(prediction.keys()):
            x = float(prediction["x"])
            y = float(prediction["y"])
            width = float(prediction["width"])
            height = float(prediction["height"])
            x1 = int(round(x - width / 2))
            y1 = int(round(y - height / 2))
            x2 = int(round(x + width / 2))
            y2 = int(round(y + height / 2))
            return [x1, y1, x2, y2]

        if {"x1", "y1", "x2", "y2"}.issubset(prediction.keys()):
            return [
                int(round(float(prediction["x1"]))),
                int(round(float(prediction["y1"]))),
                int(round(float(prediction["x2"]))),
                int(round(float(prediction["y2"]))),
            ]

        bbox = prediction.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return [int(round(float(v))) for v in bbox]

        return None

    def detect(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float = 0.25,
        imgsz: Optional[int] = None,
        augment: bool = False,
        max_det: int = 50,
    ) -> Tuple[List[dict], float]:
        start_time = time.time()
        image_bytes = self._encode_image(image_bgr)
        result = self.client.infer(image_bytes, model_id=self.model_id)
        inference_ms = (time.time() - start_time) * 1000.0

        predictions = result.get("predictions") if isinstance(result, dict) else None
        if not predictions:
            return [], inference_ms

        detections: List[dict] = []
        for pred in predictions:
            bbox = self._extract_bbox(pred)
            if not bbox:
                continue

            conf = float(pred.get("confidence", 0.0))
            if conf > 1.0:
                conf = conf / 100.0
            if conf < conf_threshold:
                continue

            label = pred.get("class") or pred.get("class_name") or self.default_label
            detections.append(
                {
                    "label": str(label),
                    "confidence": round(float(conf), 4),
                    "bbox": bbox,
                }
            )

            if max_det and len(detections) >= max_det:
                break

        return detections, inference_ms


def detect_with_fallback(
    detector,
    image_bgr: np.ndarray,
    conf_threshold: float,
    fallback_conf: Optional[float] = None,
    fallback_imgsz: Optional[int] = None,
    augment: bool = True,
    max_det: int = 100,
) -> Tuple[List[dict], float, bool]:
    detections, base_ms = detector.detect(image_bgr, conf_threshold=conf_threshold)
    if detections:
        return detections, base_ms, False

    conf = fallback_conf if fallback_conf is not None else max(0.10, conf_threshold * 0.6)
    imgsz = fallback_imgsz if fallback_imgsz is not None else max(640, detector.imgsz)
    fallback_dets, fallback_ms = detector.detect(
        image_bgr,
        conf_threshold=conf,
        imgsz=imgsz,
        augment=augment,
        max_det=max_det,
    )
    return fallback_dets, base_ms + fallback_ms, True


_base_dir = Path(__file__).resolve().parents[1]


def _select_latest_weight(paths: List[Path]) -> Optional[Path]:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def _resolve_local_plate_weights(base_dir: Path) -> Path:
    env_path = os.getenv("PLATE_WEIGHTS_PATH")
    if env_path:
        resolved = Path(env_path)
        if not resolved.is_absolute():
            resolved = base_dir / resolved
        if resolved.exists():
            return resolved.resolve()
        raise FileNotFoundError(f"PLATE_WEIGHTS_PATH not found: {resolved}")

    preferred = [
        base_dir / "runs" / "detect" / "plate_train" / "weights" / "best.pt",
        base_dir / "runs" / "detect" / "plate_train" / "weights" / "last.pt",
    ]
    latest = _select_latest_weight(preferred)
    if latest:
        return latest.resolve()

    candidates = list(base_dir.glob("runs/detect/plate_train*/weights/best.pt"))
    candidates += list(base_dir.glob("runs/detect/plate_train*/weights/last.pt"))
    latest = _select_latest_weight(candidates)
    if latest:
        return latest.resolve()

    raise FileNotFoundError(
        "No local plate weights found. Train from scratch and save to runs/detect/plate_train/weights/best.pt."
    )

vehicle_detector = YoloDetector(
    weights="yolov11n.pt",
    imgsz=512,
    class_map=VEHICLE_CLASS_MAP,
    restrict_class_ids=list(VEHICLE_CLASS_MAP.keys()),
    default_label="vehicle",
)

PLATE_IMGSZ = int(os.getenv("PLATE_IMGSZ", "640"))
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://detect.roboflow.com")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "license-plate-recognition-rxg4e/13")
PLATE_DETECTOR_BACKEND = os.getenv("PLATE_DETECTOR_BACKEND", "").strip().lower()
if not PLATE_DETECTOR_BACKEND:
    PLATE_DETECTOR_BACKEND = "roboflow" if os.getenv("ROBOFLOW_API_KEY") else "local"

plate_detector = None
plate_detector_error: Optional[str] = None

try:
    if PLATE_DETECTOR_BACKEND in {"roboflow", "remote"}:
        plate_detector = RoboflowDetector(
            api_url=ROBOFLOW_API_URL,
            api_key=os.getenv("ROBOFLOW_API_KEY", ""),
            model_id=ROBOFLOW_MODEL_ID,
            default_label="plate",
        )
    elif PLATE_DETECTOR_BACKEND in {"local", "yolo"}:
        plate_weights_path = str(_resolve_local_plate_weights(_base_dir))
        plate_detector = YoloDetector(
            weights=plate_weights_path,
            imgsz=PLATE_IMGSZ,
            class_map=PLATE_CLASS_MAP,
            restrict_class_ids=None,
            default_label="plate",
        )
    else:
        raise ValueError(
            "PLATE_DETECTOR_BACKEND must be 'roboflow' or 'local'. "
            f"Got: {PLATE_DETECTOR_BACKEND or 'empty'}."
        )
except (FileNotFoundError, RuntimeError, ValueError) as exc:
    plate_detector_error = str(exc)


def get_plate_detector():
    if plate_detector is None:
        raise RuntimeError(plate_detector_error or "Plate detector unavailable.")
    return plate_detector

