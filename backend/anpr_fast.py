import os
import re
import time
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from backend.ocr import ocr_engine
from backend.utils import image_to_base64_jpeg

INDIAN_PLATE_RE = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$")


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


DEBUG = _parse_bool(os.getenv("ANPR_DEBUG", "1"))
PROFILE_LOG = _parse_bool(os.getenv("ANPR_PROFILE_LOG", "1"))
FRAME_STRIDE = max(1, int(os.getenv("ANPR_FRAME_STRIDE", "2")))
MAX_PLATES = max(1, int(os.getenv("ANPR_MAX_PLATES", "5")))
DET_CONF = float(os.getenv("ANPR_DET_CONF", "0.30"))
OCR_CONF = float(os.getenv("ANPR_OCR_CONF", "0.30"))
OCR_CONF_LOW = float(os.getenv("ANPR_OCR_CONF_LOW", "0.40"))
IMGSZ = int(os.getenv("ANPR_IMGSZ", "640"))
PAD_X = float(os.getenv("ANPR_PAD_X", "0.20"))
PAD_Y = float(os.getenv("ANPR_PAD_Y", "0.30"))
SAVE_DEBUG = _parse_bool(os.getenv("ANPR_SAVE_DEBUG", "1"))
OCR_CACHE_IOU = float(os.getenv("ANPR_OCR_CACHE_IOU", "0.75"))
SMOOTH_WINDOW = max(1, int(os.getenv("ANPR_SMOOTH_WINDOW", "5")))
OCR_FALLBACK_CONF = float(os.getenv("ANPR_OCR_FALLBACK_CONF", "0.55"))
FINAL_MIN_CONF = float(os.getenv("ANPR_FINAL_MIN_CONF", "0.50"))
FINAL_MIN_VOTE_STRENGTH = float(os.getenv("ANPR_FINAL_MIN_VOTE", "0.60"))
STABLE_CONFIRM_FRAMES = max(1, int(os.getenv("ANPR_STABLE_CONFIRM_FRAMES", "3")))
STABLE_LOCK_FRAMES = max(1, int(os.getenv("ANPR_STABLE_LOCK_FRAMES", "10")))
NEW_PLATE_CONFIRM_FRAMES = max(1, int(os.getenv("ANPR_NEW_PLATE_CONFIRM_FRAMES", "3")))
ENABLE_SIMPLE_TRACKING = _parse_bool(os.getenv("ANPR_ENABLE_TRACKING", "1"))
PLATE_FALLBACK_IMGSZ = int(os.getenv("ANPR_PLATE_FALLBACK_IMGSZ", "768"))
PLATE_FALLBACK_CONF_FACTOR = float(os.getenv("ANPR_PLATE_FALLBACK_CONF_FACTOR", "0.60"))
PLATE_FALLBACK_MIN_CONF = float(os.getenv("ANPR_PLATE_FALLBACK_MIN_CONF", "0.10"))
PLATE_FALLBACK_AUGMENT = _parse_bool(os.getenv("ANPR_PLATE_FALLBACK_AUGMENT", "1"))
PLATE_FALLBACK_MAX_DET = max(1, int(os.getenv("ANPR_PLATE_FALLBACK_MAX_DET", "100")))


@dataclass
class PendingJob:
    frame_id: int
    future: Future


class OptimizedANPR:
    """Thread-safe, measurable ANPR pipeline.

    Design:
    - YOLO runs only in the main thread (avoids cross-thread model races).
    - OCR runs in one worker thread.
    - On frame N, detection runs while OCR from frame N-1 is already in progress.
    - Detection is executed every 2nd frame by default; skipped frames reuse last boxes.
    """

    def __init__(self) -> None:
        weights = os.getenv("ANPR_YOLO_WEIGHTS", "").strip()
        if not weights:
            weights = "yolov8n.pt"
            print("[ANPR] WARNING: ANPR_YOLO_WEIGHTS not set. Falling back to yolov8n.pt")

        self.weights = weights
        self.imgsz = IMGSZ

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(self.weights)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = self.device.startswith("cuda")
        if self.use_fp16:
            try:
                self.model.model.half()
            except Exception:
                self.use_fp16 = False

        self._ocr_pool = ThreadPoolExecutor(max_workers=1)
        self._state_lock = Lock()
        self._yolo_lock = Lock()
        self._detector_lock = Lock()

        self._frame_id = 0
        self._last_detections: List[Dict] = []
        self._last_det_ms: float = 0.0
        self._pending: Optional[PendingJob] = None
        self._last_plate_cache: Dict = {"bbox": [], "plate": "", "confidence": 0.0, "raw_ocr": "", "corrected_ocr": ""}
        self._plate_history = deque(maxlen=SMOOTH_WINDOW)
        self._last_valid_plate = ""
        self._last_valid_conf = 0.0
        self._last_valid_bbox: List[int] = []
        self._stable_plate = ""
        self._stable_count = 0
        self._lock_remaining = 0
        self._challenger_plate = ""
        self._challenger_count = 0
        self._external_plate_detector = None
        self._external_detector_strict = False

        self.debug_root = Path(__file__).resolve().parents[1] / "runs" / "anpr_debug"
        self._warmup_model()

        print(
            f"[ANPR] initialized | device={self.device} | yolo_weights={self.weights} | "
            f"ocr={ocr_engine.backend_name} | ocr_gpu={ocr_engine.use_gpu} | fp16={self.use_fp16}"
        )

    @staticmethod
    def _clamp_bbox(bbox: List[int], width: int, height: int) -> Optional[List[int]]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _pad_bbox(self, bbox: List[int], width: int, height: int) -> Optional[List[int]]:
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        px = int(bw * PAD_X)
        py = int(bh * PAD_Y)
        return self._clamp_bbox([x1 - px, y1 - py, x2 + px, y2 + py], width, height)

    def _warmup_model(self) -> None:
        warm = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            with self._yolo_lock, torch.no_grad():
                self.model.predict(
                    source=warm,
                    imgsz=self.imgsz,
                    conf=0.25,
                    max_det=1,
                    device=self.device,
                    verbose=False,
                    batch=1,
                )
            if DEBUG:
                print("[ANPR] YOLO warm-up complete.")
        except Exception as exc:
            print(f"[ANPR] warm-up failed: {exc}")

    @staticmethod
    def _bbox_iou(a: List[int], b: List[int]) -> float:
        if len(a) != 4 or len(b) != 4:
            return 0.0
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0.0:
            return 0.0
        area_a = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
        area_b = float(max(1, bx2 - bx1) * max(1, by2 - by1))
        return inter / max(1e-6, area_a + area_b - inter)

    @staticmethod
    def _resize_crop_for_ocr(crop_bgr: np.ndarray, max_w: int = 280, max_h: int = 120) -> np.ndarray:
        h, w = crop_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return crop_bgr

        scale = min(max_w / float(w), max_h / float(h), 1.0)
        if scale >= 1.0:
            return crop_bgr

        nw, nh = int(w * scale), int(h * scale)
        return cv2.resize(crop_bgr, (max(1, nw), max(1, nh)), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _preprocess_for_ocr(crop_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(up, (3, 3), 0)
        _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thr

    @staticmethod
    def _ocr_variants(crop_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        r15 = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        r20 = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        _, otsu = cv2.threshold(r20, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            r20,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )
        return [
            ("original", crop_bgr),
            ("resize_1_5x", r15),
            ("resize_2x", r20),
            ("otsu", otsu),
            ("adaptive", adaptive),
        ]

    def _run_ocr_fallback(self, crop_bgr: np.ndarray) -> Tuple[str, str, float, str]:
        best_raw = ""
        best_text = ""
        best_conf = 0.0
        best_variant = ""

        for variant_name, variant_img in self._ocr_variants(crop_bgr):
            raw_text, conf = ocr_engine.read_text(variant_img)
            corrected, is_valid = self._correct_text(raw_text)
            if is_valid and conf >= OCR_CONF:
                return raw_text, corrected, float(conf), variant_name

            if conf > best_conf:
                best_raw = raw_text
                best_text = corrected
                best_conf = float(conf)
                best_variant = variant_name

            # Confidence-gated early stop for near-valid outputs.
            if corrected and conf >= OCR_FALLBACK_CONF:
                return raw_text, corrected, float(conf), variant_name

        return best_raw, best_text, best_conf, best_variant

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    def configure_plate_detector(self, detector=None, strict: bool = False) -> None:
        """Use a provided detector for plate detection (local YOLO or Roboflow)."""
        with self._detector_lock:
            self._external_plate_detector = detector
            self._external_detector_strict = bool(strict)

    def _correct_text(self, raw_text: str) -> Tuple[str, bool]:
        cleaned = self._clean_text(raw_text)
        corrected = ocr_engine.correct_indian_plate(cleaned)
        return corrected, bool(INDIAN_PLATE_RE.fullmatch(corrected))

    @staticmethod
    def _accept_ocr_text(corrected: str, match: bool, conf: float) -> Tuple[str, bool]:
        high = max(OCR_CONF, OCR_CONF_LOW)
        low = min(OCR_CONF, OCR_CONF_LOW)
        if conf >= high:
            return corrected, bool(match)
        if conf >= low:
            return corrected, False
        return "", False

    def _history_vote(self, plate: str) -> Tuple[str, float]:
        token = plate if plate else "__none__"
        self._plate_history.append(token)
        counts = Counter(self._plate_history)
        top_token, top_count = counts.most_common(1)[0]
        vote_plate = "" if top_token == "__none__" else top_token
        vote_strength = float(top_count) / float(max(1, len(self._plate_history)))
        return vote_plate, vote_strength

    def _finalize_plate_state(self, plate: str, conf: float, bbox: List[int]) -> Tuple[str, float, List[int], bool, bool, str]:
        vote_plate, vote_strength = self._history_vote(plate)
        stabilized = False

        candidate = ""
        if plate and conf >= FINAL_MIN_CONF and vote_plate and vote_plate == plate and vote_strength >= FINAL_MIN_VOTE_STRENGTH:
            candidate = plate

        if self._lock_remaining > 0 and self._stable_plate:
            self._lock_remaining -= 1
            if candidate and candidate != self._stable_plate:
                if candidate == self._challenger_plate:
                    self._challenger_count += 1
                else:
                    self._challenger_plate = candidate
                    self._challenger_count = 1

                if self._challenger_count >= NEW_PLATE_CONFIRM_FRAMES:
                    self._stable_plate = candidate
                    self._stable_count = STABLE_CONFIRM_FRAMES
                    self._lock_remaining = STABLE_LOCK_FRAMES
                    self._challenger_plate = ""
                    self._challenger_count = 0
            else:
                self._challenger_plate = ""
                self._challenger_count = 0

            out_plate = self._stable_plate
            out_conf = self._last_valid_conf if out_plate else conf
            out_bbox = self._last_valid_bbox if self._last_valid_bbox else bbox
            stabilized = out_plate != plate
            return out_plate, out_conf, out_bbox, stabilized, bool(out_plate), "stable"

        if candidate:
            if candidate == self._stable_plate:
                self._stable_count += 1
            else:
                self._stable_plate = candidate
                self._stable_count = 1

            if self._stable_count >= STABLE_CONFIRM_FRAMES:
                self._lock_remaining = STABLE_LOCK_FRAMES

            self._last_valid_plate = candidate
            self._last_valid_conf = conf * max(0.5, vote_strength)
            if bbox:
                self._last_valid_bbox = list(bbox)

            state = "stable" if self._stable_count >= STABLE_CONFIRM_FRAMES else "uncertain"
            return candidate, self._last_valid_conf, (list(bbox) if bbox else self._last_valid_bbox), stabilized, True, state

        if self._last_valid_plate:
            stabilized = self._last_valid_plate != plate
            return self._last_valid_plate, self._last_valid_conf, self._last_valid_bbox, stabilized, False, "uncertain"

        return "", 0.0, bbox, stabilized, False, "searching"

    def _detect(self, frame_bgr: np.ndarray, conf: float) -> Tuple[List[Dict], float]:
        with self._detector_lock:
            external_detector = self._external_plate_detector
            external_strict = self._external_detector_strict
        if external_detector is not None:
            try:
                detections, det_ms = external_detector.detect(frame_bgr, conf_threshold=conf)
                if not detections:
                    fallback_conf = max(PLATE_FALLBACK_MIN_CONF, conf * PLATE_FALLBACK_CONF_FACTOR)
                    detections, fallback_ms = external_detector.detect(
                        frame_bgr,
                        conf_threshold=fallback_conf,
                        imgsz=PLATE_FALLBACK_IMGSZ,
                        augment=PLATE_FALLBACK_AUGMENT,
                        max_det=PLATE_FALLBACK_MAX_DET,
                    )
                    det_ms += fallback_ms

                normalized: List[Dict] = []
                for row in detections:
                    bbox = row.get("bbox", [])
                    if len(bbox) != 4:
                        continue
                    normalized.append(
                        {
                            "bbox": [int(v) for v in bbox],
                            "det_confidence": float(row.get("confidence", 0.0)),
                        }
                    )

                normalized.sort(key=lambda d: d["det_confidence"], reverse=True)
                return normalized[:MAX_PLATES], float(det_ms)
            except Exception as exc:
                if external_strict:
                    raise RuntimeError(f"External plate detector failed inside ANPR pipeline: {exc}") from exc
                if DEBUG:
                    print(f"[ANPR] external detector failed, using internal detector: {exc}")

        t0 = time.perf_counter()
        with self._yolo_lock, torch.no_grad():
            results = self.model.predict(
                source=frame_bgr,
                imgsz=self.imgsz,
                conf=conf,
                max_det=MAX_PLATES,
                device=self.device,
                verbose=False,
                batch=1,
            )
        det_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Dict] = []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections, det_ms

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for box, score in zip(xyxy, confs):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "det_confidence": float(score),
                }
            )

        detections.sort(key=lambda d: d["det_confidence"], reverse=True)
        return detections[:MAX_PLATES], det_ms

    def _save_debug_pair(self, frame_dir: Path, idx: int, crop: np.ndarray, prep: np.ndarray) -> Tuple[str, str]:
        frame_dir.mkdir(parents=True, exist_ok=True)
        crop_path = frame_dir / f"plate_{idx:02d}.png"
        prep_path = frame_dir / f"plate_{idx:02d}_prep.png"
        cv2.imwrite(str(crop_path), crop)
        cv2.imwrite(str(prep_path), prep)
        return str(crop_path.resolve()), str(prep_path.resolve())

    def _ocr_task(
        self,
        frame_id: int,
        frame_bgr: np.ndarray,
        detections: List[Dict],
        det_ms: float,
        reused_boxes: bool,
    ) -> Dict:
        ocr_t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_dir = self.debug_root / f"frame_{frame_id:06d}_{ts}"

        candidates: List[Dict] = []
        for i, det in enumerate(detections, start=1):
            bbox = self._pad_bbox(det["bbox"], w, h)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop = self._resize_crop_for_ocr(crop)
            prep = self._preprocess_for_ocr(crop)

            # OCR result cache: skip OCR if bbox is almost unchanged.
            cached_iou = self._bbox_iou(bbox, self._last_plate_cache.get("bbox", []))
            used_cached_ocr = cached_iou >= OCR_CACHE_IOU and bool(self._last_plate_cache.get("plate"))
            variant_used = "cache"

            if used_cached_ocr:
                raw_text = str(self._last_plate_cache.get("raw_ocr", ""))
                corrected = str(self._last_plate_cache.get("corrected_ocr", ""))
                ocr_conf = float(self._last_plate_cache.get("confidence", 0.0))
                final_text = str(self._last_plate_cache.get("plate", ""))
                match = bool(INDIAN_PLATE_RE.fullmatch(final_text))
                format_valid = bool(match)
            else:
                # Pass 1: fast single-pass on default preprocessing.
                raw_text, ocr_conf = ocr_engine.read_text(prep)
                corrected, match = self._correct_text(raw_text)
                final_text, format_valid = self._accept_ocr_text(corrected, match, float(ocr_conf))
                variant_used = "default_otsu"

                # Pass 2: multi-pass fallback when low confidence or invalid format.
                if (not final_text) or (ocr_conf < max(OCR_CONF, OCR_CONF_LOW)):
                    fb_raw, fb_text, fb_conf, fb_variant = self._run_ocr_fallback(crop)
                    if fb_text:
                        fb_match = bool(INDIAN_PLATE_RE.fullmatch(fb_text))
                        fb_final_text, fb_format_valid = self._accept_ocr_text(fb_text, fb_match, float(fb_conf))
                        # Regex adds a bonus, but never acts as a hard reject.
                        base_score = float(ocr_conf) + (0.05 if bool(match) else 0.0)
                        fb_score = float(fb_conf) + (0.05 if fb_match else 0.0)
                        if fb_final_text and (not final_text or fb_score > base_score):
                            raw_text = fb_raw
                            corrected = fb_text
                            ocr_conf = fb_conf
                            final_text = fb_final_text
                            match = fb_match
                            format_valid = fb_format_valid
                            variant_used = f"fallback_{fb_variant}"

                if final_text:
                    self._last_plate_cache = {
                        "bbox": list(bbox),
                        "plate": final_text,
                        "confidence": float(ocr_conf),
                        "raw_ocr": raw_text,
                        "corrected_ocr": corrected,
                    }

            crop_path, prep_path = "", ""
            if SAVE_DEBUG and DEBUG:
                crop_path, prep_path = self._save_debug_pair(frame_dir, i, crop, prep)

            if DEBUG:
                print(f"[ANPR][frame={frame_id}] det_conf={det['det_confidence']:.4f} bbox={bbox}")
                print("OCR RAW:", raw_text)
                print("OCR CORRECTED:", corrected)
                print("MATCH:", match)
                print("FINAL TEXT:", final_text)
                print(
                    f"[ANPR][frame={frame_id}] ocr_conf={ocr_conf:.4f} "
                    f"variant={variant_used} cached={used_cached_ocr}"
                )

            candidates.append(
                {
                    "bbox": bbox,
                    "raw_ocr": raw_text,
                    "corrected_ocr": corrected,
                    "plate": final_text,
                    "det_confidence": round(float(det["det_confidence"]), 4),
                    "ocr_confidence": round(float(ocr_conf), 4),
                    "indian_regex_match": bool(match and final_text),
                    "format_valid": bool(format_valid),
                    "variant_used": variant_used,
                    "used_cached_ocr": used_cached_ocr,
                    "debug": {
                        "crop_path": crop_path,
                        "preprocessed_path": prep_path,
                    }
                    if crop_path
                    else {},
                }
            )

        ocr_ms = (time.perf_counter() - ocr_t0) * 1000.0

        best = max(
            candidates,
            key=lambda x: (1 if x["plate"] else 0, x["ocr_confidence"], x["det_confidence"]),
            default=None,
        )

        plate = best["plate"] if best else ""
        bbox = best["bbox"] if best else []
        conf = float(best["ocr_confidence"]) if best else 0.0
        raw = best["raw_ocr"] if best else ""
        corrected = best["corrected_ocr"] if best else ""
        format_valid = bool(best["format_valid"]) if best else False
        final_plate, final_conf, final_bbox, stabilized, is_valid, state = self._finalize_plate_state(plate, conf, bbox)
        if final_plate and final_plate != corrected:
            corrected = final_plate
        if not raw and final_plate:
            raw = final_plate
        if final_plate:
            format_valid = bool(INDIAN_PLATE_RE.fullmatch(final_plate)) if corrected == final_plate else format_valid

        total_ms = det_ms + ocr_ms

        result = {
            "frame_id": frame_id,
            "plate": final_plate,
            "confidence": round(float(final_conf), 4),
            "bbox": final_bbox,
            "raw_ocr": raw,
            "corrected_ocr": corrected,
            "format_valid": format_valid,
            "stabilized": stabilized,
            "is_valid": is_valid,
            "state": state,
            "used_cached_boxes": reused_boxes,
            "timing": {
                "detection": round(det_ms, 2),
                "ocr": round(ocr_ms, 2),
                "total": round(total_ms, 2),
            },
            "detections": candidates,
        }

        if PROFILE_LOG:
            print(
                f"[ANPR][frame={frame_id}] Detection: {det_ms:.2f} ms | OCR: {ocr_ms:.2f} ms | "
                f"Total: {total_ms:.2f} ms"
            )

        return result

    def _make_empty_result(self, frame_id: int) -> Dict:
        return {
            "frame_id": frame_id,
            "plate": "",
            "confidence": 0.0,
            "bbox": [],
            "raw_ocr": "",
            "corrected_ocr": "",
            "format_valid": False,
            "stabilized": False,
            "is_valid": False,
            "state": "searching",
            "used_cached_boxes": False,
            "timing": {"detection": 0.0, "ocr": 0.0, "total": 0.0},
            "detections": [],
            "warnings": ["Pipeline warm-up: no OCR output yet."],
        }

    def process_frame(self, frame_bgr: np.ndarray, det_confidence: float = DET_CONF) -> Dict:
        """Streaming call with frame N detection and frame N-1 OCR overlap."""

        with self._state_lock:
            self._frame_id += 1
            frame_id = self._frame_id

            do_detect = (frame_id % FRAME_STRIDE == 1) or (not self._last_detections)
            if do_detect:
                detections, det_ms = self._detect(frame_bgr, det_confidence)
                self._last_detections = detections
                self._last_det_ms = det_ms
                reused = False
            else:
                detections = list(self._last_detections)
                det_ms = self._last_det_ms
                reused = True

            if ENABLE_SIMPLE_TRACKING and not detections and self._last_valid_bbox:
                detections = [
                    {
                        "bbox": list(self._last_valid_bbox),
                        "det_confidence": max(0.30, float(self._last_valid_conf)),
                    }
                ]
                reused = True

            completed = None
            if self._pending is not None:
                # Synchronize strictly with frame N-1 OCR result.
                # Detection for frame N is already done above, so this only waits for
                # the previous OCR remainder and avoids dropping/duplicating frames.
                completed = self._pending.future.result()

            self._pending = PendingJob(
                frame_id=frame_id,
                future=self._ocr_pool.submit(
                    self._ocr_task,
                    frame_id,
                    frame_bgr.copy(),
                    detections,
                    det_ms,
                    reused,
                ),
            )

            if completed is not None:
                return completed
            return self._make_empty_result(frame_id)

    def process_frame_sync(self, frame_bgr: np.ndarray, det_confidence: float = DET_CONF) -> Dict:
        """Single-image path: fully synchronous and deterministic."""

        with self._state_lock:
            self._frame_id += 1
            frame_id = self._frame_id

        detections, det_ms = self._detect(frame_bgr, det_confidence)
        if ENABLE_SIMPLE_TRACKING and not detections and self._last_valid_bbox:
            detections = [
                {
                    "bbox": list(self._last_valid_bbox),
                    "det_confidence": max(0.30, float(self._last_valid_conf)),
                }
            ]
        return self._ocr_task(frame_id, frame_bgr, detections, det_ms, reused_boxes=False)

    def flush_pending(self) -> Optional[Dict]:
        """Return the latest pending OCR result if any (for stream end flush)."""

        with self._state_lock:
            if self._pending is None:
                return None
            result = self._pending.future.result()
            self._pending = None
            return result


_pipeline = OptimizedANPR()


def process_frame(frame: np.ndarray) -> Dict:
    """Frame-stream API. Returns previous frame result once pipeline is warm."""
    return _pipeline.process_frame(frame)


def run_anpr_pipeline(
    image_bgr: np.ndarray,
    det_confidence: float = DET_CONF,
    ocr_confidence: float = OCR_CONF,
    plate_detector=None,
) -> Dict:
    del ocr_confidence
    _pipeline.configure_plate_detector(plate_detector, strict=plate_detector is not None)
    try:
        result = _pipeline.process_frame_sync(image_bgr, det_confidence=det_confidence)
    finally:
        _pipeline.configure_plate_detector(None, strict=False)

    # Compatibility keys used by existing API/frontend.
    annotated = image_bgr.copy()
    for row in result.get("detections", []):
        bbox = row.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        label = row.get("plate") or "plate"
        conf = float(row.get("det_confidence", 0.0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (34, 197, 94), 2)
        cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (34, 197, 94), 2)

    result["results"] = result.get("detections", [])
    result["timings"] = {
        "detection_ms": result["timing"]["detection"],
        "ocr_ms": result["timing"]["ocr"],
        "total_ms": result["timing"]["total"],
    }
    result["annotated_image"] = image_to_base64_jpeg(annotated)
    result["ocr_backend"] = ocr_engine.backend_name
    result["warnings"] = []
    if result.get("state") != "stable":
        result["warnings"].append(f"ANPR state: {result.get('state')}")
    if result["timing"]["total"] > 150:
        result["warnings"].append(f"Total latency above target: {result['timing']['total']} ms")
    return result


def test_single_image(path: str) -> Dict:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    result = run_anpr_pipeline(image)

    print("=" * 70)
    print(f"Image: {path}")
    print(f"Detected bbox: {result.get('bbox', [])}")
    print(f"OCR raw: {result.get('raw_ocr', '')}")
    print(f"OCR corrected: {result.get('corrected_ocr', '')}")
    print(f"Stabilized: {result.get('stabilized', False)}")
    print(f"is_valid: {result.get('is_valid', False)}")
    print(f"state: {result.get('state', 'searching')}")
    t = result.get("timing", {})
    print(f"Detection: {t.get('detection', 0.0):.2f} ms")
    print(f"OCR: {t.get('ocr', 0.0):.2f} ms")
    print(f"Total: {t.get('total', 0.0):.2f} ms")
    print("=" * 70)

    return result
