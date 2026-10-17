import os
import re
from threading import Lock
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except Exception:
    easyocr = None
    _EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    _PADDLEOCR_AVAILABLE = True
except Exception:
    PaddleOCR = None
    _PADDLEOCR_AVAILABLE = False

ALNUM_RE = re.compile(r"[^A-Z0-9]")
INDIAN_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$")

INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP",
    "WB", "BH"
}

CHAR_TO_DIGIT = {
    "O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2",
    "E": "3", "A": "4", "S": "5", "G": "6", "B": "8", "T": "7"
}

DIGIT_TO_CHAR = {
    "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A", "5": "S",
    "6": "G", "7": "T", "8": "B"
}


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class OCREngine:
    """Robust, multi-line, format-aware OCR engine for License Plate Recognition.
    
    Primary backend: EasyOCR (PyTorch native, CUDA-accelerated).
    Secondary fallback: PaddleOCR (if available and compatible).
    """

    def __init__(self) -> None:
        self._engine = None
        self._engine_type = "none"
        self._init_error = ""
        self._lock = Lock()

        force_cpu = _parse_bool(os.getenv("OCR_FORCE_CPU", "0"))
        self.use_gpu = torch.cuda.is_available() and not force_cpu
        self._init_engine()

    def _init_engine(self) -> None:
        # 1. Try EasyOCR first (high stability, PyTorch GPU acceleration)
        if _EASYOCR_AVAILABLE:
            try:
                # Suppress progress printing on Windows cp1252 consoles
                reader = easyocr.Reader(["en"], gpu=self.use_gpu, verbose=False)
                self._engine = reader
                self._engine_type = "easyocr"
                self._init_error = ""
                print(f"[OCR] EasyOCR initialized successfully (GPU={self.use_gpu}).")
                return
            except Exception as exc:
                self._init_error = f"EasyOCR init failed: {exc}"

        # 2. Try PaddleOCR fallback
        if _PADDLEOCR_AVAILABLE:
            try:
                device = "gpu:0" if self.use_gpu else "cpu"
                ocr = PaddleOCR(lang="en", device=device)
                self._engine = ocr
                self._engine_type = "paddleocr"
                self._init_error = ""
                print(f"[OCR] PaddleOCR initialized successfully (device={device}).")
                return
            except Exception as exc:
                if self._init_error:
                    self._init_error += f" | PaddleOCR init failed: {exc}"
                else:
                    self._init_error = f"PaddleOCR init failed: {exc}"

        print(f"[OCR] Warning: No OCR engine could be initialized: {self._init_error}")

    @property
    def available(self) -> bool:
        return self._engine is not None

    @property
    def backend_name(self) -> str:
        return self._engine_type if self.available else "none"

    @property
    def init_error(self) -> str:
        return self._init_error

    @staticmethod
    def clean_text(text: str) -> str:
        return ALNUM_RE.sub("", text.upper())

    @staticmethod
    def normalize_crop_dimensions(crop_bgr: np.ndarray, target_height: int = 100) -> np.ndarray:
        if crop_bgr.size == 0:
            return crop_bgr
        h, w = crop_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return crop_bgr
        
        # Scale crop so characters fit the receptive field optimal size (~96-120px)
        scale = float(target_height) / float(h)
        new_w = max(16, int(w * scale))
        return cv2.resize(crop_bgr, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def enhance_contrast(crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr.size == 0:
            return crop_bgr
        if crop_bgr.ndim == 2:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            return clahe.apply(crop_bgr)

        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        return cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)

    def parse_easyocr_boxes(self, raw_results: List) -> Tuple[str, float]:
        if not raw_results:
            return "", 0.0

        # Step 1: collect box heights to find the primary plate text height
        heights = []
        for item in raw_results:
            if len(item) < 3:
                continue
            pts, text, conf = item
            clean = ALNUM_RE.sub("", text.upper())
            if clean and clean not in {"IND", "INDIA", "IN0", "1ND", "1ND1A"} and float(conf) > 0.15:
                pts_arr = np.array(pts)
                h = float(np.max(pts_arr[:, 1]) - np.min(pts_arr[:, 1]))
                heights.append(h)

        if not heights:
            # Fallback to taking all items
            max_h = 1.0
        else:
            max_h = max(heights)

        valid_boxes: List[Dict] = []
        for item in raw_results:
            if len(item) < 3:
                continue
            pts, text, conf = item
            clean = ALNUM_RE.sub("", text.upper())
            if not clean:
                continue
            # Strip standalone "IND" or "INDIA" watermark from HSRP plates
            if clean in {"IND", "INDIA", "IN0", "1ND", "1ND1A"}:
                continue
            # Ignore tiny dealer frames, phone numbers, or web URLs (< 35% of max character height)
            pts_arr = np.array(pts)
            h = float(np.max(pts_arr[:, 1]) - np.min(pts_arr[:, 1]))
            if heights and h < (0.35 * max_h):
                continue
            if float(conf) < 0.20:
                continue

            y_center = float(np.mean(pts_arr[:, 1]))
            x_min = float(np.min(pts_arr[:, 0]))
            valid_boxes.append({"text": clean, "conf": float(conf), "y": y_center, "x": x_min, "h": h})

        if not valid_boxes:
            return "", 0.0

        # Step 2: Line grouping (support both 1-line horizontal and 2-line stacked plates)
        valid_boxes.sort(key=lambda b: b["y"])
        line_tolerance = max(15.0, max_h * 0.45)
        lines: List[List[Dict]] = []
        current_line: List[Dict] = [valid_boxes[0]]

        for box in valid_boxes[1:]:
            if abs(box["y"] - current_line[0]["y"]) > line_tolerance:
                lines.append(current_line)
                current_line = [box]
            else:
                current_line.append(box)
        lines.append(current_line)

        # Step 3: Sort horizontally within each line
        assembled_parts: List[str] = []
        confs: List[float] = []
        for line in lines:
            line.sort(key=lambda b: b["x"])
            for b in line:
                assembled_parts.append(b["text"])
                confs.append(b["conf"])

        raw_plate = "".join(assembled_parts)
        mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
        return raw_plate, mean_conf

    def read_text(self, plate_image: np.ndarray) -> Tuple[str, float]:
        if not self.available or plate_image.size == 0:
            return "", 0.0

        norm_image = self.normalize_crop_dimensions(plate_image, target_height=100)
        enhanced = self.enhance_contrast(norm_image)

        # Primary pass: Spatial EasyOCR with CLAHE and relative box filtering
        with self._lock:
            if self._engine_type == "easyocr":
                try:
                    res = self._engine.readtext(
                        enhanced,
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        detail=1,
                        paragraph=False,
                    )
                    text, conf = self.parse_easyocr_boxes(res)
                    if text and conf >= 0.35:
                        return text, conf

                    # Fallback to normalized original crop
                    res_orig = self._engine.readtext(
                        norm_image,
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        detail=1,
                        paragraph=False,
                    )
                    text_orig, conf_orig = self.parse_easyocr_boxes(res_orig)
                    if text_orig and conf_orig >= 0.35:
                        return text_orig, conf_orig
                    if text or text_orig:
                        return (text_orig, conf_orig) if conf_orig > conf else (text, conf)
                except Exception as exc:
                    print(f"[OCR] EasyOCR read error: {exc}")

            elif self._engine_type == "paddleocr":
                try:
                    if enhanced.ndim == 2:
                        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                    res = self._engine.ocr(enhanced)
                    lines = res[0] if (res and isinstance(res[0], list)) else res
                    if not lines:
                        return "", 0.0

                    items = []
                    for line in lines:
                        if len(line) >= 2 and line[1]:
                            pts = line[0]
                            txt = str(line[1][0]).strip()
                            c = float(line[1][1])
                            items.append((pts, txt, c))
                    return self.parse_easyocr_boxes(items)
                except Exception as exc:
                    print(f"[OCR] PaddleOCR read error: {exc}")

        # Secondary pass: Dedicated CRNN recognizer
        try:
            from backend.crnn_ocr import crnn_recognizer
            if crnn_recognizer.available:
                c_text, c_conf = crnn_recognizer.predict(plate_image)
                if c_text and len(c_text) >= 6:
                    return c_text, c_conf
        except Exception:
            pass

        return "", 0.0

    def correct_indian_plate(self, cleaned_text: str) -> str:
        """Intelligently fix character confusions according to Indian license plate structure."""
        if not cleaned_text:
            return ""

        text = self.clean_text(cleaned_text)

        # Strip leading IND if attached directly to plate text
        for prefix in ("IND", "IN0", "1ND", "1ND1A"):
            if text.startswith(prefix) and len(text) > len(prefix) + 4:
                text = text[len(prefix):]
                break

        if len(text) < 6:
            return text

        chars = list(text)

        # 1. State Code (First 2 chars must be letters)
        if len(chars) >= 2:
            c0 = DIGIT_TO_CHAR.get(chars[0], chars[0])
            c1 = DIGIT_TO_CHAR.get(chars[1], chars[1])
            cand = c0 + c1
            if cand in INDIAN_STATES:
                chars[0], chars[1] = c0, c1
            elif c0 + chars[1] in INDIAN_STATES:
                chars[0] = c0
            elif chars[0] + c1 in INDIAN_STATES:
                chars[1] = c1
            elif chars[0].isdigit() or chars[1].isdigit():
                chars[0] = c0
                chars[1] = c1

        # 2. Last 4 characters must be digits (Vehicle unique registration number)
        if len(chars) >= 8:
            for idx in range(len(chars) - 4, len(chars)):
                chars[idx] = CHAR_TO_DIGIT.get(chars[idx], chars[idx])
        elif len(chars) >= 6:
            # Short plates / commercial (last 3-4 are digits)
            for idx in range(max(2, len(chars) - 3), len(chars)):
                chars[idx] = CHAR_TO_DIGIT.get(chars[idx], chars[idx])

        # 3. District Code (Characters 2 & 3 must be digits)
        if len(chars) >= 4:
            chars[2] = CHAR_TO_DIGIT.get(chars[2], chars[2])
            chars[3] = CHAR_TO_DIGIT.get(chars[3], chars[3])

        # 4. Characters between district code and registration number are series letters
        if len(chars) >= 8:
            for idx in range(4, len(chars) - 4):
                chars[idx] = DIGIT_TO_CHAR.get(chars[idx], chars[idx])

        return "".join(chars)

    def read_candidates(self, variants: Dict[str, np.ndarray]) -> List[Dict]:
        candidates: List[Dict] = []
        for name, img in variants.items():
            raw, conf = self.read_text(img)
            if raw:
                candidates.append({
                    "raw_text": raw,
                    "confidence": float(conf),
                    "method_used": self.backend_name,
                    "variant": name,
                })
        return candidates


ocr_engine = OCREngine()
