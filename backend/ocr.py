import re
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch


class OCREngine:
    def __init__(self) -> None:
        self._easy_engine = None
        self.backends: List[str] = []
        self._init_backends()

    @property
    def available(self) -> bool:
        return len(self.backends) > 0

    @property
    def backend_name(self) -> str:
        if not self.backends:
            return "none"
        return "+".join(self.backends)

    def _init_backends(self) -> None:
        try:
            import easyocr

            self._easy_engine = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
            self.backends.append("easyocr")
        except Exception:
            self._easy_engine = None

    def _ensure_bgr(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def preprocess_plate(self, plate: np.ndarray) -> np.ndarray:
        plate_bgr = self._ensure_bgr(plate)
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
        thresh = cv2.resize(thresh, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        return thresh

    def clean_text(self, text: str) -> str:
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        cleaned = cleaned.replace("O", "0")
        cleaned = cleaned.replace("I", "1")
        cleaned = cleaned.replace("S", "5")
        cleaned = cleaned.replace("B", "8")
        return cleaned

    def ocr_single_pass(self, image: np.ndarray) -> Tuple[str, float]:
        if self._easy_engine is None:
            return "", 0.0

        results = self._easy_engine.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
        if not results:
            return "", 0.0

        best = max(results, key=lambda row: float(row[2]) if len(row) >= 3 else 0.0)
        if len(best) < 3:
            return "", 0.0
        return str(best[1]), float(best[2])

    def ocr_multiline_plate(self, plate: np.ndarray) -> Tuple[str, float]:
        if plate.size == 0:
            return "", 0.0

        h, _ = plate.shape[:2]

        full_proc = self.preprocess_plate(plate)
        full_text, full_conf = self.ocr_single_pass(full_proc)

        top_half = plate[0 : int(h / 2), :]
        bottom_half = plate[int(h / 2) : h, :]

        top_proc = self.preprocess_plate(top_half)
        bottom_proc = self.preprocess_plate(bottom_half)

        top_text, top_conf = self.ocr_single_pass(top_proc)
        bottom_text, bottom_conf = self.ocr_single_pass(bottom_proc)

        full_text = self.clean_text(full_text)
        top_text = self.clean_text(top_text)
        bottom_text = self.clean_text(bottom_text)

        combined_text = top_text + bottom_text

        if len(combined_text) >= 8:
            final_text = combined_text
            confidence = (top_conf + bottom_conf) / 2.0
        elif len(full_text) >= len(combined_text):
            final_text = full_text
            confidence = full_conf
        else:
            final_text = combined_text
            confidence = (top_conf + bottom_conf) / 2.0

        return final_text, float(confidence)

    def run_ocr_on_plate(self, plate_img: np.ndarray) -> Dict:
        text, conf = self.ocr_multiline_plate(plate_img)
        return {
            "plate_text": text,
            "confidence": float(conf),
            "method": "multi-line OCR",
        }

    def read_candidates(self, variants: Dict[str, np.ndarray], min_text_len: int = 4) -> List[Dict]:
        if not self.available:
            return []

        candidates: List[Dict] = []
        for variant_name, image in variants.items():
            result = self.run_ocr_on_plate(image)
            raw_text = str(result.get("plate_text", "")).strip()
            if len(raw_text) < min_text_len:
                continue
            candidates.append(
                {
                    "raw_text": raw_text,
                    "confidence": float(result.get("confidence", 0.0)),
                    "method_used": "easyocr_multi_line",
                    "variant": variant_name,
                }
            )

        dedup: Dict[str, Dict] = {}
        for cand in candidates:
            key = f"{cand['method_used']}::{cand['variant']}::{cand['raw_text']}"
            if key not in dedup or float(cand["confidence"]) > float(dedup[key]["confidence"]):
                dedup[key] = cand
        return list(dedup.values())


ocr_engine = OCREngine()
