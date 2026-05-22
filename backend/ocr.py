import os
import re
from threading import Lock
from typing import List, Tuple

import cv2
import numpy as np
import torch

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

ALNUM_RE = re.compile(r"[^A-Z0-9]")
INDIAN_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$")


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class OCREngine:
    """Singleton PaddleOCR engine (initialized once)."""

    def __init__(self) -> None:
        self._engine = None
        self._init_error = ""
        self._lock = Lock()

        force_cpu = _parse_bool(os.getenv("OCR_FORCE_CPU", "0"))
        self.use_gpu = torch.cuda.is_available() and not force_cpu
        self.cpu_threads = int(os.getenv("OCR_CPU_THREADS", "2"))
        self._init_engine()

    def _build_engine(self, use_gpu: bool):
        return PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False,
            use_gpu=use_gpu,
            cpu_threads=self.cpu_threads,
        )

    def _init_engine(self) -> None:
        if PaddleOCR is None:
            self._init_error = "paddleocr import failed"
            return

        try:
            self._engine = self._build_engine(use_gpu=self.use_gpu)
            return
        except Exception as exc:
            self._init_error = str(exc)

        if self.use_gpu:
            try:
                self._engine = self._build_engine(use_gpu=False)
                self.use_gpu = False
                self._init_error = ""
                print("[OCR] GPU init failed; using CPU fallback.")
                return
            except Exception as exc:
                self._init_error = str(exc)
                self._engine = None

    @property
    def available(self) -> bool:
        return self._engine is not None

    @property
    def backend_name(self) -> str:
        return "paddleocr" if self.available else "none"

    @property
    def init_error(self) -> str:
        return self._init_error

    @staticmethod
    def clean_text(text: str) -> str:
        return ALNUM_RE.sub("", text.upper())

    @staticmethod
    def _force_char(ch: str, expects_letter: bool) -> str:
        letter_map = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"}
        digit_map = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"}

        if expects_letter:
            if ch.isalpha():
                return ch
            return letter_map.get(ch, ch)

        if ch.isdigit():
            return ch
        return digit_map.get(ch, ch)

    def correct_indian_plate(self, cleaned_text: str) -> str:
        if not cleaned_text:
            return ""
        if INDIAN_PATTERN.fullmatch(cleaned_text):
            return cleaned_text

        templates = ["LLDDLDDDD", "LLDDLLDDDD"]
        best = ""
        best_cost = 10**9

        for template in templates:
            span = len(template)
            if len(cleaned_text) < span:
                continue

            for i in range(0, len(cleaned_text) - span + 1):
                sample = cleaned_text[i : i + span]
                fixed = []
                cost = 0

                for ch, token in zip(sample, template):
                    expects_letter = token == "L"
                    forced = self._force_char(ch, expects_letter)
                    if forced != ch:
                        cost += 1
                    fixed.append(forced)

                candidate = "".join(fixed)
                if INDIAN_PATTERN.fullmatch(candidate) and cost < best_cost:
                    best = candidate
                    best_cost = cost

        return best if best else cleaned_text

    def read_text(self, plate_image: np.ndarray) -> Tuple[str, float]:
        if self._engine is None or plate_image.size == 0:
            return "", 0.0

        if plate_image.ndim == 2:
            plate_image = cv2.cvtColor(plate_image, cv2.COLOR_GRAY2BGR)

        with self._lock:
            result = self._engine.ocr(plate_image, cls=True)

        if not result:
            return "", 0.0

        lines = result[0] if isinstance(result[0], list) else result
        if not lines:
            return "", 0.0

        rows: List[Tuple[float, str, float]] = []
        for line in lines:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue

            box = line[0]
            text_conf = line[1]
            if not text_conf or len(text_conf) < 2:
                continue

            text = str(text_conf[0]).strip()
            conf = float(text_conf[1])
            if not text:
                continue

            y_center = 0.0
            x_center = 0.0
            try:
                y_center = sum(float(p[1]) for p in box) / len(box)
                x_center = sum(float(p[0]) for p in box) / len(box)
            except Exception:
                pass

            rows.append((y_center * 10000.0 + x_center, text, conf))

        if not rows:
            return "", 0.0

        rows.sort(key=lambda x: x[0])
        raw = "".join(text for _, text, _ in rows)
        mean_conf = float(sum(conf for _, _, conf in rows) / len(rows))
        return raw, mean_conf


ocr_engine = OCREngine()
