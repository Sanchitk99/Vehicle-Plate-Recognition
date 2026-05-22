from typing import Dict

import numpy as np

from backend.anpr_fast import process_frame as process_frame
from backend.anpr_fast import run_anpr_pipeline as _run_fast


def run_anpr_pipeline(
    image_bgr: np.ndarray,
    vehicle_detector=None,
    plate_detector=None,
    det_confidence: float = 0.30,
    ocr_confidence: float = 0.30,
    ocr_engine=None,
) -> Dict:
    """Compatibility wrapper for existing API calls.

    The fast pipeline owns the OCR runtime and internal fallback detector.
    When a plate detector is provided, it is used in strict mode for this call.
    """

    del vehicle_detector, ocr_engine
    return _run_fast(
        image_bgr=image_bgr,
        det_confidence=det_confidence,
        ocr_confidence=ocr_confidence,
        plate_detector=plate_detector,
    )
