from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def add_bbox_padding(
    bbox: List[int],
    image_width: int,
    image_height: int,
    pad_x_ratio: float = 0.16,
    pad_y_ratio: float = 0.28,
) -> List[int]:
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    pad_x = int(w * pad_x_ratio)
    pad_y = int(h * pad_y_ratio)

    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(image_width, x2 + pad_x)
    py2 = min(image_height, y2 + pad_y)
    return [px1, py1, px2, py2]


def upscale_plate(crop_bgr: np.ndarray, scale: float = 3.0) -> np.ndarray:
    if crop_bgr.size == 0:
        return crop_bgr
    h, w = crop_bgr.shape[:2]
    target_w = max(1, int(w * scale))
    target_h = max(1, int(h * scale))
    return cv2.resize(crop_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def apply_gamma(image_bgr: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(0.1, float(gamma))
    inv_gamma = 1.0 / gamma
    table = np.array([((idx / 255.0) ** inv_gamma) * 255 for idx in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image_bgr, table)


def reduce_glare(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    glare_mask = cv2.inRange(v, 230, 255)
    glare_mask = cv2.medianBlur(glare_mask, 5)

    v_soft = cv2.GaussianBlur(v, (5, 5), 0)
    v_reduced = np.where(glare_mask > 0, v_soft, v).astype(np.uint8)

    merged = cv2.merge([h, s, v_reduced])
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


def sharpen_gray(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(gray, 1.65, blur, -0.65, 0)


def robust_plate_enhancement(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr.size == 0:
        return crop_bgr

    glare_reduced = reduce_glare(crop_bgr)
    gray_probe = cv2.cvtColor(glare_reduced, cv2.COLOR_BGR2GRAY)

    # Low-light normalization: brighten only when the ROI is underexposed.
    mean_luma = float(np.mean(gray_probe))
    if mean_luma < 90.0:
        glare_reduced = apply_gamma(glare_reduced, gamma=1.6)

    return glare_reduced


def build_plate_variants(crop_bgr: np.ndarray, scale: float = 3.0) -> Dict[str, np.ndarray]:
    enhanced_crop = robust_plate_enhancement(crop_bgr)
    upscaled = upscale_plate(enhanced_crop, scale=scale)
    if upscaled.size == 0:
        return {"original": crop_bgr}

    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

    clahe_obj = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe = clahe_obj.apply(bilateral)

    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )

    sharpen = sharpen_gray(clahe)
    _, sharpen_otsu = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresholded = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    enhanced = cv2.convertScaleAbs(sharpen, alpha=1.12, beta=6)

    return {
        "original": upscaled,
        "gray": gray,
        "bilateral": bilateral,
        "clahe": clahe,
        "enhanced": enhanced,
        "thresholded": thresholded,
        "otsu": otsu,
        "adaptive": adaptive,
        "sharpen": sharpen,
        "sharpen_otsu": sharpen_otsu,
    }


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def save_debug_artifacts(
    debug_root: Path,
    det_index: int,
    base_crop: np.ndarray,
    variants: Dict[str, np.ndarray],
) -> Tuple[str, Dict[str, str]]:
    det_dir = debug_root / f"det_{det_index:02d}"
    det_dir.mkdir(parents=True, exist_ok=True)

    crop_path = det_dir / "crop.png"
    cv2.imwrite(str(crop_path), _ensure_bgr(base_crop))

    variant_paths: Dict[str, str] = {}
    for name, img in variants.items():
        path = det_dir / f"{name}.png"
        cv2.imwrite(str(path), _ensure_bgr(img))
        variant_paths[name] = str(path.resolve())

    return str(crop_path.resolve()), variant_paths
