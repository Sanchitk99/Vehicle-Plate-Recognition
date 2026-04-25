import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from backend.postprocess import select_best_plate_candidate
from backend.preprocess import add_bbox_padding, build_plate_variants, save_debug_artifacts
from backend.utils import draw_detections, image_to_base64_jpeg


def _to_web_path(path: str) -> str:
    p = Path(path).resolve()
    base_dir = Path(__file__).resolve().parents[1]
    try:
        rel = p.relative_to(base_dir).as_posix()
        return f"/{rel}"
    except ValueError:
        return path


def run_anpr_pipeline(
    image_bgr: np.ndarray,
    vehicle_detector,
    plate_detector,
    det_confidence: float,
    ocr_confidence: float,
    ocr_engine,
) -> Dict:
    total_start = time.time()
    debug_root = (
        Path(__file__).resolve().parents[1]
        / "runs"
        / "anpr_debug"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    debug_root.mkdir(parents=True, exist_ok=True)

    original_path = debug_root / "original.png"
    cv2.imwrite(str(original_path), image_bgr)

    detection_start = time.time()
    vehicle_detections, vehicle_detection_ms = vehicle_detector.detect(image_bgr, conf_threshold=det_confidence)

    ocr_start = time.time()
    height, width = image_bgr.shape[:2]
    anpr_detections = []
    draw_payload = []
    warnings = []

    if not vehicle_detections:
        warnings.append("No vehicles detected. Falling back to full-frame plate search.")
        vehicle_regions = [{"bbox": [0, 0, width, height], "label": "full_frame", "confidence": 1.0}]
    else:
        vehicle_regions = vehicle_detections

    plate_detections = []
    for vehicle_idx, vehicle_det in enumerate(vehicle_regions, start=1):
        vx1, vy1, vx2, vy2 = [int(v) for v in vehicle_det["bbox"]]
        roi = image_bgr[vy1:vy2, vx1:vx2]
        if roi.size == 0:
            continue

        local_plates, _ = plate_detector.detect(roi, conf_threshold=det_confidence)
        for local_det in local_plates:
            lx1, ly1, lx2, ly2 = local_det["bbox"]
            abs_bbox = [vx1 + int(lx1), vy1 + int(ly1), vx1 + int(lx2), vy1 + int(ly2)]
            plate_detections.append(
                {
                    "vehicle_index": vehicle_idx,
                    "vehicle_bbox": [vx1, vy1, vx2, vy2],
                    "vehicle_label": vehicle_det.get("label", "vehicle"),
                    "vehicle_confidence": float(vehicle_det.get("confidence", 0.0)),
                    "plate_bbox": abs_bbox,
                    "plate_confidence": float(local_det.get("confidence", 0.0)),
                }
            )

    # Fallback ensures plate-only images are still processed if no in-vehicle plate was found.
    if not plate_detections:
        full_plates, _ = plate_detector.detect(image_bgr, conf_threshold=det_confidence)
        for local_det in full_plates:
            plate_detections.append(
                {
                    "vehicle_index": 0,
                    "vehicle_bbox": [],
                    "vehicle_label": "none",
                    "vehicle_confidence": 0.0,
                    "plate_bbox": [int(v) for v in local_det["bbox"]],
                    "plate_confidence": float(local_det.get("confidence", 0.0)),
                }
            )

    detection_ms = (time.time() - detection_start) * 1000.0

    if not plate_detections:
        warnings.append("No plate detections found.")
    if not ocr_engine.available:
        warnings.append("OCR backend unavailable. Install PaddleOCR or EasyOCR.")
    unreadable_count = 0

    for idx, det in enumerate(plate_detections, start=1):
        padded_bbox = add_bbox_padding(det["plate_bbox"], image_width=width, image_height=height)
        x1, y1, x2, y2 = padded_bbox
        crop = image_bgr[y1:y2, x1:x2]
        variants = build_plate_variants(crop, scale=3.0)

        ocr_candidates = ocr_engine.read_candidates(variants) if crop.size > 0 and ocr_engine.available else []
        best = select_best_plate_candidate(ocr_candidates, min_confidence=ocr_confidence)

        raw_text = best["raw_text"]
        cleaned_text = best["cleaned_text"]
        ocr_conf = best["confidence"]
        method_used = best["method_used"]
        used_variant = best["variant"]

        crop_path, variant_paths = save_debug_artifacts(debug_root, idx, crop, variants)
        summary_path = Path(crop_path).parent / "ocr_summary.json"
        summary_payload = {
            "raw_candidates": ocr_candidates,
            "selected": best,
            "det_confidence": float(det["plate_confidence"]),
            "vehicle_confidence": float(det["vehicle_confidence"]),
            "bbox_vehicle": det["vehicle_bbox"],
            "bbox_original": det["plate_bbox"],
            "bbox_padded": padded_bbox,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        det_inference_ms = max(0.0, detection_ms / max(1, len(plate_detections)))
        ocr_item_ms = max(0.0, ((time.time() - ocr_start) * 1000.0) / max(1, idx))

        anpr_detections.append(
            {
                "vehicle_bbox": det["vehicle_bbox"],
                "plate_bbox": padded_bbox,
                "bbox": padded_bbox,
                "det_confidence": round(float(det["plate_confidence"]), 4),
                "vehicle_confidence": round(float(det["vehicle_confidence"]), 4),
                "plate_text": cleaned_text,
                "ocr_confidence": round(float(ocr_conf), 4),
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "confidence": round(float(ocr_conf), 4),
                "method_used": method_used,
                "ocr_method": method_used,
                "inference_time": f"{(det_inference_ms + ocr_item_ms):.2f} ms",
                "source_variant": used_variant,
                "indian_format_like": bool(best["indian_format_like"]) if cleaned_text else False,
                "debug": {
                    "crop_path": crop_path,
                    "variant_paths": variant_paths,
                    "summary_path": str(summary_path.resolve()),
                    "web": {
                        "original": _to_web_path(str(original_path.resolve())),
                        "crop": _to_web_path(crop_path),
                        "variants": {name: _to_web_path(path) for name, path in variant_paths.items()},
                    },
                },
            }
        )
        if not cleaned_text:
            unreadable_count += 1

        draw_payload.append(
            {
                "label": cleaned_text if cleaned_text else "plate",
                "confidence": float(det["plate_confidence"]),
                "bbox": padded_bbox,
            }
        )

        if det["vehicle_bbox"]:
            draw_payload.append(
                {
                    "label": det["vehicle_label"],
                    "confidence": float(det["vehicle_confidence"]),
                    "bbox": det["vehicle_bbox"],
                }
            )

    ocr_ms = (time.time() - ocr_start) * 1000.0
    total_ms = (time.time() - total_start) * 1000.0

    if unreadable_count:
        warnings.append(f"OCR could not read text for {unreadable_count} plate(s).")

    fallback_draw = [
        {"label": "plate", "confidence": float(row["plate_confidence"]), "bbox": row["plate_bbox"]}
        for row in plate_detections
    ]
    annotated = draw_detections(image_bgr.copy(), draw_payload if draw_payload else fallback_draw)
    annotated_base64 = image_to_base64_jpeg(annotated)

    return {
        "results": anpr_detections,
        "detections": anpr_detections,
        "timings": {
            "vehicle_detection_ms": round(vehicle_detection_ms, 2),
            "detection_ms": round(detection_ms, 2),
            "ocr_ms": round(ocr_ms, 2),
            "total_ms": round(total_ms, 2),
        },
        "device": plate_detector.device,
        "model_size": plate_detector.model_size_text,
        "ocr_backend": ocr_engine.backend_name,
        "ocr_available": ocr_engine.available,
        "debug_dir": str(debug_root.resolve()),
        "debug_dir_web": _to_web_path(str(debug_root.resolve())),
        "warnings": warnings,
        "baseline_map": "Plate detector is fine-tuned YOLO11n with multi-pass OCR and format-aware post-processing.",
        "annotated_image": annotated_base64,
    }
