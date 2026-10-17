import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.anpr import run_anpr_pipeline
from backend.model import get_plate_detector, vehicle_detector
from backend.ocr import ocr_engine
from backend.utils import image_to_base64_jpeg


@dataclass
class TrackedPlate:
    track_id: int
    first_seen_sec: float
    last_seen_sec: float
    plate_text: str
    raw_text: str
    best_conf: float
    det_conf: float
    frames_seen: int
    last_bbox: List[int]
    indian_format_like: bool
    best_crop: Optional[np.ndarray] = None


def _bbox_iou(box1: List[int], box2: List[int]) -> float:
    if len(box1) != 4 or len(box2) != 4:
        return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return inter / float(area1 + area2 - inter)


def process_video_file(
    input_path: str,
    output_path: str,
    det_confidence: float = 0.25,
    ocr_confidence: float = 0.30,
    frame_stride: int = 2,
    max_duration_sec: float = 120.0,
) -> Dict:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if total_frames > 0 else 0.0

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    plate_detector = get_plate_detector()

    # Tracking state
    next_track_id = 1
    active_tracks: List[TrackedPlate] = []
    completed_tracks: List[TrackedPlate] = []

    frame_idx = 0
    processed_count = 0
    t0 = time.time()

    last_drawn_boxes: List[Tuple[List[int], str, float]] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_sec = frame_idx / fps
            if current_sec > max_duration_sec:
                break

            annotated_frame = frame.copy()

            # Process detection & OCR on designated stride frames
            if frame_idx % frame_stride == 0:
                processed_count += 1
                anpr_res = run_anpr_pipeline(
                    image_bgr=frame,
                    vehicle_detector=vehicle_detector,
                    plate_detector=plate_detector,
                    det_confidence=det_confidence,
                    ocr_confidence=ocr_confidence,
                    ocr_engine=ocr_engine,
                )

                current_detections = anpr_res.get("detections", [])
                matched_track_ids = set()
                new_drawn_boxes: List[Tuple[List[int], str, float]] = []

                for det in current_detections:
                    bbox = det.get("bbox", [])
                    if len(bbox) != 4:
                        continue
                    plate_text = det.get("plate_text") or det.get("plate") or ""
                    raw_text = det.get("raw_text") or det.get("raw_ocr") or ""
                    ocr_conf = float(det.get("ocr_confidence") or 0.0)
                    det_conf = float(det.get("det_confidence") or 0.0)
                    is_indian = bool(det.get("indian_format_like") or False)

                    # Crop
                    x1, y1, x2, y2 = bbox
                    crop = frame[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]

                    # Find matching active track
                    best_match: Optional[TrackedPlate] = None
                    best_iou = 0.0

                    for track in active_tracks:
                        iou = _bbox_iou(bbox, track.last_bbox)
                        if iou > 0.35 and iou > best_iou:
                            best_iou = iou
                            best_match = track

                    if best_match is not None:
                        # Update track
                        best_match.last_seen_sec = current_sec
                        best_match.frames_seen += 1
                        best_match.last_bbox = bbox
                        matched_track_ids.add(best_match.track_id)

                        # Upgrade text if this frame's OCR was better
                        if ocr_conf > best_match.best_conf or (not best_match.plate_text and plate_text):
                            best_match.plate_text = plate_text or best_match.plate_text
                            best_match.raw_text = raw_text or best_match.raw_text
                            best_match.best_conf = max(best_match.best_conf, ocr_conf)
                            best_match.det_conf = max(best_match.det_conf, det_conf)
                            best_match.indian_format_like = is_indian or best_match.indian_format_like
                            if crop.size > 0:
                                best_match.best_crop = crop.copy()

                        new_drawn_boxes.append((bbox, best_match.plate_text or "PLATE", best_match.best_conf))
                    else:
                        # New plate track
                        new_track = TrackedPlate(
                            track_id=next_track_id,
                            first_seen_sec=current_sec,
                            last_seen_sec=current_sec,
                            plate_text=plate_text,
                            raw_text=raw_text,
                            best_conf=ocr_conf,
                            det_conf=det_conf,
                            frames_seen=1,
                            last_bbox=bbox,
                            indian_format_like=is_indian,
                            best_crop=crop.copy() if crop.size > 0 else None,
                        )
                        next_track_id += 1
                        active_tracks.append(new_track)
                        matched_track_ids.add(new_track.track_id)
                        new_drawn_boxes.append((bbox, plate_text or "PLATE", ocr_conf))

                # Retire tracks that haven't been seen for > 1.5 seconds
                surviving_tracks = []
                for track in active_tracks:
                    if current_sec - track.last_seen_sec > 1.5:
                        completed_tracks.append(track)
                    else:
                        surviving_tracks.append(track)
                active_tracks = surviving_tracks
                last_drawn_boxes = new_drawn_boxes

            # Draw visual bounding box and label banners on frame
            font_scale = max(0.6, min(1.8, width / 1200.0))
            thickness = max(1, int(font_scale * 2))
            box_thick = max(2, int(width / 500.0))

            for bbox, label_str, conf_val in last_drawn_boxes:
                bx1, by1, bx2, by2 = bbox
                display_label = f"{label_str} ({int(conf_val * 100)}%)"

                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (34, 197, 94), box_thick)
                (tw, th), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                label_y1 = max(0, by1 - th - 8)
                label_y2 = by1
                label_x2 = min(width, bx1 + tw + 10)
                cv2.rectangle(annotated_frame, (bx1, label_y1), (label_x2, label_y2), (34, 197, 94), -1)
                cv2.putText(
                    annotated_frame,
                    display_label,
                    (bx1 + 5, label_y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )

            # Draw summary HUD banner in top-left
            hud_text = f"ANPR Video Stream | Frame {frame_idx}/{total_frames} | Unique Plates: {len(completed_tracks) + len(active_tracks)}"
            cv2.rectangle(annotated_frame, (10, 10), (min(width - 10, 680), 45), (15, 23, 42), -1)
            cv2.putText(
                annotated_frame,
                hud_text,
                (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            writer.write(annotated_frame)
            frame_idx += 1

    finally:
        cap.release()
        writer.release()

    # Move any remaining active tracks into completed list
    completed_tracks.extend(active_tracks)

    # Filter out spurious single-frame noise tracks
    valid_tracks = [t for t in completed_tracks if (t.frames_seen >= 2 or t.best_conf >= 0.50)]
    valid_tracks.sort(key=lambda t: t.first_seen_sec)

    unique_plates: List[Dict] = []
    for t in valid_tracks:
        thumb_b64 = image_to_base64_jpeg(t.best_crop) if t.best_crop is not None else ""
        unique_plates.append(
            {
                "track_id": t.track_id,
                "plate_text": t.plate_text or "(Unresolved)",
                "raw_text": t.raw_text,
                "confidence": round(t.best_conf, 4),
                "det_confidence": round(t.det_conf, 4),
                "first_seen_sec": round(t.first_seen_sec, 2),
                "last_seen_sec": round(t.last_seen_sec, 2),
                "duration_sec": round(t.last_seen_sec - t.first_seen_sec, 2),
                "frames_tracked": t.frames_seen,
                "indian_format_like": t.indian_format_like,
                "thumbnail": thumb_b64,
            }
        )

    proc_time = time.time() - t0
    return {
        "status": "success",
        "total_frames": frame_idx,
        "processed_frames": processed_count,
        "video_fps": round(fps, 2),
        "video_duration_sec": round(frame_idx / fps, 2),
        "processing_time_sec": round(proc_time, 2),
        "processing_fps": round(frame_idx / max(0.01, proc_time), 1),
        "unique_vehicles_count": len(unique_plates),
        "unique_plates": unique_plates,
        "output_video_path": output_path,
    }

