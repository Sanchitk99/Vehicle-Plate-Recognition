import base64
from typing import List

import cv2
import numpy as np


def read_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    np_buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode image. Please upload a valid JPEG/PNG file.")
    return image_bgr


def draw_detections(image_bgr: np.ndarray, detections: List[dict]) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        confidence = det["confidence"]
        text = f"{label} {confidence:.2f}"

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (34, 197, 94), 2)

        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_w, text_h = text_size
        box_y1 = max(0, y1 - text_h - 10)
        box_y2 = y1

        cv2.rectangle(image_bgr, (x1, box_y1), (x1 + text_w + 10, box_y2), (34, 197, 94), -1)
        cv2.putText(
            image_bgr,
            text,
            (x1 + 5, box_y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (15, 23, 42),
            2,
            cv2.LINE_AA,
        )

    return image_bgr


def image_to_base64_jpeg(image_bgr: np.ndarray) -> str:
    success, encoded = cv2.imencode(".jpg", image_bgr)
    if not success:
        raise ValueError("Could not encode annotated image.")

    img_base64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_base64}"

