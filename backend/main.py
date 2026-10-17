import os
import time
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.anpr import run_anpr_pipeline
from backend.model import detect_with_fallback, get_plate_detector, plate_detector_error, vehicle_detector
from backend.ocr import ocr_engine
from backend.utils import draw_detections, image_to_base64_jpeg, read_image_from_bytes

PLATE_FALLBACK_IMGSZ = int(os.getenv("PLATE_FALLBACK_IMGSZ", "768"))

app = FastAPI(title="ANPR API", version="2.0.0")
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
RUNS_DIR = BASE_DIR / "runs"
if RUNS_DIR.exists():
    app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")


def _validate_threshold(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise HTTPException(status_code=400, detail=f"{name} must be between 0.0 and 1.0.")


async def _read_upload_image(file: UploadFile) -> "object":
    filename = (file.filename or "").lower()
    ext = Path(filename).suffix
    content_type = (file.content_type or "").lower()

    if content_type.startswith("video/") or ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(
            status_code=400,
            detail="Video file detected! Please switch to the 'Video ANPR' tab to process video files.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    try:
        return read_image_from_bytes(file_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode image. Please upload a valid image (JPEG, PNG, WEBP, BMP).",
        ) from exc


def _build_detection_response(
    detector,
    image_bgr,
    confidence: float,
    mode: str,
    detections=None,
    inference_ms=None,
    warnings=None,
) -> Dict:
    if detections is None or inference_ms is None:
        detections, inference_ms = detector.detect(image_bgr, conf_threshold=confidence)
    annotated = draw_detections(image_bgr.copy(), detections)
    annotated_base64 = image_to_base64_jpeg(annotated)

    if mode == "vehicle":
        baseline_text = "Vehicle model uses pretrained YOLO11n (COCO classes car/bike/bus/truck)."
    else:
        plate_backend = getattr(detector, "backend_name", "yolo")
        if plate_backend == "roboflow":
            baseline_text = "Plate model uses Roboflow hosted inference."
        else:
            baseline_text = "Plate model uses a repo-trained checkpoint (no pretrained weights)."

    return {
        "mode": mode,
        "detections": detections,
        "inference_time": f"{inference_ms:.2f} ms",
        "confidence_threshold": round(confidence, 2),
        "model_size": detector.model_size_text,
        "device": detector.device,
        "baseline_map": baseline_text,
        "annotated_image": annotated_base64,
        "warnings": warnings or [],
    }


@app.get("/health")
async def health_check() -> Dict:
    plate_info: Dict[str, str] = {"weights": "unavailable", "model_size": "-", "backend": "unknown"}
    if plate_detector_error:
        plate_info["error"] = plate_detector_error
    else:
        try:
            plate = get_plate_detector()
            plate_info = {
                "weights": plate.weights_name,
                "model_size": plate.model_size_text,
                "backend": getattr(plate, "backend_name", "yolo"),
            }
        except RuntimeError as exc:
            plate_info = {"weights": "unavailable", "model_size": "-", "backend": "unknown", "error": str(exc)}

    return {
        "status": "ok",
        "device": vehicle_detector.device,
        "ocr_backend": ocr_engine.backend_name,
        "models": {
            "vehicle": {"weights": vehicle_detector.weights_name, "model_size": vehicle_detector.model_size_text},
            "plate": plate_info,
        },
    }


@app.get("/")
async def ui() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(
        index_path,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.post("/detect")
async def detect_default(file: UploadFile = File(...), confidence: float = Form(0.25)) -> Dict:
    _validate_threshold("confidence", confidence)
    image_bgr = await _read_upload_image(file)
    return _build_detection_response(vehicle_detector, image_bgr, confidence, mode="vehicle")


@app.post("/detect/vehicle")
async def detect_vehicle(file: UploadFile = File(...), confidence: float = Form(0.25)) -> Dict:
    _validate_threshold("confidence", confidence)
    image_bgr = await _read_upload_image(file)
    return _build_detection_response(vehicle_detector, image_bgr, confidence, mode="vehicle")


@app.post("/detect/plate")
async def detect_plate(file: UploadFile = File(...), confidence: float = Form(0.25)) -> Dict:
    _validate_threshold("confidence", confidence)
    image_bgr = await _read_upload_image(file)
    try:
        plate_detector = get_plate_detector()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    detections, inference_ms, used_fallback = detect_with_fallback(
        plate_detector,
        image_bgr,
        conf_threshold=confidence,
        fallback_imgsz=PLATE_FALLBACK_IMGSZ,
    )
    warnings = []
    if used_fallback:
        warnings.append("Fallback pass used for plate detection.")
    if not detections:
        warnings.append("No plates detected.")
    return _build_detection_response(
        plate_detector,
        image_bgr,
        confidence,
        mode="plate",
        detections=detections,
        inference_ms=inference_ms,
        warnings=warnings,
    )


@app.post("/anpr")
async def run_anpr(
    file: UploadFile = File(...),
    det_confidence: float = Form(0.25),
    ocr_confidence: float = Form(0.30),
) -> Dict:
    _validate_threshold("det_confidence", det_confidence)
    _validate_threshold("ocr_confidence", ocr_confidence)
    image_bgr = await _read_upload_image(file)
    try:
        plate_detector = get_plate_detector()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return run_anpr_pipeline(
        image_bgr=image_bgr,
        vehicle_detector=vehicle_detector,
        plate_detector=plate_detector,
        det_confidence=det_confidence,
        ocr_confidence=ocr_confidence,
        ocr_engine=ocr_engine,
    )


@app.post("/anpr/video")
async def run_anpr_video(
    file: UploadFile = File(...),
    det_confidence: float = Form(0.25),
    ocr_confidence: float = Form(0.30),
    frame_stride: int = Form(2),
) -> Dict:
    _validate_threshold("det_confidence", det_confidence)
    _validate_threshold("ocr_confidence", ocr_confidence)
    if frame_stride < 1 or frame_stride > 30:
        raise HTTPException(status_code=400, detail="frame_stride must be between 1 and 30.")

    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(status_code=400, detail="Supported video formats: .mp4, .avi, .mov, .mkv, .webm")

    timestamp = int(time.time() * 1000)
    upload_dir = BASE_DIR / "runs" / "video_uploads"
    output_dir = BASE_DIR / "runs" / "video_outputs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file_path = upload_dir / f"upload_{timestamp}{ext}"
    output_file_name = f"anpr_{timestamp}.mp4"
    output_file_path = output_dir / output_file_name

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty video file upload.")
    input_file_path.write_bytes(content)

    from backend.video import process_video_file

    try:
        result = process_video_file(
            input_path=str(input_file_path),
            output_path=str(output_file_path),
            det_confidence=det_confidence,
            ocr_confidence=ocr_confidence,
            frame_stride=frame_stride,
        )
        result["video_url"] = f"/runs/video_outputs/{output_file_name}"
        result["filename"] = file.filename
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {exc}") from exc
    finally:
        if input_file_path.exists():
            try:
                input_file_path.unlink()
            except Exception:
                pass

