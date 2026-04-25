from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.anpr import run_anpr_pipeline
from backend.model import plate_detector, vehicle_detector
from backend.ocr import ocr_engine
from backend.utils import draw_detections, image_to_base64_jpeg, read_image_from_bytes

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
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are supported.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    try:
        return read_image_from_bytes(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _build_detection_response(detector, image_bgr, confidence: float, mode: str) -> Dict:
    detections, inference_ms = detector.detect(image_bgr, conf_threshold=confidence)
    annotated = draw_detections(image_bgr.copy(), detections)
    annotated_base64 = image_to_base64_jpeg(annotated)

    baseline_text = (
        "Vehicle model uses pretrained YOLO11n (COCO classes car/bike/bus/truck)."
        if mode == "vehicle"
        else "Plate model uses your fine-tuned YOLO11n checkpoint."
    )

    return {
        "mode": mode,
        "detections": detections,
        "inference_time": f"{inference_ms:.2f} ms",
        "confidence_threshold": round(confidence, 2),
        "model_size": detector.model_size_text,
        "device": detector.device,
        "baseline_map": baseline_text,
        "annotated_image": annotated_base64,
    }


@app.get("/health")
async def health_check() -> Dict:
    return {
        "status": "ok",
        "device": vehicle_detector.device,
        "ocr_backend": ocr_engine.backend_name,
        "models": {
            "vehicle": {"weights": vehicle_detector.weights_name, "model_size": vehicle_detector.model_size_text},
            "plate": {"weights": plate_detector.weights_name, "model_size": plate_detector.model_size_text},
        },
    }


@app.get("/")
async def ui() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_path)


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
    return _build_detection_response(plate_detector, image_bgr, confidence, mode="plate")


@app.post("/anpr")
async def run_anpr(
    file: UploadFile = File(...),
    det_confidence: float = Form(0.25),
    ocr_confidence: float = Form(0.30),
) -> Dict:
    _validate_threshold("det_confidence", det_confidence)
    _validate_threshold("ocr_confidence", ocr_confidence)
    image_bgr = await _read_upload_image(file)
    return run_anpr_pipeline(
        image_bgr=image_bgr,
        vehicle_detector=vehicle_detector,
        plate_detector=plate_detector,
        det_confidence=det_confidence,
        ocr_confidence=ocr_confidence,
        ocr_engine=ocr_engine,
    )
