# ANPR Hackathon Demo

FastAPI + YOLO11n project with three modes:

- Vehicle detection (`/detect/vehicle`)
- Plate detection (`/detect/plate`)
- Full ANPR pipeline (`/anpr`: plate detection + OCR + text post-processing)

## Features

- Upload image (`jpg`/`png`)
- Detection confidence sliders
- Separate vehicle and plate models
- ANPR mode:
  - detects vehicles first (car/bike/bus/truck)
  - detects plate regions inside vehicle ROI
  - expands plate ROI with padding
  - crops plates
  - creates multiple preprocessed variants (gray, bilateral, CLAHE, threshold, sharpen)
  - low-light enhancement + glare reduction + blur-aware sharpening
  - runs EasyOCR with single-line + top/bottom multi-line (two-wheeler) logic
  - text normalization + Indian format-aware correction
- Debug artifacts saved per request in `runs/anpr_debug/...`
- Annotated image output + timing metrics

## Project Structure

```text
/backend
  main.py
  model.py
  anpr.py
  ocr.py
  postprocess.py
  utils.py

/frontend
  index.html
  script.js
  style.css
```

## Install

```bash
pip install -r requirements.txt
```

Notes:

- EasyOCR is the active OCR backend.
- OCR flow uses full-plate OCR plus split top/bottom OCR to support multi-line plates.

## Run

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- `http://localhost:8000/`

## API Endpoints

### `POST /detect/vehicle`

Form-data:

- `file`: image upload
- `confidence`: float (0.0 to 1.0)

### `POST /detect/plate`

Form-data:

- `file`: image upload
- `confidence`: float (0.0 to 1.0)

### `POST /anpr`

Form-data:

- `file`: image upload
- `det_confidence`: float (0.0 to 1.0)
- `ocr_confidence`: float (0.0 to 1.0, optional)

Example:

```bash
curl -X POST "http://localhost:8000/anpr" ^
  -F "file=@dataset\\images-20260424T142546Z-3-001\\images\\0000aa59-162e-4b76-af3f-9ebcd20325e0.jpg" ^
  -F "det_confidence=0.25" ^
  -F "ocr_confidence=0.30"
```

Response:

```json
{
  "detections": [
    {
      "vehicle_bbox": [x1, y1, x2, y2],
      "plate_bbox": [x1, y1, x2, y2],
      "det_confidence": 0.93,
      "plate_text": "MH12AB1234",
      "raw_text": "MH12A8I234",
      "cleaned_text": "MH12AB1234",
      "ocr_method": "easyocr_multi_line",
      "ocr_confidence": 0.88,
      "confidence": 0.88,
      "inference_time": "24.17 ms",
      "indian_format_like": true
    }
  ],
  "timings": {
    "detection_ms": 25.1,
    "ocr_ms": 42.7,
    "total_ms": 67.8
  },
  "device": "cuda:0",
  "ocr_backend": "easyocr",
  "ocr_available": true,
  "debug_dir": "D:/Project/ANPR/runs/anpr_debug/...",
  "warnings": [],
  "annotated_image": "data:image/jpeg;base64,..."
}
```

The ANPR detection object also includes fields required by leaderboard format:

- `vehicle_bbox`
- `plate_bbox`
- `plate_text`
- `confidence`
- `inference_time`
- `ocr_method`

## Leaderboard Evaluation Script

Run consolidated evaluation and generate report JSON:

```bash
python training/evaluate_leaderboard.py --imgsz 512 --sample-size 30
```

Optional arguments:

- `--vehicle-data`: YOLO dataset YAML for vehicle mAP local validation
- `--ocr-gt-csv`: CSV with OCR labels (`image,text[,x1,y1,x2,y2]`) for true character-accuracy score

Output is saved to:

- `runs/metrics/leaderboard_metrics.json`

Report includes:

- Plate detection mAP@0.5
- Vehicle detection mAP@0.5
- OCR character accuracy + exact match
- Latency (mean, p50, p95, p99)
- Model size (MB)
- FLOPs at selected `imgsz`
- Robustness retention (low-light + blur/glare)
- Threshold pass/fail summary

## Plate Training Artifacts

- Trained run: `runs/detect/plate_train`
- Checkpoints:
  - `runs/detect/plate_train/weights/best.pt`
  - `runs/detect/plate_train/weights/last.pt`
