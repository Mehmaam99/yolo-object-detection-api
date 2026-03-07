"""
========================================================
PROJECT 2: YOLO Object Detection API
========================================================
Author      : Syed Muhammad Mehmam
Tech Stack  : FastAPI + YOLOv8 + OpenCV
Description : A production-ready object detection API that accepts
              images or video frames and returns detected objects
              with bounding boxes, confidence scores, and labels.
              Inspired by the real-time factory surveillance system
              built at Xloop Digital Services.
========================================================
"""

import io
import os
import time
import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH = "yolov8n.pt"        # Nano model — fast, good accuracy, small size
UPLOAD_DIR = "./uploads"
CONFIDENCE_THRESHOLD = 0.5       # Only return detections above 50% confidence
MAX_FILE_SIZE_MB = 10

Path(UPLOAD_DIR).mkdir(exist_ok=True)

# ── Load YOLO Model ───────────────────────────────────────────────────────────
# YOLOv8n is downloaded automatically on first run (~6MB)
# In production, you'd load a custom-trained model here

print("[INFO] Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)
print(f"[INFO] Model loaded — {len(model.names)} classes available")

# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="YOLO Object Detection API",
    description="Detect objects in images using YOLOv8. Returns bounding boxes, labels, and confidence scores.",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Pydantic Models ───────────────────────────────────────────────────────────

class Detection(BaseModel):
    """Single detected object"""
    label: str              # Class name e.g. "person", "car"
    confidence: float       # Confidence score 0.0 - 1.0
    bbox: dict              # Bounding box: x1, y1, x2, y2 (pixels)
    bbox_normalized: dict   # Normalized bbox 0.0-1.0 (useful for frontend rendering)

class DetectionResponse(BaseModel):
    """Full detection response"""
    detections: list[Detection]
    total_objects: int
    inference_time_ms: float
    image_width: int
    image_height: int
    annotated_image_base64: str    # Base64 encoded image with bounding boxes drawn

# ── Helper Functions ──────────────────────────────────────────────────────────

def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.
    Uses OpenCV for drawing — same library used in production CV systems.

    Color coding:
    - Person: Red
    - Vehicle: Blue
    - Other: Green
    """
    annotated = image.copy()

    # Color map for different categories
    colors = {
        "person": (0, 0, 255),      # Red for people
        "car": (255, 0, 0),         # Blue for vehicles
        "truck": (255, 0, 0),
        "bus": (255, 0, 0),
        "default": (0, 200, 100)    # Green for everything else
    }

    for det in detections:
        x1 = det.bbox["x1"]
        y1 = det.bbox["y1"]
        x2 = det.bbox["x2"]
        y2 = det.bbox["y2"]

        color = colors.get(det.label.lower(), colors["default"])

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label_text = f"{det.label} {det.confidence:.0%}"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - h - 8), (x1 + w + 4, y1), color, -1)

        # Draw label text
        cv2.putText(
            annotated, label_text,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1
        )

    return annotated


def image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image array to base64 string for API response"""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def run_detection(image: np.ndarray, confidence: float = CONFIDENCE_THRESHOLD) -> tuple:
    """
    Run YOLOv8 inference on an image.

    Returns:
    - List of Detection objects
    - Inference time in milliseconds

    YOLOv8 Pipeline:
    1. Preprocess: resize to 640x640, normalize
    2. Forward pass through CNN backbone
    3. Post-process: NMS (Non-Maximum Suppression) to remove duplicate boxes
    4. Return filtered detections above confidence threshold
    """
    start_time = time.time()

    # Run YOLO inference
    results = model(
        image,
        conf=confidence,       # Minimum confidence threshold
        iou=0.45,              # IoU threshold for NMS
        verbose=False
    )

    inference_time = (time.time() - start_time) * 1000   # Convert to ms

    detections = []
    h, w = image.shape[:2]

    # Parse YOLO results
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            # Extract bounding box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            label = model.names[class_id]

            detections.append(Detection(
                label=label,
                confidence=round(conf, 3),
                bbox={"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                bbox_normalized={
                    "x1": round(x1/w, 4), "y1": round(y1/h, 4),
                    "x2": round(x2/w, 4), "y2": round(y2/h, 4)
                }
            ))

    # Sort by confidence (highest first)
    detections.sort(key=lambda x: x.confidence, reverse=True)

    return detections, round(inference_time, 2)

# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    confidence: float = CONFIDENCE_THRESHOLD
):
    """
    Detect objects in an uploaded image.

    - Accepts: JPG, PNG, WEBP
    - Returns: detected objects with bounding boxes + annotated image
    - Confidence threshold: 0.0 - 1.0 (default 0.5)
    """

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted")

    # Validate confidence range
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0.0 and 1.0")

    # Read and decode image
    contents = await file.read()

    # Check file size
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large. Max size: {MAX_FILE_SIZE_MB}MB")

    # Convert bytes to OpenCV image
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    h, w = image.shape[:2]

    # Run detection
    detections, inference_time = run_detection(image, confidence)

    # Draw bounding boxes on image
    annotated_image = draw_detections(image, detections)
    annotated_b64 = image_to_base64(annotated_image)

    print(f"[INFO] Detected {len(detections)} objects in {inference_time}ms")

    return DetectionResponse(
        detections=detections,
        total_objects=len(detections),
        inference_time_ms=inference_time,
        image_width=w,
        image_height=h,
        annotated_image_base64=annotated_b64
    )


@app.get("/classes")
async def get_classes():
    """Return all 80 COCO classes that YOLOv8 can detect"""
    return {
        "total_classes": len(model.names),
        "classes": list(model.names.values())
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_PATH,
        "classes": len(model.names)
    }
