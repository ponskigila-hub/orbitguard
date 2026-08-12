"""
OGB — OrbitalGuard
Detection service — wraps YOLOv8n inference.

Loads the model once at module import (lazy singleton) so the first
/detect call pays the load cost and subsequent calls reuse the instance.
"""
from __future__ import annotations

import time
import io
from pathlib import Path
from typing import Optional

from PIL import Image

from app.core.config import (
    CLASS_NAMES,
    MODEL_WEIGHTS_PATH,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IMAGE_SIZE,
    YOLO_IOU_THRESHOLD,
    YOLO_MAX_DETECTIONS,
)
from app.models.detection import BoundingBox, Detection, DetectionResponse

_model = None  # lazy singleton


def _load_model():
    """Load YOLOv8n weights. Raises RuntimeError if weights are missing."""
    global _model
    if _model is not None:
        return _model
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Run: pip install ultralytics"
        ) from exc

    if not MODEL_WEIGHTS_PATH.exists():
        raise RuntimeError(
            f"Model weights not found at {MODEL_WEIGHTS_PATH}. "
            "Train the model first using ml/training/train.py or the "
            "Colab notebook at ml/training/ogb_train_colab.ipynb, "
            "then copy the best.pt to ml/weights/ogb_yolov8n.pt."
        )
    _model = YOLO(str(MODEL_WEIGHTS_PATH))
    return _model


def run_detection(image_bytes: bytes) -> DetectionResponse:
    """
    Run YOLOv8n inference on raw image bytes.

    Returns a DetectionResponse with bounding boxes, class names, and
    confidence scores.  Does NOT return distance, velocity, or any orbital
    data — those come from the orbital pipeline only.
    """
    model = _load_model()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = image.size

    t0 = time.perf_counter()
    results = model.predict(
        source=image,
        imgsz=YOLO_IMAGE_SIZE,
        conf=YOLO_CONFIDENCE_THRESHOLD,
        iou=YOLO_IOU_THRESHOLD,
        max_det=YOLO_MAX_DETECTIONS,
        verbose=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    detections: list[Detection] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
            conf = float(box.conf[0].item())
            # xyxy pixel coords
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            # Normalised YOLO format
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bounding_box=BoundingBox(
                        x_center=cx, y_center=cy, width=bw, height=bh
                    ),
                    x1_px=x1,
                    y1_px=y1,
                    x2_px=x2,
                    y2_px=y2,
                )
            )

    summary = _build_summary(detections)

    return DetectionResponse(
        image_width=img_w,
        image_height=img_h,
        detections=detections,
        detection_count=len(detections),
        inference_latency_ms=round(latency_ms, 2),
        summary=summary,
    )


def _build_summary(detections: list[Detection]) -> str:
    """Plain-English summary passed to the AI copilot as context."""
    if not detections:
        return "No objects detected in the image."
    counts: dict[str, int] = {}
    for d in detections:
        counts[d.class_name] = counts.get(d.class_name, 0) + 1
    parts = [f"{v}× {k}" for k, v in sorted(counts.items())]
    return (
        f"Detected {len(detections)} object(s): {', '.join(parts)}. "
        "Note: this is a visual detection only — no distance, velocity, "
        "or orbital data is available from the image alone."
    )
