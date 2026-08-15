"""
OGB — OrbitalGuard
Detection service — wraps YOLOv8n inference.

Loads the model once at module import (lazy singleton) so the first
/detect call pays the load cost and subsequent calls reuse the instance.
"""
from __future__ import annotations

import math
import time
import io
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from app.core.config import (
    CLASS_MAP50,
    CLASS_NAMES,
    MODEL_DESCRIPTION,
    MODEL_VERSION,
    MODEL_WEIGHTS_PATH,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IMAGE_SIZE,
    YOLO_IOU_THRESHOLD,
    YOLO_LOW_CONFIDENCE_THRESHOLD,
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
    copilot_ctx = _build_copilot_context(detections, img_w, img_h, round(latency_ms, 2))

    return DetectionResponse(
        image_width=img_w,
        image_height=img_h,
        detections=detections,
        detection_count=len(detections),
        inference_latency_ms=round(latency_ms, 2),
        summary=summary,
        copilot_context=copilot_ctx,
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


def _frame_position(cx: float, cy: float) -> str:
    """
    Map normalised centre coordinates to a plain-language quadrant label.
    Thresholds: 0.333 / 0.667 divide the frame into a 3×3 grid.
    """
    col = "left" if cx < 0.333 else ("right" if cx > 0.667 else "centre")
    row = "upper" if cy < 0.333 else ("lower" if cy > 0.667 else "mid")
    if row == "mid" and col == "centre":
        return "near centre"
    if row == "mid":
        return f"mid-{col}"
    return f"{row}-{col}"


def _build_copilot_context(
    detections: list[Detection],
    img_w: int,
    img_h: int,
    latency_ms: float,
) -> Dict[str, Any]:
    """
    Compute enriched, deterministic context for the AI copilot.

    Everything here is derived mathematically from detection outputs —
    no LLM inference, no speculative values.
    """
    det_list = []
    low_conf_flags: list[str] = []

    for d in detections:
        bb = d.bounding_box
        area_pct = round(bb.width * bb.height * 100, 1)
        position = _frame_position(bb.x_center, bb.y_center)
        class_map50 = CLASS_MAP50.get(d.class_name)
        low_conf = d.confidence < YOLO_LOW_CONFIDENCE_THRESHOLD

        entry: Dict[str, Any] = {
            "class_name": d.class_name,
            "confidence": round(d.confidence, 3),
            "confidence_advisory": (
                "LOW — below 0.50 threshold; recommend visual verification"
                if low_conf else "OK"
            ),
            "bbox_area_pct_of_frame": area_pct,
            "frame_position": position,
            "class_map50_test": class_map50,
            "class_map50_note": (
                f"This class achieved mAP@50={class_map50:.3f} in evaluation; "
                + (
                    "treat with extra caution — lower detector reliability for this class."
                    if class_map50 is not None and class_map50 < 0.70
                    else "detector reliability for this class is good."
                )
            ) if class_map50 is not None else None,
        }
        det_list.append(entry)
        if low_conf:
            low_conf_flags.append(d.class_name)

    # Spatial relationships between detections
    spatial_notes: list[str] = []
    if len(detections) >= 2:
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                a, b = detections[i].bounding_box, detections[j].bounding_box
                dist = math.sqrt((a.x_center - b.x_center) ** 2 + (a.y_center - b.y_center) ** 2)
                dist_pct = round(dist * 100, 1)
                spatial_notes.append(
                    f"{detections[i].class_name} and {detections[j].class_name} "
                    f"centres are ~{dist_pct}% of frame diagonal apart."
                )

    return {
        "model": {
            "version": MODEL_VERSION,
            "description": MODEL_DESCRIPTION,
            "confidence_threshold": YOLO_CONFIDENCE_THRESHOLD,
            "low_confidence_advisory_threshold": YOLO_LOW_CONFIDENCE_THRESHOLD,
            "total_classes": len(CLASS_NAMES),
            "class_list": CLASS_NAMES,
            "overall_map50": 0.8156,
        },
        "image": {
            "width_px": img_w,
            "height_px": img_h,
            "inference_latency_ms": latency_ms,
        },
        "detection_count": len(detections),
        "detections": det_list,
        "spatial_relationships": spatial_notes if spatial_notes else None,
        "low_confidence_detections": low_conf_flags if low_conf_flags else None,
        "data_provenance": (
            "All values above are computed deterministically from YOLOv8n output. "
            "No orbital data, distance, velocity, or collision probability is present. "
            "This is a visual detection only."
        ),
    }
