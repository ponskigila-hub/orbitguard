"""
OGB — OrbitalGuard
ml/inference/smoke_test.py

End-to-end inference smoke test — verifies the full pipeline:
  real weights → PIL image → YOLOv8n → bounding boxes + classes + confidence

Run this once ml/weights/ogb_yolov8n.pt is in place:
    python ml/inference/smoke_test.py [--image path/to/image.jpg]

If no --image is given, picks a random image from the test split.
Prints a clear PASS / FAIL verdict and details of every detection.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_IMAGES_DIR = REPO_ROOT / "data" / "raw" / "space-debris-v2" / "test" / "images"
WEIGHTS_PATH = REPO_ROOT / "ml" / "weights" / "ogb_yolov8n.pt"

CLASS_NAMES = [
    "cheops", "debris", "double_start", "earth_observation_sat_1",
    "lisa_pathfinder", "proba_2", "proba_3_csc", "proba_3_ocs",
    "smart_1", "soho", "xmm_newton",
]


def run_smoke_test(image_path: Path) -> None:
    print("=" * 60)
    print("OGB — Inference smoke test")
    print("=" * 60)

    # ── Pre-flight checks ────────────────────────────────────────────────────
    if not WEIGHTS_PATH.exists():
        print(f"[FAIL] Weights not found: {WEIGHTS_PATH}")
        print("       Download ogb_yolov8n.pt from Drive and place it here.")
        sys.exit(1)

    if not image_path.exists():
        print(f"[FAIL] Image not found: {image_path}")
        sys.exit(1)

    try:
        from ultralytics import YOLO  # type: ignore
        from PIL import Image
    except ImportError as e:
        print(f"[FAIL] Missing dependency: {e}")
        print("       Run: pip install ultralytics pillow")
        sys.exit(1)

    print(f"Weights : {WEIGHTS_PATH}")
    print(f"Image   : {image_path}")
    print()

    # ── Load model ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    model = YOLO(str(WEIGHTS_PATH))
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Model load     : {load_ms:.0f} ms")

    # ── Load image ───────────────────────────────────────────────────────────
    image = Image.open(image_path).convert("RGB")
    print(f"Image size     : {image.size[0]}×{image.size[1]} px")

    # ── Inference ────────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    results = model.predict(
        source=image,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        verbose=False,
    )
    infer_ms = (time.perf_counter() - t1) * 1000
    print(f"Inference time : {infer_ms:.1f} ms (CPU)")

    # ── Parse detections ─────────────────────────────────────────────────────
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
            })

    print()
    print(f"Detections     : {len(detections)}")
    print("-" * 60)

    if detections:
        for i, d in enumerate(detections, 1):
            x1, y1, x2, y2 = d["bbox"]
            print(
                f"  [{i}] {d['class_name']:<30} conf={d['confidence']:.3f}  "
                f"bbox=[{x1},{y1},{x2},{y2}]"
            )
    else:
        print("  (no objects detected above conf=0.25 threshold)")
        print("  This is valid — not every test image has annotations.")

    print("-" * 60)
    # A result is a PASS as long as the model loaded, ran, and returned a
    # well-formed response. Zero detections on a single image is not a failure.
    print("\n[PASS] Inference pipeline is working end-to-end.")
    print("       Real weights → PIL image → YOLOv8n → structured output ✓")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OGB inference smoke test")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to test image (default: random from test split)",
    )
    args = parser.parse_args()

    if args.image:
        image_path = args.image
    else:
        images = sorted(TEST_IMAGES_DIR.glob("*.jpg")) + sorted(TEST_IMAGES_DIR.glob("*.png"))
        if not images:
            print(f"[FAIL] No images found in {TEST_IMAGES_DIR}")
            sys.exit(1)
        image_path = random.choice(images)

    run_smoke_test(image_path)
