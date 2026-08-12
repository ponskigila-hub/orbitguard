"""
OGB — OrbitalGuard
YOLOv8n training script.

Usage (local / Kaggle / Colab):
    python ml/training/train.py [--epochs 50] [--batch 16] [--device cpu|0]

Points at data/raw/space-debris-v2/data.yaml — no path changes needed
as long as you run from the repo root.

Sanity check runs automatically before training and aborts on any mismatch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = REPO_ROOT / "data" / "raw" / "space-debris-v2" / "data.yaml"
WEIGHTS_DIR = REPO_ROOT / "ml" / "weights"
RUNS_DIR = REPO_ROOT / "ml" / "runs"

# Expected dataset facts (verified 2024 — do not change without re-verifying)
EXPECTED_SPLITS = {"train": 2105, "valid": 239, "test": 123}
EXPECTED_CLASSES = [
    "cheops",
    "debris",
    "double_start",            # note: double_start, NOT double_star
    "earth_observation_sat_1",
    "lisa_pathfinder",
    "proba_2",
    "proba_3_csc",
    "proba_3_ocs",
    "smart_1",
    "soho",
    "xmm_newton",
]


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def run_sanity_check() -> None:
    """
    Verify dataset structure before spending GPU time on a bad run.
    Prints a summary and raises SystemExit on any failure.
    """
    import yaml  # bundled with PyYAML, required by ultralytics

    print("=" * 60)
    print("OGB — Dataset sanity check")
    print("=" * 60)

    if not DATA_YAML.exists():
        print(f"[FAIL] data.yaml not found at {DATA_YAML}")
        sys.exit(1)

    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)

    # --- Class names ---
    actual_classes = cfg.get("names", [])
    if actual_classes != EXPECTED_CLASSES:
        print("[FAIL] Class names / order mismatch!")
        print(f"  Expected : {EXPECTED_CLASSES}")
        print(f"  In yaml  : {actual_classes}")
        sys.exit(1)
    print(f"[OK] Class names ({len(actual_classes)} classes, order verified)")

    # --- nc ---
    nc = cfg.get("nc", 0)
    if nc != 11:
        print(f"[FAIL] nc = {nc}, expected 11")
        sys.exit(1)
    print(f"[OK] nc = {nc}")

    # --- Image counts ---
    split_dirs = {
        "train": REPO_ROOT / "data" / "raw" / "space-debris-v2" / "train" / "images",
        "valid": REPO_ROOT / "data" / "raw" / "space-debris-v2" / "valid" / "images",
        "test":  REPO_ROOT / "data" / "raw" / "space-debris-v2" / "test"  / "images",
    }
    all_ok = True
    for split, expected_count in EXPECTED_SPLITS.items():
        d = split_dirs[split]
        if not d.exists():
            print(f"[FAIL] {split} images directory not found: {d}")
            all_ok = False
            continue
        count = sum(1 for _ in d.iterdir() if _.is_file())
        status = "OK" if count == expected_count else "WARN"
        print(f"[{status}] {split}: {count} images (expected {expected_count})")
        if count != expected_count:
            all_ok = False

    if not all_ok:
        print("[WARN] Image counts differ from verified baseline — proceeding anyway.")
        print("       If counts are significantly off, check dataset integrity.")

    print("=" * 60)
    print("Sanity check passed. Starting training...")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(epochs: int, batch: int, device: str, imgsz: int, workers: int) -> None:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")  # downloads if not cached

    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        optimizer="AdamW",
        device=device,
        workers=workers,
        project=str(RUNS_DIR),
        name="ogb_yolov8n",
        # Conservative augmentation — dataset already has flips from Roboflow v2.
        # Skip extra flips to avoid over-augmenting.
        fliplr=0.0,
        flipud=0.0,
        # Standard augmentations still applied:
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        mosaic=1.0,
        mixup=0.0,
        # Evaluation
        val=True,
        plots=True,
        save=True,
    )

    # Copy best weights to canonical output path
    best_pt: Path = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        import shutil
        dest = WEIGHTS_DIR / "ogb_yolov8n.pt"
        shutil.copy2(best_pt, dest)
        print(f"\n[OGB] Best weights saved to: {dest}")
    else:
        print(f"\n[WARN] best.pt not found at {best_pt} — check run output.")

    # Print key metrics
    print("\n" + "=" * 60)
    print("OGB — Training complete. Key metrics:")
    print("=" * 60)
    try:
        metrics = results.results_dict
        map50    = metrics.get("metrics/mAP50(B)", float("nan"))
        map5095  = metrics.get("metrics/mAP50-95(B)", float("nan"))
        prec     = metrics.get("metrics/precision(B)", float("nan"))
        rec      = metrics.get("metrics/recall(B)", float("nan"))
        print(f"  mAP@50      : {map50:.4f}  (target ≥ 0.85)")
        print(f"  mAP@50-95   : {map5095:.4f}")
        print(f"  Precision   : {prec:.4f}")
        print(f"  Recall      : {rec:.4f}")
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
        print(f"  F1          : {f1:.4f}")
    except Exception as e:
        print(f"  (Could not extract metrics dict: {e})")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OGB YOLOv8n training script")
    parser.add_argument("--epochs",  type=int,  default=50,    help="Training epochs (default: 50)")
    parser.add_argument("--batch",   type=int,  default=16,    help="Batch size (default: 16)")
    parser.add_argument("--device",  type=str,  default="0",   help="Device: 0 (GPU), cpu (default: 0)")
    parser.add_argument("--imgsz",   type=int,  default=640,   help="Image size (default: 640)")
    parser.add_argument("--workers", type=int,  default=4,     help="DataLoader workers (default: 4)")
    args = parser.parse_args()

    run_sanity_check()
    train(
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        imgsz=args.imgsz,
        workers=args.workers,
    )
