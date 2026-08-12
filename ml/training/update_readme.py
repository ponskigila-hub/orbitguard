"""
OGB — OrbitalGuard
ml/training/update_readme.py

Run this after training to fill in real metrics in README.md.

Usage:
    python ml/training/update_readme.py [--metrics ml/weights/metrics.json]

Reads metrics.json (produced by the Colab notebook cell-07-export) and
replaces the placeholder rows in the README evaluation table with real numbers.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = REPO_ROOT / "ml" / "weights" / "metrics.json"
README = REPO_ROOT / "README.md"


def load_metrics(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def update_readme(metrics: dict, readme_path: Path) -> None:
    text = readme_path.read_text(encoding="utf-8")

    replacements = {
        r"\| \*\*mAP@50\*\* \|.*?\|":
            f"| **mAP@50** | {metrics['map50']} |",
        r"\| \*\*mAP@50-95\*\* \|.*?\|":
            f"| **mAP@50-95** | {metrics['map5095']} |",
        r"\| \*\*Precision\*\* \|.*?\|":
            f"| **Precision** | {metrics['precision']} |",
        r"\| \*\*Recall\*\* \|.*?\|":
            f"| **Recall** | {metrics['recall']} |",
        r"\| \*\*F1\*\* \|.*?\|":
            f"| **F1** | {metrics['f1']} |",
        r"\| \*\*CPU inference latency.*?\|.*?\|":
            f"| **CPU inference latency (mean)** | {metrics['cpu_latency_mean_ms']} ms "
            f"(median {metrics['cpu_latency_median_ms']} ms, p95 {metrics['cpu_latency_p95_ms']} ms) |",
    }

    changed = 0
    for pattern, replacement in replacements.items():
        new_text, n = re.subn(pattern, replacement, text)
        if n:
            text = new_text
            changed += n

    readme_path.write_text(text, encoding="utf-8")
    print(f"README updated: {changed} row(s) replaced in {readme_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args()

    if not args.metrics.exists():
        print(f"[ERROR] metrics.json not found at {args.metrics}")
        print("Run the Colab notebook to generate it, then download to ml/weights/")
        sys.exit(1)

    metrics = load_metrics(args.metrics)
    print("Loaded metrics:", json.dumps(metrics, indent=2))
    update_readme(metrics, README)
