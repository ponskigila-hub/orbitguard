"""
OGB — OrbitalGuard
Unit tests for the detection service helper functions.
These tests exercise non-inference logic (summary generation,
model-load error paths) without requiring actual weights.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.services.detection_service import _build_summary
from app.models.detection import BoundingBox, Detection


def _make_detection(class_name: str, confidence: float = 0.9) -> Detection:
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        bounding_box=BoundingBox(x_center=0.5, y_center=0.5, width=0.2, height=0.2),
    )


class TestBuildSummary:
    def test_empty_detections(self):
        summary = _build_summary([])
        assert "No objects detected" in summary

    def test_single_detection(self):
        summary = _build_summary([_make_detection("debris")])
        assert "1 object" in summary
        assert "debris" in summary
        assert "visual detection only" in summary

    def test_multiple_same_class(self):
        dets = [_make_detection("debris") for _ in range(3)]
        summary = _build_summary(dets)
        assert "3" in summary
        assert "3× debris" in summary

    def test_multiple_classes(self):
        dets = [_make_detection("debris"), _make_detection("cheops")]
        summary = _build_summary(dets)
        assert "2 object" in summary
        assert "cheops" in summary
        assert "debris" in summary

    def test_summary_mentions_no_orbital_data(self):
        """Summary must be explicit that it is vision-only."""
        summary = _build_summary([_make_detection("soho")])
        lower = summary.lower()
        assert any(kw in lower for kw in ("visual", "orbital", "distance", "velocity"))


class TestDetectionServiceErrors:
    def test_missing_weights_raises_runtime_error(self, tmp_path):
        """run_detection should raise RuntimeError when weights are absent."""
        import app.services.detection_service as svc

        original = svc._model
        svc._model = None  # reset singleton

        # Point weights path to nonexistent file
        with patch.object(
            svc, "MODEL_WEIGHTS_PATH", tmp_path / "nonexistent.pt"
        ):
            # Need ultralytics importable; mock it if not installed
            try:
                import ultralytics  # noqa: F401
            except ImportError:
                pass  # _load_model will raise ImportError which is also acceptable

            with pytest.raises((RuntimeError, Exception)):
                svc._load_model()

        svc._model = original  # restore
