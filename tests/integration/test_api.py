"""
OGB — OrbitalGuard
Integration tests for the FastAPI endpoints.
Runs against the real app with the detection/copilot services mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.main import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Return a TestClient for the OGB FastAPI app."""
    # FastAPI 0.111+ ships its own testclient that uses httpx2
    from fastapi.testclient import TestClient as FATestClient
    with FATestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "OGB" in body["service"]


# ---------------------------------------------------------------------------
# /detect
# ---------------------------------------------------------------------------

TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
    b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


class TestDetectEndpoint:
    def _mock_detection_response(self):
        from app.models.detection import DetectionResponse
        return DetectionResponse(
            image_width=640,
            image_height=640,
            detections=[],
            detection_count=0,
            inference_latency_ms=12.5,
            summary="No objects detected in the image.",
        )

    def test_unsupported_media_type(self, client):
        r = client.post(
            "/api/v1/detect",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert r.status_code == 415

    def test_empty_file(self, client):
        r = client.post(
            "/api/v1/detect",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert r.status_code == 400

    def test_valid_image_with_mocked_inference(self, client):
        with patch(
            "app.api.v1.detect.run_detection",
            return_value=self._mock_detection_response(),
        ):
            r = client.post(
                "/api/v1/detect",
                files={"file": ("test.png", TINY_PNG, "image/png")},
            )
        assert r.status_code == 200
        body = r.json()
        assert "detections" in body
        assert "detection_count" in body
        assert body["model_version"] == "ogb_yolov8n_v1"

    def test_service_unavailable_on_runtime_error(self, client):
        with patch(
            "app.api.v1.detect.run_detection",
            side_effect=RuntimeError("weights missing"),
        ):
            r = client.post(
                "/api/v1/detect",
                files={"file": ("test.png", TINY_PNG, "image/png")},
            )
        assert r.status_code == 503
        assert "weights missing" in r.json()["detail"]


# ---------------------------------------------------------------------------
# /copilot
# ---------------------------------------------------------------------------

class TestCopilotEndpoint:
    def _mock_copilot_response(self):
        from app.models.copilot import CopilotResponse
        return CopilotResponse(
            reply="I detected no objects in the image.",
            provider="gemini",
            model="gemini-2.0-flash",
        )

    def test_copilot_basic_request(self, client):
        with patch(
            "app.api.v1.copilot.run_copilot",
            return_value=self._mock_copilot_response(),
        ):
            r = client.post(
                "/api/v1/copilot",
                json={"message": "What did OGB detect?"},
            )
        assert r.status_code == 200
        body = r.json()
        assert "reply" in body
        assert body["provider"] == "gemini"

    def test_copilot_with_detection_context(self, client):
        with patch(
            "app.api.v1.copilot.run_copilot",
            return_value=self._mock_copilot_response(),
        ):
            r = client.post(
                "/api/v1/copilot",
                json={
                    "message": "Summarise the threat.",
                    "detection_context": {
                        "detection_count": 2,
                        "summary": "2 objects detected: 1× debris, 1× cheops.",
                    },
                },
            )
        assert r.status_code == 200

    def test_copilot_service_error(self, client):
        with patch(
            "app.api.v1.copilot.run_copilot",
            side_effect=RuntimeError("GEMINI_API_KEY not set"),
        ):
            r = client.post(
                "/api/v1/copilot",
                json={"message": "Hello"},
            )
        assert r.status_code == 503
