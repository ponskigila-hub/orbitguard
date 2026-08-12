"""
OGB — OrbitalGuard
POST /api/v1/detect  — image upload → YOLOv8n inference → detection response.
"""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import MAX_UPLOAD_SIZE_BYTES
from app.models.detection import DetectionResponse
from app.services.detection_service import run_detection

router = APIRouter(prefix="/detect", tags=["Detection"])

_ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


@router.post(
    "",
    response_model=DetectionResponse,
    summary="Analyse a spacecraft camera image for debris and known objects",
    description=(
        "Upload an image (JPEG, PNG, WebP, BMP, or TIFF). "
        "Returns bounding boxes, class names, and confidence scores. "
        "**Vision output only** — does not include distance, velocity, or "
        "collision probability. Those require orbital data from the V2 pipeline."
    ),
)
async def detect_objects(
    file: UploadFile = File(..., description="Spacecraft camera image"),
) -> DetectionResponse:
    # --- Content-type check ---
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Accepted: {sorted(_ALLOWED_CONTENT_TYPES)}"
            ),
        )

    # --- Size check ---
    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {len(image_bytes):,} bytes exceeds the "
                f"{MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB limit."
            ),
        )

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # --- Inference ---
    try:
        result = run_detection(image_bytes)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {exc}",
        ) from exc

    return result
