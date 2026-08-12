"""
OGB — OrbitalGuard
POST /api/v1/copilot  — operator ↔ AI Copilot chat endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.models.copilot import CopilotRequest, CopilotResponse
from app.services.copilot_service import run_copilot

router = APIRouter(prefix="/copilot", tags=["Copilot"])


@router.post(
    "",
    response_model=CopilotResponse,
    summary="Send a message to the OGB AI Copilot",
    description=(
        "The copilot receives structured detection, orbital, and risk context "
        "alongside the operator's message. It never calculates orbital mechanics "
        "or invents missing values — it explains and prioritises based on the "
        "structured data provided."
    ),
)
async def copilot_chat(request: CopilotRequest) -> CopilotResponse:
    try:
        return run_copilot(request)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Copilot error: {exc}",
        ) from exc
