"""
OGB — OrbitalGuard
POST /api/v1/orbital/analyze  — TLE propagation + optional conjunction analysis.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from app.models.orbital import OrbitalAnalyzeRequest, OrbitalAnalyzeResponse
from app.services.orbital_service import propagate_tle_state
from app.services.conjunction_service import analyze_conjunction

router = APIRouter(prefix="/orbital", tags=["Orbital"])


@router.post(
    "/analyze",
    response_model=OrbitalAnalyzeResponse,
    summary="Propagate a TLE and optionally run conjunction analysis",
    description=(
        "Supply TLE line 1 and line 2 to get the SGP4-propagated ECI state "
        "(position + velocity) at the requested timestamp. "
        "Optionally supply target TLE lines to run close-approach analysis: "
        "the service finds the Time of Closest Approach (TCA), minimum "
        "separation, relative velocity, and risk score within the specified "
        "window. **Vision output and orbital output are separate pipelines** — "
        "this endpoint does not interpret camera images."
    ),
)
async def analyze_orbital(request: OrbitalAnalyzeRequest) -> OrbitalAnalyzeResponse:
    # Default timestamp to current UTC
    ts = request.timestamp_utc
    if ts is None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    # ── Propagation ──────────────────────────────────────────────────────────
    try:
        state = propagate_tle_state(request.tle_line1, request.tle_line2, ts)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Propagation error: {exc}",
        ) from exc

    # ── Conjunction (optional) ───────────────────────────────────────────────
    conjunction_result = None
    analysis_type = "propagation_only"

    has_target = bool(request.target_tle_line1 and request.target_tle_line2)
    if has_target:
        analysis_type = "propagation_and_conjunction"
        try:
            conj = analyze_conjunction(
                sat1_line1=request.tle_line1,
                sat1_line2=request.tle_line2,
                sat2_line1=request.target_tle_line1,
                sat2_line2=request.target_tle_line2,
                t_start_utc=ts,
                window_hours=request.conjunction_window_hours,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Conjunction analysis error: {exc}",
            ) from exc

        # Map service output to Pydantic model
        conjunction_result = {
            "ok": conj.get("ok", False),
            "tca_utc": conj.get("tca_utc"),
            "d_min_km": conj.get("d_min_km"),
            "v_rel_km_s": conj.get("v_rel_km_s"),
            "position_sat1_km": conj.get("position_sat1_km"),
            "position_sat2_km": conj.get("position_sat2_km"),
            "tle_age_days_sat1": conj.get("tle_age_days_sat1"),
            "tle_age_days_sat2": conj.get("tle_age_days_sat2"),
            "risk": conj.get("risk"),
            "coarse_samples": conj.get("coarse_samples"),
            "window_hours": conj.get("window_hours"),
            "error": conj.get("error"),
        }

    return OrbitalAnalyzeResponse(
        propagated_state=state,
        conjunction=conjunction_result,
        analysis_type=analysis_type,
    )
