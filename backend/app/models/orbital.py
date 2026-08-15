"""
OGB — OrbitalGuard
Pydantic request/response models for the Orbital Intelligence API (V2).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class OrbitalAnalyzeRequest(BaseModel):
    """
    Payload for POST /api/v1/orbital/analyze.

    At minimum supply tle_line1 + tle_line2 for propagation.
    Supply target_tle_line1/2 as well to trigger conjunction analysis.
    """
    tle_line1: str = Field(
        ...,
        description="TLE line 1 of the primary object (exactly 69 chars).",
        min_length=69,
        max_length=69,
    )
    tle_line2: str = Field(
        ...,
        description="TLE line 2 of the primary object (exactly 69 chars).",
        min_length=69,
        max_length=69,
    )
    timestamp_utc: Optional[str] = Field(
        default=None,
        description=(
            "ISO-8601 UTC timestamp to propagate to. "
            "Defaults to current UTC time if not supplied."
        ),
    )
    # Conjunction fields (optional)
    target_tle_line1: Optional[str] = Field(
        default=None,
        description="TLE line 1 of the secondary object (for conjunction analysis).",
        min_length=69,
        max_length=69,
    )
    target_tle_line2: Optional[str] = Field(
        default=None,
        description="TLE line 2 of the secondary object (for conjunction analysis).",
        min_length=69,
        max_length=69,
    )
    conjunction_window_hours: float = Field(
        default=24.0,
        description="Analysis window for conjunction search [hours]. Default 24.",
        ge=0.5,
        le=168.0,  # 1 week max
    )


# ---------------------------------------------------------------------------
# Sub-models for responses
# ---------------------------------------------------------------------------

class PropagatedState(BaseModel):
    ok: bool
    position_km: Optional[List[float]] = None
    velocity_km_s: Optional[List[float]] = None
    epoch_utc: Optional[str] = None
    tle_age_days: Optional[float] = None
    propagated_at_utc: Optional[str] = None
    error: Optional[str] = None


class ConjunctionResult(BaseModel):
    ok: bool
    tca_utc: Optional[str] = None
    d_min_km: Optional[float] = None
    v_rel_km_s: Optional[float] = None
    position_sat1_km: Optional[List[float]] = None
    position_sat2_km: Optional[List[float]] = None
    tle_age_days_sat1: Optional[float] = None
    tle_age_days_sat2: Optional[float] = None
    risk: Optional[Dict[str, Any]] = None
    coarse_samples: Optional[int] = None
    window_hours: Optional[float] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class OrbitalAnalyzeResponse(BaseModel):
    """
    Full response from POST /api/v1/orbital/analyze.
    """
    propagated_state: PropagatedState
    conjunction: Optional[ConjunctionResult] = Field(
        default=None,
        description="Present only when target TLE lines were supplied.",
    )
    analysis_type: str = Field(
        description="'propagation_only' or 'propagation_and_conjunction'",
    )
