"""
OGB — OrbitalGuard
risk_service.py — real risk-score calculation.

This is the single authoritative implementation of the OGB risk formula.
It is called directly by the Copilot function-calling tool so the LLM never
performs the arithmetic itself — it only receives the result from here.

Formula (with units documented):
    Risk = min(1, (R / d_min) · log10(v_rel + 1) · exp(−Δt_epoch / 7))

    R            hard-body radius            [km]
    d_min        minimum separation          [km]
    v_rel        relative velocity           [km/s]
    Δt_epoch     TLE age                     [days]
"""
from __future__ import annotations

import math

from app.core.config import HARD_BODY_RADIUS_KM, RISK_THRESHOLDS


def calculate_risk_score(
    *,
    hard_body_radius_km: float = HARD_BODY_RADIUS_KM,
    d_min_km: float,
    v_rel_km_s: float,
    delta_t_epoch_days: float,
) -> float:
    """
    Compute the OGB risk priority score in [0, 1].

    d_min_km <= 0 is treated as a collision (returns 1.0).
    v_rel_km_s = 0 → log10(1) = 0 → score = 0 (no relative motion, no risk).
    """
    if d_min_km <= 0:
        return 1.0
    raw = (
        (hard_body_radius_km / d_min_km)
        * math.log10(v_rel_km_s + 1.0)
        * math.exp(-delta_t_epoch_days / 7.0)
    )
    return min(1.0, max(0.0, raw))


def categorise_risk(score: float) -> str:
    """
    Map a risk score to a category label using the config thresholds.
    Falls back to CRITICAL for any score > 1.0 (should not occur in practice).
    """
    for category, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= score <= hi:
            return category
    return "CRITICAL"


def calculate_risk_full(
    *,
    hard_body_radius_km: float = HARD_BODY_RADIUS_KM,
    d_min_km: float,
    v_rel_km_s: float,
    delta_t_epoch_days: float,
) -> dict:
    """
    Convenience wrapper that returns score + category together.
    Used by the Copilot tool executor so it returns one clean dict.
    """
    score = calculate_risk_score(
        hard_body_radius_km=hard_body_radius_km,
        d_min_km=d_min_km,
        v_rel_km_s=v_rel_km_s,
        delta_t_epoch_days=delta_t_epoch_days,
    )
    category = categorise_risk(score)
    return {
        "risk_score": round(score, 6),
        "risk_category": category,
        "hard_body_radius_km": hard_body_radius_km,
        "d_min_km": d_min_km,
        "v_rel_km_s": v_rel_km_s,
        "delta_t_epoch_days": delta_t_epoch_days,
        "formula": "min(1, (R/d_min) * log10(v_rel+1) * exp(-Δt/7))",
        "note": (
            "Risk score is a priority indicator, not a formally validated collision "
            "probability. Human operator makes all decisions."
        ),
    }
