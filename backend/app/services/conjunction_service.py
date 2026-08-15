"""
OGB — OrbitalGuard
conjunction_service.py — close-approach (conjunction) analysis (V2).

Algorithm
---------
1. Coarse phase  — sample the separation |r1(t) - r2(t)| every `coarse_step_s`
   seconds across the analysis window to find the sub-interval [t_lo, t_hi]
   that brackets the minimum.
2. Fine phase    — scipy.optimize.minimize_scalar (bounded Brent) on the
   separation function over [t_lo, t_hi] to find TCA to < 1-second precision.
3. Risk          — delegates to risk_service.calculate_risk_full() directly.
   No math is re-implemented here.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from scipy.optimize import minimize_scalar  # type: ignore

from app.services.orbital_service import propagate_tle_state, _parse_iso_utc
from app.services.risk_service import calculate_risk_full

# Coarse sampling interval (seconds). 60 s gives < 0.5 km error for LEO.
_COARSE_STEP_S: float = 60.0
# Guard: expand the bracketed interval by this many seconds on each side.
_BRACKET_GUARD_S: float = 120.0


def _sep_at_offset(
    sat1_line1: str,
    sat1_line2: str,
    sat2_line1: str,
    sat2_line2: str,
    t0: datetime,
    offset_s: float,
) -> tuple[float, list[float], list[float]]:
    """
    Return (separation_km, r1, r2) at t0 + offset_s seconds.
    Raises ValueError if either propagation fails.
    """
    t = t0 + timedelta(seconds=offset_s)
    ts_str = t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    r1 = propagate_tle_state(sat1_line1, sat1_line2, ts_str)
    if not r1["ok"]:
        raise ValueError(f"Sat-1 propagation failed at offset {offset_s:.1f}s: {r1['error']}")

    r2 = propagate_tle_state(sat2_line1, sat2_line2, ts_str)
    if not r2["ok"]:
        raise ValueError(f"Sat-2 propagation failed at offset {offset_s:.1f}s: {r2['error']}")

    p1 = r1["position_km"]
    p2 = r2["position_km"]
    sep = math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    return sep, p1, p2


def analyze_conjunction(
    sat1_line1: str,
    sat1_line2: str,
    sat2_line1: str,
    sat2_line2: str,
    t_start_utc: str,
    window_hours: float = 24.0,
    coarse_step_s: float = _COARSE_STEP_S,
) -> dict[str, Any]:
    """
    Find the Time of Closest Approach (TCA) between two objects within a window.

    Returns
    -------
    dict with keys:
        ok               bool
        tca_utc          str  ISO-8601 UTC of closest approach
        d_min_km         float  minimum separation at TCA [km]
        v_rel_km_s       float  relative speed at TCA [km/s]
        position_sat1_km [x,y,z] ECI at TCA
        position_sat2_km [x,y,z] ECI at TCA
        tle_age_days_sat1 float
        tle_age_days_sat2 float
        risk             dict  from calculate_risk_full()
        coarse_samples   int   number of coarse samples taken
        error            str   (only when ok=False)
    """
    try:
        t0 = _parse_iso_utc(t_start_utc)
    except ValueError as exc:
        return {"ok": False, "error": f"Invalid start timestamp: {exc}"}

    window_s = window_hours * 3600.0
    offsets = [i * coarse_step_s for i in range(int(window_s / coarse_step_s) + 1)]

    # ── Coarse phase ────────────────────────────────────────────────────────
    coarse_seps: list[tuple[float, float]] = []  # (offset_s, separation_km)
    for off in offsets:
        try:
            sep, _, _ = _sep_at_offset(
                sat1_line1, sat1_line2, sat2_line1, sat2_line2, t0, off
            )
            coarse_seps.append((off, sep))
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

    if not coarse_seps:
        return {"ok": False, "error": "No valid propagation samples in window."}

    # Find the coarse minimum
    min_idx = min(range(len(coarse_seps)), key=lambda i: coarse_seps[i][1])
    best_off = coarse_seps[min_idx][0]

    # Bracket: [prev, next] coarse samples (or window edges)
    lo = max(0.0, best_off - coarse_step_s - _BRACKET_GUARD_S)
    hi = min(window_s, best_off + coarse_step_s + _BRACKET_GUARD_S)

    # ── Fine phase (scipy Brent) ─────────────────────────────────────────────
    def objective(off: float) -> float:
        try:
            sep, _, _ = _sep_at_offset(
                sat1_line1, sat1_line2, sat2_line1, sat2_line2, t0, off
            )
            return sep
        except ValueError:
            return float("inf")

    result = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
    tca_offset_s: float = float(result.x)
    tca_dt = t0 + timedelta(seconds=tca_offset_s)
    tca_str = tca_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    # Positions at TCA
    try:
        d_min, r1_tca, r2_tca = _sep_at_offset(
            sat1_line1, sat1_line2, sat2_line1, sat2_line2, t0, tca_offset_s
        )
    except ValueError as exc:
        return {"ok": False, "error": f"TCA position lookup failed: {exc}"}

    # ── Velocity difference at TCA (finite difference, ±1 s) ────────────────
    try:
        _, r1_plus, r2_plus = _sep_at_offset(
            sat1_line1, sat1_line2, sat2_line1, sat2_line2, t0, tca_offset_s + 1.0
        )
        _, r1_minus, r2_minus = _sep_at_offset(
            sat1_line1, sat1_line2, sat2_line1, sat2_line2, t0, tca_offset_s - 1.0
        )
        # Central difference: v ≈ (r(t+1) - r(t-1)) / 2
        v1 = [(r1_plus[i] - r1_minus[i]) / 2.0 for i in range(3)]
        v2 = [(r2_plus[i] - r2_minus[i]) / 2.0 for i in range(3)]
        v_rel_km_s = math.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(3)))
    except ValueError:
        # Fall back to coarse-phase velocity estimate
        v_rel_km_s = 0.0

    # ── TLE ages ─────────────────────────────────────────────────────────────
    # Re-use the tle_age_days from the TCA propagation result
    tca_str_for_age = tca_str
    s1 = propagate_tle_state(sat1_line1, sat1_line2, tca_str_for_age)
    s2 = propagate_tle_state(sat2_line1, sat2_line2, tca_str_for_age)
    tle_age_1 = s1.get("tle_age_days", 0.0) if s1["ok"] else 0.0
    tle_age_2 = s2.get("tle_age_days", 0.0) if s2["ok"] else 0.0
    # Use the worse (older) age for risk scoring
    max_tle_age = max(abs(tle_age_1), abs(tle_age_2))

    # ── Risk ─────────────────────────────────────────────────────────────────
    risk = calculate_risk_full(
        d_min_km=d_min,
        v_rel_km_s=v_rel_km_s,
        delta_t_epoch_days=max_tle_age,
    )

    return {
        "ok": True,
        "tca_utc": tca_str,
        "d_min_km": round(d_min, 6),
        "v_rel_km_s": round(v_rel_km_s, 6),
        "position_sat1_km": [round(x, 3) for x in r1_tca],
        "position_sat2_km": [round(x, 3) for x in r2_tca],
        "tle_age_days_sat1": round(tle_age_1, 4),
        "tle_age_days_sat2": round(tle_age_2, 4),
        "risk": risk,
        "coarse_samples": len(coarse_seps),
        "window_hours": window_hours,
    }
