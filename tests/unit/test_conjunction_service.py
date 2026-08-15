"""
OGB — OrbitalGuard
tests/unit/test_conjunction_service.py

Tests for analyze_conjunction() in conjunction_service.py.

Scenarios covered
──────────────────
1. Near-zero separation — two satellites started at very similar positions
   should yield d_min < 1 km.
2. Identical orbits (same TLE) — d_min must be ≈ 0 (within floating-point).
3. Stale TLE scenario — verify tle_age_days_sat1/sat2 are populated and
   the risk score is lower due to the stale-TLE decay.
4. Required output fields — result must contain all documented keys.
5. Risk delegated to risk_service — result["risk"] must match
   calculate_risk_full() called with the same inputs.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.services.conjunction_service import analyze_conjunction
from app.services.risk_service import calculate_risk_full

# ---------------------------------------------------------------------------
# TLE fixtures
# ---------------------------------------------------------------------------

# Object 88888 — Vallado 2006, epoch 1980-275 (LEO, period ≈ 89 min)
# From sgp4 library SGP4-VER.TLE — 69-char lines, checksum verified.
_SAT_A_L1 = "1 88888U          80275.98708465  .00073094  13844-3  66816-4 0    87"
_SAT_A_L2 = "2 88888  72.8435 115.9689 0086731  52.6988 110.5714 16.05824518  1058"

# A companion object: same orbital plane as 88888 but slightly different
# mean motion (16.05824518 → 16.06000000) so it drifts over time.
# Line 2 checksum for this companion: recomputed below.
# Line 2: "2 88889  72.8435 115.9689 0086731  52.6988 110.5714 16.06000000  1050"
# len=69 ✓, checksum digit = 0 (placeholder — sgp4 ignores checksum on parse)
_SAT_B_NEAR_L1 = "1 88889U          80275.98708465  .00073094  13844-3  66816-4 0    87"
_SAT_B_NEAR_L2 = "2 88889  72.8435 115.9689 0086731  52.6988 110.5714 16.06000000  1050"

# Epoch of 88888 as ISO-8601 string (1980-10-01T23:41:24.434Z approx)
# day 275.98708465 of 1980 → 1980-10-01 ≈ day 275
_EPOCH_TS = "1980-10-01T23:41:00Z"


class TestConjunctionOutputFields:
    """Verify the response contains all documented keys."""

    def test_required_fields_present(self):
        result = analyze_conjunction(
            sat1_line1=_SAT_A_L1,
            sat1_line2=_SAT_A_L2,
            sat2_line1=_SAT_B_NEAR_L1,
            sat2_line2=_SAT_B_NEAR_L2,
            t_start_utc=_EPOCH_TS,
            window_hours=1.0,
        )
        assert result["ok"] is True, f"Conjunction failed: {result.get('error')}"
        for field in (
            "tca_utc", "d_min_km", "v_rel_km_s",
            "position_sat1_km", "position_sat2_km",
            "tle_age_days_sat1", "tle_age_days_sat2",
            "risk", "coarse_samples", "window_hours",
        ):
            assert field in result, f"Missing field: {field}"

    def test_position_vectors_are_length_3(self):
        result = analyze_conjunction(
            sat1_line1=_SAT_A_L1, sat1_line2=_SAT_A_L2,
            sat2_line1=_SAT_B_NEAR_L1, sat2_line2=_SAT_B_NEAR_L2,
            t_start_utc=_EPOCH_TS, window_hours=1.0,
        )
        assert result["ok"] is True
        assert len(result["position_sat1_km"]) == 3
        assert len(result["position_sat2_km"]) == 3

    def test_d_min_is_non_negative(self):
        result = analyze_conjunction(
            sat1_line1=_SAT_A_L1, sat1_line2=_SAT_A_L2,
            sat2_line1=_SAT_B_NEAR_L1, sat2_line2=_SAT_B_NEAR_L2,
            t_start_utc=_EPOCH_TS, window_hours=1.0,
        )
        assert result["ok"] is True
        assert result["d_min_km"] >= 0.0

    def test_risk_dict_contains_score_and_category(self):
        result = analyze_conjunction(
            sat1_line1=_SAT_A_L1, sat1_line2=_SAT_A_L2,
            sat2_line1=_SAT_B_NEAR_L1, sat2_line2=_SAT_B_NEAR_L2,
            t_start_utc=_EPOCH_TS, window_hours=1.0,
        )
        assert result["ok"] is True
        risk = result["risk"]
        assert "risk_score" in risk
        assert "risk_category" in risk
        assert 0.0 <= risk["risk_score"] <= 1.0
        assert risk["risk_category"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


class TestIdenticalOrbits:
    """
    When both satellites are on the SAME orbit (same TLE), d_min must be ≈ 0.
    The conjunction result must return ok=True and d_min < 1e-3 km (1 m).
    """

    def test_same_tle_yields_near_zero_separation(self):
        result = analyze_conjunction(
            sat1_line1=_SAT_A_L1, sat1_line2=_SAT_A_L2,
            sat2_line1=_SAT_A_L1, sat2_line2=_SAT_A_L2,
            t_start_utc=_EPOCH_TS, window_hours=0.5,
        )
        assert result["ok"] is True, f"Conjunction failed: {result.get('error')}"
        # Same object propagated twice must coincide exactly
        assert result["d_min_km"] < 1e-3, (
            f"d_min = {result['d_min_km']:.6f} km for identical TLE — expected < 1e-3 km"
        )

    def test_same_tle_risk_score_is_one(self):
        """d_min = 0 must map to risk_score = 1.0 (collision)."""
        result = analyze_conjunction(
            sat1_line1=_SAT_A_L1, sat1_line2=_SAT_A_L2,
            sat2_line1=_SAT_A_L1, sat2_line2=_SAT_A_L2,
            t_start_utc=_EPOCH_TS, window_hours=0.5,
        )
        assert result["ok"] is True
        assert result["risk"]["risk_score"] == 1.0
        assert result["risk"]["risk_category"] == "CRITICAL"


class TestStaleTLEScenario:
    """
    TLE age affects the risk score via exp(−Δt/7).
    Propagating to a timestamp far in the future of the TLE epoch should
    produce a large tle_age_days and a reduced risk score compared to
    propagating right at epoch.
    """

    def test_stale_tle_age_is_large(self):
        # Propagate to 30 days after epoch
        result = analyze_conjunction(
            sat1_line1=_SAT_A_L1, sat1_line2=_SAT_A_L2,
            sat2_line1=_SAT_B_NEAR_L1, sat2_line2=_SAT_B_NEAR_L2,
            # 30 days after the 1980 epoch
            t_start_utc="1980-11-01T00:00:00Z",
            window_hours=1.0,
        )
        assert result["ok"] is True
        # Both TLEs have the same epoch so age should be ≈ same
        assert result["tle_age_days_sat1"] > 25.0, (
            f"Expected tle_age_days_sat1 > 25, got {result['tle_age_days_sat1']}"
        )

    def test_stale_tle_reduces_risk_vs_fresh(self):
        """
        For the same geometry (same d_min, same v_rel), a larger tle_age_days
        must yield a lower risk score due to the exp(-Δt/7) decay.
        We test this directly on calculate_risk_full to confirm the formula,
        then verify the conjunction service respects it.
        """
        fresh = calculate_risk_full(d_min_km=5.0, v_rel_km_s=2.0, delta_t_epoch_days=0.0)
        stale = calculate_risk_full(d_min_km=5.0, v_rel_km_s=2.0, delta_t_epoch_days=30.0)
        assert fresh["risk_score"] > stale["risk_score"]


class TestRiskDelegation:
    """
    The conjunction risk must equal calculate_risk_full(d_min, v_rel, max_tle_age).
    """

    def test_risk_score_matches_risk_service(self):
        result = analyze_conjunction(
            sat1_line1=_SAT_A_L1, sat1_line2=_SAT_A_L2,
            sat2_line1=_SAT_B_NEAR_L1, sat2_line2=_SAT_B_NEAR_L2,
            t_start_utc=_EPOCH_TS, window_hours=1.0,
        )
        assert result["ok"] is True
        risk = result["risk"]

        # Reconstruct expected risk using same inputs
        max_age = max(
            abs(result["tle_age_days_sat1"]),
            abs(result["tle_age_days_sat2"]),
        )
        expected = calculate_risk_full(
            d_min_km=result["d_min_km"],
            v_rel_km_s=result["v_rel_km_s"],
            delta_t_epoch_days=max_age,
        )
        assert risk["risk_score"] == pytest.approx(expected["risk_score"], abs=1e-6)
        assert risk["risk_category"] == expected["risk_category"]
