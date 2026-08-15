"""
OGB — OrbitalGuard
tests/unit/test_tool_calling.py

Tests that verify:
1. The real risk_service functions produce the expected values.
2. The _execute_tool() dispatcher in copilot_service calls the real
   calculate_risk_full() — not a mock — and returns matching numbers.
3. The propagate_tle stub returns a clear NOT_IMPLEMENTED error.
4. Partial / missing orbital data scenario: _execute_tool raises KeyError
   when required fields are absent (Gemini would not call the tool in this
   case, but we verify the defensive behaviour).
5. Manual spot-check: operator supplies d_min=2, v_rel=5, tle_age=3 and
   the returned score matches manually calculating the formula.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.services.risk_service import (
    calculate_risk_full,
    calculate_risk_score,
    categorise_risk,
)
from app.services.copilot_service import _execute_tool
from app.core.config import HARD_BODY_RADIUS_KM


# ---------------------------------------------------------------------------
# 1. risk_service unit tests (real formula, not mocked)
# ---------------------------------------------------------------------------

class TestRiskServiceReal:
    """Verify risk_service.py implements the formula correctly."""

    def test_spot_check_known_values(self):
        """
        Manual calculation for d_min=2, v_rel=5, tle_age=3, R=0.01:
          (0.01/2) * log10(6) * exp(-3/7)
          = 0.005 * 0.77815 * 0.65066
          ≈ 0.002529
        """
        expected = (
            (HARD_BODY_RADIUS_KM / 2.0)
            * math.log10(5.0 + 1.0)
            * math.exp(-3.0 / 7.0)
        )
        score = calculate_risk_score(
            d_min_km=2.0,
            v_rel_km_s=5.0,
            delta_t_epoch_days=3.0,
        )
        assert score == pytest.approx(expected, rel=1e-9)

    def test_calculate_risk_full_returns_dict(self):
        result = calculate_risk_full(
            d_min_km=2.0,
            v_rel_km_s=5.0,
            delta_t_epoch_days=3.0,
        )
        assert "risk_score" in result
        assert "risk_category" in result
        assert "formula" in result
        assert 0.0 <= result["risk_score"] <= 1.0
        assert result["risk_category"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_calculate_risk_full_score_matches_score_function(self):
        """
        calculate_risk_full rounds to 6 d.p. in the dict, so compare with
        abs tolerance matching that rounding (< 5e-7).
        """
        score = calculate_risk_score(
            d_min_km=0.5,
            v_rel_km_s=7.0,
            delta_t_epoch_days=1.0,
        )
        full = calculate_risk_full(
            d_min_km=0.5,
            v_rel_km_s=7.0,
            delta_t_epoch_days=1.0,
        )
        assert full["risk_score"] == pytest.approx(score, abs=5e-7)

    def test_zero_separation_returns_one(self):
        result = calculate_risk_full(d_min_km=0.0, v_rel_km_s=5.0, delta_t_epoch_days=0.0)
        assert result["risk_score"] == 1.0
        assert result["risk_category"] == "CRITICAL"

    def test_category_thresholds_consistent(self):
        """categorise_risk must agree with the category in calculate_risk_full."""
        for d_min, v_rel, tle_age in [
            (100.0, 0.1, 0.0),   # very safe → LOW
            (1.0,   3.0, 0.0),   # moderate
            (0.1,   7.0, 0.0),   # higher risk
        ]:
            result = calculate_risk_full(
                d_min_km=d_min, v_rel_km_s=v_rel, delta_t_epoch_days=tle_age
            )
            assert categorise_risk(result["risk_score"]) == result["risk_category"]


# ---------------------------------------------------------------------------
# 2. Tool-executor round-trip tests (_execute_tool → real risk_service)
# ---------------------------------------------------------------------------

class TestExecuteToolRoundTrip:
    """
    _execute_tool must invoke the real calculate_risk_full — not a stub —
    so the returned score matches direct calculation.
    """

    def test_round_trip_matches_direct_calculation(self):
        """
        Core regression: _execute_tool("calculate_risk_score", args) must
        return the same risk_score as calling calculate_risk_full directly.
        """
        args = {
            "min_separation_km": 2.0,
            "relative_velocity_km_s": 5.0,
            "tle_age_days": 3.0,
        }
        tool_result = _execute_tool("calculate_risk_score", args)
        direct_result = calculate_risk_full(
            d_min_km=2.0,
            v_rel_km_s=5.0,
            delta_t_epoch_days=3.0,
        )
        assert tool_result["risk_score"] == pytest.approx(
            direct_result["risk_score"], abs=5e-7
        )
        assert tool_result["risk_category"] == direct_result["risk_category"]

    def test_round_trip_with_explicit_hard_body_radius(self):
        """Custom hard_body_radius_km flows through correctly."""
        args = {
            "min_separation_km": 1.0,
            "relative_velocity_km_s": 3.0,
            "tle_age_days": 0.0,
            "hard_body_radius_km": 0.05,
        }
        tool_result = _execute_tool("calculate_risk_score", args)
        direct_result = calculate_risk_full(
            hard_body_radius_km=0.05,
            d_min_km=1.0,
            v_rel_km_s=3.0,
            delta_t_epoch_days=0.0,
        )
        assert tool_result["risk_score"] == pytest.approx(
            direct_result["risk_score"], abs=5e-7
        )

    def test_result_contains_required_fields(self):
        args = {
            "min_separation_km": 5.0,
            "relative_velocity_km_s": 2.0,
            "tle_age_days": 7.0,
        }
        result = _execute_tool("calculate_risk_score", args)
        for field in ("risk_score", "risk_category", "d_min_km", "v_rel_km_s",
                      "delta_t_epoch_days", "formula", "note"):
            assert field in result, f"Missing field: {field}"

    def test_clamp_at_critical_when_zero_separation(self):
        args = {
            "min_separation_km": 0.0,
            "relative_velocity_km_s": 10.0,
            "tle_age_days": 0.0,
        }
        result = _execute_tool("calculate_risk_score", args)
        assert result["risk_score"] == 1.0
        assert result["risk_category"] == "CRITICAL"

    def test_missing_required_arg_raises(self):
        """If a required arg is absent, _execute_tool should raise KeyError
        (the Gemini API would not call the tool without required params,
        but we verify defensively)."""
        with pytest.raises(KeyError):
            _execute_tool("calculate_risk_score", {
                "min_separation_km": 2.0,
                # missing relative_velocity_km_s and tle_age_days
            })


# ---------------------------------------------------------------------------
# 3. propagate_tle stub
# ---------------------------------------------------------------------------

class TestPropagateTleStub:
    def test_returns_not_implemented(self):
        result = _execute_tool("propagate_tle", {
            "tle_line1": "1 25544U ...",
            "tle_line2": "2 25544 ...",
            "timestamp_utc": "2025-08-01T00:00:00Z",
        })
        assert result["status"] == "NOT_IMPLEMENTED"
        assert "error" in result
        assert "SGP4" in result["reason"] or "not" in result["reason"].lower()

    def test_unknown_tool_returns_error(self):
        result = _execute_tool("nonexistent_tool", {})
        assert result["status"] == "ERROR"
        assert "Unknown tool" in result["error"]


# ---------------------------------------------------------------------------
# 4. Manual spot-check — the worked example from the report
# ---------------------------------------------------------------------------

class TestManualSpotCheck:
    """
    Operator says: "object is 2 km away, moving at 5 km/s, TLE is 3 days old."
    Formula:  min(1, (0.01/2) * log10(5+1) * exp(-3/7))

    Step-by-step:
      R/d_min       = 0.01 / 2       = 0.005
      log10(v+1)    = log10(6)       ≈ 0.778151
      exp(-Δt/7)    = exp(-3/7)      ≈ 0.650641
      product       = 0.005 * 0.778151 * 0.650641 ≈ 0.002529
      clamped       = min(1, 0.002529) = 0.002529
      category      = LOW (0.00–0.24)
    """

    def test_worked_example_score(self):
        expected = (
            (HARD_BODY_RADIUS_KM / 2.0)
            * math.log10(6.0)
            * math.exp(-3.0 / 7.0)
        )
        result = _execute_tool("calculate_risk_score", {
            "min_separation_km": 2.0,
            "relative_velocity_km_s": 5.0,
            "tle_age_days": 3.0,
        })
        # risk_score is rounded to 6 d.p. in the dict
        assert result["risk_score"] == pytest.approx(expected, abs=5e-7)

    def test_worked_example_category_is_low(self):
        result = _execute_tool("calculate_risk_score", {
            "min_separation_km": 2.0,
            "relative_velocity_km_s": 5.0,
            "tle_age_days": 3.0,
        })
        assert result["risk_category"] == "LOW"

    def test_worked_example_score_is_approx_0_002529(self):
        """Sanity check the magnitude — should be in the thousandths."""
        result = _execute_tool("calculate_risk_score", {
            "min_separation_km": 2.0,
            "relative_velocity_km_s": 5.0,
            "tle_age_days": 3.0,
        })
        assert 0.002 < result["risk_score"] < 0.004
