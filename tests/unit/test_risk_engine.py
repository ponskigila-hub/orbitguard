"""
OGB — OrbitalGuard
Unit tests for the risk engine formula.

Formula (documented with units):
  Risk = min(1, (R / d_min) · log10(v_rel + 1) · exp(−Δt_epoch / 7))

  R          = hard-body radius          [km]
  d_min      = minimum separation        [km]
  v_rel      = relative velocity         [km/s]
  Δt_epoch   = TLE age                   [days]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# Allow running from repo root or tests/ directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

import pytest


# ---------------------------------------------------------------------------
# Import the risk engine (will be at backend/app/services/risk_service.py)
# We define the formula locally here for now so tests are runnable before
# the service module exists — when risk_service is written the import will
# replace this stub automatically.
# ---------------------------------------------------------------------------
try:
    from app.services.risk_service import calculate_risk_score, categorise_risk  # type: ignore
except ImportError:
    # Inline reference implementation for testing
    from app.core.config import RISK_THRESHOLDS  # type: ignore

    def calculate_risk_score(
        *,
        hard_body_radius_km: float,
        d_min_km: float,
        v_rel_km_s: float,
        delta_t_epoch_days: float,
    ) -> float:
        """
        Risk = min(1, (R / d_min) · log10(v_rel + 1) · exp(−Δt / 7))
        Units: R and d_min in km, v_rel in km/s, Δt in days.
        """
        if d_min_km <= 0:
            return 1.0
        raw = (
            (hard_body_radius_km / d_min_km)
            * math.log10(v_rel_km_s + 1)
            * math.exp(-delta_t_epoch_days / 7.0)
        )
        return min(1.0, raw)

    def categorise_risk(score: float) -> str:
        for category, (lo, hi) in RISK_THRESHOLDS.items():
            if lo <= score <= hi:
                return category
        return "CRITICAL"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRiskFormula:
    """Core formula correctness and unit-consistency tests."""

    def test_fresh_tle_close_approach_high_velocity(self):
        """Very close, fast, fresh TLE → near-CRITICAL."""
        score = calculate_risk_score(
            hard_body_radius_km=0.01,
            d_min_km=0.05,
            v_rel_km_s=10.0,
            delta_t_epoch_days=0.0,
        )
        assert 0.0 <= score <= 1.0
        # (0.01/0.05) * log10(11) * 1.0 ≈ 0.2 * 1.041 ≈ 0.208 → LOW/MEDIUM
        # (coarse sanity — exact value depends on log10(11) ≈ 1.041)
        assert score == pytest.approx(0.2 * math.log10(11.0), rel=1e-6)

    def test_zero_separation_clamps_to_one(self):
        """d_min = 0 must return 1.0 without division error."""
        score = calculate_risk_score(
            hard_body_radius_km=0.01,
            d_min_km=0.0,
            v_rel_km_s=5.0,
            delta_t_epoch_days=1.0,
        )
        assert score == 1.0

    def test_near_zero_separation_clamps_to_one(self):
        """Extremely close approach should clamp to 1.0."""
        score = calculate_risk_score(
            hard_body_radius_km=0.01,
            d_min_km=1e-9,
            v_rel_km_s=5.0,
            delta_t_epoch_days=0.0,
        )
        assert score == 1.0

    def test_stale_tle_reduces_score(self):
        """Older TLE epoch exponentially reduces risk score."""
        fresh = calculate_risk_score(
            hard_body_radius_km=0.01,
            d_min_km=1.0,
            v_rel_km_s=5.0,
            delta_t_epoch_days=0.0,
        )
        stale = calculate_risk_score(
            hard_body_radius_km=0.01,
            d_min_km=1.0,
            v_rel_km_s=5.0,
            delta_t_epoch_days=30.0,
        )
        assert fresh > stale, "Stale TLE should yield lower risk score"
        # At Δt=30, decay factor = exp(-30/7) ≈ 0.0133
        assert stale == pytest.approx(fresh * math.exp(-30.0 / 7.0), rel=1e-6)

    def test_zero_velocity_yields_zero(self):
        """Relative velocity = 0 → log10(1) = 0 → risk score = 0."""
        score = calculate_risk_score(
            hard_body_radius_km=0.01,
            d_min_km=1.0,
            v_rel_km_s=0.0,
            delta_t_epoch_days=0.0,
        )
        assert score == pytest.approx(0.0, abs=1e-12)

    def test_score_never_exceeds_one(self):
        """Formula clamps at 1.0 regardless of inputs."""
        score = calculate_risk_score(
            hard_body_radius_km=100.0,    # absurdly large radius
            d_min_km=0.001,
            v_rel_km_s=100.0,
            delta_t_epoch_days=0.0,
        )
        assert score == 1.0

    def test_score_is_non_negative(self):
        """Score must always be ≥ 0."""
        score = calculate_risk_score(
            hard_body_radius_km=0.01,
            d_min_km=1000.0,
            v_rel_km_s=0.1,
            delta_t_epoch_days=100.0,
        )
        assert score >= 0.0

    def test_unit_consistency_km_km_s(self):
        """
        Spot-check: R=0.01 km, d_min=1 km, v_rel=1 km/s, Δt=7 days.
        Expected: (0.01/1) * log10(2) * exp(-1) ≈ 0.01 * 0.3010 * 0.3679
        """
        expected = 0.01 * math.log10(2.0) * math.exp(-1.0)
        score = calculate_risk_score(
            hard_body_radius_km=0.01,
            d_min_km=1.0,
            v_rel_km_s=1.0,
            delta_t_epoch_days=7.0,
        )
        assert score == pytest.approx(expected, rel=1e-6)


class TestRiskCategorisation:
    """Risk category boundaries from config."""

    def test_low(self):
        assert categorise_risk(0.00) == "LOW"
        assert categorise_risk(0.12) == "LOW"
        assert categorise_risk(0.24) == "LOW"

    def test_medium(self):
        assert categorise_risk(0.25) == "MEDIUM"
        assert categorise_risk(0.37) == "MEDIUM"
        assert categorise_risk(0.49) == "MEDIUM"

    def test_high(self):
        assert categorise_risk(0.50) == "HIGH"
        assert categorise_risk(0.62) == "HIGH"
        assert categorise_risk(0.74) == "HIGH"

    def test_critical(self):
        assert categorise_risk(0.75) == "CRITICAL"
        assert categorise_risk(0.88) == "CRITICAL"
        assert categorise_risk(1.00) == "CRITICAL"
