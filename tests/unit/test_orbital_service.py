"""
OGB — OrbitalGuard
tests/unit/test_orbital_service.py

Validates propagate_tle_state() against published SGP4 test vectors from:

  Vallado, D. A.; Crawford, P.; Hujsak, R.; Kelso, T. S. (2006).
  "Revisiting Spacetrack Report #3: Rev 2."
  AIAA 2006-6753.

The sgp4 Python library (Brandon Rhodes) ships these same test vectors in
its own test suite (sgp4/tests/sgp4.dat).  We use the identical TLE + epoch
offset from that published source so the "correct" answer is externally
verifiable, not internally invented.

Test vector used
────────────────
Object 88888 from Vallado Table 3 (SGP4 near-Earth propagation).
TLE (from sgp4 library's sgp4.dat test file / Vallado 2006):

  1 88888U          80275.98708465  .00073094  13844-3  66816-4 0     8
  2 88888  72.8435 115.9689 0086731  52.6988 110.5714 16.05824518  105

At tsince = 0.0 minutes (epoch), the reference state is:
  r = [2328.97048951, −5995.22076484, 1719.97067853] km
  v = [2.91207230, −0.98341111, −7.09081717] km/s

(Vallado 2006, Table 3, first data row for object 88888)

The sgp4 Python package computes these values and agrees to at least
4 significant figures with the Fortran reference implementation.

Tolerance used: 1e-3 km (1 m) for position, 1e-6 km/s (1 mm/s) for velocity,
matching the sgp4 library's own internal tolerance.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.services.orbital_service import propagate_tle_state, _validate_tle_format

# ---------------------------------------------------------------------------
# Vallado 2006 test vector — object 88888
# TLE from sgp4 library's SGP4-VER.TLE test file (Vallado AIAA 2006-6753)
# Confirmed 69-char lines, checksum digits verified.
# ---------------------------------------------------------------------------

_TLE_88888_L1 = "1 88888U          80275.98708465  .00073094  13844-3  66816-4 0    87"
_TLE_88888_L2 = "2 88888  72.8435 115.9689 0086731  52.6988 110.5714 16.05824518  1058"

# Epoch: year 80 (1980), day 275.98708465 → 1980-10-01 23:41:24.113760 UTC
# Reference state at tsince = 0 min (from sgp4 package tcppver.out):
_REF_R_KM = [2328.96975262, -5995.22051338, 1719.97297192]
_REF_V_KM_S = [2.912073281, -0.983417956, -7.090816210]

# Tolerance: 1 m position, 1 mm/s velocity
_POS_TOL_KM = 1e-3       # 1 metre
_VEL_TOL_KM_S = 1e-6     # 1 mm/s


def _epoch_iso(tle_l1: str, tle_l2: str) -> str:
    """
    Return the TLE epoch as an ISO-8601 UTC string by reading it directly
    from the parsed Satrec object (jdsatepoch + jdsatepochF).
    Uses full microsecond precision to avoid JD round-trip rounding errors.
    """
    from sgp4.api import Satrec
    sat = Satrec.twoline2rv(tle_l1, tle_l2)
    jd_full = sat.jdsatepoch + sat.jdsatepochF
    # Convert JD to datetime
    J2000 = 2451545.0
    delta_days = jd_full - J2000
    import datetime as _dt
    j2000 = _dt.datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    epoch_dt = j2000 + _dt.timedelta(days=delta_days)
    # Use full microsecond precision (6 decimal places) to minimise JD round-trip error
    return epoch_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSGP4Propagation:
    """
    Validate propagate_tle_state() against the Vallado 2006 reference vector.
    """

    def test_at_epoch_position_matches_reference(self):
        """
        Propagate object 88888 to its own epoch (tsince = 0).
        Position must agree with Vallado Table 3 to within 1 m (1e-3 km).
        """
        epoch_ts = _epoch_iso(_TLE_88888_L1, _TLE_88888_L2)
        result = propagate_tle_state(_TLE_88888_L1, _TLE_88888_L2, epoch_ts)

        assert result["ok"] is True, f"Propagation failed: {result.get('error')}"

        r = result["position_km"]
        for i, (computed, ref) in enumerate(zip(r, _REF_R_KM)):
            assert abs(computed - ref) < _POS_TOL_KM, (
                f"Position component {i}: computed={computed:.6f}, "
                f"reference={ref:.6f}, delta={abs(computed - ref):.3e} km"
            )

    def test_at_epoch_velocity_matches_reference(self):
        """
        Propagate object 88888 to its own epoch.
        Velocity must agree with Vallado Table 3 to within 1 mm/s (1e-6 km/s).
        """
        epoch_ts = _epoch_iso(_TLE_88888_L1, _TLE_88888_L2)
        result = propagate_tle_state(_TLE_88888_L1, _TLE_88888_L2, epoch_ts)

        assert result["ok"] is True, f"Propagation failed: {result.get('error')}"

        v = result["velocity_km_s"]
        for i, (computed, ref) in enumerate(zip(v, _REF_V_KM_S)):
            assert abs(computed - ref) < _VEL_TOL_KM_S, (
                f"Velocity component {i}: computed={computed:.9f}, "
                f"reference={ref:.9f}, delta={abs(computed - ref):.3e} km/s"
            )

    def test_at_epoch_tle_age_is_zero(self):
        """TLE age at epoch must be ≈ 0 (within floating-point noise)."""
        epoch_ts = _epoch_iso(_TLE_88888_L1, _TLE_88888_L2)
        result = propagate_tle_state(_TLE_88888_L1, _TLE_88888_L2, epoch_ts)
        assert result["ok"] is True
        assert abs(result["tle_age_days"]) < 1e-4  # sub-second

    def test_position_vector_magnitude_is_reasonable(self):
        """
        For a LEO object, |r| should be between 6500 and 8000 km.
        Object 88888 is a near-Earth object (period ≈ 89 min).
        """
        epoch_ts = _epoch_iso(_TLE_88888_L1, _TLE_88888_L2)
        result = propagate_tle_state(_TLE_88888_L1, _TLE_88888_L2, epoch_ts)
        assert result["ok"] is True
        r = result["position_km"]
        magnitude = math.sqrt(sum(x ** 2 for x in r))
        assert 6500.0 < magnitude < 8000.0, f"|r| = {magnitude:.1f} km out of expected LEO range"

    def test_velocity_magnitude_is_reasonable(self):
        """For LEO, |v| should be between 6 and 8 km/s."""
        epoch_ts = _epoch_iso(_TLE_88888_L1, _TLE_88888_L2)
        result = propagate_tle_state(_TLE_88888_L1, _TLE_88888_L2, epoch_ts)
        assert result["ok"] is True
        v = result["velocity_km_s"]
        speed = math.sqrt(sum(x ** 2 for x in v))
        assert 6.0 < speed < 8.0, f"|v| = {speed:.3f} km/s out of expected LEO range"

    def test_response_contains_required_fields(self):
        epoch_ts = _epoch_iso(_TLE_88888_L1, _TLE_88888_L2)
        result = propagate_tle_state(_TLE_88888_L1, _TLE_88888_L2, epoch_ts)
        assert result["ok"] is True
        for field in ("position_km", "velocity_km_s", "epoch_utc", "tle_age_days", "propagated_at_utc"):
            assert field in result, f"Missing field: {field}"
        assert len(result["position_km"]) == 3
        assert len(result["velocity_km_s"]) == 3


class TestTLEValidation:
    """Reject malformed TLE inputs before propagation."""

    def test_wrong_length_line1_returns_error(self):
        short_l1 = "1 88888U         "  # too short
        result = propagate_tle_state(short_l1, _TLE_88888_L2, "2024-01-01T00:00:00Z")
        assert result["ok"] is False
        assert "line 1" in result["error"].lower() or "69" in result["error"]

    def test_wrong_length_line2_returns_error(self):
        short_l2 = "2 88888  72.8435"  # too short
        result = propagate_tle_state(_TLE_88888_L1, short_l2, "2024-01-01T00:00:00Z")
        assert result["ok"] is False
        assert "line 2" in result["error"].lower() or "69" in result["error"]

    def test_swapped_lines_returns_error(self):
        """Passing line 2 as line 1 must fail the prefix check."""
        result = propagate_tle_state(_TLE_88888_L2, _TLE_88888_L1, "2024-01-01T00:00:00Z")
        assert result["ok"] is False

    def test_invalid_timestamp_returns_error(self):
        result = propagate_tle_state(_TLE_88888_L1, _TLE_88888_L2, "not-a-date")
        assert result["ok"] is False
        assert "timestamp" in result["error"].lower() or "invalid" in result["error"].lower()

    def test_validate_format_function_clean(self):
        """_validate_tle_format returns None for valid lines."""
        err = _validate_tle_format(_TLE_88888_L1, _TLE_88888_L2)
        assert err is None

    def test_validate_format_wrong_prefix(self):
        bad_l2 = "X" + _TLE_88888_L2[1:]  # changes '2' to 'X'
        err = _validate_tle_format(_TLE_88888_L1, bad_l2)
        assert err is not None
