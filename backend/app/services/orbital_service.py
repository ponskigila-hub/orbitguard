"""
OGB — OrbitalGuard
orbital_service.py — SGP4 TLE propagation (V2).

Wraps the python-sgp4 library (Brandon Rhodes, https://github.com/brandon-rhodes/python-sgp4)
to produce ECI position and velocity vectors from a Two-Line Element set.

API used: sgp4.api (v2.x) — Satrec + jday
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

# sgp4 >= 2.23 (already in requirements.txt)
from sgp4.api import Satrec, jday  # type: ignore

# ---------------------------------------------------------------------------
# TLE format validation
# ---------------------------------------------------------------------------

# Minimal regex: line 1 starts with '1 ', line 2 starts with '2 '.
# We check length (69 chars each) and line-number prefix only — the sgp4
# library itself performs the full checksum verification on parsing.
_TLE_LINE1_RE = re.compile(r"^1 [ 0-9]{5}[A-Z ] .{52}\d$")
_TLE_LINE2_RE = re.compile(r"^2 [ 0-9]{5} .{52}\d$")


def _validate_tle_format(line1: str, line2: str) -> str | None:
    """
    Return an error string if the TLE lines are obviously malformed,
    or None if they pass basic sanity checks.
    """
    if len(line1) != 69:
        return f"TLE line 1 must be exactly 69 characters, got {len(line1)}."
    if len(line2) != 69:
        return f"TLE line 2 must be exactly 69 characters, got {len(line2)}."
    if not line1.startswith("1 "):
        return "TLE line 1 must start with '1 '."
    if not line2.startswith("2 "):
        return "TLE line 2 must start with '2 '."
    return None


# ---------------------------------------------------------------------------
# Core propagation function
# ---------------------------------------------------------------------------

def propagate_tle_state(
    tle_line1: str,
    tle_line2: str,
    timestamp_utc: str,
) -> dict[str, Any]:
    """
    Propagate a TLE to a given UTC timestamp using SGP4.

    Parameters
    ----------
    tle_line1 : str
        TLE line 1 (exactly 69 chars).
    tle_line2 : str
        TLE line 2 (exactly 69 chars).
    timestamp_utc : str
        ISO-8601 UTC timestamp, e.g. "2025-08-01T12:00:00Z".

    Returns
    -------
    dict with keys:
        ok              bool
        position_km     [x, y, z]  ECI km      (only when ok=True)
        velocity_km_s   [vx, vy, vz] ECI km/s  (only when ok=True)
        epoch_utc       str  ISO-8601 of TLE epoch
        tle_age_days    float  age of TLE at the propagation time
        error           str  (only when ok=False)
    """
    # --- strip / normalise whitespace so pasted TLEs work ---
    line1 = tle_line1.strip()
    line2 = tle_line2.strip()

    # --- format validation ---
    fmt_err = _validate_tle_format(line1, line2)
    if fmt_err:
        return {"ok": False, "error": fmt_err}

    # --- parse TLE ---
    sat = Satrec.twoline2rv(line1, line2)
    if sat.error != 0:
        return {
            "ok": False,
            "error": f"sgp4 TLE parse error code {sat.error}.",
        }

    # --- parse timestamp ---
    try:
        ts = _parse_iso_utc(timestamp_utc)
    except ValueError as exc:
        return {"ok": False, "error": f"Invalid timestamp: {exc}"}

    # --- build Julian date ---
    jd, fr = jday(
        ts.year, ts.month, ts.day,
        ts.hour, ts.minute, ts.second + ts.microsecond / 1e6,
    )

    # --- propagate ---
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        _SGP4_ERRORS = {
            1: "mean elements, e<0 or e>=1",
            2: "mean motion less than 0",
            3: "perturbed eccentricity out of bounds",
            4: "semi-latus rectum < 0",
            5: "epoch elements are sub-orbital",
            6: "satellite has decayed",
        }
        msg = _SGP4_ERRORS.get(e, f"SGP4 propagation error code {e}")
        return {"ok": False, "error": msg}

    # --- compute TLE epoch as datetime ---
    # sat.epochyr (2-digit year) + sat.epochdays (day-of-year with fraction)
    epoch_yr = int(sat.epochyr)
    full_year = 2000 + epoch_yr if epoch_yr < 57 else 1900 + epoch_yr
    epoch_jd = sat.jdsatepoch + sat.jdsatepochF
    epoch_dt = _jd_to_datetime(epoch_jd)
    tle_age_days = (ts - epoch_dt).total_seconds() / 86400.0

    return {
        "ok": True,
        "position_km": list(r),
        "velocity_km_s": list(v),
        "epoch_utc": epoch_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "tle_age_days": round(tle_age_days, 6),
        "propagated_at_utc": ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_iso_utc(s: str) -> datetime:
    """Parse an ISO-8601 UTC string to an offset-aware datetime (UTC)."""
    s = s.strip().upper().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # Try without timezone marker
        dt = datetime.fromisoformat(s.replace("+00:00", ""))
        dt = dt.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _jd_to_datetime(jd_full: float) -> datetime:
    """Convert a Julian Date (float) to a UTC datetime."""
    # JD 2451545.0 = J2000.0 = 2000-01-01 12:00:00 UTC
    J2000 = 2451545.0
    delta_days = jd_full - J2000
    delta_seconds = delta_days * 86400.0
    j2000_epoch = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return j2000_epoch + timedelta(seconds=delta_seconds)
