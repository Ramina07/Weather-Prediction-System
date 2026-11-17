"""Schema validation and dataset audit statistics for hourly pollution data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .api_client import COMPONENT_FIELDS

REQUIRED_RAW_COLUMNS = ["dt_unix", "aqi", *COMPONENT_FIELDS]
VALUE_COLUMNS = ["aqi", *COMPONENT_FIELDS]


class DataValidationError(ValueError):
    """Raised when a dataset fails a hard validation rule."""


def validate_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Check required columns and safely coerce value columns to float.

    Returns the converted frame and the number of values that could not be
    parsed per column (these become NaN and are reported, not invented).
    """
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    out = df.copy()
    bad: dict[str, int] = {}
    dt = pd.to_numeric(out["dt_unix"], errors="coerce")
    bad["dt_unix"] = int(dt.isna().sum() - out["dt_unix"].isna().sum())
    out["dt_unix"] = dt
    for c in VALUE_COLUMNS:
        before_na = out[c].isna().sum()
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
        bad[c] = int(out[c].isna().sum() - before_na)
    return out, bad


def audit_hourly(df: pd.DataFrame, tz: str, value_col: str = "pm2_5",
                 observed_col: str | None = None, min_valid_hours: int = 20) -> dict[str, Any]:
    """Compute the audit statistics required after downloading.

    ``df`` must contain ``dt_unix``; if ``observed_col`` is given, rows where it
    is False are treated as grid placeholders (missing hours).
    """
    if len(df) == 0:
        return {"n_records": 0}
    d = df if observed_col is None else df[df[observed_col].astype(bool)]
    ts = pd.to_datetime(d["dt_unix"], unit="s", utc=True)
    first, last = ts.min(), ts.max()
    expected = int((last - first) / pd.Timedelta(hours=1)) + 1
    unique = int(ts.nunique())
    dup = int(len(ts) - unique)
    local = ts.dt.tz_convert(tz)
    pm = d[value_col]
    daily = pd.DataFrame({"date": local.dt.date, "v": pm.values}).groupby("date")["v"].count()
    all_days = pd.date_range(local.min().normalize(), local.max().normalize(), freq="D").date
    daily = daily.reindex(all_days, fill_value=0)
    missing_by_field = {c: int(d[c].isna().sum()) for c in VALUE_COLUMNS if c in d.columns}
    res = {
        "first_timestamp_utc": first.isoformat(),
        "last_timestamp_utc": last.isoformat(),
        "first_timestamp_local": first.tz_convert(tz).isoformat(),
        "last_timestamp_local": last.tz_convert(tz).isoformat(),
        "n_records": int(len(d)),
        "n_unique_timestamps": unique,
        "n_duplicated_timestamps": dup,
        "expected_hourly_intervals": expected,
        "coverage_pct": round(100.0 * unique / expected, 3) if expected else 0.0,
        "missing_hours": int(expected - unique),
        "missing_by_field": missing_by_field,
        "missing_pct_by_field": {k: round(100 * v / len(d), 3) for k, v in missing_by_field.items()},
        f"{value_col}_min": float(np.nanmin(pm)) if pm.notna().any() else None,
        f"{value_col}_max": float(np.nanmax(pm)) if pm.notna().any() else None,
        "n_calendar_days_local": int(len(daily)),
        f"n_valid_days_min_{min_valid_hours}h": int((daily >= min_valid_hours).sum()),
        "n_days_with_24h": int((daily >= 24).sum()),
        "n_negative_values": {c: int((d[c] < 0).sum()) for c in VALUE_COLUMNS if c in d.columns},
        "is_sorted": bool(ts.is_monotonic_increasing),
    }
    return res


def missing_gap_lengths(df: pd.DataFrame, observed_col: str = "is_observed") -> pd.Series:
    """Lengths (hours) of consecutive missing-hour runs on a complete grid."""
    miss = ~df[observed_col].astype(bool).to_numpy()
    if not miss.any():
        return pd.Series(dtype=int)
    runs, count = [], 0
    for m in miss:
        if m:
            count += 1
        elif count:
            runs.append(count)
            count = 0
    if count:
        runs.append(count)
    return pd.Series(runs, name="gap_hours")


def format_audit(a: dict[str, Any], title: str = "Dataset audit") -> str:
    lines = [f"=== {title} ==="]
    for k, v in a.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            lines.extend(f"    {kk:<8} {vv}" for kk, vv in v.items())
        else:
            lines.append(f"{k:<34} {v}")
    return "\n".join(lines)
