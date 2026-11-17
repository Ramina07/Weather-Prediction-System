"""Explicit, reproducible cleaning of the hourly pollution table.

Global (split-independent) transformations performed here:
    schema/type validation, UTC parsing, removal of future timestamps,
    chronological sorting, duplicate removal, negative-value invalidation,
    reindexing onto a complete hourly grid (missing hours stay NaN), and
    local-time annotation.

NOT performed here (they are fitted on training data only, inside model
pipelines): imputation, scaling, feature selection. No gap is interpolated
and no PM2.5 target value is ever fabricated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, get_logger
from .data_validation import VALUE_COLUMNS, audit_hourly, missing_gap_lengths, validate_schema


def clean_hourly(raw: pd.DataFrame, tz: str, now_utc: pd.Timestamp | None = None,
                 min_valid_hours: int = 20) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean raw hourly records and return (clean_grid, quality_report)."""
    log = get_logger()
    rep: dict[str, Any] = {"steps": []}
    df, bad = validate_schema(raw)
    rep["unparseable_values"] = bad
    rep["before"] = audit_hourly(df.dropna(subset=["dt_unix"]), tz, min_valid_hours=min_valid_hours) if len(df) else {}

    n0 = len(df)
    df = df.dropna(subset=["dt_unix"])
    rep["steps"].append({"step": "drop_invalid_timestamps", "removed": int(n0 - len(df))})
    df["dt_unix"] = df["dt_unix"].astype("int64")

    now_utc = now_utc or pd.Timestamp.now(tz="UTC")
    future = df["dt_unix"] > int(now_utc.timestamp())
    rep["steps"].append({"step": "drop_future_timestamps", "removed": int(future.sum())})
    df = df[~future]

    was_sorted = bool(df["dt_unix"].is_monotonic_increasing)
    df = df.sort_values("dt_unix", kind="mergesort")
    rep["steps"].append({"step": "sort_chronologically", "was_sorted": was_sorted})

    n1 = len(df)
    df = df.drop_duplicates(subset=["dt_unix", *VALUE_COLUMNS], keep="first")
    exact = n1 - len(df)
    n2 = len(df)
    conflicting = int(df["dt_unix"].duplicated(keep=False).sum())
    df = df.drop_duplicates(subset=["dt_unix"], keep="first")
    rep["steps"].append({"step": "remove_duplicates", "exact_duplicates_removed": int(exact),
                         "conflicting_duplicate_rows": conflicting,
                         "conflicting_removed_keep_first": int(n2 - len(df)),
                         "reason": "one observation per hourly timestamp; first occurrence kept"})

    neg = {}
    for c in VALUE_COLUMNS:
        if c == "aqi":
            m = ~df[c].isin([1, 2, 3, 4, 5]) & df[c].notna()
        else:
            m = df[c] < 0
        neg[c] = int(m.sum())
        df.loc[m, c] = np.nan
    rep["steps"].append({"step": "invalidate_impossible_values", "set_to_nan": neg,
                         "rule": "negative concentrations; AQI outside 1-5"})

    if len(df) == 0:
        raise ValueError("No valid observations after cleaning.")
    first = pd.to_datetime(df["dt_unix"].iloc[0], unit="s", utc=True)
    last = pd.to_datetime(df["dt_unix"].iloc[-1], unit="s", utc=True)
    grid = pd.date_range(first, last, freq="h", tz="UTC")
    df.index = pd.to_datetime(df["dt_unix"], unit="s", utc=True)
    off_grid = int((df.index != df.index.floor("h")).sum())
    if off_grid:
        log.warning("%d timestamps are not on the hour; they are floored to the hour", off_grid)
        df.index = df.index.floor("h")
        df = df[~df.index.duplicated(keep="first")]
    g = df[VALUE_COLUMNS].reindex(grid)
    g["is_observed"] = g.index.isin(df.index)
    g.index.name = "timestamp_utc"
    rep["steps"].append({"step": "reindex_complete_hourly_grid", "grid_hours": int(len(g)),
                         "missing_hours_inserted_as_nan": int((~g["is_observed"]).sum()),
                         "off_grid_timestamps_floored": off_grid,
                         "note": "missing hours are NOT interpolated"})
    g = g.reset_index()
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    g.insert(1, "dt_unix", ((g["timestamp_utc"] - epoch) // pd.Timedelta(seconds=1)).astype("int64"))
    loc = g["timestamp_utc"].dt.tz_convert(tz)
    g.insert(2, "timestamp_local", loc)
    g.insert(3, "date_local", loc.dt.strftime("%Y-%m-%d"))
    g.insert(4, "hour_local", loc.dt.hour.astype(int))

    # Statistical inspection of extremes (flag only, never removed).
    pm = g["pm2_5"]
    lp = np.log1p(pm)
    med = np.nanmedian(lp)
    mad = np.nanmedian(np.abs(lp - med)) * 1.4826
    rz = (lp - med) / mad if mad > 0 else lp * 0
    g["pm25_extreme_flag"] = (rz > 3.5).fillna(False)
    gaps = missing_gap_lengths(g)
    rep["extremes"] = {"method": "robust z-score of log1p(PM2.5) > 3.5 (flag only, retained)",
                       "n_flagged_hours": int(g["pm25_extreme_flag"].sum()),
                       "pm25_p99": float(np.nanpercentile(pm, 99)), "pm25_max": float(np.nanmax(pm))}
    rep["gaps"] = {"n_gaps": int(len(gaps)), "max_gap_hours": int(gaps.max()) if len(gaps) else 0,
                   "gaps_over_24h": int((gaps > 24).sum()) if len(gaps) else 0,
                   "gap_length_counts": {int(k): int(v) for k, v in gaps.value_counts().sort_index().items()}}
    rep["after"] = audit_hourly(g, tz, observed_col="is_observed", min_valid_hours=min_valid_hours)
    rep["after"]["grid_rows_including_missing_hours"] = int(len(g))
    return g, rep


def save_clean(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out["timestamp_utc"] = out["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["timestamp_local"] = out["timestamp_local"].map(lambda t: t.isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def load_clean(path: Path, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["timestamp_local"] = df["timestamp_utc"].dt.tz_convert(tz)
    df["is_observed"] = df["is_observed"].astype(bool)
    return df


def save_quality_report(rep: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    b, a = rep.get("before", {}), rep.get("after", {})
    lines = ["# Data-quality report (before vs after cleaning)", "",
             "Generated by `air_quality.preprocessing.clean_hourly`. Values are computed from the "
             "actual downloaded data.", "", "| Metric | Before | After |", "|---|---|---|"]
    for k in ["n_records", "n_unique_timestamps", "n_duplicated_timestamps", "expected_hourly_intervals",
              "coverage_pct", "missing_hours", "pm2_5_min", "pm2_5_max", "n_calendar_days_local",
              "n_valid_days_min_20h", "n_days_with_24h", "is_sorted"]:
        lines.append(f"| {k} | {b.get(k, '')} | {a.get(k, '')} |")
    lines += ["", "## Missing values by field (after cleaning, observed hours only)", "",
              "| Field | Missing | % |", "|---|---|---|"]
    for k, v in a.get("missing_by_field", {}).items():
        lines.append(f"| {k} | {v} | {a['missing_pct_by_field'][k]} |")
    lines += ["", "## Cleaning steps", ""]
    for s in rep["steps"]:
        lines.append(f"- **{s['step']}**: " + ", ".join(f"{k}={v}" for k, v in s.items() if k != "step"))
    lines += ["", "## Gaps", "", f"{rep.get('gaps')}", "", "## Extremes", "", f"{rep.get('extremes')}", "",
              "Transformations above are global and split-independent. Imputation, scaling and feature "
              "selection are fitted on training data only inside scikit-learn pipelines."]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def run_preprocess(cfg: Config, operational: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    log = get_logger()
    raw_path = cfg.path("raw_operational_csv" if operational else "raw_csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}. Run the download command first.")
    raw = pd.read_csv(raw_path)
    if operational and cfg.path("raw_csv").exists():
        raw = pd.concat([pd.read_csv(cfg.path("raw_csv")), raw], ignore_index=True)
    clean, rep = clean_hourly(raw, cfg.tz, min_valid_hours=cfg["forecast"]["target_min_valid_hours"])
    out = cfg.path("clean_operational_csv" if operational else "clean_hourly_csv")
    save_clean(clean, out)
    if not operational:
        save_quality_report(rep, cfg.path("quality_report_json"), cfg.path("quality_report_md"))
    log.info("Clean hourly grid: %d rows (%d observed) -> %s", len(clean), int(clean["is_observed"].sum()), out)
    return clean, rep
