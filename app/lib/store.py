"""Persistent forecast store (SQLite, one file: ``data/app/forecast_store.db``).

Every forecast made in the app is saved here. When its target day has passed, the
real daily-mean PM2.5 is downloaded from the same API and written next to it, so the
app can report the model's *live* accuracy on days it has never seen.

Tables
------
forecasts     one row per forecast (live run, hindcast, or imported CLI forecast)
observations  verified daily-mean PM2.5 per local date (the "ground truth" for scoring)
events        small audit log (what was done and when)
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from . import paths

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at_utc    TEXT NOT NULL,
    kind              TEXT NOT NULL CHECK (kind IN ('live', 'hindcast', 'imported')),
    issue_time_local  TEXT NOT NULL,
    target_date       TEXT NOT NULL,
    model             TEXT,
    feature_config    TEXT,
    predicted         REAL NOT NULL,
    lower             REAL,
    upper             REAL,
    provider_pred     REAL,
    recent_mean_24h   REAL,
    reliable          INTEGER,
    warnings          TEXT,
    weather_data      TEXT,
    payload_json      TEXT,
    UNIQUE (kind, issue_time_local, model, feature_config)
);
CREATE INDEX IF NOT EXISTS ix_forecasts_target ON forecasts (target_date);
CREATE TABLE IF NOT EXISTS observations (
    target_date     TEXT PRIMARY KEY,
    actual          REAL,
    n_hours         INTEGER,
    fetched_at_utc  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc  TEXT NOT NULL,
    event   TEXT NOT NULL,
    detail  TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    db = Path(db_path or paths.STORE_DB)
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def log_event(event: str, detail: str = "", db_path: Path | None = None) -> None:
    with connect(db_path) as con:
        con.execute("INSERT INTO events (ts_utc, event, detail) VALUES (?, ?, ?)", (_now(), event, detail))


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def save_forecast(result: dict[str, Any], kind: str = "live", db_path: Path | None = None) -> bool:
    """Insert a forecast dict (as returned by ``prediction.make_forecast``). Returns True if new."""
    pi = result.get("prediction_interval") or {}
    prov = result.get("provider_forecast") or {}
    row = (
        _now(), kind, result["forecast_issued_local"], result["prediction_date"], result.get("model"),
        result.get("feature_config"), float(result["predicted_pm25"]), _f(pi.get("lower")), _f(pi.get("upper")),
        _f(prov.get("mean_pm25_target_date")), _f(result.get("recent_pm25_mean_24h")),
        int(bool(result.get("reliable", True))), " | ".join(result.get("warnings") or []),
        result.get("weather_data"), json.dumps(result, ensure_ascii=False, default=str),
    )
    with connect(db_path) as con:
        # a repeated run for the same issue time replaces the older one (latest data wins)
        cur = con.execute(
            """INSERT INTO forecasts (created_at_utc, kind, issue_time_local, target_date, model, feature_config,
                   predicted, lower, upper, provider_pred, recent_mean_24h, reliable, warnings, weather_data,
                   payload_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT (kind, issue_time_local, model, feature_config) DO UPDATE SET
                   created_at_utc=excluded.created_at_utc, predicted=excluded.predicted, lower=excluded.lower,
                   upper=excluded.upper, provider_pred=excluded.provider_pred,
                   recent_mean_24h=excluded.recent_mean_24h, reliable=excluded.reliable,
                   warnings=excluded.warnings, weather_data=excluded.weather_data,
                   payload_json=excluded.payload_json""", row)
        con.execute("INSERT INTO events (ts_utc, event, detail) VALUES (?, ?, ?)",
                    (_now(), f"forecast_{kind}", f"{result['prediction_date']}: {result['predicted_pm25']}"))
        return cur.rowcount > 0


def import_cli_history(pred_dir: Path | None = None, db_path: Path | None = None) -> int:
    """Copy forecasts made with the command line (``artifacts/predictions/forecast_*.json``) into the store.

    The per-forecast JSON files are used rather than ``forecast_history.csv`` because the CSV's columns changed
    between backend versions. Already imported forecasts are skipped.
    """
    d = Path(pred_dir or paths.PREDICTIONS)
    n = 0
    for f in sorted(d.glob("forecast_*.json")):
        try:
            res = json.loads(f.read_text(encoding="utf-8"))
            with connect(db_path) as con:
                exists = con.execute(
                    "SELECT 1 FROM forecasts WHERE issue_time_local=? AND model IS ? AND feature_config IS ?",
                    (res["forecast_issued_local"], res.get("model"), res.get("feature_config"))).fetchone()
            if not exists:
                save_forecast(res, kind="imported", db_path=db_path)
                n += 1
        except (KeyError, ValueError, TypeError):
            continue
    return n


def save_observation(target_date: str, actual: float | None, n_hours: int, db_path: Path | None = None) -> None:
    with connect(db_path) as con:
        con.execute("""INSERT INTO observations (target_date, actual, n_hours, fetched_at_utc) VALUES (?,?,?,?)
                       ON CONFLICT (target_date) DO UPDATE SET actual=excluded.actual, n_hours=excluded.n_hours,
                       fetched_at_utc=excluded.fetched_at_utc""", (target_date, actual, int(n_hours), _now()))


def forecasts_frame(db_path: Path | None = None) -> pd.DataFrame:
    """All forecasts joined with their verified outcome (if already known)."""
    with connect(db_path) as con:
        df = pd.read_sql_query(
            """SELECT f.*, o.actual, o.n_hours AS actual_hours, o.fetched_at_utc AS verified_at_utc
               FROM forecasts f LEFT JOIN observations o ON o.target_date = f.target_date
               ORDER BY f.target_date DESC, f.created_at_utc DESC""", con)
    if df.empty:
        return df
    df["abs_error"] = (df["predicted"] - df["actual"]).abs()
    df["error"] = df["predicted"] - df["actual"]
    df["provider_abs_error"] = (df["provider_pred"] - df["actual"]).abs()
    df["persistence_abs_error"] = (df["recent_mean_24h"] - df["actual"]).abs()
    df["in_interval"] = np.where(df["actual"].notna() & df["upper"].notna(),
                                 (df["actual"] >= df["lower"].fillna(0)) & (df["actual"] <= df["upper"]), np.nan)
    return df


def pending_dates(today_local: str, db_path: Path | None = None) -> list[str]:
    """Target dates that are complete (before today) but not yet verified."""
    with connect(db_path) as con:
        rows = con.execute(
            """SELECT DISTINCT f.target_date FROM forecasts f
               LEFT JOIN observations o ON o.target_date = f.target_date
               WHERE f.target_date < ? AND (o.target_date IS NULL OR o.actual IS NULL)
               ORDER BY f.target_date""", (today_local,)).fetchall()
    return [r[0] for r in rows]


def events_frame(limit: int = 50, db_path: Path | None = None) -> pd.DataFrame:
    with connect(db_path) as con:
        return pd.read_sql_query("SELECT ts_utc, event, detail FROM events ORDER BY id DESC LIMIT ?", con,
                                 params=(limit,))


def delete_forecast(forecast_id: int, db_path: Path | None = None) -> None:
    with connect(db_path) as con:
        con.execute("DELETE FROM forecasts WHERE id = ?", (int(forecast_id),))
        con.execute("INSERT INTO events (ts_utc, event, detail) VALUES (?, ?, ?)",
                    (_now(), "forecast_deleted", str(forecast_id)))


def summary(df: pd.DataFrame) -> dict[str, Any]:
    """Live accuracy statistics over verified forecasts (one forecast per target date: the latest live one)."""
    out: dict[str, Any] = {"n_total": int(len(df)), "n_verified": 0}
    if df.empty:
        return out
    v = df[df["actual"].notna()]
    out["n_verified"] = int(len(v))
    out["n_pending"] = int(df["actual"].isna().sum())
    if v.empty:
        return out
    out["mae"] = float(v["abs_error"].mean())
    out["bias"] = float(v["error"].mean())
    if v["provider_abs_error"].notna().any():
        out["provider_mae"] = float(v["provider_abs_error"].mean())
    if v["persistence_abs_error"].notna().any():
        out["persistence_mae"] = float(v["persistence_abs_error"].mean())
    iv = v["in_interval"].dropna()
    if len(iv):
        out["interval_hit_rate"] = float(iv.astype(float).mean())
    return out
