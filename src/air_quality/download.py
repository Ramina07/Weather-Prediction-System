"""Resumable, chunked download of OpenWeatherMap historical air pollution data.

Design:

* The requested period is split into fixed UTC chunks (default 7 days).
* Each chunk request covers ``[start, next_start - 1 s]`` so neighbouring
  chunks never share a boundary timestamp; any residual duplicates are still
  removed (and counted) when the chunks are combined.
* Each successful response is cached verbatim as JSON in ``data/raw/chunks``
  (raw-data preservation) and recorded in ``manifest.json``.
* If a chunk returns less than ``split_if_coverage_below`` of its expected
  hourly records, it is split in half and retried, down to
  ``min_chunk_hours``. Below that size the observed coverage is accepted and
  recorded as a genuine provider gap.
* Failed chunks are recorded in the manifest with the (redacted) error; they
  are never silently ignored and are retried on the next run.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .api_client import (APIAuthError, APIError, APIForbiddenError, COMPONENT_FIELDS,
                         OpenWeatherAirPollutionClient)
from .config import Config, get_logger

HOUR = 3600
RAW_COLUMNS = ["dt_unix", "timestamp_utc", "timestamp_local", "date_local", "latitude", "longitude",
               "provider", "aqi", *COMPONENT_FIELDS]


@dataclass
class DownloadSummary:
    n_chunks_planned: int
    n_chunks_downloaded: int
    n_chunks_cached: int
    n_chunks_split: int
    n_chunks_failed: int
    n_requests: int
    n_records_returned: int
    failed_intervals: list[tuple[str, str, str]]


def local_dates_to_utc_unix(start_date: str, end_date: str, tz: str, lead_in_days: int = 0) -> tuple[int, int]:
    """Convert inclusive local dates into a half-open UTC Unix interval."""
    start = pd.Timestamp(start_date).tz_localize(tz) - pd.Timedelta(days=lead_in_days)
    end = pd.Timestamp(end_date).tz_localize(tz) + pd.Timedelta(days=1)
    return int(start.tz_convert("UTC").timestamp()), int(end.tz_convert("UTC").timestamp())


def plan_chunks(start_unix: int, end_unix: int, chunk_days: int) -> list[tuple[int, int]]:
    """Half-open ``[s, e)`` chunks covering ``[start_unix, end_unix)``."""
    step = int(chunk_days * 24 * HOUR)
    chunks, s = [], int(start_unix)
    while s < end_unix:
        e = min(s + step, int(end_unix))
        chunks.append((s, e))
        s = e
    return chunks


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, dict[str, Any]] = {}
        if path.exists():
            self.entries = json.loads(path.read_text(encoding="utf-8")).get("chunks", {})

    @staticmethod
    def key(s: int, e: int) -> str:
        return f"{s}_{e}"

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"updated_utc": _iso(int(datetime.now(timezone.utc).timestamp())),
                                   "chunks": self.entries}, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def is_done(self, s: int, e: int, chunk_dir: Path) -> bool:
        ent = self.entries.get(self.key(s, e))
        if not ent or ent.get("status") not in {"ok", "split"}:
            return False
        if ent["status"] == "split":
            return False  # children are checked individually
        return (chunk_dir / ent["file"]).exists()


def download_history(cfg: Config, client: OpenWeatherAirPollutionClient, start_unix: int, end_unix: int,
                     chunk_days: int | None = None) -> DownloadSummary:
    """Download (or resume) all chunks for ``[start_unix, end_unix)``."""
    log = get_logger()
    d = cfg["download"]
    chunk_days = chunk_days or d["chunk_days"]
    lat, lon = cfg["location"]["latitude"], cfg["location"]["longitude"]
    chunk_dir = cfg.path("raw_chunks_dir")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    man = Manifest(cfg.path("manifest"))
    queue = deque(plan_chunks(start_unix, end_unix, chunk_days))
    n_planned = len(queue)
    downloaded = cached = split = 0
    n_records = 0
    failed: list[tuple[str, str, str]] = []
    req_before = client.n_requests

    while queue:
        s, e = queue.popleft()
        key = man.key(s, e)
        ent = man.entries.get(key)
        if ent and ent.get("status") == "split":
            mid = s + ((e - s) // 2 // HOUR) * HOUR
            queue.extendleft([(mid, e), (s, mid)])
            continue
        if man.is_done(s, e, chunk_dir):
            cached += 1
            n_records += man.entries[key]["n_records"]
            continue
        expected = (e - s) // HOUR
        try:
            parsed = client.history(lat, lon, s, e - 1)
        except (APIAuthError, APIForbiddenError):
            man.save()
            raise
        except APIError as exc:
            man.entries[key] = {"status": "failed", "start_utc": _iso(s), "end_utc": _iso(e),
                                "error": str(exc), "attempted_at": _iso(int(datetime.now(timezone.utc).timestamp()))}
            failed.append((_iso(s), _iso(e), str(exc)))
            log.error("Chunk %s .. %s failed: %s", _iso(s), _iso(e), exc)
            man.save()
            continue
        n = len(parsed.records)
        coverage = n / expected if expected else 0.0
        hours = (e - s) / HOUR
        if coverage < d["split_if_coverage_below"] and hours >= 2 * d["min_chunk_hours"]:
            mid = s + ((e - s) // 2 // HOUR) * HOUR
            man.entries[key] = {"status": "split", "start_utc": _iso(s), "end_utc": _iso(e),
                                "n_records_first_try": n, "expected": int(expected)}
            log.warning("Chunk %s .. %s returned %d/%d records (%.0f%%): splitting",
                        _iso(s), _iso(e), n, expected, 100 * coverage)
            queue.extendleft([(mid, e), (s, mid)])
            split += 1
            man.save()
            continue
        fname = f"chunk_{s}_{e}.json"
        (chunk_dir / fname).write_text(json.dumps(parsed.raw), encoding="utf-8")
        man.entries[key] = {"status": "ok", "start_utc": _iso(s), "end_utc": _iso(e), "file": fname,
                            "n_records": n, "expected": int(expected), "coverage": round(coverage, 4),
                            "n_missing_pm25": parsed.n_missing_pm25,
                            "downloaded_at": _iso(int(datetime.now(timezone.utc).timestamp()))}
        downloaded += 1
        n_records += n
        man.save()
        if downloaded % 20 == 0:
            log.info("... %d chunks downloaded (%d records so far)", downloaded, n_records)

    man.save()
    return DownloadSummary(n_planned, downloaded, cached, split, len(failed),
                           client.n_requests - req_before, n_records, failed)


def combine_chunks(cfg: Config, start_unix: int, end_unix: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge cached chunk JSONs inside the interval into one raw hourly table."""
    from .api_client import parse_air_pollution_payload
    man = Manifest(cfg.path("manifest"))
    chunk_dir = cfg.path("raw_chunks_dir")
    rows: list[dict[str, Any]] = []
    used = 0
    coord_lat, coord_lon = None, None
    for ent in man.entries.values():
        if ent.get("status") != "ok":
            continue
        s = int(pd.Timestamp(ent["start_utc"]).timestamp())
        e = int(pd.Timestamp(ent["end_utc"]).timestamp())
        if e <= start_unix or s >= end_unix:
            continue
        payload = json.loads((chunk_dir / ent["file"]).read_text(encoding="utf-8"))
        parsed = parse_air_pollution_payload(payload)
        coord_lat = parsed.lat if parsed.lat is not None else coord_lat
        coord_lon = parsed.lon if parsed.lon is not None else coord_lon
        rows.extend(parsed.records)
        used += 1
    info: dict[str, Any] = {"chunks_used": used, "records_before_dedup": len(rows)}
    if not rows:
        return pd.DataFrame(columns=RAW_COLUMNS), {**info, "duplicates_removed": 0}
    df = pd.DataFrame(rows).rename(columns={"dt": "dt_unix"})
    df = df[(df["dt_unix"] >= start_unix) & (df["dt_unix"] < end_unix)]
    before = len(df)
    df = df.sort_values("dt_unix").drop_duplicates(subset=["dt_unix"], keep="first")
    info["duplicates_removed"] = int(before - len(df))
    info["records_outside_interval"] = int(len(rows) - before)
    tz = cfg.tz
    ts = pd.to_datetime(df["dt_unix"], unit="s", utc=True)
    df.insert(1, "timestamp_utc", ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    local = ts.dt.tz_convert(tz)
    df.insert(2, "timestamp_local", local.map(lambda t: t.isoformat()))
    df.insert(3, "date_local", local.dt.strftime("%Y-%m-%d"))
    df.insert(4, "latitude", coord_lat if coord_lat is not None else cfg["location"]["latitude"])
    df.insert(5, "longitude", coord_lon if coord_lon is not None else cfg["location"]["longitude"])
    df.insert(6, "provider", "OpenWeatherMap Air Pollution API (history)")
    info["api_returned_coordinates"] = {"lat": coord_lat, "lon": coord_lon}
    return df[RAW_COLUMNS].reset_index(drop=True), info


def write_metadata(cfg: Config, df: pd.DataFrame, summary: DownloadSummary | None, combine_info: dict,
                   start_unix: int, end_unix: int, path: Path) -> dict[str, Any]:
    meta = {
        "provider": cfg["api"]["provider"],
        "endpoint": f"{cfg['api']['base_url']}/air_pollution/history",
        "documentation": "https://openweathermap.org/api/air-pollution",
        "location": cfg["location"],
        "api_returned_coordinates": combine_info.get("api_returned_coordinates"),
        "requested_interval_utc": [_iso(start_unix), _iso(end_unix)],
        "timestamp_convention": "dt = start of the hourly interval, Unix seconds UTC",
        "units": {f: "ug/m3" for f in COMPONENT_FIELDS} | {"aqi": "index 1 (Good) - 5 (Very Poor)"},
        "n_records": int(len(df)),
        "first_timestamp_utc": df["timestamp_utc"].iloc[0] if len(df) else None,
        "last_timestamp_utc": df["timestamp_utc"].iloc[-1] if len(df) else None,
        "combine": combine_info,
        "download": summary.__dict__ if summary else None,
        "created_utc": _iso(int(datetime.now(timezone.utc).timestamp())),
        "note": ("Values are API-provided (modelled/aggregated) concentrations for the requested "
                 "coordinate, not direct measurements from a specific ground station."),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta


def run_download(cfg: Config, client: OpenWeatherAirPollutionClient, start_date: str | None = None,
                 end_date: str | None = None, operational: bool = False) -> tuple[pd.DataFrame, DownloadSummary]:
    """Download, combine and save the raw hourly CSV + metadata."""
    log = get_logger()
    d = cfg["download"]
    if operational:
        start_date = start_date or d["operational_start_date"]
        now_local = pd.Timestamp.now(tz=cfg.tz)
        end_date = end_date or now_local.strftime("%Y-%m-%d")
        lead = d["lead_in_days"]
        out_csv = cfg.path("raw_operational_csv")
        meta_path = out_csv.with_name("dataset_metadata_operational.json")
    else:
        start_date, end_date = start_date or d["start_date"], end_date or d["end_date"]
        lead = d["lead_in_days"]
        out_csv, meta_path = cfg.path("raw_csv"), cfg.path("raw_metadata")
    s, e = local_dates_to_utc_unix(start_date, end_date, cfg.tz, lead)
    now_unix = int(datetime.now(timezone.utc).timestamp())
    e = min(e, (now_unix // HOUR) * HOUR)  # never request the future
    log.info("Downloading %s .. %s (local, lead-in %d d) = UTC %s .. %s",
             start_date, end_date, lead, _iso(s), _iso(e))
    summary = download_history(cfg, client, s, e)
    log.info("Requests: %d | downloaded chunks: %d | cached: %d | split: %d | failed: %d",
             summary.n_requests, summary.n_chunks_downloaded, summary.n_chunks_cached,
             summary.n_chunks_split, summary.n_chunks_failed)
    df, info = combine_chunks(cfg, s, e)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    write_metadata(cfg, df, summary, info, s, e, meta_path)
    log.info("Saved %d raw hourly records to %s", len(df), out_csv)
    return df, summary
