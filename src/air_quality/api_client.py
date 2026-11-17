"""Client for the OpenWeatherMap Air Pollution API (current, history, forecast).

Endpoints (official documentation: https://openweathermap.org/api/air-pollution):

* ``/data/2.5/air_pollution``           current conditions
* ``/data/2.5/air_pollution/history``   hourly history (from 27 Nov 2020)
* ``/data/2.5/air_pollution/forecast``  provider forecast (NOT our ML model)

All concentrations in ``components`` are reported by the provider in ug/m3;
``main.aqi`` is the provider's 1-5 Air Quality Index.

Error handling: 401/403 are raised immediately (no retry); 429, 5xx, timeouts
and interrupted connections are retried with exponential backoff. Every
exception message is passed through :func:`air_quality.config.redact` so the
API key can never leak through a traceback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import requests

from .config import get_logger, redact, register_secret

COMPONENT_FIELDS = ("co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3")


# --------------------------------------------------------------------------- errors
class APIError(RuntimeError):
    """Base class for all API errors (messages are always redacted)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(redact(message))
        self.status = status


class APIAuthError(APIError):
    """HTTP 401: invalid or inactive key."""


class APIForbiddenError(APIError):
    """HTTP 403: key valid but subscription does not allow this endpoint."""


class APIRateLimitError(APIError):
    """HTTP 429 after all retries."""


class APITimeoutError(APIError):
    """Request timed out after all retries."""


class APIConnectionError(APIError):
    """Network failure / interrupted connection after all retries."""


class APIResponseError(APIError):
    """Malformed JSON, unexpected schema, or missing required fields."""


# --------------------------------------------------------------------------- parsing
@dataclass
class ParsedResponse:
    records: list[dict[str, Any]]
    lat: float | None
    lon: float | None
    n_missing_pm25: int
    n_missing_fields: int
    raw: Any = None  # original JSON payload, preserved for provenance


def parse_air_pollution_payload(payload: Any) -> ParsedResponse:
    """Validate the JSON schema and flatten it into hourly records.

    Raises :class:`APIResponseError` if the payload is not a dict with a
    ``list`` array, or if an element lacks ``dt``. Missing pollutant fields are
    kept as ``None`` (counted, never invented).
    """
    if not isinstance(payload, dict):
        raise APIResponseError(f"Unexpected payload type: {type(payload).__name__}")
    if "list" not in payload or not isinstance(payload["list"], list):
        raise APIResponseError("Response is missing the 'list' array")
    coord = payload.get("coord") or {}
    lat, lon = coord.get("lat"), coord.get("lon")
    records: list[dict[str, Any]] = []
    n_missing_pm25 = 0
    n_missing_fields = 0
    for i, item in enumerate(payload["list"]):
        if not isinstance(item, dict) or "dt" not in item:
            raise APIResponseError(f"list[{i}] has no 'dt' timestamp")
        try:
            dt = int(item["dt"])
        except (TypeError, ValueError):
            raise APIResponseError(f"list[{i}].dt is not an integer Unix timestamp") from None
        comps = item.get("components") or {}
        if not isinstance(comps, dict):
            raise APIResponseError(f"list[{i}].components is not an object")
        main = item.get("main") or {}
        rec: dict[str, Any] = {"dt": dt, "aqi": _to_float(main.get("aqi"))}
        for f in COMPONENT_FIELDS:
            v = _to_float(comps.get(f))
            if v is None:
                n_missing_fields += 1
                if f == "pm2_5":
                    n_missing_pm25 += 1
            rec[f] = v
        records.append(rec)
    return ParsedResponse(records, _to_float(lat), _to_float(lon), n_missing_pm25, n_missing_fields, payload)


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- client
class OpenWeatherAirPollutionClient:
    """Rate-limited, retrying HTTP client."""

    def __init__(self, api_key: str, base_url: str = "https://api.openweathermap.org/data/2.5",
                 timeout: float = 30.0, max_retries: int = 5, backoff_base: float = 2.0,
                 min_interval: float = 1.1, session: requests.Session | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        if not api_key:
            raise ValueError("api_key must be a non-empty string")
        register_secret(api_key)
        self._key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.min_interval = min_interval
        self.session = session or requests.Session()
        self._sleep = sleep
        self._last_call = 0.0
        self.n_requests = 0
        self.log = get_logger()

    @classmethod
    def from_config(cls, cfg, api_key: str, **kw) -> "OpenWeatherAirPollutionClient":
        a = cfg["api"]
        return cls(api_key, a["base_url"], a["timeout_seconds"], a["max_retries"],
                   a["backoff_base_seconds"], a["min_seconds_between_calls"], **kw)

    # ------------------------------------------------------------------ low level
    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            self._sleep(wait)
        self._last_call = time.monotonic()

    def _get(self, path: str, params: dict[str, Any]) -> tuple[Any, int]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        params = {**params, "appid": self._key}
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.n_requests += 1
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.Timeout as exc:
                last_exc = APITimeoutError(f"Timeout after {self.timeout}s calling /{path}: {exc}")
            except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
                last_exc = APIConnectionError(f"Connection error calling /{path}: {exc}")
            except requests.RequestException as exc:
                last_exc = APIConnectionError(f"Request failed calling /{path}: {exc}")
            else:
                status = resp.status_code
                if status == 200:
                    try:
                        return resp.json(), status
                    except ValueError:
                        raise APIResponseError(f"Malformed JSON from /{path}", status) from None
                if status == 401:
                    raise APIAuthError("HTTP 401 Unauthorized: the API key is invalid or not yet "
                                       "activated (new keys can take up to ~2 hours).", status)
                if status == 403:
                    raise APIForbiddenError("HTTP 403 Forbidden: the key is not permitted to access "
                                            f"/{path} under the current subscription.", status)
                if status == 429:
                    last_exc = APIRateLimitError("HTTP 429 Too Many Requests: rate limit exceeded.", status)
                elif 500 <= status < 600:
                    last_exc = APIConnectionError(f"HTTP {status} server error from /{path}", status)
                else:
                    raise APIError(f"HTTP {status} from /{path}: {resp.text[:200]}", status)
            if attempt < self.max_retries:
                delay = self.backoff_base * (2 ** attempt)
                self.log.warning("%s -- retry %d/%d in %.1fs", last_exc, attempt + 1, self.max_retries, delay)
                self._sleep(delay)
        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------ endpoints
    def current(self, lat: float, lon: float) -> ParsedResponse:
        payload, _ = self._get("air_pollution", {"lat": lat, "lon": lon})
        return parse_air_pollution_payload(payload)

    def history(self, lat: float, lon: float, start_unix: int, end_unix: int) -> ParsedResponse:
        if end_unix <= start_unix:
            raise ValueError("end_unix must be greater than start_unix")
        payload, _ = self._get("air_pollution/history",
                               {"lat": lat, "lon": lon, "start": int(start_unix), "end": int(end_unix)})
        return parse_air_pollution_payload(payload)

    def forecast(self, lat: float, lon: float) -> ParsedResponse:
        payload, _ = self._get("air_pollution/forecast", {"lat": lat, "lon": lon})
        return parse_air_pollution_payload(payload)
