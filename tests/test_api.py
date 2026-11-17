"""API client tests with mocked HTTP responses (no real quota is consumed)."""

from __future__ import annotations

import json

import pytest
import requests

from air_quality.api_client import (APIAuthError, APIConnectionError, APIForbiddenError, APIRateLimitError,
                                    APIResponseError, APITimeoutError, OpenWeatherAirPollutionClient,
                                    parse_air_pollution_payload)
from air_quality.config import MissingAPIKeyError, get_api_key, redact

KEY = "test_secret_key_123456"

PAYLOAD = {"coord": {"lon": 71.4491, "lat": 51.1694},
           "list": [{"main": {"aqi": 2}, "components": {"co": 300.4, "no": 0.1, "no2": 12.3, "o3": 40.0, "so2": 5.1,
                                                        "pm2_5": 11.2, "pm10": 15.0, "nh3": 1.2}, "dt": 1748736000},
                    {"main": {"aqi": 1}, "components": {"co": 280.0, "no": 0.0, "no2": 10.0, "o3": 45.0, "so2": 4.0,
                                                        "pm2_5": 8.0, "pm10": 11.0, "nh3": 1.0}, "dt": 1748739600}]}


class FakeResp:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def client(responses, retries=2):
    return OpenWeatherAirPollutionClient(KEY, session=FakeSession(responses), max_retries=retries,
                                         min_interval=0, backoff_base=0, sleep=lambda s: None)


def test_parse_valid_payload():
    p = parse_air_pollution_payload(PAYLOAD)
    assert len(p.records) == 2
    assert p.records[0]["pm2_5"] == 11.2 and p.records[0]["aqi"] == 2.0
    assert p.lat == 51.1694 and p.n_missing_pm25 == 0


def test_parse_missing_pm25_is_counted_not_invented():
    pl = json.loads(json.dumps(PAYLOAD))
    del pl["list"][0]["components"]["pm2_5"]
    p = parse_air_pollution_payload(pl)
    assert p.records[0]["pm2_5"] is None
    assert p.n_missing_pm25 == 1


@pytest.mark.parametrize("bad", [[], "x", {"nolist": 1}, {"list": "notalist"}, {"list": [{"main": {}}]}])
def test_parse_malformed_schema(bad):
    with pytest.raises(APIResponseError):
        parse_air_pollution_payload(bad)


def test_empty_result_returns_no_records():
    p = client([FakeResp(200, {"coord": {}, "list": []})]).history(51, 71, 0, 3600)
    assert p.records == []


def test_http_200_history_and_params():
    c = client([FakeResp(200, PAYLOAD)])
    p = c.history(51.1694, 71.4491, 1748736000, 1748822400)
    assert len(p.records) == 2
    url, params = c.session.calls[0]
    assert url.endswith("/air_pollution/history")
    assert params["start"] == 1748736000 and params["end"] == 1748822400


def test_401_raises_without_retry():
    c = client([FakeResp(401, {"cod": 401}), FakeResp(200, PAYLOAD)])
    with pytest.raises(APIAuthError):
        c.current(51, 71)
    assert len(c.session.calls) == 1


def test_403_raises():
    with pytest.raises(APIForbiddenError):
        client([FakeResp(403, {"cod": 403})]).current(51, 71)


def test_429_retries_then_succeeds():
    c = client([FakeResp(429, {}), FakeResp(200, PAYLOAD)])
    assert len(c.current(51, 71).records) == 2
    assert len(c.session.calls) == 2


def test_429_exhausted():
    with pytest.raises(APIRateLimitError):
        client([FakeResp(429, {})] * 3, retries=2).current(51, 71)


def test_timeout_handling():
    with pytest.raises(APITimeoutError):
        client([requests.Timeout("t")] * 3, retries=2).current(51, 71)


def test_interrupted_connection_then_success():
    c = client([requests.ConnectionError("reset"), FakeResp(200, PAYLOAD)])
    assert len(c.current(51, 71).records) == 2


def test_interrupted_connection_exhausted():
    with pytest.raises(APIConnectionError):
        client([requests.exceptions.ChunkedEncodingError("cut")] * 3).current(51, 71)


def test_invalid_json():
    with pytest.raises(APIResponseError):
        client([FakeResp(200, None, text="<html>")]).current(51, 71)


def test_key_never_in_error_message():
    err = requests.ConnectionError(f"failed https://x/air_pollution?lat=1&appid={KEY}")
    with pytest.raises(APIConnectionError) as ei:
        client([err] * 3).current(51, 71)
    assert KEY not in str(ei.value)
    assert KEY not in redact(f"appid={KEY}")


def test_missing_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENWEATHER_API_KEY=your_openweathermap_api_key_here\n")
    with pytest.raises(MissingAPIKeyError):
        get_api_key(tmp_path)
    assert get_api_key(tmp_path, required=False) is None


def test_api_key_loaded_from_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    (tmp_path / ".env").write_text(f"OPENWEATHER_API_KEY={KEY}\n")
    assert get_api_key(tmp_path) == KEY
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
