"""Settlement history, spread history and the fed-funds future (no network)."""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cso import massive

SECRET = "SECRET-KEY-123"


class FakeResp:
    def __init__(self, payload=None, status=200):
        self._payload, self.status_code = payload, status
        self.ok = 200 <= status < 300

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _install(monkeypatch, responses):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append({"url": url, "params": params, "headers": headers})
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(massive.requests, "get", fake_get)
    monkeypatch.setattr(massive.time, "sleep", lambda s: None)
    return calls


def test_history_prefers_settlement_over_close_and_comes_back_ascending(monkeypatch):
    bars = {"status": "OK", "results": [          # the API returns newest first
        {"session_end_date": "2026-10-26", "settlement_price": 0, "close": 454.25},     # not settled yet: use the close
        {"session_end_date": "2026-10-23", "settlement_price": 455.0, "close": 454.0},
        {"session_end_date": "2026-10-22", "settlement_price": None, "close": 456.5},
        {"session_end_date": "2026-10-21", "close": 0},                                 # nothing usable: skipped
        {"close": 450.0},                                                               # no date: skipped
    ]}
    calls = _install(monkeypatch, [FakeResp(bars)])
    s = massive.fetch_history(SECRET, "ZCZ6")
    assert list(s.index) == [date(2026, 10, 22), date(2026, 10, 23), date(2026, 10, 26)]
    assert list(s.values) == [456.5, 455.0, 454.25]
    assert calls[0]["url"].endswith("/futures/v1/aggs/ZCZ6")
    assert calls[0]["params"] == {"resolution": "1session", "limit": 50000}             # the call shape that works
    assert calls[0]["headers"]["Authorization"] == f"Bearer {SECRET}" and SECRET not in calls[0]["url"]


def test_history_of_nothing_is_an_empty_series(monkeypatch):
    _install(monkeypatch, [FakeResp({"status": "OK", "results": []})])
    assert massive.fetch_history(SECRET, "ZCZ6").empty


def test_a_rejected_key_stops_history_at_once(monkeypatch):
    calls = _install(monkeypatch, [FakeResp({"status": "ERROR"}, status=403)])
    with pytest.raises(massive.MassiveError, match="rejected the API key"):
        massive.fetch_history(SECRET, "ZCZ6")
    assert len(calls) == 1


def test_one_failed_history_does_not_sink_the_others(monkeypatch):
    def fake(api_key, ticker, base_url=massive.BASE_URL, timeout=60.0):
        if ticker == "ZCH7":
            raise massive.MassiveError("boom")
        return pd.Series({date(2026, 10, 26): 454.25})

    monkeypatch.setattr(massive, "fetch_history", fake)
    out = massive.fetch_histories(SECRET, ["ZCZ6", "ZCH7", None, "ZCZ6"])
    assert sorted(out) == ["ZCH7", "ZCZ6"] and out["ZCH7"].empty and len(out["ZCZ6"]) == 1
    assert massive.fetch_histories(SECRET, []) == {}


def test_spread_history_uses_sessions_both_legs_traded_inside_the_window():
    near = pd.Series({date(2025, 1, 2): 440.0, date(2026, 10, 22): 456.0, date(2026, 10, 23): 455.0,
                      date(2026, 10, 26): 454.0})
    far = pd.Series({date(2026, 10, 22): 470.0, date(2026, 10, 26): 468.5})
    both = massive.spread_history(near, far, days=None)
    assert list(both.index) == [date(2026, 10, 22), date(2026, 10, 26)] and list(both.values) == [-14.0, -14.5]
    assert massive.spread_history(near, far, days=365, today=date(2026, 10, 26)).equals(both)
    assert len(massive.spread_history(near, far, days=2, today=date(2026, 10, 26))) == 1
    assert massive.spread_history(near, pd.Series(dtype=float)).empty


def test_fed_funds_is_the_front_contract_that_has_not_expired(monkeypatch):
    seen = {}

    def fake(api_key, tickers, base_url=massive.BASE_URL, timeout=12.0):
        seen["tickers"] = list(tickers)
        return {"ZQV6": massive.Quote("ZQV6", 96.20, "last", None, date(2026, 10, 30), True),
                "ZQX6": massive.Quote("ZQX6", 96.09, "last", None, date(2026, 11, 28), True),
                "ZQZ6": massive.Quote("ZQZ6", 95.945, "last", None, date(2027, 1, 4), True)}

    monkeypatch.setattr(massive, "fetch_snapshot", fake)
    assert massive.fed_funds_pct(SECRET, date(2026, 10, 26)) == (3.8, "ZQV6")
    assert seen["tickers"] == ["ZQV6", "ZQX6", "ZQZ6"]
    assert massive.fed_funds_pct(SECRET, date(2026, 11, 2)) == (3.91, "ZQX6")           # the October contract has expired
    monkeypatch.setattr(massive, "fetch_snapshot", lambda *a, **k: {})
    with pytest.raises(massive.MassiveError):
        massive.fed_funds_pct(SECRET, date(2026, 10, 26))


def test_fed_funds_tickers_roll_into_the_next_year(monkeypatch):
    seen = {}

    def fake(api_key, tickers, base_url=massive.BASE_URL, timeout=12.0):
        seen["t"] = list(tickers)
        return {}

    monkeypatch.setattr(massive, "fetch_snapshot", fake)
    with pytest.raises(massive.MassiveError):
        massive.fed_funds_pct(SECRET, date(2026, 12, 15))
    assert seen["t"] == ["ZQZ6", "ZQF7", "ZQG7"]
