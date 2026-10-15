"""Massive futures client: contract labels, credentials, parsing and error handling (no network)."""
import sys
from datetime import date
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cso import massive

TODAY = date(2026, 10, 15)
SECRET = "SECRET-KEY-123"


@pytest.mark.parametrize("label, expected", [
    ("Dec26", ("Z", 6)), ("Dec 2026", ("Z", 6)), ("DEC-26", ("Z", 6)), ("dec'26", ("Z", 6)),
    ("Mar27", ("H", 7)), ("March 2027", ("H", 7)), ("Z26", ("Z", 6)), ("Z6", ("Z", 6)),
    ("ZCZ6", ("Z", 6)), ("ZCH27", ("H", 7)), ("KEH7", ("H", 7)),
    ("Dec", ("Z", 6)), ("Jan", ("F", 7)), ("Oct", ("V", 6)), ("Sep", ("U", 7)),   # no year: next one on/after today
])
def test_parse_contract(label, expected):
    assert massive.parse_contract(label, TODAY) == expected


@pytest.mark.parametrize("bad", ["", None, "   ", "nan", "Foo26", "ZC", "13", "Dec26x"])
def test_unreadable_labels_are_none(bad):
    assert massive.parse_contract(bad, TODAY) is None
    assert massive.ticker_for("ZC", bad, TODAY) is None


def test_ticker_for():
    assert massive.ticker_for("ZC", "Dec26", TODAY) == "ZCZ6"
    assert massive.ticker_for("ZL", "Mar27", TODAY) == "ZLH7"
    assert massive.ticker_for(None, "Dec26", TODAY) is None
    assert massive.PRODUCT_CODES["Corn (¢/bu)"] == "ZC"


def test_api_key_sources_and_missing_secrets_file():
    assert massive.api_key_from({"MASSIVE_API_KEY": " abc "}, {}) == "abc"
    assert massive.api_key_from({"massive": {"api_key": "k2"}}, {}) == "k2"
    assert massive.api_key_from({}, {"MASSIVE_API_KEY": "env"}) == "env"
    assert massive.api_key_from({"MASSIVE_API_KEY": "   "}, {}) is None

    class NoSecretsFile:                       # st.secrets raises like this when there is no secrets.toml
        def __contains__(self, key):
            raise FileNotFoundError("no secrets")

    assert massive.api_key_from(NoSecretsFile(), {"MASSIVE_API_KEY": "env"}) == "env"
    assert massive.api_key_from(NoSecretsFile(), {}) is None


# --------------------------------------------------------------------------- fetching
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


PAYLOAD = {"status": "OK", "results": [
    {"details": {"ticker": "ZCZ6", "settlement_date": "2026-12-14"},
     "last_trade": {"price": 478.25, "timeframe": "DELAYED", "last_updated": 1792088398018004245},
     "session": {"settlement_price": 478.0}},
    {"details": {"ticker": "ZCH7", "settlement_date": "2027-03-12"}, "last_trade": {},
     "session": {"settlement_price": 491.0, "close": 490.5}},
    {"details": {"ticker": "ZCK7"}, "session": {"close": 495.0}},
    {"details": {"ticker": "ZCN7"}, "last_quote": {"bid": 500.0, "ask": 500.5}},
    {"details": {"ticker": "ZCU7"}},                       # nothing priced: skipped
]}


def test_prices_follow_the_fallback_chain_and_the_key_stays_in_the_header(monkeypatch):
    calls = _install(monkeypatch, [FakeResp(PAYLOAD)])
    q = massive.fetch_snapshot(SECRET, ["ZCZ6", "ZCH7", "ZCK7", "ZCN7", "ZCU7", "ZCZ6", None])
    assert len(calls) == 1
    assert calls[0]["params"] == {"ticker.any_of": "ZCH7,ZCK7,ZCN7,ZCU7,ZCZ6"}        # de-duplicated and sorted
    assert calls[0]["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert SECRET not in str(calls[0]["url"]) + str(calls[0]["params"])                # never in the URL
    assert (q["ZCZ6"].price, q["ZCZ6"].source, q["ZCZ6"].delayed) == (478.25, "last", True)
    assert q["ZCZ6"].as_of.tzinfo is not None and q["ZCZ6"].as_of.year == 2026
    assert q["ZCZ6"].expiry == date(2026, 12, 14)
    assert (q["ZCH7"].price, q["ZCH7"].source, q["ZCH7"].as_of) == (491.0, "settlement", None)
    assert (q["ZCK7"].price, q["ZCK7"].source) == (495.0, "close")
    assert (q["ZCN7"].price, q["ZCN7"].source) == (500.25, "mid")
    assert "ZCU7" not in q


def test_nothing_to_fetch_makes_no_call(monkeypatch):
    calls = _install(monkeypatch, [])
    assert massive.fetch_snapshot(SECRET, [None, ""]) == {}
    assert calls == []


def test_a_rejected_key_fails_at_once_without_leaking_it(monkeypatch):
    calls = _install(monkeypatch, [FakeResp({"status": "ERROR", "error": "bad key"}, status=401)])
    with pytest.raises(massive.MassiveError) as err:
        massive.fetch_snapshot(SECRET, ["ZCZ6"])
    assert "rejected the API key" in str(err.value) and SECRET not in str(err.value)
    assert len(calls) == 1                                   # no pointless retry


def test_one_transient_failure_is_retried(monkeypatch):
    calls = _install(monkeypatch, [FakeResp(None, status=502), FakeResp(PAYLOAD)])
    assert "ZCZ6" in massive.fetch_snapshot(SECRET, ["ZCZ6"])
    assert len(calls) == 2


@pytest.mark.parametrize("responses, text", [
    ([FakeResp(None, 502), FakeResp(None, 502)], "non-JSON"),
    ([FakeResp({"status": "ERROR", "error": "boom"}, 500), FakeResp({"status": "ERROR", "error": "boom"}, 500)],
     "reported an error: boom"),
    ([requests.ConnectionError(f"https://x/?k={SECRET}"), requests.ConnectionError("again")], "could not reach"),
])
def test_two_failures_raise_a_clean_message(monkeypatch, responses, text):
    _install(monkeypatch, responses)
    with pytest.raises(massive.MassiveError) as err:
        massive.fetch_snapshot(SECRET, ["ZCZ6"])
    assert text in str(err.value) and SECRET not in str(err.value)
