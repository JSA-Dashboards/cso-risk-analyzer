"""The live curve (contract listing + snapshot), contract labels and realized spread vol (no network)."""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cso import massive

SECRET = "SECRET-KEY-123"
TODAY = date(2026, 10, 26)


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


# the shape of the real listing: outrights beside calendar spreads, butterflies, condors and micros
LISTING = {"status": "OK", "results": [
    {"ticker": "ZCH7", "type": "single", "settlement_date": "2027-03-12", "last_trade_date": "2027-03-12"},
    {"ticker": "ZCZ6", "type": "single", "settlement_date": "2026-12-14", "last_trade_date": "2026-12-14"},
    {"ticker": "ZCZ6-ZCH7", "type": "combo", "settlement_date": "2026-12-14"},
    {"ticker": "ZC:BF Z6-H7-K7", "type": "combo", "settlement_date": "2026-12-14"},
    {"ticker": "ZC:CF H7K7N7U7", "type": "combo", "settlement_date": "2027-03-12"},
    {"ticker": "MZCZ6", "type": "single", "settlement_date": "2026-12-14"},          # a micro: another product
    {"ticker": "ZCK7", "type": "single", "last_trade_date": "2027-05-14"},             # only a last trade date
    {"ticker": "ZCN7", "type": "single", "settlement_date": "not-a-date"},             # unreadable: skipped
    {"ticker": "ZCU7", "type": "single"},                                                # no date at all: skipped
]}


# --------------------------------------------------------------------------- labels
@pytest.mark.parametrize("ticker, expected", [
    ("ZCZ6", ("ZC", "Z", 6)), ("ZLH7", ("ZL", "H", 7)), ("KEH7", ("KE", "H", 7)), ("ZMF8", ("ZM", "F", 8)),
])
def test_split_ticker(ticker, expected):
    assert massive.split_ticker(ticker) == expected


@pytest.mark.parametrize("bad", [None, "", "ZC", "ZCZ6-ZCH7", "ZC:BF H7-K7-N7", "ZCZ66", "zcz6", "MZCZ6x"])
def test_split_ticker_refuses_anything_but_a_single_month(bad):
    assert massive.split_ticker(bad) is None


@pytest.mark.parametrize("letter, digit, expiry, label", [
    ("Z", 6, date(2026, 12, 14), "Dec26"), ("H", 7, date(2027, 3, 12), "Mar27"), ("F", 7, date(2027, 1, 14), "Jan27"),
    ("F", 7, date(2026, 12, 31), "Jan27"),            # expires the month before: the year still comes from the digit
    ("Z", 9, date(2029, 12, 14), "Dec29"), ("N", 0, date(2030, 7, 12), "Jul30"),
    ("Z", 3, date(2026, 12, 14), "Dec26"),            # a digit no year around the expiry ends in: fall back to the expiry's year
])
def test_contract_label(letter, digit, expiry, label):
    assert massive.contract_label(letter, digit, expiry) == label


def test_labels_round_trip_through_the_label_parser():
    # the picker's labels are the book's labels, so a picked contract can be typed back into the spreads table
    for ticker, expiry in (("ZCZ6", date(2026, 12, 14)), ("ZCH7", date(2027, 3, 12))):
        _, letter, digit = massive.split_ticker(ticker)
        assert massive.ticker_for("ZC", massive.contract_label(letter, digit, expiry), TODAY) == ticker


# --------------------------------------------------------------------------- the listing
def test_the_listing_keeps_plain_months_of_the_product_nearest_first(monkeypatch):
    calls = _install(monkeypatch, [FakeResp(LISTING)])
    got = massive.fetch_contracts(SECRET, "ZC", TODAY)
    assert got == [("ZCZ6", date(2026, 12, 14)), ("ZCH7", date(2027, 3, 12)), ("ZCK7", date(2027, 5, 14))]
    assert len(calls) == 1 and calls[0]["url"].endswith("/futures/v1/contracts")
    assert calls[0]["params"] == {"product_code": "ZC", "active": "true", "limit": 400, "date": "2026-10-26"}
    assert calls[0]["headers"]["Authorization"] == f"Bearer {SECRET}" and SECRET not in str(calls[0]["params"]) + calls[0]["url"]


def test_a_date_with_no_listing_steps_back_to_the_latest_one(monkeypatch):
    empty = FakeResp({"status": "OK", "results": []})
    calls = _install(monkeypatch, [empty, empty, FakeResp(LISTING)])
    assert len(massive.fetch_contracts(SECRET, "ZC", date(2026, 10, 25))) == 3
    assert [c["params"]["date"] for c in calls] == ["2026-10-25", "2026-10-24", "2026-10-23"]


def test_a_product_with_no_listing_at_all_is_an_empty_curve(monkeypatch):
    calls = _install(monkeypatch, [FakeResp({"status": "OK", "results": []})] * (massive.CONTRACT_LOOKBACK_DAYS + 1))
    assert massive.fetch_contracts(SECRET, "ZC", TODAY) == []
    assert len(calls) == massive.CONTRACT_LOOKBACK_DAYS + 1


def test_a_rejected_key_stops_the_listing_at_once(monkeypatch):
    calls = _install(monkeypatch, [FakeResp({"status": "ERROR"}, status=403)])
    with pytest.raises(massive.MassiveError, match="rejected the API key") as err:
        massive.fetch_contracts(SECRET, "ZC", TODAY)
    assert SECRET not in str(err.value) and len(calls) == 1


# --------------------------------------------------------------------------- the curve
def _quote(ticker, price, expiry):
    return massive.Quote(ticker, price, "last", None, expiry, True)


def test_the_curve_joins_prices_labels_and_drops_what_has_expired_or_has_no_price(monkeypatch):
    listed = [("ZCU6", date(2026, 9, 14)), ("ZCZ6", date(2026, 12, 14)), ("ZCH7", date(2027, 3, 12)),
              ("ZCK7", date(2027, 5, 14)), ("ZCN7", date(2027, 7, 14))]
    seen = {}
    monkeypatch.setattr(massive, "fetch_contracts", lambda key, product, as_of, base_url=massive.BASE_URL, timeout=20.0: listed)

    def snap(key, tickers, base_url=massive.BASE_URL, timeout=12.0):
        seen["tickers"] = list(tickers)
        return {"ZCZ6": _quote("ZCZ6", 454.25, listed[1][1]), "ZCK7": _quote("ZCK7", 480.0, listed[3][1]),
                "ZCN7": _quote("ZCN7", 485.5, listed[4][1])}               # ZCH7 has no price

    monkeypatch.setattr(massive, "fetch_snapshot", snap)
    curve = massive.fetch_curve(SECRET, "ZC", TODAY)
    assert [(c.ticker, c.label, c.price, c.expiry) for c in curve] == [
        ("ZCZ6", "Dec26", 454.25, date(2026, 12, 14)), ("ZCK7", "May27", 480.0, date(2027, 5, 14)),
        ("ZCN7", "Jul27", 485.5, date(2027, 7, 14))]
    assert "ZCU6" not in seen["tickers"]                                  # the September contract expired in September


def test_the_curve_is_cut_to_the_requested_number_of_months(monkeypatch):
    listed = [(f"ZC{m}7", date(2027, i + 1, 14)) for i, m in enumerate("FGHJKMNQUVXZ")]
    monkeypatch.setattr(massive, "fetch_contracts", lambda *a, **k: listed)
    monkeypatch.setattr(massive, "fetch_snapshot",
                        lambda key, tickers, *a, **k: {t: _quote(t, 450.0 + i, date(2027, 1, 14)) for i, t in enumerate(tickers)})
    curve = massive.fetch_curve(SECRET, "ZC", TODAY, n=4)
    assert [c.label for c in curve] == ["Jan27", "Feb27", "Mar27", "Apr27"]


def test_an_empty_listing_is_an_empty_curve_without_a_snapshot_call(monkeypatch):
    calls = _install(monkeypatch, [])
    monkeypatch.setattr(massive, "fetch_contracts", lambda *a, **k: [])
    assert massive.fetch_curve(SECRET, "ZC", TODAY) == []
    assert calls == []                                                    # nothing listed, so nothing to price


def test_the_two_steps_the_app_caches_separately_compose_the_same_curve():
    listing = [("ZCU6", date(2026, 9, 14)), ("ZCZ6", date(2026, 12, 14)), ("ZCH7", date(2027, 3, 12)),
               ("ZCK7", date(2027, 5, 14)), ("ZCN7", date(2027, 7, 14)), ("ZCU7", date(2027, 9, 14))]
    months = massive.live_months(listing, TODAY, n=2)
    assert [t for t, _ in months] == ["ZCZ6", "ZCH7", "ZCK7", "ZCN7"]          # n + 2 spare, the expired September gone
    quotes = {"ZCZ6": _quote("ZCZ6", 454.25, date(2026, 12, 14)), "ZCK7": _quote("ZCK7", 480.0, date(2027, 5, 14)),
              "ZCN7": _quote("ZCN7", 485.5, date(2027, 7, 14))}
    curve = massive.build_curve(months, quotes, n=2)
    assert [(c.label, c.ticker, c.price) for c in curve] == [("Dec26", "ZCZ6", 454.25), ("May27", "ZCK7", 480.0)]
    assert massive.build_curve([("ZCZ6-ZCH7", date(2026, 12, 14))], {"ZCZ6-ZCH7": quotes["ZCZ6"]}) == []   # not a single month


# --------------------------------------------------------------------------- realized vol
def test_realized_vol_is_the_annualised_std_of_the_last_daily_changes():
    rng = np.random.default_rng(7)
    steps = rng.normal(0.0, 0.8, 120)
    s = pd.Series(np.cumsum(steps), index=[date(2026, 1, 1) + pd.Timedelta(days=i) for i in range(120)])
    last = s.diff().dropna().tail(60)
    assert massive.realized_vol(s, sessions=60) == pytest.approx(last.std(ddof=1) * np.sqrt(252))
    assert 8.0 < massive.realized_vol(s, sessions=60) < 17.0               # 0.8 a day is about 12.7 a year


def test_realized_vol_needs_enough_changes_and_ignores_gaps_and_order():
    short = pd.Series(np.arange(10.0), index=[date(2026, 1, 1) + pd.Timedelta(days=i) for i in range(10)])
    assert massive.realized_vol(short) is None and massive.realized_vol(pd.Series(dtype=float)) is None
    idx = [date(2026, 1, 1) + pd.Timedelta(days=i) for i in range(40)]
    vals = [float(i % 2) for i in range(40)]                              # +1, -1, +1, ... : every change has size 1
    ordered = pd.Series(vals, index=idx)
    scrambled = ordered.sample(frac=1.0, random_state=3)
    scrambled.iloc[5] = np.nan
    assert massive.realized_vol(ordered) == pytest.approx(np.sqrt(252), rel=0.03)
    assert massive.realized_vol(scrambled) is not None                    # sorted and NaN-dropped first, not rejected
