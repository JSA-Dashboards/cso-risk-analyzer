"""The Strategy builder tab: numbers on screen against the engine, no-book use, edits, carry and history panels."""
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cso import carry, massive, pricing, strategy

APP = str(ROOT / "app.py")
TODAY = date.today()


def usd(x):
    return f"${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}"


def _app(**state):
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["_authed"] = True
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def _metrics(at):
    return {m.label: m.value for m in at.metric}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _expected(level=-15.0, strike=-20.0, vol=20.0, days=36, rate=0.04, qty=10, mult=50.0):
    prem = float(pricing.price(False, level, strike, vol, days / 365, rate))
    legs = [strategy.Leg(strategy.FUTURE, qty, level), strategy.Leg(strategy.PUT, qty, strike, prem, TODAY + timedelta(days=days), vol)]
    return prem, strategy.summary(legs, mult)


def test_the_example_opens_on_a_put_over_a_long_spread_with_the_engines_numbers():
    at = _app()
    assert not at.exception
    m = _metrics(at)
    prem, sm = _expected()
    assert m["Premium paid"] == usd(sm["premium_paid"]) and sm["premium_paid"] == pytest.approx(prem * 500)
    assert m["Max loss"] == usd(sm["max_loss"])                       # the put floors the loss
    assert m["Max gain"] == "Unlimited"
    assert m["Break-even"] == f"{sm['breakevens'][0]:+.2f}"           # entry + premium, just above -15
    assert m["Spread level"] == "-15.00"


def test_editing_the_strike_moves_the_floor():
    at = _app()
    box = next(n for n in at.number_input if n.key == "cso_st_strike_Put_Dec/Mar")
    box.set_value(-25.0).run()
    assert not at.exception
    prem, sm = _expected(strike=-25.0)
    assert _metrics(at)["Max loss"] == usd(sm["max_loss"])
    assert sm["max_loss"] < _expected()[1]["max_loss"]                # a lower strike protects later: bigger floor loss


def test_a_short_spread_with_a_bought_call_is_the_mirror_image():
    at = _app(cso_st_base="Short spread", cso_st_opt="Call")
    assert not at.exception
    m = _metrics(at)
    level, strike = -15.0, -10.0                                      # default call strike: next 5 above level + 5
    prem = float(pricing.price(True, level, strike, 20.0, 36 / 365, 0.04))
    legs = [strategy.Leg(strategy.FUTURE, -10, level), strategy.Leg(strategy.CALL, 10, strike, prem, TODAY + timedelta(days=36), 20.0)]
    sm = strategy.summary(legs, 50.0)
    assert m["Max loss"] == usd(sm["max_loss"]) and m["Break-even"] == f"{sm['breakevens'][0]:+.2f}"


def test_it_works_with_no_positions_and_the_portfolio_tabs_say_so():
    spreads = pd.DataFrame([{"Spread": "Dec/Mar", "Near": "Dec26", "Far": "Mar27", "Near price": 430.0, "Far price": 445.0,
                             "Spread vol": 20.0}])
    blank = pd.DataFrame({"Spread": pd.Series(dtype=object), "Role": pd.Series(dtype=object), "Type": pd.Series(dtype=object),
                          "Qty": pd.Series(dtype=float), "Strike": pd.Series(dtype=float), "Expiry": pd.Series(dtype="datetime64[ns]"),
                          "Vol override": pd.Series(dtype=float), "Entry": pd.Series(dtype=float)})
    at = _app(cso_reset_n=0, cso_spreads_ver=0, cso_is_example=False, cso_example_touched=False,
              cso_spreads_df=spreads, cso_pos_df=blank, cso_quotes_df=pd.DataFrame())
    assert not at.exception
    assert "Premium paid" in _metrics(at)                              # the Strategy builder ran
    assert any("Strategy builder works without one" in str(i.value) for i in at.info)
    assert not at.sidebar.button == []                                 # the page did not stop early


def test_carry_and_the_traded_range_come_from_the_feed(monkeypatch):
    monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
    now = datetime(2026, 10, 26, 18, 0, tzinfo=timezone.utc)

    def snap(key, tickers, *args, **kwargs):
        book = {"ZCZ6": massive.Quote("ZCZ6", 454.25, "last", now, date(2026, 12, 14), True),
                "ZCH7": massive.Quote("ZCH7", 468.5, "last", now, date(2027, 3, 12), True),
                "ZQV6": massive.Quote("ZQV6", 96.115, "last", now, date(2026, 11, 2), True)}
        return {t: book[t] for t in tickers if t in book}

    days = [TODAY - timedelta(days=d) for d in (200, 100, 10)]
    near = pd.Series({d: p for d, p in zip(days, (440.0, 450.0, 454.0))})
    far = pd.Series({d: p for d, p in zip(days, (460.0, 462.0, 468.0))})
    monkeypatch.setattr(massive, "fetch_snapshot", snap)
    monkeypatch.setattr(massive, "fetch_histories", lambda key, tickers, **kw: {"ZCZ6": near, "ZCH7": far})
    at = _app()
    assert not at.exception
    m = _metrics(at)
    storage = carry.storage_rate("ZC", date(2026, 12, 14), date(2027, 3, 12))
    rate = (round(100 - 96.115, 3) + carry.FED_FUNDS_SPREAD_PCT) / 100
    fc = carry.full_carry(430.0, 88, storage, rate, 100)             # the table's near price (430) is what the spread uses
    assert m["Full carry"] == f"{fc['full']:.2f}"
    text = " ".join(str(x.value) for x in at.markdown) + " " + " ".join(str(c.value) for c in at.caption)
    pct = carry.pct_of_full_carry(-15.0, fc["full"])
    assert f"{pct:.0%} of full carry" in text and f":{carry.BUCKET_COLORS[carry.bucket(pct)]}-badge[" in text
    # the three sessions are -20 (440-460), -12 (450-462) and -14 (454-468): low -20, high -12
    assert "Last 12 months (3 sessions): low -20.00" in text and "high -12.00" in text


def test_without_a_key_the_carry_panel_still_works_on_the_typed_prices():
    at = _app()
    fc = carry.full_carry(430.0, 90, carry.MARKETS["ZC"][0], carry.FALLBACK_ANNUAL_RATE_PCT / 100, 100)
    assert _metrics(at)["Full carry"] == f"{fc['full']:.2f}"
