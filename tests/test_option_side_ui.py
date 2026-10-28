"""Marking each leg long or short: the option can be bought or sold, and the spread position names its long and short months."""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cso import pricing, strategy

APP = str(ROOT / "app.py")
TODAY = date.today()
MULT = 50.0


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


def _group(at, label):
    return next(b for b in at.get("button_group") if b.label == label)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _legs(kind, sign, strike, level=-15.0, vol=20.0, qty=10, base=1):
    """The example's strategy built by hand: a spread future at the entry level, plus the bought or sold option."""
    prem = float(pricing.price(kind == strategy.CALL, level, strike, vol, 36 / 365, 0.04))
    legs = [strategy.Leg(strategy.FUTURE, base * qty, level)] if base else []
    return prem, legs + [strategy.Leg(kind, sign * qty, strike, prem, TODAY + timedelta(days=36), vol)]


# --------------------------------------------------------------------------- the option leg: long (buy) or short (sell)
def test_the_option_is_bought_unless_you_say_otherwise():
    at = _app()
    assert not at.exception
    assert _group(at, "Option").value == "Buy" and _group(at, "Type").value == "Put"
    m = _metrics(at)
    assert "Premium paid" in m and "Premium received" not in m


def test_selling_a_call_against_a_long_spread_keeps_the_premium_and_caps_the_gain():
    at = _app(cso_st_side="Sell", cso_st_opt="Call")                 # the default call strike is the next 5 above level + 5: -10
    assert not at.exception
    prem, legs = _legs(strategy.CALL, -1, -10.0)
    sm = strategy.summary(legs, MULT)
    m = _metrics(at)
    assert "Premium paid" not in m
    assert m["Premium received"] == usd(-sm["premium_paid"]) and sm["premium_paid"] == pytest.approx(-prem * 500)
    assert m["Max loss"] == "Unlimited"                               # the long spread can still fall without limit
    assert m["Max gain"] == usd(sm["max_gain"]) and sm["max_gain"] == pytest.approx(500 * (5.0 + prem))
    assert m["Break-even"] == f"{sm['breakevens'][0]:+.2f}" and sm["breakevens"][0] == pytest.approx(-15.0 - prem)


def test_selling_a_put_over_a_long_spread_doubles_the_downside_instead_of_protecting_it():
    at = _app(cso_st_side="Sell")                                     # the default put strike is -20
    assert not at.exception
    prem, legs = _legs(strategy.PUT, -1, -20.0)
    sm = strategy.summary(legs, MULT)
    m = _metrics(at)
    assert m["Premium received"] == usd(-sm["premium_paid"])
    assert m["Max loss"] == "Unlimited" and m["Max gain"] == "Unlimited"
    assert m["Break-even"] == f"{sm['breakevens'][0]:+.2f}"
    below = float(strategy.at_expiry(legs, -40.0, MULT))
    bought = float(strategy.at_expiry(_legs(strategy.PUT, 1, -20.0)[1], -40.0, MULT))
    assert below < bought                                             # selling the put, not buying it, is what hurts here


def test_a_sold_option_with_no_spread_position_is_a_naked_short():
    at = _app(cso_st_side="Sell", cso_st_base="No position")
    assert not at.exception
    prem, legs = _legs(strategy.PUT, -1, -20.0, base=0)
    sm = strategy.summary(legs, MULT)
    m = _metrics(at)
    assert m["Premium received"] == usd(-sm["premium_paid"]) and m["Max gain"] == usd(sm["max_gain"]) and m["Max loss"] == "Unlimited"


def test_the_premium_wording_follows_the_side_and_a_sold_option_carries_a_warning():
    bought = _app(cso_st_psrc="My quote")
    assert any(n.label == "Premium paid (points)" for n in bought.number_input)
    sold = _app(cso_st_psrc="My quote", cso_st_side="Sell")
    assert any(n.label == "Premium received (points)" for n in sold.number_input)
    assert not any("A sold option keeps its premium" in str(c.value) for c in bought.caption)
    assert any("A sold option keeps its premium" in str(c.value) for c in sold.caption)
    model = _app(cso_st_side="Sell")
    assert any("received for 10 contract(s)" in str(c.value) for c in model.caption)


def test_the_green_and_red_table_and_chart_follow_a_sold_option_too():
    at = _app(cso_st_side="Sell", cso_st_opt="Call")
    assert not at.exception
    prem, legs = _legs(strategy.CALL, -1, -10.0)
    table = next(df.value for df in at.dataframe if "Combined at expiry" in df.value.columns)
    for _, row in table.iterrows():
        assert row["Combined at expiry"] == pytest.approx(float(strategy.at_expiry(legs, row["Spread level"], MULT)))
    assert table["Combined at expiry"].max() == pytest.approx(500 * (5.0 + prem))      # the cap the sold call puts on the gain


# --------------------------------------------------------------------------- the spread position: which month is long
def test_the_position_control_names_the_long_and_the_short_month():
    at = _app()
    pos = _group(at, "Spread position")
    assert list(pos.options) == ["Long Dec26 / short Mar27", "Long Mar27 / short Dec26", "No position"]
    assert pos.value == "Long spread"


def test_the_short_spread_is_long_the_far_month():
    at = _app(cso_st_base="Short spread", cso_st_opt="Call")
    assert not at.exception
    assert _group(at, "Spread position").value == "Short spread"
    # a short spread with a bought call is the existing mirror case: the engine's numbers, untouched
    prem, _ = _legs(strategy.CALL, 1, -10.0)
    legs = [strategy.Leg(strategy.FUTURE, -10, -15.0), strategy.Leg(strategy.CALL, 10, -10.0, prem, TODAY + timedelta(days=36), 20.0)]
    assert _metrics(at)["Max loss"] == usd(strategy.summary(legs, MULT)["max_loss"])


def test_a_far_minus_near_spread_names_the_months_the_other_way_round():
    far_near = pd.DataFrame([{"Spread": "Dec/Mar", "Near": "Dec26", "Far": "Mar27", "Near price": 430.0, "Far price": 445.0,
                              "Spread vol": 20.0, "Convention": "Far − near"}])
    blank = pd.DataFrame({"Spread": pd.Series(dtype=object), "Role": pd.Series(dtype=object), "Type": pd.Series(dtype=object),
                          "Qty": pd.Series(dtype=float), "Strike": pd.Series(dtype=float),
                          "Expiry": pd.Series(dtype="datetime64[ns]"), "Vol override": pd.Series(dtype=float),
                          "Entry": pd.Series(dtype=float)})
    at = _app(cso_reset_n=0, cso_spreads_ver=0, cso_is_example=False, cso_example_touched=False, cso_spreads_df=far_near,
              cso_pos_df=blank, cso_quotes_df=pd.DataFrame())
    assert not at.exception
    # far - near: a long spread is long the FAR month (the spread rises when the far month gains on the near one)
    assert list(_group(at, "Spread position").options) == ["Long Mar27 / short Dec26", "Long Dec26 / short Mar27", "No position"]
    assert _metrics(at)["Spread level"] == "+15.00"
