"""A second option leg: put spreads, collars and call spreads, built with the one-click buttons or by hand."""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cso import pricing, strategy
from cso.strategy import CALL, FUTURE, PUT, Leg

APP = str(ROOT / "app.py")
TODAY = date.today()
EXPIRY = TODAY + timedelta(days=36)
MULT = 50.0
SID = "Dec/Mar"                                   # the example book's spread


def usd(x):
    return f"${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}"


def _app(**state):
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["_authed"] = True
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _metrics(at):
    return {m.label: m.value for m in at.metric}


def _seg(at, key):
    return next(b for b in at.get("button_group") if b.key == key)


def _num(at, key):
    return next(n for n in at.number_input if n.key == key)


def _text(at):
    return " ".join(str(c.value) for c in at.caption) + " " + " ".join(str(m.value) for m in at.markdown)


def _legs(held, options, level=-15.0, vol=20.0):
    """The example's strategy built by hand: a spread future (held = +1 long / -1 short / 0 none) and the options
    as (kind, +1 buy / -1 sell, strike, vol offset), each priced with the model like the app does."""
    legs = [Leg(FUTURE, held * 10, level)] if held else []
    for kind, sign, k, off in options:
        prem = float(pricing.price(kind == CALL, level, k, vol + off, 36 / 365, 0.04))
        legs.append(Leg(kind, sign * 10, k, prem, EXPIRY, vol + off))
    return legs


def _specs(at):
    return [json.loads(el.proto.spec) for el in at.get("vega_lite_chart")]


def _labels(el):
    """Every text label a chart draws: the rows of a vega chart are Arrow bytes in the proto, not in the spec."""
    import pyarrow as pa
    frames = [pa.ipc.open_stream(ds.data.data).read_all().to_pandas() for ds in el.proto.datasets]
    return [v for df in frames if "label" in df for v in df["label"]]


# --------------------------------------------------------------------------- off by default, one click to build
def test_the_second_leg_is_off_until_asked_for():
    at = _app()
    assert not at.exception
    assert next(t for t in at.toggle if t.key == "cso_st_leg2").value is False
    assert not any(b.key == "cso_st_side2" for b in at.get("button_group"))
    m = _metrics(at)
    assert "Premium paid" in m and "Net premium paid" not in m
    assert [b.label for b in at.button if str(b.key).startswith("cso_qs_")] == ["Put spread", "Collar", "Call spread"]


def test_the_put_spread_button_buys_the_put_and_sells_one_ten_lower():
    at = _app()
    at.button(key="cso_qs_put").click().run()
    assert not at.exception
    assert next(t for t in at.toggle if t.key == "cso_st_leg2").value is True
    assert (_seg(at, "cso_st_side").value, _seg(at, "cso_st_opt").value) == ("Buy", "Put")
    assert (_seg(at, "cso_st_side2").value, _seg(at, "cso_st_opt2").value) == ("Sell", "Put")
    assert _num(at, f"cso_st_strike_Put_{SID}").value == -20.0 and _num(at, f"cso_st_strike2_Put_{SID}").value == -30.0
    sm = strategy.summary(_legs(1, [(PUT, 1, -20.0, 0), (PUT, -1, -30.0, 0)]), MULT)
    m = _metrics(at)
    assert m["Net premium paid"] == usd(sm["premium_paid"]) and "Premium paid" not in m
    assert m["Max loss"] == "Unlimited"                                       # below -30 the sold put stops protecting
    assert m["Break-even"] == ", ".join(f"{b:+.2f}" for b in sm["breakevens"])
    text = _text(at)
    assert "Put spread" in text and "buy 10 -20 put, sell 10 -30 put" in text


def test_the_collar_button_buys_a_put_and_sells_a_call_over_a_long_spread():
    at = _app()
    at.button(key="cso_qs_collar").click().run()
    assert not at.exception
    assert (_seg(at, "cso_st_side").value, _seg(at, "cso_st_opt").value) == ("Buy", "Put")
    assert (_seg(at, "cso_st_side2").value, _seg(at, "cso_st_opt2").value) == ("Sell", "Call")
    assert _num(at, f"cso_st_strike_Put_{SID}").value == -20.0 and _num(at, f"cso_st_strike2_Call_{SID}").value == -10.0
    sm = strategy.summary(_legs(1, [(PUT, 1, -20.0, 0), (CALL, -1, -10.0, 0)]), MULT)
    m = _metrics(at)
    assert m["Max loss"] == usd(sm["max_loss"]) and m["Max gain"] == usd(sm["max_gain"])      # bounded both ways
    assert m["Break-even"] == ", ".join(f"{b:+.2f}" for b in sm["breakevens"])
    paid = 0.0 if abs(sm["premium_paid"]) < 0.005 else sm["premium_paid"]                    # these two nearly cancel
    assert m["Net premium paid" if paid >= 0 else "Net premium received"] == usd(abs(paid))
    assert "Collar" in _text(at) and "buy 10 -20 put, sell 10 -10 call" in _text(at)


def test_a_collar_over_a_short_spread_buys_the_call_and_sells_the_put():
    at = _app(cso_st_base="Short spread")
    at.button(key="cso_qs_collar").click().run()
    assert not at.exception
    assert (_seg(at, "cso_st_side").value, _seg(at, "cso_st_opt").value) == ("Buy", "Call")
    assert (_seg(at, "cso_st_side2").value, _seg(at, "cso_st_opt2").value) == ("Sell", "Put")
    sm = strategy.summary(_legs(-1, [(CALL, 1, -10.0, 0), (PUT, -1, -20.0, 0)]), MULT)
    m = _metrics(at)
    assert m["Max loss"] == usd(sm["max_loss"]) and m["Max gain"] == usd(sm["max_gain"])
    assert "Collar" in _text(at)


def test_the_call_spread_button_buys_the_call_and_sells_one_ten_higher():
    at = _app()
    at.button(key="cso_qs_call").click().run()
    assert not at.exception
    assert _num(at, f"cso_st_strike_Call_{SID}").value == -10.0 and _num(at, f"cso_st_strike2_Call_{SID}").value == 0.0
    assert "Call spread" in _text(at) and "buy 10 -10 call, sell 10 +0 call" in _text(at)


# --------------------------------------------------------------------------- by hand
def test_turning_the_leg_on_by_hand_starts_as_a_put_spread_and_a_call_makes_it_a_collar():
    spread = _app(cso_st_leg2=True)
    assert not spread.exception
    assert (_seg(spread, "cso_st_side2").value, _seg(spread, "cso_st_opt2").value) == ("Sell", "Put")
    assert _num(spread, f"cso_st_strike2_Put_{SID}").value == -30.0                         # ten out of the money from the first strike
    assert "Put spread" in _text(spread)
    collar = _app(cso_st_leg2=True, cso_st_opt2="Call")
    assert not collar.exception
    assert _num(collar, f"cso_st_strike2_Call_{SID}").value == -10.0                        # a call goes out of the money on its own side
    assert "Collar" in _text(collar)


def test_a_credit_spread_takes_in_more_than_it_pays():
    at = _app(cso_st_leg2=True, cso_st_side="Sell")                                         # sell the -20 put, the second leg buys the -30
    assert not at.exception
    assert _seg(at, "cso_st_side2").value == "Buy"
    sm = strategy.summary(_legs(1, [(PUT, -1, -20.0, 0), (PUT, 1, -30.0, 0)]), MULT)
    m = _metrics(at)
    assert sm["premium_paid"] < 0
    assert m["Net premium received"] == usd(-sm["premium_paid"])
    assert "Put spread (credit)" in _text(at)


def test_the_second_leg_can_be_priced_at_its_own_vol():
    flat = _app(cso_st_leg2=True)
    skewed = _app(cso_st_leg2=True, cso_st_vol2off=2.0)
    assert not skewed.exception
    sm = strategy.summary(_legs(1, [(PUT, 1, -20.0, 0), (PUT, -1, -30.0, 2.0)]), MULT)
    assert _metrics(skewed)["Net premium paid"] == usd(sm["premium_paid"])
    assert _metrics(skewed)["Net premium paid"] != _metrics(flat)["Net premium paid"]


def test_my_quote_asks_for_a_premium_on_each_leg():
    at = _app(cso_st_leg2=True, cso_st_psrc="My quote")
    assert not at.exception
    first = next(n for n in at.number_input if str(n.key).startswith("cso_st_prem_"))
    second = next(n for n in at.number_input if str(n.key).startswith("cso_st_prem2_"))
    assert first.label == "Premium paid (points)" and second.label == "Premium received (points)"     # buy the -20, sell the -30
    prem = float(pricing.price(False, -15.0, -20.0, 20.0, 36 / 365, 0.04))
    assert first.value == pytest.approx(round(prem * 8) / 8)                                          # the model price to the tick


def test_the_second_legs_contracts_follow_the_position_size():
    at = _app(cso_st_leg2=True)
    assert _num(at, "cso_st_optqty2_10").value == 10
    at.number_input(key="cso_st_qty").set_value(20).run()
    assert not at.exception
    assert _num(at, "cso_st_optqty2_20").value == 20


# --------------------------------------------------------------------------- the chart and the table
def test_both_strikes_are_drawn_and_the_table_says_options():
    at = _app(cso_st_leg2=True)
    assert not at.exception
    charts = at.get("vega_lite_chart")
    payoff = next(el for el in charts if any((layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"]) == "area"
                                            for layer in json.loads(el.proto.spec).get("layer", [])))   # not the IV smile
    labels = _labels(payoff)
    assert "strike -20" in labels and "strike -30" in labels
    table = next(df.value for df in at.dataframe if "Combined at expiry" in df.value.columns)
    assert "Options" in table.columns and "Option" not in table.columns
    legs = _legs(1, [(PUT, 1, -20.0, 0), (PUT, -1, -30.0, 0)])
    for _, row in table.iterrows():
        assert row["Combined at expiry"] == pytest.approx(float(strategy.at_expiry(legs, row["Spread level"], MULT)))


def test_a_single_option_still_reads_as_before():
    at = _app()
    table = next(df.value for df in at.dataframe if "Combined at expiry" in df.value.columns)
    assert "Option" in table.columns and "Options" not in table.columns
    assert "A sold option keeps its premium" not in _text(at)


def test_a_sold_leg_brings_up_the_warning_and_a_pure_buy_does_not():
    assert "A sold option keeps its premium" in _text(_app(cso_st_leg2=True))              # the spread sells a put
    assert "A sold option keeps its premium" not in _text(_app(cso_st_leg2=True, cso_st_side2="Buy"))


# --------------------------------------------------------------------------- reruns after the buttons exist
def test_the_page_survives_several_reruns_once_the_buttons_have_been_drawn():
    # the loop at the top of the tabs re-assigns every cso_st_* value, and Streamlit refuses that for a button's key
    at = _app()
    at.button(key="cso_qs_collar").click().run()
    at.number_input(key="cso_st_qty").set_value(20).run()
    at.toggle(key="cso_st_leg2").set_value(False).run()
    at.toggle(key="cso_st_leg2").set_value(True).run()
    assert not at.exception
    assert (_seg(at, "cso_st_side2").value, _seg(at, "cso_st_opt2").value) == ("Sell", "Call")     # the legs were kept while it was off
