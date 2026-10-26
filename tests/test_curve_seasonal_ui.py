"""The leg picker and the seasonal overlay in the Strategy builder, against faked feed calls (no network).

Dates are built relative to today, so the tests keep working as real contracts expire.
"""
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cso import massive, pricing, seasonal, strategy

APP = str(ROOT / "app.py")
TODAY = date.today()
NOW = datetime.now(timezone.utc)
MULT = 50.0


def _curve(n=6):
    """The next ``n`` listed corn months after this one: ticker and (made-up) last trading day, nearest first."""
    out, d = [], TODAY.replace(day=1)
    while len(out) < n:
        d = (d + timedelta(days=32)).replace(day=1)
        letter = massive.MONTH_CODES[d.month - 1]
        if letter in "HKNUZ":
            out.append((f"ZC{letter}{d.year % 10}", date(d.year, d.month, 14)))
    return out


CURVE = _curve()
PRICE = {t: 450.0 + 5.0 * i for i, (t, _) in enumerate(CURVE)}          # a 5-point step: every neighbouring spread is -5
LABEL = {t: massive.contract_label(massive.split_ticker(t)[1], massive.split_ticker(t)[2], e) for t, e in CURVE}
NEAR, FAR = CURVE[0][0], CURVE[1][0]
EXPIRY = dict(CURVE)


def _quotes_for(prices, expiries):
    def snapshot(key, tickers, *args, **kwargs):
        out = {t: massive.Quote(t, prices[t], "last", NOW, expiries[t], True) for t in tickers if t in prices}
        out.update({t: massive.Quote(t, 96.1, "last", NOW, TODAY + timedelta(days=30), True) for t in tickers if t.startswith("ZQ")})
        return out
    return snapshot


def _walk(end, days, start, seed):
    rng = np.random.default_rng(seed)
    idx = [end - timedelta(days=days - i) for i in range(days + 1)]
    return pd.Series(start + np.cumsum(rng.normal(0, 0.7, len(idx))), index=idx)


def _histories_for(near_t, far_t, near_exp, live):
    """Live contracts run up to today. Each earlier year's near contract ends on the 12th of the same month that year, and
    its far contract a hundred days later, so every earlier pair has a full season to replay."""
    ends = {}
    for back, nt, ft in seasonal.prior_pairs(near_t, far_t, seasonal.MAX_PRIOR_YEARS):
        end = date(near_exp.year - back, near_exp.month, 12)
        ends[nt], ends[ft] = end, end + timedelta(days=100)

    def fetch(key, tickers, *args, **kwargs):
        return {t: _walk(TODAY, 300, live[t], i) if t in live else _walk(ends[t], 330, 440.0 + (i % 5) * 3, i + 50)
                for i, t in enumerate(tickers)}
    return fetch


HISTORIES = _histories_for(NEAR, FAR, EXPIRY[NEAR], PRICE)


@pytest.fixture(autouse=True)
def _feed(monkeypatch):
    monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
    monkeypatch.setattr(massive, "fetch_contracts", lambda *a, **k: list(CURVE))
    monkeypatch.setattr(massive, "fetch_snapshot", _quotes_for(PRICE, EXPIRY))
    monkeypatch.setattr(massive, "fetch_histories", HISTORIES)
    st.cache_data.clear()
    yield
    st.cache_data.clear()


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


def _specs(at):
    return [json.loads(el.proto.spec) for el in at.get("vega_lite_chart")]


def _marks(spec):
    return [layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"] for layer in spec.get("layer", [])]


def _payoff(at):
    return next(s for s in _specs(at) if "area" in _marks(s))


def _seasonal(at):
    def legend(s):
        return s["layer"][0].get("encoding", {}).get("color", {}).get("scale", {}).get("domain", []) if s.get("layer") else []
    return next((s for s in _specs(at) if "Current" in legend(s)), None)


def _replay_table(at):
    return next((df.value for df in at.dataframe if "If repeated" in df.value.columns), None)


def _far_key(near):
    return f"cso_st_far_ZC_{LABEL[CURVE[0][0]]}_{LABEL[near]}"


NEAR_KEY = f"cso_st_near_ZC_{LABEL[CURVE[0][0]]}"
LIVE = {"cso_st_src": "Live curve"}
EMPTY_BOOK = dict(
    cso_reset_n=0, cso_spreads_ver=0, cso_is_example=False, cso_example_touched=False, cso_pos_df=pd.DataFrame(),
    cso_quotes_df=pd.DataFrame(),
    cso_spreads_df=pd.DataFrame({c: pd.Series(dtype=t) for c, t in {"Spread": "object", "Near": "object", "Far": "object",
                                                                    "Near price": "float64", "Far price": "float64"}.items()}))
LIVE_SEASONAL = {**LIVE, "cso_st_seas": True}


# --------------------------------------------------------------------------- the leg picker
def test_the_book_stays_the_default_and_live_curve_is_one_click_away():
    at = _app()
    assert not at.exception
    src = _group(at, "Spread from")
    assert list(src.options) == ["My book", "Live curve"] and src.value == "My book"
    assert _metrics(at)["Spread level"] == "-15.00"                       # still the example's own typed prices


def test_the_picker_builds_the_spread_from_the_live_curve():
    at = _app(**LIVE)
    assert not at.exception
    near = next(s for s in at.selectbox if s.label == "Near contract")
    far = next(s for s in at.selectbox if s.label == "Far contract")
    assert near.value == LABEL[NEAR] and far.value == LABEL[FAR]          # the next two listed months
    assert all(LABEL[CURVE[-1][0]] not in o for o in near.options)         # the last month can only be a far leg
    assert near.options[0] == f"{LABEL[NEAR]} · {PRICE[NEAR]:g}"          # each option carries its live price
    assert _metrics(at)["Spread level"] == f"{PRICE[NEAR] - PRICE[FAR]:+.2f}"
    assert not [w for w in at.warning if "live curve" in str(w.value) or "contract months" in str(w.value)]


def test_a_different_far_contract_changes_the_spread():
    third = CURVE[2][0]
    at = _app(**LIVE, **{_far_key(NEAR): LABEL[third]})
    assert not at.exception
    assert _metrics(at)["Spread level"] == f"{PRICE[NEAR] - PRICE[third]:+.2f}"
    far = next(s for s in at.selectbox if s.label == "Far contract")
    assert not any(o.startswith(LABEL[NEAR]) for o in far.options)         # only later months are offered
    assert any(o.startswith(LABEL[CURVE[1][0]]) for o in far.options)


def test_moving_the_near_contract_resets_what_the_far_contract_can_be():
    near2 = CURVE[1][0]
    at = _app(**LIVE, **{NEAR_KEY: LABEL[near2]})
    assert not at.exception
    far = next(s for s in at.selectbox if s.label == "Far contract")
    assert far.value == LABEL[CURVE[2][0]]                                 # the month after the new near leg
    assert _metrics(at)["Spread level"] == f"{PRICE[near2] - PRICE[CURVE[2][0]]:+.2f}"


def test_the_vol_starts_at_the_realized_vol_of_the_pair_and_says_so():
    at = _app(**LIVE, **EMPTY_BOOK)               # no book spread for the pair, so nothing of the user's own to start from
    hist = HISTORIES("k", (NEAR, FAR))
    realized = massive.realized_vol(massive.spread_history(hist[NEAR], hist[FAR], None))
    assert realized is not None
    vol = next(n for n in at.number_input if n.label == "Spread vol (pts/yr)")
    assert vol.value == pytest.approx(round(realized, 1), abs=0.051)
    assert "realized vol over its last 60 sessions" in vol.help


def test_the_books_own_vol_wins_when_it_has_the_same_pair():
    pair = pd.DataFrame([{"Spread": "Mine", "Near": LABEL[NEAR], "Far": LABEL[FAR], "Near price": 1.0, "Far price": 2.0,
                          "Spread vol": 17.5}])
    at = _app(**LIVE, cso_reset_n=0, cso_spreads_ver=0, cso_is_example=False, cso_example_touched=False, cso_spreads_df=pair,
              cso_pos_df=pd.DataFrame(), cso_quotes_df=pd.DataFrame())
    assert not at.exception
    vol = next(n for n in at.number_input if n.label == "Spread vol (pts/yr)")
    assert vol.value == 17.5 and "your book's vol for this pair" in vol.help


def test_the_strategy_numbers_follow_the_picked_spread():
    at = _app(**LIVE)
    level = PRICE[NEAR] - PRICE[FAR]
    vol = next(n for n in at.number_input if n.label == "Spread vol (pts/yr)").value
    strike = next(n for n in at.number_input if n.label == "Strike").value
    prem = float(pricing.price(False, level, strike, vol, 36 / 365, 0.04))
    legs = [strategy.Leg(strategy.FUTURE, 10, level), strategy.Leg(strategy.PUT, 10, strike, prem, TODAY + timedelta(days=36), vol)]
    sm = strategy.summary(legs, MULT)
    m = _metrics(at)
    assert m["Max loss"] == f"-${abs(sm['max_loss']):,.0f}" and m["Break-even"] == f"{sm['breakevens'][0]:+.2f}"


def test_the_picker_works_with_an_empty_book_instead_of_stopping_the_page():
    at = _app(**EMPTY_BOOK)
    assert not at.exception
    assert any("can pick its spread from the live curve" in str(i.value) for i in at.info)
    assert _metrics(at)["Spread level"] == f"{PRICE[NEAR] - PRICE[FAR]:+.2f}"      # live curve is the only source
    assert "Spread from" not in [b.label for b in at.get("button_group")]            # so there is no switch


def test_a_failing_feed_says_so_instead_of_breaking_the_tab(monkeypatch):
    def down(*a, **k):
        raise massive.MassiveError("could not reach the market-data service (ConnectionError)")

    monkeypatch.setattr(massive, "fetch_contracts", down)
    at = _app(**LIVE)
    assert not at.exception
    assert any("Could not load the live curve" in str(w.value) for w in at.warning)


def test_without_a_key_there_is_no_live_curve_source(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    at = _app(**LIVE)
    assert not at.exception
    assert "Spread from" not in [b.label for b in at.get("button_group")]
    assert _metrics(at)["Spread level"] == "-15.00"
    seas = next(t for t in at.toggle if t.key == "cso_st_seas")
    assert seas.proto.disabled                                              # the overlay needs the feed


# --------------------------------------------------------------------------- the seasonal overlay
def test_the_overlay_is_off_until_asked_for():
    at = _app(**LIVE)
    assert not at.exception
    seas = next(t for t in at.toggle if t.key == "cso_st_seas")
    assert seas.value is False and _seasonal(at) is None
    assert "point" not in _marks(_payoff(at))                               # no dots without the replays
    assert not any(str(h.value) == "#### Seasonal overlay" for h in at.markdown)


def test_the_overlay_adds_the_chart_the_replay_table_and_the_dots():
    at = _app(**LIVE_SEASONAL)
    assert not at.exception
    assert "point" in _marks(_payoff(at))                                   # the replay dots
    chart = _seasonal(at)
    assert chart is not None
    domain = chart["layer"][0]["encoding"]["color"]["scale"]["domain"]
    years = [n for n in domain if n.isdigit()]
    assert domain[0] == "Current" and 1 <= len(years) <= seasonal.MAX_PRIOR_YEARS and years == sorted(years, reverse=True)
    text = " ".join(str(x.value) for x in at.markdown) + " " + " ".join(str(c.value) for c in at.caption)
    assert "If the spread repeated each earlier year's move" in text
    assert "small sample" in text and "not a forecast" in text


def test_each_replay_row_is_the_engine_applied_to_todays_level_plus_that_years_move():
    at = _app(**LIVE_SEASONAL)
    assert not at.exception
    table = _replay_table(at)
    assert table is not None and len(table) >= 1
    assert {"Earlier year", "Then", "At expiry", "Move", "If repeated", "Position only", "Strategy"} <= set(table.columns)
    level = PRICE[NEAR] - PRICE[FAR]
    vol = next(n for n in at.number_input if n.label == "Spread vol (pts/yr)").value
    strike = next(n for n in at.number_input if n.label == "Strike").value
    prem = float(pricing.price(False, level, strike, vol, 36 / 365, 0.04))
    held = [strategy.Leg(strategy.FUTURE, 10, level)]
    legs = held + [strategy.Leg(strategy.PUT, 10, strike, prem, TODAY + timedelta(days=36), vol)]
    for _, row in table.iterrows():
        assert row["If repeated"] == pytest.approx(level + row["Move"])
        assert row["Move"] == pytest.approx(row["At expiry"] - row["Then"])
        assert row["Strategy"] == pytest.approx(float(strategy.at_expiry(legs, row["If repeated"], MULT)))
        assert row["Position only"] == pytest.approx(float(strategy.at_expiry(held, row["If repeated"], MULT)))
        assert str(row["Earlier year"]).count("/") == 1                      # labelled as the pair, e.g. Dec25/Mar26


def test_fewer_earlier_years_means_fewer_replays():
    full = _app(**LIVE_SEASONAL)
    assert not full.exception
    years_key = next(n.key for n in full.number_input if str(n.key).startswith("cso_st_years_"))
    fewer = _app(**LIVE_SEASONAL, **{years_key: 1})
    assert not fewer.exception
    assert len(_replay_table(fewer)) == 1 and len(_replay_table(full)) > 1
    domain = _seasonal(fewer)["layer"][0]["encoding"]["color"]["scale"]["domain"]
    assert len([n for n in domain if n.isdigit()]) == 1


def test_an_option_expiring_after_the_near_contract_has_nothing_to_replay():
    at = _app(**LIVE_SEASONAL, cso_st_expiry=TODAY + timedelta(days=400))
    assert not at.exception
    assert any("nothing to replay" in str(i.value) for i in at.info)
    assert "point" not in _marks(_payoff(at))


def test_no_earlier_history_says_so(monkeypatch):
    monkeypatch.setattr(massive, "fetch_histories",
                        lambda key, tickers, *a, **k: {t: s for t, s in HISTORIES(key, tickers).items() if t in PRICE})
    at = _app(**LIVE_SEASONAL)
    assert not at.exception
    assert any("No earlier year of this pair has daily history" in str(i.value) for i in at.info)


def test_the_overlay_also_works_on_a_book_spread_with_typed_labels(monkeypatch):
    live = {"ZCZ6": 455.0, "ZCH7": 470.0}                                   # the example book's Dec26/Mar27 pair
    expiries = {"ZCZ6": date(2026, 12, 14), "ZCH7": date(2027, 3, 12)}
    monkeypatch.setattr(massive, "fetch_snapshot", _quotes_for(live, expiries))
    monkeypatch.setattr(massive, "fetch_histories", _histories_for("ZCZ6", "ZCH7", expiries["ZCZ6"], live))
    at = _app(cso_st_seas=True)
    assert not at.exception
    assert _seasonal(at) is not None
