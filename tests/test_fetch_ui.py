"""The "Fetch futures prices" button, with the Massive call mocked: no network and no real key."""
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cso import massive

APP = str(ROOT / "app.py")
NOW = datetime(2026, 10, 15, 19, 19, tzinfo=timezone.utc)


def _q(ticker, price, source="last"):
    return massive.Quote(ticker, price, source, NOW if source == "last" else None, date(2026, 12, 14), True)


def _app(**state):
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["_authed"] = True            # past the password gate (covered in test_gate.py)
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def _button(at):
    return next(b for b in at.button if b.key == "cso_fetch_prices")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _with_feed(monkeypatch, quotes=None, error=None):
    monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
    calls = []

    def fake(key, tickers, **kwargs):
        calls.append((key, tuple(tickers)))
        if error:
            raise massive.MassiveError(error)
        return {t: quotes[t] for t in tickers if t in quotes}

    monkeypatch.setattr(massive, "fetch_snapshot", fake)
    return calls


def test_the_button_is_off_without_a_key_and_says_why():
    at = _app()
    assert not at.exception and _button(at).disabled
    assert any("MASSIVE_API_KEY" in str(c.value) for c in at.caption)


def test_fetch_fills_near_and_far_from_the_feed(monkeypatch):
    calls = _with_feed(monkeypatch, {"ZCZ6": _q("ZCZ6", 478.25), "ZCH7": _q("ZCH7", 491.0)})
    at = _app()
    assert not _button(at).disabled
    _button(at).click().run()
    assert not at.exception
    df = at.session_state["cso_spreads_df"]
    assert (df.loc[0, "Near price"], df.loc[0, "Far price"]) == (478.25, 491.0)
    assert calls == [("test-key", ("ZCH7", "ZCZ6"))]                      # Dec26/Mar27 -> ZCZ6/ZCH7, key passed through
    kind, text = at.session_state["cso_fetch_note"]
    assert kind == "ok" and "ZCZ6 478.25" in text and "ZCH7 491" in text and "delayed" in text
    assert "test-key" not in text
    assert not any("Example data." in str(w.value) for w in at.warning)   # a fetch counts as editing the example
    assert at.session_state["cso_spreads_ver"] == 1                       # the table was rebuilt with the new prices


def test_a_contract_with_no_quote_is_reported_and_left_alone(monkeypatch):
    _with_feed(monkeypatch, {"ZCZ6": _q("ZCZ6", 478.25)})
    at = _app()
    _button(at).click().run()
    df = at.session_state["cso_spreads_df"]
    assert (df.loc[0, "Near price"], df.loc[0, "Far price"]) == (478.25, 445.0)
    kind, text = at.session_state["cso_fetch_note"]
    assert kind == "warn" and "No quote for ZCH7" in text


def test_a_service_error_is_shown_and_nothing_changes(monkeypatch):
    _with_feed(monkeypatch, error="the market-data service returned a non-JSON response (HTTP 502)")
    at = _app()
    _button(at).click().run()
    assert not at.exception
    kind, text = at.session_state["cso_fetch_note"]
    assert kind == "warn" and "Could not fetch prices" in text and "HTTP 502" in text
    df = at.session_state["cso_spreads_df"]
    assert (df.loc[0, "Near price"], df.loc[0, "Far price"]) == (430.0, 445.0)


def test_unreadable_contract_labels_are_explained_without_a_call(monkeypatch):
    calls = _with_feed(monkeypatch, {})
    at = _app()
    df = at.session_state["cso_spreads_df"].copy()
    df["Near"], df["Far"] = "front", "back"
    at.session_state["cso_spreads_df"] = df
    at.run()
    _button(at).click().run()
    kind, text = at.session_state["cso_fetch_note"]
    assert kind == "warn" and "contract labels" in text and calls == []


def test_a_typed_spread_level_is_flagged_because_it_wins(monkeypatch):
    _with_feed(monkeypatch, {"ZCZ6": _q("ZCZ6", 478.25), "ZCH7": _q("ZCH7", 491.0)})
    at = _app()
    df = at.session_state["cso_spreads_df"].copy()
    df["Spread level"] = -10.0
    at.session_state["cso_spreads_df"] = df
    at.run()
    _button(at).click().run()
    kind, text = at.session_state["cso_fetch_note"]
    assert kind == "warn" and "Spread level is typed in" in text


def test_clearing_the_example_forgets_the_fetch_note(monkeypatch):
    _with_feed(monkeypatch, {"ZCZ6": _q("ZCZ6", 478.25), "ZCH7": _q("ZCH7", 491.0)})
    at = _app()
    _button(at).click().run()
    assert at.session_state["cso_fetch_note"] is not None
    next(b for b in at.sidebar.button if b.key == "cso_clear_example").click().run()
    assert at.session_state["cso_fetch_note"] is None
