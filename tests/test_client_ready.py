"""Client-facing behaviour: the disclosure, the labelled example data, clearing it, no silent vol."""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

from cso import disclosure

APP = str(ROOT / "app.py")


def _app(**state):
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["_authed"] = True            # past the password gate (covered in test_gate.py)
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def _sidebar_button(at, key):
    return next(b for b in at.sidebar.button if b.key == key)


def _all_text(at):
    return " ".join(str(c.value) for c in at.caption) + " " + " ".join(str(w.value) for w in at.warning)


def test_disclosure_is_on_the_page_and_paragraph_two_is_the_firms_standard_text():
    at = _app()
    assert not at.exception
    text = " ".join(str(c.value) for c in at.caption)
    assert disclosure.MODEL_PARAGRAPH in text and disclosure.STANDARD_PARAGRAPH in text
    assert "does not accept orders to buy or sell any financial instruments via email" in text
    assert "Bachelier" in text and "Black-76" not in text
    # paragraph 2 is byte-for-byte the Risk Analyzer's text when that repo sits alongside this one
    ra = ROOT.parent / "jsa-risk-analyzer" / "src" / "jsa_risk" / "ui" / "theme.py"
    if ra.exists():
        import html
        import re
        paras = [re.sub(r"\s+", " ", html.unescape(p)).strip()
                 for p in re.findall(r"<p>(.*?)</p>", ra.read_text(encoding="utf-8"), re.S)]
        assert disclosure.STANDARD_PARAGRAPH == paras[1]


def test_first_visit_shows_a_labelled_example():
    at = _app()
    assert any("Example data" in str(w.value) for w in at.warning)
    assert len(at.metric) > 0                      # the worked example computes


def test_clear_empties_the_inputs_and_drops_the_banner_then_reload_restores_it():
    at = _app()
    _sidebar_button(at, "cso_clear_example").click().run()
    assert not at.exception
    assert not any("Example data" in str(w.value) for w in at.warning)
    assert any("Add a spread to begin" in str(i.value) for i in at.info)
    assert not at.metric
    assert not at.session_state["cso_is_example"]
    # the disclosure stays on a blank page
    assert disclosure.STANDARD_PARAGRAPH in " ".join(str(c.value) for c in at.caption)
    _sidebar_button(at, "cso_reload_example").click().run()
    assert any("Example data" in str(w.value) for w in at.warning)
    assert len(at.metric) > 0


def test_a_spread_without_a_vol_is_flagged_not_silently_assumed():
    spreads = pd.DataFrame([{"Spread": "Dec/Mar", "Near": "Dec", "Far": "Mar", "Near price": 430.0,
                             "Far price": 445.0}])
    pos = pd.DataFrame([{"Spread": "Dec/Mar", "Role": "Hedge / trade", "Type": "Put", "Qty": 1.0, "Strike": -20.0,
                         "Expiry": date.today() + timedelta(days=30)}])
    at = _app(cso_reset_n=0, cso_spreads_ver=0, cso_is_example=False, cso_example_touched=False,
              cso_spreads_df=spreads, cso_pos_df=pos, cso_quotes_df=pd.DataFrame())
    assert not at.exception
    assert any("no spread vol entered" in str(w.value) for w in at.warning)
