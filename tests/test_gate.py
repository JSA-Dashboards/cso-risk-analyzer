"""The shared-password gate: closed by default, opened by the password, pre-passed by the portal."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

from cso.auth import APP_PASSWORD

APP = str(ROOT / "app.py")


def _gate_inputs(at):
    return [ti for ti in at.text_input if ti.key == "cso_password_input"]


def _fresh(**state):
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def test_gate_hides_the_app_and_rejects_a_wrong_password():
    at = _fresh()
    assert not at.exception
    assert len(_gate_inputs(at)) == 1 and _gate_inputs(at)[0].label == "Password"
    assert not at.title and not at.metric and not at.tabs        # nothing of the app is rendered
    at.text_input[0].set_value("not-the-password")
    at.button[0].click().run()
    assert [e.value for e in at.error] == ["Incorrect password."]
    assert not at.title and not at.metric                          # still gated
    assert "_authed" not in at.session_state or not at.session_state["_authed"]


def test_right_password_opens_the_app():
    at = _fresh()
    at.text_input[0].set_value(APP_PASSWORD)
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert at.session_state["_authed"] is True
    assert [t.value for t in at.title] == ["CSO & spread risk analyzer"]
    assert len(at.metric) > 0
    # (not asserting the gate widgets are gone: after st.rerun() AppTest keeps the stale elements in its
    # tree although the browser replaces them; test_portal_pre_pass covers "no gate" on a clean run)


def test_portal_pre_pass_skips_the_gate():
    # the admin portal's sync inserts `st.session_state.setdefault("_authed", True)`
    at = _fresh(_authed=True)
    assert not at.exception
    assert not _gate_inputs(at)
    assert [t.value for t in at.title] == ["CSO & spread risk analyzer"]
