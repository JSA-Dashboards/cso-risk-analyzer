"""A push must take effect without a Reboot: the app reloads its own modules when they are stale."""
import sys
import types
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cso.massive  # noqa: E402,F401  (make sure the real module is loaded before we swap it)

APP = str(ROOT / "app.py")


@pytest.fixture
def restore_modules():
    saved = {n: m for n, m in sys.modules.items() if n == "cso" or n.startswith("cso.")}
    sig = getattr(sys, "_cso_code_sig", None)
    yield
    for n in [n for n in sys.modules if n == "cso" or n.startswith("cso.")]:
        del sys.modules[n]
    sys.modules.update(saved)                      # the other tests hold references to the original module objects
    sys._cso_code_sig = sig


def _run():
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["_authed"] = True
    return at.run()


def test_a_stale_module_missing_a_new_function_is_reloaded(restore_modules):
    real = sys.modules["cso.massive"]
    stale = types.ModuleType("cso.massive")        # what a Cloud process holds after a push: the OLD module,
    stale.__file__ = real.__file__                 # without functions the new file defines
    stale.MassiveError = real.MassiveError
    sys.modules["cso.massive"] = stale
    sys._cso_code_sig = None                       # first run of the new script in this process
    at = _run()
    assert not at.exception
    fresh = sys.modules["cso.massive"]
    assert fresh is not stale and hasattr(fresh, "fetch_histories") and hasattr(fresh, "fed_funds_pct")


def test_a_changed_file_since_the_last_run_reloads_the_package(restore_modules):
    _run()                                         # records the current signature
    marker = sys.modules["cso.massive"]
    sys._cso_code_sig = (("massive.py", 1),)       # pretend the files were different when this process last looked
    at = _run()
    assert not at.exception
    assert sys.modules["cso.massive"] is not marker


def test_a_fresh_process_is_left_alone(restore_modules):
    marker = sys.modules["cso.massive"]
    sys._cso_code_sig = None
    at = _run()
    assert not at.exception
    assert sys.modules["cso.massive"] is marker    # nothing stale, so nothing is reloaded (and tests keep their references)
    _run()
    assert sys.modules["cso.massive"] is marker
