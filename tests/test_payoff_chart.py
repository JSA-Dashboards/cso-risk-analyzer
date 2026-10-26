"""The payoff chart builds a valid spec: green/red fills, the three series, and the reference rules."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cso import payoff_chart, strategy
from cso.strategy import FUTURE, PUT, Leg

MULT = 50.0


def _spec(**kw):
    legs = [Leg(FUTURE, 10, -15.0), Leg(PUT, 10, -20.0, 0.75)]
    S = strategy.grid(legs, MULT, -50.0, 20.0)
    chart = payoff_chart.payoff_chart(S, strategy.at_expiry(legs, S, MULT), strategy.at_expiry(legs[:1], S, MULT),
                                      strategy.at_expiry(legs, S, MULT), current=-15.0, strikes=[(-20.0, "strike -20")],
                                      breakevens=strategy.breakevens(legs, MULT), **kw)
    return chart.to_dict()


def _marks(spec):
    return [layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"] for layer in spec["layer"]]


def test_fills_are_green_for_gains_and_red_for_losses_and_never_cross_zero():
    spec = _spec()
    areas = [layer for layer in spec["layer"] if (layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"]) == "area"]
    assert [a["mark"]["color"] for a in areas] == [payoff_chart.GREEN, payoff_chart.RED]
    datasets = spec["datasets"]
    fills = next(rows for rows in datasets.values() if rows and "gain" in rows[0])
    assert all(r["gain"] >= 0 and r["loss"] <= 0 for r in fills)
    assert any(r["gain"] > 0 for r in fills) and any(r["loss"] < 0 for r in fills)        # both colours appear
    # gain and loss never overlap at the same x, and the zero crossing is an exact grid point (both are 0 there)
    assert all(r["gain"] == 0 or r["loss"] == 0 for r in fills)
    assert any(abs(r["Spread"] - (-14.25)) < 1e-9 and r["gain"] == 0 and r["loss"] == 0 for r in fills)


def test_three_series_and_the_reference_rules_are_drawn():
    spec = _spec()
    lines = next(rows for rows in spec["datasets"].values() if rows and "Series" in rows[0])
    assert {r["Series"] for r in lines} == set(payoff_chart.SERIES)
    marks = _marks(spec)
    assert marks.count("rule") >= 4 and "text" in marks and "line" in marks               # zero, now, strike, break-even


def test_optional_layers_come_and_go():
    plain = len(_spec()["layer"])
    assert len(_spec(band=(-30.0, -5.0))["layer"]) == plain + 1                            # the traded-range band
    assert len(_spec(carry_limit=-30.0)["layer"]) == plain + 2                             # rule + label
    legs = [Leg(PUT, 1, -20.0, 0.75)]
    S = np.linspace(-40, 0, 5)
    no_today = payoff_chart.payoff_chart(S, strategy.at_expiry(legs, S, MULT), strategy.at_expiry([], S, MULT),
                                         current=-15.0).to_dict()
    lines = next(rows for rows in no_today["datasets"].values() if rows and "Series" in rows[0])
    assert {r["Series"] for r in lines} == {"Combined at expiry", "Position only"}
