"""The seasonal chart builds a valid spec: the legend order, the colours, the reference lines and what stays out of range."""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cso import payoff_chart, seasonal_chart

NEAR_EXPIRY = date(2026, 12, 14)


def _series(start_level, step, days=120):
    days_ = [NEAR_EXPIRY - timedelta(days=d) for d in range(days, -1, -2)]
    return pd.Series([start_level + step * i for i in range(len(days_))], index=days_)


def _chart(**kw):
    args = dict(current=_series(-12.0, -0.1)[: 30], priors={2025: _series(-9.0, -0.2), 2024: _series(-14.0, 0.1)},
                average=_series(-11.5, -0.05), near_expiry=NEAR_EXPIRY, window=183, today=date(2026, 10, 26),
                option_expiry=date(2026, 12, 1), strike=-20.0, breakevens=[-14.24], entry=-15.0, carry_limit=-29.8)
    args.update(kw)
    return seasonal_chart.seasonal_chart(args.pop("current"), args.pop("priors"), args.pop("average"), **args)


def _marks(spec):
    return [layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"] for layer in spec["layer"]]


def test_nothing_to_draw_is_none():
    assert seasonal_chart.seasonal_chart(None, {}, None, near_expiry=NEAR_EXPIRY, window=183, today=date(2026, 10, 26)) is None
    assert seasonal_chart.seasonal_chart(pd.Series(dtype=float), {2025: pd.Series(dtype=float)}, None,
                                         near_expiry=NEAR_EXPIRY, window=183, today=date(2026, 10, 26)) is None


def test_the_legend_reads_current_then_the_years_newest_first_then_the_average_and_the_live_line_is_drawn_last():
    spec = _chart().to_dict()
    line = spec["layer"][0]
    assert line["encoding"]["color"]["scale"]["domain"] == ["Current", "2025", "2024", "Average"]
    colors = line["encoding"]["color"]["scale"]["range"]
    assert colors[0] == payoff_chart.INK and colors[1] == seasonal_chart.PRIOR_COLORS[0]      # the most recent year is darkest
    rows = next(r for r in spec["datasets"].values() if r and "Series" in r[0])
    order = list(dict.fromkeys(r["Series"] for r in rows))                                     # first appearance = draw order
    assert order == ["2025", "2024", "Average", "Current"]


def test_the_strategys_levels_and_the_two_dates_are_drawn():
    spec = _chart().to_dict()
    texts = [r["label"] for rows in spec["datasets"].values() for r in rows if "label" in r]
    assert {"strike -20", "break-even -14.24", "entry -15.00", "full carry", "today", "option expiry"} <= set(texts)
    marks = _marks(spec)
    assert marks[0] == "line" and marks.count("rule") == 6 and marks.count("text") == 6


def test_the_y_axis_always_holds_the_strike_and_break_even_but_not_a_far_off_full_carry():
    near = _chart(strike=-60.0, carry_limit=-200.0).to_dict()["layer"][0]["encoding"]["y"]["scale"]["domain"]
    assert near[0] < -60.0 and near[1] > -5.0                                   # the strike stays on the chart, stretching it
    texts = [r["label"] for rows in _chart(carry_limit=-200.0).to_dict()["datasets"].values() for r in rows if "label" in r]
    assert "full carry" not in texts                                            # -200 would flatten everything: left off


def test_dates_outside_the_window_get_no_marker_and_optional_levels_can_be_left_out():
    spec = _chart(today=date(2027, 1, 5), option_expiry=date(2025, 1, 1), strike=None, breakevens=[], entry=None,
                  carry_limit=None).to_dict()
    texts = [r["label"] for rows in spec["datasets"].values() for r in rows if "label" in r]
    assert texts == []
    assert _marks(spec) == ["line"]


def test_the_x_axis_ends_on_the_near_contracts_last_day():
    domain = _chart().to_dict()["layer"][0]["encoding"]["x"]["scale"]["domain"]
    assert domain[1].startswith("2026-12-14") and domain[0].startswith("2026-06-14")


def test_the_payoff_chart_draws_a_dot_per_replay_green_or_red_by_result():
    from cso import strategy
    from cso.strategy import FUTURE, PUT, Leg
    legs = [Leg(FUTURE, 10, -15.0), Leg(PUT, 10, -20.0, 0.75)]
    S = strategy.grid(legs, 50.0, -50.0, 20.0)
    args = (S, strategy.at_expiry(legs, S, 50.0), strategy.at_expiry(legs[:1], S, 50.0))
    plain = payoff_chart.payoff_chart(*args, current=-15.0).to_dict()
    dotted = payoff_chart.payoff_chart(*args, current=-15.0, replays=[("'25", -9.0, 1900.0), ("'24", -22.0, -480.0)]).to_dict()
    assert len(dotted["layer"]) == len(plain["layer"]) + 2                      # the dots and their labels
    rows = next(r for r in dotted["datasets"].values() if r and "result" in r[0])
    assert [(r["year"], r["result"]) for r in rows] == [("'25", "gain"), ("'24", "loss")]
    dots = next(layer for layer in dotted["layer"] if layer["mark"]["type"] == "point")
    assert dots["encoding"]["fill"]["scale"]["range"] == [payoff_chart.GREEN, payoff_chart.RED]
    assert "color" not in dots["encoding"]                                      # a shared colour scale would recolour the dots
    assert len(payoff_chart.payoff_chart(*args, current=-15.0, replays=[]).to_dict()["layer"]) == len(plain["layer"])
