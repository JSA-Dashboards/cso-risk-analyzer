"""Strategy payoffs: bought put over a spread position, exact break-evens, caps and floors."""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cso import pricing, strategy
from cso.strategy import CALL, FUTURE, PUT, Leg

MULT = 50.0
VAL = date(2026, 10, 26)
EXP = VAL + timedelta(days=36)


def put_over_long_spread(entry=-15.0, qty=10, strike=-20.0, premium=0.75):
    return [Leg(FUTURE, qty, entry), Leg(PUT, qty, strike, premium, EXP)]


def test_a_bought_put_alone():
    legs = [Leg(PUT, 10, -20.0, 0.75, EXP)]
    assert strategy.at_expiry(legs, -30.0, MULT) == pytest.approx(10 * 50 * (10 - 0.75))
    assert strategy.at_expiry(legs, -10.0, MULT) == pytest.approx(-10 * 50 * 0.75)         # worthless: lose the premium
    s = strategy.summary(legs, MULT)
    assert s["premium_paid"] == pytest.approx(10 * 50 * 0.75)
    assert s["max_loss"] == pytest.approx(-10 * 50 * 0.75)                                  # the premium, nothing more
    assert s["max_gain"] is None                                                           # unbounded in theory
    assert s["breakevens"] == pytest.approx([-20.75])                                       # strike - premium


def test_put_over_a_long_spread_floors_the_loss_and_leaves_the_upside_open():
    legs = put_over_long_spread()
    # below the strike the put offsets the spread leg one for one, so the loss stops at (entry - strike) + premium
    floor = -10 * 50 * ((-15.0 - -20.0) + 0.75)
    assert strategy.at_expiry(legs, -35.0, MULT) == pytest.approx(floor)
    assert strategy.at_expiry(legs, -20.0, MULT) == pytest.approx(floor)
    s = strategy.summary(legs, MULT)
    assert s["max_loss"] == pytest.approx(floor) and s["max_gain"] is None
    assert s["breakevens"] == pytest.approx([-15.0 + 0.75])                                 # entry + premium
    assert strategy.at_expiry(legs, -14.25, MULT) == pytest.approx(0.0, abs=1e-9)
    assert strategy.at_expiry(legs, -10.0, MULT) == pytest.approx(10 * 50 * (5.0 - 0.75))   # upside minus the premium


def test_unchanged_spread_costs_the_premium():
    legs = put_over_long_spread()
    assert strategy.at_expiry(legs, -15.0, MULT) == pytest.approx(-10 * 50 * 0.75)


def test_short_spread_with_a_bought_call_mirrors_it():
    legs = [Leg(FUTURE, -10, -15.0), Leg(CALL, 10, -10.0, 0.75, EXP)]
    s = strategy.summary(legs, MULT)
    assert s["max_loss"] == pytest.approx(-10 * 50 * ((-10.0 - -15.0) + 0.75)) and s["max_gain"] is None
    assert s["breakevens"] == pytest.approx([-15.0 - 0.75])
    assert strategy.tail_slopes(legs, MULT) == (-500.0, 0.0)


def test_a_position_with_no_option_is_unbounded_both_ways_and_breaks_even_at_entry():
    s = strategy.summary([Leg(FUTURE, 10, -15.0)], MULT)
    assert s["max_loss"] is None and s["max_gain"] is None and s["breakevens"] == pytest.approx([-15.0])
    assert strategy.summary([], MULT) == {"premium_paid": 0, "max_loss": 0.0, "max_gain": 0.0, "breakevens": []}


def test_a_sold_put_collects_premium_and_has_unlimited_loss_on_the_carry_side():
    legs = [Leg(PUT, -10, -20.0, 0.75, EXP)]
    s = strategy.summary(legs, MULT)
    assert s["premium_paid"] == pytest.approx(-10 * 50 * 0.75)                              # negative = received
    assert s["max_gain"] == pytest.approx(10 * 50 * 0.75) and s["max_loss"] is None


def test_a_long_spread_with_a_sold_call_caps_the_gain_at_the_strike_plus_the_premium():
    legs = [Leg(FUTURE, 10, -15.0), Leg(CALL, -10, -10.0, 0.75, EXP)]
    s = strategy.summary(legs, MULT)
    assert s["premium_paid"] == pytest.approx(-10 * 50 * 0.75)                              # received
    assert s["max_gain"] == pytest.approx(10 * 50 * ((-10.0 - -15.0) + 0.75)) and s["max_loss"] is None
    assert s["breakevens"] == pytest.approx([-15.0 - 0.75])                                 # entry less the premium kept
    assert strategy.at_expiry(legs, -5.0, MULT) == pytest.approx(s["max_gain"])             # flat above the strike


def test_a_short_spread_with_a_sold_put_caps_the_gain_at_the_strike_less_the_premium():
    legs = [Leg(FUTURE, -10, -15.0), Leg(PUT, -10, -20.0, 0.75, EXP)]
    s = strategy.summary(legs, MULT)
    assert s["max_gain"] == pytest.approx(10 * 50 * ((-15.0 - -20.0) + 0.75)) and s["max_loss"] is None
    assert s["breakevens"] == pytest.approx([-15.0 + 0.75])


def test_put_spread_over_a_position_has_a_floor_and_a_cap_on_the_hedge():
    legs = put_over_long_spread() + [Leg(PUT, -10, -30.0, 0.10, EXP)]                       # sell the -30 put
    s = strategy.summary(legs, MULT)
    assert s["max_loss"] is None                              # below -30 the short put stops protecting
    assert strategy.tail_slopes(legs, MULT) == (500.0, 500.0)
    assert strategy.at_expiry(legs, -25.0, MULT) == pytest.approx(strategy.at_expiry(legs, -20.0, MULT))   # flat between strikes


def test_breakevens_are_exact_zeros_of_the_payoff():
    for legs in (put_over_long_spread(), [Leg(PUT, 5, -20.0, 0.75, EXP), Leg(CALL, 5, -5.0, 0.5, EXP)],
                 put_over_long_spread() + [Leg(PUT, -10, -30.0, 0.10, EXP)]):
        for b in strategy.breakevens(legs, MULT):
            assert strategy.at_expiry(legs, b, MULT) == pytest.approx(0.0, abs=1e-7), (legs, b)


def test_grid_contains_the_kinks_and_breakevens_so_the_shading_meets_zero_exactly():
    legs = put_over_long_spread()
    g = strategy.grid(legs, MULT, -50.0, 20.0)
    assert list(g) == sorted(set(g)) and g[0] == -50.0 and g[-1] == 20.0
    assert -20.0 in g and any(abs(g - (-14.25)) < 1e-9)
    pnl = strategy.at_expiry(legs, g, MULT)
    zero_at = g[np.isclose(pnl, 0.0, atol=1e-7)]
    assert len(zero_at) == 1 and zero_at[0] == pytest.approx(-14.25)


def test_model_value_at_expiry_equals_the_payoff_and_before_expiry_includes_time_value():
    legs = put_over_long_spread()
    S = np.linspace(-40, 10, 11)
    assert strategy.value_on(legs, S, MULT, 20.0, 0.04, EXP) == pytest.approx(strategy.at_expiry(legs, S, MULT))
    now = strategy.value_on(legs, S, MULT, 20.0, 0.04, VAL)
    price = pricing.price(False, S, -20.0, 20.0, 36 / 365, 0.04)
    assert now == pytest.approx(10 * 50 * (S + 15.0) + 10 * 50 * (price - 0.75))
    # a put at -20 with the spread at -20 is worth more than intrinsic (0) before expiry
    assert strategy.value_on([Leg(PUT, 1, -20.0, 0.0, EXP)], -20.0, MULT, 20.0, 0.04, VAL) > 0


def test_a_per_leg_vol_overrides_the_default():
    leg = Leg(PUT, 1, -20.0, 0.0, EXP, vol=35.0)
    a = strategy.value_on([leg], -15.0, MULT, 10.0, 0.04, VAL)
    b = strategy.value_on([Leg(PUT, 1, -20.0, 0.0, EXP)], -15.0, MULT, 35.0, 0.04, VAL)
    assert a == pytest.approx(b)
