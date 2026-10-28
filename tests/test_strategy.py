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


# --------------------------------------------------------------------------- two option legs: spreads and collars
def test_a_collar_over_a_long_spread_floors_the_loss_and_caps_the_gain():
    legs = [Leg(FUTURE, 10, -15.0), Leg(PUT, 10, -20.0, 0.75, EXP), Leg(CALL, -10, -10.0, 0.50, EXP)]
    s = strategy.summary(legs, MULT)
    assert s["premium_paid"] == pytest.approx(10 * 50 * (0.75 - 0.50))                      # the net premium
    assert s["max_loss"] == pytest.approx(-10 * 50 * (5.0 + 0.25))                          # entry to the put strike, plus the net premium
    assert s["max_gain"] == pytest.approx(10 * 50 * (5.0 - 0.25))                           # entry to the call strike, less it
    assert s["breakevens"] == pytest.approx([-15.0 + 0.25])
    assert strategy.tail_slopes(legs, MULT) == (0.0, 0.0)                                   # bounded both ways
    assert strategy.at_expiry(legs, -12.0, MULT) == pytest.approx(10 * 50 * (3.0 - 0.25))   # between the strikes it is just the position


def test_a_collar_over_a_short_spread_is_a_bought_call_and_a_sold_put():
    legs = [Leg(FUTURE, -10, -15.0), Leg(CALL, 10, -10.0, 0.75, EXP), Leg(PUT, -10, -20.0, 0.50, EXP)]
    s = strategy.summary(legs, MULT)
    assert s["max_loss"] == pytest.approx(-10 * 50 * (5.0 + 0.25)) and s["max_gain"] == pytest.approx(10 * 50 * (5.0 - 0.25))
    assert strategy.tail_slopes(legs, MULT) == (0.0, 0.0)


def test_a_put_spread_alone_caps_both_the_loss_and_the_gain():
    legs = [Leg(PUT, 10, -20.0, 0.75, EXP), Leg(PUT, -10, -30.0, 0.10, EXP)]
    s = strategy.summary(legs, MULT)
    assert s["max_loss"] == pytest.approx(-10 * 50 * 0.65)                                  # the net premium
    assert s["max_gain"] == pytest.approx(10 * 50 * (10.0 - 0.65))                          # the width less the net premium
    assert s["breakevens"] == pytest.approx([-20.0 - 0.65])


def test_a_call_spread_over_a_short_spread_caps_the_protection():
    legs = [Leg(FUTURE, -10, -15.0), Leg(CALL, 10, -10.0, 0.75, EXP), Leg(CALL, -10, 0.0, 0.10, EXP)]
    assert strategy.tail_slopes(legs, MULT) == (-500.0, -500.0)                             # above 0 the short call stops protecting
    assert strategy.summary(legs, MULT)["max_loss"] is None


@pytest.mark.parametrize("held, options, name", [
    (10, [(PUT, 10, -20.0)], "Long put"),
    (10, [(CALL, -10, -10.0)], "Short call"),
    (10, [(PUT, 10, -20.0), (PUT, -10, -30.0)], "Put spread"),
    (10, [(PUT, -10, -20.0), (PUT, 10, -30.0)], "Put spread (credit)"),
    (0, [(CALL, 10, -10.0), (CALL, -10, 0.0)], "Call spread"),
    (0, [(CALL, -10, -10.0), (CALL, 10, 0.0)], "Call spread (credit)"),
    (10, [(PUT, 10, -20.0), (PUT, -20, -30.0)], "Ratio put spread"),
    (10, [(PUT, 10, -20.0), (CALL, -10, -10.0)], "Collar"),
    (-10, [(CALL, 10, -10.0), (PUT, -10, -20.0)], "Collar"),
    (0, [(PUT, 10, -20.0), (CALL, -10, -10.0)], "Risk reversal"),
    (-10, [(PUT, 10, -20.0), (CALL, -10, -10.0)], "Risk reversal"),                         # a bought put over a SHORT spread doubles up
    (10, [(CALL, 10, -10.0), (PUT, -10, -20.0)], "Risk reversal"),
    (0, [(PUT, 10, -15.0), (CALL, 10, -15.0)], "Straddle"),
    (0, [(PUT, 10, -20.0), (CALL, 10, -10.0)], "Strangle"),
    (0, [(PUT, -10, -15.0), (CALL, -10, -15.0)], "Short straddle"),
    (0, [(PUT, -10, -20.0), (CALL, -10, -10.0)], "Short strangle"),
    (0, [(PUT, 10, -20.0), (PUT, 10, -30.0)], "Two puts"),
    (0, [(PUT, 10, -20.0), (PUT, -10, -20.0)], "Offsetting options"),
    (10, [(PUT, 10, -20.0), (PUT, -10, -30.0), (CALL, -10, -10.0)], "Custom"),
])
def test_the_legs_get_the_name_a_trader_would_give_them(held, options, name):
    legs = ([Leg(FUTURE, held, -15.0)] if held else []) + [Leg(kind, qty, k, 0.5, EXP) for kind, qty, k in options]
    assert strategy.name_of(legs) == name


def test_describe_says_what_each_option_leg_is():
    legs = [Leg(FUTURE, 10, -15.0), Leg(PUT, 10, -20.0, 0.75, EXP), Leg(CALL, -5, 5.0, 0.5, EXP)]
    assert strategy.describe(legs) == "buy 10 -20 put, sell 5 +5 call"
    assert strategy.describe([Leg(FUTURE, 10, -15.0)]) == ""
