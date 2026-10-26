"""Seasonal overlay engine: earlier-year pairs, alignment by days to expiry, replays, the average (no network)."""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cso import seasonal

NEAR_EXPIRY = date(2026, 12, 14)                       # the live ZCZ6


def _history(expiry, days, start_price, daily):
    """A contract's daily settlements for the ``days`` calendar days up to and including its last trading day."""
    dates = [expiry - timedelta(days=d) for d in range(days, -1, -1)]
    return pd.Series([start_price + daily * i for i in range(len(dates))], index=dates, dtype=float)


def _year(back, near_expiry_year, spread_by_dte):
    return seasonal.Year(back, near_expiry_year, date(near_expiry_year, 12, 13),
                         pd.Series(spread_by_dte.values(), index=pd.Index(list(spread_by_dte), dtype="int64")))


# --------------------------------------------------------------------------- the earlier pairs
def test_prior_pairs_step_both_legs_back_a_year_at_a_time():
    assert seasonal.prior_pairs("ZCZ6", "ZCH7", 3) == [(1, "ZCZ5", "ZCH6"), (2, "ZCZ4", "ZCH5"), (3, "ZCZ3", "ZCH4")]


def test_prior_pairs_wrap_the_year_digit():
    assert seasonal.prior_pairs("ZCZ2", "ZCH3", 3) == [(1, "ZCZ1", "ZCH2"), (2, "ZCZ0", "ZCH1"), (3, "ZCZ9", "ZCH0")]
    assert seasonal.prior_pairs("ZCN7", "ZCU7", 1) == [(1, "ZCN6", "ZCU6")]             # both legs in one calendar year


@pytest.mark.parametrize("near, far", [("ZCZ6-ZCH7", "ZCH7"), ("ZCZ6", "ZC:BF H7"), ("", "ZCH7"), (None, None)])
def test_prior_pairs_refuse_what_is_not_a_pair_of_months(near, far):
    assert seasonal.prior_pairs(near, far, 3) == []


def test_tickers_needed_is_a_sorted_unique_tuple():
    assert seasonal.tickers_needed("ZCZ6", "ZCH7", 2) == ("ZCH5", "ZCH6", "ZCZ4", "ZCZ5")
    assert seasonal.tickers_needed("ZCZ6", "bad", 2) == ()


# --------------------------------------------------------------------------- building the years
def test_each_year_is_indexed_by_days_to_its_own_near_expiry():
    z5, h6 = date(2025, 12, 12), date(2026, 3, 13)
    hist = {"ZCZ5": _history(z5, 400, 430.0, 0.05), "ZCH6": _history(h6, 400 + 91, 440.0, 0.05)}
    years = seasonal.build_years(hist, "ZCZ6", "ZCH7", NEAR_EXPIRY, 1, window=365)
    assert len(years) == 1
    y = years[0]
    assert (y.back, y.year, y.near_expiry) == (1, 2025, z5)
    assert y.by_dte.index.max() == 0 and y.by_dte.index.min() >= -365 and y.by_dte.index.is_monotonic_increasing
    near_on_expiry, far_on_expiry = hist["ZCZ5"][z5], hist["ZCH6"].get(z5)
    assert far_on_expiry is not None and y.by_dte[0] == pytest.approx(near_on_expiry - far_on_expiry)


def test_the_sign_flips_every_level_and_the_window_cuts_the_far_past():
    z5 = date(2025, 12, 12)
    hist = {"ZCZ5": _history(z5, 400, 430.0, 0.05), "ZCH6": _history(date(2026, 3, 13), 491, 440.0, 0.05)}
    plain = seasonal.build_years(hist, "ZCZ6", "ZCH7", NEAR_EXPIRY, 1, sign=1.0, window=180)[0].by_dte
    flipped = seasonal.build_years(hist, "ZCZ6", "ZCH7", NEAR_EXPIRY, 1, sign=-1.0, window=180)[0].by_dte
    assert (flipped == -plain).all() and plain.index.min() >= -180


def test_years_without_both_legs_or_in_the_wrong_month_are_left_out():
    z5, h6 = date(2025, 12, 12), date(2026, 3, 13)
    good = {"ZCZ5": _history(z5, 200, 430.0, 0.0), "ZCH6": _history(h6, 291, 440.0, 0.0)}
    assert len(seasonal.build_years(good, "ZCZ6", "ZCH7", NEAR_EXPIRY, 1)) == 1
    assert seasonal.build_years({"ZCZ5": good["ZCZ5"]}, "ZCZ6", "ZCH7", NEAR_EXPIRY, 1) == []                 # no far leg
    assert seasonal.build_years({**good, "ZCH6": pd.Series(dtype=float)}, "ZCZ6", "ZCH7", NEAR_EXPIRY, 1) == []
    never_finished = {"ZCZ5": _history(date(2025, 11, 20), 200, 430.0, 0.0), "ZCH6": good["ZCH6"]}           # ends in November
    assert seasonal.build_years(never_finished, "ZCZ6", "ZCH7", NEAR_EXPIRY, 1) == []
    assert seasonal.build_years({"ZCZ5": good["ZCZ5"], "ZCH6": _history(h6, 5, 440.0, 0.0)}, "ZCZ6", "ZCH7", NEAR_EXPIRY, 1) == []   # no overlap


def test_a_short_history_is_kept_as_far_as_it_goes():
    z1, h2 = date(2021, 12, 14), date(2022, 3, 14)
    hist = {"ZCZ1": _history(z1, 47, 430.0, 0.1), "ZCH2": _history(h2, 138, 445.0, 0.1)}
    y = seasonal.build_years(hist, "ZCZ6", "ZCH7", NEAR_EXPIRY, 5, window=365)[0]
    assert (y.back, y.year) == (5, 2021) and y.by_dte.index.min() == -47        # begins where the feed does


def test_the_current_contract_uses_the_same_footing():
    near = _history(NEAR_EXPIRY - timedelta(days=60), 100, 450.0, 0.0)          # the live contract has not expired
    far = _history(NEAR_EXPIRY - timedelta(days=60), 100, 465.0, 0.0)
    s = seasonal.current_by_dte(near, far, NEAR_EXPIRY, sign=1.0, window=365)
    assert s.index.max() == -60 and (s == -15.0).all()
    assert seasonal.current_by_dte(pd.Series(dtype=float), far, NEAR_EXPIRY).empty


# --------------------------------------------------------------------------- reading a year at a day
def test_value_at_takes_the_last_observation_inside_the_tolerance():
    s = pd.Series([-10.0, -12.0, -15.0], index=pd.Index([-60, -49, -20], dtype="int64"))
    assert seasonal.value_at(s, -49) == -12.0                                   # exact
    assert seasonal.value_at(s, -47) == -12.0                                   # two days after the last print
    assert seasonal.value_at(s, -40) is None                                    # nine days stale: too old to call it that day
    assert seasonal.value_at(s, -40, tolerance=10) == -12.0
    assert seasonal.value_at(s, -90) is None                                    # before the series starts
    assert seasonal.value_at(s, 0) is None                                      # long after it ends


# --------------------------------------------------------------------------- replays
def test_a_replay_is_each_years_move_between_the_matching_days():
    y25 = _year(1, 2025, {-60: -10.0, -49: -12.0, -20: -15.0, -13: -16.0, -5: -18.0, 0: -19.0})
    y24 = _year(2, 2024, {-60: -8.0, -49: -9.0, -13: -5.0})
    got = seasonal.replays([y25, y24], dte_now=-49, dte_expiry=-13)
    assert [(r.year, r.then, r.at_expiry, r.change) for r in got] == [(2025, -12.0, -16.0, -4.0), (2024, -9.0, -5.0, 4.0)]


def test_a_year_missing_either_day_is_skipped_and_a_backwards_window_gives_nothing():
    complete = _year(1, 2025, {-49: -12.0, -13: -16.0})
    no_end = _year(2, 2024, {-49: -9.0, -30: -8.0})                             # nothing within 5 days of -13
    no_start = _year(3, 2023, {-13: -4.0})
    assert [r.year for r in seasonal.replays([complete, no_end, no_start], -49, -13)] == [2025]
    assert seasonal.replays([complete], dte_now=-13, dte_expiry=-49) == []
    assert seasonal.replays([complete], dte_now=-13, dte_expiry=-13) == []


# --------------------------------------------------------------------------- the average and the calendar
def test_the_average_needs_two_years_and_half_of_them_present():
    a = _year(1, 2025, {d: -10.0 for d in range(-10, 1)})
    b = _year(2, 2024, {d: -20.0 for d in range(-6, 1)})
    c = _year(3, 2023, {d: -30.0 for d in range(-3, 1)})
    assert seasonal.average([a], 10) is None
    avg = seasonal.average([a, b, c], 10)
    assert avg.index.min() == -6                                                # before that only one year exists
    assert avg[-5] == pytest.approx(-15.0)                                      # a and b
    assert avg[0] == pytest.approx(-20.0)                                       # all three
    assert seasonal.average([_year(1, 2025, {-1: 1.0}), _year(2, 2024, {-9: 2.0})], 10) is None   # never two at once


def test_the_average_interpolates_between_a_years_own_sessions():
    a = _year(1, 2025, {-10: -10.0, 0: -20.0})                                  # two prints only
    b = _year(2, 2024, {d: -10.0 for d in range(-10, 1)})
    avg = seasonal.average([a, b], 10)
    assert avg[-5] == pytest.approx((-15.0 + -10.0) / 2)


def test_on_calendar_puts_days_to_expiry_on_the_live_contracts_dates():
    s = pd.Series([-12.0, -16.0, -19.0], index=pd.Index([-49, -13, 0], dtype="int64"))
    cal = seasonal.on_calendar(s, NEAR_EXPIRY)
    assert list(cal.index) == [date(2026, 10, 26), date(2026, 12, 1), date(2026, 12, 14)] and list(cal.values) == [-12.0, -16.0, -19.0]


def test_a_year_of_random_walk_survives_the_whole_chain():
    rng = np.random.default_rng(11)
    z5, h6 = date(2025, 12, 12), date(2026, 3, 13)
    near = pd.Series(430 + np.cumsum(rng.normal(0, 1, 300)), index=[z5 - timedelta(days=299 - i) for i in range(300)])
    far = pd.Series(445 + np.cumsum(rng.normal(0, 1, 300)), index=near.index)
    years = seasonal.build_years({"ZCZ5": near, "ZCH6": far}, "ZCZ6", "ZCH7", NEAR_EXPIRY, 1)
    assert len(years) == 1 and seasonal.replays(years, -49, -13) != [] and seasonal.average(years, 365) is None
