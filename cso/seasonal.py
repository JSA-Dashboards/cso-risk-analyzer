"""Seasonal overlay: the same calendar spread in earlier years, lined up by days to the near contract's expiry.

Each prior year is the pair one year earlier at a time (Dec25/Mar26 for Dec26/Mar27, and so on back). A year is
indexed by days to its near leg's last trading day (negative before it, zero on it), so every year can be laid over
the current contract's own calendar. Levels come back in whatever sign the caller asks for.

A "replay" asks what each prior year did between the days that match today and the option's expiry, and applies
that move to today's level. It is a scenario read off a handful of years, not a forecast.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from . import massive

MAX_PRIOR_YEARS = 5                 # the feed's daily bars begin in autumn 2021, so five is as far back as it goes


@dataclass(frozen=True)
class Year:
    back: int                       # years before the current contract
    year: int                       # calendar year the near leg expired in
    near_expiry: date               # that contract's last trading day
    by_dte: pd.Series               # spread level indexed by days to near_expiry (<= 0), ascending


@dataclass(frozen=True)
class Replay:
    year: int
    then: float                     # that year's level on the day matching today
    at_expiry: float                # ... and on the day matching the option's expiry
    change: float                   # at_expiry - then


def prior_pairs(near_t: str, far_t: str, years: int) -> list[tuple[int, str, str]]:
    """``(years back, near ticker, far ticker)`` for each earlier year, nearest first: ZCZ6/ZCH7 -> ZCZ5/ZCH6, ..."""
    n, f = massive.split_ticker(near_t), massive.split_ticker(far_t)
    if n is None or f is None:
        return []
    return [(k, f"{n[0]}{n[1]}{(n[2] - k) % 10}", f"{f[0]}{f[1]}{(f[2] - k) % 10}") for k in range(1, years + 1)]


def tickers_needed(near_t: str, far_t: str, years: int) -> tuple[str, ...]:
    """Every earlier contract whose history the overlay reads, as one sorted tuple (a stable cache key)."""
    return tuple(sorted({t for _, n, f in prior_pairs(near_t, far_t, years) for t in (n, f)}))


def _by_dte(spread: pd.Series, expiry: date, sign: float, window: int) -> pd.Series:
    index = pd.Index([(d - expiry).days for d in spread.index], dtype="int64")
    series = pd.Series(spread.values * sign, index=index).sort_index()
    return series[series.index >= -window]


def build_years(histories: dict[str, pd.Series], near_t: str, far_t: str, near_expiry: date, years: int, *,
                sign: float = 1.0, window: int = 365) -> list[Year]:
    """The earlier years that have history for both legs, nearest year first.

    The expiry of an earlier contract is its own last bar. A series whose last bar is not in the month the current
    near leg expires in (one year per step back) never finished or is not the contract it claims to be: left out.
    """
    out = []
    for back, near, far in prior_pairs(near_t, far_t, years):
        near_h, far_h = histories.get(near), histories.get(far)
        if near_h is None or far_h is None or near_h.empty or far_h.empty:
            continue
        expiry = near_h.index.max()
        if (expiry.year, expiry.month) != (near_expiry.year - back, near_expiry.month):
            continue
        both = massive.spread_history(near_h, far_h, days=None)
        if both.empty:
            continue
        series = _by_dte(both, expiry, sign, window)
        if not series.empty:
            out.append(Year(back, expiry.year, expiry, series))
    return out


def current_by_dte(near_h: pd.Series, far_h: pd.Series, near_expiry: date, *, sign: float = 1.0,
                   window: int = 365) -> pd.Series:
    """The live contract's spread on the same days-to-expiry footing (empty without history for both legs)."""
    both = massive.spread_history(near_h, far_h, days=None)
    return _by_dte(both, near_expiry, sign, window) if not both.empty else pd.Series(dtype=float)


def value_at(series: pd.Series, dte: int, tolerance: int = 5) -> float | None:
    """The last observation on or before ``dte``, when it is within ``tolerance`` days of it (else None)."""
    earlier = series[series.index <= dte]
    if earlier.empty or dte - int(earlier.index[-1]) > tolerance:
        return None
    return float(earlier.iloc[-1])


def replays(years: list[Year], dte_now: int, dte_expiry: int, tolerance: int = 5) -> list[Replay]:
    """Each year's move between the days matching today (``dte_now``) and the option's expiry (``dte_expiry``).

    A year without an observation near both days is skipped. Nothing when the expiry is not after today.
    """
    if dte_expiry <= dte_now:
        return []
    out = []
    for y in years:
        then, end = value_at(y.by_dte, dte_now, tolerance), value_at(y.by_dte, dte_expiry, tolerance)
        if then is not None and end is not None:
            out.append(Replay(y.year, then, end, end - then))
    return out


def average(years: list[Year], window: int) -> pd.Series | None:
    """Mean across the years on a daily grid, only where at least half of them (and never fewer than two) have data.

    Each year trades on its own session dates, so averaging the raw points would jump between "mean of five years"
    and "one lonely year": put every year on a common daily grid first.
    """
    if len(years) < 2:
        return None
    grid = pd.RangeIndex(-window, 1)
    frame = pd.DataFrame({
        y.year: y.by_dte[~y.by_dte.index.duplicated(keep="last")].reindex(grid).interpolate(limit_area="inside")
        for y in years})
    mean = frame.mean(axis=1, skipna=True)[frame.count(axis=1) >= max(2, (len(years) + 1) // 2)]
    return mean if len(mean) else None


def on_calendar(by_dte: pd.Series, near_expiry: date) -> pd.Series:
    """Re-index a days-to-expiry series onto the dates of the contract whose expiry is ``near_expiry``."""
    return pd.Series(by_dte.values, index=[near_expiry + timedelta(days=int(d)) for d in by_dte.index])
