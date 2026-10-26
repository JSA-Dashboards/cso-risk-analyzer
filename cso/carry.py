"""Full-carry math for a near/far futures pair, ported from the Cost of Carry calculator so the two agree.

Source (JSA-Dashboards/cost-of-carry-calculator @ 68a2f0e): app.py ``carry_components``, ``compute_carry_table``,
``carry_bucket``, ``COMMODITIES`` defaults and ``FED_FUNDS_SPREAD_PCT``; storage_rates.py (the CME maximum storage
schedule and ``carry_rate``); massive_api.get_fed_funds_rate (here: ``massive.fed_funds_pct``).
tests/test_carry.py compares against that repo when it sits alongside this one, so a change to the CME schedule or
the formula there shows up here as a failing test instead of a silent drift.

Spread = near price - far price (negative in a carry market). Full carry = storage + interest to hold the commodity
from the near contract's expiry to the far one's. "% of full carry" = spread / -full carry.
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import date

FED_FUNDS_SPREAD_PCT = 2.25            # JSA convention: interest = fed funds + 2.25%
FALLBACK_ANNUAL_RATE_PCT = 5.89        # used when the live fed-funds read fails

# product code -> (default storage as the calculator stores it, quote multiplier, storage unit per day)
# Grains store $/bu/day and x100 gives cents; meal ($/ton) and oil (cents/lb) are already in quote units.
MARKETS = {
    "ZC": (0.00265, 100, "¢/bu/day"),
    "ZS": (0.00265, 100, "¢/bu/day"),
    "ZW": (0.00265, 100, "¢/bu/day"),
    "KE": (0.00165, 100, "¢/bu/day"),
    "ZM": (0.12, 1, "$/ton/day"),
    "ZL": (0.005, 1, "¢/lb/day"),
}

# CME maximum storage charge, $/bu/day, ascending by effective date (see the source module for the SER notices).
SCHEDULE: dict[str, list[tuple[date, float]]] = {
    "ZC": [(date(2006, 1, 1), 0.00165), (date(2019, 12, 19), 0.00265)],
    "ZS": [(date(2006, 1, 1), 0.00165), (date(2019, 11, 19), 0.00265)],
    "ZW": [(date(2021, 3, 1), 0.00165), (date(2023, 9, 19), 0.00265), (date(2024, 3, 19), 0.00165),
           (date(2024, 5, 19), 0.00265), (date(2026, 3, 19), 0.00165), (date(2026, 12, 19), 0.00265)],
    "KE": [(date(2021, 3, 1), 0.00165), (date(2025, 5, 19), 0.00265), (date(2026, 7, 19), 0.00165),
           (date(2026, 12, 19), 0.00265)],
}
_DATES = {code: [d for d, _ in steps] for code, steps in SCHEDULE.items()}
EFFECTIVE_DAY = 19                      # storage changes take effect on the 19th of the delivery month

BUCKET_COLORS = {"high": "green", "mid": "orange", "low": "red"}   # Streamlit badge colours


def rate_on(product: str, on: date) -> float | None:
    steps = SCHEDULE.get(product)
    if not steps:
        return None
    i = bisect_right(_DATES[product], on) - 1
    return steps[i][1] if i >= 0 else None


def carry_window(near_expiry: date, far_expiry: date) -> tuple[date, date]:
    """[start, end) over which storage is paid: the 19th of the near delivery month to the 19th of the far one."""
    return (date(near_expiry.year, near_expiry.month, EFFECTIVE_DAY),
            date(far_expiry.year, far_expiry.month, EFFECTIVE_DAY))


def average_rate(product: str, start: date, end: date, fallback: float) -> float:
    """Day-weighted average maximum storage rate across [start, end)."""
    total_days = (end - start).days
    if total_days <= 0 or product not in SCHEDULE:
        return fallback
    steps = SCHEDULE[product]
    cuts = sorted({start, end} | {d for d, _ in steps if start < d < end})
    weighted = 0.0
    for a, b in zip(cuts, cuts[1:]):
        rate = rate_on(product, a)
        weighted += (b - a).days * (fallback if rate is None else rate)
    return weighted / total_days


def storage_rate(product: str, near_expiry: date, far_expiry: date) -> float:
    """The storage rate a spread is carried under (the calculator's default, "historical rates" on)."""
    default = MARKETS[product][0]
    start, end = carry_window(near_expiry, far_expiry)
    return average_rate(product, start, end, default)


def carry_components(near_price: float, days: int, storage_rate_: float, annual_rate: float,
                     multiplier: int) -> tuple[float, float]:
    """(full storage, full interest) in quote units for one near/far pair; ``annual_rate`` is a decimal."""
    storage_full = days * storage_rate_ * multiplier
    interest_full = near_price * annual_rate * days / 360
    return storage_full, interest_full


def full_carry(near_price: float, days: int, storage_rate_: float, annual_rate: float, multiplier: int) -> dict:
    storage_full, interest_full = carry_components(near_price, days, storage_rate_, annual_rate, multiplier)
    return {"storage": storage_full, "interest": interest_full, "full": storage_full + interest_full}


def pct_of_full_carry(spread: float, full: float) -> float | None:
    """spread / -full carry: 1.0 means the market pays the whole cost of carry."""
    return None if not full else spread / -full


def bucket(pct: float) -> str:
    if pct >= 0.75:
        return "high"
    if pct >= 0.50:
        return "mid"
    return "low"
