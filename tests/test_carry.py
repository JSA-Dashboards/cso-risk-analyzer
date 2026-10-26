"""Full-carry math: same numbers as the Cost of Carry calculator."""
import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cso import carry

COC = ROOT.parent / "cost-of-carry-calculator"


def test_formula_matches_the_calculators_worked_numbers():
    # Dec26/Mar27 corn: 88 days between expiries, 0.265 c/bu/day storage, 6.135% interest on a 478.25 board
    s, i = carry.carry_components(478.25, 88, 0.00265, 0.06135, 100)
    assert s == pytest.approx(88 * 0.00265 * 100)                     # 23.32 cents
    assert i == pytest.approx(478.25 * 0.06135 * 88 / 360)            # 7.17 cents
    fc = carry.full_carry(478.25, 88, 0.00265, 0.06135, 100)
    assert fc["full"] == pytest.approx(s + i) and fc["storage"] == s and fc["interest"] == i
    assert carry.pct_of_full_carry(-12.75, fc["full"]) == pytest.approx(12.75 / fc["full"])
    assert carry.pct_of_full_carry(-12.75, 0) is None


def test_non_grain_units_have_multiplier_one():
    s, _ = carry.carry_components(360.0, 90, 0.12, 0.06, 1)            # meal: $/ton/day
    assert s == pytest.approx(10.8)
    assert carry.MARKETS["ZM"][1] == 1 and carry.MARKETS["ZL"][1] == 1 and carry.MARKETS["ZC"][1] == 100


@pytest.mark.parametrize("pct, expected", [(1.2, "high"), (0.75, "high"), (0.749, "mid"), (0.5, "mid"),
                                           (0.499, "low"), (0.0, "low"), (-0.3, "low")])
def test_traffic_light_thresholds(pct, expected):
    assert carry.bucket(pct) == expected


def test_cme_storage_schedule():
    assert carry.rate_on("ZC", date(2019, 12, 18)) == 0.00165 and carry.rate_on("ZC", date(2019, 12, 19)) == 0.00265
    assert carry.rate_on("ZS", date(2026, 10, 26)) == 0.00265
    assert carry.rate_on("ZW", date(2026, 3, 18)) == 0.00265 and carry.rate_on("ZW", date(2026, 3, 19)) == 0.00165
    assert carry.rate_on("ZW", date(2026, 12, 19)) == 0.00265
    assert carry.rate_on("ZW", date(2020, 1, 1)) is None and carry.rate_on("ZM", date(2026, 1, 1)) is None


def test_carry_rate_is_a_day_weighted_average_across_the_window():
    # corn is flat 0.00265 across Dec26 -> Mar27; wheat steps up on 2026-12-19, inside a Sep26 -> Mar27 window
    assert carry.storage_rate("ZC", date(2026, 12, 14), date(2027, 3, 12)) == pytest.approx(0.00265)
    start, end = carry.carry_window(date(2026, 9, 14), date(2027, 3, 12))
    assert (start, end) == (date(2026, 9, 19), date(2027, 3, 19))
    r = carry.storage_rate("ZW", date(2026, 9, 14), date(2027, 3, 12))
    days_low, days_high = (date(2026, 12, 19) - start).days, (end - date(2026, 12, 19)).days
    assert r == pytest.approx((days_low * 0.00165 + days_high * 0.00265) / (days_low + days_high))
    assert carry.storage_rate("ZM", date(2026, 12, 14), date(2027, 3, 12)) == 0.12          # no schedule: default


@pytest.mark.skipif(not (COC / "storage_rates.py").exists(), reason="cost-of-carry-calculator not alongside")
def test_schedule_and_rates_match_the_calculators_own_module():
    spec = importlib.util.spec_from_file_location("coc_storage_rates", COC / "storage_rates.py")
    coc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coc)
    for product in carry.SCHEDULE:
        assert carry.SCHEDULE[product] == coc.SCHEDULE[product], product
    assert carry.EFFECTIVE_DAY == coc.EFFECTIVE_DAY
    for near, far in ((date(2026, 12, 14), date(2027, 3, 12)), (date(2026, 9, 14), date(2027, 3, 12)),
                      (date(2026, 7, 14), date(2026, 12, 14))):
        for product in carry.SCHEDULE:
            assert carry.storage_rate(product, near, far) == pytest.approx(
                coc.carry_rate(product, near, far, carry.MARKETS[product][0])), (product, near, far)


@pytest.mark.skipif(not (COC / "app.py").exists(), reason="cost-of-carry-calculator not alongside")
def test_constants_and_formula_match_the_calculators_source_text():
    src = (COC / "app.py").read_text(encoding="utf-8")
    assert f"FED_FUNDS_SPREAD_PCT = {carry.FED_FUNDS_SPREAD_PCT}" in src
    assert f"FALLBACK_ANNUAL_RATE_PCT = {carry.FALLBACK_ANNUAL_RATE_PCT}" in src
    assert "storage_full = days * storage_rate * multiplier" in src
    assert "interest_full = near_price * annual_rate * days / 360" in src
    for product, (default, mult, _unit) in carry.MARKETS.items():
        block = src[src.index(f'"product_code": "{product}"'):][:200]
        assert f'"default_storage": {default}' in block and f'"multiplier": {mult}' in block, product
