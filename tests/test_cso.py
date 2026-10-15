import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cso import book, pricing

VAL = date(2026, 10, 15)
EXP = VAL + timedelta(days=36)


# ------------------------------------------------------------------ pricing core
def test_put_call_parity_and_atm():
    S, K, s, T = -12.0, -15.0, 20.0, 0.1
    c = float(pricing.price(True, S, K, s, T))
    p = float(pricing.price(False, S, K, s, T))
    assert c - p == pytest.approx(S - K)
    atm = float(pricing.price(True, 0.0, 0.0, 20.0, 0.25))
    assert atm == pytest.approx(20.0 * 0.5 / np.sqrt(2 * np.pi))


def test_greeks_match_finite_differences():
    S, K, s, T, r = -10.0, -15.0, 18.0, 0.2, 0.04
    for call in (True, False):
        g = pricing.greeks(call, S, K, s, T, r)
        h = 1e-4
        up = float(pricing.price(call, S + h, K, s, T, r))
        dn = float(pricing.price(call, S - h, K, s, T, r))
        assert float(g["delta"]) == pytest.approx((up - dn) / (2 * h), abs=1e-5)
        assert float(g["gamma"]) == pytest.approx((up - 2 * float(g["price"]) + dn) / h**2, rel=1e-3)
        vu = float(pricing.price(call, S, K, s + h, T, r))
        vd = float(pricing.price(call, S, K, s - h, T, r))
        assert float(g["vega"]) == pytest.approx((vu - vd) / (2 * h), abs=1e-5)
        dt = 1e-5
        tu = float(pricing.price(call, S, K, s, T - dt, r))
        assert float(g["theta"]) == pytest.approx((tu - float(g["price"])) / dt / 365, rel=1e-3)


def test_iv_roundtrip_and_none_below_intrinsic():
    px = float(pricing.price(False, -10.0, -15.0, 17.0, 0.15, 0.03))
    assert pricing.implied_vol(False, px, -10.0, -15.0, 0.15, 0.03) == pytest.approx(17.0, rel=1e-6)
    assert pricing.implied_vol(True, 1.0, -10.0, -15.0, 0.15) is None


@pytest.mark.parametrize("delta", [-0.15, -0.5, 0.25, 0.6])
def test_strike_for_delta_roundtrip(delta):
    S, s, T, r = -8.0, 20.0, 0.12, 0.04
    K = pricing.strike_for_delta(delta, S, s, T, r)
    g = pricing.greeks(delta > 0, S, K, s, T, r)
    assert float(g["delta"]) == pytest.approx(delta, abs=1e-9)


def test_expired_option_is_intrinsic():
    assert float(pricing.price(True, -10.0, -15.0, 20.0, 0.0)) == 5.0
    assert float(pricing.price(False, -10.0, -15.0, 20.0, 0.0)) == 0.0


def test_sign_convention_duality():
    # a put on near-far at K is a call on far-near at -K
    a = float(pricing.price(False, -15.0, -20.0, 18.0, 0.1, 0.03))
    b = float(pricing.price(True, 15.0, 20.0, 18.0, 0.1, 0.03))
    assert a == pytest.approx(b)


# ------------------------------------------------------------------ spreads and vol sources
def _spread(**kw):
    d = dict(id="DZ/H", near_label="Dec26", far_label="Mar27", near_px=430.0, far_px=445.0,
             spread_vol=20.0, near_vol=0.22, far_vol=0.21, corr=0.95)
    d.update(kw)
    return book.Spread(**d)


def test_direct_vol_is_default_and_legs_is_opt_in():
    assert _spread().vol() == 20.0
    legs = _spread(vol_source="Legs")
    vn, vf = 0.22 * 430, 0.21 * 445
    assert legs.vol() == pytest.approx(np.sqrt(vn**2 + vf**2 - 2 * 0.95 * vn * vf))
    assert _spread(vol_source="Legs", corr=0.8).vol() > legs.vol()   # lower corr -> higher spread vol
    assert _spread(vol_source="Legs", near_vol=None).vol() == 20.0   # missing legs -> falls back to direct
    assert _spread().vol(dabs=3.0) == 23.0 and _spread().vol(mult=1.5) == 30.0


def test_level_conventions_and_override():
    assert _spread().level() == -15.0
    assert _spread(near_minus_far=False).level() == 15.0
    assert _spread(level_override=-12.5).level() == -12.5
    assert _spread(level_override=-12.5).level(ds=2.0) == -10.5


def test_clip_carry():
    sp = _spread(full_carry=30.0)
    assert sp.clip_carry(np.array([-50.0, -10.0, 5.0])).tolist() == [-30.0, -10.0, 5.0]
    inv = _spread(full_carry=30.0, near_minus_far=False)
    assert inv.clip_carry(np.array([50.0, 10.0])).tolist() == [30.0, 10.0]
    assert _spread().clip_carry(np.array([-99.0])).tolist() == [-99.0]


# ------------------------------------------------------------------ book
def test_leg_exposure_signs():
    sp = _spread()
    pos = [book.Position("DZ/H", "Put", 10, -15.0, EXP)]
    spreads = {sp.id: sp}
    b = book.value_book(pos, spreads, VAL, 50.0, 0.0)
    legs = book.leg_exposure(b, spreads).set_index("Futures month")["Net delta (contracts)"]
    assert legs["Dec26"] == pytest.approx(b["delta"].sum())
    assert legs["Mar27"] == pytest.approx(-b["delta"].sum())
    assert b["delta"].sum() < 0  # long put


def test_grid_matches_expiry_payoff_and_theta_sign():
    sp = _spread(spread_vol=12.0)
    spreads = {sp.id: sp}
    pos = [book.Position("DZ/H", "Put", 10, -15.0, EXP)]
    base = book.book_value(pos, spreads, VAL, 50.0, 0.0)
    g = book.grid_pnl(pos, spreads, VAL, 50.0, 0.0, [-30.0, 0.0, 30.0], 36)
    assert g["+36d"].tolist() == pytest.approx(g["At expiry"].tolist())
    assert g.loc[g["Spread move"] == -30, "At expiry"].iloc[0] == pytest.approx(10 * 50 * 30 - base)
    th = book.theta_path(pos, spreads, VAL, 50.0, 0.0, 30)
    assert th["Cumulative P&L ($)"].iloc[-1] < 0  # long option, spread unchanged -> bleeds


def test_long_cso_has_negative_theta_positive_gamma():
    sp = _spread()
    for kind in ("Call", "Put"):
        b = book.value_book([book.Position("DZ/H", kind, 1, -15.0, EXP)], {sp.id: sp}, VAL, 50.0, 0.0)
        assert b["theta"].iloc[0] < 0 and b["gamma"].iloc[0] > 0 and b["vega"].iloc[0] > 0


def test_risk_matrix_center_is_zero_and_vol_helps_long_option():
    sp = _spread()
    pos = [book.Position("DZ/H", "Put", 5, -20.0, EXP)]
    m = book.risk_matrix(pos, {sp.id: sp}, VAL, 50.0, 0.03, [-10.0, 0.0, 10.0], [-4.0, 0.0, 4.0])
    assert m.loc[0.0, 0.0] == pytest.approx(0.0)
    assert m.loc[4.0, 0.0] > 0 > m.loc[-4.0, 0.0]


def test_unrealized_vs_entry():
    sp = _spread()
    mark = float(pricing.price(False, -15.0, -20.0, 20.0, 36 / 365, 0.0))
    pos = [book.Position("DZ/H", "Put", 2, -20.0, EXP, entry=mark - 0.5),
           book.Position("DZ/H", "Spread future", 3, entry=-18.0)]
    u = book.unrealized(pos, {sp.id: sp}, VAL, 50.0, 0.0)
    assert u == pytest.approx(2 * 50 * 0.5 + 3 * 50 * (-15.0 + 18.0))


# ------------------------------------------------------------------ quotes -> vols
def test_quotes_roundtrip_and_skew_interpolation():
    sp = _spread()
    S = sp.level()
    vols = {-30.0: 24.0, -20.0: 21.0, -10.0: 19.0}
    T = 36 / 365
    rows = [dict(Spread="DZ/H", Type="Put" if k < S else "Call", Strike=k, Expiry=EXP,
                 Premium=float(pricing.price(k >= S, S, k, v, T, 0.0))) for k, v in vols.items()]
    iv = book.quotes_to_iv(pd.DataFrame(rows), {sp.id: sp}, VAL, 0.0)
    assert iv["IV"].tolist() == pytest.approx(list(vols.values()), rel=1e-5)
    assert book.interp_vol(iv, "DZ/H", EXP, -20.0, S) == pytest.approx(21.0, rel=1e-5)
    assert book.interp_vol(iv, "DZ/H", EXP, -25.0, S) == pytest.approx(22.5, rel=1e-4)   # linear between
    assert book.interp_vol(iv, "DZ/H", EXP, -50.0, S) == pytest.approx(24.0, rel=1e-5)   # flat extrapolation
    assert book.interp_vol(iv, "DZ/H", EXP + timedelta(days=7), -20.0, S) is None        # different expiry
    assert book.interp_vol(iv, "OTHER", EXP, -20.0, S) is None


def test_interp_prefers_otm_quote_at_a_strike():
    sp = _spread()
    S = sp.level()
    T = 36 / 365
    # call and put at -20 quoted at different vols; strike is below the spread, so the put is the OTM side
    rows = [dict(Spread="DZ/H", Type="Put", Strike=-20.0, Expiry=EXP, Premium=float(pricing.price(False, S, -20.0, 22.0, T, 0.0))),
            dict(Spread="DZ/H", Type="Call", Strike=-20.0, Expiry=EXP, Premium=float(pricing.price(True, S, -20.0, 30.0, T, 0.0)))]
    iv = book.quotes_to_iv(pd.DataFrame(rows), {sp.id: sp}, VAL, 0.0)
    assert book.interp_vol(iv, "DZ/H", EXP, -20.0, S) == pytest.approx(22.0, rel=1e-5)


def test_quotes_flag_bad_rows():
    sp = _spread()
    rows = [dict(Spread="DZ/H", Type="Put", Strike=-20.0, Expiry=EXP, Premium=0.0),
            dict(Spread="nope", Type="Put", Strike=-20.0, Expiry=EXP, Premium=1.0)]
    iv = book.quotes_to_iv(pd.DataFrame(rows), {sp.id: sp}, VAL, 0.0)
    assert iv["IV"].isna().all() and (iv["Note"] != "").all()


# ------------------------------------------------------------------ Monte Carlo and hedge
def test_monte_carlo_is_a_martingale_and_bounded_by_premium():
    sp = _spread(spread_vol=12.0)
    spreads = {sp.id: sp}
    pos = [book.Position("DZ/H", "Call", 10, -15.0, VAL + timedelta(days=60))]
    pnl = book.monte_carlo(pos, spreads, VAL, 50.0, 0.0, 30, n=40000)
    assert abs(pnl.mean()) < 4 * pnl.std() / np.sqrt(40000)
    assert pnl.min() >= -book.book_value(pos, spreads, VAL, 50.0, 0.0) - 1e-6


def test_carry_cap_truncates_the_carry_side():
    sp = _spread(spread_vol=40.0, full_carry=20.0)
    pos = [book.Position("DZ/H", "Spread future", 1)]
    free = book.monte_carlo(pos, {sp.id: sp}, VAL, 50.0, 0.0, 60, n=20000, carry_cap=False)
    capped = book.monte_carlo(pos, {sp.id: sp}, VAL, 50.0, 0.0, 60, n=20000, carry_cap=True)
    # worst case: spread bounded at -20 => loss from -15 is at most 5 points = $250
    assert capped.min() == pytest.approx(-250.0)
    assert free.min() < -250.0
    assert capped.max() == pytest.approx(free.max())


def test_put_hedge_cuts_downside_of_long_spread_exposure():
    sp = _spread(spread_vol=20.0)
    spreads = {sp.id: sp}
    exp_row = book.Position("DZ/H", "Spread future", 10, entry=-15.0, role=book.ROLE_EXPOSURE)
    hedge = book.Position("DZ/H", "Put", 10, -20.0, VAL + timedelta(days=60))
    unh = book.monte_carlo([exp_row], spreads, VAL, 50.0, 0.02, 30, n=30000)
    hed = book.monte_carlo([exp_row, hedge], spreads, VAL, 50.0, 0.02, 30, n=30000)
    su, sh = book.summarize(unh), book.summarize(hed)
    assert sh["5% (VaR 95)"] > su["5% (VaR 95)"]      # smaller 5th-percentile loss
    assert sh["Worst"] > su["Worst"]
    assert sh["Std dev"] < su["Std dev"]
    exp, rest = book.split_roles([exp_row, hedge])
    assert exp == [exp_row] and rest == [hedge]
