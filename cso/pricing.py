"""Bachelier (normal) pricing for options on a futures *spread*.

A calendar spread option (CSO) is an option on the price difference of two
futures months of the same commodity -- not a pair of options.  The spread can
be zero or negative (a carry market), so a lognormal model is the wrong tool;
the industry standard is the normal (Bachelier) model on the spread level with
a spread vol quoted in price units per sqrt(year) -- cents/bu for corn.

All functions are vectorised over `S` (and `sigma`) so the Monte Carlo can
revalue thousands of paths in one call.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

_EPS = 1e-12


def _parts(S, K, sigma, T, r):
    S, K, sigma = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (S, K, sigma)))
    T = float(max(T, 0.0))
    sd = sigma * np.sqrt(T)
    live = sd > _EPS
    d = np.where(live, (S - K) / np.where(live, sd, 1.0), 0.0)
    df = float(np.exp(-r * T))
    return S, K, sigma, T, sd, live, d, df


def price(is_call: bool, S, K, sigma, T: float, r: float = 0.0):
    """Premium in spread points (e.g. cents).  T in years."""
    S, K, sigma, T, sd, live, d, df = _parts(S, K, sigma, T, r)
    if is_call:
        v = (S - K) * norm.cdf(d) + sd * norm.pdf(d)
        intr = np.maximum(S - K, 0.0)
    else:
        v = (K - S) * norm.cdf(-d) + sd * norm.pdf(d)
        intr = np.maximum(K - S, 0.0)
    return df * np.where(live, v, intr)


def greeks(is_call: bool, S, K, sigma, T: float, r: float = 0.0) -> dict:
    """Per-unit Greeks.

    delta  : change in premium per +1 point in the spread
    gamma  : change in delta per +1 point in the spread
    vega   : change in premium per +1 point of spread vol (cents/yr)
    theta  : change in premium per calendar day (negative = decay)
    """
    S, K, sigma, T, sd, live, d, df = _parts(S, K, sigma, T, r)
    pdf = norm.pdf(d)
    p = price(is_call, S, K, sigma, T, r)
    if is_call:
        delta_live = df * norm.cdf(d)
        delta_dead = df * (S > K)
    else:
        delta_live = -df * norm.cdf(-d)
        delta_dead = -df * (S < K)
    sq = np.sqrt(T) if T > 0 else 1.0
    safe_sd = np.where(live, sd, 1.0)
    gamma = np.where(live, df * pdf / safe_sd, 0.0)
    vega = np.where(live, df * sq * pdf, 0.0)
    theta_yr = r * p - np.where(live, df * sigma * pdf / (2 * sq), 0.0)
    return {
        "price": p,
        "delta": np.where(live, delta_live, delta_dead),
        "gamma": gamma,
        "vega": vega,
        "theta": theta_yr / 365.0,
    }


def implied_vol(is_call: bool, premium: float, S: float, K: float, T: float, r: float = 0.0):
    """Spread vol that reproduces `premium`, or None if no solution exists."""
    if T <= 0:
        return None
    df = float(np.exp(-r * T))
    intrinsic = df * max((S - K) if is_call else (K - S), 0.0)
    if premium <= intrinsic + 1e-9:
        return None

    def f(s):
        return float(price(is_call, S, K, s, T, r)) - premium

    hi = max(1.0, abs(S - K) + 1.0)
    for _ in range(40):
        if f(hi) > 0:
            break
        hi *= 2
    else:
        return None
    return brentq(f, 1e-6, hi, xtol=1e-10)


def strike_for_delta(delta: float, S: float, sigma: float, T: float, r: float = 0.0):
    """Strike whose delta equals `delta` (>0 -> call, <0 -> put).

    This is how a "-15 delta" quote picks the strike; the delta is a strike
    selector, the strike is what the contract actually carries.
    """
    df = float(np.exp(-r * T))
    sd = sigma * np.sqrt(T)
    if delta == 0 or abs(delta) >= df:
        raise ValueError("|delta| must be between 0 and 1")
    if delta > 0:
        d = norm.ppf(delta / df)
    else:
        d = -norm.ppf(-delta / df)
    return float(S - d * sd)
