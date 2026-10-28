"""Strategy payoffs on a spread: the position you hold plus the options laid over it.

A strategy is a list of ``Leg``s. Every number here is P&L in dollars against what the legs cost: a spread
future against its entry level, an option against the premium paid (or received). Payoffs are piecewise
linear in the spread level, so break-evens, floors and caps come out exactly instead of from a sampled grid.
Adding a strategy later means adding legs, not changing any of this.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from . import pricing

FUTURE, CALL, PUT = "Spread future", "Call", "Put"
_EPS = 1e-9


@dataclass(frozen=True)
class Leg:
    kind: str                     # FUTURE | CALL | PUT
    qty: float                    # signed contracts: + long, - short
    strike: float                 # option strike, or the entry level of a spread future
    premium: float = 0.0          # points paid per unit when long (received when short); unused for futures
    expiry: date | None = None
    vol: float | None = None      # spread vol for this leg; None = the vol passed to value_on


def _intrinsic(leg: Leg, S):
    return np.maximum(S - leg.strike, 0.0) if leg.kind == CALL else np.maximum(leg.strike - S, 0.0)


def at_expiry(legs, S, mult: float) -> np.ndarray:
    """P&L ($) if the spread finishes at ``S``: futures vs entry, options at intrinsic value less premium."""
    S = np.asarray(S, dtype=float)
    total = np.zeros_like(S)
    for leg in legs:
        if leg.kind == FUTURE:
            total = total + leg.qty * mult * (S - leg.strike)
        else:
            total = total + leg.qty * mult * (_intrinsic(leg, S) - leg.premium)
    return total


def value_on(legs, S, mult: float, vol: float, rate: float, eval_date: date) -> np.ndarray:
    """P&L ($) if the spread is at ``S`` on ``eval_date``, options marked with the model (time value included)."""
    S = np.asarray(S, dtype=float)
    total = np.zeros_like(S)
    for leg in legs:
        if leg.kind == FUTURE:
            total = total + leg.qty * mult * (S - leg.strike)
            continue
        T = 0.0 if leg.expiry is None else max((leg.expiry - eval_date).days, 0) / 365.0
        sigma = leg.vol if leg.vol else vol
        price = pricing.price(leg.kind == CALL, S, leg.strike, sigma, T, rate)
        total = total + leg.qty * mult * (price - leg.premium)
    return total


def kinks(legs) -> list[float]:
    return sorted({leg.strike for leg in legs if leg.kind != FUTURE})


def tail_slopes(legs, mult: float) -> tuple[float, float]:
    """$ per point of spread as the spread heads to minus infinity and to plus infinity."""
    left = right = 0.0
    for leg in legs:
        q = leg.qty * mult
        if leg.kind == FUTURE:
            left += q
            right += q
        elif leg.kind == CALL:
            right += q
        else:
            left -= q
    return left, right


def breakevens(legs, mult: float) -> list[float]:
    """Spread levels where the payoff at expiry is zero, found exactly from the kinks."""
    ks = kinks(legs)
    left, right = tail_slopes(legs, mult)
    if not ks:
        f0 = float(at_expiry(legs, 0.0, mult))
        return [] if abs(right) < _EPS else [-f0 / right]
    vals = [float(at_expiry(legs, k, mult)) for k in ks]
    roots: list[float] = []
    if abs(left) > _EPS:
        r = ks[0] - vals[0] / left
        if r < ks[0] - _EPS:
            roots.append(r)
    for (a, b), (fa, fb) in zip(zip(ks, ks[1:]), zip(vals, vals[1:])):
        if fa * fb < 0:
            roots.append(a + (-fa) * (b - a) / (fb - fa))
    roots += [k for k, v in zip(ks, vals) if abs(v) < 1e-7]
    if abs(right) > _EPS:
        r = ks[-1] - vals[-1] / right
        if r > ks[-1] + _EPS:
            roots.append(r)
    out: list[float] = []
    for r in sorted(roots):
        if not out or abs(r - out[-1]) > 1e-7:
            out.append(r)
    return out


def summary(legs, mult: float) -> dict:
    """Premium paid, max loss and max gain at expiry (None = unlimited) and the break-evens."""
    ks = kinks(legs)
    left, right = tail_slopes(legs, mult)
    vals = [float(at_expiry(legs, k, mult)) for k in ks]
    unlimited_loss = left > _EPS or right < -_EPS
    unlimited_gain = left < -_EPS or right > _EPS
    if not vals:                      # futures only: linear, so unbounded one way or both
        vals = [0.0] if not (unlimited_loss or unlimited_gain) else []
    premium_paid = sum(leg.qty * mult * leg.premium for leg in legs if leg.kind != FUTURE)
    return {
        "premium_paid": premium_paid,
        "max_loss": None if unlimited_loss else (min(vals) if vals else 0.0),
        "max_gain": None if unlimited_gain else (max(vals) if vals else 0.0),
        "breakevens": breakevens(legs, mult),
    }


def grid(legs, mult: float, lo: float, hi: float, n: int = 401) -> np.ndarray:
    """An even grid over [lo, hi] that also contains every kink and break-even, so shading meets zero exactly."""
    xs = set(np.linspace(lo, hi, n).tolist())
    xs |= {k for k in kinks(legs) if lo <= k <= hi}
    xs |= {b for b in breakevens(legs, mult) if lo <= b <= hi}
    return np.array(sorted(xs))


def name_of(legs) -> str:
    """A trader's name for what the legs add up to: Put spread, Collar, Straddle, Risk reversal ...

    One or two options over a spread position. Two options of one kind on opposite sides are a put or call spread (a
    credit spread when the bought one is the nearer to the money). A put and a call are a straddle or strangle when
    both are bought, a short straddle or strangle when both are sold, and otherwise a collar when the bought one works
    against the position (a put over a long spread, a call over a short one) and a risk reversal when it does not.
    """
    opts = [leg for leg in legs if leg.kind != FUTURE]
    held = sum(leg.qty for leg in legs if leg.kind == FUTURE)
    if not opts:
        return "Spread position" if held else "Nothing"
    if len(opts) == 1:
        return f"{'Long' if opts[0].qty > 0 else 'Short'} {opts[0].kind.lower()}"
    if len(opts) > 2:
        return "Custom"
    a, b = opts
    if a.kind == b.kind:
        if (a.qty > 0) == (b.qty > 0):
            return f"Two {a.kind.lower()}s"
        bought, sold = (a, b) if a.qty > 0 else (b, a)
        if bought.strike == sold.strike:
            return "Offsetting options"
        debit = bought.strike > sold.strike if a.kind == PUT else bought.strike < sold.strike
        name = f"{'Ratio ' if abs(a.qty) != abs(b.qty) else ''}{a.kind.lower()} spread"
        name = name[0].upper() + name[1:]
        return name if debit else f"{name} (credit)"
    put, call = (a, b) if a.kind == PUT else (b, a)
    if put.qty > 0 and call.qty > 0:
        return "Straddle" if put.strike == call.strike else "Strangle"
    if put.qty < 0 and call.qty < 0:
        return "Short straddle" if put.strike == call.strike else "Short strangle"
    return "Collar" if held != 0 and (put.qty > 0) == (held > 0) else "Risk reversal"


def describe(legs) -> str:
    """The option legs in words: "buy 10 -20 put, sell 10 -10 call"."""
    return ", ".join(f"{'buy' if leg.qty > 0 else 'sell'} {abs(leg.qty):g} {leg.strike:+g} {leg.kind.lower()}"
                     for leg in legs if leg.kind != FUTURE)
