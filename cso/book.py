"""Spreads, positions, book valuation, scenarios, quotes -> vols, Monte Carlo.

Convention (CME grain/oilseed calendar spread options): spread = nearby futures
minus deferred futures.  A call is in the money when the spread is above the
strike; exercising a call leaves the holder long the nearby / short the deferred.
`Spread.near_minus_far=False` flips the sign for a far-minus-near convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from . import pricing

# $ per 1 point of quote per contract (cents/bu x 5,000 bu = $50 per cent)
COMMODITIES = {
    "Corn (¢/bu)": 50.0,
    "Soybeans (¢/bu)": 50.0,
    "Chicago wheat (¢/bu)": 50.0,
    "KC wheat (¢/bu)": 50.0,
    "Soybean meal ($/ton)": 100.0,
    "Soybean oil (¢/lb)": 600.0,
}

KINDS = ("Call", "Put", "Spread future")
ROLE_BOOK = "Hedge / trade"
ROLE_EXPOSURE = "Exposure"
ROLES = (ROLE_BOOK, ROLE_EXPOSURE)


@dataclass
class Spread:
    id: str
    near_label: str = ""
    far_label: str = ""
    near_px: float | None = None
    far_px: float | None = None
    spread_vol: float = 20.0            # normal vol of the spread, points/yr -- the primary input
    level_override: float | None = None  # spread level straight from the spread market
    full_carry: float | None = None      # points (positive); optional bound used by the Monte Carlo
    vol_source: str = "Direct"           # "Direct" | "Legs" (derive from leg vols + correlation)
    near_vol: float | None = None        # decimal lognormal; diagnostics / "Legs" source
    far_vol: float | None = None
    corr: float | None = None
    near_minus_far: bool = True

    def level(self, ds: float = 0.0) -> float:
        if self.level_override is not None:
            return self.level_override + ds
        d = self.near_px - self.far_px
        return (d if self.near_minus_far else -d) + ds

    def derived_vol(self, dvn=0.0, dvf=0.0, dcorr=0.0):
        """Spread vol implied by leg vols + correlation (None if inputs missing).

        Fragile by construction: it is the difference of two large, nearly equal
        numbers, so a 0.01 change in correlation moves it by ~25%.
        """
        if None in (self.near_px, self.far_px, self.near_vol, self.far_vol, self.corr):
            return None
        vn = max(self.near_vol + dvn, 0.0) * self.near_px
        vf = max(self.far_vol + dvf, 0.0) * self.far_px
        rho = float(np.clip(self.corr + dcorr, -0.999, 0.999))
        return float(np.sqrt(max(vn**2 + vf**2 - 2 * rho * vn * vf, 0.0)))

    def vol(self, dvn=0.0, dvf=0.0, dcorr=0.0, dabs=0.0, mult=1.0) -> float:
        """Effective spread vol in points/yr."""
        base = self.derived_vol(dvn, dvf, dcorr) if self.vol_source == "Legs" else None
        if base is None:
            base = self.spread_vol
        return max((base + dabs) * mult, 0.01)

    def leg_sign(self) -> tuple[float, float]:
        """(sign on near, sign on far) of a +1 spread position."""
        return (1.0, -1.0) if self.near_minus_far else (-1.0, 1.0)

    def clip_carry(self, S):
        """Bound the spread at full carry (no-op when full_carry is unset)."""
        if not self.full_carry:
            return S
        return np.maximum(S, -self.full_carry) if self.near_minus_far else np.minimum(S, self.full_carry)


@dataclass
class Position:
    spread_id: str
    kind: str                 # Call | Put | Spread future
    qty: float                # signed contracts (negative = short)
    strike: float = 0.0
    expiry: date | None = None
    vol: float | None = None  # per-row spread vol (skew)
    entry: float | None = None
    role: str = ROLE_BOOK


@dataclass
class Shock:
    ds: float = 0.0           # parallel move in every spread, points
    dabs: float = 0.0         # spread-vol shift, points
    dvn: float = 0.0          # near-leg vol shift (decimal) -- "Legs" vol source only
    dvf: float = 0.0
    dcorr: float = 0.0
    days: int = 0             # roll the valuation date forward


def _T(expiry: date, val_date: date, days: int) -> float:
    return max((expiry - val_date).days - days, 0) / 365.0


def value_book(positions, spreads: dict[str, Spread], val_date: date, mult: float,
               r: float, shock: Shock | None = None) -> pd.DataFrame:
    """One row per position: mark and signed Greeks in dollars."""
    sh = shock or Shock()
    rows = []
    for p in positions:
        sp = spreads[p.spread_id]
        S = sp.level(sh.ds)
        if p.kind == "Spread future":
            unreal = p.qty * mult * (S - p.entry) if p.entry is not None else np.nan
            rows.append(dict(role=p.role, spread=p.spread_id, kind=p.kind, qty=p.qty, strike=np.nan, expiry=None,
                             vol=np.nan, T=np.nan, premium=np.nan, value=p.qty * mult * S, unreal=unreal,
                             delta=p.qty, gamma=0.0, vega=0.0, theta=0.0))
            continue
        T = _T(p.expiry, val_date, sh.days)
        # a per-row vol (skew) ignores leg shocks but still takes the parallel shift
        if p.vol and p.vol > 0:
            sig = max(p.vol + sh.dabs, 0.01)
        else:
            sig = sp.vol(sh.dvn, sh.dvf, sh.dcorr, sh.dabs)
        g = pricing.greeks(p.kind == "Call", S, p.strike, sig, T, r)
        prem = float(g["price"])
        rows.append(dict(role=p.role, spread=p.spread_id, kind=p.kind, qty=p.qty, strike=p.strike,
                         expiry=p.expiry, vol=sig, T=T, premium=prem, value=p.qty * mult * prem,
                         unreal=p.qty * mult * (prem - p.entry) if p.entry is not None else np.nan,
                         delta=p.qty * float(g["delta"]), gamma=p.qty * float(g["gamma"]),
                         vega=p.qty * mult * float(g["vega"]), theta=p.qty * mult * float(g["theta"])))
    return pd.DataFrame(rows)


def book_value(positions, spreads, val_date, mult, r, shock=None) -> float:
    df = value_book(positions, spreads, val_date, mult, r, shock)
    return float(df["value"].sum()) if len(df) else 0.0


def leg_exposure(book: pd.DataFrame, spreads: dict[str, Spread]) -> pd.DataFrame:
    """Net futures contracts implied by spread delta, by contract month."""
    acc: dict[str, float] = {}
    for sid, d in book.groupby("spread")["delta"].sum().items():
        sp = spreads[sid]
        sn, sf = sp.leg_sign()
        acc[sp.near_label] = acc.get(sp.near_label, 0.0) + sn * d
        acc[sp.far_label] = acc.get(sp.far_label, 0.0) + sf * d
    return pd.DataFrame({"Futures month": list(acc), "Net delta (contracts)": list(acc.values())})


def expiry_value(positions, spreads, mult, ds: float) -> float:
    """Every option at intrinsic value (the spread held at the shifted level)."""
    tot = 0.0
    for p in positions:
        S = spreads[p.spread_id].level(ds)
        if p.kind == "Spread future":
            tot += p.qty * mult * S
        elif p.kind == "Call":
            tot += p.qty * mult * max(S - p.strike, 0.0)
        else:
            tot += p.qty * mult * max(p.strike - S, 0.0)
    return tot


def grid_pnl(positions, spreads, val_date, mult, r, shifts, horizon_days: int) -> pd.DataFrame:
    """Book P&L vs today's mark over a parallel spread move."""
    base = book_value(positions, spreads, val_date, mult, r)
    out = []
    for x in shifts:
        out.append({
            "Spread move": x,
            "Today": book_value(positions, spreads, val_date, mult, r, Shock(ds=x)) - base,
            f"+{horizon_days}d": book_value(positions, spreads, val_date, mult, r, Shock(ds=x, days=horizon_days)) - base,
            "At expiry": expiry_value(positions, spreads, mult, x) - base,
        })
    return pd.DataFrame(out)


def risk_matrix(positions, spreads, val_date, mult, r, shifts, dvols, days: int = 0) -> pd.DataFrame:
    """P&L over spread move (columns) x spread-vol shift (rows)."""
    base = book_value(positions, spreads, val_date, mult, r)
    data = [[book_value(positions, spreads, val_date, mult, r, Shock(ds=x, dabs=dv, days=days)) - base
             for x in shifts] for dv in dvols]
    return pd.DataFrame(data, index=pd.Index(dvols, name="Vol shift (pts)"),
                        columns=pd.Index(shifts, name="Spread move (pts)"))


def vol_table(positions, spreads, val_date, mult, r, dabs_list, twist_list) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = book_value(positions, spreads, val_date, mult, r)
    par = pd.DataFrame({"Spread vol shift (pts)": dabs_list,
                        "P&L ($)": [book_value(positions, spreads, val_date, mult, r, Shock(dabs=x)) - base
                                    for x in dabs_list]})
    rows = []
    for dn, dfv in twist_list:
        rows.append({"Near vol shift": dn, "Far vol shift": dfv,
                     "P&L ($)": book_value(positions, spreads, val_date, mult, r, Shock(dvn=dn, dvf=dfv)) - base})
    return par, pd.DataFrame(rows)


def theta_path(positions, spreads, val_date, mult, r, max_days: int) -> pd.DataFrame:
    base = book_value(positions, spreads, val_date, mult, r)
    prev = base
    rows = []
    for d in range(0, max_days + 1):
        v = book_value(positions, spreads, val_date, mult, r, Shock(days=d))
        rows.append({"Day": d, "Cumulative P&L ($)": v - base, "Daily theta ($)": v - prev if d else 0.0})
        prev = v
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- quotes -> vols / skew
def quotes_to_iv(quotes: pd.DataFrame, spreads: dict[str, Spread], val_date: date, r: float) -> pd.DataFrame:
    """Back out spread vol from market-maker premiums.  Adds Level, IV and Note columns."""
    rows = []
    for _, q in quotes.iterrows():
        sid, kind, k, exp, prem = q.get("Spread"), q.get("Type"), q.get("Strike"), q.get("Expiry"), q.get("Premium")
        row = dict(Spread=sid, Type=kind, Strike=k, Expiry=None, Premium=prem, Level=np.nan, IV=np.nan, Note="")
        if sid not in spreads or kind not in ("Call", "Put") or pd.isna(k) or pd.isna(exp) or pd.isna(prem):
            row["Note"] = "incomplete row"
            rows.append(row)
            continue
        exp = pd.to_datetime(exp).date()
        S = spreads[sid].level()
        T = max((exp - val_date).days, 0) / 365.0
        iv = pricing.implied_vol(kind == "Call", float(prem), S, float(k), T, r)
        row.update(Expiry=exp, Level=S, IV=np.nan if iv is None else iv,
                   Note="" if iv is not None else "at/below intrinsic or expired")
        rows.append(row)
    return pd.DataFrame(rows, columns=["Spread", "Type", "Strike", "Expiry", "Premium", "Level", "IV", "Note"])


def interp_vol(iv_df: pd.DataFrame, spread_id: str, expiry: date, strike: float, level: float) -> float | None:
    """Skew vol at `strike` from quotes on the same spread and expiry (flat beyond the ends).

    Out-of-the-money quotes are preferred where a strike has both a call and a put.
    """
    if iv_df is None or iv_df.empty:
        return None
    sub = iv_df[(iv_df["Spread"] == spread_id) & (iv_df["Expiry"] == expiry) & iv_df["IV"].notna()]
    if sub.empty:
        return None
    pts = {}
    for k, g in sub.groupby("Strike"):
        otm = g[(g["Type"] == "Call") == (k >= level)]
        pts[float(k)] = float((otm if len(otm) else g)["IV"].mean())
    ks = sorted(pts)
    return float(np.interp(strike, ks, [pts[k] for k in ks]))


# ----------------------------------------------------------------------- Monte Carlo
def monte_carlo(positions, spreads, val_date, mult, r, horizon_days: int, n: int = 20000,
                cross_corr: float = 0.5, volvol: float = 0.0, spot_vol_corr: float = 0.0,
                carry_cap: bool = False, seed: int = 7) -> np.ndarray:
    """P&L ($) distribution over `horizon_days`, measured from today's mark.

    Spread moves are normal at each spread's current vol (consistent with the
    Bachelier pricing; row-level skew vols affect repricing only).  Vol is
    re-marked at the horizon with a lognormal multiplier (`volvol`), optionally
    correlated with the spread move.  `carry_cap` bounds each spread at its
    `full_carry`.  The random draws depend only on `spreads` and `seed`, so
    sub-books run against the same `spreads` dict see identical paths.
    """
    rng = np.random.default_rng(seed)
    ids = list(spreads)
    k = len(ids)
    C = np.full((k, k), cross_corr)
    np.fill_diagonal(C, 1.0)
    z = rng.standard_normal((n, k)) @ np.linalg.cholesky(C).T
    w = rng.standard_normal((n, k))
    t = horizon_days / 365.0
    S, vm = {}, {}
    for i, sid in enumerate(ids):
        sp = spreads[sid]
        path = sp.level() + sp.vol() * np.sqrt(t) * z[:, i]
        S[sid] = sp.clip_carry(path) if carry_cap else path
        zv = spot_vol_corr * z[:, i] + np.sqrt(1 - spot_vol_corr**2) * w[:, i]
        vm[sid] = np.exp(volvol * np.sqrt(t) * zv - 0.5 * volvol**2 * t) if volvol > 0 else np.ones(n)
    base = book_value(positions, spreads, val_date, mult, r)
    pnl = np.full(n, -base)
    for p in positions:
        sp = spreads[p.spread_id]
        if p.kind == "Spread future":
            pnl += p.qty * mult * S[p.spread_id]
            continue
        T = _T(p.expiry, val_date, horizon_days)
        sig = (p.vol if p.vol and p.vol > 0 else sp.vol()) * vm[p.spread_id]
        pnl += p.qty * mult * pricing.price(p.kind == "Call", S[p.spread_id], p.strike, np.maximum(sig, 0.01), T, r)
    return pnl


def unrealized(positions, spreads, val_date, mult, r) -> float:
    """Today's mark minus entry cost, over rows that have an entry price."""
    b = value_book(positions, spreads, val_date, mult, r)
    return float(b["unreal"].fillna(0.0).sum()) if len(b) else 0.0


def summarize(pnl: np.ndarray) -> dict:
    q = np.percentile(pnl, [1, 5, 25, 50, 75, 95, 99])
    tail = pnl[pnl <= q[1]]
    return {"Mean": pnl.mean(), "Std dev": pnl.std(), "P(loss)": float((pnl < 0).mean()),
            "1%": q[0], "5% (VaR 95)": q[1], "25%": q[2], "Median": q[3], "75%": q[4], "95%": q[5], "99%": q[6],
            "CVaR 95": float(tail.mean()) if len(tail) else float(q[1]),
            "Worst": float(pnl.min()), "Best": float(pnl.max())}


def split_roles(positions):
    """(exposure rows, hedge/trade rows)."""
    exp = [p for p in positions if p.role == ROLE_EXPOSURE]
    return exp, [p for p in positions if p.role != ROLE_EXPOSURE]
