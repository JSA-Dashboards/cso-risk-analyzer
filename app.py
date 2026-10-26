import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cso import book, carry, massive, pricing, strategy
from cso.payoff_chart import payoff_chart
from cso.auth import require_password
from cso.disclosure import render_disclosure

st.set_page_config(page_title="CSO & spread risk analyzer", page_icon=":material/stacked_line_chart:", layout="wide")
# Hide the Streamlit Community Cloud viewer badge (creator avatar -> the account's other apps, plus the Streamlit
# logo). Cloud draws it on its OUTER page, outside this app's iframe, so CSS here can't reach it (checked live:
# still display:flex). The iframe is same-origin, so a script adds the rule to the parent document instead.
# Guarded by id; a no-op when run locally, where there is no outer page. Same snippet as the admin portal's.
st.html(
    """<script>
    (function () {
      try {
        var doc = window.parent.document;
        if (doc.getElementById('jsa-hide-cloud-badge')) return;
        var s = doc.createElement('style');
        s.id = 'jsa-hide-cloud-badge';
        s.textContent = "[class*='_profileContainer_'],[class*='_viewerBadge_']{display:none !important;}";
        doc.head.appendChild(s);
      } catch (e) {}
    })();
    </script>""",
    unsafe_allow_javascript=True,
)

require_password()      # nothing below renders until it passes (the admin portal pre-passes it)

TODAY = date.today()
NEAR_FAR, FAR_NEAR = "Near − far", "Far − near"
SPREAD_COLS = ["Spread", "Near", "Far", "Near price", "Far price", "Spread level", "Spread vol", "Full carry",
               "Convention", "Vol source", "Near vol %", "Far vol %", "Corr"]
LEG_COLS = ["Vol source", "Near vol %", "Far vol %", "Corr"]


def _eighth(x: float) -> float:
    return round(x * 8) / 8          # CSO premiums tick in 1/8 point


# --------------------------------------------------------------------------- state
LOGO = Path(__file__).resolve().parent / "assets" / "jsa_logo.png"

_SPREAD_DTYPES = {"Spread": "object", "Near": "object", "Far": "object", "Near price": "float64",
                  "Far price": "float64", "Spread level": "float64", "Spread vol": "float64",
                  "Full carry": "float64", "Convention": "object", "Vol source": "object",
                  "Near vol %": "float64", "Far vol %": "float64", "Corr": "float64"}
_POS_DTYPES = {"Spread": "object", "Role": "object", "Type": "object", "Qty": "float64", "Strike": "float64",
               "Expiry": "datetime64[ns]", "Vol override": "float64", "Entry": "float64"}
_QUOTE_DTYPES = {"Spread": "object", "Type": "object", "Strike": "float64", "Expiry": "datetime64[ns]",
                 "Premium": "float64"}


def _blank(dtypes: dict) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in dtypes.items()})


def _example_spreads() -> pd.DataFrame:
    return pd.DataFrame([{
        "Spread": "Dec/Mar", "Near": "Dec26", "Far": "Mar27", "Near price": 430.0, "Far price": 445.0,
        "Spread level": None, "Spread vol": 20.0, "Full carry": None, "Convention": NEAR_FAR,
        "Vol source": "Direct", "Near vol %": 22.0, "Far vol %": 21.0, "Corr": 0.97,
    }])


def _example_positions() -> pd.DataFrame:
    return pd.DataFrame([
        {"Spread": "Dec/Mar", "Role": book.ROLE_EXPOSURE, "Type": "Spread future", "Qty": 10.0, "Strike": None,
         "Expiry": None, "Vol override": None, "Entry": -15.0},
        {"Spread": "Dec/Mar", "Role": book.ROLE_BOOK, "Type": "Put", "Qty": 10.0, "Strike": -20.0,
         "Expiry": TODAY + timedelta(days=36), "Vol override": None, "Entry": None},
    ])


def _example_quotes() -> pd.DataFrame:
    t0, s0, ex = 36 / 365, -15.0, TODAY + timedelta(days=36)
    return pd.DataFrame([
        {"Spread": "Dec/Mar", "Type": t, "Strike": k, "Expiry": ex,
         "Premium": _eighth(float(pricing.price(t == "Call", s0, k, v, t0, 0.04)))}
        for t, k, v in (("Put", -30.0, 24.0), ("Put", -20.0, 21.0), ("Call", -10.0, 19.0))])


def _set_inputs(spreads_df, pos_df, quotes_df, example: bool) -> None:
    st.session_state.cso_spreads_df = spreads_df
    st.session_state.cso_pos_df = pos_df
    st.session_state.cso_quotes_df = quotes_df
    st.session_state.cso_spreads_ver = 0
    st.session_state.cso_reset_n = st.session_state.get("cso_reset_n", -1) + 1   # fresh editor keys
    st.session_state.cso_is_example = example
    st.session_state.cso_example_touched = False
    st.session_state.cso_fetch_note = None


def _load_blank() -> None:
    _set_inputs(_blank(_SPREAD_DTYPES), _blank(_POS_DTYPES), _blank(_QUOTE_DTYPES), False)


def _load_example() -> None:
    _set_inputs(_example_spreads(), _example_positions(), _example_quotes(), True)


if "cso_reset_n" not in st.session_state:
    _load_example()                 # first visit: open on a worked example, clearly labelled as one


def _apply_spread_vol(sid: str, vol: float) -> None:
    df = st.session_state["_cso_spreads_edited"].copy()
    m = df["Spread"] == sid
    df.loc[m, "Spread vol"] = round(vol, 2)
    df.loc[m, "Vol source"] = "Direct"
    st.session_state.cso_spreads_df = df
    st.session_state.cso_spreads_ver += 1      # new editor key => editor restarts from the updated frame
    st.session_state.cso_example_touched = True


@st.cache_data(ttl=120, show_spinner=False)
def _cached_quotes(tickers: tuple, _key: str) -> dict:
    """One Massive snapshot per two minutes for a set of contracts (the key is not part of the cache key)."""
    return massive.fetch_snapshot(_key, tickers)


@st.cache_data(ttl="6h", show_spinner="Loading the spread's recent history…")
def _cached_histories(tickers: tuple, _key: str) -> dict:
    """Daily settlements for each contract. Each call takes seconds and bars change once a day, so cache hard."""
    return massive.fetch_histories(_key, tickers)


@st.cache_data(ttl="1h", show_spinner=False)
def _cached_fed_funds(as_of: str, _key: str) -> tuple:
    """Front-month fed-funds future as a rate: (percent, ticker). It moves in basis points."""
    return massive.fed_funds_pct(_key, date.fromisoformat(as_of))


def _clock(ts: datetime) -> str:
    try:
        from zoneinfo import ZoneInfo
        return f"{ts.astimezone(ZoneInfo('America/Chicago')):%b %d %H:%M} CT"
    except Exception:                      # no tz database on this machine: say UTC instead
        return f"{ts:%b %d %H:%M} UTC"


def _fetch_prices() -> None:
    """Fill every spread's near and far price from Massive. The outcome goes to cso_fetch_note."""
    ss = st.session_state
    key = massive.api_key_from(st.secrets)
    if not key:
        ss.cso_fetch_note = ("warn", "No Massive API key is set for this app.")
        return
    product, today = massive.PRODUCT_CODES.get(ss["cso_commodity"]), ss["cso_val_date"]
    df = ss["_cso_spreads_edited"].copy()
    pairs = {i: (massive.ticker_for(product, _txt(r.get("Near")), today),
                 massive.ticker_for(product, _txt(r.get("Far")), today)) for i, r in df.iterrows()}
    wanted = tuple(sorted({t for pair in pairs.values() for t in pair if t}))
    if not wanted:
        ss.cso_fetch_note = ("warn", "Give each spread near and far contract labels like Dec26 and Mar27 so their "
                                     "prices can be looked up.")
        return
    try:
        quotes = _cached_quotes(wanted, key)
    except massive.MassiveError as exc:
        ss.cso_fetch_note = ("warn", f"Could not fetch prices: {exc}.")
        return
    used, missing, overridden = [], [], []
    for i, (near, far) in pairs.items():
        for col, ticker in (("Near price", near), ("Far price", far)):
            if ticker is None:
                continue
            q = quotes.get(ticker)
            if q is None:
                missing.append(ticker)
                continue
            df.at[i, col] = q.price
            used.append(q)
        if _num(df.at[i, "Spread level"]) is not None:
            overridden.append(_txt(df.at[i, "Spread"], "a spread"))
    if not used:
        ss.cso_fetch_note = ("warn", "No prices came back for " + ", ".join(sorted(set(missing))) + ".")
        return
    stamps = [q.as_of for q in used if q.as_of is not None]
    text = ("Prices from Massive" + (" (delayed)" if any(q.delayed for q in used) else "") +
            (f", last trade {_clock(max(stamps))}" if stamps else "") + ": " +
            ", ".join(f"{q.ticker} {q.price:g}" + ("" if q.source == "last" else f" ({q.source})")
                      for q in sorted({q.ticker: q for q in used}.values(), key=lambda q: q.ticker)) + ".")
    if missing:
        text += " No quote for " + ", ".join(sorted(set(missing))) + "."
    if overridden:
        text += (" Spread level is typed in for " + ", ".join(overridden) +
                 ", so these prices don't change it; clear that cell to use them.")
    ss.cso_fetch_note = ("warn" if (missing or overridden) else "ok", text)
    ss.cso_spreads_df = df
    ss.cso_spreads_ver += 1
    ss.cso_example_touched = True


def usd(x):
    return f"${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}"


def _num(x):
    return None if x is None or pd.isna(x) else float(x)


def _txt(x, default=""):
    return default if x is None or pd.isna(x) or str(x).strip() == "" else str(x).strip()


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Setup")
    commodity = st.selectbox("Commodity", list(book.COMMODITIES), key="cso_commodity")
    mult = book.COMMODITIES[commodity]
    st.caption(f"${mult:,.0f} per 1-point move per contract")
    val_date = st.date_input("Valuation date", TODAY, key="cso_val_date")
    rate = st.number_input("Interest rate %", 0.0, 15.0, 4.0, 0.25, key="cso_rate") / 100
    use_skew = st.toggle("Use quote-implied skew for positions", value=False, key="cso_use_skew",
                         help="Prices each option at the vol interpolated from the market-maker quotes (step 2) "
                              "for the same spread and expiry. Rows with their own Vol override keep it.")
    st.divider()
    st.caption("Inputs")
    st.button("Clear the example data", key="cso_clear_example", on_click=_load_blank, icon=":material/delete:",
              width="stretch", help="Empties all three input tables so you can enter your own.")
    st.button("Reload the example", key="cso_reload_example", on_click=_load_example, icon=":material/refresh:",
              width="stretch", help="Restores the illustrative example book.")

logo_col, title_col = st.columns([1, 7], vertical_alignment="center")
with logo_col:
    if LOGO.exists():
        st.image(str(LOGO), width=110)
with title_col:
    st.title("CSO & spread risk analyzer")
    st.caption("CME calendar spread options (spread = nearby − deferred, European exercise) priced with the "
               "normal (Bachelier) model on the spread level, so negative carry strikes work. Spread futures "
               "and any exposure being hedged are risked alongside.")
render_disclosure()

RN, SV = st.session_state.cso_reset_n, st.session_state.cso_spreads_ver
SPREADS_KEY, QUOTES_KEY, POS_KEY = f"cso_spreads_editor_{RN}_{SV}", f"cso_quotes_editor_{RN}", f"cso_pos_editor_{RN}"


def _edited(key) -> bool:
    state = st.session_state.get(key)
    return bool(state) and any(state.get(k) for k in ("edited_rows", "added_rows", "deleted_rows"))


if any(_edited(k) for k in (SPREADS_KEY, QUOTES_KEY, POS_KEY)):
    st.session_state.cso_example_touched = True
if st.session_state.cso_is_example:
    if not st.session_state.cso_example_touched:
        st.warning("**Example data.** The prices, volatilities, premiums and expiry dates below are illustrative "
                   "— not live market data. Replace them with your own, or clear them from the sidebar.",
                   icon=":material/science:")
    else:
        st.caption(":material/science: Any input you haven't changed from the example is illustrative, not "
                   "live market data.")

# --------------------------------------------------------------------------- 1 spreads
st.subheader("1 · Spreads")
_has_key = massive.api_key_from(st.secrets) is not None
fetch_col, note_col = st.columns([1.7, 4], vertical_alignment="center")
fetch_col.button("Fetch futures prices", key="cso_fetch_prices", on_click=_fetch_prices, width="stretch",
                 icon=":material/cloud_download:", disabled=not _has_key,
                 help="Looks up the near and far futures price of every spread (e.g. Dec26 and Mar27 become "
                      "ZCZ6 and ZCH7) from the market-data feed. Prices are delayed about 10 minutes.")
_note = st.session_state.get("cso_fetch_note")
if not _has_key:
    note_col.caption("To fetch prices, add MASSIVE_API_KEY to this app's Secrets.")
elif _note:
    (note_col.warning if _note[0] == "warn" else note_col.caption)(_note[1])
else:
    note_col.caption("Fills the near and far price of each spread from the market-data feed (delayed about "
                     "10 minutes) using the labels in the Near and Far columns.")
show_legs = st.toggle("Show leg vol / correlation columns", value=False, key="cso_show_legs",
                      help="Only needed for the 'Legs' vol source or the diagnostic below. Spread vol should "
                           "normally be typed in or backed out of quotes.")
spreads_df = st.data_editor(
    st.session_state.cso_spreads_df, num_rows="dynamic", key=SPREADS_KEY,
    width="stretch", column_order=[c for c in SPREAD_COLS if show_legs or c not in LEG_COLS],
    column_config={
        "Near price": st.column_config.NumberColumn(format="%.2f"),
        "Far price": st.column_config.NumberColumn(format="%.2f"),
        "Spread level": st.column_config.NumberColumn(
            format="%.3f", help="Optional: the spread's own market price. Blank = near price − far price."),
        "Spread vol": st.column_config.NumberColumn(
            format="%.2f", help="Normal vol of the spread in points/yr (¢/bu for corn). Type it in, or use "
                                "the quotes in step 2 to back it out."),
        "Full carry": st.column_config.NumberColumn(
            format="%.1f", help="Optional bound in points (e.g. 35). Lets the Monte Carlo stop the spread "
                                "at full carry; pricing and Greeks ignore it."),
        "Convention": st.column_config.SelectboxColumn(options=[NEAR_FAR, FAR_NEAR],
                                                       help="CME grain CSOs use near − far."),
        "Vol source": st.column_config.SelectboxColumn(
            options=["Direct", "Legs"], help="Legs = derive from leg vols and correlation (very sensitive to "
                                             "correlation — see the diagnostic below)."),
        "Near vol %": st.column_config.NumberColumn(format="%.1f"),
        "Far vol %": st.column_config.NumberColumn(format="%.1f"),
        "Corr": st.column_config.NumberColumn(min_value=-0.99, max_value=0.999, format="%.3f"),
    },
)
st.session_state["_cso_spreads_edited"] = spreads_df

spreads: dict[str, book.Spread] = {}
for _, r in spreads_df.iterrows():
    sid = _txt(r.get("Spread"))
    if not sid:
        continue
    lvl, npx, fpx = _num(r.get("Spread level")), _num(r.get("Near price")), _num(r.get("Far price"))
    if lvl is None and (npx is None or fpx is None):
        st.warning(f"{sid}: enter near and far prices, or a spread level.")
        continue
    if sid in spreads:
        st.warning(f"Duplicate spread name '{sid}' — the last row wins.")
    nv, fv = _num(r.get("Near vol %")), _num(r.get("Far vol %"))
    vol_in, src = _num(r.get("Spread vol")), _txt(r.get("Vol source"), "Direct")
    if vol_in is None and not (src == "Legs" and None not in (npx, fpx, nv, fv, _num(r.get("Corr")))):
        st.warning(f"{sid}: no spread vol entered, so a provisional 20.00 is used for the numbers below. Enter "
                   "your own (points per year), or back one out of quoted premiums in step 2.")
    spreads[sid] = book.Spread(
        id=sid, near_label=_txt(r.get("Near"), "Near"), far_label=_txt(r.get("Far"), "Far"),
        near_px=npx, far_px=fpx, spread_vol=vol_in or 20.0, level_override=lvl,
        full_carry=_num(r.get("Full carry")), vol_source=_txt(r.get("Vol source"), "Direct"),
        near_vol=None if nv is None else nv / 100, far_vol=None if fv is None else fv / 100,
        corr=_num(r.get("Corr")), near_minus_far=_txt(r.get("Convention"), NEAR_FAR) != FAR_NEAR)
if not spreads:
    st.info("Add a spread to begin: a name, the near and far futures prices (or the spread's own price) and a "
            "spread vol. Or reload the example from the sidebar.")
    st.stop()

summary = []
for s in spreads.values():
    dv = s.derived_vol()
    sens = None if dv is None else (s.derived_vol(dcorr=-0.01) - s.derived_vol(dcorr=0.01)) / 2
    summary.append({"Spread": s.id, "Level": s.level(), "Spread vol used": s.vol(),
                    "Legs-implied vol": dv, "Δ per ±0.01 corr": sens})
st.dataframe(pd.DataFrame(summary), hide_index=True, width="content", column_config={
    "Level": st.column_config.NumberColumn(format="%.2f"),
    "Spread vol used": st.column_config.NumberColumn(format="%.2f"),
    "Legs-implied vol": st.column_config.NumberColumn(format="%.2f", help="Diagnostic only unless Vol source = Legs"),
    "Δ per ±0.01 corr": st.column_config.NumberColumn(format="±%.2f", help="How far the legs-implied vol moves for "
                                                                           "a 0.01 change in correlation")})
for row in summary:
    d, u = row["Legs-implied vol"], row["Spread vol used"]
    if d and abs(d - u) / u > 0.25 and spreads[row["Spread"]].vol_source == "Direct":
        st.caption(f":material/info: {row['Spread']}: leg vols + correlation imply {d:.1f}, vs {u:.1f} used — a "
                   f"±0.01 move in correlation shifts that by about {row['Δ per ±0.01 corr']:.1f}, so trust "
                   "quotes or realized spread vol over this.")

# --------------------------------------------------------------------------- 2 quotes
st.subheader("2 · Market-maker quotes (optional)")
with st.expander("Back out spread vol and skew from quoted premiums", expanded=False):
    st.caption("Enter the premiums you're being quoted. Each is converted to an implied spread vol at today's "
               "spread level. Strikes in 5-point steps, premiums in ⅛ points — and because of that tick, far "
               "out-of-the-money quotes (a few eighths) give coarse vols, off by a point or two.")
    quotes_df = st.data_editor(
        st.session_state.cso_quotes_df, num_rows="dynamic", key=QUOTES_KEY, width="stretch",
        column_config={
            "Spread": st.column_config.SelectboxColumn(options=list(spreads), required=True),
            "Type": st.column_config.SelectboxColumn(options=["Call", "Put"], required=True),
            "Strike": st.column_config.NumberColumn(format="%.1f", step=1.0),
            "Expiry": st.column_config.DateColumn(),
            "Premium": st.column_config.NumberColumn("Premium (pts)", format="%.3f", step=0.125)})
    iv_df = book.quotes_to_iv(quotes_df, spreads, val_date, rate)
    if len(iv_df) and iv_df["IV"].notna().any():
        c1, c2 = st.columns([3, 2])
        c1.dataframe(iv_df, hide_index=True, width="stretch", column_config={
            "IV": st.column_config.NumberColumn("Implied vol", format="%.2f"),
            "Level": st.column_config.NumberColumn(format="%.2f")})
        sm = iv_df.dropna(subset=["IV"]).assign(Expiry=lambda d: d["Expiry"].astype(str))
        c2.altair_chart(alt.Chart(sm).mark_line(point=True).encode(
            x=alt.X("Strike:Q", scale=alt.Scale(zero=False)), y=alt.Y("IV:Q", scale=alt.Scale(zero=False),
                                                                      title="Implied spread vol"),
            color="Expiry:N", tooltip=["Spread", "Type", "Strike", "IV"]).properties(height=240), width="stretch")
        suggestions = []
        for (sid, ex), g in sm.groupby(["Spread", "Expiry"]):
            exd = pd.to_datetime(ex).date()
            v = book.interp_vol(iv_df, sid, exd, spreads[sid].level(), spreads[sid].level())
            if v:
                suggestions.append((sid, exd, v))
        if suggestions:
            st.markdown("**At-the-money implied vol** — set it as the spread vol:")
            bc = st.columns(min(len(suggestions), 4))
            for i, (sid, exd, v) in enumerate(suggestions):
                bc[i % len(bc)].button(f"{sid} · {exd:%b %d}: {v:.2f}", key=f"cso_apply_{sid}_{exd}",
                                       on_click=_apply_spread_vol, args=(sid, v), width="stretch",
                                       help="Writes this vol into the Spread vol column and switches the "
                                            "spread to the Direct source.")
    else:
        st.info("No usable quotes yet — premiums must exceed intrinsic value and expiries must be after the "
                "valuation date.")

# --------------------------------------------------------------------------- 3 positions
st.subheader("3 · Positions")
st.caption("Tag what is being **hedged** as *Exposure* (e.g. a long spread future standing in for a "
           "storage or carry position) and the options or spread futures held against it as *Hedge / trade*. "
           "Leave everything as Hedge / trade for a standalone view.")
pos_df = st.data_editor(
    st.session_state.cso_pos_df, num_rows="dynamic", key=POS_KEY, width="stretch",
    column_config={
        "Spread": st.column_config.SelectboxColumn(options=list(spreads), required=True),
        "Role": st.column_config.SelectboxColumn(options=list(book.ROLES)),
        "Type": st.column_config.SelectboxColumn(options=list(book.KINDS), required=True),
        "Qty": st.column_config.NumberColumn(help="Signed contracts: negative = short", format="%.0f"),
        "Strike": st.column_config.NumberColumn(help="Spread strike in points; negative is fine", format="%.1f"),
        "Expiry": st.column_config.DateColumn(help="CSO expiry (European; exercised only on this day)"),
        "Vol override": st.column_config.NumberColumn(help="This row's spread vol (pts/yr) — your own skew"),
        "Entry": st.column_config.NumberColumn(
            help="Option: premium paid/received (pts). Spread future: entry level. Used for P&L vs cost."),
    })
positions, problems = [], []
for i, r in pos_df.iterrows():
    sid, kind = _txt(r.get("Spread")), _txt(r.get("Type"))
    if not sid and not kind and pd.isna(r.get("Qty")):
        continue
    if sid not in spreads or kind not in book.KINDS or pd.isna(r.get("Qty")):
        problems.append(f"Row {i + 1}: needs a known spread, a type and a quantity")
        continue
    if kind != "Spread future" and (pd.isna(r.get("Strike")) or pd.isna(r.get("Expiry"))):
        problems.append(f"Row {i + 1}: options need a strike and expiry")
        continue
    expiry = pd.to_datetime(r["Expiry"]).date() if pd.notna(r.get("Expiry")) else None
    vol = _num(r.get("Vol override"))
    if kind != "Spread future" and vol is None and use_skew:
        vol = book.interp_vol(iv_df, sid, expiry, float(r["Strike"]), spreads[sid].level())
    positions.append(book.Position(
        spread_id=sid, kind=kind, qty=float(r["Qty"]), strike=_num(r.get("Strike")) or 0.0, expiry=expiry,
        vol=vol, entry=_num(r.get("Entry")), role=_txt(r.get("Role"), book.ROLE_BOOK)))
for p in problems:
    st.warning(p)
if not positions:
    st.info("Add at least one position in step 3 for the portfolio tabs. The Strategy builder works without one.")
if any(p.expiry and p.expiry <= val_date for p in positions):
    st.warning("Some options are at or past expiry on the valuation date — they're valued at intrinsic.")
used = {p.spread_id: spreads[p.spread_id] for p in positions}
exp_dates = [p.expiry for p in positions if p.expiry and p.expiry > val_date]
max_h = max((min(exp_dates) - val_date).days if exp_dates else 30, 1)
if positions:
    st.download_button("Download book (CSV)", pos_df.to_csv(index=False), "cso_book.csv", mime="text/csv")


def _heat(df):
    vmax = float(np.abs(df.values).max()) or 1.0

    def css(v):
        t = max(-1.0, min(1.0, v / vmax))
        end = (99, 190, 123) if t >= 0 else (248, 105, 107)
        c = tuple(int(255 + (end[i] - 255) * abs(t)) for i in range(3))
        return f"background-color: rgb{c}; color: #1a1a1a"
    return df.map(css)


# --------------------------------------------------------------------------- tabs
for _k in [k for k in st.session_state if str(k).startswith("cso_st_")]:
    st.session_state[_k] = st.session_state[_k]      # lazy tabs drop a closed tab's widgets; this keeps the builder's inputs

t_strategy, t_book, t_grid, t_vol, t_theta, t_mc, t_hedge, t_tools = st.tabs(
    ["Strategy builder", "Book & Greeks", "Spread moves", "Vol shocks", "Theta burn", "Monte Carlo",
     "Hedge effectiveness", "Strike & IV tools"], on_change="rerun")


def _no_book() -> bool:
    if positions:
        return False
    st.info("Add at least one position in step 3 to see this. The Strategy builder and the IV tools work without one.")
    return True


@st.fragment
def tab_strategy():
    sid = st.selectbox("Spread", list(spreads), key="cso_st_spread")
    sp = spreads[sid]
    level = sp.level()
    key = massive.api_key_from(st.secrets)
    product = massive.PRODUCT_CODES.get(commodity)
    near_t = massive.ticker_for(product, sp.near_label, val_date)
    far_t = massive.ticker_for(product, sp.far_label, val_date)

    quotes = {}
    if key and near_t and far_t:
        try:
            quotes = _cached_quotes((near_t, far_t), key)
        except massive.MassiveError:
            quotes = {}

    # ---- where the spread sits against full carry (the Cost of Carry calculator's math)
    near_px = sp.near_px if sp.near_px is not None else (quotes[near_t].price if near_t in quotes else None)
    near_exp, far_exp = (quotes[near_t].expiry if near_t in quotes else None), (quotes[far_t].expiry if far_t in quotes else None)
    mkt = carry.MARKETS[product]
    days_default = (far_exp - near_exp).days if near_exp and far_exp and far_exp > near_exp else 90
    storage_default = (carry.storage_rate(product, near_exp, far_exp) if near_exp and far_exp else mkt[0]) * mkt[1]
    rate_default, rate_note = carry.FALLBACK_ANNUAL_RATE_PCT, f"fed funds + {carry.FED_FUNDS_SPREAD_PCT}% (live read unavailable: fallback)"
    if key:
        try:
            ff, ff_ticker = _cached_fed_funds(val_date.isoformat(), key)
            rate_default = round(ff + carry.FED_FUNDS_SPREAD_PCT, 3)
            rate_note = f"{ff:.2f}% fed funds ({ff_ticker}) + {carry.FED_FUNDS_SPREAD_PCT}%"
        except massive.MassiveError:
            pass
    with st.expander("Carry assumptions"):
        a1, a2, a3 = st.columns(3)
        days = a1.number_input("Days between the two expiries", 1, 600, int(days_default), key=f"cso_st_days_{sid}_{days_default}")
        storage = a2.number_input(f"Storage ({mkt[2]})", 0.0, 50.0, float(round(storage_default, 5)), format="%.5f",
                                  key=f"cso_st_storage_{sid}_{round(storage_default, 5)}",
                                  help="CME maximum storage rate carried under across the spread's window.")
        rate_pct = a3.number_input("Interest rate %", 0.0, 30.0, float(rate_default), 0.05, key=f"cso_st_rate_{sid}_{rate_default}",
                                   help=rate_note)
    spread_nf = level if sp.near_minus_far else -level            # the calculator's convention: near - far
    carry_limit, fc, pct = None, None, None
    if near_px:
        fc = carry.full_carry(near_px, int(days), storage / mkt[1], rate_pct / 100, mkt[1])
        pct = carry.pct_of_full_carry(spread_nf, fc["full"])
        carry_limit = -fc["full"] if sp.near_minus_far else fc["full"]

    top = st.columns(4)
    top[0].metric("Spread level", f"{level:+.2f}", help=f"{sp.near_label} − {sp.far_label}" if sp.near_minus_far else f"{sp.far_label} − {sp.near_label}")
    top[1].metric("Full carry", "—" if fc is None else f"{fc['full']:.2f}",
                  help=None if fc is None else f"storage {fc['storage']:.2f} + interest {fc['interest']:.2f} points ({rate_note})")
    top[2].metric("Spread vol (pts/yr)", f"{sp.vol():.1f}")
    with top[3]:
        st.caption("Share of full carry")
        if pct is None:
            st.markdown("—")
        else:
            st.markdown(f":{carry.BUCKET_COLORS[carry.bucket(pct)]}-badge[{pct:.0%} of full carry]")

    # ---- the range the spread has actually traded in
    band = None
    if key and near_t and far_t:
        hs = _cached_histories((near_t, far_t), key)
        hist = massive.spread_history(hs.get(near_t, pd.Series(dtype=float)), hs.get(far_t, pd.Series(dtype=float)), 365, val_date)
        if not hist.empty:
            hist = hist if sp.near_minus_far else -hist
            band = (float(hist.min()), float(hist.max()))
            st.caption(f"Last 12 months ({len(hist)} sessions): low {hist.min():+.2f} on {hist.idxmin():%b %d, %Y}, "
                       f"high {hist.max():+.2f} on {hist.idxmax():%b %d, %Y}. Shaded blue on the chart.")

    # ---- the position and the option laid over it
    st.markdown("#### Position and protection")
    sn, sf = sp.leg_sign()
    legs_txt = f"{'buy' if sn > 0 else 'sell'} {sp.near_label} / {'buy' if sf > 0 else 'sell'} {sp.far_label}"
    p1, p2, p3, p4 = st.columns([1.4, 1, 1, 1])
    base = p1.segmented_control("Position being protected", ["Long spread", "Short spread", "No position"],
                                default="Long spread", key="cso_st_base",
                                help=f"Long spread = {legs_txt}. A put protects a long spread against the spread falling; "
                                     "a call protects a short one.") or "No position"
    qty = p2.number_input("Contracts", 1, 10000, 10, key="cso_st_qty")
    entry = p3.number_input("Entry level", value=float(round(level, 2)), step=0.25, format="%.2f", key=f"cso_st_entry_{sid}",
                            help="Where the spread position was put on. The default is today's level.")
    opt = p4.segmented_control("Bought option", ["Put", "Call"], default="Put", key="cso_st_opt") or "Put"

    o1, o2, o3, o4 = st.columns(4)
    default_k = float(np.floor((level - 5) / 5) * 5) if opt == "Put" else float(np.ceil((level + 5) / 5) * 5)
    strike = o1.number_input("Strike", value=default_k, step=1.0, format="%.1f", key=f"cso_st_strike_{opt}_{sid}",
                             help="Spread strike in points. CSO strikes are 5 points apart (1 for the nearest spreads).")
    opt_qty = o2.number_input("Option contracts", 1, 10000, int(qty), key=f"cso_st_optqty_{qty}")
    expiry = o3.date_input("CSO expiry", val_date + timedelta(days=36), key="cso_st_expiry",
                           help="CSOs expire before the nearby future does: enter the series' actual date.")
    vol = o4.number_input("Spread vol (pts/yr)", 0.1, 500.0, float(round(sp.vol(), 2)), 0.5,
                          key=f"cso_st_vol_{sid}_{round(sp.vol(), 2)}")
    T = max((expiry - val_date).days, 0) / 365.0
    model_prem = float(pricing.price(opt == "Call", level, strike, vol, T, rate))
    q1, q2 = st.columns([1, 2])
    psrc = q1.segmented_control("Premium", ["Model", "My quote"], default="Model", key="cso_st_psrc") or "Model"
    if psrc == "My quote":
        premium = q2.number_input("Premium paid (points)", 0.0, 500.0, float(_eighth(model_prem)), 0.125, format="%.3f",
                                  key=f"cso_st_prem_{opt}_{strike}_{round(model_prem, 3)}")
    else:
        premium = model_prem
        q2.caption(f"Model price {model_prem:.3f} points = {usd(model_prem * mult * opt_qty)} for {opt_qty} contract(s).")

    legs = []
    if base != "No position":
        legs.append(strategy.Leg(strategy.FUTURE, qty if base == "Long spread" else -qty, entry))
    held = list(legs)
    legs.append(strategy.Leg(strategy.PUT if opt == "Put" else strategy.CALL, opt_qty, strike, premium, expiry, vol))

    # ---- payoff: green where it gains, red where it loses
    half = st.number_input("Chart half-width (points)", 5.0, 300.0, 40.0, 5.0, key="cso_st_w")
    lo, hi = min(level - half, strike - 10), max(level + half, strike + 10)
    S = strategy.grid(legs, mult, lo, hi)
    combined = strategy.at_expiry(legs, S, mult)
    chart = payoff_chart(
        S, combined, strategy.at_expiry(held, S, mult), strategy.value_on(legs, S, mult, vol, rate, val_date),
        current=level, strikes=[(strike, f"strike {strike:+g}")], breakevens=strategy.breakevens(legs, mult),
        carry_limit=carry_limit if carry_limit is not None and lo <= carry_limit <= hi else None, band=band)
    st.altair_chart(chart, width="stretch")
    st.caption("**Green** = spread levels where the strategy makes money at expiry, **red** = where it loses. Solid line: "
               "at expiry. Thin blue line: today's model value. Dashed grey: the position without the option.")

    sm = strategy.summary(legs, mult)
    m = st.columns(5)
    m[0].metric("Premium paid", usd(sm["premium_paid"]), help="What the option costs up front, before any move.")
    m[1].metric("Max loss", "Unlimited" if sm["max_loss"] is None else usd(sm["max_loss"]))
    m[2].metric("Max gain", "Unlimited" if sm["max_gain"] is None else usd(sm["max_gain"]),
                help="In theory. In practice the spread can't widen far past full carry; see the next figure.")
    m[3].metric("Break-even", ", ".join(f"{b:+.2f}" for b in sm["breakevens"]) or "—")
    if carry_limit is not None:
        m[4].metric(f"At full carry ({carry_limit:+.1f})", usd(float(strategy.at_expiry(legs, carry_limit, mult))),
                help="P&L at expiry if the spread widens all the way to full carry, the practical limit of the downside.")

    # ---- the same thing as a table, shaded the same way
    step = max(1.0, float(round(half / 8)))
    levels = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
    tbl = pd.DataFrame({
        "Spread level": levels, "Move": levels - level,
        "Position only": strategy.at_expiry(held, levels, mult),
        "Option": strategy.at_expiry(legs[len(held):], levels, mult),
        "Combined at expiry": strategy.at_expiry(legs, levels, mult),
        "Combined today": strategy.value_on(legs, levels, mult, vol, rate, val_date)})
    money = ["Position only", "Option", "Combined at expiry", "Combined today"]
    st.dataframe(tbl.style.apply(_heat, subset=money[:2], axis=None).apply(_heat, subset=money[2:], axis=None)
                 .format({"Spread level": "{:+.1f}", "Move": "{:+.1f}", **{c: usd for c in money}}),
                 hide_index=True, width="stretch")      # the two groups are shaded on their own scales


@st.fragment
def tab_book():
    if _no_book():
        return
    b = book.value_book(positions, used, val_date, mult, rate)
    tot = b[["value", "delta", "gamma", "vega", "theta"]].sum()
    c = st.columns(6)
    c[0].metric("Book mark", usd(tot["value"]))
    c[1].metric("Net delta (contracts)", f"{tot['delta']:+.2f}", help="Spread contracts per +1 point in the spread")
    c[2].metric("Gamma", f"{tot['gamma']:+.3f}", help="Contracts of delta per +1 point")
    c[3].metric("Vega", usd(tot["vega"]), help="$ per +1 point of spread vol")
    c[4].metric("Theta / day", usd(tot["theta"]))
    if b["unreal"].notna().any():
        c[5].metric("Unrealized vs entry", usd(b["unreal"].sum()), help="Today's mark minus entry, rows with an Entry")
    lv = b["spread"].map(lambda s: used[s].level())
    b["otm"] = np.where(b["kind"] == "Put", lv - b["strike"], np.where(b["kind"] == "Call", b["strike"] - lv, np.nan))
    cols = ["role", "spread", "kind", "qty", "strike", "otm", "expiry", "premium", "vol", "value", "unreal",
            "delta", "gamma", "vega", "theta"]
    st.dataframe(b[cols], hide_index=True, width="stretch", column_config={
        "role": "Role", "spread": "Spread", "kind": "Type", "qty": st.column_config.NumberColumn("Qty", format="%.0f"),
        "strike": st.column_config.NumberColumn("Strike", format="%.1f"),
        "otm": st.column_config.NumberColumn("OTM by", format="%.1f", help="Points out of the money (negative = in)"),
        "expiry": "Expiry",
        "value": st.column_config.NumberColumn("Mark $", format="$%.0f"),
        "unreal": st.column_config.NumberColumn("vs entry $", format="$%.0f"),
        "premium": st.column_config.NumberColumn("Premium", format="%.3f"),
        "vol": st.column_config.NumberColumn("Spread vol", format="%.2f"),
        "delta": st.column_config.NumberColumn("Delta", format="%.3f"),
        "gamma": st.column_config.NumberColumn("Gamma", format="%.4f"),
        "vega": st.column_config.NumberColumn("Vega $", format="$%.0f"),
        "theta": st.column_config.NumberColumn("Theta $/d", format="$%.1f")})
    st.markdown("**Delta in futures terms** — what the spread delta means for each contract month "
                "(and what to trade to flatten it)")
    st.dataframe(book.leg_exposure(b, used), hide_index=True, width="content")
    st.caption("A CSO has no flat-price delta in this model: its risk is the spread level, spread vol and time. "
               "A long option has positive gamma/vega and negative theta; a short option the reverse.")


@st.fragment
def tab_grid():
    if _no_book():
        return
    c = st.columns(3)
    width = c[0].number_input("Move range ± (points)", 5.0, 500.0, 40.0, 5.0, key="cso_grid_w")
    step = c[1].number_input("Step", 0.5, 50.0, 2.0, 0.5, key="cso_grid_s")
    h = c[2].slider("Horizon (days)", 1, max_h, min(7, max_h), key="cso_grid_h")
    shifts = np.arange(-width, width + step / 2, step)
    g = book.grid_pnl(positions, used, val_date, mult, rate, shifts, h)
    long = g.melt("Spread move", var_name="Revalued", value_name="P&L ($)")
    st.altair_chart(alt.Chart(long).mark_line().encode(
        x="Spread move:Q", y="P&L ($):Q", color="Revalued:N").properties(height=380), width="stretch")
    st.caption("Parallel move applied to every spread. “At expiry” = each option at intrinsic value.")
    st.dataframe(g, hide_index=True, width="stretch")


@st.fragment
def tab_vol():
    if _no_book():
        return
    c = st.columns(3)
    rng = c[0].number_input("Spread vol shift ± (points)", 1.0, 100.0, 6.0, 1.0, key="cso_vol_rng")
    mv = c[1].number_input("Spread move ± (points)", 5.0, 200.0, 30.0, 5.0, key="cso_vol_mv")
    h = c[2].slider("Days forward", 0, max_h, 0, key="cso_vol_days")
    dvols = [float(x) for x in np.linspace(-rng, rng, 7)]
    shifts = [float(x) for x in np.linspace(-mv, mv, 7)]
    m = book.risk_matrix(positions, used, val_date, mult, rate, shifts, dvols[::-1], h)
    st.markdown("**Spread move × spread-vol shift** — P&L vs today ($)")
    m.columns = [f"{x:+.0f}" for x in m.columns]
    m.index = [f"{x:+.1f}" for x in m.index]
    st.dataframe(m.style.apply(_heat, axis=None).format(usd), width="stretch")
    par, tw = book.vol_table(positions, used, val_date, mult, rate, dvols, [])
    st.markdown("**Spread vol, parallel**")
    st.altair_chart(alt.Chart(par).mark_bar().encode(x="Spread vol shift (pts):O", y="P&L ($):Q").properties(height=240),
                    width="stretch")
    if any(s.vol_source == "Legs" and s.derived_vol() is not None for s in used.values()):
        leg = st.number_input("Leg vol shift ± (vol pts)", 1.0, 20.0, 3.0, 1.0, key="cso_vol_leg")
        twists = [(a / 100, b / 100) for a in (-leg, 0, leg) for b in (-leg, 0, leg)]
        _, tw = book.vol_table(positions, used, val_date, mult, rate, [0.0], twists)
        pv = tw.assign(**{"Near vol shift": (tw["Near vol shift"] * 100).round(1),
                          "Far vol shift": (tw["Far vol shift"] * 100).round(1)}) \
            .pivot(index="Near vol shift", columns="Far vol shift", values="P&L ($)")
        st.markdown("**Leg-vol twist** (spreads on the Legs vol source only)")
        st.dataframe(pv.style.apply(_heat, axis=None).format(usd), width="stretch")
        st.caption("Near vol up with far unchanged widens the derived spread vol; both moving together barely "
                   "does — the spread cares about the *difference*.")


@st.fragment
def tab_theta():
    if _no_book():
        return
    d = st.slider("Days forward", 1, max_h, max_h, key="cso_theta_days")
    t = book.theta_path(positions, used, val_date, mult, rate, d)
    c = st.columns(2)
    c[0].altair_chart(alt.Chart(t).mark_line().encode(x="Day:Q", y="Cumulative P&L ($):Q").properties(height=300),
                      width="stretch")
    c[1].altair_chart(alt.Chart(t).mark_bar().encode(x="Day:Q", y="Daily theta ($):Q").properties(height=300),
                      width="stretch")
    st.caption("Spreads, vols and rates held constant — pure time decay, including the pull to intrinsic near expiry.")


def _mc_controls(prefix: str, default_h: int):
    c = st.columns(5)
    h = c[0].slider("Horizon (days)", 1, max_h, min(default_h, max_h), key=f"cso_{prefix}_h")
    n = c[1].select_slider("Paths", [5000, 10000, 20000, 50000], 20000, key=f"cso_{prefix}_n")
    vv = c[2].slider("Vol of vol", 0.0, 1.5, 0.3, 0.05, key=f"cso_{prefix}_vv",
                     help="Annual lognormal vol of the spread vol itself")
    sv = c[3].slider("Spread/vol corr", -0.9, 0.9, 0.0, 0.1, key=f"cso_{prefix}_sv")
    has_fc = any(s.full_carry for s in used.values())
    cap = c[4].toggle("Cap at full carry", value=False, disabled=not has_fc, key=f"cso_{prefix}_cap",
                      help="Needs a Full carry value on the spread (step 1)." if not has_fc else
                      "Stops each spread at its full-carry bound at the horizon.")
    return h, n, vv, sv, cap


@st.fragment
def tab_mc():
    if _no_book():
        return
    h, n, vv, sv, cap = _mc_controls("mc", 20)
    c = st.columns(2)
    xc = c[0].slider("Cross-spread corr", -0.5, 1.0, 0.5, 0.05, key="cso_mc_xc", help="Only matters with several spreads")
    basis = c[1].segmented_control("Measure P&L from", ["Today's mark", "Entry price"], default="Today's mark",
                                   key="cso_mc_basis", help="Entry price uses each row's Entry (premium paid/received)")
    pnl = book.monte_carlo(positions, used, val_date, mult, rate, h, n, xc, vv, sv, cap)
    if basis == "Entry price":
        pnl = pnl + book.unrealized(positions, used, val_date, mult, rate)
    s = book.summarize(pnl)
    m = st.columns(5)
    m[0].metric("Probability of loss", f"{s['P(loss)']:.0%}")
    m[1].metric("Mean P&L", usd(s["Mean"]))
    m[2].metric("VaR 95", usd(s["5% (VaR 95)"]))
    m[3].metric("CVaR 95", usd(s["CVaR 95"]))
    m[4].metric("Worst / best", f"{usd(s['Worst'])} / {usd(s['Best'])}")
    hist, edges = np.histogram(pnl, bins=60)
    hd = pd.DataFrame({"P&L ($)": (edges[:-1] + edges[1:]) / 2, "Paths": hist})
    st.altair_chart(alt.Chart(hd).mark_bar().encode(x="P&L ($):Q", y="Paths:Q").properties(height=320), width="stretch")
    st.dataframe(pd.Series(s, name="$ / prob").to_frame().T, hide_index=True, width="stretch")
    st.caption("Spread moves are normal at each spread's current vol; row-level skew vols affect repricing only. "
               "A bought option loses against its mark most of the time over short horizons (that's theta) while "
               "its mean P&L is ≈ 0 — read the tail and the hedged-vs-unhedged comparison, not the loss odds.")


@st.fragment
def tab_hedge():
    if _no_book():
        return
    exp, hed = book.split_roles(positions)
    if not exp or not hed:
        st.info("Tag at least one row as **Exposure** (what's being hedged) and one as **Hedge / trade** to compare "
                "hedged and unhedged outcomes.")
        return
    h, n, vv, sv, cap = _mc_controls("hd", 20)
    pu = book.monte_carlo(exp, used, val_date, mult, rate, h, n, 0.5, vv, sv, cap)
    ph = book.monte_carlo(positions, used, val_date, mult, rate, h, n, 0.5, vv, sv, cap)
    su, sh = book.summarize(pu), book.summarize(ph)
    cost = book.book_value(hed, used, val_date, mult, rate)
    m = st.columns(4)
    m[0].metric("Hedge mark (cost today)", usd(cost), help="Model value of the Hedge / trade rows")
    m[1].metric("VaR 95 — hedged", usd(sh["5% (VaR 95)"]), delta=usd(sh["5% (VaR 95)"] - su["5% (VaR 95)"]),
                help="Delta is the improvement vs unhedged")
    m[2].metric("Worst path — hedged", usd(sh["Worst"]), delta=usd(sh["Worst"] - su["Worst"]))
    m[3].metric("Std dev — hedged", usd(sh["Std dev"]), delta=usd(sh["Std dev"] - su["Std dev"]), delta_color="inverse")
    edges = np.histogram_bin_edges(np.concatenate([pu, ph]), bins=60)
    dist = pd.concat([pd.DataFrame({"P&L ($)": (edges[:-1] + edges[1:]) / 2, "Paths": np.histogram(p, edges)[0],
                                    "Book": name}) for name, p in (("Unhedged (exposure only)", pu),
                                                                   ("Hedged (all rows)", ph))])
    pal = alt.Scale(domain=["Unhedged (exposure only)", "Hedged (all rows)"], range=["#f58518", "#4c78a8"])
    st.altair_chart(alt.Chart(dist).mark_area(opacity=0.55, interpolate="step").encode(
        x="P&L ($):Q", y="Paths:Q", color=alt.Color("Book:N", scale=pal)).properties(height=320), width="stretch")
    shifts = np.arange(-40.0, 40.1, 2.0)
    gu = book.grid_pnl(exp, used, val_date, mult, rate, shifts, h)
    gh = book.grid_pnl(positions, used, val_date, mult, rate, shifts, h)
    gd = pd.DataFrame({"Spread move": shifts, "Unhedged": gu[f"+{h}d"], "Hedged": gh[f"+{h}d"],
                       "Hedged at expiry": gh["At expiry"]}).melt("Spread move", var_name="Book", value_name="P&L ($)")
    st.markdown(f"**P&L over a spread move, {h} days out**")
    gpal = alt.Scale(domain=["Unhedged", "Hedged", "Hedged at expiry"], range=["#f58518", "#4c78a8", "#54a24b"])
    st.altair_chart(alt.Chart(gd).mark_line().encode(x="Spread move:Q", y="P&L ($):Q",
                                                     color=alt.Color("Book:N", scale=gpal))
                    .properties(height=300), width="stretch")
    st.dataframe(pd.DataFrame({"Unhedged": su, "Hedged": sh}).T, width="stretch")
    st.caption("Same simulated paths for both books. The hedge's expected P&L against its model mark is ≈ 0 — "
               "its real cost is the premium decay plus the bid/offer paid over the model value (compare Entry to "
               "mark in Book & Greeks). Judge it by what it does to the left tail.")


@st.fragment
def tab_tools():
    sp_id = st.selectbox("Spread", list(spreads), key="cso_tool_spread")
    sp = spreads[sp_id]
    S = sp.level()
    c = st.columns(4)
    exp = c[0].date_input("Option expiry", val_date + timedelta(days=36), key="cso_tool_exp")
    T = max((exp - val_date).days, 0) / 365
    vol = c[1].number_input("Spread vol (pts/yr)", 0.1, 500.0, float(round(sp.vol(), 2)), 0.5,
                            key=f"cso_tool_vol_{sp_id}")
    c[2].metric("Spread level", f"{S:.2f}")
    c[3].metric("Years to expiry", f"{T:.3f}")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**Delta → strike**")
        dl = st.number_input("Delta (negative = put)", -0.99, 0.99, -0.15, 0.01, key="cso_tool_delta")
        if dl == 0:
            st.write("Pick a non-zero delta.")
        else:
            K = pricing.strike_for_delta(dl, S, vol, T, rate)
            kind = "Call" if dl > 0 else "Put"
            prem = float(pricing.price(dl > 0, S, K, vol, T, rate))
            st.metric(f"{abs(dl) * 100:.0f}-delta {kind} strike", f"{K:.2f}")
            st.caption(f"Nearest 5-point strike {round(K / 5) * 5:.0f}  ·  premium {prem:.2f} pts "
                       f"({usd(prem * mult)}/contract)")
    with right, st.container(border=True):
        st.markdown("**Premium → implied spread vol**")
        k2 = st.number_input("Strike", value=float(round((S - 5) / 5) * 5), step=1.0, key="cso_tool_k")
        ty = st.segmented_control("Type", ["Call", "Put"], default="Put", key="cso_iv_type")
        prem = st.number_input("Quoted premium (pts)", 0.0, 500.0, 2.0, 0.125, key="cso_tool_prem")
        iv = pricing.implied_vol(ty == "Call", prem, S, k2, T, rate)
        if iv is None:
            st.warning("No solution — premium is at/below intrinsic value or the option is expired.")
        else:
            g = pricing.greeks(ty == "Call", S, k2, iv, T, rate)
            st.metric("Implied spread vol", f"{iv:.2f} pts/yr")
            st.caption(f"Delta {float(g['delta']):+.3f} · vega {float(g['vega']):.3f}/pt · theta {float(g['theta']):.3f}/day")


with t_strategy:
    if t_strategy.open:
        tab_strategy()
with t_book:
    if t_book.open:
        tab_book()
with t_grid:
    if t_grid.open:
        tab_grid()
with t_vol:
    if t_vol.open:
        tab_vol()
with t_theta:
    if t_theta.open:
        tab_theta()
with t_mc:
    if t_mc.open:
        tab_mc()
with t_hedge:
    if t_hedge.open:
        tab_hedge()
with t_tools:
    if t_tools.open:
        tab_tools()

with st.expander("Model and contract assumptions"):
    st.markdown(
        """
- **Product:** CME grain/oilseed calendar spread options — one option on one futures spread (**nearby − deferred**),
  not a pair of options. Exercise is **European** (only at expiry, auto-exercised if in the money) and leaves the
  holder with futures positions: a call buyer ends long the nearby / short the deferred, a put buyer the reverse.
  Grain legs are 5,000 bu ($50 per cent); premiums tick in ⅛ cent.
- **Model:** Bachelier (normal) on the spread level with a spread vol in points/yr, discounted at the rate above.
  Kirk's approximation gives the same prices to within 0.01¢ for corn-sized spreads, so the vol input — and how it
  varies by strike — is what matters, not the model.
- **Not modelled:** a hard full-carry bound in pricing (the optional Full carry only truncates the Monte Carlo),
  vol term structure across expiries beyond what the quotes imply, and bid/offer. CME CSOs are European, so
  early exercise doesn't arise; for an American-style spread option the premium this ignores is typically
  under half a tick for six months or less at 4% rates.
- **Prices:** the Fetch button pulls delayed futures prices from the market-data feed. Anything you type stays
  until you fetch again.
- **Strategy builder:** P&L is measured against what the legs cost (a spread future against its entry level, an
  option against its premium), at expiry and marked with the model today. Full carry follows the Cost of Carry
  calculator: CME maximum storage plus interest at fed funds + 2.25%, between the two contracts' expiries.
- **Check before trading:** the expiry date for each series, the listed strikes and the contract size, against
  CME's current contract specifications and expiry calendar.
        """)
