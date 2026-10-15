import sys
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cso import book, pricing

st.set_page_config(page_title="CSO & spread risk analyzer", page_icon=":material/stacked_line_chart:", layout="wide")
# Hide the Streamlit Community Cloud viewer badge (profile avatar -> the account's other apps).
st.markdown("<style>[class*='_profileContainer_']{display:none !important;}</style>", unsafe_allow_html=True)

TODAY = date.today()
NEAR_FAR, FAR_NEAR = "Near − far", "Far − near"
SPREAD_COLS = ["Spread", "Near", "Far", "Near price", "Far price", "Spread level", "Spread vol", "Full carry",
               "Convention", "Vol source", "Near vol %", "Far vol %", "Corr"]
LEG_COLS = ["Vol source", "Near vol %", "Far vol %", "Corr"]


def _eighth(x: float) -> float:
    return round(x * 8) / 8          # CSO premiums tick in 1/8 point


# --------------------------------------------------------------------------- state
if "cso_spreads_df" not in st.session_state:
    st.session_state.cso_spreads_df = pd.DataFrame([{
        "Spread": "Dec/Mar", "Near": "Dec26", "Far": "Mar27", "Near price": 430.0, "Far price": 445.0,
        "Spread level": None, "Spread vol": 20.0, "Full carry": None, "Convention": NEAR_FAR,
        "Vol source": "Direct", "Near vol %": 22.0, "Far vol %": 21.0, "Corr": 0.97,
    }])
    st.session_state.cso_spreads_ver = 0
if "cso_pos_df" not in st.session_state:
    st.session_state.cso_pos_df = pd.DataFrame([
        {"Spread": "Dec/Mar", "Role": book.ROLE_EXPOSURE, "Type": "Spread future", "Qty": 10.0, "Strike": None,
         "Expiry": None, "Vol override": None, "Entry": -15.0},
        {"Spread": "Dec/Mar", "Role": book.ROLE_BOOK, "Type": "Put", "Qty": 10.0, "Strike": -20.0,
         "Expiry": TODAY + timedelta(days=36), "Vol override": None, "Entry": None},
    ])
if "cso_quotes_df" not in st.session_state:
    _T0, _S0 = 36 / 365, -15.0
    _ex = TODAY + timedelta(days=36)
    st.session_state.cso_quotes_df = pd.DataFrame([
        {"Spread": "Dec/Mar", "Type": t, "Strike": k, "Expiry": _ex,
         "Premium": _eighth(float(pricing.price(t == "Call", _S0, k, v, _T0, 0.04)))}
        for t, k, v in (("Put", -30.0, 24.0), ("Put", -20.0, 21.0), ("Call", -10.0, 19.0))])


def _apply_spread_vol(sid: str, vol: float) -> None:
    df = st.session_state["_cso_spreads_edited"].copy()
    m = df["Spread"] == sid
    df.loc[m, "Spread vol"] = round(vol, 2)
    df.loc[m, "Vol source"] = "Direct"
    st.session_state.cso_spreads_df = df
    st.session_state.cso_spreads_ver += 1      # new editor key => editor restarts from the updated frame


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
    st.caption("Prices, vols, premiums and expiries pre-filled are **placeholders** — overwrite them with live "
               "quotes and the CME expiry calendar. Figures are theoretical model output, not executable "
               "quotes or advice; replace this note with the firm's approved disclosure before sharing "
               "anything with a client.")

st.title("CSO & spread risk analyzer")
st.caption("CME calendar spread options (spread = nearby − deferred, European exercise) priced with the normal "
           "(Bachelier) model on the spread level, so negative carry strikes work. Spread futures and any "
           "client exposure are risked alongside.")

# --------------------------------------------------------------------------- 1 spreads
st.subheader("1 · Spreads")
show_legs = st.toggle("Show leg vol / correlation columns", value=False, key="cso_show_legs",
                      help="Only needed for the 'Legs' vol source or the diagnostic below. Spread vol should "
                           "normally be typed in or backed out of quotes.")
spreads_df = st.data_editor(
    st.session_state.cso_spreads_df, num_rows="dynamic", key=f"cso_spreads_editor_{st.session_state.cso_spreads_ver}",
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
    spreads[sid] = book.Spread(
        id=sid, near_label=_txt(r.get("Near"), "Near"), far_label=_txt(r.get("Far"), "Far"),
        near_px=npx, far_px=fpx, spread_vol=_num(r.get("Spread vol")) or 20.0, level_override=lvl,
        full_carry=_num(r.get("Full carry")), vol_source=_txt(r.get("Vol source"), "Direct"),
        near_vol=None if nv is None else nv / 100, far_vol=None if fv is None else fv / 100,
        corr=_num(r.get("Corr")), near_minus_far=_txt(r.get("Convention"), NEAR_FAR) != FAR_NEAR)
if not spreads:
    st.info("Add a spread to begin.")
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
        st.session_state.cso_quotes_df, num_rows="dynamic", key="cso_quotes_editor", width="stretch",
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
st.caption("Tag what the client is **hedging** as *Exposure* (e.g. a long spread future standing in for a "
           "storage/carry position) and the options or spread futures bought against it as *Hedge / trade*. "
           "Leave everything as Hedge / trade for a standalone view.")
pos_df = st.data_editor(
    st.session_state.cso_pos_df, num_rows="dynamic", key="cso_pos_editor", width="stretch",
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
    st.info("Add at least one position to see risk.")
    st.stop()
if any(p.expiry and p.expiry <= val_date for p in positions):
    st.warning("Some options are at or past expiry on the valuation date — they're valued at intrinsic.")
used = {p.spread_id: spreads[p.spread_id] for p in positions}
exp_dates = [p.expiry for p in positions if p.expiry and p.expiry > val_date]
max_h = max((min(exp_dates) - val_date).days if exp_dates else 30, 1)
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
t_book, t_grid, t_vol, t_theta, t_mc, t_hedge, t_tools = st.tabs(
    ["Book & Greeks", "Spread moves", "Vol shocks", "Theta burn", "Monte Carlo", "Hedge effectiveness",
     "Strike & IV tools"], on_change="rerun")


@st.fragment
def tab_book():
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
  vol term structure across expiries beyond what the quotes imply, early exercise (none for these), and bid/offer.
- **To verify against CME before relying on it:** the exact expiry date for each series (check the CME expiry
  calendar), the listed strike ladder, and the contract size for each commodity selected.
        """)
