"""The payoff chart: shaded green where the position gains value and red where it loses."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

GREEN, RED, INK, MUTED, BLUE, ORANGE, CARRY = "#2e9e5b", "#d64545", "#32373c", "#8b95a1", "#0693e3", "#f58518", "#8b1e1e"
SERIES = ("Combined at expiry", "Position only", "Combined today")


def _marks(xs, labels, color, dash, text_dy=2):
    """Vertical rules with a small label each, drawn at the top of the plot."""
    if not len(xs):
        return []
    df = pd.DataFrame({"x": list(xs), "label": list(labels)})
    rule = alt.Chart(df).mark_rule(color=color, strokeWidth=1.5, strokeDash=dash).encode(x="x:Q")
    text = alt.Chart(df).mark_text(align="left", baseline="top", dx=4, dy=text_dy, fontSize=11, color=color).encode(
        x="x:Q", y=alt.value(0), text="label:N")
    return [rule, text]


def _replay_layers(replays, x_scale, y_scale):
    """One dot per prior year on the combined line (green where the strategy ends up ahead, red behind), labelled."""
    df = pd.DataFrame(list(replays), columns=["year", "x", "y"])
    df["result"] = np.where(df["y"] >= 0, "gain", "loss")
    x = alt.X("x:Q", scale=x_scale)
    y = alt.Y("y:Q", scale=y_scale)
    dots = alt.Chart(df).mark_point(filled=True, size=110, stroke="white", strokeWidth=1.5, opacity=1, clip=True).encode(
        x=x, y=y,
        fill=alt.Fill("result:N", scale=alt.Scale(domain=["gain", "loss"], range=[GREEN, RED]), legend=None),
        tooltip=[alt.Tooltip("year:N", title="Prior year's move"), alt.Tooltip("x:Q", title="Spread if repeated", format=".2f"),
                 alt.Tooltip("y:Q", title="P&L ($)", format="$,.0f")])
    labels = alt.Chart(df).mark_text(dy=-13, fontSize=10, color=INK, clip=True).encode(x=x, y=y, text="year:N")
    return [dots, labels]


def payoff_chart(spread, combined, position_only, today=None, *, current: float, strikes=(), breakevens=(),
                 carry_limit: float | None = None, band: tuple[float, float] | None = None, replays=(),
                 x_title: str = "Spread level at expiry (near − far, points)", height: int = 420):
    """Layered Altair chart of P&L ($) against the spread level.

    ``combined`` is the whole strategy at expiry and carries the green/red fill; ``position_only`` is the same
    position without the option, ``today`` the strategy marked with the model today. ``band`` is the (low, high)
    the spread has actually traded in, shaded behind everything. ``replays`` are ``(label, spread level, P&L)``
    dots, one per prior year, for where the strategy would stand if the spread repeated that year's move.
    """
    S = np.asarray(spread, dtype=float)
    c = np.asarray(combined, dtype=float)
    base = np.asarray(position_only, dtype=float)
    fills = pd.DataFrame({"Spread": S, "gain": np.maximum(c, 0.0), "loss": np.minimum(c, 0.0), "zero": 0.0})
    series = {SERIES[0]: c, SERIES[1]: base}
    if today is not None:
        series[SERIES[2]] = np.asarray(today, dtype=float)
    lines = pd.concat([pd.DataFrame({"Spread": S, "P&L ($)": v, "Series": k}) for k, v in series.items()])

    lo = min(float(v.min()) for v in series.values())
    hi = max(float(v.max()) for v in series.values())
    pad = max((hi - lo) * 0.08, 1.0)
    y_scale = alt.Scale(domain=[min(lo - pad, -pad), max(hi + pad, pad)], nice=False)
    x_scale = alt.Scale(domain=[float(S.min()), float(S.max())], nice=False)
    x = alt.X("Spread:Q", title=x_title, scale=x_scale)
    y_axis = alt.Axis(format="$,.0f", title="P&L ($)")

    layers = []
    if band is not None:
        layers.append(alt.Chart(pd.DataFrame({"x0": [band[0]], "x1": [band[1]]})).mark_rect(
            color=BLUE, opacity=0.10).encode(x=alt.X("x0:Q", scale=x_scale), x2="x1:Q"))
    layers.append(alt.Chart(fills).mark_area(color=GREEN, opacity=0.38).encode(
        x=x, y=alt.Y("gain:Q", scale=y_scale, axis=y_axis), y2="zero:Q"))
    layers.append(alt.Chart(fills).mark_area(color=RED, opacity=0.38).encode(
        x=x, y=alt.Y("loss:Q", scale=y_scale, axis=y_axis), y2="zero:Q"))
    layers.append(alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color=MUTED, strokeWidth=1).encode(
        y=alt.Y("y:Q", scale=y_scale)))
    names = list(series)
    palette = {SERIES[0]: INK, SERIES[1]: MUTED, SERIES[2]: BLUE}
    dashes = {SERIES[0]: [1, 0], SERIES[1]: [6, 4], SERIES[2]: [1, 0]}
    layers.append(alt.Chart(lines).mark_line(strokeWidth=2.2).encode(
        x=x, y=alt.Y("P&L ($):Q", scale=y_scale, axis=y_axis),
        color=alt.Color("Series:N", scale=alt.Scale(domain=names, range=[palette[n] for n in names]),
                        legend=alt.Legend(title=None, orient="top")),
        strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(domain=names, range=[dashes[n] for n in names]),
                                  legend=None),
        tooltip=[alt.Tooltip("Series:N"), alt.Tooltip("Spread:Q", format=".2f"), alt.Tooltip("P&L ($):Q", format="$,.0f")]))

    layers += _marks([current], ["now"], BLUE, [1, 0])
    layers += _marks([k for k, _ in strikes], [lab for _, lab in strikes], ORANGE, [5, 4], text_dy=16)
    layers += _marks(breakevens, [f"break-even {b:+.2f}" for b in breakevens], MUTED, [2, 3], text_dy=30)
    if carry_limit is not None:
        layers += _marks([carry_limit], ["full carry"], CARRY, [2, 2], text_dy=44)
    if len(replays):
        layers += _replay_layers(replays, x_scale, y_scale)
    return alt.layer(*layers).properties(height=height)
