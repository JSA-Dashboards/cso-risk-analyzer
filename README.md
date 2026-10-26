# CSO & spread risk analyzer

Streamlit risk tool for **CME calendar spread options** (CSOs) and spread futures, for a broker whose clients use
CSOs to manage carry/spread risk. Owner repo: `JSA-Dashboards/cso-risk-analyzer`.

> The code is public (and mirrored into the public admin-portal repo), so access control lives in the
> deployed app, not the repo. A standalone run asks for one shared password (`APP_PASSWORD` in `cso/auth.py`,
> the same generic gate as the Risk Analyzer's — it keeps casual visitors out, it is not per-user security);
> inside the admin portal the sync pre-passes it (`preauth`), so the portal's login is the only gate there.
> The app opens on a clearly labelled **example book** (clear it from the sidebar to enter your own). The
> disclosure at the top is the firm's standard text: paragraph 2 is copied unchanged from the Risk
> Analyzer (`cso/disclosure.py`), paragraph 1 is that app's model paragraph with only the model sentence
> (Black-76 → Bachelier) and the data-feed sentence changed — compliance should confirm those. Nothing
> here is trading advice.

```
pip install -r requirements-dev.txt
streamlit run app.py
python -m pytest tests -q
```

**Admin portal:** bundled as `apps/cso_risk_analyzer/` in `jsa-admin-portal` (tile under *Futures, Options &
Spreads*). That copy is a generated mirror — change it here, then run `python sync/sync_apps.py apply
cso_risk_analyzer` in the portal repo (the daily sync PR also picks it up). The app reads no database, and the one optional secret
(`MASSIVE_API_KEY`, below) is already in the portal's Secrets, so the portal needs nothing extra. Session and widget keys are prefixed `cso_` because every bundled
app shares one session.

## Secrets

One optional secret: **`MASSIVE_API_KEY`** (root level in Streamlit Secrets, like the admin portal; a `[massive]` table with
`api_key` also works, as does the environment variable). It turns on the *Fetch futures prices* button, which fills each
spread's near and far price from Massive's delayed futures snapshot; the Strategy builder also reads the fed-funds future,
the list of listed contract months (for the leg picker) and the spread's daily settlement history (the traded range, realized
vol and the seasonal overlay) from it. The key stays server-side and is sent as a header.
Without it the button is off and the app works with typed prices. Nothing else needs a secret.

## What it models

- **A CSO is one option on one number — the futures spread, nearby − deferred** (grain/oilseed CSOs, Chapter 10J for
  corn). European exercise, auto-exercised if in the money; a call buyer ends long the nearby / short the deferred.
  It is *not* a long-far/short-near pair of options, and a long CSO has negative theta, positive gamma/vega and a real
  spread delta.
- **Pricing:** Bachelier (normal) on the spread level, spread vol in points/yr, discounted at the rate input.
  Kirk's approximation agrees to ~0.01¢ on corn-sized spreads — the vol input matters, not the model.
- **Vol source:** typed in, or backed out of market-maker premiums (step 2), with a per-strike skew that can be
  applied to positions. Deriving spread vol from leg vols + correlation is available but is a diagnostic only: a 0.01
  change in correlation moves it ~25%.
- **Strategy builder (first tab):** a position (long or short spread) with a bought put or call laid over it, shown as a
  payoff chart **shaded green where the strategy gains and red where it loses**, plus break-even, max loss, the
  P&L if the spread reaches full carry, and the same grid as a green/red table. Payoffs are exact (piecewise linear,
  `cso/strategy.py` takes any list of legs, so more strategies are more legs, not new maths). Alongside it, from the
  Cost of Carry calculator: full carry (CME maximum storage + interest at fed funds + 2.25%), % of full carry with
  its traffic light, and the 12-month range the spread has actually traded in (Massive settlement history). It
  needs no book of positions.
- **Leg picker (Strategy builder, *Spread from → Live curve*):** choose the near and far contract from the feed's current
  curve (the nearest ten listed months, each with its live price) instead of typing labels. The far contract can only be
  a later month than the near one. The spread's vol starts at your book's vol if the book has the same pair, else at the
  pair's realized vol over the last 60 sessions (the box says which), and you overwrite it with the vol you are quoted. It
  works with an empty book too. *My book* stays the default, so the example and anything typed in step 1 behave as before.
- **Seasonal overlay (Strategy builder toggle):** the same pair in up to five earlier years, lined up by days to the near
  contract's last trading day and laid over the live contract (darkest line = most recent year, dotted = the average),
  with the strategy's strike, break-evens, entry and full carry across it and *today* and the *option expiry* marked.
  Below it, **what repeating each earlier year's move would have done**: that year's change between the days matching
  today and the option's expiry, added to today's level, and the strategy's P&L there (green/red, with the position
  without the option beside it). The same points are dots on the payoff chart. History comes from the feed's daily bars,
  which begin in autumn 2021, so there are at most five years and the oldest may be partial. It is a scenario read off a
  handful of years, not a forecast; the Cost of Carry calculator's 15-year Moore-style outlook needs an older archive
  and is not built here.
- **Other tabs:** Book & Greeks (+ delta in futures-month terms), spread-move grid, spread move × vol matrix, theta
  burn, Monte Carlo, **hedge effectiveness** (rows tagged *Exposure* vs *Hedge / trade*, same paths for both books),
  and delta→strike / premium→IV tools.

## Known limits / to verify before client use

- Exact expiry date per series, strike ladder and contract size: confirm on CME (not machine-verified here).
- No hard full-carry bound in pricing; the optional *Full carry* only truncates the Monte Carlo.
- No bid/offer; hedged results are measured against the model mark.
- The app opens on illustrative example data (labelled as such). The disclosure is the Risk Analyzer's standard text
  with the model and data-feed sentences adapted; compliance should confirm those sentences.

## CME sources checked (2026-10-15)

- CME grain & oilseed CSO factsheet: <https://www.cmegroup.com/trading/agricultural/files/grain-and-oilseed-calendar-spread-options.pdf>
- CBOT Rulebook Chapter 10J (corn CSOs): <https://www.cmegroup.com/content/dam/cmegroup/rulebook/CBOT/I/10J.pdf>
