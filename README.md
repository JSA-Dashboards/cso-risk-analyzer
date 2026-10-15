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
cso_risk_analyzer` in the portal repo (the daily sync PR also picks it up). The app reads no secrets and no
database, so the portal needs nothing extra. Session and widget keys are prefixed `cso_` because every bundled
app shares one session.

## Secrets

One optional secret: **`MASSIVE_API_KEY`** (root level in Streamlit Secrets, like the admin portal; a `[massive]` table with
`api_key` also works, as does the environment variable). It turns on the *Fetch futures prices* button, which fills each
spread's near and far price from Massive's delayed futures snapshot. The key stays server-side and is sent as a header.
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
- **Tabs:** Book & Greeks (+ delta in futures-month terms), spread-move grid, spread move × vol matrix, theta burn,
  Monte Carlo, **hedge effectiveness** (rows tagged *Exposure* vs *Hedge / trade*, same paths for both books), and
  delta→strike / premium→IV tools.

## Known limits / to verify before client use

- Exact expiry date per series, strike ladder and contract size: confirm on CME (not machine-verified here).
- No hard full-carry bound in pricing; the optional *Full carry* only truncates the Monte Carlo.
- No bid/offer; hedged results are measured against the model mark.
- Placeholder prices/vols/expiries are pre-filled. The in-app note is **not** the firm's approved disclosure.

## CME sources checked (2026-10-15)

- CME grain & oilseed CSO factsheet: <https://www.cmegroup.com/trading/agricultural/files/grain-and-oilseed-calendar-spread-options.pdf>
- CBOT Rulebook Chapter 10J (corn CSOs): <https://www.cmegroup.com/content/dam/cmegroup/rulebook/CBOT/I/10J.pdf>
