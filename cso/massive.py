"""Futures prices from Massive (api.massive.com) for the near and far legs of a spread.

Server-side only. The key comes from Streamlit secrets (``MASSIVE_API_KEY`` at the root, as the admin
portal keeps it, or a ``[massive] api_key`` table, as the Risk Analyzer does) or the environment. It
goes out as a Bearer header, never in a URL, and is never shown or logged.

Outright contracts are ``<product><month letter><last digit of year>``: corn Dec 2026 is ``ZCZ6``.
Grains and soybean oil come back in cents and meal in $/ton, which are the points this app prices in.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone

import requests

BASE_URL = "https://api.massive.com"

PRODUCT_CODES = {
    "Corn (¢/bu)": "ZC",
    "Soybeans (¢/bu)": "ZS",
    "Chicago wheat (¢/bu)": "ZW",
    "KC wheat (¢/bu)": "KE",
    "Soybean meal ($/ton)": "ZM",
    "Soybean oil (¢/lb)": "ZL",
}

MONTH_LETTERS = {"jan": "F", "feb": "G", "mar": "H", "apr": "J", "may": "K", "jun": "M",
                 "jul": "N", "aug": "Q", "sep": "U", "oct": "V", "nov": "X", "dec": "Z"}
LETTER_MONTH = {letter: i + 1 for i, letter in enumerate(MONTH_LETTERS.values())}

_NAME = re.compile(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s\-/'_.]*(\d{1,4})?$", re.I)
_CODE = re.compile(r"^(?:[A-Z]{1,2})?([FGHJKMNQUVXZ])[\s\-/'_.]*(\d{1,4})$", re.I)


class MassiveError(RuntimeError):
    """A failure worth showing the user. Messages never contain the key."""


@dataclass(frozen=True)
class Quote:
    ticker: str
    price: float
    source: str                 # last | settlement | close | mid
    as_of: datetime | None      # UTC time of the last trade (None when the price is a settlement/close/mid)
    expiry: date | None
    delayed: bool


# ----------------------------------------------------------------------------- contract labels
def parse_contract(label, today: date) -> tuple[str, int] | None:
    """``"Dec26"`` / ``"Dec 2026"`` / ``"Z26"`` / ``"ZCZ6"`` / ``"Dec"`` -> (month letter, last year digit).

    A label with no year means the next occurrence of that month on or after ``today``.
    """
    s = ("" if label is None else str(label)).strip()
    if not s or s.lower() == "nan":
        return None
    m = _NAME.match(s)
    if m:
        letter, digits = MONTH_LETTERS[m.group(1).lower()], m.group(2)
    else:
        m = _CODE.match(s)
        if not m:
            return None
        letter, digits = m.group(1).upper(), m.group(2)
    if digits:
        return letter, int(digits[-1])
    month = LETTER_MONTH[letter]
    year = today.year if month >= today.month else today.year + 1
    return letter, year % 10


def ticker_for(product: str | None, label, today: date) -> str | None:
    if not product:
        return None
    parsed = parse_contract(label, today)
    return None if parsed is None else f"{product}{parsed[0]}{parsed[1]}"


# ----------------------------------------------------------------------------- credentials
def api_key_from(secrets, environ=None) -> str | None:
    """The Massive key from ``st.secrets`` (root ``MASSIVE_API_KEY`` or ``[massive] api_key``) or the environment."""
    environ = os.environ if environ is None else environ
    try:
        if "MASSIVE_API_KEY" in secrets and str(secrets["MASSIVE_API_KEY"]).strip():
            return str(secrets["MASSIVE_API_KEY"]).strip()
        if "massive" in secrets:
            value = secrets["massive"].get("api_key")
            if value and str(value).strip():
                return str(value).strip()
    except Exception:       # no secrets file at all, or an unexpected shape: fall through to the environment
        pass
    value = environ.get("MASSIVE_API_KEY")
    return value.strip() if value and value.strip() else None


# ----------------------------------------------------------------------------- fetching
def _price_of(result: dict) -> tuple[float, str] | None:
    """last trade, else settlement, else close, else the bid/ask midpoint."""
    last = (result.get("last_trade") or {}).get("price")
    if last is not None:
        return float(last), "last"
    session = result.get("session") or {}
    for field, name in (("settlement_price", "settlement"), ("close", "close")):
        if session.get(field) is not None:
            return float(session[field]), name
    quote = result.get("last_quote") or {}
    if quote.get("bid") is not None and quote.get("ask") is not None:
        return (float(quote["bid"]) + float(quote["ask"])) / 2, "mid"
    return None


def _get(url: str, tickers: list[str], api_key: str, timeout: float) -> dict:
    try:
        resp = requests.get(url, params={"ticker.any_of": ",".join(tickers)}, timeout=timeout,
                            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"})
    except requests.RequestException as exc:
        raise MassiveError(f"could not reach the market-data service ({type(exc).__name__})") from None
    if resp.status_code in (401, 403):
        raise MassiveError(f"the market-data service rejected the API key (HTTP {resp.status_code})")
    try:
        payload = resp.json()
    except ValueError:
        raise MassiveError(f"the market-data service returned a non-JSON response (HTTP {resp.status_code})") from None
    if not resp.ok or payload.get("status") == "ERROR":
        detail = str(payload.get("error") or payload.get("message") or f"HTTP {resp.status_code}")[:140]
        raise MassiveError(f"the market-data service reported an error: {detail}")
    return payload


def fetch_snapshot(api_key: str, tickers, base_url: str = BASE_URL, timeout: float = 12.0) -> dict[str, Quote]:
    """Latest price for each ticker that has one. One quiet retry covers a gateway hiccup or rate limit."""
    wanted = sorted({t for t in tickers if t})
    if not wanted:
        return {}
    url = f"{base_url}/futures/v1/snapshot"
    try:
        payload = _get(url, wanted, api_key, timeout)
    except MassiveError as first:
        if "rejected the API key" in str(first):
            raise
        time.sleep(0.8)
        payload = _get(url, wanted, api_key, timeout)

    quotes: dict[str, Quote] = {}
    for result in payload.get("results") or []:
        details = result.get("details") or {}
        ticker, priced = details.get("ticker"), _price_of(result)
        if not ticker or priced is None:
            continue
        price, source = priced
        trade = result.get("last_trade") or {}
        as_of = None
        if source == "last" and trade.get("last_updated"):
            as_of = datetime.fromtimestamp(int(trade["last_updated"]) / 1e9, tz=timezone.utc)
        try:
            expiry = date.fromisoformat(details["settlement_date"]) if details.get("settlement_date") else None
        except ValueError:
            expiry = None
        quotes[ticker] = Quote(ticker, price, source, as_of, expiry, str(trade.get("timeframe", "")).upper() == "DELAYED")
    return quotes
