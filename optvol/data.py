"""Data layer: pull and CLEAN live option chains from yfinance.

yfinance quotes are noisy -- stale mids, zero-bid contracts, crossed markets,
and nonsensical implied vols are common. Feeding that straight into a model
produces a broken-looking surface, so the cleaning here is load-bearing, not
cosmetic. Every filter is recorded so the UI can report *why* contracts were
dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Columns we care about from the raw yfinance chain.
_RAW_COLS = [
    "contractSymbol",
    "strike",
    "lastPrice",
    "bid",
    "ask",
    "volume",
    "openInterest",
    "impliedVolatility",
    "inTheMoney",
    "lastTradeDate",
]

# Day-count: calendar-year convention (365), used consistently with the pricing
# module. (The original script used 252, a trading-day convention more typical
# for realised vol than for option time-to-expiry.)
_DAYS_PER_YEAR = 365.0


@dataclass
class ChainSnapshot:
    """A cleaned option chain for a single (ticker, expiry)."""

    symbol: str
    expiry: str
    spot: float
    risk_free_rate: float
    dividend_yield: float
    T: float  # years to expiry
    calls: pd.DataFrame
    puts: pd.DataFrame
    drop_report: Dict[str, int] = field(default_factory=dict)

    @property
    def combined(self) -> pd.DataFrame:
        return pd.concat([self.calls, self.puts], ignore_index=True, sort=False)


def get_risk_free_rate(default: float = 0.04) -> float:
    """Pull a short-term risk-free proxy from the 13-week T-bill (^IRX).

    ^IRX is quoted in percent (e.g. 5.2 == 5.2%). Falls back to ``default`` if
    the fetch fails so the pipeline never hard-crashes on a data hiccup.
    """
    try:
        irx = yf.Ticker("^IRX").history(period="5d")
        if not irx.empty:
            return float(irx["Close"].dropna().iloc[-1]) / 100.0
    except Exception:
        pass
    return default


def get_spot(ticker: yf.Ticker) -> float:
    hist = ticker.history(period="1d")
    if hist.empty:
        raise ValueError("Could not fetch spot price (empty history).")
    return float(hist["Close"].iloc[-1])


def get_dividend_yield(ticker: yf.Ticker, default: float = 0.0) -> float:
    try:
        dy = ticker.info.get("dividendYield")
        if dy is None:
            return default
        # Current yfinance returns this as a PERCENT number (e.g. 0.37 == 0.37%,
        # 2.5 == 2.5%), so divide by 100. A sanity cap guards against the
        # occasional garbage value blowing up the forward.
        dy = float(dy) / 100.0
        return dy if 0.0 <= dy <= 0.25 else default
    except Exception:
        return default


# US equity options expire at 16:00 ET ~= 20:00 UTC.
_EXPIRY_HOUR_UTC = 20


def _years_to_expiry(expiry: str) -> float:
    exp = datetime.strptime(expiry, "%Y-%m-%d").replace(
        hour=_EXPIRY_HOUR_UTC, tzinfo=timezone.utc
    )
    now = datetime.now(timezone.utc)
    seconds = (exp - now).total_seconds()
    return max(seconds / (_DAYS_PER_YEAR * 24 * 3600), 0.0)


# Skip near-0-DTE expiries by default: their time value is tiny and the implied
# vol is numerically unstable / dominated by quote noise.
_MIN_DEFAULT_T = 3.0 / _DAYS_PER_YEAR


def _nearest_tradeable_expiry(expiries: List[str]) -> str:
    for e in expiries:
        if _years_to_expiry(e) >= _MIN_DEFAULT_T:
            return e
    return expiries[-1]


def _clean_side(df: pd.DataFrame, option_type: str, spot: float) -> (pd.DataFrame, Dict[str, int]):
    """Apply quality filters to one side (calls or puts).

    Returns the cleaned frame plus a per-reason count of dropped rows.
    """
    report: Dict[str, int] = {}
    work = df.copy()
    # Keep only columns we know about (some may be missing on odd tickers).
    cols = [c for c in _RAW_COLS if c in work.columns]
    work = work[cols].copy()
    work["option_type"] = option_type

    n0 = len(work)

    def _drop(mask, reason):
        nonlocal work
        removed = int(mask.sum())
        if removed:
            report[reason] = removed
        work = work[~mask]

    # 1. Missing/zero bid or ask -> no two-sided market.
    _drop((work["bid"].fillna(0) <= 0) | (work["ask"].fillna(0) <= 0), "no_two_sided_quote")
    # 2. Crossed/locked market (bid >= ask).
    _drop(work["bid"] >= work["ask"], "crossed_market")
    # 3. Absurd spread (> 50% of mid) -> illiquid, untradeable mark.
    mid = (work["bid"] + work["ask"]) / 2.0
    spread = (work["ask"] - work["bid"]) / mid.replace(0, np.nan)
    _drop(spread > 0.5, "wide_spread")
    # 4. Zero open interest AND zero volume -> dead contract.
    if "openInterest" in work and "volume" in work:
        oi = work["openInterest"].fillna(0)
        vol = work["volume"].fillna(0)
        _drop((oi <= 0) & (vol <= 0), "no_liquidity")
    # 5. Implausible strikes (more than 60% away from spot rarely has signal
    #    and is dominated by quote noise).
    _drop((work["strike"] < spot * 0.4) | (work["strike"] > spot * 1.6), "far_otm")

    work = work.reset_index(drop=True)
    work["mid_price"] = (work["bid"] + work["ask"]) / 2.0
    work["moneyness"] = work["strike"] / spot
    report["_kept"] = len(work)
    report["_raw"] = n0
    return work, report


def load_chain(symbol: str, expiry: Optional[str] = None) -> ChainSnapshot:
    """Load and clean a single-expiry option chain.

    Parameters
    ----------
    symbol : str
        Ticker, e.g. "AAPL".
    expiry : str, optional
        Expiry in YYYY-MM-DD. Defaults to the nearest available expiry.
    """
    ticker = yf.Ticker(symbol)
    expiries = ticker.options
    if not expiries:
        raise ValueError(f"No listed options found for {symbol!r}.")
    if expiry is None:
        expiry = _nearest_tradeable_expiry(expiries)
    elif expiry not in expiries:
        raise ValueError(
            f"Expiry {expiry} not available for {symbol}. "
            f"Available: {', '.join(expiries[:8])}..."
        )

    spot = get_spot(ticker)
    r = get_risk_free_rate()
    q = get_dividend_yield(ticker)
    T = _years_to_expiry(expiry)

    raw = ticker.option_chain(expiry)
    calls, call_report = _clean_side(raw.calls, "call", spot)
    puts, put_report = _clean_side(raw.puts, "put", spot)

    drop_report = {f"call_{k}": v for k, v in call_report.items()}
    drop_report.update({f"put_{k}": v for k, v in put_report.items()})

    return ChainSnapshot(
        symbol=symbol,
        expiry=expiry,
        spot=spot,
        risk_free_rate=r,
        dividend_yield=q,
        T=T,
        calls=calls,
        puts=puts,
        drop_report=drop_report,
    )


def list_expiries(symbol: str) -> List[str]:
    return list(yf.Ticker(symbol).options)
