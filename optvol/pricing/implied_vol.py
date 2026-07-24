"""Implied-volatility solver.

This is the conceptual heart of the project. The original script *consumed* the
market's quoted implied volatility and fed it into Black-Scholes -- which is
circular, because IV is by definition the sigma that makes the BS price equal
the market price. Here we instead INVERT the relationship: given a market price,
we solve for the sigma that reproduces it.

Strategy: a Newton-Raphson fast path (using vega as the derivative), falling
back to Brent's method on a bracketed interval when Newton wanders out of
bounds or stalls. Brent is guaranteed to converge given a sign change, which we
establish from no-arbitrage price bounds.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from optvol.pricing.black_scholes import bs_price, _normalise_type, CALL, PUT

_MIN_VOL = 1e-4
_MAX_VOL = 5.0  # 500% vol -- generous upper bracket


def _intrinsic_bounds(S, K, T, r, opt, q):
    """No-arbitrage lower/upper bounds on a European option price."""
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    fwd = S * disc_q
    if opt == CALL:
        lower = max(fwd - K * disc_r, 0.0)
        upper = fwd
    else:
        lower = max(K * disc_r - fwd, 0.0)
        upper = K * disc_r
    return lower, upper


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = CALL,
    q: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Optional[float]:
    """Solve for the Black-Scholes implied volatility.

    Returns ``None`` when the quote is unpriceable (expired, or violates
    no-arbitrage bounds so that no real sigma reproduces it) -- this is common
    with stale/illiquid yfinance quotes and must be handled gracefully rather
    than producing a garbage number.
    """
    opt = _normalise_type(option_type)

    if T <= 0 or market_price is None or market_price <= 0:
        return None

    lower, upper = _intrinsic_bounds(S, K, T, r, opt, q)
    # Allow a tiny tolerance for rounding in the quoted price.
    if market_price < lower - 1e-8 or market_price > upper + 1e-8:
        return None

    def objective(sigma):
        return float(bs_price(S, K, T, r, sigma, opt, q)) - market_price

    # --- Newton-Raphson fast path -------------------------------------------
    # Manning's rule-of-thumb starting guess from the Brenner-Subrahmanyam
    # approximation, clamped into the bracket.
    sigma = max(_MIN_VOL, min(_MAX_VOL, math.sqrt(2 * math.pi / T) * market_price / S))
    for _ in range(max_iter):
        diff = objective(sigma)
        if abs(diff) < tol:
            return sigma
        # vega in PRICE units (per unit sigma), not per vol-point.
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        v = S * math.exp(-q * T) * _norm_pdf(d1) * math.sqrt(T)
        if v < 1e-8:
            break  # vega too small; Newton unreliable -> fall back to Brent
        sigma -= diff / v
        if sigma <= _MIN_VOL or sigma >= _MAX_VOL or not math.isfinite(sigma):
            break  # left the sensible region -> fall back to Brent

    # --- Brent fallback (bracketed, guaranteed) -----------------------------
    f_lo = objective(_MIN_VOL)
    f_hi = objective(_MAX_VOL)
    if f_lo * f_hi > 0:
        # No sign change in the bracket -> no solution in [_MIN_VOL, _MAX_VOL].
        return None
    try:
        return float(brentq(objective, _MIN_VOL, _MAX_VOL, xtol=tol, maxiter=max_iter))
    except (ValueError, RuntimeError):
        return None


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def implied_vol_vectorized(market_price, S, K, T, r, option_type, q=0.0):
    """Convenience wrapper to solve IV across array-like inputs.

    Returns a numpy array with ``np.nan`` where a quote is unpriceable.
    """
    market_price = np.atleast_1d(market_price)
    K = np.atleast_1d(K)
    T = np.atleast_1d(T)
    option_type = np.atleast_1d(option_type)
    n = max(len(market_price), len(K), len(T), len(option_type))

    def _get(a, i):
        return a[i] if len(a) > 1 else a[0]

    out = np.empty(n, dtype=float)
    for i in range(n):
        iv = implied_vol(
            float(_get(market_price, i)),
            float(S),
            float(_get(K, i)),
            float(_get(T, i)),
            float(r),
            str(_get(option_type, i)),
            float(q),
        )
        out[i] = np.nan if iv is None else iv
    return out
