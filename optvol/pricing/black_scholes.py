"""Black-Scholes-Merton pricing and analytic Greeks.

Conventions
-----------
S      : spot price of the underlying
K      : strike
T      : time to expiry in YEARS (calendar-year / 365 convention)
r      : continuously-compounded risk-free rate (annualised)
sigma  : annualised volatility
q      : continuous dividend yield (annualised), default 0

All functions are vectorised over array-like inputs via numpy, so they work on
a whole option chain at once.

Greeks are returned in their conventional "market" units:
    delta  per $1 move in spot
    gamma  per $1 move in spot
    vega   per 1 VOL POINT (i.e. per 0.01 change in sigma)
    theta  per CALENDAR DAY (per-year value / 365)
    rho    per 1% change in rates (per-unit value / 100)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

CALL = "call"
PUT = "put"


def _d1_d2(S, K, T, r, sigma, q=0.0):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    # Guard against zero/negative T or sigma producing warnings; callers that
    # need expiry-day behaviour should handle T==0 via intrinsic value.
    vol_sqrt_t = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
    return d1, d2


def _normalise_type(option_type):
    t = str(option_type).lower()
    if t in ("c", "call"):
        return CALL
    if t in ("p", "put"):
        return PUT
    raise ValueError("option_type must be 'call' or 'put'")


def bs_price(S, K, T, r, sigma, option_type=CALL, q=0.0):
    """Black-Scholes price for a European call or put."""
    opt = _normalise_type(option_type)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if opt == CALL:
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)


def delta(S, K, T, r, sigma, option_type=CALL, q=0.0):
    opt = _normalise_type(option_type)
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    if opt == CALL:
        return disc_q * norm.cdf(d1)
    return disc_q * (norm.cdf(d1) - 1.0)


def gamma(S, K, T, r, sigma, option_type=CALL, q=0.0):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    return disc_q * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, option_type=CALL, q=0.0):
    """Vega per 1 vol POINT (per 0.01 change in sigma)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    return S * disc_q * norm.pdf(d1) * np.sqrt(T) / 100.0


def theta(S, K, T, r, sigma, option_type=CALL, q=0.0):
    """Theta per CALENDAR DAY."""
    opt = _normalise_type(option_type)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    term1 = -(S * disc_q * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))
    if opt == CALL:
        annual = (
            term1
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        )
    else:
        annual = (
            term1
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        )
    return annual / 365.0


def rho(S, K, T, r, sigma, option_type=CALL, q=0.0):
    """Rho per 1% (0.01) change in the risk-free rate."""
    opt = _normalise_type(option_type)
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    if opt == CALL:
        return K * T * disc_r * norm.cdf(d2) / 100.0
    return -K * T * disc_r * norm.cdf(-d2) / 100.0


def bs_greeks(S, K, T, r, sigma, option_type=CALL, q=0.0):
    """Return all Greeks (and price) as a dict."""
    return {
        "price": bs_price(S, K, T, r, sigma, option_type, q),
        "delta": delta(S, K, T, r, sigma, option_type, q),
        "gamma": gamma(S, K, T, r, sigma, option_type, q),
        "vega": vega(S, K, T, r, sigma, option_type, q),
        "theta": theta(S, K, T, r, sigma, option_type, q),
        "rho": rho(S, K, T, r, sigma, option_type, q),
    }
