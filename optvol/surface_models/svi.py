"""SVI (Stochastic Volatility Inspired) parametric fit to the volatility smile.

SVI is the industry-standard way to parametrize a single expiry's smile with a
handful of interpretable numbers instead of raw scattered quotes. Introduced by
Jim Gatheral, it's used across sell-side and buy-side vol desks because it (a)
fits real smiles well with only 5 parameters, and (b) admits simple conditions
for ruling out arbitrage.

**Parametrization (raw SVI).** Let ``k = ln(K / F)`` be log-moneyness relative
to the forward price ``F = S * exp((r - q) * T)``. SVI models the **total
implied variance** ``w(k) = sigma_BS(k)^2 * T`` as:

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )

Five parameters per expiry:
    a      overall variance level (vertical shift)
    b      angle between the wings (>= 0; overall steepness)
    rho    wing rotation / skew (in (-1, 1); rho<0 tilts toward downside skew)
    m      horizontal shift of the smile's minimum (ATM offset)
    sigma  controls curvature/smoothness at the minimum (> 0)

We calibrate these 5 numbers per expiry by least squares against the market's
own solved implied vols (from Phase 1's inversion), then can read off a smooth,
arbitrage-checked IV at ANY strike -- including strikes the market didn't quote.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class SVIParams:
    """Calibrated raw-SVI parameters for one expiry."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float
    T: float
    forward: float
    rmse: float  # fit residual, in IV units (not variance)

    def total_variance(self, k):
        """w(k): total implied variance at log-moneyness k."""
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (
            self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma ** 2)
        )

    def implied_vol(self, K):
        """Implied vol at strike(s) K, via w(k) = sigma^2 * T."""
        K = np.asarray(K, dtype=float)
        k = np.log(K / self.forward)
        w = self.total_variance(k)
        w = np.maximum(w, 1e-12)  # guard against tiny numerical negatives
        return np.sqrt(w / self.T)

    def is_arbitrage_free(self) -> bool:
        """Necessary conditions for a static-arbitrage-free (butterfly) smile.

        These are the standard raw-SVI checks (Gatheral & Jacquier):
          * b >= 0                              (variance must not decrease
                                                   without bound with |k-m|)
          * |rho| < 1                            (valid rotation)
          * sigma > 0                            (valid curvature)
          * a + b * sigma * sqrt(1 - rho^2) >= 0 (minimum total variance >= 0,
                                                   i.e. the smile never implies
                                                   a negative variance)
        These do not guarantee full (calendar-spread) arbitrage-freedom across
        expiries, only that this one slice is internally consistent.
        """
        min_w = self.a + self.b * self.sigma * np.sqrt(max(1 - self.rho ** 2, 0.0))
        return (
            self.b >= 0
            and abs(self.rho) < 1.0
            and self.sigma > 0
            and min_w >= -1e-8
        )


def _forward_price(spot: float, r: float, q: float, T: float) -> float:
    return spot * np.exp((r - q) * T)


def fit_svi_slice(strikes, ivs, spot: float, r: float, q: float, T: float) -> SVIParams:
    """Calibrate raw SVI to one expiry's (strike, implied-vol) points.

    Uses SLSQP with explicit bounds/constraints matching the no-arbitrage
    conditions in ``SVIParams.is_arbitrage_free`` -- the optimizer is
    constrained to only ever search arbitrage-free parameter space.
    """
    strikes = np.asarray(strikes, dtype=float)
    ivs = np.asarray(ivs, dtype=float)
    mask = np.isfinite(strikes) & np.isfinite(ivs) & (ivs > 0)
    strikes, ivs = strikes[mask], ivs[mask]
    if len(strikes) < 5:
        raise ValueError("Need at least 5 clean (strike, IV) points to fit SVI.")

    F = _forward_price(spot, r, q, T)
    k = np.log(strikes / F)
    w_mkt = (ivs ** 2) * T  # target: total variance

    def unpack(x):
        a, b, rho, m, sigma = x
        return a, b, rho, m, sigma

    def model_w(x, k):
        a, b, rho, m, sigma = unpack(x)
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    def objective(x):
        resid = model_w(x, k) - w_mkt
        return float(np.mean(resid ** 2))

    # Sensible starting guess from the data itself.
    a0 = max(np.min(w_mkt), 1e-4)
    b0 = 0.1
    rho0 = 0.0
    m0 = float(np.mean(k))
    sigma0 = max(float(np.std(k)), 0.05)
    x0 = np.array([a0, b0, rho0, m0, sigma0])

    bounds = [
        (1e-8, None),    # a >= 0
        (1e-8, 5.0),     # b >= 0
        (-0.999, 0.999), # |rho| < 1
        (min(k) - 1.0, max(k) + 1.0),  # m near the observed moneyness range
        (1e-4, 5.0),     # sigma > 0
    ]

    def min_variance_constraint(x):
        a, b, rho, m, sigma = unpack(x)
        return a + b * sigma * np.sqrt(max(1 - rho ** 2, 0.0))  # >= 0

    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds,
        constraints=[{"type": "ineq", "fun": min_variance_constraint}],
        options={"maxiter": 500, "ftol": 1e-12},
    )

    a, b, rho, m, sigma = unpack(result.x)
    fitted_w = model_w(result.x, k)
    fitted_iv = np.sqrt(np.maximum(fitted_w, 1e-12) / T)
    rmse = float(np.sqrt(np.mean((fitted_iv - ivs) ** 2)))

    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma, T=T, forward=F, rmse=rmse)


def fit_svi_surface(surf) -> Dict[str, SVIParams]:
    """Fit an independent SVI slice to every expiry in a ``VolSurface``.

    Returns a dict {expiry: SVIParams}. Expiries with too few clean OTM points
    to calibrate are skipped rather than raising.
    """
    fits: Dict[str, SVIParams] = {}
    for exp in surf.expiries:
        s = surf.smile(exp)
        if len(s) < 5:
            continue
        T = float(s["T"].iloc[0])
        try:
            fits[exp] = fit_svi_slice(
                s["strike"].values, s["iv_solved"].values,
                surf.spot, surf.risk_free_rate, surf.dividend_yield, T,
            )
        except Exception:
            continue
    return fits
