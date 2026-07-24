"""Unit tests for the pricing core. No network access required."""
import math

import numpy as np
import pytest

from optvol.pricing.black_scholes import bs_price, bs_greeks, vega
from optvol.pricing.implied_vol import implied_vol


# A reference contract.
S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0


def test_put_call_parity():
    """C - P == S*e^{-qT} - K*e^{-rT} must hold exactly for any sigma."""
    c = bs_price(S, K, T, r, sigma, "call", q)
    p = bs_price(S, K, T, r, sigma, "put", q)
    lhs = c - p
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_known_atm_call_value():
    """Sanity vs a textbook value for an ATM 1y call (~10.45)."""
    c = bs_price(S, K, T, r, sigma, "call", q)
    assert c == pytest.approx(10.4506, abs=1e-3)


def test_iv_round_trip_call():
    """Price -> implied_vol must recover the original sigma."""
    price = bs_price(S, K, T, r, sigma, "call", q)
    recovered = implied_vol(price, S, K, T, r, "call", q)
    assert recovered == pytest.approx(sigma, abs=1e-5)


def test_iv_round_trip_put_otm():
    """Round-trip on an out-of-the-money put with a different vol."""
    sig = 0.45
    Kp = 80.0
    price = bs_price(S, Kp, T, r, sig, "put", q)
    recovered = implied_vol(price, S, Kp, T, r, "put", q)
    assert recovered == pytest.approx(sig, abs=1e-5)


def test_iv_round_trip_sweep():
    """Round-trip across a grid of strikes and vols."""
    for Kx in (70, 90, 100, 110, 130):
        for sig in (0.1, 0.25, 0.6, 1.2):
            for opt in ("call", "put"):
                price = bs_price(S, Kx, T, r, sig, opt, q)
                rec = implied_vol(price, S, Kx, T, r, opt, q)
                assert rec is not None, (Kx, sig, opt)
                assert rec == pytest.approx(sig, abs=1e-4), (Kx, sig, opt)


def test_iv_returns_none_below_intrinsic():
    """A price below intrinsic value has no implied vol."""
    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    bad_price = intrinsic - 1.0
    assert implied_vol(bad_price, S, K, T, r, "call", q) is None


def test_iv_returns_none_on_expired():
    assert implied_vol(5.0, S, K, 0.0, r, "call", q) is None


def test_delta_bounds():
    """Call delta in [0,1], put delta in [-1,0]."""
    cd = bs_greeks(S, K, T, r, sigma, "call", q)["delta"]
    pd_ = bs_greeks(S, K, T, r, sigma, "put", q)["delta"]
    assert 0.0 <= cd <= 1.0
    assert -1.0 <= pd_ <= 0.0


def test_vega_matches_finite_difference():
    """Analytic vega (per vol point) ~ central finite difference."""
    h = 1e-4
    up = bs_price(S, K, T, r, sigma + h, "call", q)
    dn = bs_price(S, K, T, r, sigma - h, "call", q)
    fd_per_unit = (up - dn) / (2 * h)
    analytic_per_point = vega(S, K, T, r, sigma, "call", q)
    assert analytic_per_point == pytest.approx(fd_per_unit / 100.0, rel=1e-4)


def test_vectorized_price():
    Ks = np.array([90.0, 100.0, 110.0])
    prices = bs_price(S, Ks, T, r, sigma, "call", q)
    assert prices.shape == (3,)
    assert np.all(np.diff(prices) < 0)  # call price decreases in strike
