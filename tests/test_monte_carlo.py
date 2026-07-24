"""Tests for the Monte Carlo engine. Fixed seeds keep them deterministic.

The headline test is convergence: the MC price of a European option must agree
with the exact Black-Scholes value. If that holds, the simulation machinery is
trustworthy for the path-dependent options that have no closed form.
"""
import math

import numpy as np
import pytest

from optvol.pricing.black_scholes import bs_price
from optvol.pricing.monte_carlo import (
    mc_european_price,
    mc_asian_price,
    mc_barrier_price,
    simulate_terminal,
)

S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0


def test_european_call_converges_to_bs():
    bs = bs_price(S, K, T, r, sigma, "call", q)
    mc = mc_european_price(S, K, T, r, sigma, "call", q, n_paths=200_000, seed=42)
    # Within 4 standard errors -> passes ~99.99% of the time.
    assert abs(mc.price - bs) < 4 * mc.std_error


def test_european_put_converges_to_bs():
    bs = bs_price(S, K, T, r, sigma, "put", q)
    mc = mc_european_price(S, K, T, r, sigma, "put", q, n_paths=200_000, seed=7)
    assert abs(mc.price - bs) < 4 * mc.std_error


def test_error_shrinks_with_more_paths():
    """Monte Carlo error should fall roughly like 1/sqrt(n)."""
    small = mc_european_price(S, K, T, r, sigma, "call", q, n_paths=10_000, seed=1)
    large = mc_european_price(S, K, T, r, sigma, "call", q, n_paths=160_000, seed=1)
    # 16x the paths -> ~4x smaller standard error. Allow slack.
    assert large.std_error < small.std_error / 3.0


def test_terminal_is_a_martingale():
    """Discounted expected terminal price equals the forward: E[S_T]=S0 e^{(r-q)T}."""
    S_T = simulate_terminal(S, r, q, sigma, T, n_paths=400_000, seed=3)
    expected = S * math.exp((r - q) * T)
    assert np.mean(S_T) == pytest.approx(expected, rel=2e-3)


def test_barrier_in_out_parity():
    """knock-in + knock-out = vanilla, exactly, on identical paths (same seed)."""
    common = dict(S0=S, K=K, T=T, r=r, sigma=sigma, option_type="call",
                  n_steps=100, n_paths=100_000, seed=99)
    up_out = mc_barrier_price(barrier=130, barrier_type="up-and-out", **common)
    up_in = mc_barrier_price(barrier=130, barrier_type="up-and-in", **common)
    # Reference: a down-and-out with barrier at ~0 is never knocked out, so it
    # equals the plain vanilla payoff on those same paths.
    vanilla_on_paths = mc_barrier_price(barrier=1e-9, barrier_type="down-and-out",
                                        **common)
    assert up_out.price + up_in.price == pytest.approx(vanilla_on_paths.price, abs=1e-9)


def test_asian_cheaper_than_european():
    """Averaging damps extremes, so an Asian call is worth less than vanilla."""
    euro = bs_price(S, K, T, r, sigma, "call", q)
    asian = mc_asian_price(S, K, T, r, sigma, "call", q, n_paths=100_000,
                           n_steps=100, seed=5)
    assert asian.price < euro


def test_arithmetic_asian_geq_geometric():
    """Arithmetic mean >= geometric mean, so arithmetic Asian call >= geometric."""
    common = dict(S0=S, K=K, T=T, r=r, sigma=sigma, option_type="call",
                  n_paths=100_000, n_steps=50, seed=11)
    arith = mc_asian_price(average="arithmetic", **common)
    geo = mc_asian_price(average="geometric", **common)
    assert arith.price >= geo.price - 4 * (arith.std_error + geo.std_error)


def test_knockout_cheaper_than_vanilla():
    """A knock-out option can only be worth less than the vanilla it tracks."""
    euro = bs_price(S, K, T, r, sigma, "call", q)
    ko = mc_barrier_price(S, K, T, r, sigma, barrier=130, barrier_type="up-and-out",
                          option_type="call", n_paths=100_000, n_steps=100, seed=13)
    assert ko.price < euro
