"""Monte Carlo option pricing under geometric Brownian motion (GBM).

Why Monte Carlo when we already have Black-Scholes? Because Black-Scholes only
has a closed-form answer for *vanilla* European options. Many real contracts pay
off based on the WHOLE PRICE PATH -- the average price (Asian options), or
whether a barrier was ever touched (barrier options) -- and for those there is
often no formula at all. Monte Carlo prices them by brute force: simulate many
possible future price paths, compute the payoff on each, average, and discount.

The engine here:
  * simulates GBM (the same process Black-Scholes assumes),
  * uses **antithetic variates** for variance reduction (pair each random draw Z
    with -Z; errors partially cancel, so you get a tighter estimate for free),
  * reports a **standard error** and **95% confidence interval** on every price
    (a Monte Carlo estimate is meaningless without its error bars), and
  * is validated against Black-Scholes for the European case (see tests).

Conventions match ``optvol.pricing.black_scholes``: T in years, r and q are
continuously-compounded annual rates, sigma is annualised vol.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optvol.pricing.black_scholes import _normalise_type, CALL, PUT


@dataclass
class MCResult:
    """A Monte Carlo price with its statistical uncertainty."""

    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self):
        """95% confidence interval (price +/- 1.96 standard errors)."""
        half = 1.96 * self.std_error
        return (self.price - half, self.price + half)

    def __repr__(self):
        lo, hi = self.ci95
        return (
            f"MCResult(price={self.price:.4f}, se={self.std_error:.4f}, "
            f"95% CI=[{lo:.4f}, {hi:.4f}], n={self.n_paths:,})"
        )


# ---------------------------------------------------------------------------
# Path simulation
# ---------------------------------------------------------------------------
def simulate_terminal(S0, r, q, sigma, T, n_paths, antithetic=True, seed=None):
    """Simulate terminal prices S_T only (enough for European payoffs).

    Under GBM the terminal price has an exact log-normal distribution, so we can
    draw S_T in a single step without stepping through time:

        S_T = S0 * exp((r - q - 0.5 sigma^2) T + sigma sqrt(T) Z),  Z ~ N(0,1)
    """
    rng = np.random.default_rng(seed)
    drift = (r - q - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T)
    Z = _draw_normals(rng, n_paths, antithetic)
    return S0 * np.exp(drift + diffusion * Z)


def simulate_paths(S0, r, q, sigma, T, n_steps, n_paths, antithetic=True, seed=None):
    """Simulate full GBM price paths (needed for path-dependent payoffs).

    Returns an array of shape (n_paths, n_steps + 1) including the initial price
    in column 0.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    Z = _draw_normals(rng, (n_paths, n_steps), antithetic, axis=0)
    log_increments = drift + diffusion * Z
    log_paths = np.cumsum(log_increments, axis=1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths


def _draw_normals(rng, shape, antithetic, axis=0):
    """Draw standard normals, optionally as antithetic pairs (Z and -Z)."""
    if not antithetic:
        return rng.standard_normal(shape)

    if np.isscalar(shape):
        half = shape // 2
        base = rng.standard_normal(half)
        both = np.concatenate([base, -base])
        if shape % 2:  # odd count -> top up with one more independent draw
            both = np.concatenate([both, rng.standard_normal(1)])
        return both

    # 2D: pair along the path axis (rows).
    n_paths, n_steps = shape
    half = n_paths // 2
    base = rng.standard_normal((half, n_steps))
    both = np.concatenate([base, -base], axis=0)
    if n_paths % 2:
        both = np.concatenate([both, rng.standard_normal((1, n_steps))], axis=0)
    return both


def _summarize(discounted_payoffs, n_paths) -> MCResult:
    """Turn a sample of discounted payoffs into a price + standard error."""
    price = float(np.mean(discounted_payoffs))
    # Standard error of the mean estimator.
    se = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs)))
    return MCResult(price=price, std_error=se, n_paths=n_paths)


# ---------------------------------------------------------------------------
# Pricers
# ---------------------------------------------------------------------------
def mc_european_price(S0, K, T, r, sigma, option_type=CALL, q=0.0,
                      n_paths=100_000, antithetic=True, seed=None) -> MCResult:
    """Monte Carlo price of a vanilla European option.

    This one HAS a closed form (Black-Scholes), so it exists mainly to validate
    the engine: the MC price should agree with ``bs_price`` to within the
    confidence interval. See ``tests/test_monte_carlo.py``.
    """
    opt = _normalise_type(option_type)
    S_T = simulate_terminal(S0, r, q, sigma, T, n_paths, antithetic, seed)
    payoff = np.maximum(S_T - K, 0.0) if opt == CALL else np.maximum(K - S_T, 0.0)
    return _summarize(np.exp(-r * T) * payoff, n_paths)


def mc_asian_price(S0, K, T, r, sigma, option_type=CALL, q=0.0,
                   n_steps=100, n_paths=100_000, antithetic=True, seed=None,
                   average="arithmetic") -> MCResult:
    """Arithmetic-average (Asian) option -- payoff uses the AVERAGE price.

    An Asian call pays max(avg(S) - K, 0), where the average is taken over the
    path. Averaging damps the effect of any single extreme move, so Asian
    options are cheaper than their vanilla equivalents -- and the *arithmetic*
    average has no Black-Scholes formula, which is exactly why we need MC.
    """
    opt = _normalise_type(option_type)
    paths = simulate_paths(S0, r, q, sigma, T, n_steps, n_paths, antithetic, seed)
    # Average over the monitored points after the start (columns 1..n_steps).
    if average == "geometric":
        avg = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
    else:
        avg = np.mean(paths[:, 1:], axis=1)
    payoff = np.maximum(avg - K, 0.0) if opt == CALL else np.maximum(K - avg, 0.0)
    return _summarize(np.exp(-r * T) * payoff, n_paths)


def mc_barrier_price(S0, K, T, r, sigma, barrier, barrier_type,
                     option_type=CALL, q=0.0, n_steps=100, n_paths=100_000,
                     antithetic=True, seed=None) -> MCResult:
    """Barrier option -- payoff depends on whether a barrier is ever touched.

    ``barrier_type`` is one of: 'up-and-out', 'down-and-out', 'up-and-in',
    'down-and-in'. A knock-OUT option is void if the barrier is breached; a
    knock-IN option only becomes active if it is. (in + out = vanilla; this is
    checked in the tests.) Barrier monitoring is discrete at the simulated steps.
    """
    opt = _normalise_type(option_type)
    bt = barrier_type.lower().replace("_", "-")
    if bt not in ("up-and-out", "down-and-out", "up-and-in", "down-and-in"):
        raise ValueError(f"unknown barrier_type {barrier_type!r}")

    paths = simulate_paths(S0, r, q, sigma, T, n_steps, n_paths, antithetic, seed)

    if bt.startswith("up"):
        breached = np.any(paths >= barrier, axis=1)
    else:
        breached = np.any(paths <= barrier, axis=1)

    S_T = paths[:, -1]
    vanilla = np.maximum(S_T - K, 0.0) if opt == CALL else np.maximum(K - S_T, 0.0)

    knock_in = bt.endswith("in")
    active = breached if knock_in else ~breached
    payoff = np.where(active, vanilla, 0.0)
    return _summarize(np.exp(-r * T) * payoff, n_paths)
