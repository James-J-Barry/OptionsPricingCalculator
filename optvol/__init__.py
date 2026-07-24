"""optvol — a volatility surface & model-divergence engine.

Phase 1: clean pricing core (Black-Scholes + Greeks), an implied-volatility
solver that *inverts* the market price (rather than consuming the market's own
IV), and a data layer that pulls and cleans live option chains.
"""

__version__ = "0.1.0"
