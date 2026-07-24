"""Tie the pieces together: from a cleaned chain to implied vols, Greeks, and a
model-vs-market divergence view.

The divergence story for Phase 1: we solve each contract's OWN implied vol from
its market mid (so the smile emerges), then we also price every contract with a
single FLAT volatility -- the ATM implied vol -- as a naive Black-Scholes model
would. The gap between flat-vol BS and the market price is exactly the value of
the volatility smile/skew, i.e. the market-implied information a single-number
model throws away.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from optvol.data import ChainSnapshot
from optvol.pricing.black_scholes import bs_price, bs_greeks
from optvol.pricing.implied_vol import implied_vol


def enrich_chain(snap: ChainSnapshot) -> pd.DataFrame:
    """Return one tidy DataFrame with solved IV, Greeks, and divergence cols."""
    df = snap.combined.copy()
    if df.empty:
        return df

    S, r, q, T = snap.spot, snap.risk_free_rate, snap.dividend_yield, snap.T

    # 1. Solve each contract's implied vol from its market mid price.
    solved = []
    for _, row in df.iterrows():
        solved.append(
            implied_vol(
                market_price=row["mid_price"],
                S=S,
                K=row["strike"],
                T=T,
                r=r,
                option_type=row["option_type"],
                q=q,
            )
        )
    df["iv_solved"] = solved

    # Drop contracts we couldn't invert (unpriceable / arbitrage-violating).
    df = df[df["iv_solved"].notna()].reset_index(drop=True)
    if df.empty:
        return df

    # 1b. Flag out-of-the-money contracts. The IV surface is conventionally
    # built from OTM options: they carry all the time value (so vega is
    # meaningful and the inversion is well-conditioned), whereas deep ITM
    # quotes are stale, near-intrinsic, and produce garbage IVs. We also drop
    # absurd solved IVs as a final noise guard.
    is_call = df["option_type"] == "call"
    df["is_otm"] = np.where(is_call, df["strike"] >= S, df["strike"] <= S)
    df = df[df["iv_solved"] < 3.0].reset_index(drop=True)  # <300% vol
    if df.empty:
        return df

    # 2. Sanity column: our solved IV vs yfinance's reported IV.
    if "impliedVolatility" in df:
        df["iv_market_reported"] = df["impliedVolatility"]
        df["iv_abs_diff"] = (df["iv_solved"] - df["impliedVolatility"]).abs()

    # 3. Greeks at each contract's own solved IV.
    greeks = bs_greeks(
        S, df["strike"].values, T, r, df["iv_solved"].values,
        # bs_greeks needs a single option_type; compute per-side below instead.
        option_type="call", q=q,
    )
    # Greeks depend on call/put for delta/theta/rho, so compute per row group.
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        df[name] = np.nan
    for opt in ("call", "put"):
        mask = df["option_type"] == opt
        if not mask.any():
            continue
        g = bs_greeks(
            S, df.loc[mask, "strike"].values, T, r,
            df.loc[mask, "iv_solved"].values, option_type=opt, q=q,
        )
        for name in ("delta", "gamma", "vega", "theta", "rho"):
            df.loc[mask, name] = g[name]

    # 4. Flat-vol model: price everything at the ATM implied vol.
    atm_iv = _atm_iv(df, S)
    df["iv_flat"] = atm_iv
    df["bs_flat_price"] = bs_price(
        S, df["strike"].values, T, r, atm_iv,
        option_type="call", q=q,
    )
    # Recompute per side for correct put pricing.
    for opt in ("call", "put"):
        mask = df["option_type"] == opt
        if mask.any():
            df.loc[mask, "bs_flat_price"] = bs_price(
                S, df.loc[mask, "strike"].values, T, r, atm_iv,
                option_type=opt, q=q,
            )
    df["divergence"] = df["bs_flat_price"] - df["mid_price"]
    df["divergence_pct"] = df["divergence"] / df["mid_price"].replace(0, np.nan)

    return df


def _atm_iv(df: pd.DataFrame, spot: float) -> float:
    """ATM implied vol = solved IV of the contract closest to the money."""
    idx = (df["strike"] - spot).abs().idxmin()
    return float(df.loc[idx, "iv_solved"])


def summary(df: pd.DataFrame, snap: ChainSnapshot) -> str:
    """Human-readable one-screen summary for the CLI."""
    if df.empty:
        return "No priceable contracts after cleaning."
    atm_iv = df["iv_flat"].iloc[0]
    lines = [
        f"{snap.symbol}  spot={snap.spot:.2f}  expiry={snap.expiry}  "
        f"T={snap.T:.3f}y  r={snap.risk_free_rate:.2%}  q={snap.dividend_yield:.2%}",
        f"ATM implied vol (solved): {atm_iv:.2%}",
        f"Priceable contracts: {len(df)}",
        "",
        "Largest flat-vol model vs market divergences across OTM contracts",
        "(the smile a single vol misses):",
    ]
    otm = df[df["is_otm"]] if "is_otm" in df else df
    if otm.empty:
        otm = df
    worst = otm.reindex(otm["divergence"].abs().sort_values(ascending=False).index).head(6)
    for _, row in worst.iterrows():
        lines.append(
            f"  {row['option_type']:>4} K={row['strike']:<8.2f} "
            f"mkt={row['mid_price']:>7.2f}  flat-BS={row['bs_flat_price']:>7.2f}  "
            f"diff={row['divergence']:>+7.2f}  IV={row['iv_solved']:.1%}"
        )
    return "\n".join(lines)
