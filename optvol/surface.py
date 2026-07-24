"""Multi-expiry implied-volatility surface.

Phase 1 worked with a single expiry. The volatility *surface* stitches many
expiries together: solving the implied vol of every (strike, expiry) contract
gives a 2D field IV(strike, maturity). Two cross-sections of that surface are
what traders actually look at:

  * the **smile/skew**  -- IV vs strike at a FIXED expiry (how vol varies across
    strikes), and
  * the **term structure** -- ATM IV vs maturity (how vol varies with time).

This module loads several expiries, reuses the Phase-1 `enrich_chain` pipeline
on each, and stacks the results into one long, tidy DataFrame plus a regular
interpolated grid suitable for a 3D plot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from optvol.analyze import enrich_chain
from optvol.data import load_chain, list_expiries, _years_to_expiry


def _select_expiries(all_exp: List[str], max_expiries: int, min_days: int) -> List[str]:
    """Choose expiries that SPAN maturity, skipping near-0-DTE.

    Tickers like SPY list daily expiries, so naively taking the first N gives a
    cramped surface only days wide. We drop anything under ``min_days`` (where
    IV is noisy) and then sample evenly across the remaining range so the
    surface covers weeks/months of term structure.
    """
    dated = [(e, _years_to_expiry(e)) for e in all_exp]
    usable = [e for e, t in dated if t >= min_days / 365.0]
    if not usable:
        usable = [e for e, _ in dated]
    if len(usable) <= max_expiries:
        return usable
    idx = np.linspace(0, len(usable) - 1, max_expiries).round().astype(int)
    return [usable[i] for i in sorted(set(idx))]


@dataclass
class VolSurface:
    """A cleaned, IV-solved multi-expiry surface for one ticker."""

    symbol: str
    spot: float
    risk_free_rate: float
    dividend_yield: float
    points: pd.DataFrame                       # long form: one row per contract
    expiries: List[str] = field(default_factory=list)
    drop_report: Dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------ views
    def otm(self) -> pd.DataFrame:
        """OTM-only points -- the well-conditioned set used for the surface."""
        df = self.points
        return df[df["is_otm"]] if "is_otm" in df else df

    def smile(self, expiry: str) -> pd.DataFrame:
        """One expiry's smile slice, sorted by strike (OTM points)."""
        d = self.otm()
        return d[d["expiry"] == expiry].sort_values("strike")

    def term_structure(self) -> pd.DataFrame:
        """ATM implied vol vs maturity (one row per expiry).

        ATM is taken as the OTM contract whose strike is closest to spot.
        """
        rows = []
        for exp, grp in self.otm().groupby("expiry"):
            i = (grp["strike"] - self.spot).abs().idxmin()
            rows.append(
                {
                    "expiry": exp,
                    "T": float(grp.loc[i, "T"]),
                    "days": int(round(float(grp.loc[i, "T"]) * 365)),
                    "atm_iv": float(grp.loc[i, "iv_solved"]),
                }
            )
        return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)

    def grid(self, n_money: int = 40, n_t: int = 30, money_lo: float = 0.8,
             money_hi: float = 1.2):
        """Interpolate scattered IV points onto a regular (moneyness, T) grid.

        Returns (moneyness_axis, T_axis, IV_grid) where IV_grid has shape
        (n_t, n_money). Raw option quotes are scattered and ragged across
        strikes/expiries, so we interpolate them onto an even mesh that a 3D
        surface plot can render. Gaps outside the data hull are left as NaN.
        """
        d = self.otm().dropna(subset=["moneyness", "T", "iv_solved"])
        if len(d) < 4:
            raise ValueError("Not enough points to build a surface grid.")

        m_axis = np.linspace(money_lo, money_hi, n_money)
        t_axis = np.linspace(d["T"].min(), d["T"].max(), n_t)
        MM, TT = np.meshgrid(m_axis, t_axis)

        iv = griddata(
            points=d[["moneyness", "T"]].values,
            values=d["iv_solved"].values,
            xi=(MM, TT),
            method="linear",
        )
        return m_axis, t_axis, iv


def load_surface(
    symbol: str,
    max_expiries: int = 6,
    expiries: Optional[List[str]] = None,
    min_days: int = 5,
) -> VolSurface:
    """Load and IV-solve several expiries into a single VolSurface.

    Parameters
    ----------
    symbol : str
        Ticker, e.g. "SPY".
    max_expiries : int
        How many expiries to include, sampled to SPAN the maturity range
        (more = richer surface but slower and noisier in the far months).
    expiries : list of str, optional
        Explicit expiry dates to load. Overrides ``max_expiries``/``min_days``.
    min_days : int
        Skip expiries closer than this many days (near-0-DTE IV is unstable).
    """
    all_exp = list_expiries(symbol)
    if not all_exp:
        raise ValueError(f"No listed options for {symbol!r}.")
    chosen = (
        expiries if expiries is not None
        else _select_expiries(all_exp, max_expiries, min_days)
    )

    frames: List[pd.DataFrame] = []
    drop_report: Dict[str, int] = {}
    spot = r = q = None
    loaded: List[str] = []

    for exp in chosen:
        try:
            snap = load_chain(symbol, exp)
        except Exception:
            continue
        df = enrich_chain(snap)
        if df.empty:
            continue
        df = df.copy()
        df["expiry"] = exp
        df["T"] = snap.T
        df["days_to_expiry"] = int(round(snap.T * 365))
        frames.append(df)
        loaded.append(exp)
        for k, v in snap.drop_report.items():
            drop_report[f"{exp}:{k}"] = v
        # Spot/rate/dividend are the same across expiries; capture once.
        spot, r, q = snap.spot, snap.risk_free_rate, snap.dividend_yield

    if not frames:
        raise ValueError(
            f"No priceable contracts for {symbol} across the selected expiries."
        )

    points = pd.concat(frames, ignore_index=True, sort=False)
    return VolSurface(
        symbol=symbol,
        spot=float(spot),
        risk_free_rate=float(r),
        dividend_yield=float(q),
        points=points,
        expiries=loaded,
        drop_report=drop_report,
    )
