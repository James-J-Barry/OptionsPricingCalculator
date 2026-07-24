"""Streamlit app: interactive implied-volatility surface & model-vs-market divergence.

Run with:
    streamlit run app.py

Type a ticker -> live option chains are pulled and cleaned, every contract's
implied vol is solved by inverting Black-Scholes, and the results are shown as a
3D surface, smile/skew slices, a term structure, a divergence table, and a
data-quality report.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from optvol.plots import smile_figure, surface_figure, term_structure_figure
from optvol.surface import load_surface

st.set_page_config(page_title="Volatility Surface Engine", layout="wide")


@st.cache_data(show_spinner=False, ttl=300)
def _load(symbol: str, max_expiries: int, min_days: int):
    """Cached surface load. Returns the VolSurface (cached for 5 min)."""
    return load_surface(symbol, max_expiries=max_expiries, min_days=min_days)


# ----------------------------------------------------------------- sidebar
st.sidebar.title("Volatility Surface Engine")
st.sidebar.caption(
    "Pulls live option chains, **inverts** Black-Scholes to recover each "
    "contract's implied volatility, and surfaces where a single-vol model "
    "diverges from the market."
)
symbol = st.sidebar.text_input("Ticker", value="SPY").strip().upper()
max_expiries = st.sidebar.slider("Expiries (spanning maturity)", 3, 12, 6)
min_days = st.sidebar.slider("Skip expiries under N days", 0, 30, 5)
go = st.sidebar.button("Build surface", type="primary")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: yfinance (delayed). Risk-free rate from 13-week T-bill (^IRX). "
    "Surface built from OTM contracts only."
)

# ----------------------------------------------------------------- main
st.title("Implied-Volatility Surface & Model Divergence")

if not (go or symbol):
    st.info("Enter a ticker in the sidebar and click **Build surface**.")
    st.stop()

try:
    with st.spinner(f"Loading and pricing {symbol} option chains..."):
        surf = _load(symbol, max_expiries, min_days)
except Exception as exc:  # noqa: BLE001 -- surface to the user
    st.error(f"Could not build a surface for {symbol!r}: {exc}")
    st.stop()

# --- headline metrics ------------------------------------------------------
ts = surf.term_structure()
atm_front = ts["atm_iv"].iloc[0] if not ts.empty else float("nan")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Spot", f"{surf.spot:,.2f}")
c2.metric("Front-month ATM IV", f"{atm_front:.1%}")
c3.metric("Risk-free (r)", f"{surf.risk_free_rate:.2%}")
c4.metric("Priceable contracts", f"{len(surf.otm()):,}")

# --- 3D surface ------------------------------------------------------------
st.subheader("3D implied-volatility surface")
try:
    st.plotly_chart(surface_figure(surf), use_container_width=True)
except Exception as exc:  # noqa: BLE001
    st.warning(f"Not enough clean points to interpolate a 3D surface ({exc}).")

# --- smile + term structure side by side ----------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Volatility smile / skew")
    st.plotly_chart(smile_figure(surf), use_container_width=True)
    st.caption(
        "Each curve is one expiry. A downward slope = the equity put skew "
        "(crash protection bid up). A flat line is what Black-Scholes assumes."
    )
with right:
    st.subheader("ATM term structure")
    st.plotly_chart(term_structure_figure(surf), use_container_width=True)
    st.caption(
        "How at-the-money vol changes with maturity. Upward slope (contango) is "
        "typical in calm markets; inversion often signals near-term stress."
    )

# --- divergence table ------------------------------------------------------
st.subheader("Largest model-vs-market divergences")
st.caption(
    "Each contract priced at its expiry's ATM vol (what a naive single-vol "
    "Black-Scholes user would do) vs. the real market mid. The gap is the value "
    "of the smile/skew the flat model ignores."
)
otm = surf.otm().copy()
otm["abs_div"] = otm["divergence"].abs()
cols = [
    "expiry", "option_type", "strike", "mid_price", "bs_flat_price",
    "divergence", "divergence_pct", "iv_solved", "delta", "vega",
]
cols = [c for c in cols if c in otm.columns]
table = (
    otm.sort_values("abs_div", ascending=False)[cols]
    .head(20)
    .rename(columns={
        "mid_price": "market", "bs_flat_price": "flat_BS",
        "divergence_pct": "div_%", "iv_solved": "IV",
    })
    .reset_index(drop=True)
)
fmt = {
    "strike": "{:.2f}", "market": "{:.2f}", "flat_BS": "{:.2f}",
    "divergence": "{:+.2f}", "div_%": "{:+.1%}", "IV": "{:.1%}",
    "delta": "{:+.2f}", "vega": "{:.3f}",
}
st.dataframe(
    table.style.format({k: v for k, v in fmt.items() if k in table.columns}),
    use_container_width=True,
)

# --- data quality ----------------------------------------------------------
with st.expander("Data-quality report (why contracts were dropped)"):
    if surf.drop_report:
        dq = (
            pd.Series(surf.drop_report)
            .rename_axis("expiry:reason")
            .reset_index(name="count")
        )
        st.dataframe(dq, use_container_width=True, height=300)
    else:
        st.write("No drops recorded.")
    st.caption(
        "yfinance quotes are noisy: stale marks, crossed markets, blown-out "
        "spreads, and dead contracts are filtered before any modeling."
    )
