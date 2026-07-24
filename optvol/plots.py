"""Reusable Plotly figures for the volatility surface.

Kept separate from the Streamlit app so the same plotting code can be used in
notebooks, tests, or any other front-end. Every function returns a
``plotly.graph_objects.Figure``.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from optvol.surface import VolSurface


def surface_figure(surf: VolSurface, n_money: int = 40, n_t: int = 30) -> go.Figure:
    """3D implied-volatility surface over (moneyness, days-to-expiry)."""
    m_axis, t_axis, iv = surf.grid(n_money=n_money, n_t=n_t)
    days = t_axis * 365.0
    fig = go.Figure(
        data=[
            go.Surface(
                x=m_axis,
                y=days,
                z=iv * 100.0,  # show IV in percent
                colorscale="Viridis",
                colorbar=dict(title="IV (%)"),
                hovertemplate=(
                    "moneyness=%{x:.2f}<br>days=%{y:.0f}<br>IV=%{z:.1f}%<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title=f"{surf.symbol} implied-volatility surface  (spot={surf.spot:.2f})",
        scene=dict(
            xaxis_title="moneyness (K / S)",
            yaxis_title="days to expiry",
            zaxis_title="implied vol (%)",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
    )
    return fig


def smile_figure(surf: VolSurface, expiries=None) -> go.Figure:
    """Overlaid smile/skew slices: IV vs strike for one or more expiries."""
    if expiries is None:
        expiries = surf.expiries
    fig = go.Figure()
    for exp in expiries:
        s = surf.smile(exp)
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=s["strike"],
                y=s["iv_solved"] * 100.0,
                mode="markers+lines",
                name=f"{exp} ({int(s['days_to_expiry'].iloc[0])}d)",
                hovertemplate="K=%{x}<br>IV=%{y:.1f}%<extra></extra>",
            )
        )
    fig.add_vline(
        x=surf.spot, line_dash="dot", line_color="gray",
        annotation_text="spot", annotation_position="top",
    )
    fig.update_layout(
        title=f"{surf.symbol} volatility smile / skew by expiry",
        xaxis_title="strike (K)",
        yaxis_title="implied vol (%)",
        height=450,
        legend_title="expiry",
    )
    return fig


def term_structure_figure(surf: VolSurface) -> go.Figure:
    """ATM implied vol vs maturity -- the term structure of volatility."""
    ts = surf.term_structure()
    fig = go.Figure(
        go.Scatter(
            x=ts["days"],
            y=ts["atm_iv"] * 100.0,
            mode="markers+lines",
            hovertemplate="%{x} days<br>ATM IV=%{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{surf.symbol} ATM implied-vol term structure",
        xaxis_title="days to expiry",
        yaxis_title="ATM implied vol (%)",
        height=400,
    )
    return fig
