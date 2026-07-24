"""A learned (ML) model of the implied-volatility surface.

SVI (see ``svi.py``) fits each expiry independently with a hand-chosen
5-parameter functional form. This module instead learns IV directly as a
function of the surface's natural coordinates,

    IV = f(moneyness, time-to-expiry)

with a **gradient-boosted tree ensemble** (``sklearn``'s
``GradientBoostingRegressor``). Trees need no assumption about the smile's
shape and can pick up asymmetric, non-smooth structure that a fixed
parametric form might miss -- at the cost of no arbitrage guarantees and a
real risk of overfitting scattered/noisy quotes.

This is the deliberately "un-clever" ML baseline for the Phase 4 benchmark:
plausible enough to be worth comparing, simple enough that a reviewer can see
exactly what it does and why it might overfit relative to SVI.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict


@dataclass
class SurfaceMLModel:
    """A fitted moneyness/maturity -> IV regressor, plus its CV diagnostics."""

    model: GradientBoostingRegressor
    train_rmse: float
    cv_rmse: float  # out-of-fold RMSE -- the honest generalization estimate
    n_samples: int

    def predict(self, moneyness, T):
        """Predict IV at arbitrary (moneyness, T) points."""
        moneyness = np.atleast_1d(np.asarray(moneyness, dtype=float))
        T = np.atleast_1d(np.asarray(T, dtype=float))
        if len(T) == 1 and len(moneyness) > 1:
            T = np.full_like(moneyness, T[0])
        X = np.column_stack([moneyness, T])
        return self.model.predict(X)


def fit_ml_surface(
    moneyness, T, iv, n_estimators: int = 200, max_depth: int = 3,
    learning_rate: float = 0.05, n_splits: int = 5, random_state: int = 0,
) -> SurfaceMLModel:
    """Fit a gradient-boosted regressor for IV(moneyness, T).

    Reports both the in-sample RMSE (optimistic -- the model has seen these
    points) and a K-fold **cross-validated** RMSE (the honest estimate of how
    well it generalizes to strikes/expiries it wasn't fit on). The gap between
    the two is the overfitting signal to watch for in the benchmark.
    """
    moneyness = np.asarray(moneyness, dtype=float)
    T = np.asarray(T, dtype=float)
    iv = np.asarray(iv, dtype=float)
    mask = np.isfinite(moneyness) & np.isfinite(T) & np.isfinite(iv)
    X = np.column_stack([moneyness[mask], T[mask]])
    y = iv[mask]
    if len(y) < 10:
        raise ValueError("Need at least 10 clean points to fit the ML surface model.")

    model = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=random_state,
    )
    model.fit(X, y)
    train_pred = model.predict(X)
    train_rmse = float(np.sqrt(np.mean((train_pred - y) ** 2)))

    n_splits = max(2, min(n_splits, len(y) // 5)) if len(y) >= 10 else 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_model = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=random_state,
    )
    cv_pred = cross_val_predict(cv_model, X, y, cv=kf)
    cv_rmse = float(np.sqrt(np.mean((cv_pred - y) ** 2)))

    return SurfaceMLModel(model=model, train_rmse=train_rmse, cv_rmse=cv_rmse,
                          n_samples=len(y))
