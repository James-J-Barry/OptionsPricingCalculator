"""Tests for the Phase 4 surface models: SVI parametric fit and the ML baseline.

No network access required -- everything is fit against synthetic smiles with
known ground truth, so we can assert the fitted parameters (not just the
predictions) are recovered.
"""
import numpy as np
import pytest

from optvol.surface_models.svi import SVIParams, fit_svi_slice
from optvol.surface_models.ml import fit_ml_surface


# ---------------------------------------------------------------------------
# SVI
# ---------------------------------------------------------------------------
def _synthetic_smile(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.15,
                      T=0.5, S=100.0, r=0.03, q=0.0, n=25,
                      k_lo=-0.4, k_hi=0.4, noise=0.0, seed=0):
    """Build a smile from KNOWN SVI parameters, for round-trip testing."""
    F = S * np.exp((r - q) * T)
    k = np.linspace(k_lo, k_hi, n)
    strikes = F * np.exp(k)
    w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
    iv = np.sqrt(w / T)
    if noise:
        rng = np.random.default_rng(seed)
        iv = iv + rng.normal(0, noise, n)
    return strikes, iv, dict(a=a, b=b, rho=rho, m=m, sigma=sigma, T=T, S=S, r=r, q=q)


def test_svi_recovers_known_parameters_noiseless():
    strikes, iv, truth = _synthetic_smile()
    fit = fit_svi_slice(strikes, iv, truth["S"], truth["r"], truth["q"], truth["T"])
    assert fit.a == pytest.approx(truth["a"], abs=1e-3)
    assert fit.b == pytest.approx(truth["b"], abs=1e-3)
    assert fit.rho == pytest.approx(truth["rho"], abs=1e-2)
    assert fit.m == pytest.approx(truth["m"], abs=1e-2)
    assert fit.sigma == pytest.approx(truth["sigma"], abs=1e-2)
    assert fit.rmse < 1e-3


def test_svi_fit_is_robust_to_noise():
    """With realistic quote noise, the fit should still track the true smile
    closely even if individual parameters drift a bit."""
    strikes, iv, truth = _synthetic_smile(noise=0.01, seed=1)
    fit = fit_svi_slice(strikes, iv, truth["S"], truth["r"], truth["q"], truth["T"])
    # Predicted IV should track the NOISELESS truth within noise-scale error.
    _, true_iv, _ = _synthetic_smile(noise=0.0)
    pred_iv = fit.implied_vol(strikes)
    assert np.sqrt(np.mean((pred_iv - true_iv) ** 2)) < 0.02


def test_svi_is_arbitrage_free_on_well_behaved_smile():
    strikes, iv, truth = _synthetic_smile()
    fit = fit_svi_slice(strikes, iv, truth["S"], truth["r"], truth["q"], truth["T"])
    assert fit.is_arbitrage_free()


def test_svi_flags_bad_parameters_as_not_arbitrage_free():
    # b < 0 and |rho| >= 1 both violate the no-arbitrage conditions directly.
    bad1 = SVIParams(a=0.02, b=-0.1, rho=0.0, m=0.0, sigma=0.1, T=0.5,
                     forward=100.0, rmse=0.0)
    bad2 = SVIParams(a=0.02, b=0.1, rho=1.5, m=0.0, sigma=0.1, T=0.5,
                     forward=100.0, rmse=0.0)
    bad3 = SVIParams(a=-1.0, b=0.01, rho=0.0, m=0.0, sigma=0.01, T=0.5,
                     forward=100.0, rmse=0.0)  # min variance goes negative
    assert not bad1.is_arbitrage_free()
    assert not bad2.is_arbitrage_free()
    assert not bad3.is_arbitrage_free()


def test_svi_requires_minimum_points():
    with pytest.raises(ValueError):
        fit_svi_slice([90, 100, 110], [0.2, 0.19, 0.21], 100.0, 0.03, 0.0, 0.5)


# ---------------------------------------------------------------------------
# ML surface model
# ---------------------------------------------------------------------------
def _synthetic_surface(n=300, noise=0.005, seed=0):
    rng = np.random.default_rng(seed)
    moneyness = rng.uniform(0.8, 1.2, n)
    T = rng.uniform(0.05, 1.5, n)
    iv_true = 0.18 + 0.25 * (1 - moneyness) + 0.04 * (1 - np.exp(-T / 0.5))
    iv = iv_true + rng.normal(0, noise, n)
    return moneyness, T, iv, iv_true


def test_ml_model_fits_synthetic_surface_well():
    moneyness, T, iv, iv_true = _synthetic_surface()
    res = fit_ml_surface(moneyness, T, iv)
    # Both train and cross-validated error should be small and close to the
    # injected noise level (0.005) -- if cv_rmse were huge, the model failed
    # to learn the underlying shape at all.
    assert res.train_rmse < 0.02
    assert res.cv_rmse < 0.03


def test_ml_model_cv_rmse_not_wildly_worse_than_train():
    """A sane amount of overfitting: CV error shouldn't explode vs. train error."""
    moneyness, T, iv, _ = _synthetic_surface(n=400)
    res = fit_ml_surface(moneyness, T, iv)
    assert res.cv_rmse < res.train_rmse * 5


def test_ml_model_predicts_reasonable_values():
    moneyness, T, iv, _ = _synthetic_surface()
    res = fit_ml_surface(moneyness, T, iv)
    pred = res.predict(np.array([1.0]), np.array([0.5]))
    # Should be in a sane implied-vol range, not extrapolated nonsense.
    assert 0.05 < pred[0] < 0.6


def test_ml_model_requires_minimum_points():
    with pytest.raises(ValueError):
        fit_ml_surface([1.0, 1.1], [0.5, 0.5], [0.2, 0.21])
