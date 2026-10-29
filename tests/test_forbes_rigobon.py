"""Self-checks for the Forbes-Rigobon volatility adjustment."""
import numpy as np
import pandas as pd

from src.forbes_rigobon import (adjust_correlation, market_model_fit,
                                model_mean_corr, off_diagonal)


def factor_panel(rng, n_days, n_tickers, factor_scale, loadings):
    return (loadings[None, :] * factor_scale * rng.normal(size=(n_days, 1))
            + rng.normal(size=(n_days, n_tickers)))


def test_market_model_recovers_loadings_and_volatility():
    rng = np.random.default_rng(0)
    loadings = rng.uniform(0.7, 1.3, size=30)
    panel = pd.DataFrame(factor_panel(rng, 2500, 30, 1.0, loadings))
    beta, idio, sigma_m = market_model_fit(panel)
    # regression on the equal-weighted market attenuates loadings slightly
    # (the proxy carries mean idiosyncratic noise), so allow generous slack
    assert np.corrcoef(beta, loadings)[0, 1] > 0.95
    assert np.all(idio > 0)
    assert 0.5 < sigma_m < 1.5            # market vol ~ mean loading, factor vol 1


def test_model_mean_corr_matches_empirical_in_a_one_factor_world():
    rng = np.random.default_rng(1)
    loadings = rng.uniform(0.5, 1.2, size=25)
    panel = pd.DataFrame(factor_panel(rng, 2500, 25, 1.0, loadings))
    empirical = off_diagonal(np.corrcoef(panel.to_numpy(), rowvar=False)).mean()
    beta, idio, sigma_m = market_model_fit(panel)
    assert abs(model_mean_corr(beta, sigma_m, idio) - empirical) < 0.03


def test_counterfactual_recovers_calm_correlation_after_regime_switch():
    rng = np.random.default_rng(2)
    loadings = rng.uniform(0.7, 1.3, size=30)
    calm = pd.DataFrame(factor_panel(rng, 3000, 30, 1.0, loadings))
    crisis = pd.DataFrame(factor_panel(rng, 3000, 30, 2.5, loadings))
    truth = off_diagonal(np.corrcoef(calm.to_numpy(), rowvar=False)).mean()
    measured = off_diagonal(np.corrcoef(crisis.to_numpy(), rowvar=False)).mean()

    beta, idio, _ = market_model_fit(crisis)
    sigma_m_calm = market_model_fit(calm)[2]
    counterfactual = model_mean_corr(beta, sigma_m_calm, idio)

    assert measured > truth + 0.10          # the artefact is real and material
    assert abs(counterfactual - truth) < 0.05


def test_proportional_volatility_scaling_leaves_correlation_unchanged():
    """Factor and idio rising together is pure rescaling: measured correlation is
    invariant, and only the both-volatilities counterfactual respects that."""
    rng = np.random.default_rng(4)
    loadings = rng.uniform(0.7, 1.3, size=30)
    calm = pd.DataFrame(factor_panel(rng, 3000, 30, 1.0, loadings))
    scaled = pd.DataFrame(loadings[None, :] * 2.5 * rng.normal(size=(3000, 1))
                          + 2.5 * rng.normal(size=(3000, 30)))

    truth = off_diagonal(np.corrcoef(calm.to_numpy(), rowvar=False)).mean()
    measured = off_diagonal(np.corrcoef(scaled.to_numpy(), rowvar=False)).mean()
    assert abs(measured - truth) < 0.05

    beta, idio, _ = market_model_fit(scaled)
    sigma_m_calm = market_model_fit(calm)[2]
    factor_only = model_mean_corr(beta, sigma_m_calm, idio)
    both = model_mean_corr(beta, sigma_m_calm, idio / 2.5 ** 2)
    assert factor_only < truth - 0.05          # resetting the factor alone over-deflates
    assert abs(both - truth) < 0.05            # resetting both recovers the calm level


def test_pairwise_formula_deflates_magnitude_and_spares_perfect_corr():
    rng = np.random.default_rng(3)
    corr = np.corrcoef(rng.normal(size=(1200, 8)), rowvar=False)
    adjusted = adjust_correlation(corr, np.full(8, 4.0))
    off = ~np.eye(8, dtype=bool)
    assert np.all(np.abs(adjusted[off]) <= np.abs(corr[off]) + 1e-12)
    perfect = adjust_correlation(np.array([[1.0, 0.99], [0.99, 1.0]]), np.array([9.0, 9.0]))
    assert perfect[0, 1] > 0.9              # near-perfect co-movement survives any nu
