"""Self-checks for the shared estimation pipeline."""
import numpy as np
import pandas as pd

from src.market_structure import (calculate_returns, calibrate_hyperparameters,
                                  edge_density, estimate_partial_correlations)


def test_calculate_returns_drops_dead_columns_and_fills_gaps():
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(size=(60, 3)) * 0.01, axis=0)),
                          columns=list("ABC"))
    prices.loc[:10, "B"] = np.nan          # B trades on 49/59 days: above the 50% share
    prices["C"] = 100.0                    # zero variance: dropped
    returns = calculate_returns(prices)
    assert list(returns.columns) == ["A", "B"]
    assert not returns.isna().any().any()  # gaps filled, first row gone
    assert returns.shape[0] == 59


def test_edge_density_orders_factor_strength():
    rng = np.random.default_rng(1)
    n_days, n_tickers, alpha = 300, 20, 0.30
    noise = pd.DataFrame(rng.normal(size=(n_days, n_tickers)))
    factor = pd.DataFrame(np.repeat(rng.normal(size=(n_days, 1)), n_tickers, axis=1))
    weak = edge_density(estimate_partial_correlations(noise + 0.1 * factor, alpha))
    strong = edge_density(estimate_partial_correlations(noise + 2.0 * factor, alpha))
    assert weak < strong
    assert weak < 0.10                     # near-independent: almost everything L1-zeroed


def test_partial_correlation_diagonal_is_zero_and_bounded():
    rng = np.random.default_rng(2)
    returns = pd.DataFrame(rng.normal(size=(200, 10)) * 0.01 + 0.5)
    partial = estimate_partial_correlations(returns, alpha=0.10)
    assert partial.shape == (10, 10)
    assert np.all(np.abs(partial) <= 1.0 + 1e-9)
    assert np.allclose(np.diag(partial), 0.0)


def test_calibrate_hyperparameters_uses_cache(tmp_path, monkeypatch):
    cache = tmp_path / "hyperparameters.json"
    cache.write_text('{"graphical_lasso_alpha": 0.123, "affinity_preference": 0.05}')
    out = calibrate_hyperparameters(pd.DataFrame(), cache=cache)
    assert out["graphical_lasso_alpha"] == 0.123
