"""Self-checks for the sparse-sampling bias machinery.

Subsampling a pure one-factor panel on every other day must not move its
mean correlation much; that invariance is what lets real sparse-year date
patterns be imposed on complete years to measure the bias.
"""
import numpy as np
import pandas as pd

from src.sparse_bias_check import mean_corr


def one_factor_panel(rng, n_days=250, n_tickers=20):
    factor = rng.normal(size=(n_days, 1))
    log_prices = np.cumsum(factor + rng.normal(size=(n_days, n_tickers)) * 0.5, axis=0)
    return pd.DataFrame(100 * np.exp(log_prices * 0.01))


def test_thinning_a_one_factor_panel_moves_correlation_little():
    rng = np.random.default_rng(0)
    prices = one_factor_panel(rng)
    full, thinned = mean_corr(prices), mean_corr(prices.iloc[::2])
    assert abs(full - thinned) < 0.15
    assert 0.0 < thinned < 1.0


def test_mean_corr_rises_with_factor_loading():
    rng = np.random.default_rng(1)
    weak = mean_corr(one_factor_panel(rng, n_tickers=20))
    strong_panel = one_factor_panel(rng, n_tickers=20)  # same idiosyncratic scale
    # rebuild with halved idiosyncratic noise -> tighter co-movement
    factor = rng.normal(size=(250, 1))
    tight = pd.DataFrame(100 * np.exp(np.cumsum(factor + rng.normal(size=(250, 20)) * 0.25, axis=0) * 0.01))
    assert mean_corr(tight) > weak
