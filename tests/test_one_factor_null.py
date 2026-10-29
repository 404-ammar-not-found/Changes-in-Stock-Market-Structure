"""Self-checks for the one-factor null simulation."""
import numpy as np
import pandas as pd

from src.one_factor_null import (base_loadings, model_mean_corr, observed_mean_corr,
                                 simulate, solve_scale)


def test_scale_solver_reproduces_target_for_equal_loadings():
    u = np.ones(30)
    scale = solve_scale(0.35, u)
    assert abs(model_mean_corr(scale, u) - 0.35) < 1e-6


def test_scale_solver_reproduces_target_for_heterogeneous_loadings():
    rng = np.random.default_rng(0)
    u = np.exp(rng.normal(size=50))            # spread of loading profiles
    scale = solve_scale(0.25, u)
    assert abs(model_mean_corr(scale, u) - 0.25) < 1e-6


def test_simulated_panel_matches_target_average_correlation():
    rng = np.random.default_rng(1)
    u = np.linspace(0.5, 1.5, 25)
    scale = solve_scale(0.40, u)
    panel = simulate((500, 25), scale * u, rng)
    assert panel.shape == (500, 25)
    assert abs(observed_mean_corr(panel) - 0.40) < 0.05   # Monte Carlo error only


def test_base_loadings_rank_stocks_by_their_own_co_movement():
    rng = np.random.default_rng(2)
    factor = rng.normal(size=(600, 1))
    high = 1.0 * factor + rng.normal(size=(600, 15)) * 0.5
    low = 0.2 * factor + rng.normal(size=(600, 15)) * 0.5
    panel = pd.DataFrame(np.hstack([high, low]))
    loadings = base_loadings(panel, "heterogeneous")
    assert loadings[:15].mean() > loadings[15:].mean()
