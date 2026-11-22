"""Self-checks for the one-factor null simulation."""
import numpy as np
import pandas as pd

from src.one_factor_null import (base_loadings, fit_sector_model, model_mean_corr,
                                 observed_mean_corr, shocks, simulate, simulate_sector,
                                 solve_scale)


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


def test_t_shocks_have_unit_variance_and_fat_tails():
    draws = shocks(np.random.default_rng(3), 400_000, "t")
    assert abs(draws.var() - 1.0) < 0.05
    excess_kurtosis = ((draws - draws.mean()) ** 4).mean() / draws.var() ** 2 - 3
    assert excess_kurtosis > 1.0                      # Gaussian would be ~0


def test_sector_model_reproduces_within_sector_co_movement():
    # six sectors of eight stocks plus ten unlabelled market-only stocks
    rng = np.random.default_rng(4)
    n_days, n_sectors, size = 3000, 6, 8
    market = rng.normal(size=(n_days, 1))
    blocks = [0.8 * market + 0.7 * rng.normal(size=(n_days, 1)) + rng.normal(size=(n_days, size))
              for _ in range(n_sectors)]
    blocks.append(0.8 * market + rng.normal(size=(n_days, 10)))
    panel = np.hstack(blocks)
    names = [f"S{k}_{i}" for k in range(n_sectors) for i in range(size)] + [f"U{i}" for i in range(10)]
    returns = pd.DataFrame(panel, columns=names)
    sectors = {c: c.split("_")[0] for c in names if c.startswith("S")}
    simulated = simulate_sector(fit_sector_model(returns, sectors), rng).to_numpy()

    def within_and_across(values):
        corr = np.corrcoef(values, rowvar=False)
        labelled = n_sectors * size
        same = np.equal.outer(np.arange(labelled) // size, np.arange(labelled) // size)
        np.fill_diagonal(same, False)
        block = corr[:labelled, :labelled]
        return block[same].mean(), block[~same & ~np.eye(labelled, dtype=bool)].mean()

    real_within, real_across = within_and_across(panel)
    sim_within, sim_across = within_and_across(simulated)
    assert real_within - real_across > 0.15
    assert abs(sim_within - real_within) < 0.03
    assert abs(sim_across - real_across) < 0.03
