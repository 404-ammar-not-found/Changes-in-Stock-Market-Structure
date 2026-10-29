"""Shared market-structure estimation pipeline.

Single source of truth for the return construction, Graphical Lasso estimation
and edge-density statistic used by the analysis notebook and the null-simulation
scripts. The functions mirror notebook 02 exactly; hyperparameters (L1 penalty,
Affinity Propagation preference) are calibrated once on the pooled sample and
cached, so every consumer uses identical, year-invariant estimators.
"""
from __future__ import annotations

import json
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV, graphical_lasso

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "processed" / "market_data.db"
HYPERPARAMETERS = ROOT / "data" / "processed" / "hyperparameters.json"

RANDOM_STATE = 42
RIDGE = 0.10   # smallest value keeping every year solvable (2011 fails below);
               # yearly density ranking is unchanged from 0.01 to 0.10


def load_price_matrix(min_ticker_coverage: float = 0.05, db: Path = DB) -> pd.DataFrame:
    """Wide close-price panel from the SQLite database.

    Coverage over the WHOLE 2004-2024 panel. At 0.50 this silently deletes every
    stock that did not survive the period - BSC (0.19), MER (0.22), WB (0.22),
    NCC (0.22), CFC (0.19) - reintroducing the survivorship bias that the
    index-membership filtering exists to avoid. Per-window adequacy is enforced
    separately in calculate_returns, so this only needs to drop genuine junk.
    """
    with sqlite3.connect(db) as conn:
        df = pd.read_sql("SELECT ticker, date, close FROM prices", conn, parse_dates=["date"])
    prices = df.pivot(index="date", columns="ticker", values="close").sort_index()
    coverage = prices.notna().mean()
    return prices.loc[:, coverage >= min_ticker_coverage]


def calculate_returns(price_matrix: pd.DataFrame, min_observed_share: float = 0.50) -> pd.DataFrame:
    """Simple daily returns with per-window adequacy enforced column-wise.

    Short-lived constituents (a crisis casualty may trade on only half the year's
    observed dates) are admitted by the >50% observed-share test rather than a
    global survival threshold. The 1e-8 noise breaks the ties and zero columns
    that degenerate panels would otherwise feed the estimators.
    """
    returns = price_matrix.pct_change(fill_method=None).dropna(how="all")

    returns = returns.loc[:, returns.notna().mean() > min_observed_share]
    returns = returns.fillna(0.0)

    returns = returns.loc[:, returns.std() > 1e-4]

    if not returns.empty:
        generator = np.random.default_rng(RANDOM_STATE)

        noise_df = pd.DataFrame(
            generator.normal(0, 1e-8, size=returns.shape),
            index=returns.index,
            columns=returns.columns
        )
        returns = returns + noise_df

    return returns


def standardize_returns(returns: pd.DataFrame) -> np.ndarray:
    values = returns.to_numpy(dtype=float)
    return (values - values.mean(axis=0)) / (values.std(axis=0) + 1e-9)


def estimate_partial_correlations(returns: pd.DataFrame, alpha: float,
                                  ridge: float = RIDGE) -> np.ndarray:
    """Graphical Lasso partial correlations from a returns panel.

    ridge=0.10 is the smallest value that keeps every year solvable (2011 fails
    below it). The relative ordering of yearly densities is unchanged from 0.01
    to 0.10, so this affects conditioning rather than the result.
    """
    standardized_values = standardize_returns(returns)
    empirical_correlation = np.corrcoef(standardized_values, rowvar=False)
    n_features = empirical_correlation.shape[0]
    empirical_correlation = ((1.0 - ridge) * empirical_correlation
                             + ridge * np.eye(n_features))
    _, precision_matrix = graphical_lasso(empirical_correlation, alpha=alpha, max_iter=200)
    diagonal_scale = np.sqrt(np.diag(precision_matrix))
    partial_correlations = precision_matrix / diagonal_scale[:, None] / diagonal_scale[None, :]
    np.fill_diagonal(partial_correlations, 0.0)
    return partial_correlations


def edge_density(partial_correlations: np.ndarray) -> float:
    """Share of stock pairs with a surviving conditional dependency.

    Because alpha is fixed across years, this is comparable year to year -
    unlike drawn network edges, which are capped at top-k per node and so are
    constant by construction.
    """
    off_diagonal = partial_correlations[np.triu_indices_from(partial_correlations, k=1)]
    return float(np.mean(np.abs(off_diagonal) > 1e-6))


def density_of_returns(returns: pd.DataFrame, alpha: float) -> float:
    return edge_density(estimate_partial_correlations(returns, alpha))


def calibrate_hyperparameters(price_matrix: pd.DataFrame, cache: Path = HYPERPARAMETERS,
                              use_cache: bool = True) -> dict:
    """Calibrate the L1 penalty and AP preference once, on the pooled sample.

    Both are data-dependent under sklearn defaults - refit per year, each year
    gets a different estimator and cross-year comparisons partly measure the
    estimator rather than the market. One CV fit on the pooled 2004-2024 sample
    picks both; the result is cached so the (slow) CV runs exactly once.
    """
    if use_cache and cache.exists():
        return json.loads(cache.read_text())

    pooled_returns = calculate_returns(price_matrix)
    with warnings.catch_warnings():
        # the CV grid probes non-SPD candidates on its way; that is expected
        warnings.simplefilter("ignore")
        pooled_cv = GraphicalLassoCV(n_jobs=-1).fit(standardize_returns(pooled_returns))
    pooled_correlation = pooled_returns.corr().fillna(0.0).to_numpy()
    off_diagonal = pooled_correlation[np.triu_indices_from(pooled_correlation, k=1)]

    hyperparameters = {
        "graphical_lasso_alpha": float(pooled_cv.alpha_),
        "affinity_preference": float(np.median(off_diagonal)),
        "n_pooled_days": int(pooled_returns.shape[0]),
        "n_pooled_tickers": int(pooled_returns.shape[1]),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(hyperparameters, indent=2))
    return hyperparameters
