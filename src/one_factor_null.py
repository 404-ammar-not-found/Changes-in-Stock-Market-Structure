"""Is the Graphical Lasso edge density anything more than average correlation?

For each usable year, simulate one-factor markets - beta and nothing else -
matched on the year's number of trading days, number of tickers, per-ticker
loading profile and average pairwise correlation, run the identical estimation
pipeline at the same fixed pooled alpha, and compare the observed edge density
against the null distribution. Density significantly above the null means
conditional structure beyond the market factor; no excess confirms the
deflationary reading that the network merely restates the correlation level.

Run: python src/one_factor_null.py [--sims 200] [--model heterogeneous]
Writes: figures/one_factor_null.csv, figures/one_factor_null.png
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

from src.market_structure import (DB, RIDGE, calculate_returns, calibrate_hyperparameters,
                                  density_of_returns, edge_density,
                                  estimate_partial_correlations, load_price_matrix)

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "figures" / "one_factor_null.csv"
OUT_FIG = ROOT / "figures" / "one_factor_null.png"

MIN_WINDOW_OBSERVATIONS = 50   # excludes the unrecoverable 2019/2020 windows
MIN_WINDOW_TICKERS = 10


# --------------------------------------------------------------------------
# The null model: r_it = b_i * f_t + e_it, e ~ N(0, 1), f ~ N(0, 1)
# corr(i, j) = b_i b_j / sqrt((1 + b_i^2)(1 + b_j^2))
# --------------------------------------------------------------------------

def model_mean_corr(scale: float, base_loadings: np.ndarray) -> float:
    """Average pairwise correlation implied by loadings b_i = scale * u_i."""
    b = scale * base_loadings
    corr = np.outer(b, b) / np.sqrt(np.outer(1 + b ** 2, 1 + b ** 2))
    return float(corr[np.triu_indices_from(corr, k=1)].mean())


def solve_scale(target_mean_corr: float, base_loadings: np.ndarray) -> float:
    """Bisection on the loading scale so the model reproduces the target level."""
    low, high = 1e-6, 1e4
    for _ in range(80):
        mid = np.sqrt(low * high)          # geometric bisection: scale is positive
        if model_mean_corr(mid, base_loadings) < target_mean_corr:
            low = mid
        else:
            high = mid
    return np.sqrt(low * high)


def base_loadings(returns: pd.DataFrame, model: str) -> np.ndarray:
    """Per-ticker loading profile for the null.

    'heterogeneous' gives each stock the loading that reproduces its observed
    average correlation with the rest of the panel, so the null also matches
    the cross-sectional dispersion of co-movement; 'equal' is the textbook
    single-beta market. The scale that matches the year's average correlation
    is solved separately.
    """
    corr = np.corrcoef(returns.to_numpy(dtype=float), rowvar=False)
    if model == "equal":
        return np.ones(corr.shape[0])
    m = (corr.sum(axis=1) - np.diag(corr)) / (corr.shape[0] - 1)
    m = np.clip(m, 1e-3, 0.95)
    return np.sqrt(m / (1.0 - m))


def simulate(returns_shape: tuple[int, int], loadings: np.ndarray, rng) -> pd.DataFrame:
    n_days, n_tickers = returns_shape
    factor = rng.normal(size=(n_days, 1))
    idiosyncratic = rng.normal(size=(n_days, n_tickers))
    return pd.DataFrame(loadings[None, :] * factor + idiosyncratic)


def observed_mean_corr(returns: pd.DataFrame) -> float:
    corr = np.corrcoef(returns.to_numpy(dtype=float), rowvar=False)
    return float(corr[np.triu_indices_from(corr, k=1)].mean())


# --------------------------------------------------------------------------
# Yearly driver
# --------------------------------------------------------------------------

def usable_years(price_matrix: pd.DataFrame, start_year: int = 2004, end_year: int = 2024):
    """Yield (year, returns) exactly as the animation notebook selects windows."""
    for year in range(start_year, end_year + 1):
        window = price_matrix.loc[price_matrix.index.year == year]
        if len(window) < MIN_WINDOW_OBSERVATIONS:
            continue
        returns = calculate_returns(window)
        if returns.shape[1] < MIN_WINDOW_TICKERS:
            continue
        yield year, returns


def null_densities(returns: pd.DataFrame, alpha: float, sims: int,
                   model: str, seed: int, n_jobs: int = -1) -> list[float]:
    scale = solve_scale(observed_mean_corr(returns), base_loadings(returns, model))
    loadings = scale * base_loadings(returns, model)

    def one(seed_i: int) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*slogdet.*")
            panel = simulate(returns.shape, loadings, np.random.default_rng(seed_i))
            try:
                return edge_density(estimate_partial_correlations(panel, alpha))
            except FloatingPointError:
                # The solver hits its conditioning limit on occasional draws in the
                # highest-correlation years (the README documents the same boundary
                # for the observed 2011 fit). Record and exclude; not silently retry
                # with a different estimator, which would change the null.
                return np.nan

    seeds = np.random.SeedSequence(seed).spawn(sims)
    return Parallel(n_jobs=n_jobs)(delayed(one)(int(s.generate_state(1)[0])) for s in seeds)


def main(sims: int = 200, model: str = "heterogeneous") -> pd.DataFrame:
    price_matrix = load_price_matrix()
    alpha = calibrate_hyperparameters(price_matrix)["graphical_lasso_alpha"]
    print(f"fixed pooled alpha: {alpha:.4f}; null model: {model}; sims/year: {sims}")

    rows = []
    for year, returns in usable_years(price_matrix):
        with warnings.catch_warnings():
            # the observed fits in the highest-correlation years do not fully
            # converge within 200 iterations - documented, and immaterial to the
            # comparison against the null, which runs the identical estimator
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*slogdet.*")
            observed = density_of_returns(returns, alpha)
        null = np.array(null_densities(returns, alpha, sims, model, seed=1000 + year))
        n_failed = int(np.isnan(null).sum())
        null = null[~np.isnan(null)]
        sd = float(null.std())
        if sd < 1e-12:
            # degenerate null (e.g. every beta-only draw zeroed by the L1 penalty):
            # the excess is the entire observed density, and z is infinite
            z = float("inf") if observed > null.mean() else float("nan")
        else:
            z = (observed - null.mean()) / sd
        # add-one so a fully-exceeded null reports 1/(sims+1), not 0
        p = (1 + np.sum(null >= observed)) / (len(null) + 1)
        rows.append({"year": year, "n_days": returns.shape[0], "n_tickers": returns.shape[1],
                     "mean_corr": round(observed_mean_corr(returns), 3),
                     "observed_density": round(observed, 4),
                     "null_mean": round(float(null.mean()), 4),
                     "null_sd": round(float(null.std()), 4),
                     "z": round(float(z), 2), "p_excess": round(float(p), 4),
                     "null_failed": n_failed})
        results = pd.DataFrame(rows).set_index("year")
        results.to_csv(OUT_CSV)                      # incremental: safe to interrupt
        print(results.tail(1).to_string(), flush=True)

    results = pd.DataFrame(rows).set_index("year")
    exceed = results[results.p_excess < 0.05]
    print(f"\nyears with density above the one-factor null (p < 0.05): "
          f"{len(exceed)} of {len(results)} -> {list(exceed.index)}")
    if (results.null_sd == 0).all():
        print("the beta-only null is zeroed entirely by the fixed L1 penalty in every "
              "year: any observed edge is conditional structure beyond the market factor")
    plot(results)
    return results


def plot(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    x = results.index
    ax.fill_between(x, results.null_mean - 2 * results.null_sd,
                    results.null_mean + 2 * results.null_sd,
                    color="tab:blue", alpha=0.20, label="one-factor null (mean ± 2 sd)")
    ax.plot(x, results.null_mean, color="tab:blue", ls="--", lw=1.5)
    ax.plot(x, results.observed_density, color="black", marker="o", ms=5,
            lw=1.8, label="observed")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(y) if y not in (2019, 2020) else f"{y}\n(no data)" for y in x], fontsize=9)
    ax.set_ylabel("Graphical Lasso edge density at fixed alpha")
    ax.set_title("Conditional dependence vs beta-only markets matched to each year\n"
                 "(2019-2020 unrecoverable; excluded)", fontsize=12)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)
    print(f"saved: {OUT_FIG}")


def demo():
    """Self-check: the loading solver must reproduce its target correlation."""
    rng = np.random.default_rng(0)
    u = base_loadings(pd.DataFrame(rng.normal(size=(400, 30))), "equal")
    scale = solve_scale(0.40, u)
    assert abs(model_mean_corr(scale, u) - 0.40) < 1e-6
    panel = simulate((400, 30), scale * u, rng)
    empirical = observed_mean_corr(panel)
    # tolerance is Monte Carlo error: sample pair correlations share stocks, so
    # averaging over pairs reduces noise more slowly than sqrt(#pairs)
    assert abs(empirical - 0.40) < 0.05, empirical
    print("demo ok")


if __name__ == "__main__":
    demo()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--model", choices=["heterogeneous", "equal"], default="heterogeneous")
    args = parser.parse_args()
    main(sims=args.sims, model=args.model)
