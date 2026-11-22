"""Is the Graphical Lasso edge density anything more than average correlation?

For each usable year, simulate one-factor markets - beta and nothing else -
matched on the year's number of trading days, number of tickers, per-ticker
loading profile and average pairwise correlation, run the identical estimation
pipeline at the same fixed pooled alpha, and compare the observed edge density
against the null distribution. Density significantly above the null means
conditional structure beyond the market factor; no excess confirms the
deflationary reading that the network merely restates the correlation level.

Two robustness variants test the main caveats. `--noise t` draws factor and
idiosyncratic shocks from a Student-t with 4 degrees of freedom (fat tails, same
variance). `--model sector` adds one factor per sector on top of the market, so excess
density over it is structure that neither the market nor sectors explain.

Run: python -m src.one_factor_null [--sims 200] [--model heterogeneous|equal|sector]
                                   [--noise gaussian|t]
Writes: figures/one_factor_null.csv and .png for the default model; other variants
write figures/one_factor_null_<model>_<noise>.csv. figures/null_robustness.png
compares every variant on disk.
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
from sklearn.decomposition import FactorAnalysis
from sklearn.exceptions import ConvergenceWarning

from src.market_structure import (DB, RIDGE, calculate_returns, calibrate_hyperparameters,
                                  density_of_returns, edge_density,
                                  estimate_partial_correlations, load_price_matrix)

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "figures" / "one_factor_null.csv"
OUT_FIG = ROOT / "figures" / "one_factor_null.png"

MIN_WINDOW_OBSERVATIONS = 50   # excludes the unrecoverable 2019/2020 windows
MIN_WINDOW_TICKERS = 10
T_DF = 4                       # Student-t degrees of freedom for the fat-tailed variant
SECTORS = ROOT / "data" / "raw" / "ticker_to_sector.csv"


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


def shocks(rng, size, noise: str = "gaussian") -> np.ndarray:
    """Unit-variance shocks: Gaussian, or Student-t scaled to variance 1."""
    if noise == "t":
        return rng.standard_t(T_DF, size=size) / np.sqrt(T_DF / (T_DF - 2))
    return rng.normal(size=size)


def simulate(returns_shape: tuple[int, int], loadings: np.ndarray, rng,
             noise: str = "gaussian") -> pd.DataFrame:
    n_days, n_tickers = returns_shape
    factor = shocks(rng, (n_days, 1), noise)
    idiosyncratic = shocks(rng, (n_days, n_tickers), noise)
    return pd.DataFrame(loadings[None, :] * factor + idiosyncratic)


# --------------------------------------------------------------------------
# Market + sector null: z_it = b_i m_t + g_i s_k(i),t + e_it
# --------------------------------------------------------------------------

def load_sectors() -> dict[str, str]:
    table = pd.read_csv(SECTORS)
    table["Ticker"] = table["Ticker"].astype(str).str.strip().str.upper().str.replace(".", "-")
    return {t: s for t, s in zip(table.Ticker, table.Sector) if s != "Unknown"}


def fit_sector_model(returns: pd.DataFrame, sectors: dict[str, str]) -> dict:
    """Fit a market factor plus one factor per sector on standardised returns.

    Market loadings come from regressing each stock on the equal-weighted market.
    Within each sector, a one-factor factor analysis of the market residuals gives
    the sector loadings and the leftover idiosyncratic variance. Factor analysis is
    used because regressing on a sector average is biased either way: including the
    stock inflates its loading, and leaving it out attenuates it. Stocks without a
    sector label (mostly delisted symbols) load on the market only.
    """
    z = ((returns - returns.mean()) / returns.std()).to_numpy()
    market = z.mean(axis=1)
    centred = market - market.mean()
    beta = (z - z.mean(axis=0)).T @ centred / (centred @ centred)
    residuals = z - np.outer(market, beta)
    labels = np.array([sectors.get(str(c), "") for c in returns.columns], dtype=object)
    sector_loading = np.zeros(z.shape[1])
    group = np.full(z.shape[1], -1)
    resid_sd = residuals.std(axis=0)
    scores = []
    for name in sorted(set(labels) - {""}):
        members = labels == name
        if members.sum() < 3:
            continue
        fa = FactorAnalysis(n_components=1, random_state=0).fit(residuals[:, members])
        sign = np.sign(fa.components_[0].sum())
        sector_loading[members] = fa.components_[0] * sign
        resid_sd[members] = np.sqrt(fa.noise_variance_)
        group[members] = len(scores)
        scores.append(fa.transform(residuals[:, members])[:, 0] * sign)
    # The market average contains part of every sector, so the sector factors left
    # after removing it are negatively correlated; draw them with that correlation.
    corr = np.corrcoef(np.array(scores)).reshape(len(scores), len(scores)) if scores else np.eye(0)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    root = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    return {"n_days": z.shape[0], "beta": beta, "market_sd": float(market.std()),
            "sector_loading": sector_loading, "group": group, "resid_sd": resid_sd,
            "sector_root": root}


def simulate_sector(params: dict, rng, noise: str = "gaussian") -> pd.DataFrame:
    n_days, group = params["n_days"], params["group"]
    market = params["market_sd"] * shocks(rng, (n_days, 1), noise)
    out = market * params["beta"][None, :]
    root = params["sector_root"]
    sector_factors = shocks(rng, (n_days, root.shape[1]), noise) @ root.T
    for k in range(root.shape[0]):
        members = group == k
        out[:, members] += sector_factors[:, [k]] * params["sector_loading"][members]
    out += params["resid_sd"][None, :] * shocks(rng, (n_days, len(group)), noise)
    return pd.DataFrame(out)


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
                   model: str, seed: int, n_jobs: int = -1, noise: str = "gaussian",
                   sectors: dict[str, str] | None = None) -> list[float]:
    if model == "sector":
        params = fit_sector_model(returns, sectors or {})

        def draw(rng):
            return simulate_sector(params, rng, noise)
    else:
        scale = solve_scale(observed_mean_corr(returns), base_loadings(returns, model))
        loadings = scale * base_loadings(returns, model)

        def draw(rng):
            return simulate(returns.shape, loadings, rng, noise)

    def one(seed_i: int) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*slogdet.*")
            panel = draw(np.random.default_rng(seed_i))
            try:
                return edge_density(estimate_partial_correlations(panel, alpha,
                                                                  fallback=False))
            except FloatingPointError:
                # The solver hits its conditioning limit on occasional draws in the
                # highest-correlation years (the README documents the same boundary
                # for the observed 2011 fit). Record and exclude; not silently retry
                # with a different estimator, which would change the null.
                return np.nan

    seeds = np.random.SeedSequence(seed).spawn(sims)
    return Parallel(n_jobs=n_jobs)(delayed(one)(int(s.generate_state(1)[0])) for s in seeds)


def output_path(model: str, noise: str) -> Path:
    if (model, noise) == ("heterogeneous", "gaussian"):
        return OUT_CSV
    return ROOT / "figures" / f"one_factor_null_{model}_{noise}.csv"


def main(sims: int = 200, model: str = "heterogeneous", noise: str = "gaussian",
         years: list[int] | None = None) -> pd.DataFrame:
    """Run the null for every usable year, or only `years`, merged into the saved CSV."""
    price_matrix = load_price_matrix()
    alpha = calibrate_hyperparameters(price_matrix)["graphical_lasso_alpha"]
    sectors = load_sectors() if model == "sector" else None
    out_csv = output_path(model, noise)
    print(f"fixed pooled alpha: {alpha:.4f}; null model: {model}; noise: {noise}; "
          f"sims/year: {sims}")

    rows = []
    if years and out_csv.exists():
        saved = pd.read_csv(out_csv)
        rows = saved[~saved.year.isin(years)].to_dict("records")
    for year, returns in usable_years(price_matrix):
        if years and year not in years:
            continue
        with warnings.catch_warnings():
            # the observed fits in the highest-correlation years do not fully
            # converge within 200 iterations - documented, and immaterial to the
            # comparison against the null, which runs the identical estimator
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*slogdet.*")
            observed = density_of_returns(returns, alpha)
        null = np.array(null_densities(returns, alpha, sims, model, seed=1000 + year,
                                       noise=noise, sectors=sectors))
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
        results = pd.DataFrame(rows).set_index("year").sort_index()
        results.to_csv(out_csv)                      # incremental: safe to interrupt
        print(results.tail(1).to_string(), flush=True)

    results = pd.DataFrame(rows).set_index("year").sort_index()
    exceed = results[results.p_excess < 0.05]
    print(f"\nyears with density above the one-factor null (p < 0.05): "
          f"{len(exceed)} of {len(results)} -> {list(exceed.index)}")
    if (results.null_sd == 0).all():
        print("the beta-only null is zeroed entirely by the fixed L1 penalty in every "
              "year: any observed edge is conditional structure beyond the market factor")
    if out_csv == OUT_CSV:
        plot(results)
    plot_robustness()
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
    ax.set_xticklabels([str(y) for y in x], fontsize=9, rotation=45)
    ax.set_ylabel("Graphical Lasso edge density at fixed alpha")
    ax.set_title("Conditional dependence vs beta-only markets matched to each year",
                 fontsize=12)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)
    print(f"saved: {OUT_FIG}")


VARIANTS = [("heterogeneous", "gaussian", "market only, Gaussian"),
            ("heterogeneous", "t", "market only, Student-t (4 df)"),
            ("sector", "gaussian", "market + sector factors")]


def plot_robustness() -> None:
    """Observed density against each null variant that has been run."""
    fig, ax = plt.subplots(figsize=(11, 6))
    observed = None
    for (model, noise, label), colour in zip(VARIANTS, ["tab:blue", "tab:red", "tab:green"]):
        path = output_path(model, noise)
        if not path.exists():
            continue
        frame = pd.read_csv(path).set_index("year")
        observed = frame.observed_density
        ax.fill_between(frame.index, frame.null_mean - 2 * frame.null_sd,
                        frame.null_mean + 2 * frame.null_sd, color=colour, alpha=0.15)
        ax.plot(frame.index, frame.null_mean, color=colour, ls="--", lw=1.5,
                label=f"null: {label} (mean ± 2 sd)")
    if observed is None:
        plt.close(fig)
        return
    ax.plot(observed.index, observed, color="black", marker="o", ms=5, lw=1.8, label="observed")
    ax.set_xticks(list(observed.index))
    ax.set_xticklabels([str(y) for y in observed.index], fontsize=9, rotation=45)
    ax.set_ylabel("Graphical Lasso edge density at fixed alpha")
    ax.set_title("Observed network density against three null models", fontsize=12)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / "figures" / "null_robustness.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved: {out}")


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
    parser.add_argument("--model", choices=["heterogeneous", "equal", "sector"],
                        default="heterogeneous")
    parser.add_argument("--noise", choices=["gaussian", "t"], default="gaussian")
    parser.add_argument("--years", type=int, nargs="*",
                        help="rerun only these years and merge into the saved results")
    args = parser.parse_args()
    main(sims=args.sims, model=args.model, noise=args.noise, years=args.years)
