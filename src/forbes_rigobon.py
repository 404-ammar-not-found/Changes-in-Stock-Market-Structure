"""How much of the correlation spikes is a volatility artefact?

Forbes and Rigobon (2002) show that measured correlation rises mechanically
with volatility: the correlation coefficient of a pair whose common driving
factor has become more volatile is biased upward even when the underlying
loadings are unchanged. The textbook pairwise correction assumes one passive
market, which is not the situation in an equity panel - there, every stock's
variance moves through the shared factor, and the exact inversion is not
pairwise-observable.

This project's own PC1 finding says the panel is one factor to within 0.02, so
the adjustment is done exactly in that model instead. Each year, the one-factor
decomposition r_it = b_i f_t + e_it is estimated in return units, and the
average pairwise correlation is recomputed with the year's factor volatility
replaced by the 2004-2024 median - the year's own loadings and idiosyncratic
volatilities held fixed. The gap between measured and counterfactual is the
heteroskedasticity artefact; the counterfactual series is co-movement at
constant volatility. The classic pairwise formula (per-stock variance ratios
against the pooled panel) is also reported, as a conservative lower bound on
the artefact.

Run: python -m src.forbes_rigobon
Writes: figures/forbes_rigobon.csv, figures/forbes_rigobon.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.results import load_prices, year_returns

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "figures" / "forbes_rigobon.csv"
OUT_FIG = ROOT / "figures" / "forbes_rigobon.png"


# --------------------------------------------------------------------------
# Market-model machinery (return units, not correlations)
# --------------------------------------------------------------------------

def market_model_fit(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate r_it = b_i m_t + e_it with m the equal-weighted market return.

    The factor definition (equal weights) is fixed across years, which is what
    makes the year's factor variance separately identified - a plain PC1 of the
    annual covariance could not split loadings from factor volatility, and the
    counterfactual needs exactly that split.

    Returns (betas b, idiosyncratic variances, market volatility sigma_m).
    """
    values = returns.to_numpy(dtype=float)
    market = values.mean(axis=1)
    var_market = float(market.var(ddof=1))
    # OLS slope of each stock on the market: cov(r_i, m) / var(m)
    beta = ((values - values.mean(axis=0)).T @ (market - market.mean())) / ((len(market) - 1) * var_market)
    idiosyncratic = np.clip(values.var(axis=0, ddof=1) - beta ** 2 * var_market, 1e-12, None)
    return beta, idiosyncratic, float(np.sqrt(var_market))


def model_mean_corr(beta: np.ndarray, sigma_m: float, idiosyncratic: np.ndarray) -> float:
    """Average pairwise correlation implied by the market model."""
    var_market = sigma_m ** 2
    cov = np.outer(beta, beta) * var_market
    np.fill_diagonal(cov, 0.0)
    total = beta ** 2 * var_market + idiosyncratic
    corr = cov / np.sqrt(np.outer(total, total))
    return float(corr[np.triu_indices_from(corr, k=1)].mean())


def adjust_correlation(corr: np.ndarray, variance_ratio: np.ndarray) -> np.ndarray:
    """Classic Forbes-Rigobon pairwise formula (approximate; under-corrects here).

    variance_ratio[i] is stock i's variance in this window over its benchmark.
    """
    nu = np.sqrt(np.outer(variance_ratio, variance_ratio)) - 1.0
    rho_sq = corr ** 2
    adjusted_sq = rho_sq / ((1.0 - rho_sq) * (1.0 + nu) + rho_sq)
    return np.sqrt(np.clip(adjusted_sq, 0.0, 1.0)) * np.sign(corr)


def off_diagonal(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]


# --------------------------------------------------------------------------
# Yearly driver
# --------------------------------------------------------------------------

def main() -> pd.DataFrame:
    prices = load_prices()
    pooled_variance = np.log(prices).diff().var()

    rows = []
    idio_by_year: dict[int, pd.Series] = {}
    for year in range(prices.index.year.min(), prices.index.year.max() + 1):
        returns = year_returns(prices, year)
        if len(returns) < 60 or returns.shape[1] < 10:
            continue
        corr = np.corrcoef(returns.to_numpy(), rowvar=False)
        beta, idiosyncratic, sigma_m = market_model_fit(returns)
        ratio = (returns.var() / pooled_variance[returns.columns]).to_numpy()
        pairwise = adjust_correlation(corr, ratio)
        idio_by_year[year] = pd.Series(idiosyncratic, index=returns.columns)
        rows.append({
            "year": year,
            "mean_corr": off_diagonal(corr).mean(),
            "mean_corr_fr_pairwise": off_diagonal(pairwise).mean(),
            "mean_corr_at_median_factor_vol": np.nan,   # filled once medians are known
            "mean_corr_at_typical_vol": np.nan,
            "sigma_m": sigma_m,
            "model_corr": model_mean_corr(beta, sigma_m, idiosyncratic),
            "fit": (beta, idiosyncratic),               # kept for the counterfactual pass
        })

    frame = pd.DataFrame(rows).set_index("year")
    median_sigma_m = float(frame.sigma_m.median())
    # Typical idiosyncratic level: each stock's median annual variance across the
    # years it trades; kappa_t rescales year t's whole idio profile onto that level.
    reference_idio = pd.DataFrame(idio_by_year).median(axis=1)
    for year, row in zip(frame.index, frame.itertuples()):
        beta, idiosyncratic = row.fit
        kappa = float((idio_by_year[year] / reference_idio).median())
        frame.loc[year, "mean_corr_at_median_factor_vol"] = \
            model_mean_corr(beta, median_sigma_m, idiosyncratic)
        frame.loc[year, "mean_corr_at_typical_vol"] = \
            model_mean_corr(beta, median_sigma_m, idiosyncratic / kappa)
    frame["artefact"] = frame["mean_corr"] - frame["mean_corr_at_typical_vol"]
    frame = frame.drop(columns=["fit"]).round(4)
    frame.to_csv(OUT_CSV)
    print(frame.drop(columns=["model_corr"]).to_string())

    for a, b in [(2005, 2008), (2017, 2022)]:
        raw = frame.loc[b, "mean_corr"] - frame.loc[a, "mean_corr"]
        adj = frame.loc[b, "mean_corr_at_typical_vol"] - frame.loc[a, "mean_corr_at_typical_vol"]
        print(f"\n{a} -> {b}: measured +{raw:.3f}, at-typical-volatility +{adj:.3f} "
              f"({adj / raw:.0%} of the raw move survives; {1 - adj / raw:.0%} is the "
              f"volatility artefact)")
    plot(frame)
    return frame


def plot(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    x = frame.index
    ax.plot(x, frame.mean_corr, color="tab:red", marker="o", ms=5, lw=1.8,
            label="measured mean correlation")
    ax.plot(x, frame.mean_corr_at_typical_vol, color="tab:blue", marker="s", ms=5, lw=1.8,
            label="at typical volatility (Forbes-Rigobon counterfactual)")
    ax.fill_between(x, frame.mean_corr_at_typical_vol, frame.mean_corr,
                    color="grey", alpha=0.25, label="heteroskedasticity artefact")
    ax.plot(x, frame.mean_corr_at_median_factor_vol, color="tab:purple", lw=1.2, ls="--",
            label="factor volatility reset only")
    ax.plot(x, frame.mean_corr_fr_pairwise, color="tab:green", lw=1.2, ls=":",
            label="classic pairwise formula (lower bound)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(y) if y not in (2019, 2020) else f"{y}\n(no data)" for y in x], fontsize=9)
    ax.set_ylabel("mean pairwise correlation")
    ax.set_title("Measured vs volatility-adjusted co-movement, S&P 500 constituents\n"
                 "(2019-2020 unrecoverable; excluded)", fontsize=12)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)
    print(f"saved: {OUT_FIG}")


def demo():
    """Self-check on the two volatility scenarios.

    Factor-led turbulence (only the market factor's volatility rises) inflates
    the measured correlation; resetting the factor recovers the calm level.
    Proportional turbulence (factor and idiosyncratic vol rise together) leaves
    the measured correlation unchanged - and resetting the factor alone would
    then wrongly deflate it, which is why the headline counterfactual resets
    both.
    """
    rng = np.random.default_rng(0)
    n_days, n_tickers = 3000, 30
    loadings = rng.uniform(0.7, 1.3, size=n_tickers)
    calm = loadings[None, :] * rng.normal(size=(n_days, 1)) + rng.normal(size=(n_days, n_tickers))
    factor_led = loadings[None, :] * 2.5 * rng.normal(size=(n_days, 1)) + rng.normal(size=(n_days, n_tickers))
    proportional = (loadings[None, :] * 2.5 * rng.normal(size=(n_days, 1))
                    + 2.5 * rng.normal(size=(n_days, n_tickers)))
    calm_df, led_df, prop_df = map(pd.DataFrame, (calm, factor_led, proportional))

    def measured(matrix) -> float:
        return off_diagonal(np.corrcoef(matrix, rowvar=False)).mean()

    truth = measured(calm)
    assert measured(factor_led) > truth + 0.10                 # factor-led artefact
    beta, idio, _ = market_model_fit(led_df)
    sigma_m_calm = market_model_fit(calm_df)[2]
    assert abs(model_mean_corr(beta, sigma_m_calm, idio) - truth) < 0.05

    assert abs(measured(proportional) - truth) < 0.05          # scale invariance
    beta_p, idio_p, _ = market_model_fit(prop_df)
    factor_only = model_mean_corr(beta_p, sigma_m_calm, idio_p)
    both = model_mean_corr(beta_p, sigma_m_calm, idio_p / 2.5 ** 2)
    assert factor_only < truth - 0.05                         # the trap, demonstrated
    assert abs(both - truth) < 0.05
    print("demo ok")


if __name__ == "__main__":
    demo()
    main()
