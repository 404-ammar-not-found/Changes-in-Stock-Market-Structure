"""Does a calm-year risk model understate the next crisis? An out-of-sample test.

Each year, two covariance models are calibrated on that year's returns and used to
predict the volatility of two portfolios in the following year:

- sample: the year's sample covariance. Minimum-variance weights need an inverse, which
  the sample matrix lacks when stocks outnumber days, so those use the Ledoit-Wolf
  shrunk version; shrinkage pulls correlations towards zero, which would understate
  the equal-weight portfolio's variance, so its prediction uses the plain matrix.
- typical volatility: the year's one-factor market model with market and idiosyncratic
  volatility reset to their typical level, the Forbes-Rigobon counterfactual used as a
  forecast. "Typical" is the median over the calibration year and the years before it,
  so the forecast uses no future information.

The portfolios are equal weight and minimum variance (weights from the model being
tested). The bias ratio is realised volatility over predicted: above 1 means the model
understated risk. Volatilities are annualised as variance per observation times
observations per year, which puts the sparse pre-2019 years on the same scale.

Run: python -m src.oos_risk
Writes: figures/oos_risk.csv, figures/oos_risk_from_2017.csv, figures/oos_risk.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.forbes_rigobon import market_model_fit
from src.results import load_prices, year_returns

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "figures" / "oos_risk.csv"
OUT_2017 = ROOT / "figures" / "oos_risk_from_2017.csv"
OUT_FIG = ROOT / "figures" / "oos_risk.png"


def sample_cov(returns: pd.DataFrame) -> np.ndarray:
    return np.cov(returns.to_numpy(), rowvar=False) * len(returns)


def shrunk_cov(returns: pd.DataFrame) -> np.ndarray:
    return LedoitWolf().fit(returns.to_numpy()).covariance_ * len(returns)


def factor_cov(beta: np.ndarray, market_vol: float, idio: np.ndarray) -> np.ndarray:
    return np.outer(beta, beta) * market_vol ** 2 + np.diag(idio)


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    raw = np.linalg.solve(cov, np.ones(len(cov)))
    return raw / raw.sum()


def realised_vol(returns: pd.DataFrame, weights: np.ndarray) -> float:
    return float((returns.to_numpy() @ weights).std(ddof=1) * np.sqrt(len(returns)))


def predicted_vol(cov: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov @ weights))


def calibrate(prices: pd.DataFrame) -> dict[int, dict]:
    """Per-year returns, market-model fit and annualised volatility levels."""
    fits = {}
    for year in range(2004, 2025):
        returns = year_returns(prices, year)
        n = len(returns)
        beta, idio, sigma_m = market_model_fit(returns)
        fits[year] = {"returns": returns, "beta": pd.Series(beta, index=returns.columns),
                      "idio": pd.Series(idio * n, index=returns.columns),
                      "market_vol": sigma_m * np.sqrt(n)}
    return fits


def typical_model(fits: dict, year: int, tickers: list[str]) -> np.ndarray:
    """Year's betas with volatility reset to the median of this and earlier years."""
    past = [y for y in fits if y <= year]
    market_vol = float(np.median([fits[y]["market_vol"] for y in past]))
    reference = pd.DataFrame({y: fits[y]["idio"] for y in past}).median(axis=1)
    idio = fits[year]["idio"]
    kappa = float((idio / reference[idio.index]).median())
    return factor_cov(fits[year]["beta"][tickers].to_numpy(), market_vol,
                      idio[tickers].to_numpy() / kappa)


def evaluate(fits: dict, calib: int, test: int) -> dict:
    tickers = sorted(set(fits[calib]["returns"].columns) & set(fits[test]["returns"].columns))
    calib_returns = fits[calib]["returns"][tickers]
    test_returns = fits[test]["returns"][tickers]
    typical = typical_model(fits, calib, tickers)
    # (covariance for the equal-weight prediction, covariance the optimiser uses)
    models = {"sample": (sample_cov(calib_returns), shrunk_cov(calib_returns)),
              "typical": (typical, typical)}
    row = {"calibration": calib, "test": test, "n_tickers": len(tickers)}
    equal = np.full(len(tickers), 1 / len(tickers))
    row["ew_realised"] = realised_vol(test_returns, equal)
    for name, (cov, optimiser_cov) in models.items():
        row[f"ew_pred_{name}"] = predicted_vol(cov, equal)
        weights = min_variance_weights(optimiser_cov)
        row[f"mv_pred_{name}"] = predicted_vol(optimiser_cov, weights)
        row[f"mv_realised_{name}"] = realised_vol(test_returns, weights)
    for name in models:
        row[f"ew_bias_{name}"] = row["ew_realised"] / row[f"ew_pred_{name}"]
        row[f"mv_bias_{name}"] = row[f"mv_realised_{name}"] / row[f"mv_pred_{name}"]
    return row


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    fits = calibrate(load_prices())
    years = sorted(fits)
    next_year = pd.DataFrame([evaluate(fits, c, c + 1) for c in years[:-1]]).set_index("test")
    from_2017 = pd.DataFrame([evaluate(fits, 2017, t) for t in years if t > 2017]).set_index("test")
    next_year.round(4).to_csv(OUT_CSV)
    from_2017.round(4).to_csv(OUT_2017)

    cols = ["ew_pred_sample", "ew_pred_typical", "ew_realised", "ew_bias_sample", "ew_bias_typical",
            "mv_bias_sample", "mv_bias_typical"]
    print("calibrated on 2017, equal-weight and minimum-variance portfolios:")
    print(from_2017[cols].round(3).to_string())
    print("\ncalibrated on year t-1, tested on year t:")
    print(next_year[cols].round(3).to_string())
    for name in ("sample", "typical"):
        for kind in ("ew", "mv"):
            logs = np.log(next_year[f"{kind}_bias_{name}"])
            print(f"{kind} {name:8s}: mean |log bias| {logs.abs().mean():.3f}, "
                  f"worst understatement x{np.exp(logs.max()):.2f} "
                  f"({logs.idxmax()}), worst overstatement x{np.exp(-logs.min()):.2f} "
                  f"({logs.idxmin()})")
    plot(next_year, from_2017)
    return next_year, from_2017


def plot(next_year: pd.DataFrame, from_2017: pd.DataFrame) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.5))
    for name, colour, label in [("sample", "tab:red", "sample covariance"),
                                ("typical", "tab:blue", "typical-volatility factor model")]:
        left.plot(next_year.index, next_year[f"ew_bias_{name}"], marker="o", ms=4,
                  color=colour, label=label)
    left.axhline(1, color="black", lw=1)
    left.set_yscale("log")
    left.set_yticks([0.33, 0.5, 1, 2, 3])
    left.set_yticklabels(["÷3", "÷2", "1", "×2", "×3"])
    left.set_title("Equal-weight portfolio: realised ÷ predicted volatility,\n"
                   "model calibrated on the previous year", fontsize=11)
    left.set_xticks(list(next_year.index))
    left.set_xticklabels(next_year.index, rotation=45, fontsize=8)
    left.legend(frameon=False)
    left.grid(alpha=0.3)

    x = np.arange(len(from_2017))
    right.bar(x - 0.27, from_2017.ew_pred_sample, 0.27, color="tab:red", label="predicted: sample")
    right.bar(x, from_2017.ew_pred_typical, 0.27, color="tab:blue", label="predicted: typical volatility")
    right.bar(x + 0.27, from_2017.ew_realised, 0.27, color="black", label="realised")
    right.set_xticks(x)
    right.set_xticklabels(from_2017.index)
    right.set_ylabel("annualised volatility")
    right.set_title("Equal-weight portfolio, models calibrated on 2017", fontsize=11)
    right.legend(frameon=False)
    right.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)
    print(f"saved: {OUT_FIG}")


def demo():
    """Self-check: a model calibrated in a calm regime understates a turbulent one,
    and the realised/predicted ratio recovers the size of the volatility jump."""
    rng = np.random.default_rng(0)
    n_days, n_tickers = 250, 40
    beta = rng.uniform(0.8, 1.2, n_tickers)

    def regime(market_sd):
        values = (np.outer(rng.normal(0, market_sd, n_days), beta)
                  + rng.normal(0, 0.01, (n_days, n_tickers)))
        return pd.DataFrame(values)

    calm, storm = regime(0.005), regime(0.015)
    equal = np.full(n_tickers, 1 / n_tickers)
    ratio = realised_vol(storm, equal) / predicted_vol(sample_cov(calm), equal)
    assert 2.4 < ratio < 3.6, ratio
    weights = min_variance_weights(shrunk_cov(calm))
    assert abs(weights.sum() - 1) < 1e-9
    print("demo ok")


if __name__ == "__main__":
    demo()
    main()
