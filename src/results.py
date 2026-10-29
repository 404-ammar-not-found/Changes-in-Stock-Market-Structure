"""Per-year market-integration numbers: mean pairwise correlation and PC1 share.

Run: python -m src.results   (writes figures/yearly_results.csv, prints markdown)
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "processed" / "market_data.db"
MIN_DAYS = 60          # years with fewer observed dates are reported but not trusted
MIN_COVERAGE = 0.90    # a ticker must trade on this share of the year's dates
BOOTSTRAP = 200


def load_prices() -> pd.DataFrame:
    with sqlite3.connect(DB) as conn:
        df = pd.read_sql("SELECT ticker, date, close FROM prices", conn, parse_dates=["date"])
    return df.pivot(index="date", columns="ticker", values="close").sort_index()


def year_returns(prices: pd.DataFrame, year: int) -> pd.DataFrame:
    window = prices.loc[prices.index.year == year]
    window = window.loc[:, window.notna().mean() >= MIN_COVERAGE]
    returns = np.log(window).diff().dropna(how="all").dropna(axis=1, how="any")
    return returns.loc[:, returns.std() > 0]


def stats(returns: pd.DataFrame) -> tuple[float, float]:
    """Mean off-diagonal correlation and PC1 share of variance."""
    corr = np.corrcoef(returns.to_numpy(), rowvar=False)
    upper = corr[np.triu_indices_from(corr, k=1)]
    eigenvalues = np.linalg.eigvalsh(corr)
    return upper.mean(), eigenvalues[-1] / eigenvalues.sum()


def bootstrap_ci(returns: pd.DataFrame, rng) -> tuple[float, float]:
    """Percentile CI for mean correlation, resampling days."""
    values = returns.to_numpy()
    means = []
    for _ in range(BOOTSTRAP):
        sample = values[rng.integers(0, len(values), len(values))]
        corr = np.corrcoef(sample, rowvar=False)
        means.append(corr[np.triu_indices_from(corr, k=1)].mean())
    return tuple(np.percentile(means, [2.5, 97.5]))


def main() -> pd.DataFrame:
    prices = load_prices()
    rng = np.random.default_rng(42)
    rows = []
    for year in range(prices.index.year.min(), prices.index.year.max() + 1):
        returns = year_returns(prices, year)
        n_days, n_tickers = returns.shape
        # Share of the year's trading days actually observed. The 2004-2018 pull is
        # ~40% complete and 2021 is ~36% (sparse until September, complete after), so
        # this column is the difference between a measured year and a sampled one.
        trading_days = len(pd.bdate_range(f"{year}-01-01", f"{year}-12-31"))
        row = {"year": year, "n_days": n_days, "coverage": round(n_days / trading_days, 2),
               "n_tickers": n_tickers,
               "mean_corr": np.nan, "ci_low": np.nan, "ci_high": np.nan, "pc1_share": np.nan}
        if n_days >= MIN_DAYS and n_tickers >= 10:
            row["mean_corr"], row["pc1_share"] = stats(returns)
            row["ci_low"], row["ci_high"] = bootstrap_ci(returns, rng)
        rows.append(row)

    results = pd.DataFrame(rows).set_index("year")
    results.to_csv(ROOT / "figures" / "yearly_results.csv")
    print(results.round(3).to_string())
    usable = results.dropna(subset=["mean_corr"])
    print(f"\nUsable years: {len(usable)} of {len(results)} "
          f"(need >= {MIN_DAYS} observed dates)")
    print(f"Peak mean correlation: {usable.mean_corr.idxmax()} "
          f"({usable.mean_corr.max():.3f}); trough: {usable.mean_corr.idxmin()} "
          f"({usable.mean_corr.min():.3f})")
    return results


def demo():
    """Self-check: independent noise -> ~0 correlation; one shared factor -> high."""
    rng = np.random.default_rng(0)
    independent = pd.DataFrame(rng.normal(size=(500, 30)))
    assert abs(stats(independent)[0]) < 0.05
    factor = rng.normal(size=(500, 1))
    common = pd.DataFrame(factor + rng.normal(size=(500, 30)) * 0.3)
    mean_corr, pc1 = stats(common)
    assert mean_corr > 0.8 and pc1 > 0.8, (mean_corr, pc1)
    print("demo ok")


if __name__ == "__main__":
    demo()
    main()
