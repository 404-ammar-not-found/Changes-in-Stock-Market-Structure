"""How much does the sparse 2004-2018 sampling distort mean correlation?

2022-2024 are complete, so the true daily answer is known for those years. Impose
a real sparse year's observed date pattern on them, recompute, and the gap is the
bias carried by the sparse estimates.

Run: python -m src.sparse_bias_check
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "processed" / "market_data.db"
COMPLETE_YEARS = [2022, 2023, 2024]
SPARSE_YEARS = [2004, 2008, 2012, 2016]


def load() -> pd.DataFrame:
    with sqlite3.connect(DB) as conn:
        df = pd.read_sql("SELECT ticker, date, close FROM prices", conn, parse_dates=["date"])
    return df.pivot(index="date", columns="ticker", values="close").sort_index()


def mean_corr(prices: pd.DataFrame) -> float:
    returns = np.log(prices).diff().dropna(how="all").dropna(axis=1, how="any")
    returns = returns.loc[:, returns.std() > 0]
    corr = np.corrcoef(returns.to_numpy(), rowvar=False)
    return corr[np.triu_indices_from(corr, k=1)].mean()


def year_panel(prices: pd.DataFrame, year: int, coverage: float = 0.90) -> pd.DataFrame:
    window = prices.loc[prices.index.year == year]
    return window.loc[:, window.notna().mean() >= coverage]


def main() -> None:
    prices = load()
    print(f"{'year':>6} {'true':>7} " + " ".join(f"{y:>7}" for y in SPARSE_YEARS))
    print(f"{'':>6} {'daily':>7} " + " ".join(f"{'pattern':>7}" for _ in SPARSE_YEARS))
    rows = []
    for year in COMPLETE_YEARS:
        panel = year_panel(prices, year)
        truth = mean_corr(panel)
        line = [f"{year:>6} {truth:7.3f}"]
        for sparse_year in SPARSE_YEARS:
            # Map the sparse year's ACTUAL observed dates onto this year by their
            # fractional position in the trading calendar, so the real clustering of
            # runs and gaps is reproduced rather than an even thinning.
            sparse_dates = prices.loc[prices.index.year == sparse_year].index
            calendar = pd.bdate_range(f"{sparse_year}-01-01", f"{sparse_year}-12-31")
            fraction = np.array([calendar.searchsorted(d) / len(calendar) for d in sparse_dates])
            picks = np.unique((fraction * (len(panel) - 1)).astype(int))
            estimate = mean_corr(panel.iloc[picks])
            rows.append(estimate - truth)
            line.append(f"{estimate:7.3f}")
        print(" ".join(line))
    bias = np.array(rows)
    print(f"\nbias (sparse - true): mean {bias.mean():+.3f}, range {bias.min():+.3f} to {bias.max():+.3f}")


def demo():
    """Self-check: subsampling a pure one-factor panel must not move correlation much."""
    rng = np.random.default_rng(0)
    factor = rng.normal(size=(250, 1))
    p = pd.DataFrame(100 * np.exp(np.cumsum(factor + rng.normal(size=(250, 20)) * 0.5, axis=0) * 0.01))
    full, thinned = mean_corr(p), mean_corr(p.iloc[::2])
    assert abs(full - thinned) < 0.15, (full, thinned)
    print("demo ok")


if __name__ == "__main__":
    demo()
    main()
