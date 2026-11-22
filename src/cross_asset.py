"""Did anything outside the S&P 500 protect a portfolio when the index fell?

The main analysis shows diversification inside the index fails in turbulent years. This
checks the usual alternatives over the same period with ETF total returns from yfinance:
long-dated Treasuries (TLT), intermediate Treasuries (IEF) and gold (GLD, from November
2004), each against the S&P 500 (SPY). For each year it reports the correlation of daily
returns with SPY and each asset's total return, and it reports how each asset did on
SPY's worst 5% of days.

Prices are cached in data/raw/cross_asset.csv; delete it to re-download.
Run: python -m src.cross_asset
Writes: figures/cross_asset.csv, figures/cross_asset.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "cross_asset.csv"
OUT_CSV = ROOT / "figures" / "cross_asset.csv"
OUT_FIG = ROOT / "figures" / "cross_asset.png"
ASSETS = {"TLT": "long Treasuries", "IEF": "7-10y Treasuries", "GLD": "gold"}


def load() -> pd.DataFrame:
    if not CACHE.exists():
        import yfinance as yf
        data = yf.download(["SPY", *ASSETS], start="2004-01-01", end="2025-01-01",
                           auto_adjust=True, progress=False)["Close"]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(CACHE)
    return pd.read_csv(CACHE, index_col=0, parse_dates=True)


def yearly(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices).diff()
    rows = []
    for year, block in returns.groupby(returns.index.year):
        row = {"year": year, "SPY_return": float(np.expm1(block.SPY.sum()))}
        for asset in ASSETS:
            pair = block[["SPY", asset]].dropna()
            row[f"{asset}_corr"] = pair.SPY.corr(pair[asset]) if len(pair) > 60 else np.nan
            row[f"{asset}_return"] = float(np.expm1(pair[asset].sum())) if len(pair) > 60 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")


def worst_days(prices: pd.DataFrame, share: float = 0.05) -> pd.Series:
    """Average same-day return of each asset on SPY's worst days."""
    returns = np.log(prices).diff().dropna()
    bad = returns.SPY <= returns.SPY.quantile(share)
    return returns[bad].mean()


def main() -> pd.DataFrame:
    prices = load()
    table = yearly(prices)
    table.round(3).to_csv(OUT_CSV)
    print(table.round(3).to_string())
    bad = worst_days(prices)
    print(f"\naverage daily return on SPY's worst 5% of days: "
          + ", ".join(f"{a} {bad[a]:+.2%}" for a in ["SPY", *ASSETS]))
    for year in (2008, 2020, 2022):
        row = table.loc[year]
        print(f"{year}: SPY {row.SPY_return:+.1%}; "
              + ", ".join(f"{a} {row[f'{a}_return']:+.1%} (corr {row[f'{a}_corr']:+.2f})"
                          for a in ASSETS))
    plot(table)
    return table


def plot(table: pd.DataFrame) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colours = {"TLT": "tab:blue", "IEF": "tab:cyan", "GLD": "goldenrod"}
    for asset, label in ASSETS.items():
        top.plot(table.index, table[f"{asset}_corr"], marker="o", ms=4,
                 color=colours[asset], label=label)
    top.axhline(0, color="black", lw=1)
    top.set_ylabel("correlation with S&P 500\n(daily returns, per year)")
    top.set_title("Assets outside the index: correlation with the S&P 500 and yearly returns",
                  fontsize=13)
    top.legend(frameon=False, ncol=3)
    width = 0.2
    x = np.asarray(table.index)
    bottom.bar(x - 1.5 * width, table.SPY_return, width, color="black", label="S&P 500 (SPY)")
    for i, (asset, label) in enumerate(ASSETS.items()):
        bottom.bar(x + (i - 0.5) * width, table[f"{asset}_return"], width,
                   color=colours[asset], label=label)
    bottom.axhline(0, color="black", lw=1)
    bottom.set_ylabel("total return in the year")
    bottom.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    bottom.legend(frameon=False, ncol=4, fontsize=9)
    bottom.set_xticks(list(x))
    bottom.set_xticklabels(x, rotation=45, fontsize=9)
    for ax in (top, bottom):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)
    print(f"saved: {OUT_FIG}")


def demo():
    """Self-check: a hedge that rises when SPY falls shows negative correlation and a
    positive average on SPY's worst days."""
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2010-01-01", periods=500)
    spy = rng.normal(0, 0.01, len(days))
    hedge = -0.5 * spy + rng.normal(0, 0.005, len(days))
    prices = pd.DataFrame({"SPY": np.exp(np.cumsum(spy)), "TLT": np.exp(np.cumsum(hedge)),
                           "IEF": np.exp(np.cumsum(hedge)), "GLD": np.exp(np.cumsum(spy))},
                          index=days)
    table = yearly(prices)
    assert (table.TLT_corr < -0.5).all() and (table.GLD_corr > 0.99).all()
    bad = worst_days(prices)
    assert bad.TLT > 0 > bad.SPY
    print("demo ok")


if __name__ == "__main__":
    demo()
    main()
