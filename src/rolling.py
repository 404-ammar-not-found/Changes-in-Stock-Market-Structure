"""Six-month rolling windows instead of calendar years.

Calendar years split the 2008 crisis across two windows. Here each window covers the six
months up to a month end, stepped forward one month at a time, and reports the same
statistics as the yearly tables: mean pairwise correlation, PC1 share, market volatility
and the Forbes-Rigobon correlation at typical volatility.

Volatilities are annualised as variance per observation times observations per year.
Before 2019 an "observation" can span several trading days (the pull is ~40% complete),
and the sum of a window's returns covers the whole window either way, so this puts sparse
and complete windows on the same scale.

Run: python -m src.rolling
Writes: figures/rolling.csv, figures/rolling.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.forbes_rigobon import market_model_fit, model_mean_corr
from src.results import MIN_COVERAGE, load_prices, stats

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "figures" / "rolling.csv"
OUT_FIG = ROOT / "figures" / "rolling.png"
WINDOW_MONTHS = 6
MIN_OBSERVATIONS = 40
EVENTS = {"2008-09-15": "Lehman", "2011-08-05": "US downgrade",
          "2020-02-24": "COVID sell-off", "2022-03-16": "first Fed hike"}


def window_returns(prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Log returns inside (start, end], same ticker rules as the yearly tables."""
    window = prices.loc[(prices.index > start) & (prices.index <= end)]
    window = window.loc[:, window.notna().mean() >= MIN_COVERAGE]
    returns = np.log(window).diff().dropna(how="all").dropna(axis=1, how="any")
    return returns.loc[:, returns.std() > 0]


def main() -> pd.DataFrame:
    prices = load_prices()
    ends = pd.date_range("2004-06-30", "2024-12-31", freq="ME")
    rows, fits = [], {}
    for end in ends:
        start = end - pd.DateOffset(months=WINDOW_MONTHS)
        returns = window_returns(prices, start, end)
        if len(returns) < MIN_OBSERVATIONS or returns.shape[1] < 10:
            continue
        per_year = len(returns) / (WINDOW_MONTHS / 12)
        beta, idiosyncratic, sigma_m = market_model_fit(returns)
        fits[end] = (beta, pd.Series(idiosyncratic * per_year, index=returns.columns))
        mean_corr, pc1 = stats(returns)
        rows.append({"end": end, "n_obs": len(returns), "n_tickers": returns.shape[1],
                     "mean_corr": mean_corr, "pc1_share": pc1,
                     "market_vol": sigma_m * np.sqrt(per_year)})
    frame = pd.DataFrame(rows).set_index("end")

    # Forbes-Rigobon at typical volatility, as in forbes_rigobon.py but in annual units
    typical_vol = float(frame.market_vol.median())
    reference_idio = pd.DataFrame({e: idio for e, (_, idio) in fits.items()}).median(axis=1)
    adjusted = {}
    for end, (beta, idio) in fits.items():
        kappa = float((idio / reference_idio[idio.index]).median())
        adjusted[end] = model_mean_corr(beta, typical_vol, idio.to_numpy() / kappa)
    frame["mean_corr_at_typical_vol"] = pd.Series(adjusted)
    frame = frame.round(4)
    frame.to_csv(OUT_CSV)

    peak = frame.mean_corr.idxmax()
    print(f"windows: {len(frame)}; peak {frame.mean_corr.max():.3f} in the window ending "
          f"{peak:%b %Y}; low {frame.mean_corr.min():.3f} ending {frame.mean_corr.idxmin():%b %Y}")
    print(f"correlation at typical volatility: {frame.mean_corr_at_typical_vol.min():.3f} "
          f"to {frame.mean_corr_at_typical_vol.max():.3f}")
    print("\naround Lehman (window ends):")
    print(frame.loc["2008-03":"2009-06", ["n_obs", "mean_corr", "market_vol"]].to_string())
    plot(frame)
    return frame


def plot(frame: pd.DataFrame) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1]})
    top.plot(frame.index, frame.mean_corr, color="tab:red", lw=1.6, label="measured")
    top.plot(frame.index, frame.mean_corr_at_typical_vol, color="tab:blue", lw=1.6,
             label="at typical volatility (Forbes-Rigobon)")
    top.set_ylabel("mean pairwise correlation")
    top.set_title(f"S&P 500 co-movement in {WINDOW_MONTHS}-month rolling windows", fontsize=13)
    bottom.plot(frame.index, frame.market_vol, color="black", lw=1.4)
    bottom.set_ylabel("market volatility\n(annualised)")
    for date, label in EVENTS.items():
        for ax in (top, bottom):
            ax.axvline(pd.Timestamp(date), color="grey", ls=":", lw=1)
        top.text(pd.Timestamp(date), 0.97, f" {label}", transform=top.get_xaxis_transform(),
                 fontsize=8, va="top", color="dimgrey")
    top.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.9))
    for ax in (top, bottom):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)
    print(f"saved: {OUT_FIG}")


def demo():
    """Self-check: annualising makes sparse and complete windows agree."""
    rng = np.random.default_rng(0)
    daily = pd.DataFrame(rng.normal(0, 0.01, size=(252, 5)),
                         index=pd.bdate_range("2021-01-01", periods=252))
    prices = np.exp(daily.cumsum())
    sparse = prices.iloc[::3]
    full_vol = np.log(prices).diff().std().mean() * np.sqrt(252)
    sparse_returns = np.log(sparse).diff().dropna()
    sparse_vol = sparse_returns.std().mean() * np.sqrt(len(sparse_returns))
    assert abs(sparse_vol / full_vol - 1) < 0.15, (sparse_vol, full_vol)
    print("demo ok")


if __name__ == "__main__":
    demo()
    main()
