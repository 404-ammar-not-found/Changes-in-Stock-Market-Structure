"""Fill the 2019-2021 Polygon gap with daily prices from yfinance.

The Polygon account's history now starts on 2021-09-07, and the original pull holds only
5-10% of trading days for 2019-2020. yfinance serves full daily history for tickers that
still trade, so this pulls every ticker that was an S&P 500 member at any point between
2019-01-01 and 2021-09-06. Delisted tickers (CELG, for example) are not served and stay
on their sparse Polygon rows.

`Close` is split-adjusted but not dividend-adjusted, which matches Polygon's
`adjusted=True`. merge_repull.py rescales each ticker onto the panel's price basis using
overlapping dates and rejects tickers whose prices disagree with Polygon's.

Companies that changed symbol but are the same legal entity (FB to META, for example) are
fetched under their current symbol via RENAMES. Acquisitions are deliberately left out:
an acquirer's price history is not the target's.

Resumable: tickers already on disk are skipped.
Run: python -m src.fetch_gap_yfinance
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.merge_repull import membership_intervals

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "yfinance_2019_2021"
GAP_START, GAP_END = pd.Timestamp("2019-01-01"), pd.Timestamp("2021-09-07")
# Pull a margin either side so there are overlapping dates to rescale against.
FETCH_START, FETCH_END = "2018-10-01", "2021-12-31"
# Old symbol -> current symbol of the same company. BBT is here because BB&T renamed
# itself Truist (TFC); yfinance now serves an unrelated security under "BBT".
RENAMES = {
    "ABC": "COR", "ADS": "BFH", "ANTM": "ELV", "BBT": "TFC", "BLL": "BALL", "CTL": "LUMN",
    "DWDP": "DD", "FB": "META", "FBHS": "FBIN", "FLT": "CPAY", "GPS": "GAP", "HFC": "DINO",
    "HRS": "LHX", "JEC": "J", "LB": "BBWI", "MYL": "VTRS", "NLOK": "GEN", "PEAK": "DOC",
    "PKI": "RVTY", "RE": "EG", "SYMC": "GEN", "TMK": "GL", "UTX": "RTX", "WLTW": "WTW",
}


def gap_tickers() -> list[str]:
    return sorted(t for t, spans in membership_intervals().items()
                  if any(start < GAP_END and end > GAP_START for start, end in spans))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [t for t in gap_tickers()
            if t in RENAMES or not (OUT_DIR / f"{t}.csv").exists()]
    print(f"to fetch: {len(todo)}")
    for i in range(0, len(todo), 100):
        batch = todo[i:i + 100]
        symbols = sorted({RENAMES.get(t, t) for t in batch})
        data = yf.download(symbols, start=FETCH_START, end=FETCH_END, auto_adjust=False,
                           group_by="ticker", progress=False, threads=True)
        for ticker in batch:
            symbol = RENAMES.get(ticker, ticker)
            if symbol not in data.columns.get_level_values(0):
                continue
            frame = data[symbol][["Open", "Close"]].dropna()
            if frame.empty:
                continue
            frame = frame.rename(columns={"Open": "open", "Close": "close"})
            frame.index.name = "date"
            frame.round(6).to_csv(OUT_DIR / f"{ticker}.csv")
        print(f"{min(i + 100, len(todo))}/{len(todo)}", flush=True)
    have = sum((OUT_DIR / f"{t}.csv").exists() for t in gap_tickers())
    print(f"served by yfinance: {have} of {len(gap_tickers())} gap-window members")


if __name__ == "__main__":
    main()
