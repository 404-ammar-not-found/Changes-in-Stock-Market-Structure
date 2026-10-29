"""Re-pull daily bars from Polygon into data/raw/polygon_repull_2021_2024/.

The account's plan only serves ~5 years of history (anything older returns 403
NOT_AUTHORIZED), so this recovers the recent window only. Resumable: re-running
skips tickers already written.

Run: python -m src.repull_polygon
"""
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "polygon_repull_2021_2024"
CONSTITUENTS = ROOT / "data" / "raw" / "sp500_historical_constituents.csv"
START, END = "2021-01-01", "2024-12-31"   # 403s before the plan's window are logged and skipped
COLUMNS = ["v", "vw", "o", "c", "h", "l", "t", "n", "date"]

load_dotenv(ROOT / ".env")
KEY = os.environ["POLYGON_KEY"].strip()


def constituent_tickers() -> list[str]:
    hist = pd.read_csv(CONSTITUENTS)
    hist = hist[(hist.date >= "2004-01-01") & (hist.date <= "2024-12-31")]
    return sorted(set(hist.tickers.str.split(",").explode().str.strip()))


def fetch(ticker: str) -> tuple[str, list[dict]]:
    """Returns (outcome, bars). Follows next_url; backs off on 429/5xx."""
    url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{START}/{END}"
           f"?adjusted=true&sort=asc&limit=50000")
    bars, delay = [], 2.0
    while url:
        try:
            with urllib.request.urlopen(f"{url}&apiKey={KEY}", timeout=90) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 403:
                return "not_authorized", []
            if exc.code in (429, 500, 502, 503, 504) and delay <= 32:
                time.sleep(delay)
                delay *= 2
                continue
            return f"http_{exc.code}", []
        except Exception as exc:                      # network flake
            if delay <= 32:
                time.sleep(delay)
                delay *= 2
                continue
            return f"error_{type(exc).__name__}", []
        bars.extend(payload.get("results") or [])
        url = payload.get("next_url")
    return ("ok" if bars else "empty"), bars


def to_frame(bars: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(bars)
    frame["date"] = pd.to_datetime(frame["t"], unit="ms", utc=True).dt.date
    for column in COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame.loc[:, COLUMNS]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = constituent_tickers()
    counts: dict[str, int] = {}
    for index, ticker in enumerate(tickers, 1):
        target = OUT_DIR / f"{ticker}.csv"
        if target.exists():
            counts["cached"] = counts.get("cached", 0) + 1
            continue
        outcome, bars = fetch(ticker)
        counts[outcome] = counts.get(outcome, 0) + 1
        if bars:
            to_frame(bars).to_csv(target, index=False)
        if index % 50 == 0 or index == len(tickers):
            print(f"{index}/{len(tickers)} {counts}", flush=True)
    print("done:", counts)
    check_coverage()


def check_coverage(threshold: float = 0.95) -> float:
    """A 200 OK can still carry a fraction of the range, which is how the original
    pull lost ~60% of its rows unnoticed. Compare what arrived against the trading
    days it should span, and say so loudly when it falls short."""
    dates = set()
    for path in OUT_DIR.glob("*.csv"):
        dates.update(pd.read_csv(path, usecols=["date"]).date)
    if not dates:
        print("COVERAGE: no data pulled")
        return 0.0
    served = pd.to_datetime(sorted(dates))
    expected = pd.bdate_range(served.min(), served.max())
    coverage = len(served) / len(expected)
    print(f"COVERAGE: {served.min().date()}..{served.max().date()} — "
          f"{len(served)} dates of {len(expected)} business days ({coverage:.1%})")
    if coverage < threshold:
        print(f"WARNING: below {threshold:.0%}. The API returned partial data without "
              f"erroring; do not treat this panel as daily.")
    if served.min().date() > dt.date.fromisoformat(START):
        print(f"NOTE: nothing served before {served.min().date()} — the plan's history "
              f"window starts there; earlier dates return 403.")
    return coverage


if __name__ == "__main__":
    main()
