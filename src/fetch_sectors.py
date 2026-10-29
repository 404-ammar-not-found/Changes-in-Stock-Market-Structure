"""Fill in missing ticker -> sector labels via yfinance.

The committed map covers 190 of 872 panel tickers, so ~78% of nodes render as
"Unknown" and the clusters-vs-sectors question cannot be answered. Delisted tickers
mostly stay Unknown - yfinance does not serve them - which is a known ceiling here.

Resumable: re-running only fetches tickers still missing.
Run: python src/fetch_sectors.py
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "processed" / "panel_sp500_membership"
OUT = ROOT / "data" / "raw" / "ticker_to_sector.csv"


def main() -> None:
    # the sector map lives under data/, which is gitignored, so on a fresh
    # clone there is no map to start from
    if OUT.exists():
        known = pd.read_csv(OUT)
        known["Ticker"] = known["Ticker"].astype(str).str.strip().str.upper()
        mapping = dict(zip(known["Ticker"], known["Sector"]))
    else:
        mapping = {}

    panel = sorted({p.stem.upper() for p in PANEL.glob("*.csv")})
    missing = [t for t in panel if t not in mapping]
    print(f"panel={len(panel)} labelled={len(mapping)} missing={len(missing)}")

    found = 0
    for index, ticker in enumerate(missing, 1):
        try:
            sector = yf.Ticker(ticker).info.get("sector")
        except Exception:
            sector = None
        mapping[ticker] = sector or "Unknown"
        found += bool(sector)
        if index % 50 == 0 or index == len(missing):
            pd.DataFrame(sorted(mapping.items()), columns=["Ticker", "Sector"]).to_csv(OUT, index=False)
            print(f"{index}/{len(missing)} resolved={found}", flush=True)

    pd.DataFrame(sorted(mapping.items()), columns=["Ticker", "Sector"]).to_csv(OUT, index=False)
    labelled = sum(1 for t in panel if mapping.get(t, "Unknown") != "Unknown")
    print(f"done: {labelled}/{len(panel)} panel tickers labelled ({labelled/len(panel):.0%})")


if __name__ == "__main__":
    main()
