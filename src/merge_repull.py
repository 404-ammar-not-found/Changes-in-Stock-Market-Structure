"""Splice the fresh Polygon pull onto the old one, then re-apply the project's filters.

The old pull (June 2025) and the new one are both split/dividend-adjusted, but
*as of their own pull date*: any ticker that split in between sits on a different
price scale in each. Overlapping dates give the scale factor, so the old rows are
rescaled onto the new basis before concatenation and no fake return appears at the seam.

A third source fills the 2019-01-01 to 2021-09-06 gap where yfinance serves the ticker
(see fetch_gap_yfinance.py). It is rescaled the same way and used only when its prices
agree with Polygon's on overlapping dates, which rules out a reused ticker symbol.

Run after repull_polygon.py and fetch_gap_yfinance.py:  python -m src.merge_repull
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OLD_DIR = RAW / "polygon_initial"
NEW_DIR = RAW / "polygon_repull_2021_2024"
GAP_DIR = RAW / "yfinance_2019_2021"
GAP_START, GAP_END = pd.Timestamp("2019-01-01"), pd.Timestamp("2021-09-07")
GAP_AGREEMENT = 0.01       # median |log price gap| on overlapping dates after rescaling
MERGED_DIR = ROOT / "data" / "interim" / "merged"
OUT_DIR = ROOT / "data" / "processed" / "panel_sp500_membership"
CONSTITUENTS = RAW / "sp500_historical_constituents.csv"
RESCALE_TOLERANCE = 0.02   # |ratio - 1| below this is noise, not a split


def symbol(path: Path) -> str:
    """Constituent-list spelling of a Polygon file name (BRK.B.csv -> BRK-B)."""
    return path.stem.upper().replace(".", "-")


def load_pull(path: Path) -> pd.DataFrame:
    """Both raw pulls share the Polygon aggregate schema; keep date, open, close."""
    frame = pd.read_csv(path, usecols=["date", "o", "c"], parse_dates=["date"])
    frame = frame.rename(columns={"o": "open", "c": "close"}).sort_values("date")
    return frame.loc[:, ["date", "open", "close"]]


def rescale_old(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Put old prices on the new pull's adjustment basis using overlapping dates."""
    overlap = old.merge(new, on="date", suffixes=("_old", "_new"))
    overlap = overlap[(overlap.close_old > 0) & (overlap.close_new > 0)]
    if len(overlap) < 5:
        return old, np.nan
    ratio = float(np.median(overlap.close_new / overlap.close_old))
    if not np.isfinite(ratio) or ratio <= 0 or abs(ratio - 1) < RESCALE_TOLERANCE:
        return old, ratio
    scaled = old.copy()
    scaled[["open", "close"]] *= ratio
    return scaled, ratio


def fill_gap(merged: pd.DataFrame, gap: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Replace the sparse 2019-2021 rows with yfinance's, if the two sources agree."""
    scaled, _ = rescale_old(gap, merged)
    overlap = scaled.merge(merged, on="date", suffixes=("_gap", "_panel"))
    overlap = overlap[(overlap.close_gap > 0) & (overlap.close_panel > 0)]
    if len(overlap) < 20:
        return merged, False
    if np.median(np.abs(np.log(overlap.close_gap / overlap.close_panel))) > GAP_AGREEMENT:
        return merged, False
    in_gap = (merged.date >= GAP_START) & (merged.date < GAP_END)
    fill = scaled[(scaled.date >= GAP_START) & (scaled.date < GAP_END)]
    return pd.concat([merged[~in_gap], fill], ignore_index=True), True


def membership_intervals() -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    hist = pd.read_csv(CONSTITUENTS, parse_dates=["date"]).sort_values("date")
    hist["tickers"] = hist["tickers"].str.split(",")
    long = hist.explode("tickers")
    # the list spells share classes BRK-B until 2023 and BRK.B after; use one spelling
    long["tickers"] = long["tickers"].str.strip().str.upper().str.replace(".", "-", regex=False)
    snapshots = sorted(hist["date"].unique())
    next_snapshot = {d: snapshots[i + 1] for i, d in enumerate(snapshots[:-1])}
    next_snapshot[snapshots[-1]] = pd.Timestamp.max
    intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for ticker, group in long.groupby("tickers"):
        intervals[ticker] = [(d, next_snapshot[d]) for d in sorted(group["date"])]
    return intervals


def main() -> None:
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("*.csv"):   # regenerated in full below; no leftovers
        stale.unlink()
    intervals = membership_intervals()
    rescaled, gained, written, carried = [], 0, 0, 0
    gap_filled, gap_rejected = 0, []

    # Union of both panels: tickers delisted before the new pull's window have no
    # new file and must still be carried through, or the merge reintroduces
    # survivorship bias by silently dropping every casualty (BSC, MER, WB, ...).
    # Polygon writes share classes as BRK.B, the constituents list as BRK-B. Without
    # this mapping both Berkshire and Brown-Forman fail the membership filter in every year.
    old_files = {symbol(p): p for p in OLD_DIR.glob("*.csv")}
    new_files = {symbol(p): p for p in NEW_DIR.glob("*.csv")}
    tickers = sorted(old_files.keys() | new_files.keys())
    for ticker in tickers:
        new_path = new_files.get(ticker, NEW_DIR / "missing.csv")
        old_path = old_files.get(ticker, OLD_DIR / "missing.csv")
        new = load_pull(new_path) if new_path.exists() else pd.DataFrame()
        if new.empty:
            if not old_path.exists():
                continue
            merged = load_pull(old_path)
            carried += 1
        elif old_path.exists():
            # rescale needs the overlapping dates, so trim only afterwards
            old, ratio = rescale_old(load_pull(old_path), new)
            old = old[old.date < new.date.min()]
            if np.isfinite(ratio) and abs(ratio - 1) >= RESCALE_TOLERANCE:
                rescaled.append((ticker, round(ratio, 4)))
            merged = pd.concat([old, new], ignore_index=True)
        else:
            merged = new
            gained += 1
        merged = (merged.dropna(subset=["date"]).drop_duplicates("date", keep="last")
                  .sort_values("date"))
        gap_path = GAP_DIR / f"{ticker}.csv"
        if gap_path.exists():
            gap = pd.read_csv(gap_path, parse_dates=["date"])
            merged, used = fill_gap(merged, gap)
            gap_filled += used
            if not used:
                gap_rejected.append(ticker)
            merged = merged.sort_values("date").reset_index(drop=True)
        merged = merged[(merged.date >= "2004-01-01") & (merged.date <= "2024-12-31")]
        merged.to_csv(MERGED_DIR / f"{ticker}.csv", index=False)

        if ticker not in intervals:
            continue
        keep = pd.Series(False, index=merged.index)
        for start, end in intervals[ticker]:
            keep |= (merged.date >= start) & (merged.date < end)
        filtered = merged[keep]
        if not filtered.empty:
            filtered.to_csv(OUT_DIR / f"{ticker}.csv", index=False)
            written += 1

    print(f"merged files: {len(list(MERGED_DIR.glob('*.csv')))}, "
          f"membership-filtered: {written}, new to the panel: {gained}, "
          f"carried over unchanged (delisted before the new window): {carried}")
    print(f"split-rescaled tickers: {len(rescaled)} {rescaled[:12]}")
    print(f"2019-2021 gap filled from yfinance: {gap_filled}; "
          f"rejected (prices disagree or too little overlap): {len(gap_rejected)} {gap_rejected[:12]}")


def demo():
    """Self-check: a 4:1 split between pulls must be undone, not left as a price jump."""
    dates = pd.bdate_range("2021-09-07", periods=40)
    new = pd.DataFrame({"date": dates, "open": 25.0, "close": 25.0})
    old = pd.DataFrame({"date": dates, "open": 100.0, "close": 100.0})
    scaled, ratio = rescale_old(old, new)
    assert abs(ratio - 0.25) < 1e-9, ratio
    assert abs(scaled.close.iloc[0] - 25.0) < 1e-9, scaled.close.iloc[0]
    unchanged, ratio = rescale_old(old.assign(close=25.1, open=25.1), new)
    assert abs(unchanged.close.iloc[0] - 25.1) < 1e-9   # within tolerance, left alone

    # gap fill: a matching source on a different price scale is rescaled and spliced in;
    # a reused ticker symbol (prices that disagree) is rejected
    days = pd.bdate_range("2018-11-01", "2021-10-29")
    panel = pd.DataFrame({"date": days[::10], "open": 50.0, "close": 50.0})
    trend = np.linspace(1.0, 2.0, len(days))
    source = pd.DataFrame({"date": days, "open": 100.0, "close": 100.0 * trend})
    panel["close"] = 50.0 * trend[::10]
    filled, used = fill_gap(panel, source)
    in_gap = filled[(filled.date >= GAP_START) & (filled.date < GAP_END)]
    assert used and len(in_gap) == ((days >= GAP_START) & (days < GAP_END)).sum()
    assert abs(in_gap.close.iloc[0] / (50.0 * trend[days.get_loc(in_gap.date.iloc[0])]) - 1) < 1e-9
    other = source.assign(close=np.linspace(300.0, 10.0, len(days)))
    assert not fill_gap(panel, other)[1]
    assert symbol(Path("BRK.B.csv")) == "BRK-B" and symbol(Path("aapl.csv")) == "AAPL"
    print("demo ok")


if __name__ == "__main__":
    demo()
    main()
