from pathlib import Path
import sqlite3
import pandas as pd
import time
import sys

DATA_DIR = Path(__file__).resolve().parent / "data"
DB_DIR = Path(__file__).resolve().parent.parent / "data_db"
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "market_data.db"

# prefer the re-pulled/merged panel when merge_repull.py has produced it
STOCK_FILES_DIR = DATA_DIR / "stock_data_filtered_for_s&p500_membership_merged"
if not STOCK_FILES_DIR.exists():
    STOCK_FILES_DIR = DATA_DIR / "stock_data_filtered_for_s&p500_membership"
CONSTITUENTS_PATH = DATA_DIR / "s&p500_historical_constituents.csv"
SECTOR_PATH = DATA_DIR / "ticker_to_sector.csv"

if DB_PATH.exists():
    DB_PATH.unlink()

t0 = time.time()

conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()

cur.executescript("""
PRAGMA foreign_keys = ON;
PRAGMA synchronous = OFF;
PRAGMA journal_mode = MEMORY;
PRAGMA cache_size = -80000;

CREATE TABLE IF NOT EXISTS sectors (
    sector_id TEXT PRIMARY KEY,
    sector_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stocks (
    ticker TEXT PRIMARY KEY,
    sector_id TEXT REFERENCES sectors(sector_id)
);

CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL REFERENCES stocks(ticker),
    date TEXT NOT NULL,
    open REAL,
    close REAL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS constituents (
    ticker TEXT NOT NULL REFERENCES stocks(ticker),
    from_date TEXT NOT NULL,
    to_date TEXT NOT NULL,
    PRIMARY KEY (ticker, from_date)
);
""")

# ---- sectors & stocks ----
print("Inserting sectors and stocks...", flush=True)
sector_df = pd.read_csv(SECTOR_PATH)
cur.executemany(
    "INSERT OR IGNORE INTO sectors (sector_id, sector_name) VALUES (?, ?)",
    [(r["Sector"].strip().upper().replace(" ", "_"), r["Sector"].strip()) for _, r in sector_df.iterrows()]
)
cur.executemany(
    "INSERT OR IGNORE INTO stocks (ticker, sector_id) VALUES (?, ?)",
    [(r["Ticker"].strip().upper(), r["Sector"].strip().upper().replace(" ", "_")) for _, r in sector_df.iterrows()]
)

# ---- stocks from price files (tickers missing from sector CSV) ----
price_files = sorted(STOCK_FILES_DIR.glob("*.csv"))
print(f"Inserting stocks from {len(price_files)} price files...", flush=True)
for fpath in price_files:
    cur.execute("INSERT OR IGNORE INTO stocks (ticker) VALUES (?)", (fpath.stem.upper(),))
conn.commit()

# ---- prices: batch insert in chunks ----
print("Inserting prices...", flush=True)
all_price_rows = []
total_price_files = len(price_files)
for i, fpath in enumerate(price_files):
    ticker = fpath.stem.upper()
    df = pd.read_csv(fpath)
    all_price_rows.extend(zip([ticker] * len(df), df["date"].astype(str), df["open"], df["close"]))

    if len(all_price_rows) >= 250000:
        cur.executemany(
            "INSERT OR IGNORE INTO prices (ticker, date, open, close) VALUES (?, ?, ?, ?)",
            all_price_rows
        )
        conn.commit()
        all_price_rows = []
        print(f"  {i+1}/{total_price_files} files, {cur.execute('SELECT COUNT(*) FROM prices').fetchone()[0]:,} total prices", flush=True)

if all_price_rows:
    cur.executemany("INSERT OR IGNORE INTO prices (ticker, date, open, close) VALUES (?, ?, ?, ?)", all_price_rows)
    conn.commit()
    all_price_rows = []

price_count = cur.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
print(f"  Done: {price_count:,} prices inserted", flush=True)

# ---- constituents ----
print("Processing constituents...", flush=True)
hist = pd.read_csv(CONSTITUENTS_PATH, parse_dates=["date"]).sort_values("date")
hist["tickers"] = hist["tickers"].str.split(",")
long_df = hist.explode("tickers")
long_df["tickers"] = long_df["tickers"].str.strip().str.upper()

# ensure all constituent tickers are in stocks
constituent_tickers = long_df["tickers"].unique()
for t in constituent_tickers:
    cur.execute("INSERT OR IGNORE INTO stocks (ticker) VALUES (?)", (t,))
conn.commit()

# build membership intervals using a lead approach
snapshot_dates = sorted(hist["date"].unique())
date_to_next = {d: snapshot_dates[i+1] for i, d in enumerate(snapshot_dates[:-1])}
date_to_next[snapshot_dates[-1]] = pd.Timestamp.max

long_df["to_date"] = long_df["date"].map(date_to_next)
membership_rows = [(t, d.date().isoformat(), e.date().isoformat()) for t, d, e in zip(long_df["tickers"], long_df["date"], long_df["to_date"])]

cur.executemany(
    "INSERT OR IGNORE INTO constituents (ticker, from_date, to_date) VALUES (?, ?, ?)",
    membership_rows
)
conn.commit()

# ---- summary ----
print("\nSummary:", flush=True)
for table in ["sectors", "stocks", "prices", "constituents"]:
    count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table}: {count:,} rows")

elapsed = time.time() - t0
print(f"\nDatabase: {DB_PATH}")
print(f"Time: {elapsed:.1f}s")

conn.close()
