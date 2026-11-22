# Do stocks really become more connected in a crisis?

[![tests](https://github.com/404-ammar-not-found/Changes-in-Stock-Market-Structure/actions/workflows/tests.yml/badge.svg)](https://github.com/404-ammar-not-found/Changes-in-Stock-Market-Structure/actions/workflows/tests.yml)
![Python 3.14](https://img.shields.io/badge/python-3.14-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A 21-year study (2004–2024) of how ~500 S&P 500 stocks move together, built on ~870 tickers of
daily prices with historical index membership, so the companies that failed in 2008 stay in the data.

**Short answer: no.** During a crisis, correlation between stocks more than doubles. But once
volatility is accounted for, that spike almost disappears. Stocks don't become more interconnected;
the whole market just moves more violently.

![Summary of findings](figures/summary.png)

## Key findings

| | Finding | Evidence |
|---|---|---|
| 1 | Average correlation swings four-fold, from **0.13 (2017)** to **0.55 (2011)**, with spikes in 2008–11 and 2022 | Bootstrap 95% CIs per year |
| 2 | That co-movement is **one market factor**, not many new links | The first principal component tracks mean correlation to within 0.02 every year |
| 3 | The spikes are **volatility, not structure**: at typical volatility every year sits at 0.25–0.35, and the 2005→2008 rise disappears completely | Forbes–Rigobon counterfactual |
| 4 | The crisis "network densification" is a **statistical artefact**: in 2010–11 the stock network is no denser than in 200 simulated beta-only markets | Graphical Lasso vs one-factor null |
| 5 | A thin, **stable core** of real stock-to-stock links exists in 17 of 19 years (share classes, duopolies, spin-offs) | Same null test |

## Why it matters

- **Risk models:** a model calibrated on calm-year data (2017) would have **understated 2008- or 2022-style co-movement by about 3×**. Stress tests should use crisis-year correlations.
- **Diversification:** in a crash the index behaves like one big bet. Holding more S&P stocks adds little protection; it has to come from assets driven by something else.
- **Reading signals:** before concluding that "the market has changed", check whether volatility alone explains the move. Here it explained nearly all of it.

## Approach

```
Polygon API ──► coverage check ──► splice + split reconciliation ──► SQLite
                                                                       │
     ┌──────────────────────┬───────────────────────┬──────────────────┤
     ▼                      ▼                       ▼                  ▼
 Correlation + PCA   Forbes–Rigobon        Graphical Lasso      Affinity Propagation
 (bootstrap CIs)     volatility reset      + one-factor null    clustering vs sectors
```

| Question | Method |
|---|---|
| How much do stocks move together? | Mean pairwise correlation of daily log returns; 200-sample bootstrap CIs |
| Is it one thing or many? | Variance share of the first principal component |
| Is the spike just volatility? | Re-estimate correlation with each year's market and idiosyncratic volatility reset to typical levels (Forbes & Rigobon, 2002) |
| Is there structure beyond the market factor? | Partial correlations via Graphical Lasso, tested against 200 simulated one-factor markets per year |
| Do stocks group by sector? | Affinity Propagation with hyperparameters held fixed across years so counts are comparable |

## Data challenges solved

Real-world data issues took up as much of the work as the modelling:

- **API reported success while returning only part of the data.** Polygon answered `200 OK` with
  ~40% of trading days once a request went past the account's history limit. I caught it by comparing
  row counts with the trading calendar, re-pulled what was still available, and added a completeness
  assertion to the pipeline.
- **Measured the damage instead of guessing.** Applying the sparse years' missing-day pattern to the
  complete 2022–24 data showed a bias of about **+0.03**. The trend over time is reliable; the
  levels read slightly high.
- **Survivorship bias came back through a filter.** Two innocent-looking `>= 0.50` coverage thresholds
  silently removed Bear Stearns, Merrill Lynch, Wachovia, National City and Countrywide. Both are fixed.
- **Splits mismatched between the two data pulls**, which would have created fake returns at the join.
  Five real splits (APH, MNST, CRWD, DD, PARA) were detected and confirmed with volume ratios.
- **More stocks than days** (416 stocks vs 105 days in 2011) makes the covariance matrix singular,
  so a regularised estimator was required. Hyperparameters were fixed once on pooled data rather
  than re-tuned each year, so year-on-year changes reflect the market, not the estimator.

## Animations

| Stock network by year (nodes coloured by sector) | Clusters by year |
|---|---|
| ![MDS network](figures/stock_market_structure_mds.gif) | ![Affinity propagation clusters](figures/stock_market_structure_affinity_propagation.gif) |

Node positions are fixed across frames, so movement reflects the data rather than the layout.

## Tech stack

Python · pandas · NumPy · scikit-learn (Graphical Lasso, Affinity Propagation, MDS) · SciPy ·
matplotlib · SQLite · pytest · GitHub Actions

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest                                    # estimator self-checks on synthetic data, runs in seconds

python -m src.results                     # headline table
python -m src.forbes_rigobon              # volatility counterfactual
python -m src.one_factor_null --sims 200  # structure-beyond-beta test (~30 min)
python -m src.plot_summary                # summary figure
```

`data/` is not committed because of its size. Rebuild it with a `POLYGON_KEY` in `.env` (see
[Method](docs/analysis.md#method)). The tests run without it: they check every estimator on
synthetic cases where the right answer is known.

## Repository layout

```
├── notebooks/   # 01 data prep → 02 networks → 04 results → 05 volatility → 06 beyond beta
├── src/         # pipeline and analysis modules, importable by notebooks and scripts
├── tests/       # pytest self-checks on synthetic limiting cases
├── figures/     # result tables (CSV), figures, animations
└── docs/        # full write-up with every table and caveat
```

## Limitations

- **2019–2020 are missing** (no longer available from the API), so COVID is not covered.
- Pre-2019 years cover about 40% of trading days; the bias this causes is quantified above.
- The volatility adjustment assumes a one-factor model, and the null simulations use Gaussian noise
  while real returns have fat tails.
- This is descriptive analysis, not a causal or trading study.

Every table, the full method and all caveats are in **[docs/analysis.md](docs/analysis.md)**.

## Credits

Historical constituents from [hanshof/sp500_constituents](https://github.com/hanshof/sp500_constituents);
prices from the Polygon API. I wrote the analysis and modelling, with AI assistance for the
repository restructure, the null-simulation and volatility-adjustment implementations, and the
documentation. Not financial advice.
