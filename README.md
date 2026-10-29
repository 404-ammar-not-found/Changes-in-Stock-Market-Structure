# How did the stock market change from 2004 to 2024?

**Finding.** Mean pairwise correlation of daily returns between S&P 500 members roughly doubled
from 0.20 in 2005 to 0.45 in 2008, peaked at **0.55 in 2011** — not in the crisis year — collapsed
to a 21-year low of **0.13 in 2017**, spiked again to **0.42 in the 2022 rate shock**, and sat at
0.19 in 2024. The first principal component tracks that series to within about 0.02 every year, so
what is being measured is the strength of the single market factor.

**So what.** Diversification inside the index decays precisely when it is needed. A factor risk
model calibrated on 2017 covariance would have understated 2008- or 2022-style co-movement by a
factor of three.

**Caveats up front.** The 2008 rise is real (bootstrap CIs for 2007 and 2008 barely separate); the
2011 peak is *not* statistically distinguishable from 2008. Price coverage before 2019 is ~40% of
trading days, which costs roughly ±0.05 of accuracy — see [Data quality](#data-quality). This is
descriptive analysis, not a hypothesis test.

Reproduce with `python src/results.py`.

| year | coverage | tickers | mean corr | 95% CI | PC1 share |
|-----:|---------:|--------:|----------:|:--------------|----------:|
| 2004 | 0.38 | 387 | 0.232 | 0.167 – 0.304 | 0.256 |
| 2005 | 0.38 | 391 | 0.198 | 0.156 – 0.245 | 0.225 |
| 2006 | 0.43 | 380 | 0.220 | 0.160 – 0.278 | 0.253 |
| 2007 | 0.50 | 371 | 0.348 | 0.286 – 0.403 | 0.369 |
| 2008 | 0.48 | 379 | 0.448 | 0.363 – 0.542 | 0.462 |
| 2009 | 0.43 | 390 | 0.411 | 0.357 – 0.471 | 0.440 |
| 2010 | 0.38 | 401 | 0.461 | 0.381 – 0.537 | 0.483 |
| 2011 | 0.40 | 404 | **0.553** | 0.453 – 0.638 | 0.569 |
| 2012 | 0.40 | 414 | 0.308 | 0.244 – 0.377 | 0.334 |
| 2013 | 0.40 | 418 | 0.304 | 0.234 – 0.373 | 0.331 |
| 2014 | 0.38 | 427 | 0.356 | 0.265 – 0.443 | 0.386 |
| 2015 | 0.44 | 425 | 0.347 | 0.269 – 0.435 | 0.373 |
| 2016 | 0.49 | 437 | 0.300 | 0.233 – 0.382 | 0.330 |
| 2017 | 0.43 | 447 | **0.128** | 0.097 – 0.170 | 0.162 |
| 2018 | 0.44 | 460 | 0.306 | 0.239 – 0.380 | 0.341 |
| 2019 | 0.10 | 465 | — | excluded | — |
| 2020 | 0.05 | 475 | — | excluded | — |
| 2021 | 0.35 | 474 | 0.190 | 0.138 – 0.265 | 0.252 |
| 2022 | 0.96 | 475 | 0.421 | 0.386 – 0.469 | 0.441 |
| 2023 | 0.96 | 476 | 0.265 | 0.242 – 0.299 | 0.286 |
| 2024 | 0.96 | 483 | 0.187 | 0.160 – 0.219 | 0.210 |

`coverage` is the share of the year's trading days actually present in the data. Read it before
reading anything else in the row.

## What this is testing

The question — "did the structure of the market change?" — is not directly measurable, so it is
broken into four narrower ones. Each is a separate measurement with a separate way of coming out
negative.

**1. How much do S&P 500 stocks move together, and does it change?**
Measured by mean pairwise correlation of daily log returns within each calendar year. A flat series
would mean co-movement is a constant of the market and there is no story here. It is not flat: it
ranges from 0.13 to 0.55, a four-fold spread, and the high years are 2008–2011 and 2022.

**2. Is that co-movement one thing, or many?**
Measured by the share of total variance explained by the first principal component. This is the
control on question 1. If correlation rose because the market became genuinely more interconnected,
PC1 share and mean correlation would come apart. They do not — they track to within about 0.02 in
every single year. So the honest reading is that this series measures **the strength of the single
market factor**, not the emergence of new structure. A rising correlation heatmap is mostly beta,
and this column is the evidence for saying so rather than a hedge.

**3. Is there any dependency left once the market factor is removed?**
Measured by the Graphical Lasso, which estimates *partial* correlations — the dependency between two
stocks after conditioning on every other stock, which strips out the common component that dominates
question 2. This is the measurement that survives the "isn't this just beta?" objection. It is also
the one forced on us by the data: with ~400 stocks and as few as ~100 observed days in a year, the
sample covariance matrix is singular and cannot be inverted at all, so a regularised estimator is a
requirement rather than a refinement.

**4. Does the market organise itself along sector lines, and does that survive a crisis?**
Measured by Affinity Propagation clustering on the annual correlation matrix, with clusters compared
against sector labels. The intuition being tested is that in calm years stocks should group by
industry, and in a panic — when everything falls together — sector identity should dissolve into
fewer, larger, more indiscriminate clusters. Cluster counts are only comparable across years because
`preference` is held fixed; with sklearn's default the count would move whenever the similarity
distribution moved, which is circular.

**A trap worth naming.** Two separate filters in the notebook silently deleted every stock that
failed to survive the whole period — a global coverage threshold at load, and a second one inside
the returns helper that fed the MDS layout. Bear Stearns sits at 0.19 coverage of the 2004–2024
panel, Merrill at 0.22, Wachovia at 0.22, National City at 0.22, Countrywide at 0.19; all five were
dropped by a `>= 0.50` test. Filtering to index-membership windows is necessary for avoiding
survivorship bias but nowhere near sufficient, because an innocuous-looking data-hygiene threshold
downstream can reintroduce the whole problem. Both are now set to admit short-lived constituents,
with per-window adequacy enforced separately.

**What this does not test.** There is no causal claim and no hypothesis test. The bootstrap CIs
quantify sampling uncertainty in each year's estimate, and that is all they do — they say nothing
about *why* correlation rose. Nor is the direction of causation addressed: correlation and
volatility rise together, and Forbes and Rigobon (2002) show that measured correlation increases
mechanically with volatility, so an unknown part of the 2008 and 2022 spikes is a statistical
artefact of turbulent markets rather than evidence of contagion. Establishing contagion would require
the heteroskedasticity correction from that paper, which is not implemented here.

## What the network and clustering actually found

Both are run at hyperparameters held fixed across years (see Method), so these columns are
comparable year to year. `edge density` is the share of stock pairs retaining a non-zero *partial*
correlation after the L1 penalty; `clusters` is the number of Affinity Propagation exemplars.

| year | mean corr | edge density | clusters |
|-----:|----------:|-------------:|---------:|
| 2004 | 0.232 | 0.008 | 47 |
| 2005 | 0.198 | 0.006 | 51 |
| 2006 | 0.220 | 0.008 | 45 |
| 2007 | 0.348 | 0.021 | 36 |
| 2008 | 0.448 | 0.052 | 28 |
| 2009 | 0.411 | 0.049 | 30 |
| 2010 | 0.461 | 0.057 | 26 |
| 2011 | **0.553** | **0.080** | **15** |
| 2012 | 0.308 | 0.021 | 36 |
| 2013 | 0.304 | 0.019 | 42 |
| 2014 | 0.356 | 0.034 | 33 |
| 2015 | 0.347 | 0.030 | 38 |
| 2016 | 0.300 | 0.023 | 43 |
| 2017 | **0.128** | **0.007** | **74** |
| 2018 | 0.306 | 0.025 | 46 |
| 2021 | 0.190 | 0.028 | 49 |
| 2022 | 0.421 | 0.045 | 35 |
| 2023 | 0.265 | 0.011 | 67 |
| 2024 | 0.187 | 0.006 | 82 |

The three series move together almost perfectly (Spearman, n=19):

| pair | rho | p |
|:--|--:|--:|
| mean correlation vs edge density | +0.88 | 6e-07 |
| mean correlation vs cluster count | −0.96 | 2e-10 |
| edge density vs cluster count | −0.88 | 8e-07 |

In 2011 the market fell to 15 clusters; by 2024 it had 82. That is the "sector identity dissolves in
a panic" effect, and it is visible only because `preference` is fixed — at sklearn's default the same
series is nearly flat (38 in 2004 to 63 in 2024) and shows nothing at all.

**But the deflationary reading is the honest one.** A rank correlation of −0.96 is not three methods
independently corroborating a finding; it is strong evidence that at fixed hyperparameters both the
network density and the cluster count are close to **monotone transforms of the average correlation
level**. The mechanism is not subtle: when correlations rise uniformly, more precision-matrix entries
clear a fixed L1 threshold, and more stocks look similar enough to share an exemplar. So the network
and the clustering here largely restate the correlation series in two other vocabularies rather than
revealing structure independent of it. Combined with the PC1 column — which says the correlation
series is itself mostly one factor — the defensible summary of this project is that it measures
**the strength of the market factor over 21 years, three ways**, and not that it has found hidden
structure.

**What would actually separate them**, and the obvious next step: simulate a one-factor market with
the same number of stocks, the same number of days and the same average correlation as each observed
year, run the identical Graphical Lasso, and compare densities. If the observed density
significantly exceeds the one-factor null, there is genuine conditional structure beyond beta; if it
does not, the deflationary reading is confirmed. That test is not implemented here, so question 3
above is posed rather than answered.

## Reading the figures

`figures/stock_market_structure_mds.gif` — one frame per year. Each node is a stock, positioned by
metric MDS on the full-sample correlation distance, so **positions are fixed across all frames** and
movement between years is never an artefact of the layout algorithm. Edges are the strongest partial
correlations from the Graphical Lasso; thicker means stronger. Node colour is sector.

`figures/stock_market_structure_affinity_propagation.gif` — same layout, but each node is joined to
its cluster exemplar and **coloured by cluster**, so the grouping is what you actually see. The
legend gives the cluster count for that year.

Only the 40 most connected nodes in each frame are labelled — with ~500 stocks per year, labelling
every one produces overlapping text that hides the structure it is meant to describe. In the
clustering frames those labels are effectively the exemplars.

Two things not to read into them. Each node keeps only its top-*k* strongest edges, so total edge
count is fixed by construction and cannot be used to judge whether the network "densified" — use the
table for that. And MDS coordinates carry no units; only relative position means anything. Each
frame is titled with the share of the year's trading days it was built from, which for pre-2019
frames is around 40%.

## Data quality

The original Polygon pull returned far less than it appeared to. Polygon **degrades instead of
failing** when a request crosses the account's history entitlement: it answers `200 OK` with a
partial payload rather than an error. The ingest wrapped the call in `except Exception`, which
catches a dropped connection or a 403 but cannot catch a successful response carrying 40% of the
rows, and nothing downstream compared row counts against expected trading days. So 867 clean-looking
CSVs landed on disk and the correlation matrices were computed on ~110 days a year while appearing
to cover 252.

The generalisable lesson, and the reason `src/repull_polygon.py` now ends with a coverage
assertion: **an ingest that only catches exceptions is unguarded, because the common failure mode of
a rate- or entitlement-limited API is partial success, not failure.**

What was ruled out along the way: the bars themselves are exact (close, volume and transaction
count all match a fresh pull on 432 overlapping dates), and the client library is not at fault (its
chunk boundaries don't align with the coverage regimes, and its full-range merge returns the same
817 bars as a single request).

**How much does the sparsity distort the answer?** 2022–2024 are complete, so the true daily figure
is known for those years. Imposing a real sparse year's observed date pattern on them and
recomputing (`python src/sparse_bias_check.py`) gives a bias of **+0.025 on average, ranging −0.041
to +0.076**. So the *shape* of the 2004–2018 series is trustworthy — every year there is sampled the
same way — but its *level* reads slightly high, and comparing a sparse year directly against a
complete one (2004 vs 2024, say) overstates the difference by roughly 0.03.

2019 and 2020 cannot be recovered: the account's history window now starts 2021-09-07 and everything
earlier returns 403. **COVID is therefore absent from this analysis.**

## Method

Universe is historical S&P 500 membership, so each stock contributes returns only for days it was
actually in the index; delisted constituents (BSC, MER, WB, NCC, CFC) are retained. Daily log
returns, correlations per calendar year over tickers trading on ≥90% of that year's observed dates.
CIs are 200-sample bootstraps resampling days.

1. `src/notebooks/data_preparation.ipynb` — original Polygon pull, reshape, filter to membership windows.
2. `src/repull_polygon.py` — re-pull of the still-available window, with a completeness assertion.
3. `src/merge_repull.py` — splice the two pulls; reconcile split adjustment across pull dates.
4. `src/build_db.py` — load into SQLite (`data_db/market_data.db`).
5. `src/results.py` — the table above. `src/sparse_bias_check.py` — the bias estimate.
6. `src/fetch_sectors.py` — fill in missing sector labels from yfinance.
7. `src/notebooks/market_structure_analysis.ipynb` — annual networks and animations.

Both pulls are split-adjusted *as of their own pull date*, so a ticker that split in between sits on
two different price scales. Step 3 derives the factor from overlapping dates and rescales, otherwise
a fabricated return appears at the 2021-09-07 seam. It caught five genuine corporate actions — APH
2:1, MNST 2:1, CRWD 4:1, DD 1:3 reverse, PARA 1:20 reverse — each confirmed by a reciprocal volume
ratio, which distinguishes a split from a reused ticker symbol.

### Why a network, not just a correlation matrix

Raw correlation between 500 stocks is dominated by the market factor, so a rising heatmap says
little beyond "beta went up" — and the PC1 column above shows exactly how little. The network is
estimated with the **Graphical Lasso**, which gives *partial* correlations: it conditions out the
common component and leaves the conditional dependencies. With ~400 stocks and ~110 observed days
per year the sample covariance is singular, so an L1-regularised estimator is required rather than
optional.

The L1 penalty (`alpha`) and the Affinity Propagation `preference` are calibrated **once** on the
pooled sample and held fixed for every annual frame. sklearn's defaults refit both per year —
`GraphicalLassoCV` picks a new penalty, `AffinityPropagation` defaults `preference` to the median
input similarity — which would make year-over-year edge and cluster counts partly an artefact of the
estimator rather than a fact about the market.

Note also that the network figures fix each node's degree at the top-*k* strongest edges, so edge
count is constant by construction. Read density from the table, not from the pictures.

## Limitations

- **2019–2020 missing entirely**, and 2021 is only 35% covered. COVID is out of scope.
- **Sparse sampling before 2019** biases the level by roughly +0.03 (quantified above).
- **Residual survivorship bias.** Membership filtering is correct and crisis casualties are present,
  but 38 of 903 constituent tickers never fetched, skewed toward post-bankruptcy symbols (ABKFQ,
  EKDKQ, DPHIQ, FNMA, FMCC), so the crisis years are mildly optimistic.
- **Correlation is conditional on volatility.** Forbes and Rigobon (2002) show measured correlation
  rises mechanically with volatility, so part of the 2008 and 2022 increases is heteroskedasticity
  rather than contagion. Not corrected for here.
- **Calendar-year windows are arbitrary.** The crisis straddles the 2008/2009 boundary; rolling
  windows would be the better design.
- **MDS coordinates** are fitted once on the full sample and reused for every frame, so positions are
  comparable across years, but they carry no units.
- **Sector labels** are yfinance's (Morningstar-style), not true GICS. 610 of 872 panel tickers
  (70%) are labelled; the remaining 262 are mostly delisted symbols yfinance no longer serves, so
  the `Unknown` share is concentrated in the early years — exactly where the clusters-vs-sectors
  comparison is therefore weakest.
- **Some Graphical Lasso fits do not fully converge** within 200 iterations (dual gaps of order
  0.05 in the high-correlation years). The densities are therefore approximate, and small
  differences between adjacent years should not be read as meaningful; the large swings (0.006 in
  2024 against 0.080 in 2011) are well outside that slack.
- **A ridge of 0.10** is added to each year's empirical correlation matrix before the Graphical Lasso
  fit. Without it 2011 is too ill-conditioned to solve at all (p=416 against n=105). It is applied
  identically every year, and the ranking of yearly densities is unchanged across ridge values from
  0.01 to 0.10, so it affects conditioning rather than the result.

## Data source

Historical constituents from https://github.com/hanshof/sp500_constituents — the inspiration for
this repo. Prices from the Polygon API. `src/data/` and the SQLite DB are gitignored for size;
re-create them with steps 1–4 above and a `POLYGON_KEY` in `src/notebooks/.env`. Note that step 1
can no longer reproduce the pre-2019 data on a current free-tier key.

### Disclaimer

I wrote the analysis and modelling. I used Codex to tidy the plotting helpers and file structure at
the end. Do not use any of this to inform financial decisions.
