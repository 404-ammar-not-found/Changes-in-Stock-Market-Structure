"""The whole project in one figure.

Three stacked panels sharing the year axis: the headline correlation series
with bootstrap CIs and the PC1 control; the Forbes-Rigobon volatility
counterfactual; and the Graphical Lasso edge density against the one-factor
null band. Reads the CSVs produced by results.py, forbes_rigobon.py and
one_factor_null.py.

Run: python -m src.plot_summary      (after those three)
Writes: figures/summary.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(FIGURES / name).set_index("year")


def main() -> None:
    results = load("yearly_results.csv")
    fr = load("forbes_rigobon.csv")
    null = load("one_factor_null.csv")

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    # ---- panel 1: the headline series and its PC1 control ------------------
    ax = axes[0]
    usable = results.dropna(subset=["mean_corr"])
    ax.fill_between(usable.index, usable.ci_low, usable.ci_high,
                    color="tab:blue", alpha=0.20, label="95% bootstrap CI")
    ax.plot(usable.index, usable.mean_corr, color="tab:blue", marker="o", ms=5,
            lw=1.8, label="mean pairwise correlation")
    ax.plot(usable.index, usable.pc1_share, color="tab:orange", ls="--", lw=1.5,
            marker=".", ms=5, label="PC1 variance share (control)")
    ax.set_ylabel("mean correlation / PC1 share")
    ax.set_title("Co-movement of S&P 500 constituents, 2004-2024", fontsize=14)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3)

    # ---- panel 2: Forbes-Rigobon counterfactual ----------------------------
    ax = axes[1]
    ax.plot(fr.index, fr.mean_corr, color="tab:red", marker="o", ms=5, lw=1.8,
            label="measured")
    ax.plot(fr.index, fr.mean_corr_at_typical_vol, color="tab:blue", marker="s",
            ms=5, lw=1.8, label="at typical volatility (Forbes-Rigobon counterfactual)")
    ax.fill_between(fr.index, fr.mean_corr_at_typical_vol, fr.mean_corr,
                    color="grey", alpha=0.25, label="volatility artefact")
    ax.set_ylabel("mean pairwise correlation")
    ax.set_title("The spikes are volatility, not structure", fontsize=14)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3)

    # ---- panel 3: edge density vs one-factor null --------------------------
    ax = axes[2]
    ax.fill_between(null.index, null.null_mean - 2 * null.null_sd,
                    null.null_mean + 2 * null.null_sd, color="tab:green",
                    alpha=0.20, label="one-factor null (mean ± 2 sd)")
    ax.plot(null.index, null.null_mean, color="tab:green", ls="--", lw=1.5)
    ax.plot(null.index, null.observed_density, color="black", marker="o", ms=5,
            lw=1.8, label="observed")
    ax.set_ylabel("Graphical Lasso edge density at fixed alpha")
    ax.set_title("Conditional structure beyond beta, vs matched beta-only markets", fontsize=14)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xticks(list(results.index))
    ax.set_xticklabels([str(y) if y not in (2019, 2020) else f"{y}\n(no data)"
                        for y in results.index], fontsize=9)

    fig.tight_layout()
    out = FIGURES / "summary.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
