"""Self-checks for the yearly mean-correlation / PC1 statistics.

The assertions encode the two limiting cases: independent noise must produce
near-zero mean correlation and a small PC1 share, and a single shared factor
must produce both in the high 0.8s. Anything in between is the market.
"""
import numpy as np
import pandas as pd

from src.results import bootstrap_ci, stats


def test_independent_noise_has_near_zero_correlation():
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(rng.normal(size=(500, 30)))
    mean_corr, pc1_share = stats(panel)
    assert abs(mean_corr) < 0.05
    assert pc1_share < 0.20


def test_single_factor_dominates_both_statistics():
    rng = np.random.default_rng(1)
    factor = rng.normal(size=(500, 1))
    panel = pd.DataFrame(factor + rng.normal(size=(500, 30)) * 0.3)
    mean_corr, pc1_share = stats(panel)
    assert mean_corr > 0.8
    assert pc1_share > 0.8


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(2)
    factor = rng.normal(size=(200, 1))
    panel = pd.DataFrame(factor + rng.normal(size=(200, 15)) * 0.5)
    mean_corr, _ = stats(panel)
    low, high = bootstrap_ci(panel, np.random.default_rng(3))
    assert low < mean_corr < high
    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
