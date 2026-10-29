"""Self-checks for splicing the two Polygon pulls across a split.

A 4:1 split between pull dates must be undone by rescaling onto the newer
basis rather than left as a fake -75% return at the seam; a ratio within
tolerance of 1 must be left untouched.
"""
import numpy as np
import pandas as pd

from src.merge_repull import rescale_old


def panel(dates, price):
    return pd.DataFrame({"date": dates, "open": price, "close": price})


def test_split_between_pulls_is_rescaled():
    dates = pd.bdate_range("2021-09-07", periods=40)
    old, new = panel(dates, 100.0), panel(dates, 25.0)
    scaled, ratio = rescale_old(old, new)
    assert abs(ratio - 0.25) < 1e-9
    assert abs(scaled.close.iloc[0] - 25.0) < 1e-9


def test_noise_level_ratio_is_untouched():
    dates = pd.bdate_range("2021-09-07", periods=40)
    new = panel(dates, 25.0)
    old = panel(dates, 25.1)
    unchanged, _ = rescale_old(old, new)
    assert abs(unchanged.close.iloc[0] - 25.1) < 1e-9


def test_no_overlap_returns_unscaled():
    old_dates = pd.bdate_range("2021-01-04", periods=20)
    new_dates = pd.bdate_range("2021-09-07", periods=20)
    old, new = panel(old_dates, 100.0), panel(new_dates, 100.0)
    same, ratio = rescale_old(old, new)
    assert np.isnan(ratio)
    assert same.close.iloc[0] == 100.0
