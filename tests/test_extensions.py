"""Self-checks for the rolling-window, out-of-sample risk and cross-asset modules."""
from src import cross_asset, oos_risk, rolling


def test_rolling_annualisation_puts_sparse_and_complete_windows_on_one_scale():
    rolling.demo()


def test_calm_calibrated_model_understates_a_turbulent_year_by_the_volatility_jump():
    oos_risk.demo()


def test_cross_asset_hedge_statistics():
    cross_asset.demo()
