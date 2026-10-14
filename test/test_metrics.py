"""Unit tests for performance metrics, checked against hand-computed values."""

import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from engine.trade import Trade
from performance import metrics


def curve(values):
    """Build an equity curve over consecutive days from a list of values."""
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=index, dtype=float)


def trade(entry_price, exit_price, quantity=10):
    """Build a completed trade; the dates don't matter to any test here."""
    return Trade("AAPL", datetime(2024, 1, 2), datetime(2024, 1, 9), quantity, entry_price, exit_price)


def test_total_return():
    assert metrics.total_return(curve([100, 110, 121])) == pytest.approx(0.21)


def test_annualized_return_over_exactly_one_year_equals_total_return():
    # 253 points = 252 periods = one trading year
    equity = curve(np.linspace(100, 150, 253))
    assert metrics.annualized_return(equity) == pytest.approx(0.5)


def test_annualized_return_compounds_over_two_years():
    # +44% over two years is 20% a year compounded (1.2 ** 2 = 1.44)
    equity = curve(np.linspace(100, 144, 505))
    assert metrics.annualized_return(equity) == pytest.approx(0.2)


def test_annualized_return_of_single_point_is_zero():
    assert metrics.annualized_return(curve([100])) == 0.0


def test_annualized_volatility_scales_daily_std_by_sqrt_time():
    equity = curve([100, 110, 99, 108.9])  # daily returns: +10%, -10%, +10%
    daily_std = pd.Series([0.10, -0.10, 0.10]).std()
    assert metrics.annualized_volatility(equity) == pytest.approx(daily_std * math.sqrt(252))


def test_sharpe_ratio_matches_definition():
    equity = curve([100, 110, 99, 108.9])
    returns = pd.Series([0.10, -0.10, 0.10])
    expected = (returns.mean() * 252 - 0.02) / (returns.std() * math.sqrt(252))
    assert metrics.sharpe_ratio(equity, risk_free_rate=0.02) == pytest.approx(expected)


def test_sharpe_ratio_of_flat_curve_is_zero():
    # no volatility: avoid dividing by zero
    assert metrics.sharpe_ratio(curve([100, 100, 100])) == 0.0


def test_sortino_only_counts_downside_moves():
    equity = curve([100, 110, 99, 108.9])
    # downside returns are [0, -0.10, 0]; deviation = sqrt(mean of squares)
    downside_deviation = math.sqrt(0.10**2 / 3) * math.sqrt(252)
    expected = (pd.Series([0.10, -0.10, 0.10]).mean() * 252) / downside_deviation
    assert metrics.sortino_ratio(equity) == pytest.approx(expected)


def test_sortino_of_curve_that_never_falls_is_zero():
    assert metrics.sortino_ratio(curve([100, 110, 120])) == 0.0


def test_max_drawdown_is_largest_peak_to_trough_decline():
    # peak 120 -> trough 90 is -25%; the later dip from 130 is smaller
    equity = curve([100, 120, 90, 130, 117])
    assert metrics.max_drawdown(equity) == pytest.approx(-0.25)


def test_max_drawdown_of_rising_curve_is_zero():
    assert metrics.max_drawdown(curve([100, 110, 120])) == 0.0


def test_drawdown_series_is_zero_at_each_new_high():
    drawdown = metrics.drawdown_series(curve([100, 120, 90, 130]))
    assert drawdown.tolist() == pytest.approx([0.0, 0.0, -0.25, 0.0])


def test_calmar_ratio_is_annualized_return_over_max_drawdown():
    equity = curve([100, 120, 90, 130, 117])
    expected = metrics.annualized_return(equity) / 0.25
    assert metrics.calmar_ratio(equity) == pytest.approx(expected)


def test_compute_metrics_returns_every_metric():
    result = metrics.compute_metrics(curve([100, 120, 90, 130]))
    assert set(result) == {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
    }


def test_trade_statistics():
    # pnl: +200, -100, +100
    trades = [trade(100, 120), trade(100, 90), trade(100, 110)]
    stats = metrics.trade_statistics(trades)
    assert stats["num_trades"] == 3
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["avg_trade_return"] == pytest.approx((0.20 - 0.10 + 0.10) / 3)
    assert stats["profit_factor"] == pytest.approx(300 / 100)


def test_trade_statistics_with_no_losing_trades_has_infinite_profit_factor():
    stats = metrics.trade_statistics([trade(100, 120)])
    assert stats["win_rate"] == 1.0
    assert stats["profit_factor"] == float("inf")


def test_trade_statistics_with_no_trades():
    stats = metrics.trade_statistics([])
    assert stats["num_trades"] == 0
    assert math.isnan(stats["win_rate"])


def test_format_duration():
    start, end = pd.Timestamp("2020-01-02"), pd.Timestamp("2021-01-03")
    assert metrics.format_duration(start, end) == "1 year, 1 day"
