"""Integration tests for BacktestEngine: signal timing, fills, and equity curve."""

import pandas as pd
import pytest

from engine.backtest_engine import BacktestEngine
from strategies.buy_and_hold import BuyAndHold
from strategies.strategy_base import Strategy


@pytest.fixture
def prices():
    """Four days where Open and Close differ, so a test can tell which
    price a fill actually used."""
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "Open": [10.0, 20.0, 25.0, 28.0],
            "Close": [10.0, 22.0, 30.0, 24.0],
        },
        index=index,
    )


class FixedSignals(Strategy):
    """Test double that replays a hard-coded signal per day (1 = long, 0 = flat)."""

    def __init__(self, signals):
        self.signals = signals

    def generate_signals(self, df):
        return pd.Series(self.signals, index=df.index, dtype=int)


def test_buy_and_hold_end_to_end(prices):
    equity = BacktestEngine(prices, BuyAndHold(), initial_cash=100.0).run()

    # Day 0 marks at cash, then buys 5 shares at day 1's Open (20).
    # Every later day is 5 * Close.
    expected = pd.Series([100.0, 110.0, 150.0, 120.0], index=prices.index)
    pd.testing.assert_series_equal(equity, expected, check_names=False, check_freq=False)
    assert equity.name == "equity"


def test_buy_and_hold_keeps_leftover_cash(prices):
    # 105 buys 5 whole shares at 20, leaving 5 in cash
    equity = BacktestEngine(prices, BuyAndHold(), initial_cash=105.0).run()
    assert equity.tolist() == pytest.approx([105.0, 115.0, 155.0, 125.0])


def test_sell_signal_fills_at_next_open(prices):
    # long on day 0 (fills day 1 at 20), flat on day 1 (fills day 2 at 25)
    equity = BacktestEngine(prices, FixedSignals([1, 0, 0, 0]), initial_cash=100.0).run()
    assert equity.tolist() == pytest.approx([100.0, 110.0, 125.0, 125.0])


def test_last_day_signal_is_never_filled(prices):
    # Fills happen at the next day's Open, so a final-day signal has no bar to fill on.
    equity = BacktestEngine(prices, FixedSignals([0, 0, 0, 1]), initial_cash=100.0).run()
    assert equity.tolist() == pytest.approx([100.0] * 4)


def test_no_trade_when_cash_cannot_buy_one_share(prices):
    # 15 is below the fill price of 20; whole shares only, so equity stays in cash.
    equity = BacktestEngine(prices, BuyAndHold(), initial_cash=15.0).run()
    assert equity.tolist() == pytest.approx([15.0] * 4)
