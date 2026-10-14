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


def test_short_signal_is_treated_as_flat(prices):
    # long-only engine: -1 closes the position exactly like 0 would
    equity = BacktestEngine(prices, FixedSignals([1, -1, -1, -1]), initial_cash=100.0).run()
    assert equity.tolist() == pytest.approx([100.0, 110.0, 125.0, 125.0])


def test_completed_round_trip_is_recorded_as_a_trade(prices):
    engine = BacktestEngine(prices, FixedSignals([1, 0, 0, 0]), initial_cash=100.0, symbol="AAPL")
    engine.run()

    assert len(engine.trades) == 1
    trade = engine.trades[0]
    assert trade.symbol == "AAPL"
    assert trade.entry_date == prices.index[1]
    assert trade.exit_date == prices.index[2]
    assert trade.quantity == 5
    assert trade.entry_price == 20.0
    assert trade.exit_price == 25.0
    assert trade.pnl == pytest.approx(25.0)


def test_position_still_open_at_the_end_is_not_a_completed_trade(prices):
    engine = BacktestEngine(prices, BuyAndHold(), initial_cash=100.0)
    engine.run()
    assert engine.trades == []


def test_running_twice_does_not_duplicate_trades(prices):
    engine = BacktestEngine(prices, FixedSignals([1, 0, 0, 0]), initial_cash=100.0)
    engine.run()
    engine.run()
    assert len(engine.trades) == 1


def test_slippage_makes_buys_fill_higher_and_sells_lower(prices):
    engine = BacktestEngine(prices, FixedSignals([1, 0, 0, 0]), initial_cash=100.0, slippage=0.10)
    equity = engine.run()

    # buy at 20 * 1.10 = 22 -> 4 shares, 12 cash left; sell at 25 * 0.90 = 22.5
    trade = engine.trades[0]
    assert trade.quantity == 4
    assert trade.entry_price == pytest.approx(22.0)
    assert trade.exit_price == pytest.approx(22.5)
    # day 1 marks 4 shares at the Close of 22, plus cash: 88 + 12
    assert equity.tolist() == pytest.approx([100.0, 100.0, 102.0, 102.0])


def test_commission_is_charged_on_both_sides(prices):
    engine = BacktestEngine(
        prices, FixedSignals([1, 0, 0, 0]), initial_cash=100.0, commission_rate=0.10
    )
    equity = engine.run()

    # each share costs 20 + 2 commission, so 100 buys 4 (88), leaving 12 cash
    # selling 4 at 25 brings in 100 - 10 commission = 90
    trade = engine.trades[0]
    assert trade.quantity == 4
    assert trade.entry_commission == pytest.approx(8.0)
    assert trade.exit_commission == pytest.approx(10.0)
    assert trade.pnl == pytest.approx(2.0)
    assert equity.tolist() == pytest.approx([100.0, 100.0, 102.0, 102.0])


def test_costs_never_improve_the_result(prices):
    signals = FixedSignals([1, 0, 0, 0])
    free = BacktestEngine(prices, signals, initial_cash=1000.0).run()
    costly = BacktestEngine(
        prices, signals, initial_cash=1000.0, commission_rate=0.001, slippage=0.001
    ).run()
    assert costly.iloc[-1] < free.iloc[-1]


@pytest.mark.parametrize("kwargs", [{"commission_rate": -0.01}, {"slippage": -0.01}])
def test_rejects_negative_costs(prices, kwargs):
    with pytest.raises(ValueError, match="negative"):
        BacktestEngine(prices, BuyAndHold(), **kwargs)
