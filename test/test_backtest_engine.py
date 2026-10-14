"""Integration tests for BacktestEngine: timing, fills, costs, shorts, and
several symbols at once. Prices are small enough to check by hand."""

import pandas as pd
import pytest

from engine.backtest_engine import BacktestEngine
from strategies.buy_and_hold import BuyAndHold
from strategies.strategy_base import Strategy

DAYS = pd.date_range("2024-01-01", periods=4, freq="D")


def table(columns):
    """Build a price table from {symbol: [price per day]}."""
    return pd.DataFrame(columns, index=DAYS, dtype=float)


@pytest.fixture
def one_symbol():
    """Four days of one symbol where Open and Close differ, so a test can
    tell which price a fill actually used. Returns (opens, closes)."""
    return table({"AAPL": [10, 20, 25, 28]}), table({"AAPL": [10, 22, 30, 24]})


@pytest.fixture
def two_symbols():
    """Two symbols that always open at 10 and 20, so share counts are easy."""
    opens = table({"A": [10, 10, 10, 10], "B": [20, 20, 20, 20]})
    closes = table({"A": [10, 10, 12, 12], "B": [20, 20, 20, 25]})
    return opens, closes


class FixedWeights(Strategy):
    """Test double that replays hard-coded weights: one row per day, one
    value per symbol."""

    def __init__(self, rows):
        self.rows = rows

    def generate_weights(self, closes):
        return pd.DataFrame(self.rows, index=closes.index, columns=closes.columns, dtype=float)


def run(prices, strategy, **kwargs):
    """Run one backtest on an (opens, closes) pair and return its result."""
    opens, closes = prices
    return BacktestEngine(opens, closes, strategy, **kwargs).run()


def equity(prices, strategy, **kwargs):
    return run(prices, strategy, **kwargs).equity_curve.tolist()


# --- one symbol: timing and sizing ---


def test_buy_and_hold_end_to_end(one_symbol):
    curve = run(one_symbol, BuyAndHold(), initial_cash=100.0).equity_curve

    # Day 0 marks at cash, then buys 5 shares at day 1's Open (20).
    # Every later day is 5 * Close.
    expected = pd.Series([100.0, 110.0, 150.0, 120.0], index=DAYS)
    pd.testing.assert_series_equal(curve, expected, check_names=False, check_freq=False)
    assert curve.name == "equity"


def test_buy_and_hold_keeps_leftover_cash(one_symbol):
    # 105 buys 5 whole shares at 20, leaving 5 in cash
    assert equity(one_symbol, BuyAndHold(), initial_cash=105.0) == pytest.approx([105, 115, 155, 125])


def test_a_weight_change_is_traded_at_the_next_open(one_symbol):
    # in on day 0 (fills day 1 at 20), out on day 1 (fills day 2 at 25)
    strategy = FixedWeights([[1], [0], [0], [0]])
    assert equity(one_symbol, strategy, initial_cash=100.0) == pytest.approx([100, 110, 125, 125])


def test_last_day_weight_is_never_traded(one_symbol):
    # Trades happen at the next day's Open, so the final day has no bar to trade on.
    strategy = FixedWeights([[0], [0], [0], [1]])
    assert equity(one_symbol, strategy, initial_cash=100.0) == pytest.approx([100] * 4)


def test_no_trade_when_cash_cannot_buy_one_share(one_symbol):
    # 15 is below the fill price of 20; whole shares only, so equity stays in cash.
    assert equity(one_symbol, BuyAndHold(), initial_cash=15.0) == pytest.approx([15] * 4)


def test_half_weight_buys_half_as_much(one_symbol):
    # 50 of the 100 buys 2 whole shares at 20, leaving 60 in cash
    strategy = FixedWeights([[0.5]] * 4)
    assert equity(one_symbol, strategy, initial_cash=100.0) == pytest.approx([100, 104, 120, 108])


def test_each_run_starts_from_a_clean_portfolio(one_symbol):
    # running the same engine twice must give the same answer, not carry state over
    opens, closes = one_symbol
    engine = BacktestEngine(opens, closes, FixedWeights([[1], [0], [0], [0]]), initial_cash=100.0)
    first, second = engine.run(), engine.run()
    assert len(second.trades) == 1
    pd.testing.assert_series_equal(first.equity_curve, second.equity_curve)


# --- trades ---


def test_closed_position_is_recorded_as_a_trade(one_symbol):
    trades = run(one_symbol, FixedWeights([[1], [0], [0], [0]]), initial_cash=100.0).trades

    assert len(trades) == 1
    trade = trades[0]
    assert trade.symbol == "AAPL"
    assert trade.entry_date == DAYS[1]
    assert trade.exit_date == DAYS[2]
    assert trade.cost == pytest.approx(100.0)  # 5 shares at 20
    assert trade.pnl == pytest.approx(25.0)  # sold at 25


def test_position_still_open_at_the_end_is_not_a_trade(one_symbol):
    assert run(one_symbol, BuyAndHold(), initial_cash=100.0).trades == []


# --- costs ---


def test_slippage_makes_buys_fill_higher_and_sells_lower(one_symbol):
    result = run(one_symbol, FixedWeights([[1], [0], [0], [0]]), initial_cash=100.0, slippage=0.10)

    # buy at 20 * 1.10 = 22 -> 4 shares (88), 12 cash left
    # sell at 25 * 0.90 = 22.5 -> 90 back
    assert result.trades[0].cost == pytest.approx(88.0)
    assert result.trades[0].pnl == pytest.approx(2.0)
    # day 1 marks 4 shares at the Close of 22, plus cash: 88 + 12
    assert result.equity_curve.tolist() == pytest.approx([100, 100, 102, 102])


def test_commission_is_charged_on_both_sides(one_symbol):
    result = run(
        one_symbol, FixedWeights([[1], [0], [0], [0]]), initial_cash=100.0, commission_rate=0.10
    )

    # each share costs 20 + 2 commission, so 100 buys 4 (88), leaving 12 cash
    # selling 4 at 25 brings in 100 - 10 commission = 90
    assert result.trades[0].cost == pytest.approx(88.0)
    assert result.trades[0].pnl == pytest.approx(2.0)
    assert result.equity_curve.tolist() == pytest.approx([100, 100, 102, 102])


def test_costs_never_improve_the_result(one_symbol):
    strategy = FixedWeights([[1], [0], [0], [0]])
    free = equity(one_symbol, strategy, initial_cash=1000.0)
    costly = equity(one_symbol, strategy, initial_cash=1000.0, commission_rate=0.001, slippage=0.001)
    assert costly[-1] < free[-1]


@pytest.mark.parametrize("kwargs", [{"commission_rate": -0.01}, {"slippage": -0.01}])
def test_rejects_negative_costs(one_symbol, kwargs):
    opens, closes = one_symbol
    with pytest.raises(ValueError, match="negative"):
        BacktestEngine(opens, closes, BuyAndHold(), **kwargs)


# --- short positions ---


def test_short_position_profits_when_the_price_falls_and_loses_when_it_rises(one_symbol):
    result = run(one_symbol, FixedWeights([[-1], [-1], [0], [0]]), initial_cash=100.0)

    # day 1: sell 5 shares short at 20 -> cash 200, owing 5 shares
    # day 1 close 22: 200 - 110 = 90.  day 2 close 30: 200 - 150 = 50
    # day 3: buy the 5 back at 28 (140) -> cash 60
    assert result.equity_curve.tolist() == pytest.approx([100, 90, 50, 60])
    assert result.trades[0].pnl == pytest.approx(-40.0)  # sold at 20, bought back at 28


def test_switching_from_long_to_short_closes_the_long_first(one_symbol):
    result = run(one_symbol, FixedWeights([[1], [-1], [-1], [-1]]), initial_cash=100.0)

    # day 1: buy 5 at 20.  day 2 open 25: sell those 5 (+25 profit, cash 125),
    # then short 5 more (cash 250).  close 30: 250 - 150.  close 24: 250 - 120.
    assert result.equity_curve.tolist() == pytest.approx([100, 110, 100, 130])
    assert len(result.trades) == 1  # the long; the short is still open
    assert result.trades[0].pnl == pytest.approx(25.0)


# --- several symbols ---


def test_portfolio_is_split_across_symbols(two_symbols):
    # 50 each: 5 shares of A at 10, 2 shares of B at 20 (40), 10 left in cash
    curve = equity(two_symbols, BuyAndHold(), initial_cash=100.0)
    # then 10 + 5 * A + 2 * B at each close
    assert curve == pytest.approx([100, 100, 110, 120])


def test_only_a_symbol_whose_weight_changed_is_traded(two_symbols):
    # B is dropped on day 1; A's weight stays 0.5 the whole time
    result = run(two_symbols, FixedWeights([[0.5, 0.5], [0.5, 0], [0.5, 0], [0.5, 0]]), initial_cash=100.0)

    # day 2: B's 2 shares are sold at 20 (cash 10 -> 50); A's 5 shares are untouched
    assert result.equity_curve.tolist() == pytest.approx([100, 100, 110, 110])
    assert [trade.symbol for trade in result.trades] == ["B"]


def test_sells_happen_before_buys_so_their_cash_can_be_reused(two_symbols):
    # all in A, then switch everything to B: the buy of B is only
    # affordable once A has been sold
    curve = equity(two_symbols, FixedWeights([[1, 0], [0, 1], [0, 1], [0, 1]]), initial_cash=100.0)
    # 10 shares of A -> sold for 100 -> 5 shares of B, worth 5 * 25 on the last day
    assert curve == pytest.approx([100, 100, 100, 125])


def test_long_one_symbol_and_short_the_other(two_symbols):
    curve = equity(two_symbols, FixedWeights([[0.5, -0.5]] * 4), initial_cash=100.0)
    # buy 5 A (50), short 2 B (+40): cash 90.  equity = 90 + 5 * A - 2 * B
    assert curve == pytest.approx([100, 100, 110, 100])


def test_opens_and_closes_must_line_up(two_symbols):
    opens, closes = two_symbols
    with pytest.raises(ValueError, match="same days and symbols"):
        BacktestEngine(opens, closes[["A"]], BuyAndHold())
