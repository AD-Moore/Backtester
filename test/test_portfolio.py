"""Unit tests for Portfolio: cash accounting, order execution, and equity."""

from datetime import datetime

import pytest

from engine.order import Order
from engine.portfolio import Portfolio


# Arbitrary fixed timestamp; no test here depends on its value.
DATE = datetime(2024, 1, 2)


def buy(quantity, symbol="AAPL"):
    """Build a buy order for the given quantity."""
    return Order(DATE, symbol, "buy", quantity)


def sell(quantity, symbol="AAPL"):
    """Build a sell order for the given quantity."""
    return Order(DATE, symbol, "sell", quantity)


def test_position_is_created_on_demand_and_reused():
    portfolio = Portfolio(cash=1000.0)
    pos = portfolio.position("AAPL")
    assert not pos.is_open
    # Repeat lookups return the same object, not a fresh flat position.
    assert portfolio.position("AAPL") is pos


def test_buy_moves_cash_into_position():
    portfolio = Portfolio(cash=1000.0)
    portfolio.execute(buy(5), 100.0)
    assert portfolio.cash == pytest.approx(500.0)
    assert portfolio.position("AAPL").quantity == 5


def test_buy_using_exactly_all_cash_is_allowed():
    portfolio = Portfolio(cash=1000.0)
    portfolio.execute(buy(10), 100.0)
    assert portfolio.cash == pytest.approx(0.0)


def test_buy_without_enough_cash_raises_and_changes_nothing():
    portfolio = Portfolio(cash=1000.0)
    with pytest.raises(ValueError, match="insufficient cash"):
        portfolio.execute(buy(11), 100.0)
    # A rejected order must leave both cash and the position unchanged.
    assert portfolio.cash == 1000.0
    assert not portfolio.position("AAPL").is_open


def test_sell_returns_proceeds_to_cash():
    portfolio = Portfolio(cash=1000.0)
    portfolio.execute(buy(10), 100.0)
    portfolio.execute(sell(4), 120.0)
    # 1000 - 10 * 100 + 4 * 120
    assert portfolio.cash == pytest.approx(480.0)
    assert portfolio.position("AAPL").quantity == 6


def test_oversell_raises_and_leaves_cash_untouched():
    portfolio = Portfolio(cash=1000.0)
    portfolio.execute(buy(5), 100.0)
    with pytest.raises(ValueError, match="cannot sell"):
        portfolio.execute(sell(6), 100.0)
    assert portfolio.cash == pytest.approx(500.0)
    assert portfolio.position("AAPL").quantity == 5


def test_equity_is_cash_plus_open_positions():
    portfolio = Portfolio(cash=1000.0)
    portfolio.execute(buy(2, "AAPL"), 100.0)
    portfolio.execute(buy(3, "MSFT"), 50.0)
    # cash 650 + 2 * 110 + 3 * 60
    assert portfolio.equity({"AAPL": 110.0, "MSFT": 60.0}) == pytest.approx(1050.0)


def test_equity_ignores_closed_positions_without_a_price():
    portfolio = Portfolio(cash=1000.0)
    portfolio.execute(buy(2, "AAPL"), 100.0)
    portfolio.execute(sell(2, "AAPL"), 100.0)
    # AAPL is flat, so no price is needed for it
    assert portfolio.equity({}) == pytest.approx(1000.0)
