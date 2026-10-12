"""Unit tests for Position: quantity, average cost, and realised P&L."""

import pytest

from engine.position import Position


def test_new_position_is_flat():
    pos = Position("AAPL")
    assert pos.quantity == 0
    assert pos.avg_price == 0.0
    assert not pos.is_open


def test_buy_opens_position():
    pos = Position("AAPL")
    pos.buy(10, 100.0)
    assert pos.is_open
    assert pos.quantity == 10
    assert pos.avg_price == 100.0


def test_buy_twice_averages_cost():
    pos = Position("AAPL")
    pos.buy(10, 100.0)
    pos.buy(30, 120.0)
    assert pos.quantity == 40
    assert pos.avg_price == pytest.approx(115.0)  # (1000 + 3600) / 40


def test_market_value():
    pos = Position("AAPL")
    pos.buy(10, 100.0)
    assert pos.market_value(150.0) == 1500.0


def test_partial_sell_returns_pnl_and_keeps_avg_price():
    pos = Position("AAPL")
    pos.buy(10, 100.0)
    pnl = pos.sell(4, 110.0)
    assert pnl == pytest.approx(40.0)  # 4 * (110 - 100)
    assert pos.quantity == 6
    # Selling realises P&L but does not change the cost basis of what remains.
    assert pos.avg_price == 100.0


def test_full_sell_resets_avg_price():
    pos = Position("AAPL")
    pos.buy(10, 100.0)
    pnl = pos.sell(10, 90.0)
    assert pnl == pytest.approx(-100.0)  # 10 * (90 - 100)
    assert not pos.is_open
    assert pos.avg_price == 0.0


def test_oversell_raises_and_leaves_position_untouched():
    pos = Position("AAPL")
    pos.buy(5, 100.0)
    with pytest.raises(ValueError, match="cannot sell"):
        pos.sell(6, 100.0)
    # A rejected sell must not partially apply.
    assert pos.quantity == 5
    assert pos.avg_price == 100.0
