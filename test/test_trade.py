"""Unit tests for Trade: return, win/loss, and immutability."""

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from engine.trade import Trade

# Arbitrary fixed timestamps; no test here depends on their values.
ENTRY = datetime(2024, 1, 2)
EXIT = datetime(2024, 1, 9)


def test_return_is_profit_over_money_committed():
    trade = Trade("AAPL", ENTRY, EXIT, cost=1000.0, pnl=100.0)
    assert trade.return_pct == pytest.approx(0.10)
    assert trade.is_win


def test_losing_trade():
    trade = Trade("AAPL", ENTRY, EXIT, cost=1000.0, pnl=-50.0)
    assert trade.return_pct == pytest.approx(-0.05)
    assert not trade.is_win


def test_breakeven_trade_is_not_a_win():
    assert not Trade("AAPL", ENTRY, EXIT, cost=1000.0, pnl=0.0).is_win


def test_is_immutable():
    trade = Trade("AAPL", ENTRY, EXIT, cost=1000.0, pnl=100.0)
    with pytest.raises(FrozenInstanceError):
        trade.pnl = 200.0
