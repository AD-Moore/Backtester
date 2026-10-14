"""Unit tests for Trade: P&L and return after costs, and immutability."""

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from engine.trade import Trade


# Arbitrary fixed timestamps; no test here depends on their values.
ENTRY = datetime(2024, 1, 2)
EXIT = datetime(2024, 1, 9)


def test_pnl_and_return_without_costs():
    trade = Trade("AAPL", ENTRY, EXIT, 10, 100.0, 110.0)
    assert trade.cost == pytest.approx(1000.0)
    assert trade.pnl == pytest.approx(100.0)
    assert trade.return_pct == pytest.approx(0.10)
    assert trade.is_win


def test_commissions_on_both_sides_reduce_pnl():
    trade = Trade("AAPL", ENTRY, EXIT, 10, 100.0, 110.0, entry_commission=5.0, exit_commission=5.0)
    assert trade.cost == pytest.approx(1005.0)
    assert trade.pnl == pytest.approx(90.0)  # 1100 - 5 - 1005
    assert trade.return_pct == pytest.approx(90.0 / 1005.0)


def test_costs_can_turn_a_gross_gain_into_a_loss():
    # +10 gross, -20 in commissions
    trade = Trade("AAPL", ENTRY, EXIT, 10, 100.0, 101.0, entry_commission=10.0, exit_commission=10.0)
    assert trade.pnl == pytest.approx(-10.0)
    assert not trade.is_win


def test_breakeven_trade_is_not_a_win():
    assert not Trade("AAPL", ENTRY, EXIT, 10, 100.0, 100.0).is_win


def test_is_immutable():
    trade = Trade("AAPL", ENTRY, EXIT, 10, 100.0, 110.0)
    with pytest.raises(FrozenInstanceError):
        trade.exit_price = 200.0
