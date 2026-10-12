"""Unit tests for Order: construction, validation, and immutability."""

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from engine.order import Order


# Arbitrary fixed timestamp; no test here depends on its value.
DATE = datetime(2024, 1, 2)


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_valid_order(side):
    order = Order(DATE, "AAPL", side, 10)
    assert order.side == side
    assert order.quantity == 10


def test_rejects_unknown_side():
    # Only "buy" and "sell" are supported; shorting is not modelled.
    with pytest.raises(ValueError, match="side"):
        Order(DATE, "AAPL", "short", 10)


@pytest.mark.parametrize("quantity", [0, -5])
def test_rejects_non_positive_quantity(quantity):
    with pytest.raises(ValueError, match="quantity"):
        Order(DATE, "AAPL", "buy", quantity)


def test_is_immutable():
    order = Order(DATE, "AAPL", "buy", 10)
    with pytest.raises(FrozenInstanceError):
        order.quantity = 20
