"""Unit tests for the strategies: each one's signals on small, hand-checkable prices."""

import pandas as pd
import pytest

from strategies.buy_and_hold import BuyAndHold
from strategies.mean_reversion import MeanReversion
from strategies.momentum import Momentum
from strategies.moving_average_crossover import MovingAverageCrossover


def closes(values):
    """Build a price frame with just the Close column strategies read."""
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"Close": values}, index=index, dtype=float)


def test_buy_and_hold_is_always_long():
    signals = BuyAndHold().generate_signals(closes([10, 9, 11]))
    assert signals.tolist() == [1, 1, 1]


def test_ma_crossover_is_flat_until_long_window_has_history():
    df = closes([1, 2, 3, 4, 5, 6])
    signals = MovingAverageCrossover(short_window=2, long_window=4).generate_signals(df)
    # first 3 rows have no 4-day average yet; after that the uptrend keeps short > long
    assert signals.tolist() == [0, 0, 0, 1, 1, 1]


def test_ma_crossover_goes_flat_when_trend_reverses():
    df = closes([1, 2, 3, 4, 3, 2, 1])
    signals = MovingAverageCrossover(short_window=2, long_window=4).generate_signals(df)
    # short MA drops below long MA once the decline is two days old
    assert signals.tolist() == [0, 0, 0, 1, 1, 0, 0]


def test_ma_crossover_rejects_short_window_not_below_long():
    with pytest.raises(ValueError, match="short_window"):
        MovingAverageCrossover(short_window=50, long_window=50)


def test_momentum_is_long_only_when_trailing_return_is_positive():
    df = closes([10, 11, 12, 11, 10, 12])
    signals = Momentum(lookback=2).generate_signals(df)
    # 2-day returns: n/a, n/a, +20%, 0%, -17%, +9%
    assert signals.tolist() == [0, 0, 1, 0, 0, 1]


def test_momentum_threshold_raises_the_bar():
    df = closes([10, 11, 12, 11, 10, 12])
    signals = Momentum(lookback=2, threshold=0.10).generate_signals(df)
    # only the +20% return clears a 10% threshold
    assert signals.tolist() == [0, 0, 1, 0, 0, 0]


def test_momentum_rejects_non_positive_lookback():
    with pytest.raises(ValueError, match="lookback"):
        Momentum(lookback=0)


def test_mean_reversion_buys_the_dip_and_holds_until_the_mean():
    # flat at 10, a sharp drop to 7, then a recovery back above the 3-day mean
    df = closes([10, 10, 10, 7, 7.5, 8, 10, 10])
    signals = MeanReversion(window=3, entry_z=1.0, exit_z=0.0).generate_signals(df)
    # z-score: n/a, n/a, n/a (zero std), -1.15, -0.51, +1.0, +1.14, +0.58
    # enters on the drop, stays long through -0.51 (not yet reverted), exits at +1.0
    assert signals.tolist() == [0, 0, 0, 1, 1, 0, 0, 0]


def test_mean_reversion_is_flat_when_price_never_stretches():
    df = closes([10, 10.1, 10, 10.1, 10, 10.1])
    signals = MeanReversion(window=3, entry_z=2.0).generate_signals(df)
    assert signals.tolist() == [0] * 6


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"window": 1}, "window"),
        ({"entry_z": 0}, "entry_z"),
        ({"entry_z": 2.0, "exit_z": -2.0}, "exit_z"),
    ],
)
def test_mean_reversion_rejects_bad_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MeanReversion(**kwargs)


@pytest.mark.parametrize(
    "strategy",
    [
        BuyAndHold(),
        MovingAverageCrossover(short_window=2, long_window=4),
        MeanReversion(window=3, entry_z=1.0),
        Momentum(lookback=2),
    ],
)
def test_signals_do_not_depend_on_future_prices(strategy):
    """Lookahead check: a signal for day T must be the same whether or not
    the strategy can see the days after T."""
    df = closes([10, 11, 9, 12, 8, 13, 7, 14, 10, 11])
    full = strategy.generate_signals(df)
    for end in range(1, len(df) + 1):
        truncated = strategy.generate_signals(df.iloc[:end])
        assert truncated.iloc[-1] == full.iloc[end - 1]
