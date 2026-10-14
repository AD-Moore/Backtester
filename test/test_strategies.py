"""Unit tests for the strategies: each one's weights on small, hand-checkable prices."""

import numpy as np
import pandas as pd
import pytest

from strategies.buy_and_hold import BuyAndHold
from strategies.cross_sectional_momentum import CrossSectionalMomentum
from strategies.mean_reversion import MeanReversion
from strategies.momentum import Momentum
from strategies.moving_average_crossover import MovingAverageCrossover
from strategies.pairs_trading import PairsTrading


def closes(columns):
    """Build a table of closing prices from {symbol: [price per day]}."""
    length = len(next(iter(columns.values())))
    index = pd.date_range("2024-01-01", periods=length, freq="D")
    return pd.DataFrame(columns, index=index, dtype=float)


def weights(strategy, prices):
    """The weights a strategy gives a single symbol, as a plain list."""
    return strategy.generate_weights(closes({"AAPL": prices}))["AAPL"].tolist()


# --- single-symbol rules ---


def test_buy_and_hold_is_always_fully_invested():
    assert weights(BuyAndHold(), [10, 9, 11]) == [1, 1, 1]


def test_buy_and_hold_splits_equally_across_symbols():
    table = BuyAndHold().generate_weights(closes({"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [7, 8]}))
    assert (table == 0.25).all().all()


def test_ma_crossover_holds_nothing_until_long_window_has_history():
    strategy = MovingAverageCrossover(short_window=2, long_window=4)
    # first 3 rows have no 4-day average yet; after that the uptrend keeps short > long
    assert weights(strategy, [1, 2, 3, 4, 5, 6]) == [0, 0, 0, 1, 1, 1]


def test_ma_crossover_sells_when_trend_reverses():
    strategy = MovingAverageCrossover(short_window=2, long_window=4)
    # short MA drops below long MA once the decline is two days old
    assert weights(strategy, [1, 2, 3, 4, 3, 2, 1]) == [0, 0, 0, 1, 1, 0, 0]


def test_ma_crossover_rejects_short_window_not_below_long():
    with pytest.raises(ValueError, match="short_window"):
        MovingAverageCrossover(short_window=50, long_window=50)


def test_momentum_holds_only_while_trailing_return_is_positive():
    # 2-day returns: n/a, n/a, +20%, 0%, -17%, +9%
    assert weights(Momentum(lookback=2), [10, 11, 12, 11, 10, 12]) == [0, 0, 1, 0, 0, 1]


def test_momentum_threshold_raises_the_bar():
    # only the +20% return clears a 10% threshold
    strategy = Momentum(lookback=2, threshold=0.10)
    assert weights(strategy, [10, 11, 12, 11, 10, 12]) == [0, 0, 1, 0, 0, 0]


def test_momentum_rejects_non_positive_lookback():
    with pytest.raises(ValueError, match="lookback"):
        Momentum(lookback=0)


def test_mean_reversion_buys_the_dip_and_holds_until_the_mean():
    # flat at 10, a sharp drop to 7, then a recovery back above the 3-day mean
    strategy = MeanReversion(window=3, entry_z=1.0, exit_z=0.0)
    # z-score: n/a, n/a, n/a (zero std), -1.15, -0.51, +1.0, +1.14, +0.58
    # buys on the drop, holds through -0.51 (not yet reverted), sells at +1.0
    assert weights(strategy, [10, 10, 10, 7, 7.5, 8, 10, 10]) == [0, 0, 0, 1, 1, 0, 0, 0]


def test_mean_reversion_holds_nothing_when_price_never_stretches():
    strategy = MeanReversion(window=3, entry_z=2.0)
    assert weights(strategy, [10, 10.1, 10, 10.1, 10, 10.1]) == [0] * 6


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


def test_each_symbol_is_judged_on_its_own_and_gets_an_equal_share():
    # A is rising the whole time, B is falling the whole time
    table = Momentum(lookback=1).generate_weights(closes({"A": [10, 11, 12], "B": [10, 9, 8]}))
    assert table["A"].tolist() == [0, 0.5, 0.5]  # held, at half the portfolio
    assert table["B"].tolist() == [0, 0, 0]  # never held; its half stays in cash


# --- cross-sectional momentum ---

# each day a different symbol has the best 1-day return: A, then B, then C
RACE = {"A": [10, 11, 11, 11], "B": [10, 10, 12, 12], "C": [10, 10, 10, 13]}


def test_cross_sectional_momentum_holds_the_best_performer():
    strategy = CrossSectionalMomentum(lookback=1, top_n=1, rebalance_days=1)
    table = strategy.generate_weights(closes(RACE))
    # nothing on day 0 (no return yet), then whoever just did best
    assert table.to_numpy().tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_cross_sectional_momentum_only_re_ranks_on_rebalance_days():
    strategy = CrossSectionalMomentum(lookback=1, top_n=1, rebalance_days=2)
    table = strategy.generate_weights(closes(RACE))
    # ranked on days 0 and 2 only; days 1 and 3 carry the previous choice
    assert table.to_numpy().tolist() == [[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0]]


def test_cross_sectional_momentum_splits_equally_across_the_top_n():
    strategy = CrossSectionalMomentum(lookback=1, top_n=2, rebalance_days=1)
    # day 1 returns: A +20%, B +10%, C 0%
    table = strategy.generate_weights(closes({"A": [10, 12], "B": [10, 11], "C": [10, 10]}))
    assert table.iloc[1].tolist() == [0.5, 0.5, 0]


def test_cross_sectional_momentum_needs_more_symbols_than_it_picks():
    with pytest.raises(ValueError, match="more than 2"):
        CrossSectionalMomentum(top_n=2).generate_weights(closes({"A": [1, 2], "B": [3, 4]}))


# --- pairs trading ---

# B never moves; A jumps 30% on day 3 and falls back on day 6.
# z-score of the spread (window 3): n/a, n/a, n/a, +1.15, +0.58, n/a (zero std), -1.15
PAIR = {"A": [10, 10, 10, 13, 13, 13, 10], "B": [10, 10, 10, 10, 10, 10, 10]}


def test_pairs_trading_shorts_the_expensive_side_and_buys_the_cheap_one():
    strategy = PairsTrading(window=3, entry_z=1.0, exit_z=0.5)
    table = strategy.generate_weights(closes(PAIR))
    # day 3: A is expensive -> short A, long B.  z of 0.58 is not back to
    # normal yet, so hold.  day 6: A is now cheap -> switch sides.
    assert table["A"].tolist() == [0, 0, 0, -0.5, -0.5, -0.5, 0.5]
    assert table["B"].tolist() == [0, 0, 0, 0.5, 0.5, 0.5, -0.5]


def test_pairs_trading_closes_once_the_spread_is_back_to_normal():
    strategy = PairsTrading(window=3, entry_z=1.0, exit_z=0.9)
    table = strategy.generate_weights(closes(PAIR))
    # with a wider "normal" band, the z of 0.58 on day 4 closes the trade
    assert table["A"].tolist() == [0, 0, 0, -0.5, 0, 0, 0.5]


def test_pairs_trading_legs_always_cancel_out():
    table = PairsTrading(window=3, entry_z=1.0).generate_weights(closes(PAIR))
    assert (table.sum(axis=1) == 0).all()


def test_pairs_trading_needs_exactly_two_symbols():
    with pytest.raises(ValueError, match="exactly 2"):
        PairsTrading().generate_weights(closes({"A": [1, 2]}))


def test_pairs_trading_rejects_exit_band_wider_than_entry():
    with pytest.raises(ValueError, match="exit_z"):
        PairsTrading(entry_z=1.0, exit_z=1.5)


# --- every strategy ---


@pytest.mark.parametrize(
    "strategy, symbols",
    [
        (BuyAndHold(), 3),
        (MovingAverageCrossover(short_window=2, long_window=4), 3),
        (MeanReversion(window=3, entry_z=1.0), 3),
        (Momentum(lookback=2), 3),
        (CrossSectionalMomentum(lookback=2, top_n=1, rebalance_days=3), 3),
        (PairsTrading(window=4, entry_z=1.0), 2),
    ],
)
def test_weights_do_not_depend_on_future_prices(strategy, symbols):
    """Lookahead check: the weights for day T must be the same whether or
    not the strategy can see the days after T."""
    rng = np.random.default_rng(seed=1)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.03, size=(30, symbols)), axis=0)
    table = closes({f"S{i}": prices[:, i] for i in range(symbols)})

    full = strategy.generate_weights(table)
    for end in range(1, len(table) + 1):
        truncated = strategy.generate_weights(table.iloc[:end])
        assert truncated.iloc[-1].tolist() == full.iloc[end - 1].tolist()


@pytest.mark.parametrize(
    "strategy, symbols",
    [
        (BuyAndHold(), 3),
        (MovingAverageCrossover(short_window=2, long_window=4), 3),
        (MeanReversion(window=3, entry_z=1.0), 3),
        (Momentum(lookback=2), 3),
        (CrossSectionalMomentum(lookback=2, top_n=2), 3),
    ],
)
def test_long_only_strategies_never_ask_for_more_than_the_whole_portfolio(strategy, symbols):
    rng = np.random.default_rng(seed=2)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.03, size=(40, symbols)), axis=0)
    table = strategy.generate_weights(closes({f"S{i}": prices[:, i] for i in range(symbols)}))
    assert (table >= 0).all().all()
    assert (table.sum(axis=1) <= 1 + 1e-9).all()
