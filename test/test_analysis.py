"""Tests for the parameter sweep and walk-forward test."""

import numpy as np
import pandas as pd
import pytest

from analysis.grid_search import grid_search, parameter_combinations
from analysis.walk_forward import FixedWeights, best_in_hindsight, walk_forward
from engine.backtest_engine import BacktestEngine
from performance.metrics import sharpe_ratio
from strategies.cross_sectional_momentum import CrossSectionalMomentum
from strategies.momentum import Momentum
from strategies.moving_average_crossover import MovingAverageCrossover

GRID = {"short_window": [3, 5, 10], "long_window": [10, 20]}
TRAIN_DAYS, TEST_DAYS = 100, 50


def random_walk(symbols, days=300):
    """Seeded random-walk prices as (opens, closes): enough days for
    several walk-forward windows, and the same on every run."""
    rng = np.random.default_rng(seed=0)
    index = pd.bdate_range("2020-01-01", periods=days)
    closes = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.0005, 0.02, size=(days, len(symbols))), axis=0),
        index=index,
        columns=symbols,
    )
    opens = closes.shift(1).fillna(100.0)  # open where yesterday closed
    return opens, closes


@pytest.fixture
def prices():
    return random_walk(["AAPL"])


def run_walk_forward(prices, days=None, **kwargs):
    """Walk-forward the MA crossover, optionally on only the first `days` rows."""
    opens, closes = prices
    return walk_forward(
        opens.iloc[:days], closes.iloc[:days], MovingAverageCrossover, GRID, TRAIN_DAYS, TEST_DAYS,
        **kwargs,
    )


def run_from(prices, strategy, start):
    """Backtest a strategy from row `start`, with earlier rows as history."""
    opens, closes = prices
    weights = strategy.generate_weights(closes).iloc[start:]
    return BacktestEngine(opens.iloc[start:], closes.iloc[start:], FixedWeights(weights)).run()


# --- grid search ---


def test_parameter_combinations_is_the_full_cross_product():
    assert parameter_combinations({"a": [1, 2], "b": [3]}) == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]


def test_grid_search_skips_combinations_the_strategy_rejects(prices):
    table = grid_search(*prices, MovingAverageCrossover, GRID)
    # 3 x 2 = 6 combinations, minus short=10/long=10 which is invalid
    assert len(table) == 5
    assert {"short_window": 10, "long_window": 10} not in table["params"].tolist()


def test_grid_search_puts_the_highest_sharpe_first(prices):
    table = grid_search(*prices, MovingAverageCrossover, GRID)
    assert table["sharpe_ratio"].is_monotonic_decreasing


def test_grid_search_rows_match_a_direct_backtest(prices):
    row = grid_search(*prices, MovingAverageCrossover, GRID, initial_cash=5_000.0).iloc[0]
    strategy = MovingAverageCrossover(**row["params"])
    direct = BacktestEngine(*prices, strategy, initial_cash=5_000.0).run()
    assert row["total_return"] == pytest.approx(direct.equity_curve.iloc[-1] / 5_000.0 - 1)
    assert row["num_trades"] == len(direct.trades)


def test_grid_with_no_valid_combination_raises(prices):
    with pytest.raises(ValueError, match="no valid parameter combinations"):
        grid_search(*prices, MovingAverageCrossover, {"short_window": [20], "long_window": [10]})


# --- walk-forward ---


def test_test_windows_tile_the_out_of_sample_period_with_no_gaps(prices):
    _, windows = run_walk_forward(prices)
    dates = prices[1].index

    assert len(windows) == 4  # (300 - 100) / 50
    assert windows["test_start"].iloc[0] == dates[TRAIN_DAYS]
    assert windows["test_end"].iloc[-1] == dates[-1]
    for this_end, next_start in zip(windows["test_end"], windows["test_start"].iloc[1:]):
        # each test stretch starts on the trading day after the previous one ends
        assert dates.get_loc(next_start) == dates.get_loc(this_end) + 1


def test_out_of_sample_curve_covers_only_the_period_after_the_first_training_window(prices):
    result, _ = run_walk_forward(prices)
    assert result.equity_curve.index.equals(prices[1].index[TRAIN_DAYS:])


def test_a_short_final_window_is_still_tested(prices):
    _, windows = run_walk_forward(prices, days=270)
    # 170 out-of-sample days = three full windows and one of 20 days
    assert len(windows) == 4
    assert windows["test_end"].iloc[-1] == prices[1].index[269]


def test_window_returns_compound_to_the_total_return(prices):
    result, windows = run_walk_forward(prices, initial_cash=10_000.0)
    compounded = (1 + windows["test_return"]).prod()
    assert compounded == pytest.approx(result.equity_curve.iloc[-1] / 10_000.0)


def test_choices_never_depend_on_later_data(prices):
    """Lookahead check: cutting off everything after the second test window
    must leave the first two windows' choices and equity exactly as they were."""
    full_result, full_windows = run_walk_forward(prices)
    cut_result, cut_windows = run_walk_forward(prices, days=TRAIN_DAYS + 2 * TEST_DAYS)

    assert cut_windows["params"].tolist() == full_windows["params"].iloc[:2].tolist()
    # the last day is excluded: a truncated run can't trade on that day's weights
    pd.testing.assert_series_equal(
        cut_result.equity_curve.iloc[:-1], full_result.equity_curve.iloc[: 2 * TEST_DAYS - 1]
    )


def test_single_combination_grid_matches_running_that_strategy_directly(prices):
    opens, closes = prices
    grid = {"short_window": [5], "long_window": [20]}
    result, _ = walk_forward(opens, closes, MovingAverageCrossover, grid, TRAIN_DAYS, TEST_DAYS)
    direct = run_from(prices, MovingAverageCrossover(5, 20), TRAIN_DAYS)
    pd.testing.assert_series_equal(result.equity_curve, direct.equity_curve)


def test_strategy_sees_the_history_before_its_test_window(prices):
    opens, closes = prices
    # a 60-day lookback has nothing to go on for 60 days if it only sees the test rows
    cold = BacktestEngine(opens.iloc[TRAIN_DAYS:], closes.iloc[TRAIN_DAYS:], Momentum(lookback=60)).run()
    warm, _ = walk_forward(opens, closes, Momentum, {"lookback": [60]}, TRAIN_DAYS, TEST_DAYS)
    assert cold.equity_curve.iloc[:60].nunique() == 1  # still sitting in cash
    assert warm.equity_curve.iloc[:60].nunique() > 1  # already trading


def test_walk_forward_works_on_a_multi_symbol_portfolio():
    opens, closes = random_walk(["A", "B", "C", "D"])
    grid = {"lookback": [10, 20], "top_n": [1, 2]}
    result, windows = walk_forward(opens, closes, CrossSectionalMomentum, grid, TRAIN_DAYS, TEST_DAYS)
    assert len(windows) == 4
    assert result.equity_curve.index.equals(closes.index[TRAIN_DAYS:])
    assert result.trades  # it actually rotated between symbols


def test_hindsight_is_at_least_as_good_as_any_fixed_combination(prices):
    opens, closes = prices
    params, best = best_in_hindsight(opens, closes, MovingAverageCrossover, GRID, TRAIN_DAYS)
    assert params in parameter_combinations(GRID)
    for combination in parameter_combinations(GRID):
        if combination["short_window"] >= combination["long_window"]:
            continue
        other = run_from(prices, MovingAverageCrossover(**combination), TRAIN_DAYS)
        assert sharpe_ratio(best.equity_curve) >= sharpe_ratio(other.equity_curve)


def test_too_little_data_for_the_training_window_raises(prices):
    with pytest.raises(ValueError, match="training"):
        run_walk_forward(prices, days=TRAIN_DAYS)
