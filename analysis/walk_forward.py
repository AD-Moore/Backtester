"""Walk-forward testing: the honest version of a parameter sweep.

Slide a window through history. On each step, pick the best parameters
using only the training window, then trade those parameters on the
*next* stretch of data, which the search never saw. Stitching those
test stretches together gives one out-of-sample equity curve -- what
would have happened had the parameters been re-tuned on a schedule
using only the past.

    |---- train ----|- test -|
             |---- train ----|- test -|
                      |---- train ----|- test -|
"""
import pandas as pd

from analysis.grid_search import grid_search, parameter_combinations
from engine.backtest_engine import BacktestEngine
from engine.result import BacktestResult
from performance.metrics import sharpe_ratio
from strategies.strategy_base import Strategy


class FixedWeights(Strategy):
    """Hands the engine weights that were already worked out, instead of
    computing them from prices."""

    def __init__(self, weights: pd.DataFrame):
        self.weights = weights

    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        return self.weights


def walk_forward(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    strategy_class: type[Strategy],
    param_grid: dict[str, list],
    train_days: int = 504,
    test_days: int = 126,
    **engine_settings,
) -> tuple[BacktestResult, pd.DataFrame]:
    """Re-pick parameters every `test_days` using the previous
    `train_days`, and backtest the result.

    Defaults are two trading years of training and six months of testing.
    Returns the out-of-sample backtest (which starts after the first
    training window) and a table with one row per test window.
    """
    if len(closes) < train_days + 2:
        raise ValueError(
            f"need more than {train_days + 1} days of prices for a {train_days}-day "
            f"training window, got {len(closes)}"
        )

    dates = closes.index
    test_weights = []  # the weights for each test window, in order
    windows = []

    for start in range(train_days, len(closes), test_days):
        end = min(start + test_days, len(closes))

        # 1. pick the best parameters on the training window alone
        train = slice(start - train_days, start)
        best = grid_search(
            opens.iloc[train], closes.iloc[train], strategy_class, param_grid, **engine_settings
        ).iloc[0]

        # 2. work out what those parameters would hold during the test
        #    window. The strategy sees all prices up to the window's end
        #    (so its indicators have history to work with) but none after.
        strategy = strategy_class(**best["params"])
        test_weights.append(strategy.generate_weights(closes.iloc[:end]).iloc[start:end])

        windows.append(
            {
                "test_start": dates[start],
                "test_end": dates[end - 1],
                "params": best["params"],
                "train_sharpe": best["sharpe_ratio"],
            }
        )

    # 3. join the test windows end to end and run them as ONE backtest, so
    #    positions and costs carry across window boundaries the way they
    #    would in practice
    result = BacktestEngine(
        opens.iloc[train_days:],
        closes.iloc[train_days:],
        FixedWeights(pd.concat(test_weights)),
        **engine_settings,
    ).run()

    # 4. how each test window actually turned out
    equity_at_window_end = result.equity_curve.loc[[w["test_end"] for w in windows]]
    equity_at_window_start = equity_at_window_end.shift(1, fill_value=result.equity_curve.iloc[0])
    table = pd.DataFrame(windows)
    table["test_return"] = (equity_at_window_end / equity_at_window_start - 1).to_numpy()
    return result, table


def best_in_hindsight(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    strategy_class: type[Strategy],
    param_grid: dict[str, list],
    start: int,
    **engine_settings,
) -> tuple[dict, BacktestResult]:
    """The single parameter set with the highest Sharpe ratio from row
    `start` onward, chosen by looking at that same period. This is the
    overfit number a plain sweep reports; the gap between it and the
    walk-forward result is what choosing with hindsight was worth."""
    best_params, best_result, best_sharpe = None, None, None

    for params in parameter_combinations(param_grid):
        try:
            strategy = strategy_class(**params)
        except ValueError:
            continue
        # weights from the full history, then traded only from `start`
        weights = strategy.generate_weights(closes).iloc[start:]
        result = BacktestEngine(
            opens.iloc[start:], closes.iloc[start:], FixedWeights(weights), **engine_settings
        ).run()
        sharpe = sharpe_ratio(result.equity_curve)
        if best_sharpe is None or sharpe > best_sharpe:
            best_params, best_result, best_sharpe = params, result, sharpe

    if best_params is None or best_result is None:
        raise ValueError(f"no valid parameter combinations in {param_grid}")
    return best_params, best_result
