"""Parameter sweep: backtest a strategy once per parameter combination.

On its own this is in-sample optimization -- the best row is the best
*on the data it was picked from*, which flatters it. walk_forward.py
uses the same search but scores the winner on data it never saw.
"""
import itertools

import pandas as pd

from engine.backtest_engine import BacktestEngine
from performance.metrics import compute_metrics
from strategies.strategy_base import Strategy


def parameter_combinations(param_grid: dict[str, list]) -> list[dict]:
    """Every combination of the grid's values, e.g. {"a": [1, 2], "b": [3]}
    -> [{"a": 1, "b": 3}, {"a": 2, "b": 3}]."""
    names = list(param_grid)
    return [dict(zip(names, values)) for values in itertools.product(*param_grid.values())]


def grid_search(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    strategy_class: type[Strategy],
    param_grid: dict[str, list],
    **engine_settings,
) -> pd.DataFrame:
    """Backtest every combination. Returns one row per combination with
    its parameters (as a dict, in the "params" column) and its metrics,
    sorted so the highest Sharpe ratio is the first row."""
    rows = []
    for params in parameter_combinations(param_grid):
        try:
            strategy = strategy_class(**params)
        except ValueError:
            # the strategy rejects this combination, e.g. a short window
            # longer than the long window
            continue
        result = BacktestEngine(opens, closes, strategy, **engine_settings).run()
        metrics = compute_metrics(result.equity_curve)
        rows.append({"params": params, **metrics, "num_trades": len(result.trades)})

    if not rows:
        raise ValueError(f"no valid parameter combinations in {param_grid}")

    # kind="stable": on a tie, the combination listed first in the grid wins
    table = pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False, kind="stable")
    return table.reset_index(drop=True)
