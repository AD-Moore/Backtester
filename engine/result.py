from dataclasses import dataclass

import pandas as pd

from engine.trade import Trade


@dataclass(frozen=True)
class BacktestResult:
    """What one BacktestEngine.run() produced."""

    equity_curve: pd.Series  # portfolio value at each day's close
    trades: list[Trade]  # finished positions; ones still open at the end are not included
