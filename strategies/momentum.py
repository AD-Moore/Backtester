import pandas as pd

from strategies.strategy_base import Strategy


class Momentum(Strategy):
    """Time-series momentum: long while the trailing return over
    `lookback` days is above `threshold`, flat otherwise. The bet is that
    an asset that has been rising keeps rising. Default lookback is 126
    trading days (~6 months)."""

    def __init__(self, lookback: int = 126, threshold: float = 0.0):
        if lookback < 1:
            raise ValueError("lookback must be at least 1")
        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        trailing_return = df["Close"].pct_change(self.lookback)

        # NaN > threshold is False, so the first `lookback` rows are flat.
        return (trailing_return > self.threshold).astype(int)
