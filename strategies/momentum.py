import pandas as pd

from strategies.strategy_base import Strategy


class Momentum(Strategy):
    """Time-series momentum: hold a symbol while its own trailing return
    over `lookback` days is above `threshold`, otherwise hold cash. The
    bet is that an asset that has been rising keeps rising. Default
    lookback is 126 trading days (~6 months). Each symbol is judged on
    its own and gets an equal share of the portfolio."""

    def __init__(self, lookback: int = 126, threshold: float = 0.0):
        if lookback < 1:
            raise ValueError("lookback must be at least 1")
        self.lookback = lookback
        self.threshold = threshold

    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        trailing_return = closes.pct_change(self.lookback)

        # NaN > threshold is False, so the first `lookback` rows hold nothing.
        held = (trailing_return > self.threshold).astype(float)
        return held / len(closes.columns)
