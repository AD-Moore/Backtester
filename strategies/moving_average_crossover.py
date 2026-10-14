import pandas as pd

from strategies.strategy_base import Strategy


class MovingAverageCrossover(Strategy):
    """Hold a symbol while its short MA is above its long MA, otherwise
    hold cash instead (golden/death cross). Windows are configurable,
    default 20/50. Each symbol is judged on its own and gets an equal
    share of the portfolio."""

    def __init__(self, short_window: int = 20, long_window: int = 50):
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        self.short_window = short_window
        self.long_window = long_window

    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        short_ma = closes.rolling(window=self.short_window).mean()
        long_ma = closes.rolling(window=self.long_window).mean()

        # NaN > NaN is False in pandas, so early rows (before long_ma has
        # enough history) resolve to "not held" automatically.
        held = (short_ma > long_ma).astype(float)
        return held / len(closes.columns)
