import pandas as pd

from strategies.strategy_base import Strategy


class MovingAverageCrossover(Strategy):
    """Long when the short MA is above the long MA, flat otherwise
    (golden/death cross). Windows are configurable, default 20/50."""

    def __init__(self, short_window: int = 20, long_window: int = 50):
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        short_ma = df["Close"].rolling(window=self.short_window).mean()
        long_ma = df["Close"].rolling(window=self.long_window).mean()

        # NaN > NaN is False in pandas, so early rows (before long_ma has
        # enough history) resolve to signal 0 automatically.
        return (short_ma > long_ma).astype(int)
