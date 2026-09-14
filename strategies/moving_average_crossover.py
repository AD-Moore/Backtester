import pandas as pd

from strategies.strategy_base import Strategy


class MovingAverageCrossover(Strategy):
    """Go long when the short-term moving average is above the long-term
    moving average, flat otherwise (the classic "golden cross / death
    cross" signal, just without the nicknames).

    Defaults to a 20-day / 50-day crossover, but both windows are
    configurable so you can experiment without editing this file.
    """

    def __init__(self, short_window: int = 20, long_window: int = 50):
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        short_ma = df["Close"].rolling(window=self.short_window).mean()
        long_ma = df["Close"].rolling(window=self.long_window).mean()

        # 1 where the short MA is above the long MA, 0 otherwise.
        # Rows before the long MA has enough data (first `long_window` - 1
        # rows) will be NaN > NaN comparisons, which pandas resolves to
        # False -> signal 0. That's intentional: no signal until there's
        # enough history to compute both averages.
        signals = (short_ma > long_ma).astype(int)
        return signals