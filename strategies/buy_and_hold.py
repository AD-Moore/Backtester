import pandas as pd

from strategies.strategy_base import Strategy


class BuyAndHold(Strategy):
    """Go long on day one and hold for the rest of the backtest.

    Useful less as a "real" strategy and more as a baseline — any strategy
    that can't beat buy-and-hold on a risk-adjusted basis isn't adding value.
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=df.index, dtype=int)