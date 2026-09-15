import pandas as pd

from strategies.strategy_base import Strategy


class BuyAndHold(Strategy):
    """Always long. The baseline every other strategy should beat on a
    risk-adjusted basis."""

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=df.index, dtype=int)
