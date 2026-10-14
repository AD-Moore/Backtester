import pandas as pd

from strategies.strategy_base import Strategy


class BuyAndHold(Strategy):
    """Always fully invested, split equally across the symbols. The
    baseline every other strategy should beat on a risk-adjusted basis."""

    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        equal_share = 1 / len(closes.columns)
        return pd.DataFrame(equal_share, index=closes.index, columns=closes.columns)
