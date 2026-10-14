import pandas as pd

from strategies.strategy_base import Strategy


class CrossSectionalMomentum(Strategy):
    """Hold the recent winners: rank the symbols against each other by
    trailing return and split the portfolio equally across the top few.

    Momentum (momentum.py) asks "is this symbol going up?" about each
    symbol separately. This asks "which symbols are going up the most?",
    so it is always choosing between them and needs several to choose
    from. The ranking is redone every `rebalance_days` (default 21, about
    a month) and the holdings are left alone in between.
    """

    def __init__(self, lookback: int = 126, top_n: int = 2, rebalance_days: int = 21):
        if lookback < 1 or top_n < 1 or rebalance_days < 1:
            raise ValueError("lookback, top_n and rebalance_days must all be at least 1")
        self.lookback = lookback
        self.top_n = top_n
        self.rebalance_days = rebalance_days

    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        if len(closes.columns) <= self.top_n:
            raise ValueError(
                f"cross-sectional momentum picks the top {self.top_n} symbols, so it needs "
                f"more than {self.top_n} to choose from (got {len(closes.columns)})"
            )

        # rank 1 = highest trailing return that day
        trailing_return = closes.pct_change(self.lookback)
        rank = trailing_return.rank(axis=1, ascending=False)

        # Rows before `lookback` have no return yet: NaN <= top_n is False,
        # so nothing is held.
        weights = (rank <= self.top_n).astype(float) / self.top_n

        # Keep the ranking from every rebalance_days-th row and carry it
        # forward across the days in between.
        on_rebalance_days = weights.iloc[:: self.rebalance_days]
        return on_rebalance_days.reindex(weights.index).ffill()
