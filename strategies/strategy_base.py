from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):

    @abstractmethod
    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Decide how much of the portfolio to hold in each symbol each day.

        closes has one row per day and one column per symbol. Return a
        DataFrame of the same shape where each value is a fraction of the
        portfolio: 1.0 = everything in that symbol, 0.5 = half, 0 = none,
        negative = short. Whatever a row doesn't allocate stays in cash.

        The weight on day T may only use prices up to and including day T.
        """
