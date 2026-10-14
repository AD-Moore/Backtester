import numpy as np
import pandas as pd

from strategies.strategy_base import Strategy


class MeanReversion(Strategy):
    """Buy a symbol when its price is stretched far below its rolling
    mean, sell once it has reverted back to it.

    "Stretched" is measured as a z-score: how many rolling standard
    deviations Close sits from its rolling mean (the same idea as
    Bollinger Bands). Buy when z < -entry_z, sell when z >= exit_z. Each
    symbol is judged on its own and gets an equal share of the portfolio.
    """

    def __init__(self, window: int = 20, entry_z: float = 2.0, exit_z: float = 0.0):
        if window < 2:
            raise ValueError("window must be at least 2")
        if entry_z <= 0:
            raise ValueError("entry_z must be positive")
        if exit_z <= -entry_z:
            raise ValueError("exit_z must be above -entry_z")
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        rolling = closes.rolling(window=self.window)
        z_score = (closes - rolling.mean()) / rolling.std()

        # Mark only the days that change the position (1 = buy, 0 = sell),
        # then carry each decision forward: between a buy and a sell we
        # keep holding.
        held = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
        held[z_score < -self.entry_z] = 1.0
        held[z_score >= self.exit_z] = 0.0
        held = held.ffill().fillna(0.0)
        return held / len(closes.columns)
