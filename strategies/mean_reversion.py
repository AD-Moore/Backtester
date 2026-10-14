import numpy as np
import pandas as pd

from strategies.strategy_base import Strategy


class MeanReversion(Strategy):
    """Buy when price is stretched far below its rolling mean, sell once
    it has reverted back to it.

    "Stretched" is measured as a z-score: how many rolling standard
    deviations Close sits from its rolling mean (the same idea as
    Bollinger Bands). Enter long when z < -entry_z, exit when z >= exit_z.
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

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rolling = df["Close"].rolling(window=self.window)
        z_score = (df["Close"] - rolling.mean()) / rolling.std()

        # Mark only the days that change the position, then carry each
        # decision forward: between an entry and an exit we stay long.
        signals = pd.Series(np.nan, index=df.index)
        signals[z_score < -self.entry_z] = 1
        signals[z_score >= self.exit_z] = 0
        return signals.ffill().fillna(0).astype(int)
