import numpy as np
import pandas as pd

from strategies.strategy_base import Strategy


class PairsTrading(Strategy):
    """Bet that two related symbols which have drifted apart will come
    back together.

    The "spread" is the log of the price ratio A/B. Its z-score says how
    unusual today's spread is against the last `window` days. When A has
    become unusually expensive relative to B (z > entry_z), short A and
    buy B; when unusually cheap (z < -entry_z), buy A and short B. Close
    both sides once the spread is back near normal (|z| < exit_z).

    Half the portfolio goes on each side, so the two legs roughly cancel
    out the market's overall direction -- the profit comes from the gap
    closing, not from prices rising. Needs exactly two symbols.
    """

    def __init__(self, window: int = 60, entry_z: float = 2.0, exit_z: float = 0.5):
        if window < 2:
            raise ValueError("window must be at least 2")
        if not 0 <= exit_z < entry_z:
            raise ValueError("need 0 <= exit_z < entry_z")
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        if len(closes.columns) != 2:
            raise ValueError(f"pairs trading needs exactly 2 symbols, got {len(closes.columns)}")
        a, b = closes.columns

        spread = np.log(closes[a] / closes[b])
        rolling = spread.rolling(window=self.window)
        z_score = (spread - rolling.mean()) / rolling.std()

        # Which way we are positioned in A: +1 long, -1 short, 0 out.
        # Mark only the days that change it, then carry it forward.
        side = pd.Series(np.nan, index=closes.index)
        side[z_score > self.entry_z] = -1.0  # A is expensive: short it
        side[z_score < -self.entry_z] = 1.0  # A is cheap: buy it
        side[z_score.abs() < self.exit_z] = 0.0  # back to normal: close
        side = side.ffill().fillna(0.0)

        # B always takes the opposite side to A
        return pd.DataFrame({a: 0.5 * side, b: -0.5 * side})
