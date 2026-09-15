from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return a signal per row: 1 = long, 0 = flat, -1 = short."""
