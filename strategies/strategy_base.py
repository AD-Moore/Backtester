from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    """
        #Generate trading signals from market data.
        #Returns pd.DataFrame containing a 'signal' column, where:
        #Signal convention:
        #    1  = long / buy
        #    0  = no position
        #    -1 = short / sell
    """