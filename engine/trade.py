from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Trade:
    """One finished position in one symbol, from the day it was opened to
    the day it was fully closed. Immutable, like Order -- it's a record
    of something that already happened."""

    symbol: str
    entry_date: datetime
    exit_date: datetime
    cost: float  # money committed when the position was opened, commission included
    pnl: float  # net profit: all cash received minus all cash paid, costs included

    @property
    def return_pct(self) -> float:
        return self.pnl / self.cost

    @property
    def is_win(self) -> bool:
        return self.pnl > 0
