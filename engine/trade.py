from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Trade:
    """One completed round trip (entry buy + exit sell). Immutable, like
    Order -- it's a record of something that already happened.

    Prices are the actual fill prices (slippage included); commissions
    are what was paid on each side."""

    symbol: str
    entry_date: datetime
    exit_date: datetime
    quantity: int
    entry_price: float
    exit_price: float
    entry_commission: float = 0.0
    exit_commission: float = 0.0

    @property
    def cost(self) -> float:
        """Total cash paid to open the trade."""
        return self.quantity * self.entry_price + self.entry_commission

    @property
    def pnl(self) -> float:
        """Net profit after costs on both sides."""
        proceeds = self.quantity * self.exit_price - self.exit_commission
        return proceeds - self.cost

    @property
    def return_pct(self) -> float:
        return self.pnl / self.cost

    @property
    def is_win(self) -> bool:
        return self.pnl > 0
