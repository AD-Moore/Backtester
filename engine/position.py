from dataclasses import dataclass


@dataclass
class Position:
    """Shares held and average cost for one symbol. Whole-share, long-only
    for now (see README roadmap)."""

    symbol: str
    quantity: int = 0
    avg_price: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.quantity > 0

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def buy(self, quantity: int, price: float) -> None:
        total_cost = self.avg_price * self.quantity + price * quantity
        self.quantity += quantity
        self.avg_price = total_cost / self.quantity

    def sell(self, quantity: int, price: float) -> float:
        """Reduce the position and return the realized P&L from this sale."""
        if quantity > self.quantity:
            raise ValueError(
                f"cannot sell {quantity} shares of {self.symbol}, only "
                f"{self.quantity} held"
            )
        realized_pnl = (price - self.avg_price) * quantity
        self.quantity -= quantity
        if self.quantity == 0:
            self.avg_price = 0.0
        return realized_pnl
