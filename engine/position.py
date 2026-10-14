from dataclasses import dataclass


@dataclass
class Position:
    """Shares held and average entry price for one symbol. Whole shares
    only. A positive quantity is a long position; a negative quantity is
    a short (shares sold without owning them, to be bought back later).

    To switch between long and short, close the position first: a single
    buy or sell is not allowed to carry the quantity through zero.
    """

    symbol: str
    quantity: int = 0
    avg_price: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.quantity != 0

    def market_value(self, price: float) -> float:
        """What the position is worth now. Negative for a short: it is a
        debt of shares that would cost this much to buy back."""
        return self.quantity * price

    def buy(self, quantity: int, price: float) -> float:
        """Add to a long, or buy back part of a short. Returns the
        realized P&L, which is only non-zero when buying back a short."""
        if self.quantity >= 0:
            total_cost = self.avg_price * self.quantity + price * quantity
            self.quantity += quantity
            self.avg_price = total_cost / self.quantity
            return 0.0

        shares_short = -self.quantity
        if quantity > shares_short:
            raise ValueError(
                f"cannot buy {quantity} shares of {self.symbol} to cover a short of "
                f"only {shares_short}"
            )
        # a short profits when the price has fallen since it was opened
        realized_pnl = (self.avg_price - price) * quantity
        self.quantity += quantity
        if self.quantity == 0:
            self.avg_price = 0.0
        return realized_pnl

    def sell(self, quantity: int, price: float) -> float:
        """Reduce a long, or open/add to a short. Returns the realized
        P&L, which is only non-zero when selling shares that were held."""
        if self.quantity <= 0:
            shares_short = -self.quantity
            total_proceeds = self.avg_price * shares_short + price * quantity
            self.quantity -= quantity
            self.avg_price = total_proceeds / -self.quantity
            return 0.0

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
