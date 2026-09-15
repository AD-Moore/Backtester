from dataclasses import dataclass, field

from engine.order import Order
from engine.position import Position


@dataclass
class Portfolio:
    """Owns cash and positions; the only thing allowed to mutate them.
    BacktestEngine decides what to trade, this enforces whether it's
    affordable and keeps the bookkeeping consistent."""

    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    def position(self, symbol: str) -> Position:
        """Get (creating if needed) the Position for a symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def execute(self, order: Order, fill_price: float) -> None:
        pos = self.position(order.symbol)

        if order.side == "buy":
            cost = order.quantity * fill_price
            if cost > self.cash + 1e-9:  # tolerate float rounding dust
                raise ValueError(
                    f"insufficient cash to buy {order.quantity} {order.symbol} "
                    f"at {fill_price}: need {cost:.2f}, have {self.cash:.2f}"
                )
            pos.buy(order.quantity, fill_price)
            self.cash -= cost
        else:
            pos.sell(order.quantity, fill_price)  # raises if overselling
            self.cash += order.quantity * fill_price

    def equity(self, prices: dict[str, float]) -> float:
        """Cash plus market value of every open position."""
        value = self.cash
        for symbol, pos in self.positions.items():
            if pos.is_open:
                value += pos.market_value(prices[symbol])
        return value
