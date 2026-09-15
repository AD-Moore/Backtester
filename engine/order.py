from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Order:
    """A buy/sell instruction. Immutable -- it's a record of a decision,
    not the fill itself (that's Portfolio's job)."""

    date: datetime
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int

    def __post_init__(self):
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {self.side!r}")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
