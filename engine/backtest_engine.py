import numpy as np
import pandas as pd

from engine.order import Order
from engine.portfolio import Portfolio
from engine.result import BacktestResult
from engine.trade import Trade
from strategies.strategy_base import Strategy


class BacktestEngine:
    """Walks price data day by day, trading so the portfolio matches the
    weights the strategy asks for. Works the same for one symbol or many.

    opens and closes are price tables with one row per day and one column
    per symbol (the same days and symbols in both).

    When it trades: only when a symbol's target weight changes. A weight
    that stays the same is left alone, even as prices move its real share
    of the portfolio up or down.

    Fill timing: a weight decided from day T's Close is traded at day
    T+1's Open -- trading at T's own Close would use information not yet
    available at the moment of the trade (lookahead bias).

    Costs: slippage moves every fill against you by a fraction of the
    price (buys fill higher, sells lower); commission_rate is charged on
    the traded value. Both default to 0.

    Current simplifications: whole shares only, no interest on cash, and
    no borrowing cost or margin requirement on short positions.
    """

    def __init__(
        self,
        opens: pd.DataFrame,
        closes: pd.DataFrame,
        strategy: Strategy,
        initial_cash: float = 10_000.0,
        commission_rate: float = 0.0,
        slippage: float = 0.0,
    ):
        if commission_rate < 0 or slippage < 0:
            raise ValueError("commission_rate and slippage cannot be negative")
        if not opens.index.equals(closes.index) or list(opens.columns) != list(closes.columns):
            raise ValueError("opens and closes must cover the same days and symbols")
        self.opens = opens
        self.closes = closes
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage = slippage

    def run(self) -> BacktestResult:
        symbols = list(self.closes.columns)
        dates = self.closes.index
        weights = self.strategy.generate_weights(self.closes)

        # Plain arrays for the loop: looking values up in a DataFrame on
        # every row is far slower, which adds up over a parameter sweep.
        opens = self.opens.to_numpy()
        closes = self.closes.to_numpy()
        targets = weights[symbols].fillna(0.0).to_numpy()

        portfolio = Portfolio(cash=self.initial_cash)
        equity = []
        trades = []
        open_trades: dict[str, dict] = {}  # symbol -> entry date, cost and cash flow so far

        # stop one day early: the last day has no "next day open" to trade at
        for day in range(len(dates) - 1):
            # mark equity before acting on today's weights -- today's close
            # still reflects the positions held coming into today
            equity.append(portfolio.equity(dict(zip(symbols, closes[day]))))

            previous_targets = targets[day - 1] if day > 0 else np.zeros(len(symbols))
            changed = [i for i in range(len(symbols)) if targets[day, i] != previous_targets[i]]
            if not changed:
                continue

            # Size the new positions from what the portfolio is worth at
            # tomorrow's open, which is when the trades happen.
            next_opens = opens[day + 1]
            equity_at_open = portfolio.equity(dict(zip(symbols, next_opens)))

            share_changes = []  # (symbol number, shares to buy (+) or sell (-))
            for i in changed:
                held = portfolio.position(symbols[i]).quantity
                wanted = int(targets[day, i] * equity_at_open / next_opens[i])
                if held * wanted < 0:
                    # switching between long and short: close, then reopen
                    share_changes += [(i, -held), (i, wanted)]
                elif wanted != held:
                    share_changes.append((i, wanted - held))

            # sells first, so the cash they raise can pay for the buys
            share_changes.sort(key=lambda change: change[1] > 0)

            for i, shares in share_changes:
                finished_trade = self._fill(
                    portfolio, open_trades, symbols[i], shares, next_opens[i], dates[day + 1]
                )
                if finished_trade is not None:
                    trades.append(finished_trade)

        # value the final day at its own close; nothing trades after this
        equity.append(portfolio.equity(dict(zip(symbols, closes[-1]))))

        return BacktestResult(
            equity_curve=pd.Series(equity, index=dates, name="equity", dtype=float),
            trades=trades,
        )

    def _fill(
        self,
        portfolio: Portfolio,
        open_trades: dict[str, dict],
        symbol: str,
        shares: int,
        open_price: float,
        date,
    ) -> Trade | None:
        """Buy (shares > 0) or sell (shares < 0) at the open, with costs.
        Returns a Trade if this fill closed the position, otherwise None."""
        position = portfolio.position(symbol)

        if shares > 0:
            side = "buy"
            fill_price = open_price * (1 + self.slippage)
            quantity = shares
            if position.quantity >= 0:
                # buying into a long: never more than the cash can pay for,
                # commission included
                affordable = int(portfolio.cash // (fill_price * (1 + self.commission_rate)))
                quantity = min(shares, affordable)
        else:
            side = "sell"
            fill_price = open_price * (1 - self.slippage)
            quantity = -shares

        if quantity <= 0:
            return None

        value = quantity * fill_price
        commission = value * self.commission_rate
        cash_before = portfolio.cash
        was_flat = not position.is_open
        portfolio.execute(Order(date, symbol, side, quantity), fill_price, commission)

        # A trade's profit is simply all the cash that came back out minus
        # all the cash that went in, so keep a running total while it's open.
        if was_flat:
            open_trades[symbol] = {"entry_date": date, "cost": value + commission, "cash_flow": 0.0}
        open_trades[symbol]["cash_flow"] += portfolio.cash - cash_before

        if position.is_open:
            return None
        finished = open_trades.pop(symbol)
        return Trade(symbol, finished["entry_date"], date, finished["cost"], finished["cash_flow"])


def print_equity(label: str, series: pd.Series) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    for date, value in series.items():
        print(f"{date.date()}  ${value:,.2f}")


if __name__ == "__main__":
    from data.loader import load_price_panel
    from strategies.buy_and_hold import BuyAndHold

    opens = load_price_panel("Open", ["AAPL"])
    closes = load_price_panel("Close", ["AAPL"])
    result = BacktestEngine(opens, closes, BuyAndHold()).run()

    print_equity("First 5 days", result.equity_curve.head())
    print_equity("Last 5 days", result.equity_curve.tail())
