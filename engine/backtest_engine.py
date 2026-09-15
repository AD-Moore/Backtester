import pandas as pd

from engine.order import Order
from engine.portfolio import Portfolio
from strategies.strategy_base import Strategy


class BacktestEngine:
    """Walks price data day by day: signals become orders, Portfolio
    tracks cash/positions/equity.

    Fill timing: a signal from day T's Close fills at day T+1's Open --
    filling at T's own Close would use information not yet available at
    the moment of the trade (lookahead bias).

    Current simplifications: all-in-or-flat sizing, whole shares only,
    no transaction costs/slippage, single symbol per run.
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        strategy: Strategy,
        initial_cash: float = 10_000.0,
        symbol: str = "asset",
    ):
        self.price_df = price_df
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.symbol = symbol

    def run(self) -> pd.Series:
        signals = self.strategy.generate_signals(self.price_df)
        portfolio = Portfolio(cash=self.initial_cash)
        equity_curve = []

        # stop one day early: no "next day open" to fill the last row's signal at
        for i in range(len(self.price_df) - 1):
            today = self.price_df.index[i]
            tomorrow = self.price_df.index[i + 1]

            # mark equity before acting on today's signal -- today's close
            # still reflects yesterday's position
            today_close = self.price_df.loc[today, "Close"]
            equity_curve.append((today, portfolio.equity({self.symbol: today_close})))

            current_signal = signals.loc[today]
            position = portfolio.position(self.symbol)

            if current_signal == 1 and not position.is_open:
                # flat -> long: buy at tomorrow's open
                fill_price = self.price_df.loc[tomorrow, "Open"]
                quantity = int(portfolio.cash // fill_price)  # whole shares only
                if quantity > 0:
                    order = Order(tomorrow, self.symbol, "buy", quantity)
                    portfolio.execute(order, fill_price)

            elif current_signal == 0 and position.is_open:
                # long -> flat: sell at tomorrow's open
                fill_price = self.price_df.loc[tomorrow, "Open"]
                order = Order(tomorrow, self.symbol, "sell", position.quantity)
                portfolio.execute(order, fill_price)

        # value the final day at its own close; no trades happen after this
        last_date = self.price_df.index[-1]
        last_close = self.price_df.loc[last_date, "Close"]
        equity_curve.append((last_date, portfolio.equity({self.symbol: last_close})))

        dates, values = zip(*equity_curve)
        return pd.Series(values, index=dates, name="equity")


def print_equity(label: str, series: pd.Series) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    for date, value in series.items():
        print(f"{date.date()}  ${value:,.2f}")


if __name__ == "__main__":
    from data.loader import load_ticker
    from strategies.buy_and_hold import BuyAndHold

    df = load_ticker("AAPL")
    engine = BacktestEngine(df, BuyAndHold())

    equity_curve = engine.run()

    print_equity("First 5 days", equity_curve.head())
    print_equity("Last 5 days", equity_curve.tail())
