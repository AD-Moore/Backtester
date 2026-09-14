import pandas as pd

from strategies.strategy_base import Strategy


class BacktestEngine:
    """Walk through price data day by day, turn a strategy's signals into
    trades, and track cash/shares/equity over time.

    Timing assumption: a signal generated using day T's Close can only be
    acted on at day T+1's Open. Filling at day T's own Close would mean
    trading on information that wouldn't actually be known at the moment
    of the trade — a common source of falsely good backtest results. 

    Simplifications for initial version (documented on purpose, not hidden): all-in-or-flat
    position sizing (100% of cash or 0%), whole shares only (no fractional
    shares), no transaction costs or slippage yet.
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        strategy: Strategy,
        initial_cash: float = 10_000.0,
    ):
        self.price_df = price_df
        self.strategy = strategy
        self.initial_cash = initial_cash

    def run(self) -> pd.Series:
        signals = self.strategy.generate_signals(self.price_df)

        cash = self.initial_cash
        shares = 0
        equity_curve = []

        # Stop one day early: the last row has no "next day open" to fill a
        # trade at, since there is no row after it.
        for i in range(len(self.price_df) - 1):
            today = self.price_df.index[i]
            tomorrow = self.price_df.index[i + 1]

            # Mark today's equity BEFORE acting on today's signal — any trade
            # decided today fills at tomorrow's open, so as of today's close
            # we still hold whatever position we entered today with.
            today_close = self.price_df.loc[today, "Close"]
            equity = cash + shares * today_close
            equity_curve.append((today, equity))

            current_signal = signals.loc[today]
            currently_long = shares > 0

            # Signal says "be long" but we're currently flat -> buy tomorrow's open.
            if current_signal == 1 and not currently_long:
                fill_price = self.price_df.loc[tomorrow, "Open"]
                shares = int(cash // fill_price)  # whole shares only
                cash -= shares * fill_price

            # Signal says "be flat" but we're currently long -> sell tomorrow's open.
            elif current_signal == 0 and currently_long:
                fill_price = self.price_df.loc[tomorrow, "Open"]
                cash += shares * fill_price
                shares = 0

        # Value the final day too, using its own close and whatever position
        # we're holding at that point (no more trades happen after this).
        last_date = self.price_df.index[-1]
        last_close = self.price_df.loc[last_date, "Close"]
        equity_curve.append((last_date, cash + shares * last_close))

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
    