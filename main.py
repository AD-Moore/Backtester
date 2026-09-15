from data.loader import load_ticker
from engine.backtest_engine import BacktestEngine, print_equity
from performance.metrics import print_performance_report
from strategies.buy_and_hold import BuyAndHold
from strategies.moving_average_crossover import MovingAverageCrossover


STRATEGIES = {
    "buy_and_hold": BuyAndHold(),
    "ma_20_50": MovingAverageCrossover(short_window=20, long_window=50),
}


def run(ticker: str, strategy_name: str):
    """Run one backtest, print the results, and return the equity curve
    (so callers can reuse it instead of re-running the backtest)."""
    df = load_ticker(ticker)
    strategy = STRATEGIES[strategy_name]

    engine = BacktestEngine(df, strategy)
    equity_curve = engine.run()

    print(f"\n=== {ticker} / {strategy_name} ===")
    print_equity("First 5 days", equity_curve.head())
    print_equity("Last 5 days", equity_curve.tail())
    print()
    print_performance_report(f"{ticker} / {strategy_name}", equity_curve)

    return equity_curve


if __name__ == "__main__":
    run("VGT", "buy_and_hold")
