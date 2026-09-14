from data.loader import load_ticker
from engine.backtest_engine import BacktestEngine, print_equity
from strategies.buy_and_hold import BuyAndHold
from strategies.moving_average_crossover import MovingAverageCrossover


STRATEGIES = {
    "buy_and_hold": BuyAndHold(),
    "ma_20_50": MovingAverageCrossover(short_window=20, long_window=50),
}


def run(ticker: str, strategy_name: str) -> None:
    df = load_ticker(ticker)
    strategy = STRATEGIES[strategy_name]

    engine = BacktestEngine(df, strategy)
    equity_curve = engine.run()

    print(f"\n=== {ticker} / {strategy_name} ===")
    print_equity("First 5 days", equity_curve.head())
    print_equity("Last 5 days", equity_curve.tail())


if __name__ == "__main__":
    run("AAPL", "ma_20_50")
    run("AAPL", "buy_and_hold")
    

    

