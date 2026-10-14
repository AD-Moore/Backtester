import argparse

import pandas as pd

from data.loader import load_ticker
from engine.backtest_engine import BacktestEngine
from engine.trade import Trade
from performance.metrics import compute_metrics, print_performance_report, trade_statistics
from performance.visualization import plot_equity_curves
from strategies.buy_and_hold import BuyAndHold
from strategies.mean_reversion import MeanReversion
from strategies.momentum import Momentum
from strategies.moving_average_crossover import MovingAverageCrossover


STRATEGIES = {
    "buy_and_hold": BuyAndHold(),
    "ma_20_50": MovingAverageCrossover(short_window=20, long_window=50),
    "mean_reversion": MeanReversion(window=20, entry_z=2.0, exit_z=0.0),
    "momentum": Momentum(lookback=126),
}


def run(
    ticker: str,
    strategy_name: str,
    initial_cash: float = 10_000.0,
    commission_rate: float = 0.0,
    slippage: float = 0.0,
) -> tuple[pd.Series, list[Trade]]:
    """Run one backtest and return its equity curve and completed trades."""
    df = load_ticker(ticker)
    engine = BacktestEngine(
        df,
        STRATEGIES[strategy_name],
        initial_cash=initial_cash,
        symbol=ticker,
        commission_rate=commission_rate,
        slippage=slippage,
    )
    equity_curve = engine.run()
    return equity_curve, engine.trades


def print_comparison(results: dict[str, tuple[pd.Series, list[Trade]]]) -> None:
    """One row per strategy, so they can be compared side by side."""
    header = (
        f"{'Strategy':<16}{'Return':>10}{'CAGR':>9}{'Vol':>9}{'Sharpe':>8}"
        f"{'Sortino':>9}{'Max DD':>9}{'Trades':>8}{'Win %':>8}"
    )
    print(header)
    print("-" * len(header))

    for name, (equity_curve, trades) in results.items():
        m = compute_metrics(equity_curve)
        t = trade_statistics(trades)
        win_rate = f"{t['win_rate']:.0%}" if t["num_trades"] else "-"
        print(
            f"{name:<16}{m['total_return']:>+10.1%}{m['annualized_return']:>9.1%}"
            f"{m['annualized_volatility']:>9.1%}{m['sharpe_ratio']:>8.2f}"
            f"{m['sortino_ratio']:>9.2f}{m['max_drawdown']:>9.1%}"
            f"{t['num_trades']:>8}{win_rate:>8}"
        )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest strategies on one ticker and compare them.")
    parser.add_argument("--ticker", default="AAPL", help="ticker with processed data (default: AAPL)")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(STRATEGIES),
        default=list(STRATEGIES),
        help="strategies to run (default: all)",
    )
    parser.add_argument("--cash", type=float, default=10_000.0, help="starting cash (default: 10000)")
    parser.add_argument(
        "--commission",
        type=float,
        default=0.001,
        help="commission as a fraction of traded value (default: 0.001 = 0.10%%)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0005,
        help="adverse fill-price move as a fraction (default: 0.0005 = 0.05%%)",
    )
    parser.add_argument("--detail", action="store_true", help="also print a full report per strategy")
    parser.add_argument("--no-plot", action="store_true", help="skip saving the chart")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ticker = args.ticker.upper()

    results = {
        name: run(ticker, name, args.cash, args.commission, args.slippage)
        for name in args.strategies
    }

    start, end = next(iter(results.values()))[0].index[[0, -1]]
    print(f"\n{ticker}: {start.date()} to {end.date()}, starting cash ${args.cash:,.0f}")
    print(f"Costs: {args.commission:.2%} commission, {args.slippage:.2%} slippage per fill\n")
    print_comparison(results)

    if args.detail:
        for name, (equity_curve, trades) in results.items():
            print_performance_report(f"{ticker} / {name}", equity_curve, trades=trades)

    if not args.no_plot:
        curves = {name: equity_curve for name, (equity_curve, _) in results.items()}
        chart_path = plot_equity_curves(curves, f"{ticker}: strategy comparison", f"results/{ticker}_comparison.png")
        print(f"Chart saved to {chart_path}")
