"""Command-line entry point.

    python main.py list                      the tickers and baskets available
    python main.py download --universe all   fetch data for every listed ticker
    python main.py                           compare every strategy on the default basket
    python main.py --universe multi_asset    ...on stocks, bonds, gold and commodities
    python main.py --tickers AAPL XOM        ...on any tickers you choose
    python main.py sweep                     try every parameter combination
    python main.py walkforward               test parameter picking out of sample
"""
import argparse

from rich.console import Console

from analysis.grid_search import grid_search
from analysis.walk_forward import best_in_hindsight, walk_forward
from data.config import PROCESSED_DATA_PATH
from data.downloader import DEFAULT_START, download_and_process, has_processed_data
from data.loader import load_price_panel
from data.universe import BASKETS, DEFAULT_BASKET, UNIVERSE, basket_tickers
from engine.backtest_engine import BacktestEngine
from engine.result import BacktestResult
from performance import report
from strategies.buy_and_hold import BuyAndHold
from strategies.cross_sectional_momentum import CrossSectionalMomentum
from strategies.mean_reversion import MeanReversion
from strategies.momentum import Momentum
from strategies.moving_average_crossover import MovingAverageCrossover
from strategies.pairs_trading import PairsTrading

console = Console()

# The strategies `compare` runs. To try different parameters, edit them here.
STRATEGIES = {
    "buy_and_hold": BuyAndHold(),
    "ma_crossover": MovingAverageCrossover(short_window=20, long_window=50),
    "mean_reversion": MeanReversion(window=20, entry_z=2.0, exit_z=0.0),
    "momentum": Momentum(lookback=126),
    "cross_sectional_momentum": CrossSectionalMomentum(lookback=126, top_n=2),  # 3+ tickers
    "pairs_trading": PairsTrading(window=60, entry_z=2.0, exit_z=0.5),  # exactly 2 tickers
}

# The parameter values `sweep` and `walkforward` choose between.
PARAM_GRIDS = {
    "ma_crossover": {"short_window": [10, 20, 50], "long_window": [50, 100, 200]},
    "mean_reversion": {"window": [10, 20, 50], "entry_z": [1.0, 1.5, 2.0]},
    "momentum": {"lookback": [21, 63, 126, 252]},
    "cross_sectional_momentum": {"lookback": [63, 126, 252], "top_n": [1, 2, 3]},
    "pairs_trading": {"window": [20, 60, 120], "entry_z": [1.5, 2.0, 2.5]},
}


def print_header(args: argparse.Namespace, closes, title: str) -> None:
    console.print()
    console.print(report.header(title, closes, args.cash, args.commission, args.slippage))
    console.print()


def print_assets(closes) -> None:
    """What the tickers are and how they relate, before any strategy runs."""
    console.print(report.asset_table(closes))
    if len(closes.columns) > 1:
        console.print(report.sector_table(list(closes.columns)))
        console.print()
        console.print(report.correlation_table(closes))
        console.print()


def print_results(results: dict[str, BacktestResult], title: str = "Strategy comparison") -> None:
    console.print(report.comparison_table(results, title))
    console.print()
    console.print(report.trading_table(results))
    console.print()
    console.print(report.annual_returns_table(results))
    console.print()
    console.print(report.takeaways(results))


def compare(args, opens, closes, settings) -> None:
    """Backtest each strategy over the full history and compare them."""
    results = {}
    skipped = []
    with console.status("Running backtests...") as status:
        for name in args.strategies:
            status.update(f"Running {name} on {len(closes.columns)} tickers...")
            try:
                results[name] = BacktestEngine(opens, closes, STRATEGIES[name], **settings).run()
            except ValueError as e:
                # e.g. pairs trading given one ticker: say why and carry on
                skipped.append((name, str(e)))

    print_header(args, closes, f"Strategy comparison: {args.label}")
    print_assets(closes)
    if results:
        print_results(results)
    for name, reason in skipped:
        console.print(f"[yellow]Skipped {name}:[/] {reason}")
    console.print()


def sweep(args, opens, closes, settings) -> None:
    """Backtest every parameter combination of one strategy (in-sample)."""
    strategy_class = type(STRATEGIES[args.strategy])
    with console.status(f"Sweeping {args.strategy} parameters..."):
        table = grid_search(opens, closes, strategy_class, PARAM_GRIDS[args.strategy], **settings)

    print_header(args, closes, f"Parameter sweep: {args.label}")
    console.print(report.sweep_table(table, args.strategy))
    console.print()


def walkforward(args, opens, closes, settings) -> None:
    """Walk-forward test one strategy and compare it with its benchmarks."""
    strategy_class = type(STRATEGIES[args.strategy])
    grid = PARAM_GRIDS[args.strategy]
    train_days = 504  # two trading years; the walk_forward default

    with console.status(f"Walking {args.strategy} forward (re-running the sweep for each window)..."):
        result, windows = walk_forward(opens, closes, strategy_class, grid, train_days, **settings)
        hindsight_params, hindsight = best_in_hindsight(
            opens, closes, strategy_class, grid, train_days, **settings
        )
        # everything is compared over the same out-of-sample days
        test_opens, test_closes = opens.iloc[train_days:], closes.iloc[train_days:]
        buy_and_hold = BacktestEngine(test_opens, test_closes, BuyAndHold(), **settings).run()

    print_header(args, test_closes, f"Walk-forward (out of sample): {args.label}")
    console.print(report.walk_forward_table(windows, args.strategy, train_days))
    console.print("[dim]Yellow parameters changed from the window before.[/]\n")

    results = {"walk-forward": result, "best in hindsight": hindsight, "buy_and_hold": buy_and_hold}
    print_results(results, "Walk-forward against its benchmarks")
    params = ", ".join(f"{k}={v}" for k, v in hindsight_params.items())
    console.print(f"[dim]Best in hindsight used[/] {params}\n")


def list_universe(args) -> None:
    """Show the known tickers and baskets, and which have data."""
    available = {path.stem for path in PROCESSED_DATA_PATH.glob("*.parquet")}
    console.print()
    console.print(report.baskets_table(available))
    console.print(report.universe_table(available))
    extra = sorted(available - set(UNIVERSE))
    if extra:
        console.print(f"[dim]Also downloaded, not in the universe:[/] {', '.join(extra)}\n")


def download(args) -> None:
    """Fetch and process data for the chosen tickers."""
    rows = []
    with console.status("Downloading...") as status:
        for ticker in args.tickers:
            status.update(f"Downloading {ticker} ({len(rows) + 1} of {len(args.tickers)})...")
            outcome, data = download_and_process(ticker, args.start, refresh=args.refresh)
            rows.append((ticker, outcome, data))
    console.print()
    console.print(report.download_table(rows))
    failed = [ticker for ticker, outcome, _ in rows if outcome.startswith("failed")]
    if failed:
        console.print(f"[red]{len(failed)} failed:[/] {', '.join(failed)}\n")


BACKTEST_COMMANDS = {"compare": compare, "sweep": sweep, "walkforward": walkforward}
DATA_COMMANDS = {"list": list_universe, "download": download}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest trading strategies.")
    parser.add_argument(
        "command", nargs="?", default="compare", choices=[*BACKTEST_COMMANDS, *DATA_COMMANDS],
        help="what to do (default: compare)",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--universe", choices=[*BASKETS, "all"], default=None,
        help=f"a named basket of tickers; run `list` to see them (default: {DEFAULT_BASKET})",
    )
    selection.add_argument(
        "--tickers", nargs="+", type=str.upper, help="your own tickers instead of a basket",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES),
        help="compare: which strategies to run (default: all)",
    )
    parser.add_argument(
        "--strategy", default="ma_crossover", choices=list(PARAM_GRIDS),
        help="sweep/walkforward: which strategy to search (default: ma_crossover)",
    )
    parser.add_argument("--cash", type=float, default=10_000.0, help="starting cash (default: 10000)")
    parser.add_argument(
        "--commission", type=float, default=0.001,
        help="commission as a fraction of traded value (default: 0.001 = 0.10%%)",
    )
    parser.add_argument(
        "--slippage", type=float, default=0.0005,
        help="adverse fill-price move as a fraction (default: 0.0005 = 0.05%%)",
    )
    parser.add_argument(
        "--start", default=DEFAULT_START, help=f"download: first date to fetch (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="download: re-fetch tickers that already have data",
    )
    args = parser.parse_args()

    # settle on one list of tickers, plus a label for titles
    if args.tickers:
        args.label = ", ".join(args.tickers) if len(args.tickers) <= 6 else f"{len(args.tickers)} tickers"
    else:
        args.universe = args.universe or DEFAULT_BASKET
        args.tickers = list(UNIVERSE) if args.universe == "all" else basket_tickers(args.universe)
        args.label = f"{args.universe} basket"
    return args


def check_data(args: argparse.Namespace) -> None:
    """Fail with the command that fixes it if any ticker has no data yet."""
    missing = [ticker for ticker in args.tickers if not has_processed_data(ticker)]
    if missing:
        selection = f"--universe {args.universe}" if args.universe else f"--tickers {' '.join(missing)}"
        raise FileNotFoundError(
            f"no data yet for {', '.join(missing)}. Fetch it with:\n\n"
            f"    python main.py download {selection}\n"
        )


if __name__ == "__main__":
    args = parse_args()
    settings = {
        "initial_cash": args.cash,
        "commission_rate": args.commission,
        "slippage": args.slippage,
    }
    try:
        if args.command in DATA_COMMANDS:
            DATA_COMMANDS[args.command](args)
        else:
            check_data(args)
            opens = load_price_panel("Open", args.tickers)
            closes = load_price_panel("Close", args.tickers)
            BACKTEST_COMMANDS[args.command](args, opens, closes, settings)
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[bold red]error:[/] {e}")
        raise SystemExit(1) from None
