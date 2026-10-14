"""Terminal reports, drawn with rich.

Every function here builds something printable (a Table, Panel or Group)
and returns it instead of printing, so main.py decides what is shown and
the tests can render it to a string.
"""
import math

import pandas as pd
from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from data.universe import BASKETS, UNIVERSE, asset_info, sector_counts
from engine.result import BacktestResult
from performance.metrics import TRADING_DAYS_PER_YEAR, compute_metrics, trade_statistics

# Correlations are only drawn as a grid up to this many tickers; past
# that the grid is wider than most terminals.
MAX_CORRELATION_GRID = 12

# A sector holding at least this share of the tickers gets a warning.
CONCENTRATION_WARNING = 0.4

BAR_WIDTH = 20


# ---------------------------------------------------------------- formatting


def pct(value: float, signed: bool = True, decimals: int = 1) -> Text:
    """A percentage, green when positive and red when negative."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return Text("–", style="dim")
    sign = "+" if signed else ""
    style = "green" if value > 0 else "red" if value < 0 else ""
    return Text(f"{value:{sign}.{decimals}%}", style=style)


def ratio(value: float) -> Text:
    """A ratio such as Sharpe: red below zero, bold green from 1.0."""
    if math.isnan(value):
        return Text("–", style="dim")
    if math.isinf(value):
        return Text("∞", style="green")
    style = "red" if value < 0 else "bold green" if value >= 1 else ""
    return Text(f"{value:.2f}", style=style)


def dollars(value: float) -> Text:
    return Text(f"${value:,.0f}")


def bar(fraction: float, width: int = BAR_WIDTH, style: str = "cyan") -> Text:
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return Text("█" * filled, style=style) + Text("░" * (width - filled), style="dim")


def _highlight_best(cells: list[Text], values: list[float], higher_is_better: bool) -> None:
    """Underline and bold the best value in a column, in place."""
    finite = [(v, i) for i, v in enumerate(values) if not math.isnan(v)]
    if len(finite) < 2:
        return
    _, best = max(finite) if higher_is_better else min(finite)
    cells[best].stylize("bold underline")


# ------------------------------------------------------------------ headers


def header(title: str, closes: pd.DataFrame, cash: float, commission: float, slippage: float) -> Panel:
    start, end = closes.index[0], closes.index[-1]
    years = (end - start).days / 365.25
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column()
    grid.add_row("Tickers", ", ".join(closes.columns))
    span = f"{years:.1f} years, {len(closes):,} trading days"
    grid.add_row("Period", f"{start.date()} to {end.date()}  ({span})")
    grid.add_row("Starting cash", f"${cash:,.0f}")
    grid.add_row("Costs", f"{commission:.2%} commission + {slippage:.2%} slippage per fill")
    return Panel(
        grid, title=f"[bold]{title}[/]", title_align="left",
        border_style="blue", box=box.ROUNDED, expand=False,
    )


# ------------------------------------------------------------- the assets


def asset_table(closes: pd.DataFrame) -> Table:
    """What simply holding each ticker on its own would have done, best
    CAGR first. A reference point before any strategy is run."""
    rows = []
    for ticker in closes.columns:
        m = compute_metrics(closes[ticker])
        rows.append((ticker, m))
    rows.sort(key=lambda row: -row[1]["annualized_return"])

    table = Table(
        title="The assets: each one held on its own",
        title_justify="left",
        title_style="bold",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
    )
    table.add_column("Ticker", style="bold")
    table.add_column("Name", overflow="ellipsis", max_width=30)
    table.add_column("Sector", style="magenta")
    for column in ("Return", "CAGR", "Volatility", "Sharpe", "Max DD"):
        table.add_column(column, justify="right")

    for ticker, m in rows:
        info = asset_info(ticker)
        table.add_row(
            ticker,
            info.name,
            info.sector,
            pct(m["total_return"], decimals=0),
            pct(m["annualized_return"]),
            pct(m["annualized_volatility"], signed=False),
            ratio(m["sharpe_ratio"]),
            pct(m["max_drawdown"]),
        )
    return table


def sector_table(tickers: list[str]) -> Group:
    """How the tickers spread across sectors, with a warning if one
    sector dominates."""
    counts = sector_counts(tickers)
    total = len(tickers)

    table = Table(
        title="Sector mix",
        title_justify="left",
        title_style="bold",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
    )
    table.add_column("Sector", style="magenta")
    table.add_column("Tickers")
    table.add_column("Share", justify="right")
    table.add_column("", no_wrap=True)
    for sector, count in counts.items():
        members = ", ".join(t for t in tickers if asset_info(t).sector == sector)
        share = count / total
        table.add_row(sector, members, f"{share:.0%}", bar(share))

    top_sector, top_count = next(iter(counts.items()))
    if total > 1 and top_count / total >= CONCENTRATION_WARNING:
        note = Text(
            f"⚠  {top_count / total:.0%} of the tickers are {top_sector}. Results will mostly "
            f"reflect how that one sector did; try --universe diversified or multi_asset.",
            style="yellow",
        )
    else:
        note = Text(f"✓  Spread across {len(counts)} sectors; none holds more than "
                    f"{CONCENTRATION_WARNING:.0%} of the tickers.", style="green")
    return Group(table, note)


def _correlation_style(value: float) -> str:
    if value >= 0.7:
        return "bold red"
    if value >= 0.4:
        return "yellow"
    if value >= 0.0:
        return "green"
    return "bold cyan"


def average_correlation(closes: pd.DataFrame) -> float:
    """Mean correlation of daily returns over every distinct pair."""
    corr = closes.pct_change().dropna().corr()
    n = len(corr)
    if n < 2:
        return float("nan")
    off_diagonal = corr.to_numpy().sum() - n
    return off_diagonal / (n * (n - 1))


def correlation_table(closes: pd.DataFrame) -> Group | None:
    """Correlation of daily returns between every pair of tickers. Lower
    is better for diversification: two assets that fall together offer
    no protection from each other. None if there is only one ticker."""
    if len(closes.columns) < 2:
        return None
    corr = closes.pct_change().dropna().corr()
    avg = average_correlation(closes)
    verdict = (
        "low: these assets often move independently" if avg < 0.3
        else "moderate" if avg < 0.6
        else "high: these assets tend to rise and fall together"
    )
    summary = Text.assemble(
        ("Average pairwise correlation ", "bold"),
        (f"{avg:.2f}", _correlation_style(avg)),
        f"  ({verdict})",
    )
    legend = Text.assemble(
        "  ", ("≥0.7 ", "bold red"), ("≥0.4 ", "yellow"), ("≥0 ", "green"), ("<0", "bold cyan"),
        ("  red = little diversification, blue = moves the opposite way", "dim"),
    )

    if len(corr) > MAX_CORRELATION_GRID:
        return Group(Text("Correlation", style="bold"), summary)

    table = Table(
        title="Correlation of daily returns",
        title_justify="left",
        title_style="bold",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        pad_edge=False,
    )
    table.add_column("", style="bold")
    for ticker in corr.columns:
        table.add_column(ticker, justify="right")
    for i, row_ticker in enumerate(corr.index):
        cells = []
        for j, column_ticker in enumerate(corr.columns):
            value = corr.loc[row_ticker, column_ticker]
            if i == j:
                cells.append(Text("·", style="dim"))
            else:
                cells.append(Text(f"{value:.2f}", style=_correlation_style(value)))
        table.add_row(row_ticker, *cells)
    return Group(table, summary, legend)


# ---------------------------------------------------------- the strategies


def _strategy_stats(result: BacktestResult) -> dict[str, float]:
    equity = result.equity_curve
    years = max(len(equity) - 1, 1) / TRADING_DAYS_PER_YEAR
    trades = trade_statistics(result.trades)
    return {
        "final_value": equity.iloc[-1],
        **compute_metrics(equity),
        **trades,
        "trades_per_year": trades["num_trades"] / years,
    }


def _metric_table(results: dict[str, BacktestResult], title: str, caption: str, columns: list) -> Table:
    """One row per backtest and one column per (header, stat key,
    formatter, higher_is_better) entry. The best value in each column
    with a direction is underlined."""
    table = Table(
        title=title,
        title_justify="left",
        title_style="bold",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        caption=caption,
        caption_justify="left",
        caption_style="dim",
    )
    table.add_column("Strategy", style="bold", no_wrap=True)
    for name, *_ in columns:
        table.add_column(name, justify="right", no_wrap=True)

    stats = {label: _strategy_stats(result) for label, result in results.items()}
    labels = list(results)
    cells_by_column = []
    for _, key, formatter, higher_is_better in columns:
        values = [float(stats[label][key]) for label in labels]
        cells = [formatter(value) for value in values]
        if higher_is_better is not None:
            _highlight_best(cells, values, higher_is_better)
        cells_by_column.append(cells)

    for row, label in enumerate(labels):
        table.add_row(label, *(cells[row] for cells in cells_by_column))
    return table


def comparison_table(results: dict[str, BacktestResult], title: str = "Strategy comparison") -> Table:
    """Returns and risk, one row per backtest."""
    columns = [
        ("Final value", "final_value", dollars, True),
        ("Return", "total_return", lambda v: pct(v, decimals=0), True),
        ("CAGR", "annualized_return", pct, True),
        ("Vol", "annualized_volatility", lambda v: pct(v, signed=False), False),
        ("Sharpe", "sharpe_ratio", ratio, True),
        ("Sortino", "sortino_ratio", ratio, True),
        ("Calmar", "calmar_ratio", ratio, True),
        ("Max DD", "max_drawdown", pct, True),
    ]
    caption = (
        "Best in each column underlined.  Vol = annualized volatility.  Max DD = worst fall from a peak.\n"
        "Sortino counts only downside swings as risk; Calmar is CAGR per unit of max drawdown."
    )
    return _metric_table(results, title, caption, columns)


def trading_table(results: dict[str, BacktestResult]) -> Table:
    """How each strategy got there: how often it traded and how the
    individual round trips went."""
    columns = [
        ("Trades", "num_trades", lambda v: Text(f"{v:.0f}"), None),
        ("Per year", "trades_per_year", lambda v: Text(f"{v:.1f}"), None),
        ("Win rate", "win_rate", lambda v: pct(v, signed=False, decimals=0), True),
        ("Avg trade", "avg_trade_return", pct, True),
        ("Profit factor", "profit_factor", ratio, True),
    ]
    caption = (
        "Completed round trips only; positions still open at the end aren't counted.\n"
        "Profit factor = dollars won per dollar lost (above 1 is profitable)."
    )
    return _metric_table(results, "Trading activity", caption, columns)


def annual_returns(equity_curve: pd.Series) -> pd.Series:
    """Return in each calendar year, indexed by year. The first and last
    years cover only the part of the year inside the backtest."""
    year_end = equity_curve.groupby(equity_curve.index.year).last()
    previous = year_end.shift(1, fill_value=equity_curve.iloc[0])
    return year_end / previous - 1


def annual_returns_table(results: dict[str, BacktestResult]) -> Table:
    """Calendar-year returns, one column per strategy. Shows whether a
    strategy's total came steadily or from one lucky year."""
    by_strategy = pd.DataFrame({label: annual_returns(r.equity_curve) for label, r in results.items()})
    dates = next(iter(results.values())).equity_curve.index
    partial = set()
    if (dates[0].month, dates[0].day) > (1, 7):
        partial.add(dates[0].year)
    if (dates[-1].month, dates[-1].day) < (12, 24):
        partial.add(dates[-1].year)

    table = Table(
        title="Year by year",
        title_justify="left",
        title_style="bold",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        caption="* part of a year.  Best strategy each year underlined.",
        caption_justify="left",
        caption_style="dim",
    )
    table.add_column("Year", style="bold")
    for label in by_strategy.columns:
        table.add_column(label, justify="right")

    for year, row in by_strategy.iterrows():
        cells = [pct(value) for value in row]
        _highlight_best(cells, list(row), higher_is_better=True)
        table.add_row(f"{year}{'*' if year in partial else ''}", *cells)

    # summary rows: how often each strategy made money, and its range
    table.add_section()
    table.add_row(
        Text("Up years", style="dim"),
        *(Text(f"{(col > 0).sum()} of {len(col)}", style="dim") for _, col in by_strategy.items()),
    )
    table.add_row(Text("Best", style="dim"), *(pct(col.max()) for _, col in by_strategy.items()))
    table.add_row(Text("Worst", style="dim"), *(pct(col.min()) for _, col in by_strategy.items()))
    return table


def takeaways(results: dict[str, BacktestResult], benchmark: str = "buy_and_hold") -> Panel:
    """A few plain-language lines summarising the comparison."""
    metrics = {label: compute_metrics(r.equity_curve) for label, r in results.items()}
    lines = []

    def leader(key: str, higher_is_better: bool = True) -> str:
        pick = max if higher_is_better else min
        return pick(metrics, key=lambda label: metrics[label][key])

    best_sharpe = leader("sharpe_ratio")
    best_return = leader("total_return")
    shallowest = leader("max_drawdown")
    lines.append(Text.assemble("Best risk-adjusted:  ", (best_sharpe, "bold"),
                               f"  (Sharpe {metrics[best_sharpe]['sharpe_ratio']:.2f})"))
    lines.append(Text.assemble("Most money made:     ", (best_return, "bold"),
                               f"  ({metrics[best_return]['total_return']:+.0%})"))
    lines.append(Text.assemble("Smallest drawdown:   ", (shallowest, "bold"),
                               f"  ({metrics[shallowest]['max_drawdown']:.1%})"))

    if benchmark in metrics and len(metrics) > 1:
        base = metrics[benchmark]["sharpe_ratio"]
        others = [label for label in metrics if label != benchmark]
        beat = [label for label in others if metrics[label]["sharpe_ratio"] > base]
        style = "green" if beat else "yellow"
        lines.append(Text(
            f"{len(beat)} of {len(others)} strategies beat buy-and-hold's Sharpe of {base:.2f}"
            + (f": {', '.join(beat)}" if beat else ""),
            style=style,
        ))
    return Panel(Group(*lines), title="[bold]Takeaways[/]", title_align="left",
                 border_style="green", box=box.ROUNDED, expand=False)


# ------------------------------------------------------ sweep / walk-forward


def sweep_table(table: pd.DataFrame, strategy: str) -> Group:
    out = Table(
        title=f"{strategy}: {len(table)} parameter combinations, best Sharpe first",
        title_justify="left",
        title_style="bold",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
    )
    out.add_column("#", style="dim", justify="right")
    out.add_column("Parameters", style="bold")
    for column in ("Return", "CAGR", "Sharpe", "Max DD", "Trades"):
        out.add_column(column, justify="right")
    for rank, row in enumerate(table.itertuples(), start=1):
        params = ", ".join(f"{k}={v}" for k, v in row.params.items())
        out.add_row(
            str(rank),
            params,
            pct(row.total_return, decimals=0),
            pct(row.annualized_return),
            ratio(row.sharpe_ratio),
            pct(row.max_drawdown),
            str(row.num_trades),
        )
    note = Text(
        "In-sample: the top row was picked using the same data it is scored on. "
        "Run `walkforward` to see how the picking holds up out of sample.",
        style="dim",
    )
    return Group(out, note)


def walk_forward_table(windows: pd.DataFrame, strategy: str, train_days: int) -> Table:
    out = Table(
        title=f"{strategy}: parameters re-picked from the previous {train_days} days",
        title_justify="left",
        title_style="bold",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
    )
    out.add_column("Test window")
    out.add_column("Parameters chosen", style="bold")
    out.add_column("Train Sharpe", justify="right")
    out.add_column("Test return", justify="right")
    previous = None
    for row in windows.itertuples():
        params = ", ".join(f"{k}={v}" for k, v in row.params.items())
        changed = previous is not None and row.params != previous
        out.add_row(
            f"{row.test_start.date()} → {row.test_end.date()}",
            Text(params, style="yellow" if changed else ""),
            ratio(row.train_sharpe),
            pct(row.test_return),
        )
        previous = row.params
    return out


# ----------------------------------------------------------- the universe


def universe_table(available: set[str]) -> Group:
    """Every known ticker grouped by sector, marking which have data."""
    table = Table(
        title="Ticker universe",
        title_justify="left",
        title_style="bold",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
    )
    table.add_column("Sector / asset class", style="magenta")
    table.add_column("Tickers")

    by_sector: dict[str, list[str]] = {}
    for ticker, asset in UNIVERSE.items():
        by_sector.setdefault(asset.sector, []).append(ticker)

    for sector, tickers in by_sector.items():
        labels = Text(", ").join(
            Text(f"{t}✓", style="green") if t in available else Text(t, style="dim")
            for t in tickers
        )
        table.add_row(sector, labels)
    note = Text("✓ = data downloaded.  Fetch the rest with `python main.py download --universe all`.",
                style="dim")
    return Group(table, note)


def baskets_table(available: set[str]) -> Table:
    table = Table(
        title="Baskets  (use with --universe NAME)",
        title_justify="left",
        title_style="bold",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
    )
    table.add_column("Name", style="bold")
    table.add_column("What it is")
    table.add_column("Sectors", justify="right")
    table.add_column("Data", justify="right")
    for name, (description, tickers) in BASKETS.items():
        have = sum(t in available for t in tickers)
        style = "green" if have == len(tickers) else "yellow" if have else "dim"
        table.add_row(
            name,
            f"{description}\n[dim]{' '.join(tickers)}[/]",
            str(len(sector_counts(tickers))),
            Text(f"{have}/{len(tickers)}", style=style),
        )
    return table


def download_table(rows: list[tuple[str, str, pd.DataFrame | None]]) -> Table:
    table = Table(
        title="Download",
        title_justify="left",
        title_style="bold",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
    )
    table.add_column("Ticker", style="bold")
    table.add_column("Sector", style="magenta")
    table.add_column("Status")
    table.add_column("Rows", justify="right")
    table.add_column("From")
    table.add_column("To")
    for ticker, status, data in rows:
        style = "green" if status == "downloaded" else "dim" if status == "cached" else "red"
        if data is None:
            table.add_row(ticker, asset_info(ticker).sector, Text(status, style=style), "", "", "")
        else:
            table.add_row(
                ticker,
                asset_info(ticker).sector,
                Text(status, style=style),
                f"{len(data):,}",
                str(data.index[0].date()),
                str(data.index[-1].date()),
            )
    return table

