"""Performance metrics computed from a BacktestEngine run.

Return/risk metrics come from the equity curve; trade statistics come
from the engine's list of completed trades. Costs are not a separate
metric -- they are already baked into both.
"""
import numpy as np
import pandas as pd

from engine.trade import Trade

TRADING_DAYS_PER_YEAR = 252


def total_return(equity_curve: pd.Series) -> float:
    return equity_curve.iloc[-1] / equity_curve.iloc[0] - 1


def daily_returns(equity_curve: pd.Series) -> pd.Series:
    """Day-over-day equity change. Used for volatility/Sharpe instead of
    the underlying asset's returns, since equity reflects being in or out
    of the market."""
    return equity_curve.pct_change().dropna()


def annualized_return(
    equity_curve: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """CAGR -- geometric, unlike sharpe_ratio's arithmetic mean below."""
    n_periods = len(equity_curve) - 1
    if n_periods <= 0:
        return 0.0
    growth = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = n_periods / periods_per_year
    return growth ** (1 / years) - 1


def annualized_volatility(
    equity_curve: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Std of daily returns, scaled by sqrt(time)."""
    return daily_returns(equity_curve).std() * np.sqrt(periods_per_year)


def sharpe_ratio(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Excess return per unit of volatility. risk_free_rate is annual;
    0.0 is a placeholder, not a real assumption -- pass an actual rate
    for a meaningful number."""
    returns = daily_returns(equity_curve)
    volatility = returns.std() * np.sqrt(periods_per_year)
    if volatility == 0:
        return 0.0
    excess_return = returns.mean() * periods_per_year - risk_free_rate
    return excess_return / volatility


def sortino_ratio(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Like Sharpe, but only downside moves count as risk -- a strategy
    isn't penalized for volatility that comes from large gains."""
    returns = daily_returns(equity_curve)
    downside = returns.clip(upper=0)
    downside_deviation = np.sqrt((downside**2).mean()) * np.sqrt(periods_per_year)
    if not downside_deviation > 0:  # also catches NaN from an empty series
        return 0.0
    excess_return = returns.mean() * periods_per_year - risk_free_rate
    return excess_return / downside_deviation


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Fractional decline from the running peak on each day (0 at a new high)."""
    running_max = equity_curve.cummax()
    return (equity_curve - running_max) / running_max


def max_drawdown(equity_curve: pd.Series) -> float:
    """Largest peak-to-trough decline, as a negative fraction."""
    return drawdown_series(equity_curve).min()


def calmar_ratio(equity_curve: pd.Series) -> float:
    """Annualized return per unit of worst drawdown."""
    worst = max_drawdown(equity_curve)
    if worst == 0:
        return 0.0
    return annualized_return(equity_curve) / abs(worst)


def compute_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> dict[str, float]:
    return {
        "total_return": total_return(equity_curve),
        "annualized_return": annualized_return(equity_curve),
        "annualized_volatility": annualized_volatility(equity_curve),
        "sharpe_ratio": sharpe_ratio(equity_curve, risk_free_rate),
        "sortino_ratio": sortino_ratio(equity_curve, risk_free_rate),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar_ratio": calmar_ratio(equity_curve),
    }


def trade_statistics(trades: list[Trade]) -> dict[str, float]:
    """Summary of completed round trips. A position still open at the end
    of the backtest is not a completed trade, so it isn't counted here."""
    if not trades:
        return {
            "num_trades": 0,
            "win_rate": float("nan"),
            "avg_trade_return": float("nan"),
            "profit_factor": float("nan"),
        }

    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = -sum(t.pnl for t in trades if t.pnl < 0)
    return {
        "num_trades": len(trades),
        "win_rate": sum(t.is_win for t in trades) / len(trades),
        "avg_trade_return": sum(t.return_pct for t in trades) / len(trades),
        # dollars won per dollar lost; infinite if there were no losing trades
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
    }


def format_duration(start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
    """Calendar time between two dates, e.g. '6 years, 42 days'."""
    years = end_date.year - start_date.year
    if (end_date.month, end_date.day) < (start_date.month, start_date.day):
        years -= 1
    days = (end_date - (start_date + pd.DateOffset(years=years))).days

    year_label = "year" if years == 1 else "years"
    day_label = "day" if days == 1 else "days"
    return f"{years} {year_label}, {days} {day_label}"


def print_performance_report(
    label: str,
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
    trades: list[Trade] | None = None,
) -> None:
    m = compute_metrics(equity_curve, risk_free_rate)
    initial = equity_curve.iloc[0]
    final = equity_curve.iloc[-1]
    start_date = equity_curve.index[0]
    end_date = equity_curve.index[-1]

    width = 44
    label_width = 24

    print("=" * width)
    print(f"BACKTEST RESULTS: {label}")
    print("=" * width)
    print()
    duration = format_duration(start_date, end_date)
    print(f"{'Period:':<{label_width}}{start_date.date()} to {end_date.date()} ({duration})")
    print(f"{'Initial Capital:':<{label_width}}${initial:,.2f}")
    print(f"{'Final Value:':<{label_width}}${final:,.2f}")
    print(f"{'Total Return:':<{label_width}}{m['total_return']:+.2%}")
    print()
    print(f"{'Annualized Return:':<{label_width}}{m['annualized_return']:.2%}")
    print(f"{'Annualized Volatility:':<{label_width}}{m['annualized_volatility']:.2%}")
    print(f"{'Sharpe Ratio:':<{label_width}}{m['sharpe_ratio']:.2f}")
    print(f"{'Sortino Ratio:':<{label_width}}{m['sortino_ratio']:.2f}")
    print()
    print(f"{'Max Drawdown:':<{label_width}}{m['max_drawdown']:.2%}")
    print(f"{'Calmar Ratio:':<{label_width}}{m['calmar_ratio']:.2f}")

    if trades is not None:
        t = trade_statistics(trades)
        print()
        print(f"{'Completed Trades:':<{label_width}}{t['num_trades']}")
        if t["num_trades"] > 0:
            print(f"{'Win Rate:':<{label_width}}{t['win_rate']:.2%}")
            print(f"{'Avg Trade Return:':<{label_width}}{t['avg_trade_return']:+.2%}")
            print(f"{'Profit Factor:':<{label_width}}{t['profit_factor']:.2f}")
    print("=" * width)
    print()


if __name__ == "__main__":
    from data.loader import load_price_panel
    from engine.backtest_engine import BacktestEngine
    from strategies.moving_average_crossover import MovingAverageCrossover

    opens = load_price_panel("Open", ["AAPL"])
    closes = load_price_panel("Close", ["AAPL"])
    strategy = MovingAverageCrossover(short_window=20, long_window=50)
    result = BacktestEngine(opens, closes, strategy).run()

    print_performance_report("AAPL / ma_crossover", result.equity_curve, trades=result.trades)
