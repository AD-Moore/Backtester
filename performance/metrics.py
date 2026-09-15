"""Performance metrics computed from a BacktestEngine equity curve.

Trade count, win rate, and transaction costs are intentionally left out --
the engine doesn't track individual trades or costs yet (see README).
"""
import numpy as np
import pandas as pd

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


def max_drawdown(equity_curve: pd.Series) -> float:
    """Largest peak-to-trough decline, as a negative fraction."""
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()


def compute_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> dict[str, float]:
    return {
        "total_return": total_return(equity_curve),
        "annualized_return": annualized_return(equity_curve),
        "annualized_volatility": annualized_volatility(equity_curve),
        "sharpe_ratio": sharpe_ratio(equity_curve, risk_free_rate),
        "max_drawdown": max_drawdown(equity_curve),
    }


def print_performance_report(
    label: str, equity_curve: pd.Series, risk_free_rate: float = 0.0
) -> None:
    m = compute_metrics(equity_curve, risk_free_rate)
    initial = equity_curve.iloc[0]
    final = equity_curve.iloc[-1]

    width = 44
    label_width = 24

    print("=" * width)
    print(f"BACKTEST RESULTS: {label}")
    print("=" * width)
    print()
    print(f"{'Initial Capital:':<{label_width}}${initial:,.2f}")
    print(f"{'Final Value:':<{label_width}}${final:,.2f}")
    print(f"{'Total Return:':<{label_width}}{m['total_return']:+.2%}")
    print()
    print(f"{'Annualized Return:':<{label_width}}{m['annualized_return']:.2%}")
    print(f"{'Annualized Volatility:':<{label_width}}{m['annualized_volatility']:.2%}")
    print(f"{'Sharpe Ratio:':<{label_width}}{m['sharpe_ratio']:.2f}")
    print()
    print(f"{'Max Drawdown:':<{label_width}}{m['max_drawdown']:.2%}")
    print("=" * width)
    print("(trade count / win rate / costs not tracked yet)")


if __name__ == "__main__":
    from data.loader import load_ticker
    from engine.backtest_engine import BacktestEngine
    from strategies.moving_average_crossover import MovingAverageCrossover

    df = load_ticker("AAPL")
    engine = BacktestEngine(df, MovingAverageCrossover(short_window=20, long_window=50))
    equity_curve = engine.run()

    print_performance_report("AAPL / ma_20_50", equity_curve)
