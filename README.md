# Quant Backtester

A modular Python backtesting framework for designing, testing, and evaluating algorithmic trading strategies on historical market data. Built from scratch rather than on top of an existing backtesting library, to understand how each piece — data handling, strategy logic, execution simulation, performance measurement — actually works.

---

## Current Status

**Data pipeline: complete.**
- `downloader.py` pulls historical OHLCV data via `yfinance` and stores it as raw parquet files
- `processor.py` cleans the raw data (deduplication, missing-value handling) and derives returns, log returns, and rolling volatility
- `loader.py` provides typed accessors for loading single tickers, groups of tickers, or a full multi-ticker price panel

**Strategies: in progress.**
- `strategy_base.py` defines the common `Strategy` interface (`generate_signals`, returning 1 = long, 0 = flat, -1 = short)
- `buy_and_hold.py` and `moving_average_crossover.py` (configurable short/long windows) are implemented
- Mean reversion and momentum strategies not yet implemented

**Engine: core loop working.**
- `order.py` — immutable `Order` records (symbol, side, quantity, date)
- `position.py` — tracks quantity/average cost for a single symbol, realized P&L on sell
- `portfolio.py` — owns cash and positions, executes orders, marks total equity to market
- `backtest_engine.py` — walks price data day by day, turns signals into next-day-open fills, and produces an equity curve
- Current simplifications (by design, for this initial version): all-in-or-flat sizing, whole shares only, no transaction costs/slippage, single symbol per run, long-only

**Performance: metrics implemented, visualization not started.**
- `metrics.py` computes total return, annualized return (CAGR), annualized volatility, Sharpe ratio, and max drawdown directly from the equity curve, and is wired into `main.py`'s output
- Trade count, win rate, and transaction costs are not tracked — the engine doesn't log individual trades or model costs yet
- Equity curve / drawdown charting not yet built

---

## Setup

```bash
git clone <repo-url>
cd backtester
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

```bash
python data/downloader.py   # fetch raw OHLCV data for the configured tickers
python data/processor.py    # clean raw data and compute derived columns
python main.py               # run a strategy through the backtest engine
```

```python
from data.loader import load_ticker, load_price_panel

aapl = load_ticker("AAPL")
prices = load_price_panel(field="Close")  # all tickers, one column each
```

```python
from data.loader import load_ticker
from engine.backtest_engine import BacktestEngine
from strategies.moving_average_crossover import MovingAverageCrossover

df = load_ticker("AAPL")
strategy = MovingAverageCrossover(short_window=20, long_window=50)

engine = BacktestEngine(df, strategy, initial_cash=10_000.0, symbol="AAPL")
equity_curve = engine.run()  # pd.Series of portfolio value over time
```

---

## Project Structure

```text
backtester/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── config.py
│   ├── downloader.py
│   ├── loader.py
│   └── processor.py
│
├── strategies/
│   ├── strategy_base.py
│   ├── buy_and_hold.py
│   ├── moving_average_crossover.py
│   ├── mean_reversion.py      # planned
│   └── momentum.py            # planned
│
├── engine/
│   ├── backtest_engine.py
│   ├── portfolio.py
│   ├── position.py
│   └── order.py
│
├── performance/
│   ├── metrics.py
│   ├── visualization.py        # planned
│   └── risk_metrics.py         # planned
│
├── utils/
│   ├── logger.py
│   ├── config.py               # planned
│   ├── helpers.py               # planned
│   └── indicators.py            # planned
│
├── tests/
├── results/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Roadmap

- [x] Download and cache historical market data
- [x] Clean data and derive return/volatility columns
- [x] Define a common strategy interface
- [x] Build the backtest engine (order execution, position tracking, portfolio state)
- [x] Implement buy-and-hold and moving average crossover strategies
- [x] Compute performance metrics (Sharpe, max drawdown, CAGR)
- [ ] Implement mean reversion and momentum strategies
- [ ] Track individual trades (count, win rate, avg trade return)
- [ ] Visualize results (equity curve, drawdown chart)
- [ ] Add transaction costs and slippage modeling
- [ ] Support fractional shares, short positions, and multi-symbol portfolios
- [ ] Portfolio optimization across strategies/assets

---

## Design Decisions

A few choices in this codebase are deliberate rather than incidental, worth calling out for anyone reading the code:

**One-directional layering.** `data/` → `strategies/` → `engine/` → `performance/`, with `main.py` orchestrating all of them. Each layer only knows the shape of what it receives (a DataFrame in, a Series out) — a strategy has never heard of cash, and the engine has never heard of yfinance. That's what lets strategies be swapped without touching the engine, or the data source changed without touching strategies.

**Lookahead-bias avoidance.** `BacktestEngine` fills a signal generated from day T's Close at day T+1's Open, never at T's own Close. Filling at the same day's close would mean trading on a price that wasn't actually knowable at the moment of the trade — a common way backtests silently overstate performance.

**Immutability where it represents a fact, not evolving state.** `Order` is a frozen dataclass because it's a record of a decision made at a point in time. `Position` and `Portfolio` are mutable because they represent state that changes as fills happen.

**Fail loudly, close to the source.** Empty API responses, missing columns, invalid order sides, overselling a position, insufficient cash — each raises a specific exception naming exactly what went wrong, rather than continuing with bad state.

**Simplifications are named, not hidden.** The current version assumes all-in-or-flat position sizing, whole shares only, no transaction costs or slippage, a single symbol per run, and a 0% risk-free rate in the Sharpe calculation. None of these are accidental gaps — they're a deliberately small first version, tracked on the roadmap above.

---

## Technologies

- Python
- pandas / NumPy
- yfinance
- Matplotlib (planned, for performance visualization)
- SciPy (planned, for optimization)

---

## What I'm Learning

Python for data/finance work, software design (modular architecture, separation of concerns), object-oriented design for the strategy/engine layer, and the mechanics of backtesting — including the parts that are easy to get subtly wrong, like avoiding lookahead bias and accounting for realistic trading costs.

---

## Author

Andrew D. Moore
