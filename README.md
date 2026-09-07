# Quant Backtester

A modular Python backtesting framework for designing, testing, and evaluating algorithmic trading strategies on historical market data. Built from scratch rather than on top of an existing backtesting library, to understand how each piece — data handling, strategy logic, execution simulation, performance measurement — actually works.

---

## Current Status

**Data pipeline: complete.**
- `downloader.py` pulls historical OHLCV data via `yfinance` and stores it as raw parquet files
- `processor.py` cleans the raw data (deduplication, missing-value handling) and derives returns, log returns, and rolling volatility
- `loader.py` provides typed accessors for loading single tickers, groups of tickers, or a full multi-ticker price panel

**Strategies: in progress.**
- `strategy_base.py` defines the base interface all strategies will implement
- No concrete strategies (moving average, mean reversion, momentum) implemented yet

**Engine: not started.**
- Backtest execution, portfolio tracking, and order/position modeling are planned but not yet built

**Performance: not started.**
- Metrics and visualization (Sharpe ratio, drawdown, equity curve plotting) are planned but not yet built

---

## Setup

```bash
git clone <repo-url>
cd backtester
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage (data pipeline)

```bash
python data/downloader.py   # fetch raw OHLCV data for the configured tickers
python data/processor.py    # clean raw data and compute derived columns
```

```python
from data.loader import load_ticker, load_price_panel

aapl = load_ticker("AAPL")
prices = load_price_panel(field="Close")  # all tickers, one column each
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
│   ├── moving_average.py      # planned
│   ├── mean_reversion.py      # planned
│   └── momentum.py            # planned
│
├── engine/                    # planned
│   ├── backtest_engine.py
│   ├── portfolio.py
│   ├── position.py
│   └── order.py
│
├── performance/                # planned
│   ├── metrics.py
│   ├── visualization.py
│   └── risk_metrics.py
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
- [ ] Implement moving average, mean reversion, and momentum strategies
- [ ] Build the backtest engine (order execution, position tracking, portfolio state)
- [ ] Compute performance metrics (Sharpe, max drawdown, CAGR)
- [ ] Visualize results (equity curve, drawdown chart)
- [ ] Add transaction costs and slippage modeling
- [ ] Portfolio optimization across strategies/assets

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