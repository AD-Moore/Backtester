# Quant Backtester

[![tests](https://github.com/AD-Moore/Backtester/actions/workflows/tests.yml/badge.svg)](https://github.com/AD-Moore/Backtester/actions/workflows/tests.yml)

A modular Python backtesting framework for designing, testing, and evaluating algorithmic trading strategies on historical market data. Built from scratch rather than on top of an existing backtesting library, to understand how each piece — data handling, strategy logic, execution simulation, performance measurement — actually works.

---

## What It Does

- **Data pipeline** — downloads daily OHLCV data with `yfinance`, cleans it, derives returns and rolling volatility, and caches everything as parquet
- **Six strategies behind one interface** — buy-and-hold, moving average crossover, mean reversion, time-series momentum, cross-sectional momentum, and pairs trading
- **One engine for one symbol or a whole portfolio** — walks the data day by day, trades at the *next* day's open to avoid lookahead bias, supports long and short positions, and charges commission and slippage on every fill
- **Trade log and performance report** — CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, trade count, win rate, and profit factor
- **Parameter sweep and walk-forward testing** — re-picks parameters on a rolling training window and scores them only on the data that follows, so the reported result is out of sample
- **Around 150 unit and integration tests**, linted with ruff and run on every push by GitHub Actions

---

## How It Works

The whole project rests on one idea:

> **A strategy turns prices into target weights. The engine trades to match them.**

A *weight* is the fraction of the portfolio to hold in a symbol: `1.0` is everything, `0.5` is half, `0` is none, and a negative number is a short position. Whatever isn't allocated stays in cash.

**1. Data.** `load_price_panel("Close", ["AAPL", "MSFT"])` returns a table with one row per day and one column per ticker. The engine needs two of these: opening prices and closing prices.

**2. Strategy.** Every strategy has one method, `generate_weights(closes)`, which takes the table of closing prices and returns a table of the same shape holding weights. For one ticker, a moving average crossover produces a column of `1.0` (hold it) and `0.0` (hold cash):

```text
              AAPL close    weight
2026-04-21      265.70       0.0     short MA below long MA: in cash
2026-04-22      272.68       0.0
2026-04-23      272.94       1.0     short MA crossed above: hold AAPL
2026-04-24      270.58       1.0     unchanged, so nothing to do
```

**3. Engine.** `BacktestEngine` walks through the days in order. Each day it does three things:

1. Records what the portfolio is worth at today's close.
2. Checks whether any symbol's weight is different from yesterday's.
3. If so, trades that symbol at **tomorrow's open** to reach the new weight.

Step 3 is the important one. The weight for April 23 was decided using April 23's closing price, so the earliest anyone could have acted on it is the next morning. Trading at April 23's own close would be using a price before it existed.

A weight that doesn't change is left alone, so a buy-and-hold position is bought once and never touched. Sells are done before buys, so the cash from one can pay for the other.

**4. Bookkeeping.** The engine doesn't touch cash or shares itself. It creates an `Order` and hands it to the `Portfolio`, which checks the order is affordable and updates its cash and its `Position` for that symbol. When a position goes back to zero, the engine records a `Trade` with its profit.

**5. Results.** `run()` returns a `BacktestResult` with two things: the equity curve (portfolio value each day) and the list of trades. `performance/metrics.py` turns those into Sharpe ratio, drawdown, win rate and the rest.

### Where to start reading

In this order, each file builds on the ones before it. The first seven are the core and total about 450 lines.

| | File | What it is |
|---|---|---|
| 1 | `strategies/strategy_base.py` | The one method every strategy implements |
| 2 | `strategies/moving_average_crossover.py` | A complete strategy in a dozen lines |
| 3 | `engine/order.py` | A buy or sell instruction |
| 4 | `engine/position.py` | Shares held in one symbol |
| 5 | `engine/portfolio.py` | Cash plus positions; executes orders |
| 6 | `engine/backtest_engine.py` | The day-by-day loop described above |
| 7 | `performance/metrics.py` | Equity curve in, statistics out |
| 8 | `main.py` | Wires the pieces together for the command line |
| 9 | `analysis/grid_search.py`, `analysis/walk_forward.py` | Parameter picking, and testing it honestly |

The tests in `test/` double as worked examples: each one runs a tiny case with numbers small enough to check by hand, and a comment showing the arithmetic.

---

## Results

All results start with $10,000 and pay 0.10% commission and 0.05% slippage on every fill. Data runs from 2010-01-04 to 2026-10-13.

### One symbol: AAPL

| Strategy | Total return | CAGR | Volatility | Sharpe | Max drawdown | Trades | Win rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Buy and hold | +5,206.8% | 26.8% | 28.1% | 0.99 | -43.8% | 0 | – |
| MA crossover (20/50) | +818.4% | 14.2% | 20.2% | 0.76 | -28.9% | 49 | 49% |
| Mean reversion (20-day, z = 2) | +115.0% | 4.7% | 16.5% | 0.36 | -46.3% | 60 | 68% |
| Momentum (126-day) | +1,927.5% | 19.7% | 22.9% | 0.90 | -40.6% | 46 | 41% |

- **Nothing beat buy-and-hold**, on raw return or on Sharpe. Each timing strategy spent long stretches in cash during a period when the stock mostly went up.
- **Trend-following mostly bought lower risk, not higher return.** The crossover ran at about 70% of buy-and-hold's volatility with a drawdown of -28.9% against -43.8%. Mean reversion didn't even manage that: lower volatility, but a deeper drawdown than simply holding.
- **Win rate is a poor guide to profitability.** Mean reversion won 68% of its trades and made the least money; momentum won 41% and made the most, because its winners were much larger than its losers.
- **Costs matter.** Removing commission and slippage lifts the crossover's total return from +818.4% to +966.1%.

### A portfolio: AAPL, MSFT, GOOGL, NVDA, VGT, VOO

The same strategies with the money split across six symbols (the `tech` basket). Buy-and-hold here means an equal share of each, bought once. VOO launched in September 2010, so this run starts on 2010-09-09.

| Strategy | Total return | CAGR | Volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|
| Buy and hold (equal weight) | +18,388.4% | 38.4% | 32.3% | 1.17 | -53.7% |
| MA crossover | +919.1% | 15.6% | 15.9% | 0.99 | -33.9% |
| Mean reversion | +265.2% | 8.4% | 11.9% | 0.74 | -24.2% |
| Momentum | +2,657.7% | 22.9% | 19.4% | 1.16 | -32.3% |
| Cross-sectional momentum (top 2) | +14,370.3% | 36.3% | 30.5% | 1.17 | -48.9% |

- **Diversifying helped every strategy.** Each one has a higher Sharpe ratio across six symbols than on AAPL alone, because the symbols don't all fall at the same time.
- **Momentum matched buy-and-hold's Sharpe with 60% of its volatility** (1.16 against 1.17) and a much shallower drawdown, though it earned far less in total.
- **Pairs trading lost money on every pair tried** (MSFT/GOOGL, AAPL/MSFT, VGT/VOO, MSFT/VGT, VOO/TLT). The strategy bets that two prices that drift apart will come back together; over this period, they mostly kept drifting.

### Out of sample: walk-forward

A parameter sweep will always find *some* setting that looks good on the data it was tuned on. The walk-forward test asks the harder question: if the parameters had been re-picked every six months using only the previous two years, how would they have done on the six months that followed?

The first two years are used only for training, so the out-of-sample period runs from 2012-01-03 for AAPL (2012-09-07 for the portfolios) to 2026-10-13: about 30 test windows. "Best in hindsight" is the single best fixed setting over that same period, chosen by looking at it — the number a plain sweep would report.

| | Walk-forward Sharpe | Best-in-hindsight Sharpe | Buy-and-hold Sharpe |
|---|---:|---:|---:|
| AAPL: MA crossover | 0.69 | 0.98 | 0.94 |
| AAPL: mean reversion | 0.18 | 0.65 | 0.94 |
| AAPL: momentum | 0.88 | 0.95 | 0.94 |
| Portfolio: momentum | 1.11 | 1.23 | 1.19 |
| Portfolio: cross-sectional momentum | 1.13 | 1.25 | 1.19 |
| VGT/VOO: pairs trading | -0.12 | -0.02 | 0.96 |

- **Every strategy did worse out of sample than in hindsight**, and none beat buy-and-hold. The best fixed settings looked as good as or better than buy-and-hold in four of six cases; re-picked honestly, none of them were.
- **How much was hindsight varied by strategy.** Mean reversion on AAPL lost almost three-quarters of its apparent Sharpe; momentum lost less than a tenth.
- **The parameters that won in training rarely kept winning.** The crossover's chosen windows changed 23 times across 30 test periods.
- **The portfolios are dominated by one stock.** Over this period NVDA rose about 980-fold, while the other five symbols rose between 9 and 43 times. Any strategy that held NVDA did well, and with six symbols and one extraordinary stock, the differences between these strategies can't be told apart from luck.

### Limits of these results

Mostly one long bull market for US large-cap technology, six hand-picked tickers that are known winners in hindsight, and about sixteen years of data. Differences between Sharpe ratios of this size are not statistically established. Cash earns no interest, and short positions pay no borrowing cost. The results show the framework working and what it can reveal, not strategies anyone should trade.

---

## Setup

```bash
git clone https://github.com/AD-Moore/Backtester.git
cd Backtester
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

Run the commands from the repository root.

```bash
python main.py list                       # the ticker universe and named baskets, and what's downloaded
python main.py download --universe all    # fetch and clean data for every listed ticker
python main.py                            # compare every strategy on the "diversified" basket
pytest                                    # run the test suite
ruff check .                              # lint
```

### Picking tickers

The universe in `data/universe.py` covers all 11 S&P 500 sectors, sector ETFs, small caps, international stocks, Treasuries, inflation-linked bonds, gold and broad commodities. Choose either a named basket with `--universe` or your own list with `--tickers`:

| Basket | What it is |
|---|---|
| `diversified` (default) | One large company from each of the 11 sectors |
| `sectors` | The 11 Select Sector SPDR ETFs |
| `multi_asset` | US and international stocks, real estate, bonds, gold and commodities |
| `all_weather` | S&P 500, long and medium Treasuries, gold, commodities |
| `defensive` | Staples, health care and utilities |
| `cyclical` | Energy, industrials, materials, financials and discretionary |
| `tech` | The original six technology-heavy tickers |
| `stocks_vs_bonds` | VOO and TLT, a natural pair for pairs trading |

Any ticker yfinance knows works with `--tickers`; fetch it first with `python main.py download --tickers XYZ`. Downloads skip tickers already on disk unless `--refresh` is given, and start from `--start` (default 2010-01-01).

### Commands

`compare` is the default. Its report shows each asset held on its own, the sector mix (with a warning if one sector dominates), a correlation grid, the strategies' returns and risk, their trading activity, calendar-year returns, and a short summary.

```bash
# compare: backtest strategies over the full history
python main.py --universe multi_asset
python main.py --tickers NVDA
python main.py --universe stocks_vs_bonds --strategies buy_and_hold pairs_trading
python main.py --commission 0 --slippage 0                 # no trading costs

# sweep: rank every parameter combination of one strategy (in-sample)
python main.py sweep --strategy momentum --universe sectors

# walkforward: re-pick parameters on a rolling window, score them out of sample
python main.py walkforward --strategy mean_reversion --tickers AAPL
python main.py walkforward --strategy cross_sectional_momentum --universe sectors
```

Cross-sectional momentum needs at least three tickers and pairs trading needs exactly two; `compare` skips any strategy that doesn't fit and says why.

The strategies and their parameters are listed at the top of `main.py` in `STRATEGIES`, and the values that `sweep` and `walkforward` choose between are in `PARAM_GRIDS`. Edit those to try something different.

Using the pieces directly:

```python
from data.loader import load_price_panel
from engine.backtest_engine import BacktestEngine
from performance.metrics import print_performance_report
from strategies.moving_average_crossover import MovingAverageCrossover

tickers = ["AAPL"]                       # or several, for a portfolio
opens = load_price_panel("Open", tickers)
closes = load_price_panel("Close", tickers)

strategy = MovingAverageCrossover(short_window=20, long_window=50)
result = BacktestEngine(
    opens, closes, strategy, initial_cash=10_000.0, commission_rate=0.001, slippage=0.0005
).run()

result.equity_curve   # pd.Series of portfolio value over time
result.trades         # list of finished Trade records

print_performance_report("AAPL / ma_crossover", result.equity_curve, trades=result.trades)
```

Adding a strategy means subclassing `Strategy`, returning a table of weights, and adding it to `STRATEGIES` in `main.py`:

```python
from strategies.strategy_base import Strategy

class MyStrategy(Strategy):
    def generate_weights(self, closes):
        ...  # DataFrame shaped like closes: the fraction of the portfolio in each symbol
```

---

## Project Structure

```text
Backtester/
│
├── data/
│   ├── raw/                    # downloaded parquet (git-ignored)
│   ├── processed/              # cleaned parquet (git-ignored)
│   ├── config.py
│   ├── downloader.py
│   ├── loader.py
│   ├── processor.py
│   └── universe.py             # known tickers, sectors and named baskets
│
├── strategies/
│   ├── strategy_base.py
│   ├── buy_and_hold.py
│   ├── moving_average_crossover.py
│   ├── mean_reversion.py
│   ├── momentum.py
│   ├── cross_sectional_momentum.py
│   └── pairs_trading.py
│
├── engine/
│   ├── backtest_engine.py
│   ├── portfolio.py
│   ├── position.py
│   ├── order.py
│   ├── trade.py
│   └── result.py
│
├── performance/
│   ├── metrics.py
│   └── report.py               # terminal tables (rich)
│
├── analysis/
│   ├── grid_search.py
│   └── walk_forward.py
│
├── utils/
│   └── logger.py
│
├── test/                       # pytest suite
├── .github/workflows/tests.yml
│
├── main.py
├── pyproject.toml              # pytest and ruff configuration
├── requirements.txt
└── README.md
```

---

## Design Decisions

A few choices in this codebase are deliberate rather than incidental, worth calling out for anyone reading the code:

**One interface for one symbol or many.** A strategy returns target weights, and a single-symbol strategy is just the one-column case. There is no separate "portfolio engine": the same loop runs a moving average crossover on AAPL and a long/short pairs trade. The original design had strategies return 1/0 signals for a single symbol; weights are the same idea generalised.

**One-directional layering.** `data/` → `strategies/` → `engine/` → `performance/` → `analysis/`, with `main.py` orchestrating all of them. A strategy has never heard of cash, and the engine has never heard of yfinance. That's what lets strategies be swapped without touching the engine, or the data source changed without touching strategies.

**Lookahead-bias avoidance.** A weight decided from day T's Close is traded at day T+1's Open, never at T's own Close. The test suite enforces the same property twice more: every strategy's weights for day T must be identical whether or not it is shown the days after T, and a walk-forward run's choices and equity must be unchanged when all later data is cut off.

**Trade only on a change of mind.** The engine trades a symbol only when its target weight changes. It does not rebalance daily to hold weights exact as prices drift, which would rack up costs for no decision. The trade-off is that a long-held position's real share of the portfolio wanders from its target.

**Walk-forward is one continuous backtest, not a sum of pieces.** Each test window contributes weights; the windows are joined and run through the engine once. Positions and costs therefore carry across window boundaries exactly as they would in practice.

**Costs are applied where they happen.** Slippage moves the fill price against the trade (buys fill higher, sells lower); commission is a cash charge on the traded value. A trade's profit is simply all the cash that came back minus all the cash that went in, so it is net of costs by construction.

**Immutability where it represents a fact, not evolving state.** `Order`, `Trade` and `BacktestResult` are frozen dataclasses because they're records of something that happened. `Position` and `Portfolio` are mutable because they represent state that changes as fills happen.

**Fail loudly, close to the source.** Empty API responses, missing columns, invalid order sides, selling more than is held, insufficient cash, negative costs, nonsensical strategy parameters, a pairs trade given three symbols — each raises a specific exception naming exactly what went wrong, rather than continuing with bad state.

**Tests use numbers small enough to check by hand.** Engine tests run on four days of prices where Open and Close differ, so a test can tell which price a fill used. Data pipeline tests run against temporary folders and never touch the network.

**Simplifications are named, not hidden.** Whole shares only; no interest on idle cash; no borrowing cost or margin requirement on shorts; costs are a flat percentage rather than a model of spread or market impact; pairs trading uses a fixed half-and-half split rather than an estimated hedge ratio. In a walk-forward run, each training window is scored on its own, so strategies with long lookbacks spend the start of it warming up. None of these are accidental gaps — they're tracked on the roadmap below.

---

## Roadmap

- [x] Download and cache historical market data
- [x] Clean data and derive return/volatility columns
- [x] Define a common strategy interface
- [x] Build the backtest engine (order execution, position tracking, portfolio state)
- [x] Implement buy-and-hold and moving average crossover strategies
- [x] Compute performance metrics (Sharpe, Sortino, max drawdown, CAGR, Calmar)
- [x] Implement mean reversion and momentum strategies
- [x] Track individual trades (count, win rate, avg trade return, profit factor)
- [x] Add transaction costs and slippage modeling
- [x] Command-line interface and continuous integration
- [x] Parameter sweeps and out-of-sample walk-forward testing
- [x] Multi-symbol portfolios and short positions
- [x] Cross-sectional momentum and pairs trading
- [x] A broader, less hand-picked set of tickers (including bonds and gold)
- [ ] Confidence intervals on performance metrics (bootstrap)
- [ ] Fractional shares, interest on cash, and borrowing costs on shorts
- [ ] Portfolio optimization across strategies/assets

---

## Technologies

- Python
- pandas / NumPy
- yfinance
- rich (terminal output)
- pytest, ruff, GitHub Actions

---

## What I'm Learning

Python for data/finance work, software design (modular architecture, separation of concerns), object-oriented design for the strategy/engine layer, and the mechanics of backtesting — including the parts that are easy to get subtly wrong, like avoiding lookahead bias, accounting for realistic trading costs, and telling a real result from one produced by tuning on the test data.

---

## Author

Andrew D. Moore
