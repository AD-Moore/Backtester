# Quant Backtester

A modular Python-based backtesting engine for designing, testing, and evaluating algorithmic trading strategies using historical market data.

The objective of this project is to build a scalable backtesting framework capable of evaluating trading strategies, tracking portfolio performance, calculating risk metrics, and visualizing results. As development progresses, additional features such as portfolio optimization, realistic transaction costs, and advanced quantitative models will be incorporated.

---

## Current Status

Project just started.

Current goals:

- Learn Python for finance
- Download historical stock data
- Understand how financial data is stored
- Build the first moving average indicator

---

## Long-Term Goals

- Download historical market data
- Build technical indicators
- Test trading strategies
- Simulate buying and selling stocks
- Track portfolio performance
- Calculate performance metrics
- Visualize results with charts
- Support multiple trading strategies
- Add realistic trading costs
- Build portfolio optimization tools
- Explore quantitative finance techniques

---

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- yfinance
- C++
- SciPy

---

## Planned Project Structure

The project will be developed using a modular architecture to keep components organized, reusable, and easy to expand.

```text
backtester/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── downloader.py
│   ├── loader.py
│   └── processor.py
│
├── strategies/
│   ├── strategy_base.py
│   ├── moving_average.py
│   ├── mean_reversion.py
│   └── momentum.py
│
├── engine/
│   ├── backtest_engine.py
│   ├── portfolio.py
│   ├── position.py
│   └── order.py
│
├── performance/
│   ├── metrics.py
│   ├── visualization.py
│   └── risk_metrics.py
│
├── utils/
│   ├── config.py
│   ├── helpers.py
│   ├── indicators.py
│   └── logger.py
│
├── tests/
├── results/
│
├── main.py
├── requirements.txt
└── README.md
```
---

## Learning Objectives

This project is helping me learn:

- Python programming
- Software engineering principles
- Object-oriented programming
- Git and GitHub
- Financial data analysis
- Quantitative finance
- Algorithmic trading

---

## Why I Built This

I'm building this project to gain hands-on experience with Python, software engineering, and quantitative finance. Rather than relying on existing backtesting frameworks, the goal is to understand how each component works by implementing it from scratch.

---

## Author

Andrew D. Moore