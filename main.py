import pandas as pd
from data.downloader import fetch_market_data, fetch_multiple_tickers
from data.loader import load_tickers, load_ticker, load_all, load_price_panel


data = load_tickers(["AAPL", "MSFT", "GOOGL", "NVDA"])
print(data["AAPL"].head())
prices = load_price_panel().head()
print(prices)
correlation = prices["AAPL"].corr(prices["MSFT"])
print(f"Correlation between AAPL and MSFT: {correlation:.4f}")
