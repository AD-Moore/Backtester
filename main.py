import pandas as pd
from data.downloader import fetch_market_data, fetch_multiple_tickers
from data.loader import load_tickers, load_ticker, load_all, load_price_panel


data = load_tickers(["AAPL", "MSFT", "GOOGL", "NVDA", "VOO", "VGT"])

print(data["AAPL"].iloc[20:].head(),"\n")

correlation = data["AAPL"]["Close"].corr(data["MSFT"]["Close"])
print(f"Correlation between AAPL and MSFT: {correlation:.4f}\n")

