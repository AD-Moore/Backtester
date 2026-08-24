
from data.downloader import fetch_market_data, fetch_multiple_tickers

fetch_multiple_tickers(["AAPL", "MSFT", "GOOGL"], start="2020-01-01", end="2023-01-01")

