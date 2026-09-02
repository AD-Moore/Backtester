import yfinance as yf
import pandas as pd
from pathlib import Path


# Folder where raw market data will be stored
RAW_DATA_PATH = Path("data/raw")


def fetch_multiple_tickers(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
) -> None:
    for ticker in tickers:
        fetch_market_data(ticker, start, end)


# Download historical market data for one ticker
def fetch_market_data(
    ticker: str,
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.DataFrame:

    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

    market_data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    if market_data.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Create the file path
    file_path = RAW_DATA_PATH / f"{ticker}.parquet"

    # Save raw market data
    market_data.to_parquet(file_path)

    # Return DataFrame so other parts of the program can use it
    return market_data


if __name__ == "__main__":
    fetch_multiple_tickers(
        ["AAPL", "MSFT", "GOOGL", "NVDA"],
        start="2020-01-01",
        end=None,
    )