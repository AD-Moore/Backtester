import yfinance as yf
import pandas as pd
from pathlib import Path


# Folder where raw market data will be stored
RAW_DATA_PATH = Path("data/raw")

def fetch_multiple_tickers(tickers: list[str], start: str, end: str | None) -> None:
        for ticker in tickers:
            fetch_market_data(ticker, start, end)

# define a function to fetch market data for a given ticker symbol
def fetch_market_data(
    ticker: str,
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

    # Download historical market data
    market_data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    # Create the file path
    file_path = RAW_DATA_PATH / f"{ticker}.parquet"

    # Save the DataFrame locally after creating it (see line 30)
    market_data.to_parquet(file_path)

    # Return the DataFrame so other parts of the program can use it
    return market_data

