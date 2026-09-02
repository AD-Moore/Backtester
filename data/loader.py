import pandas as pd
from pathlib import Path


# Folder where processed market data is stored
PROCESSED_DATA_PATH = Path("data/processed")

"""
    Load the processed data for one ticker.
"""
def load_ticker(ticker: str) -> pd.DataFrame:
    ticker = ticker.upper()

    file_path = PROCESSED_DATA_PATH / f"{ticker}.parquet"

    if not file_path.exists():
        raise FileNotFoundError(
            f"No processed data found for {ticker}"
        )

    return pd.read_parquet(file_path)


"""
    Load processed data for a specific list of tickers.
"""
def load_tickers(tickers: list[str]) -> dict[str, pd.DataFrame]:
    data = {}

    for ticker in tickers:
        data[ticker.upper()] = load_ticker(ticker)

    return data


"""
    Load all processed ticker files.
"""
def load_all() -> dict[str, pd.DataFrame]:
    data = {}

    for file_path in sorted(PROCESSED_DATA_PATH.glob("*.parquet")):
        ticker = file_path.stem
        data[ticker] = pd.read_parquet(file_path)

    return data



"""
    Creates a DataFrame where each column is a ticker
    and each row is a date.
"""
def load_price_panel(field: str = "Close") -> pd.DataFrame:
    
    data = load_all()

    prices = {
        ticker: df[field]
        for ticker, df in data.items()
    }

    return pd.DataFrame(prices).sort_index()