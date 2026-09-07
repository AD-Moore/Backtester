import pandas as pd

from data.config import PROCESSED_DATA_PATH


def load_ticker(ticker: str) -> pd.DataFrame:
    """Load the processed data for one ticker."""
    ticker = ticker.upper()
    file_path = PROCESSED_DATA_PATH / f"{ticker}.parquet"

    if not file_path.exists():
        raise FileNotFoundError(f"No processed data found for {ticker}")

    return pd.read_parquet(file_path)


def load_tickers(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Load processed data for a specific list of tickers."""
    return {ticker.upper(): load_ticker(ticker) for ticker in tickers}


def load_all() -> dict[str, pd.DataFrame]:
    """Load all processed ticker files found on disk."""
    return {
        file_path.stem: pd.read_parquet(file_path)
        for file_path in sorted(PROCESSED_DATA_PATH.glob("*.parquet"))
    }


def load_price_panel(field: str = "Close") -> pd.DataFrame:
    """Build a DataFrame where each column is a ticker and each row is a date."""
    data = load_all()
    prices = {ticker: df[field] for ticker, df in data.items()}
    return pd.DataFrame(prices).sort_index()