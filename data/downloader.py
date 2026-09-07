import yfinance as yf
import pandas as pd

from data.config import RAW_DATA_PATH
from utils.logger import get_logger

logger = get_logger(__name__)


def fetch_market_data(
    ticker: str,
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download historical OHLCV data for one ticker and save it as parquet."""
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

    file_path = RAW_DATA_PATH / f"{ticker}.parquet"
    market_data.to_parquet(file_path)
    logger.info(f"{ticker}: saved {len(market_data)} rows to {file_path}")

    return market_data


def fetch_multiple_tickers(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
) -> None:
    """Download historical data for a list of tickers, skipping any that fail."""
    for ticker in tickers:
        try:
            fetch_market_data(ticker, start, end)
        except ValueError as e:
            logger.error(f"{ticker}: skipped ({e})")


if __name__ == "__main__":
    fetch_multiple_tickers(
        ["AAPL", "MSFT", "GOOGL", "NVDA", "VGT", "VOO"],
        start="2020-01-01",
        end=None,
    )