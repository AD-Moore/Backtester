import pandas as pd
import yfinance as yf

from data.config import PROCESSED_DATA_PATH, RAW_DATA_PATH
from data.processor import process_ticker
from data.universe import UNIVERSE
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_START = "2010-01-01"


def fetch_market_data(
    ticker: str,
    start: str = DEFAULT_START,
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
    logger.debug(f"{ticker}: saved {len(market_data)} rows to {file_path}")

    return market_data


def fetch_multiple_tickers(
    tickers: list[str],
    start: str = DEFAULT_START,
    end: str | None = None,
) -> None:
    """Download historical data for a list of tickers, skipping any that fail."""
    for ticker in tickers:
        try:
            market_data = fetch_market_data(ticker, start, end)
            logger.info(f"{ticker}: saved {len(market_data)} rows")
        except ValueError as e:
            logger.error(f"{ticker}: skipped ({e})")


def has_processed_data(ticker: str) -> bool:
    return (PROCESSED_DATA_PATH / f"{ticker.upper()}.parquet").exists()


def download_and_process(
    ticker: str,
    start: str = DEFAULT_START,
    end: str | None = None,
    refresh: bool = False,
) -> tuple[str, pd.DataFrame | None]:
    """Make sure one ticker has processed data on disk. Returns what
    happened ("cached", "downloaded" or an error message) and the
    processed data, which is None if it failed. Already-processed tickers
    are left alone unless `refresh` is set."""
    ticker = ticker.upper()
    if has_processed_data(ticker) and not refresh:
        return "cached", pd.read_parquet(PROCESSED_DATA_PATH / f"{ticker}.parquet")
    try:
        fetch_market_data(ticker, start, end)
        return "downloaded", process_ticker(ticker)
    except Exception as e:  # network errors, bad tickers, malformed responses
        return f"failed: {e}", None


if __name__ == "__main__":
    fetch_multiple_tickers(list(UNIVERSE), start=DEFAULT_START, end=None)
