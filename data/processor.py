import numpy as np
import pandas as pd

from data.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, OHLCV_COLUMNS
from utils.logger import get_logger

logger = get_logger(__name__)


def process_ticker(ticker: str) -> pd.DataFrame:
    """Clean one ticker's raw data and derive return-based columns."""
    raw_file = RAW_DATA_PATH / f"{ticker}.parquet"
    df = pd.read_parquet(raw_file)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing_cols = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{ticker}: missing expected columns {missing_cols}")
    df = df[OHLCV_COLUMNS]

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna()

    if df.empty:
        raise ValueError(f"{ticker}: no rows left after cleaning")

    # return/log_return are NaN on the first row by construction
    df["return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility_20"] = df["log_return"].rolling(window=20).std()
    df["ticker"] = ticker

    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DATA_PATH / f"{ticker}.parquet")

    return df


def process_all_raw() -> None:
    """Process every raw ticker file, logging a summary of successes/failures."""
    raw_files = sorted(RAW_DATA_PATH.glob("*.parquet"))
    succeeded, failed = 0, 0

    for raw_file in raw_files:
        ticker = raw_file.stem
        try:
            process_ticker(ticker)
            succeeded += 1
        except Exception as e:
            logger.error(f"{ticker}: failed to process ({e})")
            failed += 1

    logger.info(f"Done: {succeeded} succeeded, {failed} failed, {len(raw_files)} total")


if __name__ == "__main__":
    process_all_raw()