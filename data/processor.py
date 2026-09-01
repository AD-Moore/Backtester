import logging

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_ticker(ticker: str) -> pd.DataFrame:
    raw_file = RAW_DATA_PATH / f"{ticker}.parquet"
    df = pd.read_parquet(raw_file)

    # yfinance can return a MultiIndex column header; flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing_cols = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{ticker}: missing expected columns {missing_cols}")
    df = df[OHLCV_COLUMNS]

    # 20-23 data cleaning steps
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna()

    if df.empty:
        raise ValueError(f"{ticker}: no rows left after cleaning")

    # adds new columns
    df["return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["ticker"] = ticker
    # note: return/log_return are NaN on the first row by construction
    # (no prior close to compare to) -- this is expected, not a bug

    # checks if file exists and creates the directory if it doesn't
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DATA_PATH / f"{ticker}.parquet")

    return df


def process_all_raw() -> None:
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