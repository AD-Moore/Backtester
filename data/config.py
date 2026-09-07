"""Shared paths and constants for the data pipeline.

Centralizing these avoids the three data modules drifting out of sync
if a path or column set ever changes.
"""
from pathlib import Path

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]