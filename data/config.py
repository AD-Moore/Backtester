"""Shared paths and constants for the data pipeline.

Centralizing these avoids the three data modules drifting out of sync
if a path or column set ever changes.
"""
from pathlib import Path

# Anchored to this file rather than the working directory, so the pipeline
# reads and writes the same folders no matter where it is run from.
DATA_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = DATA_DIR / "raw"
PROCESSED_DATA_PATH = DATA_DIR / "processed"

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
