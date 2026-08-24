# This file cleans and processes the raw market data downloaded from Yahoo Finance. It reads the raw data files, performs necessary cleaning and transformations, and saves the processed data for further analysis.
import pandas as pd
from pathlib import Path

#file path for the processed data
PROCESSED_DATA_PATH = Path("data/processed")
