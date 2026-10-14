"""Tests for the data pipeline (processor and loader), using temporary
folders in place of data/raw and data/processed. Nothing is downloaded."""

import numpy as np
import pandas as pd
import pytest

from data import config, loader, processor


@pytest.fixture
def data_dirs(tmp_path, monkeypatch):
    """Point the processor and loader at empty temporary folders."""
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    monkeypatch.setattr(processor, "RAW_DATA_PATH", raw)
    monkeypatch.setattr(processor, "PROCESSED_DATA_PATH", processed)
    monkeypatch.setattr(loader, "PROCESSED_DATA_PATH", processed)
    return raw, processed


def raw_frame(closes, index=None):
    """Raw OHLCV data where every price column equals the given closes."""
    if index is None:
        index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    data = {column: closes for column in ["Open", "High", "Low", "Close"]}
    return pd.DataFrame({**data, "Volume": [1000] * len(closes)}, index=index, dtype=float)


def write_raw(raw_dir, ticker, df):
    df.to_parquet(raw_dir / f"{ticker}.parquet")


def test_data_paths_do_not_depend_on_the_working_directory():
    assert config.RAW_DATA_PATH.is_absolute()
    assert config.RAW_DATA_PATH.parent == config.PROCESSED_DATA_PATH.parent


def test_processing_adds_return_columns(data_dirs):
    raw, processed = data_dirs
    write_raw(raw, "AAPL", raw_frame([100.0, 110.0, 99.0]))

    df = processor.process_ticker("AAPL")

    assert np.isnan(df["return"].iloc[0])  # no previous day to compare with
    assert df["return"].iloc[1:].tolist() == pytest.approx([0.10, -0.10])
    assert df["log_return"].iloc[1] == pytest.approx(np.log(1.10))
    assert (df["ticker"] == "AAPL").all()
    assert (processed / "AAPL.parquet").exists()


def test_rolling_volatility_needs_twenty_returns(data_dirs):
    raw, _ = data_dirs
    write_raw(raw, "AAPL", raw_frame(list(np.linspace(100, 130, 25))))

    volatility = processor.process_ticker("AAPL")["volatility_20"]

    # the first row has no return, so the 20th return lands on row 20
    assert volatility.iloc[:20].isna().all()
    assert volatility.iloc[20:].notna().all()


def test_rows_are_sorted_deduplicated_and_cleaned(data_dirs):
    raw, _ = data_dirs
    index = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-04"])
    df = raw_frame([103.0, 101.0, 999.0, 102.0, 104.0], index=index)
    df.loc[pd.Timestamp("2024-01-04"), "Close"] = np.nan  # an incomplete row
    write_raw(raw, "AAPL", df)

    cleaned = processor.process_ticker("AAPL")

    # sorted by date, the later duplicate (102) kept, the NaN row dropped
    assert cleaned.index.tolist() == list(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    assert cleaned["Close"].tolist() == [101.0, 102.0, 103.0]


def test_yfinance_style_two_level_columns_are_flattened(data_dirs):
    raw, _ = data_dirs
    df = raw_frame([100.0, 110.0])
    df.columns = pd.MultiIndex.from_product([df.columns, ["AAPL"]])
    write_raw(raw, "AAPL", df)

    assert "Close" in processor.process_ticker("AAPL").columns


def test_missing_column_is_reported_by_name(data_dirs):
    raw, _ = data_dirs
    write_raw(raw, "AAPL", raw_frame([100.0, 110.0]).drop(columns="Volume"))

    with pytest.raises(ValueError, match="AAPL.*Volume"):
        processor.process_ticker("AAPL")


def test_ticker_with_no_usable_rows_raises(data_dirs):
    raw, _ = data_dirs
    write_raw(raw, "AAPL", raw_frame([np.nan, np.nan]))

    with pytest.raises(ValueError, match="no rows left"):
        processor.process_ticker("AAPL")


def test_one_bad_file_does_not_stop_the_rest(data_dirs):
    raw, processed = data_dirs
    write_raw(raw, "AAPL", raw_frame([100.0, 110.0]))
    write_raw(raw, "BAD", raw_frame([np.nan, np.nan]))

    processor.process_all_raw()

    assert (processed / "AAPL.parquet").exists()
    assert not (processed / "BAD.parquet").exists()


def test_load_ticker_is_case_insensitive(data_dirs):
    raw, _ = data_dirs
    write_raw(raw, "AAPL", raw_frame([100.0, 110.0]))
    processor.process_all_raw()

    assert loader.load_ticker("aapl")["Close"].tolist() == [100.0, 110.0]


def test_loading_an_unprocessed_ticker_raises(data_dirs):
    with pytest.raises(FileNotFoundError, match="TSLA"):
        loader.load_ticker("TSLA")


def test_price_panel_has_one_column_per_ticker(data_dirs):
    raw, _ = data_dirs
    write_raw(raw, "AAPL", raw_frame([100.0, 110.0]))
    write_raw(raw, "MSFT", raw_frame([200.0, 210.0]))
    processor.process_all_raw()

    panel = loader.load_price_panel(field="Close")

    assert list(panel.columns) == ["AAPL", "MSFT"]
    assert panel["MSFT"].tolist() == [200.0, 210.0]
    assert set(loader.load_tickers(["aapl", "msft"])) == {"AAPL", "MSFT"}


def test_price_panel_can_be_limited_to_chosen_tickers_in_the_order_given(data_dirs):
    raw, _ = data_dirs
    for ticker in ["AAPL", "MSFT", "NVDA"]:
        write_raw(raw, ticker, raw_frame([100.0, 110.0]))
    processor.process_all_raw()

    panel = loader.load_price_panel("Open", ["nvda", "aapl"])

    assert list(panel.columns) == ["NVDA", "AAPL"]


def test_price_panel_keeps_only_dates_every_ticker_has(data_dirs):
    raw, _ = data_dirs
    write_raw(raw, "OLD", raw_frame([100.0, 110.0, 120.0]))  # Jan 1-3
    late_start = pd.date_range("2024-01-02", periods=2, freq="D")
    write_raw(raw, "NEW", raw_frame([50.0, 55.0], index=late_start))  # Jan 2-3
    processor.process_all_raw()

    panel = loader.load_price_panel("Close")

    assert panel.index.tolist() == list(late_start)
    assert panel["OLD"].tolist() == [110.0, 120.0]
