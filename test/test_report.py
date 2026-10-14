"""Tests for the ticker universe, the terminal report and the
download-if-missing helper. Reports are rendered to a string; nothing is
downloaded."""
import io

import numpy as np
import pandas as pd
import pytest
from rich.console import Console

from data import downloader
from data.universe import BASKETS, UNIVERSE, asset_info, basket_tickers, sector_counts
from engine.result import BacktestResult
from performance import report


def render(renderable) -> str:
    console = Console(file=io.StringIO(), width=200, color_system=None)
    console.print(renderable)
    return console.file.getvalue()


def result(values, start="2023-12-29") -> BacktestResult:
    index = pd.bdate_range(start, periods=len(values))
    return BacktestResult(equity_curve=pd.Series(values, index=index, dtype=float), trades=[])


# ------------------------------------------------------------------ universe


def test_every_basket_ticker_is_in_the_universe():
    for name, (_, tickers) in BASKETS.items():
        assert set(tickers) <= set(UNIVERSE), name
        assert len(tickers) == len(set(tickers)), f"{name} lists a ticker twice"


def test_diversified_basket_has_one_ticker_per_sector():
    counts = sector_counts(basket_tickers("diversified"))
    assert len(counts) == 11
    assert set(counts.values()) == {1}


def test_unknown_ticker_is_unclassified():
    assert asset_info("zzzz").sector == "Unclassified"
    assert asset_info("aapl").name == "Apple"


def test_unknown_basket_raises():
    with pytest.raises(ValueError, match="unknown basket"):
        basket_tickers("nope")


def test_sector_counts_largest_first():
    assert list(sector_counts(["AAPL", "XOM", "MSFT"]).items()) == [("Technology", 2), ("Energy", 1)]


# ------------------------------------------------------------------- report


def test_annual_returns_split_by_calendar_year():
    # 100 on the last day of 2023, 110 at the end of 2024, 99 later in 2025
    equity = pd.Series(
        [100.0, 110.0, 99.0],
        index=pd.to_datetime(["2023-12-29", "2024-12-31", "2025-06-30"]),
    )
    returns = report.annual_returns(equity)
    # 2023 has only the starting day, so it is flat
    assert returns.loc[2023] == pytest.approx(0.0)
    assert returns.loc[2024] == pytest.approx(0.10)
    assert returns.loc[2025] == pytest.approx(-0.10)


def test_average_correlation_of_identical_moves_is_one():
    prices = pd.Series([1.0, 2.0, 1.5, 3.0, 2.5])
    closes = pd.DataFrame({"A": prices, "B": prices * 10})
    assert report.average_correlation(closes) == pytest.approx(1.0)


def test_average_correlation_of_opposite_moves_is_minus_one():
    returns = np.array([0.01, -0.02, 0.03, -0.01])
    closes = pd.DataFrame({
        "A": 100 * np.cumprod(1 + np.r_[0, returns]),
        "B": 100 * np.cumprod(1 - np.r_[0, returns]),
    })
    assert report.average_correlation(closes) == pytest.approx(-1.0)


def test_correlation_table_needs_two_tickers():
    assert report.correlation_table(pd.DataFrame({"A": [1.0, 2.0, 3.0]})) is None


def test_sector_table_warns_when_concentrated():
    assert "are Technology" in render(report.sector_table(["AAPL", "MSFT", "NVDA", "XOM"]))
    assert "Spread across 3 sectors" in render(report.sector_table(["AAPL", "XOM", "JNJ"]))


def test_comparison_and_trading_tables_list_every_strategy():
    results = {"steady": result([100, 101, 102, 103]), "choppy": result([100, 90, 105, 95])}
    for table in (report.comparison_table(results), report.trading_table(results)):
        text = render(table)
        assert "steady" in text and "choppy" in text
    assert "+3%" in render(report.comparison_table(results))


def test_takeaways_names_the_leaders():
    results = {"buy_and_hold": result([100, 101, 102, 103]), "worse": result([100, 90, 105, 95])}
    text = render(report.takeaways(results))
    assert "Best risk-adjusted:  buy_and_hold" in text
    assert "0 of 1 strategies beat buy-and-hold" in text


def test_ratio_formats_special_values():
    assert report.ratio(float("nan")).plain == "–"
    assert report.ratio(float("inf")).plain == "∞"
    assert report.ratio(1.234).plain == "1.23"


# --------------------------------------------------------------- downloader


def test_download_skips_tickers_already_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "PROCESSED_DATA_PATH", tmp_path)
    frame = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2024-01-02"]))
    frame.to_parquet(tmp_path / "ABC.parquet")

    def no_network(*args, **kwargs):
        raise AssertionError("should not download a cached ticker")

    monkeypatch.setattr(downloader, "fetch_market_data", no_network)
    outcome, data = downloader.download_and_process("abc")
    assert outcome == "cached"
    assert len(data) == 1


def test_download_reports_failure_instead_of_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "PROCESSED_DATA_PATH", tmp_path)

    def fail(*args, **kwargs):
        raise ValueError("No data returned for BAD")

    monkeypatch.setattr(downloader, "fetch_market_data", fail)
    outcome, data = downloader.download_and_process("BAD")
    assert outcome == "failed: No data returned for BAD"
    assert data is None
