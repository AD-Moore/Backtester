"""Smoke tests for the comparison chart: it writes a file and rejects bad input."""

import pandas as pd
import pytest

from performance.visualization import SERIES_COLORS, plot_equity_curves


def curve(values):
    """Build an equity curve over consecutive days from a list of values."""
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=index, dtype=float)


def test_chart_is_written_to_disk(tmp_path):
    curves = {"a": curve([100, 110, 105]), "b": curve([100, 95, 120])}
    # nested folder that doesn't exist yet, like results/ on a fresh clone
    path = plot_equity_curves(curves, "Test", tmp_path / "charts" / "out.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError, match="no equity curves"):
        plot_equity_curves({}, "Test", tmp_path / "out.png")


def test_rejects_more_curves_than_colors(tmp_path):
    curves = {str(i): curve([100, 101]) for i in range(len(SERIES_COLORS) + 1)}
    with pytest.raises(ValueError, match="at most"):
        plot_equity_curves(curves, "Test", tmp_path / "out.png")
