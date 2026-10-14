"""Charts for comparing backtest runs: equity curves on top, drawdowns below."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render straight to a file; no display/GUI needed
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from performance.metrics import drawdown_series

# Fixed order, chosen so neighbouring colors stay distinguishable for
# colorblind readers. Assigned by position, never cycled.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"


def _style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=TEXT_MUTED, length=0, labelsize=9)


def _format_dollars(value: float, _position) -> str:
    return f"${value / 1000:,.0f}K" if abs(value) >= 1000 else f"${value:,.0f}"


def plot_equity_curves(
    equity_curves: dict[str, pd.Series], title: str, path: str | Path
) -> Path:
    """Save a two-panel chart (equity, then drawdown) comparing each
    labelled equity curve. Returns the path written."""
    if not equity_curves:
        raise ValueError("no equity curves to plot")
    if len(equity_curves) > len(SERIES_COLORS):
        raise ValueError(f"can plot at most {len(SERIES_COLORS)} curves on one chart")

    fig, (equity_ax, drawdown_ax) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, height_ratios=[2.2, 1], facecolor=SURFACE
    )

    for color, (label, equity) in zip(SERIES_COLORS, equity_curves.items()):
        # final value in the legend, so a line is identifiable without color
        legend_label = f"{label}  ${equity.iloc[-1]:,.0f}"
        equity_ax.plot(equity.index, equity, color=color, linewidth=2, label=legend_label)
        drawdown_ax.plot(equity.index, drawdown_series(equity), color=color, linewidth=1.5)

    for ax in (equity_ax, drawdown_ax):
        _style_axis(ax)

    equity_ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=13, fontweight="bold", pad=14)
    equity_ax.set_ylabel("Portfolio value", color=TEXT_SECONDARY, fontsize=9)
    equity_ax.yaxis.set_major_formatter(FuncFormatter(_format_dollars))
    equity_ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper left")

    drawdown_ax.set_ylabel("Drawdown from peak", color=TEXT_SECONDARY, fontsize=9)
    drawdown_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    drawdown_ax.set_ylim(top=0)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
