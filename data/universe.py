"""The tickers the backtester knows about, and named baskets of them.

The original six tickers were all large US technology names, which made
every result a bet on one sector in one strong decade. The universe
below covers every sector of the S&P 500 plus assets that often move
differently from US stocks (bonds, gold, broad commodities, other
countries), so a portfolio can be built that isn't one trade in disguise.

Any ticker yfinance knows can still be passed with --tickers; ones not
listed here just show up with an "Unclassified" sector.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    name: str
    sector: str  # GICS sector for stocks and sector ETFs; asset class otherwise


UNIVERSE: dict[str, Asset] = {
    # Information Technology
    "AAPL": Asset("Apple", "Technology"),
    "MSFT": Asset("Microsoft", "Technology"),
    "NVDA": Asset("NVIDIA", "Technology"),
    "VGT": Asset("Vanguard Information Tech ETF", "Technology"),
    "XLK": Asset("Technology Select Sector SPDR", "Technology"),
    # Communication Services
    "GOOGL": Asset("Alphabet", "Communication"),
    "DIS": Asset("Walt Disney", "Communication"),
    "XLC": Asset("Communication Services SPDR", "Communication"),
    # Consumer Discretionary
    "AMZN": Asset("Amazon", "Consumer Disc."),
    "HD": Asset("Home Depot", "Consumer Disc."),
    "MCD": Asset("McDonald's", "Consumer Disc."),
    "XLY": Asset("Consumer Discretionary SPDR", "Consumer Disc."),
    # Consumer Staples
    "PG": Asset("Procter & Gamble", "Consumer Staples"),
    "KO": Asset("Coca-Cola", "Consumer Staples"),
    "WMT": Asset("Walmart", "Consumer Staples"),
    "COST": Asset("Costco", "Consumer Staples"),
    "XLP": Asset("Consumer Staples SPDR", "Consumer Staples"),
    # Health Care
    "JNJ": Asset("Johnson & Johnson", "Health Care"),
    "UNH": Asset("UnitedHealth", "Health Care"),
    "LLY": Asset("Eli Lilly", "Health Care"),
    "XLV": Asset("Health Care SPDR", "Health Care"),
    # Financials
    "JPM": Asset("JPMorgan Chase", "Financials"),
    "V": Asset("Visa", "Financials"),
    "BRK-B": Asset("Berkshire Hathaway", "Financials"),
    "XLF": Asset("Financial SPDR", "Financials"),
    # Energy
    "XOM": Asset("Exxon Mobil", "Energy"),
    "CVX": Asset("Chevron", "Energy"),
    "XLE": Asset("Energy SPDR", "Energy"),
    # Industrials
    "CAT": Asset("Caterpillar", "Industrials"),
    "HON": Asset("Honeywell", "Industrials"),
    "UNP": Asset("Union Pacific", "Industrials"),
    "XLI": Asset("Industrial SPDR", "Industrials"),
    # Utilities
    "NEE": Asset("NextEra Energy", "Utilities"),
    "DUK": Asset("Duke Energy", "Utilities"),
    "XLU": Asset("Utilities SPDR", "Utilities"),
    # Real Estate
    "PLD": Asset("Prologis", "Real Estate"),
    "O": Asset("Realty Income", "Real Estate"),
    "XLRE": Asset("Real Estate SPDR", "Real Estate"),
    "VNQ": Asset("Vanguard Real Estate ETF", "Real Estate"),
    # Materials
    "LIN": Asset("Linde", "Materials"),
    "SHW": Asset("Sherwin-Williams", "Materials"),
    "XLB": Asset("Materials SPDR", "Materials"),
    # Broad equity
    "VOO": Asset("Vanguard S&P 500 ETF", "US Equity"),
    "IWM": Asset("iShares Russell 2000 (small caps)", "US Equity"),
    "VEA": Asset("Vanguard Developed Markets ETF", "Intl Equity"),
    "VWO": Asset("Vanguard Emerging Markets ETF", "Intl Equity"),
    # Bonds
    "TLT": Asset("iShares 20+ Year Treasury", "Bonds"),
    "IEF": Asset("iShares 7-10 Year Treasury", "Bonds"),
    "AGG": Asset("iShares Core US Aggregate Bond", "Bonds"),
    "TIP": Asset("iShares TIPS (inflation-linked)", "Bonds"),
    # Commodities
    "GLD": Asset("SPDR Gold Shares", "Commodities"),
    "DBC": Asset("Invesco DB Commodity Index", "Commodities"),
}

BASKETS: dict[str, tuple[str, list[str]]] = {
    "diversified": (
        "One large company from each of the 11 S&P 500 sectors",
        ["AAPL", "GOOGL", "AMZN", "PG", "JNJ", "JPM", "XOM", "CAT", "NEE", "PLD", "LIN"],
    ),
    "sectors": (
        "The 11 Select Sector SPDR ETFs: the whole S&P 500, one fund per sector",
        ["XLK", "XLC", "XLY", "XLP", "XLV", "XLF", "XLE", "XLI", "XLU", "XLRE", "XLB"],
    ),
    "multi_asset": (
        "US and international stocks, real estate, bonds, gold and commodities",
        ["VOO", "IWM", "VEA", "VWO", "VNQ", "TLT", "IEF", "TIP", "GLD", "DBC"],
    ),
    "all_weather": (
        "A small stocks/bonds/gold/commodities mix that doesn't need one economy",
        ["VOO", "TLT", "IEF", "GLD", "DBC"],
    ),
    "defensive": (
        "Staples, health care and utilities: steady demand in a downturn",
        ["PG", "KO", "WMT", "JNJ", "UNH", "NEE", "DUK", "XLP", "XLU"],
    ),
    "cyclical": (
        "Energy, industrials, materials, financials and discretionary spending",
        ["XOM", "CVX", "CAT", "UNP", "LIN", "JPM", "HD", "MCD"],
    ),
    "tech": (
        "The original six: large US technology names plus the S&P 500",
        ["AAPL", "MSFT", "GOOGL", "NVDA", "VGT", "VOO"],
    ),
    "stocks_vs_bonds": (
        "The S&P 500 against long Treasuries; a natural pair for pairs trading",
        ["VOO", "TLT"],
    ),
}

DEFAULT_BASKET = "diversified"


def asset_info(ticker: str) -> Asset:
    """The ticker's name and sector, or a placeholder for one not listed."""
    return UNIVERSE.get(ticker.upper(), Asset(ticker.upper(), "Unclassified"))


def basket_tickers(name: str) -> list[str]:
    if name not in BASKETS:
        raise ValueError(f"unknown basket {name!r}; choose from {', '.join(BASKETS)}")
    return list(BASKETS[name][1])


def sector_counts(tickers: list[str]) -> dict[str, int]:
    """How many of the tickers fall in each sector, largest first."""
    counts: dict[str, int] = {}
    for ticker in tickers:
        sector = asset_info(ticker).sector
        counts[sector] = counts.get(sector, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))
