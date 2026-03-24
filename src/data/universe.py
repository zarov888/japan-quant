"""
Japan equity universe management.
Handles ticker lists, sector mappings, and universe filtering.
"""
from __future__ import annotations

import yaml
from pathlib import Path
from dataclasses import dataclass

# TOPIX 500 core constituents (representative subset — expand as needed)
# Full list: https://www.jpx.co.jp/english/markets/indices/topix/
TOPIX_CORE = [
    # Automobiles & Transport
    "7203.T", "7267.T", "7269.T", "7261.T", "7211.T",
    # Electronics & Technology
    "6758.T", "6861.T", "6902.T", "6954.T", "6971.T",
    "6762.T", "6752.T", "6503.T", "6501.T", "6702.T",
    # Semiconductors & Equipment
    "8035.T", "6723.T", "6857.T", "7735.T", "6146.T",
    # Financials
    "8306.T", "8316.T", "8411.T", "8766.T", "8725.T",
    "8591.T", "8604.T", "8601.T",
    # Trading Houses (sogo shosha)
    "8058.T", "8031.T", "8001.T", "8002.T", "8053.T",
    # Pharma & Healthcare
    "4502.T", "4503.T", "4519.T", "4568.T", "4523.T",
    # Telecom & Internet
    "9432.T", "9433.T", "9434.T", "4689.T", "4755.T",
    # Retail & Consumer
    "9983.T", "3382.T", "8267.T", "2914.T", "2802.T",
    # Real Estate & Construction
    "8801.T", "8802.T", "1925.T", "1928.T", "1801.T",
    # Industrial & Machinery
    "6301.T", "6305.T", "7011.T", "7013.T", "6367.T",
    # Materials & Chemicals
    "4063.T", "4188.T", "4005.T", "3407.T", "5401.T",
    # Conglomerates & Other
    "9984.T", "6098.T", "4661.T", "9020.T", "9022.T",
    # Utilities
    "9501.T", "9502.T", "9503.T",
    # Shipping & Logistics
    "9101.T", "9104.T", "9107.T",
]


@dataclass
class UniverseConfig:
    tickers: list[str]
    benchmark: str
    min_market_cap_jpy: float
    min_avg_volume: int
    exclude_sectors: list[str]
    price_history_years: int
    cache_dir: str


def load_universe(config_path: str = "config/universe.yaml") -> UniverseConfig:
    """Load universe configuration from YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Universe config not found: {config_path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    universe_name = cfg.get("universe", "watchlist")
    if universe_name == "topix500":
        tickers = TOPIX_CORE.copy()
    elif universe_name == "watchlist":
        tickers = cfg.get("watchlist", [])
    else:
        tickers = cfg.get("watchlist", [])

    # Merge watchlist into universe if both specified
    watchlist = cfg.get("watchlist", [])
    for t in watchlist:
        if t not in tickers:
            tickers.append(t)

    filters = cfg.get("filters", {})
    data = cfg.get("data", {})

    return UniverseConfig(
        tickers=tickers,
        benchmark=data.get("benchmark", "^N225"),
        min_market_cap_jpy=filters.get("min_market_cap_jpy", 50_000_000_000),
        min_avg_volume=filters.get("min_avg_volume", 100_000),
        exclude_sectors=filters.get("exclude_sectors", []),
        price_history_years=data.get("price_history_years", 5),
        cache_dir=data.get("cache_dir", "data/cache"),
    )
