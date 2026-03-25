"""
Japan equity universe management.
Handles ticker lists, sector mappings, and universe filtering.
Multiple index options: TOPIX Core 30, Nikkei 225, TOPIX 500, Small Cap.
"""
from __future__ import annotations

import yaml
from pathlib import Path
from dataclasses import dataclass


# ── INDEX DEFINITIONS ─────────────────────────────────────────

# TOPIX Core 30 — mega-cap blue chips
TOPIX_CORE30 = [
    "7203.T", "6758.T", "8306.T", "9984.T", "8035.T",
    "6861.T", "6501.T", "9432.T", "7267.T", "4502.T",
    "6902.T", "8316.T", "9433.T", "6301.T", "4063.T",
    "8058.T", "8411.T", "9983.T", "6762.T", "7011.T",
    "6954.T", "8766.T", "4503.T", "2914.T", "3382.T",
    "8801.T", "6971.T", "5401.T", "9022.T", "9020.T",
]

# Nikkei 225 representative subset (120 constituents)
NIKKEI225 = [
    # Automobiles
    "7203.T", "7267.T", "7269.T", "7261.T", "7211.T", "7201.T", "7202.T",
    # Electronics & Tech
    "6758.T", "6861.T", "6902.T", "6954.T", "6971.T", "6762.T", "6752.T",
    "6503.T", "6501.T", "6702.T", "6701.T", "6753.T", "6645.T",
    # Semiconductors
    "8035.T", "6723.T", "6857.T", "7735.T", "6146.T", "6920.T",
    # Financials & Insurance
    "8306.T", "8316.T", "8411.T", "8766.T", "8725.T", "8591.T",
    "8604.T", "8601.T", "8750.T", "8795.T", "8630.T",
    # Trading Houses
    "8058.T", "8031.T", "8001.T", "8002.T", "8053.T",
    # Pharma & Healthcare
    "4502.T", "4503.T", "4519.T", "4568.T", "4523.T", "4507.T",
    "4578.T", "4506.T", "4151.T",
    # Telecom & Internet
    "9432.T", "9433.T", "9434.T", "4689.T", "4755.T", "3659.T",
    # Retail & Consumer
    "9983.T", "3382.T", "8267.T", "2914.T", "2802.T", "2801.T",
    "2503.T", "2502.T", "2269.T", "7453.T",
    # Real Estate & Construction
    "8801.T", "8802.T", "1925.T", "1928.T", "1801.T", "1802.T",
    "1803.T", "8830.T", "3289.T",
    # Industrial & Machinery
    "6301.T", "6305.T", "7011.T", "7013.T", "6367.T", "6302.T",
    "6471.T", "6473.T", "7004.T", "6361.T",
    # Materials & Chemicals
    "4063.T", "4188.T", "4005.T", "3407.T", "5401.T", "5411.T",
    "5406.T", "3405.T", "4042.T", "4043.T", "4021.T", "3401.T",
    # Glass & Ceramics
    "5332.T", "5333.T", "5201.T",
    # Paper & Pulp
    "3861.T", "3863.T",
    # Mining & Oil
    "1605.T", "5020.T", "5019.T",
    # Conglomerates & Services
    "9984.T", "6098.T", "4661.T", "9020.T", "9022.T", "2413.T",
    "4324.T", "6178.T", "9602.T",
    # Utilities
    "9501.T", "9502.T", "9503.T", "9531.T",
    # Shipping & Transport
    "9101.T", "9104.T", "9107.T", "9001.T", "9005.T", "9007.T",
    "9064.T",
    # Food & Beverage
    "2871.T", "2282.T",
    # Textiles
    "3402.T",
    # Rubber
    "5108.T",
    # Non-Ferrous Metals
    "5713.T", "5711.T", "5706.T", "5714.T",
    # Precision Instruments
    "7731.T", "7733.T", "7741.T", "7751.T", "7762.T",
]

# TOPIX 500 — broader large/mid-cap (adds ~60 more mid-caps)
TOPIX500 = list(set(NIKKEI225 + [
    # Additional mid-caps not in Nikkei 225
    "2801.T", "2871.T", "3086.T", "3099.T", "3349.T", "3626.T",
    "4021.T", "4043.T", "4183.T", "4208.T", "4307.T", "4452.T",
    "4507.T", "4543.T", "4578.T", "4631.T", "4684.T", "4704.T",
    "4901.T", "4911.T", "5101.T", "5202.T", "5233.T", "5301.T",
    "5631.T", "5703.T", "5707.T", "5801.T", "5802.T", "5803.T",
    "6103.T", "6113.T", "6201.T", "6273.T", "6326.T", "6370.T",
    "6479.T", "6504.T", "6506.T", "6586.T", "6592.T", "6594.T",
    "6674.T", "6724.T", "6752.T", "6753.T", "6755.T", "6770.T",
    "6841.T", "6856.T", "6869.T", "6976.T", "6981.T", "7003.T",
    "7012.T", "7014.T", "7205.T", "7231.T", "7259.T", "7270.T",
    "7272.T", "7282.T", "7309.T", "7741.T", "7751.T", "7762.T",
    "7832.T", "7911.T", "7912.T", "7951.T", "8015.T", "8028.T",
    "8053.T", "8252.T", "8253.T", "8267.T", "8303.T", "8304.T",
    "8308.T", "8309.T", "8331.T", "8354.T", "8355.T", "8377.T",
    "8585.T", "8628.T", "8697.T", "8713.T", "8729.T", "8768.T",
    "9005.T", "9007.T", "9008.T", "9009.T", "9021.T", "9024.T",
    "9042.T", "9064.T", "9201.T", "9202.T", "9301.T", "9602.T",
    "9613.T", "9681.T", "9735.T", "9766.T",
]))

# Japan Small Cap — companies between ~5B-50B JPY market cap
# These are prime territory for leveraged small value strategy
JAPAN_SMALL_CAP = [
    # Small-cap value / industrial
    "1414.T", "1417.T", "1518.T", "1721.T", "1726.T",
    "1813.T", "1866.T", "1870.T", "1878.T", "1882.T",
    "2127.T", "2154.T", "2175.T", "2301.T", "2379.T",
    "2412.T", "2427.T", "2462.T", "2593.T", "2607.T",
    "2695.T", "2726.T", "2734.T", "2749.T", "2767.T",
    "3034.T", "3076.T", "3091.T", "3097.T", "3105.T",
    "3167.T", "3254.T", "3291.T", "3371.T", "3387.T",
    "3434.T", "3436.T", "3465.T", "3539.T", "3591.T",
    "3597.T", "3656.T", "3697.T", "3762.T", "3769.T",
    "3844.T", "3923.T", "3941.T", "3964.T", "4044.T",
    "4080.T", "4206.T", "4245.T", "4290.T", "4312.T",
    "4369.T", "4465.T", "4544.T", "4641.T", "4658.T",
    "4718.T", "4767.T", "4792.T", "4812.T", "4927.T",
    "5214.T", "5269.T", "5288.T", "5363.T", "5451.T",
    "5602.T", "5695.T", "5741.T", "5901.T", "5929.T",
    "5949.T", "5981.T", "6044.T", "6058.T", "6140.T",
    "6208.T", "6245.T", "6298.T", "6340.T", "6406.T",
    "6455.T", "6507.T", "6584.T", "6640.T", "6651.T",
    "6727.T", "6747.T", "6785.T", "6820.T", "6866.T",
    "7148.T", "7164.T", "7181.T", "7218.T", "7230.T",
    "7238.T", "7278.T", "7313.T", "7419.T", "7480.T",
    "7532.T", "7552.T", "7575.T", "7600.T", "7745.T",
    "7839.T", "7867.T", "7917.T", "7936.T", "7979.T",
    "8014.T", "8020.T", "8029.T", "8088.T", "8098.T",
    "8113.T", "8227.T", "8279.T", "8363.T", "8385.T",
    "8511.T", "8524.T", "8566.T", "8593.T", "8609.T",
    "8698.T", "8793.T", "9003.T", "9035.T", "9052.T",
    "9065.T", "9076.T", "9104.T", "9302.T", "9386.T",
    "9506.T", "9507.T", "9508.T", "9603.T", "9678.T",
    "9699.T", "9715.T", "9743.T", "9793.T", "9831.T",
]


# Index registry — maps display name to (tickers, benchmark)
INDEX_REGISTRY = {
    "TOPIX Core 30": (TOPIX_CORE30, "^N225"),
    "Nikkei 225": (NIKKEI225, "^N225"),
    "TOPIX 500": (TOPIX500, "^N225"),
    "Japan Small Cap": (JAPAN_SMALL_CAP, "^N225"),
    "All Japan": (list(set(TOPIX500 + JAPAN_SMALL_CAP)), "^N225"),
}


@dataclass
class UniverseConfig:
    tickers: list[str]
    benchmark: str
    min_market_cap_jpy: float
    min_avg_volume: int
    exclude_sectors: list[str]
    price_history_years: int
    cache_dir: str
    index_name: str = "TOPIX 500"


def load_universe(
    config_path: str = "config/universe.yaml",
    index_override: str | None = None,
) -> UniverseConfig:
    """Load universe configuration from YAML, with optional index override."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Universe config not found: {config_path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Determine which index to use
    index_name = index_override or cfg.get("universe", "TOPIX 500")

    if index_name in INDEX_REGISTRY:
        tickers, benchmark = INDEX_REGISTRY[index_name]
        tickers = list(tickers)  # copy
    else:
        tickers = cfg.get("watchlist", [])
        benchmark = "^N225"

    # Merge watchlist extras
    watchlist = cfg.get("watchlist", [])
    for t in watchlist:
        if t not in tickers:
            tickers.append(t)

    filters = cfg.get("filters", {})
    data = cfg.get("data", {})

    return UniverseConfig(
        tickers=tickers,
        benchmark=data.get("benchmark", benchmark),
        min_market_cap_jpy=filters.get("min_market_cap_jpy", 50_000_000_000),
        min_avg_volume=filters.get("min_avg_volume", 100_000),
        exclude_sectors=filters.get("exclude_sectors", []),
        price_history_years=data.get("price_history_years", 5),
        cache_dir=data.get("cache_dir", "data/cache"),
        index_name=index_name,
    )
