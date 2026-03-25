"""
Market data fetcher for Japanese equities.
Pulls price history and fundamentals via yfinance with local caching.
"""

import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.data.universe import UniverseConfig

logger = logging.getLogger(__name__)

CACHE_EXPIRY_HOURS = 12


def _to_decimal(val):
    """Normalize yfinance yield/pct fields that sometimes come as whole numbers (2.92 = 2.92%)."""
    if val is None:
        return None
    # If > 1, it's already a percentage — convert to decimal
    return val / 100.0 if val > 1.0 else val


def _cache_path(cache_dir: str, ticker: str, kind: str) -> Path:
    """Generate cache file path for a ticker."""
    safe_name = ticker.replace(".", "_")
    return Path(cache_dir) / f"{safe_name}_{kind}.parquet"


def _cache_valid(path: Path, max_age_hours: int = CACHE_EXPIRY_HOURS) -> bool:
    """Check if cached file exists and is fresh enough."""
    if not path.exists():
        return False
    age = datetime.now().timestamp() - path.stat().st_mtime
    return age < max_age_hours * 3600


def fetch_price_history(
    ticker: str,
    years: int = 5,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """Fetch OHLCV price history for a single ticker."""
    cache = _cache_path(cache_dir, ticker, "prices")
    if _cache_valid(cache):
        logger.debug(f"Cache hit: {ticker} prices")
        return pd.read_parquet(cache)

    logger.info(f"Fetching price history: {ticker}")
    end = datetime.now()
    start = end - timedelta(days=years * 365)

    try:
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            logger.warning(f"No price data for {ticker}")
            return pd.DataFrame()

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
        return df

    except Exception as e:
        logger.error(f"Failed to fetch prices for {ticker}: {e}")
        return pd.DataFrame()


def fetch_fundamentals(
    ticker: str,
    cache_dir: str = "data/cache",
) -> dict:
    """Fetch fundamental data for a single ticker."""
    cache = _cache_path(cache_dir, ticker, "fundamentals")
    json_cache = cache.with_suffix(".json")

    if _cache_valid(json_cache):
        logger.debug(f"Cache hit: {ticker} fundamentals")
        with open(json_cache) as f:
            return json.load(f)

    logger.info(f"Fetching fundamentals: {ticker}")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        fundamentals = {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency", "JPY"),
            # Value metrics
            "pe_trailing": info.get("trailingPE"),
            "pe_forward": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            "dividend_yield": _to_decimal(info.get("dividendYield")),
            # Quality metrics
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "operating_margin": info.get("operatingMargins"),
            "profit_margin": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            # Financial strength
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "total_cash": info.get("totalCash"),
            "total_debt": info.get("totalDebt"),
            "free_cashflow": info.get("freeCashflow"),
            "enterprise_value": info.get("enterpriseValue"),
            # Verdad leveraged value metrics
            "ebitda": info.get("ebitda"),
            "total_revenue": info.get("totalRevenue"),
            "total_assets": info.get("totalAssets"),
            "gross_profits": info.get("grossProfits"),
            "long_term_debt": info.get("longTermDebt"),
            "short_ratio": info.get("shortRatio"),
            "short_pct_float": info.get("shortPercentOfFloat"),
            "net_income": info.get("netIncomeToCommon"),
            "operating_cashflow": info.get("operatingCashflow"),
            # Price data
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "fifty_day_avg": info.get("fiftyDayAverage"),
            "two_hundred_day_avg": info.get("twoHundredDayAverage"),
            "avg_volume": info.get("averageVolume"),
            "beta": info.get("beta"),
            # Metadata
            "fetched_at": datetime.now().isoformat(),
        }

        # Derived ratios — Verdad framework
        ev = fundamentals["enterprise_value"]
        mcap = fundamentals["market_cap"]
        lt_debt = fundamentals["long_term_debt"]
        ebitda = fundamentals["ebitda"]
        assets = fundamentals["total_assets"]
        revenue = fundamentals["total_revenue"]
        gross = fundamentals["gross_profits"]

        # LT Debt / EV (leverage intensity)
        fundamentals["lt_debt_to_ev"] = (
            lt_debt / ev if lt_debt and ev and ev > 0 else None
        )
        # Net Debt / EBITDA
        net_debt = (fundamentals["total_debt"] or 0) - (fundamentals["total_cash"] or 0)
        fundamentals["net_debt_to_ebitda"] = (
            net_debt / ebitda if ebitda and ebitda > 0 else None
        )
        # EBITDA / EV (earnings yield on enterprise)
        fundamentals["ebitda_to_ev"] = (
            ebitda / ev if ebitda and ev and ev > 0 else None
        )
        # Gross Profit / Assets (capital efficiency)
        fundamentals["gross_profit_to_assets"] = (
            gross / assets if gross and assets and assets > 0 else None
        )
        # LT Debt / Assets
        fundamentals["lt_debt_to_assets"] = (
            lt_debt / assets if lt_debt and assets and assets > 0 else None
        )
        # Asset Turnover (Revenue / Assets)
        fundamentals["asset_turnover"] = (
            revenue / assets if revenue and assets and assets > 0 else None
        )
        # Cash-to-market-cap ratio (Japan-specific)
        fundamentals["cash_to_mcap"] = (
            fundamentals["total_cash"] / mcap
            if fundamentals["total_cash"] and mcap and mcap > 0 else None
        )

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with open(json_cache, "w") as f:
            json.dump(fundamentals, f, indent=2, default=str)

        return fundamentals

    except Exception as e:
        logger.error(f"Failed to fetch fundamentals for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


def fetch_universe(config: UniverseConfig) -> tuple[pd.DataFrame, list[dict]]:
    """
    Fetch all data for the configured universe.
    Returns (price_df_dict, fundamentals_list).
    """
    all_fundamentals = []
    failed = []

    for i, ticker in enumerate(config.tickers):
        logger.info(f"[{i+1}/{len(config.tickers)}] {ticker}")
        fund = fetch_fundamentals(ticker, config.cache_dir)

        if "error" in fund:
            failed.append(ticker)
            continue

        # Apply pre-filters
        mcap = fund.get("market_cap")
        if mcap and mcap < config.min_market_cap_jpy:
            logger.debug(f"Skipping {ticker}: market cap {mcap:,.0f} below threshold")
            continue

        vol = fund.get("avg_volume")
        if vol and vol < config.min_avg_volume:
            logger.debug(f"Skipping {ticker}: avg volume {vol:,.0f} below threshold")
            continue

        sector = fund.get("sector", "")
        if sector in config.exclude_sectors:
            logger.debug(f"Skipping {ticker}: excluded sector {sector}")
            continue

        all_fundamentals.append(fund)

        # Rate limiting — be nice to Yahoo
        if (i + 1) % 10 == 0:
            time.sleep(1)

    if failed:
        logger.warning(f"Failed to fetch {len(failed)} tickers: {failed}")

    logger.info(
        f"Universe loaded: {len(all_fundamentals)} stocks "
        f"({len(failed)} failed, "
        f"{len(config.tickers) - len(all_fundamentals) - len(failed)} filtered)"
    )

    return all_fundamentals
