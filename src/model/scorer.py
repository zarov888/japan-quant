"""
Value scoring model for Japanese equities.
Scores each stock across value, quality, financial strength, and momentum factors.
"""
from __future__ import annotations

import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScoreBreakdown:
    ticker: str
    name: str
    sector: str
    composite: float
    value_score: float
    quality_score: float
    strength_score: float
    momentum_score: float
    factors: dict


def load_scoring_config(path: str = "config/scoring.yaml") -> dict:
    """Load scoring model configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def _normalize(value: float | None, low: float, high: float, invert: bool = False) -> float:
    """
    Normalize a value to [0, 1] range.
    If invert=True, lower raw values produce higher scores (e.g., P/E).
    Returns 0.5 if value is None (neutral).
    """
    if value is None or np.isnan(value):
        return 0.5

    value = max(low, min(high, value))
    score = (value - low) / (high - low) if high != low else 0.5

    return 1.0 - score if invert else score


def score_value(fund: dict, cfg: dict) -> tuple[float, dict]:
    """Score value factors. Lower P/E, P/B, EV/EBITDA = better."""
    weights = cfg["value"]["weights"]
    factors = {}

    # P/B — king metric for Japan
    pb = fund.get("pb_ratio")
    factors["price_to_book"] = _normalize(pb, 0.3, 3.0, invert=True)

    # P/E
    pe = fund.get("pe_trailing") or fund.get("pe_forward")
    factors["price_to_earnings"] = _normalize(pe, 3.0, 30.0, invert=True)

    # Price to FCF
    fcf = fund.get("free_cashflow")
    mcap = fund.get("market_cap")
    if fcf and mcap and fcf > 0:
        p_fcf = mcap / fcf
        factors["price_to_fcf"] = _normalize(p_fcf, 3.0, 30.0, invert=True)
    else:
        factors["price_to_fcf"] = 0.3  # penalize missing/negative FCF

    # EV/EBITDA
    ev_ebitda = fund.get("ev_to_ebitda")
    factors["ev_to_ebitda"] = _normalize(ev_ebitda, 2.0, 20.0, invert=True)

    # Dividend yield
    div_yield = fund.get("dividend_yield")
    factors["dividend_yield"] = _normalize(div_yield, 0.0, 0.06, invert=False)

    # Cash to market cap (Japan-specific)
    cash_mcap = fund.get("cash_to_mcap")
    factors["cash_to_market_cap"] = _normalize(cash_mcap, 0.0, 0.50, invert=False)

    score = sum(factors[k] * weights[k] for k in weights if k in factors)
    return score, factors


def score_quality(fund: dict, cfg: dict) -> tuple[float, dict]:
    """Score quality factors. Higher ROE, margins = better."""
    weights = cfg["quality"]["weights"]
    factors = {}

    factors["roe"] = _normalize(fund.get("roe"), 0.0, 0.25, invert=False)
    factors["operating_margin"] = _normalize(fund.get("operating_margin"), 0.0, 0.25, invert=False)

    # Earnings stability — approximate from growth consistency
    eg = fund.get("earnings_growth")
    if eg is not None:
        # Stable positive growth is best; wild swings are bad
        factors["earnings_stability"] = _normalize(abs(eg), 0.0, 1.0, invert=True) if eg < 0 else _normalize(eg, 0.0, 0.50, invert=False)
    else:
        factors["earnings_stability"] = 0.5

    factors["revenue_growth_3y"] = _normalize(fund.get("revenue_growth"), -0.10, 0.30, invert=False)

    score = sum(factors[k] * weights[k] for k in weights if k in factors)
    return score, factors


def score_financial_strength(fund: dict, cfg: dict) -> tuple[float, dict]:
    """Score balance sheet strength. Low debt, high coverage = better."""
    weights = cfg["financial_strength"]["weights"]
    factors = {}

    # D/E — yfinance reports as percentage (e.g., 50 = 0.5x)
    de = fund.get("debt_to_equity")
    if de is not None:
        de_ratio = de / 100.0 if de > 5 else de  # normalize yfinance quirk
    else:
        de_ratio = None
    factors["debt_to_equity"] = _normalize(de_ratio, 0.0, 2.0, invert=True)

    factors["current_ratio"] = _normalize(fund.get("current_ratio"), 0.5, 3.0, invert=False)

    # Interest coverage — approximate
    factors["interest_coverage"] = 0.5  # placeholder until we get income statement data

    # Net cash positive (binary)
    cash = fund.get("total_cash") or 0
    debt = fund.get("total_debt") or 0
    factors["net_cash_positive"] = 1.0 if cash > debt else 0.2

    score = sum(factors[k] * weights[k] for k in weights if k in factors)
    return score, factors


def score_momentum(fund: dict, cfg: dict) -> tuple[float, dict]:
    """Score momentum factors. Avoid falling knives."""
    weights = cfg["momentum"]["weights"]
    factors = {}

    price = fund.get("current_price")
    high_52w = fund.get("fifty_two_week_high")
    low_52w = fund.get("fifty_two_week_low")
    sma50 = fund.get("fifty_day_avg")
    sma200 = fund.get("two_hundred_day_avg")

    # Relative strength (price position in 52-week range)
    if price and high_52w and low_52w and high_52w != low_52w:
        rel_pos = (price - low_52w) / (high_52w - low_52w)
        factors["relative_strength_6m"] = _normalize(rel_pos, 0.0, 1.0)
    else:
        factors["relative_strength_6m"] = 0.5

    # Distance from 52-week high
    if price and high_52w and high_52w > 0:
        pct_from_high = price / high_52w
        factors["price_vs_52w_high"] = _normalize(pct_from_high, 0.5, 1.0)
    else:
        factors["price_vs_52w_high"] = 0.5

    # SMA 50 vs 200 (golden cross signal)
    if sma50 and sma200 and sma200 > 0:
        sma_ratio = sma50 / sma200
        factors["sma_50_vs_200"] = _normalize(sma_ratio, 0.9, 1.1)
    else:
        factors["sma_50_vs_200"] = 0.5

    score = sum(factors[k] * weights[k] for k in weights if k in factors)
    return score, factors


def score_stock(fund: dict, cfg: dict) -> ScoreBreakdown:
    """Compute composite score for a single stock."""
    gw = cfg["group_weights"]

    v_score, v_factors = score_value(fund, cfg)
    q_score, q_factors = score_quality(fund, cfg)
    s_score, s_factors = score_financial_strength(fund, cfg)
    m_score, m_factors = score_momentum(fund, cfg)

    composite = (
        gw["value"] * v_score
        + gw["quality"] * q_score
        + gw["financial_strength"] * s_score
        + gw["momentum"] * m_score
    )

    return ScoreBreakdown(
        ticker=fund["ticker"],
        name=fund.get("name", fund["ticker"]),
        sector=fund.get("sector", "Unknown"),
        composite=round(composite, 4),
        value_score=round(v_score, 4),
        quality_score=round(q_score, 4),
        strength_score=round(s_score, 4),
        momentum_score=round(m_score, 4),
        factors={
            "value": v_factors,
            "quality": q_factors,
            "financial_strength": s_factors,
            "momentum": m_factors,
        },
    )


def score_universe(fundamentals: list[dict], config_path: str = "config/scoring.yaml") -> list[ScoreBreakdown]:
    """Score all stocks in the universe. Returns sorted by composite score descending."""
    cfg = load_scoring_config(config_path)
    results = []

    for fund in fundamentals:
        if "error" in fund:
            continue
        try:
            score = score_stock(fund, cfg)
            results.append(score)
        except Exception as e:
            logger.error(f"Scoring failed for {fund.get('ticker')}: {e}")

    results.sort(key=lambda s: s.composite, reverse=True)
    logger.info(f"Scored {len(results)} stocks. Top: {results[0].ticker if results else 'N/A'} ({results[0].composite if results else 0})")
    return results


def results_to_dataframe(results: list[ScoreBreakdown]) -> pd.DataFrame:
    """Convert score results to a clean DataFrame for display/export."""
    rows = []
    for r in results:
        rows.append({
            "Ticker": r.ticker,
            "Name": r.name,
            "Sector": r.sector,
            "Composite": r.composite,
            "Value": r.value_score,
            "Quality": r.quality_score,
            "Strength": r.strength_score,
            "Momentum": r.momentum_score,
        })
    return pd.DataFrame(rows)
