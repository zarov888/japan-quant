"""
Verdad-style leveraged small value scoring model for Japanese equities.
Implements Rasmussen's PE replication thesis: leverage + size + value.
Scores across leverage/value, deleveraging, quality/efficiency, and momentum.
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
    leverage_value_score: float
    deleveraging_score: float
    quality_score: float
    momentum_score: float
    passed_primary: bool
    passed_bankruptcy: bool
    factors: dict


def load_scoring_config(path: str = "config/scoring.yaml") -> dict:
    """Load scoring model configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def _normalize(value: float | None, low: float, high: float, invert: bool = False) -> float:
    """
    Normalize a value to [0, 1] range.
    If invert=True, lower raw values produce higher scores (e.g., EV/EBITDA).
    Returns 0.5 if value is None (neutral).
    """
    if value is None or np.isnan(value):
        return 0.5

    value = max(low, min(high, value))
    score = (value - low) / (high - low) if high != low else 0.5

    return 1.0 - score if invert else score


# ── PRIMARY SCREEN ──────────────────────────────────────────

def passes_primary_screen(fund: dict, cfg: dict) -> bool:
    """
    Hard gates from Verdad's core thesis.
    Must be small, leveraged, and cheap to enter the universe.
    Returns True if stock passes all primary screens.
    """
    ps = cfg.get("primary_screen", {})

    mcap = fund.get("market_cap")
    if mcap is None:
        return False

    # Size: within target range
    mcap_min = ps.get("market_cap_min_jpy", 30e9)
    mcap_max = ps.get("market_cap_max_jpy", 300e9)
    if mcap < mcap_min or mcap > mcap_max:
        return False

    # Leverage: LT debt/EV above minimum
    lt_debt_ev = fund.get("lt_debt_to_ev")
    if lt_debt_ev is not None:
        if lt_debt_ev < ps.get("lt_debt_to_ev_min", 0.10):
            return False

    # Cheap: EV/EBITDA below ceiling
    ev_ebitda = fund.get("ev_to_ebitda")
    if ev_ebitda is not None:
        if ev_ebitda > ps.get("ev_to_ebitda_max", 8.0) or ev_ebitda < 0:
            return False

    # Net debt/EBITDA sanity check
    nd_ebitda = fund.get("net_debt_to_ebitda")
    if nd_ebitda is not None:
        if nd_ebitda > ps.get("net_debt_to_ebitda_max", 5.0):
            return False

    return True


def passes_bankruptcy_screen(fund: dict, cfg: dict) -> bool:
    """
    Verdad avoids distress: profitable, cash-flow generative, not heavily shorted.
    """
    bs = cfg.get("bankruptcy_screen", {})

    # Must be profitable
    if bs.get("require_profitable", True):
        ni = fund.get("net_income")
        if ni is not None and ni <= 0:
            return False

    # Must have positive FCF
    if bs.get("require_positive_fcf", True):
        fcf = fund.get("free_cashflow")
        if fcf is not None and fcf <= 0:
            return False

    # Must have positive operating cashflow
    if bs.get("require_positive_ocf", True):
        ocf = fund.get("operating_cashflow")
        if ocf is not None and ocf <= 0:
            return False

    # Not heavily shorted
    short_pct = fund.get("short_pct_float")
    if short_pct is not None:
        if short_pct > bs.get("max_short_pct_float", 0.10):
            return False

    short_ratio = fund.get("short_ratio")
    if short_ratio is not None:
        if short_ratio > bs.get("max_short_ratio", 8.0):
            return False

    return True


# ── SCORING FUNCTIONS ───────────────────────────────────────

def score_leverage_value(fund: dict, cfg: dict) -> tuple[float, dict]:
    """
    Core Verdad: cheap enterprise value + meaningful leverage.
    Higher EBITDA/EV, higher LT debt/EV, lower EV/EBITDA = better.
    """
    weights = cfg["leverage_value"]["weights"]
    factors = {}

    # EBITDA/EV — primary earnings yield metric
    ebitda_ev = fund.get("ebitda_to_ev")
    factors["ebitda_to_ev"] = _normalize(ebitda_ev, 0.05, 0.25, invert=False)

    # EV/EBITDA — lower is cheaper
    ev_ebitda = fund.get("ev_to_ebitda")
    factors["ev_to_ebitda"] = _normalize(ev_ebitda, 2.0, 8.0, invert=True)

    # LT Debt/EV — leverage intensity (higher = more levered equity upside)
    lt_debt_ev = fund.get("lt_debt_to_ev")
    factors["lt_debt_to_ev"] = _normalize(lt_debt_ev, 0.10, 0.50, invert=False)

    # P/B — still king in Japan
    pb = fund.get("pb_ratio")
    factors["price_to_book"] = _normalize(pb, 0.3, 2.0, invert=True)

    # Net Debt/EBITDA — lower is safer (inverted — penalize high leverage depth)
    nd_ebitda = fund.get("net_debt_to_ebitda")
    factors["net_debt_to_ebitda"] = _normalize(nd_ebitda, 0.0, 5.0, invert=True)

    score = sum(factors[k] * weights.get(k, 0) for k in factors)
    return score, factors


def score_deleveraging(fund: dict, cfg: dict) -> tuple[float, dict]:
    """
    Verdad's secondary factors: debt paydown is the #1 predictor.
    The deleveraging flywheel drives equity returns.
    """
    weights = cfg["deleveraging"]["weights"]
    factors = {}

    # Debt paydown — LT debt declining YoY
    # We approximate with debt-to-equity trend; lower D/E = deleveraging signal
    de = fund.get("debt_to_equity")
    if de is not None:
        de_ratio = de / 100.0 if de > 5 else de
        # Lower D/E suggests deleveraging has occurred or is manageable
        factors["debt_paydown"] = _normalize(de_ratio, 0.0, 2.0, invert=True)
    else:
        factors["debt_paydown"] = 0.5

    # Asset turnover (Revenue / Assets) — improving efficiency
    at = fund.get("asset_turnover")
    factors["asset_turnover"] = _normalize(at, 0.2, 1.5, invert=False)

    # Asset growth — expanding base (proxy: revenue growth)
    rg = fund.get("revenue_growth")
    factors["asset_growth"] = _normalize(rg, -0.10, 0.30, invert=False)

    # Market cap rank — smaller within range = better
    mcap = fund.get("market_cap")
    if mcap:
        # Normalize within 30B–300B JPY range, inverted (smaller = higher score)
        factors["market_cap_rank"] = _normalize(mcap, 30e9, 300e9, invert=True)
    else:
        factors["market_cap_rank"] = 0.5

    # Gross profit / assets — Novy-Marx capital efficiency
    gpa = fund.get("gross_profit_to_assets")
    factors["gross_profit_to_assets"] = _normalize(gpa, 0.05, 0.40, invert=False)

    score = sum(factors[k] * weights.get(k, 0) for k in factors)
    return score, factors


def score_quality(fund: dict, cfg: dict) -> tuple[float, dict]:
    """
    Quality and efficiency within the leveraged universe.
    High gross profit/assets, solid margins, moderate leverage on assets.
    """
    weights = cfg["quality_efficiency"]["weights"]
    factors = {}

    # Gross profit / assets — Verdad's preferred quality metric
    gpa = fund.get("gross_profit_to_assets")
    factors["gross_profit_to_assets"] = _normalize(gpa, 0.05, 0.40, invert=False)

    # Operating margin
    factors["operating_margin"] = _normalize(
        fund.get("operating_margin"), 0.0, 0.20, invert=False
    )

    # ROE
    factors["roe"] = _normalize(fund.get("roe"), 0.0, 0.20, invert=False)

    # LT debt / assets — low-to-moderate is ideal
    lt_debt_assets = fund.get("lt_debt_to_assets")
    factors["lt_debt_to_assets"] = _normalize(lt_debt_assets, 0.0, 0.50, invert=True)

    # Current ratio
    factors["current_ratio"] = _normalize(
        fund.get("current_ratio"), 0.5, 3.0, invert=False
    )

    score = sum(factors[k] * weights.get(k, 0) for k in factors)
    return score, factors


def score_momentum(fund: dict, cfg: dict) -> tuple[float, dict]:
    """Avoid falling knives. Light weight in Verdad framework."""
    weights = cfg["momentum"]["weights"]
    factors = {}

    price = fund.get("current_price")
    high_52w = fund.get("fifty_two_week_high")
    low_52w = fund.get("fifty_two_week_low")
    sma50 = fund.get("fifty_day_avg")
    sma200 = fund.get("two_hundred_day_avg")

    # Relative strength (position in 52-week range)
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

    # SMA 50 vs 200 (golden cross)
    if sma50 and sma200 and sma200 > 0:
        sma_ratio = sma50 / sma200
        factors["sma_50_vs_200"] = _normalize(sma_ratio, 0.9, 1.1)
    else:
        factors["sma_50_vs_200"] = 0.5

    score = sum(factors[k] * weights.get(k, 0) for k in factors)
    return score, factors


# ── COMPOSITE SCORING ───────────────────────────────────────

def score_stock(fund: dict, cfg: dict) -> ScoreBreakdown:
    """
    Compute Verdad composite score for a single stock.
    Applies primary screen + bankruptcy screen, then scores survivors.
    """
    gw = cfg["group_weights"]

    passed_primary = passes_primary_screen(fund, cfg)
    passed_bankruptcy = passes_bankruptcy_screen(fund, cfg)

    lv_score, lv_factors = score_leverage_value(fund, cfg)
    dl_score, dl_factors = score_deleveraging(fund, cfg)
    q_score, q_factors = score_quality(fund, cfg)
    m_score, m_factors = score_momentum(fund, cfg)

    composite = (
        gw["leverage_value"] * lv_score
        + gw["deleveraging"] * dl_score
        + gw["quality_efficiency"] * q_score
        + gw["momentum"] * m_score
    )

    # Penalize stocks that fail screens (still score them, but discount)
    if not passed_primary:
        composite *= 0.60
    if not passed_bankruptcy:
        composite *= 0.70

    return ScoreBreakdown(
        ticker=fund["ticker"],
        name=fund.get("name", fund["ticker"]),
        sector=fund.get("sector", "Unknown"),
        composite=round(composite, 4),
        leverage_value_score=round(lv_score, 4),
        deleveraging_score=round(dl_score, 4),
        quality_score=round(q_score, 4),
        momentum_score=round(m_score, 4),
        passed_primary=passed_primary,
        passed_bankruptcy=passed_bankruptcy,
        factors={
            "leverage_value": lv_factors,
            "deleveraging": dl_factors,
            "quality_efficiency": q_factors,
            "momentum": m_factors,
        },
    )


def score_universe(
    fundamentals: list[dict],
    config_path: str = "config/scoring.yaml",
) -> list[ScoreBreakdown]:
    """Score all stocks. Returns sorted by composite score descending."""
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

    passed = [r for r in results if r.passed_primary and r.passed_bankruptcy]
    logger.info(
        f"Scored {len(results)} stocks. "
        f"{len(passed)} passed all screens. "
        f"Top: {results[0].ticker if results else 'N/A'} "
        f"({results[0].composite if results else 0})"
    )
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
            "LevValue": r.leverage_value_score,
            "Delever": r.deleveraging_score,
            "Quality": r.quality_score,
            "Momentum": r.momentum_score,
            "Screen": "PASS" if (r.passed_primary and r.passed_bankruptcy) else "FAIL",
        })
    return pd.DataFrame(rows)
