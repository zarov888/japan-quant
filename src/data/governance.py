"""
Japan corporate governance reform signals.
Tracks TSE Prime compliance, cross-shareholding unwinds,
shareholder return improvements, and activist involvement.
"""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_governance_signals(fund: dict) -> dict:
    """
    Compute governance reform signals for a Japanese equity.
    Returns dict of signal scores and flags.
    """
    signals = {}

    mcap = fund.get("market_cap")
    pb = fund.get("pb_ratio")
    roe = fund.get("roe")
    div_yield = fund.get("dividend_yield")
    fcf = fund.get("free_cashflow")
    cash = fund.get("total_cash") or 0
    debt = fund.get("total_debt") or 0
    net_income = fund.get("net_income")

    # ── TSE PRIME COMPLIANCE ──────────────────────────────────
    # TSE requires P/B > 1.0 for Prime listing. Companies below 1.0
    # are under pressure to improve capital efficiency or face delisting risk.
    signals["below_pb_1"] = pb is not None and pb < 1.0
    signals["tse_pressure"] = pb is not None and pb < 1.0
    signals["pb_improvement_urgency"] = (
        "HIGH" if pb is not None and pb < 0.7
        else "MEDIUM" if pb is not None and pb < 1.0
        else "LOW"
    )

    # ── CASH HOARDING ─────────────────────────────────────────
    # Japanese companies historically hoard cash. Those with excess
    # cash relative to market cap are prime targets for activism.
    cash_to_mcap = fund.get("cash_to_mcap")
    signals["cash_rich"] = cash_to_mcap is not None and cash_to_mcap > 0.30
    signals["cash_to_mcap"] = cash_to_mcap
    signals["net_cash_positive"] = cash > debt

    # ── SHAREHOLDER RETURN ────────────────────────────────────
    # Buyback + dividend as % of net income. Japanese companies
    # are increasing payouts under governance reform pressure.
    payout_ratio = None
    if div_yield and mcap and net_income and net_income > 0:
        est_dividends = div_yield * mcap
        payout_ratio = est_dividends / net_income
    signals["est_payout_ratio"] = payout_ratio
    signals["low_payout"] = payout_ratio is not None and payout_ratio < 0.30
    signals["shareholder_return_potential"] = (
        "HIGH" if (payout_ratio is not None and payout_ratio < 0.25 and cash_to_mcap is not None and cash_to_mcap > 0.20)
        else "MEDIUM" if payout_ratio is not None and payout_ratio < 0.35
        else "LOW"
    )

    # ── ROE GAP ───────────────────────────────────────────────
    # TSE target is ROE > 8%. Gap to target indicates reform potential.
    roe_target = 0.08
    if roe is not None:
        signals["roe_gap"] = max(0, roe_target - roe)
        signals["roe_below_target"] = roe < roe_target
    else:
        signals["roe_gap"] = None
        signals["roe_below_target"] = None

    # ── CROSS-SHAREHOLDING UNWIND POTENTIAL ───────────────────
    # Proxy: companies with high cash, low ROE, low P/B are likely
    # holding cross-shareholdings that could be unwound.
    unwind_score = 0.0
    if pb is not None and pb < 1.0:
        unwind_score += 0.3
    if roe is not None and roe < 0.08:
        unwind_score += 0.25
    if cash_to_mcap is not None and cash_to_mcap > 0.20:
        unwind_score += 0.25
    if payout_ratio is not None and payout_ratio < 0.30:
        unwind_score += 0.2
    signals["cross_shareholding_unwind_score"] = round(unwind_score, 2)

    # ── ACTIVIST TARGET SCORE ─────────────────────────────────
    # Composite: undervalued + cash-rich + low returns + reform gap
    activist_score = 0.0
    if pb is not None and pb < 1.0:
        activist_score += 0.25
    if cash_to_mcap is not None and cash_to_mcap > 0.25:
        activist_score += 0.25
    if roe is not None and roe < 0.06:
        activist_score += 0.20
    if payout_ratio is not None and payout_ratio < 0.25:
        activist_score += 0.15
    if fcf is not None and fcf > 0:
        activist_score += 0.15
    signals["activist_target_score"] = round(activist_score, 2)
    signals["activist_risk"] = (
        "HIGH" if activist_score >= 0.70
        else "MEDIUM" if activist_score >= 0.45
        else "LOW"
    )

    # ── GOVERNANCE COMPOSITE ──────────────────────────────────
    # Overall reform potential — higher = more upside from governance changes
    gov_composite = (
        unwind_score * 0.35
        + activist_score * 0.30
        + (0.20 if signals["tse_pressure"] else 0.0)
        + (0.15 if signals["low_payout"] else 0.0)
    )
    signals["governance_composite"] = round(min(gov_composite, 1.0), 3)

    return signals


def add_governance_to_df(df, fund_map: dict):
    """Add governance columns to the main DataFrame."""
    gov_cols = [
        "tse_pressure", "pb_improvement_urgency", "cash_rich",
        "net_cash_positive", "est_payout_ratio", "low_payout",
        "shareholder_return_potential", "roe_below_target", "roe_gap",
        "cross_shareholding_unwind_score", "activist_target_score",
        "activist_risk", "governance_composite",
    ]

    gov_data = {}
    for ticker, fund in fund_map.items():
        gov_data[ticker] = compute_governance_signals(fund)

    for col in gov_cols:
        df[col] = df["Ticker"].map(lambda t, c=col: gov_data.get(t, {}).get(c))

    return df
