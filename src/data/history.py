"""
Score history storage — snapshots of composite scores over time.
Stores daily snapshots as parquet files for trend analysis.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

HISTORY_DIR = Path("data/history")


def save_snapshot(results: list, tag: str = "daily") -> Path:
    """Save a scoring snapshot. Returns path to saved file."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for r in results:
        rows.append({
            "ticker": r.ticker,
            "name": r.name,
            "sector": r.sector,
            "composite": r.composite,
            "leverage_value": r.leverage_value_score,
            "deleveraging": r.deleveraging_score,
            "quality": r.quality_score,
            "momentum": r.momentum_score,
            "passed_primary": r.passed_primary,
            "passed_bankruptcy": r.passed_bankruptcy,
            "timestamp": datetime.now().isoformat(),
            "tag": tag,
        })

    df = pd.DataFrame(rows)
    path = HISTORY_DIR / f"snapshot_{tag}_{ts}.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"Saved snapshot: {path} ({len(rows)} stocks)")
    return path


def load_all_snapshots() -> pd.DataFrame:
    """Load all historical snapshots into a single DataFrame."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(HISTORY_DIR.glob("snapshot_*.parquet"))

    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            logger.warning(f"Failed to read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    return combined.sort_values("timestamp")


def get_score_history(ticker: str) -> pd.DataFrame:
    """Get historical scores for a single ticker."""
    all_data = load_all_snapshots()
    if all_data.empty:
        return pd.DataFrame()
    return all_data[all_data["ticker"] == ticker].sort_values("timestamp")


def get_latest_snapshot_date() -> str | None:
    """Return the date of the most recent snapshot."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(HISTORY_DIR.glob("snapshot_*.parquet"))
    if not files:
        return None
    try:
        df = pd.read_parquet(files[-1])
        return df["timestamp"].iloc[0] if "timestamp" in df.columns else None
    except Exception:
        return None
