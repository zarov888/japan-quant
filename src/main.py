"""
Japan Value Quant Model — Main entry point.
Fetches data, scores universe, outputs ranked results.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from src.data.universe import load_universe
from src.data.fetcher import fetch_universe
from src.model.scorer import score_universe, results_to_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Japan Leveraged Small Value Model v2.0 (Verdad Framework)")
    logger.info("=" * 60)

    # Load universe config
    config = load_universe("config/universe.yaml")
    logger.info(f"Universe: {len(config.tickers)} tickers, benchmark: {config.benchmark}")

    # Fetch fundamentals
    logger.info("Fetching fundamental data...")
    fundamentals = fetch_universe(config)

    if not fundamentals:
        logger.error("No data fetched. Check network/tickers.")
        sys.exit(1)

    logger.info(f"Fetched {len(fundamentals)} stocks")

    # Score universe
    logger.info("Scoring universe...")
    results = score_universe(fundamentals)

    # Output results
    df = results_to_dataframe(results)

    print("\n" + "=" * 90)
    print("LEVERAGED SMALL VALUE SCREEN — JAPAN EQUITIES (VERDAD FRAMEWORK)")
    print("=" * 90)
    print(df.head(30).to_string(index=False))
    print("=" * 90)

    # Export
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"japan_lsv_screen_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Results exported to {csv_path}")

    # Summary stats
    passed = [r for r in results if r.passed_primary and r.passed_bankruptcy]
    print(f"\nScored: {len(results)} stocks")
    print(f"Passed all screens: {len(passed)}")
    print(f"Above 0.55 threshold: {sum(1 for r in results if r.composite >= 0.55)}")
    if results:
        top = results[0]
        flag = "PASS" if (top.passed_primary and top.passed_bankruptcy) else "FAIL"
        print(f"Top pick: {top.ticker} — {top.name} (score: {top.composite}) [{flag}]")
    else:
        print("No stocks scored.")

    # Show sector distribution of passed stocks
    passed_df = df[df["Screen"] == "PASS"].head(20)
    if not passed_df.empty:
        print(f"\nTop 20 Screened — Sector Distribution:")
        print(passed_df["Sector"].value_counts().to_string())

    return results


if __name__ == "__main__":
    main()
