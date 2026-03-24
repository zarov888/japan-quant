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
    logger.info("Japan Value Quant Model v1.0")
    logger.info("=" * 50)

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

    print("\n" + "=" * 80)
    print("TOP VALUE PICKS — JAPAN EQUITIES")
    print("=" * 80)
    print(df.head(30).to_string(index=False))
    print("=" * 80)

    # Export
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"japan_value_screen_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Results exported to {csv_path}")

    # Summary stats
    print(f"\nScored: {len(results)} stocks")
    print(f"Above 0.60 threshold: {sum(1 for r in results if r.composite >= 0.60)}")
    print(f"Top pick: {results[0].ticker} — {results[0].name} (score: {results[0].composite})")

    # Show sector distribution of top 20
    top20 = df.head(20)
    print(f"\nTop 20 Sector Distribution:")
    print(top20["Sector"].value_counts().to_string())

    return results


if __name__ == "__main__":
    main()
