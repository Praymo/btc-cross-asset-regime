"""Refresh the public BTC and ETF cache used by the research entry point."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research import download_data  # noqa: E402


if __name__ == "__main__":
    frame = download_data(force=True)
    print(f"saved {len(frame):,} rows to data/cross_asset_daily.csv")
