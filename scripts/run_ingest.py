from __future__ import annotations

import argparse
from pathlib import Path

from nfl_app.ingest.fetch import IngestConfig, fetch_and_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NFL data ingestion")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        help="Seasons to ingest, e.g. --seasons 2023 2024 2025",
    )
    parser.add_argument(
        "--weeks",
        nargs="+",
        type=int,
        help="Optional weeks to include (e.g., --weeks 1 2 3 4)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output base directory for partitioned Parquet",
    )
    args = parser.parse_args()

    cfg = IngestConfig(seasons=args.seasons or [], weeks=args.weeks, data_dir=args.data_dir)
    res = fetch_and_store(cfg)
    print(res)


if __name__ == "__main__":
    main()



