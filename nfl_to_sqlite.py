#!/usr/bin/env python3
"""
Migrate NFL data from Parquet files to SQLite database.

This script converts partitioned Parquet files (season/week structure) into a
centralized SQLite database for easier querying and prediction tracking.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

# Try to use loguru if available, otherwise fall back to standard logging
try:
    from loguru import logger
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)


class NFLDatabaseMigrator:
    """Handles migration of NFL data from Parquet to SQLite."""

    def __init__(
        self,
        parquet_dir: Path = Path("data/processed"),
        db_path: Path = Path("nfl_stats.db"),
        schema_path: Optional[Path] = Path("schema_nfl.sql"),
    ):
        self.parquet_dir = Path(parquet_dir)
        self.db_path = Path(db_path)
        self.schema_path = Path(schema_path) if schema_path else None
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def connect(self) -> None:
        """Create database connection and initialize schema."""
        logger.info(f"Creating database at {self.db_path}")
        self.conn = sqlite3.connect(self.db_path)

        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON")

        # Load schema if provided
        if self.schema_path and self.schema_path.exists():
            logger.info(f"Loading schema from {self.schema_path}")
            schema_sql = self.schema_path.read_text(encoding="utf-8")
            self.conn.executescript(schema_sql)
            self.conn.commit()
        else:
            logger.warning("No schema file found - tables must exist already")

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.commit()
            self.conn.close()
            logger.info("Database connection closed")

    def _read_partitioned_parquet(self, table_name: str) -> pd.DataFrame:
        """
        Read all Parquet files for a given table from partitioned directory structure.

        Args:
            table_name: Name of table (e.g., 'player_week', 'schedule', 'injuries')

        Returns:
            Combined DataFrame from all partitions
        """
        table_dir = self.parquet_dir / table_name
        if not table_dir.exists():
            logger.warning(f"Table directory not found: {table_dir}")
            return pd.DataFrame()

        # Find all .parquet files recursively
        parquet_files = list(table_dir.glob("**/*.parquet"))

        if not parquet_files:
            logger.warning(f"No parquet files found in {table_dir}")
            return pd.DataFrame()

        logger.info(f"Found {len(parquet_files)} parquet files for {table_name}")

        # Read and combine all partitions
        dfs = []
        for pq_file in parquet_files:
            try:
                df = pd.read_parquet(pq_file)
                dfs.append(df)
                logger.debug(f"Read {len(df)} rows from {pq_file.name}")
            except Exception as e:
                logger.error(f"Failed to read {pq_file}: {e}")

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined {len(combined):,} total rows for {table_name}")
        return combined

    def migrate_player_week_stats(self) -> int:
        """
        Migrate player weekly statistics.

        Returns:
            Number of rows inserted
        """
        logger.info("Migrating player_week_stats...")
        df = self._read_partitioned_parquet("player_week")

        if df.empty:
            logger.warning("No player_week data to migrate")
            return 0

        # Ensure column names are lowercase (already done by fetch.py)
        df.columns = [c.lower() for c in df.columns]

        # Keep only columns that exist in our schema (ignore extra columns from data source)
        expected_cols = [
            'player_id', 'player_name', 'player_display_name', 'position', 'position_group',
            'headshot_url', 'recent_team', 'season', 'week', 'season_type', 'opponent_team',
            'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
            'sacks', 'sack_yards', 'sack_fumbles', 'sack_fumbles_lost', 'passing_air_yards',
            'passing_yards_after_catch', 'passing_first_downs', 'passing_epa',
            'passing_2pt_conversions', 'pacr', 'dakota', 'carries', 'rushing_yards',
            'rushing_tds', 'rushing_fumbles', 'rushing_fumbles_lost', 'rushing_first_downs',
            'rushing_epa', 'rushing_2pt_conversions', 'receptions', 'targets',
            'receiving_yards', 'receiving_tds', 'receiving_fumbles', 'receiving_fumbles_lost',
            'receiving_air_yards', 'receiving_yards_after_catch', 'receiving_first_downs',
            'receiving_epa', 'receiving_2pt_conversions', 'racr', 'target_share',
            'air_yards_share', 'wopr', 'special_teams_tds', 'fantasy_points',
            'fantasy_points_ppr'
        ]
        # Only keep columns that exist in both the dataframe and our schema
        cols_to_keep = [c for c in expected_cols if c in df.columns]
        df = df[cols_to_keep]

        # Drop duplicates (safety check)
        before = len(df)
        df = df.drop_duplicates(subset=["player_id", "season", "week", "season_type"])
        if len(df) < before:
            logger.warning(f"Dropped {before - len(df)} duplicate rows")

        # Write to database (small chunksize due to SQLite variable limit)
        # With 54 columns, max chunksize is ~18 rows (999 var limit / 54 cols)
        rows_inserted = df.to_sql(
            "player_week_stats",
            self.conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=15,  # Safe value under SQLite's 999 variable limit
        )

        logger.info(f"Inserted {rows_inserted:,} rows into player_week_stats")
        return rows_inserted or len(df)

    def migrate_schedule(self) -> int:
        """
        Migrate game schedule data.

        Returns:
            Number of rows inserted
        """
        logger.info("Migrating schedule...")
        df = self._read_partitioned_parquet("schedule")

        if df.empty:
            logger.warning("No schedule data to migrate")
            return 0

        # Ensure column names are lowercase
        df.columns = [c.lower() for c in df.columns]

        # Drop duplicates
        before = len(df)
        df = df.drop_duplicates(subset=["game_id"])
        if len(df) < before:
            logger.warning(f"Dropped {before - len(df)} duplicate rows")

        # Write to database
        rows_inserted = df.to_sql(
            "schedule",
            self.conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
        )

        logger.info(f"Inserted {rows_inserted:,} rows into schedule")
        return rows_inserted or len(df)

    def migrate_injuries(self) -> int:
        """
        Migrate injury report data.

        Returns:
            Number of rows inserted
        """
        logger.info("Migrating injuries...")
        df = self._read_partitioned_parquet("injuries")

        if df.empty:
            logger.warning("No injuries data to migrate")
            return 0

        # Ensure column names are lowercase
        df.columns = [c.lower() for c in df.columns]

        # Drop duplicates
        before = len(df)
        df = df.drop_duplicates(
            subset=["gsis_id", "season", "week", "game_type", "team"]
        )
        if len(df) < before:
            logger.warning(f"Dropped {before - len(df)} duplicate rows")

        # Write to database
        rows_inserted = df.to_sql(
            "injuries",
            self.conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )

        logger.info(f"Inserted {rows_inserted:,} rows into injuries")
        return rows_inserted or len(df)

    def migrate_all(self) -> dict[str, int]:
        """
        Run all migrations.

        Returns:
            Dictionary with row counts for each table
        """
        logger.info("=" * 60)
        logger.info("Starting NFL Database Migration")
        logger.info("=" * 60)

        results = {
            "schedule": self.migrate_schedule(),
            "player_week_stats": self.migrate_player_week_stats(),
            "injuries": self.migrate_injuries(),
        }

        logger.info("=" * 60)
        logger.info("Migration Summary:")
        for table, count in results.items():
            logger.info(f"  {table}: {count:,} rows")
        logger.info(f"Total: {sum(results.values()):,} rows")
        logger.info("=" * 60)

        return results

    def verify_migration(self) -> None:
        """Verify migration was successful by running test queries."""
        logger.info("Verifying migration...")

        queries = {
            "Total players": "SELECT COUNT(DISTINCT player_id) FROM player_week_stats",
            "Total games": "SELECT COUNT(*) FROM schedule",
            "Seasons covered": "SELECT MIN(season), MAX(season) FROM player_week_stats",
            "Weeks covered": "SELECT MIN(week), MAX(week) FROM player_week_stats",
            "Total injuries": "SELECT COUNT(*) FROM injuries",
            "Positions": "SELECT DISTINCT position FROM player_week_stats ORDER BY position",
        }

        for description, query in queries.items():
            try:
                cursor = self.conn.execute(query)
                result = cursor.fetchall()
                logger.info(f"{description}: {result}")
            except Exception as e:
                logger.error(f"Verification query failed ({description}): {e}")

        logger.info("Verification complete")

    def get_database_stats(self) -> dict:
        """Get statistics about the database."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        stats = {}

        # Table sizes
        for table in ["player_week_stats", "schedule", "injuries", "predictions"]:
            cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table}")
            stats[f"{table}_rows"] = cursor.fetchone()[0]

        # Database file size
        stats["db_size_mb"] = self.db_path.stat().st_size / (1024 * 1024)

        return stats


def main():
    """Main migration function."""
    # Use context manager to ensure proper cleanup
    with NFLDatabaseMigrator(
        parquet_dir=Path("data/processed"),
        db_path=Path("nfl_stats.db"),
        schema_path=Path("schema_nfl.sql"),
    ) as migrator:
        # Run migration
        migrator.migrate_all()

        # Verify migration
        migrator.verify_migration()

        # Show database stats
        stats = migrator.get_database_stats()
        logger.info("\nDatabase Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

    logger.info(f"\nMigration complete! Database created at: nfl_stats.db")
    logger.info("You can now use this database for predictions and analysis.")


if __name__ == "__main__":
    main()
