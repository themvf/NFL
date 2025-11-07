"""
NFLverse Direct API Refresh Script

Fetches fresh data directly from NFLverse API using nflreadpy library
and updates the merged database. Works both locally and on Streamlit Cloud.

This eliminates the need for a local NFLverse database file.
"""

import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
import sys

try:
    import pandas as pd
    import nflreadpy as nfl
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Run: pip install nflreadpy pandas polars")
    sys.exit(1)

# Database paths
PROJECT_DIR = Path(__file__).parent
MERGED_DB = PROJECT_DIR / "data" / "nfl_merged.db"

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

# NFLverse tables to refresh
NFLVERSE_TABLES = [
    'schedules',
    'rosters',
    'injuries',
    'team_stats',
    'team_stats_week',
    'pfr_advstats_pass_week',
    'pfr_advstats_rush_week',
    'pfr_advstats_rec_week',
    'pfr_advstats_def_week',
    'ingest_metadata',
]

# PFR and custom tables to preserve (DO NOT refresh these)
PRESERVED_TABLES = [
    'plays',
    'play_participants',
    'user_notes',
    'player_injuries',
    'player_transactions',
    'upcoming_games',
    'projection_accuracy',
    'team_abbreviation_mapping',
]


def backup_database():
    """
    Create timestamped backup of merged database.

    Returns:
        Path: Path to backup file
    """
    if not MERGED_DB.exists():
        raise FileNotFoundError(f"Merged database not found: {MERGED_DB}")

    # Create backup with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = MERGED_DB.with_suffix(f'.db.backup_{timestamp}')

    shutil.copy2(MERGED_DB, backup_path)

    # Also maintain a "latest" backup (overwrite)
    latest_backup = MERGED_DB.with_suffix('.db.backup')
    shutil.copy2(MERGED_DB, latest_backup)

    return backup_path


def fetch_nflverse_data(season, progress_callback=None):
    """
    Fetch data directly from NFLverse API using nflreadpy.

    Args:
        season (int): Season to fetch
        progress_callback (callable): Optional progress reporting function

    Returns:
        dict: Dictionary of table_name -> pandas DataFrame
    """
    print(f"\nFetching NFLverse data for {season} season from API...")

    data = {}

    # Fetch schedules
    if progress_callback:
        progress_callback("schedules", 0, 9, 1)
    try:
        df = nfl.load_schedules(season)
        if hasattr(df, 'to_pandas'):  # Polars DataFrame
            df = df.to_pandas()
        data['schedules'] = df
        print(f"  ✓ Fetched schedules: {len(df)} games")
    except Exception as e:
        print(f"  ⚠ Error fetching schedules: {e}")

    # Fetch rosters
    if progress_callback:
        progress_callback("rosters", 0, 9, 2)
    try:
        df = nfl.load_rosters(season)
        if hasattr(df, 'to_pandas'):
            df = df.to_pandas()
        data['rosters'] = df
        print(f"  ✓ Fetched rosters: {len(df)} players")
    except Exception as e:
        print(f"  ⚠ Error fetching rosters: {e}")

    # Fetch injuries
    if progress_callback:
        progress_callback("injuries", 0, 9, 3)
    try:
        df = nfl.load_injuries(season)
        if hasattr(df, 'to_pandas'):
            df = df.to_pandas()
        data['injuries'] = df
        print(f"  ✓ Fetched injuries: {len(df)} entries")
    except Exception as e:
        print(f"  ⚠ Error fetching injuries: {e}")

    # Fetch team stats (season-level)
    if progress_callback:
        progress_callback("team_stats", 0, 9, 4)
    try:
        df = nfl.load_team_stats(season, summary_level='reg')
        if hasattr(df, 'to_pandas'):
            df = df.to_pandas()
        data['team_stats'] = df
        print(f"  ✓ Fetched team_stats: {len(df)} rows")
    except Exception as e:
        print(f"  ⚠ Error fetching team_stats: {e}")

    # Fetch team stats (weekly)
    if progress_callback:
        progress_callback("team_stats_week", 0, 9, 5)
    try:
        df = nfl.load_team_stats(season, summary_level='week')
        if hasattr(df, 'to_pandas'):
            df = df.to_pandas()
        data['team_stats_week'] = df
        print(f"  ✓ Fetched team_stats_week: {len(df)} rows")
    except Exception as e:
        print(f"  ⚠ Error fetching team_stats_week: {e}")

    # Fetch advanced stats
    adv_stats = [
        ('pfr_advstats_pass_week', 'pass'),
        ('pfr_advstats_rush_week', 'rush'),
        ('pfr_advstats_rec_week', 'rec'),
        ('pfr_advstats_def_week', 'def'),
    ]

    for idx, (table_name, stat_type) in enumerate(adv_stats, 6):
        if progress_callback:
            progress_callback(table_name, 0, 9, idx)
        try:
            df = nfl.load_pfr_advstats(season, stat_type=stat_type, summary_level='week')
            if hasattr(df, 'to_pandas'):
                df = df.to_pandas()
            data[table_name] = df
            print(f"  ✓ Fetched {table_name}: {len(df)} rows")
        except Exception as e:
            print(f"  ⚠ Error fetching {table_name}: {e}")

    return data


def refresh_nflverse_tables_direct(season=2025, progress_callback=None):
    """
    Refresh NFLverse tables by fetching directly from API.

    Args:
        season (int): Season to refresh (default: 2025)
        progress_callback (callable): Optional function to report progress
            Signature: callback(table_name: str, rows_updated: int, total_tables: int, current_table: int)

    Returns:
        dict: Results including success status, counts, and any errors
    """
    results = {
        'success': False,
        'backup_path': None,
        'tables_updated': {},
        'total_rows_updated': 0,
        'error': None,
        'start_time': datetime.now(),
        'end_time': None
    }

    try:
        # Verify merged database exists
        if not MERGED_DB.exists():
            raise FileNotFoundError(f"Merged database not found: {MERGED_DB}")

        # Create backup
        results['backup_path'] = backup_database()
        print(f"\n✓ Backup created: {results['backup_path'].name}")

        # Fetch fresh data from NFLverse API
        nflverse_data = fetch_nflverse_data(season, progress_callback)

        if not nflverse_data:
            raise Exception("No data fetched from NFLverse API")

        # Connect to merged database
        merged_conn = sqlite3.connect(MERGED_DB)

        try:
            # Begin transaction
            merged_conn.execute("BEGIN TRANSACTION")

            # Update each table
            for table_name, df in nflverse_data.items():
                try:
                    # Delete existing data for this season
                    try:
                        deleted = merged_conn.execute(
                            f"DELETE FROM {table_name} WHERE season = ?",
                            (season,)
                        ).rowcount
                    except sqlite3.OperationalError:
                        # Table might not have season column, delete all
                        deleted = merged_conn.execute(f"DELETE FROM {table_name}").rowcount

                    # Insert fresh data
                    if not df.empty:
                        df.to_sql(table_name, merged_conn, if_exists='append', index=False)
                        rows_updated = len(df)
                        results['tables_updated'][table_name] = rows_updated
                        results['total_rows_updated'] += rows_updated

                        if progress_callback:
                            progress_callback(
                                table_name,
                                rows_updated,
                                len(nflverse_data),
                                len(results['tables_updated'])
                            )
                    else:
                        results['tables_updated'][table_name] = 0

                except Exception as e:
                    results['tables_updated'][table_name] = f"Error: {str(e)}"
                    print(f"  ⚠ Error updating {table_name}: {e}")

            # Update refresh metadata
            try:
                merged_conn.execute("""
                    CREATE TABLE IF NOT EXISTS merge_metadata (
                        last_refresh_timestamp TEXT,
                        refresh_source TEXT,
                        rows_updated INTEGER
                    )
                """)
                merged_conn.execute("DELETE FROM merge_metadata")
                merged_conn.execute(
                    "INSERT INTO merge_metadata VALUES (?, ?, ?)",
                    (datetime.now().isoformat(), 'api-direct', results['total_rows_updated'])
                )
            except Exception as e:
                # Non-critical, continue
                pass

            # Commit transaction
            merged_conn.commit()
            results['success'] = True

        except Exception as e:
            # Rollback on error
            merged_conn.rollback()
            raise

        finally:
            merged_conn.close()

    except Exception as e:
        results['error'] = str(e)
        results['success'] = False

    results['end_time'] = datetime.now()
    results['duration_seconds'] = (results['end_time'] - results['start_time']).total_seconds()

    return results


def main():
    """Command line interface for direct API refresh."""
    print("=" * 60)
    print("NFL DIRECT API REFRESH (No local NFLverse DB required)")
    print("=" * 60)
    print()

    if not MERGED_DB.exists():
        print(f"ERROR: Merged database not found at: {MERGED_DB}")
        print("Run merge_databases.py first to create the merged database.")
        return 1

    print("Starting direct API refresh...")
    print()

    # Progress callback
    def show_progress(table, rows, total_tables, current):
        print(f"  [{current}/{total_tables}] {table:30} {rows:>6} rows updated")

    # Run refresh
    results = refresh_nflverse_tables_direct(season=2025, progress_callback=show_progress)

    print()
    if results['success']:
        print("=" * 60)
        print("REFRESH COMPLETE!")
        print("=" * 60)
        print(f"Total rows updated: {results['total_rows_updated']}")
        print(f"Duration: {results['duration_seconds']:.1f} seconds")
        print(f"Backup saved: {results['backup_path'].name}")
        print()
        print("Next steps:")
        print("  1. git add data/nfl_merged.db")
        print('  2. git commit -m "chore: refresh NFLverse data from API"')
        print("  3. git push")
        return 0
    else:
        print("=" * 60)
        print("REFRESH FAILED!")
        print("=" * 60)
        print(f"Error: {results['error']}")
        print()
        if results['backup_path']:
            print(f"Backup available at: {results['backup_path']}")
            print("To restore:")
            print(f"  cp {results['backup_path']} {MERGED_DB}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
