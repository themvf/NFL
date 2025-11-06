"""
NFLverse Data Refresh Script for Merged Database

Incrementally refreshes NFLverse tables in nfl_merged.db while preserving
pfr.db tables and custom user data (notes, projections, injuries).

Usage:
    Command line: python refresh_merged_db.py
    Streamlit:    Call refresh_nflverse_tables() function

Features:
    - Automatic backup before changes
    - Transaction-based updates (rollback on error)
    - Preserves custom user data
    - Progress reporting
    - Data integrity validation
"""

import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
import sys

# Database paths
PROJECT_DIR = Path(__file__).parent
NFLVERSE_DB = PROJECT_DIR.parent.parent / "NFL Data NFLVerse" / "NFL-Data" / "data" / "nflverse.sqlite"
MERGED_DB = PROJECT_DIR / "data" / "nfl_merged.db"

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

# NFLverse tables to refresh (excludes pfr.db tables and custom tables)
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


def get_database_status():
    """
    Get current status of databases including freshness and row counts.

    Returns:
        dict: Status information for display
    """
    status = {
        'nflverse_exists': NFLVERSE_DB.exists(),
        'merged_exists': MERGED_DB.exists(),
        'nflverse_modified': None,
        'merged_modified': None,
        'needs_refresh': False,
        'table_counts': {},
        'last_refresh': None
    }

    if NFLVERSE_DB.exists():
        status['nflverse_modified'] = datetime.fromtimestamp(NFLVERSE_DB.stat().st_mtime)
        status['nflverse_size_mb'] = NFLVERSE_DB.stat().st_size / (1024 * 1024)

    if MERGED_DB.exists():
        status['merged_modified'] = datetime.fromtimestamp(MERGED_DB.stat().st_mtime)
        status['merged_size_mb'] = MERGED_DB.stat().st_size / (1024 * 1024)

        # Check if refresh needed
        if status['nflverse_modified'] and status['merged_modified']:
            status['needs_refresh'] = status['nflverse_modified'] > status['merged_modified']

        # Get table row counts
        try:
            conn = sqlite3.connect(MERGED_DB)
            cursor = conn.cursor()

            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    status['table_counts'][table] = count
                except:
                    status['table_counts'][table] = 'Error'

            # Get last refresh timestamp if exists
            try:
                cursor.execute("SELECT last_refresh_timestamp FROM merge_metadata")
                result = cursor.fetchone()
                if result:
                    status['last_refresh'] = result[0]
            except:
                pass

            conn.close()
        except Exception as e:
            status['error'] = str(e)

    return status


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


def refresh_nflverse_tables(season=2025, progress_callback=None):
    """
    Refresh NFLverse tables in merged database while preserving pfr.db tables.

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
        # Verify source database exists
        if not NFLVERSE_DB.exists():
            raise FileNotFoundError(f"NFLverse database not found: {NFLVERSE_DB}")

        if not MERGED_DB.exists():
            raise FileNotFoundError(f"Merged database not found: {MERGED_DB}")

        # Create backup
        results['backup_path'] = backup_database()

        # Connect to databases
        nflverse_conn = sqlite3.connect(NFLVERSE_DB)
        merged_conn = sqlite3.connect(MERGED_DB)

        try:
            # Begin transaction
            merged_conn.execute("BEGIN TRANSACTION")

            total_tables = len(NFLVERSE_TABLES)

            for idx, table in enumerate(NFLVERSE_TABLES, 1):
                try:
                    # Delete existing data for this season
                    # Some tables might not have season column, try both approaches
                    try:
                        deleted = merged_conn.execute(
                            f"DELETE FROM {table} WHERE season = ?",
                            (season,)
                        ).rowcount
                    except sqlite3.OperationalError:
                        # Table might not have season column, delete all
                        deleted = merged_conn.execute(f"DELETE FROM {table}").rowcount

                    # Get fresh data from nflverse
                    try:
                        cursor = nflverse_conn.execute(
                            f"SELECT * FROM {table} WHERE season = ?",
                            (season,)
                        )
                        rows = cursor.fetchall()
                    except sqlite3.OperationalError:
                        # Table might not have season column, get all
                        cursor = nflverse_conn.execute(f"SELECT * FROM {table}")
                        rows = cursor.fetchall()

                    if rows:
                        # Get column count for placeholders
                        column_count = len(cursor.description)
                        placeholders = ','.join(['?' for _ in range(column_count)])

                        # Insert fresh data
                        merged_conn.executemany(
                            f"INSERT INTO {table} VALUES ({placeholders})",
                            rows
                        )

                        rows_updated = len(rows)
                        results['tables_updated'][table] = rows_updated
                        results['total_rows_updated'] += rows_updated
                    else:
                        results['tables_updated'][table] = 0

                    # Report progress
                    if progress_callback:
                        progress_callback(table, results['tables_updated'][table], total_tables, idx)

                except Exception as e:
                    results['tables_updated'][table] = f"Error: {str(e)}"

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
                    (datetime.now().isoformat(), 'auto-refresh', results['total_rows_updated'])
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
            nflverse_conn.close()
            merged_conn.close()

    except Exception as e:
        results['error'] = str(e)
        results['success'] = False

    results['end_time'] = datetime.now()
    results['duration_seconds'] = (results['end_time'] - results['start_time']).total_seconds()

    return results


def restore_from_backup(backup_path=None):
    """
    Restore merged database from backup.

    Args:
        backup_path (Path): Path to backup file. If None, uses latest backup.

    Returns:
        bool: True if successful
    """
    if backup_path is None:
        backup_path = MERGED_DB.with_suffix('.db.backup')

    if not Path(backup_path).exists():
        raise FileNotFoundError(f"Backup not found: {backup_path}")

    shutil.copy2(backup_path, MERGED_DB)
    return True


def verify_database_integrity():
    """
    Verify merged database integrity.

    Returns:
        dict: Validation results
    """
    results = {
        'valid': True,
        'checks': {},
        'errors': []
    }

    try:
        conn = sqlite3.connect(MERGED_DB)
        cursor = conn.cursor()

        # Check all expected tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = set(row[0] for row in cursor.fetchall())

        expected_tables = set(NFLVERSE_TABLES + PRESERVED_TABLES + ['merge_metadata', 'games'])

        missing_tables = expected_tables - existing_tables
        if missing_tables:
            results['valid'] = False
            results['errors'].append(f"Missing tables: {', '.join(missing_tables)}")

        results['checks']['tables_exist'] = len(missing_tables) == 0

        # Check critical tables have data
        critical_tables = ['schedules', 'plays', 'rosters']
        for table in critical_tables:
            if table in existing_tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                results['checks'][f'{table}_has_data'] = count > 0
                if count == 0:
                    results['valid'] = False
                    results['errors'].append(f"Table {table} is empty")

        conn.close()

    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Validation error: {str(e)}")

    return results


def main():
    """Command line interface for refresh script."""
    print("=" * 60)
    print("NFL MERGED DATABASE REFRESH")
    print("=" * 60)
    print()

    # Check database status
    status = get_database_status()

    if not status['nflverse_exists']:
        print(f"ERROR: NFLverse database not found at: {NFLVERSE_DB}")
        return 1

    if not status['merged_exists']:
        print(f"ERROR: Merged database not found at: {MERGED_DB}")
        print("Run merge_databases.py first to create the merged database.")
        return 1

    # Show status
    print(f"NFLverse DB: {NFLVERSE_DB.name}")
    print(f"  Last modified: {status['nflverse_modified']}")
    print(f"  Size: {status['nflverse_size_mb']:.2f} MB")
    print()
    print(f"Merged DB: {MERGED_DB.name}")
    print(f"  Last modified: {status['merged_modified']}")
    print(f"  Size: {status['merged_size_mb']:.2f} MB")
    print()

    if not status['needs_refresh']:
        print("INFO: Merged database appears up-to-date.")
        print("NFLverse data has not changed since last refresh.")
        print()
        response = input("Refresh anyway? (y/N): ")
        if response.lower() != 'y':
            print("Refresh cancelled.")
            return 0
    else:
        print("ALERT: Refresh needed - NFLverse has newer data!")

    print()
    print("Starting refresh...")
    print()

    # Progress callback
    def show_progress(table, rows, total_tables, current):
        print(f"  [{current}/{total_tables}] {table:30} {rows:>6} rows updated")

    # Run refresh
    results = refresh_nflverse_tables(season=2025, progress_callback=show_progress)

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
        print('  2. git commit -m "chore: refresh NFLverse data"')
        print("  3. git push")
        return 0
    else:
        print("=" * 60)
        print("REFRESH FAILED!")
        print("=" * 60)
        print(f"Error: {results['error']}")
        print()
        print(f"Backup available at: {results['backup_path']}")
        print("To restore:")
        print(f"  cp {results['backup_path']} {MERGED_DB}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
