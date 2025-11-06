"""
Fix Missing Tables/Views in NFL Merged Database

This script adds the missing 'games' view and 'merge_metadata' table
that were documented but not created during the initial merge.

Run this once to fix the database.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

# Database path
PROJECT_DIR = Path(__file__).parent
MERGED_DB = PROJECT_DIR / "data" / "nfl_merged.db"


def create_games_view(conn):
    """
    Create games VIEW as compatibility layer.
    Maps schedules table to old games table schema.
    """
    print("\nCreating 'games' view...")

    cursor = conn.cursor()
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS games AS
        SELECT
            game_id,
            gameday as game_date,
            season as season_year,
            home_team,
            away_team,
            home_score,
            away_score,
            week,
            game_type,
            weekday,
            gametime,
            stadium,
            roof,
            surface,
            temp,
            wind,
            home_rest,
            away_rest,
            div_game
        FROM schedules
    """)

    conn.commit()

    # Verify view was created
    cursor.execute("SELECT COUNT(*) FROM games")
    count = cursor.fetchone()[0]

    print(f"  ✅ Created 'games' view - {count} games available")
    return True


def create_merge_metadata_table(conn):
    """
    Create merge_metadata table to track refresh history.
    """
    print("\nCreating 'merge_metadata' table...")

    cursor = conn.cursor()

    # Check if table already exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='merge_metadata'
    """)

    if cursor.fetchone():
        print("  ℹ️  Table already exists, skipping creation")
        return True

    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS merge_metadata (
            last_refresh_timestamp TEXT,
            refresh_source TEXT,
            rows_updated INTEGER
        )
    """)

    # Insert initial metadata
    cursor.execute("""
        INSERT INTO merge_metadata VALUES (?, ?, ?)
    """, (datetime.now().isoformat(), 'initial-merge', 0))

    conn.commit()

    print(f"  ✅ Created 'merge_metadata' table with initial record")
    return True


def verify_fixes(conn):
    """
    Verify that both objects were created successfully.
    """
    print("\nVerifying fixes...")

    cursor = conn.cursor()
    issues = []

    # Check games view
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='view' AND name='games'
    """)
    if cursor.fetchone():
        cursor.execute("SELECT COUNT(*) FROM games")
        count = cursor.fetchone()[0]
        print(f"  ✅ games view exists - {count} rows")
    else:
        print(f"  ❌ games view NOT found")
        issues.append("games view missing")

    # Check merge_metadata table
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='merge_metadata'
    """)
    if cursor.fetchone():
        cursor.execute("SELECT COUNT(*) FROM merge_metadata")
        count = cursor.fetchone()[0]
        print(f"  ✅ merge_metadata table exists - {count} rows")
    else:
        print(f"  ❌ merge_metadata table NOT found")
        issues.append("merge_metadata table missing")

    return len(issues) == 0, issues


def main():
    """Main migration process"""
    print("=" * 60)
    print("FIX MISSING TABLES/VIEWS MIGRATION")
    print("=" * 60)
    print(f"\nDatabase: {MERGED_DB}")

    if not MERGED_DB.exists():
        print(f"\n❌ ERROR: Database not found at {MERGED_DB}")
        print("Please run merge_databases.py first.")
        return 1

    try:
        # Connect to database
        conn = sqlite3.connect(MERGED_DB)

        # Apply fixes
        create_games_view(conn)
        create_merge_metadata_table(conn)

        # Verify
        success, issues = verify_fixes(conn)

        conn.close()

        print("\n" + "=" * 60)
        if success:
            print("✅ MIGRATION COMPLETE!")
            print("=" * 60)
            print("\nAll missing objects have been created.")
            print("\nNext steps:")
            print("  1. Run database validation in Streamlit")
            print("  2. Verify 'games' view works in queries")
            print("  3. Commit the updated database to Git")
            return 0
        else:
            print("❌ MIGRATION FAILED!")
            print("=" * 60)
            print(f"\nIssues: {', '.join(issues)}")
            return 1

    except Exception as e:
        print(f"\n❌ ERROR during migration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
