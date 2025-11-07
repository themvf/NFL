"""
Create player_stats table for NFLverse player statistics

This script creates the player_stats table in the merged database
by fetching a sample from the NFLverse API to get the schema.
"""

import sqlite3
from pathlib import Path
import sys

try:
    import pandas as pd
    import nflreadpy as nfl
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Run: pip install nflreadpy pandas polars")
    sys.exit(1)

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

# Database path
PROJECT_DIR = Path(__file__).parent
MERGED_DB = PROJECT_DIR / "data" / "nfl_merged.db"


def create_player_stats_table(conn):
    """
    Create player_stats table using schema from NFLverse API.

    Fetches a small sample from the API to determine column names and types,
    then creates the table structure.
    """
    print("\nCreating 'player_stats' table...")

    cursor = conn.cursor()

    # Check if table already exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='player_stats'
    """)

    if cursor.fetchone():
        print("  ⚠ Table 'player_stats' already exists")
        response = input("  Drop and recreate? (y/n): ").strip().lower()
        if response == 'y':
            cursor.execute("DROP TABLE player_stats")
            print("  ✓ Dropped existing table")
        else:
            print("  ⊗ Keeping existing table")
            return False

    # Fetch a sample to get schema (just latest season, small amount)
    print("  Fetching schema from NFLverse API...")
    try:
        df = nfl.load_player_stats(2024)  # Use 2024 which is complete
        if hasattr(df, 'to_pandas'):
            df = df.to_pandas()

        # Take just one row to create schema
        sample_df = df.head(1)

        # Create table using pandas to_sql
        sample_df.to_sql('player_stats', conn, if_exists='replace', index=False)

        # Verify table was created
        cursor.execute("SELECT COUNT(*) FROM player_stats")
        count = cursor.fetchone()[0]

        # Clear the sample data
        cursor.execute("DELETE FROM player_stats")

        print(f"  ✓ Created 'player_stats' table with {len(df.columns)} columns")
        print(f"  Sample columns: {', '.join(df.columns[:10])}...")

        # Create indexes for better query performance
        print("  Creating indexes...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_player_stats_player
            ON player_stats(player_id, season, week)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_player_stats_team
            ON player_stats(team, season, week)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_player_stats_season
            ON player_stats(season, week)
        """)
        print("  ✓ Created performance indexes")

        conn.commit()
        return True

    except Exception as e:
        print(f"  ❌ Error creating table: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main creation process"""
    print("=" * 60)
    print("CREATE PLAYER_STATS TABLE")
    print("=" * 60)
    print(f"\nDatabase: {MERGED_DB}")

    if not MERGED_DB.exists():
        print(f"\n❌ ERROR: Database not found at {MERGED_DB}")
        print("Please run merge_databases.py first.")
        return 1

    try:
        # Connect to database
        conn = sqlite3.connect(MERGED_DB)

        # Create table
        success = create_player_stats_table(conn)

        if success:
            # Verify structure
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(player_stats)")
            columns = cursor.fetchall()

            print(f"\n  Table structure: {len(columns)} columns")
            print("\n  First 10 columns:")
            for col in columns[:10]:
                print(f"    - {col[1]} ({col[2]})")

        conn.close()

        if success:
            print("\n" + "=" * 60)
            print("✅ TABLE CREATION COMPLETE!")
            print("=" * 60)
            print("\nThe player_stats table is now ready.")
            print("Run nflverse_direct_refresh.py to populate with data.")
            return 0
        else:
            print("\n" + "=" * 60)
            print("⊗ TABLE CREATION SKIPPED")
            print("=" * 60)
            return 0

    except Exception as e:
        print(f"\n❌ ERROR during table creation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
