"""
NFL Database Merger Script
Merges pfr.db (play-by-play data) with nflverse.sqlite (advanced stats) into nfl_merged.db
"""

import sqlite3
from pathlib import Path
import sys

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Database paths
PFR_DB = Path(r"C:\Docs\_AI Python Projects\Cursor Projects\NFL - Copy\data\pfr.db")
MERGED_DB = Path(r"C:\Docs\_AI Python Projects\Cursor Projects\NFL - Copy\data\nfl_merged.db")

# Team abbreviation mapping: pfr format → nflverse format
TEAM_MAPPING = {
    'GNB': 'GB',   # Green Bay Packers
    'KAN': 'KC',   # Kansas City Chiefs
    'LAR': 'LA',   # Los Angeles Rams
    'LVR': 'LV',   # Las Vegas Raiders
    'NOR': 'NO',   # New Orleans Saints
    'NWE': 'NE',   # New England Patriots
    'SFO': 'SF',   # San Francisco 49ers
    'TAM': 'TB',   # Tampa Bay Buccaneers
}

def convert_team_abbr(abbr):
    """Convert pfr team abbreviation to nflverse format"""
    return TEAM_MAPPING.get(abbr, abbr)

def create_team_mapping_table(conn):
    """Create team abbreviation mapping table in merged database"""
    print("Creating team_abbreviation_mapping table...")

    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS team_abbreviation_mapping (
            pfr_abbr TEXT PRIMARY KEY,
            nflverse_abbr TEXT NOT NULL,
            team_name TEXT
        )
    """)

    # Insert mappings
    mappings = [
        ('GNB', 'GB', 'Green Bay Packers'),
        ('KAN', 'KC', 'Kansas City Chiefs'),
        ('LAR', 'LA', 'Los Angeles Rams'),
        ('LVR', 'LV', 'Las Vegas Raiders'),
        ('NOR', 'NO', 'New Orleans Saints'),
        ('NWE', 'NE', 'New England Patriots'),
        ('SFO', 'SF', 'San Francisco 49ers'),
        ('TAM', 'TB', 'Tampa Bay Buccaneers'),
    ]

    cursor.executemany(
        "INSERT OR REPLACE INTO team_abbreviation_mapping VALUES (?, ?, ?)",
        mappings
    )

    conn.commit()
    print(f"  ✓ Created mapping for {len(mappings)} teams")

def import_plays_table(pfr_conn, merged_conn):
    """Import plays table with team abbreviation conversion"""
    print("\nImporting plays table...")

    pfr_cursor = pfr_conn.cursor()
    merged_cursor = merged_conn.cursor()

    # Get plays table schema from pfr
    pfr_cursor.execute("PRAGMA table_info(plays)")
    columns = pfr_cursor.fetchall()
    column_names = [col[1] for col in columns]
    column_defs = [f"{col[1]} {col[2]}" for col in columns]

    # Create plays table in merged database
    merged_cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS plays (
            {', '.join(column_defs)}
        )
    """)

    # Get all plays
    pfr_cursor.execute("SELECT * FROM plays")
    plays = pfr_cursor.fetchall()

    # Convert team abbreviations
    converted_plays = []
    team_col_indices = []

    # Find columns that contain team abbreviations
    for i, col_name in enumerate(column_names):
        if 'team' in col_name.lower() or col_name in ['posteam', 'defteam', 'home_team', 'away_team']:
            team_col_indices.append(i)

    for play in plays:
        play_list = list(play)
        for idx in team_col_indices:
            if play_list[idx]:
                play_list[idx] = convert_team_abbr(play_list[idx])
        converted_plays.append(tuple(play_list))

    # Insert converted plays
    placeholders = ','.join(['?' for _ in column_names])
    merged_cursor.executemany(
        f"INSERT INTO plays VALUES ({placeholders})",
        converted_plays
    )

    merged_conn.commit()
    print(f"  ✓ Imported {len(converted_plays)} plays with converted team abbreviations")

def import_table(pfr_conn, merged_conn, table_name, convert_teams=False):
    """Import a table from pfr to merged database"""
    print(f"\nImporting {table_name} table...")

    pfr_cursor = pfr_conn.cursor()
    merged_cursor = merged_conn.cursor()

    # Check if table exists
    pfr_cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    if not pfr_cursor.fetchone():
        print(f"  ⚠ Table {table_name} does not exist in pfr.db, skipping")
        return

    # Get table schema
    pfr_cursor.execute(f"PRAGMA table_info({table_name})")
    columns = pfr_cursor.fetchall()
    column_names = [col[1] for col in columns]
    column_defs = [f"{col[1]} {col[2]}" for col in columns]

    # Create table in merged database
    merged_cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {', '.join(column_defs)}
        )
    """)

    # Get all rows
    pfr_cursor.execute(f"SELECT * FROM {table_name}")
    rows = pfr_cursor.fetchall()

    if convert_teams:
        # Convert team abbreviations in team column
        converted_rows = []
        team_col_idx = None
        if 'team' in column_names:
            team_col_idx = column_names.index('team')

        for row in rows:
            row_list = list(row)
            if team_col_idx is not None and row_list[team_col_idx]:
                row_list[team_col_idx] = convert_team_abbr(row_list[team_col_idx])
            converted_rows.append(tuple(row_list))
        rows = converted_rows

    # Insert rows
    if rows:
        placeholders = ','.join(['?' for _ in column_names])
        merged_cursor.executemany(
            f"INSERT INTO {table_name} VALUES ({placeholders})",
            rows
        )
        merged_conn.commit()
        print(f"  ✓ Imported {len(rows)} rows")
    else:
        print(f"  ⚠ No rows to import")

def create_game_id_mapping_view(conn):
    """Create view for easy game_id mapping between formats"""
    print("\nCreating game_id_mapping view...")

    cursor = conn.cursor()
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS game_id_mapping AS
        SELECT
            pfr as pfr_game_id,
            game_id as nflverse_game_id,
            season,
            week,
            home_team,
            away_team
        FROM schedules
        WHERE pfr IS NOT NULL
    """)

    conn.commit()
    print("  ✓ Created game_id_mapping view")

def create_projection_accuracy_table(conn):
    """Create projection_accuracy table that exists in pfr.db"""
    print("\nCreating projection_accuracy table...")

    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS projection_accuracy (
            projection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            team_abbr TEXT NOT NULL,
            opponent_abbr TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            position TEXT NOT NULL,
            projected_yds REAL NOT NULL,
            actual_yds REAL,
            multiplier REAL,
            matchup_rating TEXT,
            avg_yds_game REAL,
            median_yds REAL,
            games_played REAL,
            variance REAL,
            abs_error REAL,
            pct_error REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(player_name, team_abbr, season, week, position)
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_projection_season_week ON projection_accuracy(season, week)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_projection_player ON projection_accuracy(player_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_projection_position ON projection_accuracy(position)")

    conn.commit()
    print("  ✓ Created projection_accuracy table and indexes")

def verify_data_integrity(conn):
    """Verify merged database integrity"""
    print("\n" + "="*60)
    print("DATA INTEGRITY VERIFICATION")
    print("="*60)

    cursor = conn.cursor()

    # Get list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    print(f"\nTotal tables in merged database: {len(tables)}")
    print("\nTable Row Counts:")
    print("-" * 60)

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table:40} {count:>10} rows")

    # Check critical tables
    print("\n" + "="*60)
    print("CRITICAL TABLE CHECKS")
    print("="*60)

    # Check plays table
    cursor.execute("SELECT COUNT(*) FROM plays")
    plays_count = cursor.fetchone()[0]
    print(f"\n✓ Plays table: {plays_count} rows")

    # Check schedules table
    cursor.execute("SELECT COUNT(*) FROM schedules")
    schedules_count = cursor.fetchone()[0]
    print(f"✓ Schedules table: {schedules_count} games")

    # Check advanced stats tables
    adv_tables = [
        'pfr_advstats_pass_week',
        'pfr_advstats_rush_week',
        'pfr_advstats_rec_week',
        'pfr_advstats_def_week'
    ]

    for table in adv_tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"✓ {table}: {count} rows")

    # Sample team abbreviations to verify conversion
    print("\n" + "="*60)
    print("TEAM ABBREVIATION VERIFICATION")
    print("="*60)

    cursor.execute("""
        SELECT DISTINCT home_team FROM schedules
        WHERE home_team IN ('GB', 'KC', 'LA', 'LV', 'NO', 'NE', 'SF', 'TB')
        ORDER BY home_team
    """)
    nflverse_teams = [row[0] for row in cursor.fetchall()]
    print(f"\n✓ NFLverse teams (converted): {', '.join(nflverse_teams)}")

    # Check if any old abbreviations still exist in plays
    old_abbrs = list(TEAM_MAPPING.keys())
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='plays'")
    if cursor.fetchone():
        cursor.execute("""
            SELECT DISTINCT posteam FROM plays
            WHERE posteam IN (?, ?, ?, ?, ?, ?, ?, ?)
        """, old_abbrs)
        remaining_old = [row[0] for row in cursor.fetchall()]
        if remaining_old:
            print(f"\n⚠ WARNING: Old abbreviations still found in plays: {', '.join(remaining_old)}")
        else:
            print(f"\n✓ No old abbreviations found in plays table - conversion successful!")

def main():
    """Main merge process"""
    print("="*60)
    print("NFL DATABASE MERGER")
    print("="*60)
    print(f"\nSource: {PFR_DB}")
    print(f"Target: {MERGED_DB}")
    print()

    # Connect to both databases
    pfr_conn = sqlite3.connect(PFR_DB)
    merged_conn = sqlite3.connect(MERGED_DB)

    try:
        # Step 1: Create team mapping table
        create_team_mapping_table(merged_conn)

        # Step 2: Import plays table with team conversion
        import_plays_table(pfr_conn, merged_conn)

        # Step 3: Import other tables
        import_table(pfr_conn, merged_conn, 'play_participants', convert_teams=False)
        import_table(pfr_conn, merged_conn, 'user_notes', convert_teams=True)
        import_table(pfr_conn, merged_conn, 'player_injuries', convert_teams=True)
        import_table(pfr_conn, merged_conn, 'player_transactions', convert_teams=True)
        import_table(pfr_conn, merged_conn, 'upcoming_games', convert_teams=False)

        # Step 4: Create game_id mapping view
        create_game_id_mapping_view(merged_conn)

        # Step 5: Create projection_accuracy table
        create_projection_accuracy_table(merged_conn)

        # Step 6: Verify data integrity
        verify_data_integrity(merged_conn)

        print("\n" + "="*60)
        print("✓ DATABASE MERGE COMPLETE!")
        print("="*60)
        print(f"\nMerged database created at: {MERGED_DB}")
        print("\nNext steps:")
        print("1. Update pfr_viewer.py to use nfl_merged.db")
        print("2. Convert all team abbreviations in queries")
        print("3. Test existing features")
        print("4. Add new advanced stats visualizations")

    except Exception as e:
        print(f"\n❌ ERROR during merge: {e}")
        import traceback
        traceback.print_exc()

    finally:
        pfr_conn.close()
        merged_conn.close()

if __name__ == "__main__":
    main()
