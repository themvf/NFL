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

def create_games_view(conn):
    """
    Create games VIEW as compatibility layer.
    Maps schedules table to old games table schema for backward compatibility.
    """
    print("\nCreating games view...")

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
    print(f"  ✓ Created games view - {count} games available")

def create_merge_metadata_table(conn):
    """
    Create merge_metadata table to track refresh history.
    """
    print("\nCreating merge_metadata table...")

    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS merge_metadata (
            last_refresh_timestamp TEXT,
            refresh_source TEXT,
            rows_updated INTEGER
        )
    """)

    # Insert initial metadata
    from datetime import datetime
    cursor.execute("""
        INSERT INTO merge_metadata VALUES (?, ?, ?)
    """, (datetime.now().isoformat(), 'initial-merge', 0))

    conn.commit()
    print("  ✓ Created merge_metadata table with initial record")

def create_team_game_summary_view(conn):
    """
    Create team_game_summary VIEW as compatibility layer.
    Maps team_stats_week to old team_game_summary schema for backward compatibility.
    """
    print("\nCreating team_game_summary view...")

    cursor = conn.cursor()

    # Drop view if exists
    cursor.execute("DROP VIEW IF EXISTS team_game_summary")

    # Create compatibility view mapping NFLverse to old schema
    cursor.execute("""
        CREATE VIEW team_game_summary AS
        SELECT
            s.game_id,
            ts.season,
            ts.week,
            ts.team as team_abbr,
            ts.opponent_team as opponent_abbr,
            ts.completions as pass_comp,
            ts.attempts as pass_att,
            ts.passing_yards as pass_yds,
            ts.passing_tds as pass_td,
            ts.passing_interceptions as pass_int,
            ts.sacks_suffered as sacks,
            ts.sack_yards_lost as sack_yds,
            ts.carries as rush_att,
            ts.rushing_yards as rush_yds,
            ts.rushing_tds as rush_td,
            ts.receptions as rec,
            ts.receiving_yards as rec_yds,
            ts.receiving_tds as rec_td,
            (ts.passing_yards + ts.rushing_yards) as yards_total,
            (ts.passing_yards + ts.rushing_yards + ts.receiving_yards) /
                NULLIF(ts.attempts + ts.carries, 0) as yards_per_play,
            CASE
                WHEN ts.team = s.home_team AND s.home_score IS NOT NULL
                    THEN s.home_score
                WHEN ts.team = s.away_team AND s.away_score IS NOT NULL
                    THEN s.away_score
                ELSE (ts.passing_tds + ts.rushing_tds + ts.receiving_tds + ts.special_teams_tds + ts.def_tds) * 7
                    + ts.fg_made * 3
                    + ts.pat_made
            END as points,
            ts.def_sacks as def_sack,
            ts.def_interceptions as def_int,
            ts.def_fumbles as def_fum,
            ts.def_tds as def_td,
            ts.penalties as pen,
            ts.penalty_yards as pen_yds,
            (ts.passing_interceptions + ts.rushing_fumbles_lost + ts.receiving_fumbles_lost) as turnovers,
            (ts.attempts + ts.carries) as plays,
            (ts.passing_first_downs + ts.rushing_first_downs + ts.receiving_first_downs) as first_downs,
            NULL as third_down_conv,
            NULL as third_down_att,
            NULL as fourth_down_conv,
            NULL as fourth_down_att,
            NULL as time_of_poss,
            ts.fg_made,
            ts.fg_att as fg_attempted,
            ts.season_type
        FROM team_stats_week ts
        LEFT JOIN schedules s ON
            ts.season = s.season
            AND ts.week = s.week
            AND ts.team IN (s.home_team, s.away_team)
            AND ts.opponent_team IN (s.home_team, s.away_team)
            AND ts.team != ts.opponent_team
        WHERE ts.season_type = 'REG'
    """)

    conn.commit()

    # Verify view was created
    cursor.execute("SELECT COUNT(*) FROM team_game_summary")
    count = cursor.fetchone()[0]
    print(f"  ✓ Created team_game_summary view - {count} team-game records")

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

        # Step 5: Create games view (compatibility layer)
        create_games_view(merged_conn)

        # Step 6: Create projection_accuracy table
        create_projection_accuracy_table(merged_conn)

        # Step 7: Create merge_metadata table
        create_merge_metadata_table(merged_conn)

        # Step 8: Create team_game_summary compatibility view
        create_team_game_summary_view(merged_conn)

        # Step 9: Verify data integrity
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
