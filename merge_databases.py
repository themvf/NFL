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
    Create ENHANCED games VIEW as compatibility layer.
    Maps schedules table to old games table schema with additional columns
    including stat columns from team_game_summary for defensive rankings.
    """
    print("\nCreating enhanced games view...")

    cursor = conn.cursor()

    # Drop existing view if it exists
    cursor.execute("DROP VIEW IF EXISTS games")

    cursor.execute("""
        CREATE VIEW games AS
        SELECT
            -- Core game identifiers (using legacy column names for compatibility)
            s.game_id,
            s.season,
            s.week,
            s.gameday as game_date,
            s.home_team as home_team_abbr,
            s.away_team as away_team_abbr,
            CAST(s.home_score AS INTEGER) as home_score,
            CAST(s.away_score AS INTEGER) as away_score,

            -- Game details from schedules
            s.location,
            s.game_type,
            s.overtime,
            s.total,
            s.stadium,
            s.roof,
            s.surface,
            s.temp,
            s.wind,
            s.div_game,
            s.weekday,
            s.gametime,

            -- Betting odds
            s.spread_line,
            s.total_line,
            s.home_moneyline,
            s.away_moneyline,
            s.home_rest,
            s.away_rest,

            -- Team stats from team_game_summary (for defensive rankings)
            (SELECT rush_yds FROM team_game_summary
             WHERE game_id = s.game_id AND team_abbr = s.home_team LIMIT 1) as home_rushing_yds,
            (SELECT rush_yds FROM team_game_summary
             WHERE game_id = s.game_id AND team_abbr = s.away_team LIMIT 1) as away_rushing_yds,
            (SELECT pass_yds FROM team_game_summary
             WHERE game_id = s.game_id AND team_abbr = s.home_team LIMIT 1) as home_passing_yds,
            (SELECT pass_yds FROM team_game_summary
             WHERE game_id = s.game_id AND team_abbr = s.away_team LIMIT 1) as away_passing_yds,

            -- Legacy columns
            NULL as source_url,
            NULL as last_updated
        FROM schedules s
    """)

    conn.commit()

    # Verify view was created
    cursor.execute("SELECT COUNT(*) FROM games")
    count = cursor.fetchone()[0]
    print(f"  ✓ Created enhanced games view - {count} games with 30+ columns")

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


def create_player_box_score_view(conn):
    """Create player_box_score VIEW as compatibility layer for player stats"""
    print("\n  Creating player_box_score view...")
    cursor = conn.cursor()

    # Drop existing view if it exists
    cursor.execute("DROP VIEW IF EXISTS player_box_score")

    # Create compatibility view mapping player_stats to old schema
    cursor.execute("""
        CREATE VIEW player_box_score AS
        SELECT
            -- Player identifiers
            player_name as player,
            player_display_name as player_display,
            player_id,
            position,
            position_group,

            -- Game identifiers
            season,
            week,
            season_type,
            team,
            opponent_team as opponent,

            -- Passing stats
            completions as pass_comp,
            attempts as pass_att,
            passing_yards as pass_yds,
            passing_tds as pass_td,
            passing_interceptions as pass_int,
            sacks_suffered as sacks,
            sack_yards_lost as sack_yds,

            -- Rushing stats
            carries as rush_att,
            rushing_yards as rush_yds,
            rushing_tds as rush_td,
            rushing_fumbles as rush_fum,
            rushing_fumbles_lost as rush_fum_lost,
            rushing_first_downs as rush_first_downs,

            -- Receiving stats
            receptions as rec,
            targets,
            receiving_yards as rec_yds,
            receiving_tds as rec_td,
            receiving_fumbles as rec_fum,
            receiving_fumbles_lost as rec_fum_lost,
            receiving_first_downs as rec_first_downs,
            receiving_air_yards as air_yards,
            receiving_yards_after_catch as yac,

            -- Fantasy points
            fantasy_points,
            fantasy_points_ppr,

            -- Advanced metrics
            passing_epa,
            rushing_epa,
            receiving_epa,
            racr,
            target_share,
            air_yards_share,
            wopr,

            -- Special teams
            special_teams_tds as st_td,

            -- Total stats
            (passing_yards + rushing_yards + receiving_yards) as total_yds,
            (passing_tds + rushing_tds + receiving_tds) as total_td,
            (rushing_fumbles_lost + receiving_fumbles_lost + sack_fumbles_lost) as total_fum_lost,

            -- Two-point conversions
            passing_2pt_conversions as pass_2pt,
            rushing_2pt_conversions as rush_2pt,
            receiving_2pt_conversions as rec_2pt

        FROM player_stats
        WHERE season_type = 'REG'
    """)

    conn.commit()

    # Verify view was created
    cursor.execute("SELECT COUNT(*) FROM player_box_score")
    count = cursor.fetchone()[0]
    print(f"  ✓ Created player_box_score view - {count} player-game records")


def create_season_leader_views(conn):
    """Create all season leader aggregation views from player_box_score."""
    print("\nCreating season leader views...")
    cursor = conn.cursor()

    # Rushing leaders
    cursor.execute("DROP VIEW IF EXISTS rushing_leaders")
    cursor.execute("""
        CREATE VIEW rushing_leaders AS
        SELECT
            player, team, season,
            COUNT(DISTINCT week) as games_played,
            SUM(rush_att) as total_rush_att,
            SUM(rush_yds) as total_rush_yds,
            SUM(rush_td) as total_rush_td,
            ROUND(SUM(rush_yds) * 1.0 / NULLIF(SUM(rush_att), 0), 1) as avg_yds_per_carry,
            ROUND(SUM(rush_yds) * 1.0 / NULLIF(COUNT(DISTINCT week), 0), 1) as yds_per_game
        FROM player_box_score
        WHERE rush_att > 0
        GROUP BY player, team, season
    """)

    # Receiving leaders
    cursor.execute("DROP VIEW IF EXISTS receiving_leaders")
    cursor.execute("""
        CREATE VIEW receiving_leaders AS
        SELECT
            player, team, season,
            COUNT(DISTINCT week) as games_played,
            SUM(rec) as total_rec,
            SUM(targets) as total_targets,
            SUM(rec_yds) as total_rec_yds,
            SUM(rec_td) as total_rec_td,
            ROUND(SUM(rec_yds) * 1.0 / NULLIF(SUM(rec), 0), 1) as avg_yds_per_rec,
            ROUND(SUM(rec_yds) * 1.0 / NULLIF(COUNT(DISTINCT week), 0), 1) as yds_per_game,
            ROUND(SUM(rec) * 100.0 / NULLIF(SUM(targets), 0), 1) as catch_pct
        FROM player_box_score
        WHERE targets > 0 OR rec > 0
        GROUP BY player, team, season
    """)

    # Passing leaders
    cursor.execute("DROP VIEW IF EXISTS passing_leaders")
    cursor.execute("""
        CREATE VIEW passing_leaders AS
        SELECT
            player, team, season,
            COUNT(DISTINCT week) as games_played,
            SUM(pass_att) as total_pass_att,
            SUM(pass_comp) as total_pass_comp,
            SUM(pass_yds) as total_pass_yds,
            SUM(pass_td) as total_pass_td,
            SUM(pass_int) as total_pass_int,
            ROUND(SUM(pass_comp) * 100.0 / NULLIF(SUM(pass_att), 0), 1) as comp_pct,
            ROUND(SUM(pass_yds) * 1.0 / NULLIF(SUM(pass_att), 0), 1) as yds_per_att,
            ROUND(SUM(pass_yds) * 1.0 / NULLIF(COUNT(DISTINCT week), 0), 1) as yds_per_game
        FROM player_box_score
        WHERE pass_att > 0
        GROUP BY player, team, season
    """)

    # Touchdown leaders
    cursor.execute("DROP VIEW IF EXISTS touchdown_leaders")
    cursor.execute("""
        CREATE VIEW touchdown_leaders AS
        SELECT
            player, team, season,
            COUNT(DISTINCT week) as games_played,
            SUM(rush_td) as rush_tds,
            SUM(rec_td) as rec_tds,
            SUM(pass_td) as pass_tds,
            SUM(st_td) as st_tds,
            SUM(rush_td + rec_td + pass_td + st_td) as total_touchdowns,
            ROUND(SUM(rush_td + rec_td + pass_td + st_td) * 1.0 / NULLIF(COUNT(DISTINCT week), 0), 2) as tds_per_game
        FROM player_box_score
        WHERE (rush_td + rec_td + pass_td + st_td) > 0
        GROUP BY player, team, season
    """)

    conn.commit()

    # Verify views
    cursor.execute("SELECT COUNT(*) FROM rushing_leaders")
    rush_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM receiving_leaders")
    rec_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM passing_leaders")
    pass_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM touchdown_leaders")
    td_count = cursor.fetchone()[0]

    print(f"  ✓ Created rushing_leaders - {rush_count} records")
    print(f"  ✓ Created receiving_leaders - {rec_count} records")
    print(f"  ✓ Created passing_leaders - {pass_count} records")
    print(f"  ✓ Created touchdown_leaders - {td_count} records")


def create_touchdown_scorers_view(conn):
    """Create touchdown_scorers view for TD analysis."""
    print("\nCreating touchdown_scorers view...")
    cursor = conn.cursor()

    cursor.execute("DROP VIEW IF EXISTS touchdown_scorers")
    cursor.execute("""
        CREATE VIEW touchdown_scorers AS
        SELECT
            player_id || '_' || season || '_W' || week || '_rush' as td_id,
            COALESCE(
                (SELECT game_id FROM schedules
                 WHERE season = pbs.season AND week = pbs.week
                 AND (home_team = pbs.team OR away_team = pbs.team) LIMIT 1),
                season || '_W' || week || '_' || team
            ) as game_id,
            player, team, season, week,
            'Rushing' as touchdown_type,
            rush_td as td_count,
            0 as first_td_game, 0 as first_td_for_team
        FROM player_box_score pbs
        WHERE rush_td > 0
        UNION ALL
        SELECT
            player_id || '_' || season || '_W' || week || '_rec' as td_id,
            COALESCE(
                (SELECT game_id FROM schedules
                 WHERE season = pbs.season AND week = pbs.week
                 AND (home_team = pbs.team OR away_team = pbs.team) LIMIT 1),
                season || '_W' || week || '_' || team
            ) as game_id,
            player, team, season, week,
            'Receiving' as touchdown_type,
            rec_td as td_count,
            0 as first_td_game, 0 as first_td_for_team
        FROM player_box_score pbs
        WHERE rec_td > 0
        UNION ALL
        SELECT
            player_id || '_' || season || '_W' || week || '_pass' as td_id,
            COALESCE(
                (SELECT game_id FROM schedules
                 WHERE season = pbs.season AND week = pbs.week
                 AND (home_team = pbs.team OR away_team = pbs.team) LIMIT 1),
                season || '_W' || week || '_' || team
            ) as game_id,
            player, team, season, week,
            'Passing' as touchdown_type,
            pass_td as td_count,
            0 as first_td_game, 0 as first_td_for_team
        FROM player_box_score pbs
        WHERE pass_td > 0
    """)

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM touchdown_scorers")
    count = cursor.fetchone()[0]
    print(f"  ✓ Created touchdown_scorers view - {count} TD events")


def create_first_td_game_leaders_view(conn):
    """Create first_td_game_leaders placeholder view."""
    print("\nCreating first_td_game_leaders view (placeholder)...")
    cursor = conn.cursor()

    cursor.execute("DROP VIEW IF EXISTS first_td_game_leaders")
    cursor.execute("""
        CREATE VIEW first_td_game_leaders AS
        SELECT
            player, team, season,
            0 as first_td_count,
            'Requires play-by-play parsing' as notes
        FROM player_box_score
        WHERE (rush_td + rec_td + pass_td) > 0
        GROUP BY player, team, season
    """)

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM first_td_game_leaders")
    count = cursor.fetchone()[0]
    print(f"  ✓ Created first_td_game_leaders view (placeholder) - {count} records")


def create_box_score_summary_view(conn):
    """Create box_score_summary as alias to team_game_summary."""
    print("\nCreating box_score_summary view (alias)...")
    cursor = conn.cursor()

    cursor.execute("DROP VIEW IF EXISTS box_score_summary")
    cursor.execute("CREATE VIEW box_score_summary AS SELECT * FROM team_game_summary")

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM box_score_summary")
    count = cursor.fetchone()[0]
    print(f"  ✓ Created box_score_summary view (alias) - {count} records")


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
            SELECT DISTINCT posteam_abbr FROM plays
            WHERE posteam_abbr IN (?, ?, ?, ?, ?, ?, ?, ?)
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

        # Step 9: Create player_box_score compatibility view
        create_player_box_score_view(merged_conn)

        # Step 10: Create season leader views (rushing, receiving, passing, touchdown)
        create_season_leader_views(merged_conn)

        # Step 11: Create touchdown_scorers view for TD analysis
        create_touchdown_scorers_view(merged_conn)

        # Step 12: Create first_td_game_leaders view (placeholder)
        create_first_td_game_leaders_view(merged_conn)

        # Step 13: Create box_score_summary view (alias to team_game_summary)
        create_box_score_summary_view(merged_conn)

        # Step 14: Verify data integrity
        verify_data_integrity(merged_conn)

        print("\n" + "="*60)
        print("✓ DATABASE MERGE COMPLETE!")
        print("="*60)
        print(f"\nMerged database created at: {MERGED_DB}")
        print("\nCompatibility Views Created:")
        print("  ✅ Enhanced games view (30+ columns)")
        print("  ✅ team_game_summary (team stats)")
        print("  ✅ player_box_score (player stats)")
        print("  ✅ rushing_leaders, receiving_leaders, passing_leaders")
        print("  ✅ touchdown_leaders, touchdown_scorers")
        print("  ✅ first_td_game_leaders (placeholder)")
        print("  ✅ box_score_summary (alias)")
        print("\nNext steps:")
        print("1. Test Season Leaders page")
        print("2. Test Touchdown Analysis features")
        print("3. Test Upcoming Matches page")
        print("4. Commit and push to Git")

    except Exception as e:
        print(f"\n❌ ERROR during merge: {e}")
        import traceback
        traceback.print_exc()

    finally:
        pfr_conn.close()
        merged_conn.close()

if __name__ == "__main__":
    main()
