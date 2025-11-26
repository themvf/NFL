"""
Sync player_stats from nfl_merged.db to nfl_stats.db for Player Impact Analysis.

This ensures the Player Impact feature has access to the latest 2025 season data.
"""

import sqlite3
import pandas as pd
from pathlib import Path

# Database paths
PROJECT_DIR = Path(__file__).parent
MERGED_DB = PROJECT_DIR / "data" / "nfl_merged.db"
STATS_DB = PROJECT_DIR / "nfl_stats.db"

def sync_player_stats():
    """Copy player_stats from merged db to stats db for Player Impact."""

    print("="*60)
    print("SYNCING PLAYER STATS TO PLAYER IMPACT DATABASE")
    print("="*60)

    # Connect to both databases
    merged_conn = sqlite3.connect(str(MERGED_DB))
    stats_conn = sqlite3.connect(str(STATS_DB))

    try:
        # Read player_stats from merged db
        print("\nReading player_stats from nfl_merged.db...")
        player_stats_df = pd.read_sql_query(
            "SELECT * FROM player_stats",
            merged_conn
        )

        print(f"Found {len(player_stats_df)} player-week records")
        print(f"Seasons: {player_stats_df['season'].unique().tolist()}")

        # Check 2025 data
        df_2025 = player_stats_df[player_stats_df['season'] == 2025]
        print(f"\n2025 Season:")
        print(f"  - {len(df_2025)} player-week records")
        print(f"  - Weeks {df_2025['week'].min()} through {df_2025['week'].max()}")
        print(f"  - {df_2025['player_id'].nunique()} unique players")

        # Map column names from player_stats to player_week_stats
        # The schemas are similar but column names might differ
        column_mapping = {
            'team': 'recent_team',  # IMPORTANT: source uses 'team', target uses 'recent_team'
            'position': 'position',
            'player_id': 'player_id',
            'player_name': 'player_display_name',
            'player_display_name': 'player_display_name',
            'season': 'season',
            'week': 'week',
            'season_type': 'season_type',
            'completions': 'completions',
            'attempts': 'attempts',
            'passing_yards': 'passing_yards',
            'passing_tds': 'passing_tds',
            'interceptions': 'interceptions',
            'sacks': 'sacks',
            'sack_yards': 'sack_yards',
            'sack_fumbles': 'sack_fumbles',
            'sack_fumbles_lost': 'sack_fumbles_lost',
            'passing_air_yards': 'passing_air_yards',
            'passing_yards_after_catch': 'passing_yards_after_catch',
            'passing_first_downs': 'passing_first_downs',
            'passing_epa': 'passing_epa',
            'passing_2pt_conversions': 'passing_2pt_conversions',
            'pacr': 'pacr',
            'dakota': 'dakota',
            'carries': 'carries',
            'rushing_yards': 'rushing_yards',
            'rushing_tds': 'rushing_tds',
            'rushing_fumbles': 'rushing_fumbles',
            'rushing_fumbles_lost': 'rushing_fumbles_lost',
            'rushing_first_downs': 'rushing_first_downs',
            'rushing_epa': 'rushing_epa',
            'rushing_2pt_conversions': 'rushing_2pt_conversions',
            'receptions': 'receptions',
            'targets': 'targets',
            'receiving_yards': 'receiving_yards',
            'receiving_tds': 'receiving_tds',
            'receiving_fumbles': 'receiving_fumbles',
            'receiving_fumbles_lost': 'receiving_fumbles_lost',
            'receiving_air_yards': 'receiving_air_yards',
            'receiving_yards_after_catch': 'receiving_yards_after_catch',
            'receiving_first_downs': 'receiving_first_downs',
            'receiving_epa': 'receiving_epa',
            'receiving_2pt_conversions': 'receiving_2pt_conversions',
            'racr': 'racr',
            'target_share': 'target_share',
            'air_yards_share': 'air_yards_share',
            'wopr': 'wopr',
            'special_teams_tds': 'special_teams_tds',
            'fantasy_points': 'fantasy_points',
            'fantasy_points_ppr': 'fantasy_points_ppr',
        }

        # Rename columns
        renamed_df = player_stats_df.rename(columns=column_mapping)

        # Add missing columns with default values if needed
        required_columns = [
            'player_id', 'season', 'week', 'season_type', 'player_display_name',
            'position', 'recent_team', 'completions', 'attempts', 'passing_yards',
            'passing_tds', 'interceptions', 'carries', 'rushing_yards', 'rushing_tds',
            'receptions', 'targets', 'receiving_yards', 'receiving_tds',
            'fantasy_points', 'fantasy_points_ppr'
        ]

        for col in required_columns:
            if col not in renamed_df.columns:
                renamed_df[col] = None

        # Select only the columns that exist in player_week_stats schema
        final_df = renamed_df[required_columns]

        # Filter out rows with NULL player_id (NOT NULL constraint)
        before_count = len(final_df)
        final_df = final_df[final_df['player_id'].notna()]
        after_count = len(final_df)
        if before_count > after_count:
            print(f"Filtered out {before_count - after_count} rows with NULL player_id")

        # Delete existing data from nfl_stats.db
        print("\nDeleting old player_week_stats from nfl_stats.db...")
        stats_conn.execute("DELETE FROM player_week_stats")
        stats_conn.commit()

        # Insert new data
        print(f"Inserting {len(final_df)} rows into player_week_stats...")
        final_df.to_sql(
            'player_week_stats',
            stats_conn,
            if_exists='append',
            index=False,
            chunksize=100
        )

        # Verify
        cursor = stats_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM player_week_stats WHERE season = 2025")
        count_2025 = cursor.fetchone()[0]

        cursor.execute("SELECT DISTINCT week FROM player_week_stats WHERE season = 2025 ORDER BY week")
        weeks_2025 = [row[0] for row in cursor.fetchall()]

        print("\n" + "="*60)
        print("SYNC COMPLETE!")
        print("="*60)
        print(f"Total rows in player_week_stats: {len(final_df)}")
        print(f"2025 season rows: {count_2025}")
        print(f"2025 weeks available: {weeks_2025}")
        print("\nPlayer Impact Analysis is now ready for 2025 season!")

    finally:
        merged_conn.close()
        stats_conn.close()

if __name__ == "__main__":
    sync_player_stats()
