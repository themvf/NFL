"""
Create player_box_score compatibility view

Maps NFLverse player_stats table to the old player_box_score schema
for backward compatibility with existing queries.
"""

import sqlite3
from pathlib import Path
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


def create_player_box_score_view(conn):
    """
    Create player_box_score VIEW as compatibility layer.
    Maps player_stats to old player_box_score schema.
    """
    print("\nCreating 'player_box_score' view...")

    cursor = conn.cursor()

    # First, drop the view if it exists
    cursor.execute("DROP VIEW IF EXISTS player_box_score")

    # Create the compatibility view
    # Map NFLverse columns to old pfr.db column names
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

            -- Advanced metrics (NFLverse bonus columns)
            passing_epa,
            rushing_epa,
            receiving_epa,
            racr,
            target_share,
            air_yards_share,
            wopr,

            -- Special teams
            special_teams_tds as st_td,

            -- Total stats (calculated)
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
    print(f"  ✅ Created 'player_box_score' view - {count} player-game records available")

    # Show sample data
    cursor.execute("""
        SELECT player, team, season, week, pass_yds, rush_yds, rec_yds
        FROM player_box_score
        WHERE season = 2025
        ORDER BY (pass_yds + rush_yds + rec_yds) DESC
        LIMIT 5
    """)
    print(f"\n  Sample data (top performances):")
    for row in cursor.fetchall():
        print(f"    {row}")

    # Show player count by position
    cursor.execute("""
        SELECT position_group, COUNT(DISTINCT player) as player_count
        FROM player_box_score
        WHERE season = 2025
        GROUP BY position_group
        ORDER BY player_count DESC
    """)
    print(f"\n  Players by position (2025):")
    for row in cursor.fetchall():
        print(f"    {row[0]}: {row[1]} players")

    return True


def main():
    """Main creation process"""
    print("=" * 60)
    print("CREATE PLAYER_BOX_SCORE COMPATIBILITY VIEW")
    print("=" * 60)
    print(f"\nDatabase: {MERGED_DB}")

    if not MERGED_DB.exists():
        print(f"\n❌ ERROR: Database not found at {MERGED_DB}")
        print("Please run merge_databases.py first.")
        return 1

    try:
        # Connect to database
        conn = sqlite3.connect(MERGED_DB)

        # Create view
        create_player_box_score_view(conn)

        conn.close()

        print("\n" + "=" * 60)
        print("✅ VIEW CREATION COMPLETE!")
        print("=" * 60)
        print("\nThe player_box_score view is now available.")
        print("All existing queries using player_box_score will work.")
        print("\nNext steps:")
        print("  1. Re-enable explosive play calculations")
        print("  2. Test Upcoming Matches page")
        print("  3. Test Player Stats and Season Leaderboards")
        print("  4. Commit the updated database to Git")
        return 0

    except Exception as e:
        print(f"\n❌ ERROR during view creation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
