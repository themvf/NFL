"""
Create team_game_summary compatibility view

Maps NFLverse team_stats_week table to the old team_game_summary schema
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


def create_team_game_summary_view(conn):
    """
    Create team_game_summary VIEW as compatibility layer.
    Maps team_stats_week and schedules to old team_game_summary schema.
    """
    print("\nCreating 'team_game_summary' view...")

    cursor = conn.cursor()

    # First, drop the view if it exists
    cursor.execute("DROP VIEW IF EXISTS team_game_summary")

    # Create the compatibility view
    # Map NFLverse columns to old pfr.db column names
    cursor.execute("""
        CREATE VIEW team_game_summary AS
        SELECT
            -- Game identifiers
            s.game_id,
            ts.season,
            ts.week,
            ts.team as team_abbr,
            ts.opponent_team as opponent_abbr,

            -- Passing stats
            ts.completions as pass_comp,
            ts.attempts as pass_att,
            ts.passing_yards as pass_yds,
            ts.passing_tds as pass_td,
            ts.passing_interceptions as pass_int,
            ts.sacks_suffered as sacks,
            ts.sack_yards_lost as sack_yds,

            -- Rushing stats
            ts.carries as rush_att,
            ts.rushing_yards as rush_yds,
            ts.rushing_tds as rush_td,

            -- Receiving stats (for team totals)
            ts.receptions as rec,
            ts.receiving_yards as rec_yds,
            ts.receiving_tds as rec_td,

            -- Total yards and points
            (ts.passing_yards + ts.rushing_yards) as yards_total,
            (ts.passing_yards + ts.rushing_yards + ts.receiving_yards) /
                NULLIF(ts.attempts + ts.carries, 0) as yards_per_play,

            -- Calculate points from TDs (estimate - actual points from schedules)
            CASE
                WHEN ts.team = s.home_team AND s.home_score IS NOT NULL
                    THEN s.home_score
                WHEN ts.team = s.away_team AND s.away_score IS NOT NULL
                    THEN s.away_score
                ELSE (ts.passing_tds + ts.rushing_tds + ts.receiving_tds + ts.special_teams_tds + ts.def_tds) * 7
                    + ts.fg_made * 3
                    + ts.pat_made
            END as points,

            -- Defensive stats
            ts.def_sacks as def_sack,
            ts.def_interceptions as def_int,
            ts.def_fumbles as def_fum,
            ts.def_tds as def_td,

            -- Special teams
            ts.penalties as pen,
            ts.penalty_yards as pen_yds,

            -- Turnovers
            (ts.passing_interceptions + ts.rushing_fumbles_lost + ts.receiving_fumbles_lost) as turnovers,

            -- Plays (estimated)
            (ts.attempts + ts.carries) as plays,

            -- First downs (total)
            (ts.passing_first_downs + ts.rushing_first_downs + ts.receiving_first_downs) as first_downs,

            -- Third down conversions (not available - set to NULL)
            NULL as third_down_conv,
            NULL as third_down_att,

            -- Fourth down conversions (not available - set to NULL)
            NULL as fourth_down_conv,
            NULL as fourth_down_att,

            -- Time of possession (not available - set to NULL)
            NULL as time_of_poss,

            -- Field goals
            ts.fg_made,
            ts.fg_att as fg_attempted,

            -- Metadata
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
    print(f"  ✅ Created 'team_game_summary' view - {count} team-game records available")

    # Show sample data
    cursor.execute("""
        SELECT team_abbr, season, week, pass_yds, rush_yds, points
        FROM team_game_summary
        WHERE season = 2025
        LIMIT 3
    """)
    print(f"\n  Sample data:")
    for row in cursor.fetchall():
        print(f"    {row}")

    return True


def main():
    """Main creation process"""
    print("=" * 60)
    print("CREATE TEAM_GAME_SUMMARY COMPATIBILITY VIEW")
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
        create_team_game_summary_view(conn)

        conn.close()

        print("\n" + "=" * 60)
        print("✅ VIEW CREATION COMPLETE!")
        print("=" * 60)
        print("\nThe team_game_summary view is now available.")
        print("All existing queries using team_game_summary will work.")
        print("\nNext steps:")
        print("  1. Test Upcoming Matches page")
        print("  2. Test other views that use team statistics")
        print("  3. Commit the updated database to Git")
        return 0

    except Exception as e:
        print(f"\n❌ ERROR during view creation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
