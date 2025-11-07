"""
Create Comprehensive Compatibility Views for PFR-to-NFLverse Migration

This script creates all missing tables/views required by the application:
- Enhanced games view with missing columns
- Season leader views (rushing, receiving, passing, touchdown)
- touchdown_scorers view for TD analysis
- first_td_game_leaders view (placeholder)

Fixes 7 missing tables and adds critical columns to games view.
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


def create_enhanced_games_view(conn):
    """
    Recreate games VIEW with all missing columns from schedules
    plus calculated stat columns from team_game_summary.
    """
    print("\n  Creating enhanced 'games' view...")
    cursor = conn.cursor()

    # Drop existing view
    cursor.execute("DROP VIEW IF EXISTS games")

    # Create enhanced view with all required columns
    cursor.execute("""
        CREATE VIEW games AS
        SELECT
            -- Core game identifiers
            s.pfr as game_id,
            s.season,
            s.week,
            s.gameday as date,
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

            -- Team stats from team_game_summary (for defensive rankings)
            (SELECT rush_yds FROM team_game_summary
             WHERE game_id = s.pfr AND team_abbr = s.home_team LIMIT 1) as home_rushing_yds,
            (SELECT rush_yds FROM team_game_summary
             WHERE game_id = s.pfr AND team_abbr = s.away_team LIMIT 1) as away_rushing_yds,
            (SELECT pass_yds FROM team_game_summary
             WHERE game_id = s.pfr AND team_abbr = s.home_team LIMIT 1) as home_passing_yds,
            (SELECT pass_yds FROM team_game_summary
             WHERE game_id = s.pfr AND team_abbr = s.away_team LIMIT 1) as away_passing_yds,

            -- Legacy columns
            NULL as source_url,
            NULL as last_updated
        FROM schedules s
        WHERE s.pfr IS NOT NULL
    """)

    conn.commit()

    # Verify view was created
    cursor.execute("SELECT COUNT(*) FROM games")
    count = cursor.fetchone()[0]
    print(f"  ✅ Created enhanced 'games' view - {count} games with full column set")


def create_rushing_leaders_view(conn):
    """Create rushing_leaders view aggregated from player_box_score."""
    print("\n  Creating 'rushing_leaders' view...")
    cursor = conn.cursor()

    cursor.execute("DROP VIEW IF EXISTS rushing_leaders")

    cursor.execute("""
        CREATE VIEW rushing_leaders AS
        SELECT
            player,
            team,
            season,
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

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM rushing_leaders")
    count = cursor.fetchone()[0]
    print(f"  ✅ Created 'rushing_leaders' view - {count} player-season records")


def create_receiving_leaders_view(conn):
    """Create receiving_leaders view aggregated from player_box_score."""
    print("\n  Creating 'receiving_leaders' view...")
    cursor = conn.cursor()

    cursor.execute("DROP VIEW IF EXISTS receiving_leaders")

    cursor.execute("""
        CREATE VIEW receiving_leaders AS
        SELECT
            player,
            team,
            season,
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

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM receiving_leaders")
    count = cursor.fetchone()[0]
    print(f"  ✅ Created 'receiving_leaders' view - {count} player-season records")


def create_passing_leaders_view(conn):
    """Create passing_leaders view aggregated from player_box_score."""
    print("\n  Creating 'passing_leaders' view...")
    cursor = conn.cursor()

    cursor.execute("DROP VIEW IF EXISTS passing_leaders")

    cursor.execute("""
        CREATE VIEW passing_leaders AS
        SELECT
            player,
            team,
            season,
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

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM passing_leaders")
    count = cursor.fetchone()[0]
    print(f"  ✅ Created 'passing_leaders' view - {count} player-season records")


def create_touchdown_leaders_view(conn):
    """Create touchdown_leaders view aggregated from player_box_score."""
    print("\n  Creating 'touchdown_leaders' view...")
    cursor = conn.cursor()

    cursor.execute("DROP VIEW IF EXISTS touchdown_leaders")

    cursor.execute("""
        CREATE VIEW touchdown_leaders AS
        SELECT
            player,
            team,
            season,
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

    cursor.execute("SELECT COUNT(*) FROM touchdown_leaders")
    count = cursor.fetchone()[0]
    print(f"  ✅ Created 'touchdown_leaders' view - {count} player-season records")


def create_touchdown_scorers_view(conn):
    """
    Create touchdown_scorers view for TD analysis.
    Extracts individual TD events from player_box_score.
    """
    print("\n  Creating 'touchdown_scorers' view...")
    cursor = conn.cursor()

    cursor.execute("DROP VIEW IF EXISTS touchdown_scorers")

    cursor.execute("""
        CREATE VIEW touchdown_scorers AS
        SELECT
            -- Create synthetic entries for rushing TDs
            player_id || '_' || season || '_W' || week || '_rush' as td_id,
            COALESCE(
                (SELECT game_id FROM schedules
                 WHERE season = pbs.season
                 AND week = pbs.week
                 AND (home_team = pbs.team OR away_team = pbs.team)
                 LIMIT 1),
                season || '_W' || week || '_' || team
            ) as game_id,
            player,
            team,
            season,
            week,
            'Rushing' as touchdown_type,
            rush_td as td_count,
            0 as first_td_game,
            0 as first_td_for_team
        FROM player_box_score pbs
        WHERE rush_td > 0

        UNION ALL

        SELECT
            -- Create synthetic entries for receiving TDs
            player_id || '_' || season || '_W' || week || '_rec' as td_id,
            COALESCE(
                (SELECT game_id FROM schedules
                 WHERE season = pbs.season
                 AND week = pbs.week
                 AND (home_team = pbs.team OR away_team = pbs.team)
                 LIMIT 1),
                season || '_W' || week || '_' || team
            ) as game_id,
            player,
            team,
            season,
            week,
            'Receiving' as touchdown_type,
            rec_td as td_count,
            0 as first_td_game,
            0 as first_td_for_team
        FROM player_box_score pbs
        WHERE rec_td > 0

        UNION ALL

        SELECT
            -- Create synthetic entries for passing TDs
            player_id || '_' || season || '_W' || week || '_pass' as td_id,
            COALESCE(
                (SELECT game_id FROM schedules
                 WHERE season = pbs.season
                 AND week = pbs.week
                 AND (home_team = pbs.team OR away_team = pbs.team)
                 LIMIT 1),
                season || '_W' || week || '_' || team
            ) as game_id,
            player,
            team,
            season,
            week,
            'Passing' as touchdown_type,
            pass_td as td_count,
            0 as first_td_game,
            0 as first_td_for_team
        FROM player_box_score pbs
        WHERE pass_td > 0
    """)

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM touchdown_scorers")
    count = cursor.fetchone()[0]
    print(f"  ✅ Created 'touchdown_scorers' view - {count} TD events")


def create_first_td_game_leaders_view(conn):
    """
    Create first_td_game_leaders view (placeholder).
    Proper implementation requires play-by-play data parsing.
    """
    print("\n  Creating 'first_td_game_leaders' view (placeholder)...")
    cursor = conn.cursor()

    cursor.execute("DROP VIEW IF EXISTS first_td_game_leaders")

    cursor.execute("""
        CREATE VIEW first_td_game_leaders AS
        SELECT
            player,
            team,
            season,
            0 as first_td_count,
            'Requires play-by-play parsing' as notes
        FROM player_box_score
        WHERE (rush_td + rec_td + pass_td) > 0
        GROUP BY player, team, season
    """)

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM first_td_game_leaders")
    count = cursor.fetchone()[0]
    print(f"  ✅ Created 'first_td_game_leaders' view (placeholder) - {count} player-season records")


def create_box_score_summary_view(conn):
    """
    Create box_score_summary view as alias to team_game_summary.
    This allows old queries referencing box_score_summary to work.
    """
    print("\n  Creating 'box_score_summary' view (alias)...")
    cursor = conn.cursor()

    cursor.execute("DROP VIEW IF EXISTS box_score_summary")

    # Simply mirror team_game_summary structure
    cursor.execute("""
        CREATE VIEW box_score_summary AS
        SELECT * FROM team_game_summary
    """)

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM box_score_summary")
    count = cursor.fetchone()[0]
    print(f"  ✅ Created 'box_score_summary' view (alias) - {count} records")


def verify_all_views(conn):
    """Verify all created views exist and have data."""
    print("\n" + "="*60)
    print("VERIFYING ALL COMPATIBILITY VIEWS")
    print("="*60)

    cursor = conn.cursor()

    views_to_check = [
        'games',
        'team_game_summary',
        'player_box_score',
        'rushing_leaders',
        'receiving_leaders',
        'passing_leaders',
        'touchdown_leaders',
        'touchdown_scorers',
        'first_td_game_leaders',
        'box_score_summary'
    ]

    all_good = True
    for view_name in views_to_check:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {view_name}")
            count = cursor.fetchone()[0]
            status = "✅" if count > 0 else "⚠️"
            print(f"  {status} {view_name:30} {count:>6} records")
            if count == 0 and view_name not in ['first_td_game_leaders']:
                all_good = False
        except sqlite3.OperationalError as e:
            print(f"  ❌ {view_name:30} ERROR: {e}")
            all_good = False

    return all_good


def main():
    """Main migration process"""
    print("=" * 60)
    print("CREATE COMPREHENSIVE COMPATIBILITY VIEWS")
    print("=" * 60)
    print(f"\nDatabase: {MERGED_DB}")

    if not MERGED_DB.exists():
        print(f"\n❌ ERROR: Database not found at {MERGED_DB}")
        print("Please run merge_databases.py first.")
        return 1

    try:
        # Connect to database
        conn = sqlite3.connect(MERGED_DB)

        # Create all views
        print("\n" + "=" * 60)
        print("CREATING VIEWS")
        print("=" * 60)

        create_enhanced_games_view(conn)
        create_rushing_leaders_view(conn)
        create_receiving_leaders_view(conn)
        create_passing_leaders_view(conn)
        create_touchdown_leaders_view(conn)
        create_touchdown_scorers_view(conn)
        create_first_td_game_leaders_view(conn)
        create_box_score_summary_view(conn)

        # Verify all views
        all_good = verify_all_views(conn)

        conn.close()

        print("\n" + "=" * 60)
        if all_good:
            print("✅ ALL COMPATIBILITY VIEWS CREATED SUCCESSFULLY!")
        else:
            print("⚠️ SOME VIEWS HAVE ISSUES - CHECK ABOVE")
        print("=" * 60)

        print("\nViews Created:")
        print("  1. Enhanced 'games' view with 20+ additional columns")
        print("  2. rushing_leaders - Season rushing statistics")
        print("  3. receiving_leaders - Season receiving statistics")
        print("  4. passing_leaders - Season passing statistics")
        print("  5. touchdown_leaders - Season TD statistics")
        print("  6. touchdown_scorers - Individual TD events")
        print("  7. first_td_game_leaders - First TD tracking (placeholder)")
        print("  8. box_score_summary - Alias to team_game_summary")

        print("\nNext steps:")
        print("  1. Update pfr_viewer.py (if needed)")
        print("  2. Update merge_databases.py to include these views")
        print("  3. Test all features in Streamlit")
        print("  4. Commit and deploy")

        return 0 if all_good else 1

    except Exception as e:
        print(f"\n❌ ERROR during view creation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
