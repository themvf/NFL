# -*- coding: utf-8 -*-
"""
Yardage-Only DVOA (Defense-adjusted Value Over Average)
A practical framework for measuring player efficiency adjusted for opponent strength.

This is NOT Football Outsiders' proprietary DVOA, but a principled approximation
built solely from yardage data.

Key Concepts:
- Opportunity normalization (per carry / per target)
- Opponent difficulty adjustment
- Comparison to league baseline
"""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Database path - relative to script location
DB_PATH = Path(__file__).parent / "data" / "nfl_merged.db"


@dataclass
class DefensiveDVOA:
    """Team defensive DVOA metrics."""
    team: str
    games: int
    total_yards_allowed: float
    total_opportunities: int  # carries for rush, targets for pass
    yards_per_opp: float
    league_avg_ypo: float
    dvoa_pct: float  # Positive = bad defense (allows more), Negative = good defense
    adjustment_factor: float  # Used to adjust player yards


@dataclass
class PlayerDVOA:
    """Player yardage-only DVOA metrics."""
    player_id: str
    player_name: str
    position: str
    team: str
    games: int
    total_raw_yards: float
    total_adjusted_yards: float
    total_opportunities: int  # carries or targets
    raw_ypo: float  # Raw yards per opportunity
    adjusted_ypo: float  # Opponent-adjusted yards per opportunity
    league_avg_ypo: float
    dvoa_pct: float  # Positive = better than average
    dvoa_grade: str  # A+ to F


def get_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def calculate_league_rushing_baseline(season: int, through_week: Optional[int] = None) -> Dict:
    """
    Calculate league-wide rushing baseline statistics.

    Returns:
        Dict with league_avg_ypc, total_yards, total_carries, games
    """
    conn = get_connection()

    week_filter = f"AND week <= {through_week}" if through_week else ""

    query = f"""
        SELECT
            SUM(rushing_yards) as total_yards,
            SUM(carries) as total_carries,
            COUNT(DISTINCT team || '-' || week) as team_games
        FROM player_stats
        WHERE season = ?
            AND season_type = 'REG'
            AND position = 'RB'
            AND carries > 0
            {week_filter}
    """

    df = pd.read_sql_query(query, conn, params=(season,))
    conn.close()

    total_yards = df['total_yards'].iloc[0] or 0
    total_carries = df['total_carries'].iloc[0] or 0

    league_avg_ypc = total_yards / total_carries if total_carries > 0 else 0

    return {
        'league_avg_ypc': league_avg_ypc,
        'total_yards': total_yards,
        'total_carries': total_carries,
        'team_games': df['team_games'].iloc[0] or 0
    }


def calculate_league_receiving_baseline(season: int, through_week: Optional[int] = None) -> Dict:
    """
    Calculate league-wide receiving baseline statistics.

    Returns:
        Dict with league_avg_ypt (yards per target), total_yards, total_targets
    """
    conn = get_connection()

    week_filter = f"AND week <= {through_week}" if through_week else ""

    query = f"""
        SELECT
            SUM(receiving_yards) as total_yards,
            SUM(targets) as total_targets
        FROM player_stats
        WHERE season = ?
            AND season_type = 'REG'
            AND position IN ('WR', 'TE', 'RB')
            AND targets > 0
            {week_filter}
    """

    df = pd.read_sql_query(query, conn, params=(season,))
    conn.close()

    total_yards = df['total_yards'].iloc[0] or 0
    total_targets = df['total_targets'].iloc[0] or 0

    league_avg_ypt = total_yards / total_targets if total_targets > 0 else 0

    return {
        'league_avg_ypt': league_avg_ypt,
        'total_yards': total_yards,
        'total_targets': total_targets
    }


def calculate_defensive_rush_dvoa(
    season: int,
    through_week: Optional[int] = None
) -> List[DefensiveDVOA]:
    """
    Calculate defensive rushing DVOA for all teams.

    Negative DVOA = Good defense (allows fewer yards than average)
    Positive DVOA = Bad defense (allows more yards than average)

    Returns:
        List of DefensiveDVOA objects sorted by DVOA (best to worst)
    """
    conn = get_connection()

    # Get league baseline first
    baseline = calculate_league_rushing_baseline(season, through_week)
    league_avg_ypc = baseline['league_avg_ypc']

    week_filter = f"AND week <= {through_week}" if through_week else ""

    # Calculate defensive stats (yards ALLOWED by each team)
    query = f"""
        SELECT
            opponent_team as team,
            COUNT(DISTINCT week) as games,
            SUM(rushing_yards) as total_yards_allowed,
            SUM(carries) as total_carries_allowed
        FROM player_stats
        WHERE season = ?
            AND season_type = 'REG'
            AND position = 'RB'
            AND carries > 0
            {week_filter}
        GROUP BY opponent_team
        ORDER BY team
    """

    df = pd.read_sql_query(query, conn, params=(season,))
    conn.close()

    results = []
    for _, row in df.iterrows():
        yards_allowed = row['total_yards_allowed'] or 0
        carries_allowed = row['total_carries_allowed'] or 0

        ypc_allowed = yards_allowed / carries_allowed if carries_allowed > 0 else league_avg_ypc

        # DVOA calculation: (allowed / league_avg - 1) * 100
        # Positive = bad defense, Negative = good defense
        dvoa_pct = ((ypc_allowed / league_avg_ypc) - 1) * 100 if league_avg_ypc > 0 else 0

        # Adjustment factor for player DVOA calculation
        # Strong defense (negative DVOA) = factor > 1 (boost player yards)
        # Weak defense (positive DVOA) = factor < 1 (reduce player yards)
        adjustment_factor = 1 - (dvoa_pct / 100)

        results.append(DefensiveDVOA(
            team=row['team'],
            games=row['games'],
            total_yards_allowed=yards_allowed,
            total_opportunities=carries_allowed,
            yards_per_opp=ypc_allowed,
            league_avg_ypo=league_avg_ypc,
            dvoa_pct=round(dvoa_pct, 1),
            adjustment_factor=round(adjustment_factor, 3)
        ))

    # Sort by DVOA (best defense first = most negative)
    return sorted(results, key=lambda x: x.dvoa_pct)


def calculate_defensive_pass_dvoa(
    season: int,
    through_week: Optional[int] = None
) -> List[DefensiveDVOA]:
    """
    Calculate defensive passing DVOA for all teams.
    Uses yards per target as the base metric.

    Returns:
        List of DefensiveDVOA objects sorted by DVOA (best to worst)
    """
    conn = get_connection()

    baseline = calculate_league_receiving_baseline(season, through_week)
    league_avg_ypt = baseline['league_avg_ypt']

    week_filter = f"AND week <= {through_week}" if through_week else ""

    query = f"""
        SELECT
            opponent_team as team,
            COUNT(DISTINCT week) as games,
            SUM(receiving_yards) as total_yards_allowed,
            SUM(targets) as total_targets_allowed
        FROM player_stats
        WHERE season = ?
            AND season_type = 'REG'
            AND position IN ('WR', 'TE', 'RB')
            AND targets > 0
            {week_filter}
        GROUP BY opponent_team
        ORDER BY team
    """

    df = pd.read_sql_query(query, conn, params=(season,))
    conn.close()

    results = []
    for _, row in df.iterrows():
        yards_allowed = row['total_yards_allowed'] or 0
        targets_allowed = row['total_targets_allowed'] or 0

        ypt_allowed = yards_allowed / targets_allowed if targets_allowed > 0 else league_avg_ypt

        dvoa_pct = ((ypt_allowed / league_avg_ypt) - 1) * 100 if league_avg_ypt > 0 else 0
        adjustment_factor = 1 - (dvoa_pct / 100)

        results.append(DefensiveDVOA(
            team=row['team'],
            games=row['games'],
            total_yards_allowed=yards_allowed,
            total_opportunities=targets_allowed,
            yards_per_opp=ypt_allowed,
            league_avg_ypo=league_avg_ypt,
            dvoa_pct=round(dvoa_pct, 1),
            adjustment_factor=round(adjustment_factor, 3)
        ))

    return sorted(results, key=lambda x: x.dvoa_pct)


def _get_dvoa_grade(dvoa_pct: float) -> str:
    """Convert DVOA percentage to letter grade."""
    if dvoa_pct >= 25:
        return 'A+'
    elif dvoa_pct >= 15:
        return 'A'
    elif dvoa_pct >= 10:
        return 'A-'
    elif dvoa_pct >= 5:
        return 'B+'
    elif dvoa_pct >= 0:
        return 'B'
    elif dvoa_pct >= -5:
        return 'B-'
    elif dvoa_pct >= -10:
        return 'C+'
    elif dvoa_pct >= -15:
        return 'C'
    elif dvoa_pct >= -20:
        return 'C-'
    elif dvoa_pct >= -25:
        return 'D'
    else:
        return 'F'


def calculate_player_rushing_dvoa(
    season: int,
    through_week: Optional[int] = None,
    min_carries: int = 30,
    position_filter: Optional[str] = None
) -> List[PlayerDVOA]:
    """
    Calculate yardage-only rushing DVOA for players.

    For each game:
    1. Get opponent defensive run DVOA
    2. Adjust player yards by opponent difficulty
    3. Aggregate across games
    4. Compare to league baseline

    Args:
        season: NFL season year
        through_week: Optional week limit
        min_carries: Minimum carries to qualify
        position_filter: Optional position filter (e.g., 'RB')

    Returns:
        List of PlayerDVOA objects sorted by DVOA (best to worst)
    """
    conn = get_connection()

    # Get defensive DVOA for opponent adjustments
    def_dvoa_list = calculate_defensive_rush_dvoa(season, through_week)
    def_dvoa_map = {d.team: d.adjustment_factor for d in def_dvoa_list}

    # Get league baseline
    baseline = calculate_league_rushing_baseline(season, through_week)
    league_avg_ypc = baseline['league_avg_ypc']

    week_filter = f"AND week <= {through_week}" if through_week else ""
    position_clause = f"AND position = '{position_filter}'" if position_filter else "AND position IN ('RB', 'QB', 'WR')"

    # Get player game-by-game rushing stats
    query = f"""
        SELECT
            player_id,
            player_display_name as player_name,
            position,
            team,
            week,
            opponent_team,
            carries,
            rushing_yards
        FROM player_stats
        WHERE season = ?
            AND season_type = 'REG'
            AND carries > 0
            {position_clause}
            {week_filter}
        ORDER BY player_id, week
    """

    df = pd.read_sql_query(query, conn, params=(season,))
    conn.close()

    # Calculate adjusted yards per game
    player_stats = {}

    for _, row in df.iterrows():
        pid = row['player_id']

        if pid not in player_stats:
            player_stats[pid] = {
                'player_name': row['player_name'],
                'position': row['position'],
                'team': row['team'],
                'games': 0,
                'total_raw_yards': 0,
                'total_adjusted_yards': 0,
                'total_carries': 0
            }

        # Get opponent adjustment factor (default to 1.0 if unknown)
        opp = row['opponent_team']
        adj_factor = def_dvoa_map.get(opp, 1.0)

        raw_yards = row['rushing_yards'] or 0
        adjusted_yards = raw_yards * adj_factor

        player_stats[pid]['games'] += 1
        player_stats[pid]['total_raw_yards'] += raw_yards
        player_stats[pid]['total_adjusted_yards'] += adjusted_yards
        player_stats[pid]['total_carries'] += row['carries']
        player_stats[pid]['team'] = row['team']  # Update to latest team

    # Calculate DVOA for each player
    results = []
    for pid, stats in player_stats.items():
        if stats['total_carries'] < min_carries:
            continue

        raw_ypc = stats['total_raw_yards'] / stats['total_carries']
        adjusted_ypc = stats['total_adjusted_yards'] / stats['total_carries']

        # Player DVOA: (adjusted_efficiency / league_avg - 1) * 100
        dvoa_pct = ((adjusted_ypc / league_avg_ypc) - 1) * 100 if league_avg_ypc > 0 else 0

        results.append(PlayerDVOA(
            player_id=pid,
            player_name=stats['player_name'],
            position=stats['position'],
            team=stats['team'],
            games=stats['games'],
            total_raw_yards=round(stats['total_raw_yards'], 1),
            total_adjusted_yards=round(stats['total_adjusted_yards'], 1),
            total_opportunities=stats['total_carries'],
            raw_ypo=round(raw_ypc, 2),
            adjusted_ypo=round(adjusted_ypc, 2),
            league_avg_ypo=round(league_avg_ypc, 2),
            dvoa_pct=round(dvoa_pct, 1),
            dvoa_grade=_get_dvoa_grade(dvoa_pct)
        ))

    # Sort by DVOA (best first)
    return sorted(results, key=lambda x: x.dvoa_pct, reverse=True)


def calculate_player_receiving_dvoa(
    season: int,
    through_week: Optional[int] = None,
    min_targets: int = 20,
    position_filter: Optional[str] = None
) -> List[PlayerDVOA]:
    """
    Calculate yardage-only receiving DVOA for players.
    Uses yards per target as the base metric.

    Args:
        season: NFL season year
        through_week: Optional week limit
        min_targets: Minimum targets to qualify
        position_filter: Optional position filter (e.g., 'WR', 'TE', 'RB')

    Returns:
        List of PlayerDVOA objects sorted by DVOA (best to worst)
    """
    conn = get_connection()

    # Get defensive DVOA for opponent adjustments
    def_dvoa_list = calculate_defensive_pass_dvoa(season, through_week)
    def_dvoa_map = {d.team: d.adjustment_factor for d in def_dvoa_list}

    # Get league baseline
    baseline = calculate_league_receiving_baseline(season, through_week)
    league_avg_ypt = baseline['league_avg_ypt']

    week_filter = f"AND week <= {through_week}" if through_week else ""
    position_clause = f"AND position = '{position_filter}'" if position_filter else "AND position IN ('WR', 'TE', 'RB')"

    query = f"""
        SELECT
            player_id,
            player_display_name as player_name,
            position,
            team,
            week,
            opponent_team,
            targets,
            receiving_yards
        FROM player_stats
        WHERE season = ?
            AND season_type = 'REG'
            AND targets > 0
            {position_clause}
            {week_filter}
        ORDER BY player_id, week
    """

    df = pd.read_sql_query(query, conn, params=(season,))
    conn.close()

    player_stats = {}

    for _, row in df.iterrows():
        pid = row['player_id']

        if pid not in player_stats:
            player_stats[pid] = {
                'player_name': row['player_name'],
                'position': row['position'],
                'team': row['team'],
                'games': 0,
                'total_raw_yards': 0,
                'total_adjusted_yards': 0,
                'total_targets': 0
            }

        opp = row['opponent_team']
        adj_factor = def_dvoa_map.get(opp, 1.0)

        raw_yards = row['receiving_yards'] or 0
        adjusted_yards = raw_yards * adj_factor

        player_stats[pid]['games'] += 1
        player_stats[pid]['total_raw_yards'] += raw_yards
        player_stats[pid]['total_adjusted_yards'] += adjusted_yards
        player_stats[pid]['total_targets'] += row['targets']
        player_stats[pid]['team'] = row['team']

    results = []
    for pid, stats in player_stats.items():
        if stats['total_targets'] < min_targets:
            continue

        raw_ypt = stats['total_raw_yards'] / stats['total_targets']
        adjusted_ypt = stats['total_adjusted_yards'] / stats['total_targets']

        dvoa_pct = ((adjusted_ypt / league_avg_ypt) - 1) * 100 if league_avg_ypt > 0 else 0

        results.append(PlayerDVOA(
            player_id=pid,
            player_name=stats['player_name'],
            position=stats['position'],
            team=stats['team'],
            games=stats['games'],
            total_raw_yards=round(stats['total_raw_yards'], 1),
            total_adjusted_yards=round(stats['total_adjusted_yards'], 1),
            total_opportunities=stats['total_targets'],
            raw_ypo=round(raw_ypt, 2),
            adjusted_ypo=round(adjusted_ypt, 2),
            league_avg_ypo=round(league_avg_ypt, 2),
            dvoa_pct=round(dvoa_pct, 1),
            dvoa_grade=_get_dvoa_grade(dvoa_pct)
        ))

    return sorted(results, key=lambda x: x.dvoa_pct, reverse=True)


def get_player_game_log_with_dvoa(
    player_id: str,
    season: int,
    stat_type: str = 'rushing'  # 'rushing' or 'receiving'
) -> pd.DataFrame:
    """
    Get player's game-by-game stats with DVOA adjustments.
    Useful for detailed analysis.

    Returns:
        DataFrame with raw and adjusted stats per game
    """
    conn = get_connection()

    # Get defensive DVOA map
    if stat_type == 'rushing':
        def_dvoa_list = calculate_defensive_rush_dvoa(season)
        baseline = calculate_league_rushing_baseline(season)
        yards_col = 'rushing_yards'
        opp_col = 'carries'
    else:
        def_dvoa_list = calculate_defensive_pass_dvoa(season)
        baseline = calculate_league_receiving_baseline(season)
        yards_col = 'receiving_yards'
        opp_col = 'targets'

    def_dvoa_map = {d.team: (d.dvoa_pct, d.adjustment_factor) for d in def_dvoa_list}
    league_avg = baseline['league_avg_ypc'] if stat_type == 'rushing' else baseline['league_avg_ypt']

    query = f"""
        SELECT
            player_display_name as player_name,
            team,
            week,
            opponent_team,
            {opp_col} as opportunities,
            {yards_col} as raw_yards
        FROM player_stats
        WHERE player_id = ?
            AND season = ?
            AND season_type = 'REG'
            AND {opp_col} > 0
        ORDER BY week
    """

    df = pd.read_sql_query(query, conn, params=(player_id, season))
    conn.close()

    # Add DVOA columns
    def get_opp_dvoa(opp):
        return def_dvoa_map.get(opp, (0, 1.0))[0]

    def get_adj_factor(opp):
        return def_dvoa_map.get(opp, (0, 1.0))[1]

    df['opp_def_dvoa'] = df['opponent_team'].apply(get_opp_dvoa)
    df['adjustment_factor'] = df['opponent_team'].apply(get_adj_factor)
    df['adjusted_yards'] = (df['raw_yards'] * df['adjustment_factor']).round(1)
    df['raw_ypo'] = (df['raw_yards'] / df['opportunities']).round(2)
    df['adjusted_ypo'] = (df['adjusted_yards'] / df['opportunities']).round(2)
    df['league_avg_ypo'] = round(league_avg, 2)
    df['game_dvoa'] = (((df['adjusted_ypo'] / league_avg) - 1) * 100).round(1)

    return df


def get_dvoa_summary(season: int, through_week: Optional[int] = None) -> Dict:
    """
    Get comprehensive DVOA summary for the season.

    Returns:
        Dict with league baselines, top performers, and defensive rankings
    """
    rush_baseline = calculate_league_rushing_baseline(season, through_week)
    recv_baseline = calculate_league_receiving_baseline(season, through_week)

    def_rush_dvoa = calculate_defensive_rush_dvoa(season, through_week)
    def_pass_dvoa = calculate_defensive_pass_dvoa(season, through_week)

    player_rush_dvoa = calculate_player_rushing_dvoa(season, through_week, min_carries=30)
    player_recv_dvoa = calculate_player_receiving_dvoa(season, through_week, min_targets=20)

    return {
        'season': season,
        'through_week': through_week,
        'league_baselines': {
            'rushing': {
                'avg_ypc': round(rush_baseline['league_avg_ypc'], 2),
                'total_yards': rush_baseline['total_yards'],
                'total_carries': rush_baseline['total_carries']
            },
            'receiving': {
                'avg_ypt': round(recv_baseline['league_avg_ypt'], 2),
                'total_yards': recv_baseline['total_yards'],
                'total_targets': recv_baseline['total_targets']
            }
        },
        'top_rushing_defenses': [
            {'team': d.team, 'dvoa': d.dvoa_pct, 'ypc_allowed': d.yards_per_opp}
            for d in def_rush_dvoa[:5]
        ],
        'worst_rushing_defenses': [
            {'team': d.team, 'dvoa': d.dvoa_pct, 'ypc_allowed': d.yards_per_opp}
            for d in def_rush_dvoa[-5:][::-1]
        ],
        'top_rushing_players': [
            {'name': p.player_name, 'team': p.team, 'dvoa': p.dvoa_pct, 'grade': p.dvoa_grade}
            for p in player_rush_dvoa[:10]
        ],
        'top_receiving_players': [
            {'name': p.player_name, 'team': p.team, 'dvoa': p.dvoa_pct, 'grade': p.dvoa_grade}
            for p in player_recv_dvoa[:10]
        ]
    }


# Convenience function for DataFrame exports
def defensive_rush_dvoa_to_df(season: int, through_week: Optional[int] = None) -> pd.DataFrame:
    """Convert defensive rushing DVOA to DataFrame."""
    data = calculate_defensive_rush_dvoa(season, through_week)
    return pd.DataFrame([{
        'Team': d.team,
        'Games': d.games,
        'Rush Yds Allowed': int(d.total_yards_allowed),
        'Carries Allowed': d.total_opportunities,
        'YPC Allowed': round(d.yards_per_opp, 2),
        'League Avg YPC': round(d.league_avg_ypo, 2),
        'DVOA %': d.dvoa_pct,
        'Adj Factor': d.adjustment_factor
    } for d in data])


def defensive_pass_dvoa_to_df(season: int, through_week: Optional[int] = None) -> pd.DataFrame:
    """Convert defensive passing DVOA to DataFrame."""
    data = calculate_defensive_pass_dvoa(season, through_week)
    return pd.DataFrame([{
        'Team': d.team,
        'Games': d.games,
        'Recv Yds Allowed': int(d.total_yards_allowed),
        'Targets Allowed': d.total_opportunities,
        'Y/Tgt Allowed': round(d.yards_per_opp, 2),
        'League Avg Y/Tgt': round(d.league_avg_ypo, 2),
        'DVOA %': d.dvoa_pct,
        'Adj Factor': d.adjustment_factor
    } for d in data])


def player_rushing_dvoa_to_df(
    season: int,
    through_week: Optional[int] = None,
    min_carries: int = 30
) -> pd.DataFrame:
    """Convert player rushing DVOA to DataFrame."""
    data = calculate_player_rushing_dvoa(season, through_week, min_carries)
    return pd.DataFrame([{
        'Player': p.player_name,
        'Pos': p.position,
        'Team': p.team,
        'Games': p.games,
        'Carries': p.total_opportunities,
        'Raw Yards': int(p.total_raw_yards),
        'Adj Yards': int(p.total_adjusted_yards),
        'Raw YPC': p.raw_ypo,
        'Adj YPC': p.adjusted_ypo,
        'Lg Avg YPC': p.league_avg_ypo,
        'DVOA %': p.dvoa_pct,
        'Grade': p.dvoa_grade
    } for p in data])


def player_receiving_dvoa_to_df(
    season: int,
    through_week: Optional[int] = None,
    min_targets: int = 20,
    position: Optional[str] = None
) -> pd.DataFrame:
    """Convert player receiving DVOA to DataFrame."""
    data = calculate_player_receiving_dvoa(season, through_week, min_targets, position)
    return pd.DataFrame([{
        'Player': p.player_name,
        'Pos': p.position,
        'Team': p.team,
        'Games': p.games,
        'Targets': p.total_opportunities,
        'Raw Yards': int(p.total_raw_yards),
        'Adj Yards': int(p.total_adjusted_yards),
        'Raw Y/Tgt': p.raw_ypo,
        'Adj Y/Tgt': p.adjusted_ypo,
        'Lg Avg Y/Tgt': p.league_avg_ypo,
        'DVOA %': p.dvoa_pct,
        'Grade': p.dvoa_grade
    } for p in data])


if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("YARDAGE-ONLY DVOA TEST")
    print("=" * 60)

    season = 2025

    print(f"\nLeague Rushing Baseline (Season {season}):")
    baseline = calculate_league_rushing_baseline(season)
    print(f"  League Avg YPC: {baseline['league_avg_ypc']:.2f}")
    print(f"  Total Yards: {baseline['total_yards']:,}")
    print(f"  Total Carries: {baseline['total_carries']:,}")

    print(f"\n--- Top 5 Rushing Defenses (By DVOA) ---")
    def_dvoa = calculate_defensive_rush_dvoa(season)
    for d in def_dvoa[:5]:
        print(f"  {d.team}: DVOA {d.dvoa_pct:+.1f}% | YPC Allowed: {d.yards_per_opp:.2f} | Adj Factor: {d.adjustment_factor:.3f}")

    print(f"\n--- Bottom 5 Rushing Defenses ---")
    for d in def_dvoa[-5:]:
        print(f"  {d.team}: DVOA {d.dvoa_pct:+.1f}% | YPC Allowed: {d.yards_per_opp:.2f} | Adj Factor: {d.adjustment_factor:.3f}")

    print(f"\n--- Top 10 RBs by Rushing DVOA ---")
    player_dvoa = calculate_player_rushing_dvoa(season, min_carries=30, position_filter='RB')
    for p in player_dvoa[:10]:
        print(f"  {p.dvoa_grade} | {p.player_name} ({p.team}): DVOA {p.dvoa_pct:+.1f}% | Raw YPC: {p.raw_ypo:.2f} | Adj YPC: {p.adjusted_ypo:.2f}")
