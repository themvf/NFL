# -*- coding: utf-8 -*-
"""
WR Projections Module
Generates DVOA-adjusted receiving yard projections for wide receivers.

Projection Formula:
- Receiving: ProjectedTargets * CatchRate * DVOA-adjusted YPT

Uses spread data and game script to adjust for expected passing volume.
"""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Import DVOA functions
import yardage_dvoa as ydvoa

# Database path - relative to script location
DB_PATH = Path(__file__).parent / "data" / "nfl_merged.db"


@dataclass
class WRProjection:
    """Complete WR projection with all formula components."""
    player_name: str
    team: str
    opponent: str
    is_home: bool

    # Receiving components
    baseline_targets: float
    team_target_base: float
    spread_adjustment: float
    projected_team_targets: float
    target_share: float
    projected_targets: float
    league_avg_ypt: float
    wr_recv_dvoa: float
    def_pass_dvoa: float
    projected_ypt: float
    catch_rate: float
    projected_recv_yards: float


def get_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def get_player_baseline_stats(
    player_name: str,
    team: str,
    season: int,
    week: int,
    lookback: int = 4
) -> Dict:
    """
    Get player baseline statistics from recent games.

    Returns:
        Dict with median_targets, target_share, catch_rate
    """
    conn = get_connection()

    # Get player's recent game stats
    start_week = max(1, week - lookback)

    query = """
        SELECT
            week,
            targets,
            receptions,
            receiving_yards
        FROM player_stats
        WHERE player_display_name = ?
            AND team = ?
            AND season = ?
            AND week >= ?
            AND week < ?
            AND season_type = 'REG'
        ORDER BY week DESC
    """

    player_df = pd.read_sql_query(query, conn, params=(player_name, team, season, start_week, week))

    # Get team totals for share calculation
    team_query = """
        SELECT
            week,
            SUM(targets) as team_targets
        FROM player_stats
        WHERE team = ?
            AND season = ?
            AND week >= ?
            AND week < ?
            AND season_type = 'REG'
            AND position IN ('WR', 'TE', 'RB')
        GROUP BY week
    """

    team_df = pd.read_sql_query(team_query, conn, params=(team, season, start_week, week))
    conn.close()

    # Calculate baseline stats
    if len(player_df) == 0:
        return {
            'median_targets': 0,
            'target_share': 0,
            'catch_rate': 0.65  # Default WR catch rate
        }

    # Use median for stability
    median_targets = player_df['targets'].median() if len(player_df) >= 2 else player_df['targets'].mean()

    # Calculate shares
    total_player_targets = player_df['targets'].sum()
    total_team_targets = team_df['team_targets'].sum() if len(team_df) > 0 else 1

    target_share = total_player_targets / total_team_targets if total_team_targets > 0 else 0

    # Calculate catch rate
    total_receptions = player_df['receptions'].sum()
    catch_rate = total_receptions / total_player_targets if total_player_targets > 0 else 0.65

    return {
        'median_targets': round(median_targets, 1),
        'target_share': round(target_share, 3),
        'catch_rate': round(catch_rate, 3)
    }


def get_team_target_base(team: str, season: int, week: int, lookback: int = 4) -> float:
    """
    Get team's baseline pass attempts per game from recent games.

    Returns:
        Median team pass attempts per game (targets across all positions)
    """
    conn = get_connection()
    start_week = max(1, week - lookback)

    query = """
        SELECT
            week,
            SUM(targets) as team_targets
        FROM player_stats
        WHERE team = ?
            AND season = ?
            AND week >= ?
            AND week < ?
            AND season_type = 'REG'
        GROUP BY week
    """

    df = pd.read_sql_query(query, conn, params=(team, season, start_week, week))
    conn.close()

    if len(df) == 0:
        return 32.0  # League average default

    return round(df['team_targets'].median(), 1)


def get_spread_adjustment(spread_line: float, is_home_team: bool) -> float:
    """
    Calculate spread adjustment for game script (passing volume).

    For WRs, underdogs PASS MORE (trailing teams pass to catch up).
    This is the OPPOSITE of RB logic.

    From schedules.spread_line:
    - Positive = Home team favorite
    - Negative = Away team favorite

    Formula for WRs:
    - team_spread = spread_line (for home), -spread_line (for away)
    - SpreadAdj = 0.5 * team_spread (underdogs get MORE targets)
    - Capped at +/-5 targets

    Examples:
    - Home team is -7.0 favorite (spread_line = 7.0): home gets -3.5 targets, away gets +3.5
    - Away team is -3.0 favorite (spread_line = -3.0): away gets -1.5 targets, home gets +1.5
    """
    # Convert spread_line to betting spread from team's perspective
    if is_home_team:
        team_spread = -spread_line  # If spread_line=7.0, home is -7.0 favorite
    else:
        team_spread = spread_line   # If spread_line=7.0, away is +7.0 underdog

    # Formula: SpreadAdj = 0.5 * Spread
    # For favorite (negative spread): SpreadAdj = negative (fewer targets)
    # For underdog (positive spread): SpreadAdj = positive (more targets)
    raw_adjustment = 0.5 * team_spread

    # Cap at +/- 5 targets
    capped_adjustment = max(-5.0, min(5.0, raw_adjustment))

    return round(capped_adjustment, 1)


def get_matchup_spread(away_team: str, home_team: str, season: int, week: int) -> Optional[float]:
    """
    Get spread line from schedules table.

    Returns:
        spread_line (positive = home favorite, negative = away favorite)
        None if not found
    """
    conn = get_connection()

    query = """
        SELECT spread_line
        FROM schedules
        WHERE away_team = ?
            AND home_team = ?
            AND season = ?
            AND week = ?
        LIMIT 1
    """

    df = pd.read_sql_query(query, conn, params=(away_team, home_team, season, week))
    conn.close()

    if len(df) == 0 or pd.isna(df['spread_line'].iloc[0]):
        return None

    return float(df['spread_line'].iloc[0])


def get_player_dvoa(player_name: str, season: int, week: int) -> float:
    """
    Get player's receiving DVOA percentage.

    Returns:
        DVOA percentage (e.g., 12.5 means +12.5% above league average)
    """
    try:
        dvoa_list = ydvoa.calculate_player_receiving_dvoa(season, week, min_targets=1, position_filter='WR')

        for p in dvoa_list:
            if p.player_name == player_name:
                return p.dvoa_pct

        return 0.0  # League average if not found
    except Exception:
        return 0.0


def get_defensive_dvoa(team: str, season: int, week: int) -> float:
    """
    Get team's defensive pass DVOA percentage.

    Returns:
        DVOA percentage (negative = good defense, positive = bad defense)
    """
    try:
        dvoa_list = ydvoa.calculate_defensive_pass_dvoa(season, week)

        for d in dvoa_list:
            if d.team == team:
                return d.dvoa_pct

        return 0.0  # League average if not found
    except Exception:
        return 0.0


def calculate_wr_projection(
    player_name: str,
    team: str,
    opponent: str,
    spread_line: Optional[float],
    is_home: bool,
    season: int,
    week: int
) -> WRProjection:
    """
    Calculate complete WR projection with all formula components.

    Formula:
    - Receiving:
        ProjectedTargets = 0.6 * BaselineTargets + 0.4 * (ProjectedTeamTargets * TargetShare)
        ProjectedTeamTargets = TeamTargetBase + SpreadAdj
        ProjectedYPT = LgYPT * (1 + WR_DVOA/100) / (1 - DEF_DVOA/100)
        ProjectedRecvYds = ProjectedTargets * CatchRate * ProjectedYPT
    """
    # Get baseline stats
    baseline = get_player_baseline_stats(player_name, team, season, week)

    # Get team baseline
    team_target_base = get_team_target_base(team, season, week)

    # Calculate spread adjustment (underdogs pass more)
    spread_adj = get_spread_adjustment(spread_line, is_home) if spread_line is not None else 0.0

    # --- RECEIVING PROJECTION ---
    projected_team_targets = team_target_base + spread_adj
    projected_targets = 0.6 * baseline['median_targets'] + 0.4 * (projected_team_targets * baseline['target_share'])

    # Cap projected targets at realistic bounds
    MIN_TARGETS = 0.0
    MAX_TARGETS = 18.0  # Elite WR1 ceiling
    projected_targets = max(MIN_TARGETS, min(MAX_TARGETS, projected_targets))

    # Get receiving baselines
    recv_baseline = ydvoa.calculate_league_receiving_baseline(season, week)
    league_avg_ypt = recv_baseline['league_avg_ypt']

    # Get receiving DVOAs
    wr_recv_dvoa = get_player_dvoa(player_name, season, week)
    def_pass_dvoa = get_defensive_dvoa(opponent, season, week)

    # Cap defensive DVOA to avoid division issues
    safe_def_pass_dvoa = min(def_pass_dvoa, 99.0)

    # Calculate projected YPT (yards per target)
    projected_ypt = league_avg_ypt * (1 + wr_recv_dvoa/100) / (1 - safe_def_pass_dvoa/100)

    # Cap projected yards per target at realistic bounds
    MIN_YPT = 4.0   # Floor for WR receiving efficiency
    MAX_YPT = 15.0  # Elite deep threat ceiling
    projected_ypt = max(MIN_YPT, min(MAX_YPT, projected_ypt))

    # Calculate projected receiving yards
    projected_recv_yards = projected_targets * baseline['catch_rate'] * projected_ypt

    return WRProjection(
        player_name=player_name,
        team=team,
        opponent=opponent,
        is_home=is_home,
        # Receiving
        baseline_targets=baseline['median_targets'],
        team_target_base=team_target_base,
        spread_adjustment=spread_adj,
        projected_team_targets=round(projected_team_targets, 1),
        target_share=baseline['target_share'],
        projected_targets=round(projected_targets, 1),
        league_avg_ypt=round(league_avg_ypt, 2),
        wr_recv_dvoa=round(wr_recv_dvoa, 1),
        def_pass_dvoa=round(def_pass_dvoa, 1),
        projected_ypt=round(projected_ypt, 2),
        catch_rate=baseline['catch_rate'],
        projected_recv_yards=round(projected_recv_yards, 1)
    )


def get_team_wrs(team: str, season: int, week: int, min_targets: int = 15) -> List[str]:
    """
    Get list of WRs for a team with significant targets this season.

    Returns:
        List of player names sorted by total targets
    """
    conn = get_connection()

    query = """
        SELECT
            player_display_name,
            SUM(targets) as total_targets
        FROM player_stats
        WHERE team = ?
            AND season = ?
            AND week < ?
            AND season_type = 'REG'
            AND position = 'WR'
        GROUP BY player_display_name
        HAVING SUM(targets) >= ?
        ORDER BY total_targets DESC
    """

    df = pd.read_sql_query(query, conn, params=(team, season, week, min_targets))
    conn.close()

    return df['player_display_name'].tolist()


def get_matchup_projections(
    away_team: str,
    home_team: str,
    season: int,
    week: int,
    min_targets: int = 15
) -> Tuple[List[WRProjection], List[WRProjection]]:
    """
    Get WR projections for both teams in a matchup.

    Returns:
        Tuple of (away_team_projections, home_team_projections)
    """
    # Get spread
    spread_line = get_matchup_spread(away_team, home_team, season, week)

    # Get WRs for each team
    away_wrs = get_team_wrs(away_team, season, week, min_targets)
    home_wrs = get_team_wrs(home_team, season, week, min_targets)

    # Calculate projections
    away_projections = []
    for wr in away_wrs:
        proj = calculate_wr_projection(wr, away_team, home_team, spread_line, False, season, week)
        away_projections.append(proj)

    home_projections = []
    for wr in home_wrs:
        proj = calculate_wr_projection(wr, home_team, away_team, spread_line, True, season, week)
        home_projections.append(proj)

    # Sort by total projected yards
    away_projections.sort(key=lambda x: x.projected_recv_yards, reverse=True)
    home_projections.sort(key=lambda x: x.projected_recv_yards, reverse=True)

    return away_projections, home_projections


def format_spread_display(spread_line: Optional[float], home_team: str) -> str:
    """Format spread for display."""
    if spread_line is None:
        return "Spread: Not available"

    if spread_line > 0:
        return f"Spread: {home_team} -{abs(spread_line):.1f} (home favorite)"
    elif spread_line < 0:
        return f"Spread: {home_team} +{abs(spread_line):.1f} (home underdog)"
    else:
        return "Spread: Pick'em"


if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("WR PROJECTIONS TEST")
    print("=" * 60)

    season = 2025
    week = 14
    away_team = "DET"
    home_team = "GB"

    print(f"\nMatchup: {away_team} @ {home_team} - Week {week}")

    spread = get_matchup_spread(away_team, home_team, season, week)
    print(f"{format_spread_display(spread, home_team)}")

    away_projs, home_projs = get_matchup_projections(away_team, home_team, season, week, min_targets=10)

    print(f"\n{away_team} WR Projections:")
    for p in away_projs[:5]:
        print(f"  {p.player_name}: {p.projected_recv_yards:.1f} rec yds")
        print(f"    Targets: {p.baseline_targets:.1f} baseline -> {p.projected_targets:.1f} proj (spread adj: {p.spread_adjustment:+.1f})")
        print(f"    YPT: {p.league_avg_ypt:.2f} lg avg -> {p.projected_ypt:.2f} proj (WR DVOA: {p.wr_recv_dvoa:+.1f}%, DEF DVOA: {p.def_pass_dvoa:+.1f}%)")
        print(f"    Catch Rate: {p.catch_rate:.1%}")

    print(f"\n{home_team} WR Projections:")
    for p in home_projs[:5]:
        print(f"  {p.player_name}: {p.projected_recv_yards:.1f} rec yds")
        print(f"    Targets: {p.baseline_targets:.1f} baseline -> {p.projected_targets:.1f} proj (spread adj: {p.spread_adjustment:+.1f})")
        print(f"    YPT: {p.league_avg_ypt:.2f} lg avg -> {p.projected_ypt:.2f} proj (WR DVOA: {p.wr_recv_dvoa:+.1f}%, DEF DVOA: {p.def_pass_dvoa:+.1f}%)")
        print(f"    Catch Rate: {p.catch_rate:.1%}")
