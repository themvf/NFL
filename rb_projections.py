# -*- coding: utf-8 -*-
"""
RB Projections Module
Generates DVOA-adjusted rushing and receiving yard projections for running backs.

Projection Formulas:
- Rushing: ProjectedCarries * DVOA-adjusted YPC
- Receiving: ProjectedTargets * CatchRate * DVOA-adjusted YPT

Uses spread data to adjust for expected game script.
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
class RBProjection:
    """Complete RB projection with all formula components."""
    player_name: str
    team: str
    opponent: str
    is_home: bool

    # Rushing components
    baseline_carries: float
    team_rush_base: float
    spread_adjustment: float
    projected_team_rush: float
    rush_share: float
    projected_carries: float
    league_avg_ypc: float
    rb_rush_dvoa: float
    def_rush_dvoa: float
    projected_ypc: float
    projected_rush_yards: float

    # Receiving components
    baseline_targets: float
    team_target_base: float
    target_share: float
    projected_targets: float
    league_avg_ypt: float
    rb_recv_dvoa: float
    def_pass_dvoa: float
    projected_ypt: float
    catch_rate: float
    projected_recv_yards: float

    # Totals
    projected_total_yards: float


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
        Dict with median_carries, median_targets, rush_share, target_share, catch_rate
    """
    conn = get_connection()

    # Get player's recent game stats
    start_week = max(1, week - lookback)

    query = """
        SELECT
            week,
            carries,
            rushing_yards,
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
            SUM(carries) as team_carries,
            SUM(targets) as team_targets
        FROM player_stats
        WHERE team = ?
            AND season = ?
            AND week >= ?
            AND week < ?
            AND season_type = 'REG'
            AND position = 'RB'
        GROUP BY week
    """

    team_df = pd.read_sql_query(team_query, conn, params=(team, season, start_week, week))
    conn.close()

    # Calculate baseline stats
    if len(player_df) == 0:
        return {
            'median_carries': 0,
            'median_targets': 0,
            'rush_share': 0,
            'target_share': 0,
            'catch_rate': 0.75  # Default catch rate
        }

    # Use median for stability
    median_carries = player_df['carries'].median() if len(player_df) >= 2 else player_df['carries'].mean()
    median_targets = player_df['targets'].median() if len(player_df) >= 2 else player_df['targets'].mean()

    # Calculate shares
    total_player_carries = player_df['carries'].sum()
    total_player_targets = player_df['targets'].sum()
    total_team_carries = team_df['team_carries'].sum() if len(team_df) > 0 else 1
    total_team_targets = team_df['team_targets'].sum() if len(team_df) > 0 else 1

    rush_share = total_player_carries / total_team_carries if total_team_carries > 0 else 0
    target_share = total_player_targets / total_team_targets if total_team_targets > 0 else 0

    # Calculate catch rate
    total_receptions = player_df['receptions'].sum()
    catch_rate = total_receptions / total_player_targets if total_player_targets > 0 else 0.75

    return {
        'median_carries': round(median_carries, 1),
        'median_targets': round(median_targets, 1),
        'rush_share': round(rush_share, 3),
        'target_share': round(target_share, 3),
        'catch_rate': round(catch_rate, 3)
    }


def get_team_rush_base(team: str, season: int, week: int, lookback: int = 4) -> float:
    """
    Get team's baseline carries per game from recent games.

    Returns:
        Median team carries per game
    """
    conn = get_connection()
    start_week = max(1, week - lookback)

    query = """
        SELECT
            week,
            SUM(carries) as team_carries
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
        return 25.0  # League average default

    return round(df['team_carries'].median(), 1)


def get_team_rb_target_base(team: str, season: int, week: int, lookback: int = 4) -> float:
    """
    Get team's baseline RB targets per game from recent games.

    Returns:
        Median team RB targets per game
    """
    conn = get_connection()
    start_week = max(1, week - lookback)

    query = """
        SELECT
            week,
            SUM(targets) as rb_targets
        FROM player_stats
        WHERE team = ?
            AND season = ?
            AND week >= ?
            AND week < ?
            AND season_type = 'REG'
            AND position = 'RB'
        GROUP BY week
    """

    df = pd.read_sql_query(query, conn, params=(team, season, start_week, week))
    conn.close()

    if len(df) == 0:
        return 5.0  # League average default

    return round(df['rb_targets'].median(), 1)


def get_spread_adjustment(spread_line: float, is_home_team: bool) -> float:
    """
    Calculate spread adjustment for game script.

    From schedules.spread_line:
    - Positive = Home team favorite
    - Negative = Away team favorite

    For formula:
    - team_spread = spread_line (for home), -spread_line (for away)
    - SpreadAdj = 0.8 * (-team_spread), capped at +/-8

    Examples:
    - Home team is -6.5 favorite (spread_line = 6.5): home gets +5.2 carries, away gets -5.2
    - Away team is -3.0 favorite (spread_line = -3.0): away gets +2.4 carries, home gets -2.4
    """
    # Convert spread_line to betting spread from team's perspective
    # spread_line positive = home favorite
    # In betting: favorites have NEGATIVE spreads (giving points)
    # In betting: underdogs have POSITIVE spreads (getting points)
    if is_home_team:
        team_spread = -spread_line  # If spread_line=9.5, home is -9.5 favorite
    else:
        team_spread = spread_line   # If spread_line=9.5, away is +9.5 underdog

    # Formula: SpreadAdj = 0.8 * (-Spread)
    # For favorite (negative spread): SpreadAdj = positive (more carries)
    # For underdog (positive spread): SpreadAdj = negative (fewer carries)
    raw_adjustment = 0.8 * (-team_spread)

    # Cap at +/- 8 carries
    capped_adjustment = max(-8.0, min(8.0, raw_adjustment))

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


def get_player_dvoa(player_name: str, season: int, week: int, stat_type: str = 'rushing') -> float:
    """
    Get player's DVOA percentage.

    Args:
        stat_type: 'rushing' or 'receiving'

    Returns:
        DVOA percentage (e.g., 8.5 means +8.5% above league average)
    """
    try:
        if stat_type == 'rushing':
            dvoa_list = ydvoa.calculate_player_rushing_dvoa(season, week, min_carries=1, position_filter='RB')
        else:
            dvoa_list = ydvoa.calculate_player_receiving_dvoa(season, week, min_targets=1, position_filter='RB')

        for p in dvoa_list:
            if p.player_name == player_name:
                return p.dvoa_pct

        return 0.0  # League average if not found
    except Exception:
        return 0.0


def get_defensive_dvoa(team: str, season: int, week: int, stat_type: str = 'rushing') -> float:
    """
    Get team's defensive DVOA percentage.

    Args:
        stat_type: 'rushing' or 'passing'

    Returns:
        DVOA percentage (negative = good defense, positive = bad defense)
    """
    try:
        if stat_type == 'rushing':
            dvoa_list = ydvoa.calculate_defensive_rush_dvoa(season, week)
        else:
            dvoa_list = ydvoa.calculate_defensive_pass_dvoa(season, week)

        for d in dvoa_list:
            if d.team == team:
                return d.dvoa_pct

        return 0.0  # League average if not found
    except Exception:
        return 0.0


def calculate_rb_projection(
    player_name: str,
    team: str,
    opponent: str,
    spread_line: Optional[float],
    is_home: bool,
    season: int,
    week: int
) -> RBProjection:
    """
    Calculate complete RB projection with all formula components.

    Formulas:
    - Rushing:
        ProjectedCarries = 0.6 * BaselineCarries + 0.4 * (ProjectedTeamRush * RushShare)
        ProjectedTeamRush = TeamRushBase + SpreadAdj
        ProjectedYPC = LgYPC * (1 + RB_DVOA/100) / (1 - DEF_DVOA/100)
        ProjectedRushYds = ProjectedCarries * ProjectedYPC

    - Receiving:
        ProjectedTargets = 0.6 * BaselineTargets + 0.4 * (ProjectedTeamTargets * TargetShare)
        ProjectedYPT = LgYPT * (1 + RB_Recv_DVOA/100) / (1 - DEF_Pass_DVOA/100)
        ProjectedRecvYds = ProjectedTargets * CatchRate * ProjectedYPT
    """
    # Get baseline stats
    baseline = get_player_baseline_stats(player_name, team, season, week)

    # Get team baselines
    team_rush_base = get_team_rush_base(team, season, week)
    team_target_base = get_team_rb_target_base(team, season, week)

    # Calculate spread adjustment
    spread_adj = get_spread_adjustment(spread_line, is_home) if spread_line is not None else 0.0

    # --- RUSHING PROJECTION ---
    projected_team_rush = team_rush_base + spread_adj
    projected_carries = 0.6 * baseline['median_carries'] + 0.4 * (projected_team_rush * baseline['rush_share'])

    # Cap projected carries at realistic bounds
    MIN_CARRIES = 0.0
    MAX_CARRIES = 35.0  # Elite bell cow workload ceiling
    projected_carries = max(MIN_CARRIES, min(MAX_CARRIES, projected_carries))

    # Get league baseline
    rush_baseline = ydvoa.calculate_league_rushing_baseline(season, week)
    league_avg_ypc = rush_baseline['league_avg_ypc']

    # Get DVOAs
    rb_rush_dvoa = get_player_dvoa(player_name, season, week, 'rushing')
    def_rush_dvoa = get_defensive_dvoa(opponent, season, week, 'rushing')

    # Cap defensive DVOA to avoid division issues (if DEF_DVOA >= 100%, use 99%)
    safe_def_rush_dvoa = min(def_rush_dvoa, 99.0)

    # Calculate projected YPC
    # ProjectedYPC = LgYPC * (1 + RB_DVOA/100) / (1 - DEF_DVOA/100)
    projected_ypc = league_avg_ypc * (1 + rb_rush_dvoa/100) / (1 - safe_def_rush_dvoa/100)

    # Cap projected YPC at realistic bounds (prevent extreme projections)
    MIN_YPC = 2.0  # Floor for any rushing performance
    MAX_YPC = 8.0  # Elite single-game ceiling
    projected_ypc = max(MIN_YPC, min(MAX_YPC, projected_ypc))

    # Calculate projected rush yards
    projected_rush_yards = projected_carries * projected_ypc

    # --- RECEIVING PROJECTION ---
    projected_team_targets = team_target_base  # Could add spread adj here too
    projected_targets = 0.6 * baseline['median_targets'] + 0.4 * (projected_team_targets * baseline['target_share'])

    # Cap projected targets at realistic bounds
    MIN_TARGETS = 0.0
    MAX_TARGETS = 12.0  # Elite pass-catching back ceiling
    projected_targets = max(MIN_TARGETS, min(MAX_TARGETS, projected_targets))

    # Get receiving baselines
    recv_baseline = ydvoa.calculate_league_receiving_baseline(season, week)
    league_avg_ypt = recv_baseline['league_avg_ypt']

    # Get receiving DVOAs
    rb_recv_dvoa = get_player_dvoa(player_name, season, week, 'receiving')
    def_pass_dvoa = get_defensive_dvoa(opponent, season, week, 'passing')

    # Cap defensive DVOA
    safe_def_pass_dvoa = min(def_pass_dvoa, 99.0)

    # Calculate projected YPT (yards per target, not per reception)
    projected_ypt = league_avg_ypt * (1 + rb_recv_dvoa/100) / (1 - safe_def_pass_dvoa/100)

    # Cap projected yards per target at realistic bounds
    MIN_YPT = 3.0   # Floor for RB receiving efficiency
    MAX_YPT = 12.0  # Elite pass-catching back ceiling
    projected_ypt = max(MIN_YPT, min(MAX_YPT, projected_ypt))

    # Calculate projected receiving yards
    # Note: we use targets * YPT directly (YPT already accounts for incompletions)
    projected_recv_yards = projected_targets * projected_ypt

    # Total projected yards
    projected_total_yards = projected_rush_yards + projected_recv_yards

    return RBProjection(
        player_name=player_name,
        team=team,
        opponent=opponent,
        is_home=is_home,
        # Rushing
        baseline_carries=baseline['median_carries'],
        team_rush_base=team_rush_base,
        spread_adjustment=spread_adj,
        projected_team_rush=round(projected_team_rush, 1),
        rush_share=baseline['rush_share'],
        projected_carries=round(projected_carries, 1),
        league_avg_ypc=round(league_avg_ypc, 2),
        rb_rush_dvoa=round(rb_rush_dvoa, 1),
        def_rush_dvoa=round(def_rush_dvoa, 1),
        projected_ypc=round(projected_ypc, 2),
        projected_rush_yards=round(projected_rush_yards, 1),
        # Receiving
        baseline_targets=baseline['median_targets'],
        team_target_base=team_target_base,
        target_share=baseline['target_share'],
        projected_targets=round(projected_targets, 1),
        league_avg_ypt=round(league_avg_ypt, 2),
        rb_recv_dvoa=round(rb_recv_dvoa, 1),
        def_pass_dvoa=round(def_pass_dvoa, 1),
        projected_ypt=round(projected_ypt, 2),
        catch_rate=baseline['catch_rate'],
        projected_recv_yards=round(projected_recv_yards, 1),
        # Total
        projected_total_yards=round(projected_total_yards, 1)
    )


def get_team_rbs(team: str, season: int, week: int, min_carries: int = 15) -> List[str]:
    """
    Get list of RBs for a team with significant carries this season.

    Returns:
        List of player names sorted by total carries
    """
    conn = get_connection()

    query = """
        SELECT
            player_display_name,
            SUM(carries) as total_carries
        FROM player_stats
        WHERE team = ?
            AND season = ?
            AND week < ?
            AND season_type = 'REG'
            AND position = 'RB'
        GROUP BY player_display_name
        HAVING SUM(carries) >= ?
        ORDER BY total_carries DESC
    """

    df = pd.read_sql_query(query, conn, params=(team, season, week, min_carries))
    conn.close()

    return df['player_display_name'].tolist()


def get_matchup_projections(
    away_team: str,
    home_team: str,
    season: int,
    week: int,
    min_carries: int = 15
) -> Tuple[List[RBProjection], List[RBProjection]]:
    """
    Get RB projections for both teams in a matchup.

    Returns:
        Tuple of (away_team_projections, home_team_projections)
    """
    # Get spread
    spread_line = get_matchup_spread(away_team, home_team, season, week)

    # Get RBs for each team
    away_rbs = get_team_rbs(away_team, season, week, min_carries)
    home_rbs = get_team_rbs(home_team, season, week, min_carries)

    # Calculate projections
    away_projections = []
    for rb in away_rbs:
        proj = calculate_rb_projection(rb, away_team, home_team, spread_line, False, season, week)
        away_projections.append(proj)

    home_projections = []
    for rb in home_rbs:
        proj = calculate_rb_projection(rb, home_team, away_team, spread_line, True, season, week)
        home_projections.append(proj)

    # Sort by total projected yards
    away_projections.sort(key=lambda x: x.projected_total_yards, reverse=True)
    home_projections.sort(key=lambda x: x.projected_total_yards, reverse=True)

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
    print("RB PROJECTIONS TEST")
    print("=" * 60)

    season = 2025
    week = 15
    away_team = "NYJ"
    home_team = "JAX"

    print(f"\nMatchup: {away_team} @ {home_team} - Week {week}")

    spread = get_matchup_spread(away_team, home_team, season, week)
    print(f"{format_spread_display(spread, home_team)}")

    away_projs, home_projs = get_matchup_projections(away_team, home_team, season, week, min_carries=10)

    print(f"\n{away_team} RB Projections:")
    for p in away_projs[:3]:
        print(f"  {p.player_name}: {p.projected_total_yards:.1f} total yds ({p.projected_rush_yards:.1f} rush + {p.projected_recv_yards:.1f} recv)")
        print(f"    Carries: {p.baseline_carries:.1f} baseline -> {p.projected_carries:.1f} proj (spread adj: {p.spread_adjustment:+.1f})")
        print(f"    YPC: {p.league_avg_ypc:.2f} lg avg -> {p.projected_ypc:.2f} proj (RB DVOA: {p.rb_rush_dvoa:+.1f}%, DEF DVOA: {p.def_rush_dvoa:+.1f}%)")

    print(f"\n{home_team} RB Projections:")
    for p in home_projs[:3]:
        print(f"  {p.player_name}: {p.projected_total_yards:.1f} total yds ({p.projected_rush_yards:.1f} rush + {p.projected_recv_yards:.1f} recv)")
        print(f"    Carries: {p.baseline_carries:.1f} baseline -> {p.projected_carries:.1f} proj (spread adj: {p.spread_adjustment:+.1f})")
        print(f"    YPC: {p.league_avg_ypc:.2f} lg avg -> {p.projected_ypc:.2f} proj (RB DVOA: {p.rb_rush_dvoa:+.1f}%, DEF DVOA: {p.def_rush_dvoa:+.1f}%)")
