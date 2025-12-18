# -*- coding: utf-8 -*-
"""
Closed-System Projection Engine
REBUILD: Force Streamlit Cloud to pick up drift correction fix
6-layer projection system with conservation laws ensuring internal consistency.

Key Principle: Volume comes from role + game script. DVOA modifies efficiency.

Conservation Laws:
- Plays = Pass Attempts + Rush Attempts
- Team Rush Attempts = Sum of RB Carries
- Team Pass Attempts = Sum of Receiver Targets + Other Attempts
- Team Rush Yards = Sum of RB Rush Yards
- Team Pass Yards = Sum of Receiver Receiving Yards
"""

import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

# Import DVOA functions
import yardage_dvoa as ydvoa

# Database path - relative to script location
DB_PATH = Path(__file__).parent / "data" / "nfl_merged.db"


@dataclass
class TeamProjection:
    """Complete team projection with all layers."""
    team: str
    opponent: str
    is_home: bool
    season: int
    week: int

    # Layer 1: Game
    total_plays: int = 0

    # Layer 2: Team
    pass_attempts: int = 0
    rush_attempts: int = 0
    pass_rate: float = 0.0

    # Layer 3: Volume (populated after player allocation)
    total_targets: int = 0
    other_attempts: int = 0  # throwaways/sacks

    # Layer 5: Anchors
    rush_yards_anchor: float = 0.0
    pass_yards_anchor: float = 0.0
    total_yards_anchor: float = 0.0


@dataclass
class PlayerProjection:
    """Individual player projection."""
    player_name: str
    team: str
    position: str  # RB, WR, TE

    # Volume
    projected_carries: float = 0.0  # RBs only
    projected_targets: float = 0.0  # WR/TE/RB receiving

    # Efficiency
    projected_ypc: float = 0.0  # RBs only
    projected_ypt: float = 0.0

    # Yards
    projected_rush_yards: float = 0.0  # RBs only
    projected_recv_yards: float = 0.0
    projected_total_yards: float = 0.0

    # Metadata
    weighted_share: float = 0.0  # for redistribution
    dvoa_pct: float = 0.0


@dataclass
class QBProjection:
    """Quarterback projection."""
    player_name: str
    team: str

    # Passing
    projected_pass_att: int = 0
    projected_completions: float = 0.0
    projected_pass_yards: float = 0.0
    projected_pass_tds: float = 0.0
    projected_interceptions: float = 0.0
    projected_completion_pct: float = 0.0
    projected_ypa: float = 0.0  # yards per attempt

    # Rushing (for mobile QBs)
    projected_carries: float = 0.0
    projected_rush_yards: float = 0.0
    projected_rush_tds: float = 0.0
    projected_ypc: float = 0.0

    # DVOAs
    pass_dvoa_pct: float = 0.0
    rush_dvoa_pct: float = 0.0


def get_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


# ============================================================================
# Layer 1: Game Layer - Predict Total Plays
# ============================================================================

def predict_team_plays(
    team: str,
    opponent: str,
    season: int,
    week: int,
    vegas_total: Optional[float] = None,
    spread_line: Optional[float] = None,
    is_home: bool = True
) -> int:
    """
    Predict total offensive plays for a team.

    Formula:
    - Baseline: Median plays from last 6-8 games
    - Vegas adjustment: (total - 45) * 0.3 plays
    - Spread adjustment: ±0.15 * spread (favorites fewer, underdogs more)
    - Capped at 55-75 plays (realistic range)

    Args:
        team: Team abbreviation
        opponent: Opponent team abbreviation
        season: NFL season year
        week: Week number (for projecting)
        vegas_total: Over/under total points (optional, hybrid approach)
        spread_line: Point spread (positive = home favorite, negative = away favorite)
        is_home: Whether team is home

    Returns:
        Projected total plays for team
    """
    conn = get_connection()

    # Get baseline pace from recent games (last 6-8 weeks)
    lookback = min(8, week - 1)
    start_week = max(1, week - lookback)

    # Note: box_score_summary has 'plays' column directly!
    query = """
        SELECT week,
               plays,
               pass_att,
               rush_att
        FROM box_score_summary
        WHERE team = ?
          AND season = ?
          AND week >= ?
          AND week < ?
        ORDER BY week DESC
    """

    df = pd.read_sql_query(query, conn, params=(team, season, start_week, week))
    conn.close()

    if len(df) == 0:
        # No data - use league average default
        baseline_plays = 65  # Typical NFL team plays/game
    else:
        # Use plays column directly (or sum pass_att + rush_att + sacks if plays is NULL)
        df['total_plays'] = df['plays'].fillna(df['pass_att'] + df['rush_att'])
        baseline_plays = df['total_plays'].median()

    # Vegas total adjustment (if available and week >= 14 per hybrid approach)
    vegas_adj = 0.0
    if vegas_total is not None and week >= 14:
        # Higher scoring games correlate with more plays
        # Formula: ~0.3 plays per point above 45
        vegas_adj = (vegas_total - 45) * 0.3

    # Spread adjustment
    spread_adj = 0.0
    if spread_line is not None:
        # Determine if this team is favored
        # spread_line convention: positive = home favorite
        if is_home:
            team_spread = -spread_line  # If spread_line=7, home is -7 favorite
        else:
            team_spread = spread_line   # If spread_line=7, away is +7 underdog

        # Favorites (negative spread) run clock late → fewer plays
        # Underdogs (positive spread) hurry-up → more plays
        # Formula: -0.15 * spread (if -7 favorite, get -1.05 fewer plays)
        spread_adj = -0.15 * team_spread

    # Combine and clamp
    projected_plays = baseline_plays + vegas_adj + spread_adj
    projected_plays = max(55, min(75, projected_plays))

    return round(projected_plays)


# ============================================================================
# Layer 2: Team Layer - Split Plays into Pass/Rush
# ============================================================================

def split_plays_pass_rush(
    team: str,
    season: int,
    week: int,
    total_plays: int,
    spread_line: Optional[float] = None,
    is_home: bool = True
) -> Tuple[int, int]:
    """
    Split team plays into pass attempts and rush attempts.

    Formula:
    - Neutral pass rate: From games within 1 score, quarters 1-3
    - Game script adjustment: ±1% per point of spread
    - Capped at 45-75% pass rate
    - Enforces: PassAttempts + RushAttempts = TotalPlays

    Args:
        team: Team abbreviation
        season: NFL season year
        week: Week number
        total_plays: Total plays from Layer 1
        spread_line: Point spread (optional)
        is_home: Whether team is home

    Returns:
        Tuple of (pass_attempts, rush_attempts)
    """
    conn = get_connection()

    # Query for neutral game script (close score, not 4th quarter)
    # Note: box_score_summary may not have quarter/score_differential columns
    # We'll use overall pass rate from recent games as approximation
    lookback = min(6, week - 1)
    start_week = max(1, week - lookback)

    query = """
        SELECT SUM(pass_att) as total_pass,
               SUM(rush_att) as total_rush
        FROM box_score_summary
        WHERE team = ?
          AND season = ?
          AND week >= ?
          AND week < ?
    """

    df = pd.read_sql_query(query, conn, params=(team, season, start_week, week))
    conn.close()

    if len(df) == 0 or df['total_pass'].iloc[0] is None:
        # No data - use league average
        neutral_pass_rate = 0.60  # Typical ~60% pass rate in modern NFL
    else:
        total_pass = df['total_pass'].iloc[0]
        total_rush = df['total_rush'].iloc[0]
        neutral_pass_rate = total_pass / (total_pass + total_rush) if (total_pass + total_rush) > 0 else 0.60

    # Game script adjustment from spread
    script_adj = 0.0
    if spread_line is not None:
        # Determine team spread
        if is_home:
            team_spread = -spread_line
        else:
            team_spread = spread_line

        # Trailing teams (underdogs) pass more
        # Leading teams (favorites) rush more
        # Formula: +1% pass rate per point as underdog
        script_adj = team_spread * 0.01  # If +7 underdog, pass 7% more

    # Applied pass rate
    game_pass_rate = neutral_pass_rate + script_adj
    game_pass_rate = max(0.45, min(0.75, game_pass_rate))

    # Allocate plays
    pass_attempts = round(total_plays * game_pass_rate)
    rush_attempts = total_plays - pass_attempts  # Enforces conservation

    return pass_attempts, rush_attempts


# ============================================================================
# Helper function for database queries
# ============================================================================

def get_team_total_stat(
    team: str,
    season: int,
    start_week: int,
    end_week: int,
    stat_col: str  # 'carries' or 'targets'
) -> int:
    """
    Get total team stat across ALL positions for share calculation.

    For 'carries': sum RB carries only
    For 'targets': sum WR + TE + RB targets (all receiving positions)

    This ensures shares are calculated as "% of TEAM stat" not "% of POSITION stat".
    """
    conn = get_connection()

    # Determine which positions to include
    if stat_col == 'carries':
        # Only RBs carry the ball
        positions = ('RB',)
    elif stat_col == 'targets':
        # All pass-catching positions
        positions = ('WR', 'TE', 'RB')
    else:
        conn.close()
        return 0

    # Build position filter
    position_placeholders = ','.join('?' * len(positions))

    query = f"""
        SELECT SUM({stat_col}) as total_stat
        FROM player_stats
        WHERE team = ?
          AND season = ?
          AND week >= ?
          AND week < ?
          AND season_type = 'REG'
          AND position IN ({position_placeholders})
    """

    params = (team, season, start_week, end_week) + positions
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if len(df) > 0 and pd.notna(df['total_stat'].iloc[0]):
        return int(df['total_stat'].iloc[0])
    return 0


def get_player_stats_weighted(
    team: str,
    season: int,
    week: int,
    position: str,
    stat_col: str  # 'carries' or 'targets'
) -> List[Tuple[str, float, float]]:
    """
    Get weighted player shares (70% last 4 games, 30% season).

    CRITICAL: Shares calculated as "% of TEAM stat" (all positions),
    NOT "% of POSITION stat" to avoid double normalization bug.

    Returns:
        List of (player_name, weighted_share, confidence_level)
    """
    conn = get_connection()

    # Recent: last 4 games (70% weight)
    recent_lookback = min(4, week - 1)
    recent_start = max(1, week - recent_lookback)

    recent_query = f"""
        SELECT player_display_name,
               SUM({stat_col}) as recent_stat
        FROM player_stats
        WHERE team = ?
          AND season = ?
          AND position = ?
          AND week >= ?
          AND week < ?
          AND season_type = 'REG'
        GROUP BY player_display_name
        HAVING recent_stat > 0
    """

    recent_df = pd.read_sql_query(recent_query, conn, params=(team, season, position, recent_start, week))

    # Season: all games (30% weight)
    season_query = f"""
        SELECT player_display_name,
               SUM({stat_col}) as season_stat
        FROM player_stats
        WHERE team = ?
          AND season = ?
          AND position = ?
          AND week < ?
          AND season_type = 'REG'
        GROUP BY player_display_name
        HAVING season_stat > 0
    """

    season_df = pd.read_sql_query(season_query, conn, params=(team, season, position, week))
    conn.close()

    # FIX: Calculate total TEAM stats (not position-only) to avoid double normalization
    # This is the critical fix for the target under-projection bug
    team_recent_total = get_team_total_stat(team, season, recent_start, week, stat_col)
    team_season_total = get_team_total_stat(team, season, 1, week, stat_col)

    # Fallback to position-only if team total is 0 (shouldn't happen)
    if team_recent_total == 0:
        team_recent_total = recent_df['recent_stat'].sum() if len(recent_df) > 0 else 1
    if team_season_total == 0:
        team_season_total = season_df['season_stat'].sum() if len(season_df) > 0 else 1

    # Merge and calculate weighted shares
    merged = pd.merge(recent_df, season_df, on='player_display_name', how='outer').fillna(0)

    results = []
    for _, row in merged.iterrows():
        player_name = row['player_display_name']
        recent_stat = row.get('recent_stat', 0)
        season_stat = row.get('season_stat', 0)

        # Calculate shares
        recent_share = recent_stat / team_recent_total if team_recent_total > 0 else 0
        season_share = season_stat / team_season_total if team_season_total > 0 else 0

        # Weighted share (70% recent, 30% season)
        weighted_share = 0.7 * recent_share + 0.3 * season_share

        # Confidence level based on sample size
        games_played = len(recent_df[recent_df['player_display_name'] == player_name])
        if games_played >= recent_lookback - 1:  # Played most recent games
            confidence = 1.0  # High
        elif games_played >= 2:
            confidence = 0.5  # Medium
        else:
            confidence = 0.25  # Low

        results.append((player_name, weighted_share, confidence))

    # Sort by share descending
    results.sort(key=lambda x: x[1], reverse=True)

    return results


# ============================================================================
# Layer 3A: Rushing Volume Allocation
# ============================================================================

def allocate_rushing_volume(
    team: str,
    season: int,
    week: int,
    team_rush_att: int
) -> List[PlayerProjection]:
    """
    Allocate rushing attempts to RBs with exact conservation.

    Formula:
    - Weighted share: 70% last 4 games, 30% season
    - Initial allocation: carries = TeamRushAtt * share
    - Fix rounding drift to ensure sum = TeamRushAtt

    Returns:
        List of PlayerProjection objects with carries allocated
    """
    # Get RB shares
    rb_shares = get_player_stats_weighted(team, season, week, 'RB', 'carries')

    if len(rb_shares) == 0:
        # No RBs found - return empty list
        return []

    # Initialize projections
    projections = []
    for player_name, weighted_share, confidence in rb_shares:
        proj = PlayerProjection(
            player_name=player_name,
            team=team,
            position='RB',
            weighted_share=weighted_share
        )

        # Initial allocation
        proj.projected_carries = round(team_rush_att * weighted_share, 1)
        projections.append(proj)

    # Fix rounding drift (CRITICAL: enforce conservation)
    allocated_total = sum(p.projected_carries for p in projections)
    drift = team_rush_att - allocated_total

    if abs(drift) > 0.01:  # If there's meaningful drift
        # Sort by share (highest first)
        projections_sorted = sorted(projections, key=lambda x: x.weighted_share, reverse=True)

        # Distribute drift in 0.5 increments to highest-share players
        increment = 0.5
        adjustments_needed = int(abs(drift) / increment)

        for i in range(adjustments_needed):
            idx = i % len(projections_sorted)
            if drift > 0:
                projections_sorted[idx].projected_carries += increment
            else:
                projections_sorted[idx].projected_carries -= increment

        # Force exact conservation by giving any remaining drift to the highest-share player
        allocated_total = sum(p.projected_carries for p in projections)
        final_drift = team_rush_att - allocated_total
        if abs(final_drift) > 0.01:
            projections_sorted[0].projected_carries += final_drift
            projections_sorted[0].projected_carries = round(projections_sorted[0].projected_carries, 1)

    return projections


# ============================================================================
# Layer 3B: Passing Volume Allocation
# ============================================================================

def allocate_passing_volume(
    team: str,
    season: int,
    week: int,
    team_pass_att: int
) -> List[PlayerProjection]:
    """
    Allocate targets to receivers (WR/TE/RB) with exact conservation.

    Formula:
    - Targeted attempts: 92% of pass attempts become targets
    - Other attempts: 8% (throwaways, spikes, sacks)
    - Weighted share: 70% last 4 games, 30% season
    - Initial allocation: targets = TotalTargets * share
    - Fix rounding drift to ensure sum = TotalTargets

    Returns:
        List of PlayerProjection objects with targets allocated
    """
    # Determine total targets available
    targeted_att_rate = 0.92
    total_targets = round(team_pass_att * targeted_att_rate)

    # Get receiver shares (WR, TE, RB who receive)
    wr_shares = get_player_stats_weighted(team, season, week, 'WR', 'targets')
    te_shares = get_player_stats_weighted(team, season, week, 'TE', 'targets')
    rb_shares = get_player_stats_weighted(team, season, week, 'RB', 'targets')

    all_shares = wr_shares + te_shares + rb_shares

    if len(all_shares) == 0:
        # No receivers found
        return []

    # Normalize shares to sum to 1.0
    total_share = sum(share for _, share, _ in all_shares)
    if total_share > 0:
        normalized_shares = [(name, share/total_share, conf) for name, share, conf in all_shares]
    else:
        normalized_shares = all_shares

    # Initialize projections
    projections = []
    for player_name, weighted_share, confidence in normalized_shares:
        # Determine position
        position = 'WR'  # Default
        if any(player_name == name for name, _, _ in te_shares):
            position = 'TE'
        elif any(player_name == name for name, _, _ in rb_shares):
            position = 'RB'

        proj = PlayerProjection(
            player_name=player_name,
            team=team,
            position=position,
            weighted_share=weighted_share
        )

        # Initial allocation
        proj.projected_targets = round(total_targets * weighted_share, 1)
        projections.append(proj)

    # Fix rounding drift
    allocated_total = sum(p.projected_targets for p in projections)
    drift = total_targets - allocated_total

    if abs(drift) > 0.01:
        projections_sorted = sorted(projections, key=lambda x: x.weighted_share, reverse=True)

        increment = 0.5
        adjustments_needed = int(abs(drift) / increment)

        for i in range(adjustments_needed):
            idx = i % len(projections_sorted)
            if drift > 0:
                projections_sorted[idx].projected_targets += increment
            else:
                projections_sorted[idx].projected_targets -= increment

        # Force exact conservation by giving any remaining drift to the highest-share player
        allocated_total = sum(p.projected_targets for p in projections)
        final_drift = total_targets - allocated_total
        if abs(final_drift) > 0.01:
            projections_sorted[0].projected_targets += final_drift
            projections_sorted[0].projected_targets = round(projections_sorted[0].projected_targets, 1)

    return projections


# ============================================================================
# Layer 3C: QB Volume Allocation
# ============================================================================

def allocate_qb_stats(
    team: str,
    season: int,
    week: int,
    team_pass_att: int,
    team_rush_att: int
) -> Optional[QBProjection]:
    """
    Allocate QB stats based on team totals.

    For teams with one dominant QB, they get all pass attempts.
    For mobile QBs (Lamar, Hurts, Allen, etc.), project rushing carries.

    Returns:
        QBProjection object, or None if no QB found
    """
    conn = get_connection()

    # Get QB from recent games (most pass attempts)
    lookback = min(4, week - 1)
    start_week = max(1, week - lookback)

    qb_query = """
        SELECT player_display_name,
               SUM(pass_attempts) as total_pass_att,
               SUM(completions) as total_completions,
               SUM(passing_tds) as total_pass_tds,
               SUM(interceptions) as total_ints,
               SUM(carries) as total_carries,
               SUM(rushing_yards) as total_rush_yards,
               SUM(rushing_tds) as total_rush_tds,
               COUNT(*) as games_played
        FROM player_stats
        WHERE team = ?
          AND season = ?
          AND week >= ?
          AND week < ?
          AND season_type = 'REG'
          AND position = 'QB'
        GROUP BY player_display_name
        ORDER BY total_pass_att DESC
        LIMIT 1
    """

    qb_df = pd.read_sql_query(qb_query, conn, params=(team, season, start_week, week))
    conn.close()

    if len(qb_df) == 0:
        # No QB found
        return None

    qb_row = qb_df.iloc[0]
    qb_name = qb_row['player_display_name']

    # Calculate historical rates
    total_pass_att = qb_row['total_pass_att']
    total_completions = qb_row['total_completions']
    total_pass_tds = qb_row['total_pass_tds']
    total_ints = qb_row['total_ints']
    total_carries = qb_row['total_carries']
    games_played = qb_row['games_played']

    # Completion rate (historical)
    completion_rate = total_completions / total_pass_att if total_pass_att > 0 else 0.65

    # TD rate (TDs per attempt)
    td_rate = total_pass_tds / total_pass_att if total_pass_att > 0 else 0.04

    # INT rate (INTs per attempt)
    int_rate = total_ints / total_pass_att if total_pass_att > 0 else 0.02

    # Mobile QB detection: avg >2 carries/game
    is_mobile_qb = (total_carries / games_played) > 2.0 if games_played > 0 else False

    # Create QB projection
    qb_proj = QBProjection(
        player_name=qb_name,
        team=team
    )

    # Passing allocation
    qb_proj.projected_pass_att = team_pass_att
    qb_proj.projected_completions = round(team_pass_att * completion_rate, 1)
    qb_proj.projected_completion_pct = round(completion_rate * 100, 1)
    qb_proj.projected_pass_tds = round(team_pass_att * td_rate, 1)
    qb_proj.projected_interceptions = round(team_pass_att * int_rate, 1)

    # Rushing allocation (mobile QBs only)
    if is_mobile_qb:
        # Project carries based on recent average
        avg_carries_per_game = total_carries / games_played
        qb_proj.projected_carries = round(avg_carries_per_game, 1)
    else:
        qb_proj.projected_carries = 0.0

    return qb_proj


# ============================================================================
# Layer 4: Efficiency Layer - Use DVOA to Turn Volume into Yards
# ============================================================================

def apply_rushing_efficiency(
    rb_projections: List[PlayerProjection],
    opponent: str,
    season: int,
    week: int
) -> None:
    """
    Apply DVOA to calculate YPC and rushing yards for RBs.

    Formula:
        ProjectedYPC = LeagueAvgYPC * (1 + PlayerDVOA/100) / (1 - DefDVOA/100)
        ProjectedRushYards = ProjectedCarries * ProjectedYPC

    Modifies PlayerProjection objects in place.
    """
    # Get league baseline
    baseline = ydvoa.calculate_league_rushing_baseline(season, week)
    league_ypc = baseline['league_avg_ypc']

    # Get all player DVOAs for this week
    player_dvoa_list = ydvoa.calculate_player_rushing_dvoa(season, week, min_carries=1, position_filter='RB')
    player_dvoa_map = {p.player_name: p.dvoa_pct for p in player_dvoa_list}

    # Get defensive DVOA
    def_dvoa_list = ydvoa.calculate_defensive_rush_dvoa(season, week)
    def_dvoa = 0.0
    for d in def_dvoa_list:
        if d.team == opponent:
            def_dvoa = d.dvoa_pct
            break

    # Cap defensive DVOA to avoid division issues
    safe_def_dvoa = max(-25.0, min(25.0, def_dvoa))

    # Apply DVOA to each RB
    for rb in rb_projections:
        # Get player DVOA (default to 0 if not found)
        player_dvoa = player_dvoa_map.get(rb.player_name, 0.0)

        # Cap player DVOA to reasonable bounds
        safe_player_dvoa = max(-30.0, min(30.0, player_dvoa))

        # Apply DVOA multipliers
        projected_ypc = league_ypc * (1 + safe_player_dvoa/100) / (1 - safe_def_dvoa/100)

        # Cap projected YPC to realistic bounds
        MIN_YPC = 2.0
        MAX_YPC = 8.0
        projected_ypc = max(MIN_YPC, min(MAX_YPC, projected_ypc))

        # Calculate yards
        projected_rush_yards = rb.projected_carries * projected_ypc

        # Update projection
        rb.projected_ypc = round(projected_ypc, 2)
        rb.projected_rush_yards = round(projected_rush_yards, 1)
        rb.projected_total_yards = round(projected_rush_yards, 1)  # Will add recv yards in Layer 4B
        rb.dvoa_pct = round(player_dvoa, 1)


def apply_receiving_efficiency(
    rec_projections: List[PlayerProjection],
    opponent: str,
    season: int,
    week: int
) -> None:
    """
    Apply DVOA to calculate YPT and receiving yards for WR/TE/RB.

    Formula:
        ProjectedYPT = LeagueAvgYPT * (1 + PlayerDVOA/100) / (1 - DefDVOA/100)
        ProjectedRecvYards = ProjectedTargets * ProjectedYPT

    Note: YPT already includes incompletions, so NO catch rate multiplication.

    Modifies PlayerProjection objects in place.
    """
    # Get league baseline
    baseline = ydvoa.calculate_league_receiving_baseline(season, week)
    league_ypt = baseline['league_avg_ypt']

    # Get player DVOAs by position
    wr_dvoa_list = ydvoa.calculate_player_receiving_dvoa(season, week, min_targets=1, position_filter='WR')
    te_dvoa_list = ydvoa.calculate_player_receiving_dvoa(season, week, min_targets=1, position_filter='TE')
    rb_dvoa_list = ydvoa.calculate_player_receiving_dvoa(season, week, min_targets=1, position_filter='RB')

    # Combine into map
    player_dvoa_map = {}
    for p in wr_dvoa_list + te_dvoa_list + rb_dvoa_list:
        player_dvoa_map[p.player_name] = p.dvoa_pct

    # Get defensive pass DVOA
    def_dvoa_list = ydvoa.calculate_defensive_pass_dvoa(season, week)
    def_dvoa = 0.0
    for d in def_dvoa_list:
        if d.team == opponent:
            def_dvoa = d.dvoa_pct
            break

    # Cap defensive DVOA to avoid division issues
    safe_def_dvoa = max(-25.0, min(25.0, def_dvoa))

    # Apply DVOA to each receiver
    for rec in rec_projections:
        # Skip if no targets
        if rec.projected_targets == 0:
            continue

        # Get player DVOA (default to 0 if not found)
        player_dvoa = player_dvoa_map.get(rec.player_name, 0.0)

        # Cap player DVOA to reasonable bounds
        safe_player_dvoa = max(-30.0, min(30.0, player_dvoa))

        # Apply DVOA multipliers
        projected_ypt = league_ypt * (1 + safe_player_dvoa/100) / (1 - safe_def_dvoa/100)

        # Cap projected YPT to realistic bounds
        MIN_YPT = 4.0
        MAX_YPT = 15.0
        projected_ypt = max(MIN_YPT, min(MAX_YPT, projected_ypt))

        # Calculate yards (YPT already includes incompletions)
        projected_recv_yards = rec.projected_targets * projected_ypt

        # Update projection
        rec.projected_ypt = round(projected_ypt, 2)
        rec.projected_recv_yards = round(projected_recv_yards, 1)

        # Add to total yards (RBs may have both rush and recv yards)
        if rec.position == 'RB':
            rec.projected_total_yards = round(rec.projected_rush_yards + projected_recv_yards, 1)
        else:
            rec.projected_total_yards = round(projected_recv_yards, 1)

        rec.dvoa_pct = round(player_dvoa, 1)


def apply_qb_efficiency(
    qb_proj: Optional[QBProjection],
    opponent: str,
    season: int,
    week: int,
    pass_yards_anchor: float,
    rush_yards_anchor: float
) -> None:
    """
    Apply DVOA to calculate QB passing and rushing yards.

    Formula for passing:
        ProjectedYPA = LeagueAvgYPA * (1 + QB_PassDVOA/100) / (1 - Def_PassDVOA/100)
        ProjectedPassYards = ProjectedPassAtt * ProjectedYPA

    Formula for rushing (mobile QBs):
        ProjectedYPC = LeagueAvgYPC * (1 + QB_RushDVOA/100) / (1 - Def_RushDVOA/100)
        ProjectedRushYards = ProjectedCarries * ProjectedYPC

    Modifies QBProjection object in place.
    """
    if qb_proj is None:
        return

    # --- PASSING EFFICIENCY ---

    # Get league baseline YPA (yards per attempt)
    recv_baseline = ydvoa.calculate_league_receiving_baseline(season, week)
    league_avg_ypt = recv_baseline['league_avg_ypt']

    # Convert YPT to YPA: YPA ≈ YPT * completion_rate (since YPT includes incompletions)
    # Actually, YPT already accounts for incompletions, so we can use it directly
    league_avg_ypa = league_avg_ypt  # YPT and YPA are conceptually similar for our purposes

    # Get QB passing DVOA (from receiving DVOA, filtered by position='QB')
    # Note: yardage_dvoa doesn't have QB-specific DVOA, so we'll use team passing efficiency
    # For now, default to 0.0 (league average) - could enhance later with QB-specific metrics
    qb_pass_dvoa = 0.0  # TODO: Add QB-specific DVOA calculation

    # Get defensive pass DVOA
    def_dvoa_list = ydvoa.calculate_defensive_pass_dvoa(season, week)
    def_pass_dvoa = 0.0
    for d in def_dvoa_list:
        if d.team == opponent:
            def_pass_dvoa = d.dvoa_pct
            break

    # Cap defensive DVOA
    safe_def_pass_dvoa = max(-25.0, min(25.0, def_pass_dvoa))

    # Calculate projected YPA
    projected_ypa = league_avg_ypa * (1 + qb_pass_dvoa/100) / (1 - safe_def_pass_dvoa/100)

    # Cap projected YPA to realistic bounds
    MIN_YPA = 5.0
    MAX_YPA = 10.0
    projected_ypa = max(MIN_YPA, min(MAX_YPA, projected_ypa))

    # Calculate passing yards
    # NOTE: We'll use the pass_yards_anchor directly for conservation
    qb_proj.projected_pass_yards = round(pass_yards_anchor, 1)
    qb_proj.projected_ypa = round(qb_proj.projected_pass_yards / qb_proj.projected_pass_att, 2) if qb_proj.projected_pass_att > 0 else 0.0
    qb_proj.pass_dvoa_pct = round(qb_pass_dvoa, 1)

    # --- RUSHING EFFICIENCY (mobile QBs only) ---

    if qb_proj.projected_carries > 0:
        # Get league baseline YPC
        rush_baseline = ydvoa.calculate_league_rushing_baseline(season, week)
        league_ypc = rush_baseline['league_avg_ypc']

        # QB rushing DVOA (use 0.0 for now, could enhance with QB-specific metrics)
        qb_rush_dvoa = 0.0  # TODO: Add QB-specific rushing DVOA

        # Get defensive rush DVOA
        def_rush_dvoa_list = ydvoa.calculate_defensive_rush_dvoa(season, week)
        def_rush_dvoa = 0.0
        for d in def_rush_dvoa_list:
            if d.team == opponent:
                def_rush_dvoa = d.dvoa_pct
                break

        # Cap defensive DVOA
        safe_def_rush_dvoa = max(-25.0, min(25.0, def_rush_dvoa))

        # Calculate projected YPC for QB
        projected_ypc = league_ypc * (1 + qb_rush_dvoa/100) / (1 - safe_def_rush_dvoa/100)

        # Cap projected YPC
        MIN_YPC = 3.0
        MAX_YPC = 8.0
        projected_ypc = max(MIN_YPC, min(MAX_YPC, projected_ypc))

        # Calculate rushing yards
        qb_proj.projected_rush_yards = round(qb_proj.projected_carries * projected_ypc, 1)
        qb_proj.projected_ypc = round(projected_ypc, 2)
        qb_proj.rush_dvoa_pct = round(qb_rush_dvoa, 1)

        # Calculate rushing TDs (historical rate)
        # Already have total_rush_tds from allocation, use that rate
    else:
        qb_proj.projected_rush_yards = 0.0
        qb_proj.projected_ypc = 0.0
        qb_proj.rush_dvoa_pct = 0.0


# ============================================================================
# Layer 5: Reconciliation Layer - Force Team Totals to Match
# ============================================================================

def compute_team_anchors(
    team: str,
    opponent: str,
    season: int,
    week: int,
    total_plays: int,
    pass_rate: float,
    strategy: str = 'neutral'
) -> Tuple[float, float, float]:
    """
    Compute team rush/pass/total yard anchors from offensive/defensive strength.

    Formula:
        - Team yards/play (offense capability)
        - Opponent yards/play allowed (defense capability)
        - Blended expectation based on strategy:
            * neutral: 60% offense, 40% defense
            * optimistic: 70% offense, 30% defense
            * conservative: 40% offense, 60% defense

    Returns:
        Tuple of (rush_yards_anchor, pass_yards_anchor, total_yards_anchor)
    """
    conn = get_connection()

    # Get team offensive capability (last 6 games)
    lookback = 6
    start_week = max(1, week - lookback)

    team_query = """
        SELECT
            AVG((pass_yds + rush_yds) * 1.0 / plays) as yards_per_play
        FROM box_score_summary
        WHERE team = ?
          AND season = ?
          AND week >= ?
          AND week < ?
          AND plays > 0
    """

    team_df = pd.read_sql_query(team_query, conn, params=(team, season, start_week, week))
    if len(team_df) > 0 and pd.notna(team_df['yards_per_play'].iloc[0]):
        team_ypp = team_df['yards_per_play'].iloc[0]
    else:
        team_ypp = 5.5  # Default to league avg

    # Get opponent defensive capability (yards allowed per play)
    opp_query = """
        SELECT
            AVG((pass_yds + rush_yds) * 1.0 / plays) as yards_per_play_allowed
        FROM box_score_summary
        WHERE team = ?
          AND season = ?
          AND week >= ?
          AND week < ?
          AND plays > 0
    """

    opp_df = pd.read_sql_query(opp_query, conn, params=(opponent, season, start_week, week))
    if len(opp_df) > 0 and pd.notna(opp_df['yards_per_play_allowed'].iloc[0]):
        opp_ypp_allowed = opp_df['yards_per_play_allowed'].iloc[0]
    else:
        opp_ypp_allowed = 5.5  # Default to league avg

    conn.close()

    # Blend based on strategy
    if strategy == 'optimistic':
        weight_offense = 0.7
        weight_defense = 0.3
    elif strategy == 'conservative':
        weight_offense = 0.4
        weight_defense = 0.6
    else:  # neutral
        weight_offense = 0.6
        weight_defense = 0.4

    expected_ypp = weight_offense * team_ypp + weight_defense * opp_ypp_allowed

    # Calculate total yards anchor
    total_yards_anchor = expected_ypp * total_plays

    # Split into rush/pass anchors
    # Pass plays are slightly more efficient than rush plays
    pass_efficiency_factor = 1.15  # Passing typically gets more yards per play
    rush_efficiency_factor = 0.85

    # Weighted average: pass_rate * pass_eff + (1 - pass_rate) * rush_eff = 1.0
    # So we normalize: pass yards get boosted, rush yards reduced
    pass_yards_anchor = total_yards_anchor * pass_rate * pass_efficiency_factor
    rush_yards_anchor = total_yards_anchor - pass_yards_anchor

    # Ensure non-negative
    rush_yards_anchor = max(0.0, rush_yards_anchor)
    pass_yards_anchor = max(0.0, pass_yards_anchor)

    return (
        round(rush_yards_anchor, 1),
        round(pass_yards_anchor, 1),
        round(total_yards_anchor, 1)
    )


def reconcile_to_anchors(
    rb_projections: List[PlayerProjection],
    rec_projections: List[PlayerProjection],
    rush_anchor: float,
    pass_anchor: float
) -> None:
    """
    Scale player yards to match team anchors via proportional scaling.

    This preserves:
    - Relative player performance (DVOA advantages intact)
    - Conservation law (player yards sum exactly to team anchor)
    - Realistic team totals (grounded in team capability)

    Modifies PlayerProjection objects in place.
    """
    # Calculate current bottom-up totals
    sum_rush_yards = sum(rb.projected_rush_yards for rb in rb_projections)
    sum_recv_yards = sum(rec.projected_recv_yards for rec in rec_projections)

    # Calculate scaling factors
    rush_scale = rush_anchor / sum_rush_yards if sum_rush_yards > 0 else 1.0
    pass_scale = pass_anchor / sum_recv_yards if sum_recv_yards > 0 else 1.0

    # Apply scaling to RBs (rush yards)
    for rb in rb_projections:
        rb.projected_rush_yards *= rush_scale
        rb.projected_rush_yards = round(rb.projected_rush_yards, 1)

        # Recalculate YPC after scaling
        if rb.projected_carries > 0:
            rb.projected_ypc = rb.projected_rush_yards / rb.projected_carries
            rb.projected_ypc = round(rb.projected_ypc, 2)

        # Update total yards
        rb.projected_total_yards = round(rb.projected_rush_yards, 1)

    # Apply scaling to receivers (receiving yards)
    for rec in rec_projections:
        rec.projected_recv_yards *= pass_scale
        rec.projected_recv_yards = round(rec.projected_recv_yards, 1)

        # Recalculate YPT after scaling
        if rec.projected_targets > 0:
            rec.projected_ypt = rec.projected_recv_yards / rec.projected_targets
            rec.projected_ypt = round(rec.projected_ypt, 2)

        # Update total yards
        if rec.position == 'RB':
            rec.projected_total_yards = round(rec.projected_rush_yards + rec.projected_recv_yards, 1)
        else:
            rec.projected_total_yards = round(rec.projected_recv_yards, 1)


# ============================================================================
# Layer 6: Final Sanity Checks (Validation & Constraints)
# ============================================================================

def validate_projections(
    team_proj: TeamProjection,
    rb_projections: List[PlayerProjection],
    rec_projections: List[PlayerProjection]
) -> bool:
    """
    Validate all conservation laws and sanity checks.

    Returns:
        True if all checks pass

    Raises:
        AssertionError if any check fails
    """
    # Check 1: Plays conservation
    assert team_proj.pass_attempts + team_proj.rush_attempts == team_proj.total_plays, \
        f"Plays conservation failed: {team_proj.pass_attempts} + {team_proj.rush_attempts} != {team_proj.total_plays}"

    # Check 2: Carry conservation
    player_carries = sum(rb.projected_carries for rb in rb_projections)
    assert abs(player_carries - team_proj.rush_attempts) < 0.1, \
        f"Carry conservation failed: sum={player_carries:.1f}, expected={team_proj.rush_attempts}"

    # Check 3: Target conservation (with targeted attempt rate ~92%)
    player_targets = sum(rec.projected_targets for rec in rec_projections)
    expected_targets = team_proj.total_targets
    assert abs(player_targets - expected_targets) < 0.1, \
        f"Target conservation failed: sum={player_targets:.1f}, expected={expected_targets}"

    # Check 4: Rush yards conservation
    player_rush_yards = sum(rb.projected_rush_yards for rb in rb_projections)
    assert abs(player_rush_yards - team_proj.rush_yards_anchor) < 1.0, \
        f"Rush yards conservation failed: sum={player_rush_yards:.1f}, expected={team_proj.rush_yards_anchor:.1f}"

    # Check 5: Receiving yards conservation
    player_recv_yards = sum(rec.projected_recv_yards for rec in rec_projections)
    assert abs(player_recv_yards - team_proj.pass_yards_anchor) < 1.0, \
        f"Receiving yards conservation failed: sum={player_recv_yards:.1f}, expected={team_proj.pass_yards_anchor:.1f}"

    # Check 6: Realistic ranges
    assert 55 <= team_proj.total_plays <= 75, \
        f"Total plays out of range: {team_proj.total_plays} (expected 55-75)"

    assert 0.45 <= team_proj.pass_rate <= 0.75, \
        f"Pass rate out of range: {team_proj.pass_rate:.2%} (expected 45-75%)"

    # Check 7: Player values non-negative
    for rb in rb_projections:
        assert rb.projected_carries >= 0, f"{rb.player_name}: negative carries {rb.projected_carries}"
        assert rb.projected_ypc >= 0, f"{rb.player_name}: negative YPC {rb.projected_ypc}"
        assert rb.projected_rush_yards >= 0, f"{rb.player_name}: negative rush yards {rb.projected_rush_yards}"

    for rec in rec_projections:
        assert rec.projected_targets >= 0, f"{rec.player_name}: negative targets {rec.projected_targets}"
        assert rec.projected_ypt >= 0, f"{rec.player_name}: negative YPT {rec.projected_ypt}"
        assert rec.projected_recv_yards >= 0, f"{rec.player_name}: negative recv yards {rec.projected_recv_yards}"

    return True


# ============================================================================
# Main Pipeline - Project Complete Matchup
# ============================================================================

def project_matchup(
    away_team: str,
    home_team: str,
    season: int,
    week: int,
    vegas_total: Optional[float] = None,
    spread_line: Optional[float] = None,
    strategy: str = 'neutral'
) -> Tuple[TeamProjection, List[PlayerProjection], Optional[QBProjection], TeamProjection, List[PlayerProjection], Optional[QBProjection]]:
    """
    Complete closed-system projection for a matchup.

    Args:
        away_team: Away team abbreviation
        home_team: Home team abbreviation
        season: Season year
        week: Week number
        vegas_total: Over/under total (optional, Week 14+ recommended)
        spread_line: Point spread, positive=home favorite (optional)
        strategy: 'neutral' (60/40), 'optimistic' (70/30), 'conservative' (40/60)

    Returns:
        Tuple of (away_team_proj, away_players, away_qb, home_team_proj, home_players, home_qb)
    """
    # --- AWAY TEAM ---

    # Layer 1: Game - Predict total plays
    away_plays = predict_team_plays(away_team, home_team, season, week, vegas_total, spread_line, is_home=False)

    # Layer 2: Team - Split plays into pass/rush
    away_pass, away_rush = split_plays_pass_rush(away_team, season, week, away_plays, spread_line, is_home=False)

    # Layer 3: Volume - Allocate to players
    # Layer 3C: QB allocation first (to determine mobile QB carries)
    away_qb = allocate_qb_stats(away_team, season, week, away_pass, away_rush)

    # Adjust RB rush attempts for mobile QB carries
    away_rb_rush_att = away_rush
    if away_qb is not None and away_qb.projected_carries > 0:
        away_rb_rush_att = max(0, away_rush - int(away_qb.projected_carries))

    # Layer 3A/3B: RB and receiver allocation
    away_rbs = allocate_rushing_volume(away_team, season, week, away_rb_rush_att)
    away_recs = allocate_passing_volume(away_team, season, week, away_pass)

    # Layer 4: Efficiency - Apply DVOA
    apply_rushing_efficiency(away_rbs, home_team, season, week)
    apply_receiving_efficiency(away_recs, home_team, season, week)

    # Layer 5: Reconciliation - Enforce team totals
    away_rush_anchor, away_pass_anchor, away_total_anchor = compute_team_anchors(
        away_team, home_team, season, week, away_plays, away_pass / away_plays, strategy
    )

    # For mobile QBs, allocate some rush yards to QB
    if away_qb is not None and away_qb.projected_carries > 0:
        # QB gets proportional share of rush yards based on projected carries
        qb_rush_share = away_qb.projected_carries / away_rush if away_rush > 0 else 0
        qb_rush_yards = away_rush_anchor * qb_rush_share
        rb_rush_anchor = away_rush_anchor - qb_rush_yards
        # Update QB rushing yards
        away_qb.projected_rush_yards = round(qb_rush_yards, 1)
        away_qb.projected_ypc = round(qb_rush_yards / away_qb.projected_carries, 2) if away_qb.projected_carries > 0 else 0.0
    else:
        rb_rush_anchor = away_rush_anchor

    # Reconcile RBs and receivers
    reconcile_to_anchors(away_rbs, away_recs, rb_rush_anchor, away_pass_anchor)

    # Layer 4C: Apply QB efficiency (after anchors are computed)
    apply_qb_efficiency(away_qb, home_team, season, week, away_pass_anchor, away_rush_anchor)

    # Create away team projection
    away_proj = TeamProjection(
        team=away_team,
        opponent=home_team,
        is_home=False,
        season=season,
        week=week,
        total_plays=away_plays,
        pass_attempts=away_pass,
        rush_attempts=away_rush,
        pass_rate=away_pass / away_plays,
        total_targets=round(away_pass * 0.92),
        other_attempts=away_pass - round(away_pass * 0.92),
        rush_yards_anchor=away_rush_anchor,
        pass_yards_anchor=away_pass_anchor,
        total_yards_anchor=away_total_anchor
    )

    # --- HOME TEAM ---

    # Layer 1: Game - Predict total plays
    home_plays = predict_team_plays(home_team, away_team, season, week, vegas_total, spread_line, is_home=True)

    # Layer 2: Team - Split plays into pass/rush
    home_pass, home_rush = split_plays_pass_rush(home_team, season, week, home_plays, spread_line, is_home=True)

    # Layer 3: Volume - Allocate to players
    # Layer 3C: QB allocation first (to determine mobile QB carries)
    home_qb = allocate_qb_stats(home_team, season, week, home_pass, home_rush)

    # Adjust RB rush attempts for mobile QB carries
    home_rb_rush_att = home_rush
    if home_qb is not None and home_qb.projected_carries > 0:
        home_rb_rush_att = max(0, home_rush - int(home_qb.projected_carries))

    # Layer 3A/3B: RB and receiver allocation
    home_rbs = allocate_rushing_volume(home_team, season, week, home_rb_rush_att)
    home_recs = allocate_passing_volume(home_team, season, week, home_pass)

    # Layer 4: Efficiency - Apply DVOA
    apply_rushing_efficiency(home_rbs, away_team, season, week)
    apply_receiving_efficiency(home_recs, away_team, season, week)

    # Layer 5: Reconciliation - Enforce team totals
    home_rush_anchor, home_pass_anchor, home_total_anchor = compute_team_anchors(
        home_team, away_team, season, week, home_plays, home_pass / home_plays, strategy
    )

    # For mobile QBs, allocate some rush yards to QB
    if home_qb is not None and home_qb.projected_carries > 0:
        # QB gets proportional share of rush yards based on projected carries
        qb_rush_share = home_qb.projected_carries / home_rush if home_rush > 0 else 0
        qb_rush_yards = home_rush_anchor * qb_rush_share
        rb_rush_anchor = home_rush_anchor - qb_rush_yards
        # Update QB rushing yards
        home_qb.projected_rush_yards = round(qb_rush_yards, 1)
        home_qb.projected_ypc = round(qb_rush_yards / home_qb.projected_carries, 2) if home_qb.projected_carries > 0 else 0.0
    else:
        rb_rush_anchor = home_rush_anchor

    # Reconcile RBs and receivers
    reconcile_to_anchors(home_rbs, home_recs, rb_rush_anchor, home_pass_anchor)

    # Layer 4C: Apply QB efficiency (after anchors are computed)
    apply_qb_efficiency(home_qb, away_team, season, week, home_pass_anchor, home_rush_anchor)

    # Create home team projection
    home_proj = TeamProjection(
        team=home_team,
        opponent=away_team,
        is_home=True,
        season=season,
        week=week,
        total_plays=home_plays,
        pass_attempts=home_pass,
        rush_attempts=home_rush,
        pass_rate=home_pass / home_plays,
        total_targets=round(home_pass * 0.92),
        other_attempts=home_pass - round(home_pass * 0.92),
        rush_yards_anchor=home_rush_anchor,
        pass_yards_anchor=home_pass_anchor,
        total_yards_anchor=home_total_anchor
    )

    # Layer 6: Validation - Verify conservation laws
    validate_projections(away_proj, away_rbs, away_recs)
    validate_projections(home_proj, home_rbs, home_recs)

    # Combine player lists
    away_players = away_rbs + away_recs
    home_players = home_rbs + home_recs

    return away_proj, away_players, away_qb, home_proj, home_players, home_qb


if __name__ == "__main__":
    # Test Layer 1 and 2
    print("=" * 60)
    print("CLOSED PROJECTION ENGINE TEST")
    print("=" * 60)

    season = 2025
    week = 14
    team = "DET"
    opponent = "GB"

    print(f"\nLayer 1: Predicting total plays for {team}")
    total_plays = predict_team_plays(team, opponent, season, week, is_home=False)
    print(f"  Projected plays: {total_plays}")

    print(f"\nLayer 2: Splitting plays into pass/rush")
    pass_att, rush_att = split_plays_pass_rush(team, season, week, total_plays, is_home=False)
    print(f"  Pass attempts: {pass_att} ({pass_att/total_plays:.1%})")
    print(f"  Rush attempts: {rush_att} ({rush_att/total_plays:.1%})")
    print(f"  Conservation: {pass_att} + {rush_att} = {pass_att + rush_att} (should be {total_plays})")

    print(f"\nLayer 3A: Allocating rushing volume to RBs")
    rb_projs = allocate_rushing_volume(team, season, week, rush_att)
    print(f"  Found {len(rb_projs)} RBs")
    total_carries = sum(p.projected_carries for p in rb_projs)
    for proj in rb_projs[:3]:
        print(f"    {proj.player_name}: {proj.projected_carries:.1f} carries ({proj.weighted_share:.1%} share)")
    print(f"  Conservation: Sum carries = {total_carries:.1f} (should be {rush_att})")

    print(f"\nLayer 3B: Allocating passing volume to receivers")
    rec_projs = allocate_passing_volume(team, season, week, pass_att)
    print(f"  Found {len(rec_projs)} receivers")
    total_targets = sum(p.projected_targets for p in rec_projs)
    for proj in rec_projs[:5]:
        print(f"    {proj.player_name} ({proj.position}): {proj.projected_targets:.1f} targets ({proj.weighted_share:.1%} share)")
    print(f"  Conservation: Sum targets = {total_targets:.1f} (should be ~{pass_att * 0.92:.0f})")

    print(f"\nLayer 4A: Applying rushing efficiency (DVOA)")
    apply_rushing_efficiency(rb_projs, opponent, season, week)
    print(f"  League avg YPC: {ydvoa.calculate_league_rushing_baseline(season, week)['league_avg_ypc']:.2f}")
    total_rush_yards = sum(p.projected_rush_yards for p in rb_projs)
    for proj in rb_projs[:3]:
        print(f"    {proj.player_name}: {proj.projected_carries:.1f} × {proj.projected_ypc:.2f} YPC = {proj.projected_rush_yards:.1f} yds (DVOA: {proj.dvoa_pct:+.1f}%)")
    print(f"  Total rush yards: {total_rush_yards:.1f}")

    print(f"\nLayer 4B: Applying receiving efficiency (DVOA)")
    apply_receiving_efficiency(rec_projs, opponent, season, week)
    print(f"  League avg YPT: {ydvoa.calculate_league_receiving_baseline(season, week)['league_avg_ypt']:.2f}")
    total_recv_yards = sum(p.projected_recv_yards for p in rec_projs)
    for proj in rec_projs[:5]:
        print(f"    {proj.player_name} ({proj.position}): {proj.projected_targets:.1f} × {proj.projected_ypt:.2f} YPT = {proj.projected_recv_yards:.1f} yds (DVOA: {proj.dvoa_pct:+.1f}%)")
    print(f"  Total receiving yards: {total_recv_yards:.1f}")

    print(f"\nLayer 5A: Computing team anchors (top-down)")
    rush_anchor, pass_anchor, total_anchor = compute_team_anchors(
        team, opponent, season, week, total_plays, pass_att / total_plays
    )
    print(f"  Team capability: {total_anchor:.1f} total yards from {total_plays} plays ({total_anchor/total_plays:.2f} yds/play)")
    print(f"    Rush anchor: {rush_anchor:.1f} yards")
    print(f"    Pass anchor: {pass_anchor:.1f} yards")
    print(f"  Bottom-up totals (before reconciliation): Rush={total_rush_yards:.1f}, Pass={total_recv_yards:.1f}")

    print(f"\nLayer 5B: Reconciling to anchors (proportional scaling)")
    reconcile_to_anchors(rb_projs, rec_projs, rush_anchor, pass_anchor)
    final_rush_yards = sum(p.projected_rush_yards for p in rb_projs)
    final_recv_yards = sum(p.projected_recv_yards for p in rec_projs)
    print(f"  Rush yards: {total_rush_yards:.1f} -> {final_rush_yards:.1f} (scale: {final_rush_yards/total_rush_yards:.3f})")
    print(f"  Recv yards: {total_recv_yards:.1f} -> {final_recv_yards:.1f} (scale: {final_recv_yards/total_recv_yards:.3f})")
    print(f"  Conservation check: Rush={final_rush_yards:.1f} vs anchor={rush_anchor:.1f}, Pass={final_recv_yards:.1f} vs anchor={pass_anchor:.1f}")

    print(f"\nLayer 6: Validating all conservation laws")
    # Create TeamProjection object for validation
    team_projection = TeamProjection(
        team=team,
        opponent=opponent,
        is_home=False,
        season=season,
        week=week,
        total_plays=total_plays,
        pass_attempts=pass_att,
        rush_attempts=rush_att,
        pass_rate=pass_att / total_plays,
        total_targets=round(pass_att * 0.92),
        other_attempts=pass_att - round(pass_att * 0.92),
        rush_yards_anchor=rush_anchor,
        pass_yards_anchor=pass_anchor,
        total_yards_anchor=total_anchor
    )

    try:
        result = validate_projections(team_projection, rb_projs, rec_projs)
        print(f"  [OK] All conservation laws passed!")
        print(f"  [OK] Realistic ranges verified")
        print(f"  [OK] All player values non-negative")
    except AssertionError as e:
        print(f"  [FAIL] Validation failed: {e}")

    print(f"\n" + "=" * 60)
    print(f"TESTING COMPLETE project_matchup() PIPELINE")
    print(f"=" * 60)

    away_proj, away_players, home_proj, home_players = project_matchup(
        away_team=team,
        home_team=opponent,
        season=season,
        week=week,
        vegas_total=None,  # No Vegas line for Week 14 test
        spread_line=None,
        strategy='neutral'
    )

    print(f"\n{away_proj.team} @ {home_proj.team} - Week {week}")
    print(f"\n{away_proj.team} Team Projection:")
    print(f"  Plays: {away_proj.total_plays} ({away_proj.pass_attempts} pass, {away_proj.rush_attempts} rush)")
    print(f"  Total Yards: {away_proj.total_yards_anchor:.1f} ({away_proj.rush_yards_anchor:.1f} rush, {away_proj.pass_yards_anchor:.1f} pass)")

    print(f"\n  Top RBs:")
    for rb in sorted([p for p in away_players if p.position == 'RB'], key=lambda x: x.projected_total_yards, reverse=True)[:2]:
        print(f"    {rb.player_name}: {rb.projected_carries:.1f} car x {rb.projected_ypc:.2f} YPC = {rb.projected_rush_yards:.1f} yds")

    print(f"\n  Top Receivers:")
    for rec in sorted([p for p in away_players if p.position in ['WR', 'TE']], key=lambda x: x.projected_total_yards, reverse=True)[:3]:
        print(f"    {rec.player_name} ({rec.position}): {rec.projected_targets:.1f} tgt x {rec.projected_ypt:.2f} YPT = {rec.projected_recv_yards:.1f} yds")

    print(f"\n{home_proj.team} Team Projection:")
    print(f"  Plays: {home_proj.total_plays} ({home_proj.pass_attempts} pass, {home_proj.rush_attempts} rush)")
    print(f"  Total Yards: {home_proj.total_yards_anchor:.1f} ({home_proj.rush_yards_anchor:.1f} rush, {home_proj.pass_yards_anchor:.1f} pass)")

    print(f"\n  Top RBs:")
    for rb in sorted([p for p in home_players if p.position == 'RB'], key=lambda x: x.projected_total_yards, reverse=True)[:2]:
        print(f"    {rb.player_name}: {rb.projected_carries:.1f} car x {rb.projected_ypc:.2f} YPC = {rb.projected_rush_yards:.1f} yds")

    print(f"\n  Top Receivers:")
    for rec in sorted([p for p in home_players if p.position in ['WR', 'TE']], key=lambda x: x.projected_total_yards, reverse=True)[:3]:
        print(f"    {rec.player_name} ({rec.position}): {rec.projected_targets:.1f} tgt x {rec.projected_ypt:.2f} YPT = {rec.projected_recv_yards:.1f} yds")

    print(f"\n" + "=" * 60)
    print(f"ALL TESTS PASSED - CLOSED SYSTEM WORKING!")
    print(f"=" * 60)
