#!/usr/bin/env python3
"""
NFL Player Impact Analytics Module

Analyzes how player absences affect:
1. Team performance (scoring, offensive efficiency)
2. Teammate usage redistribution (targets, carries, opportunities)
3. Opponent performance changes

NFL-Specific Considerations:
- Weekly games (18 weeks vs 82 NBA games)
- Position-specific stats (QB/RB/WR/TE have different metrics)
- Massive impact from key positions (QB, RB1, WR1)
- Smaller sample sizes require lower thresholds
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd


@dataclass
class PlayerAbsenceStats:
    """Statistics comparing team performance with/without a player."""

    player_name: str
    player_id: str
    position: str
    team: str
    season: int

    # Game counts
    games_played: int
    games_absent: int
    total_team_games: int

    # Team performance WITH player
    team_avg_pts_with: float
    team_avg_pass_yds_with: float
    team_avg_rush_yds_with: float

    # Team performance WITHOUT player
    team_avg_pts_without: float
    team_avg_pass_yds_without: float
    team_avg_rush_yds_without: float

    # Impact deltas (negative = team worse without player)
    pts_delta: float
    pass_yds_delta: float
    rush_yds_delta: float


@dataclass
class TeammateImpact:
    """How a teammate's stats change when the key player is absent."""

    teammate_name: str
    teammate_id: str
    position: str

    # WITH key player present
    avg_fantasy_pts_with: float
    avg_targets_with: float  # For pass catchers
    avg_carries_with: float  # For RBs
    avg_yards_with: float

    # WITHOUT key player (when they're absent)
    avg_fantasy_pts_without: float
    avg_targets_without: float
    avg_carries_without: float
    avg_yards_without: float

    # Deltas (positive = teammate BENEFITS from absence - DFS GOLD!)
    fantasy_pts_delta: float
    targets_delta: float
    carries_delta: float
    yards_delta: float

    # Sample sizes
    games_together: int
    games_apart: int

    # DFS Value Score (0-100, higher = better DFS play when star is out)
    dfs_value_score: float


@dataclass
class OpponentImpact:
    """How opponents perform differently when a key player is absent."""

    # Opponent stats when key player PRESENT
    avg_opp_pts_with: float
    avg_opp_pass_yds_with: float
    avg_opp_rush_yds_with: float

    # Opponent stats when key player ABSENT
    avg_opp_pts_without: float
    avg_opp_pass_yds_without: float
    avg_opp_rush_yds_without: float

    # Deltas (positive = opponents perform BETTER without the player)
    opp_pts_delta: float
    opp_pass_yds_delta: float
    opp_rush_yds_delta: float


def get_significant_players(
    conn: sqlite3.Connection,
    season: int = 2024,
    season_type: str = "REG",
    min_games: int = 5,
    min_avg_fantasy_pts: float = 8.0,
) -> pd.DataFrame:
    """
    Get list of significant players for analysis.

    Args:
        conn: Database connection
        season: NFL season year
        season_type: REG, POST, PRE
        min_games: Minimum games played
        min_avg_fantasy_pts: Minimum fantasy points per game

    Returns:
        DataFrame with columns: player_id, player_name, position, team, avg_fantasy_pts, games
    """
    query = """
    SELECT
        player_id,
        player_display_name as player_name,
        position,
        recent_team as team,
        COUNT(*) as games,
        ROUND(AVG(fantasy_points_ppr), 2) as avg_fantasy_pts,
        ROUND(AVG(targets), 1) as avg_targets,
        ROUND(AVG(carries), 1) as avg_carries
    FROM player_week_stats
    WHERE season = ?
      AND season_type = ?
      AND fantasy_points_ppr IS NOT NULL
    GROUP BY player_id, player_display_name, position, recent_team
    HAVING COUNT(*) >= ?
       AND AVG(fantasy_points_ppr) >= ?
    ORDER BY avg_fantasy_pts DESC
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=[season, season_type, min_games, min_avg_fantasy_pts]
    )

    return df


def get_player_absences(
    conn: sqlite3.Connection,
    player_id: str,
    season: int = 2024,
    season_type: str = "REG",
) -> Tuple[List[int], List[int]]:
    """
    Identify weeks where a player was present vs. absent.

    Args:
        conn: Database connection
        player_id: NFL player ID
        season: Season year
        season_type: REG, POST, PRE

    Returns:
        (weeks_with_player, weeks_without_player) - Two lists of week numbers
    """
    # First, get the player's team(s) for this season
    team_query = """
    SELECT DISTINCT recent_team
    FROM player_week_stats
    WHERE player_id = ? AND season = ? AND season_type = ?
    """

    team_df = pd.read_sql_query(
        team_query,
        conn,
        params=[player_id, season, season_type]
    )

    if team_df.empty:
        return [], []

    player_teams = team_df['recent_team'].tolist()

    # Get all weeks where the team(s) actually played games (from schedule table)
    # This excludes bye weeks automatically and ensures we only count real games
    placeholders = ','.join('?' * len(player_teams))

    # Map season_type to game_type for schedule table
    if season_type == "POST":
        # For playoffs, include all playoff game types
        schedule_query = f"""
        SELECT DISTINCT week
        FROM schedule
        WHERE season = ?
          AND game_type IN ('WC', 'DIV', 'CON', 'SB')
          AND (away_team IN ({placeholders}) OR home_team IN ({placeholders}))
        ORDER BY week
        """
        params = [season] + player_teams + player_teams
    else:
        # For REG and PRE, use exact game_type match
        schedule_query = f"""
        SELECT DISTINCT week
        FROM schedule
        WHERE season = ?
          AND game_type = ?
          AND (away_team IN ({placeholders}) OR home_team IN ({placeholders}))
        ORDER BY week
        """
        params = [season, season_type] + player_teams + player_teams

    all_weeks_df = pd.read_sql_query(schedule_query, conn, params=params)
    all_weeks = set(all_weeks_df['week'].tolist()) if not all_weeks_df.empty else set()

    # Get weeks where player actually had stats (played and got touches)
    player_weeks_query = """
    SELECT DISTINCT week
    FROM player_week_stats
    WHERE player_id = ?
      AND season = ?
      AND season_type = ?
    """

    player_weeks_df = pd.read_sql_query(
        player_weeks_query,
        conn,
        params=[player_id, season, season_type]
    )
    weeks_played = set(player_weeks_df['week'].tolist())

    # Weeks absent = team game weeks - weeks player had stats
    # NOTE: This assumes players without stats were inactive/absent
    # Players who were active but had 0 touches will incorrectly show as absent
    weeks_absent = all_weeks - weeks_played

    return sorted(list(weeks_played)), sorted(list(weeks_absent))


def calculate_team_impact(
    conn: sqlite3.Connection,
    player_id: str,
    player_name: str,
    season: int = 2024,
    season_type: str = "REG",
) -> Optional[PlayerAbsenceStats]:
    """
    Calculate how team performs with/without this player.

    Args:
        conn: Database connection
        player_id: NFL player ID
        player_name: Player display name
        season: Season year
        season_type: REG, POST, PRE

    Returns:
        PlayerAbsenceStats object or None if insufficient data
    """
    weeks_with, weeks_without = get_player_absences(conn, player_id, season, season_type)

    # Need at least 1 game with and 1 game without for comparison
    if len(weeks_with) < 1 or len(weeks_without) < 1:
        return None

    # Get player's team and position
    player_info_query = """
    SELECT position, recent_team
    FROM player_week_stats
    WHERE player_id = ? AND season = ? AND season_type = ?
    LIMIT 1
    """

    player_info = pd.read_sql_query(
        player_info_query,
        conn,
        params=[player_id, season, season_type]
    )

    if player_info.empty:
        return None

    position = player_info.iloc[0]['position']
    team = player_info.iloc[0]['recent_team']

    # Get team stats for weeks WITH player
    team_with_query = """
    SELECT
        AVG(team_total_pts) as avg_pts,
        AVG(team_pass_yds) as avg_pass_yds,
        AVG(team_rush_yds) as avg_rush_yds
    FROM (
        SELECT
            week,
            SUM(CASE
                WHEN position = 'QB' THEN passing_yards
                WHEN position IN ('RB', 'WR', 'TE') THEN receiving_yards
                ELSE 0
            END) as team_pass_yds,
            SUM(rushing_yards) as team_rush_yds,
            SUM(passing_tds * 4 + rushing_tds * 6 + receiving_tds * 6) as team_total_pts
        FROM player_week_stats
        WHERE recent_team = ?
          AND season = ?
          AND season_type = ?
          AND week IN ({})
        GROUP BY week
    )
    """.format(','.join('?' * len(weeks_with)))

    team_with_stats = pd.read_sql_query(
        team_with_query,
        conn,
        params=[team, season, season_type] + weeks_with
    )

    # Get team stats for weeks WITHOUT player
    team_without_query = """
    SELECT
        AVG(team_total_pts) as avg_pts,
        AVG(team_pass_yds) as avg_pass_yds,
        AVG(team_rush_yds) as avg_rush_yds
    FROM (
        SELECT
            week,
            SUM(CASE
                WHEN position = 'QB' THEN passing_yards
                WHEN position IN ('RB', 'WR', 'TE') THEN receiving_yards
                ELSE 0
            END) as team_pass_yds,
            SUM(rushing_yards) as team_rush_yds,
            SUM(passing_tds * 4 + rushing_tds * 6 + receiving_tds * 6) as team_total_pts
        FROM player_week_stats
        WHERE recent_team = ?
          AND season = ?
          AND season_type = ?
          AND week IN ({})
        GROUP BY week
    )
    """.format(','.join('?' * len(weeks_without)))

    team_without_stats = pd.read_sql_query(
        team_without_query,
        conn,
        params=[team, season, season_type] + weeks_without
    )

    # Calculate deltas
    pts_with = team_with_stats.iloc[0]['avg_pts'] or 0
    pts_without = team_without_stats.iloc[0]['avg_pts'] or 0
    pass_yds_with = team_with_stats.iloc[0]['avg_pass_yds'] or 0
    pass_yds_without = team_without_stats.iloc[0]['avg_pass_yds'] or 0
    rush_yds_with = team_with_stats.iloc[0]['avg_rush_yds'] or 0
    rush_yds_without = team_without_stats.iloc[0]['avg_rush_yds'] or 0

    return PlayerAbsenceStats(
        player_name=player_name,
        player_id=player_id,
        position=position,
        team=team,
        season=season,
        games_played=len(weeks_with),
        games_absent=len(weeks_without),
        total_team_games=len(weeks_with) + len(weeks_without),
        team_avg_pts_with=pts_with,
        team_avg_pass_yds_with=pass_yds_with,
        team_avg_rush_yds_with=rush_yds_with,
        team_avg_pts_without=pts_without,
        team_avg_pass_yds_without=pass_yds_without,
        team_avg_rush_yds_without=rush_yds_without,
        pts_delta=pts_without - pts_with,
        pass_yds_delta=pass_yds_without - pass_yds_with,
        rush_yds_delta=rush_yds_without - rush_yds_with,
    )


def calculate_teammate_redistribution(
    conn: sqlite3.Connection,
    player_id: str,
    season: int = 2024,
    season_type: str = "REG",
    min_games: int = 1,
) -> List[TeammateImpact]:
    """
    Calculate how teammates' stats change when this player is absent.

    This is THE KEY FUNCTION for DFS - identifies value plays when stars are out!

    Args:
        conn: Database connection
        player_id: NFL player ID
        season: Season year
        season_type: REG, POST, PRE
        min_games: Minimum games together/apart

    Returns:
        List of TeammateImpact objects sorted by DFS value score (best first)
    """
    weeks_with, weeks_without = get_player_absences(conn, player_id, season, season_type)

    # Need absences to analyze impact
    if len(weeks_without) < min_games:
        return []

    # Get player's team
    team_query = """
    SELECT DISTINCT recent_team
    FROM player_week_stats
    WHERE player_id = ? AND season = ? AND season_type = ?
    LIMIT 1
    """

    team_df = pd.read_sql_query(team_query, conn, params=[player_id, season, season_type])
    if team_df.empty:
        return []

    team = team_df.iloc[0]['recent_team']

    # Get ALL teammates who played in both scenarios
    teammates_query = """
    SELECT DISTINCT player_id, player_display_name, position
    FROM player_week_stats
    WHERE recent_team = ?
      AND season = ?
      AND season_type = ?
      AND player_id != ?
      AND position IN ('QB', 'RB', 'WR', 'TE')  -- Fantasy-relevant positions only
    """

    teammates = pd.read_sql_query(
        teammates_query,
        conn,
        params=[team, season, season_type, player_id]
    )

    teammate_impacts = []

    for _, teammate_row in teammates.iterrows():
        teammate_id = teammate_row['player_id']
        teammate_name = teammate_row['player_display_name']
        teammate_pos = teammate_row['position']

        # Stats WITH key player (weeks_with)
        if weeks_with:
            with_query = """
            SELECT
                AVG(fantasy_points_ppr) as avg_fantasy_pts,
                AVG(targets) as avg_targets,
                AVG(carries) as avg_carries,
                AVG(receiving_yards + rushing_yards + passing_yards) as avg_yards,
                COUNT(*) as games
            FROM player_week_stats
            WHERE player_id = ?
              AND season = ?
              AND season_type = ?
              AND week IN ({})
            """.format(','.join('?' * len(weeks_with)))

            with_stats = pd.read_sql_query(
                with_query,
                conn,
                params=[teammate_id, season, season_type] + weeks_with
            )
        else:
            with_stats = pd.DataFrame([{
                'avg_fantasy_pts': 0,
                'avg_targets': 0,
                'avg_carries': 0,
                'avg_yards': 0,
                'games': 0
            }])

        # Stats WITHOUT key player (weeks_without)
        if weeks_without:
            without_query = """
            SELECT
                AVG(fantasy_points_ppr) as avg_fantasy_pts,
                AVG(targets) as avg_targets,
                AVG(carries) as avg_carries,
                AVG(receiving_yards + rushing_yards + passing_yards) as avg_yards,
                COUNT(*) as games
            FROM player_week_stats
            WHERE player_id = ?
              AND season = ?
              AND season_type = ?
              AND week IN ({})
            """.format(','.join('?' * len(weeks_without)))

            without_stats = pd.read_sql_query(
                without_query,
                conn,
                params=[teammate_id, season, season_type] + weeks_without
            )
        else:
            without_stats = pd.DataFrame([{
                'avg_fantasy_pts': 0,
                'avg_targets': 0,
                'avg_carries': 0,
                'avg_yards': 0,
                'games': 0
            }])

        games_together = with_stats.iloc[0]['games'] or 0
        games_apart = without_stats.iloc[0]['games'] or 0

        # Skip if insufficient sample size
        if games_together < min_games or games_apart < min_games:
            continue

        # Extract averages
        fantasy_with = with_stats.iloc[0]['avg_fantasy_pts'] or 0
        fantasy_without = without_stats.iloc[0]['avg_fantasy_pts'] or 0
        targets_with = with_stats.iloc[0]['avg_targets'] or 0
        targets_without = without_stats.iloc[0]['avg_targets'] or 0
        carries_with = with_stats.iloc[0]['avg_carries'] or 0
        carries_without = without_stats.iloc[0]['avg_carries'] or 0
        yards_with = with_stats.iloc[0]['avg_yards'] or 0
        yards_without = without_stats.iloc[0]['avg_yards'] or 0

        # Calculate deltas
        fantasy_delta = fantasy_without - fantasy_with
        targets_delta = targets_without - targets_with
        carries_delta = carries_without - carries_with
        yards_delta = yards_without - yards_with

        # Calculate DFS Value Score (0-100)
        # Higher = better DFS play when key player is out
        dfs_score = 50  # Base score

        # Fantasy points delta (most important)
        if fantasy_delta >= 5:
            dfs_score += 30
        elif fantasy_delta >= 3:
            dfs_score += 20
        elif fantasy_delta >= 1:
            dfs_score += 10
        elif fantasy_delta <= -3:
            dfs_score -= 20

        # Opportunity increase (targets/carries)
        if teammate_pos in ('WR', 'TE') and targets_delta >= 2:
            dfs_score += 15
        elif teammate_pos == 'RB' and carries_delta >= 3:
            dfs_score += 15

        # Sample size bonus (more data = more confident)
        if games_apart >= 3:
            dfs_score += 10
        elif games_apart >= 2:
            dfs_score += 5

        # Cap score at 100
        dfs_score = min(100, max(0, dfs_score))

        teammate_impacts.append(TeammateImpact(
            teammate_name=teammate_name,
            teammate_id=teammate_id,
            position=teammate_pos,
            avg_fantasy_pts_with=fantasy_with,
            avg_targets_with=targets_with,
            avg_carries_with=carries_with,
            avg_yards_with=yards_with,
            avg_fantasy_pts_without=fantasy_without,
            avg_targets_without=targets_without,
            avg_carries_without=carries_without,
            avg_yards_without=yards_without,
            fantasy_pts_delta=fantasy_delta,
            targets_delta=targets_delta,
            carries_delta=carries_delta,
            yards_delta=yards_delta,
            games_together=games_together,
            games_apart=games_apart,
            dfs_value_score=dfs_score
        ))

    # Sort by DFS value score (best DFS plays first)
    teammate_impacts.sort(key=lambda x: x.dfs_value_score, reverse=True)

    return teammate_impacts


def calculate_opponent_impact(
    conn: sqlite3.Connection,
    player_id: str,
    season: int = 2024,
    season_type: str = "REG",
) -> Optional[OpponentImpact]:
    """
    Calculate how opponents perform differently when this player is absent.

    Useful for understanding defensive impact of players.

    Args:
        conn: Database connection
        player_id: NFL player ID
        season: Season year
        season_type: REG, POST, PRE

    Returns:
        OpponentImpact object or None if insufficient data
    """
    weeks_with, weeks_without = get_player_absences(conn, player_id, season, season_type)

    if len(weeks_with) < 1 or len(weeks_without) < 1:
        return None

    # Get player's team
    team_query = """
    SELECT DISTINCT recent_team
    FROM player_week_stats
    WHERE player_id = ? AND season = ? AND season_type = ?
    LIMIT 1
    """

    team_df = pd.read_sql_query(team_query, conn, params=[player_id, season, season_type])
    if team_df.empty:
        return None

    team = team_df.iloc[0]['recent_team']

    # Get opponent stats from schedule (simplified - would need schedule table join)
    # For now, return None as this requires schedule data
    # TODO: Implement with schedule table join

    return None
