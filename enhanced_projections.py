"""
Enhanced Receiving Yards & TD Projections with DFS Scoring

This module provides advanced player projections that incorporate:
- Defense-adjusted team projections (60% offense, 40% defense)
- Recency-weighted target shares (70% last 4 games, 30% season)
- Position-specific defensive matchups (WR vs TE vulnerabilities)
- Air Yards vs YAC style matchup multipliers
- TD probability modeling (red zone targets, team efficiency, defensive vulnerability)
- Comprehensive DFS scoring (0-100 scale)
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

# Database path
PROJECT_DIR = Path(__file__).parent
DB_PATH = PROJECT_DIR / "data" / "nfl_merged.db"

# Constants
TEAM_YARDS_CAP = 425  # 90th percentile max team receiving yards
TEAM_YARDS_FLOOR = 100  # Minimum team receiving yards
TD_CAP_PER_GAME = 2.0  # Maximum TDs for individual player
RECENT_GAMES_WINDOW = 4  # Games to use for recency weighting
RECENCY_WEIGHT = 0.70  # Weight given to recent games vs season average


def get_offensive_team_stats(season=2025):
    """
    Get offensive team statistics for receiving yards, air yards, YAC.

    Returns DataFrame with columns:
    - team
    - games
    - total_rec_yards
    - avg_rec_yards_per_game
    - total_air_yards
    - total_yac
    - air_pct
    - yac_pct
    """
    conn = sqlite3.connect(str(DB_PATH))

    query = '''
    SELECT
        team,
        COUNT(DISTINCT week) as games,
        SUM(receiving_yards) as total_rec_yards,
        SUM(receiving_air_yards) as total_air_yards,
        SUM(receiving_yards_after_catch) as total_yac
    FROM player_stats
    WHERE season = ?
      AND season_type = 'REG'
      AND receiving_yards > 0
    GROUP BY team
    '''

    df = pd.read_sql_query(query, conn, params=(season,))
    conn.close()

    # Calculate per-game averages and percentages
    df['avg_rec_yards_per_game'] = (df['total_rec_yards'] / df['games']).round(1)
    df['air_pct'] = ((df['total_air_yards'] / df['total_rec_yards'].replace(0, 1)) * 100).round(1)
    df['yac_pct'] = ((df['total_yac'] / df['total_rec_yards'].replace(0, 1)) * 100).round(1)

    return df


def get_defensive_team_stats(season=2025):
    """
    Get defensive team statistics for receiving yards allowed, by position.

    Returns DataFrame with columns:
    - defense (team)
    - games
    - total_rec_allowed
    - avg_rec_allowed_per_game
    - wr_yards_allowed_per_game
    - te_yards_allowed_per_game
    - wr_to_te_ratio (positional vulnerability indicator)
    - air_allowed_pct
    - yac_allowed_pct
    - tds_allowed_to_wr_per_game
    - tds_allowed_to_te_per_game
    """
    conn = sqlite3.connect(str(DB_PATH))

    # Get total receiving yards allowed
    query_total = '''
    SELECT
        opponent_team as defense,
        COUNT(DISTINCT week) as games,
        SUM(receiving_yards) as total_rec_allowed,
        SUM(receiving_air_yards) as total_air_allowed,
        SUM(receiving_yards_after_catch) as total_yac_allowed
    FROM player_stats
    WHERE season = ?
      AND season_type = 'REG'
      AND receiving_yards > 0
    GROUP BY opponent_team
    '''

    df_total = pd.read_sql_query(query_total, conn, params=(season,))

    # Get position-specific yards allowed
    query_position = '''
    SELECT
        opponent_team as defense,
        position,
        SUM(receiving_yards) / COUNT(DISTINCT week) as yards_per_game,
        SUM(receiving_tds) / COUNT(DISTINCT week) as tds_per_game
    FROM player_stats
    WHERE season = ?
      AND season_type = 'REG'
      AND position IN ('WR', 'TE')
      AND receiving_yards > 0
    GROUP BY opponent_team, position
    '''

    df_position = pd.read_sql_query(query_position, conn, params=(season,))
    conn.close()

    # Pivot to get WR and TE columns
    df_wr = df_position[df_position['position'] == 'WR'][['defense', 'yards_per_game', 'tds_per_game']].rename(
        columns={'yards_per_game': 'wr_yards_allowed_per_game', 'tds_per_game': 'tds_allowed_to_wr_per_game'}
    )
    df_te = df_position[df_position['position'] == 'TE'][['defense', 'yards_per_game', 'tds_per_game']].rename(
        columns={'yards_per_game': 'te_yards_allowed_per_game', 'tds_per_game': 'tds_allowed_to_te_per_game'}
    )

    # Merge everything together
    df = df_total.merge(df_wr, on='defense', how='left')
    df = df.merge(df_te, on='defense', how='left')

    # Calculate derived metrics
    df['avg_rec_allowed_per_game'] = (df['total_rec_allowed'] / df['games']).round(1)
    df['wr_to_te_ratio'] = (df['wr_yards_allowed_per_game'] / df['te_yards_allowed_per_game'].replace(0, 1)).round(2)
    df['air_allowed_pct'] = ((df['total_air_allowed'] / df['total_rec_allowed'].replace(0, 1)) * 100).round(1)
    df['yac_allowed_pct'] = ((df['total_yac_allowed'] / df['total_rec_allowed'].replace(0, 1)) * 100).round(1)

    # Fill NaN values
    df = df.fillna(0)

    return df


def get_player_target_share(player_name, team, season=2025, recent_only=False):
    """
    Get player's target share (season average or recent games).

    Args:
        player_name: Player display name
        team: Player's team
        season: Season year
        recent_only: If True, only use last N games (defined by RECENT_GAMES_WINDOW)

    Returns dict with:
    - target_share: Percentage of team targets (0-100)
    - avg_rec_yards: Average receiving yards per game
    - avg_receptions: Average receptions per game
    - avg_tds: Average TDs per game
    - games: Number of games
    """
    conn = sqlite3.connect(str(DB_PATH))

    if recent_only:
        # Get last N games
        query = '''
        SELECT
            week,
            target_share,
            receiving_yards,
            receptions,
            receiving_tds
        FROM player_stats
        WHERE player_display_name = ?
          AND team = ?
          AND season = ?
          AND season_type = 'REG'
          AND targets > 0
        ORDER BY week DESC
        LIMIT ?
        '''
        df = pd.read_sql_query(query, conn, params=(player_name, team, season, RECENT_GAMES_WINDOW))
    else:
        # Get full season
        query = '''
        SELECT
            week,
            target_share,
            receiving_yards,
            receptions,
            receiving_tds
        FROM player_stats
        WHERE player_display_name = ?
          AND team = ?
          AND season = ?
          AND season_type = 'REG'
          AND targets > 0
        ORDER BY week DESC
        '''
        df = pd.read_sql_query(query, conn, params=(player_name, team, season))

    conn.close()

    if df.empty:
        return None

    return {
        'target_share': (df['target_share'].mean() * 100).round(1),
        'avg_rec_yards': df['receiving_yards'].mean().round(1),
        'avg_receptions': df['receptions'].mean().round(1),
        'avg_tds': df['receiving_tds'].mean().round(2),
        'games': len(df)
    }


def get_player_red_zone_share(player_name, team, season=2025):
    """
    Get player's red zone target share and TD efficiency.

    Note: This is a simplified version. True red zone data requires play-by-play.
    We approximate using TD rate and target share.

    Returns dict with:
    - estimated_rz_share: Estimated red zone target share
    - td_rate: TDs per reception percentage
    - td_per_game: Average TDs per game
    """
    conn = sqlite3.connect(str(DB_PATH))

    query = '''
    SELECT
        SUM(receiving_tds) as total_tds,
        SUM(receptions) as total_receptions,
        SUM(targets) as total_targets,
        COUNT(DISTINCT week) as games
    FROM player_stats
    WHERE player_display_name = ?
      AND team = ?
      AND season = ?
      AND season_type = 'REG'
    '''

    df = pd.read_sql_query(query, conn, params=(player_name, team, season))
    conn.close()

    if df.empty or df.iloc[0]['total_receptions'] == 0:
        return None

    row = df.iloc[0]
    td_rate = (row['total_tds'] / row['total_receptions'] * 100) if row['total_receptions'] > 0 else 0
    td_per_game = row['total_tds'] / row['games'] if row['games'] > 0 else 0

    # Rough estimation: players with high TD rates likely get more red zone targets
    # This is a proxy - ideal would be actual red zone target data
    estimated_rz_share = min(td_rate * 0.5, 35)  # Cap at 35% estimated RZ share

    return {
        'estimated_rz_share': round(estimated_rz_share, 1),
        'td_rate': round(td_rate, 1),
        'td_per_game': round(td_per_game, 2)
    }


def calculate_position_matchup_multiplier(position, opponent_defense_stats):
    """
    Calculate position-specific matchup multiplier based on defensive tendencies.

    Args:
        position: 'WR' or 'TE'
        opponent_defense_stats: Row from defensive stats DataFrame

    Returns multiplier (0.5 to 1.5):
    - < 1.0 = tough matchup (defense locks down this position)
    - 1.0 = neutral matchup
    - > 1.0 = favorable matchup (defense vulnerable to this position)
    """
    if opponent_defense_stats is None or opponent_defense_stats.empty:
        return 1.0

    stats = opponent_defense_stats.iloc[0] if isinstance(opponent_defense_stats, pd.DataFrame) else opponent_defense_stats

    wr_to_te_ratio = stats.get('wr_to_te_ratio', 1.0)

    if position == 'WR':
        # If ratio > 2.0, defense allows way more to WRs than TEs → good for WRs
        # If ratio < 0.5, defense locks down WRs → bad for WRs
        if wr_to_te_ratio > 2.5:
            return 1.3  # Very favorable for WRs
        elif wr_to_te_ratio > 1.8:
            return 1.15  # Favorable for WRs
        elif wr_to_te_ratio < 0.7:
            return 0.7  # Tough for WRs (they focus on stopping WRs)
        else:
            return 1.0

    elif position == 'TE':
        # Inverse logic for TEs
        # If ratio > 2.5, defense allows little to TEs → bad for TEs
        # If ratio < 1.0, defense allows more to TEs → good for TEs
        if wr_to_te_ratio > 3.0:
            return 0.66  # Very tough for TEs (TE-lockdown defense)
        elif wr_to_te_ratio > 2.0:
            return 0.85  # Tough for TEs
        elif wr_to_te_ratio < 1.0:
            return 1.25  # Favorable for TEs
        else:
            return 1.0

    return 1.0


def calculate_style_matchup_multiplier(offense_stats, defense_stats):
    """
    Calculate style matchup multiplier based on Air Yards vs YAC alignment.

    Args:
        offense_stats: Row from offensive stats DataFrame
        defense_stats: Row from defensive stats DataFrame

    Returns multiplier (0.8 to 1.2):
    - Air-heavy offense vs air-vulnerable defense = 1.2
    - YAC-heavy offense vs YAC-vulnerable defense = 1.2
    - Mismatches (air offense vs air-strong defense) = 0.9
    """
    if offense_stats is None or defense_stats is None:
        return 1.0

    off_stats = offense_stats.iloc[0] if isinstance(offense_stats, pd.DataFrame) else offense_stats
    def_stats = defense_stats.iloc[0] if isinstance(defense_stats, pd.DataFrame) else defense_stats

    off_air_pct = off_stats.get('air_pct', 50)
    off_yac_pct = off_stats.get('yac_pct', 50)
    def_air_pct = def_stats.get('air_allowed_pct', 50)
    def_yac_pct = def_stats.get('yac_allowed_pct', 50)

    # Check for style alignment
    # Air offense (>55%) vs air-vulnerable defense (>55%)
    if off_air_pct > 55 and def_air_pct > 55:
        return 1.20  # Great matchup for deep ball offense

    # YAC offense (>50%) vs YAC-vulnerable defense (>50%)
    elif off_yac_pct > 50 and def_yac_pct > 50:
        return 1.15  # Great matchup for YAC offense

    # Air offense vs YAC-vulnerable defense (mismatch)
    elif off_air_pct > 55 and def_yac_pct > 50:
        return 0.95  # Slight mismatch

    # YAC offense vs air-vulnerable defense (mismatch)
    elif off_yac_pct > 50 and def_air_pct > 55:
        return 0.95  # Slight mismatch

    # Air offense vs air-strong defense (<45% air allowed)
    elif off_air_pct > 55 and def_air_pct < 45:
        return 0.85  # Tough matchup

    else:
        return 1.0  # Neutral


def calculate_team_receiving_yards_projection(offense_team, defense_team, offense_stats_df, defense_stats_df):
    """
    Calculate defense-adjusted team receiving yards projection.

    Formula: (Offense Avg * 0.6) + (Defense Allowed Avg * 0.4)
    Then apply sanity caps.

    Returns projected team receiving yards (capped at TEAM_YARDS_CAP).
    """
    # Get team stats
    off_stats = offense_stats_df[offense_stats_df['team'] == offense_team]
    def_stats = defense_stats_df[defense_stats_df['defense'] == defense_team]

    if off_stats.empty or def_stats.empty:
        return None

    off_avg = off_stats.iloc[0]['avg_rec_yards_per_game']
    def_avg = def_stats.iloc[0]['avg_rec_allowed_per_game']

    # 60% offense capability, 40% defensive vulnerability
    projection = (off_avg * 0.6) + (def_avg * 0.4)

    # Apply caps
    projection = max(TEAM_YARDS_FLOOR, min(TEAM_YARDS_CAP, projection))

    return round(projection, 1)


def calculate_td_projection(player_name, team, opponent_team, position, season=2025):
    """
    Calculate TD projection for a player based on:
    - Historical TD rate (recency-weighted)
    - Red zone target share
    - Team offensive efficiency (PPG)
    - Opponent defensive TD vulnerability by position

    Returns projected TDs per game (capped at TD_CAP_PER_GAME).
    """
    conn = sqlite3.connect(str(DB_PATH))

    # Get player's recent TD performance (last 4 games)
    query_recent = '''
    SELECT
        AVG(receiving_tds) as avg_tds_recent
    FROM (
        SELECT receiving_tds
        FROM player_stats
        WHERE player_display_name = ?
          AND team = ?
          AND season = ?
          AND season_type = 'REG'
        ORDER BY week DESC
        LIMIT ?
    )
    '''
    df_recent = pd.read_sql_query(query_recent, conn, params=(player_name, team, season, RECENT_GAMES_WINDOW))

    # Get player's season TD rate
    query_season = '''
    SELECT
        SUM(receiving_tds) / COUNT(DISTINCT week) as avg_tds_season
    FROM player_stats
    WHERE player_display_name = ?
      AND team = ?
      AND season = ?
      AND season_type = 'REG'
    '''
    df_season = pd.read_sql_query(query_season, conn, params=(player_name, team, season))

    # Get opponent's TD vulnerability for this position
    query_defense = '''
    SELECT
        SUM(receiving_tds) / COUNT(DISTINCT week) as tds_allowed_per_game
    FROM player_stats
    WHERE opponent_team = ?
      AND position = ?
      AND season = ?
      AND season_type = 'REG'
    '''
    df_defense = pd.read_sql_query(query_defense, conn, params=(opponent_team, position, season))

    conn.close()

    # Calculate weighted TD projection
    avg_tds_recent = df_recent.iloc[0]['avg_tds_recent'] if not df_recent.empty and df_recent.iloc[0]['avg_tds_recent'] else 0
    avg_tds_season = df_season.iloc[0]['avg_tds_season'] if not df_season.empty and df_season.iloc[0]['avg_tds_season'] else 0
    tds_allowed = df_defense.iloc[0]['tds_allowed_per_game'] if not df_defense.empty and df_defense.iloc[0]['tds_allowed_per_game'] else 0

    # Weighted average: 70% recent, 30% season
    player_td_rate = (avg_tds_recent * RECENCY_WEIGHT) + (avg_tds_season * (1 - RECENCY_WEIGHT))

    # Blend player rate (60%) with defensive vulnerability (40%)
    projected_tds = (player_td_rate * 0.6) + (tds_allowed * 0.4)

    # Red zone boost: if player has high TD rate (>15%), give 1.2x boost
    rz_data = get_player_red_zone_share(player_name, team, season)
    if rz_data and rz_data['td_rate'] > 15:
        projected_tds *= 1.2

    # Apply cap
    projected_tds = min(TD_CAP_PER_GAME, projected_tds)

    return round(projected_tds, 2)


def calculate_enhanced_receiving_projection(
    player_name,
    team,
    opponent_team,
    position,
    season=2025
):
    """
    Calculate comprehensive enhanced receiving projection for a player.

    Returns dict with:
    - projected_rec_yards: Enhanced receiving yards projection
    - projected_rec_yards_low: Low confidence bound (75%)
    - projected_rec_yards_high: High confidence bound (125%)
    - projected_tds: TD projection
    - projected_receptions: Receptions projection
    - target_share_pct: Recency-weighted target share
    - position_matchup_multiplier: Position matchup quality
    - style_matchup_multiplier: Air/YAC style matchup quality
    - matchup_grade: Overall matchup grade (A+ to D)
    """
    # Get team stats
    offense_stats_df = get_offensive_team_stats(season)
    defense_stats_df = get_defensive_team_stats(season)

    # Calculate team receiving yards projection
    team_proj = calculate_team_receiving_yards_projection(
        team, opponent_team, offense_stats_df, defense_stats_df
    )

    if team_proj is None:
        return None

    # Get player target shares (recency-weighted)
    recent_share_data = get_player_target_share(player_name, team, season, recent_only=True)
    season_share_data = get_player_target_share(player_name, team, season, recent_only=False)

    if not recent_share_data or not season_share_data:
        return None

    # Weighted target share: 70% recent, 30% season
    weighted_target_share = (
        (recent_share_data['target_share'] * RECENCY_WEIGHT) +
        (season_share_data['target_share'] * (1 - RECENCY_WEIGHT))
    )

    # Get offense and defense stats for this matchup
    off_stats = offense_stats_df[offense_stats_df['team'] == team]
    def_stats = defense_stats_df[defense_stats_df['defense'] == opponent_team]

    # Calculate multipliers
    position_mult = calculate_position_matchup_multiplier(position, def_stats)
    style_mult = calculate_style_matchup_multiplier(off_stats, def_stats)

    # Calculate player receiving yards projection
    # Team projection * target share * position matchup * style matchup
    base_proj = team_proj * (weighted_target_share / 100) * position_mult * style_mult

    # Confidence ranges
    proj_low = base_proj * 0.75
    proj_high = base_proj * 1.25

    # Calculate TD projection
    projected_tds = calculate_td_projection(player_name, team, opponent_team, position, season)

    # Calculate receptions projection (based on historical catch rate)
    avg_receptions = (
        (recent_share_data['avg_receptions'] * RECENCY_WEIGHT) +
        (season_share_data['avg_receptions'] * (1 - RECENCY_WEIGHT))
    )

    # Matchup grade based on combined multipliers
    combined_mult = position_mult * style_mult
    if combined_mult >= 1.4:
        matchup_grade = "A+"
    elif combined_mult >= 1.25:
        matchup_grade = "A"
    elif combined_mult >= 1.1:
        matchup_grade = "B+"
    elif combined_mult >= 0.95:
        matchup_grade = "B"
    elif combined_mult >= 0.85:
        matchup_grade = "C"
    else:
        matchup_grade = "D"

    return {
        'projected_rec_yards': round(base_proj, 1),
        'projected_rec_yards_low': round(proj_low, 1),
        'projected_rec_yards_high': round(proj_high, 1),
        'projected_tds': projected_tds,
        'projected_receptions': round(avg_receptions, 1),
        'target_share_pct': round(weighted_target_share, 1),
        'position_matchup_multiplier': round(position_mult, 2),
        'style_matchup_multiplier': round(style_mult, 2),
        'matchup_grade': matchup_grade
    }


def calculate_dfs_score(projection_data, player_salary=None):
    """
    Calculate comprehensive DFS score (0-100) based on 5 components:

    1. Projection Score (30 pts): Projected fantasy points
    2. Value Score (25 pts): Fantasy points per $1K salary
    3. Matchup Score (20 pts): Position + style matchup quality
    4. Volume Score (15 pts): Target share and trend
    5. Consistency Score (10 pts): Projection range tightness

    Args:
        projection_data: Dict from calculate_enhanced_receiving_projection()
        player_salary: DFS salary (optional, defaults to $6000 if not provided)

    Returns dict with:
    - dfs_score: Total DFS score (0-100)
    - score_breakdown: Dict with individual component scores
    - dfs_rating: Text rating (Elite/Excellent/Good/Moderate/Avoid)
    """
    if projection_data is None:
        return None

    # Default salary if not provided
    if player_salary is None:
        player_salary = 6000

    # Component 1: Projection Score (30 pts)
    # Calculate projected fantasy points (PPR scoring)
    projected_fp = (
        projection_data['projected_rec_yards'] * 0.1 +  # 0.1 pts per yard
        projection_data['projected_receptions'] * 1.0 +  # 1 pt per catch (PPR)
        projection_data['projected_tds'] * 6.0  # 6 pts per TD
    )

    # Scale to 30 points (assume 25 FP = max 30 pts)
    projection_score = min(30, (projected_fp / 25) * 30)

    # Component 2: Value Score (25 pts)
    # FP per $1K salary (assume 3.5x = excellent value)
    fp_per_1k = projected_fp / (player_salary / 1000)
    value_score = min(25, (fp_per_1k / 3.5) * 25)

    # Component 3: Matchup Score (20 pts)
    # Based on position + style multipliers
    combined_mult = (projection_data['position_matchup_multiplier'] *
                     projection_data['style_matchup_multiplier'])
    # Scale: 1.4+ = 20 pts, 0.7 = 0 pts
    matchup_score = max(0, min(20, ((combined_mult - 0.7) / 0.7) * 20))

    # Component 4: Volume Score (15 pts)
    # Based on target share (assume 25% = max 15 pts)
    volume_score = min(15, (projection_data['target_share_pct'] / 25) * 15)

    # Component 5: Consistency Score (10 pts)
    # Tighter range = better consistency
    # Range as % of base projection
    proj_range = (projection_data['projected_rec_yards_high'] -
                  projection_data['projected_rec_yards_low'])
    range_pct = proj_range / projection_data['projected_rec_yards'] if projection_data['projected_rec_yards'] > 0 else 1

    # Lower range % = higher consistency (invert scale)
    # 50% range = 10 pts, 100% range = 0 pts
    consistency_score = max(0, min(10, ((1 - range_pct) / 0.5) * 10))

    # Total DFS Score
    dfs_score = (projection_score + value_score + matchup_score +
                 volume_score + consistency_score)

    # Rating
    if dfs_score >= 85:
        rating = "ELITE SMASH SPOT"
    elif dfs_score >= 75:
        rating = "EXCELLENT VALUE"
    elif dfs_score >= 65:
        rating = "GOOD PLAY"
    elif dfs_score >= 50:
        rating = "MODERATE"
    else:
        rating = "AVOID"

    return {
        'dfs_score': round(dfs_score, 1),
        'projected_fantasy_points': round(projected_fp, 1),
        'score_breakdown': {
            'projection': round(projection_score, 1),
            'value': round(value_score, 1),
            'matchup': round(matchup_score, 1),
            'volume': round(volume_score, 1),
            'consistency': round(consistency_score, 1)
        },
        'dfs_rating': rating
    }


# Test function
if __name__ == "__main__":
    print("Testing Enhanced Projections System...")
    print("=" * 60)

    # Test with Trey McBride @ TB
    print("\nTest 1: Trey McBride (ARI) @ TB")
    proj = calculate_enhanced_receiving_projection(
        player_name="Trey McBride",
        team="ARI",
        opponent_team="TB",
        position="TE",
        season=2025
    )

    if proj:
        print(f"Projected Rec Yards: {proj['projected_rec_yards']} (Range: {proj['projected_rec_yards_low']}-{proj['projected_rec_yards_high']})")
        print(f"Projected TDs: {proj['projected_tds']}")
        print(f"Projected Receptions: {proj['projected_receptions']}")
        print(f"Target Share: {proj['target_share_pct']}%")
        print(f"Position Matchup: {proj['position_matchup_multiplier']}x")
        print(f"Style Matchup: {proj['style_matchup_multiplier']}x")
        print(f"Matchup Grade: {proj['matchup_grade']}")

        dfs = calculate_dfs_score(proj, player_salary=6500)
        if dfs:
            print(f"\nDFS Score: {dfs['dfs_score']}/100")
            print(f"Projected Fantasy Points: {dfs['projected_fantasy_points']}")
            print(f"Rating: {dfs['dfs_rating']}")
            print(f"Breakdown: Proj={dfs['score_breakdown']['projection']}, Value={dfs['score_breakdown']['value']}, Matchup={dfs['score_breakdown']['matchup']}, Volume={dfs['score_breakdown']['volume']}, Consistency={dfs['score_breakdown']['consistency']}")
    else:
        print("X No projection data available")

    print("\n" + "=" * 60)
