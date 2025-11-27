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
TEAM_RUSH_YARDS_CAP = 200  # Max team rushing yards per game
TEAM_RUSH_YARDS_FLOOR = 50  # Minimum team rushing yards
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


def calculate_wr_breakout_score(player_name, team, opponent_team, season=2025):
    """
    Calculate WR Breakout Score (0-100) - identifies high-upside dartthrow plays.

    Perfect for finding UNDER-THE-RADAR WRs who could smash their projection based on:
    1. Rising opportunity trend (targets/receptions increasing)
    2. High variance/volatility (boom-or-bust upside)
    3. Favorable matchup style (YAC/Air alignment with defense)
    4. Low ownership/WR3-WR4 types with spike potential

    Components (100 points total):
    - Opportunity Trend (30 pts): Recent target growth vs season average
    - Volatility Score (25 pts): Standard deviation of yardage (high variance = high upside)
    - Matchup Alignment (25 pts): Air/YAC style match + defensive vulnerability
    - Ceiling Potential (20 pts): Red zone upside + big play ability

    Returns dict with:
    - breakout_score: 0-100 (70+ = Elite dartthrow, 50-69 = Good upside, <50 = Low upside)
    - breakout_rating: Text rating
    - trend_indicator: "Rising Star" / "Heating Up" / "Stable" / "Cooling"
    - upside_factors: List of reasons this player could smash
    """
    conn = sqlite3.connect(str(DB_PATH))

    # Get player's game-by-game data for trend and variance analysis
    query = '''
    SELECT
        week,
        targets,
        receptions,
        receiving_yards,
        receiving_tds,
        receiving_air_yards,
        receiving_yards_after_catch
    FROM player_stats
    WHERE player_display_name = ?
      AND team = ?
      AND season = ?
      AND season_type = 'REG'
    ORDER BY week DESC
    LIMIT 12
    '''

    df = pd.read_sql_query(query, conn, params=(player_name, team, season))
    conn.close()

    if df.empty or len(df) < 3:
        return None  # Need at least 3 games for trend analysis

    # COMPONENT 1: Opportunity Trend Score (30 points)
    # Compare recent 4 games vs earlier games
    recent_games = df.head(4)
    earlier_games = df.tail(len(df) - 4) if len(df) > 4 else df

    recent_targets_avg = recent_games['targets'].mean()
    earlier_targets_avg = earlier_games['targets'].mean() if len(earlier_games) > 0 else recent_targets_avg

    # Calculate trend percentage
    if earlier_targets_avg > 0:
        trend_pct = ((recent_targets_avg - earlier_targets_avg) / earlier_targets_avg) * 100
    else:
        trend_pct = 0

    # Score the trend (positive trend = points)
    if trend_pct >= 50:  # Targets up 50%+
        opportunity_score = 30
        trend_indicator = "🔥 Rising Star"
    elif trend_pct >= 25:  # Targets up 25%+
        opportunity_score = 24
        trend_indicator = "📈 Heating Up"
    elif trend_pct >= 10:  # Targets up 10%+
        opportunity_score = 18
        trend_indicator = "↗️ Trending Up"
    elif trend_pct >= -10:  # Stable targets
        opportunity_score = 12
        trend_indicator = "➡️ Stable"
    else:  # Declining targets
        opportunity_score = 5
        trend_indicator = "↘️ Cooling"

    # COMPONENT 2: Volatility Score (25 points)
    # High variance = boom-or-bust upside (good for DFS dartthrows)
    yards_std = df['receiving_yards'].std()
    yards_mean = df['receiving_yards'].mean()

    # Coefficient of variation (CV) = std / mean
    cv = (yards_std / yards_mean) if yards_mean > 5 else 0

    # High CV (>0.8) = volatile boom-or-bust player
    # Moderate CV (0.5-0.8) = some variance
    # Low CV (<0.5) = consistent (bad for dartthrow upside)
    if cv >= 0.9:
        volatility_score = 25  # Ultra-volatile (massive ceiling)
        volatility_label = "Explosive Upside"
    elif cv >= 0.7:
        volatility_score = 20  # High volatility
        volatility_label = "High Ceiling"
    elif cv >= 0.5:
        volatility_score = 14  # Moderate volatility
        volatility_label = "Moderate Variance"
    else:
        volatility_score = 7  # Low volatility (consistent = low ceiling)
        volatility_label = "Low Variance"

    # COMPONENT 3: Matchup Alignment Score (25 points)
    # Air/YAC alignment + defensive vulnerability
    offense_df = get_offensive_team_stats(season)
    defense_df = get_defensive_team_stats(season)

    off_stats = offense_df[offense_df['team'] == team]
    def_stats = defense_df[defense_df['defense'] == opponent_team]

    matchup_score = 12  # Default neutral
    matchup_factors = []

    if not off_stats.empty and not def_stats.empty:
        # Get style multiplier
        style_mult = calculate_style_matchup_multiplier(off_stats, def_stats)

        # Convert multiplier to score (1.0 = 12 pts, 1.2 = 25 pts, 0.8 = 5 pts)
        matchup_score = int(((style_mult - 0.8) / 0.4) * 20 + 5)
        matchup_score = max(5, min(25, matchup_score))

        # Check for specific matchup advantages
        def_air_pct = def_stats.iloc[0]['air_allowed_pct']
        def_yac_pct = def_stats.iloc[0]['yac_allowed_pct']
        off_air_pct = off_stats.iloc[0]['air_pct']

        if def_air_pct > 55 and off_air_pct > 55:
            matchup_factors.append("Deep Ball Matchup")
        if def_yac_pct > 50:
            matchup_factors.append("YAC Exploitable")

        # Check defensive vulnerability to WRs
        def_wr_yards = def_stats.iloc[0]['wr_yards_allowed_per_game']
        if def_wr_yards > 250:
            matchup_factors.append("WR-Soft Defense")
            matchup_score = min(25, matchup_score + 3)

    # COMPONENT 4: Ceiling Potential (20 points)
    # Big play ability + red zone upside
    ceiling_score = 0
    ceiling_factors = []

    # Check for big-play games (100+ yards or 2+ TDs)
    big_games = df[(df['receiving_yards'] >= 100) | (df['receiving_tds'] >= 2)]
    if len(big_games) > 0:
        ceiling_score += 8
        ceiling_factors.append(f"{len(big_games)} Big Games")

    # Air yards percentage (deep threat indicator)
    if df['receiving_air_yards'].sum() > 0:
        air_pct = (df['receiving_air_yards'].sum() / df['receiving_yards'].sum()) * 100
        if air_pct > 60:
            ceiling_score += 7
            ceiling_factors.append("Deep Threat")
        elif air_pct > 45:
            ceiling_score += 4

    # TD upside
    td_games = df[df['receiving_tds'] > 0]
    if len(td_games) >= 2:
        ceiling_score += 5
        ceiling_factors.append("TD Upside")

    # Total Breakout Score
    breakout_score = opportunity_score + volatility_score + matchup_score + ceiling_score

    # Rating
    if breakout_score >= 75:
        rating = "🚀 ELITE DARTTHROW"
    elif breakout_score >= 65:
        rating = "💎 STRONG UPSIDE"
    elif breakout_score >= 50:
        rating = "📊 GOOD UPSIDE"
    elif breakout_score >= 35:
        rating = "⚖️ MODERATE UPSIDE"
    else:
        rating = "😴 LOW UPSIDE"

    # Compile upside factors
    upside_factors = []
    if trend_pct >= 25:
        upside_factors.append(f"Targets +{trend_pct:.0f}% (trending up)")
    if cv >= 0.7:
        upside_factors.append(f"{volatility_label} (CV: {cv:.2f})")
    upside_factors.extend(matchup_factors)
    upside_factors.extend(ceiling_factors)

    return {
        'breakout_score': round(breakout_score, 1),
        'breakout_rating': rating,
        'trend_indicator': trend_indicator,
        'opportunity_trend_pct': round(trend_pct, 1),
        'volatility_cv': round(cv, 2),
        'recent_targets_avg': round(recent_targets_avg, 1),
        'upside_factors': upside_factors,
        'score_breakdown': {
            'opportunity': opportunity_score,
            'volatility': volatility_score,
            'matchup': matchup_score,
            'ceiling': ceiling_score
        }
    }


# ============================================================================
# RB Enhanced Projections Functions
# ============================================================================

def get_rb_rushing_team_stats(season=2025):
    """
    Get offensive team rushing statistics.

    Returns DataFrame with columns:
    - team
    - games
    - total_rush_yards
    - avg_rush_yards_per_game
    - total_carries
    - avg_carries_per_game
    """
    conn = sqlite3.connect(str(DB_PATH))

    query = '''
    SELECT
        team,
        COUNT(DISTINCT week) as games,
        SUM(rushing_yards) as total_rush_yards,
        SUM(carries) as total_carries
    FROM player_stats
    WHERE season = ?
      AND season_type = 'REG'
      AND rushing_yards > 0
    GROUP BY team
    '''

    df = pd.read_sql_query(query, conn, params=(season,))
    conn.close()

    # Calculate per-game averages
    df['avg_rush_yards_per_game'] = (df['total_rush_yards'] / df['games']).round(1)
    df['avg_carries_per_game'] = (df['total_carries'] / df['games']).round(1)

    return df


def get_rb_defensive_rush_stats(season=2025):
    """
    Get defensive team rushing statistics allowed.

    Returns DataFrame with columns:
    - defense (team)
    - games
    - total_rush_allowed
    - avg_rush_allowed_per_game
    - rush_tds_allowed_per_game
    - avg_rb_receptions_allowed (for dual-threat RBs)
    - avg_rb_rec_yards_allowed
    - rec_tds_allowed_per_game
    """
    conn = sqlite3.connect(str(DB_PATH))

    query = '''
    SELECT
        opponent_team as defense,
        COUNT(DISTINCT week) as games,
        SUM(rushing_yards) as total_rush_allowed,
        SUM(rushing_tds) as rush_tds_allowed,
        SUM(receptions) as total_rb_receptions,
        SUM(receiving_yards) as total_rb_rec_yards,
        SUM(receiving_tds) as rec_tds_allowed
    FROM player_stats
    WHERE season = ?
      AND season_type = 'REG'
      AND position = 'RB'
    GROUP BY opponent_team
    '''

    df = pd.read_sql_query(query, conn, params=(season,))
    conn.close()

    # Calculate per-game averages
    df['avg_rush_allowed_per_game'] = (df['total_rush_allowed'] / df['games']).round(1)
    df['rush_tds_allowed_per_game'] = (df['rush_tds_allowed'] / df['games']).round(2)
    df['avg_rb_receptions_allowed'] = (df['total_rb_receptions'] / df['games']).round(1)
    df['avg_rb_rec_yards_allowed'] = (df['total_rb_rec_yards'] / df['games']).round(1)
    df['rec_tds_allowed_per_game'] = (df['rec_tds_allowed'] / df['games']).round(2)

    return df


def get_rb_carry_share(player_name, team, season=2025, recent_only=False):
    """
    Get RB's carry share and receiving role (season average or recent games).

    Args:
        player_name: Player display name
        team: Player's team
        season: Season year
        recent_only: If True, only use last N games

    Returns dict with:
    - carry_share: Estimated percentage of team carries (0-100)
    - avg_rush_yards: Average rushing yards per game
    - avg_carries: Average carries per game
    - avg_rec_yards: Average receiving yards per game
    - avg_receptions: Average receptions per game
    - avg_total_tds: Average total TDs per game (rush + rec)
    - games: Number of games
    """
    conn = sqlite3.connect(str(DB_PATH))

    if recent_only:
        query = '''
        SELECT
            week,
            carries,
            rushing_yards,
            rushing_tds,
            receptions,
            receiving_yards,
            receiving_tds
        FROM player_stats
        WHERE player_display_name = ?
          AND team = ?
          AND season = ?
          AND season_type = 'REG'
          AND (carries > 0 OR receptions > 0)
        ORDER BY week DESC
        LIMIT ?
        '''
        df = pd.read_sql_query(query, conn, params=(player_name, team, season, RECENT_GAMES_WINDOW))
    else:
        query = '''
        SELECT
            week,
            carries,
            rushing_yards,
            rushing_tds,
            receptions,
            receiving_yards,
            receiving_tds
        FROM player_stats
        WHERE player_display_name = ?
          AND team = ?
          AND season = ?
          AND season_type = 'REG'
          AND (carries > 0 OR receptions > 0)
        ORDER BY week DESC
        '''
        df = pd.read_sql_query(query, conn, params=(player_name, team, season))

    conn.close()

    if df.empty:
        return None

    # Estimate carry share (typical team has ~28 carries per game)
    typical_team_carries = 28
    avg_carries = df['carries'].mean()
    carry_share = (avg_carries / typical_team_carries * 100) if typical_team_carries > 0 else 0

    return {
        'carry_share': round(carry_share, 1),
        'avg_rush_yards': df['rushing_yards'].mean().round(1),
        'avg_carries': df['carries'].mean().round(1),
        'avg_rec_yards': df['receiving_yards'].mean().round(1),
        'avg_receptions': df['receptions'].mean().round(1),
        'avg_total_tds': (df['rushing_tds'] + df['receiving_tds']).mean().round(2),
        'games': len(df)
    }


def calculate_rb_rushing_td_projection(player_name, team, opponent_team, season=2025):
    """
    Calculate rushing TD projection for RB.

    Returns projected rushing TDs per game (capped at TD_CAP_PER_GAME).
    """
    conn = sqlite3.connect(str(DB_PATH))

    # Get player's recent TD performance
    query_recent = '''
    SELECT
        AVG(rushing_tds) as avg_tds_recent
    FROM (
        SELECT rushing_tds
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
        SUM(rushing_tds) / COUNT(DISTINCT week) as avg_tds_season
    FROM player_stats
    WHERE player_display_name = ?
      AND team = ?
      AND season = ?
      AND season_type = 'REG'
    '''
    df_season = pd.read_sql_query(query_season, conn, params=(player_name, team, season))

    # Get opponent's rush TD vulnerability
    query_defense = '''
    SELECT
        SUM(rushing_tds) / COUNT(DISTINCT week) as rush_tds_allowed_per_game
    FROM player_stats
    WHERE opponent_team = ?
      AND position = 'RB'
      AND season = ?
      AND season_type = 'REG'
    '''
    df_defense = pd.read_sql_query(query_defense, conn, params=(opponent_team, season))

    conn.close()

    # Calculate weighted TD projection
    avg_tds_recent = df_recent.iloc[0]['avg_tds_recent'] if not df_recent.empty and df_recent.iloc[0]['avg_tds_recent'] else 0
    avg_tds_season = df_season.iloc[0]['avg_tds_season'] if not df_season.empty and df_season.iloc[0]['avg_tds_season'] else 0
    tds_allowed = df_defense.iloc[0]['rush_tds_allowed_per_game'] if not df_defense.empty and df_defense.iloc[0]['rush_tds_allowed_per_game'] else 0

    # Weighted average: 70% recent, 30% season
    player_td_rate = (avg_tds_recent * RECENCY_WEIGHT) + (avg_tds_season * (1 - RECENCY_WEIGHT))

    # Blend player rate (65%) with defensive vulnerability (35%)
    projected_rush_tds = (player_td_rate * 0.65) + (tds_allowed * 0.35)

    # Apply cap
    projected_rush_tds = min(TD_CAP_PER_GAME, projected_rush_tds)

    return round(projected_rush_tds, 2)


def calculate_enhanced_rb_projection(
    player_name,
    team,
    opponent_team,
    season=2025
):
    """
    Calculate comprehensive enhanced projection for RB.

    Returns dict with:
    - projected_rush_yards: Enhanced rushing yards projection
    - projected_rush_yards_low: Low confidence bound (75%)
    - projected_rush_yards_high: High confidence bound (125%)
    - projected_rush_tds: Rushing TD projection
    - projected_rec_tds: Receiving TD projection (using existing function)
    - projected_carries: Carries projection
    - projected_receptions: Receptions projection
    - carry_share_pct: Recency-weighted carry share
    - matchup_grade: Overall matchup grade (A+ to D)
    """
    # Get team stats
    rush_offense_df = get_rb_rushing_team_stats(season)
    rush_defense_df = get_rb_defensive_rush_stats(season)

    # Get team-level projection
    off_stats = rush_offense_df[rush_offense_df['team'] == team]
    def_stats = rush_defense_df[rush_defense_df['defense'] == opponent_team]

    if off_stats.empty or def_stats.empty:
        return None

    off_avg = off_stats.iloc[0]['avg_rush_yards_per_game']
    def_avg = def_stats.iloc[0]['avg_rush_allowed_per_game']

    # 60% offense capability, 40% defensive vulnerability
    team_rush_proj = (off_avg * 0.6) + (def_avg * 0.4)

    # Apply caps
    team_rush_proj = max(TEAM_RUSH_YARDS_FLOOR, min(TEAM_RUSH_YARDS_CAP, team_rush_proj))

    # Get player carry shares (recency-weighted)
    recent_share_data = get_rb_carry_share(player_name, team, season, recent_only=True)
    season_share_data = get_rb_carry_share(player_name, team, season, recent_only=False)

    if not recent_share_data or not season_share_data:
        return None

    # Weighted carry share: 70% recent, 30% season
    weighted_carry_share = (
        (recent_share_data['carry_share'] * RECENCY_WEIGHT) +
        (season_share_data['carry_share'] * (1 - RECENCY_WEIGHT))
    )

    # Calculate player rushing yards projection
    base_rush_proj = team_rush_proj * (weighted_carry_share / 100)

    # Confidence ranges
    proj_low = base_rush_proj * 0.75
    proj_high = base_rush_proj * 1.25

    # Calculate TD projections
    projected_rush_tds = calculate_rb_rushing_td_projection(player_name, team, opponent_team, season)

    # Receiving TDs for dual-threat RBs
    projected_rec_tds = calculate_td_projection(player_name, team, opponent_team, 'RB', season)

    # Calculate carries and receptions projections
    avg_carries = (
        (recent_share_data['avg_carries'] * RECENCY_WEIGHT) +
        (season_share_data['avg_carries'] * (1 - RECENCY_WEIGHT))
    )

    avg_receptions = (
        (recent_share_data['avg_receptions'] * RECENCY_WEIGHT) +
        (season_share_data['avg_receptions'] * (1 - RECENCY_WEIGHT))
    )

    # Comprehensive matchup grade based on TOTAL fantasy production allowed
    # Includes: rush yards, rush TDs, receiving yards, receiving TDs, receptions
    # This evaluates the COMPLETE RB matchup, not just rushing defense

    # Get defensive stats for RBs (all production types)
    rb_rec_allowed = def_stats.iloc[0]['avg_rb_receptions_allowed']
    rb_rec_yards_allowed = def_stats.iloc[0]['avg_rb_rec_yards_allowed']
    rush_tds_allowed = def_stats.iloc[0]['rush_tds_allowed_per_game']
    rec_tds_allowed = def_stats.iloc[0]['rec_tds_allowed_per_game']

    # Calculate total fantasy points allowed to RBs per game (PPR scoring)
    # Rush: 0.1 pts/yard + 6 pts/TD
    # Receiving: 0.1 pts/yard + 1 pt/rec + 6 pts/TD
    def_fp_allowed = (
        def_avg * 0.1 +                    # Rush yards
        rush_tds_allowed * 6.0 +           # Rush TDs
        rb_rec_yards_allowed * 0.1 +       # Receiving yards
        rb_rec_allowed * 1.0 +             # Receptions (PPR)
        rec_tds_allowed * 6.0              # Receiving TDs
    )

    # League average RB fantasy points allowed ~15-18 per game
    league_avg_rb_fp_allowed = 16.5

    # Grade based on total fantasy production allowed (higher = better matchup)
    if def_fp_allowed >= league_avg_rb_fp_allowed * 1.15:
        matchup_grade = "A"      # Allows 19+ FP/game (soft defense)
    elif def_fp_allowed >= league_avg_rb_fp_allowed * 1.05:
        matchup_grade = "B+"     # Allows 17.3+ FP/game
    elif def_fp_allowed >= league_avg_rb_fp_allowed * 0.95:
        matchup_grade = "B"      # Allows 15.7+ FP/game (average)
    elif def_fp_allowed >= league_avg_rb_fp_allowed * 0.85:
        matchup_grade = "C"      # Allows 14+ FP/game (tough)
    else:
        matchup_grade = "D"      # Allows <14 FP/game (elite defense)

    return {
        'projected_rush_yards': round(base_rush_proj, 1),
        'projected_rush_yards_low': round(proj_low, 1),
        'projected_rush_yards_high': round(proj_high, 1),
        'projected_rush_tds': projected_rush_tds,
        'projected_rec_tds': projected_rec_tds,
        'projected_carries': round(avg_carries, 1),
        'projected_receptions': round(avg_receptions, 1),
        'carry_share_pct': round(weighted_carry_share, 1),
        'matchup_grade': matchup_grade
    }


def calculate_rb_dfs_score(projection_data, player_salary=None):
    """
    Calculate comprehensive DFS score (0-100) for RBs based on 5 components:

    1. Projection Score (30 pts): Projected fantasy points
    2. Value Score (25 pts): Fantasy points per $1K salary
    3. Matchup Score (20 pts): Rush defense matchup quality
    4. Volume Score (15 pts): Carry share and receiving role
    5. Consistency Score (10 pts): Projection range tightness

    Args:
        projection_data: Dict from calculate_enhanced_rb_projection()
        player_salary: DFS salary (optional, defaults to $6500 if not provided)

    Returns dict with:
    - dfs_score: Total DFS score (0-100)
    - score_breakdown: Dict with individual component scores
    - dfs_rating: Text rating (Elite/Excellent/Good/Moderate/Avoid)
    """
    if projection_data is None:
        return None

    # Default salary if not provided
    if player_salary is None:
        player_salary = 6500

    # Component 1: Projection Score (30 pts)
    # Calculate projected fantasy points (PPR scoring)
    # Rushing: 0.1 pts/yard + 6 pts/TD
    # Receiving: 0.1 pts/yard + 1 pt/catch + 6 pts/TD
    projected_fp = (
        projection_data['projected_rush_yards'] * 0.1 +
        projection_data['projected_rush_tds'] * 6.0 +
        projection_data.get('projected_rec_yards', projection_data['projected_receptions'] * 8) * 0.1 +  # Estimate rec yards
        projection_data['projected_receptions'] * 1.0 +
        projection_data['projected_rec_tds'] * 6.0
    )

    # Scale to 30 points (assume 20 FP = max 30 pts for RBs)
    projection_score = min(30, (projected_fp / 20) * 30)

    # Component 2: Value Score (25 pts)
    fp_per_1k = projected_fp / (player_salary / 1000)
    value_score = min(25, (fp_per_1k / 3.0) * 25)

    # Component 3: Matchup Score (20 pts)
    grade_to_score = {'A+': 20, 'A': 18, 'B+': 15, 'B': 12, 'C': 8, 'D': 4, 'F': 0}
    matchup_score = grade_to_score.get(projection_data['matchup_grade'], 10)

    # Component 4: Volume Score (15 pts)
    # Based on carry share (assume 35% = max 15 pts)
    volume_score = min(15, (projection_data['carry_share_pct'] / 35) * 15)

    # Component 5: Consistency Score (10 pts)
    proj_range = (projection_data['projected_rush_yards_high'] -
                  projection_data['projected_rush_yards_low'])
    range_pct = proj_range / projection_data['projected_rush_yards'] if projection_data['projected_rush_yards'] > 0 else 1

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
