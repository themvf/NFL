# -*- coding: utf-8 -*-
"""
Historical Trends & Betting Analytics Module
Analyzes historical NFL game outcomes for patterns and betting insights.
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Database configuration
DB_PATH = Path(__file__).parent / "data" / "nfl_merged.db"


# ============================================================================
# Data Analysis Functions
# ============================================================================

def get_completed_games(seasons: List[int]) -> pd.DataFrame:
    """
    Get all completed games from schedules table.

    Args:
        seasons: List of seasons to include

    Returns:
        DataFrame with completed game data
    """
    conn = sqlite3.connect(DB_PATH)

    seasons_str = ','.join(map(str, seasons))

    query = f"""
        SELECT
            season,
            week,
            weekday,
            home_team,
            away_team,
            home_score,
            away_score,
            spread_line,
            total_line,
            div_game
        FROM schedules
        WHERE game_type = 'REG'
        AND season IN ({seasons_str})
        AND home_score IS NOT NULL
        AND away_score IS NOT NULL
        AND spread_line IS NOT NULL
        ORDER BY season, week
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def calculate_home_win_rate(seasons: List[int]) -> Dict:
    """
    Calculate home team win rate.

    Returns:
        Dict with overall rate and yearly breakdown
    """
    games = get_completed_games(seasons)

    if games.empty:
        return {
            'overall_rate': 0,
            'total_games': 0,
            'home_wins': 0,
            'by_season': pd.DataFrame()
        }

    # Overall rate
    games['home_won'] = games['home_score'] > games['away_score']
    overall_rate = games['home_won'].mean() * 100

    # By season
    by_season = games.groupby('season').agg({
        'home_won': ['sum', 'count', 'mean']
    }).reset_index()
    by_season.columns = ['season', 'home_wins', 'total_games', 'win_rate']
    by_season['win_rate'] = by_season['win_rate'] * 100

    return {
        'overall_rate': overall_rate,
        'total_games': len(games),
        'home_wins': games['home_won'].sum(),
        'by_season': by_season
    }


def calculate_favorite_win_rate(seasons: List[int], min_spread: Optional[float] = None,
                                max_spread: Optional[float] = None) -> Dict:
    """
    Calculate favorite win rate.
    Negative spread = home favorite, positive = away favorite.

    Args:
        seasons: List of seasons
        min_spread: Minimum absolute spread value (e.g., 3.0 for 3+ point favorites)
        max_spread: Maximum absolute spread value

    Returns:
        Dict with favorite performance metrics
    """
    games = get_completed_games(seasons)

    if games.empty:
        return {
            'overall_rate': 0,
            'total_games': 0,
            'favorite_wins': 0,
            'by_spread_range': pd.DataFrame()
        }

    # Determine favorite and whether they won
    # NFLverse convention: POSITIVE spread = home favorite, NEGATIVE = away favorite
    games['favorite_is_home'] = games['spread_line'] > 0
    games['favorite_spread'] = games['spread_line'].abs()

    # Apply spread filters
    if min_spread is not None:
        games = games[games['favorite_spread'] >= min_spread]
    if max_spread is not None:
        games = games[games['favorite_spread'] <= max_spread]

    # Did favorite win?
    games['favorite_won'] = (
        ((games['favorite_is_home']) & (games['home_score'] > games['away_score'])) |
        ((~games['favorite_is_home']) & (games['away_score'] > games['home_score']))
    )

    # Overall rate
    overall_rate = games['favorite_won'].mean() * 100 if not games.empty else 0

    # By spread range
    games['spread_range'] = pd.cut(
        games['favorite_spread'],
        bins=[0, 3, 7, 10, 20],
        labels=['0-3', '3-7', '7-10', '10+'],
        include_lowest=True
    )

    by_spread = games.groupby('spread_range', observed=True).agg({
        'favorite_won': ['sum', 'count', 'mean']
    }).reset_index()
    by_spread.columns = ['spread_range', 'wins', 'total_games', 'win_rate']
    by_spread['win_rate'] = by_spread['win_rate'] * 100

    return {
        'overall_rate': overall_rate,
        'total_games': len(games),
        'favorite_wins': games['favorite_won'].sum(),
        'by_spread_range': by_spread
    }


def calculate_home_favorite_split(seasons: List[int]) -> Dict:
    """
    Calculate win rates for home favorites vs home underdogs.

    Returns:
        Dict with split metrics
    """
    games = get_completed_games(seasons)

    if games.empty:
        return {
            'home_fav_rate': 0,
            'home_dog_rate': 0,
            'home_fav_games': 0,
            'home_dog_games': 0
        }

    games['home_won'] = games['home_score'] > games['away_score']
    # NFLverse convention: POSITIVE spread = home favorite, NEGATIVE = away favorite
    games['home_is_favorite'] = games['spread_line'] > 0

    # Home favorites
    home_favs = games[games['home_is_favorite']]
    home_fav_rate = home_favs['home_won'].mean() * 100 if not home_favs.empty else 0

    # Home underdogs
    home_dogs = games[~games['home_is_favorite']]
    home_dog_rate = home_dogs['home_won'].mean() * 100 if not home_dogs.empty else 0

    return {
        'home_fav_rate': home_fav_rate,
        'home_dog_rate': home_dog_rate,
        'home_fav_games': len(home_favs),
        'home_dog_games': len(home_dogs)
    }


def calculate_day_of_week_metrics(seasons: List[int]) -> pd.DataFrame:
    """
    Calculate favorite win rate by day of week.

    Returns:
        DataFrame with metrics by weekday
    """
    games = get_completed_games(seasons)

    if games.empty:
        return pd.DataFrame()

    # Determine favorite and whether they won
    # NFLverse convention: POSITIVE spread = home favorite, NEGATIVE = away favorite
    games['favorite_is_home'] = games['spread_line'] > 0
    games['favorite_won'] = (
        ((games['favorite_is_home']) & (games['home_score'] > games['away_score'])) |
        ((~games['favorite_is_home']) & (games['away_score'] > games['home_score']))
    )

    # Group by weekday
    by_day = games.groupby('weekday').agg({
        'favorite_won': ['sum', 'count', 'mean'],
        'home_score': 'mean',
        'away_score': 'mean'
    }).reset_index()

    by_day.columns = ['weekday', 'fav_wins', 'total_games', 'fav_win_rate', 'avg_home_score', 'avg_away_score']
    by_day['fav_win_rate'] = by_day['fav_win_rate'] * 100

    # Sort by common day order
    day_order = ['Thursday', 'Sunday', 'Monday', 'Saturday', 'Friday', 'Tuesday', 'Wednesday']
    by_day['day_order'] = by_day['weekday'].map({day: i for i, day in enumerate(day_order)})
    by_day = by_day.sort_values('day_order').drop('day_order', axis=1)

    return by_day


def calculate_over_under_rate(seasons: List[int]) -> Dict:
    """
    Calculate over/under hit rates.

    Returns:
        Dict with over/under metrics
    """
    games = get_completed_games(seasons)

    if games.empty:
        return {
            'over_rate': 0,
            'under_rate': 0,
            'push_rate': 0,
            'total_games': 0,
            'by_season': pd.DataFrame()
        }

    games['total_points'] = games['home_score'] + games['away_score']
    games['result'] = games.apply(
        lambda row: 'Over' if row['total_points'] > row['total_line']
        else ('Under' if row['total_points'] < row['total_line'] else 'Push'),
        axis=1
    )

    # Overall rates
    over_rate = (games['result'] == 'Over').mean() * 100
    under_rate = (games['result'] == 'Under').mean() * 100
    push_rate = (games['result'] == 'Push').mean() * 100

    # By season
    by_season = games.groupby('season')['result'].value_counts(normalize=True).unstack(fill_value=0) * 100
    by_season = by_season.reset_index()

    return {
        'over_rate': over_rate,
        'under_rate': under_rate,
        'push_rate': push_rate,
        'total_games': len(games),
        'by_season': by_season
    }


def calculate_score_margin_distribution(seasons: List[int]) -> pd.DataFrame:
    """
    Calculate distribution of final score margins.

    Returns:
        DataFrame with margin categories and counts
    """
    games = get_completed_games(seasons)

    if games.empty:
        return pd.DataFrame()

    games['margin'] = (games['home_score'] - games['away_score']).abs()

    # Create margin bins
    games['margin_category'] = pd.cut(
        games['margin'],
        bins=[0, 1, 3, 6, 10, 14, 21, 100],
        labels=['1 pt', '2-3 pts (FG)', '4-6 pts', '7-10 pts', '11-14 pts (2 TD)', '15-21 pts', '22+ pts'],
        include_lowest=True
    )

    distribution = games['margin_category'].value_counts().reset_index()
    distribution.columns = ['margin_category', 'count']
    distribution['percentage'] = (distribution['count'] / len(games) * 100).round(1)
    distribution = distribution.sort_values('margin_category')

    return distribution


def calculate_underdog_blowout_wins(seasons: List[int], min_margin: float = 7.5) -> Dict:
    """
    Calculate rate of underdogs winning by 7.5+ points (significant upsets).

    Args:
        seasons: List of seasons
        min_margin: Minimum margin of victory (default 7.5 points)

    Returns:
        Dict with underdog blowout win stats
    """
    games = get_completed_games(seasons)

    if games.empty:
        return {
            'overall_rate': 0,
            'total_underdog_games': 0,
            'underdog_blowout_wins': 0,
            'by_season': pd.DataFrame(),
            'examples': pd.DataFrame()
        }

    # Determine underdog and calculate margin
    # NFLverse convention: POSITIVE spread = home favorite, NEGATIVE = away favorite
    games['underdog_is_home'] = games['spread_line'] < 0
    games['underdog_won'] = (
        ((games['underdog_is_home']) & (games['home_score'] > games['away_score'])) |
        ((~games['underdog_is_home']) & (games['away_score'] > games['home_score']))
    )

    # Calculate margin of victory for underdog wins
    games['underdog_margin'] = games.apply(
        lambda row: (row['home_score'] - row['away_score']) if row['underdog_is_home']
        else (row['away_score'] - row['home_score']),
        axis=1
    )

    # Filter to underdog blowout wins (7.5+ point margin)
    games['underdog_blowout_win'] = (games['underdog_won']) & (games['underdog_margin'] >= min_margin)

    # Calculate overall rate
    overall_rate = games['underdog_blowout_win'].mean() * 100
    total_underdog_games = len(games)
    underdog_blowout_wins = games['underdog_blowout_win'].sum()

    # By season
    by_season = games.groupby('season').agg({
        'underdog_blowout_win': ['sum', 'count', 'mean']
    }).reset_index()
    by_season.columns = ['season', 'upset_wins', 'total_games', 'upset_rate']
    by_season['upset_rate'] = by_season['upset_rate'] * 100

    # Get examples of recent blowout upsets
    examples = games[games['underdog_blowout_win']].copy()
    examples['underdog_team'] = examples.apply(
        lambda row: row['home_team'] if row['underdog_is_home'] else row['away_team'],
        axis=1
    )
    examples['favorite_team'] = examples.apply(
        lambda row: row['away_team'] if row['underdog_is_home'] else row['home_team'],
        axis=1
    )
    examples['score'] = examples.apply(
        lambda row: f"{row['home_team']} {int(row['home_score'])} - {int(row['away_score'])} {row['away_team']}",
        axis=1
    )
    examples = examples[['season', 'week', 'underdog_team', 'favorite_team', 'underdog_margin', 'score', 'spread_line']].sort_values(['season', 'week'], ascending=[False, False]).head(20)

    return {
        'overall_rate': overall_rate,
        'total_underdog_games': total_underdog_games,
        'underdog_blowout_wins': underdog_blowout_wins,
        'by_season': by_season,
        'examples': examples
    }


def get_most_common_scores(seasons: List[int], limit: int = 20) -> pd.DataFrame:
    """
    Get most common final score combinations.

    Args:
        seasons: List of seasons
        limit: Number of top scores to return

    Returns:
        DataFrame of score combinations
    """
    games = get_completed_games(seasons)

    if games.empty:
        return pd.DataFrame()

    # Create sorted score combinations (higher score first for consistency)
    games['score_combo'] = games.apply(
        lambda row: f"{max(row['home_score'], row['away_score'])}-{min(row['home_score'], row['away_score'])}",
        axis=1
    )

    score_counts = games['score_combo'].value_counts().head(limit).reset_index()
    score_counts.columns = ['score', 'frequency']
    score_counts['percentage'] = (score_counts['frequency'] / len(games) * 100).round(2)

    return score_counts


# ============================================================================
# Visualization Functions
# ============================================================================

def render_home_field_analysis(seasons: List[int]):
    """Render home field advantage analysis."""
    st.subheader("🏠 Home Field Advantage Analysis")

    metrics = calculate_home_win_rate(seasons)

    if metrics['total_games'] == 0:
        st.info("No completed games found for selected seasons.")
        return

    # Big number display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Home Win Rate",
            f"{metrics['overall_rate']:.1f}%",
            help="Percentage of games won by the home team"
        )
    with col2:
        st.metric("Home Wins", f"{metrics['home_wins']:,}")
    with col3:
        st.metric("Total Games", f"{metrics['total_games']:,}")

    # Trend by season
    if not metrics['by_season'].empty:
        st.markdown("#### Home Win Rate by Season")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=metrics['by_season']['season'],
            y=metrics['by_season']['win_rate'],
            mode='lines+markers',
            name='Home Win Rate',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10)
        ))

        # Add league average line
        fig.add_hline(
            y=metrics['overall_rate'],
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Overall Avg: {metrics['overall_rate']:.1f}%",
            annotation_position="right"
        )

        fig.update_layout(
            xaxis_title="Season",
            yaxis_title="Home Win Rate (%)",
            height=400,
            hovermode='x unified',
            yaxis=dict(range=[45, 65])  # Typical range for home field advantage
        )

        st.plotly_chart(fig, use_container_width=True)

        # Data table
        with st.expander("📊 View Season-by-Season Data"):
            display_df = metrics['by_season'].copy()
            display_df.columns = ['Season', 'Home Wins', 'Total Games', 'Win Rate (%)']
            display_df['Win Rate (%)'] = display_df['Win Rate (%)'].round(1)
            st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_favorite_analysis(seasons: List[int]):
    """Render favorite performance analysis."""
    st.subheader("⭐ Favorite Performance Analysis")

    metrics = calculate_favorite_win_rate(seasons)

    if metrics['total_games'] == 0:
        st.info("No completed games found for selected seasons.")
        return

    # Overall metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Favorite Win Rate",
            f"{metrics['overall_rate']:.1f}%",
            help="Percentage of games won by the favorite (team with negative spread)"
        )
    with col2:
        st.metric("Favorite Wins", f"{metrics['favorite_wins']:,}")
    with col3:
        st.metric("Total Games", f"{metrics['total_games']:,}")

    # Win rate by spread size
    if not metrics['by_spread_range'].empty:
        st.markdown("#### Favorite Win Rate by Spread Size")

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=metrics['by_spread_range']['spread_range'].astype(str),
            y=metrics['by_spread_range']['win_rate'],
            text=metrics['by_spread_range']['win_rate'].round(1).astype(str) + '%',
            textposition='outside',
            marker_color='#2ca02c',
            hovertemplate='<b>Spread Range: %{x}</b><br>Win Rate: %{y:.1f}%<br>Games: %{customdata}<extra></extra>',
            customdata=metrics['by_spread_range']['total_games']
        ))

        fig.update_layout(
            xaxis_title="Spread Range (Points)",
            yaxis_title="Favorite Win Rate (%)",
            height=400,
            yaxis=dict(range=[0, 100])
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption("💡 **Insight:** Larger favorites (10+ point spreads) typically win at very high rates, while small favorites (0-3 points) are less reliable.")


def render_home_fav_underdog_split(seasons: List[int]):
    """Render home favorite vs home underdog split."""
    st.markdown("#### Home Favorites vs Home Underdogs")

    split = calculate_home_favorite_split(seasons)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Home Favorites Win Rate",
            f"{split['home_fav_rate']:.1f}%",
            help=f"Based on {split['home_fav_games']} games"
        )
        st.caption(f"📊 {split['home_fav_games']:,} games")

    with col2:
        st.metric(
            "Home Underdogs Win Rate",
            f"{split['home_dog_rate']:.1f}%",
            help=f"Based on {split['home_dog_games']} games"
        )
        st.caption(f"📊 {split['home_dog_games']:,} games")

    st.caption("💡 **Insight:** Home favorites benefit from both being favored AND having home field advantage, leading to high win rates. Home underdogs must overcome both disadvantages.")


def render_underdog_blowout_wins(seasons: List[int]):
    """Render underdog blowout wins (7.5+ point upsets)."""
    st.markdown("#### 💥 Underdog Blowout Wins (7.5+ Points)")

    metrics = calculate_underdog_blowout_wins(seasons)

    # Key Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Upset Rate",
            f"{metrics['overall_rate']:.1f}%",
            help="% of all games where underdog won by 7.5+ points"
        )

    with col2:
        st.metric(
            "Total Upsets",
            f"{metrics['underdog_blowout_wins']:,}",
            help="Total underdog blowout wins"
        )

    with col3:
        st.metric(
            "Total Games",
            f"{metrics['total_underdog_games']:,}",
            help="Total completed games analyzed"
        )

    # Trend by season
    if not metrics['by_season'].empty:
        st.markdown("**Underdog Blowout Wins by Season**")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=metrics['by_season']['season'],
            y=metrics['by_season']['upset_rate'],
            mode='lines+markers',
            name='Upset Rate',
            line=dict(color='#d62728', width=3),
            marker=dict(size=10),
            hovertemplate='<b>%{x}</b><br>Upset Rate: %{y:.1f}%<br>Upsets: %{customdata}<extra></extra>',
            customdata=metrics['by_season']['upset_wins']
        ))

        fig.update_layout(
            xaxis_title="Season",
            yaxis_title="Upset Rate (%)",
            height=350,
            yaxis=dict(range=[0, max(metrics['by_season']['upset_rate'].max() * 1.2, 10)])
        )

        st.plotly_chart(fig, use_container_width=True)

    # Recent examples
    if not metrics['examples'].empty:
        with st.expander("📋 Recent Underdog Blowout Wins", expanded=False):
            display_df = metrics['examples'].copy()
            display_df['spread_line'] = display_df['spread_line'].round(1)
            display_df['underdog_margin'] = display_df['underdog_margin'].round(1)

            display_df.columns = ['Season', 'Week', 'Underdog', 'Favorite', 'Margin', 'Final Score', 'Spread']

            st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption("💡 **Insight:** Underdog blowout wins (7.5+ points) represent significant upsets where the underdog not only won, but dominated. These games often indicate mismatches in betting lines or exceptional underdog performances.")


def render_day_of_week_analysis(seasons: List[int]):
    """Render day of week analysis."""
    st.subheader("📅 Day of Week Analysis")

    day_metrics = calculate_day_of_week_metrics(seasons)

    if day_metrics.empty:
        st.info("No completed games found for selected seasons.")
        return

    # Bar chart of favorite win rates by day
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=day_metrics['weekday'],
        y=day_metrics['fav_win_rate'],
        text=day_metrics['fav_win_rate'].round(1).astype(str) + '%',
        textposition='outside',
        marker_color='#ff7f0e',
        hovertemplate='<b>%{x}</b><br>Favorite Win Rate: %{y:.1f}%<br>Games: %{customdata}<extra></extra>',
        customdata=day_metrics['total_games']
    ))

    fig.update_layout(
        title="Favorite Win Rate by Day of Week",
        xaxis_title="Day of Week",
        yaxis_title="Favorite Win Rate (%)",
        height=400,
        yaxis=dict(range=[0, 100])
    )

    st.plotly_chart(fig, use_container_width=True)

    # Data table
    with st.expander("📊 View Detailed Day-of-Week Statistics"):
        display_df = day_metrics[['weekday', 'fav_win_rate', 'total_games', 'avg_home_score', 'avg_away_score']].copy()
        display_df.columns = ['Day', 'Fav Win Rate (%)', 'Total Games', 'Avg Home Score', 'Avg Away Score']
        display_df['Fav Win Rate (%)'] = display_df['Fav Win Rate (%)'].round(1)
        display_df['Avg Home Score'] = display_df['Avg Home Score'].round(1)
        display_df['Avg Away Score'] = display_df['Avg Away Score'].round(1)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption("💡 **Insight:** Thursday Night Football and Monday Night Football often show different patterns than Sunday games due to short rest periods.")


def render_over_under_analysis(seasons: List[int]):
    """Render over/under analysis."""
    st.subheader("🎯 Over/Under Analysis")

    ou_metrics = calculate_over_under_rate(seasons)

    if ou_metrics['total_games'] == 0:
        st.info("No completed games found for selected seasons.")
        return

    # Big numbers
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Over Rate", f"{ou_metrics['over_rate']:.1f}%")
    with col2:
        st.metric("Under Rate", f"{ou_metrics['under_rate']:.1f}%")
    with col3:
        st.metric("Push Rate", f"{ou_metrics['push_rate']:.1f}%")
    with col4:
        st.metric("Total Games", f"{ou_metrics['total_games']:,}")

    # Pie chart
    fig = go.Figure(data=[go.Pie(
        labels=['Over', 'Under', 'Push'],
        values=[ou_metrics['over_rate'], ou_metrics['under_rate'], ou_metrics['push_rate']],
        hole=.4,
        marker_colors=['#d62728', '#1f77b4', '#7f7f7f']
    )])

    fig.update_layout(
        title="Over/Under Distribution",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption("💡 **Insight:** Over/under markets are typically efficient, with relatively balanced results over time. Look for season-specific trends or situational edges.")


def render_close_game_analysis(seasons: List[int]):
    """Render close game analysis."""
    st.subheader("📏 Score Margin Distribution")

    distribution = calculate_score_margin_distribution(seasons)

    if distribution.empty:
        st.info("No completed games found for selected seasons.")
        return

    # Bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=distribution['margin_category'].astype(str),
        y=distribution['count'],
        text=distribution['percentage'].astype(str) + '%',
        textposition='outside',
        marker_color='#9467bd',
        hovertemplate='<b>%{x}</b><br>Games: %{y}<br>Percentage: %{text}<extra></extra>'
    ))

    fig.update_layout(
        title="Distribution of Final Score Margins",
        xaxis_title="Score Margin",
        yaxis_title="Number of Games",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    # Close games percentage
    close_games = distribution[distribution['margin_category'].isin(['1 pt', '2-3 pts (FG)', '4-6 pts'])]
    close_pct = close_games['percentage'].sum()

    st.metric("Games Decided by 6 Points or Less", f"{close_pct:.1f}%")

    st.caption("💡 **Insight:** A significant percentage of NFL games are decided by one possession or less, making point spreads crucial for betting analysis.")


def render_common_scores_analysis(seasons: List[int]):
    """Render most common scores analysis."""
    st.subheader("🔢 Most Common Final Scores")

    score_counts = get_most_common_scores(seasons, limit=30)

    if score_counts.empty:
        st.info("No completed games found for selected seasons.")
        return

    # Display as table with formatting
    display_df = score_counts.copy()
    display_df.columns = ['Final Score', 'Frequency', 'Percentage (%)']

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Final Score": st.column_config.TextColumn("Final Score", width="medium"),
            "Frequency": st.column_config.NumberColumn("Frequency", format="%d"),
            "Percentage (%)": st.column_config.NumberColumn("Percentage", format="%.2f%%")
        }
    )

    st.caption("💡 **Insight:** Certain scores (like 20-17, 24-20, 27-24) occur more frequently due to common scoring patterns (TDs + FGs). Use this for same-game parlay and exact score betting strategies.")


# ============================================================================
# Main Render Function
# ============================================================================

def render_historical_trends():
    """Main function to render Historical Trends & Betting Analytics page."""
    st.header("📊 Historical Trends & Betting Analytics")

    st.markdown("""
    Analyze historical NFL game outcomes to identify patterns useful for predictions and betting strategies.
    All data based on regular season games with completed scores and betting lines.
    """)

    # Season filter
    st.sidebar.header("📅 Season Filter")

    # Get available seasons from database
    conn = sqlite3.connect(DB_PATH)
    seasons_df = pd.read_sql_query(
        "SELECT DISTINCT season FROM schedules WHERE game_type = 'REG' ORDER BY season DESC",
        conn
    )
    conn.close()

    available_seasons = seasons_df['season'].tolist() if not seasons_df.empty else []

    if not available_seasons:
        st.error("No season data available in database.")
        return

    selected_seasons = st.sidebar.multiselect(
        "Select Seasons",
        available_seasons,
        default=available_seasons[:3] if len(available_seasons) >= 3 else available_seasons,
        help="Select one or more seasons to analyze"
    )

    if not selected_seasons:
        st.warning("Please select at least one season to analyze.")
        return

    # Tab structure
    tabs = st.tabs([
        "🏠 Home Field",
        "⭐ Favorites",
        "📅 Day of Week",
        "🎯 Over/Under",
        "📏 Close Games",
        "🔢 Common Scores"
    ])

    with tabs[0]:
        render_home_field_analysis(selected_seasons)

    with tabs[1]:
        render_favorite_analysis(selected_seasons)
        st.divider()
        render_home_fav_underdog_split(selected_seasons)
        st.divider()
        render_underdog_blowout_wins(selected_seasons)

    with tabs[2]:
        render_day_of_week_analysis(selected_seasons)

    with tabs[3]:
        render_over_under_analysis(selected_seasons)

    with tabs[4]:
        render_close_game_analysis(selected_seasons)

    with tabs[5]:
        render_common_scores_analysis(selected_seasons)
