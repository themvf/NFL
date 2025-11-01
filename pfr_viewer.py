# -*- coding: utf-8 -*-
"""
NFL Data Viewer - PFR Database Edition
Displays NFL game data from the Pro-Football-Reference SQLite database.
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Tuple
import re
import time
import logging
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from google.cloud import storage
import json

# Configure logging
logging.basicConfig(
    filename='nfl_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Database configuration - relative path for deployment
DB_PATH = Path(__file__).parent / "data" / "pfr.db"

# Page configuration (must be first st command)
st.set_page_config(
    page_title="NFL Data Viewer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# GCS Configuration (from Streamlit secrets) - loaded after page config
GCS_BUCKET_NAME = st.secrets.get("gcs_bucket_name", "")
GCS_DB_BLOB_NAME = "pfr.db"

# ============================================================================
# Google Cloud Storage Functions
# ============================================================================

def get_gcs_client():
    """Initialize GCS client from Streamlit secrets."""
    try:
        if "gcs_service_account" in st.secrets:
            credentials_dict = dict(st.secrets["gcs_service_account"])
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            return storage.Client(credentials=credentials, project=credentials_dict.get("project_id"))
        else:
            # Try default credentials (for local development)
            return storage.Client()
    except Exception as e:
        logging.error(f"Failed to initialize GCS client: {e}")
        return None


def download_db_from_gcs():
    """Download database from Google Cloud Storage."""
    if not GCS_BUCKET_NAME:
        logging.info("GCS bucket name not configured, using local database")
        return False

    try:
        client = get_gcs_client()
        if not client:
            logging.warning("GCS client not available, using local database")
            return False

        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_DB_BLOB_NAME)

        # Create data directory if it doesn't exist
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Download the database
        blob.download_to_filename(str(DB_PATH))
        logging.info(f"Successfully downloaded database from GCS: {GCS_BUCKET_NAME}/{GCS_DB_BLOB_NAME}")
        return True
    except Exception as e:
        logging.error(f"Failed to download database from GCS: {e}")
        return False


def upload_db_to_gcs():
    """Upload database to Google Cloud Storage."""
    if not GCS_BUCKET_NAME:
        logging.info("GCS bucket name not configured, skipping upload")
        return False

    try:
        client = get_gcs_client()
        if not client:
            logging.warning("GCS client not available, skipping upload")
            return False

        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_DB_BLOB_NAME)

        # Upload the database
        blob.upload_from_filename(str(DB_PATH))
        logging.info(f"Successfully uploaded database to GCS: {GCS_BUCKET_NAME}/{GCS_DB_BLOB_NAME}")
        return True
    except Exception as e:
        logging.error(f"Failed to upload database to GCS: {e}")
        return False


# Verify database exists or download from GCS
if not DB_PATH.exists():
    st.info("📥 Downloading database from cloud storage...")
    if not download_db_from_gcs():
        st.error(f"❌ Database not found at {DB_PATH} and could not download from GCS")
        st.info("Please ensure the database exists in Google Cloud Storage or locally.")
        st.stop()


# ============================================================================
# Database Query Functions
# ============================================================================

@st.cache_data(ttl=60)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Execute SQL query and return results as DataFrame."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database query error: {e}")
        return pd.DataFrame()


def get_team_logo_url(team_abbr: str) -> str:
    """
    Get ESPN CDN URL for NFL team logo.

    Args:
        team_abbr: 3-letter team abbreviation from database

    Returns:
        URL to team logo PNG (500x500 transparent background)
    """
    # Map database abbreviations to ESPN CDN abbreviations
    TEAM_LOGO_MAP = {
        'ARI': 'ari', 'ATL': 'atl', 'BAL': 'bal', 'BUF': 'buf',
        'CAR': 'car', 'CHI': 'chi', 'CIN': 'cin', 'CLE': 'cle',
        'DAL': 'dal', 'DEN': 'den', 'DET': 'det', 'GNB': 'gb',
        'HOU': 'hou', 'IND': 'ind', 'JAX': 'jax', 'KAN': 'kc',
        'LAC': 'lac', 'LAR': 'lar', 'LVR': 'lv', 'MIA': 'mia',
        'MIN': 'min', 'NOR': 'no', 'NWE': 'ne', 'NYG': 'nyg',
        'NYJ': 'nyj', 'PHI': 'phi', 'PIT': 'pit', 'SEA': 'sea',
        'SFO': 'sf', 'TAM': 'tb', 'TEN': 'ten', 'WAS': 'wsh'
    }

    espn_abbr = TEAM_LOGO_MAP.get(team_abbr, team_abbr.lower())
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{espn_abbr}.png"


@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_player_ids():
    """Load player ID mapping from nflverse."""
    try:
        import pandas as pd
        # Load player IDs from nflverse GitHub
        url = "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv"
        df = pd.read_csv(url)
        # Create mapping: player name -> espn_id
        player_map = {}
        for _, row in df.iterrows():
            if pd.notna(row.get('espn_id')):
                # Store by display name
                if pd.notna(row.get('display_name')):
                    player_map[row['display_name']] = str(int(row['espn_id']))
                # Also store by short name if available
                if pd.notna(row.get('short_name')):
                    player_map[row['short_name']] = str(int(row['espn_id']))
        return player_map
    except Exception as e:
        st.warning(f"Could not load player IDs: {e}")
        return {}


def get_player_headshot_url(player_name: str, team_abbr: str = None) -> str:
    """
    Get ESPN CDN URL for NFL player headshot.

    Args:
        player_name: Player's full name (e.g., "Patrick Mahomes")
        team_abbr: Team abbreviation (fallback to team logo if player not found)

    Returns:
        URL to player headshot or team logo as fallback
    """
    player_map = load_player_ids()

    # Try to find player ID
    espn_id = player_map.get(player_name)

    if espn_id:
        return f"https://a.espncdn.com/i/headshots/nfl/players/full/{espn_id}.png"
    else:
        # Fallback to team logo if we have team_abbr
        if team_abbr:
            return get_team_logo_url(team_abbr)
        # Ultimate fallback: generic player icon
        return "https://a.espncdn.com/i/headshots/nophoto.png"


@st.cache_data(ttl=300)
def get_seasons() -> List[int]:
    """Get list of available seasons."""
    df = query("SELECT DISTINCT season FROM games WHERE season IS NOT NULL ORDER BY season DESC")
    return df['season'].tolist() if not df.empty else []


@st.cache_data(ttl=300)
def get_weeks(season: int) -> List[int]:
    """Get list of available weeks for a season."""
    df = query("SELECT DISTINCT week FROM games WHERE season=? AND week IS NOT NULL ORDER BY week", (season,))
    return df['week'].tolist() if not df.empty else []


def init_notes_table():
    """Initialize user_notes table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_notes (
                note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                note_text TEXT NOT NULL,
                tags TEXT,
                season INTEGER,
                week INTEGER
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error initializing notes table: {e}")


def init_injuries_table():
    """Initialize player_injuries table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_injuries (
                injury_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                team_abbr TEXT NOT NULL,
                season INTEGER NOT NULL,
                injury_type TEXT NOT NULL,
                start_week INTEGER,
                end_week INTEGER,
                injury_description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player_name, team_abbr, season)
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error initializing injuries table: {e}")


def init_transactions_table():
    """Initialize player_transactions table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_transactions (
                transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                transaction_type TEXT NOT NULL,
                from_team TEXT,
                to_team TEXT,
                season INTEGER NOT NULL,
                effective_week INTEGER NOT NULL,
                transaction_date DATE,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player_name, season, effective_week)
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error initializing transactions table: {e}")


def init_upcoming_games_table():
    """Initialize upcoming_games table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS upcoming_games (
                game_id TEXT PRIMARY KEY,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                date TEXT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                day_of_week TEXT,
                primetime INTEGER DEFAULT 0,
                location TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add location column to existing tables if it doesn't exist
        cursor.execute("PRAGMA table_info(upcoming_games)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'location' not in columns:
            cursor.execute("ALTER TABLE upcoming_games ADD COLUMN location TEXT")

        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error initializing upcoming games table: {e}")


def extract_tags(note_text):
    """
    Extract hashtags from note text and categorize them.
    Returns dict with all_tags, teams, and custom_tags.
    """
    # Find all hashtags (# followed by word characters)
    hashtags = re.findall(r'#(\w+)', note_text)

    # Get all team abbreviations from database
    all_teams = get_teams()
    team_set = set(t.upper() for t in all_teams)

    # Categorize tags
    teams = [tag for tag in hashtags if tag.upper() in team_set]
    custom_tags = [tag for tag in hashtags if tag.upper() not in team_set]

    return {
        'all_tags': hashtags,
        'teams': teams,
        'custom_tags': custom_tags
    }


def save_note(note_text, season=None, week=None):
    """
    Save a new note to the database.
    Returns note_id if successful, None otherwise.
    """
    try:
        # Extract tags from note text
        tag_data = extract_tags(note_text)
        tags_str = ','.join(tag_data['all_tags']) if tag_data['all_tags'] else ''

        # Insert into database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_notes (note_text, tags, season, week)
            VALUES (?, ?, ?, ?)
        """, (note_text, tags_str, season, week))
        note_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Upload database to GCS after successful save
        upload_db_to_gcs()
        return note_id
    except Exception as e:
        st.error(f"Error saving note: {e}")
        return None


def get_notes(team_filter=None, tag_filter=None, season_filter=None, week_filter=None, search_text=None):
    """
    Retrieve notes with optional filtering.

    Parameters:
    - team_filter: Filter by team abbreviation (e.g., 'NWE')
    - tag_filter: Filter by custom tag
    - season_filter: Filter by season
    - week_filter: Filter by week
    - search_text: Full-text search in note_text

    Returns DataFrame with notes.
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # Build query with filters
        sql = "SELECT note_id, created_at, updated_at, note_text, tags, season, week FROM user_notes WHERE 1=1"
        params = []

        if team_filter:
            sql += " AND (tags LIKE ? OR tags LIKE ? OR tags LIKE ? OR tags = ?)"
            params.extend([f"{team_filter},%", f"%,{team_filter},%", f"%,{team_filter}", team_filter])

        if tag_filter:
            sql += " AND (tags LIKE ? OR tags LIKE ? OR tags LIKE ? OR tags = ?)"
            params.extend([f"{tag_filter},%", f"%,{tag_filter},%", f"%,{tag_filter}", tag_filter])

        if season_filter is not None:
            sql += " AND season = ?"
            params.append(season_filter)

        if week_filter is not None:
            sql += " AND week = ?"
            params.append(week_filter)

        if search_text:
            sql += " AND note_text LIKE ?"
            params.append(f"%{search_text}%")

        sql += " ORDER BY created_at DESC"

        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()

        return df
    except Exception as e:
        st.error(f"Error retrieving notes: {e}")
        return pd.DataFrame()


def update_note(note_id, note_text):
    """
    Update an existing note.
    Returns True if successful, False otherwise.
    """
    try:
        # Extract new tags from updated text
        tag_data = extract_tags(note_text)
        tags_str = ','.join(tag_data['all_tags']) if tag_data['all_tags'] else ''

        # Update database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE user_notes
            SET note_text = ?, tags = ?, updated_at = CURRENT_TIMESTAMP
            WHERE note_id = ?
        """, (note_text, tags_str, note_id))
        conn.commit()
        conn.close()

        # Upload database to GCS after successful save
        upload_db_to_gcs()
        return True
    except Exception as e:
        st.error(f"Error updating note: {e}")
        return False


def delete_note(note_id):
    """
    Delete a note by ID.
    Returns True if successful, False otherwise.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_notes WHERE note_id = ?", (note_id,))
        conn.commit()
        conn.close()

        # Upload database to GCS after successful save
        upload_db_to_gcs()
        return True
    except Exception as e:
        st.error(f"Error deleting note: {e}")
        return False


# ============================================================================
# Section: Persistent Injury Management Functions
# ============================================================================

def add_persistent_injury(player_name, team, season, injury_type, start_week=None, end_week=None, description=None):
    """
    Add or update a persistent injury with retry logic for database locks.
    Returns (injury_id, error_message) tuple.
    """
    max_retries = 5
    retry_delay = 0.1  # 100ms

    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=10.0)
            cursor = conn.cursor()

            logging.info(f"Saving injury: {player_name} ({team}) Season {season} - {injury_type}")

            cursor.execute("""
                INSERT OR REPLACE INTO player_injuries
                (player_name, team_abbr, season, injury_type, start_week, end_week, injury_description, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (player_name, team, season, injury_type, start_week, end_week, description))

            injury_id = cursor.lastrowid
            conn.commit()

            # Verify the save
            cursor.execute("""
                SELECT injury_id, player_name, team_abbr FROM player_injuries
                WHERE player_name = ? AND team_abbr = ? AND season = ?
            """, (player_name, team, season))
            result = cursor.fetchone()

            conn.close()

            if result:
                logging.info(f"Successfully saved injury ID {result[0]} for {player_name}")
                # Upload database to GCS after successful save
                upload_db_to_gcs()
                return result[0], None
            else:
                logging.error(f"Injury saved but could not verify: {player_name}")
                return injury_id, "Saved but verification failed"

        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                logging.warning(f"Database locked, retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
            else:
                error_msg = f"Database error after {attempt + 1} attempts: {e}"
                logging.error(error_msg)
                return None, error_msg

        except Exception as e:
            error_msg = f"Unexpected error saving injury: {type(e).__name__}: {e}"
            logging.error(error_msg)
            return None, error_msg

    error_msg = f"Failed to save injury after {max_retries} attempts"
    logging.error(error_msg)
    return None, error_msg


def get_persistent_injuries(team=None, season=None, injury_type=None):
    """Get persistent injuries with optional filtering."""
    try:
        conn = sqlite3.connect(DB_PATH)
        sql = "SELECT * FROM player_injuries WHERE 1=1"
        params = []

        if team:
            sql += " AND team_abbr = ?"
            params.append(team)
        if season:
            sql += " AND season = ?"
            params.append(season)
        if injury_type:
            sql += " AND injury_type = ?"
            params.append(injury_type)

        sql += " ORDER BY updated_at DESC"
        df = pd.read_sql_query(sql, conn, params=params if params else None)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error retrieving injuries: {e}")
        return pd.DataFrame()


def is_player_on_injury_list(player_name, team, season, week):
    """Check if player is on persistent injury list for given week."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT injury_type, start_week, end_week FROM player_injuries
            WHERE player_name = ? AND team_abbr = ? AND season = ?
        """, (player_name, team, season))
        result = cursor.fetchone()
        conn.close()

        if not result:
            return False

        injury_type, start_week, end_week = result

        # Season-ending always returns True
        if injury_type == 'SEASON_ENDING' or end_week == 99:
            return True

        # Check if current week is within injury window
        if start_week and end_week and week:
            return start_week <= week <= end_week

        return True  # If no weeks specified, assume currently injured
    except Exception as e:
        st.error(f"Error checking injury status: {e}")
        return False


def remove_persistent_injury(player_name, team, season):
    """Remove a persistent injury (player returned from injury)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM player_injuries
            WHERE player_name = ? AND team_abbr = ? AND season = ?
        """, (player_name, team, season))
        conn.commit()
        conn.close()
        # Upload database to GCS after successful save
        upload_db_to_gcs()
        return True
    except Exception as e:
        st.error(f"Error removing injury: {e}")
        return False


def get_injured_players_for_team(team, season, week):
    """Get all injured players for a team (session + persistent)."""
    injured = set()

    # Get persistent injuries
    persistent_df = get_persistent_injuries(team=team, season=season)
    for _, row in persistent_df.iterrows():
        if is_player_on_injury_list(row['player_name'], team, season, week):
            injured.add(row['player_name'])

    # Add session-only injuries
    session_injured = get_injured_players_for_session()
    for key in session_injured:
        if team in key:
            player = key.replace(f"{team}_", "")
            injured.add(player)

    return injured


def update_persistent_injury(injury_id, injury_type=None, start_week=None, end_week=None, description=None):
    """Update an existing persistent injury."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        updates = []
        params = []

        if injury_type is not None:
            updates.append("injury_type = ?")
            params.append(injury_type)
        if start_week is not None:
            updates.append("start_week = ?")
            params.append(start_week)
        if end_week is not None:
            updates.append("end_week = ?")
            params.append(end_week)
        if description is not None:
            updates.append("injury_description = ?")
            params.append(description)

        if not updates:
            return False

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(injury_id)

        sql = f"UPDATE player_injuries SET {', '.join(updates)} WHERE injury_id = ?"
        cursor.execute(sql, params)
        conn.commit()
        conn.close()
        # Upload database to GCS after successful save
        upload_db_to_gcs()
        return True
    except Exception as e:
        st.error(f"Error updating injury: {e}")
        return False


def get_database_stats():
    """
    Get statistics about the database for the status panel.
    Returns dict with injury_count, game_count, last_modified, db_size_mb.
    """
    try:
        import os

        # Get file stats
        db_path = DB_PATH
        if os.path.exists(db_path):
            stat_info = os.stat(db_path)
            last_modified = datetime.fromtimestamp(stat_info.st_mtime)
            db_size_mb = stat_info.st_size / (1024 * 1024)
        else:
            last_modified = None
            db_size_mb = 0

        # Get record counts
        conn = sqlite3.connect(DB_PATH, timeout=5.0)
        cursor = conn.cursor()

        # Count injuries
        cursor.execute("SELECT COUNT(*) FROM player_injuries")
        injury_count = cursor.fetchone()[0]

        # Count upcoming games
        cursor.execute("SELECT COUNT(*) FROM upcoming_games")
        game_count = cursor.fetchone()[0]

        conn.close()

        return {
            'injury_count': injury_count,
            'game_count': game_count,
            'last_modified': last_modified,
            'db_size_mb': db_size_mb
        }
    except Exception as e:
        logging.error(f"Error getting database stats: {e}")
        return {
            'injury_count': 0,
            'game_count': 0,
            'last_modified': None,
            'db_size_mb': 0,
            'error': str(e)
        }


# ============================================================================
# Section: Session-Based Injury Management Functions
# ============================================================================

def get_injured_players_for_session():
    """Get set of injured players from session state."""
    if 'injured_players' not in st.session_state:
        st.session_state.injured_players = set()
    return st.session_state.injured_players


def toggle_player_injury(player_name, team):
    """Toggle injury status for a player."""
    key = f"{team}_{player_name}"
    injured = get_injured_players_for_session()
    if key in injured:
        injured.remove(key)
    else:
        injured.add(key)
    st.session_state.injured_players = injured


def is_player_injured(player_name, team):
    """Check if a player is marked as injured."""
    key = f"{team}_{player_name}"
    return key in get_injured_players_for_session()


def clear_all_injuries():
    """Clear all injury flags."""
    st.session_state.injured_players = set()


def redistribute_stats(player_stats_list, team, stat_columns, season=None, week=None):
    """
    Redistribute stats from injured players to healthy players proportionally.
    Checks both session-based and persistent injuries.

    Parameters:
    - player_stats_list: List of dicts with player stats (includes 'Player', 'team', and stat columns)
    - team: Team abbreviation
    - stat_columns: List of tuples (column_name, display_name) to redistribute
                   e.g., [('Expected Yds', 'yards'), ('Expected TDs', 'TDs')]
    - season: Season number (optional, for persistent injury checking)
    - week: Week number (optional, for persistent injury checking)

    Returns:
    - adjusted_stats: List of dicts with adjusted stats
    - redistribution_summary: Dict with redistribution details
    """
    if not player_stats_list:
        return player_stats_list, {}

    # Get all injured players (session + persistent)
    if season and week:
        all_injured = get_injured_players_for_team(team, season, week)
    else:
        # Fallback to session-only if season/week not provided
        all_injured = set()
        session_injured = get_injured_players_for_session()
        for key in session_injured:
            if team in key:
                player = key.replace(f"{team}_", "")
                all_injured.add(player)

    # Identify injured and healthy players
    injured_players = []
    healthy_players = []

    for player_stat in player_stats_list:
        player_name = player_stat['Player']
        if player_name in all_injured:
            injured_players.append(player_stat)
            # Mark if persistent injury
            if season and week:
                player_stat['injury_persistent'] = is_player_on_injury_list(player_name, team, season, week)
            else:
                player_stat['injury_persistent'] = False
        else:
            healthy_players.append(player_stat)

    # If no injured players or no healthy players, return original
    if not injured_players or not healthy_players:
        return player_stats_list, {}

    # Calculate total stats from injured players
    injured_totals = {}
    for col, display_name in stat_columns:
        injured_totals[col] = sum(p.get(col, 0) for p in injured_players)

    # Calculate total stats from healthy players (for proportional distribution)
    healthy_totals = {}
    for col, display_name in stat_columns:
        healthy_totals[col] = sum(p.get(col, 0) for p in healthy_players)

    # Redistribute stats proportionally
    adjusted_stats = []
    redistribution_details = []

    for player_stat in player_stats_list:
        adjusted_player = player_stat.copy()
        player_name = adjusted_player['Player']

        if player_name in all_injured:
            # Mark as injured, zero out stats
            adjusted_player['injured'] = True
            adjusted_player['gains'] = {}
            # Preserve injury_persistent flag if it exists
            if 'injury_persistent' not in adjusted_player and season and week:
                adjusted_player['injury_persistent'] = is_player_on_injury_list(player_name, team, season, week)
        else:
            # Calculate proportional gains
            adjusted_player['injured'] = False
            adjusted_player['gains'] = {}

            for col, display_name in stat_columns:
                if healthy_totals[col] > 0:
                    # Proportion of this player's contribution to healthy total
                    player_proportion = player_stat.get(col, 0) / healthy_totals[col]
                    # Redistribute injured player stats proportionally
                    gain = player_proportion * injured_totals[col]
                    adjusted_player['gains'][col] = gain
                    adjusted_player[col] = player_stat.get(col, 0) + gain

                    redistribution_details.append({
                        'player': player_name,
                        'stat': display_name,
                        'gain': gain
                    })
                else:
                    adjusted_player['gains'][col] = 0

        adjusted_stats.append(adjusted_player)

    # Create redistribution summary
    summary = {
        'injured_players': [p['Player'] for p in injured_players],
        'injured_totals': injured_totals,
        'healthy_count': len(healthy_players),
        'details': redistribution_details
    }

    return adjusted_stats, summary


def calculate_smart_expected_stats(player_name, team, season, week, stat_type, opponent_factor=1.0, strategy='season_avg'):
    """
    Calculate expected player stats using smart aggregation strategies.

    Parameters:
    - player_name: Player's name
    - team: Team abbreviation
    - season: Season number
    - week: Week number (for filtering data up to this week)
    - stat_type: 'passing', 'rushing', or 'receiving'
    - opponent_factor: Defensive adjustment factor (default 1.0)
    - strategy: Aggregation strategy to use
        - 'season_avg': Full season average (default)
        - 'recent_form': Weighted average favoring last 3-5 games
        - 'conservative': Lower quartile (25th percentile)
        - 'optimistic': Upper quartile (75th percentile)
        - 'vs_similar': Against similar-ranked defenses

    Returns:
    - Dictionary with expected, downside, and upside projections for yards and TDs
    """
    # Determine which stats to query based on stat_type
    if stat_type == 'passing':
        yds_col, td_col, att_col = 'pass_yds', 'pass_td', 'pass_att'
        filter_col = 'pass_att > 0'
    elif stat_type == 'rushing':
        yds_col, td_col, att_col = 'rush_yds', 'rush_td', 'rush_att'
        filter_col = 'rush_att > 0'
    elif stat_type == 'receiving':
        yds_col, td_col, att_col = 'rec_yds', 'rec_td', 'rec'
        filter_col = 'rec > 0'
    else:
        raise ValueError(f"Invalid stat_type: {stat_type}. Must be 'passing', 'rushing', or 'receiving'")

    # Query player's game-by-game stats
    sql = f"""
        SELECT
            week,
            {yds_col} as yards,
            {td_col} as tds,
            {att_col} as attempts,
            team
        FROM player_box_score
        WHERE player = ? AND season = ? AND {filter_col}
    """
    params = [player_name, season]

    if week:
        sql += " AND week <= ?"
        params.append(week)

    sql += " ORDER BY week DESC"

    df = query(sql, tuple(params))

    if df.empty:
        # Return zeros if no data
        result = {
            'expected_yds': 0,
            'downside_yds': 0,
            'upside_yds': 0,
            'expected_tds': 0.0,
            'downside_tds': 0.0,
            'upside_tds': 0.0,
            'sample_size': 0,
            'confidence': 'none'
        }
        # Add receptions for receiving stat_type
        if stat_type == 'receiving':
            result.update({
                'expected_rec': 0.0,
                'downside_rec': 0.0,
                'upside_rec': 0.0
            })
        return result

    # Calculate confidence based on sample size
    sample_size = len(df)
    if sample_size >= 8:
        confidence = 'high'
    elif sample_size >= 4:
        confidence = 'medium'
    else:
        confidence = 'low'

    # Apply aggregation strategy
    if strategy == 'season_avg':
        # Season average: mean ± std (balanced expectation)
        expected_yds = df['yards'].mean()
        downside_yds = max(0, df['yards'].mean() - df['yards'].std())
        upside_yds = df['yards'].mean() + df['yards'].std()
        expected_tds = df['tds'].mean()
        downside_tds = max(0, df['tds'].mean() - df['tds'].std())
        upside_tds = df['tds'].mean() + df['tds'].std()

    elif strategy == 'recent_form':
        # Weight recent games more heavily (exponential decay)
        # Last game gets weight 1.0, second-to-last gets 0.8, etc.
        weights = [0.6 ** i for i in range(len(df))]
        total_weight = sum(weights)

        weighted_yds = sum(df.iloc[i]['yards'] * weights[i] for i in range(len(df))) / total_weight
        weighted_tds = sum(df.iloc[i]['tds'] * weights[i] for i in range(len(df))) / total_weight

        # Use last 5 games for variance calculation
        recent_df = df.head(5)
        yds_std = recent_df['yards'].std() if len(recent_df) > 1 else df['yards'].std()
        tds_std = recent_df['tds'].std() if len(recent_df) > 1 else df['tds'].std()

        expected_yds = weighted_yds
        downside_yds = max(0, weighted_yds - yds_std)
        upside_yds = weighted_yds + yds_std
        expected_tds = weighted_tds
        downside_tds = max(0, weighted_tds - tds_std)
        upside_tds = weighted_tds + tds_std

    elif strategy == 'conservative':
        # Use 25th percentile as expected, 10th as downside, median as upside
        expected_yds = df['yards'].quantile(0.25)
        downside_yds = df['yards'].quantile(0.10)
        upside_yds = df['yards'].median()
        expected_tds = df['tds'].quantile(0.25)
        downside_tds = df['tds'].quantile(0.10)
        upside_tds = df['tds'].median()

    elif strategy == 'optimistic':
        # Use 75th percentile as expected, median as downside, 90th as upside
        expected_yds = df['yards'].quantile(0.75)
        downside_yds = df['yards'].median()
        upside_yds = df['yards'].quantile(0.90)
        expected_tds = df['tds'].quantile(0.75)
        downside_tds = df['tds'].median()
        upside_tds = df['tds'].quantile(0.90)

    elif strategy == 'vs_similar':
        # For now, fall back to season_avg (would require defensive ranking data)
        # TODO: Implement matchup-based filtering
        expected_yds = df['yards'].mean()
        downside_yds = max(0, df['yards'].mean() - df['yards'].std())
        upside_yds = df['yards'].mean() + df['yards'].std()
        expected_tds = df['tds'].mean()
        downside_tds = max(0, df['tds'].mean() - df['tds'].std())
        upside_tds = df['tds'].mean() + df['tds'].std()

    else:
        # Default to season_avg
        expected_yds = df['yards'].mean()
        downside_yds = max(0, df['yards'].mean() - df['yards'].std())
        upside_yds = df['yards'].mean() + df['yards'].std()
        expected_tds = df['tds'].mean()
        downside_tds = max(0, df['tds'].mean() - df['tds'].std())
        upside_tds = df['tds'].mean() + df['tds'].std()

    # Calculate receptions for receiving stat_type (don't apply opponent factor - receptions are more about targets/role)
    if stat_type == 'receiving':
        expected_rec = df['attempts'].mean()
        downside_rec = max(0, df['attempts'].mean() - df['attempts'].std())
        upside_rec = df['attempts'].mean() + df['attempts'].std()

    # Apply opponent adjustment factor
    result = {
        'expected_yds': int(expected_yds * opponent_factor) if pd.notna(expected_yds) else 0,
        'downside_yds': int(downside_yds * opponent_factor) if pd.notna(downside_yds) else 0,
        'upside_yds': int(upside_yds * opponent_factor) if pd.notna(upside_yds) else 0,
        'expected_tds': round(expected_tds * opponent_factor, 1) if pd.notna(expected_tds) else 0.0,
        'downside_tds': round(downside_tds * opponent_factor, 1) if pd.notna(downside_tds) else 0.0,
        'upside_tds': round(upside_tds * opponent_factor, 1) if pd.notna(upside_tds) else 0.0,
        'sample_size': sample_size,
        'confidence': confidence
    }

    # Add reception stats for receiving
    if stat_type == 'receiving':
        result.update({
            'expected_rec': round(expected_rec, 1) if pd.notna(expected_rec) else 0.0,
            'downside_rec': round(downside_rec, 1) if pd.notna(downside_rec) else 0.0,
            'upside_rec': round(upside_rec, 1) if pd.notna(upside_rec) else 0.0
        })

    return result


@st.cache_data(ttl=300)
def get_teams(season: Optional[int] = None) -> List[str]:
    """Get list of teams, optionally filtered by season."""
    if season:
        sql = """
            SELECT DISTINCT team FROM (
                SELECT home_team_abbr as team FROM games WHERE season=?
                UNION
                SELECT away_team_abbr as team FROM games WHERE season=?
            ) WHERE team IS NOT NULL ORDER BY team
        """
        df = query(sql, (season, season))
    else:
        sql = """
            SELECT DISTINCT team FROM (
                SELECT home_team_abbr as team FROM games
                UNION
                SELECT away_team_abbr as team FROM games
            ) WHERE team IS NOT NULL ORDER BY team
        """
        df = query(sql)
    return df['team'].tolist() if not df.empty else []


# ============================================================================
# Section: Player Transaction Management Functions
# ============================================================================

def add_transaction(player_name, transaction_type, season, effective_week, from_team=None, to_team=None, notes=None):
    """
    Record a player transaction.
    Returns transaction_id if successful, None otherwise.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO player_transactions
            (player_name, transaction_type, from_team, to_team, season, effective_week, transaction_date, notes)
            VALUES (?, ?, ?, ?, ?, ?, DATE('now'), ?)
        """, (player_name, transaction_type, from_team, to_team, season, effective_week, notes))
        transaction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return transaction_id
    except Exception as e:
        st.error(f"Error adding transaction: {e}")
        return None


def get_player_current_team(player_name, season, as_of_week=None):
    """
    Get player's current team based on transaction history.
    Returns team abbreviation or None if free agent.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        sql = """
            SELECT to_team, effective_week FROM player_transactions
            WHERE player_name = ? AND season = ?
        """
        params = [player_name, season]

        if as_of_week:
            sql += " AND effective_week <= ?"
            params.append(as_of_week)

        sql += " ORDER BY effective_week DESC LIMIT 1"

        cursor.execute(sql, params)
        result = cursor.fetchone()
        conn.close()

        if result:
            return result[0]  # to_team (None if released)

        # No transaction found - infer from game data
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        sql = "SELECT team FROM player_box_score WHERE player = ? AND season = ?"
        params = [player_name, season]

        if as_of_week:
            sql += " AND week <= ?"
            params.append(as_of_week)

        sql += " ORDER BY week DESC LIMIT 1"

        cursor.execute(sql, params)
        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None
    except Exception as e:
        st.error(f"Error getting player current team: {e}")
        return None


def get_player_transactions(player_name=None, season=None, team=None, transaction_type=None):
    """Get transaction history with optional filtering."""
    try:
        conn = sqlite3.connect(DB_PATH)
        sql = "SELECT * FROM player_transactions WHERE 1=1"
        params = []

        if player_name:
            sql += " AND player_name = ?"
            params.append(player_name)
        if season:
            sql += " AND season = ?"
            params.append(season)
        if team:
            sql += " AND (from_team = ? OR to_team = ?)"
            params.extend([team, team])
        if transaction_type:
            sql += " AND transaction_type = ?"
            params.append(transaction_type)

        sql += " ORDER BY season DESC, effective_week DESC"

        df = pd.read_sql_query(sql, conn, params=params if params else None)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error retrieving transactions: {e}")
        return pd.DataFrame()


def get_team_roster(team, season, as_of_week=None):
    """
    Get active roster for a team at a specific point in time.
    Returns DataFrame with player names and metadata.
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # Get all players who played for this team
        sql = """
            SELECT DISTINCT player, team, season
            FROM player_box_score
            WHERE team = ? AND season = ?
        """
        params = [team, season]

        if as_of_week:
            sql += " AND week <= ?"
            params.append(as_of_week)

        players_df = pd.read_sql_query(sql, conn, params=params)
        conn.close()

        # Filter out players who were traded away
        current_roster = []
        for _, player_row in players_df.iterrows():
            player_name = player_row['player']
            current_team = get_player_current_team(player_name, season, as_of_week)

            if current_team == team:
                current_roster.append({
                    'player': player_name,
                    'team': team,
                    'season': season
                })

        return pd.DataFrame(current_roster)
    except Exception as e:
        st.error(f"Error getting team roster: {e}")
        return pd.DataFrame()


def delete_transaction(transaction_id):
    """Delete a transaction by ID."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM player_transactions WHERE transaction_id = ?", (transaction_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error deleting transaction: {e}")
        return False


def get_player_transaction_indicator(player_name, team, season, as_of_week=None):
    """
    Get transaction indicator for a player showing when they joined/left a team.

    Parameters:
    - player_name: Player's name
    - team: Team abbreviation
    - season: Season number
    - as_of_week: Week to check transactions up to (None = all season)

    Returns:
    - Dictionary with:
        - 'indicator': Visual indicator string (e.g., "▶️ W3", "◀️ W5", "⚠️ NEW")
        - 'joined_week': Week player joined team (None if started with team)
        - 'left_week': Week player left team (None if still on team)
        - 'is_valid': Boolean - True if player was on team during analysis period
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # Get all transactions for this player in this season
        sql = """
            SELECT transaction_type, from_team, to_team, effective_week
            FROM player_transactions
            WHERE player_name = ? AND season = ?
        """
        params = [player_name, season]

        if as_of_week:
            sql += " AND effective_week <= ?"
            params.append(as_of_week)

        sql += " ORDER BY effective_week ASC"

        cursor = conn.cursor()
        cursor.execute(sql, params)
        transactions = cursor.fetchall()
        conn.close()

        joined_week = None
        left_week = None
        indicator = ""
        is_valid = True

        for trans_type, from_team_val, to_team_val, eff_week in transactions:
            # Check if player joined this team
            if to_team_val == team:
                joined_week = eff_week
                indicator = f"▶️ W{eff_week}"

            # Check if player left this team
            if from_team_val == team and to_team_val != team:
                left_week = eff_week
                indicator = f"◀️ W{eff_week}"

        # Check if player was never on this team during the period
        if left_week and not joined_week:
            # Player left but we don't know when they joined - might have started with team
            is_valid = True
        elif joined_week and as_of_week and joined_week > as_of_week:
            # Player joined after the analysis period
            is_valid = False
        elif left_week and as_of_week and left_week < as_of_week:
            # Player left before the analysis period - but might have useful data
            is_valid = True  # Still valid for historical analysis

        return {
            'indicator': indicator,
            'joined_week': joined_week,
            'left_week': left_week,
            'is_valid': is_valid
        }

    except Exception as e:
        # If there's an error, assume player is valid
        return {
            'indicator': '',
            'joined_week': None,
            'left_week': None,
            'is_valid': True
        }


def detect_unreported_transactions(season):
    """
    Analyze player_box_score data to find players who switched teams
    without recorded transactions. Returns DataFrame of potential transactions.
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # Find players who appeared on multiple teams in the same season
        sql = """
            SELECT player, season, MIN(week) as first_week, MAX(week) as last_week,
                   GROUP_CONCAT(DISTINCT team) as teams, COUNT(DISTINCT team) as team_count
            FROM player_box_score
            WHERE season = ?
            GROUP BY player, season
            HAVING team_count > 1
            ORDER BY player
        """

        multi_team_players = pd.read_sql_query(sql, conn, params=[season])

        potential_transactions = []

        for _, row in multi_team_players.iterrows():
            player_name = row['player']

            # Get week-by-week team history
            sql = """
                SELECT DISTINCT week, team
                FROM player_box_score
                WHERE player = ? AND season = ?
                ORDER BY week
            """
            cursor = conn.cursor()
            cursor.execute(sql, (player_name, season))
            week_history = cursor.fetchall()

            # Find transition points
            prev_team = None
            for week, team in week_history:
                if prev_team and team != prev_team:
                    # Check if transaction already recorded
                    cursor.execute("""
                        SELECT COUNT(*) FROM player_transactions
                        WHERE player_name = ? AND season = ? AND effective_week = ?
                    """, (player_name, season, week))

                    if cursor.fetchone()[0] == 0:
                        potential_transactions.append({
                            'player_name': player_name,
                            'from_team': prev_team,
                            'to_team': team,
                            'effective_week': week,
                            'season': season
                        })

                prev_team = team

        conn.close()
        return pd.DataFrame(potential_transactions)
    except Exception as e:
        st.error(f"Error detecting transactions: {e}")
        return pd.DataFrame()


# ============================================================================
# Upcoming Games Functions
# ============================================================================

def upload_upcoming_schedule_csv(csv_data, season):
    """
    Parse and upload upcoming games from CSV data with retry logic.

    Parameters:
    - csv_data: Pandas DataFrame from CSV
    - season: Season number

    Returns:
    - Tuple of (success_count, error_messages)
    """
    success_count = 0
    errors = []
    max_retries = 5
    retry_delay = 0.1

    logging.info(f"Starting CSV upload: {len(csv_data)} games for season {season}")

    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=10.0)
            cursor = conn.cursor()

            for idx, row in csv_data.iterrows():
                try:
                    # Validate required fields
                    if pd.isna(row.get('Week')) or pd.isna(row.get('Home')) or pd.isna(row.get('Away')):
                        errors.append(f"Missing required fields in row {idx + 2}: {row.to_dict()}")
                        continue

                    week = int(row['Week'])
                    home_team = str(row['Home']).strip()
                    away_team = str(row['Away']).strip()
                    day_of_week = str(row.get('Day', '')).strip() if not pd.isna(row.get('Day')) else ''
                    primetime = 1 if str(row.get('Primetime', '')).strip().lower() == 'yes' else 0
                    location = str(row.get('Location', '')).strip() if not pd.isna(row.get('Location')) else ''

                    # Create game_id from week and teams
                    game_id = f"{season}W{week:02d}{away_team}{home_team}"

                    # Insert or replace game
                    cursor.execute("""
                        INSERT OR REPLACE INTO upcoming_games
                        (game_id, season, week, date, home_team, away_team, day_of_week, primetime, location)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        game_id,
                        season,
                        week,
                        None,  # No date in CSV
                        home_team,
                        away_team,
                        day_of_week,
                        primetime,
                        location
                    ))
                    success_count += 1

                except Exception as e:
                    errors.append(f"Error processing row {idx + 2}: {str(e)}")

            conn.commit()
            conn.close()

            logging.info(f"Successfully uploaded {success_count} games")
            # Upload database to GCS after successful save
            upload_db_to_gcs()
            return success_count, errors

        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                logging.warning(f"Database locked during CSV upload, retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(retry_delay * (attempt + 1))
                continue
            else:
                error_msg = f"Database error after {attempt + 1} attempts: {str(e)}"
                logging.error(error_msg)
                errors.append(error_msg)
                return success_count, errors

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logging.error(error_msg)
            errors.append(error_msg)
            return success_count, errors

    errors.append(f"Failed to upload CSV after {max_retries} attempts")
    return success_count, errors


def get_upcoming_games(season=None, week=None):
    """Get upcoming games from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)

        sql = "SELECT * FROM upcoming_games WHERE 1=1"
        params = []

        if season:
            sql += " AND season = ?"
            params.append(season)

        if week:
            sql += " AND week = ?"
            params.append(week)

        sql += " ORDER BY week, date, game_id"

        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        return df

    except Exception as e:
        st.error(f"Error fetching upcoming games: {e}")
        return pd.DataFrame()


def delete_upcoming_game(game_id):
    """Delete an upcoming game by ID."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM upcoming_games WHERE game_id = ?", (game_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error deleting game: {e}")
        return False


def clear_upcoming_games(season):
    """Clear all upcoming games for a season."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM upcoming_games WHERE season = ?", (season,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error clearing games: {e}")
        return False


# ============================================================================
# Player Projection Functions
# ============================================================================

def calculate_defensive_stats(season, max_week):
    """
    Calculate defensive stats for each team (yards allowed by position).

    Returns dict: {team: {pass_allowed, rush_allowed, rec_to_rb, rec_to_wr, rec_to_te}}
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # Get all teams
        teams_query = "SELECT DISTINCT team FROM player_box_score WHERE season = ?"
        teams_df = pd.read_sql_query(teams_query, conn, params=(season,))
        teams = teams_df['team'].tolist()

        defensive_stats = {}

        for team in teams:
            # Pass yards allowed (opponent QBs)
            # Join with games to find players who played AGAINST this team
            pass_query = """
                SELECT AVG(p.pass_yds) as avg_pass_allowed
                FROM player_box_score p
                JOIN games g ON p.season = g.season AND p.week = g.week
                WHERE p.season = ? AND p.week < ? AND p.pass_att > 10
                  AND (
                    (p.team = g.away_team_abbr AND g.home_team_abbr = ?) OR
                    (p.team = g.home_team_abbr AND g.away_team_abbr = ?)
                  )
            """
            pass_df = pd.read_sql_query(pass_query, conn, params=(season, max_week, team, team))

            # Rush yards allowed (opponent RBs)
            rush_query = """
                SELECT AVG(p.rush_yds) as avg_rush_allowed
                FROM player_box_score p
                JOIN games g ON p.season = g.season AND p.week = g.week
                WHERE p.season = ? AND p.week < ? AND p.rush_att >= 5
                  AND (
                    (p.team = g.away_team_abbr AND g.home_team_abbr = ?) OR
                    (p.team = g.home_team_abbr AND g.away_team_abbr = ?)
                  )
            """
            rush_df = pd.read_sql_query(rush_query, conn, params=(season, max_week, team, team))

            # Receiving yards allowed to RBs (opponents with rush + rec)
            rec_rb_query = """
                SELECT AVG(p.rec_yds) as avg_rec_to_rb
                FROM player_box_score p
                JOIN games g ON p.season = g.season AND p.week = g.week
                WHERE p.season = ? AND p.week < ?
                  AND p.rush_att >= 5 AND p.targets > 0
                  AND (
                    (p.team = g.away_team_abbr AND g.home_team_abbr = ?) OR
                    (p.team = g.home_team_abbr AND g.away_team_abbr = ?)
                  )
            """
            rec_rb_df = pd.read_sql_query(rec_rb_query, conn, params=(season, max_week, team, team))

            # Receiving yards allowed to WRs
            rec_wr_query = """
                SELECT AVG(p.rec_yds) as avg_rec_to_wr
                FROM player_box_score p
                JOIN games g ON p.season = g.season AND p.week = g.week
                WHERE p.season = ? AND p.week < ?
                  AND p.targets >= 4 AND p.rush_att < 3
                  AND (
                    (p.team = g.away_team_abbr AND g.home_team_abbr = ?) OR
                    (p.team = g.home_team_abbr AND g.away_team_abbr = ?)
                  )
            """
            rec_wr_df = pd.read_sql_query(rec_wr_query, conn, params=(season, max_week, team, team))

            # Receiving yards allowed to TEs
            rec_te_query = """
                SELECT AVG(p.rec_yds) as avg_rec_to_te
                FROM player_box_score p
                JOIN games g ON p.season = g.season AND p.week = g.week
                WHERE p.season = ? AND p.week < ?
                  AND p.targets >= 2 AND p.targets < 10 AND p.rush_att < 2
                  AND (
                    (p.team = g.away_team_abbr AND g.home_team_abbr = ?) OR
                    (p.team = g.home_team_abbr AND g.away_team_abbr = ?)
                  )
            """
            rec_te_df = pd.read_sql_query(rec_te_query, conn, params=(season, max_week, team, team))

            defensive_stats[team] = {
                'pass_allowed': pass_df['avg_pass_allowed'].iloc[0] if not pass_df.empty and not pd.isna(pass_df['avg_pass_allowed'].iloc[0]) else 240,
                'rush_allowed': rush_df['avg_rush_allowed'].iloc[0] if not rush_df.empty and not pd.isna(rush_df['avg_rush_allowed'].iloc[0]) else 80,
                'rec_to_rb': rec_rb_df['avg_rec_to_rb'].iloc[0] if not rec_rb_df.empty and not pd.isna(rec_rb_df['avg_rec_to_rb'].iloc[0]) else 20,
                'rec_to_wr': rec_wr_df['avg_rec_to_wr'].iloc[0] if not rec_wr_df.empty and not pd.isna(rec_wr_df['avg_rec_to_wr'].iloc[0]) else 60,
                'rec_to_te': rec_te_df['avg_rec_to_te'].iloc[0] if not rec_te_df.empty and not pd.isna(rec_te_df['avg_rec_to_te'].iloc[0]) else 40
            }

        conn.close()
        return defensive_stats

    except Exception as e:
        st.error(f"Error calculating defensive stats: {e}")
        return {}


def calculate_player_medians(season, max_week, teams_playing=None):
    """
    Calculate median stats for all players by position.

    Returns DataFrame with columns: player, team, position_type, median stats, games_played
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # Base query for all players
        base_query = """
            SELECT
                player,
                team,
                pass_yds,
                pass_td,
                pass_cmp,
                pass_att,
                rush_yds,
                rush_att,
                rec_yds,
                rec,
                targets,
                rush_td,
                rec_td
            FROM player_box_score
            WHERE season = ? AND week < ?
        """

        params = [season, max_week]

        if teams_playing:
            placeholders = ','.join(['?' for _ in teams_playing])
            base_query += f" AND team IN ({placeholders})"
            params.extend(teams_playing)

        df = pd.read_sql_query(base_query, conn, params=params)
        conn.close()

        if df.empty:
            return pd.DataFrame()

        # Classify position and calculate medians
        results = []

        for (player, team), group in df.groupby(['player', 'team']):
            games_played = len(group)

            if games_played < 2:  # Need at least 2 games
                continue

            # Determine position
            avg_pass_att = group['pass_att'].mean()
            avg_rush_att = group['rush_att'].mean()
            avg_targets = group['targets'].mean()

            position = None
            stats = {'player': player, 'team': team, 'games_played': games_played}

            # QB: High pass attempts
            if avg_pass_att > 10:
                position = 'QB'
                stats.update({
                    'avg_pass_yds': group['pass_yds'].mean(),
                    'median_pass_yds': group['pass_yds'].median(),
                    'median_pass_td': group['pass_td'].median(),
                    'median_pass_cmp_pct': (group['pass_cmp'].sum() / group['pass_att'].sum() * 100) if group['pass_att'].sum() > 0 else 0
                })

            # RB: High rush attempts
            elif avg_rush_att >= 5:
                position = 'RB'
                stats.update({
                    'avg_total_yds': (group['rush_yds'] + group['rec_yds']).mean(),
                    'median_rush_yds': group['rush_yds'].median(),
                    'median_rec_yds': group['rec_yds'].median(),
                    'median_total_yds': (group['rush_yds'] + group['rec_yds']).median(),
                    'median_total_td': (group['rush_td'] + group['rec_td']).median(),
                    'median_targets': group['targets'].median()
                })

            # WR: High targets, low rushing
            elif avg_targets >= 4 and avg_rush_att < 3:
                position = 'WR'
                stats.update({
                    'avg_rec_yds': group['rec_yds'].mean(),
                    'median_rec_yds': group['rec_yds'].median(),
                    'median_targets': group['targets'].median(),
                    'median_rec_td': group['rec_td'].median(),
                    'median_rec': group['rec'].median()
                })

            # TE: Moderate targets
            elif avg_targets >= 2 and avg_targets < 10 and avg_rush_att < 2:
                position = 'TE'
                stats.update({
                    'avg_rec_yds': group['rec_yds'].mean(),
                    'median_rec_yds': group['rec_yds'].median(),
                    'median_targets': group['targets'].median(),
                    'median_rec_td': group['rec_td'].median(),
                    'median_rec': group['rec'].median()
                })

            if position:
                stats['position_type'] = position
                results.append(stats)

        return pd.DataFrame(results)

    except Exception as e:
        st.error(f"Error calculating player medians: {e}")
        return pd.DataFrame()


def generate_player_projections(season, week, teams_playing):
    """
    Generate defensive-adjusted projections for all players.

    Returns dict: {position: DataFrame with projections}
    """
    try:
        # Calculate defensive stats
        defensive_stats = calculate_defensive_stats(season, week)

        # Calculate league averages for normalization
        league_avg = {
            'pass': sum([d['pass_allowed'] for d in defensive_stats.values()]) / len(defensive_stats) if defensive_stats else 240,
            'rush': sum([d['rush_allowed'] for d in defensive_stats.values()]) / len(defensive_stats) if defensive_stats else 80,
            'rec_rb': sum([d['rec_to_rb'] for d in defensive_stats.values()]) / len(defensive_stats) if defensive_stats else 20,
            'rec_wr': sum([d['rec_to_wr'] for d in defensive_stats.values()]) / len(defensive_stats) if defensive_stats else 60,
            'rec_te': sum([d['rec_to_te'] for d in defensive_stats.values()]) / len(defensive_stats) if defensive_stats else 40
        }

        # Get player medians
        player_medians = calculate_player_medians(season, week, teams_playing)

        if player_medians.empty:
            return {}

        # Get matchups and injured players from database
        conn = sqlite3.connect(DB_PATH)

        matchups_query = """
            SELECT home_team, away_team
            FROM upcoming_games
            WHERE season = ? AND week = ?
        """
        matchups_df = pd.read_sql_query(matchups_query, conn, params=(season, week))

        # Get injured players for this week
        injuries_query = """
            SELECT player_name, team_abbr
            FROM player_injuries
            WHERE season = ?
              AND ? >= start_week
              AND ? <= end_week
        """
        injuries_df = pd.read_sql_query(injuries_query, conn, params=(season, week, week))

        conn.close()

        # Create set of injured players for fast lookup
        injured_players = set(injuries_df['player_name'].tolist()) if not injuries_df.empty else set()

        # Create team to opponent mapping
        team_opponent = {}
        for _, row in matchups_df.iterrows():
            team_opponent[row['home_team']] = row['away_team']
            team_opponent[row['away_team']] = row['home_team']

        # Apply defensive multipliers
        projections = {'QB': [], 'RB': [], 'WR': [], 'TE': [], 'SKILL': []}

        for _, player in player_medians.iterrows():
            # Skip injured players
            if player['player'] in injured_players:
                continue

            opponent = team_opponent.get(player['team'])
            if not opponent or opponent not in defensive_stats:
                continue

            opponent_def = defensive_stats[opponent]
            position = player['position_type']

            if position == 'QB':
                multiplier = opponent_def['pass_allowed'] / league_avg['pass']
                projected_yds = player['median_pass_yds'] * multiplier

                projections['QB'].append({
                    'Player': player['player'],
                    'Team': player['team'],
                    'Opponent': opponent,
                    'Avg Yds/Game': round(player['avg_pass_yds'], 1),
                    'Median Pass Yds': round(player['median_pass_yds'], 1),
                    'Def Allows': round(opponent_def['pass_allowed'], 1),
                    'Projected Yds': round(projected_yds, 1),
                    'Multiplier': round(multiplier, 2),
                    'Games': player['games_played']
                })

            elif position == 'RB':
                rush_mult = opponent_def['rush_allowed'] / league_avg['rush']
                rec_mult = opponent_def['rec_to_rb'] / league_avg['rec_rb']

                proj_rush = player['median_rush_yds'] * rush_mult
                proj_rec = player['median_rec_yds'] * rec_mult
                proj_total = proj_rush + proj_rec

                avg_mult = (rush_mult + rec_mult) / 2

                projections['RB'].append({
                    'Player': player['player'],
                    'Team': player['team'],
                    'Opponent': opponent,
                    'Avg Yds/Game': round(player['avg_total_yds'], 1),
                    'Median Rush': round(player['median_rush_yds'], 1),
                    'Median Rec': round(player['median_rec_yds'], 1),
                    'Projected Total': round(proj_total, 1),
                    'Multiplier': round(avg_mult, 2),
                    'Games': player['games_played']
                })

                projections['SKILL'].append({
                    'Player': player['player'],
                    'Team': f"{player['team']} (RB)",
                    'Opponent': opponent,
                    'Avg Yds/Game': round(player['avg_total_yds'], 1),
                    'Median Yds': round(player['median_total_yds'], 1),
                    'Projected Yds': round(proj_total, 1),
                    'Multiplier': round(avg_mult, 2),
                    'Games': player['games_played']
                })

            elif position == 'WR':
                multiplier = opponent_def['rec_to_wr'] / league_avg['rec_wr']
                projected_yds = player['median_rec_yds'] * multiplier

                projections['WR'].append({
                    'Player': player['player'],
                    'Team': player['team'],
                    'Opponent': opponent,
                    'Avg Yds/Game': round(player['avg_rec_yds'], 1),
                    'Median Rec Yds': round(player['median_rec_yds'], 1),
                    'Median Tgts': round(player['median_targets'], 1),
                    'Projected Yds': round(projected_yds, 1),
                    'Multiplier': round(multiplier, 2),
                    'Games': player['games_played']
                })

                projections['SKILL'].append({
                    'Player': player['player'],
                    'Team': f"{player['team']} (WR)",
                    'Opponent': opponent,
                    'Avg Yds/Game': round(player['avg_rec_yds'], 1),
                    'Median Yds': round(player['median_rec_yds'], 1),
                    'Projected Yds': round(projected_yds, 1),
                    'Multiplier': round(multiplier, 2),
                    'Games': player['games_played']
                })

            elif position == 'TE':
                multiplier = opponent_def['rec_to_te'] / league_avg['rec_te']
                projected_yds = player['median_rec_yds'] * multiplier

                projections['TE'].append({
                    'Player': player['player'],
                    'Team': player['team'],
                    'Opponent': opponent,
                    'Avg Yds/Game': round(player['avg_rec_yds'], 1),
                    'Median Rec Yds': round(player['median_rec_yds'], 1),
                    'Median Tgts': round(player['median_targets'], 1),
                    'Projected Yds': round(projected_yds, 1),
                    'Multiplier': round(multiplier, 2),
                    'Games': player['games_played']
                })

                projections['SKILL'].append({
                    'Player': player['player'],
                    'Team': f"{player['team']} (TE)",
                    'Opponent': opponent,
                    'Avg Yds/Game': round(player['avg_rec_yds'], 1),
                    'Median Yds': round(player['median_rec_yds'], 1),
                    'Projected Yds': round(projected_yds, 1),
                    'Multiplier': round(multiplier, 2),
                    'Games': player['games_played']
                })

        # Convert to DataFrames and sort
        result = {}
        for pos, data in projections.items():
            if data:
                df = pd.DataFrame(data)
                sort_col = 'Projected Yds' if 'Projected Yds' in df.columns else 'Projected Total'
                result[pos] = df.sort_values(sort_col, ascending=False)
            else:
                result[pos] = pd.DataFrame()

        return result

    except Exception as e:
        st.error(f"Error generating projections: {e}")
        return {}


def get_matchup_rating(multiplier):
    """Convert multiplier to matchup rating and color."""
    if multiplier >= 1.15:
        return '🔥 Great', 'background-color: #90EE90'
    elif multiplier >= 1.05:
        return '✅ Good', 'background-color: #D3FFD3'
    elif multiplier >= 0.95:
        return '⚪ Average', ''
    elif multiplier >= 0.85:
        return '⚠️ Tough', 'background-color: #FFD3D3'
    else:
        return '🛑 Brutal', 'background-color: #FF9090'


# ============================================================================
# Advanced Metrics Functions
# ============================================================================

def calculate_win_loss_record(team_abbr: str, season: int, week: Optional[int] = None) -> dict:
    """Calculate team's win-loss record and winning percentage."""
    sql = """
        SELECT
            g.game_id,
            g.home_team_abbr,
            g.away_team_abbr,
            g.home_score,
            g.away_score
        FROM games g
        WHERE g.season = ?
        AND (g.home_team_abbr = ? OR g.away_team_abbr = ?)
    """
    params = [season, team_abbr, team_abbr]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    games = query(sql, tuple(params))

    if games.empty:
        return {'wins': 0, 'losses': 0, 'win_pct': 0.0, 'games_played': 0}

    wins = 0
    losses = 0

    for _, row in games.iterrows():
        is_home = row['home_team_abbr'] == team_abbr
        team_score = row['home_score'] if is_home else row['away_score']
        opp_score = row['away_score'] if is_home else row['home_score']

        if team_score > opp_score:
            wins += 1
        elif team_score < opp_score:
            losses += 1

    games_played = wins + losses
    win_pct = wins / games_played if games_played > 0 else 0.0

    return {
        'wins': wins,
        'losses': losses,
        'win_pct': win_pct,
        'games_played': games_played
    }


def calculate_sos_adjusted_record(team_abbr: str, season: int, week: Optional[int] = None, all_team_powers: dict = None) -> dict:
    """
    Calculate strength-of-schedule adjusted win-loss record.
    Weights wins/losses by opponent power rating.
    """
    sql = """
        SELECT
            g.game_id,
            g.week,
            g.home_team_abbr,
            g.away_team_abbr,
            g.home_score,
            g.away_score
        FROM games g
        WHERE g.season = ?
        AND (g.home_team_abbr = ? OR g.away_team_abbr = ?)
    """
    params = [season, team_abbr, team_abbr]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    games = query(sql, tuple(params))

    if games.empty or not all_team_powers:
        return {'sos_adj_win_pct': 0.0, 'avg_opp_power': 0.0, 'quality_wins': 0, 'bad_losses': 0}

    total_quality_score = 0
    total_opp_power = 0
    games_count = 0
    quality_wins = 0  # Wins against teams with power > 50
    bad_losses = 0    # Losses against teams with power < 50

    for _, row in games.iterrows():
        is_home = row['home_team_abbr'] == team_abbr
        opponent = row['away_team_abbr'] if is_home else row['home_team_abbr']
        team_score = row['home_score'] if is_home else row['away_score']
        opp_score = row['away_score'] if is_home else row['home_score']

        # Get opponent's power rating (use 50 as default if not available)
        opp_power = all_team_powers.get(opponent, 50)
        total_opp_power += opp_power

        # Result: 1 for win, 0 for loss
        result = 1 if team_score > opp_score else 0

        # Quality score: weight result by opponent strength
        # Strong opponents (power > 50): wins worth more, losses hurt less
        # Weak opponents (power < 50): wins worth less, losses hurt more
        quality_multiplier = opp_power / 50  # Normalize around average (50)
        if result == 1:
            total_quality_score += quality_multiplier
            if opp_power > 50:
                quality_wins += 1
        else:
            # For losses, inverse the effect (losing to weak teams hurts more)
            if opp_power < 50:
                bad_losses += 1

        games_count += 1

    # Calculate adjusted win percentage
    # This normalizes the quality score to a win percentage scale
    sos_adj_win_pct = total_quality_score / games_count if games_count > 0 else 0.0
    avg_opp_power = total_opp_power / games_count if games_count > 0 else 0.0

    return {
        'sos_adj_win_pct': sos_adj_win_pct,
        'avg_opp_power': avg_opp_power,
        'quality_wins': quality_wins,
        'bad_losses': bad_losses,
        'games_played': games_count
    }


def calculate_quality_victory_margin(team_abbr: str, season: int, week: Optional[int] = None, all_team_powers: dict = None) -> dict:
    """
    Calculate quality victory margin - rewards blowout wins against tough opponents.

    For wins with margin > 14 points (2 TDs), applies quality multiplier based on opponent strength.
    Formula: (margin - 14) * (opponent_power / 50) for each qualifying win

    Returns:
        dict with quality_margin_total, quality_margin_per_game, blowout_wins
    """
    if not all_team_powers:
        return {
            'quality_margin_total': 0.0,
            'quality_margin_per_game': 0.0,
            'blowout_wins': 0
        }

    sql = """
        SELECT
            g.game_id,
            g.week,
            g.home_team_abbr,
            g.away_team_abbr,
            g.home_score,
            g.away_score
        FROM games g
        WHERE g.season = ?
        AND (g.home_team_abbr = ? OR g.away_team_abbr = ?)
    """
    params = [season, team_abbr, team_abbr]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    games = query(sql, tuple(params))

    if games.empty:
        return {
            'quality_margin_total': 0.0,
            'quality_margin_per_game': 0.0,
            'blowout_wins': 0
        }

    quality_margin_total = 0.0
    blowout_wins = 0
    games_count = len(games)

    for _, game in games.iterrows():
        is_home = game['home_team_abbr'] == team_abbr
        team_score = game['home_score'] if is_home else game['away_score']
        opp_score = game['away_score'] if is_home else game['home_score']
        opp_team = game['away_team_abbr'] if is_home else game['home_team_abbr']

        margin = team_score - opp_score

        # Only count wins with margin > 14 (2+ TDs)
        if margin > 14:
            opp_power = all_team_powers.get(opp_team, 50)  # Default to average if not found

            # Quality multiplier: (margin - 14) * (opponent_power / 50)
            # This rewards big wins against good teams, doesn't reward running up score on weak teams
            quality_multiplier = opp_power / 50
            quality_margin = (margin - 14) * quality_multiplier

            quality_margin_total += quality_margin
            blowout_wins += 1

    quality_margin_per_game = quality_margin_total / games_count if games_count > 0 else 0.0

    return {
        'quality_margin_total': quality_margin_total,
        'quality_margin_per_game': quality_margin_per_game,
        'blowout_wins': blowout_wins
    }


def calculate_road_dominance(team_abbr: str, season: int, week: Optional[int] = None) -> dict:
    """
    Calculate road dominance - performance in away games.

    Formula: (Road Win% × 0.6 + Road Pt Diff/7 × 0.4) × 10
    Combines road record and road scoring margin to reward teams that perform well on the road.

    Returns:
        dict with road_win_pct, road_pt_diff_per_game, road_score
    """
    sql = """
        SELECT
            g.game_id,
            g.home_team_abbr,
            g.away_team_abbr,
            g.home_score,
            g.away_score
        FROM games g
        WHERE g.season = ?
        AND g.away_team_abbr = ?
    """
    params = [season, team_abbr]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    road_games = query(sql, tuple(params))

    if road_games.empty:
        return {
            'road_win_pct': 0.0,
            'road_pt_diff_per_game': 0.0,
            'road_score': 0.0,
            'road_games': 0
        }

    road_wins = 0
    total_road_pt_diff = 0

    for _, game in road_games.iterrows():
        away_score = game['away_score']
        home_score = game['home_score']

        if away_score > home_score:
            road_wins += 1

        total_road_pt_diff += (away_score - home_score)

    road_games_count = len(road_games)
    road_win_pct = road_wins / road_games_count if road_games_count > 0 else 0.0
    road_pt_diff_per_game = total_road_pt_diff / road_games_count if road_games_count > 0 else 0.0

    # Calculate road dominance score: (Win% × 0.6 + Pt Diff/7 × 0.4) × 10
    road_score = (road_win_pct * 0.6 + (road_pt_diff_per_game / 7) * 0.4) * 10

    return {
        'road_win_pct': road_win_pct,
        'road_pt_diff_per_game': road_pt_diff_per_game,
        'road_score': road_score,
        'road_games': road_games_count
    }


def calculate_high_scoring_consistency(team_abbr: str, season: int, week: Optional[int] = None) -> dict:
    """
    Calculate high scoring consistency - percentage of games scoring 30+ points.

    Formula: (Games with 30+ points / Total games) × 8
    Rewards elite offenses that consistently put up points.

    Returns:
        dict with high_scoring_rate, high_scoring_score, games_30plus
    """
    sql = """
        SELECT
            g.game_id,
            g.home_team_abbr,
            g.away_team_abbr,
            g.home_score,
            g.away_score
        FROM games g
        WHERE g.season = ?
        AND (g.home_team_abbr = ? OR g.away_team_abbr = ?)
    """
    params = [season, team_abbr, team_abbr]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    games = query(sql, tuple(params))

    if games.empty:
        return {
            'high_scoring_rate': 0.0,
            'high_scoring_score': 0.0,
            'games_30plus': 0
        }

    games_30plus = 0

    for _, game in games.iterrows():
        is_home = game['home_team_abbr'] == team_abbr
        team_score = game['home_score'] if is_home else game['away_score']

        if team_score >= 30:
            games_30plus += 1

    games_count = len(games)
    high_scoring_rate = games_30plus / games_count if games_count > 0 else 0.0
    high_scoring_score = high_scoring_rate * 8

    return {
        'high_scoring_rate': high_scoring_rate,
        'high_scoring_score': high_scoring_score,
        'games_30plus': games_30plus
    }


def calculate_recent_form(team_abbr: str, season: int, week: Optional[int] = None) -> dict:
    """
    Calculate recent form - weighted performance of last 3 games.

    Formula: ((margin_1 × 0.5 + margin_2 × 0.3 + margin_3 × 0.2) / 7) × 10
    Weights: Last game (50%), 2 games ago (30%), 3 games ago (20%)
    Captures momentum and recent trajectory.

    Returns:
        dict with recent_form_score, weighted_margin, last_3_margins
    """
    sql = """
        SELECT
            g.game_id,
            g.week,
            g.home_team_abbr,
            g.away_team_abbr,
            g.home_score,
            g.away_score
        FROM games g
        WHERE g.season = ?
        AND (g.home_team_abbr = ? OR g.away_team_abbr = ?)
    """
    params = [season, team_abbr, team_abbr]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    sql += " ORDER BY g.week DESC LIMIT 3"

    recent_games = query(sql, tuple(params))

    if recent_games.empty or len(recent_games) < 3:
        return {
            'recent_form_score': 0.0,
            'weighted_margin': 0.0,
            'last_3_margins': [],
            'games_available': len(recent_games) if not recent_games.empty else 0
        }

    margins = []
    weights = [0.5, 0.3, 0.2]  # Most recent to oldest

    for _, game in recent_games.iterrows():
        is_home = game['home_team_abbr'] == team_abbr
        team_score = game['home_score'] if is_home else game['away_score']
        opp_score = game['away_score'] if is_home else game['home_score']
        margin = team_score - opp_score
        margins.append(margin)

    # Calculate weighted margin
    weighted_margin = sum(m * w for m, w in zip(margins, weights))

    # Calculate recent form score: (weighted_margin / 7) × 10
    recent_form_score = (weighted_margin / 7) * 10

    return {
        'recent_form_score': recent_form_score,
        'weighted_margin': weighted_margin,
        'last_3_margins': margins,
        'games_available': 3
    }


# ============================================================================
# Matchup Difficulty Calculations
# ============================================================================

def calculate_defensive_ranking(team_abbr: str, season: int, week: Optional[int], stat_type: str = 'rush') -> dict:
    """
    Calculate a team's defensive ranking for matchup difficulty.

    Args:
        team_abbr: Team abbreviation
        season: Season year
        week: Week number (None for full season)
        stat_type: 'rush' for rush defense, 'pass' for pass defense

    Returns:
        dict with ranking, yards_allowed_per_game, total_teams
    """
    # Get all teams' defensive stats
    if stat_type == 'rush':
        sql = """
            SELECT
                CASE
                    WHEN g.home_team_abbr = ? THEN g.away_team_abbr
                    ELSE g.home_team_abbr
                END as defense_team,
                SUM(CASE
                    WHEN g.home_team_abbr = ? THEN g.away_rushing_yds
                    ELSE g.home_rushing_yds
                END) as yards_allowed,
                COUNT(*) as games
            FROM games g
            WHERE g.season = ?
        """
        params = [team_abbr, team_abbr, season]
        if week:
            sql += " AND g.week <= ?"
            params.append(week)
        sql += " GROUP BY defense_team"

    else:  # pass defense
        sql = """
            SELECT
                CASE
                    WHEN g.home_team_abbr = ? THEN g.away_team_abbr
                    ELSE g.home_team_abbr
                END as defense_team,
                SUM(CASE
                    WHEN g.home_team_abbr = ? THEN g.away_passing_yds
                    ELSE g.home_passing_yds
                END) as yards_allowed,
                COUNT(*) as games
            FROM games g
            WHERE g.season = ?
        """
        params = [team_abbr, team_abbr, season]
        if week:
            sql += " AND g.week <= ?"
            params.append(week)
        sql += " GROUP BY defense_team"

    # Get league-wide defensive stats
    league_sql = """
        SELECT
            defense_team,
            SUM(yards_allowed) as yards_allowed,
            SUM(games) as games
        FROM (
            SELECT
                g.home_team_abbr as defense_team,
                SUM(g.away_{}_yds) as yards_allowed,
                COUNT(*) as games
            FROM games g
            WHERE g.season = ?
    """.format('rushing' if stat_type == 'rush' else 'passing')

    league_params = [season]
    if week:
        league_sql += " AND g.week <= ?"
        league_params.append(week)

    league_sql += """
            GROUP BY g.home_team_abbr
            UNION ALL
            SELECT
                g.away_team_abbr as defense_team,
                SUM(g.home_{}_yds) as yards_allowed,
                COUNT(*) as games
            FROM games g
            WHERE g.season = ?
    """.format('rushing' if stat_type == 'rush' else 'passing')

    league_params.append(season)
    if week:
        league_sql += " AND g.week <= ?"
        league_params.append(week)

    league_sql += """
        ) combined
        GROUP BY defense_team
        ORDER BY yards_allowed ASC
    """

    league_def_stats = query(league_sql, tuple(league_params))

    if league_def_stats.empty:
        return {'ranking': 16, 'yards_allowed_per_game': 0, 'total_teams': 32}

    # Calculate yards per game for each team
    league_def_stats['ypg'] = league_def_stats['yards_allowed'] / league_def_stats['games']
    league_def_stats = league_def_stats.sort_values('ypg')
    league_def_stats['rank'] = range(1, len(league_def_stats) + 1)

    # Find this team's ranking
    team_stats = league_def_stats[league_def_stats['defense_team'] == team_abbr]

    if team_stats.empty:
        return {
            'ranking': len(league_def_stats) // 2,
            'yards_allowed_per_game': 0,
            'total_teams': len(league_def_stats)
        }

    return {
        'ranking': int(team_stats.iloc[0]['rank']),
        'yards_allowed_per_game': float(team_stats.iloc[0]['ypg']),
        'total_teams': len(league_def_stats)
    }


def get_matchup_multiplier(opponent_rank: int, total_teams: int) -> dict:
    """
    Convert defensive ranking to matchup difficulty multiplier.

    Args:
        opponent_rank: Opponent's defensive rank (1 = best defense, 32 = worst)
        total_teams: Total number of teams in league

    Returns:
        dict with multiplier, difficulty_label, color_code
    """
    # Calculate percentile (1 = best defense, 100 = worst defense)
    percentile = (opponent_rank / total_teams) * 100

    # Curved scale for more granular difficulty
    if percentile <= 6.25:  # Top 2 teams (rank 1-2 out of 32)
        multiplier = 0.70
        label = "Elite Defense"
        color = "🔴"
    elif percentile <= 12.5:  # Top 4 teams (rank 3-4)
        multiplier = 0.80
        label = "Strong Defense"
        color = "🔴"
    elif percentile <= 18.75:  # Top 6 teams (rank 5-6)
        multiplier = 0.90
        label = "Above Avg Defense"
        color = "🟠"
    elif percentile <= 81.25:  # Middle teams (rank 7-26)
        # Linear scale from 0.95x to 1.05x
        mid_range = (percentile - 18.75) / (81.25 - 18.75)  # 0 to 1
        multiplier = 0.95 + (mid_range * 0.10)  # 0.95 to 1.05
        label = "Average Defense"
        color = "🟡"
    elif percentile <= 87.5:  # Bottom 6 teams (rank 27-28)
        multiplier = 1.10
        label = "Below Avg Defense"
        color = "🟢"
    elif percentile <= 93.75:  # Bottom 4 teams (rank 29-30)
        multiplier = 1.20
        label = "Weak Defense"
        color = "🟢"
    else:  # Bottom 2 teams (rank 31-32)
        multiplier = 1.30
        label = "Worst Defense"
        color = "🟢"

    return {
        'multiplier': round(multiplier, 2),
        'difficulty_label': label,
        'color_code': color
    }


# ============================================================================
# Z-Score Normalization and League Statistics
# ============================================================================

def z_score(value: float, mean: float, std: float, cap: float = 3.0) -> float:
    """
    Calculate z-score (standard deviations from mean) with optional capping.

    Formula: (value - mean) / std
    Cap at ±3.0 standard deviations to prevent extreme outliers from dominating.
    Returns 0 if std is 0 (all values identical).
    """
    if std == 0:
        return 0.0

    z = (value - mean) / std

    # Cap at ±3.0 to prevent extreme outliers (99.7% of data falls within ±3 SD)
    if cap:
        z = max(-cap, min(cap, z))

    return z


def calculate_defensive_success_rate(team_abbr: str, season: int, week: Optional[int] = None) -> float:
    """
    Calculate defensive success rate by looking at opponents' yards per play against this team.
    Lower opponent YPP = better defense = higher defensive success rate.
    """
    sql = """
        SELECT
            AVG(CASE
                WHEN g.home_team_abbr = ? THEN tgs.yards_per_play
                WHEN g.away_team_abbr = ? THEN tgs.yards_per_play
                ELSE NULL
            END) as opp_ypp
        FROM team_game_summary tgs
        JOIN games g ON tgs.game_id = g.game_id
        WHERE g.season = ?
        AND (
            (g.home_team_abbr = ? AND tgs.team_abbr = g.away_team_abbr)
            OR
            (g.away_team_abbr = ? AND tgs.team_abbr = g.home_team_abbr)
        )
    """
    params = [team_abbr, team_abbr, season, team_abbr, team_abbr]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    stats = query(sql, tuple(params))

    if stats.empty or stats['opp_ypp'].iloc[0] is None:
        return 0.5  # Default to league average

    opp_ypp = stats['opp_ypp'].iloc[0]

    # Convert to defensive success rate using inverted logistic
    # Lower opponent YPP = higher defensive success rate
    import math
    center = 5.4  # League average YPP
    steepness = 1.5

    try:
        # Invert the calculation: good defense allows low YPP
        def_success = 1 / (1 + math.exp(steepness * (opp_ypp - center) / center))
        return def_success
    except (OverflowError, ZeroDivisionError):
        return 0.5


def calculate_defensive_explosive_rate(team_abbr: str, season: int, week: Optional[int] = None) -> float:
    """
    Calculate defensive explosive play rate by looking at opponents' explosive plays against this team.
    Lower opponent explosive rate = better defense.
    """
    # Get opponent big plays against this team
    sql = """
        SELECT
            SUM(CASE WHEN pbs.rec > 0 AND pbs.rec_yds >= 40 THEN 1 ELSE 0 END) as opp_big_pass,
            SUM(CASE WHEN pbs.rush_att > 0 AND pbs.rush_yds >= 40 THEN 1 ELSE 0 END) as opp_big_rush,
            COUNT(DISTINCT g.game_id) as games
        FROM player_box_score pbs
        JOIN games g ON pbs.game_id = g.game_id
        WHERE g.season = ?
        AND (
            (g.home_team_abbr = ? AND pbs.team = g.away_team_abbr)
            OR
            (g.away_team_abbr = ? AND pbs.team = g.home_team_abbr)
        )
    """
    params = [season, team_abbr, team_abbr]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    stats = query(sql, tuple(params))

    if stats.empty:
        return 0.05  # League average defensive explosive rate

    opp_big_pass = stats['opp_big_pass'].iloc[0] or 0
    opp_big_rush = stats['opp_big_rush'].iloc[0] or 0
    games = stats['games'].iloc[0] or 1

    # Calculate opponent explosive rate allowed
    opp_explosive_per_game = (opp_big_pass + opp_big_rush) / games
    estimated_plays = 65
    opp_explosive_rate = opp_explosive_per_game / estimated_plays

    return opp_explosive_rate


def calculate_league_statistics(season: int, week: Optional[int] = None, all_teams: list = None) -> dict:
    """
    Calculate league-wide mean and standard deviation for all power rating components.
    Used for z-score normalization.

    Returns dict with mean and std for each metric.
    """
    if not all_teams:
        # Get all teams for this season
        teams_query = f"SELECT DISTINCT home_team_abbr as team FROM games WHERE season={season} ORDER BY home_team_abbr"
        teams_df = query(teams_query)
        all_teams = teams_df['team'].tolist() if not teams_df.empty else []

    if not all_teams:
        return {}

    # Collect metrics for all teams
    metrics = {
        'win_pct': [],
        'net_epa': [],
        'sr_diff': [],
        'xpl_diff': [],
        'quality_margin': [],
        'road_dom': [],
        'high_scoring': [],
        'recent_form': []
    }

    for team in all_teams:
        try:
            # Win percentage (will be SOS-adjusted later in second pass)
            team_record = calculate_win_loss_record(team, season, week)
            metrics['win_pct'].append(team_record.get('win_pct', 0))

            # EPA
            team_epa = calculate_team_epa(team, season, week)
            net_epa = team_epa.get('off_epa_per_play', 0) - team_epa.get('def_epa_per_play', 0)
            metrics['net_epa'].append(net_epa)

            # Success Rate Diff
            off_sr = calculate_success_rates(team, season, week).get('overall', 0)
            def_sr = calculate_defensive_success_rate(team, season, week)
            metrics['sr_diff'].append(off_sr - def_sr)

            # Explosive Rate Diff
            off_xpl = calculate_explosive_plays(team, season, week).get('explosive_rate', 0)
            def_xpl = calculate_defensive_explosive_rate(team, season, week)
            metrics['xpl_diff'].append(off_xpl - def_xpl)

            # Other components
            metrics['quality_margin'].append(0)  # Calculated in second pass with all_team_powers
            team_road = calculate_road_dominance(team, season, week)
            metrics['road_dom'].append(team_road.get('road_score', 0))
            team_high = calculate_high_scoring_consistency(team, season, week)
            metrics['high_scoring'].append(team_high.get('high_scoring_score', 0))
            team_form = calculate_recent_form(team, season, week)
            metrics['recent_form'].append(team_form.get('recent_form_score', 0))
        except:
            # Skip teams with incomplete data
            continue

    # Calculate mean and std for each metric
    import statistics
    league_stats = {}

    for metric_name, values in metrics.items():
        if len(values) > 1:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 1.0

            # CRITICAL FIX: Ensure minimum std to prevent extreme z-scores
            # If std is too small, z-scores become unreasonably large
            min_std = {
                'net_epa': 0.05,      # Minimum std for EPA
                'sr_diff': 0.05,      # Minimum std for success rate
                'xpl_diff': 0.01,     # Minimum std for explosive rate
                'win_pct': 0.1,       # Minimum std for win%
                'quality_margin': 0.5,
                'road_dom': 1.0,
                'high_scoring': 0.5,
                'recent_form': 2.0
            }

            if metric_name in min_std:
                std_val = max(std_val, min_std[metric_name])

            league_stats[metric_name] = {
                'mean': mean_val,
                'std': std_val
            }
        else:
            league_stats[metric_name] = {'mean': 0.0, 'std': 1.0}

    return league_stats


def calculate_team_power_rating(team_abbr: str, season: int, week: Optional[int] = None,
                                all_team_powers: dict = None, league_stats: dict = None) -> float:
    """
    Calculate a team's power rating for a specific week using z-score normalized components.

    WINS DOMINANT 2025 FORMULA:
    Power Rating = [z(SOS-Adj Win%) × 0.55 + z(Quality Margin) × 0.15 + z(Net EPA) × 0.20 +
                   z(SR Diff) × 0.05 + z(Xpl Diff) × 0.03 + z(Road Dom) × 0.01 + z(High Scoring) × 0.01] × 10

    Key philosophy:
    - WINS ARE EVERYTHING (70% combined): Win% 55% + Quality Wins 15%
    - SOS-Adjusted Win% at 55% makes record the dominant factor
    - Quality Victory Margin at 15% heavily rewards beating good teams badly
    - Net EPA reduced to 20% - efficiency is secondary to winning
    - Explosive plays reduced to 3% (minimal impact, prevents over-weighting volatility)
    - Road/Scoring reduced to 1% each (tiebreakers only)
    - Z-score capping at ±3.0 to prevent extreme outliers
    - Minimum std thresholds to prevent statistical anomalies
    - Weights sum to 1.0, then scaled by 10 for 0-100 range

    Result: Teams with better records and quality wins DOMINATE rankings.

    Returns the final power rating score.
    """
    # Get team stats
    team_record = calculate_win_loss_record(team_abbr, season, week)
    team_epa = calculate_team_epa(team_abbr, season, week)
    team_quality_margin = calculate_quality_victory_margin(team_abbr, season, week, all_team_powers)
    team_road_dom = calculate_road_dominance(team_abbr, season, week)
    team_high_scoring = calculate_high_scoring_consistency(team_abbr, season, week)
    team_recent_form = calculate_recent_form(team_abbr, season, week)

    # Calculate differential metrics (Off - Def)
    off_sr = calculate_success_rates(team_abbr, season, week).get('overall', 0)
    def_sr = calculate_defensive_success_rate(team_abbr, season, week)
    sr_diff = off_sr - def_sr

    off_xpl = calculate_explosive_plays(team_abbr, season, week).get('explosive_rate', 0)
    def_xpl = calculate_defensive_explosive_rate(team_abbr, season, week)
    xpl_diff = off_xpl - def_xpl

    net_epa = team_epa.get('off_epa_per_play', 0) - team_epa.get('def_epa_per_play', 0)

    # Calculate SOS adjustment if all_team_powers provided
    if all_team_powers:
        team_sos = calculate_sos_adjusted_record(team_abbr, season, week, all_team_powers)
        win_pct = team_sos.get('sos_adj_win_pct', team_record.get('win_pct', 0))
    else:
        win_pct = team_record.get('win_pct', 0)

    # If league stats provided, use z-score normalization
    if league_stats:
        z_win_pct = z_score(win_pct,
                           league_stats.get('win_pct', {}).get('mean', 0.5),
                           league_stats.get('win_pct', {}).get('std', 0.2))
        z_net_epa = z_score(net_epa,
                           league_stats.get('net_epa', {}).get('mean', 0),
                           league_stats.get('net_epa', {}).get('std', 0.1))
        z_sr_diff = z_score(sr_diff,
                           league_stats.get('sr_diff', {}).get('mean', 0),
                           league_stats.get('sr_diff', {}).get('std', 0.1))
        z_xpl_diff = z_score(xpl_diff,
                            league_stats.get('xpl_diff', {}).get('mean', 0),
                            league_stats.get('xpl_diff', {}).get('std', 0.02))
        z_quality = z_score(team_quality_margin.get('quality_margin_per_game', 0),
                           league_stats.get('quality_margin', {}).get('mean', 0),
                           league_stats.get('quality_margin', {}).get('std', 1))
        z_road = z_score(team_road_dom.get('road_score', 0),
                        league_stats.get('road_dom', {}).get('mean', 0),
                        league_stats.get('road_dom', {}).get('std', 2))
        z_high_scoring = z_score(team_high_scoring.get('high_scoring_score', 0),
                                league_stats.get('high_scoring', {}).get('mean', 0),
                                league_stats.get('high_scoring', {}).get('std', 1))
        z_recent = z_score(team_recent_form.get('recent_form_score', 0),
                          league_stats.get('recent_form', {}).get('mean', 0),
                          league_stats.get('recent_form', {}).get('std', 5))

        # Calculate power rating with z-scored components
        # Weights sum to 1.0 for interpretability
        # WINS DOMINANT 2025: Wins and quality wins are EVERYTHING (70% combined)
        power = (
            z_win_pct * 0.55 +          # 55%: Win-loss record (SOS-adjusted) - ABSOLUTELY DOMINANT
            z_quality * 0.15 +          # 15%: Quality victory margin - beating good teams badly matters
            z_net_epa * 0.20 +          # 20%: Net EPA (efficiency) - tertiary importance
            z_sr_diff * 0.05 +          # 5%: Success rate differential - minor factor
            z_xpl_diff * 0.03 +         # 3%: Explosive play differential - minimal impact
            z_road * 0.01 +             # 1%: Road dominance - tiebreaker only
            z_high_scoring * 0.01       # 1%: High scoring consistency - tiebreaker only
        )

        # Scale to ~0-100 range for interpretability (multiply by 10)
        power_scaled = power * 10
    else:
        # Fallback: raw values without z-scoring (for backwards compatibility)
        # Use same proportional weights as z-scored version
        power = (
            win_pct * 0.55 +
            team_quality_margin.get('quality_margin_per_game', 0) * 0.15 +
            net_epa * 0.20 +
            sr_diff * 0.05 +
            xpl_diff * 0.03 +
            team_road_dom.get('road_score', 0) * 0.01 +
            team_high_scoring.get('high_scoring_score', 0) * 0.01
        )
        power_scaled = power * 10

    return power_scaled


def calculate_point_differential(team_abbr: str, season: int, week: Optional[int] = None) -> dict:
    """Calculate team's point differential (points scored - points allowed)."""
    sql = """
        SELECT
            g.game_id,
            g.home_team_abbr,
            g.away_team_abbr,
            g.home_score,
            g.away_score
        FROM games g
        WHERE g.season = ?
        AND (g.home_team_abbr = ? OR g.away_team_abbr = ?)
    """
    params = [season, team_abbr, team_abbr]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    games = query(sql, tuple(params))

    if games.empty:
        return {'pts_for': 0.0, 'pts_against': 0.0, 'pt_diff': 0.0, 'pt_diff_per_game': 0.0}

    total_pts_for = 0
    total_pts_against = 0

    for _, row in games.iterrows():
        is_home = row['home_team_abbr'] == team_abbr
        team_score = row['home_score'] if is_home else row['away_score']
        opp_score = row['away_score'] if is_home else row['home_score']

        total_pts_for += team_score
        total_pts_against += opp_score

    games_played = len(games)
    pt_diff = total_pts_for - total_pts_against
    pt_diff_per_game = pt_diff / games_played if games_played > 0 else 0.0

    return {
        'pts_for': float(total_pts_for),
        'pts_against': float(total_pts_against),
        'pt_diff': float(pt_diff),
        'pt_diff_per_game': pt_diff_per_game,
        'avg_pts_for': total_pts_for / games_played if games_played > 0 else 0.0,
        'avg_pts_against': total_pts_against / games_played if games_played > 0 else 0.0
    }


def calculate_team_epa(team_abbr: str, season: int, week: Optional[int] = None) -> dict:
    """
    Calculate team efficiency metrics using aggregated box score stats.
    Returns offensive and defensive efficiency per game.
    """
    # Get team's offensive stats
    sql = """
        SELECT
            AVG(pass_yds + rush_yds) as avg_total_yards,
            AVG(pass_yds) as avg_pass_yds,
            AVG(rush_yds) as avg_rush_yds,
            AVG(points) as avg_points,
            COUNT(*) as games
        FROM team_game_summary tgs
        JOIN games g ON tgs.game_id = g.game_id
        WHERE tgs.team_abbr = ?
        AND g.season = ?
    """
    params = [team_abbr, season]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    off_stats = query(sql, tuple(params))

    # Get defensive stats (points/yards allowed)
    def_sql = """
        SELECT
            AVG(CASE
                WHEN g.home_team_abbr = ? THEN g.away_score
                ELSE g.home_score
            END) as avg_points_allowed,
            COUNT(*) as games
        FROM games g
        WHERE g.season = ?
        AND (g.home_team_abbr = ? OR g.away_team_abbr = ?)
    """
    def_params = [team_abbr, season, team_abbr, team_abbr]
    if week:
        def_sql += " AND g.week <= ?"
        def_params.append(week)

    def_stats = query(def_sql, tuple(def_params))

    if off_stats.empty:
        return {
            'off_epa_per_play': 0,
            'def_epa_per_play': 0,
            'off_epa_total': 0,
            'def_epa_total': 0,
            'off_plays': 0,
            'def_plays': 0
        }

    # Calculate efficiency scores with improved scaling
    # Using std dev instead of dividing by mean for better differentiation
    avg_yards = off_stats['avg_total_yards'].iloc[0] or 0
    avg_points = off_stats['avg_points'].iloc[0] or 0
    avg_pts_allowed = def_stats['avg_points_allowed'].iloc[0] if not def_stats.empty else 23
    games = off_stats['games'].iloc[0] or 1

    # League averages and standard deviations
    league_avg_yards = 340
    league_std_yards = 50  # ~50 yard std dev per game
    league_avg_points = 22
    league_std_points = 7  # ~7 point std dev per game

    # Calculate z-scores (standard deviations from mean)
    yards_z = (avg_yards - league_avg_yards) / league_std_yards
    points_z = (avg_points - league_avg_points) / league_std_points
    def_points_z = (avg_pts_allowed - league_avg_points) / league_std_points

    # Offensive EPA: combine yards and points efficiency
    # Scale to typical EPA range: -0.2 to +0.2 per play
    off_epa_per_play = ((yards_z + points_z) / 2) * 0.1

    # Defensive EPA: negative is better (fewer points allowed)
    # Scale to typical EPA range
    def_epa_per_play = -def_points_z * 0.1

    return {
        'off_epa_per_play': off_epa_per_play,
        'def_epa_per_play': def_epa_per_play,
        'off_epa_total': off_epa_per_play * 65 * games,  # ~65 plays/game
        'def_epa_total': def_epa_per_play * 65 * games,
        'off_plays': int(games * 65),
        'def_plays': int(games * 65),
        'avg_yards': avg_yards,
        'avg_points': avg_points,
        'avg_pts_allowed': avg_pts_allowed
    }


def calculate_success_rates(team_abbr: str, season: int, week: Optional[int] = None) -> dict:
    """Calculate success rate proxies using efficiency stats."""
    sql = """
        SELECT
            AVG(CAST(pass_yds AS FLOAT) / NULLIF(pass_comp, 0)) as yards_per_comp,
            AVG(CAST(rush_yds AS FLOAT) / NULLIF(rush_att, 0)) as yards_per_carry,
            AVG(CAST(pass_comp AS FLOAT) / NULLIF(pass_att, 0)) as completion_pct,
            AVG(yards_per_play) as avg_ypp,
            COUNT(*) as games
        FROM team_game_summary tgs
        JOIN games g ON tgs.game_id = g.game_id
        WHERE tgs.team_abbr = ?
        AND g.season = ?
    """
    params = [team_abbr, season]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    stats = query(sql, tuple(params))

    if stats.empty:
        return {}

    ypc = stats['yards_per_carry'].iloc[0] or 0
    ypr = stats['yards_per_comp'].iloc[0] or 0
    comp_pct = stats['completion_pct'].iloc[0] or 0
    ypp = stats['avg_ypp'].iloc[0] or 0

    # Calculate success rate using logistic curve for realistic 40-60% range
    # Uses formula: 1 / (1 + e^(-k*(x - x0))) where x0 is center, k controls steepness
    import math

    def logistic_success(value: float, center: float, steepness: float = 2.0) -> float:
        """Convert metric to success rate using logistic function."""
        if value == 0:
            return 0.0
        try:
            return 1 / (1 + math.exp(-steepness * (value - center) / center))
        except (OverflowError, ZeroDivisionError):
            return 0.5

    # Calculate success rates using logistic curves
    # Centers are set at "average" values, will produce ~40-60% for most teams
    rush_success = logistic_success(ypc, 4.3, 1.5)  # 4.3 YPC is average
    pass_success = logistic_success(ypr, 11.5, 1.5)  # 11.5 YPR is average
    overall_success = logistic_success(ypp, 5.4, 1.5)  # 5.4 YPP is average
    third_down_proxy = logistic_success(comp_pct, 0.65, 3.0)  # 65% completion is average

    return {
        'overall': overall_success,
        'by_down': {1: rush_success, 2: pass_success, 3: third_down_proxy},
        'early_down': (rush_success + pass_success) / 2,
        'passing_down': third_down_proxy,
        'red_zone': overall_success * 0.90,  # Estimate red zone as slightly lower
        'total_plays': 0,
        'ypp': ypp,
        'ypc': ypc,
        'ypr': ypr
    }


def calculate_explosive_plays(team_abbr: str, season: int, week: Optional[int] = None) -> dict:
    """Calculate explosive play proxies from player box scores (20+ yard plays)."""
    # Get player stats to find big plays using actual column names
    sql = """
        SELECT
            SUM(CASE WHEN rec > 0 AND rec_yds >= 40 THEN 1 ELSE 0 END) as big_pass_plays,
            SUM(CASE WHEN rush_att > 0 AND rush_yds >= 40 THEN 1 ELSE 0 END) as big_rush_plays,
            AVG(rec_yds) as avg_rec_yds,
            AVG(rush_yds) as avg_rush_yds,
            COUNT(DISTINCT pbs.game_id) as games
        FROM player_box_score pbs
        JOIN games g ON pbs.game_id = g.game_id
        WHERE pbs.team = ?
        AND g.season = ?
    """
    params = [team_abbr, season]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    stats = query(sql, tuple(params))

    if stats.empty:
        return {}

    big_pass = stats['big_pass_plays'].iloc[0] or 0
    big_rush = stats['big_rush_plays'].iloc[0] or 0
    games = stats['games'].iloc[0] or 1

    # Estimate explosive play rate (big plays per game / estimated plays per game)
    explosive_per_game = (big_pass + big_rush) / games
    estimated_plays = 65  # avg plays per game
    explosive_rate = explosive_per_game / estimated_plays

    return {
        'explosive_rate': explosive_rate,
        'explosive_pass_rate': (big_pass / games) / 35,  # ~35 pass attempts per game
        'explosive_rush_rate': (big_rush / games) / 30,  # ~30 rush attempts per game
        'total_explosive': int(big_pass + big_rush),
        'avg_explosive_yards': 45  # Estimate for 40+ yard plays
    }


def get_team_defensive_stats(team_abbr: str, season: int, week: Optional[int] = None) -> dict:
    """Get aggregated defensive stats for a team from advanced_stats_raw."""
    sql = """
        SELECT
            g.game_id,
            asr.row_json
        FROM advanced_stats_raw asr
        JOIN games g ON asr.game_id = g.game_id
        WHERE asr.stat_type = 'advanced_defense'
        AND g.season = ?
    """
    params = [season]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    rows = query(sql, tuple(params))

    if rows.empty:
        return {
            'total_tackles': 0, 'tackles_per_game': 0, 'missed_tackles': 0, 'missed_tackle_pct': 0,
            'total_sacks': 0, 'sacks_per_game': 0, 'total_ints': 0, 'ints_per_game': 0,
            'total_pressures': 0, 'pressures_per_game': 0, 'total_qb_hits': 0, 'qb_hits_per_game': 0,
            'pass_rating_allowed': 0, 'completion_pct_allowed': 0, 'yards_per_target': 0,
            'games': 0
        }

    # Parse JSON and aggregate by game_id and team
    import json
    game_stats = {}

    for _, row in rows.iterrows():
        data = json.loads(row['row_json'])
        game_id = row['game_id']
        team = data.get('team', '')

        if team != team_abbr:
            continue

        if game_id not in game_stats:
            game_stats[game_id] = {
                'tackles': 0, 'missed_tackles': 0, 'sacks': 0, 'ints': 0,
                'pressures': 0, 'qb_hits': 0, 'hurries': 0,
                'targets': 0, 'completions': 0, 'comp_yds': 0, 'pass_tds': 0
            }

        game_stats[game_id]['tackles'] += float(data.get('tackles_combined', 0) or 0)
        game_stats[game_id]['missed_tackles'] += float(data.get('tackles_missed', 0) or 0)
        game_stats[game_id]['sacks'] += float(data.get('sacks', 0) or 0)
        game_stats[game_id]['ints'] += float(data.get('def_int', 0) or 0)
        game_stats[game_id]['pressures'] += float(data.get('pressures', 0) or 0)
        game_stats[game_id]['qb_hits'] += float(data.get('qb_knockdown', 0) or 0)
        game_stats[game_id]['hurries'] += float(data.get('qb_hurry', 0) or 0)
        game_stats[game_id]['targets'] += float(data.get('def_targets', 0) or 0)
        game_stats[game_id]['completions'] += float(data.get('def_cmp', 0) or 0)
        game_stats[game_id]['comp_yds'] += float(data.get('def_cmp_yds', 0) or 0)
        game_stats[game_id]['pass_tds'] += float(data.get('def_cmp_td', 0) or 0)

    if not game_stats:
        return {
            'total_tackles': 0, 'tackles_per_game': 0, 'missed_tackles': 0, 'missed_tackle_pct': 0,
            'total_sacks': 0, 'sacks_per_game': 0, 'total_ints': 0, 'ints_per_game': 0,
            'total_pressures': 0, 'pressures_per_game': 0, 'total_qb_hits': 0, 'qb_hits_per_game': 0,
            'pass_rating_allowed': 0, 'completion_pct_allowed': 0, 'yards_per_target': 0,
            'games': 0
        }

    games = len(game_stats)
    total_tackles = sum(g['tackles'] for g in game_stats.values())
    total_missed = sum(g['missed_tackles'] for g in game_stats.values())
    total_sacks = sum(g['sacks'] for g in game_stats.values())
    total_ints = sum(g['ints'] for g in game_stats.values())
    total_pressures = sum(g['pressures'] for g in game_stats.values())
    total_qb_hits = sum(g['qb_hits'] for g in game_stats.values())
    total_targets = sum(g['targets'] for g in game_stats.values())
    total_completions = sum(g['completions'] for g in game_stats.values())
    total_comp_yds = sum(g['comp_yds'] for g in game_stats.values())
    total_pass_tds = sum(g['pass_tds'] for g in game_stats.values())

    # Calculate pass rating allowed (simplified)
    comp_pct = (total_completions / total_targets * 100) if total_targets > 0 else 0
    yds_per_att = (total_comp_yds / total_targets) if total_targets > 0 else 0
    td_pct = (total_pass_tds / total_targets * 100) if total_targets > 0 else 0

    # Simplified passer rating formula
    c = max(0, min(2.375, (comp_pct - 30) / 20))
    y = max(0, min(2.375, (yds_per_att - 3) / 4))
    t = max(0, min(2.375, td_pct / 5))
    i = max(0, min(2.375, 2.375 - (0 / 25)))  # No INT data per target
    pass_rating = ((c + y + t + i) / 6) * 100

    return {
        'total_tackles': total_tackles,
        'tackles_per_game': total_tackles / games if games > 0 else 0,
        'missed_tackles': total_missed,
        'missed_tackle_pct': (total_missed / (total_tackles + total_missed) * 100) if (total_tackles + total_missed) > 0 else 0,
        'total_sacks': total_sacks,
        'sacks_per_game': total_sacks / games if games > 0 else 0,
        'total_ints': total_ints,
        'ints_per_game': total_ints / games if games > 0 else 0,
        'total_pressures': total_pressures,
        'pressures_per_game': total_pressures / games if games > 0 else 0,
        'total_qb_hits': total_qb_hits,
        'qb_hits_per_game': total_qb_hits / games if games > 0 else 0,
        'pass_rating_allowed': pass_rating,
        'completion_pct_allowed': comp_pct,
        'yards_per_target': yds_per_att,
        'games': games
    }


def get_defensive_leaders(team_abbr: str, season: int, week: Optional[int] = None, stat_type: str = 'tackles') -> pd.DataFrame:
    """Get top defensive players for a team."""
    sql = """
        SELECT
            g.game_id,
            g.week,
            asr.row_json
        FROM advanced_stats_raw asr
        JOIN games g ON asr.game_id = g.game_id
        WHERE asr.stat_type = 'advanced_defense'
        AND g.season = ?
    """
    params = [season]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    rows = query(sql, tuple(params))

    if rows.empty:
        return pd.DataFrame()

    import json
    player_stats = {}

    for _, row in rows.iterrows():
        data = json.loads(row['row_json'])
        team = data.get('team', '')

        if team != team_abbr:
            continue

        player = data.get('player', '')
        if not player or player == 'Team Total':
            continue

        if player not in player_stats:
            player_stats[player] = {
                'player': player,
                'team': team,
                'tackles': 0,
                'missed_tackles': 0,
                'sacks': 0,
                'ints': 0,
                'pressures': 0,
                'qb_hits': 0,
                'hurries': 0,
                'targets': 0,
                'completions': 0,
                'comp_yds': 0,
                'games': 0
            }

        player_stats[player]['tackles'] += float(data.get('tackles_combined', 0) or 0)
        player_stats[player]['missed_tackles'] += float(data.get('tackles_missed', 0) or 0)
        player_stats[player]['sacks'] += float(data.get('sacks', 0) or 0)
        player_stats[player]['ints'] += float(data.get('def_int', 0) or 0)
        player_stats[player]['pressures'] += float(data.get('pressures', 0) or 0)
        player_stats[player]['qb_hits'] += float(data.get('qb_knockdown', 0) or 0)
        player_stats[player]['hurries'] += float(data.get('qb_hurry', 0) or 0)
        player_stats[player]['targets'] += float(data.get('def_targets', 0) or 0)
        player_stats[player]['completions'] += float(data.get('def_cmp', 0) or 0)
        player_stats[player]['comp_yds'] += float(data.get('def_cmp_yds', 0) or 0)
        player_stats[player]['games'] += 1

    df = pd.DataFrame(player_stats.values())

    if df.empty:
        return df

    # Calculate per-game stats
    df['tackles_per_game'] = df['tackles'] / df['games']
    df['sacks_per_game'] = df['sacks'] / df['games']
    df['ints_per_game'] = df['ints'] / df['games']
    df['pressures_per_game'] = df['pressures'] / df['games']
    df['total_disruptions'] = df['sacks'] + df['qb_hits'] + df['hurries']
    df['comp_pct_allowed'] = (df['completions'] / df['targets'] * 100).fillna(0)
    df['yds_per_target'] = (df['comp_yds'] / df['targets']).fillna(0)

    return df


def calculate_third_down_metrics(team_abbr: str, season: int, week: Optional[int] = None) -> dict:
    """Estimate 3rd down metrics using completion % and efficiency."""
    sql = """
        SELECT
            AVG(CAST(pass_comp AS FLOAT) / NULLIF(pass_att, 0)) as completion_pct,
            AVG(yards_per_play) as avg_ypp,
            COUNT(*) as games
        FROM team_game_summary tgs
        JOIN games g ON tgs.game_id = g.game_id
        WHERE tgs.team_abbr = ?
        AND g.season = ?
    """
    params = [team_abbr, season]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    stats = query(sql, tuple(params))

    if stats.empty:
        return {}

    comp_pct = stats['completion_pct'].iloc[0] or 0
    ypp = stats['avg_ypp'].iloc[0] or 0
    games = stats['games'].iloc[0] or 1

    # Estimate conversion rate from completion % and YPP
    # Higher completion % and YPP correlate with 3rd down conversions
    estimated_conversion = (comp_pct * 0.6 + min(1.0, ypp / 5.5) * 0.4)

    # Estimate ~12 third down attempts per game
    est_attempts_per_game = 12
    est_conversions_per_game = estimated_conversion * est_attempts_per_game

    # Estimate distance-based conversions (typical distribution)
    short_conv = min(1.0, estimated_conversion * 1.5)  # Short is easier
    medium_conv = estimated_conversion
    long_conv = estimated_conversion * 0.5  # Long is harder

    return {
        'conversion_rate': estimated_conversion,
        'short_conversion': short_conv,
        'medium_conversion': medium_conv,
        'long_conversion': long_conv,
        'total_attempts': int(est_attempts_per_game * games),
        'total_conversions': int(est_conversions_per_game * games)
    }


def calculate_drive_efficiency(team_abbr: str, season: int, week: Optional[int] = None) -> dict:
    """Calculate drive efficiency metrics using team game summary data (estimated)."""
    # Since play-by-play data doesn't have proper team attribution, use team_game_summary
    # to estimate drive efficiency metrics
    sql = """
        SELECT
            AVG(points) as avg_points,
            AVG(yards_total) as avg_yards,
            COUNT(*) as games
        FROM team_game_summary tgs
        JOIN games g ON tgs.game_id = g.game_id
        WHERE tgs.team_abbr = ?
        AND g.season = ?
    """
    params = [team_abbr, season]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    stats = query(sql, tuple(params))

    if stats.empty or stats['games'].iloc[0] == 0:
        return {
            'points_per_drive': 0,
            'red_zone_td_pct': 0,
            'red_zone_score_pct': 0,
            'drive_success_rate': 0,
            'avg_start_position': 50,
            'short_field_ppd': 0,
            'long_field_ppd': 0,
            'data_available': False
        }

    avg_points = stats['avg_points'].iloc[0] or 0
    avg_yards = stats['avg_yards'].iloc[0] or 0
    # Estimate TDs from points (assume ~70% of points come from TDs)
    avg_tds = (avg_points * 0.7) / 7  # Rough estimate

    # Estimate drives per game (~12 drives per team per game on average)
    est_drives_per_game = 12
    ppd = avg_points / est_drives_per_game if est_drives_per_game > 0 else 0

    # Estimate red zone efficiency based on TD rate and points
    # Assume ~20% of drives reach red zone, and TD rate shows RZ efficiency
    est_rz_trips = est_drives_per_game * 0.20
    rz_td_pct = min(1.0, (avg_tds / est_rz_trips) if est_rz_trips > 0 else 0)
    # Estimate score rate (TDs + FGs) - assume ~60% of RZ trips result in points
    rz_score_pct = min(1.0, rz_td_pct * 1.5)

    # Estimate drive success rate (drives ending in points)
    # Based on points scored vs theoretical maximum
    drive_success_rate = min(1.0, avg_points / 35)  # 35 points ~= very good game

    # Field position estimates (can't calculate without play-by-play)
    # Use league average of ~27 yard line (own 27)
    avg_start = 27

    # Estimate short/long field based on overall efficiency
    # Better teams likely score more on short fields
    short_field_ppd = ppd * 1.3  # ~30% boost on short fields
    long_field_ppd = ppd * 0.85  # ~15% penalty on long fields

    return {
        'points_per_drive': ppd,
        'red_zone_td_pct': rz_td_pct,
        'red_zone_score_pct': rz_score_pct,
        'drive_success_rate': drive_success_rate,
        'avg_start_position': avg_start,
        'short_field_ppd': short_field_ppd,
        'long_field_ppd': long_field_ppd,
        'data_available': True
    }


def calculate_tempo_metrics(team_abbr: str, season: int, week: Optional[int] = None) -> dict:
    """Calculate tempo and pace of play metrics (estimated from total plays if detailed data unavailable)."""
    # Try to get play count per game from team game summary (has plays column)
    sql = """
        SELECT
            AVG(plays) as avg_plays,
            COUNT(*) as games
        FROM team_game_summary tgs
        JOIN games g ON tgs.game_id = g.game_id
        WHERE tgs.team_abbr = ?
        AND g.season = ?
    """
    params = [team_abbr, season]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    stats = query(sql, tuple(params))

    if stats.empty or stats['games'].iloc[0] == 0:
        return {
            'plays_per_game': 65,
            'seconds_per_play': 30,
            'pace_rating': 50,
            'q1_pace': 16,
            'q2_pace': 16,
            'q3_pace': 16,
            'q4_pace': 16,
            'data_available': False
        }

    avg_plays = stats['avg_plays'].iloc[0] or 65

    # Estimate seconds per play (approximate from total game time)
    # Typical game: ~60 minutes of play clock time
    est_seconds_per_play = (60 * 60) / avg_plays if avg_plays > 0 else 30

    # Pace rating (0-100, higher = faster)
    # 75+ plays/game = very fast (100), 55 plays/game = very slow (0)
    pace_rating = min(100, max(0, ((avg_plays - 55) / 20) * 100))

    # Estimate quarter distribution (typically 25% per quarter)
    est_quarter_plays = avg_plays / 4

    return {
        'plays_per_game': avg_plays,
        'seconds_per_play': min(40, est_seconds_per_play),
        'pace_rating': pace_rating,
        'q1_pace': est_quarter_plays,
        'q2_pace': est_quarter_plays,
        'q3_pace': est_quarter_plays,
        'q4_pace': est_quarter_plays,
        'data_available': True
    }


def calculate_clutch_metrics(team_abbr: str, season: int, week: Optional[int] = None) -> dict:
    """Calculate clutch and momentum performance metrics."""
    # Get game results
    sql = """
        SELECT
            g.game_id,
            g.home_team_abbr,
            g.away_team_abbr,
            g.home_score,
            g.away_score,
            CASE
                WHEN g.home_team_abbr = ? THEN
                    CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END
                ELSE
                    CASE WHEN g.away_score > g.home_score THEN 1 ELSE 0 END
            END as won,
            ABS(g.home_score - g.away_score) as margin
        FROM games g
        WHERE g.season = ?
        AND (g.home_team_abbr = ? OR g.away_team_abbr = ?)
    """
    params = [team_abbr, season, team_abbr, team_abbr]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    games_data = query(sql, tuple(params))

    if games_data.empty:
        return {
            'clutch_win_rate': 0.5,
            'close_game_record': "0-0",
            'blowout_rate': 0,
            'comeback_rate': 0,
            'lead_protection_rate': 0
        }

    total_games = len(games_data)
    wins = games_data['won'].sum()

    # Close games (within 7 points)
    close_games = games_data[games_data['margin'] <= 7]
    close_wins = close_games['won'].sum()
    close_losses = len(close_games) - close_wins

    # Blowouts (won by 14+)
    blowouts = len(games_data[(games_data['won'] == 1) & (games_data['margin'] > 14)])
    blowout_rate = blowouts / total_games if total_games > 0 else 0

    # Estimate comeback/lead protection (proxy: close game performance)
    clutch_win_rate = close_wins / len(close_games) if len(close_games) > 0 else 0.5

    return {
        'clutch_win_rate': clutch_win_rate,
        'close_game_record': f"{int(close_wins)}-{int(close_losses)}",
        'blowout_rate': blowout_rate,
        'comeback_rate': clutch_win_rate * 0.6,  # Estimate
        'lead_protection_rate': clutch_win_rate * 1.2 if clutch_win_rate < 0.8 else 0.9
    }


def calculate_playcalling_tendencies(team_abbr: str, season: int, week: Optional[int] = None) -> dict:
    """Calculate play-calling tendencies using pass/rush ratios (estimated without detailed play data)."""
    # Use team_game_summary to estimate tendencies from pass/rush attempts
    sql = """
        SELECT
            AVG(pass_att) as avg_pass_att,
            AVG(rush_att) as avg_rush_att,
            AVG(pass_yds) as avg_pass_yds,
            AVG(rush_yds) as avg_rush_yds,
            COUNT(*) as games
        FROM team_game_summary tgs
        JOIN games g ON tgs.game_id = g.game_id
        WHERE tgs.team_abbr = ?
        AND g.season = ?
    """
    params = [team_abbr, season]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    stats = query(sql, tuple(params))

    if stats.empty or stats['games'].iloc[0] == 0:
        return {
            'first_down_pass_rate': 0.5,
            'second_long_pass_rate': 0.7,
            'third_long_pass_rate': 0.8,
            'predictability_score': 50,
            'early_down_aggression': 50,
            'data_available': False
        }

    avg_pass = stats['avg_pass_att'].iloc[0] or 0
    avg_rush = stats['avg_rush_att'].iloc[0] or 0
    total_plays = avg_pass + avg_rush

    # Estimate overall pass rate
    overall_pass_rate = avg_pass / total_plays if total_plays > 0 else 0.5

    # Estimate down-specific tendencies
    # First down: typically more balanced (slightly run-heavy for most teams)
    first_pass_rate = overall_pass_rate * 0.85  # ~15% less passing on 1st down

    # Second & long: more passing
    second_long_pass_rate = min(0.95, overall_pass_rate * 1.3)

    # Third & long: heavily pass
    third_long_pass_rate = min(0.98, overall_pass_rate * 1.5)

    # Predictability based on how extreme the pass/rush balance is
    # Balanced = 50/50, extreme = 70/30 or more
    balance_deviation = abs(overall_pass_rate - 0.5)
    predictability = min(100, balance_deviation * 200)  # 0 = balanced, 100 = extreme

    # Early down aggression
    aggression_score = overall_pass_rate * 100

    return {
        'first_down_pass_rate': first_pass_rate,
        'second_long_pass_rate': second_long_pass_rate,
        'third_long_pass_rate': third_long_pass_rate,
        'predictability_score': predictability,
        'early_down_aggression': aggression_score,
        'data_available': True
    }


def calculate_matchup_advantages(team1_abbr: str, team2_abbr: str, season: int, week: Optional[int] = None) -> dict:
    """Calculate strength vs weakness matchup advantages between two teams."""
    # Get offensive and defensive rankings
    teams_sql = """
        SELECT
            team_abbr,
            AVG(pass_yds) as avg_pass_yds,
            AVG(rush_yds) as avg_rush_yds,
            AVG(points) as avg_points,
            AVG(yards_total) as avg_total_yards
        FROM team_game_summary tgs
        JOIN games g ON tgs.game_id = g.game_id
        WHERE g.season = ?
    """
    params = [season]
    if week:
        teams_sql += " AND g.week <= ?"
        params.append(week)
    teams_sql += " GROUP BY team_abbr"

    all_teams = query(teams_sql, tuple(params))

    if all_teams.empty:
        return {
            'pass_matchup_advantage': 0,
            'rush_matchup_advantage': 0,
            'overall_advantage': 0,
            't1_advantages': [],
            't2_advantages': []
        }

    # Get team1 and team2 stats
    t1_stats = all_teams[all_teams['team_abbr'] == team1_abbr]
    t2_stats = all_teams[all_teams['team_abbr'] == team2_abbr]

    if t1_stats.empty or t2_stats.empty:
        return {
            'pass_matchup_advantage': 0,
            'rush_matchup_advantage': 0,
            'overall_advantage': 0,
            't1_advantages': [],
            't2_advantages': []
        }

    # Calculate percentile ranks
    t1_pass_rank = (all_teams['avg_pass_yds'] < t1_stats['avg_pass_yds'].iloc[0]).sum() / len(all_teams)
    t2_pass_rank = (all_teams['avg_pass_yds'] < t2_stats['avg_pass_yds'].iloc[0]).sum() / len(all_teams)

    t1_rush_rank = (all_teams['avg_rush_yds'] < t1_stats['avg_rush_yds'].iloc[0]).sum() / len(all_teams)
    t2_rush_rank = (all_teams['avg_rush_yds'] < t2_stats['avg_rush_yds'].iloc[0]).sum() / len(all_teams)

    # Matchup advantage (positive = team1 advantage, negative = team2 advantage)
    pass_advantage = (t1_pass_rank - t2_pass_rank) * 100
    rush_advantage = (t1_rush_rank - t2_rush_rank) * 100
    overall_advantage = (pass_advantage + rush_advantage) / 2

    # Identify specific advantages
    t1_advantages = []
    t2_advantages = []

    if pass_advantage > 10:
        t1_advantages.append("Passing Offense")
    elif pass_advantage < -10:
        t2_advantages.append("Passing Offense")

    if rush_advantage > 10:
        t1_advantages.append("Rushing Offense")
    elif rush_advantage < -10:
        t2_advantages.append("Rushing Offense")

    return {
        'pass_matchup_advantage': pass_advantage,
        'rush_matchup_advantage': rush_advantage,
        'overall_advantage': overall_advantage,
        't1_advantages': t1_advantages,
        't2_advantages': t2_advantages
    }


# ============================================================================
# UI Components
# ============================================================================

def render_sidebar() -> Tuple[str, Optional[int], Optional[int], Optional[str]]:
    """Render sidebar with filters and navigation."""
    st.sidebar.title("🏈 NFL Data Viewer")
    st.sidebar.caption(f"Database: {DB_PATH.name}")

    # Navigation
    st.sidebar.header("Navigation")
    view = st.sidebar.selectbox(
        "Select View",
        [
            "Games Browser",
            "Charts",
            "Team Overview",
            "Team Comparison",
            "Power Rankings",
            "Stats & Trends",
            "Advanced Team Analytics",
            "Matchup Predictor",
            "Upcoming Matches",
            "Skill Yards Grid",
            "Skill TDs Grid",
            "First TD Grid",
            "First TD Detail",
            "TD Against",
            "Player Stats",
            "Season Leaderboards",
            "Play-by-Play Viewer",
            "Game Detail",
            "Notes Manager",
            "Transaction Manager"
        ],
        index=0
    )

    st.sidebar.divider()

    # Filters
    st.sidebar.header("Filters")

    seasons = get_seasons()
    season = st.sidebar.selectbox(
        "Season",
        seasons,
        index=0 if seasons else 0
    ) if seasons else None

    weeks = get_weeks(season) if season else []
    week = st.sidebar.selectbox(
        "Week",
        ["All"] + weeks,
        index=0
    ) if weeks else None
    week = None if week == "All" else week

    teams = get_teams(season)
    team = st.sidebar.selectbox(
        "Team",
        ["All"] + teams,
        index=0
    ) if teams else None
    team = None if team == "All" else team

    # Quick Notes section
    st.sidebar.divider()
    st.sidebar.header("Quick Notes")

    note_text = st.sidebar.text_area(
        "Add a note",
        placeholder="Use #TeamAbbr or #CustomTag to tag your notes...",
        height=100,
        key="quick_note_input"
    )

    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        if st.button("Save Note", key="save_quick_note", use_container_width=True):
            if note_text.strip():
                note_id = save_note(note_text, season=season, week=week)
                if note_id:
                    st.success("Note saved!")
                    st.rerun()
            else:
                st.warning("Please enter a note")

    with col2:
        if st.button("View All", key="view_all_notes", use_container_width=True):
            st.session_state['view_override'] = 'Notes Manager'
            st.rerun()

    # Show recent notes count
    recent_notes = get_notes()
    if not recent_notes.empty:
        st.sidebar.caption(f"Total notes: {len(recent_notes)}")

    # Database Status Panel
    st.sidebar.divider()
    st.sidebar.header("Database Status")

    # Add refresh button and sync button
    col1, col2, col3 = st.sidebar.columns([2, 1, 1])
    with col2:
        refresh_stats = st.button("🔄", key="refresh_db_stats", help="Refresh database stats")
    with col3:
        sync_from_cloud = st.button("☁️", key="sync_from_cloud", help="Download latest database from GCS")

    # Handle sync from cloud
    if sync_from_cloud:
        if GCS_BUCKET_NAME:
            with st.spinner("Downloading latest database from cloud..."):
                # Delete local database to force fresh download
                if DB_PATH.exists():
                    DB_PATH.unlink()

                # Download from GCS
                if download_db_from_gcs():
                    st.success("✅ Database updated from cloud!")
                    st.cache_data.clear()  # Clear all cached queries
                    time.sleep(1)
                    st.rerun()  # Reload the app with new database
                else:
                    st.error("❌ Failed to download database from cloud")
        else:
            st.warning("Cloud storage not configured")

    # Get database stats (refresh if button clicked)
    if refresh_stats or 'db_stats' not in st.session_state:
        st.session_state.db_stats = get_database_stats()

    stats = st.session_state.db_stats

    # Display stats
    if 'error' in stats:
        st.sidebar.error(f"Error loading stats: {stats['error']}")
    else:
        # Record counts with icons
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Injuries", stats['injury_count'])
        with col2:
            st.metric("Games", stats['game_count'])

        # Last modified time
        if stats['last_modified']:
            st.sidebar.caption(f"Last modified: {stats['last_modified'].strftime('%Y-%m-%d %H:%M:%S')}")

        # Database size
        st.sidebar.caption(f"DB Size: {stats['db_size_mb']:.2f} MB")

    return view, season, week, team


# ============================================================================
# Section: Games Browser
# ============================================================================

def render_games_browser(season: Optional[int], week: Optional[int]):
    """Display list of games with filters."""
    st.header("🏈 Games Browser")

    if not season:
        st.warning("No season data available.")
        return

    # Build query
    sql = "SELECT game_id, season, week, home_team_abbr, away_team_abbr, home_score, away_score, source_url FROM games WHERE season=?"
    params = [season]

    if week:
        sql += " AND week=?"
        params.append(week)

    sql += " ORDER BY week, game_id"

    df = query(sql, tuple(params))

    if df.empty:
        st.info(f"No games found for season {season}" + (f" week {week}" if week else ""))
        return

    # Display summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Games", len(df))
    with col2:
        if 'week' in df.columns:
            st.metric("Weeks", df['week'].nunique())
    with col3:
        teams = set(df['home_team_abbr'].tolist() + df['away_team_abbr'].tolist())
        st.metric("Teams", len(teams))

    st.divider()

    # Format display
    display_df = df.copy()
    display_df['Matchup'] = display_df['away_team_abbr'] + ' @ ' + display_df['home_team_abbr']
    display_df['Score'] = display_df['away_score'].astype(str) + ' - ' + display_df['home_score'].astype(str)

    # Show table
    st.dataframe(
        display_df[['season', 'week', 'Matchup', 'Score', 'game_id']],
        use_container_width=True,
        hide_index=True
    )


# ============================================================================
# Section: Team Overview
# ============================================================================

def render_team_overview(season: Optional[int], week: Optional[int]):
    """Display team statistics overview."""
    st.header("📊 Team Overview")

    if not season:
        st.warning("No season data available.")
        return

    # Query box score summary
    sql = "SELECT * FROM box_score_summary WHERE season=?"
    params = [season]

    if week:
        sql += " AND week<=?"
        params.append(week)

    df = query(sql, tuple(params))

    if df.empty:
        st.info(f"No team data found for season {season}")
        return

    # Aggregate by team
    agg_df = df.groupby('team').agg({
        'plays': 'sum',
        'yards_total': 'sum',
        'yards_per_play': 'mean',
        'rush_att': 'sum',
        'rush_yds': 'sum',
        'pass_att': 'sum',
        'pass_comp': 'sum',
        'pass_yds': 'sum',
        'points': 'sum'
    }).reset_index()

    # Calculate wins/losses from games
    games_sql = """
        SELECT
            home_team_abbr as team,
            CASE WHEN home_score > away_score THEN 1 ELSE 0 END as win
        FROM games WHERE season=? AND home_score IS NOT NULL
        UNION ALL
        SELECT
            away_team_abbr as team,
            CASE WHEN away_score > home_score THEN 1 ELSE 0 END as win
        FROM games WHERE season=? AND away_score IS NOT NULL
    """
    games_params = [season, season]
    if week:
        games_sql = games_sql.replace("home_score IS NOT NULL", "home_score IS NOT NULL AND week<=?")
        games_sql = games_sql.replace("away_score IS NOT NULL", "away_score IS NOT NULL AND week<=?")
        games_params = [season, week, season, week]

    games_df = query(games_sql, tuple(games_params))

    if not games_df.empty:
        wins_df = games_df.groupby('team')['win'].agg(['sum', 'count']).reset_index()
        wins_df.columns = ['team', 'wins', 'games']
        wins_df['losses'] = wins_df['games'] - wins_df['wins']
        agg_df = agg_df.merge(wins_df[['team', 'wins', 'losses', 'games']], on='team', how='left')

    # Sort by points
    agg_df = agg_df.sort_values('points', ascending=False)

    # Display
    st.dataframe(
        agg_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "yards_per_play": st.column_config.NumberColumn(format="%.2f"),
            "points": st.column_config.NumberColumn(format="%d")
        }
    )


# ============================================================================
# Section: Team Comparison
# ============================================================================

def render_team_comparison(season: Optional[int], week: Optional[int]):
    """Compare two teams side-by-side with deep insights."""
    st.header("🔀 Team Comparison")

    if not season:
        st.warning("No season data available.")
        return

    teams = get_teams(season)
    if len(teams) < 2:
        st.warning("Need at least 2 teams for comparison.")
        return

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", teams, index=0, key="team1")
    with col2:
        team2 = st.selectbox("Team 2", teams, index=min(1, len(teams)-1), key="team2")

    # Query team game stats
    sql = "SELECT * FROM box_score_summary WHERE season=? AND team IN (?, ?)"
    params = [season, team1, team2]
    if week:
        sql += " AND week<=?"
        params.append(week)
    df = query(sql, tuple(params))

    # Query player stats
    player_sql = "SELECT * FROM player_box_score WHERE season=? AND team IN (?, ?)"
    player_params = [season, team1, team2]
    if week:
        player_sql += " AND week<=?"
        player_params.append(week)
    players_df = query(player_sql, tuple(player_params))

    # Query head-to-head games
    h2h_sql = """
        SELECT g.*, tgs1.points as team1_pts, tgs2.points as team2_pts
        FROM games g
        LEFT JOIN team_game_summary tgs1 ON g.game_id = tgs1.game_id AND tgs1.team_abbr = ?
        LEFT JOIN team_game_summary tgs2 ON g.game_id = tgs2.game_id AND tgs2.team_abbr = ?
        WHERE g.season = ?
        AND ((g.home_team_abbr = ? AND g.away_team_abbr = ?) OR (g.home_team_abbr = ? AND g.away_team_abbr = ?))
    """
    h2h_params = (team1, team2, season, team1, team2, team2, team1)
    if week:
        h2h_sql += " AND g.week <= ?"
        h2h_params = h2h_params + (week,)
    h2h_df = query(h2h_sql, h2h_params)

    if df.empty:
        st.info("No data available for selected teams.")
        return

    # Aggregate team stats (totals)
    stats = df.groupby('team').agg({
        'plays': 'sum',
        'yards_total': 'sum',
        'yards_per_play': 'mean',
        'rush_att': 'sum',
        'rush_yds': 'sum',
        'pass_att': 'sum',
        'pass_comp': 'sum',
        'pass_yds': 'sum',
        'points': 'sum'
    }).reset_index()

    # Calculate per-game statistics and consistency metrics
    game_stats = df.groupby('team').agg({
        'points': ['mean', 'median', 'std', 'min', 'max', 'count'],
        'yards_total': ['mean', 'median', 'std', 'min', 'max'],
        'yards_per_play': ['mean', 'std'],
        'plays': 'mean'
    }).reset_index()

    # Flatten multi-level column names
    game_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in game_stats.columns.values]
    game_stats = game_stats.rename(columns={'team': 'team'})

    # Get W-L records - calculate from games table
    games_sql = """
        SELECT team,
               SUM(win) as wins,
               COUNT(*) - SUM(win) as losses,
               0 as ties
        FROM (
            SELECT home_team_abbr as team,
                   CASE WHEN home_score > away_score THEN 1 ELSE 0 END as win
            FROM games WHERE season=? AND home_team_abbr IN (?, ?) AND home_score IS NOT NULL
            UNION ALL
            SELECT away_team_abbr as team,
                   CASE WHEN away_score > home_score THEN 1 ELSE 0 END as win
            FROM games WHERE season=? AND away_team_abbr IN (?, ?) AND away_score IS NOT NULL
        )
        GROUP BY team
    """
    records_params = [season, team1, team2, season, team1, team2]
    if week:
        # Update both subqueries to include week filter
        games_sql = """
            SELECT team,
                   SUM(win) as wins,
                   COUNT(*) - SUM(win) as losses,
                   0 as ties
            FROM (
                SELECT home_team_abbr as team,
                       CASE WHEN home_score > away_score THEN 1 ELSE 0 END as win
                FROM games WHERE season=? AND week<=? AND home_team_abbr IN (?, ?) AND home_score IS NOT NULL
                UNION ALL
                SELECT away_team_abbr as team,
                       CASE WHEN away_score > home_score THEN 1 ELSE 0 END as win
                FROM games WHERE season=? AND week<=? AND away_team_abbr IN (?, ?) AND away_score IS NOT NULL
            )
            GROUP BY team
        """
        records_params = [season, week, team1, team2, season, week, team1, team2]
    records = query(games_sql, tuple(records_params))

    # Display head-to-head result
    if not h2h_df.empty:
        st.subheader("🏈 Head-to-Head Matchup")
        for _, game in h2h_df.iterrows():
            t1_score = game['team1_pts'] if pd.notna(game['team1_pts']) else 0
            t2_score = game['team2_pts'] if pd.notna(game['team2_pts']) else 0
            winner = team1 if t1_score > t2_score else (team2 if t2_score > t1_score else "TIE")
            st.markdown(f"**Week {game['week']}**: {team1} {int(t1_score)} - {int(t2_score)} {team2} {'✅ ' + winner if winner != 'TIE' else '🤝 TIE'}")

    st.divider()

    # Display relevant notes for selected teams
    st.subheader("📝 Relevant Notes")

    # Get notes for both teams
    team1_notes = get_notes(team_filter=team1, season_filter=season)
    team2_notes = get_notes(team_filter=team2, season_filter=season)

    # Combine and sort by created_at
    all_notes = pd.concat([team1_notes, team2_notes]).drop_duplicates(subset=['note_id'])
    if not all_notes.empty:
        all_notes = all_notes.sort_values('created_at', ascending=False)

    if all_notes.empty:
        st.info(f"No notes found for {team1} or {team2}. Add notes using the Quick Notes section in the sidebar!")
    else:
        note_col1, note_col2 = st.columns(2)

        with note_col1:
            st.markdown(f"**{team1} Notes**")
            if team1_notes.empty:
                st.caption("No notes for this team")
            else:
                for idx, note in team1_notes.head(3).iterrows():
                    with st.expander(f"📅 {note['created_at'][:16]}", expanded=False):
                        st.write(note['note_text'])
                        if note['tags']:
                            tags_list = note['tags'].split(',')
                            tags_formatted = ' '.join([f"`#{tag}`" for tag in tags_list])
                            st.caption(tags_formatted)

        with note_col2:
            st.markdown(f"**{team2} Notes**")
            if team2_notes.empty:
                st.caption("No notes for this team")
            else:
                for idx, note in team2_notes.head(3).iterrows():
                    with st.expander(f"📅 {note['created_at'][:16]}", expanded=False):
                        st.write(note['note_text'])
                        if note['tags']:
                            tags_list = note['tags'].split(',')
                            tags_formatted = ' '.join([f"`#{tag}`" for tag in tags_list])
                            st.caption(tags_formatted)

        # Link to view all notes
        if len(all_notes) > 6:
            st.caption(f"Showing most recent 3 notes per team. [View all {len(all_notes)} notes in Notes Manager](#)")

    st.divider()

    # Statistical Analysis Section
    st.subheader("📈 Statistical Analysis")

    t1_game_stats = game_stats[game_stats['team'] == team1]
    t2_game_stats = game_stats[game_stats['team'] == team2]

    if not t1_game_stats.empty and not t2_game_stats.empty:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"### {team1}")
            st.metric("Points/Game", f"{t1_game_stats['points_mean'].iloc[0]:.1f}")
            st.metric("Median Points", f"{t1_game_stats['points_median'].iloc[0]:.1f}")
            st.metric("Scoring Range", f"{int(t1_game_stats['points_min'].iloc[0])}-{int(t1_game_stats['points_max'].iloc[0])}")
            st.metric("Consistency (σ)", f"{t1_game_stats['points_std'].iloc[0]:.1f}")
            st.metric("Yards/Game", f"{t1_game_stats['yards_total_mean'].iloc[0]:.0f}")
            st.metric("YPP Avg", f"{t1_game_stats['yards_per_play_mean'].iloc[0]:.2f}")

        with col3:
            st.markdown(f"### {team2}")
            st.metric("Points/Game", f"{t2_game_stats['points_mean'].iloc[0]:.1f}")
            st.metric("Median Points", f"{t2_game_stats['points_median'].iloc[0]:.1f}")
            st.metric("Scoring Range", f"{int(t2_game_stats['points_min'].iloc[0])}-{int(t2_game_stats['points_max'].iloc[0])}")
            st.metric("Consistency (σ)", f"{t2_game_stats['points_std'].iloc[0]:.1f}")
            st.metric("Yards/Game", f"{t2_game_stats['yards_total_mean'].iloc[0]:.0f}")
            st.metric("YPP Avg", f"{t2_game_stats['yards_per_play_mean'].iloc[0]:.2f}")

        with col2:
            st.markdown("### Advantage")
            ppg_diff = t1_game_stats['points_mean'].iloc[0] - t2_game_stats['points_mean'].iloc[0]
            st.metric("PPG Leader", team1 if ppg_diff > 0 else team2, f"{abs(ppg_diff):+.1f}")

            ypg_diff = t1_game_stats['yards_total_mean'].iloc[0] - t2_game_stats['yards_total_mean'].iloc[0]
            st.metric("YPG Leader", team1 if ypg_diff > 0 else team2, f"{abs(ypg_diff):+.0f}")

            # Consistency comparison (lower std dev = more consistent)
            t1_std = t1_game_stats['points_std'].iloc[0]
            t2_std = t2_game_stats['points_std'].iloc[0]
            more_consistent = team1 if t1_std < t2_std else team2
            st.metric("More Consistent", more_consistent, f"σ: {min(t1_std, t2_std):.1f}")

    st.divider()

    # Strength of Schedule Analysis
    st.subheader("🛡️ Strength of Schedule Analysis")
    st.caption("Schedule difficulty from defensive matchup and overall opponent strength perspectives")

    # Get opponents for both teams
    t1_opponents_sql = """
        SELECT g.week, g.game_id,
               CASE
                   WHEN g.home_team_abbr = ? THEN g.away_team_abbr
                   ELSE g.home_team_abbr
               END as opponent
        FROM games g
        WHERE g.season = ?
        AND (g.home_team_abbr = ? OR g.away_team_abbr = ?)
        ORDER BY g.week
    """
    t1_opps = query(t1_opponents_sql, (team1, season, team1, team1))

    t2_opponents_sql = """
        SELECT g.week, g.game_id,
               CASE
                   WHEN g.home_team_abbr = ? THEN g.away_team_abbr
                   ELSE g.home_team_abbr
               END as opponent
        FROM games g
        WHERE g.season = ?
        AND (g.home_team_abbr = ? OR g.away_team_abbr = ?)
        ORDER BY g.week
    """
    t2_opps = query(t2_opponents_sql, (team2, season, team2, team2))

    # Calculate power ratings for opponent strength analysis
    all_teams_query_sos = f"SELECT DISTINCT home_team_abbr as team FROM games WHERE season={season}"
    if week:
        all_teams_query_sos += f" AND week < {week}"
    all_teams_sos_df = query(all_teams_query_sos)
    all_teams_sos = all_teams_sos_df['team'].tolist()

    # Calculate league-wide statistics for power ratings
    league_stats_sos = calculate_league_statistics(season, week, all_teams_sos)

    # Calculate baseline power ratings for all teams
    all_team_powers_sos = {}
    teams_data_sos = []
    for team in all_teams_sos:
        try:
            power = calculate_team_power_rating(team, season, week, all_team_powers=None, league_stats=league_stats_sos)
            team_record = calculate_win_loss_record(team, season, week)
            teams_data_sos.append({
                'team': team,
                'power': power,
                'wins': team_record.get('wins', 0),
                'losses': team_record.get('losses', 0)
            })
        except:
            team_record = calculate_win_loss_record(team, season, week)
            teams_data_sos.append({
                'team': team,
                'power': 50,
                'wins': team_record.get('wins', 0),
                'losses': team_record.get('losses', 0)
            })

    # ORDINAL RANKING: Sort by record first, then by power rating
    # This ensures teams with better records ALWAYS rank higher
    teams_data_sos.sort(key=lambda x: (-x['wins'], x['losses'], -x['power']))

    # Normalize power ratings to 1-100 scale with median = 50
    # Apply percentile ranking based on the SORTED order (wins dominant)
    if len(teams_data_sos) > 0:
        for rank_idx, team_data in enumerate(teams_data_sos):
            # Percentile based on position in sorted list (wins-dominant ranking)
            rank_pct = (rank_idx / len(teams_data_sos)) * 100
            # Invert so best team (rank_idx=0) gets ~100, worst gets ~1
            normalized_power = max(1, min(100, 100 - rank_pct))
            all_team_powers_sos[team_data['team']] = normalized_power

    # Calculate average opponent power for both teams
    t1_opp_powers = []
    t1_quality_opps = 0
    t1_weak_opps = 0
    if not t1_opps.empty:
        for opp in t1_opps['opponent'].tolist():
            opp_power = all_team_powers_sos.get(opp, 50)
            t1_opp_powers.append(opp_power)
            if opp_power > 55:
                t1_quality_opps += 1
            elif opp_power < 45:
                t1_weak_opps += 1

    t2_opp_powers = []
    t2_quality_opps = 0
    t2_weak_opps = 0
    if not t2_opps.empty:
        for opp in t2_opps['opponent'].tolist():
            opp_power = all_team_powers_sos.get(opp, 50)
            t2_opp_powers.append(opp_power)
            if opp_power > 55:
                t2_quality_opps += 1
            elif opp_power < 45:
                t2_weak_opps += 1

    t1_avg_opp_power = sum(t1_opp_powers) / len(t1_opp_powers) if t1_opp_powers else 50
    t2_avg_opp_power = sum(t2_opp_powers) / len(t2_opp_powers) if t2_opp_powers else 50

    # Calculate league-wide defensive averages
    league_def_sql = """
        SELECT
            AVG(points) as league_avg_pts,
            AVG(pass_yds) as league_avg_pass,
            AVG(rush_yds) as league_avg_rush,
            AVG(yards_total) as league_avg_total
        FROM team_game_summary
        WHERE season = ?
    """
    league_def = query(league_def_sql, (season,))

    if not league_def.empty and not t1_opps.empty and not t2_opps.empty:
        league_avg_pts = league_def['league_avg_pts'].iloc[0]
        league_avg_pass = league_def['league_avg_pass'].iloc[0]
        league_avg_rush = league_def['league_avg_rush'].iloc[0]
        league_avg_total = league_def['league_avg_total'].iloc[0]

        # Calculate opponent defensive stats for Team 1
        if not t1_opps.empty:
            t1_opp_list = tuple(t1_opps['opponent'].tolist())
            t1_opp_def_sql = f"""
                SELECT team_abbr,
                       AVG(points) as avg_pts_allowed,
                       AVG(pass_yds) as avg_pass_allowed,
                       AVG(rush_yds) as avg_rush_allowed,
                       AVG(yards_total) as avg_total_allowed
                FROM team_game_summary
                WHERE season = ? AND team_abbr IN ({','.join(['?']*len(t1_opp_list))})
                GROUP BY team_abbr
            """
            t1_opp_def = query(t1_opp_def_sql, (season,) + t1_opp_list)

        # Calculate opponent defensive stats for Team 2
        if not t2_opps.empty:
            t2_opp_list = tuple(t2_opps['opponent'].tolist())
            t2_opp_def_sql = f"""
                SELECT team_abbr,
                       AVG(points) as avg_pts_allowed,
                       AVG(pass_yds) as avg_pass_allowed,
                       AVG(rush_yds) as avg_rush_allowed,
                       AVG(yards_total) as avg_total_allowed
                FROM team_game_summary
                WHERE season = ? AND team_abbr IN ({','.join(['?']*len(t2_opp_list))})
                GROUP BY team_abbr
            """
            t2_opp_def = query(t2_opp_def_sql, (season,) + t2_opp_list)

        # Display SOS metrics in two sections
        st.markdown("### 💪 Opponent Power Rating (Overall Team Strength)")
        st.caption("Based on comprehensive power ratings - aligns with Power Rankings methodology")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{team1}**")

            # Determine difficulty based on avg opponent power
            power_difficulty = "Easy" if t1_avg_opp_power < 45 else ("Medium" if t1_avg_opp_power < 52 else ("Hard" if t1_avg_opp_power < 58 else "Elite"))

            st.metric("Avg Opponent Power", f"{t1_avg_opp_power:.1f}", help="Average power rating of opponents faced (50 = league average)")
            st.metric("Schedule Difficulty", power_difficulty)
            st.metric("Quality Opponents", f"{t1_quality_opps}", help="Opponents with power rating > 55")
            st.metric("Weak Opponents", f"{t1_weak_opps}", help="Opponents with power rating < 45")

        with col2:
            st.markdown(f"**{team2}**")

            power_difficulty = "Easy" if t2_avg_opp_power < 45 else ("Medium" if t2_avg_opp_power < 52 else ("Hard" if t2_avg_opp_power < 58 else "Elite"))

            st.metric("Avg Opponent Power", f"{t2_avg_opp_power:.1f}", help="Average power rating of opponents faced (50 = league average)")
            st.metric("Schedule Difficulty", power_difficulty)
            st.metric("Quality Opponents", f"{t2_quality_opps}", help="Opponents with power rating > 55")
            st.metric("Weak Opponents", f"{t2_weak_opps}", help="Opponents with power rating < 45")

        st.markdown("---")
        st.markdown("### 🛡️ Defensive Matchup Context")
        st.caption("How tough were the DEFENSES faced - useful for offensive projections")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{team1}**")
            if not t1_opp_def.empty:
                t1_avg_opp_pts = t1_opp_def['avg_pts_allowed'].mean()
                t1_avg_opp_pass = t1_opp_def['avg_pass_allowed'].mean()
                t1_avg_opp_rush = t1_opp_def['avg_rush_allowed'].mean()
                t1_avg_opp_total = t1_opp_def['avg_total_allowed'].mean()

                # Lower opponent stats = tougher schedule (they allow less)
                # SOS Score: 50 = avg, >50 = harder, <50 = easier
                t1_sos_overall = 100 - ((t1_avg_opp_pts / league_avg_pts) * 50)
                t1_sos_pass = 100 - ((t1_avg_opp_pass / 250) * 50)  # 250 yds/game = league avg
                t1_sos_rush = 100 - ((t1_avg_opp_rush / 120) * 50)  # 120 yds/game = league avg

                difficulty = "Easy" if t1_sos_overall < 40 else ("Medium" if t1_sos_overall < 55 else ("Hard" if t1_sos_overall < 70 else "Elite"))

                st.metric("Defense Strength", f"{t1_sos_overall:.0f}/100", help="Higher = tougher defensive opponents")
                st.metric("Pass Defense SOS", f"{t1_sos_pass:.0f}/100", help="Strength of pass defenses faced")
                st.metric("Rush Defense SOS", f"{t1_sos_rush:.0f}/100", help="Strength of rush defenses faced")

                st.markdown("**Opponent Defensive Averages:**")
                st.caption(f"Pts Allowed: {t1_avg_opp_pts:.1f}/game")
                st.caption(f"Pass Yds Allowed: {t1_avg_opp_pass:.0f}/game")
                st.caption(f"Rush Yds Allowed: {t1_avg_opp_rush:.0f}/game")

        with col2:
            st.markdown(f"**{team2}**")
            if not t2_opp_def.empty:
                t2_avg_opp_pts = t2_opp_def['avg_pts_allowed'].mean()
                t2_avg_opp_pass = t2_opp_def['avg_pass_allowed'].mean()
                t2_avg_opp_rush = t2_opp_def['avg_rush_allowed'].mean()
                t2_avg_opp_total = t2_opp_def['avg_total_allowed'].mean()

                t2_sos_overall = 100 - ((t2_avg_opp_pts / league_avg_pts) * 50)
                t2_sos_pass = 100 - ((t2_avg_opp_pass / 250) * 50)
                t2_sos_rush = 100 - ((t2_avg_opp_rush / 120) * 50)

                difficulty = "Easy" if t2_sos_overall < 40 else ("Medium" if t2_sos_overall < 55 else ("Hard" if t2_sos_overall < 70 else "Elite"))

                st.metric("Defense Strength", f"{t2_sos_overall:.0f}/100", help="Higher = tougher defensive opponents")
                st.metric("Pass Defense SOS", f"{t2_sos_pass:.0f}/100", help="Strength of pass defenses faced")
                st.metric("Rush Defense SOS", f"{t2_sos_rush:.0f}/100", help="Strength of rush defenses faced")

                st.markdown("**Opponent Defensive Averages:**")
                st.caption(f"Pts Allowed: {t2_avg_opp_pts:.1f}/game")
                st.caption(f"Pass Yds Allowed: {t2_avg_opp_pass:.0f}/game")
                st.caption(f"Rush Yds Allowed: {t2_avg_opp_rush:.0f}/game")

        # Context-Adjusted Performance
        st.markdown("---")
        st.markdown("### 📊 Context-Adjusted Performance")
        st.caption("How teams perform relative to the defenses they've faced")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{team1}**")
            if not t1_game_stats.empty and not t1_opp_def.empty:
                t1_ppg = t1_game_stats['points_mean'].iloc[0]
                t1_pass_pg = stats[stats['team'] == team1]['pass_yds'].iloc[0] / t1_game_stats['points_count'].iloc[0]
                t1_rush_pg = stats[stats['team'] == team1]['rush_yds'].iloc[0] / t1_game_stats['points_count'].iloc[0]

                pts_context = t1_ppg - t1_avg_opp_pts
                pass_context = t1_pass_pg - t1_avg_opp_pass
                rush_context = t1_rush_pg - t1_avg_opp_rush

                # Calculate efficiency multipliers
                pass_eff = (t1_pass_pg / t1_avg_opp_pass) if t1_avg_opp_pass > 0 else 1.0
                rush_eff = (t1_rush_pg / t1_avg_opp_rush) if t1_avg_opp_rush > 0 else 1.0

                st.metric("Points vs Opp Avg", f"{t1_ppg:.1f}/game", f"{pts_context:+.1f} vs defense")

                st.markdown("**Passing Efficiency:**")
                st.metric(f"Pass Yds vs Opp Avg", f"{t1_pass_pg:.0f}/game", f"{pass_context:+.0f} yds")
                st.caption(f"Opponents allow {t1_avg_opp_pass:.0f} pass yds/game • Efficiency: {pass_eff:.2f}x")

                st.markdown("**Rushing Efficiency:**")
                st.metric(f"Rush Yds vs Opp Avg", f"{t1_rush_pg:.0f}/game", f"{rush_context:+.0f} yds")
                st.caption(f"Opponents allow {t1_avg_opp_rush:.0f} rush yds/game • Efficiency: {rush_eff:.2f}x")

                # Quality indicators
                quality_indicators = []
                if pts_context > 5:
                    quality_indicators.append("⭐ Dominant scoring vs opponent defenses")
                if pass_context > 30:
                    quality_indicators.append("⭐ Elite passing vs tough pass defenses")
                if rush_context > 15:
                    quality_indicators.append("⭐ Elite rushing vs tough run defenses")
                if pass_eff > 1.15:
                    quality_indicators.append("💪 15%+ better than opp pass D avg")
                if rush_eff > 1.15:
                    quality_indicators.append("💪 15%+ better than opp rush D avg")

                if quality_indicators:
                    st.markdown("**Quality Performance:**")
                    for qi in quality_indicators:
                        st.markdown(f"- {qi}")

        with col2:
            st.markdown(f"**{team2}**")
            if not t2_game_stats.empty and not t2_opp_def.empty:
                t2_ppg = t2_game_stats['points_mean'].iloc[0]
                t2_pass_pg = stats[stats['team'] == team2]['pass_yds'].iloc[0] / t2_game_stats['points_count'].iloc[0]
                t2_rush_pg = stats[stats['team'] == team2]['rush_yds'].iloc[0] / t2_game_stats['points_count'].iloc[0]

                pts_context = t2_ppg - t2_avg_opp_pts
                pass_context = t2_pass_pg - t2_avg_opp_pass
                rush_context = t2_rush_pg - t2_avg_opp_rush

                # Calculate efficiency multipliers
                pass_eff = (t2_pass_pg / t2_avg_opp_pass) if t2_avg_opp_pass > 0 else 1.0
                rush_eff = (t2_rush_pg / t2_avg_opp_rush) if t2_avg_opp_rush > 0 else 1.0

                st.metric("Points vs Opp Avg", f"{t2_ppg:.1f}/game", f"{pts_context:+.1f} vs defense")

                st.markdown("**Passing Efficiency:**")
                st.metric(f"Pass Yds vs Opp Avg", f"{t2_pass_pg:.0f}/game", f"{pass_context:+.0f} yds")
                st.caption(f"Opponents allow {t2_avg_opp_pass:.0f} pass yds/game • Efficiency: {pass_eff:.2f}x")

                st.markdown("**Rushing Efficiency:**")
                st.metric(f"Rush Yds vs Opp Avg", f"{t2_rush_pg:.0f}/game", f"{rush_context:+.0f} yds")
                st.caption(f"Opponents allow {t2_avg_opp_rush:.0f} rush yds/game • Efficiency: {rush_eff:.2f}x")

                # Quality indicators
                quality_indicators = []
                if pts_context > 5:
                    quality_indicators.append("⭐ Dominant scoring vs opponent defenses")
                if pass_context > 30:
                    quality_indicators.append("⭐ Elite passing vs tough pass defenses")
                if rush_context > 15:
                    quality_indicators.append("⭐ Elite rushing vs tough run defenses")
                if pass_eff > 1.15:
                    quality_indicators.append("💪 15%+ better than opp pass D avg")
                if rush_eff > 1.15:
                    quality_indicators.append("💪 15%+ better than opp rush D avg")

                if quality_indicators:
                    st.markdown("**Quality Performance:**")
                    for qi in quality_indicators:
                        st.markdown(f"- {qi}")

    st.divider()

    # Game Results Section
    st.subheader("📊 Season Game Results")

    # Query game results for both teams
    def get_team_game_results(team, season, week):
        """Get game results for a team"""
        sql = f"""
            SELECT
                g.week,
                CASE
                    WHEN g.home_team_abbr = '{team}' THEN g.away_team_abbr
                    ELSE g.home_team_abbr
                END as opponent,
                CASE
                    WHEN g.home_team_abbr = '{team}' THEN g.home_score
                    ELSE g.away_score
                END as team_score,
                CASE
                    WHEN g.home_team_abbr = '{team}' THEN g.away_score
                    ELSE g.home_score
                END as opponent_score,
                CASE
                    WHEN g.home_team_abbr = '{team}' THEN (tgs_home.pass_yds + tgs_home.rush_yds)
                    ELSE (tgs_away.pass_yds + tgs_away.rush_yds)
                END as team_yards,
                CASE
                    WHEN g.home_team_abbr = '{team}' THEN (tgs_away.pass_yds + tgs_away.rush_yds)
                    ELSE (tgs_home.pass_yds + tgs_home.rush_yds)
                END as opponent_yards
            FROM games g
            LEFT JOIN team_game_summary tgs_home ON g.game_id = tgs_home.game_id AND g.home_team_abbr = tgs_home.team_abbr
            LEFT JOIN team_game_summary tgs_away ON g.game_id = tgs_away.game_id AND g.away_team_abbr = tgs_away.team_abbr
            WHERE g.season = {season}
                AND (g.home_team_abbr = '{team}' OR g.away_team_abbr = '{team}')
        """
        if week:
            sql += f" AND g.week < {week}"
        sql += " ORDER BY g.week"

        df = query(sql)
        if not df.empty:
            df['Total'] = df['team_score'] + df['opponent_score']
            df['Result'] = df.apply(lambda row: '✅ W' if row['team_score'] > row['opponent_score'] else '❌ L', axis=1)
            df['Week'] = df['week']
            df['Opponent'] = df['opponent']
            df['Team Score'] = df['team_score']
            df['Opponent Score'] = df['opponent_score']
            df['Team Yards'] = df['team_yards']
            df['Opp Yards'] = df['opponent_yards']
            return df[['Week', 'Opponent', 'Team Score', 'Opponent Score', 'Total', 'Team Yards', 'Opp Yards', 'Result']]
        return pd.DataFrame()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{team1} Games**")
        t1_results = get_team_game_results(team1, season, week)
        if not t1_results.empty:
            st.dataframe(t1_results, hide_index=True, use_container_width=True)
        else:
            st.info("No games played yet")

    with col2:
        st.markdown(f"**{team2} Games**")
        t2_results = get_team_game_results(team2, season, week)
        if not t2_results.empty:
            st.dataframe(t2_results, hide_index=True, use_container_width=True)
        else:
            st.info("No games played yet")

    st.divider()

    # Matchup Prediction Section
    st.subheader("🎯 Matchup Prediction")

    # Calculate metrics for both teams
    t1_record = calculate_win_loss_record(team1, season, week)
    t2_record = calculate_win_loss_record(team2, season, week)

    t1_pt_diff = calculate_point_differential(team1, season, week)
    t2_pt_diff = calculate_point_differential(team2, season, week)

    t1_epa = calculate_team_epa(team1, season, week)
    t2_epa = calculate_team_epa(team2, season, week)

    t1_success = calculate_success_rates(team1, season, week)
    t2_success = calculate_success_rates(team2, season, week)

    t1_explosive = calculate_explosive_plays(team1, season, week)
    t2_explosive = calculate_explosive_plays(team2, season, week)

    # Step 1: Get all teams and calculate league statistics for z-score normalization
    all_teams_query = f"SELECT DISTINCT home_team_abbr as team FROM games WHERE season={season}"
    if week:
        all_teams_query += f" AND week <= {week}"
    all_teams_df = query(all_teams_query)
    all_teams = all_teams_df['team'].tolist()

    # Calculate league-wide statistics
    league_stats = calculate_league_statistics(season, week, all_teams)

    # Step 2: Calculate baseline power ratings for all teams (for SOS calculation)
    all_team_powers = {}
    teams_data = []
    for team in all_teams:
        # Calculate baseline without quality margin (needs all_team_powers)
        power = calculate_team_power_rating(team, season, week, all_team_powers=None, league_stats=league_stats)
        team_record = calculate_win_loss_record(team, season, week)
        teams_data.append({
            'team': team,
            'power': power,
            'wins': team_record.get('wins', 0),
            'losses': team_record.get('losses', 0)
        })

    # ORDINAL RANKING: Sort by record first, then by power rating
    # This ensures teams with better records ALWAYS rank higher
    teams_data.sort(key=lambda x: (-x['wins'], x['losses'], -x['power']))

    # Normalize power ratings to 1-100 scale with median = 50
    # Apply percentile ranking based on the SORTED order (wins dominant)
    if len(teams_data) > 0:
        for rank_idx, team_data in enumerate(teams_data):
            # Percentile based on position in sorted list (wins-dominant ranking)
            rank_pct = (rank_idx / len(teams_data)) * 100
            # Invert so best team (rank_idx=0) gets ~100, worst gets ~1
            normalized_power = max(1, min(100, 100 - rank_pct))
            all_team_powers[team_data['team']] = normalized_power

    # Step 3: Calculate quality margin-adjusted league stats
    quality_margins = []
    for team in all_teams:
        try:
            qm = calculate_quality_victory_margin(team, season, week, all_team_powers)
            quality_margins.append(qm.get('quality_margin_per_game', 0))
        except:
            quality_margins.append(0)

    if len(quality_margins) > 1:
        import statistics
        league_stats['quality_margin'] = {
            'mean': statistics.mean(quality_margins),
            'std': statistics.stdev(quality_margins) if len(quality_margins) > 1 else 1.0
        }

    # Step 4: Calculate SOS-adjusted records
    t1_sos = calculate_sos_adjusted_record(team1, season, week, all_team_powers)
    t2_sos = calculate_sos_adjusted_record(team2, season, week, all_team_powers)

    # Step 5: Calculate all components for both teams
    t1_quality_margin = calculate_quality_victory_margin(team1, season, week, all_team_powers)
    t2_quality_margin = calculate_quality_victory_margin(team2, season, week, all_team_powers)

    t1_road_dom = calculate_road_dominance(team1, season, week)
    t2_road_dom = calculate_road_dominance(team2, season, week)
    t1_high_scoring = calculate_high_scoring_consistency(team1, season, week)
    t2_high_scoring = calculate_high_scoring_consistency(team2, season, week)
    t1_recent_form = calculate_recent_form(team1, season, week)
    t2_recent_form = calculate_recent_form(team2, season, week)

    # Calculate differential metrics
    t1_off_sr = t1_success.get('overall', 0)
    t1_def_sr = calculate_defensive_success_rate(team1, season, week)
    t1_sr_diff = t1_off_sr - t1_def_sr

    t2_off_sr = t2_success.get('overall', 0)
    t2_def_sr = calculate_defensive_success_rate(team2, season, week)
    t2_sr_diff = t2_off_sr - t2_def_sr

    t1_off_xpl = t1_explosive.get('explosive_rate', 0)
    t1_def_xpl = calculate_defensive_explosive_rate(team1, season, week)
    t1_xpl_diff = t1_off_xpl - t1_def_xpl

    t2_off_xpl = t2_explosive.get('explosive_rate', 0)
    t2_def_xpl = calculate_defensive_explosive_rate(team2, season, week)
    t2_xpl_diff = t2_off_xpl - t2_def_xpl

    t1_net_epa = t1_epa.get('off_epa_per_play', 0) - t1_epa.get('def_epa_per_play', 0)
    t2_net_epa = t2_epa.get('off_epa_per_play', 0) - t2_epa.get('def_epa_per_play', 0)

    # Step 6: Get normalized power ratings from all_team_powers (1-100 scale)
    t1_power = all_team_powers.get(team1, 50)  # Default to 50 if team not found
    t2_power = all_team_powers.get(team2, 50)

    # Calculate win probability (logistic function with adjusted scaling)
    rating_diff = t1_power - t2_power
    win_prob_t1 = 1 / (1 + 10 ** (-rating_diff / 10))  # Adjusted from /2 to /10 for new scale

    # Predict scores using team averages + adjustment
    t1_avg_pts = t1_pt_diff.get('avg_pts_for', 22)
    t2_avg_pts = t2_pt_diff.get('avg_pts_for', 22)
    t1_score = t1_avg_pts + (rating_diff * 0.15)  # Small adjustment based on rating diff
    t2_score = t2_avg_pts - (rating_diff * 0.15)

    # Display prediction
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"### {team1}")
        st.metric("Win Probability", f"{win_prob_t1*100:.1f}%")
        st.metric("Predicted Score", f"{max(0, t1_score):.0f}")
        st.metric("Power Rating", f"{t1_power:.2f}")

    with col2:
        st.markdown("### Prediction")
        winner = team1 if win_prob_t1 > 0.5 else team2
        confidence = "High" if abs(win_prob_t1 - 0.5) > 0.2 else ("Medium" if abs(win_prob_t1 - 0.5) > 0.1 else "Low")
        st.metric("Predicted Winner", winner)
        st.metric("Confidence", confidence)
        margin = abs(t1_score - t2_score)
        st.metric("Predicted Margin", f"{margin:.0f} pts")

    with col3:
        st.markdown(f"### {team2}")
        st.metric("Win Probability", f"{(1-win_prob_t1)*100:.1f}%")
        st.metric("Predicted Score", f"{max(0, t2_score):.0f}")
        st.metric("Power Rating", f"{t2_power:.2f}")

    st.divider()

    # Betting Lines Section
    st.markdown("### 💰 Expected Betting Lines")

    # Calculate betting metrics
    spread = t1_score - t2_score
    total = t1_score + t2_score

    # Convert win probability to American odds
    def prob_to_american_odds(prob):
        """Convert probability to American odds format."""
        if prob >= 0.5:
            # Favorite (negative odds)
            if prob >= 0.99:
                return -10000
            return int(-100 * prob / (1 - prob))
        else:
            # Underdog (positive odds)
            if prob <= 0.01:
                return 10000
            return int(100 * (1 - prob) / prob)

    t1_moneyline = prob_to_american_odds(win_prob_t1)
    t2_moneyline = prob_to_american_odds(1 - win_prob_t1)

    # Display betting lines
    bet_col1, bet_col2, bet_col3 = st.columns(3)

    with bet_col1:
        st.markdown("**Expected Spread**")
        if spread > 0:
            st.metric(f"{team1} (Favorite)", f"-{abs(spread):.1f}")
            st.caption(f"{team2}: +{abs(spread):.1f}")
        else:
            st.metric(f"{team2} (Favorite)", f"-{abs(spread):.1f}")
            st.caption(f"{team1}: +{abs(spread):.1f}")

    with bet_col2:
        st.markdown("**Expected Total**")
        st.metric("Over/Under", f"{total:.1f}")
        st.caption(f"({t1_score:.0f} + {t2_score:.0f})")

    with bet_col3:
        st.markdown("**Expected Moneyline**")
        if t1_moneyline < 0:
            st.metric(f"{team1} (Favorite)", f"{t1_moneyline:+d}")
            st.caption(f"{team2}: {t2_moneyline:+d}")
        else:
            st.metric(f"{team2} (Favorite)", f"{t2_moneyline:+d}")
            st.caption(f"{team1}: {t1_moneyline:+d}")

    st.caption("📊 Betting lines are model-generated expectations based on team metrics and may not reflect actual market lines.")

    # Key factors
    st.markdown("### 📊 Key Prediction Factors")

    # Data quality check
    has_limited_data = (t1_record.get('games_played', 0) < 4 or t2_record.get('games_played', 0) < 4)
    if has_limited_data:
        st.warning(f"⚠️ Limited sample size: {team1} ({t1_record.get('games_played', 0)} games), {team2} ({t2_record.get('games_played', 0)} games). Predictions may be less reliable early in the season.")

    # Show power rating breakdown with new components
    st.markdown("**Power Rating Components:**")
    breakdown_cols = st.columns(2)

    with breakdown_cols[0]:
        st.markdown(f"**{team1}** (Rating: {t1_power:.2f})")
        t1_wl_contrib = t1_sos.get('sos_adj_win_pct', t1_record.get('win_pct', 0)) * 25
        t1_pd_contrib = (t1_pt_diff.get('pt_diff_per_game', 0) / 7) * 25
        t1_off_contrib = t1_epa.get('off_epa_per_play', 0) * 20 * 10
        t1_def_contrib = (-t1_epa.get('def_epa_per_play', 0)) * 20 * 10
        t1_success_contrib = t1_success.get('overall', 0) * 15
        t1_explosive_contrib = t1_explosive.get('explosive_rate', 0) * 5 * 100
        t1_quality_contrib = t1_quality_margin.get('quality_margin_per_game', 0) * 15
        t1_road_contrib = t1_road_dom.get('road_score', 0)
        t1_high_scoring_contrib = t1_high_scoring.get('high_scoring_score', 0)
        t1_recent_contrib = t1_recent_form.get('recent_form_score', 0)

        st.markdown(f"- **Win-Loss (SOS Adj)**: {t1_wl_contrib:.2f} ({t1_record.get('wins', 0)}-{t1_record.get('losses', 0)}, SOS: {t1_sos.get('sos_adj_win_pct', 0):.1%} × 25)")
        st.caption(f"   Avg Opp Power: {t1_sos.get('avg_opp_power', 0):.1f} | Quality Wins: {t1_sos.get('quality_wins', 0)} | Bad Losses: {t1_sos.get('bad_losses', 0)}")
        st.markdown(f"- **Point Differential**: {t1_pd_contrib:.2f} ({t1_pt_diff.get('pt_diff_per_game', 0):+.1f}/game × 25)")
        st.markdown(f"- **Offensive EPA**: {t1_off_contrib:.2f} ({t1_epa.get('off_epa_per_play', 0):+.3f}/play × 200)")
        st.markdown(f"- **Defensive EPA**: {t1_def_contrib:.2f} ({-t1_epa.get('def_epa_per_play', 0):+.3f}/play × 200)")
        st.markdown(f"- **Success Rate**: {t1_success_contrib:.2f} ({t1_success.get('overall', 0):.1%} × 15)")
        st.markdown(f"- **Explosive Rate**: {t1_explosive_contrib:.2f} ({t1_explosive.get('explosive_rate', 0):.1%} × 500)")
        st.markdown(f"- **Quality Victory Margin**: {t1_quality_contrib:.2f} ({t1_quality_margin.get('blowout_wins', 0)} blowout wins, {t1_quality_margin.get('quality_margin_per_game', 0):.2f}/game × 15)")
        st.markdown(f"- **Road Dominance**: {t1_road_contrib:.2f} ({t1_road_dom.get('road_games', 0)} road games, {t1_road_dom.get('road_win_pct', 0):.1%} win%, {t1_road_dom.get('road_pt_diff_per_game', 0):+.1f} pt diff)")
        st.markdown(f"- **High Scoring Consistency**: {t1_high_scoring_contrib:.2f} ({t1_high_scoring.get('games_30plus', 0)} games 30+, {t1_high_scoring.get('high_scoring_rate', 0):.1%} × 8)")
        st.markdown(f"- **Recent Form**: {t1_recent_contrib:.2f} (Last 3: {', '.join([f'{m:+d}' for m in t1_recent_form.get('last_3_margins', [])])})")

    with breakdown_cols[1]:
        st.markdown(f"**{team2}** (Rating: {t2_power:.2f})")
        t2_wl_contrib = t2_sos.get('sos_adj_win_pct', t2_record.get('win_pct', 0)) * 25
        t2_pd_contrib = (t2_pt_diff.get('pt_diff_per_game', 0) / 7) * 25
        t2_off_contrib = t2_epa.get('off_epa_per_play', 0) * 20 * 10
        t2_def_contrib = (-t2_epa.get('def_epa_per_play', 0)) * 20 * 10
        t2_success_contrib = t2_success.get('overall', 0) * 15
        t2_explosive_contrib = t2_explosive.get('explosive_rate', 0) * 5 * 100
        t2_quality_contrib = t2_quality_margin.get('quality_margin_per_game', 0) * 15
        t2_road_contrib = t2_road_dom.get('road_score', 0)
        t2_high_scoring_contrib = t2_high_scoring.get('high_scoring_score', 0)
        t2_recent_contrib = t2_recent_form.get('recent_form_score', 0)

        st.markdown(f"- **Win-Loss (SOS Adj)**: {t2_wl_contrib:.2f} ({t2_record.get('wins', 0)}-{t2_record.get('losses', 0)}, SOS: {t2_sos.get('sos_adj_win_pct', 0):.1%} × 25)")
        st.caption(f"   Avg Opp Power: {t2_sos.get('avg_opp_power', 0):.1f} | Quality Wins: {t2_sos.get('quality_wins', 0)} | Bad Losses: {t2_sos.get('bad_losses', 0)}")
        st.markdown(f"- **Point Differential**: {t2_pd_contrib:.2f} ({t2_pt_diff.get('pt_diff_per_game', 0):+.1f}/game × 25)")
        st.markdown(f"- **Offensive EPA**: {t2_off_contrib:.2f} ({t2_epa.get('off_epa_per_play', 0):+.3f}/play × 200)")
        st.markdown(f"- **Defensive EPA**: {t2_def_contrib:.2f} ({-t2_epa.get('def_epa_per_play', 0):+.3f}/play × 200)")
        st.markdown(f"- **Success Rate**: {t2_success_contrib:.2f} ({t2_success.get('overall', 0):.1%} × 15)")
        st.markdown(f"- **Explosive Rate**: {t2_explosive_contrib:.2f} ({t2_explosive.get('explosive_rate', 0):.1%} × 500)")
        st.markdown(f"- **Quality Victory Margin**: {t2_quality_contrib:.2f} ({t2_quality_margin.get('blowout_wins', 0)} blowout wins, {t2_quality_margin.get('quality_margin_per_game', 0):.2f}/game × 15)")
        st.markdown(f"- **Road Dominance**: {t2_road_contrib:.2f} ({t2_road_dom.get('road_games', 0)} road games, {t2_road_dom.get('road_win_pct', 0):.1%} win%, {t2_road_dom.get('road_pt_diff_per_game', 0):+.1f} pt diff)")
        st.markdown(f"- **High Scoring Consistency**: {t2_high_scoring_contrib:.2f} ({t2_high_scoring.get('games_30plus', 0)} games 30+, {t2_high_scoring.get('high_scoring_rate', 0):.1%} × 8)")
        st.markdown(f"- **Recent Form**: {t2_recent_contrib:.2f} (Last 3: {', '.join([f'{m:+d}' for m in t2_recent_form.get('last_3_margins', [])])})")

    st.markdown("---")
    st.markdown("**Comparative Advantages:**")

    factors = []

    # Win-loss record comparison (SOS-adjusted)
    sos_diff = abs(t1_sos.get('sos_adj_win_pct', 0) - t2_sos.get('sos_adj_win_pct', 0))
    if sos_diff > 0.15:
        leader = team1 if t1_sos.get('sos_adj_win_pct', 0) > t2_sos.get('sos_adj_win_pct', 0) else team2
        leader_rec = t1_record if leader == team1 else t2_record
        leader_sos = t1_sos if leader == team1 else t2_sos
        advantage = "significant" if sos_diff > 0.35 else "moderate"
        factors.append(f"🏆 **{leader}** has a {advantage} record advantage ({leader_rec.get('wins', 0)}-{leader_rec.get('losses', 0)}, SOS-Adj: {leader_sos.get('sos_adj_win_pct', 0):.1%}, Avg Opp: {leader_sos.get('avg_opp_power', 0):.1f})")

    # Point differential comparison
    pt_diff_diff = abs(t1_pt_diff.get('pt_diff_per_game', 0) - t2_pt_diff.get('pt_diff_per_game', 0))
    if pt_diff_diff > 3:
        leader = team1 if t1_pt_diff.get('pt_diff_per_game', 0) > t2_pt_diff.get('pt_diff_per_game', 0) else team2
        leader_diff = t1_pt_diff.get('pt_diff_per_game', 0) if leader == team1 else t2_pt_diff.get('pt_diff_per_game', 0)
        advantage = "significant" if pt_diff_diff > 7 else "moderate"
        factors.append(f"📊 **{leader}** has a {advantage} point differential advantage ({leader_diff:+.1f} pts/game)")

    # Offensive EPA comparison
    off_epa_diff = abs(t1_epa.get('off_epa_per_play', 0) - t2_epa.get('off_epa_per_play', 0))
    if off_epa_diff > 0.03:
        leader = team1 if t1_epa.get('off_epa_per_play', 0) > t2_epa.get('off_epa_per_play', 0) else team2
        advantage = "significant" if off_epa_diff > 0.08 else "moderate" if off_epa_diff > 0.05 else "slight"
        factors.append(f"✅ **{leader}** has a {advantage} offensive efficiency advantage (+{off_epa_diff:.3f} EPA/play)")

    # Defensive EPA comparison
    def_epa_diff = abs(t1_epa.get('def_epa_per_play', 0) - t2_epa.get('def_epa_per_play', 0))
    if def_epa_diff > 0.03:
        leader = team1 if t1_epa.get('def_epa_per_play', 0) < t2_epa.get('def_epa_per_play', 0) else team2
        advantage = "significant" if def_epa_diff > 0.08 else "moderate" if def_epa_diff > 0.05 else "slight"
        factors.append(f"🛡️ **{leader}** has a {advantage} defensive advantage ({def_epa_diff:.3f} EPA/play better)")

    if factors:
        for factor in factors:
            st.markdown(factor)
    else:
        st.info("This matchup appears very evenly matched across all key metrics.")

    # Prediction methodology note
    st.markdown("---")
    with st.expander("ℹ️ Prediction Methodology"):
        st.markdown("""
        **How the prediction is calculated:**

        ### WINS DOMINANT 2025 FORMULA:

        **Power Rating** = [z(SOS-Adj Win%) × 0.55 + z(Quality Margin) × 0.15 + z(Net EPA) × 0.20 +
                           z(SR Diff) × 0.05 + z(Xpl Diff) × 0.03 + z(Road Dom) × 0.01 + z(High Scoring) × 0.01] × 10

        **🏆 WINS ARE EVERYTHING (70% Combined):**
        - SOS-Adjusted Win%: **55%** — Winning is absolutely dominant
        - Quality Victory Margin: **15%** — Beating good teams badly matters enormously
        - **Total: 70%** — Your record determines your ranking

        **Key Philosophy:**
        - ✅ **WINS DOMINATE** — 70% weight ensures teams with better records rank higher
        - ✅ **Quality Wins Rewarded** — 15% on quality margin = beating tough teams badly is highly valuable
        - ✅ **Explosive Plays Minimized** — Reduced to 3% to prevent over-weighting volatility
        - ✅ **Z-Score Normalization** — all components standardized to league distribution
        - ✅ **Outlier Protection** — Z-scores capped at ±3.0, minimum std thresholds

        **Component Details (Normalized Weights Sum to 1.0):**

        1. **SOS-Adjusted Win%** (55% weight — ABSOLUTELY DOMINANT):
           - Raw win% weighted by opponent power ratings
           - Wins vs strong teams count more, losses vs weak teams hurt more
           - Z-scored: (team_win% − league_avg) / std_dev
           - **Dominant factor** — winning is everything

        2. **Quality Victory Margin** (15% weight — CRITICAL):
           - Rewards blowout wins (>14 point margin) scaled by opponent strength
           - Formula: (margin − 14) × (opponent_power / 50) per game
           - Beating strong teams by large margins significantly increases rating
           - Beating weak teams by large margins has minimal impact
           - Z-scored to league distribution

        3. **Net EPA** (20% weight — Tertiary):
           - Net EPA = Offensive EPA per play − Defensive EPA per play
           - Single metric captures both offense and defense
           - Z-scored to league distribution
           - Secondary to winning

        4. **Success Rate Diff** (5% weight — Minor):
           - SR Diff = Offensive Success Rate − Defensive Success Rate
           - Offensive SR: logistic function on team's yards per play
           - Defensive SR: inverted logistic on opponent yards allowed
           - Z-scored for fair comparison

        5. **Explosive Rate Diff** (3% weight — Minimal):
           - Xpl Diff = Offensive Explosive% − Defensive Explosive%
           - Explosive = 40+ yard plays per 65 plays
           - **Minimal weight** — prevents over-emphasis on volatility
           - Z-scored to prevent outlier dominance

        6. **Road Dominance** (1% weight — Tiebreaker):
           - Formula: (Road Win% × 0.6 + Road Pt Diff/7 × 0.4) × 10
           - Rewards teams that perform well away from home
           - Z-scored to league distribution

        7. **High Scoring Consistency** (1% weight — Tiebreaker):
           - Percentage of games scoring 30+ points
           - Formula: (Games 30+ / Total games) × 8
           - Rewards elite offenses
           - Z-scored to league distribution

        **Final Score:** Sum of weighted z-scores × 10 → scaled to ~0-100 range
        - All z-scores capped at ±3.0 to prevent extreme outliers
        - Minimum std thresholds prevent statistical anomalies

        ---

        **Win Probability:** Logistic function based on power rating difference
        - Formula: 1 / (1 + 10^(−rating_diff/10))

        **Predicted Score:** Team avg points/game ± (rating_diff × 0.15)

        **Confidence Level:**
        - High: Win probability > 70% or < 30%
           - Medium: Win probability 60-70% or 30-40%
           - Low: Win probability 40-60% (toss-up)

        5. **Data Quality**:
           - Predictions are more reliable with 4+ games played
           - Early season predictions may not reflect true team strength
        """)

    st.divider()

    # Tornado Chart - Side-by-side comparison
    st.subheader("📊 Statistical Comparison (Tornado Chart)")

    if not t1_game_stats.empty and not t2_game_stats.empty:
        categories = ['Points/Game', 'Yards/Game', 'Rush Yds', 'Pass Yds', 'YPP']

        t1_values = [
            t1_game_stats['points_mean'].iloc[0],
            t1_game_stats['yards_total_mean'].iloc[0],
            stats[stats['team'] == team1]['rush_yds'].iloc[0] / t1_game_stats['points_count'].iloc[0],
            stats[stats['team'] == team1]['pass_yds'].iloc[0] / t1_game_stats['points_count'].iloc[0],
            t1_game_stats['yards_per_play_mean'].iloc[0]
        ]

        t2_values = [
            t2_game_stats['points_mean'].iloc[0],
            t2_game_stats['yards_total_mean'].iloc[0],
            stats[stats['team'] == team2]['rush_yds'].iloc[0] / t2_game_stats['points_count'].iloc[0],
            stats[stats['team'] == team2]['pass_yds'].iloc[0] / t2_game_stats['points_count'].iloc[0],
            t2_game_stats['yards_per_play_mean'].iloc[0]
        ]

        fig_tornado = go.Figure()

        # Team 1 bars (left side, negative values for visual effect)
        fig_tornado.add_trace(go.Bar(
            y=categories,
            x=[-v for v in t1_values],
            name=team1,
            orientation='h',
            marker=dict(color='#1f77b4'),
            text=[f'{v:.1f}' for v in t1_values],
            textposition='inside'
        ))

        # Team 2 bars (right side, positive values)
        fig_tornado.add_trace(go.Bar(
            y=categories,
            x=t2_values,
            name=team2,
            orientation='h',
            marker=dict(color='#ff7f0e'),
            text=[f'{v:.1f}' for v in t2_values],
            textposition='inside'
        ))

        fig_tornado.update_layout(
            barmode='relative',
            height=400,
            xaxis=dict(title='', showgrid=False),
            yaxis=dict(title=''),
            showlegend=True,
            margin=dict(l=20, r=20, t=20, b=20)
        )

        st.plotly_chart(fig_tornado, use_container_width=True)

    st.divider()

    # Performance Trends
    st.subheader("📉 Performance Trends (Week-by-Week)")

    if not df.empty:
        # Get week-by-week data
        t1_weeks = df[df['team'] == team1].sort_values('week')
        t2_weeks = df[df['team'] == team2].sort_values('week')

        if not t1_weeks.empty and not t2_weeks.empty:
            # Points trend
            fig_trends = go.Figure()

            fig_trends.add_trace(go.Scatter(
                x=t1_weeks['week'],
                y=t1_weeks['points'],
                mode='lines+markers',
                name=f'{team1} Points',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ))

            fig_trends.add_trace(go.Scatter(
                x=t2_weeks['week'],
                y=t2_weeks['points'],
                mode='lines+markers',
                name=f'{team2} Points',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=8)
            ))

            fig_trends.update_layout(
                title="Points Scored by Week",
                xaxis_title="Week",
                yaxis_title="Points",
                height=400,
                hovermode='x unified',
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig_trends, use_container_width=True)

            # Yards trend
            fig_yards = go.Figure()

            fig_yards.add_trace(go.Scatter(
                x=t1_weeks['week'],
                y=t1_weeks['yards_total'],
                mode='lines+markers',
                name=f'{team1} Yards',
                line=dict(color='#1f77b4', width=3, dash='dot'),
                marker=dict(size=8)
            ))

            fig_yards.add_trace(go.Scatter(
                x=t2_weeks['week'],
                y=t2_weeks['yards_total'],
                mode='lines+markers',
                name=f'{team2} Yards',
                line=dict(color='#ff7f0e', width=3, dash='dot'),
                marker=dict(size=8)
            ))

            fig_yards.update_layout(
                title="Total Yards by Week",
                xaxis_title="Week",
                yaxis_title="Yards",
                height=400,
                hovermode='x unified',
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig_yards, use_container_width=True)

    st.divider()

    # Radar Chart - Multi-dimensional comparison
    st.subheader("🎯 Multi-Dimensional Comparison (Radar Chart)")

    if not t1_game_stats.empty and not t2_game_stats.empty:
        # Normalize stats to 0-100 scale for radar chart
        def normalize(val, min_val, max_val):
            if max_val == min_val:
                return 50
            return ((val - min_val) / (max_val - min_val)) * 100

        # Get values for normalization
        ppg_max = max(t1_game_stats['points_mean'].iloc[0], t2_game_stats['points_mean'].iloc[0])
        ppg_min = min(t1_game_stats['points_mean'].iloc[0], t2_game_stats['points_mean'].iloc[0])

        ypg_max = max(t1_game_stats['yards_total_mean'].iloc[0], t2_game_stats['yards_total_mean'].iloc[0])
        ypg_min = min(t1_game_stats['yards_total_mean'].iloc[0], t2_game_stats['yards_total_mean'].iloc[0])

        ypp_max = max(t1_game_stats['yards_per_play_mean'].iloc[0], t2_game_stats['yards_per_play_mean'].iloc[0])
        ypp_min = min(t1_game_stats['yards_per_play_mean'].iloc[0], t2_game_stats['yards_per_play_mean'].iloc[0])

        # Consistency: inverse of std dev (higher is better)
        t1_consistency = 1 / (t1_game_stats['points_std'].iloc[0] + 0.1)  # +0.1 to avoid division by zero
        t2_consistency = 1 / (t2_game_stats['points_std'].iloc[0] + 0.1)
        cons_max = max(t1_consistency, t2_consistency)
        cons_min = min(t1_consistency, t2_consistency)

        # Scoring range (inverse - smaller range is better)
        t1_range_inv = 100 / (t1_game_stats['points_max'].iloc[0] - t1_game_stats['points_min'].iloc[0] + 1)
        t2_range_inv = 100 / (t2_game_stats['points_max'].iloc[0] - t2_game_stats['points_min'].iloc[0] + 1)
        range_max = max(t1_range_inv, t2_range_inv)
        range_min = min(t1_range_inv, t2_range_inv)

        categories_radar = ['Points/Game', 'Yards/Game', 'Efficiency<br>(YPP)', 'Consistency', 'Scoring<br>Stability']

        t1_radar = [
            normalize(t1_game_stats['points_mean'].iloc[0], ppg_min, ppg_max),
            normalize(t1_game_stats['yards_total_mean'].iloc[0], ypg_min, ypg_max),
            normalize(t1_game_stats['yards_per_play_mean'].iloc[0], ypp_min, ypp_max),
            normalize(t1_consistency, cons_min, cons_max),
            normalize(t1_range_inv, range_min, range_max)
        ]

        t2_radar = [
            normalize(t2_game_stats['points_mean'].iloc[0], ppg_min, ppg_max),
            normalize(t2_game_stats['yards_total_mean'].iloc[0], ypg_min, ypg_max),
            normalize(t2_game_stats['yards_per_play_mean'].iloc[0], ypp_min, ypp_max),
            normalize(t2_consistency, cons_min, cons_max),
            normalize(t2_range_inv, range_min, range_max)
        ]

        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=t1_radar + [t1_radar[0]],  # Close the polygon
            theta=categories_radar + [categories_radar[0]],
            fill='toself',
            name=team1,
            line=dict(color='#1f77b4', width=2),
            fillcolor='rgba(31, 119, 180, 0.3)'
        ))

        fig_radar.add_trace(go.Scatterpolar(
            r=t2_radar + [t2_radar[0]],
            theta=categories_radar + [categories_radar[0]],
            fill='toself',
            name=team2,
            line=dict(color='#ff7f0e', width=2),
            fillcolor='rgba(255, 127, 14, 0.3)'
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=True,
            height=500,
            margin=dict(l=80, r=80, t=40, b=40)
        )

        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # Momentum Analysis
    st.subheader("🔥 Momentum & Recent Form")

    if not df.empty:
        # Get last 3 games for each team
        t1_recent = df[df['team'] == team1].sort_values('week', ascending=False).head(3)
        t2_recent = df[df['team'] == team2].sort_values('week', ascending=False).head(3)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### {team1}")
            if not t1_recent.empty:
                recent_ppg = t1_recent['points'].mean()
                recent_ypg = t1_recent['yards_total'].mean()
                season_ppg = t1_game_stats['points_mean'].iloc[0]

                trend = "↗️ Improving" if recent_ppg > season_ppg else ("↘️ Declining" if recent_ppg < season_ppg else "→ Steady")

                st.metric("Last 3 Games PPG", f"{recent_ppg:.1f}", f"{recent_ppg - season_ppg:+.1f}")
                st.metric("Last 3 Games YPG", f"{recent_ypg:.0f}")
                st.metric("Form Trend", trend)

                # Show last 3 game results
                st.markdown("**Recent Results:**")
                for _, game in t1_recent.iterrows():
                    st.text(f"Week {int(game['week'])}: {int(game['points'])} pts, {int(game['yards_total'])} yds")

        with col2:
            st.markdown(f"### {team2}")
            if not t2_recent.empty:
                recent_ppg = t2_recent['points'].mean()
                recent_ypg = t2_recent['yards_total'].mean()
                season_ppg = t2_game_stats['points_mean'].iloc[0]

                trend = "↗️ Improving" if recent_ppg > season_ppg else ("↘️ Declining" if recent_ppg < season_ppg else "→ Steady")

                st.metric("Last 3 Games PPG", f"{recent_ppg:.1f}", f"{recent_ppg - season_ppg:+.1f}")
                st.metric("Last 3 Games YPG", f"{recent_ypg:.0f}")
                st.metric("Form Trend", trend)

                # Show last 3 game results
                st.markdown("**Recent Results:**")
                for _, game in t2_recent.iterrows():
                    st.text(f"Week {int(game['week'])}: {int(game['points'])} pts, {int(game['yards_total'])} yds")

    st.divider()

    # Team Overview Comparison
    st.subheader("📊 Season Totals Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### " + team1)
        t1_stats = stats[stats['team'] == team1]
        t1_game_stats_sum = game_stats[game_stats['team'] == team1]
        t1_record = records[records['team'] == team1]
        if not t1_record.empty:
            st.metric("Record", f"{int(t1_record['wins'].iloc[0])}-{int(t1_record['losses'].iloc[0])}-{int(t1_record['ties'].iloc[0])}")
        if not t1_stats.empty and not t1_game_stats_sum.empty:
            st.metric("Points", int(t1_stats['points'].iloc[0]))
            total_yds = int(t1_stats['yards_total'].iloc[0])
            ypg = t1_game_stats_sum['yards_total_mean'].iloc[0]
            st.metric("Total Yards", f"{total_yds:,} ({ypg:.0f}/game)")
            st.metric("Yards/Play", f"{t1_stats['yards_per_play'].iloc[0]:.2f}")

    with col3:
        st.markdown("### " + team2)
        t2_stats = stats[stats['team'] == team2]
        t2_game_stats_sum = game_stats[game_stats['team'] == team2]
        t2_record = records[records['team'] == team2]
        if not t2_record.empty:
            st.metric("Record", f"{int(t2_record['wins'].iloc[0])}-{int(t2_record['losses'].iloc[0])}-{int(t2_record['ties'].iloc[0])}")
        if not t2_stats.empty and not t2_game_stats_sum.empty:
            st.metric("Points", int(t2_stats['points'].iloc[0]))
            total_yds = int(t2_stats['yards_total'].iloc[0])
            ypg = t2_game_stats_sum['yards_total_mean'].iloc[0]
            st.metric("Total Yards", f"{total_yds:,} ({ypg:.0f}/game)")
            st.metric("Yards/Play", f"{t2_stats['yards_per_play'].iloc[0]:.2f}")

    with col2:
        st.markdown("### Advantage")
        if not t1_stats.empty and not t2_stats.empty:
            pts_diff = int(t1_stats['points'].iloc[0]) - int(t2_stats['points'].iloc[0])
            st.metric("Points", team1 if pts_diff > 0 else team2, f"{abs(pts_diff):+d}")

            yds_diff = int(t1_stats['yards_total'].iloc[0]) - int(t2_stats['yards_total'].iloc[0])
            st.metric("Yards", team1 if yds_diff > 0 else team2, f"{abs(yds_diff):+d}")

            ypp_diff = t1_stats['yards_per_play'].iloc[0] - t2_stats['yards_per_play'].iloc[0]
            st.metric("YPP", team1 if ypp_diff > 0 else team2, f"{abs(ypp_diff):+.2f}")

    # Add defensive summary stats
    st.markdown("### Defensive Summary")
    t1_def_overview = get_team_defensive_stats(team1, season, week)
    t2_def_overview = get_team_defensive_stats(team2, season, week)

    # Check if defensive data is available
    has_def_summary_data = (t1_def_overview['games'] > 0 or t2_def_overview['games'] > 0)

    if not has_def_summary_data:
        st.info(f"⚠️ Defensive summary not available - detailed defensive data only exists for select teams (currently ARI and SEA for 2025 season)")
    else:
        def_col1, def_col2, def_col3 = st.columns(3)

        with def_col1:
            st.markdown(f"**{team1} Defense**")
            st.metric("Sacks", f"{t1_def_overview['total_sacks']:.1f}", help=f"{t1_def_overview['sacks_per_game']:.2f}/game")
            st.metric("INTs", f"{t1_def_overview['total_ints']:.0f}", help=f"{t1_def_overview['ints_per_game']:.2f}/game")
            st.metric("Tackles", f"{t1_def_overview['total_tackles']:.0f}", help=f"{t1_def_overview['tackles_per_game']:.1f}/game")

        with def_col3:
            st.markdown(f"**{team2} Defense**")
            st.metric("Sacks", f"{t2_def_overview['total_sacks']:.1f}", help=f"{t2_def_overview['sacks_per_game']:.2f}/game")
            st.metric("INTs", f"{t2_def_overview['total_ints']:.0f}", help=f"{t2_def_overview['ints_per_game']:.2f}/game")
            st.metric("Tackles", f"{t2_def_overview['total_tackles']:.0f}", help=f"{t2_def_overview['tackles_per_game']:.1f}/game")

        with def_col2:
            st.markdown("**Advantage**")
            sacks_diff = t1_def_overview['total_sacks'] - t2_def_overview['total_sacks']
            st.metric("Sacks", team1 if sacks_diff > 0 else team2, f"{abs(sacks_diff):+.1f}")

            ints_diff = t1_def_overview['total_ints'] - t2_def_overview['total_ints']
            st.metric("INTs", team1 if ints_diff > 0 else team2, f"{abs(ints_diff):+.0f}")

            tackles_diff = t1_def_overview['total_tackles'] - t2_def_overview['total_tackles']
            st.metric("Tackles", team1 if tackles_diff > 0 else team2, f"{abs(tackles_diff):+.0f}")

    st.divider()

    # Home/Away Splits and Margin Analysis
    st.subheader("🏠 Home/Away Performance & Margins")

    # Query home/away splits with margins
    home_away_sql = """
        SELECT
            team,
            SUM(CASE WHEN location = 'home' AND win = 1 THEN 1 ELSE 0 END) as home_wins,
            SUM(CASE WHEN location = 'home' AND win = 0 THEN 1 ELSE 0 END) as home_losses,
            SUM(CASE WHEN location = 'away' AND win = 1 THEN 1 ELSE 0 END) as away_wins,
            SUM(CASE WHEN location = 'away' AND win = 0 THEN 1 ELSE 0 END) as away_losses,
            AVG(CASE WHEN location = 'home' THEN margin ELSE NULL END) as home_margin,
            AVG(CASE WHEN location = 'away' THEN margin ELSE NULL END) as away_margin,
            AVG(CASE WHEN win = 1 THEN margin ELSE NULL END) as avg_win_margin,
            AVG(CASE WHEN win = 0 THEN -margin ELSE NULL END) as avg_loss_margin,
            MAX(CASE WHEN win = 1 THEN margin ELSE NULL END) as biggest_win,
            MAX(CASE WHEN win = 0 THEN -margin ELSE NULL END) as biggest_loss,
            SUM(CASE WHEN ABS(margin) <= 7 AND win = 1 THEN 1 ELSE 0 END) as close_wins,
            SUM(CASE WHEN ABS(margin) <= 7 AND win = 0 THEN 1 ELSE 0 END) as close_losses,
            SUM(CASE WHEN margin >= 14 AND win = 1 THEN 1 ELSE 0 END) as blowout_wins,
            SUM(CASE WHEN margin <= -14 AND win = 0 THEN 1 ELSE 0 END) as blowout_losses
        FROM (
            SELECT
                home_team_abbr as team,
                'home' as location,
                CASE WHEN home_score > away_score THEN 1 ELSE 0 END as win,
                (home_score - away_score) as margin
            FROM games
            WHERE season=? AND home_team_abbr IN (?, ?) AND home_score IS NOT NULL
            UNION ALL
            SELECT
                away_team_abbr as team,
                'away' as location,
                CASE WHEN away_score > home_score THEN 1 ELSE 0 END as win,
                (away_score - home_score) as margin
            FROM games
            WHERE season=? AND away_team_abbr IN (?, ?) AND away_score IS NOT NULL
        )
        GROUP BY team
    """
    home_away_params = [season, team1, team2, season, team1, team2]
    if week:
        home_away_sql = home_away_sql.replace("AND home_score IS NOT NULL", "AND week<=? AND home_score IS NOT NULL")
        home_away_sql = home_away_sql.replace("AND away_score IS NOT NULL", "AND week<=? AND away_score IS NOT NULL")
        home_away_params = [season, week, team1, team2, season, week, team1, team2]

    home_away_df = query(home_away_sql, tuple(home_away_params))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {team1}")
        if not home_away_df.empty:
            t1_data = home_away_df[home_away_df['team'] == team1]
            if not t1_data.empty:
                row = t1_data.iloc[0]

                # Home/Away Records
                st.markdown("**📍 Home/Away Splits**")
                st.metric("Home Record", f"{int(row['home_wins'])}-{int(row['home_losses'])}")
                st.metric("Away Record", f"{int(row['away_wins'])}-{int(row['away_losses'])}")
                if pd.notna(row['home_margin']) and pd.notna(row['away_margin']):
                    st.metric("Home Avg Margin", f"{row['home_margin']:+.1f} pts")
                    st.metric("Away Avg Margin", f"{row['away_margin']:+.1f} pts")

                st.markdown("---")

                # Margin Analysis
                st.markdown("**📊 Margin Analysis**")
                if pd.notna(row['avg_win_margin']):
                    st.metric("Avg Win Margin", f"+{row['avg_win_margin']:.1f} pts")
                if pd.notna(row['avg_loss_margin']):
                    st.metric("Avg Loss Margin", f"-{row['avg_loss_margin']:.1f} pts")
                if pd.notna(row['biggest_win']):
                    st.metric("Biggest Win", f"+{int(row['biggest_win'])} pts")
                if pd.notna(row['biggest_loss']):
                    st.metric("Biggest Loss", f"-{int(row['biggest_loss'])} pts")

                # Competitiveness
                total_games = int(row['home_wins'] + row['home_losses'] + row['away_wins'] + row['away_losses'])
                close_games = int(row['close_wins'] + row['close_losses'])
                if total_games > 0:
                    comp_pct = 100 * close_games / total_games
                    st.metric("One-Score Games", f"{close_games} ({comp_pct:.0f}%)")
                st.metric("Close Wins (≤7)", int(row['close_wins']))
                st.metric("Close Losses (≤7)", int(row['close_losses']))

    with col2:
        st.markdown(f"### {team2}")
        if not home_away_df.empty:
            t2_data = home_away_df[home_away_df['team'] == team2]
            if not t2_data.empty:
                row = t2_data.iloc[0]

                # Home/Away Records
                st.markdown("**📍 Home/Away Splits**")
                st.metric("Home Record", f"{int(row['home_wins'])}-{int(row['home_losses'])}")
                st.metric("Away Record", f"{int(row['away_wins'])}-{int(row['away_losses'])}")
                if pd.notna(row['home_margin']) and pd.notna(row['away_margin']):
                    st.metric("Home Avg Margin", f"{row['home_margin']:+.1f} pts")
                    st.metric("Away Avg Margin", f"{row['away_margin']:+.1f} pts")

                st.markdown("---")

                # Margin Analysis
                st.markdown("**📊 Margin Analysis**")
                if pd.notna(row['avg_win_margin']):
                    st.metric("Avg Win Margin", f"+{row['avg_win_margin']:.1f} pts")
                if pd.notna(row['avg_loss_margin']):
                    st.metric("Avg Loss Margin", f"-{row['avg_loss_margin']:.1f} pts")
                if pd.notna(row['biggest_win']):
                    st.metric("Biggest Win", f"+{int(row['biggest_win'])} pts")
                if pd.notna(row['biggest_loss']):
                    st.metric("Biggest Loss", f"-{int(row['biggest_loss'])} pts")

                # Competitiveness
                total_games = int(row['home_wins'] + row['home_losses'] + row['away_wins'] + row['away_losses'])
                close_games = int(row['close_wins'] + row['close_losses'])
                if total_games > 0:
                    comp_pct = 100 * close_games / total_games
                    st.metric("One-Score Games", f"{close_games} ({comp_pct:.0f}%)")
                st.metric("Close Wins (≤7)", int(row['close_wins']))
                st.metric("Close Losses (≤7)", int(row['close_losses']))

    st.divider()

    # Offensive comparison
    st.subheader("⚔️ Offensive Comparison")
    offense_cols = st.columns(3)

    with offense_cols[0]:
        st.markdown(f"**{team1}**")
        if not t1_stats.empty and not t1_game_stats_sum.empty:
            # Get games count for per-game averages
            games_count = t1_game_stats_sum['points_count'].iloc[0]

            rush_yds_total = int(t1_stats['rush_yds'].iloc[0])
            rush_yds_pg = rush_yds_total / games_count if games_count > 0 else 0
            rush_att_total = int(t1_stats['rush_att'].iloc[0])
            rush_att_pg = rush_att_total / games_count if games_count > 0 else 0
            rush_ypa = rush_yds_total / rush_att_total if rush_att_total > 0 else 0

            pass_yds_total = int(t1_stats['pass_yds'].iloc[0])
            pass_yds_pg = pass_yds_total / games_count if games_count > 0 else 0
            pass_comp_pct = 100 * t1_stats['pass_comp'].iloc[0] / t1_stats['pass_att'].iloc[0] if t1_stats['pass_att'].iloc[0] > 0 else 0

            st.metric("Rush Yards", f"{rush_yds_total:,} ({rush_yds_pg:.0f}/game)")
            st.metric("Rush Att", f"{rush_att_total} ({rush_att_pg:.0f}/game)")
            st.metric("Rush YPA", f"{rush_ypa:.2f}")
            st.metric("Pass Yards", f"{pass_yds_total:,} ({pass_yds_pg:.0f}/game)")
            st.metric("Completions", f"{int(t1_stats['pass_comp'].iloc[0])}/{int(t1_stats['pass_att'].iloc[0])}")
            st.metric("Comp %", f"{pass_comp_pct:.1f}%")

    with offense_cols[2]:
        st.markdown(f"**{team2}**")
        if not t2_stats.empty and not t2_game_stats_sum.empty:
            # Get games count for per-game averages
            games_count = t2_game_stats_sum['points_count'].iloc[0]

            rush_yds_total = int(t2_stats['rush_yds'].iloc[0])
            rush_yds_pg = rush_yds_total / games_count if games_count > 0 else 0
            rush_att_total = int(t2_stats['rush_att'].iloc[0])
            rush_att_pg = rush_att_total / games_count if games_count > 0 else 0
            rush_ypa = rush_yds_total / rush_att_total if rush_att_total > 0 else 0

            pass_yds_total = int(t2_stats['pass_yds'].iloc[0])
            pass_yds_pg = pass_yds_total / games_count if games_count > 0 else 0
            pass_comp_pct = 100 * t2_stats['pass_comp'].iloc[0] / t2_stats['pass_att'].iloc[0] if t2_stats['pass_att'].iloc[0] > 0 else 0

            st.metric("Rush Yards", f"{rush_yds_total:,} ({rush_yds_pg:.0f}/game)")
            st.metric("Rush Att", f"{rush_att_total} ({rush_att_pg:.0f}/game)")
            st.metric("Rush YPA", f"{rush_ypa:.2f}")
            st.metric("Pass Yards", f"{pass_yds_total:,} ({pass_yds_pg:.0f}/game)")
            st.metric("Completions", f"{int(t2_stats['pass_comp'].iloc[0])}/{int(t2_stats['pass_att'].iloc[0])}")
            st.metric("Comp %", f"{pass_comp_pct:.1f}%")

    # Top Players Comparison
    if not players_df.empty:
        st.subheader("⭐ Top Players")

        tab1, tab2, tab3 = st.tabs(["Passing Leaders", "Rushing Leaders", "Receiving Leaders"])

        with tab1:
            pass_leaders = players_df[players_df['pass_att'] > 0].groupby(['team', 'player']).agg({
                'pass_cmp': 'sum',
                'pass_att': ['sum', 'count'],
                'pass_yds': 'sum',
                'pass_td': 'sum',
                'pass_int': 'sum',
                'pass_rating': 'mean'
            }).reset_index()
            pass_leaders.columns = ['team', 'player', 'pass_cmp', 'pass_att', 'games', 'pass_yds', 'pass_td', 'pass_int', 'pass_rating']
            pass_leaders['Comp%'] = (pass_leaders['pass_cmp'] / pass_leaders['pass_att'] * 100).round(1)
            pass_leaders = pass_leaders.sort_values('pass_yds', ascending=False)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{team1} Passing**")
                t1_pass = pass_leaders[pass_leaders['team'] == team1].head(3)
                if not t1_pass.empty:
                    t1_pass_display = t1_pass[['player', 'games', 'pass_yds', 'pass_td', 'pass_int', 'Comp%', 'pass_rating']].copy()
                    t1_pass_display.columns = ['Player', 'Games', 'Yards', 'TD', 'INT', 'Comp%', 'Rating']
                    t1_pass_display['Games'] = t1_pass_display['Games'].astype(int)
                    t1_pass_display['Yards'] = t1_pass_display['Yards'].astype(int)
                    t1_pass_display['TD'] = t1_pass_display['TD'].astype(int)
                    t1_pass_display['INT'] = t1_pass_display['INT'].astype(int)
                    t1_pass_display['Rating'] = t1_pass_display['Rating'].round(1)
                    st.dataframe(t1_pass_display, hide_index=True, use_container_width=True)

            with col2:
                st.markdown(f"**{team2} Passing**")
                t2_pass = pass_leaders[pass_leaders['team'] == team2].head(3)
                if not t2_pass.empty:
                    t2_pass_display = t2_pass[['player', 'games', 'pass_yds', 'pass_td', 'pass_int', 'Comp%', 'pass_rating']].copy()
                    t2_pass_display.columns = ['Player', 'Games', 'Yards', 'TD', 'INT', 'Comp%', 'Rating']
                    t2_pass_display['Games'] = t2_pass_display['Games'].astype(int)
                    t2_pass_display['Yards'] = t2_pass_display['Yards'].astype(int)
                    t2_pass_display['TD'] = t2_pass_display['TD'].astype(int)
                    t2_pass_display['INT'] = t2_pass_display['INT'].astype(int)
                    t2_pass_display['Rating'] = t2_pass_display['Rating'].round(1)
                    st.dataframe(t2_pass_display, hide_index=True, use_container_width=True)

        with tab2:
            rush_leaders = players_df[players_df['rush_att'] > 0].groupby(['team', 'player']).agg({
                'rush_att': 'sum',
                'rush_yds': ['sum', 'count'],
                'rush_td': 'sum',
                'rush_long': 'max'
            }).reset_index()
            rush_leaders.columns = ['team', 'player', 'rush_att', 'rush_yds', 'games', 'rush_td', 'rush_long']
            rush_leaders['YPA'] = (rush_leaders['rush_yds'] / rush_leaders['rush_att']).round(1)
            rush_leaders = rush_leaders.sort_values('rush_yds', ascending=False)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{team1} Rushing**")
                t1_rush = rush_leaders[rush_leaders['team'] == team1].head(5)
                if not t1_rush.empty:
                    t1_rush_display = t1_rush[['player', 'games', 'rush_yds', 'rush_att', 'YPA', 'rush_td', 'rush_long']].copy()
                    t1_rush_display.columns = ['Player', 'Games', 'Yards', 'Att', 'YPA', 'TD', 'Long']
                    t1_rush_display['Games'] = t1_rush_display['Games'].astype(int)
                    t1_rush_display['Yards'] = t1_rush_display['Yards'].astype(int)
                    t1_rush_display['Att'] = t1_rush_display['Att'].astype(int)
                    t1_rush_display['TD'] = t1_rush_display['TD'].astype(int)
                    t1_rush_display['Long'] = t1_rush_display['Long'].astype(int)
                    st.dataframe(t1_rush_display, hide_index=True, use_container_width=True)

            with col2:
                st.markdown(f"**{team2} Rushing**")
                t2_rush = rush_leaders[rush_leaders['team'] == team2].head(5)
                if not t2_rush.empty:
                    t2_rush_display = t2_rush[['player', 'games', 'rush_yds', 'rush_att', 'YPA', 'rush_td', 'rush_long']].copy()
                    t2_rush_display.columns = ['Player', 'Games', 'Yards', 'Att', 'YPA', 'TD', 'Long']
                    t2_rush_display['Games'] = t2_rush_display['Games'].astype(int)
                    t2_rush_display['Yards'] = t2_rush_display['Yards'].astype(int)
                    t2_rush_display['Att'] = t2_rush_display['Att'].astype(int)
                    t2_rush_display['TD'] = t2_rush_display['TD'].astype(int)
                    t2_rush_display['Long'] = t2_rush_display['Long'].astype(int)
                    st.dataframe(t2_rush_display, hide_index=True, use_container_width=True)

        with tab3:
            rec_leaders = players_df[players_df['rec'] > 0].groupby(['team', 'player']).agg({
                'rec': 'sum',
                'targets': 'sum',
                'rec_yds': ['sum', 'count'],
                'rec_td': 'sum',
                'rec_long': 'max'
            }).reset_index()
            rec_leaders.columns = ['team', 'player', 'rec', 'targets', 'rec_yds', 'games', 'rec_td', 'rec_long']
            rec_leaders['YPR'] = (rec_leaders['rec_yds'] / rec_leaders['rec']).round(1)
            rec_leaders['Catch%'] = (rec_leaders['rec'] / rec_leaders['targets'] * 100).round(0)
            rec_leaders = rec_leaders.sort_values('rec_yds', ascending=False)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{team1} Receiving**")
                t1_rec = rec_leaders[rec_leaders['team'] == team1].head(5)
                if not t1_rec.empty:
                    t1_rec_display = t1_rec[['player', 'games', 'rec_yds', 'targets', 'rec', 'YPR', 'rec_td', 'Catch%', 'rec_long']].copy()
                    t1_rec_display.columns = ['Player', 'Games', 'Yards', 'Tgt', 'Rec', 'YPR', 'TD', 'Catch%', 'Long']
                    t1_rec_display['Games'] = t1_rec_display['Games'].astype(int)
                    t1_rec_display['Yards'] = t1_rec_display['Yards'].astype(int)
                    t1_rec_display['Tgt'] = t1_rec_display['Tgt'].astype(int)
                    t1_rec_display['Rec'] = t1_rec_display['Rec'].astype(int)
                    t1_rec_display['TD'] = t1_rec_display['TD'].astype(int)
                    t1_rec_display['Catch%'] = t1_rec_display['Catch%'].astype(int)
                    t1_rec_display['Long'] = t1_rec_display['Long'].astype(int)
                    st.dataframe(t1_rec_display, hide_index=True, use_container_width=True)

            with col2:
                st.markdown(f"**{team2} Receiving**")
                t2_rec = rec_leaders[rec_leaders['team'] == team2].head(5)
                if not t2_rec.empty:
                    t2_rec_display = t2_rec[['player', 'games', 'rec_yds', 'targets', 'rec', 'YPR', 'rec_td', 'Catch%', 'rec_long']].copy()
                    t2_rec_display.columns = ['Player', 'Games', 'Yards', 'Tgt', 'Rec', 'YPR', 'TD', 'Catch%', 'Long']
                    t2_rec_display['Games'] = t2_rec_display['Games'].astype(int)
                    t2_rec_display['Yards'] = t2_rec_display['Yards'].astype(int)
                    t2_rec_display['Tgt'] = t2_rec_display['Tgt'].astype(int)
                    t2_rec_display['Rec'] = t2_rec_display['Rec'].astype(int)
                    t2_rec_display['TD'] = t2_rec_display['TD'].astype(int)
                    t2_rec_display['Catch%'] = t2_rec_display['Catch%'].astype(int)
                    t2_rec_display['Long'] = t2_rec_display['Long'].astype(int)
                    st.dataframe(t2_rec_display, hide_index=True, use_container_width=True)

        st.divider()

        # Player Statistical Analysis
        st.subheader("📊 Player Statistical Analysis")
        st.caption("Individual player performance breakdown - identifying likely statistical leaders")

        tab1, tab2, tab3 = st.tabs(["Passing Leaders", "Rushing Leaders", "Receiving Leaders"])

        with tab1:
            # Individual passing statistics
            pass_stats = players_df[players_df['pass_att'] > 0].groupby(['team', 'player']).agg({
                'pass_yds': ['mean', 'median', 'max', 'min', 'sum', 'count'],
                'pass_td': ['mean', 'sum'],
                'pass_int': ['sum'],
                'pass_cmp': ['sum'],
                'pass_att': ['sum'],
                'pass_rating': ['mean']
            }).reset_index()

            # Flatten column names
            pass_stats.columns = ['team', 'player', 'avg_yds', 'med_yds', 'max_yds', 'min_yds', 'total_yds', 'games',
                                  'avg_td', 'total_td', 'total_int', 'total_cmp', 'total_att', 'avg_rating']
            pass_stats = pass_stats.sort_values('total_yds', ascending=False)

            # Get game-by-game data with opponents for last 3 games
            # Note: Don't select 'week' from games table since players_df already has it
            games_query = f"SELECT game_id, home_team_abbr, away_team_abbr FROM games WHERE season={season}"
            games_info = query(games_query)
            pass_players_with_games = players_df[players_df['pass_att'] > 0].merge(games_info, on='game_id', how='left')

            # Add opponent column
            def get_pass_opponent(row):
                if row['team'] == row['home_team_abbr']:
                    return row['away_team_abbr']
                else:
                    return row['home_team_abbr']

            def is_pass_away_game(row):
                return row['team'] == row['away_team_abbr']

            pass_players_with_games['opponent'] = pass_players_with_games.apply(get_pass_opponent, axis=1)
            pass_players_with_games['is_away'] = pass_players_with_games.apply(is_pass_away_game, axis=1)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### {team1} Passers")
                t1_passers = pass_stats[pass_stats['team'] == team1].head(3)
                if not t1_passers.empty:
                    t1_pass_display = t1_passers.copy()
                    t1_pass_display['Comp%'] = (t1_pass_display['total_cmp'] / t1_pass_display['total_att'] * 100).round(1)
                    t1_pass_display = t1_pass_display[['player', 'games', 'total_yds', 'avg_yds', 'total_td', 'avg_td', 'total_int', 'Comp%', 'avg_rating']].copy()
                    t1_pass_display.columns = ['Player', 'Games', 'Tot Yds', 'Avg Yds', 'Tot TD', 'Avg TD', 'INT', 'Comp%', 'Rating']
                    t1_pass_display['Games'] = t1_pass_display['Games'].astype(int)
                    t1_pass_display['Tot Yds'] = t1_pass_display['Tot Yds'].astype(int)
                    t1_pass_display['Avg Yds'] = t1_pass_display['Avg Yds'].round(1)
                    t1_pass_display['Tot TD'] = t1_pass_display['Tot TD'].astype(int)
                    t1_pass_display['Avg TD'] = t1_pass_display['Avg TD'].round(1)
                    t1_pass_display['INT'] = t1_pass_display['INT'].astype(int)
                    t1_pass_display['Rating'] = t1_pass_display['Rating'].round(1)
                    st.dataframe(t1_pass_display, hide_index=True, use_container_width=True)

                    # Last 3 games details for QBs
                    with st.expander("📊 Last 3 Games Details"):
                        for _, p in t1_passers.iterrows():
                            player_games = pass_players_with_games[(pass_players_with_games['team'] == p['team']) &
                                                                    (pass_players_with_games['player'] == p['player'])].sort_values('week')
                            if len(player_games) >= 3:
                                last_3 = player_games.tail(3)
                                st.markdown(f"**{p['player']}**")

                                # Build dataframe for last 3 games
                                game_data = []
                                for _, row in last_3.iterrows():
                                    team_total_att = pass_players_with_games[(pass_players_with_games['game_id'] == row['game_id']) &
                                                                             (pass_players_with_games['team'] == row['team'])]['pass_att'].sum()
                                    att_pct = (row['pass_att'] / team_total_att * 100) if team_total_att > 0 else 0
                                    location = "@ " if row['is_away'] else "vs "
                                    cmp_att = f"{int(row['pass_cmp'])}/{int(row['pass_att'])}"
                                    game_data.append({
                                        'Week': int(row['week']),
                                        'Opponent': f"{location}{row['opponent']}",
                                        'Yards': int(row['pass_yds']),
                                        'Cmp/Att': cmp_att,
                                        'TD': int(row['pass_td']),
                                        'INT': int(row['pass_int']),
                                        'Rating': round(row['pass_rating'], 1) if pd.notna(row['pass_rating']) else 0.0,
                                        'Att %': f"{att_pct:.0f}%"
                                    })

                                last_3_df = pd.DataFrame(game_data)
                                st.dataframe(last_3_df, hide_index=True, use_container_width=True)
                                st.markdown("---")

            with col2:
                st.markdown(f"### {team2} Passers")
                t2_passers = pass_stats[pass_stats['team'] == team2].head(3)
                if not t2_passers.empty:
                    t2_pass_display = t2_passers.copy()
                    t2_pass_display['Comp%'] = (t2_pass_display['total_cmp'] / t2_pass_display['total_att'] * 100).round(1)
                    t2_pass_display = t2_pass_display[['player', 'games', 'total_yds', 'avg_yds', 'total_td', 'avg_td', 'total_int', 'Comp%', 'avg_rating']].copy()
                    t2_pass_display.columns = ['Player', 'Games', 'Tot Yds', 'Avg Yds', 'Tot TD', 'Avg TD', 'INT', 'Comp%', 'Rating']
                    t2_pass_display['Games'] = t2_pass_display['Games'].astype(int)
                    t2_pass_display['Tot Yds'] = t2_pass_display['Tot Yds'].astype(int)
                    t2_pass_display['Avg Yds'] = t2_pass_display['Avg Yds'].round(1)
                    t2_pass_display['Tot TD'] = t2_pass_display['Tot TD'].astype(int)
                    t2_pass_display['Avg TD'] = t2_pass_display['Avg TD'].round(1)
                    t2_pass_display['INT'] = t2_pass_display['INT'].astype(int)
                    t2_pass_display['Rating'] = t2_pass_display['Rating'].round(1)
                    st.dataframe(t2_pass_display, hide_index=True, use_container_width=True)

                    # Last 3 games details for QBs
                    with st.expander("📊 Last 3 Games Details"):
                        for _, p in t2_passers.iterrows():
                            player_games = pass_players_with_games[(pass_players_with_games['team'] == p['team']) &
                                                                    (pass_players_with_games['player'] == p['player'])].sort_values('week')
                            if len(player_games) >= 3:
                                last_3 = player_games.tail(3)
                                st.markdown(f"**{p['player']}**")

                                # Build dataframe for last 3 games
                                game_data = []
                                for _, row in last_3.iterrows():
                                    team_total_att = pass_players_with_games[(pass_players_with_games['game_id'] == row['game_id']) &
                                                                             (pass_players_with_games['team'] == row['team'])]['pass_att'].sum()
                                    att_pct = (row['pass_att'] / team_total_att * 100) if team_total_att > 0 else 0
                                    location = "@ " if row['is_away'] else "vs "
                                    cmp_att = f"{int(row['pass_cmp'])}/{int(row['pass_att'])}"
                                    game_data.append({
                                        'Week': int(row['week']),
                                        'Opponent': f"{location}{row['opponent']}",
                                        'Yards': int(row['pass_yds']),
                                        'Cmp/Att': cmp_att,
                                        'TD': int(row['pass_td']),
                                        'INT': int(row['pass_int']),
                                        'Rating': round(row['pass_rating'], 1) if pd.notna(row['pass_rating']) else 0.0,
                                        'Att %': f"{att_pct:.0f}%"
                                    })

                                last_3_df = pd.DataFrame(game_data)
                                st.dataframe(last_3_df, hide_index=True, use_container_width=True)
                                st.markdown("---")

        with tab2:
            # Individual rushing statistics
            rush_stats = players_df[players_df['rush_att'] > 0].groupby(['team', 'player']).agg({
                'rush_yds': ['mean', 'median', 'max', 'min', 'sum', 'count', 'std'],
                'rush_att': ['sum', 'min', 'max'],
                'rush_td': ['mean', 'sum'],
                'rush_long': ['max']
            }).reset_index()

            # Flatten column names
            rush_stats.columns = ['team', 'player', 'avg_yds', 'med_yds', 'max_yds', 'min_yds', 'total_yds', 'games', 'std_yds',
                                  'total_att', 'min_att', 'max_att', 'avg_td', 'total_td', 'long']
            rush_stats = rush_stats.sort_values('total_yds', ascending=False)

            # Get game-by-game data with opponents for last 3 games
            # Note: Don't select 'week' from games table since players_df already has it
            games_query = f"SELECT game_id, home_team_abbr, away_team_abbr FROM games WHERE season={season}"
            games_info = query(games_query)
            players_with_games = players_df[players_df['rush_att'] > 0].merge(games_info, on='game_id', how='left')

            # Get touchdown_scorers data for first TD flags
            td_scorers_query = f"SELECT game_id, player, team, touchdown_type, first_td_game, first_td_for_team FROM touchdown_scorers WHERE season={season} AND touchdown_type='Rushing'"
            td_scorers = query(td_scorers_query)
            if not td_scorers.empty:
                # Aggregate by game_id, player, team to handle multiple TDs in same game
                td_scorers_agg = td_scorers.groupby(['game_id', 'player', 'team']).agg({
                    'first_td_game': 'max',
                    'first_td_for_team': 'max'
                }).reset_index()
                players_with_games = players_with_games.merge(
                    td_scorers_agg,
                    on=['game_id', 'player', 'team'],
                    how='left'
                )
            else:
                players_with_games['first_td_game'] = 0
                players_with_games['first_td_for_team'] = 0

            # Add opponent column
            def get_opponent(row):
                if row['team'] == row['home_team_abbr']:
                    return row['away_team_abbr']
                else:
                    return row['home_team_abbr']

            def is_away_game(row):
                return row['team'] == row['away_team_abbr']

            players_with_games['opponent'] = players_with_games.apply(get_opponent, axis=1)
            players_with_games['is_away'] = players_with_games.apply(is_away_game, axis=1)

            # Fill NaN values in first TD columns with 0
            players_with_games['first_td_game'] = players_with_games['first_td_game'].fillna(0).astype(int)
            players_with_games['first_td_for_team'] = players_with_games['first_td_for_team'].fillna(0).astype(int)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### {team1} Rushers")
                t1_rushers = rush_stats[rush_stats['team'] == team1].head(5)
                if not t1_rushers.empty:
                    t1_rush_display = t1_rushers.copy()
                    t1_rush_display['YPC'] = (t1_rush_display['total_yds'] / t1_rush_display['total_att']).round(2)
                    t1_rush_display['Att/G'] = (t1_rush_display['total_att'] / t1_rush_display['games']).round(1)
                    t1_rush_display = t1_rush_display[['player', 'games', 'total_yds', 'avg_yds', 'Att/G', 'total_td', 'YPC', 'long']].copy()
                    t1_rush_display.columns = ['Player', 'Games', 'Tot Yds', 'Avg Yds', 'Att/G', 'TD', 'YPC', 'Long']
                    t1_rush_display['Games'] = t1_rush_display['Games'].astype(int)
                    t1_rush_display['Tot Yds'] = t1_rush_display['Tot Yds'].astype(int)
                    t1_rush_display['Avg Yds'] = t1_rush_display['Avg Yds'].round(1)
                    t1_rush_display['TD'] = t1_rush_display['TD'].astype(int)
                    t1_rush_display['Long'] = t1_rush_display['Long'].astype(int)
                    st.dataframe(t1_rush_display, hide_index=True, use_container_width=True)

                    # Last 3 games details
                    with st.expander("📊 Last 3 Games Details"):
                        for _, p in t1_rushers.iterrows():
                            player_games = players_with_games[(players_with_games['team'] == p['team']) &
                                                             (players_with_games['player'] == p['player'])].sort_values('week')
                            if len(player_games) >= 3:
                                last_3 = player_games.tail(3)
                                st.markdown(f"**{p['player']}**")

                                # Build dataframe for last 3 games
                                game_data = []
                                for _, row in last_3.iterrows():
                                    team_total = players_with_games[(players_with_games['game_id'] == row['game_id']) &
                                                                   (players_with_games['team'] == row['team'])]['rush_att'].sum()
                                    att_pct = (row['rush_att'] / team_total * 100) if team_total > 0 else 0
                                    location = "@ " if row['is_away'] else "vs "
                                    game_data.append({
                                        'Week': int(row['week']),
                                        'Opponent': f"{location}{row['opponent']}",
                                        'Yards': int(row['rush_yds']),
                                        'Att': int(row['rush_att']),
                                        'TD': int(row['rush_td']),
                                        '1st Game': 'Yes' if row['first_td_game'] == 1 else 'No',
                                        '1st Team': 'Yes' if row['first_td_for_team'] == 1 else 'No',
                                        'Att %': f"{att_pct:.0f}%"
                                    })

                                last_3_df = pd.DataFrame(game_data)
                                st.dataframe(last_3_df, hide_index=True, use_container_width=True)
                                st.markdown("---")

            with col2:
                st.markdown(f"### {team2} Rushers")
                t2_rushers = rush_stats[rush_stats['team'] == team2].head(5)
                if not t2_rushers.empty:
                    t2_rush_display = t2_rushers.copy()
                    t2_rush_display['YPC'] = (t2_rush_display['total_yds'] / t2_rush_display['total_att']).round(2)
                    t2_rush_display['Att/G'] = (t2_rush_display['total_att'] / t2_rush_display['games']).round(1)
                    t2_rush_display = t2_rush_display[['player', 'games', 'total_yds', 'avg_yds', 'Att/G', 'total_td', 'YPC', 'long']].copy()
                    t2_rush_display.columns = ['Player', 'Games', 'Tot Yds', 'Avg Yds', 'Att/G', 'TD', 'YPC', 'Long']
                    t2_rush_display['Games'] = t2_rush_display['Games'].astype(int)
                    t2_rush_display['Tot Yds'] = t2_rush_display['Tot Yds'].astype(int)
                    t2_rush_display['Avg Yds'] = t2_rush_display['Avg Yds'].round(1)
                    t2_rush_display['TD'] = t2_rush_display['TD'].astype(int)
                    t2_rush_display['Long'] = t2_rush_display['Long'].astype(int)
                    st.dataframe(t2_rush_display, hide_index=True, use_container_width=True)

                    # Last 3 games details
                    with st.expander("📊 Last 3 Games Details"):
                        for _, p in t2_rushers.iterrows():
                            player_games = players_with_games[(players_with_games['team'] == p['team']) &
                                                             (players_with_games['player'] == p['player'])].sort_values('week')
                            if len(player_games) >= 3:
                                last_3 = player_games.tail(3)
                                st.markdown(f"**{p['player']}**")

                                # Build dataframe for last 3 games
                                game_data = []
                                for _, row in last_3.iterrows():
                                    team_total = players_with_games[(players_with_games['game_id'] == row['game_id']) &
                                                                   (players_with_games['team'] == row['team'])]['rush_att'].sum()
                                    att_pct = (row['rush_att'] / team_total * 100) if team_total > 0 else 0
                                    location = "@ " if row['is_away'] else "vs "
                                    game_data.append({
                                        'Week': int(row['week']),
                                        'Opponent': f"{location}{row['opponent']}",
                                        'Yards': int(row['rush_yds']),
                                        'Att': int(row['rush_att']),
                                        'TD': int(row['rush_td']),
                                        '1st Game': 'Yes' if row['first_td_game'] == 1 else 'No',
                                        '1st Team': 'Yes' if row['first_td_for_team'] == 1 else 'No',
                                        'Att %': f"{att_pct:.0f}%"
                                    })

                                last_3_df = pd.DataFrame(game_data)
                                st.dataframe(last_3_df, hide_index=True, use_container_width=True)
                                st.markdown("---")

        with tab3:
            # Individual receiving statistics
            rec_stats = players_df[players_df['rec'] > 0].groupby(['team', 'player']).agg({
                'rec_yds': ['mean', 'median', 'max', 'min', 'sum', 'count', 'std'],
                'rec': ['sum', 'min', 'max'],
                'targets': ['sum'],
                'rec_td': ['mean', 'sum'],
                'rec_long': ['max']
            }).reset_index()

            # Flatten column names
            rec_stats.columns = ['team', 'player', 'avg_yds', 'med_yds', 'max_yds', 'min_yds', 'total_yds', 'games', 'std_yds',
                                 'total_rec', 'min_rec', 'max_rec', 'total_tgt', 'avg_td', 'total_td', 'long']
            rec_stats = rec_stats.sort_values('total_yds', ascending=False)

            # Get game-by-game data with opponents for last 3 games
            # Note: Don't select 'week' from games table since players_df already has it
            rec_games_query = f"SELECT game_id, home_team_abbr, away_team_abbr FROM games WHERE season={season}"
            rec_games_info = query(rec_games_query)
            rec_players_with_games = players_df[players_df['rec'] > 0].merge(rec_games_info, on='game_id', how='left')

            # Get touchdown_scorers data for first TD flags
            rec_td_scorers_query = f"SELECT game_id, player, team, touchdown_type, first_td_game, first_td_for_team FROM touchdown_scorers WHERE season={season} AND touchdown_type='Receiving'"
            rec_td_scorers = query(rec_td_scorers_query)
            if not rec_td_scorers.empty:
                # Aggregate by game_id, player, team to handle multiple TDs in same game
                rec_td_scorers_agg = rec_td_scorers.groupby(['game_id', 'player', 'team']).agg({
                    'first_td_game': 'max',
                    'first_td_for_team': 'max'
                }).reset_index()
                rec_players_with_games = rec_players_with_games.merge(
                    rec_td_scorers_agg,
                    on=['game_id', 'player', 'team'],
                    how='left'
                )
            else:
                rec_players_with_games['first_td_game'] = 0
                rec_players_with_games['first_td_for_team'] = 0

            # Add opponent column
            rec_players_with_games['opponent'] = rec_players_with_games.apply(
                lambda row: row['away_team_abbr'] if row['team'] == row['home_team_abbr'] else row['home_team_abbr'], axis=1
            )
            rec_players_with_games['is_away'] = rec_players_with_games.apply(
                lambda row: row['team'] == row['away_team_abbr'], axis=1
            )

            # Fill NaN values in first TD columns with 0
            rec_players_with_games['first_td_game'] = rec_players_with_games['first_td_game'].fillna(0).astype(int)
            rec_players_with_games['first_td_for_team'] = rec_players_with_games['first_td_for_team'].fillna(0).astype(int)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### {team1} Receivers")
                t1_receivers = rec_stats[rec_stats['team'] == team1].head(5)
                if not t1_receivers.empty:
                    t1_rec_display = t1_receivers.copy()
                    t1_rec_display['YPR'] = (t1_rec_display['total_yds'] / t1_rec_display['total_rec']).round(1)
                    t1_rec_display['Catch%'] = (t1_rec_display['total_rec'] / t1_rec_display['total_tgt'] * 100).round(1)
                    t1_rec_display['Rec/G'] = (t1_rec_display['total_rec'] / t1_rec_display['games']).round(1)
                    t1_rec_display = t1_rec_display[['player', 'games', 'total_yds', 'avg_yds', 'total_rec', 'Rec/G', 'total_td', 'YPR', 'Catch%', 'long']].copy()
                    t1_rec_display.columns = ['Player', 'Games', 'Tot Yds', 'Avg Yds', 'Tot Rec', 'Rec/G', 'TD', 'YPR', 'Catch%', 'Long']
                    t1_rec_display['Games'] = t1_rec_display['Games'].astype(int)
                    t1_rec_display['Tot Yds'] = t1_rec_display['Tot Yds'].astype(int)
                    t1_rec_display['Avg Yds'] = t1_rec_display['Avg Yds'].round(1)
                    t1_rec_display['Tot Rec'] = t1_rec_display['Tot Rec'].astype(int)
                    t1_rec_display['TD'] = t1_rec_display['TD'].astype(int)
                    t1_rec_display['Long'] = t1_rec_display['Long'].astype(int)
                    st.dataframe(t1_rec_display, hide_index=True, use_container_width=True)

                    # Last 3 games details
                    with st.expander("📊 Last 3 Games Details"):
                        for _, p in t1_receivers.iterrows():
                            player_games = rec_players_with_games[(rec_players_with_games['team'] == p['team']) &
                                                                  (rec_players_with_games['player'] == p['player'])].sort_values('week')
                            if len(player_games) >= 3:
                                last_3 = player_games.tail(3)
                                st.markdown(f"**{p['player']}**")

                                # Build dataframe for last 3 games
                                game_data = []
                                for _, row in last_3.iterrows():
                                    team_total_tgt = rec_players_with_games[(rec_players_with_games['game_id'] == row['game_id']) &
                                                                            (rec_players_with_games['team'] == row['team'])]['targets'].sum()
                                    team_total_rec = rec_players_with_games[(rec_players_with_games['game_id'] == row['game_id']) &
                                                                            (rec_players_with_games['team'] == row['team'])]['rec'].sum()
                                    tgt_pct = (row['targets'] / team_total_tgt * 100) if team_total_tgt > 0 else 0
                                    rec_pct = (row['rec'] / team_total_rec * 100) if team_total_rec > 0 else 0
                                    location = "@ " if row['is_away'] else "vs "
                                    game_data.append({
                                        'Week': int(row['week']),
                                        'Opponent': f"{location}{row['opponent']}",
                                        'Yards': int(row['rec_yds']),
                                        'Rec': int(row['rec']),
                                        'TD': int(row['rec_td']),
                                        '1st Game': 'Yes' if row['first_td_game'] == 1 else 'No',
                                        '1st Team': 'Yes' if row['first_td_for_team'] == 1 else 'No',
                                        'Tgt %': f"{tgt_pct:.0f}%",
                                        'Rec %': f"{rec_pct:.0f}%"
                                    })

                                last_3_df = pd.DataFrame(game_data)
                                st.dataframe(last_3_df, hide_index=True, use_container_width=True)
                                st.markdown("---")

            with col2:
                st.markdown(f"### {team2} Receivers")
                t2_receivers = rec_stats[rec_stats['team'] == team2].head(5)
                if not t2_receivers.empty:
                    t2_rec_display = t2_receivers.copy()
                    t2_rec_display['YPR'] = (t2_rec_display['total_yds'] / t2_rec_display['total_rec']).round(1)
                    t2_rec_display['Catch%'] = (t2_rec_display['total_rec'] / t2_rec_display['total_tgt'] * 100).round(1)
                    t2_rec_display['Rec/G'] = (t2_rec_display['total_rec'] / t2_rec_display['games']).round(1)
                    t2_rec_display = t2_rec_display[['player', 'games', 'total_yds', 'avg_yds', 'total_rec', 'Rec/G', 'total_td', 'YPR', 'Catch%', 'long']].copy()
                    t2_rec_display.columns = ['Player', 'Games', 'Tot Yds', 'Avg Yds', 'Tot Rec', 'Rec/G', 'TD', 'YPR', 'Catch%', 'Long']
                    t2_rec_display['Games'] = t2_rec_display['Games'].astype(int)
                    t2_rec_display['Tot Yds'] = t2_rec_display['Tot Yds'].astype(int)
                    t2_rec_display['Avg Yds'] = t2_rec_display['Avg Yds'].round(1)
                    t2_rec_display['Tot Rec'] = t2_rec_display['Tot Rec'].astype(int)
                    t2_rec_display['TD'] = t2_rec_display['TD'].astype(int)
                    t2_rec_display['Long'] = t2_rec_display['Long'].astype(int)
                    st.dataframe(t2_rec_display, hide_index=True, use_container_width=True)

                    # Last 3 games details
                    with st.expander("📊 Last 3 Games Details"):
                        for _, p in t2_receivers.iterrows():
                            player_games = rec_players_with_games[(rec_players_with_games['team'] == p['team']) &
                                                                  (rec_players_with_games['player'] == p['player'])].sort_values('week')
                            if len(player_games) >= 3:
                                last_3 = player_games.tail(3)
                                st.markdown(f"**{p['player']}**")

                                # Build dataframe for last 3 games
                                game_data = []
                                for _, row in last_3.iterrows():
                                    team_total_tgt = rec_players_with_games[(rec_players_with_games['game_id'] == row['game_id']) &
                                                                            (rec_players_with_games['team'] == row['team'])]['targets'].sum()
                                    team_total_rec = rec_players_with_games[(rec_players_with_games['game_id'] == row['game_id']) &
                                                                            (rec_players_with_games['team'] == row['team'])]['rec'].sum()
                                    tgt_pct = (row['targets'] / team_total_tgt * 100) if team_total_tgt > 0 else 0
                                    rec_pct = (row['rec'] / team_total_rec * 100) if team_total_rec > 0 else 0
                                    location = "@ " if row['is_away'] else "vs "
                                    game_data.append({
                                        'Week': int(row['week']),
                                        'Opponent': f"{location}{row['opponent']}",
                                        'Yards': int(row['rec_yds']),
                                        'Rec': int(row['rec']),
                                        'TD': int(row['rec_td']),
                                        '1st Game': 'Yes' if row['first_td_game'] == 1 else 'No',
                                        '1st Team': 'Yes' if row['first_td_for_team'] == 1 else 'No',
                                        'Tgt %': f"{tgt_pct:.0f}%",
                                        'Rec %': f"{rec_pct:.0f}%"
                                    })

                                last_3_df = pd.DataFrame(game_data)
                                st.dataframe(last_3_df, hide_index=True, use_container_width=True)
                                st.markdown("---")

        st.divider()

        # TD Against Section
        st.subheader("🛡️ TD Against")
        st.caption("Touchdowns allowed by each team's defense - lower numbers indicate better defensive performance")

        # First, get league-wide TD Against stats for all teams to calculate rankings
        league_td_sql = """
            WITH team_games AS (
                SELECT
                    team_abbr,
                    COUNT(DISTINCT game_id) as games_played
                FROM (
                    SELECT home_team_abbr as team_abbr, game_id FROM games WHERE season = ? {week_filter}
                    UNION ALL
                    SELECT away_team_abbr as team_abbr, game_id FROM games WHERE season = ? {week_filter}
                )
                GROUP BY team_abbr
            ),
            tds_by_team AS (
                SELECT
                    CASE
                        WHEN ts.team = g.home_team_abbr THEN g.away_team_abbr
                        WHEN ts.team = g.away_team_abbr THEN g.home_team_abbr
                    END as defense_team,
                    SUM(CASE WHEN ts.touchdown_type = 'Rushing' THEN 1 ELSE 0 END) as rush_tds_allowed,
                    SUM(CASE WHEN ts.touchdown_type = 'Receiving' THEN 1 ELSE 0 END) as rec_tds_allowed
                FROM touchdown_scorers ts
                JOIN games g ON ts.game_id = g.game_id
                WHERE ts.season = ? {week_filter}
                GROUP BY defense_team
            )
            SELECT
                tg.team_abbr,
                tg.games_played,
                COALESCE(tbt.rush_tds_allowed, 0) as rush_tds_allowed,
                COALESCE(tbt.rec_tds_allowed, 0) as rec_tds_allowed,
                COALESCE(tbt.rush_tds_allowed, 0) + COALESCE(tbt.rec_tds_allowed, 0) as total_tds_allowed
            FROM team_games tg
            LEFT JOIN tds_by_team tbt ON tg.team_abbr = tbt.defense_team
        """

        # Add week filter if specified
        week_filter = f"AND week <= {week}" if week else ""
        league_td_sql = league_td_sql.replace("{week_filter}", week_filter)

        # Execute query to get all teams' defensive stats
        league_td_df = query(league_td_sql, (season, season, season))

        if not league_td_df.empty:
            # Calculate per-game rates to handle bye weeks
            league_td_df['rush_tds_per_game'] = (league_td_df['rush_tds_allowed'] / league_td_df['games_played']).round(2)
            league_td_df['rec_tds_per_game'] = (league_td_df['rec_tds_allowed'] / league_td_df['games_played']).round(2)
            league_td_df['total_tds_per_game'] = (league_td_df['total_tds_allowed'] / league_td_df['games_played']).round(2)

            # Calculate rankings (lower is better for defense)
            league_td_df['rush_rank'] = league_td_df['rush_tds_per_game'].rank(method='min').astype(int)
            league_td_df['rec_rank'] = league_td_df['rec_tds_per_game'].rank(method='min').astype(int)
            league_td_df['total_rank'] = league_td_df['total_tds_per_game'].rank(method='min').astype(int)

            # Get stats for both teams
            t1_stats = league_td_df[league_td_df['team_abbr'] == team1]
            t2_stats = league_td_df[league_td_df['team_abbr'] == team2]

        # Display TD Against stats with rankings
        td_col1, td_col2, td_col3 = st.columns([2, 1, 2])

        with td_col1:
            st.markdown(f"### {team1} Defense")
            if not t1_stats.empty:
                row = t1_stats.iloc[0]
                games = int(row['games_played'])
                rush_tds = int(row['rush_tds_allowed'])
                rec_tds = int(row['rec_tds_allowed'])
                total_tds = int(row['total_tds_allowed'])
                rush_rank = int(row['rush_rank'])
                rec_rank = int(row['rec_rank'])
                total_rank = int(row['total_rank'])
                rush_per_game = row['rush_tds_per_game']
                rec_per_game = row['rec_tds_per_game']
                total_per_game = row['total_tds_per_game']

                st.metric(
                    f"Rush TDs Allowed (Rank: {rush_rank})",
                    f"{rush_tds} ({rush_per_game}/game)",
                    help=f"Touchdowns allowed on rushing plays - {games} games played"
                )
                st.metric(
                    f"Receiving TDs Allowed (Rank: {rec_rank})",
                    f"{rec_tds} ({rec_per_game}/game)",
                    help=f"Touchdowns allowed on passing plays - {games} games played"
                )
                st.metric(
                    f"Total TDs Allowed (Rank: {total_rank})",
                    f"{total_tds} ({total_per_game}/game)",
                    help=f"Combined rush + receiving TDs allowed - {games} games played"
                )
            else:
                st.info("No defensive TD data available")

        with td_col2:
            st.markdown("###  ")
            st.markdown("<div style='text-align: center; padding-top: 40px; font-size: 24px;'>⚔️</div>", unsafe_allow_html=True)

        with td_col3:
            st.markdown(f"### {team2} Defense")
            if not t2_stats.empty:
                row = t2_stats.iloc[0]
                games = int(row['games_played'])
                rush_tds = int(row['rush_tds_allowed'])
                rec_tds = int(row['rec_tds_allowed'])
                total_tds = int(row['total_tds_allowed'])
                rush_rank = int(row['rush_rank'])
                rec_rank = int(row['rec_rank'])
                total_rank = int(row['total_rank'])
                rush_per_game = row['rush_tds_per_game']
                rec_per_game = row['rec_tds_per_game']
                total_per_game = row['total_tds_per_game']

                st.metric(
                    f"Rush TDs Allowed (Rank: {rush_rank})",
                    f"{rush_tds} ({rush_per_game}/game)",
                    help=f"Touchdowns allowed on rushing plays - {games} games played"
                )
                st.metric(
                    f"Receiving TDs Allowed (Rank: {rec_rank})",
                    f"{rec_tds} ({rec_per_game}/game)",
                    help=f"Touchdowns allowed on passing plays - {games} games played"
                )
                st.metric(
                    f"Total TDs Allowed (Rank: {total_rank})",
                    f"{total_tds} ({total_per_game}/game)",
                    help=f"Combined rush + receiving TDs allowed - {games} games played"
                )
            else:
                st.info("No defensive TD data available")

        st.divider()

        # Expected Stats Section
        st.subheader("🎯 Expected Stats")
        st.caption("Projected player performance based on season medians adjusted for opponent defensive strength")

        # Get league-wide defensive averages
        league_def_sql = """
            SELECT
                AVG(pass_yds) as league_avg_pass,
                AVG(rush_yds) as league_avg_rush,
                AVG(rec_yds) as league_avg_rec
            FROM player_box_score
            WHERE season = ?
        """
        league_def = query(league_def_sql, (season,))

        if not league_def.empty and not players_df.empty:
            league_avg_pass = league_def['league_avg_pass'].iloc[0] if league_def['league_avg_pass'].iloc[0] else 1
            league_avg_rush = league_def['league_avg_rush'].iloc[0] if league_def['league_avg_rush'].iloc[0] else 1
            league_avg_rec = league_def['league_avg_rec'].iloc[0] if league_def['league_avg_rec'].iloc[0] else 1

            # Get opponent defensive stats (team1 faces team2's defense and vice versa)
            def get_opponent_def_stats(opponent_team, season, week):
                """Get opponent's defensive stats (yards allowed)."""
                sql = """
                    SELECT
                        AVG(pass_yds) as avg_pass_allowed,
                        AVG(rush_yds) as avg_rush_allowed,
                        AVG(rec_yds) as avg_rec_allowed
                    FROM player_box_score
                    WHERE season = ? AND team != ?
                """
                params = [season, opponent_team]
                if week:
                    sql += " AND week <= ?"
                    params.append(week)
                result = query(sql, tuple(params))
                if result.empty:
                    return {'pass': 1, 'rush': 1, 'rec': 1}
                return {
                    'pass': result['avg_pass_allowed'].iloc[0] if result['avg_pass_allowed'].iloc[0] else 1,
                    'rush': result['avg_rush_allowed'].iloc[0] if result['avg_rush_allowed'].iloc[0] else 1,
                    'rec': result['avg_rec_allowed'].iloc[0] if result['avg_rec_allowed'].iloc[0] else 1
                }

            t1_opponent_def = get_opponent_def_stats(team2, season, week)
            t2_opponent_def = get_opponent_def_stats(team1, season, week)

            # Calculate adjustment factors
            t1_pass_factor = t1_opponent_def['pass'] / league_avg_pass if league_avg_pass > 0 else 1
            t1_rush_factor = t1_opponent_def['rush'] / league_avg_rush if league_avg_rush > 0 else 1
            t1_rec_factor = t1_opponent_def['rec'] / league_avg_rec if league_avg_rec > 0 else 1

            t2_pass_factor = t2_opponent_def['pass'] / league_avg_pass if league_avg_pass > 0 else 1
            t2_rush_factor = t2_opponent_def['rush'] / league_avg_rush if league_avg_rush > 0 else 1
            t2_rec_factor = t2_opponent_def['rec'] / league_avg_rec if league_avg_rec > 0 else 1

            # Smart Aggregation Strategy Selector
            st.markdown("**📊 Projection Strategy**")
            strategy_col1, strategy_col2 = st.columns([3, 2])

            with strategy_col1:
                projection_strategy = st.selectbox(
                    "Select how to calculate expected stats:",
                    options=['season_avg', 'recent_form', 'conservative', 'optimistic'],
                    format_func=lambda x: {
                        'season_avg': '📈 Season Average (Balanced - uses full season average)',
                        'recent_form': '🔥 Recent Form (Last 3-5 games weighted heavily)',
                        'conservative': '🛡️ Conservative (25th percentile - safer floor)',
                        'optimistic': '🚀 Optimistic (75th percentile - upside focus)'
                    }[x],
                    index=0,
                    key='projection_strategy',
                    help="Different strategies for calculating expected performance:\n"
                         "• Season Average: Uses mean (average) of all games - matches player's avg stats\n"
                         "• Recent Form: Weights recent games more heavily (good for hot/cold streaks)\n"
                         "• Conservative: Lower percentile projections (good for floor analysis)\n"
                         "• Optimistic: Higher percentile projections (good for ceiling analysis)"
                )

            with strategy_col2:
                st.caption("**Strategy Impact:**")
                strategy_descriptions = {
                    'season_avg': "Balanced view of full season",
                    'recent_form': "Emphasizes last 3-5 games",
                    'conservative': "Lower floor, safer projections",
                    'optimistic': "Higher ceiling, upside scenarios"
                }
                st.caption(f"_{strategy_descriptions.get(projection_strategy, '')}_")

            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.info("**Confidence:** ✅ High (8+ games) | ⚠️ Medium (4-7 games) | ❗ Low (1-3 games) | ❌ No data")
            with col_info2:
                st.info("**Transactions:** ▶️ Joined team | ◀️ Left team | Week shown (e.g., W3 = Week 3)")

            st.divider()

            tab1, tab2, tab3 = st.tabs(["Expected Passing", "Expected Rushing", "Expected Receiving"])

            with tab1:
                # Expected Passing Stats
                st.markdown("### Expected Passing Performance")

                # Injury Controls
                control_col1, control_col2, control_col3 = st.columns([2, 1, 1])
                with control_col1:
                    injured_players = get_injured_players_for_session()
                    injured_count = sum(1 for key in injured_players if team1 in key or team2 in key)
                    st.caption(f"👤 Players marked OUT: {injured_count}")
                with control_col2:
                    if st.button("🔄 Reset All Injuries", key="reset_pass_injuries"):
                        clear_all_injuries()
                        st.rerun()
                with control_col3:
                    show_adjusted = st.checkbox("Show Adjusted", value=True, key="show_adjusted_pass")

                st.divider()

                pass_stats = players_df[players_df['pass_att'] > 0].groupby(['team', 'player']).agg({
                    'pass_yds': ['median', 'mean', 'std'],
                    'pass_td': ['median', 'mean', 'std'],
                    'pass_cmp': 'median'
                }).reset_index()
                pass_stats.columns = ['team', 'player', 'pass_yds_median', 'pass_yds_mean', 'pass_yds_std',
                                     'pass_td_median', 'pass_td_mean', 'pass_td_std', 'pass_cmp']
                pass_stats = pass_stats.sort_values('pass_yds_median', ascending=False)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**{team1} vs {team2} Defense**")
                    t1_passers = pass_stats[pass_stats['team'] == team1]
                    if not t1_passers.empty:
                        # Build expected stats list using smart aggregation
                        t1_expected = []
                        for _, p in t1_passers.iterrows():
                            # Check for transactions
                            trans_info = get_player_transaction_indicator(p['player'], team1, season, week)

                            # Skip players who weren't on team during analysis period
                            if not trans_info['is_valid']:
                                continue

                            # Skip players who are injured for this week
                            if is_player_on_injury_list(p['player'], team1, season, week):
                                continue

                            # Use smart aggregation function
                            smart_stats = calculate_smart_expected_stats(
                                player_name=p['player'],
                                team=team1,
                                season=season,
                                week=week,
                                stat_type='passing',
                                opponent_factor=t1_pass_factor,
                                strategy=projection_strategy
                            )

                            # Build display dictionary with confidence and transaction indicators
                            confidence_emoji = {'high': '✅', 'medium': '⚠️', 'low': '❗', 'none': '❌'}
                            confidence_label = confidence_emoji.get(smart_stats['confidence'], '')

                            # Combine player name with indicators
                            player_display = f"{p['player']} {confidence_label}"
                            if trans_info['indicator']:
                                player_display += f" {trans_info['indicator']}"

                            t1_expected.append({
                                'Player': player_display,
                                'Games': smart_stats['sample_size'],
                                'Expected Yds': smart_stats['expected_yds'],
                                'Downside Yds': smart_stats['downside_yds'],
                                'Upside Yds': smart_stats['upside_yds'],
                                'Expected TDs': smart_stats['expected_tds'],
                                'Downside TDs': smart_stats['downside_tds'],
                                'Upside TDs': smart_stats['upside_tds']
                            })

                        # Add injury checkboxes
                        st.markdown("**Mark Players as OUT:**")
                        injury_cols = st.columns(min(5, len(t1_expected)))
                        for idx, (col, player_stat) in enumerate(zip(injury_cols * (len(t1_expected) // len(injury_cols) + 1), t1_expected)):
                            with col:
                                player_name = player_stat['Player']
                                is_injured = is_player_injured(player_name, team1)
                                if st.checkbox(
                                    f"{'❌ ' if is_injured else ''}{player_name[:10]}...",
                                    value=is_injured,
                                    key=f"injury_t1_pass_{player_name}"
                                ):
                                    if not is_injured:
                                        toggle_player_injury(player_name, team1)
                                        st.rerun()
                                else:
                                    if is_injured:
                                        toggle_player_injury(player_name, team1)
                                        st.rerun()

                        # Apply redistribution if injuries exist and "Show Adjusted" is checked
                        if show_adjusted:
                            stat_columns = [
                                ('Downside Yds', 'yards'),
                                ('Expected Yds', 'yards'),
                                ('Upside Yds', 'yards'),
                                ('Downside TDs', 'TDs'),
                                ('Expected TDs', 'TDs'),
                                ('Upside TDs', 'TDs')
                            ]
                            adjusted_stats, summary = redistribute_stats(t1_expected, team1, stat_columns, season, week)

                            # Display adjusted stats
                            if summary:
                                display_stats = []
                                for adj_stat in adjusted_stats:
                                    player_name = adj_stat['Player']
                                    if adj_stat['injured']:
                                        # Show injured player with strikethrough effect (grayed out)
                                        display_stats.append({
                                            'Player': f"⚠️ {player_name}",
                                            'Downside Yds': '—',
                                            'Expected Yds': '—',
                                            'Upside Yds': '—',
                                            'Downside TDs': '—',
                                            'Expected TDs': '—',
                                            'Upside TDs': '—'
                                        })
                                    else:
                                        # Show healthy player with gains
                                        downside_yds_gain = adj_stat['gains'].get('Downside Yds', 0)
                                        expected_yds_gain = adj_stat['gains'].get('Expected Yds', 0)
                                        upside_yds_gain = adj_stat['gains'].get('Upside Yds', 0)
                                        downside_td_gain = adj_stat['gains'].get('Downside TDs', 0)
                                        expected_td_gain = adj_stat['gains'].get('Expected TDs', 0)
                                        upside_td_gain = adj_stat['gains'].get('Upside TDs', 0)

                                        display_stats.append({
                                            'Player': player_name,
                                            'Downside Yds': f"{int(adj_stat['Downside Yds'])} (+{int(downside_yds_gain)})" if downside_yds_gain > 0.5 else str(int(adj_stat['Downside Yds'])),
                                            'Expected Yds': f"{int(adj_stat['Expected Yds'])} (+{int(expected_yds_gain)})" if expected_yds_gain > 0.5 else str(int(adj_stat['Expected Yds'])),
                                            'Upside Yds': f"{int(adj_stat['Upside Yds'])} (+{int(upside_yds_gain)})" if upside_yds_gain > 0.5 else str(int(adj_stat['Upside Yds'])),
                                            'Downside TDs': f"{adj_stat['Downside TDs']:.1f} (+{downside_td_gain:.1f})" if downside_td_gain > 0.05 else f"{adj_stat['Downside TDs']:.1f}",
                                            'Expected TDs': f"{adj_stat['Expected TDs']:.1f} (+{expected_td_gain:.1f})" if expected_td_gain > 0.05 else f"{adj_stat['Expected TDs']:.1f}",
                                            'Upside TDs': f"{adj_stat['Upside TDs']:.1f} (+{upside_td_gain:.1f})" if upside_td_gain > 0.05 else f"{adj_stat['Upside TDs']:.1f}"
                                        })

                                st.dataframe(pd.DataFrame(display_stats), hide_index=True, use_container_width=True)

                                # Show redistribution summary
                                if summary['injured_players']:
                                    st.caption(f"📊 Redistribution: {', '.join(summary['injured_players'])} OUT")
                            else:
                                # No injuries, show original stats
                                display_df = pd.DataFrame(t1_expected)[['Player', 'Games', 'Downside Yds', 'Expected Yds', 'Upside Yds', 'Downside TDs', 'Expected TDs', 'Upside TDs']]
                                st.dataframe(display_df, hide_index=True, use_container_width=True)
                        else:
                            # Show original stats without adjustments
                            display_df = pd.DataFrame(t1_expected)[['Player', 'Games', 'Downside Yds', 'Expected Yds', 'Upside Yds', 'Downside TDs', 'Expected TDs', 'Upside TDs']]
                            st.dataframe(display_df, hide_index=True, use_container_width=True)

                        matchup_indicator = "Favorable" if t1_pass_factor > 1 else "Difficult"
                        st.caption(f"Matchup: {matchup_indicator} (Factor: {t1_pass_factor:.2f}x)")

                with col2:
                    st.markdown(f"**{team2} vs {team1} Defense**")
                    t2_passers = pass_stats[pass_stats['team'] == team2]
                    if not t2_passers.empty:
                        # Build expected stats list using smart aggregation
                        t2_expected = []
                        for _, p in t2_passers.iterrows():
                            # Check for transactions
                            trans_info = get_player_transaction_indicator(p['player'], team2, season, week)

                            # Skip players who weren't on team during analysis period
                            if not trans_info['is_valid']:
                                continue

                            # Skip players who are injured for this week
                            if is_player_on_injury_list(p['player'], team2, season, week):
                                continue

                            # Use smart aggregation function
                            smart_stats = calculate_smart_expected_stats(
                                player_name=p['player'],
                                team=team2,
                                season=season,
                                week=week,
                                stat_type='passing',
                                opponent_factor=t2_pass_factor,
                                strategy=projection_strategy
                            )

                            # Build display dictionary with confidence and transaction indicators
                            confidence_emoji = {'high': '✅', 'medium': '⚠️', 'low': '❗', 'none': '❌'}
                            confidence_label = confidence_emoji.get(smart_stats['confidence'], '')

                            # Combine player name with indicators
                            player_display = f"{p['player']} {confidence_label}"
                            if trans_info['indicator']:
                                player_display += f" {trans_info['indicator']}"

                            t2_expected.append({
                                'Player': player_display,
                                'team': team2,
                                'Games': smart_stats['sample_size'],
                                'Expected Yds': smart_stats['expected_yds'],
                                'Downside Yds': smart_stats['downside_yds'],
                                'Upside Yds': smart_stats['upside_yds'],
                                'Expected TDs': smart_stats['expected_tds'],
                                'Downside TDs': smart_stats['downside_tds'],
                                'Upside TDs': smart_stats['upside_tds']
                            })

                        # Add injury checkboxes
                        st.markdown("**Mark Players as OUT:**")
                        injury_cols = st.columns(min(5, len(t2_expected)))
                        for idx, (col, player_stat) in enumerate(zip(injury_cols * (len(t2_expected) // len(injury_cols) + 1), t2_expected)):
                            with col:
                                player_name = player_stat['Player']
                                is_injured = is_player_injured(player_name, team2)
                                if st.checkbox(
                                    f"{'❌ ' if is_injured else ''}{player_name[:10]}...",
                                    value=is_injured,
                                    key=f"injury_t2_pass_{player_name}"
                                ):
                                    if not is_injured:
                                        toggle_player_injury(player_name, team2)
                                        st.rerun()
                                else:
                                    if is_injured:
                                        toggle_player_injury(player_name, team2)
                                        st.rerun()

                        # Apply redistribution if injuries exist and "Show Adjusted" is checked
                        if show_adjusted:
                            stat_columns = [
                                ('Downside Yds', 'yards'),
                                ('Expected Yds', 'yards'),
                                ('Upside Yds', 'yards'),
                                ('Downside TDs', 'TDs'),
                                ('Expected TDs', 'TDs'),
                                ('Upside TDs', 'TDs')
                            ]
                            adjusted_stats, summary = redistribute_stats(t2_expected, team2, stat_columns, season, week)

                            # Display adjusted stats
                            if summary:
                                display_stats = []
                                for adj_stat in adjusted_stats:
                                    player_name = adj_stat['Player']
                                    if adj_stat['injured']:
                                        # Show injured player with strikethrough effect (grayed out)
                                        display_stats.append({
                                            'Player': f"⚠️ {player_name}",
                                            'Downside Yds': '—',
                                            'Expected Yds': '—',
                                            'Upside Yds': '—',
                                            'Downside TDs': '—',
                                            'Expected TDs': '—',
                                            'Upside TDs': '—'
                                        })
                                    else:
                                        # Show healthy player with gains
                                        downside_yds_gain = adj_stat['gains'].get('Downside Yds', 0)
                                        expected_yds_gain = adj_stat['gains'].get('Expected Yds', 0)
                                        upside_yds_gain = adj_stat['gains'].get('Upside Yds', 0)
                                        downside_td_gain = adj_stat['gains'].get('Downside TDs', 0)
                                        expected_td_gain = adj_stat['gains'].get('Expected TDs', 0)
                                        upside_td_gain = adj_stat['gains'].get('Upside TDs', 0)

                                        display_stats.append({
                                            'Player': player_name,
                                            'Downside Yds': f"{int(adj_stat['Downside Yds'])} (+{int(downside_yds_gain)})" if downside_yds_gain > 0.5 else str(int(adj_stat['Downside Yds'])),
                                            'Expected Yds': f"{int(adj_stat['Expected Yds'])} (+{int(expected_yds_gain)})" if expected_yds_gain > 0.5 else str(int(adj_stat['Expected Yds'])),
                                            'Upside Yds': f"{int(adj_stat['Upside Yds'])} (+{int(upside_yds_gain)})" if upside_yds_gain > 0.5 else str(int(adj_stat['Upside Yds'])),
                                            'Downside TDs': f"{adj_stat['Downside TDs']:.1f} (+{downside_td_gain:.1f})" if downside_td_gain > 0.05 else f"{adj_stat['Downside TDs']:.1f}",
                                            'Expected TDs': f"{adj_stat['Expected TDs']:.1f} (+{expected_td_gain:.1f})" if expected_td_gain > 0.05 else f"{adj_stat['Expected TDs']:.1f}",
                                            'Upside TDs': f"{adj_stat['Upside TDs']:.1f} (+{upside_td_gain:.1f})" if upside_td_gain > 0.05 else f"{adj_stat['Upside TDs']:.1f}"
                                        })

                                st.dataframe(pd.DataFrame(display_stats), hide_index=True, use_container_width=True)

                                # Show redistribution summary
                                if summary['injured_players']:
                                    st.caption(f"📊 Redistribution: {', '.join(summary['injured_players'])} OUT")
                            else:
                                # No injuries, show original stats
                                display_df = pd.DataFrame(t2_expected)[['Player', 'Games', 'Downside Yds', 'Expected Yds', 'Upside Yds', 'Downside TDs', 'Expected TDs', 'Upside TDs']]
                                st.dataframe(display_df, hide_index=True, use_container_width=True)
                        else:
                            # Show original stats without adjustments
                            display_df = pd.DataFrame(t2_expected)[['Player', 'Games', 'Downside Yds', 'Expected Yds', 'Upside Yds', 'Downside TDs', 'Expected TDs', 'Upside TDs']]
                            st.dataframe(display_df, hide_index=True, use_container_width=True)

                        matchup_indicator = "Favorable" if t2_pass_factor > 1 else "Difficult"
                        st.caption(f"Matchup: {matchup_indicator} (Factor: {t2_pass_factor:.2f}x)")

            with tab2:
                # Expected Rushing Stats
                st.markdown("### Expected Rushing Performance")

                # Injury Controls
                control_col1, control_col2, control_col3 = st.columns([2, 1, 1])
                with control_col1:
                    injured_players = get_injured_players_for_session()
                    injured_count = sum(1 for key in injured_players if team1 in key or team2 in key)
                    st.caption(f"👤 Players marked OUT: {injured_count}")
                with control_col2:
                    if st.button("🔄 Reset All Injuries", key="reset_rush_injuries"):
                        clear_all_injuries()
                        st.rerun()
                with control_col3:
                    show_adjusted_rush = st.checkbox("Show Adjusted", value=True, key="show_adjusted_rush")

                st.divider()

                # Get top rushers for each team using total yards
                rush_stats = players_df[players_df['rush_att'] > 0].groupby(['team', 'player']).agg({
                    'rush_yds': 'sum'
                }).reset_index()
                rush_stats = rush_stats.sort_values('rush_yds', ascending=False)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**{team1} vs {team2} Defense**")
                    t1_rushers = rush_stats[rush_stats['team'] == team1]
                    if not t1_rushers.empty:
                        # Build expected stats list using smart aggregation
                        t1_expected = []
                        for _, p in t1_rushers.iterrows():
                            # Check for transactions
                            trans_info = get_player_transaction_indicator(p['player'], team1, season, week)

                            # Skip players who weren't on team during analysis period
                            if not trans_info['is_valid']:
                                continue

                            # Skip players who are injured for this week
                            if is_player_on_injury_list(p['player'], team1, season, week):
                                continue

                            # Use smart aggregation function
                            smart_stats = calculate_smart_expected_stats(
                                player_name=p['player'],
                                team=team1,
                                season=season,
                                week=week,
                                stat_type='rushing',
                                opponent_factor=t1_rush_factor,
                                strategy=projection_strategy
                            )

                            # Build display dictionary with confidence and transaction indicators
                            confidence_emoji = {'high': '✅', 'medium': '⚠️', 'low': '❗', 'none': '❌'}
                            confidence_label = confidence_emoji.get(smart_stats['confidence'], '')

                            # Combine player name with indicators
                            player_display = f"{p['player']} {confidence_label}"
                            if trans_info['indicator']:
                                player_display += f" {trans_info['indicator']}"

                            t1_expected.append({
                                'Player': player_display,
                                'team': team1,
                                'Games': smart_stats['sample_size'],
                                'Expected Yds': smart_stats['expected_yds'],
                                'Downside Yds': smart_stats['downside_yds'],
                                'Upside Yds': smart_stats['upside_yds'],
                                'Expected TDs': smart_stats['expected_tds'],
                                'Downside TDs': smart_stats['downside_tds'],
                                'Upside TDs': smart_stats['upside_tds']
                            })

                        # Add injury checkboxes
                        st.markdown("**Mark Players as OUT:**")
                        injury_cols = st.columns(min(5, len(t1_expected)))
                        for idx, (col, player_stat) in enumerate(zip(injury_cols * (len(t1_expected) // len(injury_cols) + 1), t1_expected)):
                            with col:
                                player_name = player_stat['Player']
                                is_injured = is_player_injured(player_name, team1)
                                if st.checkbox(
                                    f"{'❌ ' if is_injured else ''}{player_name[:10]}...",
                                    value=is_injured,
                                    key=f"injury_t1_rush_{player_name}"
                                ):
                                    if not is_injured:
                                        toggle_player_injury(player_name, team1)
                                        st.rerun()
                                else:
                                    if is_injured:
                                        toggle_player_injury(player_name, team1)
                                        st.rerun()

                        # Apply redistribution if injuries exist and "Show Adjusted" is checked
                        if show_adjusted_rush:
                            stat_columns = [
                                ('Downside Yds', 'yards'),
                                ('Expected Yds', 'yards'),
                                ('Upside Yds', 'yards'),
                                ('Downside TDs', 'TDs'),
                                ('Expected TDs', 'TDs'),
                                ('Upside TDs', 'TDs')
                            ]
                            adjusted_stats, summary = redistribute_stats(t1_expected, team1, stat_columns, season, week)

                            # Display adjusted stats
                            if summary:
                                display_stats = []
                                for adj_stat in adjusted_stats:
                                    player_name = adj_stat['Player']
                                    if adj_stat['injured']:
                                        # Show injured player with strikethrough effect (grayed out)
                                        display_stats.append({
                                            'Player': f"⚠️ {player_name}",
                                            'Downside Yds': '—',
                                            'Expected Yds': '—',
                                            'Upside Yds': '—',
                                            'Downside TDs': '—',
                                            'Expected TDs': '—',
                                            'Upside TDs': '—'
                                        })
                                    else:
                                        # Show healthy player with gains
                                        downside_yds_gain = adj_stat['gains'].get('Downside Yds', 0)
                                        expected_yds_gain = adj_stat['gains'].get('Expected Yds', 0)
                                        upside_yds_gain = adj_stat['gains'].get('Upside Yds', 0)
                                        downside_td_gain = adj_stat['gains'].get('Downside TDs', 0)
                                        expected_td_gain = adj_stat['gains'].get('Expected TDs', 0)
                                        upside_td_gain = adj_stat['gains'].get('Upside TDs', 0)

                                        display_stats.append({
                                            'Player': player_name,
                                            'Downside Yds': f"{int(adj_stat['Downside Yds'])} (+{int(downside_yds_gain)})" if downside_yds_gain > 0.5 else str(int(adj_stat['Downside Yds'])),
                                            'Expected Yds': f"{int(adj_stat['Expected Yds'])} (+{int(expected_yds_gain)})" if expected_yds_gain > 0.5 else str(int(adj_stat['Expected Yds'])),
                                            'Upside Yds': f"{int(adj_stat['Upside Yds'])} (+{int(upside_yds_gain)})" if upside_yds_gain > 0.5 else str(int(adj_stat['Upside Yds'])),
                                            'Downside TDs': f"{adj_stat['Downside TDs']:.1f} (+{downside_td_gain:.1f})" if downside_td_gain > 0.05 else f"{adj_stat['Downside TDs']:.1f}",
                                            'Expected TDs': f"{adj_stat['Expected TDs']:.1f} (+{expected_td_gain:.1f})" if expected_td_gain > 0.05 else f"{adj_stat['Expected TDs']:.1f}",
                                            'Upside TDs': f"{adj_stat['Upside TDs']:.1f} (+{upside_td_gain:.1f})" if upside_td_gain > 0.05 else f"{adj_stat['Upside TDs']:.1f}"
                                        })

                                st.dataframe(pd.DataFrame(display_stats), hide_index=True, use_container_width=True)

                                # Show redistribution summary
                                if summary['injured_players']:
                                    st.caption(f"📊 Redistribution: {', '.join(summary['injured_players'])} OUT")
                            else:
                                # No injuries, show original stats
                                display_df = pd.DataFrame(t1_expected)[['Player', 'Games', 'Downside Yds', 'Expected Yds', 'Upside Yds', 'Downside TDs', 'Expected TDs', 'Upside TDs']]
                                st.dataframe(display_df, hide_index=True, use_container_width=True)
                        else:
                            # Show original stats without adjustments
                            display_df = pd.DataFrame(t1_expected)[['Player', 'Games', 'Downside Yds', 'Expected Yds', 'Upside Yds', 'Downside TDs', 'Expected TDs', 'Upside TDs']]
                            st.dataframe(display_df, hide_index=True, use_container_width=True)

                        matchup_indicator = "Favorable" if t1_rush_factor > 1 else "Difficult"
                        st.caption(f"Matchup: {matchup_indicator} (Factor: {t1_rush_factor:.2f}x)")

                with col2:
                    st.markdown(f"**{team2} vs {team1} Defense**")
                    t2_rushers = rush_stats[rush_stats['team'] == team2]
                    if not t2_rushers.empty:
                        # Build expected stats list using smart aggregation
                        t2_expected = []
                        for _, p in t2_rushers.iterrows():
                            # Check for transactions
                            trans_info = get_player_transaction_indicator(p['player'], team2, season, week)

                            # Skip players who weren't on team during analysis period
                            if not trans_info['is_valid']:
                                continue

                            # Skip players who are injured for this week
                            if is_player_on_injury_list(p['player'], team2, season, week):
                                continue

                            # Use smart aggregation function
                            smart_stats = calculate_smart_expected_stats(
                                player_name=p['player'],
                                team=team2,
                                season=season,
                                week=week,
                                stat_type='rushing',
                                opponent_factor=t2_rush_factor,
                                strategy=projection_strategy
                            )

                            # Build display dictionary with confidence and transaction indicators
                            confidence_emoji = {'high': '✅', 'medium': '⚠️', 'low': '❗', 'none': '❌'}
                            confidence_label = confidence_emoji.get(smart_stats['confidence'], '')

                            # Combine player name with indicators
                            player_display = f"{p['player']} {confidence_label}"
                            if trans_info['indicator']:
                                player_display += f" {trans_info['indicator']}"

                            t2_expected.append({
                                'Player': player_display,
                                'team': team2,
                                'Games': smart_stats['sample_size'],
                                'Expected Yds': smart_stats['expected_yds'],
                                'Downside Yds': smart_stats['downside_yds'],
                                'Upside Yds': smart_stats['upside_yds'],
                                'Expected TDs': smart_stats['expected_tds'],
                                'Downside TDs': smart_stats['downside_tds'],
                                'Upside TDs': smart_stats['upside_tds']
                            })

                        # Add injury checkboxes
                        st.markdown("**Mark Players as OUT:**")
                        injury_cols = st.columns(min(5, len(t2_expected)))
                        for idx, (col, player_stat) in enumerate(zip(injury_cols * (len(t2_expected) // len(injury_cols) + 1), t2_expected)):
                            with col:
                                player_name = player_stat['Player']
                                is_injured = is_player_injured(player_name, team2)
                                if st.checkbox(
                                    f"{'❌ ' if is_injured else ''}{player_name[:10]}...",
                                    value=is_injured,
                                    key=f"injury_t2_rush_{player_name}"
                                ):
                                    if not is_injured:
                                        toggle_player_injury(player_name, team2)
                                        st.rerun()
                                else:
                                    if is_injured:
                                        toggle_player_injury(player_name, team2)
                                        st.rerun()

                        # Apply redistribution if injuries exist and "Show Adjusted" is checked
                        if show_adjusted_rush:
                            stat_columns = [
                                ('Downside Yds', 'yards'),
                                ('Expected Yds', 'yards'),
                                ('Upside Yds', 'yards'),
                                ('Downside TDs', 'TDs'),
                                ('Expected TDs', 'TDs'),
                                ('Upside TDs', 'TDs')
                            ]
                            adjusted_stats, summary = redistribute_stats(t2_expected, team2, stat_columns, season, week)

                            # Display adjusted stats
                            if summary:
                                display_stats = []
                                for adj_stat in adjusted_stats:
                                    player_name = adj_stat['Player']
                                    if adj_stat['injured']:
                                        # Show injured player with strikethrough effect (grayed out)
                                        display_stats.append({
                                            'Player': f"⚠️ {player_name}",
                                            'Downside Yds': '—',
                                            'Expected Yds': '—',
                                            'Upside Yds': '—',
                                            'Downside TDs': '—',
                                            'Expected TDs': '—',
                                            'Upside TDs': '—'
                                        })
                                    else:
                                        # Show healthy player with gains
                                        downside_yds_gain = adj_stat['gains'].get('Downside Yds', 0)
                                        expected_yds_gain = adj_stat['gains'].get('Expected Yds', 0)
                                        upside_yds_gain = adj_stat['gains'].get('Upside Yds', 0)
                                        downside_td_gain = adj_stat['gains'].get('Downside TDs', 0)
                                        expected_td_gain = adj_stat['gains'].get('Expected TDs', 0)
                                        upside_td_gain = adj_stat['gains'].get('Upside TDs', 0)

                                        display_stats.append({
                                            'Player': player_name,
                                            'Downside Yds': f"{int(adj_stat['Downside Yds'])} (+{int(downside_yds_gain)})" if downside_yds_gain > 0.5 else str(int(adj_stat['Downside Yds'])),
                                            'Expected Yds': f"{int(adj_stat['Expected Yds'])} (+{int(expected_yds_gain)})" if expected_yds_gain > 0.5 else str(int(adj_stat['Expected Yds'])),
                                            'Upside Yds': f"{int(adj_stat['Upside Yds'])} (+{int(upside_yds_gain)})" if upside_yds_gain > 0.5 else str(int(adj_stat['Upside Yds'])),
                                            'Downside TDs': f"{adj_stat['Downside TDs']:.1f} (+{downside_td_gain:.1f})" if downside_td_gain > 0.05 else f"{adj_stat['Downside TDs']:.1f}",
                                            'Expected TDs': f"{adj_stat['Expected TDs']:.1f} (+{expected_td_gain:.1f})" if expected_td_gain > 0.05 else f"{adj_stat['Expected TDs']:.1f}",
                                            'Upside TDs': f"{adj_stat['Upside TDs']:.1f} (+{upside_td_gain:.1f})" if upside_td_gain > 0.05 else f"{adj_stat['Upside TDs']:.1f}"
                                        })

                                st.dataframe(pd.DataFrame(display_stats), hide_index=True, use_container_width=True)

                                # Show redistribution summary
                                if summary['injured_players']:
                                    st.caption(f"📊 Redistribution: {', '.join(summary['injured_players'])} OUT")
                            else:
                                # No injuries, show original stats
                                display_df = pd.DataFrame(t2_expected)[['Player', 'Games', 'Downside Yds', 'Expected Yds', 'Upside Yds', 'Downside TDs', 'Expected TDs', 'Upside TDs']]
                                st.dataframe(display_df, hide_index=True, use_container_width=True)
                        else:
                            # Show original stats without adjustments
                            display_df = pd.DataFrame(t2_expected)[['Player', 'Games', 'Downside Yds', 'Expected Yds', 'Upside Yds', 'Downside TDs', 'Expected TDs', 'Upside TDs']]
                            st.dataframe(display_df, hide_index=True, use_container_width=True)

                        matchup_indicator = "Favorable" if t2_rush_factor > 1 else "Difficult"
                        st.caption(f"Matchup: {matchup_indicator} (Factor: {t2_rush_factor:.2f}x)")

            with tab3:
                # Expected Receiving Stats
                st.markdown("### Expected Receiving Performance")

                # Injury Controls
                control_col1, control_col2, control_col3 = st.columns([2, 1, 1])
                with control_col1:
                    injured_players = get_injured_players_for_session()
                    injured_count = sum(1 for key in injured_players if team1 in key or team2 in key)
                    st.caption(f"👤 Players marked OUT: {injured_count}")
                with control_col2:
                    if st.button("🔄 Reset All Injuries", key="reset_rec_injuries"):
                        clear_all_injuries()
                        st.rerun()
                with control_col3:
                    show_adjusted = st.checkbox("Show Adjusted", value=True, key="show_adjusted_rec")

                st.divider()

                # Get top receivers for each team using total yards
                rec_stats = players_df[players_df['rec'] > 0].groupby(['team', 'player']).agg({
                    'rec_yds': 'sum'
                }).reset_index()
                rec_stats = rec_stats.sort_values('rec_yds', ascending=False)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**{team1} vs {team2} Defense**")
                    t1_receivers = rec_stats[rec_stats['team'] == team1]
                    if not t1_receivers.empty:
                        # Build expected stats list using smart aggregation
                        t1_expected = []
                        for _, p in t1_receivers.iterrows():
                            # Check for transactions
                            trans_info = get_player_transaction_indicator(p['player'], team1, season, week)

                            # Skip players who weren't on team during analysis period
                            if not trans_info['is_valid']:
                                continue

                            # Skip players who are injured for this week
                            if is_player_on_injury_list(p['player'], team1, season, week):
                                continue

                            # Use smart aggregation function
                            smart_stats = calculate_smart_expected_stats(
                                player_name=p['player'],
                                team=team1,
                                season=season,
                                week=week,
                                stat_type='receiving',
                                opponent_factor=t1_rec_factor,
                                strategy=projection_strategy
                            )

                            # Build display dictionary with confidence and transaction indicators
                            confidence_emoji = {'high': '✅', 'medium': '⚠️', 'low': '❗', 'none': '❌'}
                            confidence_label = confidence_emoji.get(smart_stats['confidence'], '')

                            # Combine player name with indicators
                            player_display = f"{p['player']} {confidence_label}"
                            if trans_info['indicator']:
                                player_display += f" {trans_info['indicator']}"

                            t1_expected.append({
                                'Player': player_display,
                                'Games': smart_stats['sample_size'],
                                'Expected Yds': smart_stats['expected_yds'],
                                'Downside Yds': smart_stats['downside_yds'],
                                'Upside Yds': smart_stats['upside_yds'],
                                'Expected Rec': smart_stats['expected_rec'],
                                'Downside Rec': smart_stats['downside_rec'],
                                'Upside Rec': smart_stats['upside_rec'],
                                'Expected TDs': smart_stats['expected_tds'],
                                'Downside TDs': smart_stats['downside_tds'],
                                'Upside TDs': smart_stats['upside_tds']
                            })

                        # Add injury checkboxes
                        st.markdown("**Mark Players as OUT:**")
                        injury_cols = st.columns(min(5, len(t1_expected)))
                        for idx, (col, player_stat) in enumerate(zip(injury_cols * (len(t1_expected) // len(injury_cols) + 1), t1_expected)):
                            with col:
                                player_name = player_stat['Player']
                                is_injured = is_player_injured(player_name, team1)
                                if st.checkbox(
                                    f"{'❌ ' if is_injured else ''}{player_name[:10]}...",
                                    value=is_injured,
                                    key=f"injury_t1_rec_{player_name}"
                                ):
                                    if not is_injured:
                                        toggle_player_injury(player_name, team1)
                                        st.rerun()
                                else:
                                    if is_injured:
                                        toggle_player_injury(player_name, team1)
                                        st.rerun()

                        # Apply redistribution if injuries exist and "Show Adjusted" is checked
                        if show_adjusted:
                            stat_columns = [
                                ('Downside Yds', 'yards'),
                                ('Expected Yds', 'yards'),
                                ('Upside Yds', 'yards'),
                                ('Downside Rec', 'receptions'),
                                ('Expected Rec', 'receptions'),
                                ('Upside Rec', 'receptions'),
                                ('Downside TDs', 'TDs'),
                                ('Expected TDs', 'TDs'),
                                ('Upside TDs', 'TDs')
                            ]
                            adjusted_stats, summary = redistribute_stats(t1_expected, team1, stat_columns, season, week)

                            # Display adjusted stats
                            if summary:
                                display_stats = []
                                for adj_stat in adjusted_stats:
                                    player_name = adj_stat['Player']
                                    if adj_stat['injured']:
                                        # Show injured player with strikethrough effect (grayed out)
                                        display_stats.append({
                                            'Player': f"⚠️ {player_name}",
                                            'Downside Yds': '—',
                                            'Expected Yds': '—',
                                            'Upside Yds': '—',
                                            'Downside Rec': '—',
                                            'Expected Rec': '—',
                                            'Upside Rec': '—',
                                            'Downside TDs': '—',
                                            'Expected TDs': '—',
                                            'Upside TDs': '—'
                                        })
                                    else:
                                        # Show healthy player with gains
                                        downside_yds_gain = adj_stat['gains'].get('Downside Yds', 0)
                                        expected_yds_gain = adj_stat['gains'].get('Expected Yds', 0)
                                        upside_yds_gain = adj_stat['gains'].get('Upside Yds', 0)
                                        downside_rec_gain = adj_stat['gains'].get('Downside Rec', 0)
                                        expected_rec_gain = adj_stat['gains'].get('Expected Rec', 0)
                                        upside_rec_gain = adj_stat['gains'].get('Upside Rec', 0)
                                        downside_td_gain = adj_stat['gains'].get('Downside TDs', 0)
                                        expected_td_gain = adj_stat['gains'].get('Expected TDs', 0)
                                        upside_td_gain = adj_stat['gains'].get('Upside TDs', 0)

                                        display_stats.append({
                                            'Player': player_name,
                                            'Downside Yds': f"{int(adj_stat['Downside Yds'])} (+{int(downside_yds_gain)})" if downside_yds_gain > 0.5 else str(int(adj_stat['Downside Yds'])),
                                            'Expected Yds': f"{int(adj_stat['Expected Yds'])} (+{int(expected_yds_gain)})" if expected_yds_gain > 0.5 else str(int(adj_stat['Expected Yds'])),
                                            'Upside Yds': f"{int(adj_stat['Upside Yds'])} (+{int(upside_yds_gain)})" if upside_yds_gain > 0.5 else str(int(adj_stat['Upside Yds'])),
                                            'Downside Rec': f"{adj_stat['Downside Rec']:.1f} (+{downside_rec_gain:.1f})" if downside_rec_gain > 0.05 else f"{adj_stat['Downside Rec']:.1f}",
                                            'Expected Rec': f"{adj_stat['Expected Rec']:.1f} (+{expected_rec_gain:.1f})" if expected_rec_gain > 0.05 else f"{adj_stat['Expected Rec']:.1f}",
                                            'Upside Rec': f"{adj_stat['Upside Rec']:.1f} (+{upside_rec_gain:.1f})" if upside_rec_gain > 0.05 else f"{adj_stat['Upside Rec']:.1f}",
                                            'Downside TDs': f"{adj_stat['Downside TDs']:.1f} (+{downside_td_gain:.1f})" if downside_td_gain > 0.05 else f"{adj_stat['Downside TDs']:.1f}",
                                            'Expected TDs': f"{adj_stat['Expected TDs']:.1f} (+{expected_td_gain:.1f})" if expected_td_gain > 0.05 else f"{adj_stat['Expected TDs']:.1f}",
                                            'Upside TDs': f"{adj_stat['Upside TDs']:.1f} (+{upside_td_gain:.1f})" if upside_td_gain > 0.05 else f"{adj_stat['Upside TDs']:.1f}"
                                        })

                                st.dataframe(pd.DataFrame(display_stats), hide_index=True, use_container_width=True)

                                # Show redistribution summary
                                if summary['injured_players']:
                                    st.caption(f"📊 Redistribution: {', '.join(summary['injured_players'])} OUT")
                            else:
                                # No injuries, show original stats
                                display_df = pd.DataFrame(t1_expected)[['Player', 'Games', 'Downside Yds', 'Expected Yds', 'Upside Yds', 'Downside Rec', 'Expected Rec', 'Upside Rec', 'Downside TDs', 'Expected TDs', 'Upside TDs']]
                                st.dataframe(display_df, hide_index=True, use_container_width=True)
                        else:
                            # Show original stats without adjustments
                            display_df = pd.DataFrame(t1_expected)[['Player', 'Games', 'Downside Yds', 'Expected Yds', 'Upside Yds', 'Downside Rec', 'Expected Rec', 'Upside Rec', 'Downside TDs', 'Expected TDs', 'Upside TDs']]
                            st.dataframe(display_df, hide_index=True, use_container_width=True)

                        matchup_indicator = "Favorable" if t1_rec_factor > 1 else "Difficult"
                        st.caption(f"Matchup: {matchup_indicator} (Factor: {t1_rec_factor:.2f}x)")

                with col2:
                    st.markdown(f"**{team2} vs {team1} Defense**")
                    t2_receivers = rec_stats[rec_stats['team'] == team2]
                    if not t2_receivers.empty:
                        # Build expected stats list using smart aggregation
                        t2_expected = []
                        for _, p in t2_receivers.iterrows():
                            # Check for transactions
                            trans_info = get_player_transaction_indicator(p['player'], team2, season, week)

                            # Skip players who weren't on team during analysis period
                            if not trans_info['is_valid']:
                                continue

                            # Skip players who are injured for this week
                            if is_player_on_injury_list(p['player'], team2, season, week):
                                continue

                            # Use smart aggregation function
                            smart_stats = calculate_smart_expected_stats(
                                player_name=p['player'],
                                team=team2,
                                season=season,
                                week=week,
                                stat_type='receiving',
                                opponent_factor=t2_rec_factor,
                                strategy=projection_strategy
                            )

                            # Build display dictionary with confidence and transaction indicators
                            confidence_emoji = {'high': '✅', 'medium': '⚠️', 'low': '❗', 'none': '❌'}
                            confidence_label = confidence_emoji.get(smart_stats['confidence'], '')

                            # Combine player name with indicators
                            player_display = f"{p['player']} {confidence_label}"
                            if trans_info['indicator']:
                                player_display += f" {trans_info['indicator']}"

                            t2_expected.append({
                                'Player': player_display,
                                'Games': smart_stats['sample_size'],
                                'Expected Yds': smart_stats['expected_yds'],
                                'Downside Yds': smart_stats['downside_yds'],
                                'Upside Yds': smart_stats['upside_yds'],
                                'Expected Rec': smart_stats['expected_rec'],
                                'Downside Rec': smart_stats['downside_rec'],
                                'Upside Rec': smart_stats['upside_rec'],
                                'Expected TDs': smart_stats['expected_tds'],
                                'Downside TDs': smart_stats['downside_tds'],
                                'Upside TDs': smart_stats['upside_tds']
                            })

                        # Add injury checkboxes
                        st.markdown("**Mark Players as OUT:**")
                        injury_cols = st.columns(len(t2_expected))
                        for idx, (col, player_stat) in enumerate(zip(injury_cols, t2_expected)):
                            with col:
                                player_name = player_stat['Player']
                                is_injured = is_player_injured(player_name, team2)
                                if st.checkbox(
                                    f"{'❌ ' if is_injured else ''}{player_name[:10]}...",
                                    value=is_injured,
                                    key=f"injury_t2_rec_{player_name}"
                                ):
                                    if not is_injured:
                                        toggle_player_injury(player_name, team2)
                                        st.rerun()
                                else:
                                    if is_injured:
                                        toggle_player_injury(player_name, team2)
                                        st.rerun()

                        # Apply redistribution if injuries exist and "Show Adjusted" is checked
                        if show_adjusted:
                            stat_columns = [
                                ('Downside Yds', 'yards'),
                                ('Expected Yds', 'yards'),
                                ('Upside Yds', 'yards'),
                                ('Downside Rec', 'receptions'),
                                ('Expected Rec', 'receptions'),
                                ('Upside Rec', 'receptions'),
                                ('Downside TDs', 'TDs'),
                                ('Expected TDs', 'TDs'),
                                ('Upside TDs', 'TDs')
                            ]
                            adjusted_stats, summary = redistribute_stats(t2_expected, team2, stat_columns, season, week)

                            # Display adjusted stats
                            if summary:
                                display_stats = []
                                for adj_stat in adjusted_stats:
                                    player_name = adj_stat['Player']
                                    if adj_stat['injured']:
                                        # Show injured player with strikethrough effect (grayed out)
                                        display_stats.append({
                                            'Player': f"⚠️ {player_name}",
                                            'Downside Yds': '—',
                                            'Expected Yds': '—',
                                            'Upside Yds': '—',
                                            'Downside Rec': '—',
                                            'Expected Rec': '—',
                                            'Upside Rec': '—',
                                            'Downside TDs': '—',
                                            'Expected TDs': '—',
                                            'Upside TDs': '—'
                                        })
                                    else:
                                        # Show healthy player with gains
                                        downside_yds_gain = adj_stat['gains'].get('Downside Yds', 0)
                                        expected_yds_gain = adj_stat['gains'].get('Expected Yds', 0)
                                        upside_yds_gain = adj_stat['gains'].get('Upside Yds', 0)
                                        downside_rec_gain = adj_stat['gains'].get('Downside Rec', 0)
                                        expected_rec_gain = adj_stat['gains'].get('Expected Rec', 0)
                                        upside_rec_gain = adj_stat['gains'].get('Upside Rec', 0)
                                        downside_td_gain = adj_stat['gains'].get('Downside TDs', 0)
                                        expected_td_gain = adj_stat['gains'].get('Expected TDs', 0)
                                        upside_td_gain = adj_stat['gains'].get('Upside TDs', 0)

                                        display_stats.append({
                                            'Player': player_name,
                                            'Downside Yds': f"{int(adj_stat['Downside Yds'])} (+{int(downside_yds_gain)})" if downside_yds_gain > 0.5 else str(int(adj_stat['Downside Yds'])),
                                            'Expected Yds': f"{int(adj_stat['Expected Yds'])} (+{int(expected_yds_gain)})" if expected_yds_gain > 0.5 else str(int(adj_stat['Expected Yds'])),
                                            'Upside Yds': f"{int(adj_stat['Upside Yds'])} (+{int(upside_yds_gain)})" if upside_yds_gain > 0.5 else str(int(adj_stat['Upside Yds'])),
                                            'Downside Rec': f"{adj_stat['Downside Rec']:.1f} (+{downside_rec_gain:.1f})" if downside_rec_gain > 0.05 else f"{adj_stat['Downside Rec']:.1f}",
                                            'Expected Rec': f"{adj_stat['Expected Rec']:.1f} (+{expected_rec_gain:.1f})" if expected_rec_gain > 0.05 else f"{adj_stat['Expected Rec']:.1f}",
                                            'Upside Rec': f"{adj_stat['Upside Rec']:.1f} (+{upside_rec_gain:.1f})" if upside_rec_gain > 0.05 else f"{adj_stat['Upside Rec']:.1f}",
                                            'Downside TDs': f"{adj_stat['Downside TDs']:.1f} (+{downside_td_gain:.1f})" if downside_td_gain > 0.05 else f"{adj_stat['Downside TDs']:.1f}",
                                            'Expected TDs': f"{adj_stat['Expected TDs']:.1f} (+{expected_td_gain:.1f})" if expected_td_gain > 0.05 else f"{adj_stat['Expected TDs']:.1f}",
                                            'Upside TDs': f"{adj_stat['Upside TDs']:.1f} (+{upside_td_gain:.1f})" if upside_td_gain > 0.05 else f"{adj_stat['Upside TDs']:.1f}"
                                        })

                                st.dataframe(pd.DataFrame(display_stats), hide_index=True, use_container_width=True)

                                # Show redistribution summary
                                if summary['injured_players']:
                                    st.caption(f"📊 Redistribution: {', '.join(summary['injured_players'])} OUT")
                            else:
                                # No injuries, show original stats
                                display_df = pd.DataFrame(t2_expected)[['Player', 'Games', 'Downside Yds', 'Expected Yds', 'Upside Yds', 'Downside Rec', 'Expected Rec', 'Upside Rec', 'Downside TDs', 'Expected TDs', 'Upside TDs']]
                                st.dataframe(display_df, hide_index=True, use_container_width=True)
                        else:
                            # Show original stats without adjustments
                            display_df = pd.DataFrame(t2_expected)[['Player', 'Games', 'Downside Yds', 'Expected Yds', 'Upside Yds', 'Downside Rec', 'Expected Rec', 'Upside Rec', 'Downside TDs', 'Expected TDs', 'Upside TDs']]
                            st.dataframe(display_df, hide_index=True, use_container_width=True)

                        matchup_indicator = "Favorable" if t2_rec_factor > 1 else "Difficult"
                        st.caption(f"Matchup: {matchup_indicator} (Factor: {t2_rec_factor:.2f}x)")
        else:
            st.info("Insufficient data to calculate expected stats.")

        st.divider()

        # Defensive Performance Section
        st.subheader("🛡️ Defensive Performance")
        st.caption("Defensive stats, impact players, and unit effectiveness")

        # Get defensive stats for both teams
        t1_def_stats = get_team_defensive_stats(team1, season, week)
        t2_def_stats = get_team_defensive_stats(team2, season, week)

        # Check if defensive data is available
        has_def_data = (t1_def_stats['games'] > 0 or t2_def_stats['games'] > 0)

        if not has_def_data:
            st.info(f"⚠️ Detailed defensive player stats not available for this matchup. Defensive data may not be scraped for {team1} and/or {team2} yet.")
        else:
            # Team Defensive Stats Overview
            st.markdown("### Team Defensive Stats")

            def_col1, def_col2 = st.columns(2)

            with def_col1:
                st.markdown(f"**{team1} Defense**")
                st.metric("Tackles/Game", f"{t1_def_stats['tackles_per_game']:.1f}")
                st.metric("Sacks/Game", f"{t1_def_stats['sacks_per_game']:.2f}",
                         help="Sacks per game")
                st.metric("INTs/Game", f"{t1_def_stats['ints_per_game']:.2f}",
                         help="Interceptions per game")
                st.metric("Pressures/Game", f"{t1_def_stats['pressures_per_game']:.1f}",
                         help="QB pressures, hits, and hurries per game")
                st.metric("Pass Rating Allowed", f"{t1_def_stats['pass_rating_allowed']:.1f}",
                         help="Opponent passer rating when targeting these defenders")
                st.metric("Missed Tackle %", f"{t1_def_stats['missed_tackle_pct']:.1f}%",
                         help="Percentage of tackle attempts that are missed")

            with def_col2:
                st.markdown(f"**{team2} Defense**")
                st.metric("Tackles/Game", f"{t2_def_stats['tackles_per_game']:.1f}")
                st.metric("Sacks/Game", f"{t2_def_stats['sacks_per_game']:.2f}",
                         help="Sacks per game")
                st.metric("INTs/Game", f"{t2_def_stats['ints_per_game']:.2f}",
                         help="Interceptions per game")
                st.metric("Pressures/Game", f"{t2_def_stats['pressures_per_game']:.1f}",
                         help="QB pressures, hits, and hurries per game")
                st.metric("Pass Rating Allowed", f"{t2_def_stats['pass_rating_allowed']:.1f}",
                         help="Opponent passer rating when targeting these defenders")
                st.metric("Missed Tackle %", f"{t2_def_stats['missed_tackle_pct']:.1f}%",
                         help="Percentage of tackle attempts that are missed")

            st.markdown("---")

            # Defensive Leaders
            st.markdown("### Defensive Leaders")

            # Get defensive leaders for both teams
            t1_def_leaders = get_defensive_leaders(team1, season, week)
            t2_def_leaders = get_defensive_leaders(team2, season, week)

            if not t1_def_leaders.empty and not t2_def_leaders.empty:
                tab1, tab2, tab3 = st.tabs(["Top Tacklers", "Pass Rushers", "Coverage"])

                with tab1:
                    st.markdown("**Leading Tacklers**")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"### {team1}")
                        t1_tacklers = t1_def_leaders.sort_values('tackles', ascending=False).head(5)
                        for _, p in t1_tacklers.iterrows():
                            missed_pct = (p['missed_tackles'] / (p['tackles'] + p['missed_tackles']) * 100) if (p['tackles'] + p['missed_tackles']) > 0 else 0
                            st.markdown(f"**{p['player']}** ({int(p['games'])} games)")
                            st.markdown(f"- **Total:** {int(p['tackles'])} tackles, {int(p['missed_tackles'])} missed")
                            st.markdown(f"- **Per Game:** {p['tackles_per_game']:.1f} tackles/game")
                            st.markdown(f"- **Missed %:** {missed_pct:.1f}%")
                            st.markdown("---")

                    with col2:
                        st.markdown(f"### {team2}")
                        t2_tacklers = t2_def_leaders.sort_values('tackles', ascending=False).head(5)
                        for _, p in t2_tacklers.iterrows():
                            missed_pct = (p['missed_tackles'] / (p['tackles'] + p['missed_tackles']) * 100) if (p['tackles'] + p['missed_tackles']) > 0 else 0
                            st.markdown(f"**{p['player']}** ({int(p['games'])} games)")
                            st.markdown(f"- **Total:** {int(p['tackles'])} tackles, {int(p['missed_tackles'])} missed")
                            st.markdown(f"- **Per Game:** {p['tackles_per_game']:.1f} tackles/game")
                            st.markdown(f"- **Missed %:** {missed_pct:.1f}%")
                            st.markdown("---")

                with tab2:
                    st.markdown("**Pass Rush Leaders**")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"### {team1}")
                        t1_rushers = t1_def_leaders.sort_values('total_disruptions', ascending=False).head(5)
                        for _, p in t1_rushers.iterrows():
                            if p['total_disruptions'] > 0:
                                st.markdown(f"**{p['player']}** ({int(p['games'])} games)")
                                st.markdown(f"- **Sacks:** {p['sacks']:.1f}")
                                st.markdown(f"- **QB Hits:** {int(p['qb_hits'])}")
                                st.markdown(f"- **Hurries:** {int(p['hurries'])}")
                                st.markdown(f"- **Pressures:** {int(p['pressures'])}")
                                st.markdown(f"- **Total Disruptions:** {int(p['total_disruptions'])}")
                                st.markdown("---")

                    with col2:
                        st.markdown(f"### {team2}")
                        t2_rushers = t2_def_leaders.sort_values('total_disruptions', ascending=False).head(5)
                        for _, p in t2_rushers.iterrows():
                            if p['total_disruptions'] > 0:
                                st.markdown(f"**{p['player']}** ({int(p['games'])} games)")
                                st.markdown(f"- **Sacks:** {p['sacks']:.1f}")
                                st.markdown(f"- **QB Hits:** {int(p['qb_hits'])}")
                                st.markdown(f"- **Hurries:** {int(p['hurries'])}")
                                st.markdown(f"- **Pressures:** {int(p['pressures'])}")
                                st.markdown(f"- **Total Disruptions:** {int(p['total_disruptions'])}")
                                st.markdown("---")

                with tab3:
                    st.markdown("**Coverage Leaders**")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"### {team1}")
                        # Sort by targets (most coverage snaps) and then by effectiveness
                        t1_coverage = t1_def_leaders[t1_def_leaders['targets'] >= 5].sort_values('targets', ascending=False).head(5)
                        for _, p in t1_coverage.iterrows():
                            st.markdown(f"**{p['player']}** ({int(p['games'])} games)")
                            st.markdown(f"- **Targets:** {int(p['targets'])}")
                            st.markdown(f"- **Completions:** {int(p['completions'])} ({p['comp_pct_allowed']:.1f}%)")
                            st.markdown(f"- **Yards Allowed:** {int(p['comp_yds'])} ({p['yds_per_target']:.1f}/target)")
                            st.markdown(f"- **INTs:** {int(p['ints'])}")
                            st.markdown("---")

                    with col2:
                        st.markdown(f"### {team2}")
                        t2_coverage = t2_def_leaders[t2_def_leaders['targets'] >= 5].sort_values('targets', ascending=False).head(5)
                        for _, p in t2_coverage.iterrows():
                            st.markdown(f"**{p['player']}** ({int(p['games'])} games)")
                            st.markdown(f"- **Targets:** {int(p['targets'])}")
                            st.markdown(f"- **Completions:** {int(p['completions'])} ({p['comp_pct_allowed']:.1f}%)")
                            st.markdown(f"- **Yards Allowed:** {int(p['comp_yds'])} ({p['yds_per_target']:.1f}/target)")
                            st.markdown(f"- **INTs:** {int(p['ints'])}")
                            st.markdown("---")

            else:
                st.info("Defensive player data not available for this matchup.")

        st.divider()

        # Performance Trends & Momentum
        st.subheader("📈 Performance Trends & Momentum")
        st.caption("Analyzing recent form, splits, and trajectory")

        # Get recent games (last 5 weeks)
        max_week = players_df['week'].max()
        recent_weeks = max(1, max_week - 4)
        recent_df = players_df[players_df['week'] >= recent_weeks].copy()

        tab1, tab2, tab3 = st.tabs(["Recent Form", "Home/Away Splits", "Trend Analysis"])

        with tab1:
            st.markdown("### Last 5 Games vs Season Average")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"#### {team1} Key Players")

                # Passing recent form
                t1_passers_recent = recent_df[(recent_df['team'] == team1) & (recent_df['pass_att'] > 0)]
                if not t1_passers_recent.empty:
                    t1_pass_recent_avg = t1_passers_recent.groupby('player')['pass_yds'].agg(['mean', 'count']).reset_index()
                    t1_pass_season_avg = players_df[(players_df['team'] == team1) & (players_df['pass_att'] > 0)].groupby('player')['pass_yds'].mean().reset_index()
                    t1_pass_season_avg.columns = ['player', 'season_avg']
                    t1_pass_form = t1_pass_recent_avg.merge(t1_pass_season_avg, on='player')
                    t1_pass_form['diff'] = t1_pass_form['mean'] - t1_pass_form['season_avg']
                    t1_pass_form = t1_pass_form[t1_pass_form['count'] >= 3].sort_values('mean', ascending=False)

                    if not t1_pass_form.empty:
                        st.markdown("**Passing:**")
                        for _, p in t1_pass_form.head(2).iterrows():
                            trend = "🔥" if p['diff'] > 20 else "📈" if p['diff'] > 0 else "📉" if p['diff'] < -20 else "➡️"
                            st.markdown(f"{trend} **{p['player']}**: {p['mean']:.1f} yds/game (recent) vs {p['season_avg']:.1f} (season) | {p['diff']:+.1f}")

                # Rushing recent form
                t1_rushers_recent = recent_df[(recent_df['team'] == team1) & (recent_df['rush_att'] > 0)]
                if not t1_rushers_recent.empty:
                    t1_rush_recent_avg = t1_rushers_recent.groupby('player')['rush_yds'].agg(['mean', 'count']).reset_index()
                    t1_rush_season_avg = players_df[(players_df['team'] == team1) & (players_df['rush_att'] > 0)].groupby('player')['rush_yds'].mean().reset_index()
                    t1_rush_season_avg.columns = ['player', 'season_avg']
                    t1_rush_form = t1_rush_recent_avg.merge(t1_rush_season_avg, on='player')
                    t1_rush_form['diff'] = t1_rush_form['mean'] - t1_rush_form['season_avg']
                    t1_rush_form = t1_rush_form[t1_rush_form['count'] >= 3].sort_values('mean', ascending=False)

                    if not t1_rush_form.empty:
                        st.markdown("**Rushing:**")
                        for _, p in t1_rush_form.head(3).iterrows():
                            trend = "🔥" if p['diff'] > 15 else "📈" if p['diff'] > 0 else "📉" if p['diff'] < -15 else "➡️"
                            st.markdown(f"{trend} **{p['player']}**: {p['mean']:.1f} yds/game (recent) vs {p['season_avg']:.1f} (season) | {p['diff']:+.1f}")

                # Receiving recent form
                t1_receivers_recent = recent_df[(recent_df['team'] == team1) & (recent_df['rec'] > 0)]
                if not t1_receivers_recent.empty:
                    t1_rec_recent_avg = t1_receivers_recent.groupby('player')['rec_yds'].agg(['mean', 'count']).reset_index()
                    t1_rec_season_avg = players_df[(players_df['team'] == team1) & (players_df['rec'] > 0)].groupby('player')['rec_yds'].mean().reset_index()
                    t1_rec_season_avg.columns = ['player', 'season_avg']
                    t1_rec_form = t1_rec_recent_avg.merge(t1_rec_season_avg, on='player')
                    t1_rec_form['diff'] = t1_rec_form['mean'] - t1_rec_form['season_avg']
                    t1_rec_form = t1_rec_form[t1_rec_form['count'] >= 3].sort_values('mean', ascending=False)

                    if not t1_rec_form.empty:
                        st.markdown("**Receiving:**")
                        for _, p in t1_rec_form.head(3).iterrows():
                            trend = "🔥" if p['diff'] > 15 else "📈" if p['diff'] > 0 else "📉" if p['diff'] < -15 else "➡️"
                            st.markdown(f"{trend} **{p['player']}**: {p['mean']:.1f} yds/game (recent) vs {p['season_avg']:.1f} (season) | {p['diff']:+.1f}")

            with col2:
                st.markdown(f"#### {team2} Key Players")

                # Passing recent form
                t2_passers_recent = recent_df[(recent_df['team'] == team2) & (recent_df['pass_att'] > 0)]
                if not t2_passers_recent.empty:
                    t2_pass_recent_avg = t2_passers_recent.groupby('player')['pass_yds'].agg(['mean', 'count']).reset_index()
                    t2_pass_season_avg = players_df[(players_df['team'] == team2) & (players_df['pass_att'] > 0)].groupby('player')['pass_yds'].mean().reset_index()
                    t2_pass_season_avg.columns = ['player', 'season_avg']
                    t2_pass_form = t2_pass_recent_avg.merge(t2_pass_season_avg, on='player')
                    t2_pass_form['diff'] = t2_pass_form['mean'] - t2_pass_form['season_avg']
                    t2_pass_form = t2_pass_form[t2_pass_form['count'] >= 3].sort_values('mean', ascending=False)

                    if not t2_pass_form.empty:
                        st.markdown("**Passing:**")
                        for _, p in t2_pass_form.head(2).iterrows():
                            trend = "🔥" if p['diff'] > 20 else "📈" if p['diff'] > 0 else "📉" if p['diff'] < -20 else "➡️"
                            st.markdown(f"{trend} **{p['player']}**: {p['mean']:.1f} yds/game (recent) vs {p['season_avg']:.1f} (season) | {p['diff']:+.1f}")

                # Rushing recent form
                t2_rushers_recent = recent_df[(recent_df['team'] == team2) & (recent_df['rush_att'] > 0)]
                if not t2_rushers_recent.empty:
                    t2_rush_recent_avg = t2_rushers_recent.groupby('player')['rush_yds'].agg(['mean', 'count']).reset_index()
                    t2_rush_season_avg = players_df[(players_df['team'] == team2) & (players_df['rush_att'] > 0)].groupby('player')['rush_yds'].mean().reset_index()
                    t2_rush_season_avg.columns = ['player', 'season_avg']
                    t2_rush_form = t2_rush_recent_avg.merge(t2_rush_season_avg, on='player')
                    t2_rush_form['diff'] = t2_rush_form['mean'] - t2_rush_form['season_avg']
                    t2_rush_form = t2_rush_form[t2_rush_form['count'] >= 3].sort_values('mean', ascending=False)

                    if not t2_rush_form.empty:
                        st.markdown("**Rushing:**")
                        for _, p in t2_rush_form.head(3).iterrows():
                            trend = "🔥" if p['diff'] > 15 else "📈" if p['diff'] > 0 else "📉" if p['diff'] < -15 else "➡️"
                            st.markdown(f"{trend} **{p['player']}**: {p['mean']:.1f} yds/game (recent) vs {p['season_avg']:.1f} (season) | {p['diff']:+.1f}")

                # Receiving recent form
                t2_receivers_recent = recent_df[(recent_df['team'] == team2) & (recent_df['rec'] > 0)]
                if not t2_receivers_recent.empty:
                    t2_rec_recent_avg = t2_receivers_recent.groupby('player')['rec_yds'].agg(['mean', 'count']).reset_index()
                    t2_rec_season_avg = players_df[(players_df['team'] == team2) & (players_df['rec'] > 0)].groupby('player')['rec_yds'].mean().reset_index()
                    t2_rec_season_avg.columns = ['player', 'season_avg']
                    t2_rec_form = t2_rec_recent_avg.merge(t2_rec_season_avg, on='player')
                    t2_rec_form['diff'] = t2_rec_form['mean'] - t2_rec_form['season_avg']
                    t2_rec_form = t2_rec_form[t2_rec_form['count'] >= 3].sort_values('mean', ascending=False)

                    if not t2_rec_form.empty:
                        st.markdown("**Receiving:**")
                        for _, p in t2_rec_form.head(3).iterrows():
                            trend = "🔥" if p['diff'] > 15 else "📈" if p['diff'] > 0 else "📉" if p['diff'] < -15 else "➡️"
                            st.markdown(f"{trend} **{p['player']}**: {p['mean']:.1f} yds/game (recent) vs {p['season_avg']:.1f} (season) | {p['diff']:+.1f}")

        with tab2:
            st.markdown("### Home vs Away Performance Splits")

            # Join with games to get location
            games_query = f"SELECT game_id, home_team_abbr, away_team_abbr FROM games WHERE season={season}"
            games_loc = query(games_query)

            # Add location to players_df
            players_with_loc = players_df.merge(games_loc, on='game_id', how='left')
            players_with_loc['location'] = players_with_loc.apply(
                lambda row: 'home' if row['team'] == row['home_team_abbr'] else 'away', axis=1
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"#### {team1} Home/Away Splits")

                t1_data = players_with_loc[players_with_loc['team'] == team1]

                # Passing splits
                t1_pass = t1_data[t1_data['pass_att'] > 0].groupby(['player', 'location'])['pass_yds'].mean().reset_index()
                t1_pass_pivot = t1_pass.pivot(index='player', columns='location', values='pass_yds').reset_index()
                if 'home' in t1_pass_pivot.columns and 'away' in t1_pass_pivot.columns:
                    t1_pass_pivot = t1_pass_pivot.dropna()
                    if not t1_pass_pivot.empty:
                        st.markdown("**QB Performance:**")
                        for _, p in t1_pass_pivot.iterrows():
                            diff = p['home'] - p['away']
                            better = "🏠" if diff > 0 else "✈️"
                            st.markdown(f"{better} **{p['player']}**: Home: {p['home']:.1f} yds | Away: {p['away']:.1f} yds | Diff: {diff:+.1f}")

                # Rushing splits
                t1_rush = t1_data[t1_data['rush_att'] > 0].groupby(['player', 'location'])['rush_yds'].mean().reset_index()
                t1_rush_pivot = t1_rush.pivot(index='player', columns='location', values='rush_yds').reset_index()
                if 'home' in t1_rush_pivot.columns and 'away' in t1_rush_pivot.columns:
                    t1_rush_pivot = t1_rush_pivot.dropna()
                    if not t1_rush_pivot.empty:
                        st.markdown("**Top Rushers:**")
                        for _, p in t1_rush_pivot.head(3).iterrows():
                            diff = p['home'] - p['away']
                            better = "🏠" if diff > 0 else "✈️"
                            st.markdown(f"{better} **{p['player']}**: Home: {p['home']:.1f} yds | Away: {p['away']:.1f} yds | Diff: {diff:+.1f}")

                # Receiving splits
                t1_rec = t1_data[t1_data['rec'] > 0].groupby(['player', 'location'])['rec_yds'].mean().reset_index()
                t1_rec_pivot = t1_rec.pivot(index='player', columns='location', values='rec_yds').reset_index()
                if 'home' in t1_rec_pivot.columns and 'away' in t1_rec_pivot.columns:
                    t1_rec_pivot = t1_rec_pivot.dropna()
                    if not t1_rec_pivot.empty:
                        st.markdown("**Top Receivers:**")
                        for _, p in t1_rec_pivot.head(3).iterrows():
                            diff = p['home'] - p['away']
                            better = "🏠" if diff > 0 else "✈️"
                            st.markdown(f"{better} **{p['player']}**: Home: {p['home']:.1f} yds | Away: {p['away']:.1f} yds | Diff: {diff:+.1f}")

            with col2:
                st.markdown(f"#### {team2} Home/Away Splits")

                t2_data = players_with_loc[players_with_loc['team'] == team2]

                # Passing splits
                t2_pass = t2_data[t2_data['pass_att'] > 0].groupby(['player', 'location'])['pass_yds'].mean().reset_index()
                t2_pass_pivot = t2_pass.pivot(index='player', columns='location', values='pass_yds').reset_index()
                if 'home' in t2_pass_pivot.columns and 'away' in t2_pass_pivot.columns:
                    t2_pass_pivot = t2_pass_pivot.dropna()
                    if not t2_pass_pivot.empty:
                        st.markdown("**QB Performance:**")
                        for _, p in t2_pass_pivot.iterrows():
                            diff = p['home'] - p['away']
                            better = "🏠" if diff > 0 else "✈️"
                            st.markdown(f"{better} **{p['player']}**: Home: {p['home']:.1f} yds | Away: {p['away']:.1f} yds | Diff: {diff:+.1f}")

                # Rushing splits
                t2_rush = t2_data[t2_data['rush_att'] > 0].groupby(['player', 'location'])['rush_yds'].mean().reset_index()
                t2_rush_pivot = t2_rush.pivot(index='player', columns='location', values='rush_yds').reset_index()
                if 'home' in t2_rush_pivot.columns and 'away' in t2_rush_pivot.columns:
                    t2_rush_pivot = t2_rush_pivot.dropna()
                    if not t2_rush_pivot.empty:
                        st.markdown("**Top Rushers:**")
                        for _, p in t2_rush_pivot.head(3).iterrows():
                            diff = p['home'] - p['away']
                            better = "🏠" if diff > 0 else "✈️"
                            st.markdown(f"{better} **{p['player']}**: Home: {p['home']:.1f} yds | Away: {p['away']:.1f} yds | Diff: {diff:+.1f}")

                # Receiving splits
                t2_rec = t2_data[t2_data['rec'] > 0].groupby(['player', 'location'])['rec_yds'].mean().reset_index()
                t2_rec_pivot = t2_rec.pivot(index='player', columns='location', values='rec_yds').reset_index()
                if 'home' in t2_rec_pivot.columns and 'away' in t2_rec_pivot.columns:
                    t2_rec_pivot = t2_rec_pivot.dropna()
                    if not t2_rec_pivot.empty:
                        st.markdown("**Top Receivers:**")
                        for _, p in t2_rec_pivot.head(3).iterrows():
                            diff = p['home'] - p['away']
                            better = "🏠" if diff > 0 else "✈️"
                            st.markdown(f"{better} **{p['player']}**: Home: {p['home']:.1f} yds | Away: {p['away']:.1f} yds | Diff: {diff:+.1f}")

        with tab3:
            st.markdown("### Performance Trajectory Analysis")
            st.caption("Analyzing if players are trending up, down, or staying consistent")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"#### {team1} Trending Players")

                # Calculate week-over-week trends for top players
                t1_players = players_df[players_df['team'] == team1].copy()
                t1_players = t1_players.sort_values(['player', 'week'])

                # Rushing trends
                t1_rush_weekly = t1_players[t1_players['rush_att'] > 0].groupby('player').filter(lambda x: len(x) >= 4)
                if not t1_rush_weekly.empty:
                    st.markdown("**Rushing Trends:**")
                    for player in t1_rush_weekly['player'].unique()[:3]:
                        player_data = t1_rush_weekly[t1_rush_weekly['player'] == player].sort_values('week')
                        if len(player_data) >= 4:
                            # Simple linear trend
                            weeks = player_data['week'].values
                            yds = player_data['rush_yds'].values
                            if len(weeks) > 1:
                                slope = (yds[-2:].mean() - yds[:2].mean()) / (weeks[-2:].mean() - weeks[:2].mean()) if (weeks[-2:].mean() - weeks[:2].mean()) != 0 else 0
                                trend_icon = "📈" if slope > 5 else "📉" if slope < -5 else "➡️"
                                recent_avg = yds[-3:].mean()
                                st.markdown(f"{trend_icon} **{player}**: Recent avg: {recent_avg:.1f} yds | Trend: {slope:+.1f} yds/week")

                # Receiving trends
                t1_rec_weekly = t1_players[t1_players['rec'] > 0].groupby('player').filter(lambda x: len(x) >= 4)
                if not t1_rec_weekly.empty:
                    st.markdown("**Receiving Trends:**")
                    for player in t1_rec_weekly.groupby('player')['rec_yds'].sum().nlargest(3).index:
                        player_data = t1_rec_weekly[t1_rec_weekly['player'] == player].sort_values('week')
                        if len(player_data) >= 4:
                            weeks = player_data['week'].values
                            yds = player_data['rec_yds'].values
                            if len(weeks) > 1:
                                slope = (yds[-2:].mean() - yds[:2].mean()) / (weeks[-2:].mean() - weeks[:2].mean()) if (weeks[-2:].mean() - weeks[:2].mean()) != 0 else 0
                                trend_icon = "📈" if slope > 5 else "📉" if slope < -5 else "➡️"
                                recent_avg = yds[-3:].mean()
                                st.markdown(f"{trend_icon} **{player}**: Recent avg: {recent_avg:.1f} yds | Trend: {slope:+.1f} yds/week")

            with col2:
                st.markdown(f"#### {team2} Trending Players")

                t2_players = players_df[players_df['team'] == team2].copy()
                t2_players = t2_players.sort_values(['player', 'week'])

                # Rushing trends
                t2_rush_weekly = t2_players[t2_players['rush_att'] > 0].groupby('player').filter(lambda x: len(x) >= 4)
                if not t2_rush_weekly.empty:
                    st.markdown("**Rushing Trends:**")
                    for player in t2_rush_weekly['player'].unique()[:3]:
                        player_data = t2_rush_weekly[t2_rush_weekly['player'] == player].sort_values('week')
                        if len(player_data) >= 4:
                            weeks = player_data['week'].values
                            yds = player_data['rush_yds'].values
                            if len(weeks) > 1:
                                slope = (yds[-2:].mean() - yds[:2].mean()) / (weeks[-2:].mean() - weeks[:2].mean()) if (weeks[-2:].mean() - weeks[:2].mean()) != 0 else 0
                                trend_icon = "📈" if slope > 5 else "📉" if slope < -5 else "➡️"
                                recent_avg = yds[-3:].mean()
                                st.markdown(f"{trend_icon} **{player}**: Recent avg: {recent_avg:.1f} yds | Trend: {slope:+.1f} yds/week")

                # Receiving trends
                t2_rec_weekly = t2_players[t2_players['rec'] > 0].groupby('player').filter(lambda x: len(x) >= 4)
                if not t2_rec_weekly.empty:
                    st.markdown("**Receiving Trends:**")
                    for player in t2_rec_weekly.groupby('player')['rec_yds'].sum().nlargest(3).index:
                        player_data = t2_rec_weekly[t2_rec_weekly['player'] == player].sort_values('week')
                        if len(player_data) >= 4:
                            weeks = player_data['week'].values
                            yds = player_data['rec_yds'].values
                            if len(weeks) > 1:
                                slope = (yds[-2:].mean() - yds[:2].mean()) / (weeks[-2:].mean() - weeks[:2].mean()) if (weeks[-2:].mean() - weeks[:2].mean()) != 0 else 0
                                trend_icon = "📈" if slope > 5 else "📉" if slope < -5 else "➡️"
                                recent_avg = yds[-3:].mean()
                                st.markdown(f"{trend_icon} **{player}**: Recent avg: {recent_avg:.1f} yds | Trend: {slope:+.1f} yds/week")

        st.divider()

        # Depth Analysis
        st.subheader("📊 Offensive Depth Comparison")

        col1, col2 = st.columns(2)

        # Calculate depth metrics
        with col1:
            st.markdown(f"### {team1} Depth Chart")

            # Rushing depth
            t1_rushers = players_df[(players_df['team'] == team1) & (players_df['rush_att'] > 0)]
            t1_rush_agg = t1_rushers.groupby('player')['rush_yds'].sum().sort_values(ascending=False)
            t1_rush_total = t1_rush_agg.sum()

            if not t1_rush_agg.empty and t1_rush_total > 0:
                top_rusher_pct = (t1_rush_agg.iloc[0] / t1_rush_total) * 100
                st.metric("Rushing Contributors", len(t1_rush_agg))
                st.metric("Top Rusher %", f"{top_rusher_pct:.1f}%")
                st.metric("100+ Rush Yd Players", len(t1_rush_agg[t1_rush_agg >= 100]))

            # Receiving depth
            t1_receivers = players_df[(players_df['team'] == team1) & (players_df['rec'] > 0)]
            t1_rec_agg = t1_receivers.groupby('player')['rec_yds'].sum().sort_values(ascending=False)
            t1_rec_total = t1_rec_agg.sum()

            if not t1_rec_agg.empty and t1_rec_total > 0:
                top_receiver_pct = (t1_rec_agg.iloc[0] / t1_rec_total) * 100
                st.metric("Receiving Targets", len(t1_rec_agg))
                st.metric("Top Receiver %", f"{top_receiver_pct:.1f}%")
                st.metric("100+ Rec Yd Players", len(t1_rec_agg[t1_rec_agg >= 100]))

            # Distribution quality
            balance_score = 100 - ((top_rusher_pct + top_receiver_pct) / 2)
            st.metric("Balance Score", f"{balance_score:.0f}/100", help="Higher = more balanced distribution")

        with col2:
            st.markdown(f"### {team2} Depth Chart")

            # Rushing depth
            t2_rushers = players_df[(players_df['team'] == team2) & (players_df['rush_att'] > 0)]
            t2_rush_agg = t2_rushers.groupby('player')['rush_yds'].sum().sort_values(ascending=False)
            t2_rush_total = t2_rush_agg.sum()

            if not t2_rush_agg.empty and t2_rush_total > 0:
                top_rusher_pct = (t2_rush_agg.iloc[0] / t2_rush_total) * 100
                st.metric("Rushing Contributors", len(t2_rush_agg))
                st.metric("Top Rusher %", f"{top_rusher_pct:.1f}%")
                st.metric("100+ Rush Yd Players", len(t2_rush_agg[t2_rush_agg >= 100]))

            # Receiving depth
            t2_receivers = players_df[(players_df['team'] == team2) & (players_df['rec'] > 0)]
            t2_rec_agg = t2_receivers.groupby('player')['rec_yds'].sum().sort_values(ascending=False)
            t2_rec_total = t2_rec_agg.sum()

            if not t2_rec_agg.empty and t2_rec_total > 0:
                top_receiver_pct = (t2_rec_agg.iloc[0] / t2_rec_total) * 100
                st.metric("Receiving Targets", len(t2_rec_agg))
                st.metric("Top Receiver %", f"{top_receiver_pct:.1f}%")
                st.metric("100+ Rec Yd Players", len(t2_rec_agg[t2_rec_agg >= 100]))

            # Distribution quality
            balance_score = 100 - ((top_rusher_pct + top_receiver_pct) / 2)
            st.metric("Balance Score", f"{balance_score:.0f}/100", help="Higher = more balanced distribution")

        st.divider()

        # Reliability & Risk Metrics
        st.subheader("🎯 Reliability & Risk Metrics")
        st.caption("Understanding player consistency, floor, ceiling, and game script dependency")

        tab1, tab2, tab3 = st.tabs(["Floor/Ceiling Analysis", "Bust Rate & Consistency", "Game Script Dependency"])

        with tab1:
            st.markdown("### Floor & Ceiling Analysis")
            st.caption("Realistic best-case and worst-case projections per game")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"#### {team1} Player Ranges")

                # Rushing floor/ceiling
                t1_rush_players = players_df[(players_df['team'] == team1) & (players_df['rush_att'] > 0)]
                if not t1_rush_players.empty:
                    t1_rush_range = t1_rush_players.groupby('player')['rush_yds'].agg([
                        ('floor', lambda x: x.quantile(0.25)),  # 25th percentile
                        ('median', 'median'),
                        ('ceiling', lambda x: x.quantile(0.75)),  # 75th percentile
                        ('avg', 'mean'),
                        ('games', 'count')
                    ]).reset_index()
                    t1_rush_range = t1_rush_range[t1_rush_range['games'] >= 3].sort_values('avg', ascending=False)

                    if not t1_rush_range.empty:
                        st.markdown("**Rushing:**")
                        for _, p in t1_rush_range.head(3).iterrows():
                            range_spread = p['ceiling'] - p['floor']
                            reliability = "🟢 Consistent" if range_spread < 30 else "🟡 Variable" if range_spread < 50 else "🔴 Volatile"
                            st.markdown(f"**{p['player']}** ({int(p['games'])} games)")
                            st.markdown(f"- Floor (25%): {p['floor']:.1f} yds | Median: {p['median']:.1f} | Ceiling (75%): {p['ceiling']:.1f}")
                            st.markdown(f"- Avg: {p['avg']:.1f} yds | {reliability} (Range: {range_spread:.1f})")
                            st.markdown("---")

                # Receiving floor/ceiling
                t1_rec_players = players_df[(players_df['team'] == team1) & (players_df['rec'] > 0)]
                if not t1_rec_players.empty:
                    t1_rec_range = t1_rec_players.groupby('player')['rec_yds'].agg([
                        ('floor', lambda x: x.quantile(0.25)),
                        ('median', 'median'),
                        ('ceiling', lambda x: x.quantile(0.75)),
                        ('avg', 'mean'),
                        ('games', 'count')
                    ]).reset_index()
                    t1_rec_range = t1_rec_range[t1_rec_range['games'] >= 3].sort_values('avg', ascending=False)

                    if not t1_rec_range.empty:
                        st.markdown("**Receiving:**")
                        for _, p in t1_rec_range.head(3).iterrows():
                            range_spread = p['ceiling'] - p['floor']
                            reliability = "🟢 Consistent" if range_spread < 30 else "🟡 Variable" if range_spread < 50 else "🔴 Volatile"
                            st.markdown(f"**{p['player']}** ({int(p['games'])} games)")
                            st.markdown(f"- Floor (25%): {p['floor']:.1f} yds | Median: {p['median']:.1f} | Ceiling (75%): {p['ceiling']:.1f}")
                            st.markdown(f"- Avg: {p['avg']:.1f} yds | {reliability} (Range: {range_spread:.1f})")
                            st.markdown("---")

            with col2:
                st.markdown(f"#### {team2} Player Ranges")

                # Rushing floor/ceiling
                t2_rush_players = players_df[(players_df['team'] == team2) & (players_df['rush_att'] > 0)]
                if not t2_rush_players.empty:
                    t2_rush_range = t2_rush_players.groupby('player')['rush_yds'].agg([
                        ('floor', lambda x: x.quantile(0.25)),
                        ('median', 'median'),
                        ('ceiling', lambda x: x.quantile(0.75)),
                        ('avg', 'mean'),
                        ('games', 'count')
                    ]).reset_index()
                    t2_rush_range = t2_rush_range[t2_rush_range['games'] >= 3].sort_values('avg', ascending=False)

                    if not t2_rush_range.empty:
                        st.markdown("**Rushing:**")
                        for _, p in t2_rush_range.head(3).iterrows():
                            range_spread = p['ceiling'] - p['floor']
                            reliability = "🟢 Consistent" if range_spread < 30 else "🟡 Variable" if range_spread < 50 else "🔴 Volatile"
                            st.markdown(f"**{p['player']}** ({int(p['games'])} games)")
                            st.markdown(f"- Floor (25%): {p['floor']:.1f} yds | Median: {p['median']:.1f} | Ceiling (75%): {p['ceiling']:.1f}")
                            st.markdown(f"- Avg: {p['avg']:.1f} yds | {reliability} (Range: {range_spread:.1f})")
                            st.markdown("---")

                # Receiving floor/ceiling
                t2_rec_players = players_df[(players_df['team'] == team2) & (players_df['rec'] > 0)]
                if not t2_rec_players.empty:
                    t2_rec_range = t2_rec_players.groupby('player')['rec_yds'].agg([
                        ('floor', lambda x: x.quantile(0.25)),
                        ('median', 'median'),
                        ('ceiling', lambda x: x.quantile(0.75)),
                        ('avg', 'mean'),
                        ('games', 'count')
                    ]).reset_index()
                    t2_rec_range = t2_rec_range[t2_rec_range['games'] >= 3].sort_values('avg', ascending=False)

                    if not t2_rec_range.empty:
                        st.markdown("**Receiving:**")
                        for _, p in t2_rec_range.head(3).iterrows():
                            range_spread = p['ceiling'] - p['floor']
                            reliability = "🟢 Consistent" if range_spread < 30 else "🟡 Variable" if range_spread < 50 else "🔴 Volatile"
                            st.markdown(f"**{p['player']}** ({int(p['games'])} games)")
                            st.markdown(f"- Floor (25%): {p['floor']:.1f} yds | Median: {p['median']:.1f} | Ceiling (75%): {p['ceiling']:.1f}")
                            st.markdown(f"- Avg: {p['avg']:.1f} yds | {reliability} (Range: {range_spread:.1f})")
                            st.markdown("---")

        with tab2:
            st.markdown("### Bust Rate & Performance Consistency")
            st.caption("How often do players fail to meet expectations?")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"#### {team1} Consistency Metrics")

                # Rushing bust rate
                if not t1_rush_players.empty:
                    st.markdown("**Rushing:**")
                    for player in t1_rush_players['player'].unique()[:3]:
                        player_games = t1_rush_players[t1_rush_players['player'] == player]['rush_yds']
                        if len(player_games) >= 3:
                            avg_yds = player_games.mean()
                            threshold = 50  # Bust = under 50 yards
                            bust_rate = (player_games < threshold).sum() / len(player_games) * 100
                            low_games = (player_games < avg_yds * 0.5).sum()  # Well below average
                            consistency_score = 100 - player_games.std() if len(player_games) > 1 else 0

                            rating = "⭐⭐⭐" if bust_rate < 20 else "⭐⭐" if bust_rate < 40 else "⭐"
                            st.markdown(f"**{player}** {rating}")
                            st.markdown(f"- Bust Rate: {bust_rate:.1f}% (under {threshold} yds)")
                            st.markdown(f"- Games well below avg: {low_games}/{len(player_games)}")
                            st.markdown(f"- Std Dev: {player_games.std():.1f} yds")
                            st.markdown("---")

                # Receiving bust rate
                if not t1_rec_players.empty:
                    st.markdown("**Receiving:**")
                    rec_player_list = t1_rec_players.groupby('player')['rec_yds'].sum().nlargest(3).index
                    for player in rec_player_list:
                        player_games = t1_rec_players[t1_rec_players['player'] == player]['rec_yds']
                        if len(player_games) >= 3:
                            avg_yds = player_games.mean()
                            threshold = 40  # Bust = under 40 yards
                            bust_rate = (player_games < threshold).sum() / len(player_games) * 100
                            low_games = (player_games < avg_yds * 0.5).sum()

                            rating = "⭐⭐⭐" if bust_rate < 20 else "⭐⭐" if bust_rate < 40 else "⭐"
                            st.markdown(f"**{player}** {rating}")
                            st.markdown(f"- Bust Rate: {bust_rate:.1f}% (under {threshold} yds)")
                            st.markdown(f"- Games well below avg: {low_games}/{len(player_games)}")
                            st.markdown(f"- Std Dev: {player_games.std():.1f} yds")
                            st.markdown("---")

            with col2:
                st.markdown(f"#### {team2} Consistency Metrics")

                # Rushing bust rate
                if not t2_rush_players.empty:
                    st.markdown("**Rushing:**")
                    for player in t2_rush_players['player'].unique()[:3]:
                        player_games = t2_rush_players[t2_rush_players['player'] == player]['rush_yds']
                        if len(player_games) >= 3:
                            avg_yds = player_games.mean()
                            threshold = 50
                            bust_rate = (player_games < threshold).sum() / len(player_games) * 100
                            low_games = (player_games < avg_yds * 0.5).sum()

                            rating = "⭐⭐⭐" if bust_rate < 20 else "⭐⭐" if bust_rate < 40 else "⭐"
                            st.markdown(f"**{player}** {rating}")
                            st.markdown(f"- Bust Rate: {bust_rate:.1f}% (under {threshold} yds)")
                            st.markdown(f"- Games well below avg: {low_games}/{len(player_games)}")
                            st.markdown(f"- Std Dev: {player_games.std():.1f} yds")
                            st.markdown("---")

                # Receiving bust rate
                if not t2_rec_players.empty:
                    st.markdown("**Receiving:**")
                    rec_player_list = t2_rec_players.groupby('player')['rec_yds'].sum().nlargest(3).index
                    for player in rec_player_list:
                        player_games = t2_rec_players[t2_rec_players['player'] == player]['rec_yds']
                        if len(player_games) >= 3:
                            avg_yds = player_games.mean()
                            threshold = 40
                            bust_rate = (player_games < threshold).sum() / len(player_games) * 100
                            low_games = (player_games < avg_yds * 0.5).sum()

                            rating = "⭐⭐⭐" if bust_rate < 20 else "⭐⭐" if bust_rate < 40 else "⭐"
                            st.markdown(f"**{player}** {rating}")
                            st.markdown(f"- Bust Rate: {bust_rate:.1f}% (under {threshold} yds)")
                            st.markdown(f"- Games well below avg: {low_games}/{len(player_games)}")
                            st.markdown(f"- Std Dev: {player_games.std():.1f} yds")
                            st.markdown("---")

        with tab3:
            st.markdown("### Game Script Dependency")
            st.caption("How do players perform when their team is winning vs losing?")

            # Add score differential to players_df
            players_with_score = players_df.merge(games_loc, on='game_id', how='left')

            # Calculate score differential at game time
            # Get final scores
            final_scores = query(f"SELECT game_id, home_score, away_score FROM games WHERE season={season}")
            players_with_score = players_with_score.merge(final_scores, on='game_id', how='left')

            # Determine if team won
            players_with_score['team_won'] = players_with_score.apply(
                lambda row: (row['home_score'] > row['away_score']) if row['team'] == row['home_team_abbr']
                else (row['away_score'] > row['home_score']),
                axis=1
            )

            # Calculate margin
            players_with_score['margin'] = players_with_score.apply(
                lambda row: (row['home_score'] - row['away_score']) if row['team'] == row['home_team_abbr']
                else (row['away_score'] - row['home_score']),
                axis=1
            )

            # Categorize game script
            players_with_score['game_script'] = players_with_score['margin'].apply(
                lambda x: 'blowout_win' if x > 14 else 'close_win' if x > 0 else 'close_loss' if x > -14 else 'blowout_loss'
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"#### {team1} Game Script Performance")

                t1_script = players_with_score[players_with_score['team'] == team1]

                # Rushing by game script
                t1_rush_script = t1_script[t1_script['rush_att'] > 0]
                if not t1_rush_script.empty:
                    st.markdown("**Top Rushers:**")
                    for player in t1_rush_script['player'].unique()[:2]:
                        player_data = t1_rush_script[t1_rush_script['player'] == player]
                        wins = player_data[player_data['team_won'] == True]['rush_yds'].mean()
                        losses = player_data[player_data['team_won'] == False]['rush_yds'].mean()

                        if pd.notna(wins) and pd.notna(losses):
                            diff = wins - losses
                            dependency = "📈 Better in wins" if diff > 10 else "📉 Better in losses" if diff < -10 else "➡️ Script neutral"
                            st.markdown(f"**{player}** - {dependency}")
                            st.markdown(f"- Wins: {wins:.1f} yds/game | Losses: {losses:.1f} yds/game | Diff: {diff:+.1f}")
                            st.markdown("---")

                # Receiving by game script
                t1_rec_script = t1_script[t1_script['rec'] > 0]
                if not t1_rec_script.empty:
                    st.markdown("**Top Receivers:**")
                    rec_players = t1_rec_script.groupby('player')['rec_yds'].sum().nlargest(2).index
                    for player in rec_players:
                        player_data = t1_rec_script[t1_rec_script['player'] == player]
                        wins = player_data[player_data['team_won'] == True]['rec_yds'].mean()
                        losses = player_data[player_data['team_won'] == False]['rec_yds'].mean()

                        if pd.notna(wins) and pd.notna(losses):
                            diff = wins - losses
                            dependency = "📈 Better in wins" if diff > 10 else "📉 Better in losses" if diff < -10 else "➡️ Script neutral"
                            st.markdown(f"**{player}** - {dependency}")
                            st.markdown(f"- Wins: {wins:.1f} yds/game | Losses: {losses:.1f} yds/game | Diff: {diff:+.1f}")
                            st.markdown("---")

            with col2:
                st.markdown(f"#### {team2} Game Script Performance")

                t2_script = players_with_score[players_with_score['team'] == team2]

                # Rushing by game script
                t2_rush_script = t2_script[t2_script['rush_att'] > 0]
                if not t2_rush_script.empty:
                    st.markdown("**Top Rushers:**")
                    for player in t2_rush_script['player'].unique()[:2]:
                        player_data = t2_rush_script[t2_rush_script['player'] == player]
                        wins = player_data[player_data['team_won'] == True]['rush_yds'].mean()
                        losses = player_data[player_data['team_won'] == False]['rush_yds'].mean()

                        if pd.notna(wins) and pd.notna(losses):
                            diff = wins - losses
                            dependency = "📈 Better in wins" if diff > 10 else "📉 Better in losses" if diff < -10 else "➡️ Script neutral"
                            st.markdown(f"**{player}** - {dependency}")
                            st.markdown(f"- Wins: {wins:.1f} yds/game | Losses: {losses:.1f} yds/game | Diff: {diff:+.1f}")
                            st.markdown("---")

                # Receiving by game script
                t2_rec_script = t2_script[t2_script['rec'] > 0]
                if not t2_rec_script.empty:
                    st.markdown("**Top Receivers:**")
                    rec_players = t2_rec_script.groupby('player')['rec_yds'].sum().nlargest(2).index
                    for player in rec_players:
                        player_data = t2_rec_script[t2_rec_script['player'] == player]
                        wins = player_data[player_data['team_won'] == True]['rec_yds'].mean()
                        losses = player_data[player_data['team_won'] == False]['rec_yds'].mean()

                        if pd.notna(wins) and pd.notna(losses):
                            diff = wins - losses
                            dependency = "📈 Better in wins" if diff > 10 else "📉 Better in losses" if diff < -10 else "➡️ Script neutral"
                            st.markdown(f"**{player}** - {dependency}")
                            st.markdown(f"- Wins: {wins:.1f} yds/game | Losses: {losses:.1f} yds/game | Diff: {diff:+.1f}")
                            st.markdown("---")

        st.divider()

        # Team Distribution & Balance
        st.subheader("⚖️ Team Distribution & Balance")
        st.caption("Target/touch concentration, red zone specialists, and explosive play analysis")

        tab1, tab2 = st.tabs(["Touch Concentration", "Red Zone & Explosive Plays"])

        with tab1:
            st.markdown("### Workload Distribution Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"#### {team1} Touch Distribution")

                # Calculate total touches (rush att + targets)
                t1_touches = players_df[players_df['team'] == team1].copy()
                t1_touches['touches'] = t1_touches['rush_att'].fillna(0) + t1_touches['targets'].fillna(0)
                t1_touch_dist = t1_touches[t1_touches['touches'] > 0].groupby('player')['touches'].sum().sort_values(ascending=False)

                if not t1_touch_dist.empty:
                    total_touches = t1_touch_dist.sum()
                    top_player_pct = (t1_touch_dist.iloc[0] / total_touches) * 100
                    top_3_pct = (t1_touch_dist.head(3).sum() / total_touches) * 100

                    concentration_rating = "🔴 High Risk" if top_player_pct > 30 else "🟡 Moderate" if top_player_pct > 20 else "🟢 Balanced"

                    st.metric("Top Player Touch %", f"{top_player_pct:.1f}%")
                    st.metric("Top 3 Players Touch %", f"{top_3_pct:.1f}%")
                    st.markdown(f"**Concentration: {concentration_rating}**")

                    st.markdown("**Top Touch Leaders:**")
                    for player, touches in t1_touch_dist.head(5).items():
                        pct = (touches / total_touches) * 100
                        st.markdown(f"- {player}: {int(touches)} touches ({pct:.1f}%)")

            with col2:
                st.markdown(f"#### {team2} Touch Distribution")

                t2_touches = players_df[players_df['team'] == team2].copy()
                t2_touches['touches'] = t2_touches['rush_att'].fillna(0) + t2_touches['targets'].fillna(0)
                t2_touch_dist = t2_touches[t2_touches['touches'] > 0].groupby('player')['touches'].sum().sort_values(ascending=False)

                if not t2_touch_dist.empty:
                    total_touches = t2_touch_dist.sum()
                    top_player_pct = (t2_touch_dist.iloc[0] / total_touches) * 100
                    top_3_pct = (t2_touch_dist.head(3).sum() / total_touches) * 100

                    concentration_rating = "🔴 High Risk" if top_player_pct > 30 else "🟡 Moderate" if top_player_pct > 20 else "🟢 Balanced"

                    st.metric("Top Player Touch %", f"{top_player_pct:.1f}%")
                    st.metric("Top 3 Players Touch %", f"{top_3_pct:.1f}%")
                    st.markdown(f"**Concentration: {concentration_rating}**")

                    st.markdown("**Top Touch Leaders:**")
                    for player, touches in t2_touch_dist.head(5).items():
                        pct = (touches / total_touches) * 100
                        st.markdown(f"- {player}: {int(touches)} touches ({pct:.1f}%)")

        with tab2:
            st.markdown("### Red Zone & Explosive Plays")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"#### {team1} Scoring & Big Plays")

                # TD concentration
                t1_td_scorers = players_df[players_df['team'] == team1].copy()
                t1_td_scorers['total_td'] = t1_td_scorers['pass_td'].fillna(0) + t1_td_scorers['rush_td'].fillna(0) + t1_td_scorers['rec_td'].fillna(0)
                t1_td_leaders = t1_td_scorers[t1_td_scorers['total_td'] > 0].groupby('player')['total_td'].sum().sort_values(ascending=False)

                if not t1_td_leaders.empty:
                    st.markdown("**Red Zone Specialists (TD Leaders):**")
                    for player, tds in t1_td_leaders.head(5).items():
                        st.markdown(f"- {player}: {int(tds)} TDs")

                # Explosive plays (20+ yard plays for rush/rec)
                t1_explosive_rush = players_df[(players_df['team'] == team1) & (players_df['rush_long'] >= 20)]
                t1_explosive_rec = players_df[(players_df['team'] == team1) & (players_df['rec_long'] >= 20)]

                if not t1_explosive_rush.empty or not t1_explosive_rec.empty:
                    st.markdown("**Explosive Play Producers:**")
                    explosive_players = {}
                    for _, row in t1_explosive_rush.iterrows():
                        player = row['player']
                        explosive_players[player] = explosive_players.get(player, 0) + 1
                    for _, row in t1_explosive_rec.iterrows():
                        player = row['player']
                        explosive_players[player] = explosive_players.get(player, 0) + 1

                    for player, count in sorted(explosive_players.items(), key=lambda x: x[1], reverse=True)[:5]:
                        st.markdown(f"- {player}: {count} explosive plays (20+ yds)")

            with col2:
                st.markdown(f"#### {team2} Scoring & Big Plays")

                # TD concentration
                t2_td_scorers = players_df[players_df['team'] == team2].copy()
                t2_td_scorers['total_td'] = t2_td_scorers['pass_td'].fillna(0) + t2_td_scorers['rush_td'].fillna(0) + t2_td_scorers['rec_td'].fillna(0)
                t2_td_leaders = t2_td_scorers[t2_td_scorers['total_td'] > 0].groupby('player')['total_td'].sum().sort_values(ascending=False)

                if not t2_td_leaders.empty:
                    st.markdown("**Red Zone Specialists (TD Leaders):**")
                    for player, tds in t2_td_leaders.head(5).items():
                        st.markdown(f"- {player}: {int(tds)} TDs")

                # Explosive plays
                t2_explosive_rush = players_df[(players_df['team'] == team2) & (players_df['rush_long'] >= 20)]
                t2_explosive_rec = players_df[(players_df['team'] == team2) & (players_df['rec_long'] >= 20)]

                if not t2_explosive_rush.empty or not t2_explosive_rec.empty:
                    st.markdown("**Explosive Play Producers:**")
                    explosive_players = {}
                    for _, row in t2_explosive_rush.iterrows():
                        player = row['player']
                        explosive_players[player] = explosive_players.get(player, 0) + 1
                    for _, row in t2_explosive_rec.iterrows():
                        player = row['player']
                        explosive_players[player] = explosive_players.get(player, 0) + 1

                    for player, count in sorted(explosive_players.items(), key=lambda x: x[1], reverse=True)[:5]:
                        st.markdown(f"- {player}: {count} explosive plays (20+ yds)")

        st.divider()

        # Usage & Availability
        st.subheader("📋 Usage & Availability Trends")
        st.caption("Games played, workload trends, and usage patterns over the season")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"#### {team1} Player Availability & Usage")

            # Games played and consistency
            t1_availability = players_df[players_df['team'] == team1].copy()
            t1_availability['has_stats'] = (
                (t1_availability['rush_att'] > 0) |
                (t1_availability['rec'] > 0) |
                (t1_availability['pass_att'] > 0)
            )

            t1_games_played = t1_availability[t1_availability['has_stats']].groupby('player').size().reset_index(name='games')
            t1_games_played = t1_games_played.sort_values('games', ascending=False)

            total_weeks = players_df['week'].max()

            st.markdown("**Player Availability:**")
            for _, p in t1_games_played.head(8).iterrows():
                availability_pct = (p['games'] / total_weeks) * 100
                status = "✅" if availability_pct >= 90 else "⚠️" if availability_pct >= 70 else "🔴"
                st.markdown(f"{status} **{p['player']}**: {int(p['games'])}/{total_weeks} games ({availability_pct:.0f}%)")

            # Touch/target trend analysis
            st.markdown("**Workload Trends (Last 5 Weeks):**")
            recent_t1 = players_df[(players_df['team'] == team1) & (players_df['week'] >= max(1, max_week - 4))].copy()
            recent_t1['touches'] = recent_t1['rush_att'].fillna(0) + recent_t1['targets'].fillna(0)
            early_t1 = players_df[(players_df['team'] == team1) & (players_df['week'] <= 5)].copy()
            early_t1['touches'] = early_t1['rush_att'].fillna(0) + early_t1['targets'].fillna(0)

            recent_avg = recent_t1.groupby('player')['touches'].mean()
            early_avg = early_t1.groupby('player')['touches'].mean()

            for player in recent_avg.index[:5]:
                if player in early_avg.index:
                    recent_val = recent_avg[player]
                    early_val = early_avg[player]
                    diff = recent_val - early_val
                    trend = "📈 Increasing" if diff > 2 else "📉 Decreasing" if diff < -2 else "➡️ Stable"
                    st.markdown(f"- **{player}**: {recent_val:.1f} touches/game (recent) | {trend}")

        with col2:
            st.markdown(f"#### {team2} Player Availability & Usage")

            # Games played and consistency
            t2_availability = players_df[players_df['team'] == team2].copy()
            t2_availability['has_stats'] = (
                (t2_availability['rush_att'] > 0) |
                (t2_availability['rec'] > 0) |
                (t2_availability['pass_att'] > 0)
            )

            t2_games_played = t2_availability[t2_availability['has_stats']].groupby('player').size().reset_index(name='games')
            t2_games_played = t2_games_played.sort_values('games', ascending=False)

            st.markdown("**Player Availability:**")
            for _, p in t2_games_played.head(8).iterrows():
                availability_pct = (p['games'] / total_weeks) * 100
                status = "✅" if availability_pct >= 90 else "⚠️" if availability_pct >= 70 else "🔴"
                st.markdown(f"{status} **{p['player']}**: {int(p['games'])}/{total_weeks} games ({availability_pct:.0f}%)")

            # Touch/target trend analysis
            st.markdown("**Workload Trends (Last 5 Weeks):**")
            recent_t2 = players_df[(players_df['team'] == team2) & (players_df['week'] >= max(1, max_week - 4))].copy()
            recent_t2['touches'] = recent_t2['rush_att'].fillna(0) + recent_t2['targets'].fillna(0)
            early_t2 = players_df[(players_df['team'] == team2) & (players_df['week'] <= 5)].copy()
            early_t2['touches'] = early_t2['rush_att'].fillna(0) + early_t2['targets'].fillna(0)

            recent_avg = recent_t2.groupby('player')['touches'].mean()
            early_avg = early_t2.groupby('player')['touches'].mean()

            for player in recent_avg.index[:5]:
                if player in early_avg.index:
                    recent_val = recent_avg[player]
                    early_val = early_avg[player]
                    diff = recent_val - early_val
                    trend = "📈 Increasing" if diff > 2 else "📉 Decreasing" if diff < -2 else "➡️ Stable"
                    st.markdown(f"- **{player}**: {recent_val:.1f} touches/game (recent) | {trend}")

        st.divider()

        # Advanced Statistics Section 1: Drive Efficiency
        st.subheader("🎯 Drive Efficiency & Scoring Probability")
        st.caption("How efficiently do teams convert drives into points?")

        t1_drive = calculate_drive_efficiency(team1, season, week)
        t2_drive = calculate_drive_efficiency(team2, season, week)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### {team1} Drive Metrics")
            st.metric("Points Per Drive", f"{t1_drive['points_per_drive']:.2f}")
            st.metric("Red Zone TD %", f"{t1_drive['red_zone_td_pct']*100:.1f}%")
            st.metric("Red Zone Score %", f"{t1_drive['red_zone_score_pct']*100:.1f}%")
            st.metric("Drive Success Rate", f"{t1_drive['drive_success_rate']*100:.1f}%")
            st.metric("Avg Start Position", f"{t1_drive['avg_start_position']:.0f} yd line")

            st.markdown("**Field Position Impact:**")
            st.markdown(f"- Short Field PPD: {t1_drive['short_field_ppd']:.2f}")
            st.markdown(f"- Long Field PPD: {t1_drive['long_field_ppd']:.2f}")

        with col2:
            st.markdown(f"### {team2} Drive Metrics")
            st.metric("Points Per Drive", f"{t2_drive['points_per_drive']:.2f}")
            st.metric("Red Zone TD %", f"{t2_drive['red_zone_td_pct']*100:.1f}%")
            st.metric("Red Zone Score %", f"{t2_drive['red_zone_score_pct']*100:.1f}%")
            st.metric("Drive Success Rate", f"{t2_drive['drive_success_rate']*100:.1f}%")
            st.metric("Avg Start Position", f"{t2_drive['avg_start_position']:.0f} yd line")

            st.markdown("**Field Position Impact:**")
            st.markdown(f"- Short Field PPD: {t2_drive['short_field_ppd']:.2f}")
            st.markdown(f"- Long Field PPD: {t2_drive['long_field_ppd']:.2f}")

        st.divider()

        # Advanced Statistics Section 2: Tempo & Pace
        st.subheader("⏱️ Tempo & Pace Analysis")
        st.caption("Understanding playing speed and its impact")

        t1_tempo = calculate_tempo_metrics(team1, season, week)
        t2_tempo = calculate_tempo_metrics(team2, season, week)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### {team1} Pace Metrics")

            pace_emoji = "🚀" if t1_tempo['pace_rating'] > 70 else "➡️" if t1_tempo['pace_rating'] > 40 else "🐌"
            st.metric("Plays Per Game", f"{t1_tempo['plays_per_game']:.1f}")
            st.metric("Pace Rating", f"{pace_emoji} {t1_tempo['pace_rating']:.0f}/100")
            st.metric("Seconds Per Play", f"{t1_tempo['seconds_per_play']:.1f}s")

            st.markdown("**Quarter-by-Quarter Pace:**")
            st.markdown(f"- Q1: {t1_tempo['q1_pace']:.1f} plays")
            st.markdown(f"- Q2: {t1_tempo['q2_pace']:.1f} plays")
            st.markdown(f"- Q3: {t1_tempo['q3_pace']:.1f} plays")
            st.markdown(f"- Q4: {t1_tempo['q4_pace']:.1f} plays")

        with col2:
            st.markdown(f"### {team2} Pace Metrics")

            pace_emoji = "🚀" if t2_tempo['pace_rating'] > 70 else "➡️" if t2_tempo['pace_rating'] > 40 else "🐌"
            st.metric("Plays Per Game", f"{t2_tempo['plays_per_game']:.1f}")
            st.metric("Pace Rating", f"{pace_emoji} {t2_tempo['pace_rating']:.0f}/100")
            st.metric("Seconds Per Play", f"{t2_tempo['seconds_per_play']:.1f}s")

            st.markdown("**Quarter-by-Quarter Pace:**")
            st.markdown(f"- Q1: {t2_tempo['q1_pace']:.1f} plays")
            st.markdown(f"- Q2: {t2_tempo['q2_pace']:.1f} plays")
            st.markdown(f"- Q3: {t2_tempo['q3_pace']:.1f} plays")
            st.markdown(f"- Q4: {t2_tempo['q4_pace']:.1f} plays")

        st.divider()

        # Advanced Statistics Section 3: Clutch Performance
        st.subheader("🔄 Momentum & Clutch Performance")
        st.caption("Performance under pressure and in critical situations")

        t1_clutch = calculate_clutch_metrics(team1, season, week)
        t2_clutch = calculate_clutch_metrics(team2, season, week)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### {team1} Clutch Metrics")

            clutch_rating = "⭐⭐⭐" if t1_clutch['clutch_win_rate'] > 0.6 else "⭐⭐" if t1_clutch['clutch_win_rate'] > 0.4 else "⭐"
            st.metric("Clutch Rating", clutch_rating)
            st.metric("Clutch Win Rate", f"{t1_clutch['clutch_win_rate']*100:.1f}%")
            st.metric("Close Game Record", t1_clutch['close_game_record'])
            st.metric("Blowout Rate", f"{t1_clutch['blowout_rate']*100:.1f}%")

            st.markdown("**Situational Performance:**")
            st.markdown(f"- Comeback Rate: {t1_clutch['comeback_rate']*100:.1f}%")
            st.markdown(f"- Lead Protection: {t1_clutch['lead_protection_rate']*100:.1f}%")

        with col2:
            st.markdown(f"### {team2} Clutch Metrics")

            clutch_rating = "⭐⭐⭐" if t2_clutch['clutch_win_rate'] > 0.6 else "⭐⭐" if t2_clutch['clutch_win_rate'] > 0.4 else "⭐"
            st.metric("Clutch Rating", clutch_rating)
            st.metric("Clutch Win Rate", f"{t2_clutch['clutch_win_rate']*100:.1f}%")
            st.metric("Close Game Record", t2_clutch['close_game_record'])
            st.metric("Blowout Rate", f"{t2_clutch['blowout_rate']*100:.1f}%")

            st.markdown("**Situational Performance:**")
            st.markdown(f"- Comeback Rate: {t2_clutch['comeback_rate']*100:.1f}%")
            st.markdown(f"- Lead Protection: {t2_clutch['lead_protection_rate']*100:.1f}%")

        st.divider()

        # Advanced Statistics Section 4: Play-Calling Tendencies
        st.subheader("🧠 Play-Calling Tendencies & Predictability")
        st.caption("Offensive strategy patterns and defensive adaptability")

        t1_playcall = calculate_playcalling_tendencies(team1, season, week)
        t2_playcall = calculate_playcalling_tendencies(team2, season, week)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### {team1} Tendencies")

            predictability_color = "🔴" if t1_playcall['predictability_score'] > 70 else "🟡" if t1_playcall['predictability_score'] > 50 else "🟢"
            st.metric("Predictability Score", f"{predictability_color} {t1_playcall['predictability_score']:.0f}/100")
            st.metric("Early Down Aggression", f"{t1_playcall['early_down_aggression']:.0f}/100")

            st.markdown("**Down & Distance Tendencies:**")
            st.markdown(f"- 1st Down Pass %: {t1_playcall['first_down_pass_rate']*100:.1f}%")
            st.markdown(f"- 2nd & Long Pass %: {t1_playcall['second_long_pass_rate']*100:.1f}%")
            st.markdown(f"- 3rd & Long Pass %: {t1_playcall['third_long_pass_rate']*100:.1f}%")

            balance = "Balanced" if 40 < t1_playcall['first_down_pass_rate']*100 < 60 else "Pass Heavy" if t1_playcall['first_down_pass_rate'] > 0.6 else "Run Heavy"
            st.markdown(f"**Overall Philosophy:** {balance}")

        with col2:
            st.markdown(f"### {team2} Tendencies")

            predictability_color = "🔴" if t2_playcall['predictability_score'] > 70 else "🟡" if t2_playcall['predictability_score'] > 50 else "🟢"
            st.metric("Predictability Score", f"{predictability_color} {t2_playcall['predictability_score']:.0f}/100")
            st.metric("Early Down Aggression", f"{t2_playcall['early_down_aggression']:.0f}/100")

            st.markdown("**Down & Distance Tendencies:**")
            st.markdown(f"- 1st Down Pass %: {t2_playcall['first_down_pass_rate']*100:.1f}%")
            st.markdown(f"- 2nd & Long Pass %: {t2_playcall['second_long_pass_rate']*100:.1f}%")
            st.markdown(f"- 3rd & Long Pass %: {t2_playcall['third_long_pass_rate']*100:.1f}%")

            balance = "Balanced" if 40 < t2_playcall['first_down_pass_rate']*100 < 60 else "Pass Heavy" if t2_playcall['first_down_pass_rate'] > 0.6 else "Run Heavy"
            st.markdown(f"**Overall Philosophy:** {balance}")

        st.divider()

        # Advanced Statistics Section 5: Matchup Matrix
        st.subheader("💪 Strength vs Weakness Matchup Matrix")
        st.caption("Advanced head-to-head analysis identifying exploitable advantages")

        matchup = calculate_matchup_advantages(team1, team2, season, week)

        # Overall advantage meter
        if matchup['overall_advantage'] > 15:
            advantage_display = f"🟢 **{team1}** has a significant advantage"
        elif matchup['overall_advantage'] > 5:
            advantage_display = f"🟡 **{team1}** has a slight advantage"
        elif matchup['overall_advantage'] < -15:
            advantage_display = f"🟢 **{team2}** has a significant advantage"
        elif matchup['overall_advantage'] < -5:
            advantage_display = f"🟡 **{team2}** has a slight advantage"
        else:
            advantage_display = "⚖️ **Evenly Matched**"

        st.markdown(f"### Overall Matchup: {advantage_display}")
        st.progress(min(1.0, max(0.0, (matchup['overall_advantage'] + 50) / 100)))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### {team1} Advantages")
            st.metric("Pass Matchup", f"{matchup['pass_matchup_advantage']:+.1f}")
            st.metric("Rush Matchup", f"{matchup['rush_matchup_advantage']:+.1f}")

            if matchup['t1_advantages']:
                st.markdown("**Key Advantages:**")
                for adv in matchup['t1_advantages']:
                    st.markdown(f"- ✅ {adv}")
            else:
                st.markdown("*No significant advantages identified*")

        with col2:
            st.markdown(f"### {team2} Advantages")
            st.metric("Pass Matchup", f"{-matchup['pass_matchup_advantage']:+.1f}")
            st.metric("Rush Matchup", f"{-matchup['rush_matchup_advantage']:+.1f}")

            if matchup['t2_advantages']:
                st.markdown("**Key Advantages:**")
                for adv in matchup['t2_advantages']:
                    st.markdown(f"- ✅ {adv}")
            else:
                st.markdown("*No significant advantages identified*")



# ============================================================================
# Section: Power Rankings
# ============================================================================

def render_power_rankings(season: Optional[int], week: Optional[int]):
    """Display power rankings for all teams with week-by-week progression."""
    st.header("⚡ Power Rankings")
    st.caption("AGGRESSIVE REBALANCE 2025: Wins are DOMINANT (45%) vs EPA (25%). Z-scores capped at ±3.0. Teams with better records ALWAYS rank higher.")

    if not season:
        st.warning("No season data available.")
        return

    # Get all teams
    teams_query = f"SELECT DISTINCT home_team_abbr as team FROM games WHERE season={season} ORDER BY home_team_abbr"
    teams_df = query(teams_query)

    if teams_df.empty:
        st.warning("No teams found for this season.")
        return

    all_teams = teams_df['team'].unique()

    # Determine max week (use selected week if provided, otherwise get latest)
    if week:
        max_week = week
    else:
        max_week_query = f"SELECT MAX(week) as max_week FROM games WHERE season={season}"
        max_week_df = query(max_week_query)
        max_week = int(max_week_df['max_week'].iloc[0]) if not max_week_df.empty else 1

    # Calculate power ratings for each team for each week
    st.info(f"Calculating power ratings for {len(all_teams)} teams across {max_week} week(s)...")

    ratings_data = []

    for current_week in range(1, max_week + 1):
        # Step 1: Calculate league statistics for this week
        league_stats = calculate_league_statistics(season, current_week, list(all_teams))

        # Step 2: Calculate baseline power ratings for all teams (for SOS calculation)
        all_team_powers = {}
        for team in all_teams:
            try:
                # Calculate baseline without quality margin (needs all_team_powers)
                power = calculate_team_power_rating(team, season, current_week, all_team_powers=None, league_stats=league_stats)
                all_team_powers[team] = power
            except:
                all_team_powers[team] = 0.0

        # Step 3: Calculate quality margin-adjusted league stats
        quality_margins = []
        for team in all_teams:
            try:
                qm = calculate_quality_victory_margin(team, season, current_week, all_team_powers)
                quality_margins.append(qm.get('quality_margin_per_game', 0))
            except:
                quality_margins.append(0)

        if len(quality_margins) > 1:
            import statistics
            league_stats['quality_margin'] = {
                'mean': statistics.mean(quality_margins),
                'std': statistics.stdev(quality_margins) if len(quality_margins) > 1 else 1.0
            }

        # Step 4: Calculate final SOS-adjusted power ratings with z-scoring
        week_powers = []
        week_teams_data = []  # Store team data for this week for sorting

        for team in all_teams:
            try:
                final_power = calculate_team_power_rating(team, season, current_week, all_team_powers, league_stats)
                team_record = calculate_win_loss_record(team, season, current_week)
                wins = team_record.get('wins', 0)
                losses = team_record.get('losses', 0)

                week_powers.append(final_power)
                week_teams_data.append({
                    'team': team,
                    'power': final_power,
                    'wins': wins,
                    'losses': losses
                })
            except:
                week_powers.append(0.0)
                week_teams_data.append({
                    'team': team,
                    'power': 0.0,
                    'wins': 0,
                    'losses': 0
                })

        # ORDINAL RANKING: Sort by record first, then by power rating
        # This ensures teams with better records ALWAYS rank higher
        week_teams_data.sort(key=lambda x: (-x['wins'], x['losses'], -x['power']))

        # Normalize power ratings for this week to 1-100 scale with median = 50
        # Apply percentile ranking based on the SORTED order (wins dominant)
        if len(week_teams_data) > 0:
            for rank_idx, team_data in enumerate(week_teams_data):
                # Percentile based on position in sorted list (wins-dominant ranking)
                rank_pct = (rank_idx / len(week_teams_data)) * 100
                # Invert so best team (rank_idx=0) gets ~100, worst gets ~1
                normalized_power = max(1, min(100, 100 - rank_pct))

                # Add to ratings_data
                ratings_data.append({
                    'Team': team_data['team'],
                    'Week': current_week,
                    'Power Rating': normalized_power,
                    'Wins': team_data['wins'],
                    'Losses': team_data['losses']
                })

    # Convert to DataFrame
    ratings_df = pd.DataFrame(ratings_data)

    # Pivot to get teams as rows, weeks as columns
    pivot_df = ratings_df.pivot(index='Team', columns='Week', values='Power Rating')

    # Add current rating column (last week)
    pivot_df['Current Rating'] = pivot_df[max_week]

    # Sort by current rating (descending)
    pivot_df = pivot_df.sort_values('Current Rating', ascending=False)

    # Add rank column
    pivot_df.insert(0, 'Rank', range(1, len(pivot_df) + 1))

    # Reorder columns: Rank, Current Rating, then weeks in order
    week_cols = [col for col in pivot_df.columns if isinstance(col, int)]
    week_cols.sort()
    column_order = ['Rank', 'Current Rating'] + week_cols
    pivot_df = pivot_df[column_order]

    # ========== CALCULATE ELITE TEAM INDICATORS ==========
    # Identify elite teams (quality wins, high scoring, limited opponent scoring)
    elite_team_indicators = {}

    for team in all_teams:
        try:
            # Get all games for this team
            games_sql = """
                SELECT
                    home_team_abbr,
                    away_team_abbr,
                    home_score,
                    away_score
                FROM games
                WHERE season = ?
                AND week <= ?
                AND (home_team_abbr = ? OR away_team_abbr = ?)
            """
            games_df = query(games_sql, (season, max_week, team, team))

            if games_df.empty:
                continue

            # Calculate metrics
            points_scored = []
            points_allowed = []
            quality_wins = 0

            for _, game in games_df.iterrows():
                is_home = game['home_team_abbr'] == team
                opponent = game['away_team_abbr'] if is_home else game['home_team_abbr']
                team_score = game['home_score'] if is_home else game['away_score']
                opp_score = game['away_score'] if is_home else game['home_score']

                points_scored.append(team_score)
                points_allowed.append(opp_score)

                # Check for quality win (win against team with power > 50)
                if team_score > opp_score:
                    opp_power = all_team_powers.get(opponent, 50)
                    if opp_power > 50:
                        quality_wins += 1

            ppg = sum(points_scored) / len(points_scored) if points_scored else 0
            papg = sum(points_allowed) / len(points_allowed) if points_allowed else 0

            # Check if team meets elite criteria
            is_elite = (ppg >= 25.0 and papg < 20.0 and quality_wins >= 2)

            if is_elite:
                elite_team_indicators[team] = "🏆"
            elif ppg >= 25.0 and papg < 20.0:
                elite_team_indicators[team] = "⚡"  # High scoring + strong defense
            elif ppg >= 25.0 and quality_wins >= 2:
                elite_team_indicators[team] = "🔥"  # High scoring + quality wins
            elif papg < 20.0 and quality_wins >= 2:
                elite_team_indicators[team] = "🛡️"  # Strong defense + quality wins

        except:
            continue

    # Display the table
    st.subheader(f"Power Rankings - Season {season}")

    # Show elite team legend if any elite teams exist
    if elite_team_indicators:
        st.caption("🏆 = Elite (All 3: Quality Wins, High Scoring, Strong Defense) | ⚡ = High Scoring + Strong Defense | 🔥 = High Scoring + Quality Wins | 🛡️ = Strong Defense + Quality Wins")

    # Format the dataframe for display
    display_df = pivot_df.copy()
    display_df['Current Rating'] = display_df['Current Rating'].apply(lambda x: f"{x:.2f}")
    for col in week_cols:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}")

    # Reset index to show team names
    display_df = display_df.reset_index()

    # Add elite indicators to team names
    display_df['Team'] = display_df['Team'].apply(lambda x: f"{elite_team_indicators.get(x, '')} {x}".strip())

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Team": st.column_config.TextColumn("Team", width="medium"),
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Current Rating": st.column_config.TextColumn("Current", width="small"),
        }
    )

    # Add visualization
    st.subheader("📊 Power Rating Trends")

    # Show top 10 teams
    top_teams = pivot_df.head(10).index.tolist()

    # Prepare data for plotting
    plot_data = ratings_df[ratings_df['Team'].isin(top_teams)]

    # Create line chart
    import plotly.express as px
    fig = px.line(
        plot_data,
        x='Week',
        y='Power Rating',
        color='Team',
        title=f'Top 10 Teams - Power Rating Progression',
        markers=True
    )
    fig.update_layout(
        xaxis_title='Week',
        yaxis_title='Power Rating',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Component Breakdown Section
    st.subheader("🔍 Component Breakdown")
    st.caption("See how each metric contributes to the final power rating")

    # Team selector for breakdown
    breakdown_team = st.selectbox(
        "Select Team for Detailed Breakdown",
        pivot_df.index.tolist(),
        key="breakdown_team_select"
    )

    if breakdown_team:
        # Calculate detailed components for the selected team
        team_record = calculate_win_loss_record(breakdown_team, season, max_week)
        team_epa = calculate_team_epa(breakdown_team, season, max_week)
        team_quality_margin = calculate_quality_victory_margin(breakdown_team, season, max_week, all_team_powers)
        team_road_dom = calculate_road_dominance(breakdown_team, season, max_week)
        team_high_scoring = calculate_high_scoring_consistency(breakdown_team, season, max_week)
        team_recent_form = calculate_recent_form(breakdown_team, season, max_week)

        # Calculate differential metrics
        off_sr = calculate_success_rates(breakdown_team, season, max_week).get('overall', 0)
        def_sr = calculate_defensive_success_rate(breakdown_team, season, max_week)
        sr_diff = off_sr - def_sr

        off_xpl = calculate_explosive_plays(breakdown_team, season, max_week).get('explosive_rate', 0)
        def_xpl = calculate_defensive_explosive_rate(breakdown_team, season, max_week)
        xpl_diff = off_xpl - def_xpl

        net_epa = team_epa.get('off_epa_per_play', 0) - team_epa.get('def_epa_per_play', 0)

        # Get SOS-adjusted win%
        if all_team_powers:
            team_sos = calculate_sos_adjusted_record(breakdown_team, season, max_week, all_team_powers)
            win_pct = team_sos.get('sos_adj_win_pct', team_record.get('win_pct', 0))
        else:
            win_pct = team_record.get('win_pct', 0)

        # Calculate z-scores
        z_win_pct = z_score(win_pct, league_stats.get('win_pct', {}).get('mean', 0.5), league_stats.get('win_pct', {}).get('std', 0.2))
        z_net_epa = z_score(net_epa, league_stats.get('net_epa', {}).get('mean', 0), league_stats.get('net_epa', {}).get('std', 0.1))
        z_sr_diff = z_score(sr_diff, league_stats.get('sr_diff', {}).get('mean', 0), league_stats.get('sr_diff', {}).get('std', 0.1))
        z_xpl_diff = z_score(xpl_diff, league_stats.get('xpl_diff', {}).get('mean', 0), league_stats.get('xpl_diff', {}).get('std', 0.02))
        z_quality = z_score(team_quality_margin.get('quality_margin_per_game', 0), league_stats.get('quality_margin', {}).get('mean', 0), league_stats.get('quality_margin', {}).get('std', 1))
        z_road = z_score(team_road_dom.get('road_score', 0), league_stats.get('road_dom', {}).get('mean', 0), league_stats.get('road_dom', {}).get('std', 2))
        z_high_scoring = z_score(team_high_scoring.get('high_scoring_score', 0), league_stats.get('high_scoring', {}).get('mean', 0), league_stats.get('high_scoring', {}).get('std', 1))
        z_recent = z_score(team_recent_form.get('recent_form_score', 0), league_stats.get('recent_form', {}).get('mean', 0), league_stats.get('recent_form', {}).get('std', 5))

        # Calculate weighted contributions (WINS DOMINANT 2025 - 70% on Wins)
        contributions = {
            'SOS-Adj Win%': {'raw': win_pct, 'z_score': z_win_pct, 'weight': 0.55, 'contribution': z_win_pct * 0.55},
            'Quality Margin': {'raw': team_quality_margin.get('quality_margin_per_game', 0), 'z_score': z_quality, 'weight': 0.15, 'contribution': z_quality * 0.15},
            'Net EPA': {'raw': net_epa, 'z_score': z_net_epa, 'weight': 0.20, 'contribution': z_net_epa * 0.20},
            'Success Rate Diff': {'raw': sr_diff, 'z_score': z_sr_diff, 'weight': 0.05, 'contribution': z_sr_diff * 0.05},
            'Explosive Rate Diff': {'raw': xpl_diff, 'z_score': z_xpl_diff, 'weight': 0.03, 'contribution': z_xpl_diff * 0.03},
            'Road Dominance': {'raw': team_road_dom.get('road_score', 0), 'z_score': z_road, 'weight': 0.01, 'contribution': z_road * 0.01},
            'High Scoring': {'raw': team_high_scoring.get('high_scoring_score', 0), 'z_score': z_high_scoring, 'weight': 0.01, 'contribution': z_high_scoring * 0.01}
        }

        # Display breakdown
        st.markdown(f"### {breakdown_team} - Week {max_week}")
        st.markdown(f"**Record:** {team_record.get('wins', 0)}-{team_record.get('losses', 0)} ({win_pct:.3f} SOS-adjusted)")
        st.markdown(f"**Final Power Rating:** {pivot_df.loc[breakdown_team, 'Current Rating']}")

        # Create breakdown table
        breakdown_data = []
        total_contribution = 0
        for component, values in contributions.items():
            breakdown_data.append({
                'Component': component,
                'Raw Value': f"{values['raw']:.4f}",
                'Z-Score': f"{values['z_score']:.2f}",
                'Weight': f"{values['weight']:.0%}",
                'Contribution': f"{values['contribution']:.3f}",
                'Scaled (×10)': f"{values['contribution'] * 10:.2f}"
            })
            total_contribution += values['contribution']

        breakdown_df = pd.DataFrame(breakdown_data)
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

        st.markdown(f"**Total Weighted Sum:** {total_contribution:.3f}")
        st.markdown(f"**Final Rating (×10):** {total_contribution * 10:.2f}")

        # Add interpretation
        st.info(f"""
        **Interpretation (AGGRESSIVE REBALANCE 2025):**
        - Z-scores show standard deviations from league average (0 = average, +1 = 1 SD above average, -1 = 1 SD below average)
        - Z-scores are capped at ±3.0 to prevent extreme outliers
        - **WINS ARE NOW DOMINANT:** 45% weight vs EPA 25% (was 15%/40% originally)
        - A team with 0 wins will have a very negative Win% z-score (typically -2 to -3), contributing around **-0.9 to -1.35** to the total
        - Even with excellent efficiency (EPA z-score +2), the contribution is only +0.5, which CANNOT offset poor wins
        - **Teams with better records will ALWAYS rank higher** - efficiency only matters for tiebreaking similar records
        - This ensures Power Rankings reflect actual team success, not just statistical efficiency
        """)

    # Show methodology
    with st.expander("📖 How Power Ratings are Calculated"):
        st.markdown("""
        ### WINS DOMINANT 2025 POWER RATING FORMULA:

        **Power Rating** = [z(SOS-Adj Win%) × 0.55 + z(Quality Margin) × 0.15 + z(Net EPA) × 0.20 +
                           z(SR Diff) × 0.05 + z(Xpl Diff) × 0.03 + z(Road Dom) × 0.01 + z(High Scoring) × 0.01] × 10

        **🏆 WINS ARE EVERYTHING (70% Combined):**
        - 🔥 **SOS-Adjusted Win%: 55%** — Winning is ABSOLUTELY DOMINANT
        - 🏅 **Quality Victory Margin: 15%** — Beating good teams badly matters enormously
        - **Total Wins-Related: 70%** — Your record determines your ranking

        **Efficiency Metrics (30% Combined):**
        - 📊 **Net EPA: 20%** — Efficiency is tertiary to winning
        - 📉 **Success Rate Diff: 5%** — Minor factor
        - 💥 **Explosive Plays: 3%** — Minimal impact (prevents over-weighting volatility)
        - 🛣️ **Road Dominance: 1%** — Tiebreaker only
        - 🎯 **High Scoring: 1%** — Tiebreaker only

        **Core Philosophy:**
        - ✅ **WINS DOMINATE** — 70% weight ensures teams with better records ALWAYS rank higher
        - ✅ **Quality Wins Rewarded** — 15% on quality margin means beating tough teams badly is highly valuable
        - ✅ **Explosive Plays Minimized** — Reduced from 8% to 3% to prevent volatility
        - ✅ **SOS Adjustment** — Win% weighted by opponent power rating
        - ✅ **Z-Score Normalization** — All components standardized to league distribution
        - ✅ **Outlier Protection** — Capped z-scores at ±3.0 & minimum std thresholds

        **Components (Weights Sum to 1.0):**
        1. **SOS-Adjusted Win%** (55% — ABSOLUTELY DOMINANT): Win% × opponent power rating
        2. **Quality Victory Margin** (15% — CRITICAL): Blowout wins (>14 pts) vs tough opponents
        3. **Net EPA** (20% — Tertiary): Off EPA − Def EPA per play; z-scored & capped
        4. **Success Rate Diff** (5%): Off SR − Def SR; minor factor; z-scored & capped
        5. **Explosive Rate Diff** (3%): Off Xpl − Def Xpl; minimal impact; z-scored & capped
        6. **Road Dominance** (1%): Away game performance; tiebreaker only
        7. **High Scoring Consistency** (1%): Games with 30+ points; tiebreaker only

        **Final Score:** Sum of weighted z-scores × 10 → scaled to ~0-100 range

        **Why This Works (Mathematical Proof):**
        - A 4-2 team: z_win ≈ +0.5 to +1.0, contributing **+0.225 to +0.45**
        - A 0-6 team: z_win ≈ -2.0 to -3.0, contributing **-0.90 to -1.35**
        - **Gap: 1.125 to 1.8** just from wins alone
        - Even if 0-6 team has perfect EPA (z=+3.0 capped): +0.75 contribution
        - **Result: 4-2 team ALWAYS ranks higher** regardless of efficiency stats
        - Efficiency metrics only matter for tiebreaking teams with similar records
        """)


# ============================================================================
# Section: Stats & Trends (Anomaly Detection)
# ============================================================================

def render_stats_trends(season: Optional[int], week: Optional[int]):
    """Display statistical anomalies and trends across the league."""
    st.header("📊 Stats & Trends - Anomaly Detection")
    st.caption("Identifying statistical outliers, trends, and interesting patterns league-wide")

    if not season:
        st.warning("Please select a season")
        return

    # Get all teams
    all_teams_query = f"SELECT DISTINCT home_team_abbr as team FROM games WHERE season={season}"
    if week:
        all_teams_query += f" AND week <= {week}"
    all_teams_df = query(all_teams_query)
    all_teams = all_teams_df['team'].tolist()

    if not all_teams:
        st.warning("No data available for selected filters")
        return

    # Calculate league statistics and power ratings
    league_stats = calculate_league_statistics(season, week, all_teams)

    # Calculate power ratings for all teams
    all_team_powers = {}
    all_team_records = {}
    all_team_data = []

    for team in all_teams:
        try:
            power = calculate_team_power_rating(team, season, week, all_team_powers=None, league_stats=league_stats)
            record = calculate_win_loss_record(team, season, week)
            epa = calculate_team_epa(team, season, week)
            sos = calculate_sos_adjusted_record(team, season, week, all_team_powers)

            all_team_powers[team] = power
            all_team_records[team] = record

            all_team_data.append({
                'team': team,
                'power': power,
                'wins': record.get('wins', 0),
                'losses': record.get('losses', 0),
                'win_pct': record.get('win_pct', 0),
                'off_epa': epa.get('off_epa_per_play', 0),
                'def_epa': epa.get('def_epa_per_play', 0),
                'net_epa': epa.get('off_epa_per_play', 0) - epa.get('def_epa_per_play', 0)
            })
        except:
            continue

    # ORDINAL RANKING: Sort by record first, then by power rating
    # This ensures teams with better records ALWAYS rank higher
    all_team_data.sort(key=lambda x: (-x['wins'], x['losses'], -x['power']))

    # Normalize power ratings to 1-100 scale with median = 50
    # Apply percentile ranking based on the SORTED order (wins dominant)
    if len(all_team_data) > 0:
        for rank_idx, team_data in enumerate(all_team_data):
            # Percentile based on position in sorted list (wins-dominant ranking)
            rank_pct = (rank_idx / len(all_team_data)) * 100
            # Invert so best team (rank_idx=0) gets ~100, worst gets ~1
            normalized_power = max(1, min(100, 100 - rank_pct))

            # Update both the dictionary and the list
            all_team_powers[team_data['team']] = normalized_power
            team_data['power'] = normalized_power

    team_df = pd.DataFrame(all_team_data)

    # ========== ELITE TEAMS ANALYSIS ==========
    # Calculate quality wins, scoring offense, and scoring defense for each team

    elite_team_data = []

    for team in all_teams:
        try:
            # Get all games for this team
            games_sql = """
                SELECT
                    game_id,
                    week,
                    home_team_abbr,
                    away_team_abbr,
                    home_score,
                    away_score
                FROM games
                WHERE season = ?
                AND (home_team_abbr = ? OR away_team_abbr = ?)
            """
            params = [season, team, team]
            if week:
                games_sql += " AND week <= ?"
                params.append(week)

            games_df = query(games_sql, tuple(params))

            if games_df.empty:
                continue

            # Calculate points scored and allowed
            points_scored = []
            points_allowed = []
            quality_wins = 0

            for _, game in games_df.iterrows():
                is_home = game['home_team_abbr'] == team
                opponent = game['away_team_abbr'] if is_home else game['home_team_abbr']
                team_score = game['home_score'] if is_home else game['away_score']
                opp_score = game['away_score'] if is_home else game['home_score']

                points_scored.append(team_score)
                points_allowed.append(opp_score)

                # Check if this is a quality win (win against team with power > 50)
                if team_score > opp_score:  # Team won
                    opp_power = all_team_powers.get(opponent, 50)
                    if opp_power > 50:
                        quality_wins += 1

            # Calculate averages
            ppg = sum(points_scored) / len(points_scored) if points_scored else 0
            papg = sum(points_allowed) / len(points_allowed) if points_allowed else 0

            elite_team_data.append({
                'team': team,
                'ppg': ppg,
                'papg': papg,
                'quality_wins': quality_wins,
                'power': all_team_powers.get(team, 50)
            })
        except:
            continue

    elite_df = pd.DataFrame(elite_team_data)

    # Define elite criteria
    ELITE_PPG_THRESHOLD = 25.0  # High scoring: 25+ ppg
    ELITE_PAPG_THRESHOLD = 20.0  # Strong defense: <20 ppg allowed
    ELITE_QUALITY_WINS_THRESHOLD = 2  # At least 2 quality wins

    # Identify elite teams meeting ALL criteria
    if not elite_df.empty:
        elite_teams = elite_df[
            (elite_df['ppg'] >= ELITE_PPG_THRESHOLD) &
            (elite_df['papg'] < ELITE_PAPG_THRESHOLD) &
            (elite_df['quality_wins'] >= ELITE_QUALITY_WINS_THRESHOLD)
        ].sort_values('power', ascending=False)

        # Display Elite Teams section prominently
        st.markdown("---")
        st.markdown("## 🏆 ELITE TEAMS")
        st.caption("Teams excelling in all three categories: Quality Wins, High Scoring, Limiting Opponent Scoring")

        if not elite_teams.empty:
            st.success(f"**{len(elite_teams)} Elite Team(s) Identified**")

            # Display in columns
            cols = st.columns(min(3, len(elite_teams)))

            for idx, (_, team_data) in enumerate(elite_teams.iterrows()):
                with cols[idx % 3]:
                    st.markdown(f"### {team_data['team']}")
                    st.metric("Power Rating", f"{team_data['power']:.1f}")
                    st.metric("🔥 Points Per Game", f"{team_data['ppg']:.1f}")
                    st.metric("🛡️ Points Allowed", f"{team_data['papg']:.1f}")
                    st.metric("🏅 Quality Wins", f"{team_data['quality_wins']}")
                    st.caption(f"Differential: +{team_data['ppg'] - team_data['papg']:.1f} ppg")

            # Show detailed table
            st.markdown("### Elite Teams Details")
            st.dataframe(
                elite_teams[['team', 'power', 'ppg', 'papg', 'quality_wins']].rename(columns={
                    'team': 'Team',
                    'power': 'Power Rating',
                    'ppg': 'Points/Game',
                    'papg': 'Points Allowed/Game',
                    'quality_wins': 'Quality Wins'
                }),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning(f"No teams currently meet all elite criteria (≥{ELITE_PPG_THRESHOLD} PPG, <{ELITE_PAPG_THRESHOLD} PAPG, ≥{ELITE_QUALITY_WINS_THRESHOLD} quality wins)")

        # Show teams close to elite status
        st.markdown("---")
        st.markdown("### 🎯 Close to Elite Status")
        st.caption("Teams meeting 2 of 3 elite criteria")

        near_elite = elite_df[
            (
                ((elite_df['ppg'] >= ELITE_PPG_THRESHOLD) & (elite_df['papg'] < ELITE_PAPG_THRESHOLD)) |
                ((elite_df['ppg'] >= ELITE_PPG_THRESHOLD) & (elite_df['quality_wins'] >= ELITE_QUALITY_WINS_THRESHOLD)) |
                ((elite_df['papg'] < ELITE_PAPG_THRESHOLD) & (elite_df['quality_wins'] >= ELITE_QUALITY_WINS_THRESHOLD))
            ) &
            ~elite_df['team'].isin(elite_teams['team'] if not elite_teams.empty else [])
        ].sort_values('power', ascending=False)

        if not near_elite.empty:
            for _, team_data in near_elite.iterrows():
                criteria_met = []
                if team_data['ppg'] >= ELITE_PPG_THRESHOLD:
                    criteria_met.append("🔥 High Scoring")
                if team_data['papg'] < ELITE_PAPG_THRESHOLD:
                    criteria_met.append("🛡️ Strong Defense")
                if team_data['quality_wins'] >= ELITE_QUALITY_WINS_THRESHOLD:
                    criteria_met.append("🏅 Quality Wins")

                st.markdown(f"**{team_data['team']}** (Power: {team_data['power']:.1f}) - {' + '.join(criteria_met)}")
                st.caption(f"   {team_data['ppg']:.1f} PPG | {team_data['papg']:.1f} PAPG | {team_data['quality_wins']} quality wins")
        else:
            st.info("No teams meet 2 of 3 criteria")

        st.markdown("---")

    # Tab organization
    tabs = st.tabs(["🔥 Performance Anomalies", "📈 Statistical Outliers", "🎯 Trend Analysis", "⚠️ Flags", "👤 Player Anomalies", "📊 Scoring Insights"])

    # ========== TAB 1: Performance Anomalies ==========
    with tabs[0]:
        st.subheader("🔥 Performance Anomalies")
        st.caption("Teams whose records don't match their underlying performance metrics")

        # Calculate expected wins based on EPA
        team_df['expected_wins_ratio'] = (team_df['net_epa'] - team_df['net_epa'].min()) / (team_df['net_epa'].max() - team_df['net_epa'].min())
        games_played = team_df['wins'] + team_df['losses']
        team_df['expected_wins'] = team_df['expected_wins_ratio'] * games_played
        team_df['win_differential'] = team_df['wins'] - team_df['expected_wins']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🍀 Overperforming (Lucky)")
            st.caption("Winning more than EPA suggests - regression candidates")
            overperformers = team_df[team_df['win_differential'] > 1.5].sort_values('win_differential', ascending=False)

            if not overperformers.empty:
                for _, row in overperformers.iterrows():
                    delta_wins = row['win_differential']
                    st.metric(
                        f"{row['team']} ({row['wins']}-{row['losses']})",
                        f"+{delta_wins:.1f} wins",
                        help=f"Expected: {row['expected_wins']:.1f} wins based on Net EPA"
                    )
                    st.caption(f"   Power: {row['power']:.1f} | Net EPA: {row['net_epa']:+.3f}")
            else:
                st.info("No significant overperformers")

        with col2:
            st.markdown("### 📉 Underperforming (Unlucky)")
            st.caption("Winning less than EPA suggests - bounce-back candidates")
            underperformers = team_df[team_df['win_differential'] < -1.5].sort_values('win_differential')

            if not underperformers.empty:
                for _, row in underperformers.iterrows():
                    delta_wins = row['win_differential']
                    st.metric(
                        f"{row['team']} ({row['wins']}-{row['losses']})",
                        f"{delta_wins:.1f} wins",
                        help=f"Expected: {row['expected_wins']:.1f} wins based on Net EPA"
                    )
                    st.caption(f"   Power: {row['power']:.1f} | Net EPA: {row['net_epa']:+.3f}")
            else:
                st.info("No significant underperformers")

        # Power vs Record Discrepancy
        st.markdown("---")
        st.markdown("### ⚖️ Power Rating vs Record Mismatches")
        team_df['rank_by_wins'] = team_df['win_pct'].rank(method='min', ascending=False)
        team_df['rank_by_power'] = team_df['power'].rank(method='min', ascending=False)
        team_df['rank_difference'] = abs(team_df['rank_by_wins'] - team_df['rank_by_power'])

        mismatches = team_df[team_df['rank_difference'] >= 5].sort_values('rank_difference', ascending=False).head(10)

        if not mismatches.empty:
            st.dataframe(
                mismatches[['team', 'wins', 'losses', 'power', 'rank_by_wins', 'rank_by_power', 'rank_difference']].rename(columns={
                    'team': 'Team',
                    'wins': 'W',
                    'losses': 'L',
                    'power': 'Power',
                    'rank_by_wins': 'Rank by Record',
                    'rank_by_power': 'Rank by Power',
                    'rank_difference': 'Rank Gap'
                }),
                hide_index=True,
                use_container_width=True
            )

    # ========== TAB 2: Statistical Outliers ==========
    with tabs[1]:
        st.subheader("📈 Statistical Outliers")
        st.caption("Teams with extreme performances (>2 standard deviations from mean)")

        # Calculate z-scores
        team_df['off_epa_z'] = (team_df['off_epa'] - team_df['off_epa'].mean()) / team_df['off_epa'].std()
        team_df['def_epa_z'] = (team_df['def_epa'] - team_df['def_epa'].mean()) / team_df['def_epa'].std()
        team_df['net_epa_z'] = (team_df['net_epa'] - team_df['net_epa'].mean()) / team_df['net_epa'].std()

        # Elite offenses
        elite_off = team_df[team_df['off_epa_z'] > 2.0].sort_values('off_epa', ascending=False)
        # Elite defenses (defensive EPA should be negative, so < -2.0 in z-score terms)
        elite_def = team_df[team_df['def_epa_z'] < -2.0].sort_values('def_epa')
        # Terrible offenses
        poor_off = team_df[team_df['off_epa_z'] < -2.0].sort_values('off_epa')
        # Terrible defenses
        poor_def = team_df[team_df['def_epa_z'] > 2.0].sort_values('def_epa', ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔥 Elite Offenses (z > 2.0)")
            if not elite_off.empty:
                for _, row in elite_off.iterrows():
                    st.metric(
                        f"{row['team']}",
                        f"EPA: {row['off_epa']:.3f}",
                        f"z-score: {row['off_epa_z']:.2f}"
                    )
            else:
                st.info("No statistically elite offenses (z > 2.0)")

            st.markdown("### 🛡️ Elite Defenses (z < -2.0)")
            if not elite_def.empty:
                for _, row in elite_def.iterrows():
                    st.metric(
                        f"{row['team']}",
                        f"EPA: {row['def_epa']:.3f}",
                        f"z-score: {row['def_epa_z']:.2f}"
                    )
            else:
                st.info("No statistically elite defenses (z < -2.0)")

        with col2:
            st.markdown("### 📉 Struggling Offenses (z < -2.0)")
            if not poor_off.empty:
                for _, row in poor_off.iterrows():
                    st.metric(
                        f"{row['team']}",
                        f"EPA: {row['off_epa']:.3f}",
                        f"z-score: {row['off_epa_z']:.2f}"
                    )
            else:
                st.info("No statistically poor offenses (z < -2.0)")

            st.markdown("### ⚠️ Struggling Defenses (z > 2.0)")
            if not poor_def.empty:
                for _, row in poor_def.iterrows():
                    st.metric(
                        f"{row['team']}",
                        f"EPA: {row['def_epa']:.3f}",
                        f"z-score: {row['def_epa_z']:.2f}"
                    )
            else:
                st.info("No statistically poor defenses (z > 2.0)")

    # ========== TAB 3: Trend Analysis ==========
    with tabs[2]:
        st.subheader("🎯 Trend Analysis")
        st.caption("Recent performance trends and momentum indicators")

        # Calculate recent form for all teams
        trend_data = []
        for team in all_teams:
            try:
                recent_form = calculate_recent_form(team, season, week)
                team_record = all_team_records.get(team, {})

                trend_data.append({
                    'team': team,
                    'recent_form_score': recent_form.get('recent_form_score', 0),
                    'last_3_margin': recent_form.get('last_3_avg_margin', 0),
                    'wins': team_record.get('wins', 0),
                    'losses': team_record.get('losses', 0)
                })
            except:
                continue

        trend_df = pd.DataFrame(trend_data)

        if not trend_df.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🔥 Hot Teams (Strong Recent Form)")
                hot_teams = trend_df[trend_df['recent_form_score'] > 5].sort_values('recent_form_score', ascending=False).head(8)

                if not hot_teams.empty:
                    for _, row in hot_teams.iterrows():
                        st.metric(
                            f"{row['team']} ({row['wins']}-{row['losses']})",
                            f"Form: {row['recent_form_score']:.1f}",
                            f"Avg Margin: {row['last_3_margin']:+.1f}"
                        )
                else:
                    st.info("No teams with exceptionally strong recent form")

            with col2:
                st.markdown("### ❄️ Cold Teams (Weak Recent Form)")
                cold_teams = trend_df[trend_df['recent_form_score'] < -5].sort_values('recent_form_score').head(8)

                if not cold_teams.empty:
                    for _, row in cold_teams.iterrows():
                        st.metric(
                            f"{row['team']} ({row['wins']}-{row['losses']})",
                            f"Form: {row['recent_form_score']:.1f}",
                            f"Avg Margin: {row['last_3_margin']:+.1f}"
                        )
                else:
                    st.info("No teams with exceptionally weak recent form")

    # ========== TAB 4: Red/Green Flags ==========
    with tabs[3]:
        st.subheader("⚠️ Red Flags & Green Flags")
        st.caption("Warning signs and positive indicators for each team")

        # Calculate flags for all teams
        flag_data = []
        for team in all_teams:
            try:
                record = all_team_records.get(team, {})
                power = all_team_powers.get(team, 50)
                sos_data = calculate_sos_adjusted_record(team, season, week, all_team_powers)

                red_flags = []
                green_flags = []

                # Check for red flags
                if sos_data.get('bad_losses', 0) > 0:
                    red_flags.append(f"❌ {sos_data['bad_losses']} bad loss(es) (vs weak teams)")

                if sos_data.get('avg_opp_power', 50) < 47 and record.get('win_pct', 0) > 0.5:
                    red_flags.append(f"⚠️ Weak schedule (Avg Opp: {sos_data['avg_opp_power']:.1f})")

                win_diff = team_df[team_df['team'] == team]['win_differential'].iloc[0] if not team_df[team_df['team'] == team].empty else 0
                if win_diff > 2:
                    red_flags.append(f"🍀 Overperforming (+{win_diff:.1f} wins vs EPA)")

                # Check for green flags
                if sos_data.get('quality_wins', 0) >= 2:
                    green_flags.append(f"✅ {sos_data['quality_wins']} quality win(s) (vs strong teams)")

                if sos_data.get('avg_opp_power', 50) > 53:
                    green_flags.append(f"💪 Tough schedule (Avg Opp: {sos_data['avg_opp_power']:.1f})")

                if win_diff < -1.5:
                    green_flags.append(f"📈 Underperforming ({win_diff:.1f} wins vs EPA - bounce-back candidate)")

                if red_flags or green_flags:
                    flag_data.append({
                        'team': team,
                        'record': f"{record.get('wins', 0)}-{record.get('losses', 0)}",
                        'power': power,
                        'red_flags': red_flags,
                        'green_flags': green_flags
                    })
            except:
                continue

        # Display teams with flags
        for team_flags in sorted(flag_data, key=lambda x: x['power'], reverse=True):
            with st.expander(f"{team_flags['team']} ({team_flags['record']}) - Power: {team_flags['power']:.1f}"):
                if team_flags['red_flags']:
                    st.markdown("**⚠️ Red Flags:**")
                    for flag in team_flags['red_flags']:
                        st.markdown(f"- {flag}")

                if team_flags['green_flags']:
                    st.markdown("**✅ Green Flags:**")
                    for flag in team_flags['green_flags']:
                        st.markdown(f"- {flag}")

    # ========== TAB 5: Player Anomalies ==========
    with tabs[4]:
        st.subheader("👤 Player Anomalies & Trends")
        st.caption("Individual player statistical outliers, trends, and performance patterns")

        # Filter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            min_games = st.number_input("Min Games Played", min_value=1, max_value=20, value=4)
        with col2:
            stat_filter = st.selectbox("Stat Type", ["All", "Passing", "Rushing", "Receiving"])
        with col3:
            team_filter = st.selectbox("Team Filter", ["All Teams"] + sorted(all_teams))

        # Query player box score data
        player_sql = """
            SELECT
                pbs.player,
                pbs.team,
                g.week,
                pbs.pass_yds,
                pbs.pass_td,
                pbs.pass_att,
                pbs.rush_yds,
                pbs.rush_td,
                pbs.rush_att,
                pbs.rec_yds,
                pbs.rec_td,
                pbs.rec
            FROM player_box_score pbs
            JOIN games g ON pbs.game_id = g.game_id
            WHERE g.season = ?
        """
        params = [season]
        if week:
            player_sql += " AND g.week <= ?"
            params.append(week)
        if team_filter != "All Teams":
            player_sql += " AND pbs.team = ?"
            params.append(team_filter)

        player_sql += " ORDER BY pbs.player, g.week"

        player_data = query(player_sql, tuple(params))

        if player_data.empty:
            st.warning("No player data available")
        else:
            # Process player statistics
            player_stats = []

            for player_name in player_data['player'].unique():
                player_games = player_data[player_data['player'] == player_name]

                if len(player_games) < min_games:
                    continue

                # Calculate season totals and averages
                games_played = len(player_games)
                team = player_games['team'].iloc[-1]  # Most recent team

                # Passing stats
                pass_yds_total = player_games['pass_yds'].sum()
                pass_td_total = player_games['pass_td'].sum()
                pass_att_total = player_games['pass_att'].sum()
                pass_yds_avg = pass_yds_total / games_played if games_played > 0 else 0
                pass_td_avg = pass_td_total / games_played if games_played > 0 else 0

                # Rushing stats
                rush_yds_total = player_games['rush_yds'].sum()
                rush_td_total = player_games['rush_td'].sum()
                rush_att_total = player_games['rush_att'].sum()
                rush_yds_avg = rush_yds_total / games_played if games_played > 0 else 0
                rush_td_avg = rush_td_total / games_played if games_played > 0 else 0
                rush_ypc = rush_yds_total / rush_att_total if rush_att_total > 0 else 0

                # Receiving stats
                rec_yds_total = player_games['rec_yds'].sum()
                rec_td_total = player_games['rec_td'].sum()
                rec_total = player_games['rec'].sum()
                rec_yds_avg = rec_yds_total / games_played if games_played > 0 else 0
                rec_td_avg = rec_td_total / games_played if games_played > 0 else 0
                rec_avg = rec_total / games_played if games_played > 0 else 0
                ypr = rec_yds_total / rec_total if rec_total > 0 else 0

                # Last 3 games for trend analysis
                last_3 = player_games.tail(3)
                last_3_pass_yds = last_3['pass_yds'].mean() if len(last_3) > 0 else 0
                last_3_rush_yds = last_3['rush_yds'].mean() if len(last_3) > 0 else 0
                last_3_rec_yds = last_3['rec_yds'].mean() if len(last_3) > 0 else 0
                last_3_rush_att = last_3['rush_att'].mean() if len(last_3) > 0 else 0
                last_3_rec = last_3['rec'].mean() if len(last_3) > 0 else 0

                # Variance/consistency
                pass_yds_std = player_games['pass_yds'].std() if games_played > 1 else 0
                rush_yds_std = player_games['rush_yds'].std() if games_played > 1 else 0
                rec_yds_std = player_games['rec_yds'].std() if games_played > 1 else 0

                # Coefficient of variation (lower = more consistent)
                pass_cv = pass_yds_std / pass_yds_avg if pass_yds_avg > 0 else 0
                rush_cv = rush_yds_std / rush_yds_avg if rush_yds_avg > 0 else 0
                rec_cv = rec_yds_std / rec_yds_avg if rec_yds_avg > 0 else 0

                # Determine primary position based on volume
                if pass_att_total > 20:
                    position = "QB"
                    primary_yds = pass_yds_avg
                    primary_td = pass_td_avg
                    primary_std = pass_yds_std
                    primary_cv = pass_cv
                    last_3_avg = last_3_pass_yds
                elif rush_att_total > rec_total and rush_att_total > 10:
                    position = "RB"
                    primary_yds = rush_yds_avg
                    primary_td = rush_td_avg
                    primary_std = rush_yds_std
                    primary_cv = rush_cv
                    last_3_avg = last_3_rush_yds
                elif rec_total > 0:
                    position = "WR/TE"
                    primary_yds = rec_yds_avg
                    primary_td = rec_td_avg
                    primary_std = rec_yds_std
                    primary_cv = rec_cv
                    last_3_avg = last_3_rec_yds
                else:
                    continue  # Skip players without clear position

                player_stats.append({
                    'player': player_name,
                    'team': team,
                    'position': position,
                    'games': games_played,
                    'pass_yds_avg': pass_yds_avg,
                    'pass_td_avg': pass_td_avg,
                    'rush_yds_avg': rush_yds_avg,
                    'rush_td_avg': rush_td_avg,
                    'rush_ypc': rush_ypc,
                    'rush_att_avg': rush_att_total / games_played,
                    'rec_yds_avg': rec_yds_avg,
                    'rec_td_avg': rec_td_avg,
                    'rec_avg': rec_avg,
                    'ypr': ypr,
                    'primary_yds': primary_yds,
                    'primary_td': primary_td,
                    'primary_std': primary_std,
                    'primary_cv': primary_cv,
                    'last_3_avg': last_3_avg,
                    'last_3_rush_att': last_3_rush_att,
                    'last_3_rec': last_3_rec
                })

            player_df = pd.DataFrame(player_stats)

            if player_df.empty:
                st.warning(f"No players meet the criteria (min {min_games} games)")
            else:
                # Apply stat filter
                if stat_filter != "All":
                    if stat_filter == "Passing":
                        player_df = player_df[player_df['position'] == 'QB']
                    elif stat_filter == "Rushing":
                        player_df = player_df[player_df['position'] == 'RB']
                    elif stat_filter == "Receiving":
                        player_df = player_df[player_df['position'] == 'WR/TE']

                # Calculate z-scores by position
                for pos in player_df['position'].unique():
                    pos_mask = player_df['position'] == pos
                    pos_data = player_df[pos_mask]

                    if len(pos_data) > 1:
                        mean_yds = pos_data['primary_yds'].mean()
                        std_yds = pos_data['primary_yds'].std()
                        std_yds = max(std_yds, 1.0)  # Prevent division by zero

                        player_df.loc[pos_mask, 'yds_z'] = (pos_data['primary_yds'] - mean_yds) / std_yds

                # Section 1: Performance Outliers by Position
                st.markdown("### 🔥 Performance Outliers by Position")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Quarterbacks (Elite)**")
                    qb_elite = player_df[(player_df['position'] == 'QB') & (player_df['yds_z'] > 1.5)].sort_values('pass_yds_avg', ascending=False).head(5)
                    if not qb_elite.empty:
                        for _, p in qb_elite.iterrows():
                            st.metric(f"{p['player']} ({p['team']})", f"{p['pass_yds_avg']:.1f} yds/g", f"z: {p['yds_z']:.2f}")
                    else:
                        st.info("No elite QBs (z > 1.5)")

                with col2:
                    st.markdown("**Running Backs (Elite)**")
                    rb_elite = player_df[(player_df['position'] == 'RB') & (player_df['yds_z'] > 1.5)].sort_values('rush_yds_avg', ascending=False).head(5)
                    if not rb_elite.empty:
                        for _, p in rb_elite.iterrows():
                            st.metric(f"{p['player']} ({p['team']})", f"{p['rush_yds_avg']:.1f} yds/g", f"{p['rush_ypc']:.2f} ypc")
                    else:
                        st.info("No elite RBs (z > 1.5)")

                with col3:
                    st.markdown("**Receivers (Elite)**")
                    rec_elite = player_df[(player_df['position'] == 'WR/TE') & (player_df['yds_z'] > 1.5)].sort_values('rec_yds_avg', ascending=False).head(5)
                    if not rec_elite.empty:
                        for _, p in rec_elite.iterrows():
                            st.metric(f"{p['player']} ({p['team']})", f"{p['rec_yds_avg']:.1f} yds/g", f"{p['rec_avg']:.1f} rec/g")
                    else:
                        st.info("No elite WR/TEs (z > 1.5)")

                # Section 2: Volume vs Efficiency
                st.markdown("---")
                st.markdown("### 📊 Volume vs Efficiency Discrepancies")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**High Volume, Low Efficiency**")
                    high_vol_low_eff = player_df[
                        (player_df['position'] == 'RB') &
                        (player_df['rush_att_avg'] > 15) &
                        (player_df['rush_ypc'] < 3.8)
                    ].sort_values('rush_att_avg', ascending=False).head(5)

                    if not high_vol_low_eff.empty:
                        for _, p in high_vol_low_eff.iterrows():
                            st.metric(
                                f"{p['player']} ({p['team']})",
                                f"{p['rush_att_avg']:.1f} att/g",
                                f"{p['rush_ypc']:.2f} ypc ⚠️"
                            )
                    else:
                        st.info("No high-volume, low-efficiency RBs")

                with col2:
                    st.markdown("**High Efficiency, Low Volume**")
                    high_eff_low_vol = player_df[
                        (player_df['position'] == 'RB') &
                        (player_df['rush_att_avg'] < 12) &
                        (player_df['rush_att_avg'] > 5) &
                        (player_df['rush_ypc'] > 5.0)
                    ].sort_values('rush_ypc', ascending=False).head(5)

                    if not high_eff_low_vol.empty:
                        for _, p in high_eff_low_vol.iterrows():
                            st.metric(
                                f"{p['player']} ({p['team']})",
                                f"{p['rush_ypc']:.2f} ypc 🔥",
                                f"{p['rush_att_avg']:.1f} att/g"
                            )
                    else:
                        st.info("No high-efficiency, low-volume RBs")

                # Section 3: Boom/Bust Analysis
                st.markdown("---")
                st.markdown("### 🎯 Boom/Bust Analysis (Consistency)")
                st.caption("Lower CV (coefficient of variation) = more consistent")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Most Consistent (Low Variance)**")
                    consistent = player_df[player_df['primary_cv'] > 0].sort_values('primary_cv').head(8)
                    if not consistent.empty:
                        for _, p in consistent.iterrows():
                            st.metric(
                                f"{p['player']} ({p['position']})",
                                f"{p['primary_yds']:.1f} yds/g",
                                f"CV: {p['primary_cv']:.2f} ✅"
                            )

                with col2:
                    st.markdown("**Most Volatile (High Variance)**")
                    volatile = player_df[player_df['primary_cv'] > 0].sort_values('primary_cv', ascending=False).head(8)
                    if not volatile.empty:
                        for _, p in volatile.iterrows():
                            st.metric(
                                f"{p['player']} ({p['position']})",
                                f"{p['primary_yds']:.1f} yds/g",
                                f"CV: {p['primary_cv']:.2f} ⚠️"
                            )

                # Section 4: Player Trends (Last 3 Games)
                st.markdown("---")
                st.markdown("### 📈 Player Trends (Last 3 Games vs Season Avg)")

                player_df['trend_pct'] = ((player_df['last_3_avg'] - player_df['primary_yds']) / player_df['primary_yds'] * 100).fillna(0)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**🔥 Hot Streaks (Trending Up)**")
                    hot = player_df[player_df['trend_pct'] > 20].sort_values('trend_pct', ascending=False).head(8)
                    if not hot.empty:
                        for _, p in hot.iterrows():
                            st.metric(
                                f"{p['player']} ({p['position']})",
                                f"L3: {p['last_3_avg']:.1f} yds/g",
                                f"+{p['trend_pct']:.0f}% vs season"
                            )
                    else:
                        st.info("No significant hot streaks")

                with col2:
                    st.markdown("**❄️ Cold Streaks (Trending Down)**")
                    cold = player_df[player_df['trend_pct'] < -20].sort_values('trend_pct').head(8)
                    if not cold.empty:
                        for _, p in cold.iterrows():
                            st.metric(
                                f"{p['player']} ({p['position']})",
                                f"L3: {p['last_3_avg']:.1f} yds/g",
                                f"{p['trend_pct']:.0f}% vs season"
                            )
                    else:
                        st.info("No significant cold streaks")

                # Section 5: Volume Trends
                st.markdown("---")
                st.markdown("### 📊 Volume Trends (Usage Changes)")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**RB Workload Trends**")
                    player_df['rush_att_trend'] = ((player_df['last_3_rush_att'] - player_df['rush_att_avg']) / player_df['rush_att_avg'] * 100).fillna(0)
                    rb_trend = player_df[
                        (player_df['position'] == 'RB') &
                        (player_df['rush_att_avg'] > 5) &
                        (abs(player_df['rush_att_trend']) > 25)
                    ].sort_values('rush_att_trend', ascending=False).head(6)

                    if not rb_trend.empty:
                        for _, p in rb_trend.iterrows():
                            direction = "📈" if p['rush_att_trend'] > 0 else "📉"
                            st.metric(
                                f"{p['player']} {direction}",
                                f"L3: {p['last_3_rush_att']:.1f} att/g",
                                f"{p['rush_att_trend']:+.0f}%"
                            )

                with col2:
                    st.markdown("**WR/TE Target Trends**")
                    player_df['rec_trend'] = ((player_df['last_3_rec'] - player_df['rec_avg']) / player_df['rec_avg'] * 100).fillna(0)
                    rec_trend = player_df[
                        (player_df['position'] == 'WR/TE') &
                        (player_df['rec_avg'] > 3) &
                        (abs(player_df['rec_trend']) > 25)
                    ].sort_values('rec_trend', ascending=False).head(6)

                    if not rec_trend.empty:
                        for _, p in rec_trend.iterrows():
                            direction = "📈" if p['rec_trend'] > 0 else "📉"
                            st.metric(
                                f"{p['player']} {direction}",
                                f"L3: {p['last_3_rec']:.1f} rec/g",
                                f"{p['rec_trend']:+.0f}%"
                            )

    # ========== TAB 6: Scoring Insights ==========
    with tabs[5]:
        st.subheader("📊 Scoring Insights")
        st.caption("Comprehensive scoring statistics, trends, and interesting patterns")

        # Query all games for scoring analysis
        games_sql = """
            SELECT
                home_team_abbr,
                away_team_abbr,
                home_score,
                away_score,
                week
            FROM games
            WHERE season = ?
        """
        params_games = [season]
        if week:
            games_sql += " AND week <= ?"
            params_games.append(week)

        games_df = query(games_sql, tuple(params_games))

        if games_df.empty:
            st.warning("No games data available")
        else:
            # Calculate total scores
            games_df['total_score'] = games_df['home_score'] + games_df['away_score']
            games_df['margin'] = abs(games_df['home_score'] - games_df['away_score'])

            # Section 1: League-Wide Scoring Trends
            st.markdown("### 📈 League-Wide Scoring Trends")

            col1, col2, col3, col4 = st.columns(4)

            # Calculate league-wide statistics
            all_scores = pd.concat([games_df['home_score'], games_df['away_score']])
            avg_score = all_scores.mean()
            median_score = all_scores.median()
            std_score = all_scores.std()

            avg_total = games_df['total_score'].mean()
            median_total = games_df['total_score'].median()
            std_total = games_df['total_score'].std()

            with col1:
                st.metric("Avg Points/Team", f"{avg_score:.1f}")
                st.metric("Median Points/Team", f"{median_score:.1f}")
                st.metric("Std Dev", f"{std_score:.1f}")

            with col2:
                st.metric("Avg Total Score", f"{avg_total:.1f}")
                st.metric("Median Total", f"{median_total:.1f}")
                st.metric("Std Dev Total", f"{std_total:.1f}")

            with col3:
                high_game = games_df.loc[games_df['total_score'].idxmax()]
                low_game = games_df.loc[games_df['total_score'].idxmin()]
                st.metric("Highest Scoring Game", f"{high_game['total_score']:.0f}")
                st.caption(f"Week {high_game['week']}: {high_game['home_team_abbr']} vs {high_game['away_team_abbr']}")
                st.metric("Lowest Scoring Game", f"{low_game['total_score']:.0f}")
                st.caption(f"Week {low_game['week']}: {low_game['home_team_abbr']} vs {low_game['away_team_abbr']}")

            with col4:
                high_scoring_games = len(all_scores[all_scores > 40])
                total_games = len(games_df) * 2  # Each game has 2 team performances
                low_scoring_games = len(all_scores[all_scores < 10])

                st.metric("40+ Point Games", high_scoring_games)
                st.caption(f"{high_scoring_games / total_games * 100:.1f}% of performances")
                st.metric("<10 Point Games", low_scoring_games)
                st.caption(f"{low_scoring_games / total_games * 100:.1f}% of performances")

            # Section 2: Home vs Away Scoring
            st.markdown("---")
            st.markdown("### 🏠 Home vs Away Scoring")

            # Calculate per-team home/away stats
            team_scoring = []
            for team in all_teams:
                home_games = games_df[games_df['home_team_abbr'] == team]
                away_games = games_df[games_df['away_team_abbr'] == team]

                home_avg = home_games['home_score'].mean() if not home_games.empty else 0
                away_avg = away_games['away_score'].mean() if not away_games.empty else 0
                home_std = home_games['home_score'].std() if len(home_games) > 1 else 0
                away_std = away_games['away_score'].std() if len(away_games) > 1 else 0

                team_scoring.append({
                    'team': team,
                    'home_avg': home_avg,
                    'away_avg': away_avg,
                    'home_std': home_std,
                    'away_std': away_std,
                    'split': home_avg - away_avg,
                    'overall_avg': (home_avg + away_avg) / 2,
                    'overall_std': (home_std + away_std) / 2,
                    'home_games': len(home_games),
                    'away_games': len(away_games)
                })

            team_scoring_df = pd.DataFrame(team_scoring)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Best Home Scorers**")
                home_scorers = team_scoring_df.sort_values('home_avg', ascending=False).head(8)
                for _, t in home_scorers.iterrows():
                    st.metric(
                        f"{t['team']}",
                        f"{t['home_avg']:.1f} pts/g",
                        f"+{t['split']:.1f} vs away" if t['split'] > 0 else f"{t['split']:.1f} vs away"
                    )

            with col2:
                st.markdown("**Best Road Scorers**")
                road_scorers = team_scoring_df.sort_values('away_avg', ascending=False).head(8)
                for _, t in road_scorers.iterrows():
                    st.metric(
                        f"{t['team']}",
                        f"{t['away_avg']:.1f} pts/g",
                        f"{-t['split']:.1f} vs home" if t['split'] < 0 else f"+{-t['split']:.1f} vs home"
                    )

            st.markdown("**Biggest Home/Away Splits**")
            splits = team_scoring_df.sort_values('split', ascending=False).head(6)
            col1, col2 = st.columns([2, 1])
            with col1:
                for _, t in splits.iterrows():
                    pct_diff = (t['split'] / t['overall_avg'] * 100) if t['overall_avg'] > 0 else 0
                    st.metric(
                        f"{t['team']}",
                        f"{t['split']:+.1f} pts difference",
                        f"{pct_diff:+.0f}%"
                    )

            # Section 3: Scoring Consistency
            st.markdown("---")
            st.markdown("### 🎯 Scoring Consistency")
            st.caption("Lower standard deviation = more consistent scoring")

            # Calculate coefficient of variation
            team_scoring_df['cv'] = team_scoring_df['overall_std'] / team_scoring_df['overall_avg']

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Most Consistent Scorers**")
                consistent = team_scoring_df[team_scoring_df['overall_avg'] > 15].sort_values('overall_std').head(8)
                for _, t in consistent.iterrows():
                    st.metric(
                        f"{t['team']}",
                        f"{t['overall_avg']:.1f} pts/g",
                        f"σ: {t['overall_std']:.1f}"
                    )

            with col2:
                st.markdown("**Most Volatile Scorers**")
                volatile = team_scoring_df.sort_values('overall_std', ascending=False).head(8)
                for _, t in volatile.iterrows():
                    st.metric(
                        f"{t['team']}",
                        f"{t['overall_avg']:.1f} pts/g",
                        f"σ: {t['overall_std']:.1f} ⚠️"
                    )

            # Section 4: Scoring Patterns
            st.markdown("---")
            st.markdown("### 📊 Scoring Patterns")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**High-Scoring Offenses**")
                high_scorers = team_scoring_df[team_scoring_df['overall_avg'] >= 25].sort_values('overall_avg', ascending=False)
                if not high_scorers.empty:
                    for _, t in high_scorers.iterrows():
                        st.metric(f"{t['team']}", f"{t['overall_avg']:.1f} pts/g")
                else:
                    st.info("No teams averaging 25+ pts/g")

            with col2:
                st.markdown("**Struggling Offenses**")
                low_scorers = team_scoring_df[team_scoring_df['overall_avg'] < 18].sort_values('overall_avg')
                if not low_scorers.empty:
                    for _, t in low_scorers.iterrows():
                        st.metric(f"{t['team']}", f"{t['overall_avg']:.1f} pts/g")
                else:
                    st.info("No teams averaging <18 pts/g")

            with col3:
                st.markdown("**Game Types**")
                high_scoring_total = len(games_df[games_df['total_score'] >= 50])
                defensive_battles = len(games_df[games_df['total_score'] < 35])
                close_games = len(games_df[games_df['margin'] <= 8])
                blowouts = len(games_df[games_df['margin'] >= 21])

                total_games_played = len(games_df)

                st.metric("50+ Combined", high_scoring_total)
                st.caption(f"{high_scoring_total / total_games_played * 100:.1f}% of games")
                st.metric("<35 Combined", defensive_battles)
                st.caption(f"{defensive_battles / total_games_played * 100:.1f}% of games")
                st.metric("1-Score Games (≤8)", close_games)
                st.caption(f"{close_games / total_games_played * 100:.1f}% of games")
                st.metric("Blowouts (≥21)", blowouts)
                st.caption(f"{blowouts / total_games_played * 100:.1f}% of games")

            # Section 5: Scoring Distribution
            st.markdown("---")
            st.markdown("### 📉 Scoring Distribution")

            # Create scoring buckets
            bins = [0, 10, 14, 17, 21, 24, 28, 35, 100]
            labels = ['0-9', '10-13', '14-16', '17-20', '21-23', '24-27', '28-34', '35+']
            all_scores_series = pd.Series(all_scores)
            score_dist = pd.cut(all_scores_series, bins=bins, labels=labels, include_lowest=True).value_counts().sort_index()

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("**Points Scored Distribution**")
                dist_df = pd.DataFrame({
                    'Range': score_dist.index,
                    'Games': score_dist.values,
                    'Percentage': (score_dist.values / score_dist.sum() * 100).round(1)
                })
                st.dataframe(dist_df, hide_index=True, use_container_width=True)

            with col2:
                st.markdown("**Key Insights**")
                most_common_range = score_dist.idxmax()
                most_common_pct = (score_dist.max() / score_dist.sum() * 100)
                st.metric("Most Common Range", most_common_range)
                st.caption(f"{most_common_pct:.1f}% of all scores")

                over_30 = all_scores[all_scores >= 30].count()
                over_30_pct = (over_30 / len(all_scores) * 100)
                st.metric("30+ Point Games", over_30)
                st.caption(f"{over_30_pct:.1f}% of performances")


# ============================================================================
# Section: Skill Yards Grid
# ============================================================================

def render_skill_yards_grid(season: Optional[int], week: Optional[int]):
    """Display RB/WR/TE combined rush + receiving yards matrix."""
    st.header("📈 Skill Yards Grid (RB/WR/TE)")

    if not season:
        st.warning("No season data available.")
        return

    # Query player stats
    sql = """
        SELECT player, team, week,
               COALESCE(rush_yds, 0) + COALESCE(rec_yds, 0) as total_yards
        FROM player_box_score
        WHERE season=? AND (rush_att > 0 OR targets > 0)
    """
    params = [season]
    if week:
        sql += " AND week<=?"
        params.append(week)

    df = query(sql, tuple(params))

    if df.empty:
        st.info("No player data available.")
        return

    # Pivot to create grid
    pivot = df.pivot_table(index=['player', 'team'], columns='week', values='total_yards', fill_value=0, aggfunc='sum')
    pivot = pivot.reset_index()

    # Add total column (sum of all week columns)
    week_columns = [col for col in pivot.columns if isinstance(col, (int, float)) or str(col).isdigit()]
    if week_columns:
        pivot['Total'] = pivot[week_columns].sum(axis=1)

        # Reorder columns to put Total after team
        cols = ['player', 'team', 'Total'] + week_columns
        pivot = pivot[cols]

    # Display
    st.dataframe(
        pivot.sort_values(by='Total' if 'Total' in pivot.columns else pivot.columns[-1], ascending=False).head(50),
        use_container_width=True,
        hide_index=True
    )


# ============================================================================
# Section: Skill TDs Grid
# ============================================================================

def render_skill_tds_grid(season: Optional[int], week: Optional[int]):
    """Display RB/WR/TE touchdowns matrix."""
    st.header("🎯 Skill TDs Grid (RB/WR/TE)")

    if not season:
        st.warning("No season data available.")
        return

    # Query player stats
    sql = """
        SELECT player, team, week,
               COALESCE(rush_td, 0) + COALESCE(rec_td, 0) as total_tds
        FROM player_box_score
        WHERE season=? AND (rush_td > 0 OR rec_td > 0)
    """
    params = [season]
    if week:
        sql += " AND week<=?"
        params.append(week)

    df = query(sql, tuple(params))

    if df.empty:
        st.info("No touchdown data available.")
        return

    # Pivot to create grid
    pivot = df.pivot_table(index=['player', 'team'], columns='week', values='total_tds', fill_value=0, aggfunc='sum')
    pivot = pivot.reset_index()

    # Add total column (sum of all week columns)
    week_columns = [col for col in pivot.columns if isinstance(col, (int, float)) or str(col).isdigit()]
    if week_columns:
        pivot['Total'] = pivot[week_columns].sum(axis=1)

        # Reorder columns to put Total after team
        cols = ['player', 'team', 'Total'] + week_columns
        pivot = pivot[cols]

    # Display
    st.dataframe(
        pivot.sort_values(by='Total' if 'Total' in pivot.columns else pivot.columns[-1], ascending=False).head(50),
        use_container_width=True,
        hide_index=True
    )


# ============================================================================
# Section: First TD Grid
# ============================================================================

def render_first_td_grid(season: Optional[int], week: Optional[int]):
    """Display weekly first touchdown scorers grid."""
    st.header("⭐ First TD Grid")

    if not season:
        st.warning("No season data available.")
        return

    # Query first TDs with deduplication
    sql = """
        SELECT
            week,
            team,
            player,
            touchdown_type,
            qtr,
            MAX(time) as time,
            MAX(yards_gained) as yards_gained,
            MAX(first_td_game) as first_td_game,
            MAX(first_td_for_team) as first_td_for_team
        FROM touchdown_scorers
        WHERE season=? AND (first_td_game=1 OR first_td_for_team=1)
    """
    params = [season]
    if week:
        sql += " AND week<=?"
        params.append(week)

    sql += " GROUP BY week, team, player, touchdown_type, qtr ORDER BY week, qtr, time"

    df = query(sql, tuple(params))

    if df.empty:
        st.info("No first touchdown data available.")
        return

    # Display options
    td_type = st.radio("Filter by:", ["First TD of Game", "First TD for Team", "Both"], horizontal=True)

    if td_type == "First TD of Game":
        df = df[df['first_td_game'] == 1]
    elif td_type == "First TD for Team":
        df = df[df['first_td_for_team'] == 1]

    # Group by week
    st.dataframe(
        df[['week', 'team', 'player', 'touchdown_type', 'qtr', 'time', 'yards_gained']].sort_values('week'),
        use_container_width=True,
        hide_index=True
    )


# ============================================================================
# Section: First TD Detail
# ============================================================================

def render_first_td_detail(season: Optional[int], week: Optional[int]):
    """Display detailed touchdown list."""
    st.header("📋 First TD Detail")

    if not season:
        st.warning("No season data available.")
        return

    # Query all TDs with deduplication
    # Group by key fields to eliminate duplicates from database
    sql = """
        SELECT
            week,
            team,
            player,
            touchdown_type,
            qtr,
            MAX(time) as time,
            MAX(yards_gained) as yards_gained,
            MAX(first_td_game) as first_td_game,
            MAX(first_td_for_team) as first_td_for_team
        FROM touchdown_scorers
        WHERE season=?
    """
    params = [season]
    if week:
        sql += " AND week=?"
        params.append(week)

    sql += " GROUP BY week, team, player, touchdown_type, qtr ORDER BY week, qtr, time"

    df = query(sql, tuple(params))

    if df.empty:
        st.info("No touchdown data available.")
        return

    # Display summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total TDs", len(df))
    with col2:
        st.metric("Rushing TDs", len(df[df['touchdown_type'] == 'Rushing']))
    with col3:
        st.metric("Receiving TDs", len(df[df['touchdown_type'] == 'Receiving']))

    st.divider()

    # Display table with highlighting
    display_df = df.copy()
    display_df['First Game'] = display_df['first_td_game'].apply(lambda x: '⭐' if x == 1 else '')
    display_df['First Team'] = display_df['first_td_for_team'].apply(lambda x: '🎯' if x == 1 else '')

    st.dataframe(
        display_df[['week', 'team', 'player', 'touchdown_type', 'qtr', 'time', 'yards_gained', 'First Game', 'First Team']],
        use_container_width=True,
        hide_index=True
    )


# ============================================================================
# Section: TD Against (League-Wide Defensive Stats)
# ============================================================================

def render_td_against(season: Optional[int], week: Optional[int]):
    """Display league-wide defensive touchdown statistics."""
    st.header("🛡️ TD Against - Defensive Rankings")
    st.caption("Touchdowns allowed by each team's defense - lower numbers indicate better performance")

    if not season:
        st.warning("No season data available.")
        return

    # Build query to get TDs allowed by each defense
    # This joins games with touchdown_scorers to get TDs scored against each team
    sql = """
        WITH team_games AS (
            SELECT
                team_abbr,
                COUNT(DISTINCT game_id) as games_played
            FROM (
                SELECT home_team_abbr as team_abbr, game_id FROM games WHERE season = ? {week_filter}
                UNION ALL
                SELECT away_team_abbr as team_abbr, game_id FROM games WHERE season = ? {week_filter}
            )
            GROUP BY team_abbr
        ),
        tds_allowed AS (
            SELECT
                CASE
                    WHEN ts.team = g.home_team_abbr THEN g.away_team_abbr
                    WHEN ts.team = g.away_team_abbr THEN g.home_team_abbr
                END as defense_team,
                ts.touchdown_type,
                COUNT(*) as td_count
            FROM touchdown_scorers ts
            JOIN games g ON ts.game_id = g.game_id
            WHERE ts.season = ? {week_filter}
            GROUP BY defense_team, ts.touchdown_type
        )
        SELECT
            tg.team_abbr as Team,
            tg.games_played as Games,
            COALESCE(SUM(CASE WHEN ta.touchdown_type = 'Rushing' THEN ta.td_count ELSE 0 END), 0) as rush_tds_allowed,
            COALESCE(SUM(CASE WHEN ta.touchdown_type = 'Receiving' THEN ta.td_count ELSE 0 END), 0) as rec_tds_allowed,
            COALESCE(SUM(ta.td_count), 0) as total_tds_allowed
        FROM team_games tg
        LEFT JOIN tds_allowed ta ON tg.team_abbr = ta.defense_team
        GROUP BY tg.team_abbr, tg.games_played
        ORDER BY total_tds_allowed ASC
    """

    # Add week filter if specified
    week_filter = f"AND week <= {week}" if week else ""
    sql = sql.replace("{week_filter}", week_filter)

    # Execute query
    params = [season, season, season]
    df = query(sql, tuple(params))

    if df.empty:
        st.info("No touchdown data available for this season.")
        return

    # Calculate per-game averages
    df['Rush TDs/Game'] = (df['rush_tds_allowed'] / df['Games']).round(2)
    df['Rec TDs/Game'] = (df['rec_tds_allowed'] / df['Games']).round(2)
    df['Total TDs/Game'] = (df['total_tds_allowed'] / df['Games']).round(2)

    # Add rank columns
    df['Rush Rank'] = df['rush_tds_allowed'].rank(method='min').astype(int)
    df['Rec Rank'] = df['rec_tds_allowed'].rank(method='min').astype(int)
    df['Overall Rank'] = df['total_tds_allowed'].rank(method='min').astype(int)

    # Display summary metrics
    st.subheader("📊 League Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_total = df['total_tds_allowed'].mean()
        st.metric("Avg TDs Allowed", f"{avg_total:.1f}")

    with col2:
        avg_rush = df['rush_tds_allowed'].mean()
        st.metric("Avg Rush TDs", f"{avg_rush:.1f}")

    with col3:
        avg_rec = df['rec_tds_allowed'].mean()
        st.metric("Avg Rec TDs", f"{avg_rec:.1f}")

    with col4:
        best_defense = df.iloc[0]['Team']
        st.metric("Best Defense", best_defense)

    st.divider()

    # Sort options
    st.subheader("🏈 Team Rankings")
    sort_col1, sort_col2 = st.columns([2, 1])

    with sort_col1:
        sort_by = st.selectbox(
            "Sort by:",
            ["Total TDs Allowed", "Rush TDs Allowed", "Receiving TDs Allowed", "Total TDs/Game", "Rush TDs/Game", "Rec TDs/Game"],
            key="td_against_sort"
        )

    with sort_col2:
        sort_order = st.radio("Order:", ["Ascending (Best)", "Descending (Worst)"], horizontal=True, key="td_against_order")

    # Apply sorting
    sort_column_map = {
        "Total TDs Allowed": "total_tds_allowed",
        "Rush TDs Allowed": "rush_tds_allowed",
        "Receiving TDs Allowed": "rec_tds_allowed",
        "Total TDs/Game": "Total TDs/Game",
        "Rush TDs/Game": "Rush TDs/Game",
        "Rec TDs/Game": "Rec TDs/Game"
    }

    sort_col = sort_column_map[sort_by]
    ascending = sort_order == "Ascending (Best)"

    display_df = df.sort_values(by=sort_col, ascending=ascending).copy()

    # Prepare display columns
    display_df = display_df[[
        'Overall Rank', 'Team', 'Games',
        'rush_tds_allowed', 'Rush Rank', 'Rush TDs/Game',
        'rec_tds_allowed', 'Rec Rank', 'Rec TDs/Game',
        'total_tds_allowed', 'Total TDs/Game'
    ]]

    # Rename columns for display
    display_df.columns = [
        'Rank', 'Team', 'Games',
        'Rush TDs', 'Rush Rank', 'Rush/Gm',
        'Rec TDs', 'Rec Rank', 'Rec/Gm',
        'Total TDs', 'Total/Gm'
    ]

    # Convert to integers where appropriate
    display_df['Rank'] = display_df['Rank'].astype(int)
    display_df['Games'] = display_df['Games'].astype(int)
    display_df['Rush TDs'] = display_df['Rush TDs'].astype(int)
    display_df['Rush Rank'] = display_df['Rush Rank'].astype(int)
    display_df['Rec TDs'] = display_df['Rec TDs'].astype(int)
    display_df['Rec Rank'] = display_df['Rec Rank'].astype(int)
    display_df['Total TDs'] = display_df['Total TDs'].astype(int)

    # Display the table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=600
    )

    st.divider()

    # Best/Worst performers
    st.subheader("🎯 Best & Worst Defenses")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🏆 Top 5 Best Overall Defenses**")
        best_overall = df.nsmallest(5, 'total_tds_allowed')[['Team', 'total_tds_allowed', 'Total TDs/Game']].copy()
        best_overall.columns = ['Team', 'Total TDs', 'TDs/Game']
        best_overall['Total TDs'] = best_overall['Total TDs'].astype(int)
        st.dataframe(best_overall, hide_index=True, use_container_width=True)

        st.markdown("**🛡️ Top 5 Best Rush Defenses**")
        best_rush = df.nsmallest(5, 'rush_tds_allowed')[['Team', 'rush_tds_allowed', 'Rush TDs/Game']].copy()
        best_rush.columns = ['Team', 'Rush TDs', 'Rush/Game']
        best_rush['Rush TDs'] = best_rush['Rush TDs'].astype(int)
        st.dataframe(best_rush, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**⚠️ Bottom 5 Worst Overall Defenses**")
        worst_overall = df.nlargest(5, 'total_tds_allowed')[['Team', 'total_tds_allowed', 'Total TDs/Game']].copy()
        worst_overall.columns = ['Team', 'Total TDs', 'TDs/Game']
        worst_overall['Total TDs'] = worst_overall['Total TDs'].astype(int)
        st.dataframe(worst_overall, hide_index=True, use_container_width=True)

        st.markdown("**🔥 Bottom 5 Worst Pass Defenses**")
        worst_rec = df.nlargest(5, 'rec_tds_allowed')[['Team', 'rec_tds_allowed', 'Rec TDs/Game']].copy()
        worst_rec.columns = ['Team', 'Rec TDs', 'Rec/Game']
        worst_rec['Rec TDs'] = worst_rec['Rec TDs'].astype(int)
        st.dataframe(worst_rec, hide_index=True, use_container_width=True)


# ============================================================================
# Section: Player Stats
# ============================================================================

def render_player_stats(season: Optional[int], week: Optional[int], team: Optional[str]):
    """Display individual player statistics."""
    st.header("👤 Player Stats")

    if not season:
        st.warning("No season data available.")
        return

    # Build query
    sql = "SELECT * FROM player_box_score WHERE season=?"
    params = [season]

    if week:
        sql += " AND week=?"
        params.append(week)

    if team:
        sql += " AND team=?"
        params.append(team)

    df = query(sql, tuple(params))

    if df.empty:
        st.info("No player data available.")
        return

    # Add search filter
    search = st.text_input("Search player name:", "")
    if search:
        df = df[df['player'].str.contains(search, case=False, na=False)]

    # Display stats
    st.dataframe(
        df[[
            'player', 'team', 'week',
            'pass_cmp', 'pass_att', 'pass_yds', 'pass_td', 'pass_int',
            'rush_att', 'rush_yds', 'rush_td',
            'targets', 'rec', 'rec_yds', 'rec_td'
        ]].sort_values('week'),
        use_container_width=True,
        hide_index=True
    )


# ============================================================================
# Section: Season Leaderboards
# ============================================================================

def render_season_leaderboards(season: Optional[int]):
    """Display season leaderboards using pre-built views."""
    st.header("🏆 Season Leaderboards")

    if not season:
        st.warning("No season data available.")
        return

    # Tabs for different leader categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Rushing Leaders",
        "Receiving Leaders",
        "Passing Leaders",
        "TD Leaders",
        "First TD Leaders"
    ])

    with tab1:
        st.subheader("Rushing Leaders")
        df = query("SELECT * FROM rushing_leaders WHERE season=? ORDER BY total_rush_yds DESC LIMIT 25", (season,))
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No rushing data available.")

    with tab2:
        st.subheader("Receiving Leaders")
        df = query("SELECT * FROM receiving_leaders WHERE season=? ORDER BY total_rec_yds DESC LIMIT 25", (season,))
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No receiving data available.")

    with tab3:
        st.subheader("Passing Leaders")
        df = query("SELECT * FROM passing_leaders WHERE season=? ORDER BY total_pass_yds DESC LIMIT 25", (season,))
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No passing data available.")

    with tab4:
        st.subheader("Touchdown Leaders")
        df = query("SELECT * FROM touchdown_leaders WHERE season=? ORDER BY total_touchdowns DESC LIMIT 25", (season,))
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No touchdown data available.")

    with tab5:
        st.subheader("First TD of Game Leaders")
        df = query("SELECT * FROM first_td_game_leaders WHERE season=? ORDER BY first_td_count DESC LIMIT 25", (season,))
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No first TD data available.")


# ============================================================================
# Section: Play-by-Play Viewer
# ============================================================================

def render_play_by_play(season: Optional[int], week: Optional[int], team: Optional[str]):
    """Display play-by-play for selected game."""
    st.header("▶️ Play-by-Play Viewer")

    if not season:
        st.warning("No season data available.")
        return

    # Game selector
    games_sql = "SELECT game_id, week, home_team_abbr, away_team_abbr FROM games WHERE season=?"
    games_params = [season]
    if week:
        games_sql += " AND week=?"
        games_params.append(week)

    games_df = query(games_sql, tuple(games_params))

    if games_df.empty:
        st.info("No games available.")
        return

    # Create game options
    games_df['display'] = 'Week ' + games_df['week'].astype(str) + ': ' + games_df['away_team_abbr'] + ' @ ' + games_df['home_team_abbr']
    game_options = dict(zip(games_df['display'], games_df['game_id']))

    selected_game_display = st.selectbox("Select Game:", list(game_options.keys()))
    selected_game = game_options[selected_game_display]

    # Query plays
    sql = "SELECT * FROM plays WHERE game_id=? ORDER BY play_index"
    df = query(sql, (selected_game,))

    if df.empty:
        st.info("No plays available for this game.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_qtr = st.multiselect("Quarter:", df['qtr'].dropna().unique().tolist())
    with col2:
        filter_down = st.multiselect("Down:", df['down'].dropna().unique().tolist())
    with col3:
        show_td_only = st.checkbox("Touchdowns only")

    # Apply filters
    filtered = df.copy()
    if filter_qtr:
        filtered = filtered[filtered['qtr'].isin(filter_qtr)]
    if filter_down:
        filtered = filtered[filtered['down'].isin(filter_down)]
    if show_td_only:
        filtered = filtered[filtered['is_touchdown'] == 1]
    if team:
        filtered = filtered[(filtered['posteam_abbr'] == team) | (filtered['defteam_abbr'] == team)]

    # Display
    st.dataframe(
        filtered[['play_index', 'qtr', 'time', 'down', 'togo', 'yardline_text', 'posteam_abbr', 'detail', 'yards_gained']],
        use_container_width=True,
        hide_index=True
    )


# ============================================================================
# Section: Game Detail
# ============================================================================

def render_game_detail(season: Optional[int], week: Optional[int]):
    """Deep dive into a single game."""
    st.header("🔍 Game Detail")

    if not season:
        st.warning("No season data available.")
        return

    # Game selector
    games_sql = "SELECT game_id, week, home_team_abbr, away_team_abbr, home_score, away_score FROM games WHERE season=?"
    games_params = [season]
    if week:
        games_sql += " AND week=?"
        games_params.append(week)

    games_df = query(games_sql, tuple(games_params))

    if games_df.empty:
        st.info("No games available.")
        return

    games_df['display'] = 'Week ' + games_df['week'].astype(str) + ': ' + games_df['away_team_abbr'] + ' @ ' + games_df['home_team_abbr'] + ' (' + games_df['away_score'].astype(str) + '-' + games_df['home_score'].astype(str) + ')'
    game_options = dict(zip(games_df['display'], games_df['game_id']))

    selected_game_display = st.selectbox("Select Game:", list(game_options.keys()))
    selected_game = game_options[selected_game_display]

    # Game summary
    game_info = games_df[games_df['game_id'] == selected_game].iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Away Team", game_info['away_team_abbr'])
    with col2:
        st.metric("Away Score", int(game_info['away_score']))
    with col3:
        st.metric("Home Team", game_info['home_team_abbr'])
    with col4:
        st.metric("Home Score", int(game_info['home_score']))

    st.divider()

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Box Score", "Touchdowns", "Play Count"])

    with tab1:
        box_df = query("SELECT * FROM box_score_summary WHERE game_id=?", (selected_game,))
        if not box_df.empty:
            st.dataframe(box_df, use_container_width=True, hide_index=True)

    with tab2:
        td_df = query("""
            SELECT team, player, touchdown_type, qtr,
                   MAX(time) as time,
                   MAX(yards_gained) as yards_gained
            FROM touchdown_scorers
            WHERE game_id=?
            GROUP BY team, player, touchdown_type, qtr
            ORDER BY qtr, time
        """, (selected_game,))
        if not td_df.empty:
            st.dataframe(td_df, use_container_width=True, hide_index=True)
        else:
            st.info("No touchdowns in this game.")

    with tab3:
        plays_df = query("SELECT COUNT(*) as total_plays, SUM(CASE WHEN is_pass=1 THEN 1 ELSE 0 END) as pass_plays, SUM(CASE WHEN is_rush=1 THEN 1 ELSE 0 END) as rush_plays FROM plays WHERE game_id=?", (selected_game,))
        if not plays_df.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Plays", int(plays_df['total_plays'].iloc[0]))
            with col2:
                st.metric("Pass Plays", int(plays_df['pass_plays'].iloc[0]))
            with col3:
                st.metric("Rush Plays", int(plays_df['rush_plays'].iloc[0]))


# ============================================================================
# Section: Advanced Team Analytics
# ============================================================================

def render_advanced_team_analytics(season: Optional[int], week: Optional[int], team: Optional[str]):
    """Display advanced NFL metrics and analytics."""
    st.header("🔬 Advanced Team Analytics")

    if not season:
        st.warning("No season data available.")
        return

    # Team selector
    teams = get_teams(season)
    if not teams:
        st.warning("No teams available.")
        return

    selected_team = team if team else st.selectbox("Select Team", teams, key="adv_team")

    st.subheader(f"{selected_team} Advanced Metrics")

    # Calculate all metrics
    epa_metrics = calculate_team_epa(selected_team, season, week)
    success_metrics = calculate_success_rates(selected_team, season, week)
    explosive_metrics = calculate_explosive_plays(selected_team, season, week)
    third_down_metrics = calculate_third_down_metrics(selected_team, season, week)

    # EPA Section
    st.markdown("### 📊 EPA (Expected Points Added)")

    if epa_metrics:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Offensive EPA/Play", f"{epa_metrics.get('off_epa_per_play', 0):.3f}")
            st.metric("Total Offensive EPA", f"{epa_metrics.get('off_epa_total', 0):.1f}")

        with col2:
            st.metric("Defensive EPA/Play Allowed", f"{epa_metrics.get('def_epa_per_play', 0):.3f}")
            st.metric("Total Defensive EPA", f"{epa_metrics.get('def_epa_total', 0):.1f}")

        with col3:
            net_epa = epa_metrics.get('off_epa_per_play', 0) - epa_metrics.get('def_epa_per_play', 0)
            st.metric("Net EPA/Play", f"{net_epa:.3f}")
            st.caption("Higher = Better overall team efficiency")
    else:
        st.info("No EPA data available for this team/season")

    st.divider()

    # Success Rate Section
    st.markdown("### ✅ Success Rate Metrics")
    if success_metrics:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Overall Success Rate", f"{success_metrics['overall']*100:.1f}%")

        with col2:
            st.metric("Early Down Success", f"{success_metrics['early_down']*100:.1f}%")

        with col3:
            st.metric("Passing Down Success", f"{success_metrics['passing_down']*100:.1f}%")

        with col4:
            st.metric("Red Zone Success", f"{success_metrics['red_zone']*100:.1f}%")

        # Success by down chart
        if success_metrics.get('by_down'):
            st.markdown("**Success Rate by Down:**")
            down_data = success_metrics['by_down']
            down_df = pd.DataFrame({
                'Down': [f"{int(d)}" for d in down_data.keys()],
                'Success Rate': [v * 100 for v in down_data.values()]
            })

            fig = go.Figure(data=[
                go.Bar(x=down_df['Down'], y=down_df['Success Rate'], marker_color='#1f77b4')
            ])
            fig.update_layout(
                yaxis_title="Success Rate (%)",
                xaxis_title="Down",
                height=300,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Explosive Plays Section
    st.markdown("### 💥 Explosive Plays (20+ Yards)")
    if explosive_metrics:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Explosive Play Rate", f"{explosive_metrics['explosive_rate']*100:.1f}%")

        with col2:
            st.metric("Explosive Pass Rate", f"{explosive_metrics['explosive_pass_rate']*100:.1f}%")

        with col3:
            st.metric("Explosive Rush Rate", f"{explosive_metrics['explosive_rush_rate']*100:.1f}%")

        with col4:
            st.metric("Avg Explosive Yards", f"{explosive_metrics['avg_explosive_yards']:.1f}")

        st.caption(f"Total Explosive Plays: {explosive_metrics['total_explosive']}")

    st.divider()

    # Third Down Section
    st.markdown("### 🎯 Third Down Efficiency")
    if third_down_metrics:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("3rd Down Conversion", f"{third_down_metrics['conversion_rate']*100:.1f}%")
            st.caption(f"{third_down_metrics['total_conversions']} / {third_down_metrics['total_attempts']}")

        with col2:
            st.metric("3rd & Short (≤3)", f"{third_down_metrics['short_conversion']*100:.1f}%")

        with col3:
            st.metric("3rd & Medium (4-7)", f"{third_down_metrics['medium_conversion']*100:.1f}%")

        with col4:
            st.metric("3rd & Long (8+)", f"{third_down_metrics['long_conversion']*100:.1f}%")


# ============================================================================
# Section: Matchup Predictor
# ============================================================================

def render_matchup_predictor(season: Optional[int], week: Optional[int]):
    """Predict game outcomes using advanced metrics."""
    st.header("🎯 Matchup Predictor")

    if not season:
        st.warning("No season data available.")
        return

    teams = get_teams(season)
    if len(teams) < 2:
        st.warning("Need at least 2 teams for prediction.")
        return

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", teams, index=0, key="pred_team1")
    with col2:
        team2 = st.selectbox("Team 2", teams, index=min(1, len(teams)-1), key="pred_team2")

    if st.button("🔮 Generate Prediction", type="primary"):
        # Calculate metrics for both teams
        t1_epa = calculate_team_epa(team1, season, week)
        t2_epa = calculate_team_epa(team2, season, week)

        t1_success = calculate_success_rates(team1, season, week)
        t2_success = calculate_success_rates(team1, season, week)

        t1_explosive = calculate_explosive_plays(team1, season, week)
        t2_explosive = calculate_explosive_plays(team2, season, week)

        # Calculate power ratings
        t1_power = (
            t1_epa['off_epa_per_play'] * 40 +
            (-t1_epa['def_epa_per_play']) * 40 +
            t1_success.get('overall', 0) * 10 +
            t1_explosive.get('explosive_rate', 0) * 10
        )

        t2_power = (
            t2_epa['off_epa_per_play'] * 40 +
            (-t2_epa['def_epa_per_play']) * 40 +
            t2_success.get('overall', 0) * 10 +
            t2_explosive.get('explosive_rate', 0) * 10
        )

        # Calculate win probability (logistic function)
        rating_diff = t1_power - t2_power
        win_prob_t1 = 1 / (1 + 10 ** (-rating_diff / 2))

        # Predict scores
        league_avg = 22.5
        t1_score = league_avg + (rating_diff * 15)
        t2_score = league_avg - (rating_diff * 15)

        # Display prediction
        st.divider()
        st.subheader("🔮 Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"### {team1}")
            st.metric("Win Probability", f"{win_prob_t1*100:.1f}%")
            st.metric("Predicted Score", f"{max(0, t1_score):.0f}")
            st.metric("Power Rating", f"{t1_power:.2f}")

        with col2:
            st.markdown("### Prediction")
            winner = team1 if win_prob_t1 > 0.5 else team2
            confidence = "High" if abs(win_prob_t1 - 0.5) > 0.2 else ("Medium" if abs(win_prob_t1 - 0.5) > 0.1 else "Low")
            st.metric("Predicted Winner", winner)
            st.metric("Confidence", confidence)

        with col3:
            st.markdown(f"### {team2}")
            st.metric("Win Probability", f"{(1-win_prob_t1)*100:.1f}%")
            st.metric("Predicted Score", f"{max(0, t2_score):.0f}")
            st.metric("Power Rating", f"{t2_power:.2f}")

        st.divider()

        # Key factors
        st.markdown("### 📊 Key Factors")

        factors = []
        if abs(t1_epa['off_epa_per_play'] - t2_epa['off_epa_per_play']) > 0.1:
            leader = team1 if t1_epa['off_epa_per_play'] > t2_epa['off_epa_per_play'] else team2
            factors.append(f"✅ **{leader}** has superior offensive efficiency (EPA)")

        if abs(t1_epa['def_epa_per_play'] - t2_epa['def_epa_per_play']) > 0.1:
            leader = team1 if t1_epa['def_epa_per_play'] < t2_epa['def_epa_per_play'] else team2
            factors.append(f"🛡️ **{leader}** has a tougher defense (lower EPA allowed)")

        if abs(t1_explosive.get('explosive_rate', 0) - t2_explosive.get('explosive_rate', 0)) > 0.03:
            leader = team1 if t1_explosive.get('explosive_rate', 0) > t2_explosive.get('explosive_rate', 0) else team2
            factors.append(f"💥 **{leader}** generates more explosive plays")

        if factors:
            for factor in factors:
                st.markdown(factor)
        else:
            st.info("This matchup appears evenly matched across key metrics.")


# ============================================================================
# Section: Notes Manager
# ============================================================================

def render_notes_manager(season: Optional[int], week: Optional[int]):
    """Display notes manager with filtering, search, edit, and delete capabilities."""
    st.header("📝 Notes Manager")

    # Filters section
    st.subheader("Filters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Team filter
        all_teams = get_teams()
        team_filter = st.selectbox(
            "Filter by Team",
            ["All"] + all_teams,
            key="notes_team_filter"
        )
        team_filter = None if team_filter == "All" else team_filter

    with col2:
        # Season filter
        seasons = get_seasons()
        season_filter = st.selectbox(
            "Filter by Season",
            ["All"] + seasons,
            index=0,
            key="notes_season_filter"
        )
        season_filter = None if season_filter == "All" else season_filter

    with col3:
        # Week filter
        if season_filter:
            weeks = get_weeks(season_filter)
            week_filter = st.selectbox(
                "Filter by Week",
                ["All"] + weeks,
                key="notes_week_filter"
            )
            week_filter = None if week_filter == "All" else week_filter
        else:
            week_filter = None
            st.selectbox("Filter by Week", ["All"], disabled=True, key="notes_week_filter_disabled")

    with col4:
        # Custom tag filter
        tag_filter = st.text_input(
            "Filter by Tag",
            placeholder="e.g., Injury, Important",
            key="notes_tag_filter"
        )
        tag_filter = tag_filter.strip() if tag_filter else None

    # Search bar
    search_text = st.text_input(
        "Search notes",
        placeholder="Search note content...",
        key="notes_search"
    )
    search_text = search_text.strip() if search_text else None

    st.divider()

    # Get filtered notes
    notes_df = get_notes(
        team_filter=team_filter,
        tag_filter=tag_filter,
        season_filter=season_filter,
        week_filter=week_filter,
        search_text=search_text
    )

    # Display results
    if notes_df.empty:
        st.info("No notes found. Add a note using the Quick Notes section in the sidebar!")
    else:
        st.subheader(f"Notes ({len(notes_df)})")

        # Display each note
        for idx, row in notes_df.iterrows():
            with st.expander(
                f"📅 {row['created_at'][:16]} - Season {row['season']}, Week {row['week']}" if row['season'] and row['week']
                else f"📅 {row['created_at'][:16]}"
            ):
                # Display note content
                st.markdown(f"**Note:**")
                st.write(row['note_text'])

                # Display tags if any
                if row['tags']:
                    tags_list = row['tags'].split(',')
                    tags_formatted = ' '.join([f"`#{tag}`" for tag in tags_list])
                    st.markdown(f"**Tags:** {tags_formatted}")

                # Edit and Delete buttons
                col_edit, col_delete, col_spacer = st.columns([1, 1, 4])

                with col_edit:
                    if st.button("✏️ Edit", key=f"edit_{row['note_id']}"):
                        st.session_state[f"editing_{row['note_id']}"] = True
                        st.rerun()

                with col_delete:
                    if st.button("🗑️ Delete", key=f"delete_{row['note_id']}"):
                        if delete_note(row['note_id']):
                            st.success("Note deleted!")
                            st.rerun()

                # Edit mode
                if st.session_state.get(f"editing_{row['note_id']}", False):
                    st.divider()
                    st.markdown("**Edit Note:**")
                    edited_text = st.text_area(
                        "Note content",
                        value=row['note_text'],
                        height=100,
                        key=f"edit_text_{row['note_id']}"
                    )

                    col_save, col_cancel = st.columns([1, 1])
                    with col_save:
                        if st.button("💾 Save", key=f"save_{row['note_id']}"):
                            if edited_text.strip():
                                if update_note(row['note_id'], edited_text):
                                    st.success("Note updated!")
                                    del st.session_state[f"editing_{row['note_id']}"]
                                    st.rerun()
                            else:
                                st.warning("Note cannot be empty")

                    with col_cancel:
                        if st.button("❌ Cancel", key=f"cancel_{row['note_id']}"):
                            del st.session_state[f"editing_{row['note_id']}"]
                            st.rerun()


# ============================================================================
# Section: Transaction Manager
# ============================================================================

def render_transaction_manager(season: Optional[int], week: Optional[int]):
    """Display transaction manager with roster tracking and transaction recording."""
    st.header("🔄 Transaction Manager")

    # Tab layout for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Active Rosters", "Add Transaction", "Transaction History", "Injuries", "Upcoming Games"])

    with tab1:
        st.subheader("Current Team Rosters")

        # Filters
        col1, col2 = st.columns([1, 2])
        with col1:
            teams = get_teams(season) if season else []
            team_filter = st.selectbox(
                "Filter by Team",
                ["All Teams"] + teams,
                key="roster_team_filter"
            )

        with col2:
            # Auto-detect unreported transactions
            if season and st.button("🔍 Auto-Detect Transactions", key="detect_trans"):
                with st.spinner("Analyzing game data..."):
                    detected = detect_unreported_transactions(season)
                    if not detected.empty:
                        st.session_state['detected_transactions'] = detected
                        st.success(f"Found {len(detected)} potential transactions!")
                    else:
                        st.info("No unreported transactions detected.")

        st.divider()

        # Show detected transactions if any
        if 'detected_transactions' in st.session_state:
            detected = st.session_state['detected_transactions']
            if not detected.empty:
                st.warning(f"⚠️ {len(detected)} Unreported Transactions Detected")
                for idx, trans in detected.iterrows():
                    st.markdown(f"**{trans['player_name']}**: {trans['from_team']} → {trans['to_team']} (Week {trans['effective_week']})")
                    if st.button(f"Add Transaction", key=f"add_detected_{idx}"):
                        add_transaction(
                            trans['player_name'],
                            'TRADE',
                            trans['season'],
                            trans['effective_week'],
                            from_team=trans['from_team'],
                            to_team=trans['to_team'],
                            notes="Auto-detected from game data"
                        )
                        st.success(f"Added transaction for {trans['player_name']}")
                        # Remove from detected list
                        detected = detected.drop(idx)
                        if detected.empty:
                            del st.session_state['detected_transactions']
                        else:
                            st.session_state['detected_transactions'] = detected
                        st.rerun()
                st.divider()

        # Display rosters
        if season:
            if team_filter == "All Teams":
                display_teams = teams[:5]  # Show first 5 teams to avoid clutter
                st.caption(f"Showing first 5 teams. Select a specific team to view full roster.")
            else:
                display_teams = [team_filter]

            for team in display_teams:
                with st.expander(f"📋 {team} Roster", expanded=(team_filter != "All Teams")):
                    roster_df = get_team_roster(team, season, as_of_week=week)

                    if not roster_df.empty:
                        # Get additional stats for each player
                        player_stats = []
                        for _, player_row in roster_df.iterrows():
                            player_name = player_row['player']

                            # Get games played
                            stats_query = f"""
                                SELECT COUNT(DISTINCT week) as games
                                FROM player_box_score
                                WHERE player = '{player_name}' AND team = '{team}' AND season = {season}
                            """
                            if week:
                                stats_query += f" AND week <= {week}"

                            games_df = query(stats_query)
                            games = games_df['games'].iloc[0] if not games_df.empty else 0

                            # Check if acquired mid-season
                            transactions = get_player_transactions(player_name=player_name, season=season)
                            acquired_week = None
                            for _, trans in transactions.iterrows():
                                if trans['to_team'] == team and trans['effective_week'] > 1:
                                    acquired_week = trans['effective_week']
                                    break

                            # Check injury status
                            is_injured = is_player_on_injury_list(player_name, team, season, week if week else 18)
                            injury_status = "🏥 INJURED" if is_injured else "✅ Healthy"

                            player_stats.append({
                                'Player': player_name,
                                'Games': games,
                                'Acquired': f"Wk {acquired_week}" if acquired_week else "—",
                                'Injury Status': injury_status,
                                'Status': "Active"
                            })

                        stats_df = pd.DataFrame(player_stats)
                        st.dataframe(stats_df, hide_index=True, use_container_width=True)
                        st.caption(f"Total players: {len(stats_df)}")
                    else:
                        st.info("No players found for this team.")
        else:
            st.warning("Please select a season to view rosters.")

    with tab2:
        st.subheader("➕ Add New Transaction")

        # Get all unique players from the season
        if season:
            players_query = f"SELECT DISTINCT player FROM player_box_score WHERE season = {season} ORDER BY player"
            players_df = query(players_query)
            all_players = players_df['player'].tolist() if not players_df.empty else []
        else:
            all_players = []

        with st.form("add_transaction_form"):
            col1, col2 = st.columns(2)

            with col1:
                player_name = st.selectbox(
                    "Player Name *",
                    [""] + all_players,
                    help="Select player from dropdown"
                )

                transaction_type = st.selectbox(
                    "Transaction Type *",
                    ["TRADE", "SIGNING", "RELEASE", "WAIVER_CLAIM"],
                    help="Type of transaction"
                )

                from_team = st.selectbox(
                    "From Team",
                    ["None (Free Agent)"] + (get_teams(season) if season else []),
                    help="Leave as 'None' for signings"
                )

            with col2:
                to_team = st.selectbox(
                    "To Team",
                    ["None (Released)"] + (get_teams(season) if season else []),
                    help="Leave as 'None' for releases"
                )

                trans_season = st.number_input(
                    "Season *",
                    min_value=2020,
                    max_value=2030,
                    value=season if season else 2024
                )

                effective_week = st.number_input(
                    "Effective Week *",
                    min_value=1,
                    max_value=18,
                    value=week if week else 1,
                    help="Week when transaction takes effect"
                )

            notes_text = st.text_area(
                "Notes (Optional)",
                placeholder="E.g., 'Traded for draft pick', 'Signed as free agent'",
                height=100
            )

            submitted = st.form_submit_button("💾 Save Transaction", type="primary")

            if submitted:
                if not player_name:
                    st.error("Please select a player")
                elif from_team == "None (Free Agent)" and transaction_type not in ["SIGNING", "WAIVER_CLAIM"]:
                    st.error("Free agent can only be used with SIGNING or WAIVER_CLAIM")
                elif to_team == "None (Released)" and transaction_type != "RELEASE":
                    st.error("Released status can only be used with RELEASE transaction")
                else:
                    # Process team values
                    from_team_val = None if from_team == "None (Free Agent)" else from_team
                    to_team_val = None if to_team == "None (Released)" else to_team

                    # Add transaction
                    trans_id = add_transaction(
                        player_name,
                        transaction_type,
                        trans_season,
                        effective_week,
                        from_team=from_team_val,
                        to_team=to_team_val,
                        notes=notes_text if notes_text else None
                    )

                    if trans_id:
                        st.success(f"✅ Transaction added for {player_name}")
                        st.balloons()
                        st.rerun()

    with tab3:
        st.subheader("📜 Transaction History")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            hist_season = st.selectbox(
                "Filter by Season",
                ["All"] + (get_seasons() if get_seasons() else []),
                key="hist_season_filter"
            )
            hist_season_val = None if hist_season == "All" else hist_season

        with col2:
            hist_team = st.selectbox(
                "Filter by Team",
                ["All"] + (get_teams() if get_teams() else []),
                key="hist_team_filter"
            )
            hist_team_val = None if hist_team == "All" else hist_team

        with col3:
            hist_type = st.selectbox(
                "Filter by Type",
                ["All", "TRADE", "SIGNING", "RELEASE", "WAIVER_CLAIM"],
                key="hist_type_filter"
            )
            hist_type_val = None if hist_type == "All" else hist_type

        st.divider()

        # Get transactions
        transactions_df = get_player_transactions(
            season=hist_season_val,
            team=hist_team_val,
            transaction_type=hist_type_val
        )

        if not transactions_df.empty:
            st.caption(f"Total transactions: {len(transactions_df)}")

            # Display transactions
            for idx, trans in transactions_df.iterrows():
                from_display = trans['from_team'] if trans['from_team'] else "Free Agent"
                to_display = trans['to_team'] if trans['to_team'] else "Released"

                # Transaction summary
                if trans['transaction_type'] == 'TRADE':
                    summary = f"**{trans['player_name']}** - TRADE: {from_display} → {to_display}"
                elif trans['transaction_type'] == 'SIGNING':
                    summary = f"**{trans['player_name']}** - SIGNED by {to_display}"
                elif trans['transaction_type'] == 'RELEASE':
                    summary = f"**{trans['player_name']}** - RELEASED by {from_display}"
                else:
                    summary = f"**{trans['player_name']}** - {trans['transaction_type']}: {from_display} → {to_display}"

                with st.expander(f"Week {trans['effective_week']}, {trans['season']}: {summary}"):
                    col_info, col_actions = st.columns([3, 1])

                    with col_info:
                        st.markdown(f"**Type:** {trans['transaction_type']}")
                        st.markdown(f"**Date:** {trans['transaction_date']}")
                        if trans['notes']:
                            st.markdown(f"**Notes:** {trans['notes']}")

                    with col_actions:
                        if st.button("🗑️ Delete", key=f"delete_trans_{trans['transaction_id']}"):
                            if delete_transaction(trans['transaction_id']):
                                st.success("Transaction deleted!")
                                st.rerun()
        else:
            st.info("No transactions found with the selected filters.")

    with tab4:
        st.subheader("🏥 Injury Management")

        # Sub-tabs for Add and View injuries
        injury_tab1, injury_tab2 = st.tabs(["Add/Update Injury", "Active Injuries"])

        with injury_tab1:
            st.markdown("### Add or Update Player Injury")

            # Initialize form counter for resetting
            if 'injury_form_counter' not in st.session_state:
                st.session_state.injury_form_counter = 0

            # Get all unique players from the season
            if season:
                players_query = f"SELECT DISTINCT player FROM player_box_score WHERE season = {season} ORDER BY player"
                players_df = query(players_query)
                all_players = players_df['player'].tolist() if not players_df.empty else []
            else:
                all_players = []

            with st.form(f"add_injury_form_{st.session_state.injury_form_counter}"):
                col1, col2 = st.columns(2)

                with col1:
                    inj_player = st.selectbox(
                        "Player Name *",
                        [""] + all_players,
                        help="Select player to mark as injured"
                    )

                    # Get player's current team
                    if inj_player and season:
                        current_team = get_player_current_team(inj_player, season, as_of_week=week)
                        if current_team:
                            st.info(f"Current Team: {current_team}")
                            inj_team = current_team
                        else:
                            teams = get_teams(season) if season else []
                            inj_team = st.selectbox("Team *", teams)
                    else:
                        teams = get_teams(season) if season else []
                        inj_team = st.selectbox("Team *", teams) if teams else None

                    inj_type = st.selectbox(
                        "Injury Type *",
                        ["OUT", "IR", "DOUBTFUL", "QUESTIONABLE", "PUP"],
                        help="OUT: Game-time decision, IR: Season-ending"
                    )

                with col2:
                    inj_season = st.number_input(
                        "Season *",
                        min_value=2020,
                        max_value=2030,
                        value=season if season else 2024
                    )

                    inj_start_week = st.number_input(
                        "Start Week",
                        min_value=1,
                        max_value=18,
                        value=week if week else 1,
                        help="Week when injury began (optional)"
                    )

                    inj_end_week = st.number_input(
                        "End Week",
                        min_value=1,
                        max_value=18,
                        value=18,
                        help="Expected return week (18 = season-ending)"
                    )

                inj_description = st.text_area(
                    "Injury Description (Optional)",
                    placeholder="E.g., 'Torn ACL', 'Concussion protocol', 'Ankle sprain'",
                    height=100
                )

                submitted = st.form_submit_button("💾 Save Injury", type="primary")

                if submitted:
                    if not inj_player:
                        st.error("Please select a player")
                    elif not inj_team:
                        st.error("Please select a team")
                    else:
                        # Save with detailed error handling
                        injury_id, error_msg = add_persistent_injury(
                            inj_player,
                            inj_team,
                            inj_season,
                            inj_type,
                            start_week=inj_start_week,
                            end_week=inj_end_week,
                            description=inj_description if inj_description else None
                        )

                        if injury_id:
                            # Store success message in session state to persist across rerun
                            st.session_state.injury_success = f"✅ Successfully saved injury for {inj_player} (ID: {injury_id})"
                            st.session_state.injury_last_saved = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            # Increment form counter to reset form
                            st.session_state.injury_form_counter += 1
                            st.balloons()
                            st.rerun()
                        else:
                            # Show detailed error
                            st.error(f"❌ Failed to save injury for {inj_player}")
                            if error_msg:
                                st.error(f"Error details: {error_msg}")
                            st.warning("Please check the log file (nfl_app.log) for more details")

            # Show success message if it exists
            if 'injury_success' in st.session_state:
                st.success(st.session_state.injury_success)
                if 'injury_last_saved' in st.session_state:
                    st.caption(f"Saved at: {st.session_state.injury_last_saved}")
                # Clear the message after displaying
                del st.session_state.injury_success
                if 'injury_last_saved' in st.session_state:
                    del st.session_state.injury_last_saved

        with injury_tab2:
            st.markdown("### Active Injuries")

            # Filters
            col1, col2, col3 = st.columns(3)

            with col1:
                inj_season_filter = st.selectbox(
                    "Filter by Season",
                    ["All"] + (get_seasons() if get_seasons() else []),
                    key="inj_season_filter"
                )
                inj_season_val = None if inj_season_filter == "All" else inj_season_filter

            with col2:
                inj_team_filter = st.selectbox(
                    "Filter by Team",
                    ["All"] + (get_teams() if get_teams() else []),
                    key="inj_team_filter"
                )
                inj_team_val = None if inj_team_filter == "All" else inj_team_filter

            with col3:
                inj_type_filter = st.selectbox(
                    "Filter by Type",
                    ["All", "OUT", "IR", "DOUBTFUL", "QUESTIONABLE", "PUP"],
                    key="inj_type_filter"
                )
                inj_type_val = None if inj_type_filter == "All" else inj_type_filter

            st.divider()

            # Get injuries
            injuries_df = get_persistent_injuries(
                season=inj_season_val,
                team=inj_team_val,
                injury_type=inj_type_val
            )

            if not injuries_df.empty:
                st.caption(f"Total active injuries: {len(injuries_df)}")

                # Display injuries
                for idx, inj in injuries_df.iterrows():
                    # Check if player is currently on injury list for current week
                    current_week = week if week else 18
                    is_active = is_player_on_injury_list(
                        inj['player_name'],
                        inj['team_abbr'],
                        inj['season'],
                        current_week
                    )

                    status_badge = "🔴 ACTIVE" if is_active else "⚪ INACTIVE"

                    # Build title
                    weeks_display = ""
                    if inj['start_week'] and inj['end_week']:
                        weeks_display = f"Weeks {inj['start_week']}-{inj['end_week']}"
                    elif inj['start_week']:
                        weeks_display = f"Since Week {inj['start_week']}"
                    elif inj['end_week']:
                        weeks_display = f"Until Week {inj['end_week']}"

                    title = f"{status_badge} **{inj['player_name']}** ({inj['team_abbr']}) - {inj['injury_type']}"
                    if weeks_display:
                        title += f" | {weeks_display}"

                    with st.expander(title):
                        col_info, col_actions = st.columns([3, 1])

                        with col_info:
                            st.markdown(f"**Player:** {inj['player_name']}")
                            st.markdown(f"**Team:** {inj['team_abbr']}")
                            st.markdown(f"**Season:** {inj['season']}")
                            st.markdown(f"**Injury Type:** {inj['injury_type']}")
                            if inj['injury_description']:
                                st.markdown(f"**Description:** {inj['injury_description']}")
                            st.markdown(f"**Added:** {inj['created_at']}")
                            st.markdown(f"**Last Updated:** {inj['updated_at']}")

                        with col_actions:
                            if st.button("🗑️ Remove", key=f"delete_inj_{inj['injury_id']}"):
                                if remove_persistent_injury(inj['injury_id']):
                                    st.success("Injury removed!")
                                    st.rerun()
            else:
                st.info("No injuries found with the selected filters.")

    with tab5:
        st.subheader("📅 Upcoming Games Schedule")
        st.caption("Upload upcoming games JSON to power the Game Preview feature")

        # Create sub-tabs for Upload and View
        upload_tab, view_tab, reference_tab = st.tabs(["📤 Upload Schedule", "📋 Current Schedule", "📖 CSV Format"])

        with upload_tab:
            st.markdown("### Upload CSV Schedule")

            if not season:
                st.warning("⚠️ Please select a season from the sidebar first.")
            else:
                uploaded_file = st.file_uploader(
                    "Choose a CSV file",
                    type=['csv'],
                    help="Upload a CSV file containing upcoming games (Week, Away, Home, Day, Primetime, Location)"
                )

                if uploaded_file is not None:
                    try:
                        csv_data = pd.read_csv(uploaded_file)

                        # Validate required columns
                        required_cols = ['Week', 'Away', 'Home']
                        missing_cols = [col for col in required_cols if col not in csv_data.columns]

                        if missing_cols:
                            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                        else:
                            st.success(f"✅ Loaded {len(csv_data)} games from file")

                            # Preview first few games
                            st.markdown("**Preview (first 5 games):**")
                            st.dataframe(csv_data.head(), use_container_width=True)

                            # Upload button
                            if st.button("📥 Upload to Database", type="primary"):
                                with st.spinner("Uploading schedule..."):
                                    success_count, errors = upload_upcoming_schedule_csv(csv_data, season)

                                    if errors:
                                        st.error(f"⚠️ Encountered {len(errors)} errors:")
                                        for err in errors[:5]:  # Show first 5 errors
                                            st.error(err)

                                    if success_count > 0:
                                        st.success(f"✅ Successfully uploaded {success_count} games!")
                                        st.rerun()
                                    else:
                                        st.error("❌ No games were uploaded successfully")

                    except Exception as e:
                        st.error(f"❌ Error reading file: {str(e)}")

        with view_tab:
            st.markdown("### Scheduled Upcoming Games")

            # Get upcoming games
            upcoming_df = get_upcoming_games(season=season)

            if not upcoming_df.empty:
                # Show count by week
                st.metric("Total Games Scheduled", len(upcoming_df))

                # Group by week
                weeks = sorted(upcoming_df['week'].unique())

                for wk in weeks:
                    week_games = upcoming_df[upcoming_df['week'] == wk]

                    with st.expander(f"📅 Week {wk} ({len(week_games)} games)", expanded=(wk == weeks[0])):
                        for _, game in week_games.iterrows():
                            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

                            with col1:
                                matchup = f"**{game['away_team']} @ {game['home_team']}**"
                                prime_badge = " 🌟" if game['primetime'] else ""
                                st.markdown(f"{matchup}{prime_badge}")
                                # Show day and location
                                location_text = f"📍 {game['location']}" if game.get('location') else ""
                                day_info = game['day_of_week'] if game.get('day_of_week') else ""
                                caption = f"{day_info}  {location_text}".strip()
                                if caption:
                                    st.caption(caption)

                            with col2:
                                if game.get('location') and game['location'] != game['home_team']:
                                    st.caption("🌍 Neutral")

                            with col3:
                                if game['primetime']:
                                    st.caption("Primetime")

                            with col4:
                                if st.button("🗑️", key=f"del_{game['game_id']}", help="Delete game"):
                                    if delete_upcoming_game(game['game_id']):
                                        st.success("Game deleted!")
                                        st.rerun()

                st.divider()

                # Clear all button
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("🗑️ Clear All Games", type="secondary"):
                        if clear_upcoming_games(season):
                            st.success("All games cleared!")
                            st.rerun()

            else:
                st.info("📭 No upcoming games scheduled for this season. Upload a CSV file to get started!")

        with reference_tab:
            st.markdown("### CSV Format Reference")

            st.markdown("""
Upload a CSV file with the following columns:

**Required Columns:**
- **Week** (integer): NFL week number (1-18)
- **Away** (string): Away team abbreviation
- **Home** (string): Home team abbreviation

**Optional Columns:**
- **Day** (string): Day name (e.g., "Sunday", "Monday", "Thursday")
- **Primetime** (string): "Yes" for primetime games, empty otherwise
- **Location** (string): Game location - use team abbreviation or special location (e.g., "London", "Germany", "Brazil")

**Example CSV:**
""")

            example_csv = pd.DataFrame([
                {"Week": 1, "Away": "DAL", "Home": "PHI", "Day": "Thursday", "Primetime": "Yes", "Location": "PHI"},
                {"Week": 1, "Away": "KAN", "Home": "LAC", "Day": "Friday", "Primetime": "Yes", "Location": "Brazil"},
                {"Week": 1, "Away": "PIT", "Home": "NYJ", "Day": "Sunday", "Primetime": "", "Location": "NYJ"},
                {"Week": 5, "Away": "MIN", "Home": "CLE", "Day": "Sunday", "Primetime": "", "Location": "London"},
                {"Week": 8, "Away": "PHI", "Home": "DAL", "Day": "Sunday", "Primetime": "Yes", "Location": "DAL"}
            ])

            st.dataframe(example_csv, use_container_width=True, hide_index=True)

            st.markdown("""
**Notes:**
- **Location**: Typically matches the Home team, but use special locations for international games (London, Germany, Brazil, Mexico, etc.)
- **Neutral Site Games**: For neutral site games, set Location to the actual game location (different from Home team)
- **Primetime**: Use "Yes" for primetime games (SNF, MNF, TNF), leave empty for regular games
- Team abbreviations can be any consistent format (2-3 letters)
""")


# ============================================================================
# Section: Upcoming Matches
# ============================================================================

def render_upcoming_matches(season: Optional[int], week: Optional[int]):
    """Display upcoming games schedule with week filter."""
    st.header("📅 Upcoming Matches")

    # Get all available weeks from upcoming_games table
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get all unique weeks and seasons
        cursor.execute("""
            SELECT DISTINCT season, week
            FROM upcoming_games
            ORDER BY season DESC, week ASC
        """)
        available_data = cursor.fetchall()

        if not available_data:
            st.info("No upcoming games schedule uploaded yet. Upload schedule via Transaction Manager → Upcoming Games.")
            conn.close()
            return

        # Get unique seasons and weeks
        available_seasons = sorted(list(set([row[0] for row in available_data])), reverse=True)

        # Season selector
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_season = st.selectbox(
                "Season",
                available_seasons,
                index=0 if season in available_seasons else 0,
                key="upcoming_season_select"
            )

        # Get weeks for selected season
        weeks_for_season = sorted([row[1] for row in available_data if row[0] == selected_season])

        with col2:
            selected_week = st.selectbox(
                "Week",
                ["All Weeks"] + weeks_for_season,
                index=0,
                key="upcoming_week_select"
            )

        # Build query
        if selected_week == "All Weeks":
            cursor.execute("""
                SELECT date, week, home_team, away_team, day_of_week, primetime, location
                FROM upcoming_games
                WHERE season = ?
                ORDER BY week ASC, date ASC
            """, (selected_season,))
        else:
            cursor.execute("""
                SELECT date, week, home_team, away_team, day_of_week, primetime, location
                FROM upcoming_games
                WHERE season = ? AND week = ?
                ORDER BY date ASC
            """, (selected_season, selected_week))

        games = cursor.fetchall()
        conn.close()

        if not games:
            st.warning(f"No games found for the selected filter.")
            return

        # Convert to DataFrame
        df = pd.DataFrame(games, columns=['Date', 'Week', 'Home Team', 'Away Team', 'Day', 'Primetime', 'Location'])

        # Format primetime column
        df['Primetime'] = df['Primetime'].apply(lambda x: '⭐' if x == 1 else '')

        # Create matchup column with location info
        def format_matchup(row):
            matchup = f"{row['Away Team']} @ {row['Home Team']}"
            if row['Location'] and row['Location'] != row['Home Team']:
                matchup += f" ({row['Location']})"
            return matchup

        df['Matchup'] = df.apply(format_matchup, axis=1)

        # Display summary metrics
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Games", len(df))
        with col2:
            st.metric("Weeks", df['Week'].nunique())
        with col3:
            primetime_count = (df['Primetime'] == '⭐').sum()
            st.metric("Primetime Games", primetime_count)
        with col4:
            teams = set(df['Home Team'].tolist() + df['Away Team'].tolist())
            st.metric("Teams", len(teams))

        st.divider()

        # Group by week if showing all weeks
        if selected_week == "All Weeks":
            for week_num in sorted(df['Week'].unique()):
                week_games = df[df['Week'] == week_num].copy()

                with st.expander(f"Week {week_num} ({len(week_games)} games)", expanded=(week_num == weeks_for_season[0])):
                    # Display games for this week
                    display_df = week_games[['Date', 'Day', 'Matchup', 'Primetime']].copy()

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Date": st.column_config.DateColumn("Date", format="MMM DD, YYYY"),
                            "Day": st.column_config.TextColumn("Day"),
                            "Matchup": st.column_config.TextColumn("Matchup", width="medium"),
                            "Primetime": st.column_config.TextColumn("Prime", width="small")
                        }
                    )
        else:
            # Display single week
            st.subheader(f"Week {selected_week} Schedule")

            display_df = df[['Date', 'Day', 'Matchup', 'Primetime']].copy()

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date": st.column_config.DateColumn("Date", format="MMM DD, YYYY"),
                    "Day": st.column_config.TextColumn("Day"),
                    "Matchup": st.column_config.TextColumn("Matchup", width="medium"),
                    "Primetime": st.column_config.TextColumn("Prime", width="small")
                }
            )

            # Add player projections for this week
            st.divider()
            st.subheader("📊 Player Projections (Defensive Matchup Adjusted)")

            # Get teams playing this week
            teams_playing = list(set(df['Home Team'].tolist() + df['Away Team'].tolist()))

            # Generate projections
            with st.spinner("Calculating matchup-adjusted projections..."):
                projections = generate_player_projections(selected_season, selected_week, teams_playing)

            if projections and any(not proj_df.empty for proj_df in projections.values()):
                # Create tabs for each position
                proj_tabs = st.tabs(["🎯 Top QBs", "🏃 Top RBs", "🙌 Top WRs", "💪 Top TEs", "⭐ Top Skill Players"])

                # QB Tab
                with proj_tabs[0]:
                    if not projections.get('QB', pd.DataFrame()).empty:
                        st.markdown("##### Quarterbacks - Matchup-Adjusted Passing Yard Projections")

                        qb_df = projections['QB'].head(20).copy()

                        # Add matchup rating column
                        qb_df['Matchup'] = qb_df['Multiplier'].apply(lambda x: get_matchup_rating(x)[0])

                        # Style the dataframe with colors
                        def style_matchup(row):
                            _, color = get_matchup_rating(row['Multiplier'])
                            return [color] * len(row) if color else [''] * len(row)

                        styled_df = qb_df.style.apply(style_matchup, axis=1)

                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No QB data available for this week")

                # RB Tab
                with proj_tabs[1]:
                    if not projections.get('RB', pd.DataFrame()).empty:
                        st.markdown("##### Running Backs - Matchup-Adjusted Total Yard Projections")

                        rb_df = projections['RB'].head(20).copy()
                        rb_df['Matchup'] = rb_df['Multiplier'].apply(lambda x: get_matchup_rating(x)[0])

                        def style_matchup(row):
                            _, color = get_matchup_rating(row['Multiplier'])
                            return [color] * len(row) if color else [''] * len(row)

                        styled_df = rb_df.style.apply(style_matchup, axis=1)

                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No RB data available for this week")

                # WR Tab
                with proj_tabs[2]:
                    if not projections.get('WR', pd.DataFrame()).empty:
                        st.markdown("##### Wide Receivers - Matchup-Adjusted Receiving Yard Projections")

                        wr_df = projections['WR'].head(20).copy()
                        wr_df['Matchup'] = wr_df['Multiplier'].apply(lambda x: get_matchup_rating(x)[0])

                        def style_matchup(row):
                            _, color = get_matchup_rating(row['Multiplier'])
                            return [color] * len(row) if color else [''] * len(row)

                        styled_df = wr_df.style.apply(style_matchup, axis=1)

                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No WR data available for this week")

                # TE Tab
                with proj_tabs[3]:
                    if not projections.get('TE', pd.DataFrame()).empty:
                        st.markdown("##### Tight Ends - Matchup-Adjusted Receiving Yard Projections")

                        te_df = projections['TE'].head(20).copy()
                        te_df['Matchup'] = te_df['Multiplier'].apply(lambda x: get_matchup_rating(x)[0])

                        def style_matchup(row):
                            _, color = get_matchup_rating(row['Multiplier'])
                            return [color] * len(row) if color else [''] * len(row)

                        styled_df = te_df.style.apply(style_matchup, axis=1)

                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No TE data available for this week")

                # Skill Players Combined Tab
                with proj_tabs[4]:
                    if not projections.get('SKILL', pd.DataFrame()).empty:
                        st.markdown("##### Top Skill Position Players (RB/WR/TE) - All Positions Combined")

                        skill_df = projections['SKILL'].head(20).copy()
                        skill_df['Matchup'] = skill_df['Multiplier'].apply(lambda x: get_matchup_rating(x)[0])

                        def style_matchup(row):
                            _, color = get_matchup_rating(row['Multiplier'])
                            return [color] * len(row) if color else [''] * len(row)

                        styled_df = skill_df.style.apply(style_matchup, axis=1)

                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No skill position data available for this week")

                # Add legend
                st.divider()
                st.markdown("""
**Matchup Rating Legend:**
- 🔥 **Great** (1.15+ multiplier): Highly favorable matchup - defense allows significantly more yards than average
- ✅ **Good** (1.05-1.14): Favorable matchup - defense allows more yards than average
- ⚪ **Average** (0.95-1.04): Neutral matchup - defense performs near league average
- ⚠️ **Tough** (0.85-0.94): Difficult matchup - defense allows fewer yards than average
- 🛑 **Brutal** (<0.85): Very difficult matchup - elite defense, significantly limits yards

*Projections are calculated by multiplying player's median yards by the defensive matchup factor. The multiplier shows how the opponent's defense compares to league average.*
                """)

            else:
                st.info("Not enough data to generate projections for this week. Make sure player stats are available for previous weeks.")

    except Exception as e:
        st.error(f"Error loading upcoming games: {e}")


# ============================================================================
# Charts Section
# ============================================================================

def render_charts_view(season: Optional[int], week: Optional[int]):
    """Render comprehensive charts and visualizations."""
    st.header("📊 Analytics Charts")
    st.markdown("Visual analytics to identify efficiency, trends, and opportunities")

    # Chart selection
    chart_type = st.selectbox(
        "Select Chart",
        [
            "Team Offense Efficiency (Yards vs Points)",
            "Team Balance (Points Scored vs Allowed)",
            "Defense Yards Allowed (Pass vs Rush)",
            "RB Efficiency (Rush Yards vs TDs)",
            "RB Yards per Carry",
            "Skill Player Total Yards vs Touches",
            "WR Efficiency (Targets vs Yards per Route Run)",
            "QB Passing TDs vs Interceptions",
            "QB Passing Yards vs Attempts"
        ]
    )

    st.divider()

    if chart_type == "Team Offense Efficiency (Yards vs Points)":
        render_team_offense_efficiency_chart(season, week)
    elif chart_type == "Team Balance (Points Scored vs Allowed)":
        render_team_balance_chart(season, week)
    elif chart_type == "Defense Yards Allowed (Pass vs Rush)":
        render_defense_yards_allowed_chart(season, week)
    elif chart_type == "RB Efficiency (Rush Yards vs TDs)":
        render_rb_efficiency_chart(season, week)
    elif chart_type == "RB Yards per Carry":
        render_rb_yards_per_carry_chart(season, week)
    elif chart_type == "Skill Player Total Yards vs Touches":
        render_skill_player_yards_touches_chart(season, week)
    elif chart_type == "WR Efficiency (Targets vs Yards per Route Run)":
        render_wr_efficiency_chart(season, week)
    elif chart_type == "QB Passing TDs vs Interceptions":
        render_qb_td_int_chart(season, week)
    elif chart_type == "QB Passing Yards vs Attempts":
        render_qb_yards_attempts_chart(season, week)


def render_team_offense_efficiency_chart(season: Optional[int], week: Optional[int]):
    """Chart 1: Yards per Game vs Points per Game - Scoring Efficiency"""
    st.subheader("⚡ Team Offense Efficiency")
    st.markdown("""
    **Goal:** Identify teams that turn yardage into actual scoring efficiency.
    - **High yards, high points:** Dominant offense
    - **High yards, low points:** Inefficient in red zone
    - **Low yards, high points:** Explosive/efficient plays
    """)

    try:
        # Build query with week filter
        week_filter = f"AND week <= {week}" if week else ""

        sql_query = f"""
        SELECT
            team_abbr,
            COUNT(DISTINCT game_id) as games,
            AVG(yards_total) as avg_yards,
            AVG(points) as avg_points,
            SUM(points) as total_points,
            SUM(yards_total) as total_yards
        FROM team_game_summary
        WHERE season = ?
        {week_filter}
        GROUP BY team_abbr
        HAVING games > 0
        ORDER BY avg_points DESC
        """

        df = query(sql_query, (season,))

        if df.empty:
            st.info("No data available for selected filters")
            return

        # Create scatter plot
        fig = go.Figure()

        # Add invisible markers for hover functionality
        fig.add_trace(go.Scatter(
            x=df['avg_yards'],
            y=df['avg_points'],
            mode='markers',
            marker=dict(
                size=1,
                color=df['avg_points'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="PPG"),
                opacity=0
            ),
            text=df['team_abbr'],
            customdata=df['team_abbr'],
            hovertemplate='<b>%{customdata}</b><br>' +
                         'Yards/Game: %{x:.1f}<br>' +
                         'Points/Game: %{y:.1f}<br>' +
                         '<extra></extra>',
            showlegend=False
        ))

        # Add league average lines
        avg_yards = df['avg_yards'].mean()
        avg_points = df['avg_points'].mean()

        fig.add_hline(y=avg_points, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="League Avg PPG", annotation_position="right")
        fig.add_vline(x=avg_yards, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="League Avg YPG", annotation_position="top")

        # Build layout with images
        layout_images = []
        for idx, row in df.iterrows():
            layout_images.append(dict(
                source=get_team_logo_url(row['team_abbr']),
                xref="x",
                yref="y",
                x=row['avg_yards'],
                y=row['avg_points'],
                sizex=15,  # 15 yards wide
                sizey=1.5, # 1.5 points tall
                xanchor="center",
                yanchor="middle",
                layer="above",
                opacity=0.9
            ))

        fig.update_layout(
            title=f"Yards per Game vs Points per Game ({season} Season)",
            xaxis_title="Total Yards per Game",
            yaxis_title="Points per Game",
            height=600,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            images=layout_images
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        with st.expander("📋 View Data Table"):
            display_df = df[['team_abbr', 'games', 'avg_yards', 'avg_points']].copy()
            display_df.columns = ['Team', 'Games', 'Yards/Game', 'Points/Game']
            display_df = display_df.round(1)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error generating chart: {e}")


def render_team_balance_chart(season: Optional[int], week: Optional[int]):
    """Chart 2: Points Scored vs Points Allowed - Team Balance"""
    st.subheader("⚖️ Team Balance Chart")
    st.markdown("""
    **Goal:** Classic balance chart showing team dominance.
    - **Top-right quadrant:** Elite teams (high scoring, strong defense)
    - **Top-left:** Good defense, struggling offense
    - **Bottom-right:** High-powered offense, weak defense
    - **Bottom-left:** Struggling on both sides
    """)

    try:
        week_filter = f"AND week <= {week}" if week else ""

        # Get offensive stats (points scored)
        sql_offense = f"""
        SELECT
            team_abbr,
            AVG(points) as avg_points_scored
        FROM team_game_summary
        WHERE season = ?
        {week_filter}
        GROUP BY team_abbr
        """

        # Get defensive stats (points allowed) - opponent's points
        sql_defense = f"""
        SELECT
            t1.team_abbr,
            AVG(t2.points) as avg_points_allowed
        FROM team_game_summary t1
        JOIN team_game_summary t2 ON t1.game_id = t2.game_id AND t1.team_abbr != t2.team_abbr
        WHERE t1.season = ?
        {week_filter}
        GROUP BY t1.team_abbr
        """

        df_offense = query(sql_offense, (season,))
        df_defense = query(sql_defense, (season,))

        # Merge datasets
        df = df_offense.merge(df_defense, on='team_abbr')

        if df.empty:
            st.info("No data available for selected filters")
            return

        # Create scatter plot
        fig = go.Figure()

        # Calculate dominance score (higher scored, lower allowed = better)
        df['dominance'] = df['avg_points_scored'] - df['avg_points_allowed']

        # Add invisible markers for hover functionality
        fig.add_trace(go.Scatter(
            x=df['avg_points_scored'],
            y=df['avg_points_allowed'],
            mode='markers',
            marker=dict(
                size=1,
                color=df['dominance'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Point Diff"),
                opacity=0
            ),
            text=df['team_abbr'],
            customdata=df['dominance'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Points Scored/Game: %{x:.1f}<br>' +
                         'Points Allowed/Game: %{y:.1f}<br>' +
                         'Point Differential: %{customdata:.1f}<br>' +
                         '<extra></extra>',
            showlegend=False
        ))

        # Add league average lines
        avg_scored = df['avg_points_scored'].mean()
        avg_allowed = df['avg_points_allowed'].mean()

        fig.add_hline(y=avg_allowed, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="League Avg Allowed", annotation_position="right")
        fig.add_vline(x=avg_scored, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="League Avg Scored", annotation_position="top")

        # Add quadrant labels
        fig.add_annotation(x=df['avg_points_scored'].max() * 0.95, y=df['avg_points_allowed'].min() * 1.05,
                          text="Elite", showarrow=False, font=dict(size=14, color="green"))
        fig.add_annotation(x=df['avg_points_scored'].min() * 1.05, y=df['avg_points_allowed'].min() * 1.05,
                          text="Strong Defense", showarrow=False, font=dict(size=14, color="blue"))
        fig.add_annotation(x=df['avg_points_scored'].max() * 0.95, y=df['avg_points_allowed'].max() * 0.95,
                          text="High Scoring", showarrow=False, font=dict(size=14, color="orange"))
        fig.add_annotation(x=df['avg_points_scored'].min() * 1.05, y=df['avg_points_allowed'].max() * 0.95,
                          text="Struggling", showarrow=False, font=dict(size=14, color="red"))

        # Build layout with images
        layout_images = []
        for idx, row in df.iterrows():
            layout_images.append(dict(
                source=get_team_logo_url(row['team_abbr']),
                xref="x",
                yref="y",
                x=row['avg_points_scored'],
                y=row['avg_points_allowed'],
                sizex=1.5,  # 1.5 points wide
                sizey=1.5,  # 1.5 points tall
                xanchor="center",
                yanchor="middle",
                layer="above",
                opacity=0.9
            ))

        fig.update_layout(
            title=f"Points Scored vs Points Allowed ({season} Season)",
            xaxis_title="Points Scored per Game",
            yaxis_title="Points Allowed per Game (Lower is Better)",
            height=600,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            images=layout_images
        )

        # Reverse y-axis so lower (better defense) is at top
        fig.update_yaxes(autorange="reversed")

        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        with st.expander("📋 View Data Table"):
            display_df = df[['team_abbr', 'avg_points_scored', 'avg_points_allowed', 'dominance']].copy()
            display_df.columns = ['Team', 'PPG Scored', 'PPG Allowed', 'Point Diff']
            display_df = display_df.round(1).sort_values('Point Diff', ascending=False)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error generating chart: {e}")


def render_defense_yards_allowed_chart(season: Optional[int], week: Optional[int]):
    """Chart: Defense Yards Allowed (Passing vs Rushing)."""
    st.subheader("🛡️ Defense Yards Allowed")
    st.markdown("""
    **Goal:** Identify defensive strengths and weaknesses.
    - **Bottom-left quadrant:** Elite defense (low pass & rush yards allowed)
    - **Bottom-right:** Weak against run, strong against pass
    - **Top-left:** Weak against pass, strong against run
    - **Top-right:** Struggling defense overall
    """)

    try:
        week_filter = f"AND g.week <= {week}" if week else ""

        # Get defensive stats by aggregating opponent's player stats
        # Pass yards allowed = sum of opponent QB pass yards against this team
        sql_defense = f"""
        SELECT
            pb.team as defense_team,
            COUNT(DISTINCT pb.game_id) as games,
            SUM(CASE WHEN opp.pass_yds > 0 THEN opp.pass_yds ELSE 0 END) as total_pass_yds_allowed,
            SUM(CASE WHEN opp.rush_yds > 0 THEN opp.rush_yds ELSE 0 END) as total_rush_yds_allowed
        FROM player_box_score pb
        JOIN games g ON pb.game_id = g.game_id
        JOIN player_box_score opp ON pb.game_id = opp.game_id AND pb.team != opp.team
        WHERE g.season = ?
        {week_filter}
        GROUP BY pb.team
        """

        df = query(sql_defense, (season,))

        if df.empty:
            st.info("No defensive data available for selected filters")
            return

        # Calculate per-game averages
        df['avg_pass_yds_allowed'] = (df['total_pass_yds_allowed'] / df['games']).round(1)
        df['avg_rush_yds_allowed'] = (df['total_rush_yds_allowed'] / df['games']).round(1)
        df.rename(columns={'defense_team': 'team_abbr'}, inplace=True)

        if df.empty:
            st.info("No defensive data available for selected filters")
            return

        # Calculate total yards allowed
        df['total_yds_allowed'] = df['avg_pass_yds_allowed'] + df['avg_rush_yds_allowed']

        # Create scatter plot
        fig = go.Figure()

        # Add invisible markers for hover functionality
        fig.add_trace(go.Scatter(
            x=df['avg_rush_yds_allowed'],
            y=df['avg_pass_yds_allowed'],
            mode='markers',
            marker=dict(
                size=1,
                color=df['total_yds_allowed'],
                colorscale='RdYlGn_r',  # Reversed: red=bad (more yards), green=good (fewer yards)
                showscale=True,
                colorbar=dict(title="Total Yds"),
                opacity=0
            ),
            text=df['team_abbr'],
            customdata=df['total_yds_allowed'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Pass Yards Allowed/Game: %{y:.1f}<br>' +
                         'Rush Yards Allowed/Game: %{x:.1f}<br>' +
                         'Total Yards Allowed: %{customdata:.1f}<br>' +
                         '<extra></extra>',
            showlegend=False
        ))

        # Add league average lines
        avg_pass = df['avg_pass_yds_allowed'].mean()
        avg_rush = df['avg_rush_yds_allowed'].mean()

        fig.add_hline(y=avg_pass, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="League Avg Pass", annotation_position="right")
        fig.add_vline(x=avg_rush, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="League Avg Rush", annotation_position="top")

        # Add quadrant labels
        fig.add_annotation(x=df['avg_rush_yds_allowed'].min() * 1.02, y=df['avg_pass_yds_allowed'].min() * 1.02,
                          text="Elite Defense", showarrow=False, font=dict(size=14, color="green"))
        fig.add_annotation(x=df['avg_rush_yds_allowed'].max() * 0.98, y=df['avg_pass_yds_allowed'].min() * 1.02,
                          text="Run Defense Issue", showarrow=False, font=dict(size=14, color="orange"))
        fig.add_annotation(x=df['avg_rush_yds_allowed'].min() * 1.02, y=df['avg_pass_yds_allowed'].max() * 0.98,
                          text="Pass Defense Issue", showarrow=False, font=dict(size=14, color="orange"))
        fig.add_annotation(x=df['avg_rush_yds_allowed'].max() * 0.98, y=df['avg_pass_yds_allowed'].max() * 0.98,
                          text="Struggling", showarrow=False, font=dict(size=14, color="red"))

        # Calculate dynamic sizing for team logos
        x_range = df['avg_rush_yds_allowed'].max() - df['avg_rush_yds_allowed'].min()
        y_range = df['avg_pass_yds_allowed'].max() - df['avg_pass_yds_allowed'].min()

        logo_size_x = max(x_range * 0.06, 5)  # At least 5 yards wide
        logo_size_y = max(y_range * 0.06, 5)  # At least 5 yards tall

        # Build layout with team logo images
        layout_images = []
        for idx, row in df.iterrows():
            layout_images.append(dict(
                source=get_team_logo_url(row['team_abbr']),
                xref="x",
                yref="y",
                x=row['avg_rush_yds_allowed'],
                y=row['avg_pass_yds_allowed'],
                sizex=logo_size_x,
                sizey=logo_size_y,
                xanchor="center",
                yanchor="middle",
                layer="above",
                opacity=0.9
            ))

        fig.update_layout(
            title=f"Defense: Pass Yards vs Rush Yards Allowed ({season} Season)",
            xaxis_title="Rush Yards Allowed per Game",
            yaxis_title="Pass Yards Allowed per Game",
            height=600,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            images=layout_images
        )

        # Reverse both axes so lower (better defense) is in bottom-left
        fig.update_xaxes(autorange="reversed")
        fig.update_yaxes(autorange="reversed")

        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        with st.expander("📋 View Defensive Stats Table"):
            display_df = df[['team_abbr', 'avg_pass_yds_allowed', 'avg_rush_yds_allowed', 'total_yds_allowed']].copy()
            display_df.columns = ['Team', 'Pass Yds/G', 'Rush Yds/G', 'Total Yds/G']
            display_df = display_df.round(1).sort_values('Total Yds/G', ascending=True)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error generating chart: {e}")


def render_rb_efficiency_chart(season: Optional[int], week: Optional[int]):
    """Chart 3: Rushing Yards vs Rushing TDs - RB Efficiency"""
    st.subheader("🏃 Running Back Efficiency")
    st.markdown("""
    **Goal:** Identify volume vs scoring efficiency among running backs.
    - **Minimum 20 touches** to filter for relevant players
    - **High yards, high TDs:** Workhorse RBs
    - **High TDs, lower yards:** Goal-line specialists
    """)

    # Minimum touches filter
    min_touches = st.slider("Minimum Rush Attempts", 10, 50, 20, 5)

    try:
        week_filter = f"AND g.week <= {week}" if week else ""

        sql_rb = f"""
        SELECT
            pb.player,
            pb.team,
            SUM(pb.rush_att) as total_attempts,
            SUM(pb.rush_yds) as total_yards,
            SUM(pb.rush_td) as total_tds,
            COUNT(DISTINCT pb.game_id) as games,
            AVG(pb.rush_yds) as avg_yards_per_game,
            CAST(SUM(pb.rush_yds) AS FLOAT) / NULLIF(SUM(pb.rush_att), 0) as yards_per_carry
        FROM player_box_score pb
        JOIN games g ON pb.game_id = g.game_id
        WHERE g.season = ?
        {week_filter}
        AND pb.rush_att > 0
        GROUP BY pb.player, pb.team
        HAVING total_attempts >= ?
        ORDER BY total_yards DESC
        """

        df = query(sql_rb, (season, min_touches))

        if df.empty:
            st.info(f"No running backs with at least {min_touches} attempts found")
            return

        # Create scatter plot
        fig = go.Figure()

        # Add invisible markers for hover functionality
        fig.add_trace(go.Scatter(
            x=df['total_yards'],
            y=df['total_tds'],
            mode='markers',
            marker=dict(
                size=1,
                color=df['yards_per_carry'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="YPC"),
                opacity=0
            ),
            text=df['player'],
            customdata=df[['player', 'team', 'total_attempts', 'yards_per_carry']],
            hovertemplate='<b>%{customdata[0]}</b> (%{customdata[1]})<br>' +
                         'Rush Yards: %{x}<br>' +
                         'Rush TDs: %{y}<br>' +
                         'Attempts: %{customdata[2]}<br>' +
                         'Yards/Carry: %{customdata[3]:.2f}<br>' +
                         '<extra></extra>',
            showlegend=False
        ))

        # Calculate dynamic sizing based on data ranges
        x_range = df['total_yards'].max() - df['total_yards'].min()
        y_range = df['total_tds'].max() - df['total_tds'].min()

        # Use fixed size that works well for player headshots
        headshot_size_x = max(x_range * 0.04, 20)  # At least 20 yards wide
        headshot_size_y = max(y_range * 0.15, 0.5)  # At least 0.5 TDs tall

        # Build layout with player headshot images
        layout_images = []
        for idx, row in df.iterrows():
            layout_images.append(dict(
                source=get_player_headshot_url(row['player'], row['team']),
                xref="x",
                yref="y",
                x=row['total_yards'],
                y=row['total_tds'],
                sizex=headshot_size_x,
                sizey=headshot_size_y,
                xanchor="center",
                yanchor="middle",
                layer="above",
                opacity=0.9
            ))

        fig.update_layout(
            title=f"Rushing Yards vs Touchdowns ({season} Season)",
            xaxis_title="Total Rushing Yards",
            yaxis_title="Total Rushing Touchdowns",
            height=600,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            images=layout_images
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        with st.expander("📋 View Data Table"):
            display_df = df[['player', 'team', 'games', 'total_attempts', 'total_yards', 'total_tds', 'yards_per_carry']].copy()
            display_df.columns = ['Player', 'Team', 'Games', 'Attempts', 'Yards', 'TDs', 'YPC']
            display_df['YPC'] = display_df['YPC'].round(2)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error generating chart: {e}")


def render_wr_efficiency_chart(season: Optional[int], week: Optional[int]):
    """Chart 4: Targets vs Yards per Route Run - WR Efficiency"""
    st.subheader("🎯 Wide Receiver Efficiency")
    st.markdown("""
    **Goal:** Identify true efficiency vs volume receivers.
    - **Note:** Yards per Route Run requires route data (approximated by receptions)
    - **High targets, high efficiency:** Elite WR1s
    - **High targets, lower efficiency:** Volume receivers
    - **Low targets, high efficiency:** Deep threats/efficient role players
    """)

    st.warning("⚠️ YPRR calculation is approximated as Yards/Reception since route data is not available in this dataset")

    # Minimum targets filter
    min_targets = st.slider("Minimum Targets", 5, 30, 10, 5)

    try:
        week_filter = f"AND g.week <= {week}" if week else ""

        sql_wr = f"""
        SELECT
            pb.player,
            pb.team,
            SUM(pb.targets) as total_targets,
            SUM(pb.rec) as total_receptions,
            SUM(pb.rec_yds) as total_yards,
            SUM(pb.rec_td) as total_tds,
            COUNT(DISTINCT pb.game_id) as games,
            CAST(SUM(pb.rec_yds) AS FLOAT) / NULLIF(SUM(pb.rec), 0) as yards_per_reception,
            CAST(SUM(pb.rec) AS FLOAT) / NULLIF(SUM(pb.targets), 0) as catch_rate
        FROM player_box_score pb
        JOIN games g ON pb.game_id = g.game_id
        WHERE g.season = ?
        {week_filter}
        AND pb.targets > 0
        GROUP BY pb.player, pb.team
        HAVING total_targets >= ?
        ORDER BY total_yards DESC
        """

        df = query(sql_wr, (season, min_targets))

        if df.empty:
            st.info(f"No receivers with at least {min_targets} targets found")
            return

        # Create scatter plot
        fig = go.Figure()

        # Add invisible markers for hover functionality
        fig.add_trace(go.Scatter(
            x=df['total_targets'],
            y=df['yards_per_reception'],
            mode='markers',
            marker=dict(
                size=1,
                color=df['catch_rate'] * 100,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Catch %"),
                opacity=0,
                cmin=40,
                cmax=100
            ),
            text=df['player'],
            customdata=df[['player', 'team', 'total_yards', 'total_receptions', 'total_tds', 'catch_rate']],
            hovertemplate='<b>%{customdata[0]}</b> (%{customdata[1]})<br>' +
                         'Targets: %{x}<br>' +
                         'Yards/Reception: %{y:.2f}<br>' +
                         'Total Yards: %{customdata[2]}<br>' +
                         'Receptions: %{customdata[3]}<br>' +
                         'Catch Rate: %{customdata[5]:.1%}<br>' +
                         'TDs: %{customdata[4]}<br>' +
                         '<extra></extra>',
            showlegend=False
        ))

        # Add league average lines
        avg_targets = df['total_targets'].mean()
        avg_ypr = df['yards_per_reception'].mean()

        fig.add_hline(y=avg_ypr, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="Avg Y/R", annotation_position="right")
        fig.add_vline(x=avg_targets, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="Avg Targets", annotation_position="top")

        # Calculate dynamic sizing based on data ranges
        x_range = df['total_targets'].max() - df['total_targets'].min()
        y_range = df['yards_per_reception'].max() - df['yards_per_reception'].min()

        # Use fixed size that works well for player headshots
        headshot_size_x = max(x_range * 0.04, 3)  # At least 3 targets wide
        headshot_size_y = max(y_range * 0.10, 0.5)  # At least 0.5 Y/R tall

        # Build layout with player headshot images
        layout_images = []
        for idx, row in df.iterrows():
            layout_images.append(dict(
                source=get_player_headshot_url(row['player'], row['team']),
                xref="x",
                yref="y",
                x=row['total_targets'],
                y=row['yards_per_reception'],
                sizex=headshot_size_x,
                sizey=headshot_size_y,
                xanchor="center",
                yanchor="middle",
                layer="above",
                opacity=0.9
            ))

        fig.update_layout(
            title=f"Targets vs Yards per Reception ({season} Season)",
            xaxis_title="Total Targets",
            yaxis_title="Yards per Reception (Proxy for YPRR)",
            height=600,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            images=layout_images
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        with st.expander("📋 View Data Table"):
            display_df = df[['player', 'team', 'games', 'total_targets', 'total_receptions',
                            'total_yards', 'total_tds', 'yards_per_reception', 'catch_rate']].copy()
            display_df.columns = ['Player', 'Team', 'Games', 'Targets', 'Rec', 'Yards', 'TDs', 'Y/R', 'Catch %']
            display_df['Y/R'] = display_df['Y/R'].round(2)
            display_df['Catch %'] = (display_df['Catch %'] * 100).round(1)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error generating chart: {e}")


# ============================================================================
# Section: QB Passing Charts
# ============================================================================

def render_qb_td_int_chart(season: Optional[int], week: Optional[int]):
    """Chart: QB Passing Touchdowns vs Interceptions (axes swapped)."""
    st.subheader("🏈 QB Passing Touchdowns vs. Interceptions")
    st.markdown("*Shows the TD-to-INT ratio for QBs. Top-right quadrant represents QBs with high TDs and low INTs.*")

    if not season:
        st.warning("No season data available.")
        return

    # Minimum attempts filter
    min_attempts = st.slider("Minimum Pass Attempts", 10, 300, 50, 10, key="qb_td_int_min_att")

    # Build query to get QB passing stats
    week_filter = f"AND g.week <= {week}" if week else ""

    sql_qb = f"""
    SELECT
        pb.player,
        pb.team,
        SUM(pb.pass_yds) as total_pass_yds,
        SUM(pb.pass_td) as total_pass_tds,
        SUM(pb.pass_int) as total_interceptions,
        SUM(pb.pass_att) as total_pass_att,
        COUNT(DISTINCT pb.game_id) as games
    FROM player_box_score pb
    JOIN games g ON pb.game_id = g.game_id
    WHERE g.season = ?
    {week_filter}
    AND pb.pass_att > 0
    GROUP BY pb.player, pb.team
    HAVING total_pass_att >= ?
    ORDER BY total_pass_yds DESC
    """

    try:
        df = query(sql_qb, (season, min_attempts))

        if df.empty:
            st.info(f"No QB passing data available with at least {min_attempts} attempts.")
            return

        # Filter out QBs with 0 TDs and 0 INTs
        chart_df = df[(df['total_pass_tds'] > 0) | (df['total_interceptions'] > 0)].copy()

        if not chart_df.empty:
            import plotly.express as px

            # Swapped: X = TDs, Y = INTs (was X = INTs, Y = TDs)
            fig = px.scatter(
                chart_df,
                x='total_pass_tds',
                y='total_interceptions',
                hover_data=['player', 'team', 'total_pass_yds', 'games'],
                text='player',
                labels={
                    'total_pass_tds': 'Passing Touchdowns',
                    'total_interceptions': 'Interceptions'
                },
                title=f"QB Passing TDs vs Interceptions ({season})"
            )
            fig.update_traces(textposition='top center', marker=dict(size=12))
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Show data table
            with st.expander("📋 View QB Stats Table"):
                display_df = df[['player', 'team', 'games', 'total_pass_att', 'total_pass_yds',
                                'total_pass_tds', 'total_interceptions']].copy()
                display_df.columns = ['Player', 'Team', 'Games', 'Attempts', 'Yards', 'TDs', 'INTs']
                display_df['Y/A'] = (display_df['Yards'] / display_df['Attempts']).round(2)
                display_df['TD:INT'] = (display_df['TDs'] / display_df['INTs'].replace(0, 1)).round(2)
                st.dataframe(display_df.sort_values('TDs', ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No QB data available with passing TDs or interceptions.")

    except Exception as e:
        st.error(f"Error generating chart: {e}")


def render_qb_yards_attempts_chart(season: Optional[int], week: Optional[int]):
    """Chart: QB Passing Yards vs Attempts (axes swapped)."""
    st.subheader("🎯 QB Passing Yards vs. Attempts")
    st.markdown("*Shows passing efficiency. Steeper slopes indicate higher yards per attempt.*")

    if not season:
        st.warning("No season data available.")
        return

    # Minimum attempts filter
    min_attempts = st.slider("Minimum Pass Attempts", 10, 300, 50, 10, key="qb_yards_att_min_att")

    # Build query to get QB passing stats
    week_filter = f"AND g.week <= {week}" if week else ""

    sql_qb = f"""
    SELECT
        pb.player,
        pb.team,
        SUM(pb.pass_yds) as total_pass_yds,
        SUM(pb.pass_td) as total_pass_tds,
        SUM(pb.pass_int) as total_interceptions,
        SUM(pb.pass_att) as total_pass_att,
        COUNT(DISTINCT pb.game_id) as games
    FROM player_box_score pb
    JOIN games g ON pb.game_id = g.game_id
    WHERE g.season = ?
    {week_filter}
    AND pb.pass_att > 0
    GROUP BY pb.player, pb.team
    HAVING total_pass_att >= ?
    ORDER BY total_pass_yds DESC
    """

    try:
        df = query(sql_qb, (season, min_attempts))

        if df.empty:
            st.info(f"No QB passing data available with at least {min_attempts} attempts.")
            return

        # Calculate yards per attempt for color coding
        chart_df = df.copy()

        if not chart_df.empty:
            chart_df['yards_per_attempt'] = (chart_df['total_pass_yds'] / chart_df['total_pass_att']).round(2)

            import plotly.express as px

            # Swapped: X = Yards, Y = Attempts (was X = Attempts, Y = Yards)
            fig = px.scatter(
                chart_df,
                x='total_pass_yds',
                y='total_pass_att',
                hover_data=['player', 'team', 'yards_per_attempt', 'total_pass_tds', 'games'],
                text='player',
                color='yards_per_attempt',
                labels={
                    'total_pass_yds': 'Passing Yards',
                    'total_pass_att': 'Pass Attempts',
                    'yards_per_attempt': 'Yards/Attempt'
                },
                title=f"QB Passing Yards vs Attempts ({season})",
                color_continuous_scale='Viridis'
            )
            fig.update_traces(textposition='top center', marker=dict(size=12))
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Show data table
            with st.expander("📋 View QB Stats Table"):
                display_df = df[['player', 'team', 'games', 'total_pass_att', 'total_pass_yds',
                                'total_pass_tds', 'total_interceptions']].copy()
                display_df.columns = ['Player', 'Team', 'Games', 'Attempts', 'Yards', 'TDs', 'INTs']
                display_df['Y/A'] = (display_df['Yards'] / display_df['Attempts']).round(2)
                display_df['TD:INT'] = (display_df['TDs'] / display_df['INTs'].replace(0, 1)).round(2)
                st.dataframe(display_df.sort_values('Yards', ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No QB data available with sufficient pass attempts (minimum 10).")

    except Exception as e:
        st.error(f"Error generating chart: {e}")


# ============================================================================
# Section: RB Rushing Charts
# ============================================================================

def render_rb_yards_per_carry_chart(season: Optional[int], week: Optional[int]):
    """Chart: RB Rushing Yards vs Yards per Carry."""
    st.subheader("🏃 RB Yards per Carry")
    st.markdown("*Shows rushing efficiency. Higher Y/C indicates more efficient runners.*")

    if not season:
        st.warning("No season data available.")
        return

    # Minimum carries filter
    min_carries = st.slider("Minimum Rushing Attempts", 10, 200, 30, 10, key="rb_ypc_min_carries")

    # Build query to get RB rushing stats
    week_filter = f"AND g.week <= {week}" if week else ""

    sql_rb = f"""
    SELECT
        pb.player,
        pb.team,
        SUM(pb.rush_att) as total_rush_att,
        SUM(pb.rush_yds) as total_rush_yds,
        SUM(pb.rush_td) as total_rush_tds,
        COUNT(DISTINCT pb.game_id) as games
    FROM player_box_score pb
    JOIN games g ON pb.game_id = g.game_id
    WHERE g.season = ?
    {week_filter}
    AND pb.rush_att > 0
    GROUP BY pb.player, pb.team
    HAVING total_rush_att >= ?
    ORDER BY total_rush_yds DESC
    """

    try:
        df = query(sql_rb, (season, min_carries))

        if df.empty:
            st.info(f"No RB rushing data available with at least {min_carries} carries.")
            return

        # Calculate yards per carry
        chart_df = df.copy()
        chart_df['yards_per_carry'] = (chart_df['total_rush_yds'] / chart_df['total_rush_att']).round(2)

        if not chart_df.empty:
            import plotly.express as px

            # X = Yards, Y = Yards per Carry
            fig = px.scatter(
                chart_df,
                x='total_rush_yds',
                y='yards_per_carry',
                hover_data=['player', 'team', 'total_rush_att', 'total_rush_tds', 'games'],
                text='player',
                color='total_rush_tds',
                labels={
                    'total_rush_yds': 'Rushing Yards',
                    'yards_per_carry': 'Yards per Carry',
                    'total_rush_tds': 'Rush TDs'
                },
                title=f"RB Rushing Yards vs Yards per Carry ({season})",
                color_continuous_scale='Reds'
            )
            fig.update_traces(textposition='top center', marker=dict(size=12))
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Show data table
            with st.expander("📋 View RB Stats Table"):
                display_df = df[['player', 'team', 'games', 'total_rush_att', 'total_rush_yds', 'total_rush_tds']].copy()
                display_df.columns = ['Player', 'Team', 'Games', 'Carries', 'Yards', 'TDs']
                display_df['Y/C'] = (display_df['Yards'] / display_df['Carries']).round(2)
                display_df['Yards/Game'] = (display_df['Yards'] / display_df['Games']).round(1)
                st.dataframe(display_df.sort_values('Yards', ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info(f"No RB data available with at least {min_carries} carries.")

    except Exception as e:
        st.error(f"Error generating chart: {e}")


def render_skill_player_yards_touches_chart(season: Optional[int], week: Optional[int]):
    """Chart: Skill Player Total Yards (Rush + Rec) vs Total Touches (Carries + Receptions)."""
    st.subheader("💨 Skill Player Total Yards vs Touches")
    st.markdown("*Shows offensive involvement and production. X-axis = Total Yards (Rush + Rec), Y-axis = Total Touches (Carries + Receptions).*")

    if not season:
        st.warning("No season data available.")
        return

    # Minimum touches filter
    min_touches = st.slider("Minimum Total Touches", 10, 200, 30, 10, key="skill_total_touches")

    # Build query to get combined stats
    week_filter = f"AND g.week <= {week}" if week else ""

    sql_skill = f"""
    SELECT
        pb.player,
        pb.team,
        SUM(COALESCE(pb.rush_att, 0)) as total_rush_att,
        SUM(COALESCE(pb.rush_yds, 0)) as total_rush_yds,
        SUM(COALESCE(pb.rush_td, 0)) as total_rush_tds,
        SUM(COALESCE(pb.rec, 0)) as total_receptions,
        SUM(COALESCE(pb.rec_yds, 0)) as total_rec_yds,
        SUM(COALESCE(pb.rec_td, 0)) as total_rec_tds,
        COUNT(DISTINCT pb.game_id) as games
    FROM player_box_score pb
    JOIN games g ON pb.game_id = g.game_id
    WHERE g.season = ?
    {week_filter}
    AND (COALESCE(pb.rush_att, 0) > 0 OR COALESCE(pb.rec, 0) > 0 OR COALESCE(pb.targets, 0) > 0)
    GROUP BY pb.player, pb.team
    HAVING (SUM(COALESCE(pb.rush_att, 0)) + SUM(COALESCE(pb.rec, 0))) > 0
    ORDER BY (SUM(COALESCE(pb.rush_yds, 0)) + SUM(COALESCE(pb.rec_yds, 0))) DESC
    """

    try:
        df = query(sql_skill, (season,))

        if df.empty:
            st.info("No skill player data available for the selected season/week.")
            return

        # Calculate combined totals (already summed with COALESCE, so no NULLs)
        df['total_touches'] = df['total_rush_att'] + df['total_receptions']
        df['total_yards'] = df['total_rush_yds'] + df['total_rec_yds']
        df['total_tds'] = df['total_rush_tds'] + df['total_rec_tds']

        # Avoid division by zero
        df['yards_per_touch'] = df.apply(
            lambda row: round(row['total_yards'] / row['total_touches'], 2) if row['total_touches'] > 0 else 0,
            axis=1
        )

        # Filter by minimum touches
        chart_df = df[df['total_touches'] >= min_touches].copy()

        if not chart_df.empty:
            import plotly.express as px

            # X = Total Yards, Y = Total Touches, Size = Total TDs
            fig = px.scatter(
                chart_df,
                x='total_yards',
                y='total_touches',
                size='total_tds',
                hover_data=['player', 'team', 'yards_per_touch', 'total_tds', 'games'],
                text='player',
                color='yards_per_touch',
                labels={
                    'total_yards': 'Total Yards (Rush + Rec)',
                    'total_touches': 'Total Touches (Carries + Receptions)',
                    'yards_per_touch': 'Yards/Touch',
                    'total_tds': 'Total TDs'
                },
                title=f"Skill Player Total Yards vs Touches ({season})",
                color_continuous_scale='Viridis',
                size_max=30
            )
            fig.update_traces(textposition='top center')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Show data table
            with st.expander("📋 View Skill Player Stats Table"):
                display_df = chart_df[['player', 'team', 'games', 'total_rush_att', 'total_rush_yds',
                                       'total_receptions', 'total_rec_yds', 'total_touches',
                                       'total_yards', 'total_tds']].copy()
                display_df.columns = ['Player', 'Team', 'Games', 'Carries', 'Rush Yds',
                                     'Rec', 'Rec Yds', 'Total Touches', 'Total Yds', 'Total TDs']
                display_df['Yds/Touch'] = (display_df['Total Yds'] / display_df['Total Touches']).round(2)
                display_df['Yds/Game'] = (display_df['Total Yds'] / display_df['Games']).round(1)
                st.dataframe(display_df.sort_values('Total Yds', ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info(f"No skill player data available with at least {min_touches} touches.")

    except Exception as e:
        st.error(f"Error generating chart: {e}")


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main application entry point."""

    # Initialize database tables on every startup to ensure they exist
    try:
        init_notes_table()
        init_injuries_table()
        init_transactions_table()
        init_upcoming_games_table()
    except Exception as e:
        st.error(f"Database initialization error: {e}")
        st.info("Some features may not work correctly. Please check database permissions.")

    # Render sidebar and get selections
    view, season, week, team = render_sidebar()

    # Handle view override from session state (e.g., from Quick Notes "View All" button)
    if 'view_override' in st.session_state:
        view = st.session_state['view_override']
        del st.session_state['view_override']

    # Render selected view
    if view == "Games Browser":
        render_games_browser(season, week)

    elif view == "Charts":
        render_charts_view(season, week)

    elif view == "Team Overview":
        render_team_overview(season, week)

    elif view == "Team Comparison":
        render_team_comparison(season, week)

    elif view == "Power Rankings":
        render_power_rankings(season, week)

    elif view == "Stats & Trends":
        render_stats_trends(season, week)

    elif view == "Advanced Team Analytics":
        render_advanced_team_analytics(season, week, team)

    elif view == "Matchup Predictor":
        render_matchup_predictor(season, week)

    elif view == "Upcoming Matches":
        render_upcoming_matches(season, week)

    elif view == "Skill Yards Grid":
        render_skill_yards_grid(season, week)

    elif view == "Skill TDs Grid":
        render_skill_tds_grid(season, week)

    elif view == "First TD Grid":
        render_first_td_grid(season, week)

    elif view == "First TD Detail":
        render_first_td_detail(season, week)

    elif view == "TD Against":
        render_td_against(season, week)

    elif view == "Player Stats":
        render_player_stats(season, week, team)

    elif view == "Season Leaderboards":
        render_season_leaderboards(season)

    elif view == "Play-by-Play Viewer":
        render_play_by_play(season, week, team)

    elif view == "Game Detail":
        render_game_detail(season, week)

    elif view == "Notes Manager":
        render_notes_manager(season, week)

    elif view == "Transaction Manager":
        render_transaction_manager(season, week)


if __name__ == "__main__":
    main()
