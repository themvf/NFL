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
from historical_trends import render_historical_trends

# Configure logging
logging.basicConfig(
    filename='nfl_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Database configuration - relative path for deployment
DB_PATH = Path(__file__).parent / "data" / "nfl_merged.db"

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
        'DAL': 'dal', 'DEN': 'den', 'DET': 'det', 'GB': 'gb',
        'HOU': 'hou', 'IND': 'ind', 'JAX': 'jax', 'KC': 'kc',
        'LAC': 'lac', 'LA': 'lar', 'LV': 'lv', 'MIA': 'mia',
        'MIN': 'min', 'NO': 'no', 'NE': 'ne', 'NYG': 'nyg',
        'NYJ': 'nyj', 'PHI': 'phi', 'PIT': 'pit', 'SEA': 'sea',
        'SF': 'sf', 'TB': 'tb', 'TEN': 'ten', 'WAS': 'wsh'
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


def init_projection_accuracy_table():
    """Initialize projection_accuracy table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projection_accuracy (
                projection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                team_abbr TEXT NOT NULL,
                opponent_abbr TEXT NOT NULL,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                position TEXT NOT NULL,
                projected_yds REAL NOT NULL,
                actual_yds REAL,
                multiplier REAL,
                matchup_rating TEXT,
                avg_yds_game REAL,
                median_yds REAL,
                games_played REAL,
                variance REAL,
                abs_error REAL,
                pct_error REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player_name, team_abbr, season, week, position)
            )
        """)

        # Create indexes for better query performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_projection_season_week
            ON projection_accuracy(season, week)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_projection_player
            ON projection_accuracy(player_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_projection_position
            ON projection_accuracy(position)
        """)

        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error initializing projection accuracy table: {e}")


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
    - team_filter: Filter by team abbreviation (e.g., 'NE')
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
                SELECT player_name, team_abbr, season FROM player_injuries
                WHERE player_name = ? AND team_abbr = ? AND season = ?
            """, (player_name, team, season))
            result = cursor.fetchone()

            conn.close()

            if result:
                logging.info(f"Successfully saved injury for {player_name} (rowid: {injury_id})")
                # Upload database to GCS after successful save (non-blocking)
                try:
                    upload_db_to_gcs()
                except Exception as gcs_error:
                    logging.warning(f"GCS upload failed but injury saved: {gcs_error}")
                    # Still return success since database save worked
                # Return lastrowid as the ID (injury_id column may be NULL if no AUTOINCREMENT)
                return injury_id if injury_id else 1, None
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

        # No transaction found - infer from game data (use player_stats for consistency)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        sql = "SELECT team FROM player_stats WHERE player_display_name = ? AND season = ?"
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

        # Get all teams from player_stats table
        teams_query = f"SELECT DISTINCT team FROM player_stats WHERE season = {season}"
        teams_df = pd.read_sql_query(teams_query, conn)
        teams = teams_df['team'].tolist()

        defensive_stats = {}

        for team in teams:
            # Pass yards and TDs allowed (opponent QBs)
            # Use opponent_team field to find players who played AGAINST this team
            pass_query = f"""
                SELECT
                    AVG(passing_yards) as avg_pass_allowed,
                    SUM(passing_tds) as total_pass_td_allowed
                FROM player_stats
                WHERE season = {season} AND week < {max_week} AND attempts > 10
                  AND opponent_team = '{team}'
            """
            pass_df = pd.read_sql_query(pass_query, conn)

            # Rush yards and TDs allowed (opponent RBs only, excludes QBs)
            # Only count players with 50+ total touches to exclude QB scrambles
            # Calculate per-game totals (all RBs combined), then average across games
            rush_query = f"""
                SELECT
                    AVG(game_yards) as avg_rush_allowed,
                    SUM(game_tds) as total_rush_td_allowed
                FROM (
                    SELECT
                        ps.week,
                        SUM(ps.rushing_yards) as game_yards,
                        SUM(ps.rushing_tds) as game_tds
                    FROM player_stats ps
                    INNER JOIN (
                        SELECT player_display_name, team
                        FROM player_stats
                        WHERE season = {season} AND week < {max_week} AND carries >= 5
                        GROUP BY player_display_name, team
                        HAVING SUM(carries + targets) >= 50
                    ) qualified_rbs
                    ON ps.player_display_name = qualified_rbs.player_display_name
                       AND ps.team = qualified_rbs.team
                    WHERE ps.season = {season} AND ps.week < {max_week}
                      AND ps.opponent_team = '{team}'
                    GROUP BY ps.week
                ) per_game
            """
            rush_df = pd.read_sql_query(rush_query, conn)

            # Receiving yards allowed to RBs (opponents with rush + rec)
            # Calculate per-game totals, then average
            rec_rb_query = f"""
                SELECT AVG(game_total) as avg_rec_to_rb
                FROM (
                    SELECT week, SUM(receiving_yards) as game_total
                    FROM player_stats
                    WHERE season = {season} AND week < {max_week}
                      AND carries >= 5 AND targets > 0
                      AND opponent_team = '{team}'
                    GROUP BY week
                ) per_game
            """
            rec_rb_df = pd.read_sql_query(rec_rb_query, conn)

            # Receiving yards and TDs allowed to WRs
            # Calculate per-game totals, then average
            rec_wr_query = f"""
                SELECT
                    AVG(game_yards) as avg_rec_to_wr,
                    SUM(game_tds) as total_rec_td_to_wr
                FROM (
                    SELECT
                        week,
                        SUM(receiving_yards) as game_yards,
                        SUM(receiving_tds) as game_tds
                    FROM player_stats
                    WHERE season = {season} AND week < {max_week}
                      AND targets >= 4 AND carries < 3
                      AND opponent_team = '{team}'
                    GROUP BY week
                ) per_game
            """
            rec_wr_df = pd.read_sql_query(rec_wr_query, conn)

            # Receiving yards allowed to TEs
            # Calculate per-game totals, then average
            rec_te_query = f"""
                SELECT AVG(game_total) as avg_rec_to_te
                FROM (
                    SELECT week, SUM(receiving_yards) as game_total
                    FROM player_stats
                    WHERE season = {season} AND week < {max_week}
                      AND targets >= 2 AND targets < 10 AND carries < 2
                      AND opponent_team = '{team}'
                    GROUP BY week
                ) per_game
            """
            rec_te_df = pd.read_sql_query(rec_te_query, conn)

            # Get advanced defensive stats from pfr_advstats_def_week
            def_adv_query = f"""
                SELECT
                    SUM(def_ints) as total_def_ints,
                    SUM(def_times_blitzed) as total_blitzes,
                    SUM(def_times_hurried) as total_hurries,
                    SUM(def_sacks) as total_sacks
                FROM pfr_advstats_def_week
                WHERE season = {season} AND week < {max_week} AND team = '{team}'
            """
            def_adv_df = pd.read_sql_query(def_adv_query, conn)

            # QB rushing yards and TDs allowed (QBs only - high pass attempts)
            # Calculate per-game average of total QB rushing yards and TDs
            qb_rush_allowed_query = f"""
                SELECT
                    AVG(game_rush_yds) as avg_qb_rush_allowed,
                    SUM(game_rush_tds) as total_qb_rush_tds_allowed
                FROM (
                    SELECT
                        week,
                        SUM(rushing_yards) as game_rush_yds,
                        SUM(rushing_tds) as game_rush_tds
                    FROM player_stats
                    WHERE season = {season} AND week < {max_week}
                      AND attempts > 10
                      AND opponent_team = '{team}'
                    GROUP BY week
                ) per_game
            """
            qb_rush_df = pd.read_sql_query(qb_rush_allowed_query, conn)

            defensive_stats[team] = {
                'pass_allowed': pass_df['avg_pass_allowed'].iloc[0] if not pass_df.empty and not pd.isna(pass_df['avg_pass_allowed'].iloc[0]) else 240,
                'pass_td_allowed': pass_df['total_pass_td_allowed'].iloc[0] if not pass_df.empty and not pd.isna(pass_df['total_pass_td_allowed'].iloc[0]) else 0,
                'rush_allowed': rush_df['avg_rush_allowed'].iloc[0] if not rush_df.empty and not pd.isna(rush_df['avg_rush_allowed'].iloc[0]) else 80,
                'rush_td_allowed': rush_df['total_rush_td_allowed'].iloc[0] if not rush_df.empty and not pd.isna(rush_df['total_rush_td_allowed'].iloc[0]) else 0,
                'rec_to_rb': rec_rb_df['avg_rec_to_rb'].iloc[0] if not rec_rb_df.empty and not pd.isna(rec_rb_df['avg_rec_to_rb'].iloc[0]) else 20,
                'rec_to_wr': rec_wr_df['avg_rec_to_wr'].iloc[0] if not rec_wr_df.empty and not pd.isna(rec_wr_df['avg_rec_to_wr'].iloc[0]) else 60,
                'rec_td_to_wr': rec_wr_df['total_rec_td_to_wr'].iloc[0] if not rec_wr_df.empty and not pd.isna(rec_wr_df['total_rec_td_to_wr'].iloc[0]) else 0,
                'rec_to_te': rec_te_df['avg_rec_to_te'].iloc[0] if not rec_te_df.empty and not pd.isna(rec_te_df['avg_rec_to_te'].iloc[0]) else 40,
                'qb_rush_allowed': qb_rush_df['avg_qb_rush_allowed'].iloc[0] if not qb_rush_df.empty and not pd.isna(qb_rush_df['avg_qb_rush_allowed'].iloc[0]) else 10,
                'qb_rush_tds_allowed': qb_rush_df['total_qb_rush_tds_allowed'].iloc[0] if not qb_rush_df.empty and not pd.isna(qb_rush_df['total_qb_rush_tds_allowed'].iloc[0]) else 0,
                'def_ints': def_adv_df['total_def_ints'].iloc[0] if not def_adv_df.empty and not pd.isna(def_adv_df['total_def_ints'].iloc[0]) else 0,
                'def_blitzes': def_adv_df['total_blitzes'].iloc[0] if not def_adv_df.empty and not pd.isna(def_adv_df['total_blitzes'].iloc[0]) else 0,
                'def_hurries': def_adv_df['total_hurries'].iloc[0] if not def_adv_df.empty and not pd.isna(def_adv_df['total_hurries'].iloc[0]) else 0,
                'def_sacks': def_adv_df['total_sacks'].iloc[0] if not def_adv_df.empty and not pd.isna(def_adv_df['total_sacks'].iloc[0]) else 0
            }

        conn.close()
        return defensive_stats

    except Exception as e:
        st.error(f"Error calculating defensive stats: {e}")
        return {}


def calculate_rb_rankings_and_opponent_quality(season, max_week):
    """
    Calculate RB rankings by average rushing yards and track opponent quality faced by each defense.
    Uses 50+ total touches threshold to filter out backups and QBs.

    Returns:
        rb_rankings: dict {player_name: rank} where 1 = best RB
        defense_opponent_quality: dict {team: avg_opponent_rb_rank}
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # Get all RBs with 50+ total touches (excludes QBs and deep backups)
        rb_query = f"""
            SELECT
                player_display_name as player,
                team,
                AVG(rushing_yards) as avg_rush_yds,
                SUM(carries + targets) as total_touches,
                COUNT(*) as games
            FROM player_stats
            WHERE season = {season} AND week < {max_week}
              AND carries >= 5
            GROUP BY player_display_name, team
            HAVING total_touches >= 50
            ORDER BY avg_rush_yds DESC
        """
        rb_df = pd.read_sql_query(rb_query, conn)

        # Create rankings: 1 = highest avg rushing yards
        rb_rankings = {}
        for rank, row in enumerate(rb_df.itertuples(), 1):
            rb_rankings[row.player] = {
                'rank': rank,
                'avg_rush_yds': row.avg_rush_yds,
                'team': row.team
            }

        # Track which RBs each defense faced (only RBs with 50+ touches)
        # This excludes QBs and backup RBs
        defense_opponents_query = f"""
            SELECT
                ps.opponent_team as defense,
                ps.player_display_name as rb_name,
                AVG(ps.rushing_yards) as avg_yds
            FROM player_stats ps
            INNER JOIN (
                SELECT player_display_name, team
                FROM player_stats
                WHERE season = {season} AND week < {max_week} AND carries >= 5
                GROUP BY player_display_name, team
                HAVING SUM(carries + targets) >= 50
            ) qualified_rbs
            ON ps.player_display_name = qualified_rbs.player_display_name
               AND ps.team = qualified_rbs.team
            WHERE ps.season = {season} AND ps.week < {max_week}
            GROUP BY ps.opponent_team, ps.player_display_name
        """
        def_opponents_df = pd.read_sql_query(defense_opponents_query, conn)

        # Calculate average opponent RB rank for each defense
        defense_opponent_quality = {}
        for defense in def_opponents_df['defense'].unique():
            rbs_faced = def_opponents_df[def_opponents_df['defense'] == defense]['rb_name'].tolist()

            # Get ranks of RBs faced (only those in our rankings)
            ranks_faced = [rb_rankings[rb]['rank'] for rb in rbs_faced if rb in rb_rankings]

            if ranks_faced:
                defense_opponent_quality[defense] = {
                    'avg_opponent_rank': sum(ranks_faced) / len(ranks_faced),
                    'num_rbs_faced': len(ranks_faced),
                    'best_rb_rank': min(ranks_faced),
                    'worst_rb_rank': max(ranks_faced)
                }
            else:
                defense_opponent_quality[defense] = {
                    'avg_opponent_rank': 0,
                    'num_rbs_faced': 0,
                    'best_rb_rank': 0,
                    'worst_rb_rank': 0
                }

        conn.close()
        return rb_rankings, defense_opponent_quality

    except Exception as e:
        st.error(f"Error calculating RB rankings: {e}")
        return {}, {}


def calculate_player_medians(season, max_week, teams_playing=None):
    """
    Calculate median stats for all players by position.

    Returns DataFrame with columns: player, team, position_type, median stats, games_played
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # Base query for all players - updated for player_stats table
        base_query = f"""
            SELECT
                player_display_name as player,
                team,
                passing_yards as pass_yds,
                passing_tds as pass_td,
                passing_interceptions as pass_int,
                completions as pass_comp,
                attempts as pass_att,
                rushing_yards as rush_yds,
                carries as rush_att,
                receiving_yards as rec_yds,
                receptions as rec,
                targets,
                rushing_tds as rush_td,
                receiving_tds as rec_td,
                receiving_air_yards,
                receiving_yards_after_catch
            FROM player_stats
            WHERE season = {season} AND week < {max_week}
        """

        if teams_playing:
            teams_list = "', '".join(teams_playing)
            base_query += f" AND team IN ('{teams_list}')"

        df = pd.read_sql_query(base_query, conn)
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
                    'median_rush_td': group['rush_td'].median(),
                    'median_rush_yds': group['rush_yds'].median(),
                    'total_pass_td': group['pass_td'].sum(),
                    'total_rush_td': group['rush_td'].sum(),
                    'total_pass_int': group['pass_int'].sum(),
                    'total_pass_att': group['pass_att'].sum(),
                    'median_pass_comp_pct': (group['pass_comp'].sum() / group['pass_att'].sum() * 100) if group['pass_att'].sum() > 0 else 0
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
                    'median_rush_td': group['rush_td'].median(),
                    'median_rec_td': group['rec_td'].median(),
                    'total_rush_td': group['rush_td'].sum(),
                    'total_rec_td': group['rec_td'].sum(),
                    'median_targets': group['targets'].median(),
                    'avg_rec_air_yds': group['receiving_air_yards'].mean(),
                    'avg_rec_yac': group['receiving_yards_after_catch'].mean(),
                    # Additional stats for comprehensive RB scoring
                    'total_rush_yds': group['rush_yds'].sum(),
                    'total_rec_yds': group['rec_yds'].sum(),
                    'total_carries': group['rush_att'].sum(),
                    'total_targets': group['targets'].sum(),
                    'total_receptions': group['rec'].sum(),
                    'avg_rush_yds': group['rush_yds'].mean(),
                    'avg_targets': group['targets'].mean()
                })

            # WR: High targets, low rushing
            elif avg_targets >= 4 and avg_rush_att < 3:
                position = 'WR'

                # Calculate last 3 games averages
                last_3_games = group.tail(3) if len(group) >= 3 else group

                stats.update({
                    'avg_rec_yds': group['rec_yds'].mean(),
                    'median_rec_yds': group['rec_yds'].median(),
                    'median_targets': group['targets'].median(),
                    'median_rec_td': group['rec_td'].median(),
                    'median_rush_td': group['rush_td'].median(),
                    'total_rush_td': group['rush_td'].sum(),
                    'total_rec_td': group['rec_td'].sum(),
                    'total_targets': group['targets'].sum(),
                    'total_receptions': group['rec'].sum(),
                    'last_3_avg_targets': last_3_games['targets'].mean(),
                    'last_3_avg_receptions': last_3_games['rec'].mean(),
                    'median_rec': group['rec'].median(),
                    'avg_rec_air_yds': group['receiving_air_yards'].mean(),
                    'avg_rec_yac': group['receiving_yards_after_catch'].mean(),
                    'avg_targets': group['targets'].mean(),
                    'total_rec_yds': group['rec_yds'].sum()
                })

            # TE: Moderate targets
            elif avg_targets >= 2 and avg_targets < 10 and avg_rush_att < 2:
                position = 'TE'
                stats.update({
                    'avg_rec_yds': group['rec_yds'].mean(),
                    'median_rec_yds': group['rec_yds'].median(),
                    'median_targets': group['targets'].median(),
                    'median_rec_td': group['rec_td'].median(),
                    'total_rec_td': group['rec_td'].sum(),
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

        # Calculate RB rankings and opponent quality
        rb_rankings, defense_opponent_quality = calculate_rb_rankings_and_opponent_quality(season, week)

        # Calculate defensive rankings for yards allowed by position
        # Lower yards = better defense = lower rank number (1 = best, 32 = worst)
        def_pass_ranking = {}
        def_rush_ranking = {}
        def_rec_ranking = {}
        def_total_ranking = {}  # For skill players (rush + rec combined)

        if defensive_stats:
            # Pass defense ranking
            sorted_pass = sorted(defensive_stats.items(), key=lambda x: x[1]['pass_allowed'])
            for rank, (team, _) in enumerate(sorted_pass, 1):
                def_pass_ranking[team] = rank

            # Rush defense ranking
            sorted_rush = sorted(defensive_stats.items(), key=lambda x: x[1]['rush_allowed'])
            for rank, (team, _) in enumerate(sorted_rush, 1):
                def_rush_ranking[team] = rank

            # Receiving defense ranking (WRs)
            sorted_rec = sorted(defensive_stats.items(), key=lambda x: x[1]['rec_to_wr'])
            for rank, (team, _) in enumerate(sorted_rec, 1):
                def_rec_ranking[team] = rank

            # Total yards ranking for skill players (rush + rec combined)
            sorted_total = sorted(defensive_stats.items(), key=lambda x: x[1]['rush_allowed'] + x[1]['rec_to_wr'])
            for rank, (team, _) in enumerate(sorted_total, 1):
                def_total_ranking[team] = rank

        # Calculate league averages for normalization
        league_avg = {
            'pass': sum([d['pass_allowed'] for d in defensive_stats.values()]) / len(defensive_stats) if defensive_stats else 240,
            'rush': sum([d['rush_allowed'] for d in defensive_stats.values()]) / len(defensive_stats) if defensive_stats else 80,
            'rec_rb': sum([d['rec_to_rb'] for d in defensive_stats.values()]) / len(defensive_stats) if defensive_stats else 20,
            'rec_wr': sum([d['rec_to_wr'] for d in defensive_stats.values()]) / len(defensive_stats) if defensive_stats else 60,
            'rec_te': sum([d['rec_to_te'] for d in defensive_stats.values()]) / len(defensive_stats) if defensive_stats else 40,
            'qb_rush': sum([d['qb_rush_allowed'] for d in defensive_stats.values()]) / len(defensive_stats) if defensive_stats else 10,
            'qb_rush_tds': sum([d['qb_rush_tds_allowed'] for d in defensive_stats.values()]) / len(defensive_stats) if defensive_stats else 0.5,
            'def_ints': sum([d['def_ints'] for d in defensive_stats.values()]) / len(defensive_stats) if defensive_stats else 8,
            'def_blitzes': sum([d['def_blitzes'] for d in defensive_stats.values()]) / len(defensive_stats) if defensive_stats else 50,
            'def_sacks': sum([d['def_sacks'] for d in defensive_stats.values()]) / len(defensive_stats) if defensive_stats else 20,
            'pass_tds': sum([d['pass_td_allowed'] for d in defensive_stats.values()]) / len(defensive_stats) if defensive_stats else 15,
            'def_rush_tds': sum([d['rush_td_allowed'] for d in defensive_stats.values()]) / len(defensive_stats) if defensive_stats else 1.0,
            'def_rec_tds': sum([d['rec_td_to_wr'] for d in defensive_stats.values()]) / len(defensive_stats) if defensive_stats else 1.0
        }

        # Get player medians
        player_medians = calculate_player_medians(season, week, teams_playing)

        if player_medians.empty:
            return {}

        # Get matchups and injured players from database
        conn = sqlite3.connect(DB_PATH)

        # Query from schedules table instead of upcoming_games
        matchups_query = f"""
            SELECT home_team, away_team
            FROM schedules
            WHERE season = {season} AND week = {week} AND game_type = 'REG'
        """
        matchups_df = pd.read_sql_query(matchups_query, conn)

        # Get injured players for this week
        injuries_query = f"""
            SELECT player_name, team_abbr
            FROM player_injuries
            WHERE season = {season}
              AND {week} >= start_week
              AND {week} <= end_week
        """
        injuries_df = pd.read_sql_query(injuries_query, conn)

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
                # Composite multiplier incorporating 5 defensive factors
                # 1. Passing yards allowed (40% weight - primary factor)
                pass_yds_mult = opponent_def['pass_allowed'] / league_avg['pass']

                # 2. Passing TDs allowed (25% weight - scoring opportunities)
                pass_td_mult = opponent_def['pass_td_allowed'] / league_avg['pass_tds']

                # 3. QB rushing yards allowed (20% weight - dual-threat opportunity)
                qb_rush_mult = opponent_def['qb_rush_allowed'] / league_avg['qb_rush']

                # 4. Defensive INTs (10% weight - inverse because lower is better for QB)
                # Invert the ratio: defenses with MORE INTs get lower multiplier
                int_mult = league_avg['def_ints'] / opponent_def['def_ints'] if opponent_def['def_ints'] > 0 else 1.0

                # 5. Defensive sacks (5% weight - inverse because lower is better for QB)
                # Invert the ratio: defenses with MORE sacks get lower multiplier
                sack_mult = league_avg['def_sacks'] / opponent_def['def_sacks'] if opponent_def['def_sacks'] > 0 else 1.0

                # Weighted composite multiplier
                multiplier = (
                    pass_yds_mult * 0.40 +
                    pass_td_mult * 0.25 +
                    qb_rush_mult * 0.20 +
                    int_mult * 0.10 +
                    sack_mult * 0.05
                )

                # Apply multiplier to median passing yards
                projected_yds = player['median_pass_yds'] * multiplier

                # Calculate comprehensive QB score
                qb_score = calculate_comprehensive_qb_score(
                    qb_yards_per_game=player['avg_pass_yds'],
                    qb_tds_total=player['total_pass_td'],
                    qb_ints_total=player['total_pass_int'],
                    qb_attempts_total=player['total_pass_att'],
                    qb_rush_tds_total=player['total_rush_td'],
                    qb_games=player['games_played'],
                    def_pass_allowed=opponent_def['pass_allowed'],
                    def_pass_tds_allowed=opponent_def['pass_td_allowed'],
                    def_ints=opponent_def['def_ints'],
                    def_sacks=opponent_def['def_sacks'],
                    def_hurries=opponent_def['def_hurries'],
                    def_blitzes=opponent_def['def_blitzes'],
                    league_avg_pass_yds=league_avg['pass'],
                    league_avg_pass_tds=league_avg['pass_tds'],
                    league_avg_def_ints=league_avg['def_ints'],
                    league_avg_def_sacks=league_avg['def_sacks']
                )

                # Calculate INT rate
                int_rate = (player['total_pass_int'] / player['total_pass_att'] * 100) if player['total_pass_att'] > 0 else 0
                tds_per_game = player['total_pass_td'] / player['games_played'] if player['games_played'] > 0 else 0

                # Generate storyline
                tier, recommendation = generate_comprehensive_qb_storyline(
                    qb_score=qb_score,
                    qb_name=player['player'],
                    qb_yards_per_game=player['avg_pass_yds'],
                    qb_tds_per_game=tds_per_game,
                    qb_int_rate=int_rate,
                    qb_rush_tds=player['total_rush_td'],
                    def_pass_allowed=opponent_def['pass_allowed'],
                    def_sacks=opponent_def['def_sacks']
                )

                # Calculate pressure score
                pressure_score = opponent_def['def_sacks'] + (opponent_def['def_hurries'] * 0.5) + (opponent_def['def_blitzes'] * 0.3)

                # Calculate QB Defensive Composite Score (0-100 scale - higher = worse defense = better for QB)
                qb_def_score = calculate_qb_defensive_composite_score(
                    def_pass_yards_allowed=opponent_def['pass_allowed'],
                    def_pass_tds_allowed=opponent_def['pass_td_allowed'],
                    def_qb_rush_yards_allowed=opponent_def['qb_rush_allowed'],
                    def_qb_rush_tds_allowed=opponent_def['qb_rush_tds_allowed'],
                    def_blitzes=opponent_def['def_blitzes'],
                    def_sacks=opponent_def['def_sacks'],
                    league_avg_pass_yards=league_avg['pass'],
                    league_avg_pass_tds=league_avg['pass_tds'],
                    league_avg_qb_rush_yards=league_avg['qb_rush'],
                    league_avg_qb_rush_tds=league_avg['qb_rush_tds'],
                    league_avg_blitzes=league_avg['def_blitzes'],
                    league_avg_sacks=league_avg['def_sacks']
                )

                # Calculate Matchup Score (weighted combination of QB talent and defensive matchup)
                # 60% QB Score (talent) + 40% QB Def Score (matchup favorability)
                # This gives more weight to QB skill since elite QBs produce even vs tough defenses
                matchup_score = (qb_score * 0.6) + (qb_def_score * 0.4)

                projections['QB'].append({
                    'Player': player['player'],
                    'Team': player['team'],
                    'Opponent': opponent,
                    'QB Score': round(qb_score, 1),
                    'QB Def Score': round(qb_def_score, 1),
                    'Matchup Score': round(matchup_score, 1),
                    'Tier': tier,
                    'Avg Yds/Game': round(player['avg_pass_yds'], 1),
                    'Median Pass Yds': round(player['median_pass_yds'], 1),
                    'Pass TDs': int(player['total_pass_td']),
                    'TDs/Game': round(tds_per_game, 2),
                    'Pass INTs': int(player['total_pass_int']),
                    'INT Rate %': round(int_rate, 1),
                    'Rush TDs': int(player['total_rush_td']),
                    'Rush Yds': round(player['median_rush_yds'], 1),
                    'Def Allows': round(opponent_def['pass_allowed'], 1),
                    'Def Pass TDs': int(opponent_def['pass_td_allowed']),
                    'Def QB Rush Yds': round(opponent_def['qb_rush_allowed'], 1),
                    'Def INTs': int(opponent_def['def_ints']),
                    'Def Sacks': int(opponent_def['def_sacks']),
                    'Pressure Score': round(pressure_score, 1),
                    'Def Pass Rank': def_pass_ranking.get(opponent, 16),
                    'Projected Yds': round(projected_yds, 1),
                    'Multiplier': round(multiplier, 2),
                    'Pass Yds Mult': round(pass_yds_mult, 2),
                    'Pass TD Mult': round(pass_td_mult, 2),
                    'QB Rush Mult': round(qb_rush_mult, 2),
                    'INT Mult': round(int_mult, 2),
                    'Sack Mult': round(sack_mult, 2),
                    'Recommendation': recommendation,
                    'Games': round(float(player['games_played']), 1)
                })

            elif position == 'RB':
                # Calculate separate RB talent score and defensive matchup score
                rb_talent_score = calculate_rb_talent_score(
                    rb_rush_yds_per_game=player['avg_rush_yds'],
                    rb_rush_tds_total=player['total_rush_td'],
                    rb_rec_tds_total=player['total_rec_td'],
                    rb_targets_per_game=player['avg_targets'],
                    rb_games=player['games_played']
                )

                rb_def_score = calculate_rb_defensive_matchup_score(
                    def_rush_allowed=opponent_def['rush_allowed'],
                    def_rush_tds_allowed=opponent_def['rush_td_allowed'],
                    league_avg_def_rush_tds=league_avg['def_rush_tds']
                )

                # Calculate Matchup Score (weighted combination of talent and matchup)
                # 60% RB talent + 40% defensive matchup
                # Elite RBs produce even vs tough defenses, so weight talent more heavily
                matchup_score = (rb_talent_score * 0.6) + (rb_def_score * 0.4)

                # Calculate projected rushing yards based on defensive matchup
                # Use actual defensive yards allowed relative to league average
                # Defense allows 100 yds, league avg is 80 = 1.25x multiplier (favorable matchup)
                # Defense allows 60 yds, league avg is 80 = 0.75x multiplier (tough matchup)
                def_multiplier = opponent_def['rush_allowed'] / league_avg['rush'] if league_avg['rush'] > 0 else 1.0
                projected_rush_yds = player['avg_rush_yds'] * def_multiplier

                # Calculate TD Probability %
                rb_total_tds = player['total_rush_td'] + player['total_rec_td']
                league_avg_total_tds = league_avg['def_rush_tds']  # Use as baseline for all TDs
                td_probability = calculate_rb_td_probability(
                    rb_total_tds=rb_total_tds,
                    rb_games=player['games_played'],
                    def_rush_tds_allowed=opponent_def['rush_td_allowed'],
                    def_rec_tds_to_rb=opponent_def.get('rec_td_to_rb', 0),  # RB receiving TDs allowed
                    league_avg_tds=league_avg_total_tds,
                    td_std_dev=None  # TODO: Calculate from game-by-game data in future
                )

                # Keep legacy comprehensive score for backward compatibility
                rb_score = calculate_comprehensive_rb_score(
                    rb_rush_yds_per_game=player['avg_rush_yds'],
                    rb_rush_tds_total=player['total_rush_td'],
                    rb_rec_tds_total=player['total_rec_td'],
                    rb_targets_per_game=player['avg_targets'],
                    rb_games=player['games_played'],
                    def_rush_allowed=opponent_def['rush_allowed'],
                    def_rush_tds_allowed=opponent_def['rush_td_allowed'],
                    league_avg_rush_yds=league_avg['rush'],
                    league_avg_def_rush_tds=league_avg['def_rush_tds']
                )

                # Calculate per-game rates
                rb_rush_tds_per_game = player['total_rush_td'] / player['games_played'] if player['games_played'] > 0 else 0
                rb_rec_tds_per_game = player['total_rec_td'] / player['games_played'] if player['games_played'] > 0 else 0
                rb_total_tds_per_game = rb_rush_tds_per_game + rb_rec_tds_per_game
                rb_touches_per_game = (player['total_carries'] + player['total_targets']) / player['games_played'] if player['games_played'] > 0 else 0

                # Generate storyline
                tier, recommendation = generate_comprehensive_rb_storyline(
                    rb_score=rb_score,
                    rb_name=player['player'],
                    rb_rush_yds_per_game=player['avg_rush_yds'],
                    rb_rush_tds_per_game=rb_rush_tds_per_game,
                    rb_rec_tds_per_game=rb_rec_tds_per_game,
                    rb_targets_per_game=player['avg_targets'],
                    rb_total_tds_per_game=rb_total_tds_per_game,
                    def_rush_allowed=opponent_def['rush_allowed'],
                    def_rush_tds_allowed=opponent_def['rush_td_allowed']
                )

                # Keep old projection logic for backward compatibility
                rush_mult = opponent_def['rush_allowed'] / league_avg['rush']
                rec_mult = opponent_def['rec_to_rb'] / league_avg['rec_rb']
                proj_rush = player['median_rush_yds'] * rush_mult
                proj_rec = player['median_rec_yds'] * rec_mult
                proj_total = proj_rush + proj_rec
                avg_mult = (rush_mult + rec_mult) / 2
                combined_median = player['median_rush_yds'] + player['median_rec_yds']

                # Get RB ranking and opponent quality
                rb_rank = rb_rankings.get(player['player'], {}).get('rank', 999)
                opp_quality = defense_opponent_quality.get(opponent, {})
                avg_opp_rank = opp_quality.get('avg_opponent_rank', 0)

                # Get defensive run style
                try:
                    def_style_info = get_defensive_run_style_matchup(
                        offense_team=player['team'],
                        defense_team=opponent,
                        season=season,
                        week=week
                    )
                    def_run_style = def_style_info['style']
                    def_run_style_insight = def_style_info['rb_matchup_insight']
                except Exception:
                    def_run_style = "Unknown"
                    def_run_style_insight = "No data available"

                projections['RB'].append({
                    'RB Score': round(rb_talent_score, 1),
                    'RB Def Score': round(rb_def_score, 1),
                    'Matchup Score': round(matchup_score, 1),
                    'Tier': tier,
                    'Player': player['player'],
                    'Team': player['team'],
                    'Opponent': opponent,
                    'Rush Yds/Gm': round(player['avg_rush_yds'], 1),
                    'Projected Rush Yds': round(projected_rush_yds, 1),
                    'Rec Yds/Gm': round(player.get('total_rec_yds', 0) / player['games_played'], 1) if player['games_played'] > 0 else 0,
                    'Rush TDs/Gm': round(rb_rush_tds_per_game, 2),
                    'Rec TDs/Gm': round(rb_rec_tds_per_game, 2),
                    'Total TDs/Gm': round(rb_total_tds_per_game, 2),
                    'TD Probability %': td_probability,
                    'Targets/Gm': round(player['avg_targets'], 1),
                    'Touches/Gm': round(rb_touches_per_game, 1),
                    'Def Rush Yds': round(opponent_def['rush_allowed'], 1),
                    'Def Rush TDs': round(opponent_def['rush_td_allowed'], 1),
                    'Def Run Style': def_run_style,
                    'Def Style Insight': def_run_style_insight,
                    'Recommendation': recommendation,
                    # Keep legacy columns for backward compatibility
                    'RB Rank': rb_rank,
                    'Avg Yds/Game': round(player['avg_total_yds'], 1),
                    'Median Rush': round(player['median_rush_yds'], 1),
                    'Median Rec': round(player['median_rec_yds'], 1),
                    'Total Median': round(combined_median, 1),
                    'Rush TDs': int(player['total_rush_td']),
                    'Rec TDs': int(player['total_rec_td']),
                    'Total TDs': int(player['total_rush_td'] + player['total_rec_td']),
                    'Def Avg Opp RB Rank': round(avg_opp_rank, 1) if avg_opp_rank > 0 else 0,
                    'Def Rush Rank': def_rush_ranking.get(opponent, 16),
                    'Projected Total': round(proj_total, 1),
                    'Multiplier': round((rush_mult + rec_mult) / 2, 1),
                    'Games': round(float(player['games_played']), 1)
                })

                # Calculate defensive TD percentages for RB row in SKILL table
                def_rush_tds = int(opponent_def['rush_td_allowed'])
                def_rec_tds = int(opponent_def['rec_td_to_wr'])
                def_total_tds = def_rush_tds + def_rec_tds
                pct_rush_tds = round((def_rush_tds / def_total_tds * 100), 1) if def_total_tds > 0 else 0
                pct_rec_tds = round((def_rec_tds / def_total_tds * 100), 1) if def_total_tds > 0 else 0

                projections['SKILL'].append({
                    'Player': player['player'],
                    'Team': f"{player['team']} (RB)",
                    'Opponent': opponent,
                    'RB Score': round(rb_talent_score, 1),
                    'RB Def Score': round(rb_def_score, 1),
                    'Matchup Score': round(matchup_score, 1),
                    'Tier': tier,
                    'Rush Yds/Gm': round(player['avg_rush_yds'], 1),
                    'Projected Rush Yds': round(projected_rush_yds, 1),
                    'Rec Yds/Gm': round(player.get('total_rec_yds', 0) / player['games_played'], 1) if player['games_played'] > 0 else 0,
                    'Rush TDs/Gm': round(rb_rush_tds_per_game, 2),
                    'Rec TDs/Gm': round(rb_rec_tds_per_game, 2),
                    'Total TDs/Gm': round(rb_total_tds_per_game, 2),
                    'TD Probability %': td_probability,
                    'Touches/Gm': round(rb_touches_per_game, 1),
                    'Targets/Gm': round(player['avg_targets'], 1),
                    'Recommendation': recommendation,
                    # Legacy columns for backward compatibility
                    'Avg Yds/Game': round(player['avg_total_yds'], 1),
                    'Median Rush Yds': round(player['median_rush_yds'], 1),
                    'Median Rec Yds': round(player['median_rec_yds'], 1),
                    'Median Yds': round(player['median_rush_yds'] + player['median_rec_yds'], 1),
                    'Avg Air Yds': round(player.get('avg_rec_air_yds', 0), 1),
                    'Avg YAC': round(player.get('avg_rec_yac', 0), 1),
                    'Rush TDs': int(player['total_rush_td']),
                    'Rec TDs': int(player['total_rec_td']),
                    'Def Total TDs': def_total_tds,
                    'Def Rush TDs': def_rush_tds,
                    'Def Rec TDs': def_rec_tds,
                    '% Rush TDs': pct_rush_tds,
                    '% Rec TDs': pct_rec_tds,
                    'Def Total Rank': def_total_ranking.get(opponent, 16),
                    'Projected Yds': round(proj_total, 1),
                    'Multiplier': round(avg_mult, 1),
                    'Games': round(float(player['games_played']), 1)
                })

            elif position == 'WR':
                # Calculate WR matchup score components
                wr_rec_yds_per_game = player['avg_rec_yds']
                wr_rec_tds_total = player['total_rec_td']
                wr_targets_per_game = player.get('avg_targets', player['median_targets'])
                wr_games = player['games_played']
                def_rec_yds_allowed = opponent_def['rec_to_wr']
                def_rec_tds_allowed = opponent_def['rec_td_to_wr']
                def_rec_rank = def_rec_ranking.get(opponent, 16)

                # Calculate target share % (approximate based on typical team targets per game ~35)
                # More accurate calculation would require team total targets from database
                wr_total_targets = player['total_targets']
                typical_team_targets_per_game = 35  # NFL average
                wr_target_share_pct = (wr_targets_per_game / typical_team_targets_per_game * 100) if typical_team_targets_per_game > 0 else 0

                # Calculate receptions per game for catch rate
                wr_receptions_per_game = player['total_receptions'] / wr_games if wr_games > 0 else 0
                wr_avg_yac = player.get('avg_rec_yac', 0)

                # Calculate separate WR talent score and defensive matchup score
                wr_talent_score = calculate_wr_talent_score(
                    wr_rec_yds_per_game, wr_rec_tds_total, wr_targets_per_game,
                    wr_target_share_pct, wr_games, wr_receptions_per_game, wr_avg_yac
                )

                wr_def_score = calculate_wr_defensive_matchup_score(
                    def_rec_yds_allowed, def_rec_tds_allowed,
                    def_rec_rank, league_avg['def_rec_tds']
                )

                # Calculate Matchup Score (weighted combination of talent and matchup)
                # 60% WR talent + 40% defensive matchup
                # Elite WRs produce even vs tough coverage, so weight talent more heavily
                matchup_score = (wr_talent_score * 0.6) + (wr_def_score * 0.4)

                # Keep legacy comprehensive score for backward compatibility
                wr_score = calculate_comprehensive_wr_score(
                    wr_rec_yds_per_game, wr_rec_tds_total, wr_targets_per_game,
                    wr_target_share_pct, wr_games, def_rec_yds_allowed,
                    def_rec_tds_allowed, def_rec_rank, league_avg['rec_wr'],
                    league_avg['def_rec_tds']
                )

                # Calculate per-game rates
                wr_rec_tds_per_game = wr_rec_tds_total / wr_games if wr_games > 0 else 0

                # Calculate TD Probability %
                td_probability = calculate_wr_td_probability(
                    wr_total_rec_tds=wr_rec_tds_total,
                    wr_games=wr_games,
                    def_rec_tds_allowed=def_rec_tds_allowed,
                    league_avg_rec_tds=league_avg['def_rec_tds'],
                    wr_targets_per_game=wr_targets_per_game,
                    td_std_dev=None  # TODO: Calculate from game-by-game data in future
                )

                # Generate tier and recommendation
                tier, recommendation = generate_comprehensive_wr_storyline(
                    wr_score, player['player'], wr_rec_yds_per_game, wr_rec_tds_per_game,
                    wr_targets_per_game, wr_target_share_pct, def_rec_yds_allowed,
                    def_rec_tds_allowed, def_rec_rank
                )

                # Legacy multiplier for backwards compatibility
                multiplier = opponent_def['rec_to_wr'] / league_avg['rec_wr']
                projected_yds = player['median_rec_yds'] * multiplier

                projections['WR'].append({
                    'Player': player['player'],
                    'Team': player['team'],
                    'Opponent': opponent,
                    'WR Score': round(wr_talent_score, 1),
                    'WR Def Score': round(wr_def_score, 1),
                    'Matchup Score': round(matchup_score, 1),
                    'Tier': tier,
                    'Rec Yds/Gm': round(wr_rec_yds_per_game, 1),
                    'Rec TDs/Gm': round(wr_rec_tds_per_game, 2),
                    'TD Probability %': td_probability,
                    'Targets/Gm': round(wr_targets_per_game, 1),
                    'Target Share %': round(wr_target_share_pct, 1),
                    'Avg Yds/Game': round(player['avg_rec_yds'], 1),
                    'Median Rec Yds': round(player['median_rec_yds'], 1),
                    'Median Tgts': round(player['median_targets'], 1),
                    'Total Targets': int(player['total_targets']),
                    'Total Receptions': int(player['total_receptions']),
                    'Last 3 Avg Tgts': round(player['last_3_avg_targets'], 1),
                    'Last 3 Avg Rec': round(player['last_3_avg_receptions'], 1),
                    'Rush TDs': int(player['total_rush_td']),
                    'Rec TDs': int(player['total_rec_td']),
                    'Def Rec Yds': round(opponent_def['rec_to_wr'], 1),
                    'Def Rec TDs': int(opponent_def['rec_td_to_wr']),
                    'Def Rec Rank': def_rec_ranking.get(opponent, 16),
                    'Projected Yds': round(projected_yds, 1),
                    'Multiplier': round(multiplier, 1),
                    'Games': round(float(player['games_played']), 1),
                    'Recommendation': recommendation
                })

                # Calculate defensive TD percentages for WR row in SKILL table
                def_rush_tds_wr = int(opponent_def['rush_td_allowed'])
                def_rec_tds_wr = int(opponent_def['rec_td_to_wr'])
                def_total_tds_wr = def_rush_tds_wr + def_rec_tds_wr
                pct_rush_tds_wr = round((def_rush_tds_wr / def_total_tds_wr * 100), 1) if def_total_tds_wr > 0 else 0
                pct_rec_tds_wr = round((def_rec_tds_wr / def_total_tds_wr * 100), 1) if def_total_tds_wr > 0 else 0

                projections['SKILL'].append({
                    'Player': player['player'],
                    'Team': f"{player['team']} (WR)",
                    'Opponent': opponent,
                    'WR Score': round(wr_talent_score, 1),
                    'WR Def Score': round(wr_def_score, 1),
                    'Matchup Score': round(matchup_score, 1),
                    'Tier': tier,
                    'Rec Yds/Gm': round(wr_rec_yds_per_game, 1),
                    'Rec TDs/Gm': round(wr_rec_tds_per_game, 2),
                    'TD Probability %': td_probability,
                    'Targets/Gm': round(wr_targets_per_game, 1),
                    'Target Share %': round(wr_target_share_pct, 1),
                    'Recommendation': recommendation,
                    # Legacy columns for backward compatibility
                    'Avg Yds/Game': round(player['avg_rec_yds'], 1),
                    'Median Rush Yds': 0,
                    'Median Rec Yds': round(player['median_rec_yds'], 1),
                    'Median Yds': round(player['median_rec_yds'], 1),
                    'Avg Air Yds': round(player.get('avg_rec_air_yds', 0), 1),
                    'Avg YAC': round(player.get('avg_rec_yac', 0), 1),
                    'Rush TDs': int(player['total_rush_td']),
                    'Rec TDs': int(player['total_rec_td']),
                    'Def Total TDs': def_total_tds_wr,
                    'Def Rush TDs': def_rush_tds_wr,
                    'Def Rec TDs': def_rec_tds_wr,
                    '% Rush TDs': pct_rush_tds_wr,
                    '% Rec TDs': pct_rec_tds_wr,
                    'Def Total Rank': def_total_ranking.get(opponent, 16),
                    'Projected Yds': round(projected_yds, 1),
                    'Multiplier': round(multiplier, 1),
                    'Games': round(float(player['games_played']), 1)
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
                    'Multiplier': round(multiplier, 1),
                    'Games': round(float(player['games_played']), 1)
                })

                projections['SKILL'].append({
                    'Player': player['player'],
                    'Team': f"{player['team']} (TE)",
                    'Opponent': opponent,
                    'Avg Yds/Game': round(player['avg_rec_yds'], 1),
                    'Median Yds': round(player['median_rec_yds'], 1),
                    'Projected Yds': round(projected_yds, 1),
                    'Multiplier': round(multiplier, 1),
                    'Games': round(float(player['games_played']), 1)
                })

        # Convert to DataFrames and sort
        result = {}
        for pos, data in projections.items():
            if data:
                df = pd.DataFrame(data)
                # Use position-specific scoring for QB, RB, and WR, otherwise use projected yards/total
                if pos == 'QB' and 'Matchup Score' in df.columns:
                    sort_col = 'Matchup Score'
                elif pos == 'RB' and 'Matchup Score' in df.columns:
                    sort_col = 'Matchup Score'
                elif pos == 'WR' and 'Matchup Score' in df.columns:
                    sort_col = 'Matchup Score'
                else:
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
# Air Yards / YAC Matchup Analysis Functions
# ============================================================================

def calculate_air_yac_matchup_stats(season, max_week=None):
    """
    Calculate Air Yards vs YAC matchup statistics for offense and defense.

    Returns DataFrame with columns:
    - team: Team abbreviation
    - offense_air_yards: Average air yards per game (offense)
    - offense_yac_yards: Average YAC per game (offense)
    - offense_passing_yards: Total passing yards per game
    - offense_air_share: Percentage of passing yards from air yards
    - offense_yac_share: Percentage of passing yards from YAC
    - defense_air_allowed: Average air yards allowed per game
    - defense_yac_allowed: Average YAC allowed per game
    - defense_air_share: Percentage of yards allowed via air
    - defense_yac_share: Percentage of yards allowed via YAC
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # Build week filter
        week_filter = f"AND week <= {max_week}" if max_week else ""

        # Calculate offensive air yards and YAC stats
        offense_query = f"""
        SELECT
            team,
            COUNT(DISTINCT week) as games,
            SUM(receiving_air_yards) as total_air_yards,
            SUM(receiving_yards_after_catch) as total_yac,
            SUM(receiving_yards) as total_receiving_yards
        FROM player_stats
        WHERE season = {season}
        {week_filter}
        AND (receiving_air_yards > 0 OR receiving_yards_after_catch > 0)
        GROUP BY team
        """

        offense_df = pd.read_sql_query(offense_query, conn)

        # Calculate per-game averages and shares for offense
        offense_df['offense_air_yards'] = (offense_df['total_air_yards'] / offense_df['games']).round(1)
        offense_df['offense_yac_yards'] = (offense_df['total_yac'] / offense_df['games']).round(1)
        offense_df['offense_passing_yards'] = (offense_df['total_receiving_yards'] / offense_df['games']).round(1)

        # Calculate air yards on completions only (not all targets)
        # Air Yards on Completions = Total Receiving Yards - YAC
        offense_df['air_yards_on_completions'] = (
            offense_df['total_receiving_yards'] - offense_df['total_yac']
        )

        # Calculate shares (avoid division by zero)
        # Air % = Air Yards on Completions / Total Receiving Yards * 100
        # YAC % = YAC / Total Receiving Yards * 100
        # These two percentages will always sum to 100%
        offense_df['offense_air_share'] = (
            (offense_df['air_yards_on_completions'] /
             offense_df['total_receiving_yards'].replace(0, 1)) * 100
        ).round(1)
        offense_df['offense_yac_share'] = (
            (offense_df['total_yac'] /
             offense_df['total_receiving_yards'].replace(0, 1)) * 100
        ).round(1)

        # Calculate defensive stats (opponent's offensive stats)
        defense_query = f"""
        SELECT
            opponent_team as team,
            COUNT(DISTINCT week) as games,
            SUM(receiving_air_yards) as total_air_allowed,
            SUM(receiving_yards_after_catch) as total_yac_allowed,
            SUM(receiving_yards) as total_receiving_allowed
        FROM player_stats
        WHERE season = {season}
        {week_filter}
        AND (receiving_air_yards > 0 OR receiving_yards_after_catch > 0)
        GROUP BY opponent_team
        """

        defense_df = pd.read_sql_query(defense_query, conn)

        # Calculate per-game averages and shares for defense
        defense_df['defense_air_allowed'] = (defense_df['total_air_allowed'] / defense_df['games']).round(1)
        defense_df['defense_yac_allowed'] = (defense_df['total_yac_allowed'] / defense_df['games']).round(1)
        defense_df['defense_passing_allowed'] = (defense_df['total_receiving_allowed'] / defense_df['games']).round(1)

        # Calculate air yards on completions only (not all targets)
        # Air Yards on Completions = Total Receiving Yards - YAC
        defense_df['air_yards_allowed_on_completions'] = (
            defense_df['total_receiving_allowed'] - defense_df['total_yac_allowed']
        )

        # Calculate shares (avoid division by zero)
        # Air % = Air Yards on Completions / Total Receiving Yards * 100
        # YAC % = YAC / Total Receiving Yards * 100
        # These two percentages will always sum to 100%
        defense_df['defense_air_share'] = (
            (defense_df['air_yards_allowed_on_completions'] /
             defense_df['total_receiving_allowed'].replace(0, 1)) * 100
        ).round(1)
        defense_df['defense_yac_share'] = (
            (defense_df['total_yac_allowed'] /
             defense_df['total_receiving_allowed'].replace(0, 1)) * 100
        ).round(1)

        # Merge offense and defense stats
        result_df = pd.merge(
            offense_df[['team', 'offense_air_yards', 'offense_yac_yards', 'offense_passing_yards',
                       'offense_air_share', 'offense_yac_share']],
            defense_df[['team', 'defense_air_allowed', 'defense_yac_allowed', 'defense_passing_allowed',
                       'defense_air_share', 'defense_yac_share']],
            on='team',
            how='outer'
        )

        # Fill NaN values with 0
        result_df = result_df.fillna(0)

        conn.close()
        return result_df

    except Exception as e:
        st.error(f"Error calculating Air/YAC matchup stats: {e}")
        return pd.DataFrame()


def categorize_offensive_style(air_share):
    """
    Categorize offensive passing style based on air yards percentage.

    Benchmarks:
    - Deep Ball (>60% air): Team throws deep, relies on air yards
    - Balanced (45-60% air): Mix of deep and short passes
    - Short Passing (<45% air): YAC-dependent, underneath routes
    """
    if air_share >= 60:
        return "Deep Ball"
    elif air_share >= 45:
        return "Balanced"
    else:
        return "Short Passing"


def categorize_defensive_weakness(air_share_allowed, yac_share_allowed):
    """
    Categorize defensive vulnerability based on what they allow.

    Benchmarks:
    - Air Vulnerable (>60% air allowed): Allows deep completions
    - YAC Vulnerable (>55% YAC allowed): Allows yards after catch
    - Balanced Defense: Neither extreme
    """
    if air_share_allowed >= 60:
        return "Air Vulnerable"
    elif yac_share_allowed >= 55:
        return "YAC Vulnerable"
    else:
        return "Balanced Defense"


def generate_air_yac_storyline(offense_air_share, defense_yac_share, defense_air_share, offense_yac_share):
    """
    Generate narrative storyline based on air yards vs YAC matchup with refined benchmarks.

    Offensive Style Buckets:
    - Deep Ball: >60% air share (vertical passing attack)
    - Balanced: 45-60% air share (mix of deep and short)
    - Short Passing: <45% air share (YAC-dependent, underneath routes)

    Defensive Weakness Buckets:
    - Air Vulnerable: >60% air allowed (gives up deep balls)
    - YAC Vulnerable: >55% YAC allowed (gives up yards after catch)
    - Balanced Defense: Neither extreme

    Favorable Matchups (Offense exploits defensive weakness):
    - Deep Ball offense vs YAC Vulnerable defense
    - Short Passing offense vs Air Vulnerable defense
    - Deep Ball offense vs Air Vulnerable defense (most explosive)

    Unfavorable Matchups (Mismatch):
    - Deep Ball offense vs defense that limits air yards (<45% air allowed)
    - Short Passing offense vs defense that limits YAC (<45% YAC allowed)
    """

    # Categorize offense and defense
    off_style = categorize_offensive_style(offense_air_share)
    def_weakness = categorize_defensive_weakness(defense_air_share, defense_yac_share)

    # FAVORABLE MATCHUPS (Offense exploits defensive weakness)
    if off_style == "Deep Ball" and defense_yac_share >= 55:
        return "🎯 EXPLOIT: Deep Ball vs YAC-Leaky", f"Vertical attack ({offense_air_share:.0f}% air) faces defense allowing YAC ({defense_yac_share:.0f}%) - big play potential"

    elif off_style == "Deep Ball" and defense_air_share >= 60:
        return "🔥 EXPLOIT: Deep Ball vs Air-Prone", f"Vertical attack ({offense_air_share:.0f}% air) vs defense weak against deep balls ({defense_air_share:.0f}% air allowed) - ELITE matchup"

    elif off_style == "Short Passing" and defense_air_share >= 60:
        return "✅ FAVORABLE: Short Game vs Air-Prone", f"YAC offense ({offense_yac_share:.0f}% YAC) vs defense allowing deep balls - favorable mismatch"

    # UNFAVORABLE MATCHUPS (Offense strength neutralized)
    elif off_style == "Deep Ball" and defense_air_share < 45:
        return "🛡️ TOUGH: Deep Ball vs Tight Coverage", f"Vertical attack ({offense_air_share:.0f}% air) vs defense limiting air yards ({defense_air_share:.0f}%) - challenging matchup"

    elif off_style == "Short Passing" and defense_yac_share < 45:
        return "🚧 TOUGH: Short Game vs YAC-Stingy", f"YAC offense ({offense_yac_share:.0f}%) vs defense preventing YAC ({defense_yac_share:.0f}%) - limited upside"

    # BALANCED/NEUTRAL MATCHUPS
    elif off_style == "Balanced" and def_weakness == "Balanced Defense":
        return "⚖️ NEUTRAL: Balanced Matchup", f"Balanced offense ({offense_air_share:.0f}% air, {offense_yac_share:.0f}% YAC) vs balanced defense"

    elif off_style == "Deep Ball":
        return f"✈️ Deep Ball Offense", f"Vertical passing attack ({offense_air_share:.0f}% air, {offense_yac_share:.0f}% YAC)"

    elif off_style == "Short Passing":
        return f"🏃 YAC-Dependent Offense", f"Underneath routes and YAC ({offense_air_share:.0f}% air, {offense_yac_share:.0f}% YAC)"

    else:
        return "⚪ Standard Matchup", f"Air: {offense_air_share:.0f}% vs Def Air Allowed: {defense_air_share:.0f}%"


# ============================================================================
# QB Pressure / Sack Matchup Analysis Functions
# ============================================================================

def calculate_qb_pressure_stats(season, max_week=None):
    """
    Calculate QB pressure statistics from PFR advanced stats.

    Returns DataFrame with columns:
    - player: QB name
    - team: Team abbreviation
    - games: Games played
    - pressures: Total pressures faced
    - sacks: Total sacks taken
    - hurries: Total hurries
    - hits: Total QB hits
    - pressure_rate: % of dropbacks with pressure
    - sacks_per_pressure: Sacks / pressures (vulnerability metric)
    - bad_throw_pct: Bad throw percentage under pressure
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # Build week filter
        week_filter = f"AND week <= {max_week}" if max_week else ""

        # Get QB pressure stats from pfr_advstats_pass_week
        query = f"""
        SELECT
            pfr_player_name as player,
            team,
            COUNT(DISTINCT week) as games,
            SUM(COALESCE(times_pressured, 0)) as pressures,
            SUM(COALESCE(times_sacked, 0)) as sacks,
            SUM(COALESCE(times_hurried, 0)) as hurries,
            SUM(COALESCE(times_hit, 0)) as hits,
            SUM(COALESCE(times_blitzed, 0)) as blitzes,
            AVG(COALESCE(times_pressured_pct, 0)) * 100 as pressure_rate,
            AVG(COALESCE(passing_bad_throw_pct, 0)) * 100 as bad_throw_pct
        FROM pfr_advstats_pass_week
        WHERE season = {season}
        {week_filter}
        GROUP BY pfr_player_name, team
        HAVING pressures > 0
        """

        df = pd.read_sql_query(query, conn)

        if not df.empty:
            # Calculate sacks per pressure (vulnerability metric)
            df['sacks_per_pressure'] = (df['sacks'] / df['pressures'].replace(0, 1) * 100).round(1)

            # Calculate per-game averages
            df['pressures_per_game'] = (df['pressures'] / df['games']).round(1)
            df['sacks_per_game'] = (df['sacks'] / df['games']).round(1)

            # Round other percentages
            df['pressure_rate'] = df['pressure_rate'].round(1)
            df['bad_throw_pct'] = df['bad_throw_pct'].round(1)

        conn.close()
        return df

    except Exception as e:
        st.error(f"Error calculating QB pressure stats: {e}")
        return pd.DataFrame()


def calculate_defense_pressure_stats(season, max_week=None):
    """
    Calculate defensive pressure statistics from PFR advanced stats.

    Returns DataFrame with columns:
    - team: Team abbreviation
    - games: Games played
    - pressures: Total pressures generated
    - sacks: Total sacks
    - hurries: Total hurries
    - qb_hits: Total QB hits
    - blitzes: Total blitzes sent
    - pressure_rate: Average pressure rate
    - sack_rate: Sacks per game
    - blitz_rate: Blitzes per game
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # Build week filter
        week_filter = f"AND week <= {max_week}" if max_week else ""

        # Aggregate defensive pressure stats by team
        query = f"""
        SELECT
            team,
            COUNT(DISTINCT week) as games,
            SUM(COALESCE(def_pressures, 0)) as pressures,
            SUM(COALESCE(def_sacks, 0)) as sacks,
            SUM(COALESCE(def_times_hurried, 0)) as hurries,
            SUM(COALESCE(def_times_hitqb, 0)) as qb_hits,
            SUM(COALESCE(def_times_blitzed, 0)) as blitzes
        FROM pfr_advstats_def_week
        WHERE season = {season}
        {week_filter}
        GROUP BY team
        """

        df = pd.read_sql_query(query, conn)

        if not df.empty:
            # Calculate per-game rates
            df['pressures_per_game'] = (df['pressures'] / df['games']).round(1)
            df['sacks_per_game'] = (df['sacks'] / df['games']).round(1)
            df['hurries_per_game'] = (df['hurries'] / df['games']).round(1)
            df['qb_hits_per_game'] = (df['qb_hits'] / df['games']).round(1)
            df['blitzes_per_game'] = (df['blitzes'] / df['games']).round(1)

            # Calculate pressure efficiency (sacks per pressure)
            df['sack_per_pressure_pct'] = (df['sacks'] / df['pressures'].replace(0, 1) * 100).round(1)

        conn.close()
        return df

    except Exception as e:
        st.error(f"Error calculating defense pressure stats: {e}")
        return pd.DataFrame()


def generate_qb_pressure_storyline(qb_pressure_rate, def_pressure_rate, sacks_per_pressure):
    """
    Generate narrative storyline based on QB pressure vulnerability vs defensive pressure strength.

    Logic:
    - High QB pressure rate (>30%) + High def pressure rate (>12/game) = DANGER ZONE
    - Low QB pressure rate (<20%) + High def pressure rate = Tough Test
    - High QB pressure rate + Low def pressure rate (<8/game) = Exploitable
    - Low QB pressure rate + Low def pressure rate = Clean Pocket
    - Otherwise = Competitive Matchup
    """

    if qb_pressure_rate > 30 and def_pressure_rate > 12:
        return "🚨 DANGER ZONE", f"QB struggles with pressure ({qb_pressure_rate:.1f}%) vs aggressive pass rush ({def_pressure_rate:.1f}/gm)"
    elif qb_pressure_rate < 20 and def_pressure_rate > 12:
        return "⚔️ Tough Test", f"Good protection ({qb_pressure_rate:.1f}%) faces strong pass rush ({def_pressure_rate:.1f}/gm)"
    elif qb_pressure_rate > 30 and def_pressure_rate < 8:
        return "🎯 Exploit able", f"QB vulnerable to pressure ({qb_pressure_rate:.1f}%) vs weak pass rush ({def_pressure_rate:.1f}/gm)"
    elif qb_pressure_rate < 20 and def_pressure_rate < 8:
        return "✅ Clean Pocket", f"Good protection ({qb_pressure_rate:.1f}%) vs weak pass rush ({def_pressure_rate:.1f}/gm) - QB should thrive"
    elif sacks_per_pressure > 30:
        return "⚠️ High Sack Risk", f"QB takes sacks on {sacks_per_pressure:.1f}% of pressures - pocket awareness concern"
    else:
        return "⚪ Competitive Battle", "Standard pressure matchup"


# ============================================================================
# Rushing TD Efficiency Matchup Analysis Functions
# ============================================================================

def calculate_rushing_td_matchup_stats(season, max_week=None):
    """
    Calculate Rushing TD matchup statistics for offense and defense.

    Returns DataFrame with columns:
    - team: Team abbreviation
    - offense_rush_tds: Rushing TDs scored per game
    - offense_rush_td_rate: TD rate (TDs per carry)
    - defense_rush_tds_allowed: Rushing TDs allowed per game
    - games: Games played
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # Build week filter
        week_filter = f"AND week <= {max_week}" if max_week else ""

        # Get offensive rushing TD stats
        offense_query = f"""
        SELECT
            team,
            COUNT(DISTINCT week) as games,
            SUM(rushing_tds) as total_rush_tds,
            SUM(carries) as total_carries
        FROM player_stats
        WHERE season = {season}
        {week_filter}
        AND position IN ('RB', 'QB', 'WR', 'TE')
        GROUP BY team
        """

        offense_df = pd.read_sql_query(offense_query, conn)

        if not offense_df.empty:
            # Calculate per-game averages and rates
            offense_df['offense_rush_tds'] = (offense_df['total_rush_tds'] / offense_df['games']).round(2)
            offense_df['offense_rush_td_rate'] = (
                (offense_df['total_rush_tds'] / offense_df['total_carries'].replace(0, 1)) * 100
            ).round(2)

        # Get defensive rushing TD stats (TDs allowed)
        defense_query = f"""
        SELECT
            opponent_team as team,
            COUNT(DISTINCT week) as games,
            SUM(rushing_tds) as total_rush_tds_allowed
        FROM player_stats
        WHERE season = {season}
        {week_filter}
        AND position IN ('RB', 'QB', 'WR', 'TE')
        GROUP BY opponent_team
        """

        defense_df = pd.read_sql_query(defense_query, conn)

        if not defense_df.empty:
            # Calculate per-game averages
            defense_df['defense_rush_tds_allowed'] = (
                defense_df['total_rush_tds_allowed'] / defense_df['games']
            ).round(2)

        # Merge offense and defense stats
        result_df = pd.merge(
            offense_df[['team', 'offense_rush_tds', 'offense_rush_td_rate']],
            defense_df[['team', 'defense_rush_tds_allowed']],
            on='team',
            how='outer'
        )

        # Fill NaN values with 0
        result_df = result_df.fillna(0)

        conn.close()
        return result_df

    except Exception as e:
        st.error(f"Error calculating Rushing TD matchup stats: {e}")
        return pd.DataFrame()


def categorize_rushing_td_offense(rush_tds_per_game):
    """
    Categorize offensive rushing TD production.

    Benchmarks (based on API rushing_tds metric):
    - Goal Line Pounder (>0.8 TDs/game): Elite TD scoring rate
    - Balanced Rusher (0.4-0.8 TDs/game): Standard touchdown production
    - Volume Back (<0.4 TDs/game): High touches but limited TD conversion
    """
    if rush_tds_per_game >= 0.8:
        return "Goal Line Pounder"
    elif rush_tds_per_game >= 0.4:
        return "Balanced Rusher"
    else:
        return "Volume Back"


def categorize_rushing_td_defense(rush_tds_allowed_per_game):
    """
    Categorize defensive rushing TD vulnerability.

    Benchmarks (based on API rushing_tds allowed):
    - TD-Prone (>1.2 TDs/game allowed): Allows high TD rate to opposing rushers
    - Average (0.7-1.2 TDs/game): League standard TD prevention
    - Stingy (<0.7 TDs/game): Elite at limiting rushing TDs
    """
    if rush_tds_allowed_per_game >= 1.2:
        return "TD-Prone"
    elif rush_tds_allowed_per_game >= 0.7:
        return "Average"
    else:
        return "Stingy"


def generate_rushing_td_storyline(offense_rush_tds, defense_rush_tds_allowed):
    """
    Generate narrative storyline based on rushing TD production vs defensive vulnerability.

    Favorable Matchups (High TD upside):
    - Goal Line Pounder vs TD-Prone defense (most explosive)
    - Volume Back vs TD-Prone defense (breakout potential)
    - Goal Line Pounder vs Average defense

    Unfavorable Matchups (Limited TD upside):
    - Goal Line Pounder vs Stingy defense (strength vs strength)
    - Volume Back vs Stingy defense (worst case)
    """

    off_category = categorize_rushing_td_offense(offense_rush_tds)
    def_category = categorize_rushing_td_defense(defense_rush_tds_allowed)

    # EXPLOIT MATCHUPS (High TD upside)
    if off_category == "Goal Line Pounder" and defense_rush_tds_allowed >= 1.2:
        return "🎯 EXPLOIT: High TD Rate vs TD-Prone", f"Elite TD scorer ({offense_rush_tds:.1f} TDs/gm) vs defense allowing {defense_rush_tds_allowed:.1f} TDs/gm - Elite TD upside"

    elif off_category == "Volume Back" and defense_rush_tds_allowed >= 1.2:
        return "🔥 SMASH SPOT: Volume vs Leaky", f"High-touch back faces porous rush defense ({defense_rush_tds_allowed:.1f} TDs allowed/gm) - Breakout TD potential"

    elif off_category == "Goal Line Pounder" and 0.7 <= defense_rush_tds_allowed < 1.2:
        return "✅ FAVORABLE: High TD Rate vs Average", f"TD-dependent back ({offense_rush_tds:.1f} TDs/gm) vs league-average defense ({defense_rush_tds_allowed:.1f} allowed/gm) - Solid TD odds"

    # TOUGH MATCHUPS (Limited TD upside)
    elif off_category == "Goal Line Pounder" and defense_rush_tds_allowed < 0.7:
        return "🛡️ TOUGH: TD-Dependent vs Stingy", f"High TD rate ({offense_rush_tds:.1f}/gm) vs stingy defense ({defense_rush_tds_allowed:.1f} allowed/gm) - Limited ceiling"

    elif off_category == "Volume Back" and defense_rush_tds_allowed < 0.7:
        return "🚧 YARDAGE ONLY: Volume vs Stingy", f"Back gets touches but defense limits TDs ({defense_rush_tds_allowed:.1f} allowed/gm) - Expect yards, not scores"

    # NEUTRAL MATCHUPS
    elif off_category == "Balanced Rusher":
        return "⚖️ BALANCED: Standard Rusher", f"Standard TD production ({offense_rush_tds:.1f}/gm) vs defense allowing {defense_rush_tds_allowed:.1f} TDs"

    else:
        return "⚪ STANDARD: Neutral Matchup", f"Rush TDs: {offense_rush_tds:.1f}/gm vs Def Allows: {defense_rush_tds_allowed:.1f}/gm"


# ============================================================================
# RB Rushing Yards Matchup Analysis Functions
# ============================================================================

def calculate_rushing_yards_matchup_stats(season, max_week=None):
    """
    Calculate RB Rushing Yards matchup statistics for offense and defense.

    Returns DataFrame with columns:
    - team: Team abbreviation
    - offense_rush_yards: Rushing yards per game
    - defense_rush_yards_allowed: Rushing yards allowed per game
    - games: Games played
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        week_filter = f"AND week <= {max_week}" if max_week else ""

        # Get offensive rushing yards stats
        offense_query = f"""
        SELECT
            team,
            COUNT(DISTINCT week) as games,
            SUM(rushing_yards) as total_rush_yards
        FROM player_stats
        WHERE season = {season}
        {week_filter}
        AND position IN ('RB', 'FB')
        GROUP BY team
        """

        offense_df = pd.read_sql_query(offense_query, conn)

        if not offense_df.empty:
            offense_df['offense_rush_yards'] = (offense_df['total_rush_yards'] / offense_df['games']).round(1)

        # Get defensive rushing yards stats (yards allowed)
        defense_query = f"""
        SELECT
            opponent_team as team,
            COUNT(DISTINCT week) as games,
            SUM(rushing_yards) as total_rush_yards_allowed
        FROM player_stats
        WHERE season = {season}
        {week_filter}
        AND position IN ('RB', 'FB')
        GROUP BY opponent_team
        """

        defense_df = pd.read_sql_query(defense_query, conn)

        if not defense_df.empty:
            defense_df['defense_rush_yards_allowed'] = (
                defense_df['total_rush_yards_allowed'] / defense_df['games']
            ).round(1)

        # Merge offense and defense stats
        result_df = pd.merge(
            offense_df[['team', 'offense_rush_yards']],
            defense_df[['team', 'defense_rush_yards_allowed']],
            on='team',
            how='outer'
        )

        result_df = result_df.fillna(0)
        conn.close()
        return result_df

    except Exception as e:
        st.error(f"Error calculating Rushing Yards matchup stats: {e}")
        return pd.DataFrame()


def categorize_rushing_yards_offense(rush_yards_per_game):
    """
    Categorize offensive rushing yards production.

    Benchmarks (based on API rushing_yards metric):
    - Elite Ground Game (≥140 yards/game): High-volume rushing attack
    - Balanced Run Game (100-140 yards/game): Standard rushing volume
    - Pass-Heavy Offense (<100 yards/game): Conservative rushing approach
    """
    if rush_yards_per_game >= 140:
        return "Elite Ground Game"
    elif rush_yards_per_game >= 100:
        return "Balanced Run Game"
    else:
        return "Pass-Heavy Offense"


def categorize_rushing_yards_defense(rush_yards_allowed_per_game):
    """
    Categorize defensive rushing yards vulnerability.

    Benchmarks (based on API rushing_yards allowed):
    - Run Funnel Defense (≥130 yards/game allowed): Allows high rushing volume
    - Average Run Defense (100-130 yards/game): League standard run defense
    - Elite Run Defense (<100 yards/game): Dominant at limiting rushing yards
    """
    if rush_yards_allowed_per_game >= 130:
        return "Run Funnel Defense"
    elif rush_yards_allowed_per_game >= 100:
        return "Average Run Defense"
    else:
        return "Elite Run Defense"


def generate_rushing_yards_storyline(offense_rush_yards, defense_rush_yards_allowed):
    """
    Generate narrative storyline based on rushing yards production vs defensive vulnerability.

    Favorable Matchups (High volume upside):
    - Elite Ground Game vs Run Funnel (most explosive)
    - Pass-Heavy vs Run Funnel (game script opportunity)
    - Elite Ground Game vs Average Run Defense

    Unfavorable Matchups (Limited volume):
    - Elite Ground Game vs Elite Run Defense (grind it out)
    - Pass-Heavy vs Elite Run Defense (lowest volume)
    """

    off_category = categorize_rushing_yards_offense(offense_rush_yards)
    def_category = categorize_rushing_yards_defense(defense_rush_yards_allowed)

    # EXPLOIT MATCHUPS (High volume upside)
    if off_category == "Elite Ground Game" and defense_rush_yards_allowed >= 130:
        return "🚀 CEILING EXPLOSION", f"Elite rushing attack ({offense_rush_yards:.0f} yds/gm) vs run funnel defense ({defense_rush_yards_allowed:.0f} allowed/gm) - Massive yardage upside"

    elif off_category == "Pass-Heavy Offense" and defense_rush_yards_allowed >= 130:
        return "📈 VOLUME SPIKE", f"Low-volume rush attack ({offense_rush_yards:.0f} yds/gm) vs run funnel ({defense_rush_yards_allowed:.0f} allowed/gm) - Game script opportunity"

    elif off_category == "Elite Ground Game" and 100 <= defense_rush_yards_allowed < 130:
        return "✅ FAVORABLE", f"Strong rushing attack ({offense_rush_yards:.0f} yds/gm) vs average run defense ({defense_rush_yards_allowed:.0f} allowed/gm) - Good yardage floor"

    # TOUGH MATCHUPS (Limited volume)
    elif off_category == "Elite Ground Game" and defense_rush_yards_allowed < 100:
        return "💪 GRIND IT OUT", f"Elite rush attack ({offense_rush_yards:.0f} yds/gm) faces elite run defense ({defense_rush_yards_allowed:.0f} allowed/gm) - Limited ceiling"

    elif off_category == "Pass-Heavy Offense" and defense_rush_yards_allowed < 100:
        return "⚠️ LOW VOLUME", f"Weak rushing attack ({offense_rush_yards:.0f} yds/gm) vs elite run defense ({defense_rush_yards_allowed:.0f} allowed/gm) - Avoid in yardage formats"

    # NEUTRAL MATCHUPS
    elif off_category == "Balanced Run Game":
        return "⚖️ BALANCED", f"Balanced rushing attack ({offense_rush_yards:.0f} yds/gm) vs defense allowing {defense_rush_yards_allowed:.0f} yards/gm"

    else:
        return "⚪ STANDARD", f"Rush Yards: {offense_rush_yards:.0f}/gm vs Def Allows: {defense_rush_yards_allowed:.0f}/gm"


# ============================================================================
# Player Rush TD vs Defense Analysis Functions
# ============================================================================

def calculate_player_rush_td_vs_defense_stats(season, max_week=None):
    """
    Calculate individual player Rush TDs vs defensive Rush TDs allowed.

    Returns DataFrame with columns:
    - team: Team abbreviation
    - player: Player name
    - position: Player position
    - player_rush_tds: Player's total rushing TDs
    - player_games: Games played by player
    - player_rush_tds_per_game: Player's rushing TDs per game
    - defense_rush_tds_allowed: Defensive rushing TDs allowed per game
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # Build week filter
        week_filter = f"AND week <= {max_week}" if max_week else ""

        # Get player rushing TD stats (RBs and QBs primarily)
        player_query = f"""
        SELECT
            team,
            player_display_name as player,
            position,
            COUNT(DISTINCT week) as player_games,
            SUM(rushing_tds) as player_rush_tds,
            SUM(carries) as total_carries
        FROM player_stats
        WHERE season = {season}
        {week_filter}
        AND position IN ('RB', 'QB', 'WR', 'TE')
        AND carries > 0
        GROUP BY team, player_display_name, position
        HAVING SUM(rushing_tds) > 0
        """

        player_df = pd.read_sql_query(player_query, conn)

        if not player_df.empty:
            # Calculate per-game averages
            player_df['player_rush_tds_per_game'] = (
                player_df['player_rush_tds'] / player_df['player_games']
            ).round(2)

        # Get defensive rushing TD stats (TDs allowed per game)
        defense_query = f"""
        SELECT
            opponent_team as team,
            COUNT(DISTINCT week) as games,
            SUM(rushing_tds) as total_rush_tds_allowed
        FROM player_stats
        WHERE season = {season}
        {week_filter}
        AND position IN ('RB', 'QB', 'WR', 'TE')
        GROUP BY opponent_team
        """

        defense_df = pd.read_sql_query(defense_query, conn)

        if not defense_df.empty:
            # Calculate per-game averages
            defense_df['defense_rush_tds_allowed'] = (
                defense_df['total_rush_tds_allowed'] / defense_df['games']
            ).round(2)

        # Merge player stats with team defensive stats
        result_df = pd.merge(
            player_df[['team', 'player', 'position', 'player_rush_tds', 'player_games', 'player_rush_tds_per_game']],
            defense_df[['team', 'defense_rush_tds_allowed']],
            on='team',
            how='left'
        )

        # Fill NaN values with league average
        if not result_df.empty:
            league_avg_def = defense_df['defense_rush_tds_allowed'].mean()
            result_df['defense_rush_tds_allowed'] = result_df['defense_rush_tds_allowed'].fillna(league_avg_def)

        conn.close()
        return result_df

    except Exception as e:
        st.error(f"Error calculating Player Rush TD vs Defense stats: {e}")
        return pd.DataFrame()


def categorize_player_rush_tds(rush_tds_per_game):
    """Categorize player rushing TD production."""
    if rush_tds_per_game >= 0.75:
        return "Elite TD Scorer"
    elif rush_tds_per_game >= 0.5:
        return "Strong TD Producer"
    elif rush_tds_per_game >= 0.25:
        return "Moderate TD Threat"
    else:
        return "Limited TD Production"


def categorize_defense_rush_tds(rush_tds_allowed_per_game):
    """Categorize defensive rushing TD prevention."""
    if rush_tds_allowed_per_game >= 1.2:
        return "TD-Prone Defense"
    elif rush_tds_allowed_per_game >= 0.8:
        return "Average TD Defense"
    else:
        return "Stingy TD Defense"


def generate_player_rush_td_storyline(player_rush_tds_per_game, defense_rush_tds_allowed, player_name, position):
    """
    Generate narrative storyline for player rush TD matchup vs defense.

    Best Matchups (High TD probability):
    - Elite/Strong TD scorer vs TD-Prone Defense
    - Elite TD scorer vs Average Defense

    Avoid Matchups (Low TD probability):
    - Limited TD production vs Stingy Defense
    - Moderate production vs Stingy Defense
    """

    player_category = categorize_player_rush_tds(player_rush_tds_per_game)
    def_category = categorize_defense_rush_tds(defense_rush_tds_allowed)

    # SMASH MATCHUPS (Highest TD probability)
    if player_rush_tds_per_game >= 0.5 and defense_rush_tds_allowed >= 1.2:
        return "🎯 TD SMASH SPOT", f"{player_name} ({position}) scores {player_rush_tds_per_game:.2f} rush TDs/gm vs defense allowing {defense_rush_tds_allowed:.2f} TDs/gm - Elite TD opportunity"

    elif player_rush_tds_per_game >= 0.75 and defense_rush_tds_allowed >= 0.8:
        return "✅ GREAT MATCHUP", f"Elite TD scorer {player_name} ({player_rush_tds_per_game:.2f} TDs/gm) vs generous defense ({defense_rush_tds_allowed:.2f} allowed/gm)"

    elif player_rush_tds_per_game >= 0.5 and defense_rush_tds_allowed >= 0.8:
        return "💎 FAVORABLE", f"{player_name} ({player_rush_tds_per_game:.2f} TDs/gm) has solid TD upside vs defense allowing {defense_rush_tds_allowed:.2f} TDs/gm"

    # TOUGH MATCHUPS (Lower TD probability)
    elif player_rush_tds_per_game < 0.25 and defense_rush_tds_allowed < 0.8:
        return "🛑 AVOID", f"{player_name} has limited TD production ({player_rush_tds_per_game:.2f}/gm) vs stingy defense ({defense_rush_tds_allowed:.2f} allowed/gm)"

    elif player_rush_tds_per_game < 0.5 and defense_rush_tds_allowed < 0.8:
        return "⚠️ TOUGH MATCHUP", f"{player_name} ({player_rush_tds_per_game:.2f} TDs/gm) faces tough TD defense ({defense_rush_tds_allowed:.2f} allowed/gm)"

    elif player_rush_tds_per_game >= 0.75 and defense_rush_tds_allowed < 0.8:
        return "⚔️ CHALLENGE", f"Elite scorer {player_name} ({player_rush_tds_per_game:.2f} TDs/gm) vs strong TD defense ({defense_rush_tds_allowed:.2f} allowed/gm)"

    # NEUTRAL MATCHUPS
    else:
        return "⚖️ BALANCED", f"{player_name} ({player_rush_tds_per_game:.2f} TDs/gm) vs defense allowing {defense_rush_tds_allowed:.2f} TDs/gm"


# ============================================================================
# Comprehensive QB Matchup Scoring System
# ============================================================================

def calculate_comprehensive_qb_score(qb_yards_per_game, qb_tds_total, qb_ints_total, qb_attempts_total,
                                     qb_rush_tds_total, qb_games, def_pass_allowed, def_pass_tds_allowed,
                                     def_ints, def_sacks, def_hurries, def_blitzes,
                                     league_avg_pass_yds, league_avg_pass_tds, league_avg_def_ints, league_avg_def_sacks):
    """
    Calculate comprehensive QB matchup score (0-100 scale) based on multiple factors.

    Scoring Components:
    - Passing Yards Production: 20 points (top 5 = 20pts, top 12 = 15pts, top 20 = 10pts, rest scaled)
    - Passing TD Production: 20 points (top 5 = 20pts, top 12 = 15pts, top 20 = 10pts, rest scaled)
    - INT Avoidance: 15 points (< 1.5% = 15pts, 1.5-2.5% = 10pts, > 2.5% = 5pts)
    - Rushing TD Bonus: 10 points (10+ rush TDs = 10pts, 5-9 = 7pts, 1-4 = 3pts, 0 = 0pts)
    - Defensive Pass Yards Allowed: 15 points (≥ 250 = 15pts, ≥ 240 = 12pts, ≥ 230 = 8pts, < 220 = 3pts)
    - Defensive Pressure: 10 points (low sacks+hurries = 10pts, scaled down for high pressure)
    - Defensive Turnover Creation: 10 points (low INTs = 10pts, scaled down for ball hawks)

    Args:
        qb_yards_per_game: QB's avg passing yards per game
        qb_tds_total: QB's total passing TDs
        qb_ints_total: QB's total interceptions
        qb_attempts_total: QB's total pass attempts
        qb_rush_tds_total: QB's total rushing TDs (bonus)
        qb_games: Number of games played
        def_pass_allowed: Opponent's avg pass yards allowed per game
        def_pass_tds_allowed: Opponent's total pass TDs allowed
        def_ints: Opponent's total interceptions forced
        def_sacks: Opponent's total sacks
        def_hurries: Opponent's total QB hurries
        def_blitzes: Opponent's total blitzes
        league_avg_pass_yds: League average passing yards per game
        league_avg_pass_tds: League average passing TDs total
        league_avg_def_ints: League average defensive INTs
        league_avg_def_sacks: League average defensive sacks

    Returns:
        float: Score from 0-100
    """
    score = 0

    # 1. PASSING YARDS PRODUCTION (20 points)
    if qb_yards_per_game >= 275:  # Elite (top 5)
        score += 20
    elif qb_yards_per_game >= 250:  # Strong (top 12)
        score += 15
    elif qb_yards_per_game >= 225:  # Average (top 20)
        score += 10
    elif qb_yards_per_game >= 200:  # Below average
        score += 5
    # else: 0 points

    # 2. PASSING TD PRODUCTION (20 points)
    tds_per_game = qb_tds_total / qb_games if qb_games > 0 else 0
    if tds_per_game >= 2.0:  # Elite (top 5)
        score += 20
    elif tds_per_game >= 1.5:  # Strong (top 12)
        score += 15
    elif tds_per_game >= 1.0:  # Average (top 20)
        score += 10
    elif tds_per_game >= 0.5:  # Below average
        score += 5
    # else: 0 points

    # 3. INT AVOIDANCE (15 points) - Lower is better
    int_rate = (qb_ints_total / qb_attempts_total * 100) if qb_attempts_total > 0 else 3.0
    if int_rate <= 1.5:  # Safe
        score += 15
    elif int_rate <= 2.5:  # Average
        score += 10
    else:  # Risky (> 2.5%)
        score += 5

    # 4. RUSHING TD BONUS (10 points)
    if qb_rush_tds_total >= 10:  # Elite dual-threat (Hurts, Allen, Daniels)
        score += 10
    elif qb_rush_tds_total >= 5:  # Strong dual-threat
        score += 7
    elif qb_rush_tds_total >= 1:  # Occasional rusher
        score += 3
    # else: 0 points

    # 5. DEFENSIVE PASS YARDS ALLOWED (15 points) - Higher is better for QB
    if def_pass_allowed >= 250:  # Generous (bottom 10 defense)
        score += 15
    elif def_pass_allowed >= 240:  # Favorable (bottom 20)
        score += 12
    elif def_pass_allowed >= 230:  # Slightly above average
        score += 8
    elif def_pass_allowed >= 220:  # Average
        score += 5
    else:  # Stingy (< 220)
        score += 3

    # 6. DEFENSIVE PRESSURE (10 points) - Lower pressure is better for QB
    pressure_score = def_sacks + (def_hurries * 0.5) + (def_blitzes * 0.3)
    if pressure_score < 20:  # Low pressure defense
        score += 10
    elif pressure_score < 30:  # Average pressure
        score += 7
    elif pressure_score < 40:  # High pressure
        score += 4
    else:  # Very high pressure
        score += 2

    # 7. DEFENSIVE TURNOVER CREATION (10 points) - Lower INTs is better for QB
    if def_ints < league_avg_def_ints * 0.8:  # Passive secondary
        score += 10
    elif def_ints < league_avg_def_ints * 1.0:  # Average
        score += 7
    elif def_ints < league_avg_def_ints * 1.2:  # Ball hawks
        score += 4
    else:  # Elite INT defense
        score += 2

    return round(score, 1)


def categorize_qb_passing_yards(yards_per_game):
    """Categorize QB by passing yards per game."""
    if yards_per_game >= 275:
        return "🔥 Elite", "elite"
    elif yards_per_game >= 250:
        return "✅ Strong", "strong"
    elif yards_per_game >= 225:
        return "⚖️ Average", "average"
    else:
        return "⚠️ Limited", "limited"


def categorize_qb_passing_tds(tds_per_game):
    """Categorize QB by passing TDs per game."""
    if tds_per_game >= 2.0:
        return "🔥 Elite", "elite"
    elif tds_per_game >= 1.5:
        return "✅ Strong", "strong"
    elif tds_per_game >= 1.0:
        return "⚖️ Average", "average"
    else:
        return "⚠️ Limited", "limited"


def categorize_qb_int_rate(int_rate):
    """Categorize QB by interception rate (lower is better)."""
    if int_rate <= 1.5:
        return "✅ Safe", "safe"
    elif int_rate <= 2.5:
        return "⚖️ Average", "average"
    else:
        return "⚠️ Risky", "risky"


def categorize_defense_pass_allowed(pass_allowed):
    """Categorize defense by passing yards allowed (higher is better for QB)."""
    if pass_allowed >= 250:
        return "🎯 Generous", "generous"
    elif pass_allowed >= 240:
        return "✅ Favorable", "favorable"
    elif pass_allowed >= 230:
        return "⚖️ Average", "average"
    else:
        return "🛡️ Stingy", "stingy"


def categorize_defense_pressure(sacks, hurries, blitzes):
    """Categorize defense by pressure (lower is better for QB)."""
    pressure_score = sacks + (hurries * 0.5) + (blitzes * 0.3)
    if pressure_score < 20:
        return "✅ Low Pressure", "low"
    elif pressure_score < 30:
        return "⚖️ Average Pressure", "average"
    else:
        return "⚠️ High Pressure", "high"


def generate_comprehensive_qb_storyline(qb_score, qb_name, qb_yards_per_game, qb_tds_per_game,
                                       qb_int_rate, qb_rush_tds, def_pass_allowed, def_sacks):
    """
    Generate comprehensive QB matchup storyline based on score.

    7 Tier System:
    - 85-100: 🔥🔥🔥 ELITE SMASH SPOT
    - 75-84:  🔥🔥 PREMIUM MATCHUP
    - 65-74:  🔥 SMASH SPOT
    - 55-64:  ✅ SOLID START
    - 45-54:  ⚖️ BALANCED
    - 35-44:  ⚠️ RISKY PLAY
    - 0-34:   🛑 AVOID
    """
    # Build description components
    yards_desc = f"{qb_yards_per_game:.1f} yds/gm"
    tds_desc = f"{qb_tds_per_game:.2f} TDs/gm"
    int_desc = f"{qb_int_rate:.1f}% INT"
    rush_td_desc = f", {qb_rush_tds} rush TDs" if qb_rush_tds > 0 else ""
    def_desc = f"vs {def_pass_allowed:.1f} yds allowed, {def_sacks} sacks"

    # Determine tier and recommendation
    if qb_score >= 85:
        tier = "🔥🔥🔥 ELITE SMASH SPOT"
        recommendation = f"{qb_name} is a MUST-START QB1. Elite production ({yards_desc}, {tds_desc}{rush_td_desc}) against exploitable defense ({def_desc}). Expect ceiling performance."
    elif qb_score >= 75:
        tier = "🔥🔥 PREMIUM MATCHUP"
        recommendation = f"{qb_name} is a premium QB1 play. Strong production ({yards_desc}, {tds_desc}{rush_td_desc}) with favorable matchup ({def_desc}). High floor and ceiling."
    elif qb_score >= 65:
        tier = "🔥 SMASH SPOT"
        recommendation = f"{qb_name} is a strong start. Solid production ({yards_desc}, {tds_desc}{rush_td_desc}) in advantageous spot ({def_desc}). Good upside play."
    elif qb_score >= 55:
        tier = "✅ SOLID START"
        recommendation = f"{qb_name} is a safe QB1/QB2. Reliable production ({yards_desc}, {tds_desc}{rush_td_desc}) vs {def_desc}. Good floor with TD upside."
    elif qb_score >= 45:
        tier = "⚖️ BALANCED"
        recommendation = f"{qb_name} is a neutral matchup ({yards_desc}, {tds_desc}{rush_td_desc}) vs {def_desc}. Start based on roster needs and other options."
    elif qb_score >= 35:
        tier = "⚠️ RISKY PLAY"
        recommendation = f"{qb_name} has risk factors. Limited production ({yards_desc}, {tds_desc}, {int_desc}{rush_td_desc}) and/or tough matchup ({def_desc}). Boom-bust QB2."
    else:  # < 35
        tier = "🛑 AVOID"
        recommendation = f"{qb_name} is a fade candidate. Poor production ({yards_desc}, {tds_desc}, {int_desc}{rush_td_desc}) vs difficult matchup ({def_desc}). Bench if possible."

    return tier, recommendation


def calculate_qb_defensive_composite_score(def_pass_yards_allowed, def_pass_tds_allowed,
                                           def_qb_rush_yards_allowed, def_qb_rush_tds_allowed,
                                           def_blitzes, def_sacks,
                                           league_avg_pass_yards, league_avg_pass_tds,
                                           league_avg_qb_rush_yards, league_avg_qb_rush_tds,
                                           league_avg_blitzes, league_avg_sacks):
    """
    Calculate QB Defensive Composite Score (0-100 scale).

    **Higher score = WORSE defense (better matchup for QB)**
    **Lower score = BETTER defense (worse matchup for QB)**

    This is different from the multiplier - it's an absolute 0-100 rating where:
    - 85-100: Elite matchup (defense is terrible at stopping QBs)
    - 70-84: Great matchup (defense allows significant QB production)
    - 55-69: Good matchup (defense is below average vs QBs)
    - 45-54: Average matchup (league average defense)
    - 30-44: Tough matchup (defense is above average vs QBs)
    - 15-29: Very tough matchup (strong defense)
    - 0-14: Elite defense (nightmare matchup for QBs)

    Scoring Components (weighted):
    - Pass Yards Allowed: 30 points (primary yardage factor)
    - Pass TDs Allowed: 25 points (scoring opportunities)
    - QB Rush Yards Allowed: 20 points (dual-threat opportunity)
    - QB Rush TDs Allowed: 10 points (rushing TD bonus)
    - Blitzes (inverted): 10 points (more blitzes = lower score for defense)
    - Sacks (inverted): 5 points (more sacks = lower score for defense)

    Total: 100 points
    """
    score = 0

    # 1. PASS YARDS ALLOWED (30 points) - Higher yards allowed = better for QB
    if def_pass_yards_allowed >= league_avg_pass_yards * 1.15:  # 15%+ above average
        score += 30  # Elite pass funnel
    elif def_pass_yards_allowed >= league_avg_pass_yards * 1.08:  # 8-15% above average
        score += 24  # Great pass funnel
    elif def_pass_yards_allowed >= league_avg_pass_yards * 1.02:  # 2-8% above average
        score += 18  # Above average
    elif def_pass_yards_allowed >= league_avg_pass_yards * 0.98:  # Within 2% of average
        score += 15  # Average
    elif def_pass_yards_allowed >= league_avg_pass_yards * 0.92:  # 2-8% below average
        score += 10  # Below average
    elif def_pass_yards_allowed >= league_avg_pass_yards * 0.85:  # 8-15% below average
        score += 5   # Stingy
    else:  # 15%+ below average
        score += 0   # Elite pass defense

    # 2. PASS TDS ALLOWED (25 points) - More TDs allowed = better for QB
    pass_td_ratio = def_pass_tds_allowed / league_avg_pass_tds if league_avg_pass_tds > 0 else 1.0
    if pass_td_ratio >= 1.3:  # 30%+ more TDs allowed
        score += 25  # TD-prone defense
    elif pass_td_ratio >= 1.15:  # 15-30% more TDs
        score += 20
    elif pass_td_ratio >= 1.05:  # 5-15% more TDs
        score += 15
    elif pass_td_ratio >= 0.95:  # Within 5% of average
        score += 12
    elif pass_td_ratio >= 0.85:  # 5-15% fewer TDs
        score += 8
    elif pass_td_ratio >= 0.7:  # 15-30% fewer TDs
        score += 4
    else:  # 30%+ fewer TDs allowed
        score += 0  # TD stingy

    # 3. QB RUSH YARDS ALLOWED (20 points) - More rush yards allowed = better for dual-threat QBs
    qb_rush_ratio = def_qb_rush_yards_allowed / league_avg_qb_rush_yards if league_avg_qb_rush_yards > 0 else 1.0
    if qb_rush_ratio >= 1.5:  # 50%+ more QB rush yards allowed
        score += 20  # Can't contain mobile QBs
    elif qb_rush_ratio >= 1.25:  # 25-50% more
        score += 16
    elif qb_rush_ratio >= 1.1:  # 10-25% more
        score += 12
    elif qb_rush_ratio >= 0.9:  # Within 10% of average
        score += 10
    elif qb_rush_ratio >= 0.75:  # 10-25% fewer
        score += 6
    elif qb_rush_ratio >= 0.5:  # 25-50% fewer
        score += 3
    else:  # 50%+ fewer QB rush yards
        score += 0  # Excellent QB rush containment

    # 4. QB RUSH TDS ALLOWED (10 points) - More rush TDs allowed = better for QBs
    qb_rush_td_ratio = def_qb_rush_tds_allowed / league_avg_qb_rush_tds if league_avg_qb_rush_tds > 0 else 1.0
    if qb_rush_td_ratio >= 2.0:  # 2x+ more rush TDs to QBs
        score += 10
    elif qb_rush_td_ratio >= 1.5:  # 1.5-2x more
        score += 8
    elif qb_rush_td_ratio >= 1.2:  # 20-50% more
        score += 6
    elif qb_rush_td_ratio >= 0.8:  # Within 20%
        score += 5
    elif qb_rush_td_ratio >= 0.5:  # 20-50% fewer
        score += 3
    else:  # 50%+ fewer
        score += 0

    # 5. BLITZES (10 points - INVERTED) - Fewer blitzes = better for QB (less pressure)
    blitz_ratio = def_blitzes / league_avg_blitzes if league_avg_blitzes > 0 else 1.0
    if blitz_ratio <= 0.7:  # 30%+ fewer blitzes (passive defense)
        score += 10  # Great for QB - less aggression
    elif blitz_ratio <= 0.85:  # 15-30% fewer
        score += 8
    elif blitz_ratio <= 0.95:  # 5-15% fewer
        score += 6
    elif blitz_ratio <= 1.05:  # Within 5%
        score += 5
    elif blitz_ratio <= 1.15:  # 5-15% more blitzes
        score += 3
    elif blitz_ratio <= 1.3:  # 15-30% more
        score += 1
    else:  # 30%+ more blitzes (very aggressive)
        score += 0  # Tough for QB

    # 6. SACKS (5 points - INVERTED) - Fewer sacks = better for QB
    sack_ratio = def_sacks / league_avg_sacks if league_avg_sacks > 0 else 1.0
    if sack_ratio <= 0.7:  # 30%+ fewer sacks
        score += 5  # Weak pass rush
    elif sack_ratio <= 0.85:  # 15-30% fewer
        score += 4
    elif sack_ratio <= 0.95:  # 5-15% fewer
        score += 3
    elif sack_ratio <= 1.05:  # Within 5%
        score += 2.5
    elif sack_ratio <= 1.15:  # 5-15% more sacks
        score += 1.5
    elif sack_ratio <= 1.3:  # 15-30% more
        score += 0.5
    else:  # 30%+ more sacks
        score += 0  # Elite pass rush

    return round(score, 1)


# ============================================================================
# COMPREHENSIVE RB MATCHUP SCORING SYSTEM
# ============================================================================

def calculate_comprehensive_rb_score(rb_rush_yds_per_game, rb_rush_tds_total, rb_rec_tds_total,
                                     rb_targets_per_game, rb_games, def_rush_allowed,
                                     def_rush_tds_allowed, league_avg_rush_yds, league_avg_def_rush_tds):
    """
    Calculate comprehensive RB matchup score (0-100 scale) based on multiple factors.

    Scoring Components:
    - Rushing Yards Production: 25 points (balanced approach)
    - Rushing TD Production: 20 points (balanced approach)
    - Receiving Role/PPR Value: 20 points (full receiving analysis)
    - Receiving TD Bonus: 10 points (dual-threat backs)
    - Defensive Rush Defense: 15 points (rush yards allowed primary factor)
    - Defensive TD Vulnerability: 10 points (TD-prone defenses)

    Total: 100 points
    """
    score = 0

    # Calculate per-game rates
    rb_rush_tds_per_game = rb_rush_tds_total / rb_games if rb_games > 0 else 0
    rb_rec_tds_per_game = rb_rec_tds_total / rb_games if rb_games > 0 else 0
    rb_total_tds_per_game = rb_rush_tds_per_game + rb_rec_tds_per_game

    # 1. RUSHING YARDS PRODUCTION (25 points) - Balanced Yards Approach
    if rb_rush_yds_per_game >= 100:  # Elite (top 5)
        score += 25
    elif rb_rush_yds_per_game >= 80:  # Strong (top 12)
        score += 20
    elif rb_rush_yds_per_game >= 60:  # Average (top 20)
        score += 15
    elif rb_rush_yds_per_game >= 40:  # Below average
        score += 10
    else:  # Limited role
        score += 5

    # 2. RUSHING TD PRODUCTION (20 points) - Balanced TDs Approach
    if rb_rush_tds_per_game >= 0.6:  # Elite (10+ per season)
        score += 20
    elif rb_rush_tds_per_game >= 0.4:  # Strong (7+ per season)
        score += 15
    elif rb_rush_tds_per_game >= 0.2:  # Average (3+ per season)
        score += 10
    elif rb_rush_tds_per_game > 0:  # Occasional
        score += 5
    # else: 0 points for no TDs

    # 3. RECEIVING ROLE/PPR VALUE (20 points) - Full Receiving Analysis
    if rb_targets_per_game >= 5:  # Elite pass-catching back (CMC, Kamara type)
        score += 20
    elif rb_targets_per_game >= 3:  # Strong receiving role (Gibbs, Bijan type)
        score += 15
    elif rb_targets_per_game >= 2:  # Moderate receiving (Most RB1s)
        score += 10
    elif rb_targets_per_game >= 1:  # Limited receiving
        score += 5
    # else: 0 points for no targets

    # 4. RECEIVING TD BONUS (10 points) - Dual-Threat Back Bonus
    if rb_rec_tds_per_game >= 0.3:  # Elite (5+ rec TDs per season)
        score += 10
    elif rb_rec_tds_per_game >= 0.1:  # Strong (2+ rec TDs per season)
        score += 7
    elif rb_rec_tds_per_game > 0:  # Occasional
        score += 3
    # else: 0 points for no rec TDs

    # 5. DEFENSIVE RUSH DEFENSE (15 points) - Rush Yards Allowed Primary Factor
    if def_rush_allowed >= 110:  # Generous run defense (worst 5)
        score += 15
    elif def_rush_allowed >= 95:  # Favorable (bottom 12)
        score += 12
    elif def_rush_allowed >= 80:  # Average
        score += 8
    else:  # Stingy (<80 yds allowed)
        score += 4

    # 6. DEFENSIVE TD VULNERABILITY (10 points) - TD-Prone Defenses
    if def_rush_tds_allowed >= league_avg_def_rush_tds * 1.2:  # TD-prone (20%+ above avg)
        score += 10
    elif def_rush_tds_allowed >= league_avg_def_rush_tds * 0.8:  # Average
        score += 6
    else:  # Lockdown (<80% of avg)
        score += 2

    return round(score, 1)


def generate_comprehensive_rb_storyline(rb_score, rb_name, rb_rush_yds_per_game, rb_rush_tds_per_game,
                                       rb_rec_tds_per_game, rb_targets_per_game, rb_total_tds_per_game,
                                       def_rush_allowed, def_rush_tds_allowed):
    """
    Generate comprehensive RB matchup storyline based on score.

    7 Tier System (same as QBs):
    - 85-100: 🔥🔥🔥 ELITE SMASH SPOT
    - 75-84:  🔥🔥 PREMIUM MATCHUP
    - 65-74:  🔥 SMASH SPOT
    - 55-64:  ✅ SOLID START
    - 45-54:  ⚖️ BALANCED
    - 35-44:  ⚠️ RISKY PLAY
    - 0-34:   🛑 AVOID
    """
    # Build description components
    rush_yds_desc = f"{rb_rush_yds_per_game:.1f} rush yds/gm"
    rush_tds_desc = f"{rb_rush_tds_per_game:.2f} rush TDs/gm"
    rec_tds_desc = f"{rb_rec_tds_per_game:.2f} rec TDs/gm" if rb_rec_tds_per_game > 0 else ""
    targets_desc = f"{rb_targets_per_game:.1f} tgts/gm"
    total_tds_desc = f"{rb_total_tds_per_game:.2f} total TDs/gm"
    def_desc = f"vs {def_rush_allowed:.1f} rush yds allowed, {def_rush_tds_allowed:.1f} rush TDs allowed"

    # Determine tier and recommendation
    if rb_score >= 85:
        tier = "🔥🔥🔥 ELITE SMASH SPOT"
        recommendation = f"{rb_name} is a MUST-START RB1. Elite workhorse production ({rush_yds_desc}, {total_tds_desc}, {targets_desc}) against exploitable run defense ({def_desc}). Expect ceiling performance with multiple TD upside."
    elif rb_score >= 75:
        tier = "🔥🔥 PREMIUM MATCHUP"
        recommendation = f"{rb_name} is a premium RB1 play. Strong production ({rush_yds_desc}, {total_tds_desc}, {targets_desc}) with favorable matchup ({def_desc}). High floor and ceiling - start with confidence."
    elif rb_score >= 65:
        tier = "🔥 SMASH SPOT"
        recommendation = f"{rb_name} is a strong start. Solid production ({rush_yds_desc}, {rush_tds_desc}, {targets_desc}) in advantageous spot ({def_desc}). Good RB1/RB2 upside play."
    elif rb_score >= 55:
        tier = "✅ SOLID START"
        recommendation = f"{rb_name} is a safe RB2/FLEX. Reliable production ({rush_yds_desc}, {rush_tds_desc}, {targets_desc}) vs {def_desc}. Good floor with TD upside."
    elif rb_score >= 45:
        tier = "⚖️ BALANCED"
        recommendation = f"{rb_name} is a neutral matchup ({rush_yds_desc}, {rush_tds_desc}, {targets_desc}) vs {def_desc}. Start based on roster needs and other options. FLEX consideration."
    elif rb_score >= 35:
        tier = "⚠️ RISKY PLAY"
        recommendation = f"{rb_name} has risk factors. Limited production ({rush_yds_desc}, {rush_tds_desc}, {targets_desc}) and/or tough matchup ({def_desc}). Boom-bust FLEX play with low floor."
    else:  # < 35
        tier = "🛑 AVOID"
        recommendation = f"{rb_name} is a fade candidate. Poor production ({rush_yds_desc}, {rush_tds_desc}, {targets_desc}) vs difficult matchup ({def_desc}). Bench unless desperate for bye week fill-in."

    return tier, recommendation


def calculate_rb_talent_score(rb_rush_yds_per_game, rb_rush_tds_total, rb_rec_tds_total,
                               rb_targets_per_game, rb_games):
    """
    Calculate RB talent/production score (0-100) based purely on player performance.
    This isolates RB ability from defensive matchup quality.

    Scoring Components (75 points total):
    - Rushing Yards Production: 25 points
    - Rushing TD Production: 20 points
    - Receiving Role/PPR Value: 20 points
    - Receiving TD Bonus: 10 points
    """
    score = 0

    # Calculate per-game rates
    rb_rush_tds_per_game = rb_rush_tds_total / rb_games if rb_games > 0 else 0
    rb_rec_tds_per_game = rb_rec_tds_total / rb_games if rb_games > 0 else 0

    # 1. RUSHING YARDS PRODUCTION (25 points)
    if rb_rush_yds_per_game >= 100:  # Elite (top 5)
        score += 25
    elif rb_rush_yds_per_game >= 80:  # Strong (top 12)
        score += 20
    elif rb_rush_yds_per_game >= 60:  # Average (top 20)
        score += 15
    elif rb_rush_yds_per_game >= 40:  # Below average
        score += 10
    else:  # Limited role
        score += 5

    # 2. RUSHING TD PRODUCTION (20 points)
    if rb_rush_tds_per_game >= 0.6:  # Elite (10+ per season)
        score += 20
    elif rb_rush_tds_per_game >= 0.4:  # Strong (7+ per season)
        score += 15
    elif rb_rush_tds_per_game >= 0.2:  # Average (3+ per season)
        score += 10
    elif rb_rush_tds_per_game > 0:  # Occasional
        score += 5
    # else: 0 points for no TDs

    # 3. RECEIVING ROLE/PPR VALUE (20 points)
    if rb_targets_per_game >= 5:  # Elite pass-catching back
        score += 20
    elif rb_targets_per_game >= 3:  # Strong receiving role
        score += 15
    elif rb_targets_per_game >= 2:  # Moderate receiving
        score += 10
    elif rb_targets_per_game >= 1:  # Limited receiving
        score += 5
    # else: 0 points for no targets

    # 4. RECEIVING TD BONUS (10 points)
    if rb_rec_tds_per_game >= 0.3:  # Elite (5+ rec TDs per season)
        score += 10
    elif rb_rec_tds_per_game >= 0.1:  # Strong (2+ rec TDs per season)
        score += 7
    elif rb_rec_tds_per_game > 0:  # Occasional
        score += 3
    # else: 0 points for no rec TDs

    # Scale to 0-100 (current max is 75, so multiply by 100/75 = 1.333)
    scaled_score = (score / 75) * 100

    return round(scaled_score, 1)


def calculate_rb_defensive_matchup_score(def_rush_allowed, def_rush_tds_allowed, league_avg_def_rush_tds):
    """
    Calculate RB defensive matchup score (0-100) based purely on defensive weakness.
    Higher score = worse defense = better matchup for RB.

    Scoring Components (25 points total):
    - Defensive Rush Defense: 15 points (rush yards allowed)
    - Defensive TD Vulnerability: 10 points (TD-prone defenses)
    """
    score = 0

    # 1. DEFENSIVE RUSH DEFENSE (15 points)
    if def_rush_allowed >= 110:  # Generous run defense (worst 5)
        score += 15
    elif def_rush_allowed >= 95:  # Favorable (bottom 12)
        score += 12
    elif def_rush_allowed >= 80:  # Average
        score += 8
    else:  # Stingy (<80 yds allowed)
        score += 4

    # 2. DEFENSIVE TD VULNERABILITY (10 points)
    if def_rush_tds_allowed >= league_avg_def_rush_tds * 1.2:  # TD-prone (20%+ above avg)
        score += 10
    elif def_rush_tds_allowed >= league_avg_def_rush_tds * 0.8:  # Average
        score += 6
    else:  # Lockdown (<80% of avg)
        score += 2

    # Scale to 0-100 (current max is 25, so multiply by 100/25 = 4)
    scaled_score = (score / 25) * 100

    return round(scaled_score, 1)


def calculate_rb_td_probability(rb_total_tds, rb_games, def_rush_tds_allowed,
                                 def_rec_tds_to_rb, league_avg_tds, td_std_dev=None):
    """
    Calculate RB TD Probability % combining player TD production, defensive TD vulnerability,
    league averages, and TD consistency.

    Formula combines:
    1. RB Total TDs per game (rush + receiving TDs)
    2. Defense allows rush TDs (per game)
    3. Defense allows receiving TDs to RBs (per game)
    4. RB's average TDs per game (baseline production)
    5. RB's TD consistency (optional - lower std dev = more consistent)

    Returns: Probability percentage (0-100%)
    """
    # Calculate RB's average TD rate
    rb_tds_per_game = rb_total_tds / rb_games if rb_games > 0 else 0

    # Base probability from RB talent (40% weight)
    # Scale RB production relative to elite tier (1.0 TD/game = elite, 0.5 = average)
    rb_talent_factor = min(rb_tds_per_game / 1.0, 1.0) * 40  # Max 40%

    # Defensive vulnerability factor (35% weight)
    # Total defensive TDs allowed (rush + receiving to RBs)
    total_def_tds_allowed = def_rush_tds_allowed + def_rec_tds_to_rb

    # Scale defensive vulnerability relative to league average
    if league_avg_tds > 0:
        def_multiplier = total_def_tds_allowed / league_avg_tds
    else:
        def_multiplier = 1.0

    # Generous defense (1.5x+ league avg) = max 35%, stingy (0.5x avg) = 10%
    def_vulnerability_factor = min(max((def_multiplier - 0.5) / 1.0 * 35, 10), 35)

    # League baseline factor (15% weight)
    # Every RB has some baseline TD probability based on role
    league_baseline = 15

    # TD consistency bonus (10% weight) - Optional
    consistency_bonus = 0
    if td_std_dev is not None and rb_tds_per_game > 0:
        # Coefficient of Variation = std_dev / mean
        # Lower CV = more consistent
        cv = td_std_dev / rb_tds_per_game if rb_tds_per_game > 0 else 999

        if cv <= 0.5:  # Very consistent (scores TDs regularly)
            consistency_bonus = 10
        elif cv <= 0.75:  # Consistent
            consistency_bonus = 7
        elif cv <= 1.0:  # Moderate consistency
            consistency_bonus = 5
        elif cv <= 1.5:  # Inconsistent
            consistency_bonus = 3
        else:  # Very inconsistent (boom/bust)
            consistency_bonus = 1
    else:
        # If no std dev provided, use moderate baseline
        consistency_bonus = 5

    # Total TD Probability %
    td_probability = rb_talent_factor + def_vulnerability_factor + league_baseline + consistency_bonus

    # Cap at 100%
    td_probability = min(td_probability, 100)

    return round(td_probability, 1)


def calculate_wr_td_probability(wr_total_rec_tds, wr_games, def_rec_tds_allowed,
                                  league_avg_rec_tds, wr_targets_per_game=0, td_std_dev=None):
    """
    Calculate WR TD Probability % combining player TD production, defensive TD vulnerability,
    league averages, target volume, and TD consistency.

    Formula combines:
    1. WR Total receiving TDs per game
    2. Defense allows receiving TDs to WRs (per game)
    3. WR's target volume (opportunity factor)
    4. WR's average TDs per game (baseline production)
    5. WR's TD consistency (optional - lower std dev = more consistent)

    Returns: Probability percentage (0-100%)
    """
    # Calculate WR's average TD rate
    wr_tds_per_game = wr_total_rec_tds / wr_games if wr_games > 0 else 0

    # Base probability from WR talent (35% weight)
    # Scale WR production relative to elite tier (0.6 TD/game = elite WR1, 0.3 = WR2)
    # WRs score TDs less frequently than RBs, so adjust thresholds
    wr_talent_factor = min(wr_tds_per_game / 0.6, 1.0) * 35  # Max 35%

    # Defensive vulnerability factor (30% weight)
    # Scale defensive rec TDs allowed relative to league average
    if league_avg_rec_tds > 0:
        def_multiplier = def_rec_tds_allowed / league_avg_rec_tds
    else:
        def_multiplier = 1.0

    # Generous defense (1.5x+ league avg) = max 30%, stingy (0.5x avg) = 8%
    def_vulnerability_factor = min(max((def_multiplier - 0.5) / 1.0 * 30, 8), 30)

    # Target volume bonus (20% weight)
    # More targets = more TD opportunities
    # Elite WR1: 10+ targets/game, WR2: 7 targets, WR3: 4 targets
    if wr_targets_per_game >= 10:  # WR1 elite volume
        target_bonus = 20
    elif wr_targets_per_game >= 8:  # WR1 volume
        target_bonus = 16
    elif wr_targets_per_game >= 6:  # WR2 volume
        target_bonus = 12
    elif wr_targets_per_game >= 4:  # WR3 volume
        target_bonus = 8
    else:  # Limited role
        target_bonus = 4

    # League baseline factor (10% weight)
    # Every WR has some baseline TD probability based on role
    league_baseline = 10

    # TD consistency bonus (5% weight) - Optional
    consistency_bonus = 0
    if td_std_dev is not None and wr_tds_per_game > 0:
        # Coefficient of Variation = std_dev / mean
        # Lower CV = more consistent
        cv = td_std_dev / wr_tds_per_game if wr_tds_per_game > 0 else 999

        if cv <= 0.5:  # Very consistent (scores TDs regularly)
            consistency_bonus = 5
        elif cv <= 0.75:  # Consistent
            consistency_bonus = 4
        elif cv <= 1.0:  # Moderate consistency
            consistency_bonus = 3
        elif cv <= 1.5:  # Inconsistent
            consistency_bonus = 2
        else:  # Very inconsistent (boom/bust)
            consistency_bonus = 1
    else:
        # If no std dev provided, use moderate baseline
        consistency_bonus = 3

    # Total TD Probability %
    td_probability = wr_talent_factor + def_vulnerability_factor + target_bonus + league_baseline + consistency_bonus

    # Cap at 100%
    td_probability = min(td_probability, 100)

    return round(td_probability, 1)


# ============================================================================
# DEFENSIVE RUN STYLE CLASSIFICATION SYSTEM
# ============================================================================

def calculate_defensive_run_metrics(season, week=None, window_size=5):
    """
    Calculate defensive run metrics for all teams using available data.

    Metrics calculated:
    - stuff_rate_forced: % of rushes stopped at or behind LOS
    - explosive_run_pct_allowed: % of rushes ≥10 yards
    - ybc_allowed: yards before contact per rush allowed
    - yac_allowed: yards after contact per rush allowed
    - avg_rush_yards_allowed: average rush yards per attempt

    Args:
        season (int): Season to analyze
        week (int, optional): If provided, uses rolling window ending at this week
        window_size (int): Number of weeks to include in rolling window (default: 5)

    Returns:
        pd.DataFrame: Team defensive metrics with columns:
            team, stuff_rate, explosive_run_pct, ybc_allowed, yac_allowed,
            avg_rush_yds_allowed, rushes_faced
    """
    conn = sqlite3.connect(DB_PATH)

    # Build week filter for rolling window
    if week:
        week_start = max(1, week - window_size + 1)
        week_filter = f"AND week BETWEEN {week_start} AND {week}"
    else:
        week_filter = ""

    # Query: Advanced metrics (YBC, YAC) from pfr_advstats_rush_week
    # NOTE: plays table is empty, so we only use pfr_advstats_rush_week
    # This table has player-level data, we need to aggregate by defensive team
    adv_query = f"""
        SELECT
            opponent AS team,
            AVG(rushing_yards_before_contact_avg) AS ybc_allowed,
            AVG(rushing_yards_after_contact_avg) AS yac_allowed,
            SUM(carries) / COUNT(DISTINCT week) AS carries_per_game,
            COUNT(DISTINCT week) AS games_played,
            COUNT(*) AS player_games
        FROM pfr_advstats_rush_week
        WHERE season = {season}
            AND carries > 0
            {week_filter}
        GROUP BY opponent
    """

    try:
        metrics_df = pd.read_sql_query(adv_query, conn)

        if metrics_df.empty:
            return pd.DataFrame()

        # Select final columns
        result = metrics_df[[
            'team', 'ybc_allowed', 'yac_allowed',
            'carries_per_game', 'games_played'
        ]].copy()

        return result

    except Exception as e:
        print(f"Error calculating defensive run metrics: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def classify_defensive_run_style(metrics_df):
    """
    Classify each defense into run style categories based on YBC and YAC percentiles.

    Simplified classification using only available metrics from pfr_advstats_rush_week:

    Styles:
    1. Bulldozer: Strong at LOS (low YBC) + good tackling (low YAC) - dominant run defense
    2. Spill-and-Swarm: Weak at LOS (high YBC) but good tackling (low YAC) - absorb contact then swarm
    3. Soft Shell: Weak at LOS (high YBC) - give up yards before contact
    4. Leakier Front: Poor tackling (high YAC) - allow yards after contact
    5. Balanced: Doesn't fit other categories

    Args:
        metrics_df (pd.DataFrame): Output from calculate_defensive_run_metrics()

    Returns:
        pd.DataFrame: Original metrics plus style, percentile ranks, and explainers
    """
    if metrics_df.empty:
        return metrics_df

    result = metrics_df.copy()

    # Calculate percentiles (higher percentile = better defense)
    # Lower YBC = better defense (stop at LOS)
    result['ybc_percentile'] = (1 - result['ybc_allowed'].rank(pct=True)) * 100

    # Lower YAC = better defense (better tackling)
    result['yac_percentile'] = (1 - result['yac_allowed'].rank(pct=True)) * 100

    # Classify each team
    styles = []
    explainers = []

    for _, row in result.iterrows():
        style = "Balanced"
        explainer_parts = []

        # Bulldozer: Strong at LOS (low YBC) + good tackling (low YAC)
        if row['ybc_percentile'] >= 60 and row['yac_percentile'] >= 50:
            style = "🚜 Bulldozer"
            explainer_parts.append(f"Low YBC ({row['ybc_allowed']:.2f}, p{row['ybc_percentile']:.0f})")
            explainer_parts.append(f"Good tackling ({row['yac_allowed']:.2f}, p{row['yac_percentile']:.0f})")

        # Spill-and-Swarm: Weak at LOS (high YBC) but good tackling (low YAC)
        elif row['ybc_percentile'] <= 40 and row['yac_percentile'] >= 50:
            style = "🌊 Spill-and-Swarm"
            explainer_parts.append(f"High YBC ({row['ybc_allowed']:.2f}, p{row['ybc_percentile']:.0f})")
            explainer_parts.append(f"Good tackling ({row['yac_allowed']:.2f}, p{row['yac_percentile']:.0f})")

        # Soft Shell: Weak at LOS (high YBC)
        elif row['ybc_percentile'] <= 40:
            style = "🛡️ Soft Shell"
            explainer_parts.append(f"High YBC ({row['ybc_allowed']:.2f}, p{row['ybc_percentile']:.0f})")

        # Leakier Front: Poor tackling (high YAC)
        elif row['yac_percentile'] <= 40:
            style = "🚨 Leakier Front"
            explainer_parts.append(f"High YAC ({row['yac_allowed']:.2f}, p{row['yac_percentile']:.0f})")

        # If no specific style identified, explain balanced
        if not explainer_parts:
            explainer_parts.append(f"YBC: {row['ybc_allowed']:.2f} (p{row['ybc_percentile']:.0f})")
            explainer_parts.append(f"YAC: {row['yac_allowed']:.2f} (p{row['yac_percentile']:.0f})")

        styles.append(style)
        explainers.append(" | ".join(explainer_parts))

    result['defensive_style'] = styles
    result['style_explainer'] = explainers

    return result


def get_defensive_run_style_matchup(offense_team, defense_team, season, week=None):
    """
    Get defensive run style for a specific matchup and generate RB matchup insight.

    Args:
        offense_team (str): Offensive team abbreviation
        defense_team (str): Defensive team abbreviation
        season (int): Season
        week (int, optional): Week number for rolling window

    Returns:
        dict: Defensive style info including:
            - style: Style classification
            - explainer: Human-readable explanation
            - metrics: Raw metrics dict
            - rb_matchup_insight: Tailored RB advice
    """
    # Calculate metrics for all teams
    metrics_df = calculate_defensive_run_metrics(season, week)

    if metrics_df.empty:
        return {
            'style': 'Unknown',
            'explainer': 'Insufficient data',
            'metrics': {},
            'rb_matchup_insight': 'No data available'
        }

    # Classify styles
    styled_df = classify_defensive_run_style(metrics_df)

    # Get defense row
    def_row = styled_df[styled_df['team'] == defense_team]

    if def_row.empty:
        return {
            'style': 'Unknown',
            'explainer': 'Team not found',
            'metrics': {},
            'rb_matchup_insight': 'No data available for this team'
        }

    def_row = def_row.iloc[0]

    # Generate RB matchup insight based on style
    style = def_row['defensive_style']
    insights = {
        "🚜 Bulldozer": "TOUGH RB MATCHUP - Strong at point of attack. Target pass-catching RBs or look elsewhere.",
        "🌊 Spill-and-Swarm": "MODERATE RB MATCHUP - Limits big plays but allows consistent gains. Volume RBs preferred.",
        "🛡️ Soft Shell": "FAVORABLE RB MATCHUP - Allows yards at LOS. Target rushing volume and YPC upside.",
        "🚨 Leakier Front": "SMASH RB MATCHUP - Allows explosive runs and missed tackles. Target home run RBs.",
        "Balanced": "NEUTRAL RB MATCHUP - No clear defensive identity. Defer to player talent."
    }

    return {
        'style': style,
        'explainer': def_row['style_explainer'],
        'metrics': {
            'stuff_rate': def_row['stuff_rate'],
            'explosive_run_pct': def_row['explosive_run_pct'],
            'ybc_allowed': def_row['ybc_allowed'],
            'yac_allowed': def_row['yac_allowed'],
            'avg_rush_yds_allowed': def_row['avg_rush_yds_allowed'],
            'rushes_faced': def_row['rushes_faced']
        },
        'rb_matchup_insight': insights.get(style, insights["Balanced"])
    }


# ============================================================================
# COMPREHENSIVE WR MATCHUP SCORING SYSTEM
# ============================================================================

def calculate_wr_talent_score(wr_rec_yds_per_game, wr_rec_tds_total, wr_targets_per_game,
                              wr_target_share_pct, wr_games, wr_receptions_per_game,
                              wr_avg_yac):
    """
    Calculate WR talent/production score (0-100) based purely on player performance.
    This isolates WR ability from defensive matchup quality.

    Scoring Components (75 points total):
    - Receiving Yards Production: 25 points
    - Receiving TD Production: 20 points
    - Target Volume/Role: 15 points
    - Target Share %: 10 points
    - Reception Rate (Catch %): 5 points (efficiency)
    """
    score = 0

    # Calculate per-game rates
    wr_rec_tds_per_game = wr_rec_tds_total / wr_games if wr_games > 0 else 0
    catch_rate = (wr_receptions_per_game / wr_targets_per_game * 100) if wr_targets_per_game > 0 else 0

    # 1. RECEIVING YARDS PRODUCTION (25 points)
    if wr_rec_yds_per_game >= 90:  # Elite (top 5 - 1500+ yd pace)
        score += 25
    elif wr_rec_yds_per_game >= 75:  # Strong (top 12 - 1250+ yd pace)
        score += 20
    elif wr_rec_yds_per_game >= 60:  # Average (top 20 - 1000+ yd pace)
        score += 15
    elif wr_rec_yds_per_game >= 45:  # Below average (750+ yd pace)
        score += 10
    else:  # Limited role (<750 yd pace)
        score += 5

    # 2. RECEIVING TD PRODUCTION (20 points)
    if wr_rec_tds_per_game >= 0.6:  # Elite (10+ TDs per season)
        score += 20
    elif wr_rec_tds_per_game >= 0.4:  # Strong (7+ TDs per season)
        score += 15
    elif wr_rec_tds_per_game >= 0.3:  # Average (5+ TDs per season)
        score += 10
    elif wr_rec_tds_per_game > 0:  # Occasional (1-4 TDs per season)
        score += 5
    # else: 0 points for no TDs

    # 3. TARGET VOLUME/ROLE (15 points)
    if wr_targets_per_game >= 10:  # Elite target share (true WR1)
        score += 15
    elif wr_targets_per_game >= 8:  # Strong (alpha WR)
        score += 12
    elif wr_targets_per_game >= 6:  # Average (WR2)
        score += 9
    elif wr_targets_per_game >= 4:  # Below average (WR3)
        score += 5
    else:  # Limited role
        score += 2

    # 4. TARGET SHARE % (10 points)
    if wr_target_share_pct >= 28:  # Elite (true alpha - 28%+)
        score += 10
    elif wr_target_share_pct >= 22:  # Strong (clear WR1 - 22%+)
        score += 8
    elif wr_target_share_pct >= 17:  # Average (WR1/WR2 - 17%+)
        score += 5
    elif wr_target_share_pct >= 12:  # Below average (WR3 - 12%+)
        score += 2
    # else: 0 points for limited share

    # 5. RECEPTION RATE/EFFICIENCY (5 points) - Catch % bonus
    if catch_rate >= 70:  # Elite hands/efficiency
        score += 5
    elif catch_rate >= 65:  # Strong
        score += 3
    elif catch_rate >= 60:  # Average
        score += 2
    elif catch_rate >= 55:  # Below average
        score += 1
    # else: 0 points

    # Scale to 0-100 (current max is 75, so multiply by 100/75 = 1.333)
    scaled_score = (score / 75) * 100

    return round(scaled_score, 1)


def calculate_wr_defensive_matchup_score(def_rec_yds_allowed, def_rec_tds_allowed,
                                         def_rec_rank, league_avg_def_rec_tds):
    """
    Calculate WR defensive matchup score (0-100) based purely on defensive weakness.
    Higher score = worse defense = better matchup for WR.

    Scoring Components (25 points total):
    - Defensive Receiving Yards Allowed: 15 points
    - Defensive TD Vulnerability: 10 points (TDs allowed to WRs)
    """
    score = 0

    # 1. DEFENSIVE RECEIVING VULNERABILITY (15 points) - Pass yards allowed to WRs
    if def_rec_yds_allowed >= 80:  # Generous (worst 5 pass defenses)
        score += 15
    elif def_rec_yds_allowed >= 70:  # Favorable (bottom 12)
        score += 12
    elif def_rec_yds_allowed >= 60:  # Average
        score += 8
    else:  # Lockdown (<60 yds to WRs)
        score += 4

    # 2. DEFENSIVE TD VULNERABILITY (10 points) - Receiving TDs allowed to WRs
    if def_rec_tds_allowed >= league_avg_def_rec_tds * 1.2:  # TD-prone (20%+ above avg)
        score += 10
    elif def_rec_tds_allowed >= league_avg_def_rec_tds * 0.8:  # Average
        score += 6
    else:  # Lockdown (<80% of avg)
        score += 2

    # Scale to 0-100 (current max is 25, so multiply by 100/25 = 4)
    scaled_score = (score / 25) * 100

    return round(scaled_score, 1)


def calculate_comprehensive_wr_score(wr_rec_yds_per_game, wr_rec_tds_total, wr_targets_per_game,
                                     wr_target_share_pct, wr_games, def_rec_yds_allowed,
                                     def_rec_tds_allowed, def_rec_rank, league_avg_rec_yds,
                                     league_avg_def_rec_tds):
    """
    Calculate comprehensive WR matchup score (0-100 scale) based on multiple factors.

    Scoring Components:
    - Receiving Yards Production: 25 points (balanced yards approach)
    - Receiving TD Production: 20 points (balanced TDs approach)
    - Target Volume/Role: 20 points (opportunity analysis)
    - Target Share %: 10 points (WR1 role identification)
    - Defensive Receiving Vulnerability: 15 points (pass yards allowed)
    - Defensive TD Vulnerability: 10 points (receiving TDs allowed to WRs)

    Total: 100 points
    """
    score = 0

    # Calculate per-game rates
    wr_rec_tds_per_game = wr_rec_tds_total / wr_games if wr_games > 0 else 0

    # 1. RECEIVING YARDS PRODUCTION (25 points) - Balanced Yards Approach
    if wr_rec_yds_per_game >= 90:  # Elite (top 5 - 1500+ yd pace)
        score += 25
    elif wr_rec_yds_per_game >= 75:  # Strong (top 12 - 1250+ yd pace)
        score += 20
    elif wr_rec_yds_per_game >= 60:  # Average (top 20 - 1000+ yd pace)
        score += 15
    elif wr_rec_yds_per_game >= 45:  # Below average (750+ yd pace)
        score += 10
    else:  # Limited role (<750 yd pace)
        score += 5

    # 2. RECEIVING TD PRODUCTION (20 points) - Balanced TDs Approach
    if wr_rec_tds_per_game >= 0.6:  # Elite (10+ TDs per season)
        score += 20
    elif wr_rec_tds_per_game >= 0.4:  # Strong (7+ TDs per season)
        score += 15
    elif wr_rec_tds_per_game >= 0.3:  # Average (5+ TDs per season)
        score += 10
    elif wr_rec_tds_per_game > 0:  # Occasional (1-4 TDs per season)
        score += 5
    # else: 0 points for no TDs

    # 3. TARGET VOLUME/ROLE (20 points) - Opportunity Analysis
    if wr_targets_per_game >= 10:  # Elite target hog (alpha WR1)
        score += 20
    elif wr_targets_per_game >= 8:  # Strong volume (WR1)
        score += 16
    elif wr_targets_per_game >= 6:  # Good volume (WR2)
        score += 12
    elif wr_targets_per_game >= 4:  # Moderate volume (WR3/FLEX)
        score += 8
    else:  # Limited volume (<4 targets)
        score += 3

    # 4. TARGET SHARE % (10 points) - WR1 Role Identification
    if wr_target_share_pct >= 25:  # True alpha WR1 (25%+ of team targets)
        score += 10
    elif wr_target_share_pct >= 20:  # Strong WR1 (20%+ of team targets)
        score += 8
    elif wr_target_share_pct >= 15:  # WR2 role (15%+ of team targets)
        score += 6
    elif wr_target_share_pct >= 10:  # WR3 role (10%+ of team targets)
        score += 3
    # else: 0 points for low target share

    # 5. DEFENSIVE RECEIVING VULNERABILITY (15 points) - Pass Yards Allowed
    if def_rec_yds_allowed >= 70:  # Generous pass defense (worst 5)
        score += 15
    elif def_rec_yds_allowed >= 65:  # Favorable (bottom 12)
        score += 12
    elif def_rec_yds_allowed >= 60:  # Average
        score += 8
    else:  # Lockdown (<60 yds allowed to WRs)
        score += 4

    # 6. DEFENSIVE TD VULNERABILITY (10 points) - Receiving TDs Allowed to WRs
    if def_rec_tds_allowed >= league_avg_def_rec_tds * 1.3:  # TD-prone (30%+ above avg)
        score += 10
    elif def_rec_tds_allowed >= league_avg_def_rec_tds * 0.8:  # Average
        score += 6
    else:  # Lockdown (<80% of avg)
        score += 2

    return round(score, 1)


def generate_comprehensive_wr_storyline(wr_score, wr_name, wr_rec_yds_per_game, wr_rec_tds_per_game,
                                        wr_targets_per_game, wr_target_share_pct, def_rec_yds_allowed,
                                        def_rec_tds_allowed, def_rec_rank):
    """
    Generate comprehensive WR matchup storyline based on score.

    7 Tier System (same as QBs/RBs):
    - 85-100: 🔥🔥🔥 ELITE SMASH SPOT
    - 75-84:  🔥🔥 PREMIUM MATCHUP
    - 65-74:  🔥 SMASH SPOT
    - 55-64:  ✅ SOLID START
    - 45-54:  ⚖️ BALANCED
    - 35-44:  ⚠️ RISKY PLAY
    - 0-34:   🛑 AVOID
    """
    # Build description components
    rec_yds_desc = f"{wr_rec_yds_per_game:.1f} rec yds/gm"
    rec_tds_desc = f"{wr_rec_tds_per_game:.2f} rec TDs/gm"
    targets_desc = f"{wr_targets_per_game:.1f} tgts/gm"
    target_share_desc = f"{wr_target_share_pct:.1f}% target share"
    def_desc = f"vs #{def_rec_rank} pass D ({def_rec_yds_allowed:.1f} rec yds allowed, {def_rec_tds_allowed:.1f} rec TDs allowed to WRs)"

    # Determine tier and recommendation
    if wr_score >= 85:
        tier = "🔥🔥🔥 ELITE SMASH SPOT"
        recommendation = f"{wr_name} is a MUST-START WR1. Elite alpha production ({rec_yds_desc}, {rec_tds_desc}, {targets_desc}, {target_share_desc}) against exploitable pass defense ({def_desc}). Expect ceiling performance with multiple TD upside."
    elif wr_score >= 75:
        tier = "🔥🔥 PREMIUM MATCHUP"
        recommendation = f"{wr_name} is a premium WR1 play. Strong production ({rec_yds_desc}, {rec_tds_desc}, {targets_desc}, {target_share_desc}) with favorable matchup ({def_desc}). High floor and ceiling - start with confidence."
    elif wr_score >= 65:
        tier = "🔥 SMASH SPOT"
        recommendation = f"{wr_name} is a strong start. Solid production ({rec_yds_desc}, {rec_tds_desc}, {targets_desc}) in advantageous spot ({def_desc}). Good WR1/WR2 upside play."
    elif wr_score >= 55:
        tier = "✅ SOLID START"
        recommendation = f"{wr_name} is a safe WR2/FLEX. Reliable production ({rec_yds_desc}, {rec_tds_desc}, {targets_desc}) vs {def_desc}. Good floor with TD upside."
    elif wr_score >= 45:
        tier = "⚖️ BALANCED"
        recommendation = f"{wr_name} is a neutral matchup ({rec_yds_desc}, {rec_tds_desc}, {targets_desc}) vs {def_desc}. Start based on roster needs and other options. FLEX consideration."
    elif wr_score >= 35:
        tier = "⚠️ RISKY PLAY"
        recommendation = f"{wr_name} has risk factors. Limited production ({rec_yds_desc}, {rec_tds_desc}, {targets_desc}) and/or tough matchup ({def_desc}). Boom-bust FLEX play with low floor."
    else:  # < 35
        tier = "🛑 AVOID"
        recommendation = f"{wr_name} is a fade candidate. Poor production ({rec_yds_desc}, {rec_tds_desc}, {targets_desc}) vs difficult matchup ({def_desc}). Bench unless desperate for bye week fill-in."

    return tier, recommendation


# ============================================================================
# RB Pass-Catching Exploitation Matchup Analysis Functions
# ============================================================================

def calculate_rb_receiving_matchup_stats(season, max_week=None):
    """
    Calculate RB receiving matchup statistics for offense and defense.

    Returns DataFrame with columns:
    - team: Team abbreviation
    - rb_targets_per_game: RB targets per game
    - rb_receptions_per_game: RB receptions per game
    - rb_rec_yards_per_game: RB receiving yards per game
    - defense_rec_to_rb: Receiving yards allowed to RBs per game (from defensive stats)
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # Build week filter
        week_filter = f"AND week <= {max_week}" if max_week else ""

        # Get offensive RB receiving stats
        offense_query = f"""
        SELECT
            team,
            COUNT(DISTINCT week) as games,
            SUM(targets) as total_rb_targets,
            SUM(receptions) as total_rb_receptions,
            SUM(receiving_yards) as total_rb_rec_yards
        FROM player_stats
        WHERE season = {season}
        {week_filter}
        AND position = 'RB'
        GROUP BY team
        """

        offense_df = pd.read_sql_query(offense_query, conn)

        if not offense_df.empty:
            # Calculate per-game averages
            offense_df['rb_targets_per_game'] = (offense_df['total_rb_targets'] / offense_df['games']).round(1)
            offense_df['rb_receptions_per_game'] = (offense_df['total_rb_receptions'] / offense_df['games']).round(1)
            offense_df['rb_rec_yards_per_game'] = (offense_df['total_rb_rec_yards'] / offense_df['games']).round(1)

        # Get defensive RB receiving stats (yards allowed to RBs)
        defense_query = f"""
        SELECT
            opponent_team as team,
            COUNT(DISTINCT week) as games,
            SUM(receiving_yards) as total_rec_yards_to_rb
        FROM player_stats
        WHERE season = {season}
        {week_filter}
        AND position = 'RB'
        GROUP BY opponent_team
        """

        defense_df = pd.read_sql_query(defense_query, conn)

        if not defense_df.empty:
            # Calculate per-game averages
            defense_df['defense_rec_to_rb'] = (
                defense_df['total_rec_yards_to_rb'] / defense_df['games']
            ).round(1)

        # Merge offense and defense stats
        result_df = pd.merge(
            offense_df[['team', 'rb_targets_per_game', 'rb_receptions_per_game', 'rb_rec_yards_per_game']],
            defense_df[['team', 'defense_rec_to_rb']],
            on='team',
            how='outer'
        )

        # Fill NaN values with 0
        result_df = result_df.fillna(0)

        conn.close()
        return result_df

    except Exception as e:
        st.error(f"Error calculating RB receiving matchup stats: {e}")
        return pd.DataFrame()


def categorize_rb_receiving_offense(rb_targets_per_game):
    """
    Categorize offensive RB receiving usage.

    Benchmarks:
    - Bell Cow Pass-Catcher (>6 targets/game): Heavily involved in passing game
    - Situational Receiver (3-6 targets/game): Third-down/passing situations
    - Ground-Only (<3 targets/game): Limited passing work
    """
    if rb_targets_per_game >= 6:
        return "Bell Cow Pass-Catcher"
    elif rb_targets_per_game >= 3:
        return "Situational Receiver"
    else:
        return "Ground-Only"


def categorize_rb_receiving_defense(rec_yards_to_rb_per_game):
    """
    Categorize defensive RB receiving vulnerability.

    Benchmarks:
    - RB-Vulnerable (>45 yds/game to RBs): Struggles covering backs
    - Average (30-45 yds/game): League standard
    - Lockdown LBs (<30 yds/game): Elite RB coverage
    """
    if rec_yards_to_rb_per_game >= 45:
        return "RB-Vulnerable"
    elif rec_yards_to_rb_per_game >= 30:
        return "Average"
    else:
        return "Lockdown LBs"


def generate_rb_receiving_storyline(rb_targets_per_game, defense_rec_to_rb):
    """
    Generate narrative storyline based on RB receiving usage vs defensive vulnerability to RB targets.

    Favorable Matchups (PPR gold):
    - Bell Cow Pass-Catcher vs RB-Vulnerable defense
    - Situational Receiver vs RB-Vulnerable defense
    - Bell Cow Pass-Catcher vs Average defense

    Unfavorable Matchups (Limited PPR value):
    - Bell Cow Pass-Catcher vs Lockdown LBs
    - Situational Receiver vs Lockdown LBs
    """

    off_category = categorize_rb_receiving_offense(rb_targets_per_game)
    def_category = categorize_rb_receiving_defense(defense_rec_to_rb)

    # EXPLOIT MATCHUPS (PPR gold)
    if off_category == "Bell Cow Pass-Catcher" and defense_rec_to_rb >= 45:
        return "🎯 PPR SMASH: Pass-Catcher vs RB-Vulnerable", f"RB averages {rb_targets_per_game:.1f} targets vs defense allowing {defense_rec_to_rb:.1f} rec yds to RBs - PPR gem"

    elif off_category == "Situational Receiver" and defense_rec_to_rb >= 45:
        return "💰 CHECKDOWN CITY: Passing Situations", f"RB gets {rb_targets_per_game:.1f} targets in passing downs vs vulnerable coverage ({defense_rec_to_rb:.1f} yds allowed) - Volume spike"

    elif off_category == "Bell Cow Pass-Catcher" and 30 <= defense_rec_to_rb < 45:
        return "✅ SOLID FLOOR: Pass-Catcher vs Average", f"High-target RB ({rb_targets_per_game:.1f}/gm) vs league-average coverage - Safe PPR floor"

    # TOUGH MATCHUPS (Limited PPR value)
    elif off_category == "Bell Cow Pass-Catcher" and defense_rec_to_rb < 30:
        return "🚫 AVOID: Pass-Catcher vs Lockdown LBs", f"Primary receiving role neutralized vs elite RB coverage ({defense_rec_to_rb:.1f} yds allowed) - Fade in PPR"

    elif off_category == "Situational Receiver" and defense_rec_to_rb < 30:
        return "🛡️ LIMITED: Situational vs Lockdown", f"Limited targets ({rb_targets_per_game:.1f}/gm) vs strong coverage ({defense_rec_to_rb:.1f} yds) - Low ceiling"

    # NEUTRAL/GROUND-ONLY MATCHUPS
    elif off_category == "Ground-Only":
        return "🚧 GROUND GAME: Rushing Specialist", f"RB rarely targeted ({rb_targets_per_game:.1f}/gm) - Non-factor in passing game"

    else:
        return "⚪ STANDARD: Neutral Matchup", f"RB Targets: {rb_targets_per_game:.1f}/gm vs Def Allows: {defense_rec_to_rb:.1f} yds to RBs"


# ============================================================================
# QB Passing TD Matchup Analysis Functions
# ============================================================================

def calculate_passing_td_matchup_stats(season, max_week=None):
    """
    Calculate QB Passing TD matchup statistics for offense and defense.

    Returns DataFrame with columns:
    - team: Team abbreviation
    - offense_pass_tds: Passing TDs scored per game
    - defense_pass_tds_allowed: Passing TDs allowed per game
    - games: Games played
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        week_filter = f"AND week <= {max_week}" if max_week else ""

        # Get offensive passing TD stats
        offense_query = f"""
        SELECT
            team,
            COUNT(DISTINCT week) as games,
            SUM(passing_tds) as total_pass_tds
        FROM player_stats
        WHERE season = {season}
        {week_filter}
        AND position IN ('QB')
        GROUP BY team
        """

        offense_df = pd.read_sql_query(offense_query, conn)

        if not offense_df.empty:
            offense_df['offense_pass_tds'] = (offense_df['total_pass_tds'] / offense_df['games']).round(2)

        # Get defensive passing TD stats (TDs allowed)
        defense_query = f"""
        SELECT
            opponent_team as team,
            COUNT(DISTINCT week) as games,
            SUM(passing_tds) as total_pass_tds_allowed
        FROM player_stats
        WHERE season = {season}
        {week_filter}
        AND position IN ('QB')
        GROUP BY opponent_team
        """

        defense_df = pd.read_sql_query(defense_query, conn)

        if not defense_df.empty:
            defense_df['defense_pass_tds_allowed'] = (
                defense_df['total_pass_tds_allowed'] / defense_df['games']
            ).round(2)

        # Merge offense and defense stats
        result_df = pd.merge(
            offense_df[['team', 'offense_pass_tds']],
            defense_df[['team', 'defense_pass_tds_allowed']],
            on='team',
            how='outer'
        )

        result_df = result_df.fillna(0)
        conn.close()
        return result_df

    except Exception as e:
        st.error(f"Error calculating Passing TD matchup stats: {e}")
        return pd.DataFrame()


def categorize_passing_td_offense(pass_tds_per_game):
    """
    Categorize offensive passing TD production.

    Benchmarks (based on API passing_tds metric):
    - Elite TD Producer (>2.5 TDs/game): High-volume TD scoring
    - Balanced Passer (1.5-2.5 TDs/game): Standard touchdown production
    - Game Manager (<1.5 TDs/game): Conservative/limited TD passing
    """
    if pass_tds_per_game >= 2.5:
        return "Elite TD Producer"
    elif pass_tds_per_game >= 1.5:
        return "Balanced Passer"
    else:
        return "Game Manager"


def categorize_passing_td_defense(pass_tds_allowed_per_game):
    """
    Categorize defensive passing TD vulnerability.

    Benchmarks (based on API passing_tds allowed):
    - TD Funnel Defense (>2.5 TDs/game allowed): Allows high TD rate to QBs
    - Average Coverage (1.5-2.5 TDs/game): League standard TD prevention
    - Lockdown Secondary (<1.5 TDs/game): Elite at limiting passing TDs
    """
    if pass_tds_allowed_per_game >= 2.5:
        return "TD Funnel Defense"
    elif pass_tds_allowed_per_game >= 1.5:
        return "Average Coverage"
    else:
        return "Lockdown Secondary"


def generate_passing_td_storyline(offense_pass_tds, defense_pass_tds_allowed):
    """
    Generate narrative storyline based on passing TD production vs defensive vulnerability.

    Favorable Matchups (High TD upside):
    - Elite TD Producer vs TD Funnel Defense (most explosive)
    - Game Manager vs TD Funnel Defense (streaming opportunity)
    - Elite TD Producer vs Average Coverage

    Unfavorable Matchups (Limited TD upside):
    - Elite TD Producer vs Lockdown Secondary (strength vs strength)
    - Game Manager vs Lockdown Secondary (worst case)
    """

    off_category = categorize_passing_td_offense(offense_pass_tds)
    def_category = categorize_passing_td_defense(defense_pass_tds_allowed)

    # EXPLOIT MATCHUPS (High TD upside)
    if off_category == "Elite TD Producer" and defense_pass_tds_allowed >= 2.5:
        return "🎯 TD SMASH SPOT", f"Elite TD scorer ({offense_pass_tds:.1f} TDs/gm) vs TD-vulnerable defense ({defense_pass_tds_allowed:.1f} allowed/gm) - Elite TD upside"

    elif off_category == "Game Manager" and defense_pass_tds_allowed >= 2.5:
        return "💎 SNEAKY VALUE", f"Conservative QB ({offense_pass_tds:.1f} TDs/gm) faces generous secondary ({defense_pass_tds_allowed:.1f} allowed/gm) - Streaming opportunity"

    elif off_category == "Elite TD Producer" and 1.5 <= defense_pass_tds_allowed < 2.5:
        return "✅ FAVORABLE", f"Elite TD producer ({offense_pass_tds:.1f} TDs/gm) vs average defense ({defense_pass_tds_allowed:.1f} allowed/gm) - Solid TD odds"

    # TOUGH MATCHUPS (Limited TD upside)
    elif off_category == "Elite TD Producer" and defense_pass_tds_allowed < 1.5:
        return "⚔️ TOUGH TEST", f"Elite QB ({offense_pass_tds:.1f} TDs/gm) vs lockdown secondary ({defense_pass_tds_allowed:.1f} allowed/gm) - Limited ceiling"

    elif off_category == "Game Manager" and defense_pass_tds_allowed < 1.5:
        return "🛑 AVOID", f"Game manager ({offense_pass_tds:.1f} TDs/gm) vs elite coverage ({defense_pass_tds_allowed:.1f} allowed/gm) - Low TD potential"

    # NEUTRAL MATCHUPS
    elif off_category == "Balanced Passer":
        return "⚖️ BALANCED", f"Standard TD production ({offense_pass_tds:.1f} TDs/gm) vs defense allowing {defense_pass_tds_allowed:.1f} TDs/gm"

    else:
        return "⚪ STANDARD", f"Pass TDs: {offense_pass_tds:.1f}/gm vs Def Allows: {defense_pass_tds_allowed:.1f}/gm"


# ============================================================================
# QB Passing Yards Matchup Analysis Functions
# ============================================================================

def calculate_passing_yards_matchup_stats(season, max_week=None):
    """
    Calculate QB Passing Yards matchup statistics for offense and defense.

    Returns DataFrame with columns:
    - team: Team abbreviation
    - offense_pass_yards: Passing yards per game
    - defense_pass_yards_allowed: Passing yards allowed per game
    - games: Games played
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        week_filter = f"AND week <= {max_week}" if max_week else ""

        # Get offensive passing yards stats
        offense_query = f"""
        SELECT
            team,
            COUNT(DISTINCT week) as games,
            SUM(passing_yards) as total_pass_yards
        FROM player_stats
        WHERE season = {season}
        {week_filter}
        AND position IN ('QB')
        GROUP BY team
        """

        offense_df = pd.read_sql_query(offense_query, conn)

        if not offense_df.empty:
            offense_df['offense_pass_yards'] = (offense_df['total_pass_yards'] / offense_df['games']).round(1)

        # Get defensive passing yards stats (yards allowed)
        defense_query = f"""
        SELECT
            opponent_team as team,
            COUNT(DISTINCT week) as games,
            SUM(passing_yards) as total_pass_yards_allowed
        FROM player_stats
        WHERE season = {season}
        {week_filter}
        AND position IN ('QB')
        GROUP BY opponent_team
        """

        defense_df = pd.read_sql_query(defense_query, conn)

        if not defense_df.empty:
            defense_df['defense_pass_yards_allowed'] = (
                defense_df['total_pass_yards_allowed'] / defense_df['games']
            ).round(1)

        # Merge offense and defense stats
        result_df = pd.merge(
            offense_df[['team', 'offense_pass_yards']],
            defense_df[['team', 'defense_pass_yards_allowed']],
            on='team',
            how='outer'
        )

        result_df = result_df.fillna(0)
        conn.close()
        return result_df

    except Exception as e:
        st.error(f"Error calculating Passing Yards matchup stats: {e}")
        return pd.DataFrame()


def categorize_passing_yards_offense(pass_yards_per_game):
    """
    Categorize offensive passing yards production.

    Benchmarks (based on API passing_yards metric):
    - Air Raid Offense (≥280 yards/game): High-volume passing attack
    - Balanced Passing (220-280 yards/game): Standard passing volume
    - Run-First Offense (<220 yards/game): Conservative passing approach
    """
    if pass_yards_per_game >= 280:
        return "Air Raid Offense"
    elif pass_yards_per_game >= 220:
        return "Balanced Passing"
    else:
        return "Run-First Offense"


def categorize_passing_yards_defense(pass_yards_allowed_per_game):
    """
    Categorize defensive passing yards vulnerability.

    Benchmarks (based on API passing_yards allowed):
    - Pass Funnel Defense (≥270 yards/game allowed): Allows high passing volume
    - Average Secondary (220-270 yards/game): League standard pass defense
    - Elite Coverage (<220 yards/game): Dominant at limiting passing yards
    """
    if pass_yards_allowed_per_game >= 270:
        return "Pass Funnel Defense"
    elif pass_yards_allowed_per_game >= 220:
        return "Average Secondary"
    else:
        return "Elite Coverage"


def generate_passing_yards_storyline(offense_pass_yards, defense_pass_yards_allowed):
    """
    Generate narrative storyline based on passing yards production vs defensive vulnerability.

    Favorable Matchups (High volume upside):
    - Air Raid vs Pass Funnel (most explosive)
    - Run-First vs Pass Funnel (game script opportunity)
    - Air Raid vs Average Secondary

    Unfavorable Matchups (Limited volume):
    - Air Raid vs Elite Coverage (grind it out)
    - Run-First vs Elite Coverage (lowest volume)
    """

    off_category = categorize_passing_yards_offense(offense_pass_yards)
    def_category = categorize_passing_yards_defense(defense_pass_yards_allowed)

    # EXPLOIT MATCHUPS (High volume upside)
    if off_category == "Air Raid Offense" and defense_pass_yards_allowed >= 270:
        return "🚀 CEILING EXPLOSION", f"High-volume passing ({offense_pass_yards:.0f} yds/gm) vs generous secondary ({defense_pass_yards_allowed:.0f} yds allowed/gm) - Elite yardage ceiling"

    elif off_category == "Run-First Offense" and defense_pass_yards_allowed >= 270:
        return "📈 VOLUME SPIKE", f"Run-heavy offense ({offense_pass_yards:.0f} yds/gm) forced to pass vs weak secondary ({defense_pass_yards_allowed:.0f} yds allowed/gm) - Game script opportunity"

    elif off_category == "Air Raid Offense" and 220 <= defense_pass_yards_allowed < 270:
        return "✅ SOLID VOLUME", f"Air raid attack ({offense_pass_yards:.0f} yds/gm) vs average coverage ({defense_pass_yards_allowed:.0f} yds allowed/gm) - Reliable yardage"

    # TOUGH MATCHUPS (Limited volume)
    elif off_category == "Air Raid Offense" and defense_pass_yards_allowed < 220:
        return "💪 GRIND IT OUT", f"High-volume QB ({offense_pass_yards:.0f} yds/gm) vs elite coverage ({defense_pass_yards_allowed:.0f} yds allowed/gm) - Tough sledding"

    elif off_category == "Run-First Offense" and defense_pass_yards_allowed < 220:
        return "⚠️ LOW VOLUME", f"Conservative passing ({offense_pass_yards:.0f} yds/gm) vs dominant coverage ({defense_pass_yards_allowed:.0f} yds allowed/gm) - Minimal upside"

    # NEUTRAL MATCHUPS
    elif off_category == "Balanced Passing":
        return "⚖️ BALANCED", f"Standard passing volume ({offense_pass_yards:.0f} yds/gm) vs defense allowing {defense_pass_yards_allowed:.0f} yds/gm"

    else:
        return "⚪ STANDARD", f"Pass Yards: {offense_pass_yards:.0f}/gm vs Def Allows: {defense_pass_yards_allowed:.0f}/gm"


# ============================================================================
# QB TD Rate vs Defensive INT Rate Matchup Analysis Functions
# ============================================================================

def calculate_qb_td_rate_vs_int_stats(season, max_week=None):
    """
    Calculate QB TD Rate vs Defensive INT Rate matchup statistics.

    Requires merging player_stats (for QB TD rate) and pfr_advstats_def_week (for defensive INTs).

    Returns DataFrame with columns:
    - team: Team abbreviation
    - qb_td_rate: TD percentage per attempt
    - defense_ints_per_game: Interceptions forced per game
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        week_filter = f"AND week <= {max_week}" if max_week else ""

        # Get offensive QB TD rate (TDs per attempt)
        offense_query = f"""
        SELECT
            team,
            SUM(passing_tds) as total_tds,
            SUM(attempts) as total_attempts
        FROM player_stats
        WHERE season = {season}
        {week_filter}
        AND position IN ('QB')
        GROUP BY team
        """

        offense_df = pd.read_sql_query(offense_query, conn)

        if not offense_df.empty:
            offense_df['qb_td_rate'] = (
                (offense_df['total_tds'] / offense_df['total_attempts'].replace(0, 1)) * 100
            ).round(2)

        # Get defensive INT rate from pfr_advstats_def_week table
        defense_query = f"""
        SELECT
            team,
            COUNT(DISTINCT week) as games,
            SUM(CAST(def_ints AS REAL)) as total_ints
        FROM pfr_advstats_def_week
        WHERE season = {season}
        {week_filter}
        GROUP BY team
        """

        defense_df = pd.read_sql_query(defense_query, conn)

        if not defense_df.empty:
            defense_df['defense_ints_per_game'] = (
                defense_df['total_ints'] / defense_df['games']
            ).round(2)

        # Merge offense and defense stats
        result_df = pd.merge(
            offense_df[['team', 'qb_td_rate']],
            defense_df[['team', 'defense_ints_per_game']],
            on='team',
            how='outer'
        )

        result_df = result_df.fillna(0)
        conn.close()
        return result_df

    except Exception as e:
        st.error(f"Error calculating QB TD Rate vs INT stats: {e}")
        return pd.DataFrame()


def categorize_qb_td_rate(td_rate_percent):
    """
    Categorize QB TD efficiency.

    Benchmarks (based on API passing_tds / attempts):
    - Elite Efficiency (≥5.5% TD rate): Top-tier TD efficiency
    - Average Efficiency (3.5-5.5% TD rate): Standard QB efficiency
    - Struggling QB (<3.5% TD rate): Below-average TD conversion
    """
    if td_rate_percent >= 5.5:
        return "Elite Efficiency"
    elif td_rate_percent >= 3.5:
        return "Average Efficiency"
    else:
        return "Struggling QB"


def categorize_defensive_int_rate(ints_per_game):
    """
    Categorize defensive INT creation.

    Benchmarks (based on pfr_advstats_def_week.def_ints):
    - Ball Hawk Defense (≥1.2 INTs/game): Aggressive turnover creation
    - Average Turnover Creation (0.7-1.2 INTs/game): League standard
    - Passive Coverage (<0.7 INTs/game): Limited turnover generation
    """
    if ints_per_game >= 1.2:
        return "Ball Hawk Defense"
    elif ints_per_game >= 0.7:
        return "Average Turnover Creation"
    else:
        return "Passive Coverage"


def generate_qb_td_rate_vs_int_storyline(qb_td_rate, defense_ints_per_game):
    """
    Generate narrative storyline based on QB TD efficiency vs defensive INT creation.

    Favorable Matchups (High ceiling, low risk):
    - Elite Efficiency vs Passive Coverage (safest ceiling play)
    - Average Efficiency vs Passive Coverage
    - Elite Efficiency vs Average Turnover Creation

    Unfavorable Matchups (High risk):
    - Elite Efficiency vs Ball Hawk Defense (boom-bust)
    - Struggling QB vs Ball Hawk Defense (danger zone)
    """

    off_category = categorize_qb_td_rate(qb_td_rate)
    def_category = categorize_defensive_int_rate(defense_ints_per_game)

    # FAVORABLE MATCHUPS (Safe ceiling plays)
    if off_category == "Elite Efficiency" and defense_ints_per_game < 0.7:
        return "⚡ HIGH CEILING, LOW RISK", f"Elite TD rate ({qb_td_rate:.1f}%) vs passive coverage ({defense_ints_per_game:.1f} INTs/gm) - Safe ceiling play"

    elif off_category == "Average Efficiency" and defense_ints_per_game < 0.7:
        return "✅ SAFE FLOOR", f"Standard efficiency ({qb_td_rate:.1f}%) vs passive defense ({defense_ints_per_game:.1f} INTs/gm) - Low turnover risk"

    elif off_category == "Elite Efficiency" and 0.7 <= defense_ints_per_game < 1.2:
        return "💎 SOLID PLAY", f"Elite efficiency ({qb_td_rate:.1f}%) vs average coverage ({defense_ints_per_game:.1f} INTs/gm) - Good TD upside"

    # RISKY MATCHUPS (Turnover danger)
    elif off_category == "Elite Efficiency" and defense_ints_per_game >= 1.2:
        return "🎲 BOOM-BUST", f"Efficient QB ({qb_td_rate:.1f}%) vs ball-hawk defense ({defense_ints_per_game:.1f} INTs/gm) - High variance"

    elif off_category == "Struggling QB" and defense_ints_per_game >= 1.2:
        return "💀 DANGER ZONE", f"Struggling QB ({qb_td_rate:.1f}%) vs aggressive defense ({defense_ints_per_game:.1f} INTs/gm) - Multiple turnover risk"

    elif off_category == "Struggling QB" and 0.7 <= defense_ints_per_game < 1.2:
        return "⚠️ RISKY", f"Low efficiency ({qb_td_rate:.1f}%) vs average coverage ({defense_ints_per_game:.1f} INTs/gm) - Turnover concern"

    # NEUTRAL MATCHUPS
    elif off_category == "Average Efficiency":
        return "⚖️ BALANCED", f"Average efficiency ({qb_td_rate:.1f}%) vs {defense_ints_per_game:.1f} INTs/gm - Standard risk profile"

    else:
        return "⚪ STANDARD", f"TD Rate: {qb_td_rate:.1f}% vs Def INTs: {defense_ints_per_game:.1f}/gm"


# ============================================================================
# QB Efficiency Matchup Analysis Functions (Composite Metrics)
# ============================================================================

def calculate_qb_efficiency_matchup_stats(season, max_week=None):
    """
    Calculate QB Efficiency matchup statistics using composite metrics.

    Offensive Efficiency: Passing yards per game + INT rate (lower is better)
    Defensive Efficiency: Passing yards allowed + INTs forced + sacks

    Requires merging player_stats and pfr_advstats_def_week.

    Returns DataFrame with columns:
    - team: Team abbreviation
    - qb_pass_yards_per_game: Passing yards per game
    - qb_int_rate: INT rate percentage (INTs per attempt)
    - qb_efficiency_score: Composite efficiency score
    - defense_pass_yards_allowed: Passing yards allowed per game
    - defense_ints_per_game: INTs forced per game
    - defense_sacks_per_game: Sacks per game
    - defense_efficiency_score: Composite defensive efficiency score
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        week_filter = f"AND week <= {max_week}" if max_week else ""

        # Get offensive QB efficiency stats
        offense_query = f"""
        SELECT
            team,
            COUNT(DISTINCT week) as games,
            SUM(passing_yards) as total_pass_yards,
            SUM(passing_interceptions) as total_ints,
            SUM(attempts) as total_attempts
        FROM player_stats
        WHERE season = {season}
        {week_filter}
        AND position IN ('QB')
        GROUP BY team
        """

        offense_df = pd.read_sql_query(offense_query, conn)

        if not offense_df.empty:
            offense_df['qb_pass_yards_per_game'] = (
                offense_df['total_pass_yards'] / offense_df['games']
            ).round(1)
            offense_df['qb_int_rate'] = (
                (offense_df['total_ints'] / offense_df['total_attempts'].replace(0, 1)) * 100
            ).round(2)

            # Efficiency score: Higher yards good, lower INT rate good
            # Normalize: yards/game scaled, subtract INT penalty
            offense_df['qb_efficiency_score'] = (
                offense_df['qb_pass_yards_per_game'] - (offense_df['qb_int_rate'] * 20)
            ).round(1)

        # Get defensive passing yards allowed (from player_stats)
        defense_yards_query = f"""
        SELECT
            opponent_team as team,
            COUNT(DISTINCT week) as games,
            SUM(passing_yards) as total_pass_yards_allowed
        FROM player_stats
        WHERE season = {season}
        {week_filter}
        AND position IN ('QB')
        GROUP BY opponent_team
        """

        defense_yards_df = pd.read_sql_query(defense_yards_query, conn)

        if not defense_yards_df.empty:
            defense_yards_df['defense_pass_yards_allowed'] = (
                defense_yards_df['total_pass_yards_allowed'] / defense_yards_df['games']
            ).round(1)

        # Get defensive INT and sack stats (from pfr_advstats_def_week)
        defense_advanced_query = f"""
        SELECT
            team,
            COUNT(DISTINCT week) as games,
            SUM(CAST(def_ints AS REAL)) as total_ints,
            SUM(CAST(def_sacks AS REAL)) as total_sacks
        FROM pfr_advstats_def_week
        WHERE season = {season}
        {week_filter}
        GROUP BY team
        """

        defense_advanced_df = pd.read_sql_query(defense_advanced_query, conn)

        if not defense_advanced_df.empty:
            defense_advanced_df['defense_ints_per_game'] = (
                defense_advanced_df['total_ints'] / defense_advanced_df['games']
            ).round(2)
            defense_advanced_df['defense_sacks_per_game'] = (
                defense_advanced_df['total_sacks'] / defense_advanced_df['games']
            ).round(2)

        # Merge defensive stats
        defense_df = pd.merge(
            defense_yards_df[['team', 'defense_pass_yards_allowed']],
            defense_advanced_df[['team', 'defense_ints_per_game', 'defense_sacks_per_game']],
            on='team',
            how='outer'
        )

        if not defense_df.empty:
            # Defensive efficiency score: Lower yards allowed good, higher INTs/sacks good
            # Normalize: subtract yards allowed, add INT/sack bonuses
            defense_df['defense_efficiency_score'] = (
                -(defense_df['defense_pass_yards_allowed'].fillna(0)) +
                (defense_df['defense_ints_per_game'].fillna(0) * 30) +
                (defense_df['defense_sacks_per_game'].fillna(0) * 15)
            ).round(1)

        # Merge offense and defense stats
        result_df = pd.merge(
            offense_df[['team', 'qb_pass_yards_per_game', 'qb_int_rate', 'qb_efficiency_score']],
            defense_df[['team', 'defense_pass_yards_allowed', 'defense_ints_per_game',
                       'defense_sacks_per_game', 'defense_efficiency_score']],
            on='team',
            how='outer'
        )

        result_df = result_df.fillna(0)
        conn.close()
        return result_df

    except Exception as e:
        st.error(f"Error calculating QB Efficiency matchup stats: {e}")
        return pd.DataFrame()


def categorize_qb_efficiency(pass_yards_per_game, int_rate):
    """
    Categorize QB offensive efficiency based on yards and turnover safety.

    Benchmarks:
    - Elite Efficient (high yards, <2% INT rate): Best combo
    - Volume Chucker (high yards, >3% INT): High output but turnover prone
    - Safe Manager (moderate yards, <2% INT): Conservative but secure
    - Struggling (low yards or high INT rate): Below average
    """
    if pass_yards_per_game >= 260 and int_rate < 2.0:
        return "Elite Efficient"
    elif pass_yards_per_game >= 260 and int_rate >= 3.0:
        return "Volume Chucker"
    elif 200 <= pass_yards_per_game < 260 and int_rate < 2.0:
        return "Safe Manager"
    else:
        return "Struggling"


def categorize_defensive_passing_efficiency(yards_allowed, ints_per_game, sacks_per_game):
    """
    Categorize defensive passing efficiency based on composite metrics.

    Benchmarks:
    - Generous (high yards allowed, low pressure): Easy matchup
    - Opportunistic (moderate yards, high INTs): Creates turnovers
    - Dominant (low yards, high pressure): Elite pass defense
    - Average: Standard defense
    """
    if yards_allowed >= 260 and ints_per_game < 0.7 and sacks_per_game < 2.0:
        return "Generous"
    elif ints_per_game >= 1.0 or sacks_per_game >= 3.0:
        return "Opportunistic"
    elif yards_allowed < 220 and (ints_per_game >= 0.7 or sacks_per_game >= 2.0):
        return "Dominant"
    else:
        return "Average"


def generate_qb_efficiency_storyline(qb_yards, qb_int_rate, def_yards_allowed, def_ints, def_sacks):
    """
    Generate narrative storyline based on composite QB efficiency vs defensive efficiency.

    Favorable Matchups (High ceiling, low risk):
    - Elite Efficient vs Generous defense
    - Safe Manager vs Generous defense
    - Elite Efficient vs Average defense

    Unfavorable Matchups (Low ceiling or high risk):
    - Any QB vs Dominant defense
    - Volume Chucker vs Opportunistic defense (turnover trap)
    - Struggling vs any strong defense
    """

    off_category = categorize_qb_efficiency(qb_yards, qb_int_rate)
    def_category = categorize_defensive_passing_efficiency(def_yards_allowed, def_ints, def_sacks)

    # EXPLOIT MATCHUPS (Best scenarios)
    if off_category == "Elite Efficient" and def_category == "Generous":
        return "💎 SAFE CEILING SPOT", f"Elite efficiency ({qb_yards:.0f} yds, {qb_int_rate:.1f}% INT) vs generous D ({def_yards_allowed:.0f} yds, {def_ints:.1f} INTs, {def_sacks:.1f} sacks) - Smash play"

    elif off_category == "Safe Manager" and def_category == "Generous":
        return "✅ VOLUME OPPORTUNITY", f"Conservative QB ({qb_yards:.0f} yds, {qb_int_rate:.1f}% INT) vs soft coverage ({def_yards_allowed:.0f} yds allowed) - Safe volume boost"

    elif off_category == "Elite Efficient" and def_category == "Average":
        return "🔥 FAVORABLE", f"Elite QB ({qb_yards:.0f} yds, {qb_int_rate:.1f}% INT) vs average D - Reliable production"

    # RISKY/TOUGH MATCHUPS
    elif def_category == "Dominant":
        return "🛡️ DEFENSIVE WALL", f"QB faces dominant D ({def_yards_allowed:.0f} yds allowed, {def_ints:.1f} INTs, {def_sacks:.1f} sacks) - Limited ceiling"

    elif off_category == "Volume Chucker" and def_category == "Opportunistic":
        return "⚠️ TURNOVER TRAP", f"Turnover-prone QB ({qb_int_rate:.1f}% INT) vs aggressive D ({def_ints:.1f} INTs/gm) - Multiple INT risk"

    elif off_category == "Struggling":
        return "🚫 AVOID", f"Struggling QB ({qb_yards:.0f} yds, {qb_int_rate:.1f}% INT) - Low floor and ceiling"

    # NEUTRAL MATCHUPS
    elif off_category in ["Safe Manager", "Elite Efficient"]:
        return "⚖️ BALANCED", f"QB efficiency vs defense - Standard matchup ({qb_yards:.0f} yds/gm, {qb_int_rate:.1f}% INT)"

    else:
        return "⚪ STANDARD", f"Yards: {qb_yards:.0f}/gm, INT: {qb_int_rate:.1f}% vs Def: {def_yards_allowed:.0f} yds, {def_ints:.1f} INTs, {def_sacks:.1f} sacks"


# ============================================================================
# Comprehensive Matchup Recommendation System
# ============================================================================

def score_storyline(label):
    """
    Convert storyline label to numeric score (1-5 scale).

    Used for weighted averaging across multiple matchup dimensions.

    Args:
        label (str): Storyline label like "🎯 TD SMASH SPOT" or "🛡️ TOUGH"

    Returns:
        int: Score from 1 (worst) to 5 (best)
    """
    label_upper = label.upper()

    # Elite/Exploit matchups (5 points)
    if any(keyword in label_upper for keyword in ['EXPLOIT', 'SMASH', 'ELITE', 'EXPLOSION', 'SAFE CEILING']):
        return 5

    # Favorable/Value matchups (4 points)
    elif any(keyword in label_upper for keyword in ['FAVORABLE', 'SOLID', 'VALUE', 'SNEAKY', 'OPPORTUNITY', 'SPIKE']):
        return 4

    # Neutral/Balanced matchups (3 points)
    elif any(keyword in label_upper for keyword in ['BALANCED', 'NEUTRAL', 'STANDARD', 'AVERAGE']):
        return 3

    # Tough/Risky matchups (2 points)
    elif any(keyword in label_upper for keyword in ['TOUGH', 'RISKY', 'LIMITED', 'GRIND', 'BOOM-BUST']):
        return 2

    # Avoid/Danger matchups (1 point)
    elif any(keyword in label_upper for keyword in ['AVOID', 'DANGER', 'WALL', 'TRAP', 'YARDAGE ONLY', 'GROUND GAME']):
        return 1

    # Default to neutral if no keywords match
    else:
        return 3


def get_top_players(team, position, season, max_week=None):
    """
    Get top 1-2 players for a team/position from rosters or player_stats table.

    Args:
        team (str): Team abbreviation
        position (str): 'QB', 'RB', 'WR', or 'TE'
        season (int): Season year
        max_week (int, optional): Max week to consider

    Returns:
        list: List of player display names (max 2 players)
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # Try rosters table first
        query_roster = f"""
        SELECT DISTINCT player_display_name
        FROM rosters
        WHERE team = ?
        AND position = ?
        AND season = ?
        LIMIT 2
        """

        df = pd.read_sql_query(query_roster, conn, params=(team, position, season))

        # If rosters table is empty, try player_stats table with actual usage
        if df.empty:
            week_filter = f"AND week <= {max_week}" if max_week else ""

            # Get players who actually played, sorted by total usage
            if position == 'QB':
                stat_col = 'passing_attempts'
            elif position == 'RB':
                stat_col = 'rushing_attempts'
            else:  # WR, TE
                stat_col = 'targets'

            query_stats = f"""
            SELECT player_display_name, SUM({stat_col}) as total_usage
            FROM player_stats
            WHERE team = ?
            AND position = ?
            AND season = ?
            {week_filter}
            GROUP BY player_display_name
            ORDER BY total_usage DESC
            LIMIT 2
            """

            df = pd.read_sql_query(query_stats, conn, params=(team, position, season))

        conn.close()

        if df.empty:
            return []

        # Return just the names (drop total_usage column if it exists)
        if 'player_display_name' in df.columns:
            return df['player_display_name'].tolist()
        else:
            return []

    except Exception as e:
        # Log error for debugging but don't crash
        import traceback
        print(f"Error in get_top_players({team}, {position}, {season}): {e}")
        print(traceback.format_exc())
        return []


def synthesize_qb_matchup(team, opponent, season, max_week=None):
    """
    Synthesize QB matchup by combining multiple storylines.

    Combines:
    - QB Pressure (sacks/pressure rate)
    - Passing TD matchup
    - Passing Yards matchup
    - QB Efficiency vs INT Risk

    Args:
        team (str): Offensive team
        opponent (str): Defensive team
        season (int): Season year
        max_week (int, optional): Max week to consider

    Returns:
        dict with keys:
        - 'rating': str ('ELITE', 'FAVORABLE', 'NEUTRAL', 'AVOID')
        - 'summary': str (narrative summary)
        - 'players': list of player names
        - 'key_factors': list of (storyline_label, score) tuples
        - 'weighted_score': float (for sorting)
    """
    try:
        # Calculate all QB-related matchup stats
        qb_pressure_stats = calculate_qb_pressure_stats(season, max_week)
        def_pressure_stats = calculate_defense_pressure_stats(season, max_week)
        td_stats = calculate_passing_td_matchup_stats(season, max_week)
        yards_stats = calculate_passing_yards_matchup_stats(season, max_week)
        efficiency_stats = calculate_qb_td_rate_vs_int_stats(season, max_week)

        # Extract team vs opponent data
        # For QB pressure: Get starting QB (player-level data)
        team_qbs = qb_pressure_stats[qb_pressure_stats['team'] == team]
        team_qb = team_qbs.sort_values('games', ascending=False).iloc[0] if not team_qbs.empty else None

        # For defense pressure: Get team-level data
        opp_def = def_pressure_stats[def_pressure_stats['team'] == opponent]

        team_td = td_stats[td_stats['team'] == team]
        opp_td = td_stats[td_stats['team'] == opponent]

        team_yards = yards_stats[yards_stats['team'] == team]
        opp_yards = yards_stats[yards_stats['team'] == opponent]

        team_eff = efficiency_stats[efficiency_stats['team'] == team]
        opp_eff = efficiency_stats[efficiency_stats['team'] == opponent]

        # Generate individual storylines
        storylines = {}

        if team_qb is not None and not opp_def.empty:
            pressure_label, _ = generate_qb_pressure_storyline(
                team_qb['pressure_rate'],
                opp_def.iloc[0]['pressures_per_game'],
                team_qb['sacks_per_pressure']
            )
            storylines['pressure'] = (pressure_label, score_storyline(pressure_label))

        if not team_td.empty and not opp_td.empty:
            td_label, _ = generate_passing_td_storyline(
                team_td.iloc[0]['offense_pass_tds'],
                opp_td.iloc[0]['defense_pass_tds_allowed']
            )
            storylines['td'] = (td_label, score_storyline(td_label))

        if not team_yards.empty and not opp_yards.empty:
            yards_label, _ = generate_passing_yards_storyline(
                team_yards.iloc[0]['offense_pass_yards'],
                opp_yards.iloc[0]['defense_pass_yards_allowed']
            )
            storylines['yards'] = (yards_label, score_storyline(yards_label))

        if not team_eff.empty and not opp_eff.empty:
            eff_label, _ = generate_qb_td_rate_vs_int_storyline(
                team_eff.iloc[0]['qb_td_rate'],
                opp_eff.iloc[0]['defense_ints_per_game']
            )
            storylines['efficiency'] = (eff_label, score_storyline(eff_label))

        # If no data, return default
        if not storylines:
            return {
                'rating': 'N/A',
                'summary': 'Insufficient data for QB matchup analysis',
                'players': [],
                'key_factors': [],
                'weighted_score': 0
            }

        # Calculate weighted score
        weights = {
            'pressure': 0.20,
            'td': 0.35,
            'yards': 0.25,
            'efficiency': 0.20
        }

        total_weight = sum(weights[k] for k in storylines.keys())
        weighted_score = sum(storylines[k][1] * weights.get(k, 0) for k in storylines.keys()) / total_weight

        # Determine overall rating (5-level system)
        if weighted_score >= 4.25:
            rating = 'SMASH'
        elif weighted_score >= 3.75:
            rating = 'GOOD'
        elif weighted_score >= 2.75:
            rating = 'NEUTRAL'
        elif weighted_score >= 2.0:
            rating = 'RISKY'
        else:
            rating = 'AVOID'

        # Get top QB players
        players = get_top_players(team, 'QB', season, max_week)

        # Generate narrative summary (simplified for now - will enhance in Phase 2)
        key_factors = [(label, score) for label, score in storylines.values()]
        summary = f"{rating} QB matchup vs {opponent}. " + ", ".join([label for label, _ in key_factors[:2]])

        return {
            'rating': rating,
            'summary': summary,
            'players': players,
            'key_factors': key_factors,
            'weighted_score': weighted_score
        }

    except Exception as e:
        return {
            'rating': 'ERROR',
            'summary': f'Error analyzing QB matchup: {str(e)}',
            'players': [],
            'key_factors': [],
            'weighted_score': 0
        }


def synthesize_rb_matchup(team, opponent, season, max_week=None):
    """
    Synthesize RB matchup by combining rushing yards, rushing TD, and receiving storylines.

    Combines:
    - Rushing Yards Matchup
    - Rushing TD Efficiency
    - RB Pass-Catching (PPR value)

    Args:
        team (str): Offensive team
        opponent (str): Defensive team
        season (int): Season year
        max_week (int, optional): Max week to consider

    Returns:
        dict with same structure as synthesize_qb_matchup()
    """
    try:
        # Calculate RB-related matchup stats
        rush_yards_stats = calculate_rushing_yards_matchup_stats(season, max_week)
        rush_td_stats = calculate_rushing_td_matchup_stats(season, max_week)
        rb_rec_stats = calculate_rb_receiving_matchup_stats(season, max_week)

        # Extract team vs opponent data
        team_rush_yds = rush_yards_stats[rush_yards_stats['team'] == team]
        opp_rush_yds = rush_yards_stats[rush_yards_stats['team'] == opponent]

        team_rush_td = rush_td_stats[rush_td_stats['team'] == team]
        opp_rush_td = rush_td_stats[rush_td_stats['team'] == opponent]

        team_rec = rb_rec_stats[rb_rec_stats['team'] == team]
        opp_rec = rb_rec_stats[rb_rec_stats['team'] == opponent]

        # Generate individual storylines
        storylines = {}

        if not team_rush_yds.empty and not opp_rush_yds.empty:
            rush_yds_label, _ = generate_rushing_yards_storyline(
                team_rush_yds.iloc[0]['offense_rush_yards'],
                opp_rush_yds.iloc[0]['defense_rush_yards_allowed']
            )
            storylines['rushing_yards'] = (rush_yds_label, score_storyline(rush_yds_label))

        if not team_rush_td.empty and not opp_rush_td.empty:
            rush_td_label, _ = generate_rushing_td_storyline(
                team_rush_td.iloc[0]['offense_rush_tds'],
                opp_rush_td.iloc[0]['defense_rush_tds_allowed']
            )
            storylines['rushing_td'] = (rush_td_label, score_storyline(rush_td_label))

        if not team_rec.empty and not opp_rec.empty:
            rec_label, _ = generate_rb_receiving_storyline(
                team_rec.iloc[0]['rb_targets_per_game'],
                opp_rec.iloc[0]['defense_rec_to_rb']
            )
            storylines['receiving'] = (rec_label, score_storyline(rec_label))

        # If no data, return default
        if not storylines:
            return {
                'rating': 'N/A',
                'summary': 'Insufficient data for RB matchup analysis',
                'players': [],
                'key_factors': [],
                'weighted_score': 0
            }

        # Calculate weighted score (Yards=40%, TDs=35%, Receiving=25%)
        weights = {
            'rushing_yards': 0.40,
            'rushing_td': 0.35,
            'receiving': 0.25
        }

        total_weight = sum(weights[k] for k in storylines.keys())
        weighted_score = sum(storylines[k][1] * weights.get(k, 0) for k in storylines.keys()) / total_weight

        # Determine overall rating (5-level system)
        if weighted_score >= 4.25:
            rating = 'SMASH'
        elif weighted_score >= 3.75:
            rating = 'GOOD'
        elif weighted_score >= 2.75:
            rating = 'NEUTRAL'
        elif weighted_score >= 2.0:
            rating = 'RISKY'
        else:
            rating = 'AVOID'

        # Get top RB players
        players = get_top_players(team, 'RB', season, max_week)

        # Generate narrative summary
        key_factors = [(label, score) for label, score in storylines.values()]
        summary = f"{rating} RB matchup vs {opponent}. " + ", ".join([label for label, _ in key_factors])

        return {
            'rating': rating,
            'summary': summary,
            'players': players,
            'key_factors': key_factors,
            'weighted_score': weighted_score
        }

    except Exception as e:
        return {
            'rating': 'ERROR',
            'summary': f'Error analyzing RB matchup: {str(e)}',
            'players': [],
            'key_factors': [],
            'weighted_score': 0
        }


def synthesize_wr_matchup(team, opponent, season, max_week=None):
    """
    Synthesize WR/TE matchup using Air Yards vs YAC storyline.

    Args:
        team (str): Offensive team
        opponent (str): Defensive team
        season (int): Season year
        max_week (int, optional): Max week to consider

    Returns:
        dict with same structure as synthesize_qb_matchup()
    """
    try:
        # Calculate WR-related matchup stats
        air_yac_stats = calculate_air_yac_matchup_stats(season, max_week)

        # Extract team vs opponent data
        team_data = air_yac_stats[air_yac_stats['team'] == team]
        opp_data = air_yac_stats[air_yac_stats['team'] == opponent]

        # Generate storyline
        if not team_data.empty and not opp_data.empty:
            air_yac_label, _ = generate_air_yac_storyline(
                team_data.iloc[0]['offense_air_share'],
                opp_data.iloc[0]['defense_yac_share'],
                opp_data.iloc[0]['defense_air_share'],
                team_data.iloc[0]['offense_yac_share']
            )
            score = score_storyline(air_yac_label)

            # Determine rating based on single storyline score (5-level system)
            if score >= 4.25:
                rating = 'SMASH'
            elif score >= 3.75:
                rating = 'GOOD'
            elif score >= 2.75:
                rating = 'NEUTRAL'
            elif score >= 2.0:
                rating = 'RISKY'
            else:
                rating = 'AVOID'

            # Get top WR players
            players = get_top_players(team, 'WR', season, max_week)

            # Generate narrative summary
            summary = f"{rating} WR matchup vs {opponent}. {air_yac_label}"

            return {
                'rating': rating,
                'summary': summary,
                'players': players,
                'key_factors': [(air_yac_label, score)],
                'weighted_score': float(score)
            }
        else:
            return {
                'rating': 'N/A',
                'summary': 'Insufficient data for WR matchup analysis',
                'players': [],
                'key_factors': [],
                'weighted_score': 0
            }

    except Exception as e:
        return {
            'rating': 'ERROR',
            'summary': f'Error analyzing WR matchup: {str(e)}',
            'players': [],
            'key_factors': [],
            'weighted_score': 0
        }


def format_player_list(players):
    """
    Format player names for display.

    Args:
        players (list): List of player display names

    Returns:
        str: Formatted string like "(Lamar Jackson)" or "(D.Henry, T.Pollard)"
    """
    if not players:
        return ""
    elif len(players) == 1:
        return f"({players[0]})"
    else:
        # Abbreviate first names for space
        short_names = [p.split()[0][0] + "." + " ".join(p.split()[1:]) if len(p.split()) > 1 else p for p in players[:2]]
        return f"({', '.join(short_names)})"


def format_rating_badge(rating):
    """
    Return colored emoji badge for rating.

    Args:
        rating (str): 'SMASH', 'GOOD', 'NEUTRAL', 'RISKY', 'AVOID', 'N/A', or 'ERROR'

    Returns:
        str: Emoji + rating text
    """
    badges = {
        'SMASH': '🟢 SMASH',
        'GOOD': '🟡 GOOD',
        'NEUTRAL': '⚪ NEUTRAL',
        'RISKY': '🟠 RISKY',
        'AVOID': '🔴 AVOID',
        'N/A': '⚫ N/A',
        'ERROR': '❌ ERROR',
        # Legacy support for old ratings
        'ELITE': '🟢 SMASH',
        'FAVORABLE': '🟡 GOOD'
    }
    return badges.get(rating, '⚪ ' + rating)


def render_matchup_recommendations_table(season, week, upcoming_games_df):
    """
    Render comprehensive matchup recommendations table.

    Synthesizes all storylines into position-specific ratings displayed in a table.

    Args:
        season (int): Season year
        week (int): Week number
        upcoming_games_df (DataFrame): Upcoming games with home_team, away_team columns
    """
    if upcoming_games_df is None or upcoming_games_df.empty:
        st.info("No upcoming games available for matchup recommendations.")
        return

    try:
        summary_data = []

        for _, game in upcoming_games_df.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']

            # Synthesize matchups for both teams
            home_qb = synthesize_qb_matchup(home_team, away_team, season, week)
            home_rb = synthesize_rb_matchup(home_team, away_team, season, week)
            home_wr = synthesize_wr_matchup(home_team, away_team, season, week)

            away_qb = synthesize_qb_matchup(away_team, home_team, season, week)
            away_rb = synthesize_rb_matchup(away_team, home_team, season, week)
            away_wr = synthesize_wr_matchup(away_team, home_team, season, week)

            # Determine best opportunity for home team
            home_scores = {
                'QB': home_qb['weighted_score'],
                'RB': home_rb['weighted_score'],
                'WR': home_wr['weighted_score']
            }
            home_best_pos = max(home_scores, key=home_scores.get) if max(home_scores.values()) > 0 else 'N/A'
            home_best_score = home_scores.get(home_best_pos, 0)

            # Show scores if positions are close (within 0.3 of each other)
            sorted_scores = sorted(home_scores.items(), key=lambda x: x[1], reverse=True)
            if home_best_pos != 'N/A' and len(sorted_scores) > 1 and (sorted_scores[0][1] - sorted_scores[1][1]) < 0.3:
                home_best = f"{home_best_pos} ({home_best_score:.1f})"
            else:
                home_best = home_best_pos

            # Select summary based on best position (use home_best_pos without score)
            home_position_data = {'QB': home_qb, 'RB': home_rb, 'WR': home_wr}
            home_best_matchup = home_position_data.get(home_best_pos, home_qb)  # Fallback to QB if N/A
            home_summary = home_best_matchup['summary'][:80] + "..." if len(home_best_matchup['summary']) > 80 else home_best_matchup['summary']

            # Determine best opportunity for away team
            away_scores = {
                'QB': away_qb['weighted_score'],
                'RB': away_rb['weighted_score'],
                'WR': away_wr['weighted_score']
            }
            away_best_pos = max(away_scores, key=away_scores.get) if max(away_scores.values()) > 0 else 'N/A'
            away_best_score = away_scores.get(away_best_pos, 0)

            # Show scores if positions are close (within 0.3 of each other)
            sorted_away_scores = sorted(away_scores.items(), key=lambda x: x[1], reverse=True)
            if away_best_pos != 'N/A' and len(sorted_away_scores) > 1 and (sorted_away_scores[0][1] - sorted_away_scores[1][1]) < 0.3:
                away_best = f"{away_best_pos} ({away_best_score:.1f})"
            else:
                away_best = away_best_pos

            # Select summary based on best position (use away_best_pos without score)
            away_position_data = {'QB': away_qb, 'RB': away_rb, 'WR': away_wr}
            away_best_matchup = away_position_data.get(away_best_pos, away_qb)  # Fallback to QB if N/A
            away_summary = away_best_matchup['summary'][:80] + "..." if len(away_best_matchup['summary']) > 80 else away_best_matchup['summary']

            # Add home team row
            summary_data.append({
                'Game': f"{away_team} @ {home_team}",
                'Team': home_team,
                'QB': format_rating_badge(home_qb['rating']) + " " + format_player_list(home_qb['players']),
                'RB': format_rating_badge(home_rb['rating']) + " " + format_player_list(home_rb['players']),
                'WR': format_rating_badge(home_wr['rating']) + " " + format_player_list(home_wr['players']),
                'Best': home_best,
                'Summary': home_summary
            })

            # Add away team row
            summary_data.append({
                'Game': f"{away_team} @ {home_team}",
                'Team': away_team,
                'QB': format_rating_badge(away_qb['rating']) + " " + format_player_list(away_qb['players']),
                'RB': format_rating_badge(away_rb['rating']) + " " + format_player_list(away_rb['players']),
                'WR': format_rating_badge(away_wr['rating']) + " " + format_player_list(away_wr['players']),
                'Best': away_best,
                'Summary': away_summary
            })

        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)

        # Display table
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption("💡 **Tip:** Ratings combine all relevant matchup storylines for each position using weighted scoring.")

    except Exception as e:
        st.error(f"Error rendering matchup recommendations: {e}")
        import traceback
        st.error(traceback.format_exc())


# ============================================================================
# Projection Accuracy Tracking Functions
# ============================================================================

def save_projections_for_week(season, week):
    """
    Save current projections to database before games start.
    Returns count of projections saved.
    """
    try:
        # Initialize table if needed
        init_projection_accuracy_table()

        # Get upcoming games for this week
        conn = sqlite3.connect(DB_PATH)
        query_str = """
            SELECT DISTINCT home_team, away_team
            FROM upcoming_games
            WHERE season = ? AND week = ?
        """
        games_df = pd.read_sql_query(query_str, conn, params=(season, week))
        conn.close()

        if games_df.empty:
            return 0

        # Get teams playing this week
        teams_playing = list(set(games_df['home_team'].tolist() + games_df['away_team'].tolist()))

        # Generate projections
        projections = generate_player_projections(season, week, teams_playing)

        # Save projections to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        count = 0

        # Process each position
        for position in ['QB', 'RB', 'WR', 'TE']:
            if position not in projections or projections[position].empty:
                continue

            df = projections[position]

            for _, row in df.iterrows():
                try:
                    # Get matchup rating
                    matchup_rating, _ = get_matchup_rating(row.get('Multiplier', 1.0))

                    # Determine projected yards based on position
                    if position == 'QB':
                        projected_yds = row['Projected Yds']
                        avg_yds = row.get('Avg Yds/Game', 0)
                        median_yds = row.get('Median Pass Yds', 0)
                    elif position == 'RB':
                        projected_yds = row['Projected Total']
                        avg_yds = row.get('Avg Yds/Game', 0)
                        median_yds = row.get('Total Median', 0)
                    else:  # WR, TE
                        projected_yds = row['Projected Yds']
                        avg_yds = row.get('Avg Yds/Game', 0)
                        median_yds = row.get('Median Rec Yds', 0) if position == 'WR' else row.get('Median Rec Yds', 0)

                    # Insert or replace projection
                    cursor.execute("""
                        INSERT OR REPLACE INTO projection_accuracy (
                            player_name, team_abbr, opponent_abbr, season, week,
                            position, projected_yds, multiplier, matchup_rating,
                            avg_yds_game, median_yds, games_played
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['Player'],
                        row['Team'],
                        row['Opponent'],
                        season,
                        week,
                        position,
                        float(projected_yds),
                        float(row.get('Multiplier', 1.0)),
                        matchup_rating,
                        float(avg_yds),
                        float(median_yds),
                        float(row.get('Games', 0))
                    ))
                    count += 1
                except Exception as e:
                    logging.error(f"Error saving projection for {row.get('Player', 'Unknown')}: {e}")
                    continue

        conn.commit()
        conn.close()

        # Upload to GCS
        upload_db_to_gcs()

        return count

    except Exception as e:
        logging.error(f"Error in save_projections_for_week: {e}")
        return 0


def update_projection_actuals(season, week):
    """
    Update projections with actual results after games complete.
    Returns count of projections updated.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get projections that need updating
        cursor.execute("""
            SELECT projection_id, player_name, team_abbr, season, week, position, projected_yds
            FROM projection_accuracy
            WHERE season = ? AND week = ? AND actual_yds IS NULL
        """, (season, week))

        projections = cursor.fetchall()
        count = 0

        for proj in projections:
            proj_id, player_name, team_abbr, seas, wk, position, projected_yds = proj

            try:
                # Query actual performance based on position
                if position == 'QB':
                    query_str = """
                        SELECT COALESCE(SUM(pass_yds), 0) as actual_yds
                        FROM player_box_score
                        WHERE player = ? AND team = ? AND season = ? AND week = ?
                    """
                elif position == 'RB':
                    query_str = """
                        SELECT COALESCE(SUM(rush_yds), 0) + COALESCE(SUM(rec_yds), 0) as actual_yds
                        FROM player_box_score
                        WHERE player = ? AND team = ? AND season = ? AND week = ?
                    """
                elif position in ['WR', 'TE']:
                    query_str = """
                        SELECT COALESCE(SUM(rec_yds), 0) as actual_yds
                        FROM player_box_score
                        WHERE player = ? AND team = ? AND season = ? AND week = ?
                    """
                else:
                    continue

                cursor.execute(query_str, (player_name, team_abbr, seas, wk))
                result = cursor.fetchone()

                if result and result[0] is not None:
                    actual_yds = float(result[0])

                    # Calculate accuracy metrics
                    variance = actual_yds - projected_yds
                    abs_error = abs(variance)
                    pct_error = (variance / actual_yds * 100) if actual_yds > 0 else 0

                    # Update the projection
                    cursor.execute("""
                        UPDATE projection_accuracy
                        SET actual_yds = ?,
                            variance = ?,
                            abs_error = ?,
                            pct_error = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE projection_id = ?
                    """, (actual_yds, variance, abs_error, pct_error, proj_id))

                    count += 1

            except Exception as e:
                logging.error(f"Error updating actual for {player_name}: {e}")
                continue

        conn.commit()
        conn.close()

        # Upload to GCS
        upload_db_to_gcs()

        return count

    except Exception as e:
        logging.error(f"Error in update_projection_actuals: {e}")
        return 0


def get_projection_accuracy_data(season=None, week=None, position=None):
    """
    Query projection_accuracy table with filters.
    Returns DataFrame for display and visualization.
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # Build query with filters
        where_clauses = ["actual_yds IS NOT NULL"]
        params = []

        if season:
            where_clauses.append("season = ?")
            params.append(season)
        if week:
            where_clauses.append("week = ?")
            params.append(week)
        if position:
            where_clauses.append("position = ?")
            params.append(position)

        where_str = " AND ".join(where_clauses)

        query_str = f"""
            SELECT
                player_name, team_abbr, opponent_abbr, season, week, position,
                projected_yds, actual_yds, multiplier, matchup_rating,
                avg_yds_game, median_yds, games_played,
                variance, abs_error, pct_error,
                created_at, updated_at
            FROM projection_accuracy
            WHERE {where_str}
            ORDER BY season DESC, week DESC, abs_error ASC
        """

        df = pd.read_sql_query(query_str, conn, params=params)
        conn.close()

        return df

    except Exception as e:
        logging.error(f"Error in get_projection_accuracy_data: {e}")
        return pd.DataFrame()


def get_accuracy_metrics(season=None, week=None, position=None, matchup_rating=None):
    """
    Calculate aggregate accuracy statistics with filters.
    Returns dict with MAE, RMSE, bias, hit rates, etc.
    """
    try:
        df = get_projection_accuracy_data(season, week, position)

        if matchup_rating:
            df = df[df['matchup_rating'] == matchup_rating]

        if df.empty:
            return {}

        # Calculate metrics
        mae = df['abs_error'].mean()
        rmse = np.sqrt((df['variance'] ** 2).mean())
        bias = df['variance'].mean()

        # Hit rates
        hit_10 = (df['abs_error'] <= 10).sum() / len(df) * 100
        hit_20 = (df['abs_error'] <= 20).sum() / len(df) * 100
        hit_30 = (df['abs_error'] <= 30).sum() / len(df) * 100

        # Over/Under
        over = (df['variance'] > 0).sum()
        under = (df['variance'] < 0).sum()

        # Correlation
        correlation = df['projected_yds'].corr(df['actual_yds'])

        return {
            'count': len(df),
            'mae': mae,
            'rmse': rmse,
            'bias': bias,
            'hit_rate_10': hit_10,
            'hit_rate_20': hit_20,
            'hit_rate_30': hit_30,
            'over_count': over,
            'under_count': under,
            'correlation': correlation,
            'avg_projected': df['projected_yds'].mean(),
            'avg_actual': df['actual_yds'].mean()
        }

    except Exception as e:
        logging.error(f"Error in get_accuracy_metrics: {e}")
        return {}


def get_player_accuracy_leaderboard(position=None, min_projections=5):
    """
    Rank players by projection accuracy.
    Returns DataFrame sorted by lowest MAE.
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        where_clause = "WHERE actual_yds IS NOT NULL"
        params = []

        if position:
            where_clause += " AND position = ?"
            params.append(position)

        query_str = f"""
            SELECT
                player_name,
                position,
                COUNT(*) as projection_count,
                AVG(projected_yds) as avg_projected,
                AVG(actual_yds) as avg_actual,
                AVG(abs_error) as mae,
                AVG(CASE WHEN variance > 0 THEN 1 ELSE 0 END) * 100 as over_pct
            FROM projection_accuracy
            {where_clause}
            GROUP BY player_name, position
            HAVING projection_count >= ?
            ORDER BY mae ASC
        """

        params.append(min_projections)
        df = pd.read_sql_query(query_str, conn, params=params)
        conn.close()

        # Calculate accuracy percentage
        if not df.empty:
            df['accuracy_pct'] = 100 - (df['mae'] / df['avg_actual'] * 100)
            df['accuracy_pct'] = df['accuracy_pct'].clip(lower=0, upper=100)

        return df

    except Exception as e:
        logging.error(f"Error in get_player_accuracy_leaderboard: {e}")
        return pd.DataFrame()


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

        # Skip games without scores (not yet played)
        if team_score is None or opp_score is None or pd.isna(team_score) or pd.isna(opp_score):
            continue

        margin = int(team_score) - int(opp_score)
        margins.append(margin)

    # If we don't have enough completed games after filtering, return default
    if len(margins) < 3:
        return {
            'recent_form_score': 0.0,
            'weighted_margin': 0.0,
            'last_3_margins': margins,
            'games_available': len(margins)
        }

    # Calculate weighted margin
    weighted_margin = sum(m * w for m, w in zip(margins, weights))

    # Calculate recent form score: (weighted_margin / 7) × 10
    recent_form_score = (weighted_margin / 7) * 10

    return {
        'recent_form_score': recent_form_score,
        'weighted_margin': weighted_margin,
        'last_3_margins': margins,
        'games_available': len(margins)
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
    # Get opponent teams that played against this team
    sql = """
        SELECT
            SUM(CASE WHEN pass_yds >= 300 THEN 1 ELSE 0 END) as big_pass_games_allowed,
            SUM(CASE WHEN rush_yds >= 100 THEN 1 ELSE 0 END) as big_rush_games_allowed,
            SUM(CASE WHEN rec_yds >= 100 THEN 1 ELSE 0 END) as big_rec_games_allowed,
            COUNT(DISTINCT week) as games_played
        FROM player_box_score
        WHERE opponent = ? AND season = ?
    """
    params = [team_abbr, season]

    if week:
        sql += " AND week <= ?"
        params.append(week)

    df = query(sql, tuple(params))

    if df.empty or df.iloc[0]['games_played'] == 0:
        return 0.05  # League average

    row = df.iloc[0]
    games = row['games_played']
    total_big_plays_allowed = row['big_pass_games_allowed'] + row['big_rush_games_allowed'] + row['big_rec_games_allowed']

    return total_big_plays_allowed / games if games > 0 else 0.05


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
    import numpy as np
    league_stats = {}

    for metric_name, values in metrics.items():
        if len(values) > 1:
            # Convert all values to float to avoid 'float has no numerator' errors
            # Use numpy to avoid Python statistics module issues with float types
            float_values = np.array(values, dtype=np.float64)
            mean_val = float(np.mean(float_values))
            std_val = float(np.std(float_values, ddof=1)) if len(float_values) > 1 else 1.0

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
        AND g.home_score IS NOT NULL
        AND g.away_score IS NOT NULL
    """
    params = [season, team_abbr, team_abbr]
    if week:
        sql += " AND g.week <= ?"
        params.append(week)

    games = query(sql, tuple(params))

    if games.empty:
        return {
            'pts_for': 0.0,
            'pts_against': 0.0,
            'pt_diff': 0.0,
            'pt_diff_per_game': 0.0,
            'avg_pts_for': 22.0,  # NFL average fallback
            'avg_pts_against': 22.0
        }

    total_pts_for = 0
    total_pts_against = 0

    for _, row in games.iterrows():
        is_home = row['home_team_abbr'] == team_abbr
        team_score = row['home_score'] if is_home else row['away_score']
        opp_score = row['away_score'] if is_home else row['home_score']

        # Skip if scores are None (shouldn't happen with SQL filter, but defensive)
        if team_score is None or opp_score is None:
            continue

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
        'avg_pts_for': float(total_pts_for / games_played) if games_played > 0 else 22.0,
        'avg_pts_against': float(total_pts_against / games_played) if games_played > 0 else 22.0
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
    """
    Calculate explosive play proxies from player box scores (20+ yard plays).

    Uses heuristic: 40+ yard pass/rush attempts or 100+ yard receiving games
    as indicators of explosive plays.
    """
    sql = """
        SELECT
            SUM(CASE WHEN pass_yds >= 300 THEN 1 ELSE 0 END) as big_pass_games,
            SUM(CASE WHEN rush_yds >= 100 THEN 1 ELSE 0 END) as big_rush_games,
            SUM(CASE WHEN rec_yds >= 100 THEN 1 ELSE 0 END) as big_rec_games,
            COUNT(DISTINCT week) as games_played
        FROM player_box_score
        WHERE team = ? AND season = ?
    """
    params = [team_abbr, season]

    if week:
        sql += " AND week <= ?"
        params.append(week)

    df = query(sql, tuple(params))

    if df.empty or df.iloc[0]['games_played'] == 0:
        return {
            'explosive_play_rate': 0.0,
            'big_pass_games': 0,
            'big_rush_games': 0,
            'big_rec_games': 0,
            'games': 0
        }

    row = df.iloc[0]
    games = row['games_played']
    total_big_plays = row['big_pass_games'] + row['big_rush_games'] + row['big_rec_games']

    return {
        'explosive_play_rate': total_big_plays / games if games > 0 else 0.0,
        'big_pass_games': row['big_pass_games'],
        'big_rush_games': row['big_rush_games'],
        'big_rec_games': row['big_rec_games'],
        'games': games
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
            "Historical Trends",
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
            "Projection Analytics",
            "Database Manager",
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

    # Quick matchup selector for specific week
    st.subheader("🗓️ Quick Week Matchup Selector")

    # Initialize session state for team selections if not exists
    if 'comparison_team1_idx' not in st.session_state:
        st.session_state.comparison_team1_idx = 0
    if 'comparison_team2_idx' not in st.session_state:
        st.session_state.comparison_team2_idx = min(1, len(teams)-1)

    # Get available weeks and matchups from schedule
    try:
        conn = sqlite3.connect(DB_PATH)
        weeks_query = f"""
            SELECT DISTINCT week
            FROM schedules
            WHERE season = {season} AND game_type = 'REG'
            ORDER BY week
        """
        weeks_df = pd.read_sql_query(weeks_query, conn)
        available_weeks = weeks_df['week'].tolist() if not weeks_df.empty else []

        if available_weeks:
            col_week, col_matchup, col_button = st.columns([1, 3, 1])

            with col_week:
                selected_matchup_week = st.selectbox(
                    "Select Week",
                    ["Manual Selection"] + available_weeks,
                    key="quick_matchup_week",
                    help="Choose a week to see matchups, or 'Manual Selection' to pick teams manually"
                )

            # If a week is selected, show matchups for that week
            if selected_matchup_week != "Manual Selection":
                matchups_query = f"""
                    SELECT home_team, away_team, game_id, gameday
                    FROM schedules
                    WHERE season = {season} AND week = {selected_matchup_week} AND game_type = 'REG'
                    ORDER BY gameday
                """
                matchups_df = pd.read_sql_query(matchups_query, conn)

                if not matchups_df.empty:
                    # Create matchup options list
                    matchup_options = []
                    for _, game in matchups_df.iterrows():
                        matchup_str = f"{game['away_team']} @ {game['home_team']}"
                        if pd.notna(game['gameday']):
                            matchup_str += f" ({game['gameday']})"
                        matchup_options.append({
                            'display': matchup_str,
                            'away': game['away_team'],
                            'home': game['home_team']
                        })

                    with col_matchup:
                        selected_matchup_idx = st.selectbox(
                            "Select Matchup",
                            range(len(matchup_options)),
                            format_func=lambda x: matchup_options[x]['display'],
                            key="quick_matchup_game"
                        )

                    with col_button:
                        st.write("")  # Spacing
                        if st.button("Load Matchup", type="primary", use_container_width=True):
                            # Update session state based on selected matchup
                            selected_matchup = matchup_options[selected_matchup_idx]
                            if selected_matchup['away'] in teams:
                                st.session_state.comparison_team1_idx = teams.index(selected_matchup['away'])
                            if selected_matchup['home'] in teams:
                                st.session_state.comparison_team2_idx = teams.index(selected_matchup['home'])
                            st.rerun()

        conn.close()
    except Exception as e:
        st.warning(f"Could not load week matchups: {e}")

    st.divider()
    st.subheader("📊 Team Selection")

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox(
            "Team 1",
            teams,
            index=st.session_state.comparison_team1_idx,
            key="team1"
        )
        # Update session state when manually changed
        st.session_state.comparison_team1_idx = teams.index(team1)
    with col2:
        team2 = st.selectbox(
            "Team 2",
            teams,
            index=st.session_state.comparison_team2_idx,
            key="team2"
        )
        # Update session state when manually changed
        st.session_state.comparison_team2_idx = teams.index(team2)

    # Query team game stats
    sql = "SELECT * FROM box_score_summary WHERE season=? AND team IN (?, ?)"
    params = [season, team1, team2]
    if week:
        sql += " AND week<=?"
        params.append(week)
    df = query(sql, tuple(params))

    # Query player stats
    player_sql = f"""
        SELECT
            player_display_name as player,
            team,
            week,
            season,
            opponent_team as opponent,
            completions as pass_comp,
            attempts as pass_att,
            passing_yards as pass_yds,
            passing_tds as pass_td,
            passing_interceptions as pass_int,
            carries as rush_att,
            rushing_yards as rush_yds,
            rushing_tds as rush_td,
            receptions as rec,
            targets,
            receiving_yards as rec_yds,
            receiving_tds as rec_td
        FROM player_stats
        WHERE season={season} AND team IN ('{team1}', '{team2}')
    """
    if week:
        player_sql += f" AND week<={week}"
    players_df = query(player_sql)

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

    # Advanced Defensive Metrics Visualization
    st.markdown("### 🛡️ Advanced Defensive Analysis")
    st.caption("Pass rush pressure, blitz effectiveness, and defensive efficiency metrics")

    try:
        # Query advanced defensive metrics from pfr_advstats_def_week table
        conn = sqlite3.connect(DB_PATH)
        week_filter = f" AND week <= {week}" if week else ""

        adv_def_query = f"""
            SELECT
                team,
                COUNT(DISTINCT game_id) as games,
                ROUND(SUM(COALESCE(def_sacks, 0)), 1) as total_sacks,
                ROUND(SUM(COALESCE(def_times_hurried, 0)), 1) as total_hurries,
                ROUND(SUM(COALESCE(def_times_hitqb, 0)), 1) as total_qb_hits,
                ROUND(SUM(COALESCE(def_times_blitzed, 0)), 1) as total_blitzes,
                ROUND(SUM(COALESCE(def_sacks, 0)) / NULLIF(COUNT(DISTINCT game_id), 0), 2) as sacks_per_game,
                ROUND(SUM(COALESCE(def_times_hurried, 0)) / NULLIF(COUNT(DISTINCT game_id), 0), 2) as hurries_per_game,
                ROUND(SUM(COALESCE(def_times_hitqb, 0)) / NULLIF(COUNT(DISTINCT game_id), 0), 2) as qb_hits_per_game,
                ROUND(SUM(COALESCE(def_times_blitzed, 0)) / NULLIF(COUNT(DISTINCT game_id), 0), 2) as blitzes_per_game,
                ROUND(
                    (SUM(COALESCE(def_sacks, 0)) +
                     SUM(COALESCE(def_times_hurried, 0)) +
                     SUM(COALESCE(def_times_hitqb, 0))) / NULLIF(COUNT(DISTINCT game_id), 0), 2
                ) as pressure_plays_per_game,
                ROUND(AVG(COALESCE(def_passer_rating_allowed, 0)), 1) as passer_rating_allowed,
                ROUND(AVG(COALESCE(def_missed_tackle_pct, 0)) * 100, 1) as missed_tackle_pct
            FROM pfr_advstats_def_week
            WHERE season = {season}{week_filter}
              AND team IN ('{team1}', '{team2}')
            GROUP BY team
            ORDER BY sacks_per_game DESC
        """

        adv_def_stats = pd.read_sql_query(adv_def_query, conn)
        conn.close()

        if not adv_def_stats.empty and len(adv_def_stats) == 2:
            # Separate data for each team
            t1_data = adv_def_stats[adv_def_stats['team'] == team1].iloc[0] if team1 in adv_def_stats['team'].values else None
            t2_data = adv_def_stats[adv_def_stats['team'] == team2].iloc[0] if team2 in adv_def_stats['team'].values else None

            if t1_data is not None and t2_data is not None:
                # CHART 1: Pass Rush Pressure Comparison (Grouped Bar Chart)
                st.markdown("#### 🔥 Pass Rush Pressure Metrics")

                fig_pressure = go.Figure()

                categories = ['Sacks/Game', 'QB Hurries/Game', 'QB Hits/Game']
                t1_values = [t1_data['sacks_per_game'], t1_data['hurries_per_game'], t1_data['qb_hits_per_game']]
                t2_values = [t2_data['sacks_per_game'], t2_data['hurries_per_game'], t2_data['qb_hits_per_game']]

                fig_pressure.add_trace(go.Bar(
                    name=team1,
                    x=categories,
                    y=t1_values,
                    marker_color='#e74c3c',
                    text=[f'{v:.2f}' for v in t1_values],
                    textposition='outside'
                ))

                fig_pressure.add_trace(go.Bar(
                    name=team2,
                    x=categories,
                    y=t2_values,
                    marker_color='#3498db',
                    text=[f'{v:.2f}' for v in t2_values],
                    textposition='outside'
                ))

                fig_pressure.update_layout(
                    title=f"Pass Rush Pressure Comparison ({season} Season)",
                    xaxis_title="Metric",
                    yaxis_title="Per Game Average",
                    barmode='group',
                    height=400,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig_pressure, use_container_width=True)

                # Display summary metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        f"{team1} Total Pressure Plays/Game",
                        f"{t1_data['pressure_plays_per_game']:.2f}",
                        help="Combined sacks + hurries + QB hits per game"
                    )
                with col2:
                    st.metric(
                        f"{team2} Total Pressure Plays/Game",
                        f"{t2_data['pressure_plays_per_game']:.2f}",
                        help="Combined sacks + hurries + QB hits per game"
                    )

                st.divider()

                # CHART 2: Blitz Effectiveness Scatter Plot
                st.markdown("#### ⚡ Blitz Effectiveness Analysis")

                fig_blitz = go.Figure()

                # Add data points for both teams
                fig_blitz.add_trace(go.Scatter(
                    x=[t1_data['blitzes_per_game']],
                    y=[t1_data['sacks_per_game']],
                    mode='markers+text',
                    name=team1,
                    marker=dict(size=20, color='#e74c3c'),
                    text=[team1],
                    textposition='top center',
                    textfont=dict(size=14, color='#e74c3c', family='Arial Black'),
                    hovertemplate=f'<b>{team1}</b><br>Blitzes/Game: %{{x:.2f}}<br>Sacks/Game: %{{y:.2f}}<extra></extra>'
                ))

                fig_blitz.add_trace(go.Scatter(
                    x=[t2_data['blitzes_per_game']],
                    y=[t2_data['sacks_per_game']],
                    mode='markers+text',
                    name=team2,
                    marker=dict(size=20, color='#3498db'),
                    text=[team2],
                    textposition='top center',
                    textfont=dict(size=14, color='#3498db', family='Arial Black'),
                    hovertemplate=f'<b>{team2}</b><br>Blitzes/Game: %{{x:.2f}}<br>Sacks/Game: %{{y:.2f}}<extra></extra>'
                ))

                # Add average reference lines
                avg_blitzes = (t1_data['blitzes_per_game'] + t2_data['blitzes_per_game']) / 2
                avg_sacks = (t1_data['sacks_per_game'] + t2_data['sacks_per_game']) / 2

                fig_blitz.add_hline(y=avg_sacks, line_dash="dash", line_color="gray",
                                   annotation_text=f"Avg Sacks ({avg_sacks:.2f})", annotation_position="right")
                fig_blitz.add_vline(x=avg_blitzes, line_dash="dash", line_color="gray",
                                   annotation_text=f"Avg Blitzes ({avg_blitzes:.2f})", annotation_position="top")

                fig_blitz.update_layout(
                    title="Blitz Frequency vs Sack Production",
                    xaxis_title="Blitzes Per Game",
                    yaxis_title="Sacks Per Game",
                    height=450,
                    showlegend=False
                )

                st.plotly_chart(fig_blitz, use_container_width=True)

                # Interpretation
                st.caption("""
                **💡 Interpretation:**
                - **Top-right:** High blitz rate + high sack production (aggressive & effective)
                - **Top-left:** Low blitz rate + high sack production (efficient pass rush without blitzing)
                - **Bottom-right:** High blitz rate + low sack production (blitzing without results)
                - **Bottom-left:** Low blitz rate + low sack production (conservative approach)
                """)

                st.divider()

                # CHART 3: Defensive Efficiency Metrics (Horizontal Bar Chart)
                st.markdown("#### 💎 Defensive Efficiency Metrics")

                fig_efficiency = go.Figure()

                # Passer Rating Allowed (lower is better - invert for visualization)
                fig_efficiency.add_trace(go.Bar(
                    name='QB Rating Allowed',
                    y=[team1, team2],
                    x=[t1_data['passer_rating_allowed'], t2_data['passer_rating_allowed']],
                    orientation='h',
                    marker_color=['#e74c3c', '#3498db'],
                    text=[f"{t1_data['passer_rating_allowed']:.1f}", f"{t2_data['passer_rating_allowed']:.1f}"],
                    textposition='outside',
                    hovertemplate='%{y}<br>QB Rating Allowed: %{x:.1f}<extra></extra>'
                ))

                fig_efficiency.update_layout(
                    title="QB Passer Rating Allowed (Lower is Better)",
                    xaxis_title="Passer Rating",
                    yaxis_title="Team",
                    height=250,
                    showlegend=False,
                    xaxis=dict(autorange='reversed')  # Reverse so lower (better) appears on right
                )

                st.plotly_chart(fig_efficiency, use_container_width=True)

                # Missed Tackle Percentage
                fig_tackles = go.Figure()

                fig_tackles.add_trace(go.Bar(
                    name='Missed Tackle %',
                    y=[team1, team2],
                    x=[t1_data['missed_tackle_pct'], t2_data['missed_tackle_pct']],
                    orientation='h',
                    marker_color=['#e74c3c', '#3498db'],
                    text=[f"{t1_data['missed_tackle_pct']:.1f}%", f"{t2_data['missed_tackle_pct']:.1f}%"],
                    textposition='outside',
                    hovertemplate='%{y}<br>Missed Tackle %: %{x:.1f}%<extra></extra>'
                ))

                fig_tackles.update_layout(
                    title="Missed Tackle Percentage (Lower is Better)",
                    xaxis_title="Missed Tackle %",
                    yaxis_title="Team",
                    height=250,
                    showlegend=False,
                    xaxis=dict(autorange='reversed')  # Reverse so lower (better) appears on right
                )

                st.plotly_chart(fig_tackles, use_container_width=True)

                # Summary metrics in columns
                st.markdown("##### 📊 Defensive Summary")
                col1, col2, col3 = st.columns(3)

                with col1:
                    better_pressure = team1 if t1_data['pressure_plays_per_game'] > t2_data['pressure_plays_per_game'] else team2
                    st.info(f"**Pass Rush Leader:** {better_pressure}")

                with col2:
                    better_coverage = team1 if t1_data['passer_rating_allowed'] < t2_data['passer_rating_allowed'] else team2
                    st.info(f"**Coverage Leader:** {better_coverage}")

                with col3:
                    better_tackling = team1 if t1_data['missed_tackle_pct'] < t2_data['missed_tackle_pct'] else team2
                    st.info(f"**Tackling Leader:** {better_tackling}")

            else:
                st.info("Advanced defensive data available for only one team")
        else:
            st.info("Advanced defensive metrics not available for selected teams/season")

    except Exception as e:
        st.caption(f"⚠️ Advanced defensive charts unavailable: {e}")
        import traceback
        st.error(traceback.format_exc())

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

    # Enhanced Defensive Analysis Section
    st.subheader("🛡️ Comprehensive Defensive Analysis")
    st.caption("Detailed defensive metrics including pressure, turnovers, and rankings")

    # Calculate defensive stats using NFLverse data (always available)
    def calculate_team_defensive_metrics(team, season, week):
        """Calculate defensive metrics from team_game_summary and player_stats tables."""
        try:
            conn = sqlite3.connect(DB_PATH)

            # Build week filter
            week_filter = f" AND week < {week}" if week else ""

            # Query defensive stats from team_game_summary (opponent stats)
            defense_query = f"""
                SELECT
                    AVG(pass_yds) as avg_pass_yds_allowed,
                    AVG(rush_yds) as avg_rush_yds_allowed,
                    AVG(yards_total) as avg_total_yds_allowed,
                    AVG(points) as avg_pts_allowed,
                    COUNT(*) as games
                FROM team_game_summary
                WHERE season = {season}{week_filter}
                  AND opponent_team = '{team}'
            """
            def_df = pd.read_sql_query(defense_query, conn)

            # Query turnover stats from player_stats (defensive players against this team)
            turnover_query = f"""
                SELECT
                    SUM(CASE WHEN position = 'DB' OR position = 'LB' OR position = 'DL'
                             THEN COALESCE(interceptions, 0) ELSE 0 END) as total_ints,
                    COUNT(DISTINCT week) as games_played
                FROM player_stats
                WHERE season = {season}{week_filter}
                  AND opponent_team = '{team}'
            """
            turnover_df = pd.read_sql_query(turnover_query, conn)

            conn.close()

            # Calculate metrics
            games = def_df['games'].iloc[0] if not def_df.empty and def_df['games'].iloc[0] > 0 else 0

            if games > 0:
                return {
                    'games': games,
                    'avg_pass_yds_allowed': def_df['avg_pass_yds_allowed'].iloc[0],
                    'avg_rush_yds_allowed': def_df['avg_rush_yds_allowed'].iloc[0],
                    'avg_total_yds_allowed': def_df['avg_total_yds_allowed'].iloc[0],
                    'avg_pts_allowed': def_df['avg_pts_allowed'].iloc[0],
                    'total_ints': turnover_df['total_ints'].iloc[0] if not turnover_df.empty else 0,
                    'ints_per_game': turnover_df['total_ints'].iloc[0] / games if not turnover_df.empty and games > 0 else 0
                }
            else:
                return {'games': 0}

        except Exception as e:
            return {'games': 0}

    # Get comprehensive defensive metrics for both teams
    t1_def_metrics = calculate_team_defensive_metrics(team1, season, week)
    t2_def_metrics = calculate_team_defensive_metrics(team2, season, week)

    if t1_def_metrics['games'] > 0 and t2_def_metrics['games'] > 0:
        # Display defensive metrics in expandable sections
        with st.expander("🎯 Pass Defense Metrics", expanded=True):
            pass_col1, pass_col2, pass_col3 = st.columns(3)

            with pass_col1:
                st.markdown(f"**{team1} Pass Defense**")
                st.metric("Pass Yds Allowed/Gm", f"{t1_def_metrics['avg_pass_yds_allowed']:.1f}")
                st.metric("INTs/Game", f"{t1_def_metrics['ints_per_game']:.2f}")

            with pass_col3:
                st.markdown(f"**{team2} Pass Defense**")
                st.metric("Pass Yds Allowed/Gm", f"{t2_def_metrics['avg_pass_yds_allowed']:.1f}")
                st.metric("INTs/Game", f"{t2_def_metrics['ints_per_game']:.2f}")

            with pass_col2:
                st.markdown("**Advantage**")
                # Lower is better for defense
                pass_yds_diff = t2_def_metrics['avg_pass_yds_allowed'] - t1_def_metrics['avg_pass_yds_allowed']
                leader = team1 if pass_yds_diff > 0 else team2
                st.metric("Pass D", leader, f"{abs(pass_yds_diff):.1f} yds/gm better")

                ints_diff = t1_def_metrics['ints_per_game'] - t2_def_metrics['ints_per_game']
                int_leader = team1 if ints_diff > 0 else team2
                st.metric("Takeaways", int_leader, f"{abs(ints_diff):+.2f} INT/gm")

        with st.expander("🏃 Run Defense Metrics"):
            run_col1, run_col2, run_col3 = st.columns(3)

            with run_col1:
                st.markdown(f"**{team1} Run Defense**")
                st.metric("Rush Yds Allowed/Gm", f"{t1_def_metrics['avg_rush_yds_allowed']:.1f}")

            with run_col3:
                st.markdown(f"**{team2} Run Defense**")
                st.metric("Rush Yds Allowed/Gm", f"{t2_def_metrics['avg_rush_yds_allowed']:.1f}")

            with run_col2:
                st.markdown("**Advantage**")
                run_yds_diff = t2_def_metrics['avg_rush_yds_allowed'] - t1_def_metrics['avg_rush_yds_allowed']
                run_leader = team1 if run_yds_diff > 0 else team2
                st.metric("Run D", run_leader, f"{abs(run_yds_diff):.1f} yds/gm better")

        with st.expander("📊 Overall Defensive Efficiency"):
            eff_col1, eff_col2, eff_col3 = st.columns(3)

            with eff_col1:
                st.markdown(f"**{team1} Defense**")
                st.metric("Total Yds Allowed/Gm", f"{t1_def_metrics['avg_total_yds_allowed']:.1f}")
                st.metric("Points Allowed/Gm", f"{t1_def_metrics['avg_pts_allowed']:.1f}")

            with eff_col3:
                st.markdown(f"**{team2} Defense**")
                st.metric("Total Yds Allowed/Gm", f"{t2_def_metrics['avg_total_yds_allowed']:.1f}")
                st.metric("Points Allowed/Gm", f"{t2_def_metrics['avg_pts_allowed']:.1f}")

            with eff_col2:
                st.markdown("**Advantage**")
                total_yds_diff = t2_def_metrics['avg_total_yds_allowed'] - t1_def_metrics['avg_total_yds_allowed']
                total_leader = team1 if total_yds_diff > 0 else team2
                st.metric("Total D", total_leader, f"{abs(total_yds_diff):.1f} yds/gm")

                pts_diff = t2_def_metrics['avg_pts_allowed'] - t1_def_metrics['avg_pts_allowed']
                pts_leader = team1 if pts_diff > 0 else team2
                st.metric("Scoring D", pts_leader, f"{abs(pts_diff):.1f} pts/gm")

        # Defensive Rankings
        st.markdown("### 📈 Defensive Rankings")
        st.caption("League rankings based on yards allowed (1 = best defense)")

        try:
            # Get all teams and calculate defensive rankings
            all_teams_def = []
            teams_list = get_teams(season)

            for team in teams_list:
                team_def = calculate_team_defensive_metrics(team, season, week)
                if team_def['games'] > 0:
                    all_teams_def.append({
                        'team': team,
                        'pass_yds': team_def['avg_pass_yds_allowed'],
                        'rush_yds': team_def['avg_rush_yds_allowed'],
                        'total_yds': team_def['avg_total_yds_allowed'],
                        'pts': team_def['avg_pts_allowed']
                    })

            # Sort and rank (lower is better for defense)
            all_teams_def.sort(key=lambda x: x['total_yds'])

            # Find rankings for our teams
            t1_rank = next((i+1 for i, t in enumerate(all_teams_def) if t['team'] == team1), None)
            t2_rank = next((i+1 for i, t in enumerate(all_teams_def) if t['team'] == team2), None)

            if t1_rank and t2_rank:
                rank_col1, rank_col2, rank_col3 = st.columns(3)

                with rank_col1:
                    st.markdown(f"**{team1}**")
                    st.metric("Overall Defense Rank", f"#{t1_rank}")

                with rank_col3:
                    st.markdown(f"**{team2}**")
                    st.metric("Overall Defense Rank", f"#{t2_rank}")

                with rank_col2:
                    st.markdown("**Comparison**")
                    rank_diff = abs(t1_rank - t2_rank)
                    better_team = team1 if t1_rank < t2_rank else team2
                    if rank_diff > 0:
                        st.metric("Better Defense", better_team, f"{rank_diff} spots better")
                    else:
                        st.info("Tied ranking")

        except Exception as e:
            st.caption(f"Rankings calculation unavailable: {e}")


    else:
        st.info("⚠️ Comprehensive defensive metrics require game data. Play at least 1 game to see detailed defensive analysis.")

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
                'pass_comp': 'sum',
                'pass_att': ['sum', 'count'],
                'pass_yds': 'sum',
                'pass_td': 'sum',
                'pass_int': 'sum'
            }).reset_index()
            pass_leaders.columns = ['team', 'player', 'pass_comp', 'pass_att', 'games', 'pass_yds', 'pass_td', 'pass_int']
            pass_leaders['Comp%'] = (pass_leaders['pass_comp'] / pass_leaders['pass_att'] * 100).round(1)
            pass_leaders = pass_leaders.sort_values('pass_yds', ascending=False)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{team1} Passing**")
                t1_pass = pass_leaders[pass_leaders['team'] == team1].head(3)
                if not t1_pass.empty:
                    t1_pass_display = t1_pass[['player', 'games', 'pass_yds', 'pass_td', 'pass_int', 'Comp%']].copy()
                    t1_pass_display.columns = ['Player', 'Games', 'Yards', 'TD', 'INT', 'Comp%']
                    t1_pass_display['Games'] = t1_pass_display['Games'].astype(int)
                    t1_pass_display['Yards'] = t1_pass_display['Yards'].astype(int)
                    t1_pass_display['TD'] = t1_pass_display['TD'].astype(int)
                    t1_pass_display['INT'] = t1_pass_display['INT'].astype(int)
                    st.dataframe(t1_pass_display, hide_index=True, use_container_width=True)

            with col2:
                st.markdown(f"**{team2} Passing**")
                t2_pass = pass_leaders[pass_leaders['team'] == team2].head(3)
                if not t2_pass.empty:
                    t2_pass_display = t2_pass[['player', 'games', 'pass_yds', 'pass_td', 'pass_int', 'Comp%']].copy()
                    t2_pass_display.columns = ['Player', 'Games', 'Yards', 'TD', 'INT', 'Comp%']
                    t2_pass_display['Games'] = t2_pass_display['Games'].astype(int)
                    t2_pass_display['Yards'] = t2_pass_display['Yards'].astype(int)
                    t2_pass_display['TD'] = t2_pass_display['TD'].astype(int)
                    t2_pass_display['INT'] = t2_pass_display['INT'].astype(int)
                    st.dataframe(t2_pass_display, hide_index=True, use_container_width=True)

        with tab2:
            rush_leaders = players_df[players_df['rush_att'] > 0].groupby(['team', 'player']).agg({
                'rush_att': 'sum',
                'rush_yds': ['sum', 'count'],
                'rush_td': 'sum'
            }).reset_index()
            rush_leaders.columns = ['team', 'player', 'rush_att', 'rush_yds', 'games', 'rush_td']
            rush_leaders['YPA'] = (rush_leaders['rush_yds'] / rush_leaders['rush_att']).round(1)
            rush_leaders = rush_leaders.sort_values('rush_yds', ascending=False)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{team1} Rushing**")
                t1_rush = rush_leaders[rush_leaders['team'] == team1].head(5)
                if not t1_rush.empty:
                    t1_rush_display = t1_rush[['player', 'games', 'rush_yds', 'rush_att', 'YPA', 'rush_td']].copy()
                    t1_rush_display.columns = ['Player', 'Games', 'Yards', 'Att', 'YPA', 'TD']
                    t1_rush_display['Games'] = t1_rush_display['Games'].astype(int)
                    t1_rush_display['Yards'] = t1_rush_display['Yards'].astype(int)
                    t1_rush_display['Att'] = t1_rush_display['Att'].astype(int)
                    t1_rush_display['TD'] = t1_rush_display['TD'].astype(int)
                    st.dataframe(t1_rush_display, hide_index=True, use_container_width=True)

            with col2:
                st.markdown(f"**{team2} Rushing**")
                t2_rush = rush_leaders[rush_leaders['team'] == team2].head(5)
                if not t2_rush.empty:
                    t2_rush_display = t2_rush[['player', 'games', 'rush_yds', 'rush_att', 'YPA', 'rush_td']].copy()
                    t2_rush_display.columns = ['Player', 'Games', 'Yards', 'Att', 'YPA', 'TD']
                    t2_rush_display['Games'] = t2_rush_display['Games'].astype(int)
                    t2_rush_display['Yards'] = t2_rush_display['Yards'].astype(int)
                    t2_rush_display['Att'] = t2_rush_display['Att'].astype(int)
                    t2_rush_display['TD'] = t2_rush_display['TD'].astype(int)
                    st.dataframe(t2_rush_display, hide_index=True, use_container_width=True)

        with tab3:
            rec_leaders = players_df[players_df['rec'] > 0].groupby(['team', 'player']).agg({
                'rec': 'sum',
                'targets': 'sum',
                'rec_yds': ['sum', 'count'],
                'rec_td': 'sum'
            }).reset_index()
            rec_leaders.columns = ['team', 'player', 'rec', 'targets', 'rec_yds', 'games', 'rec_td']
            rec_leaders['YPR'] = (rec_leaders['rec_yds'] / rec_leaders['rec']).round(1)
            rec_leaders['Catch%'] = (rec_leaders['rec'] / rec_leaders['targets'] * 100).round(0)
            rec_leaders = rec_leaders.sort_values('rec_yds', ascending=False)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{team1} Receiving**")
                t1_rec = rec_leaders[rec_leaders['team'] == team1].head(5)
                if not t1_rec.empty:
                    t1_rec_display = t1_rec[['player', 'games', 'rec_yds', 'targets', 'rec', 'YPR', 'rec_td', 'Catch%']].copy()
                    t1_rec_display.columns = ['Player', 'Games', 'Yards', 'Tgt', 'Rec', 'YPR', 'TD', 'Catch%']
                    t1_rec_display['Games'] = t1_rec_display['Games'].astype(int)
                    t1_rec_display['Yards'] = t1_rec_display['Yards'].astype(int)
                    t1_rec_display['Tgt'] = t1_rec_display['Tgt'].astype(int)
                    t1_rec_display['Rec'] = t1_rec_display['Rec'].astype(int)
                    t1_rec_display['TD'] = t1_rec_display['TD'].astype(int)
                    t1_rec_display['Catch%'] = t1_rec_display['Catch%'].astype(int)
                    st.dataframe(t1_rec_display, hide_index=True, use_container_width=True)

            with col2:
                st.markdown(f"**{team2} Receiving**")
                t2_rec = rec_leaders[rec_leaders['team'] == team2].head(5)
                if not t2_rec.empty:
                    t2_rec_display = t2_rec[['player', 'games', 'rec_yds', 'targets', 'rec', 'YPR', 'rec_td', 'Catch%']].copy()
                    t2_rec_display.columns = ['Player', 'Games', 'Yards', 'Tgt', 'Rec', 'YPR', 'TD', 'Catch%']
                    t2_rec_display['Games'] = t2_rec_display['Games'].astype(int)
                    t2_rec_display['Yards'] = t2_rec_display['Yards'].astype(int)
                    t2_rec_display['Tgt'] = t2_rec_display['Tgt'].astype(int)
                    t2_rec_display['Rec'] = t2_rec_display['Rec'].astype(int)
                    t2_rec_display['TD'] = t2_rec_display['TD'].astype(int)
                    t2_rec_display['Catch%'] = t2_rec_display['Catch%'].astype(int)
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
                'pass_comp': ['sum'],
                'pass_att': ['sum']
            }).reset_index()

            # Flatten column names
            pass_stats.columns = ['team', 'player', 'avg_yds', 'med_yds', 'max_yds', 'min_yds', 'total_yds', 'games',
                                  'avg_td', 'total_td', 'total_int', 'total_comp', 'total_att']
            pass_stats = pass_stats.sort_values('total_yds', ascending=False)

            # Get game-by-game data with opponents for last 3 games
            # Note: player_stats already has opponent column from query
            pass_players_with_games = players_df[players_df['pass_att'] > 0].copy()

            # Load advanced QB stats from pfr_advstats_pass_week
            qb_adv_query = f"""
                SELECT
                    season,
                    week,
                    team,
                    pfr_player_name as player,
                    times_blitzed,
                    times_sacked,
                    times_pressured_pct,
                    passing_bad_throw_pct,
                    passing_drops,
                    passing_drop_pct
                FROM pfr_advstats_pass_week
                WHERE season={season} AND team IN ('{team1}', '{team2}')
            """
            if week:
                qb_adv_query += f" AND week<={week}"

            qb_adv_stats = query(qb_adv_query)

            # Get schedule info to determine home/away games
            schedules_query = f"""
                SELECT week, season, away_team, home_team
                FROM schedules
                WHERE season={season}
            """
            schedules_info = query(schedules_query)

            # Create a temporary merge column combining week, season, team, opponent
            # This helps us join with schedules to determine home/away
            def determine_is_away(row):
                # Find the game in schedules
                game = schedules_info[
                    (schedules_info['week'] == row['week']) &
                    (schedules_info['season'] == row['season']) &
                    (
                        ((schedules_info['home_team'] == row['team']) & (schedules_info['away_team'] == row['opponent'])) |
                        ((schedules_info['away_team'] == row['team']) & (schedules_info['home_team'] == row['opponent']))
                    )
                ]
                if not game.empty:
                    return row['team'] == game.iloc[0]['away_team']
                return False  # Default to home if not found

            pass_players_with_games['is_away'] = pass_players_with_games.apply(determine_is_away, axis=1)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### {team1} Passers")
                t1_passers = pass_stats[pass_stats['team'] == team1].head(3)
                if not t1_passers.empty:
                    t1_pass_display = t1_passers.copy()
                    t1_pass_display['Comp%'] = (t1_pass_display['total_comp'] / t1_pass_display['total_att'] * 100).round(1)
                    t1_pass_display = t1_pass_display[['player', 'games', 'total_yds', 'avg_yds', 'total_td', 'avg_td', 'total_int', 'Comp%']].copy()
                    t1_pass_display.columns = ['Player', 'Games', 'Tot Yds', 'Avg Yds', 'Tot TD', 'Avg TD', 'INT', 'Comp%']
                    t1_pass_display['Games'] = t1_pass_display['Games'].astype(int)
                    t1_pass_display['Tot Yds'] = t1_pass_display['Tot Yds'].astype(int)
                    t1_pass_display['Avg Yds'] = t1_pass_display['Avg Yds'].round(1)
                    t1_pass_display['Tot TD'] = t1_pass_display['Tot TD'].astype(int)
                    t1_pass_display['Avg TD'] = t1_pass_display['Avg TD'].round(1)
                    t1_pass_display['INT'] = t1_pass_display['INT'].astype(int)
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
                                    team_total_att = pass_players_with_games[
                                        (pass_players_with_games['week'] == row['week']) &
                                        (pass_players_with_games['season'] == row['season']) &
                                        (pass_players_with_games['team'] == row['team'])
                                    ]['pass_att'].sum()
                                    att_pct = (row['pass_att'] / team_total_att * 100) if team_total_att > 0 else 0
                                    location = "@ " if row['is_away'] else "vs "
                                    cmp_att = f"{int(row['pass_comp'])}/{int(row['pass_att'])}"

                                    # Get advanced stats for this player/week
                                    adv_stats = qb_adv_stats[
                                        (qb_adv_stats['player'] == row['player']) &
                                        (qb_adv_stats['week'] == row['week']) &
                                        (qb_adv_stats['season'] == row['season']) &
                                        (qb_adv_stats['team'] == row['team'])
                                    ]

                                    # Extract advanced stats or use defaults
                                    rush_td = int(row['rush_td']) if row['rush_td'] > 0 else 0
                                    blitzed = int(adv_stats['times_blitzed'].iloc[0]) if not adv_stats.empty and pd.notna(adv_stats['times_blitzed'].iloc[0]) else 0
                                    sacked = int(adv_stats['times_sacked'].iloc[0]) if not adv_stats.empty and pd.notna(adv_stats['times_sacked'].iloc[0]) else 0
                                    pressure_pct = f"{adv_stats['times_pressured_pct'].iloc[0]*100:.1f}%" if not adv_stats.empty and pd.notna(adv_stats['times_pressured_pct'].iloc[0]) else "0.0%"
                                    bad_throw_pct = f"{adv_stats['passing_bad_throw_pct'].iloc[0]*100:.1f}%" if not adv_stats.empty and pd.notna(adv_stats['passing_bad_throw_pct'].iloc[0]) else "0.0%"

                                    # Format drops as "# (X%)" for clarity
                                    if not adv_stats.empty and pd.notna(adv_stats['passing_drops'].iloc[0]):
                                        drops_num = int(adv_stats['passing_drops'].iloc[0])
                                        drops_pct = adv_stats['passing_drop_pct'].iloc[0] * 100
                                        drop_display = f"{drops_num} ({drops_pct:.1f}%)"
                                    else:
                                        drop_display = "0 (0.0%)"

                                    game_data.append({
                                        'Week': int(row['week']),
                                        'Opponent': f"{location}{row['opponent']}",
                                        'Yards': int(row['pass_yds']),
                                        'Cmp/Att': cmp_att,
                                        'TD': int(row['pass_td']),
                                        'INT': int(row['pass_int']),
                                        'Rush TD': rush_td,
                                        'Blitzed': blitzed,
                                        'Sacked': sacked,
                                        'Press%': pressure_pct,
                                        'BadTh%': bad_throw_pct,
                                        'Drops': drop_display,
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
                    t2_pass_display['Comp%'] = (t2_pass_display['total_comp'] / t2_pass_display['total_att'] * 100).round(1)
                    t2_pass_display = t2_pass_display[['player', 'games', 'total_yds', 'avg_yds', 'total_td', 'avg_td', 'total_int', 'Comp%']].copy()
                    t2_pass_display.columns = ['Player', 'Games', 'Tot Yds', 'Avg Yds', 'Tot TD', 'Avg TD', 'INT', 'Comp%']
                    t2_pass_display['Games'] = t2_pass_display['Games'].astype(int)
                    t2_pass_display['Tot Yds'] = t2_pass_display['Tot Yds'].astype(int)
                    t2_pass_display['Avg Yds'] = t2_pass_display['Avg Yds'].round(1)
                    t2_pass_display['Tot TD'] = t2_pass_display['Tot TD'].astype(int)
                    t2_pass_display['Avg TD'] = t2_pass_display['Avg TD'].round(1)
                    t2_pass_display['INT'] = t2_pass_display['INT'].astype(int)
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
                                    team_total_att = pass_players_with_games[
                                        (pass_players_with_games['week'] == row['week']) &
                                        (pass_players_with_games['season'] == row['season']) &
                                        (pass_players_with_games['team'] == row['team'])
                                    ]['pass_att'].sum()
                                    att_pct = (row['pass_att'] / team_total_att * 100) if team_total_att > 0 else 0
                                    location = "@ " if row['is_away'] else "vs "
                                    cmp_att = f"{int(row['pass_comp'])}/{int(row['pass_att'])}"

                                    # Get advanced stats for this player/week
                                    adv_stats = qb_adv_stats[
                                        (qb_adv_stats['player'] == row['player']) &
                                        (qb_adv_stats['week'] == row['week']) &
                                        (qb_adv_stats['season'] == row['season']) &
                                        (qb_adv_stats['team'] == row['team'])
                                    ]

                                    # Extract advanced stats or use defaults
                                    rush_td = int(row['rush_td']) if row['rush_td'] > 0 else 0
                                    blitzed = int(adv_stats['times_blitzed'].iloc[0]) if not adv_stats.empty and pd.notna(adv_stats['times_blitzed'].iloc[0]) else 0
                                    sacked = int(adv_stats['times_sacked'].iloc[0]) if not adv_stats.empty and pd.notna(adv_stats['times_sacked'].iloc[0]) else 0
                                    pressure_pct = f"{adv_stats['times_pressured_pct'].iloc[0]*100:.1f}%" if not adv_stats.empty and pd.notna(adv_stats['times_pressured_pct'].iloc[0]) else "0.0%"
                                    bad_throw_pct = f"{adv_stats['passing_bad_throw_pct'].iloc[0]*100:.1f}%" if not adv_stats.empty and pd.notna(adv_stats['passing_bad_throw_pct'].iloc[0]) else "0.0%"

                                    # Format drops as "# (X%)" for clarity
                                    if not adv_stats.empty and pd.notna(adv_stats['passing_drops'].iloc[0]):
                                        drops_num = int(adv_stats['passing_drops'].iloc[0])
                                        drops_pct = adv_stats['passing_drop_pct'].iloc[0] * 100
                                        drop_display = f"{drops_num} ({drops_pct:.1f}%)"
                                    else:
                                        drop_display = "0 (0.0%)"

                                    game_data.append({
                                        'Week': int(row['week']),
                                        'Opponent': f"{location}{row['opponent']}",
                                        'Yards': int(row['pass_yds']),
                                        'Cmp/Att': cmp_att,
                                        'TD': int(row['pass_td']),
                                        'INT': int(row['pass_int']),
                                        'Rush TD': rush_td,
                                        'Blitzed': blitzed,
                                        'Sacked': sacked,
                                        'Press%': pressure_pct,
                                        'BadTh%': bad_throw_pct,
                                        'Drops': drop_display,
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
                'rush_td': ['mean', 'sum']
            }).reset_index()

            # Flatten column names
            rush_stats.columns = ['team', 'player', 'avg_yds', 'med_yds', 'max_yds', 'min_yds', 'total_yds', 'games', 'std_yds',
                                  'total_att', 'min_att', 'max_att', 'avg_td', 'total_td']
            rush_stats = rush_stats.sort_values('total_yds', ascending=False)

            # Get game-by-game data with opponents for last 3 games
            # Note: player_stats already has opponent column from query
            players_with_games = players_df[players_df['rush_att'] > 0].copy()

            # Get schedule info to determine home/away games
            schedules_query = f"""
                SELECT week, season, away_team, home_team
                FROM schedules
                WHERE season={season}
            """
            schedules_info = query(schedules_query)

            # Determine if away game
            def determine_is_away(row):
                game = schedules_info[
                    (schedules_info['week'] == row['week']) &
                    (schedules_info['season'] == row['season']) &
                    (
                        ((schedules_info['home_team'] == row['team']) & (schedules_info['away_team'] == row['opponent'])) |
                        ((schedules_info['away_team'] == row['team']) & (schedules_info['home_team'] == row['opponent']))
                    )
                ]
                if not game.empty:
                    return row['team'] == game.iloc[0]['away_team']
                return False

            players_with_games['is_away'] = players_with_games.apply(determine_is_away, axis=1)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### {team1} Rushers")
                t1_rushers = rush_stats[rush_stats['team'] == team1].head(5)
                if not t1_rushers.empty:
                    t1_rush_display = t1_rushers.copy()
                    t1_rush_display['YPC'] = (t1_rush_display['total_yds'] / t1_rush_display['total_att']).round(2)
                    t1_rush_display['Att/G'] = (t1_rush_display['total_att'] / t1_rush_display['games']).round(1)
                    t1_rush_display = t1_rush_display[['player', 'games', 'total_yds', 'avg_yds', 'Att/G', 'total_td', 'YPC', 'max_yds']].copy()
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
                                    team_total = players_with_games[
                                        (players_with_games['week'] == row['week']) &
                                        (players_with_games['season'] == row['season']) &
                                        (players_with_games['team'] == row['team'])
                                    ]['rush_att'].sum()
                                    att_pct = (row['rush_att'] / team_total * 100) if team_total > 0 else 0
                                    location = "@ " if row['is_away'] else "vs "
                                    game_data.append({
                                        'Week': int(row['week']),
                                        'Opponent': f"{location}{row['opponent']}",
                                        'Yards': int(row['rush_yds']),
                                        'Att': int(row['rush_att']),
                                        'TD': int(row['rush_td']),
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
                    t2_rush_display = t2_rush_display[['player', 'games', 'total_yds', 'avg_yds', 'Att/G', 'total_td', 'YPC', 'max_yds']].copy()
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
                                    team_total = players_with_games[
                                        (players_with_games['week'] == row['week']) &
                                        (players_with_games['season'] == row['season']) &
                                        (players_with_games['team'] == row['team'])
                                    ]['rush_att'].sum()
                                    att_pct = (row['rush_att'] / team_total * 100) if team_total > 0 else 0
                                    location = "@ " if row['is_away'] else "vs "
                                    game_data.append({
                                        'Week': int(row['week']),
                                        'Opponent': f"{location}{row['opponent']}",
                                        'Yards': int(row['rush_yds']),
                                        'Att': int(row['rush_att']),
                                        'TD': int(row['rush_td']),
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
                'rec_td': ['mean', 'sum']
            }).reset_index()

            # Flatten column names
            rec_stats.columns = ['team', 'player', 'avg_yds', 'med_yds', 'max_yds', 'min_yds', 'total_yds', 'games', 'std_yds',
                                 'total_rec', 'min_rec', 'max_rec', 'total_tgt', 'avg_td', 'total_td']
            rec_stats = rec_stats.sort_values('total_yds', ascending=False)

            # Get game-by-game data with opponents for last 3 games
            # Note: player_stats already has opponent column from query
            rec_players_with_games = players_df[players_df['rec'] > 0].copy()

            # Get schedule info to determine home/away games
            schedules_query = f"""
                SELECT week, season, away_team, home_team
                FROM schedules
                WHERE season={season}
            """
            schedules_info = query(schedules_query)

            # Determine if away game
            def determine_is_away_rec(row):
                game = schedules_info[
                    (schedules_info['week'] == row['week']) &
                    (schedules_info['season'] == row['season']) &
                    (
                        ((schedules_info['home_team'] == row['team']) & (schedules_info['away_team'] == row['opponent'])) |
                        ((schedules_info['away_team'] == row['team']) & (schedules_info['home_team'] == row['opponent']))
                    )
                ]
                if not game.empty:
                    return row['team'] == game.iloc[0]['away_team']
                return False

            rec_players_with_games['is_away'] = rec_players_with_games.apply(determine_is_away_rec, axis=1)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### {team1} Receivers")
                t1_receivers = rec_stats[rec_stats['team'] == team1].head(5)
                if not t1_receivers.empty:
                    t1_rec_display = t1_receivers.copy()
                    t1_rec_display['YPR'] = (t1_rec_display['total_yds'] / t1_rec_display['total_rec']).round(1)
                    t1_rec_display['Catch%'] = (t1_rec_display['total_rec'] / t1_rec_display['total_tgt'] * 100).round(1)
                    t1_rec_display['Rec/G'] = (t1_rec_display['total_rec'] / t1_rec_display['games']).round(1)
                    t1_rec_display = t1_rec_display[['player', 'games', 'total_yds', 'avg_yds', 'total_rec', 'Rec/G', 'total_td', 'YPR', 'Catch%', 'max_yds']].copy()
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
                                    team_total_tgt = rec_players_with_games[
                                        (rec_players_with_games['week'] == row['week']) &
                                        (rec_players_with_games['season'] == row['season']) &
                                        (rec_players_with_games['team'] == row['team'])
                                    ]['targets'].sum()
                                    team_total_rec = rec_players_with_games[
                                        (rec_players_with_games['week'] == row['week']) &
                                        (rec_players_with_games['season'] == row['season']) &
                                        (rec_players_with_games['team'] == row['team'])
                                    ]['rec'].sum()
                                    tgt_pct = (row['targets'] / team_total_tgt * 100) if team_total_tgt > 0 else 0
                                    rec_pct = (row['rec'] / team_total_rec * 100) if team_total_rec > 0 else 0
                                    location = "@ " if row['is_away'] else "vs "
                                    game_data.append({
                                        'Week': int(row['week']),
                                        'Opponent': f"{location}{row['opponent']}",
                                        'Tgts': int(row['targets']),
                                        'Yards': int(row['rec_yds']),
                                        'Rec': int(row['rec']),
                                        'TD': int(row['rec_td']),
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
                    t2_rec_display = t2_rec_display[['player', 'games', 'total_yds', 'avg_yds', 'total_rec', 'Rec/G', 'total_td', 'YPR', 'Catch%', 'max_yds']].copy()
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
                                    team_total_tgt = rec_players_with_games[
                                        (rec_players_with_games['week'] == row['week']) &
                                        (rec_players_with_games['season'] == row['season']) &
                                        (rec_players_with_games['team'] == row['team'])
                                    ]['targets'].sum()
                                    team_total_rec = rec_players_with_games[
                                        (rec_players_with_games['week'] == row['week']) &
                                        (rec_players_with_games['season'] == row['season']) &
                                        (rec_players_with_games['team'] == row['team'])
                                    ]['rec'].sum()
                                    tgt_pct = (row['targets'] / team_total_tgt * 100) if team_total_tgt > 0 else 0
                                    rec_pct = (row['rec'] / team_total_rec * 100) if team_total_rec > 0 else 0
                                    location = "@ " if row['is_away'] else "vs "
                                    game_data.append({
                                        'Week': int(row['week']),
                                        'Opponent': f"{location}{row['opponent']}",
                                        'Tgts': int(row['targets']),
                                        'Yards': int(row['rec_yds']),
                                        'Rec': int(row['rec']),
                                        'TD': int(row['rec_td']),
                                        'Tgt %': f"{tgt_pct:.0f}%",
                                        'Rec %': f"{rec_pct:.0f}%"
                                    })

                                last_3_df = pd.DataFrame(game_data)
                                st.dataframe(last_3_df, hide_index=True, use_container_width=True)
                                st.markdown("---")

        st.divider()

        # Strategic Matchup Analysis Section
        st.subheader("🎯 Strategic Matchup Analysis")
        st.caption("Advanced team-level matchup charts identifying strategic advantages and vulnerabilities")

        # Create a fake upcoming_games dataframe for team comparison context
        # This allows the charts to filter to just these two teams
        upcoming_comparison = pd.DataFrame([{
            'home_team': team1,
            'away_team': team2,
            'week': week if week else 'Season'
        }])

        # QB Pressure Matchup Chart
        with st.expander("🛡️ QB Pressure Matchup - Pass Protection vs Pass Rush", expanded=True):
            st.caption("**Analysis:** Offensive line protection quality vs defensive pass rush effectiveness")
            render_qb_pressure_matchup_chart(season, week, upcoming_comparison)

        # Air Yards vs YAC Matchup Chart
        with st.expander("✈️ Air Yards vs YAC - Passing Philosophy vs Coverage Style", expanded=True):
            st.caption("**Analysis:** Downfield vs underneath passing attack and yards after catch allowed")
            render_air_yac_matchup_chart(season, week, upcoming_comparison)

        # Team Balance Chart
        with st.expander("⚖️ Team Balance - Offensive Play Calling Tendencies", expanded=True):
            st.caption("**Analysis:** Rush vs pass play distribution reveals offensive philosophy and predictability")
            render_team_balance_chart(season, week)

        # Additional Matchup Charts (collapsed by default)
        with st.expander("🏈 Rushing TD Matchup - Red Zone Ground Game Analysis", expanded=False):
            st.caption("**Analysis:** Offensive rushing TD production vs defensive rushing TD prevention in red zone situations")
            render_rushing_td_matchup_chart(season, week, upcoming_comparison)

        with st.expander("🎯 RB Receiving Matchup - Pass-Catching Back Opportunities", expanded=False):
            st.caption("**Analysis:** RB receiving yards production vs linebacker/safety coverage vulnerability on passing downs")
            render_rb_receiving_matchup_chart(season, week, upcoming_comparison)

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
                    'pass_comp': 'median'
                }).reset_index()
                pass_stats.columns = ['team', 'player', 'pass_yds_median', 'pass_yds_mean', 'pass_yds_std',
                                     'pass_td_median', 'pass_td_mean', 'pass_td_std', 'pass_comp']
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

            # Join with schedules to get location (using week and teams)
            schedules_query = f"SELECT week, season, home_team, away_team FROM schedules WHERE season={season}"
            schedules_info = query(schedules_query)

            # Determine home/away for each player game
            def determine_location(row):
                matching_games = schedules_info[
                    (schedules_info['week'] == row['week']) &
                    (schedules_info['season'] == row['season']) &
                    (
                        ((schedules_info['home_team'] == row['team']) & (schedules_info['away_team'] == row['opponent'])) |
                        ((schedules_info['away_team'] == row['team']) & (schedules_info['home_team'] == row['opponent']))
                    )
                ]
                if not matching_games.empty:
                    return 'home' if row['team'] == matching_games.iloc[0]['home_team'] else 'away'
                return 'unknown'

            players_with_loc = players_df.copy()
            players_with_loc['location'] = players_with_loc.apply(determine_location, axis=1)

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

            # Get game scores using schedules and team_stats_week tables
            scores_query = f"""
                SELECT 
                    ts.season,
                    ts.week,
                    ts.team as team_abbr,
                    ts.points as team_score,
                    s.home_team,
                    s.away_team,
                    CASE 
                        WHEN ts.team = s.home_team THEN 
                            (SELECT points FROM team_stats_week WHERE season=ts.season AND week=ts.week AND team=s.away_team)
                        ELSE 
                            (SELECT points FROM team_stats_week WHERE season=ts.season AND week=ts.week AND team=s.home_team)
                    END as opp_score
                FROM team_stats_week ts
                JOIN schedules s ON ts.season = s.season AND ts.week = s.week 
                    AND (ts.team = s.home_team OR ts.team = s.away_team)
                WHERE ts.season = {season}
                    AND ts.team IN ('{team1}', '{team2}')
            """
            if week:
                scores_query += f" AND ts.week <= {week}"
            
            game_scores = query(scores_query)

            # Merge scores with players_df using week, season, and team
            players_with_score = players_df.merge(
                game_scores, 
                left_on=['week', 'season', 'team'], 
                right_on=['week', 'season', 'team_abbr'],
                how='left'
            )

            # Determine if team won
            players_with_score['team_won'] = players_with_score['team_score'] > players_with_score['opp_score']

            # Calculate margin
            players_with_score['margin'] = players_with_score['team_score'] - players_with_score['opp_score']

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

                # TODO: Explosive plays tracking disabled - rush_long and rec_long not in NFLverse schema
                # Need to calculate from individual game logs or play-by-play data
                # st.markdown("**Explosive Play Producers:**")
                # st.caption("Feature temporarily disabled - requires play-by-play data")

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

                # TODO: Explosive plays tracking disabled - rush_long and rec_long not in NFLverse schema
                # Need to calculate from individual game logs or play-by-play data
                # st.markdown("**Explosive Play Producers:**")
                # st.caption("Feature temporarily disabled - requires play-by-play data")

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
            'pass_comp', 'pass_att', 'pass_yds', 'pass_td', 'pass_int',
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
# Section: Projection Analytics
# ============================================================================

def render_projection_analytics(season: Optional[int], week: Optional[int]):
    """Display projection accuracy tracking and analytics."""
    st.header("📊 Projection Analytics")
    st.markdown("Track and analyze the accuracy of player yardage projections")

    # Initialize table
    init_projection_accuracy_table()

    # Create tabs
    tabs = st.tabs([
        "💾 Save Projections",
        "🔄 Update Results",
        "📊 Accuracy Dashboard",
        "🏆 Player Leaderboard",
        "📈 Projected vs Actual"
    ])

    # Tab 1: Save Projections
    with tabs[0]:
        st.subheader("💾 Save Projections")
        st.markdown("Save current week's projections before games start")

        col1, col2 = st.columns([1, 1])

        with col1:
            save_season = st.selectbox("Season", options=get_seasons(), key="save_season", index=0)
        with col2:
            save_week = st.selectbox("Week", options=list(range(1, 19)), key="save_week")

        # Check if projections already exist
        try:
            conn = sqlite3.connect(DB_PATH)
            existing_count = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM projection_accuracy WHERE season = ? AND week = ?",
                conn, params=(save_season, save_week)
            )['count'].iloc[0]
            conn.close()

            if existing_count > 0:
                st.info(f"ℹ️ {existing_count} projections already saved for Week {save_week}. Saving again will overwrite them.")
        except:
            existing_count = 0

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("💾 Save Projections", type="primary", use_container_width=True):
                with st.spinner("Saving projections..."):
                    count = save_projections_for_week(save_season, save_week)
                    if count > 0:
                        st.success(f"✅ Successfully saved {count} projections for Week {save_week}")
                        st.rerun()
                    else:
                        st.error("❌ No projections found. Make sure games are scheduled for this week.")

        # Show coverage status
        st.divider()
        st.markdown("### 📋 Projection Coverage")

        try:
            conn = sqlite3.connect(DB_PATH)
            coverage_df = pd.read_sql_query("""
                SELECT
                    season,
                    week,
                    COUNT(DISTINCT player_name) as players,
                    COUNT(*) as projections,
                    MAX(created_at) as saved_at
                FROM projection_accuracy
                GROUP BY season, week
                ORDER BY season DESC, week DESC
                LIMIT 10
            """, conn)
            conn.close()

            if not coverage_df.empty:
                coverage_df.columns = ['Season', 'Week', 'Players', 'Projections', 'Saved At']
                st.dataframe(coverage_df, use_container_width=True, hide_index=True)
            else:
                st.info("No projections saved yet")
        except Exception as e:
            st.error(f"Error loading coverage: {e}")

    # Tab 2: Update Results
    with tabs[1]:
        st.subheader("🔄 Update Actual Results")
        st.markdown("Update projections with actual game performance")

        col1, col2 = st.columns([1, 1])

        with col1:
            update_season = st.selectbox("Season", options=get_seasons(), key="update_season", index=0)
        with col2:
            update_week = st.selectbox("Week", options=list(range(1, 19)), key="update_week")

        # Check status
        try:
            conn = sqlite3.connect(DB_PATH)
            status_df = pd.read_sql_query("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN actual_yds IS NULL THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN actual_yds IS NOT NULL THEN 1 ELSE 0 END) as completed
                FROM projection_accuracy
                WHERE season = ? AND week = ?
            """, conn, params=(update_season, update_week))
            conn.close()

            if status_df['total'].iloc[0] > 0:
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Total Projections", int(status_df['total'].iloc[0]))
                col_m2.metric("Pending Update", int(status_df['pending'].iloc[0]))
                col_m3.metric("Completed", int(status_df['completed'].iloc[0]))
            else:
                st.warning(f"⚠️ No projections saved for Week {update_week}")
        except:
            st.warning("⚠️ No projections found")

        if st.button("🔄 Update Actual Results", type="primary", use_container_width=True):
            with st.spinner("Updating actuals..."):
                count = update_projection_actuals(update_season, update_week)
                if count > 0:
                    st.success(f"✅ Successfully updated {count} projections")
                    st.rerun()
                elif count == 0:
                    st.info("ℹ️ No new results to update. Games may not be completed yet.")
                else:
                    st.error("❌ Error updating results")

    # Tab 3: Accuracy Dashboard
    with tabs[2]:
        st.subheader("📊 Accuracy Dashboard")

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            dash_season = st.selectbox("Season", options=[None] + get_seasons(), key="dash_season", format_func=lambda x: "All Seasons" if x is None else str(x))
        with col2:
            dash_week = st.selectbox("Week", options=[None] + list(range(1, 19)), key="dash_week", format_func=lambda x: "All Weeks" if x is None else f"Week {x}")
        with col3:
            dash_position = st.selectbox("Position", options=[None, "QB", "RB", "WR", "TE"], key="dash_position", format_func=lambda x: "All Positions" if x is None else x)

        # Get metrics
        metrics = get_accuracy_metrics(dash_season, dash_week, dash_position)

        if metrics:
            # Key Metrics
            st.markdown("### 🎯 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Projections", metrics['count'])
            col2.metric("MAE (Yards)", f"{metrics['mae']:.1f}")
            col3.metric("RMSE", f"{metrics['rmse']:.1f}")
            col4.metric("Bias", f"{metrics['bias']:.1f}", help="Positive = over-projected, Negative = under-projected")

            # Hit Rates
            st.markdown("### 🎲 Accuracy Rates")
            col1, col2, col3 = st.columns(3)
            col1.metric("Within ±10 yards", f"{metrics['hit_rate_10']:.1f}%")
            col2.metric("Within ±20 yards", f"{metrics['hit_rate_20']:.1f}%")
            col3.metric("Within ±30 yards", f"{metrics['hit_rate_30']:.1f}%")

            # Over/Under
            st.markdown("### ⚖️ Projection Tendency")
            col1, col2, col3 = st.columns(3)
            col1.metric("Over-Projected", metrics['over_count'])
            col2.metric("Under-Projected", metrics['under_count'])
            col3.metric("Correlation", f"{metrics['correlation']:.3f}")

            # Visualizations
            st.divider()
            st.markdown("### 📈 Visualizations")

            df = get_projection_accuracy_data(dash_season, dash_week, dash_position)

            if not df.empty:
                viz_col1, viz_col2 = st.columns(2)

                with viz_col1:
                    # Scatter: Projected vs Actual
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(
                        x=df['projected_yds'],
                        y=df['actual_yds'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=df['abs_error'],
                            colorscale='RdYlGn_r',
                            showscale=True,
                            colorbar=dict(title="Error")
                        ),
                        text=df['player_name'],
                        hovertemplate='<b>%{text}</b><br>Projected: %{x:.0f}<br>Actual: %{y:.0f}<extra></extra>'
                    ))
                    # Add trend line
                    fig1.add_trace(go.Scatter(
                        x=[df['projected_yds'].min(), df['projected_yds'].max()],
                        y=[df['projected_yds'].min(), df['projected_yds'].max()],
                        mode='lines',
                        line=dict(dash='dash', color='gray'),
                        showlegend=False
                    ))
                    fig1.update_layout(
                        title="Projected vs Actual Yards",
                        xaxis_title="Projected Yards",
                        yaxis_title="Actual Yards",
                        height=400
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                with viz_col2:
                    # Histogram: Error Distribution
                    fig2 = go.Figure()
                    fig2.add_trace(go.Histogram(
                        x=df['variance'],
                        nbinsx=30,
                        marker=dict(color='steelblue')
                    ))
                    fig2.update_layout(
                        title="Error Distribution",
                        xaxis_title="Variance (Actual - Projected)",
                        yaxis_title="Count",
                        height=400
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # Accuracy by Position
                if dash_position is None:
                    st.markdown("### 📊 Accuracy by Position")
                    pos_metrics = []
                    for pos in ['QB', 'RB', 'WR', 'TE']:
                        pos_m = get_accuracy_metrics(dash_season, dash_week, pos)
                        if pos_m:
                            pos_metrics.append({
                                'Position': pos,
                                'Count': pos_m['count'],
                                'MAE': pos_m['mae'],
                                'RMSE': pos_m['rmse'],
                                'Hit Rate (±20)': pos_m['hit_rate_20']
                            })

                    if pos_metrics:
                        pos_df = pd.DataFrame(pos_metrics)
                        st.dataframe(pos_df, use_container_width=True, hide_index=True)

            # Advanced Stats (Expandable)
            with st.expander("📊 Advanced Statistics"):
                st.markdown(f"""
                **Detailed Metrics:**
                - Mean Absolute Error (MAE): {metrics['mae']:.2f} yards
                - Root Mean Square Error (RMSE): {metrics['rmse']:.2f} yards
                - Average Bias: {metrics['bias']:.2f} yards
                - Correlation Coefficient: {metrics['correlation']:.3f}
                - Average Projected: {metrics['avg_projected']:.1f} yards
                - Average Actual: {metrics['avg_actual']:.1f} yards
                - Over-Projection Rate: {(metrics['over_count']/metrics['count']*100):.1f}%
                """)
        else:
            st.info("No projection data available for the selected filters")

    # Tab 4: Player Leaderboard
    with tabs[3]:
        st.subheader("🏆 Player Accuracy Leaderboard")
        st.markdown("Players ranked by projection accuracy (lowest MAE)")

        col1, col2 = st.columns(2)
        with col1:
            lb_position = st.selectbox("Position", options=[None, "QB", "RB", "WR", "TE"], key="lb_position", format_func=lambda x: "All Positions" if x is None else x)
        with col2:
            min_proj = st.number_input("Min Projections", min_value=1, max_value=20, value=5)

        leaderboard_df = get_player_accuracy_leaderboard(lb_position, min_proj)

        if not leaderboard_df.empty:
            # Format dataframe
            display_df = leaderboard_df.copy()
            display_df.columns = ['Player', 'Pos', 'Projections', 'Avg Projected', 'Avg Actual', 'MAE', 'Over %', 'Accuracy %']
            display_df['Avg Projected'] = display_df['Avg Projected'].round(1)
            display_df['Avg Actual'] = display_df['Avg Actual'].round(1)
            display_df['MAE'] = display_df['MAE'].round(1)
            display_df['Over %'] = display_df['Over %'].round(1)
            display_df['Accuracy %'] = display_df['Accuracy %'].round(1)

            # Color-code by accuracy
            def color_accuracy(row):
                acc = row['Accuracy %']
                if acc >= 85:
                    return ['background-color: #90EE90'] * len(row)
                elif acc >= 75:
                    return ['background-color: #D3FFD3'] * len(row)
                elif acc <= 60:
                    return ['background-color: #FFD3D3'] * len(row)
                return [''] * len(row)

            styled_df = display_df.style.apply(color_accuracy, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.info(f"No players with at least {min_proj} projections")

    # Tab 5: Projected vs Actual
    with tabs[4]:
        st.subheader("📈 Projected vs Actual Details")
        st.markdown("View all projections with actual results")

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            detail_season = st.selectbox("Season", options=[None] + get_seasons(), key="detail_season", format_func=lambda x: "All" if x is None else str(x))
        with col2:
            detail_week = st.selectbox("Week", options=[None] + list(range(1, 19)), key="detail_week", format_func=lambda x: "All" if x is None else f"Week {x}")
        with col3:
            detail_position = st.selectbox("Position", options=[None, "QB", "RB", "WR", "TE"], key="detail_position", format_func=lambda x: "All" if x is None else x)

        # Get data
        detail_df = get_projection_accuracy_data(detail_season, detail_week, detail_position)

        if not detail_df.empty:
            # Format and display
            display_df = detail_df[['player_name', 'team_abbr', 'opponent_abbr', 'position',
                                    'week', 'projected_yds', 'actual_yds', 'variance',
                                    'abs_error', 'pct_error', 'matchup_rating']].copy()
            display_df.columns = ['Player', 'Team', 'Opp', 'Pos', 'Week',
                                  'Projected', 'Actual', 'Diff', 'Error', '% Error', 'Matchup']

            # Round numeric columns
            for col in ['Projected', 'Actual', 'Diff', 'Error']:
                display_df[col] = display_df[col].round(1)
            display_df['% Error'] = display_df['% Error'].round(1)

            # Color-code by error
            def color_error(row):
                error = abs(row['Error'])
                if error <= 10:
                    return ['background-color: #90EE90'] * len(row)
                elif error <= 20:
                    return ['background-color: #D3FFD3'] * len(row)
                elif error >= 40:
                    return ['background-color: #FFD3D3'] * len(row)
                return [''] * len(row)

            styled_df = display_df.style.apply(color_error, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=600)

            # Export button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"projection_accuracy_{detail_season}_{detail_week}.csv",
                mime="text/csv"
            )
        else:
            st.info("No data available for the selected filters")


# ============================================================================
# Section: Database Manager
# ============================================================================

def render_database_manager():
    """Display database management interface with refresh, info, backup, and validation tools."""
    import refresh_merged_db
    import nflverse_direct_refresh
    from datetime import datetime
    from pathlib import Path

    st.header("🗄️ Database Manager")
    st.markdown("Manage the merged NFL database, including NFLverse data refresh, backups, and integrity checks.")

    # Create tabs for different management functions
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔄 Refresh NFLverse Data",
        "📊 Database Info",
        "💾 Backup Management",
        "✅ Validation"
    ])

    # Tab 1: Refresh NFLverse Data
    with tab1:
        st.subheader("🌐 Refresh NFLverse Data from API")
        st.markdown("""
        Fetch fresh data directly from NFLverse API and update the merged database.
        This works on both local and Streamlit Cloud environments.

        **Data updated:**
        - Schedules (games, scores, betting lines)
        - Rosters (player lists with IDs)
        - Injuries (official injury reports)
        - Team statistics (offensive & defensive)
        - Advanced statistics (passing, rushing, receiving, defensive)

        **Data preserved:**
        - Play-by-play data
        - User notes and custom tracking
        """)

        st.divider()

        # Refresh controls
        col1, col2 = st.columns([3, 1])

        with col1:
            season_input = st.number_input(
                "Season to refresh",
                min_value=2020,
                max_value=2030,
                value=2025,
                help="Fetch and update data for this season from NFLverse API"
            )

        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            refresh_button = st.button("🔄 Refresh from API", type="primary", use_container_width=True)

        # Execute refresh
        if refresh_button:
            st.info("🌐 Fetching data directly from NFLverse API... This may take 30-60 seconds.")

            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Progress callback
            def show_progress(table, rows, total_tables, current):
                progress = current / total_tables
                progress_bar.progress(progress)
                status_text.text(f"[{current}/{total_tables}] {table}: {rows:,} rows updated")

            # Execute refresh using direct API
            with st.spinner("Creating backup and fetching data..."):
                results = nflverse_direct_refresh.refresh_nflverse_tables_direct(
                    season=season_input,
                    progress_callback=show_progress
                )

            # Display results
            progress_bar.empty()
            status_text.empty()

            if results['success']:
                st.success(f"✅ Refresh completed successfully!")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Rows Updated", f"{results['total_rows_updated']:,}")
                with col2:
                    st.metric("Duration", f"{results['duration_seconds']:.1f}s")

                st.info(f"**Backup saved:** {results['backup_path'].name}")

                # Show detailed table updates
                with st.expander("📋 View Detailed Updates"):
                    for table, count in results['tables_updated'].items():
                        if isinstance(count, int):
                            st.write(f"• {table}: {count:,} rows")
                        else:
                            st.write(f"• {table}: {count}")

                st.divider()
                st.markdown("### 📤 Next Steps (Local Only)")
                st.info("""
                If running locally, commit the updated database to Git:
                """)
                st.code("""
git add data/nfl_merged.db
git commit -m "chore: refresh NFLverse data from API"
git push
                """, language="bash")
                st.caption("💡 On Streamlit Cloud, data refreshes automatically on next deployment")

            else:
                st.error(f"❌ Refresh failed: {results['error']}")
                if results['backup_path']:
                    st.warning(f"Backup available at: {results['backup_path']}")
                    st.info("The database was not modified due to the error.")

                st.divider()
                st.markdown("### 🔧 Troubleshooting")
                with st.expander("Common Issues"):
                    st.markdown("""
                    **Connection Error:**
                    - Check internet connection
                    - NFLverse API may be temporarily unavailable

                    **Missing Data:**
                    - Some data may not be available for all seasons
                    - Recent seasons (current year) have most complete data

                    **Database Locked:**
                    - Close any other applications accessing the database
                    - Wait a moment and try again
                    """)

    # Tab 2: Database Info
    with tab2:
        st.subheader("Database Information")

        try:
            status = refresh_merged_db.get_database_status()

            # Database file info
            st.markdown("### Database Files")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**NFLverse Database**")
                if status['nflverse_exists']:
                    st.write(f"📁 Size: {status.get('nflverse_size_mb', 0):.2f} MB")
                    st.write(f"📅 Modified: {status['nflverse_modified'].strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    st.write("❌ Not found")

            with col2:
                st.markdown("**Merged Database**")
                if status['merged_exists']:
                    st.write(f"📁 Size: {status.get('merged_size_mb', 0):.2f} MB")
                    st.write(f"📅 Modified: {status['merged_modified'].strftime('%Y-%m-%d %H:%M:%S')}")
                    if status.get('last_refresh'):
                        st.write(f"🔄 Last Refresh: {status['last_refresh']}")
                else:
                    st.write("❌ Not found")

            # Table row counts
            if status['merged_exists'] and status['table_counts']:
                st.divider()
                st.markdown("### Table Row Counts")

                # Separate tables into categories
                nflverse_tables = [t for t in refresh_merged_db.NFLVERSE_TABLES if t in status['table_counts']]
                pfr_tables = [t for t in refresh_merged_db.PRESERVED_TABLES if t in status['table_counts']]
                other_tables = [t for t in status['table_counts'] if t not in nflverse_tables and t not in pfr_tables]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**NFLverse Tables** *(Refreshable)*")
                    for table in sorted(nflverse_tables):
                        count = status['table_counts'][table]
                        if isinstance(count, int):
                            st.write(f"• {table}: {count:,}")
                        else:
                            st.write(f"• {table}: {count}")

                with col2:
                    st.markdown("**PFR Tables** *(Preserved)*")
                    for table in sorted(pfr_tables):
                        count = status['table_counts'][table]
                        if isinstance(count, int):
                            st.write(f"• {table}: {count:,}")
                        else:
                            st.write(f"• {table}: {count}")

                with col3:
                    st.markdown("**Other Tables**")
                    for table in sorted(other_tables):
                        count = status['table_counts'][table]
                        if isinstance(count, int):
                            st.write(f"• {table}: {count:,}")
                        else:
                            st.write(f"• {table}: {count}")

        except Exception as e:
            st.error(f"Error retrieving database information: {e}")

    # Tab 3: Backup Management
    with tab3:
        st.subheader("Backup Management")

        try:
            # Find all backup files
            data_dir = Path(__file__).parent / "data"
            backup_files = sorted(
                data_dir.glob("nfl_merged.db.backup*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            if backup_files:
                st.markdown(f"Found **{len(backup_files)}** backup file(s)")
                st.divider()

                # Display backups
                for backup in backup_files:
                    with st.expander(f"📦 {backup.name}"):
                        stat = backup.stat()
                        modified = datetime.fromtimestamp(stat.st_mtime)
                        size_mb = stat.st_size / (1024 * 1024)

                        col1, col2, col3 = st.columns([2, 2, 1])

                        with col1:
                            st.write(f"**Size:** {size_mb:.2f} MB")
                        with col2:
                            st.write(f"**Created:** {modified.strftime('%Y-%m-%d %H:%M:%S')}")
                        with col3:
                            restore_key = f"restore_{backup.name}"
                            if st.button("↩️ Restore", key=restore_key):
                                try:
                                    with st.spinner("Restoring backup..."):
                                        refresh_merged_db.restore_from_backup(backup)
                                    st.success(f"✅ Restored from {backup.name}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Restore failed: {e}")
            else:
                st.info("No backup files found.")

            st.divider()

            # Manual backup creation
            st.markdown("### Create Manual Backup")
            if st.button("💾 Create Backup Now", type="secondary"):
                try:
                    with st.spinner("Creating backup..."):
                        backup_path = refresh_merged_db.backup_database()
                    st.success(f"✅ Backup created: {backup_path.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Backup failed: {e}")

        except Exception as e:
            st.error(f"Error managing backups: {e}")

    # Tab 4: Validation
    with tab4:
        st.subheader("Database Validation")
        st.markdown("Verify the integrity and completeness of the merged database.")

        if st.button("🔍 Run Validation", type="primary"):
            try:
                with st.spinner("Validating database..."):
                    results = refresh_merged_db.verify_database_integrity()

                st.divider()

                # Overall status
                if results['valid']:
                    st.success("✅ Database validation passed!")
                else:
                    st.error("❌ Database validation failed!")

                # Individual checks
                st.markdown("### Validation Checks")

                for check_name, passed in results['checks'].items():
                    if passed:
                        st.write(f"✅ {check_name.replace('_', ' ').title()}")
                    else:
                        st.write(f"❌ {check_name.replace('_', ' ').title()}")

                # Errors
                if results['errors']:
                    st.divider()
                    st.markdown("### Errors Detected")
                    for error in results['errors']:
                        st.error(error)

            except Exception as e:
                st.error(f"Validation error: {e}")


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
                        if st.button("🗑️ Delete", key=f"delete_trans_{idx}_{trans['transaction_id']}"):
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

            # Get all unique players from the season (using player_stats table for consistency with projections)
            if season:
                players_query = f"SELECT DISTINCT player_display_name as player FROM player_stats WHERE season = {season} ORDER BY player_display_name"
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
                                st.error(f"**Error details:** {error_msg}")
                            else:
                                st.error("**Error details:** No error message returned (check database table schema)")

                            # Show debug information
                            with st.expander("🔍 Debug Information"):
                                st.code(f"""
Player: {inj_player}
Team: {inj_team}
Season: {inj_season}
Injury Type: {inj_type}
Start Week: {inj_start_week}
End Week: {inj_end_week}
Description: {inj_description if inj_description else 'None'}
                                """.strip())

                            st.warning("Please check the log file (nfl_app.log) for more details or try with a different player name")

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
                            if st.button("🗑️ Remove", key=f"delete_inj_{idx}_{inj['injury_id']}"):
                                if remove_persistent_injury(inj['player_name'], inj['team_abbr'], inj['season']):
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
                {"Week": 1, "Away": "KC", "Home": "LAC", "Day": "Friday", "Primetime": "Yes", "Location": "Brazil"},
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
    """Display upcoming games schedule with week filter using NFLverse schedules table."""
    st.header("📅 Upcoming Matches")

    # Get all available weeks from schedules table (NFLverse data)
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get all unique weeks and seasons from schedules where games haven't been played yet
        # (games without scores are future games)
        cursor.execute("""
            SELECT DISTINCT season, week
            FROM schedules
            WHERE game_type = 'REG'
            ORDER BY season DESC, week ASC
        """)
        available_data = cursor.fetchall()

        if not available_data:
            st.info("No schedule data available.")
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

        # Get weeks for selected season (convert to int to avoid float type issues)
        weeks_for_season = sorted([int(row[1]) for row in available_data if row[0] == selected_season])

        with col2:
            selected_week = st.selectbox(
                "Week",
                ["All Weeks"] + weeks_for_season,
                index=0,
                key="upcoming_week_select"
            )

        # Build query - get schedule with new NFLverse data
        if selected_week == "All Weeks":
            cursor.execute("""
                SELECT
                    gameday,
                    week,
                    home_team,
                    away_team,
                    weekday,
                    gametime,
                    stadium,
                    roof,
                    surface,
                    temp,
                    wind,
                    spread_line,
                    total_line,
                    home_score,
                    away_score,
                    div_game,
                    away_rest,
                    home_rest,
                    location
                FROM schedules
                WHERE season = ? AND game_type = 'REG'
                ORDER BY week ASC, gameday ASC
            """, (selected_season,))
        else:
            cursor.execute("""
                SELECT
                    gameday,
                    week,
                    home_team,
                    away_team,
                    weekday,
                    gametime,
                    stadium,
                    roof,
                    surface,
                    temp,
                    wind,
                    spread_line,
                    total_line,
                    home_score,
                    away_score,
                    div_game,
                    away_rest,
                    home_rest,
                    location
                FROM schedules
                WHERE season = ? AND week = ? AND game_type = 'REG'
                ORDER BY gameday ASC
            """, (selected_season, selected_week))

        games = cursor.fetchall()
        conn.close()

        if not games:
            st.warning(f"No games found for the selected filter.")
            return

        # Convert to DataFrame with new NFLverse columns
        df = pd.DataFrame(games, columns=[
            'Date', 'Week', 'Home Team', 'Away Team', 'Day', 'Time',
            'Stadium', 'Roof', 'Surface', 'Temp', 'Wind',
            'Spread', 'Total', 'Home Score', 'Away Score', 'Div Game',
            'Away Rest', 'Home Rest', 'Location'
        ])

        # Convert Week column to int to prevent float type issues with Streamlit widgets
        df['Week'] = df['Week'].astype(int)

        # Determine if game is completed or upcoming
        df['Status'] = df.apply(lambda row: 'Final' if pd.notna(row['Home Score']) else 'Scheduled', axis=1)

        # Calculate power rankings for each team using the same 4-step process as Power Rankings view
        # Use the selected season and the week before the first game for power rating calculation
        # Convert to int to avoid float type issues with Streamlit
        power_rating_week = int(df['Week'].min()) if selected_week == "All Weeks" else int(selected_week)

        # Get all unique teams in the schedule
        all_teams = set(df['Home Team'].tolist() + df['Away Team'].tolist())

        # Step 1: Calculate league statistics
        league_stats = calculate_league_statistics(selected_season, power_rating_week, list(all_teams))

        # Step 2: Calculate baseline power ratings for all teams (for SOS calculation)
        all_team_powers = {}
        for team in all_teams:
            try:
                power = calculate_team_power_rating(team, selected_season, power_rating_week, all_team_powers=None, league_stats=league_stats)
                all_team_powers[team] = power
            except:
                all_team_powers[team] = 0.0

        # Step 3: Calculate quality margin-adjusted league stats
        quality_margins = []
        for team in all_teams:
            try:
                qm = calculate_quality_victory_margin(team, selected_season, power_rating_week, all_team_powers)
                quality_margins.append(qm.get('quality_margin_per_game', 0))
            except:
                quality_margins.append(0)

        if len(quality_margins) > 1:
            import statistics
            league_stats['quality_margin'] = {
                'mean': statistics.mean(quality_margins),
                'std': statistics.stdev(quality_margins) if len(quality_margins) > 1 else 1.0
            }

        # Step 4: Calculate final power ratings with quality margin adjustments
        week_teams_data = []
        for team in all_teams:
            try:
                final_power = calculate_team_power_rating(team, selected_season, power_rating_week, all_team_powers, league_stats)
                team_record = calculate_win_loss_record(team, selected_season, power_rating_week)
                wins = team_record.get('wins', 0)
                losses = team_record.get('losses', 0)

                week_teams_data.append({
                    'team': team,
                    'power': final_power,
                    'wins': wins,
                    'losses': losses
                })
            except:
                week_teams_data.append({
                    'team': team,
                    'power': 0.0,
                    'wins': 0,
                    'losses': 0
                })

        # Sort by record first, then by power rating (same as Power Rankings view)
        week_teams_data.sort(key=lambda x: (-x['wins'], x['losses'], -x['power']))

        # Normalize power ratings to 1-100 scale based on rank position
        final_team_powers = {}
        if len(week_teams_data) > 0:
            for rank_idx, team_data in enumerate(week_teams_data):
                # Percentile based on position in sorted list (wins-dominant ranking)
                rank_pct = (rank_idx / len(week_teams_data)) * 100
                # Invert so best team (rank_idx=0) gets ~100, worst gets ~1
                normalized_power = max(1, min(100, 100 - rank_pct))
                final_team_powers[team_data['team']] = normalized_power

        def get_power_ranking(team, week):
            return round(final_team_powers.get(team, 50.0), 1)

        # Add power ranking columns
        df['Home Power'] = df.apply(lambda row: get_power_ranking(row['Home Team'], row['Week']), axis=1)
        df['Away Power'] = df.apply(lambda row: get_power_ranking(row['Away Team'], row['Week']), axis=1)

        # Add analysis columns for completed games
        def check_power_winner(row):
            """Check if team with higher power score won the game"""
            if pd.isna(row['Home Score']) or pd.isna(row['Away Score']):
                return ""  # Game not played yet

            home_won = row['Home Score'] > row['Away Score']
            power_favorite_is_home = row['Home Power'] > row['Away Power']

            # If home team won and had higher power, or away team won and had higher power
            if (home_won and power_favorite_is_home) or (not home_won and not power_favorite_is_home):
                return "✓"
            return "✗"

        def check_spread_cover(row):
            """Check if favorite covered the spread"""
            if pd.isna(row['Home Score']) or pd.isna(row['Away Score']) or pd.isna(row['Spread']):
                return ""  # Game not played or no spread available

            # Spread is from home team perspective (negative means home favored)
            # Example: -3.5 means home team favored by 3.5
            actual_margin = row['Home Score'] - row['Away Score']  # Positive if home won

            # Home team covers if: actual_margin > spread (since spread is negative for favorites)
            # Away team covers if: actual_margin < spread
            if row['Spread'] < 0:  # Home team favored
                # Home needs to win by more than the spread magnitude
                return "✓" if actual_margin > abs(row['Spread']) else "✗"
            else:  # Away team favored (or pick'em if 0)
                # Away needs to win by more than spread, or home needs to lose by less
                return "✓" if actual_margin < -abs(row['Spread']) else "✗"

        def check_over_under(row):
            """Check if total went over the O/U line"""
            if pd.isna(row['Home Score']) or pd.isna(row['Away Score']) or pd.isna(row['Total']):
                return ""  # Game not played or no total available

            actual_total = row['Home Score'] + row['Away Score']
            return "✓" if actual_total > row['Total'] else "✗"

        df['Power Winner ✓'] = df.apply(check_power_winner, axis=1)
        df['Favorite Covered ✓'] = df.apply(check_spread_cover, axis=1)
        df['Over ✓'] = df.apply(check_over_under, axis=1)

        # Create matchup column with location info
        def format_matchup(row):
            matchup = f"{row['Away Team']} @ {row['Home Team']}"
            if row['Location'] and row['Location'] != row['Home Team']:
                matchup += f" ({row['Location']})"
            return matchup

        df['Matchup'] = df.apply(format_matchup, axis=1)

        # Display summary metrics
        st.divider()
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Games", len(df))
        with col2:
            st.metric("Weeks", df['Week'].nunique())
        with col3:
            completed = (df['Status'] == 'Final').sum()
            st.metric("Completed", completed)
        with col4:
            upcoming = (df['Status'] == 'Scheduled').sum()
            st.metric("Upcoming", upcoming)
        with col5:
            div_games = (df['Div Game'] == 1).sum() if 'Div Game' in df.columns else 0
            st.metric("Division Games", div_games)

        st.divider()

        # Group by week if showing all weeks
        if selected_week == "All Weeks":
            # df['Week'] already converted to int, but ensure unique() returns ints not floats
            for week_num in sorted(int(w) for w in df['Week'].unique()):
                week_games = df[df['Week'] == week_num].copy()

                with st.expander(f"Week {week_num} ({len(week_games)} games)", expanded=(week_num == weeks_for_season[0])):
                    # Display games for this week with enhanced data
                    display_df = week_games[[
                        'Date', 'Time', 'Day', 'Home Team', 'Home Power', 'Home Score',
                        'Away Team', 'Away Power', 'Away Score',
                        'Status', 'Spread', 'Power Winner ✓', 'Favorite Covered ✓',
                        'Total', 'Over ✓', 'Stadium', 'Roof', 'Temp'
                    ]].copy()

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Date": st.column_config.DateColumn("Date", format="MMM DD"),
                            "Time": st.column_config.TextColumn("Time", width="small"),
                            "Day": st.column_config.TextColumn("Day", width="small"),
                            "Home Team": st.column_config.TextColumn("Home", width="small"),
                            "Home Power": st.column_config.NumberColumn("H Pwr", width="small", format="%.1f"),
                            "Home Score": st.column_config.NumberColumn("H Scr", width="small", format="%.0f"),
                            "Away Team": st.column_config.TextColumn("Away", width="small"),
                            "Away Power": st.column_config.NumberColumn("A Pwr", width="small", format="%.1f"),
                            "Away Score": st.column_config.NumberColumn("A Scr", width="small", format="%.0f"),
                            "Status": st.column_config.TextColumn("Status", width="small"),
                            "Spread": st.column_config.NumberColumn("Spread", width="small", format="%.1f"),
                            "Power Winner ✓": st.column_config.TextColumn("Pwr ✓", width="small"),
                            "Favorite Covered ✓": st.column_config.TextColumn("Cov ✓", width="small"),
                            "Total": st.column_config.NumberColumn("O/U", width="small", format="%.1f"),
                            "Over ✓": st.column_config.TextColumn("O ✓", width="small"),
                            "Stadium": st.column_config.TextColumn("Stadium", width="medium"),
                            "Roof": st.column_config.TextColumn("Roof", width="small"),
                            "Temp": st.column_config.NumberColumn("Temp", width="small", format="%.0f°")
                        }
                    )
        else:
            # Display single week with enhanced data
            st.subheader(f"Week {selected_week} Schedule")

            display_df = df[[
                'Date', 'Time', 'Day', 'Home Team', 'Home Power', 'Home Score',
                'Away Team', 'Away Power', 'Away Score',
                'Status', 'Spread', 'Power Winner ✓', 'Favorite Covered ✓',
                'Total', 'Over ✓', 'Stadium', 'Roof', 'Surface', 'Temp', 'Wind'
            ]].copy()

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date": st.column_config.DateColumn("Date", format="MMM DD, YYYY"),
                    "Time": st.column_config.TextColumn("Time", width="small"),
                    "Day": st.column_config.TextColumn("Day", width="small"),
                    "Home Team": st.column_config.TextColumn("Home", width="small"),
                    "Home Power": st.column_config.NumberColumn("H Pwr", width="small", format="%.1f"),
                    "Home Score": st.column_config.NumberColumn("H Scr", width="small", format="%.0f"),
                    "Away Team": st.column_config.TextColumn("Away", width="small"),
                    "Away Power": st.column_config.NumberColumn("A Pwr", width="small", format="%.1f"),
                    "Away Score": st.column_config.NumberColumn("A Scr", width="small", format="%.0f"),
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Spread": st.column_config.NumberColumn("Spread", width="small", format="%.1f"),
                    "Power Winner ✓": st.column_config.TextColumn("Pwr ✓", width="small"),
                    "Favorite Covered ✓": st.column_config.TextColumn("Cov ✓", width="small"),
                    "Total": st.column_config.NumberColumn("O/U", width="small", format="%.1f"),
                    "Over ✓": st.column_config.TextColumn("O ✓", width="small"),
                    "Stadium": st.column_config.TextColumn("Stadium", width="medium"),
                    "Roof": st.column_config.TextColumn("Roof", width="small"),
                    "Surface": st.column_config.TextColumn("Surface", width="small"),
                    "Temp": st.column_config.NumberColumn("Temp", width="small", format="%.0f°"),
                    "Wind": st.column_config.NumberColumn("Wind", width="small", format="%.0f mph")
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
                # Create tabs for each position (removed TEs)
                proj_tabs = st.tabs(["🎯 Top QBs", "🏃 Top RBs", "🙌 Top WRs", "⭐ Top Skill Players"])

                # QB Tab
                with proj_tabs[0]:
                    if not projections.get('QB', pd.DataFrame()).empty:
                        st.markdown("##### Quarterbacks - Comprehensive Matchup Analysis")
                        st.caption("Ranked by QB Score (0-100): Multi-factor evaluation of QB production vs. defensive matchup")

                        qb_df = projections['QB'].head(30).copy()

                        # Style the dataframe with tier-based colors
                        def style_qb_tier(row):
                            tier = row['Tier']
                            if '🔥🔥🔥' in tier:  # ELITE SMASH SPOT
                                return ['background-color: #0A5F0F; color: white'] * len(row)
                            elif '🔥🔥' in tier:  # PREMIUM MATCHUP
                                return ['background-color: #228B22; color: white'] * len(row)
                            elif '🔥' in tier:  # SMASH SPOT
                                return ['background-color: #90EE90'] * len(row)
                            elif '✅' in tier:  # SOLID START
                                return ['background-color: #E8F5E9'] * len(row)
                            elif '⚠️' in tier:  # RISKY PLAY
                                return ['background-color: #FFE4B5'] * len(row)
                            elif '🛑' in tier:  # AVOID
                                return ['background-color: #FFB6C1'] * len(row)
                            else:  # BALANCED
                                return [''] * len(row)

                        styled_df = qb_df.style.apply(style_qb_tier, axis=1)

                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "QB Score": st.column_config.NumberColumn("QB Score", format="%.1f", help="Comprehensive 0-100 score evaluating production + matchup"),
                                "Tier": st.column_config.TextColumn("Tier", width="medium"),
                                "Avg Yds/Game": st.column_config.NumberColumn("Avg Yds/Game", format="%.1f"),
                                "TDs/Game": st.column_config.NumberColumn("TDs/Game", format="%.2f"),
                                "INT Rate %": st.column_config.NumberColumn("INT Rate %", format="%.1f", help="Interceptions per 100 attempts"),
                                "Pass TDs": st.column_config.NumberColumn("Pass TDs", format="%.0f"),
                                "Pass INTs": st.column_config.NumberColumn("Pass INTs", format="%.0f"),
                                "Rush TDs": st.column_config.NumberColumn("Rush TDs", format="%.0f"),
                                "Def Allows": st.column_config.NumberColumn("Def Allows", format="%.1f", help="Pass yards allowed per game"),
                                "Def Sacks": st.column_config.NumberColumn("Def Sacks", format="%.0f"),
                                "Def INTs": st.column_config.NumberColumn("Def INTs", format="%.0f"),
                                "Pressure Score": st.column_config.NumberColumn("Pressure Score", format="%.1f", help="Sacks + 0.5*Hurries + 0.3*Blitzes"),
                                "Projected Yds": st.column_config.NumberColumn("Projected Yds", format="%.1f"),
                                "Games": st.column_config.NumberColumn("Games", format="%.1f")
                            }
                        )

                        # Show storylines/recommendations in expandable section
                        with st.expander("📊 View Detailed QB Recommendations"):
                            for _, qb in qb_df.iterrows():
                                st.markdown(f"**{qb['Player']} ({qb['Team']}) vs {qb['Opponent']}** - Score: {qb['QB Score']}")
                                st.markdown(f"_{qb['Recommendation']}_")
                                st.markdown("---")
                    else:
                        st.info("No QB data available for this week")

                # RB Tab
                with proj_tabs[1]:
                    if not projections.get('RB', pd.DataFrame()).empty:
                        st.markdown("##### Running Backs - Comprehensive Matchup Analysis")
                        st.caption("Ranked by RB Score (0-100): Multi-factor evaluation of rushing production + receiving role + TD scoring vs defensive matchup quality")

                        rb_df = projections['RB'].head(30).copy()

                        # Style with tier-based colors (same as QB)
                        def style_rb_tier(row):
                            tier = row['Tier']
                            if '🔥🔥🔥' in tier:  # ELITE SMASH SPOT
                                return ['background-color: #0A5F0F; color: white'] * len(row)
                            elif '🔥🔥' in tier:  # PREMIUM MATCHUP
                                return ['background-color: #228B22; color: white'] * len(row)
                            elif '🔥' in tier:  # SMASH SPOT
                                return ['background-color: #90EE90'] * len(row)
                            elif '✅' in tier:  # SOLID START
                                return ['background-color: #E8F5E9'] * len(row)
                            elif '⚖️' in tier:  # BALANCED
                                return [''] * len(row)
                            elif '⚠️' in tier:  # RISKY PLAY
                                return ['background-color: #FFE4B5'] * len(row)
                            elif '🛑' in tier:  # AVOID
                                return ['background-color: #FFB6C1'] * len(row)
                            else:
                                return [''] * len(row)

                        styled_df = rb_df.style.apply(style_rb_tier, axis=1)

                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "RB Score": st.column_config.NumberColumn("RB Score", format="%.1f",
                                    help="Comprehensive 0-100 score: rush yards (25pts) + rush TDs (20pts) + receiving role (20pts) + rec TDs (10pts) + defensive matchup (25pts)"),
                                "Tier": st.column_config.TextColumn("Tier", width="medium"),
                                "Player": st.column_config.TextColumn("Player"),
                                "Team": st.column_config.TextColumn("Team"),
                                "Opponent": st.column_config.TextColumn("Opp"),
                                "Rush Yds/Gm": st.column_config.NumberColumn("Rush Yds/Gm", format="%.1f"),
                                "Rec Yds/Gm": st.column_config.NumberColumn("Rec Yds/Gm", format="%.1f"),
                                "Rush TDs/Gm": st.column_config.NumberColumn("Rush TDs/Gm", format="%.2f"),
                                "Rec TDs/Gm": st.column_config.NumberColumn("Rec TDs/Gm", format="%.2f"),
                                "Total TDs/Gm": st.column_config.NumberColumn("Total TDs/Gm", format="%.2f"),
                                "Targets/Gm": st.column_config.NumberColumn("Targets/Gm", format="%.1f",
                                    help="Targets per game (PPR value indicator)"),
                                "Touches/Gm": st.column_config.NumberColumn("Touches/Gm", format="%.1f",
                                    help="Carries + Targets per game (total opportunity)"),
                                "Def Rush Yds": st.column_config.NumberColumn("Def Rush Yds", format="%.1f"),
                                "Def Rush TDs": st.column_config.NumberColumn("Def Rush TDs", format="%.1f")
                            }
                        )

                        # Show storylines/recommendations in expandable section
                        with st.expander("📊 View Detailed RB Recommendations"):
                            for _, rb in rb_df.iterrows():
                                st.markdown(f"**{rb['Player']} ({rb['Team']}) vs {rb['Opponent']}** - Score: {rb['RB Score']}")
                                st.markdown(f"_{rb['Recommendation']}_")
                                st.markdown("---")
                    else:
                        st.info("No RB data available for this week")

                # WR Tab
                with proj_tabs[2]:
                    if not projections.get('WR', pd.DataFrame()).empty:
                        st.markdown("##### Wide Receivers - Comprehensive Matchup Analysis")
                        st.caption("Ranked by WR Score (0-100): Multi-factor evaluation of WR production vs. defensive matchup")

                        wr_df = projections['WR'].head(30).copy()

                        # Style the dataframe with tier-based colors
                        def style_wr_tier(row):
                            tier = row['Tier']
                            if '🔥🔥🔥' in tier:  # ELITE SMASH SPOT
                                return ['background-color: #0A5F0F; color: white'] * len(row)
                            elif '🔥🔥' in tier:  # PREMIUM MATCHUP
                                return ['background-color: #228B22; color: white'] * len(row)
                            elif '🔥' in tier:  # SMASH SPOT
                                return ['background-color: #90EE90'] * len(row)
                            elif '✅' in tier:  # SOLID START
                                return ['background-color: #E8F5E9'] * len(row)
                            elif '⚠️' in tier:  # RISKY PLAY
                                return ['background-color: #FFE4B5'] * len(row)
                            elif '🛑' in tier:  # AVOID
                                return ['background-color: #FFB6C1'] * len(row)
                            else:  # BALANCED
                                return [''] * len(row)

                        styled_df = wr_df.style.apply(style_wr_tier, axis=1)

                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "WR Score": st.column_config.NumberColumn("WR Score", format="%.1f", help="Comprehensive 0-100 score evaluating production + matchup"),
                                "Tier": st.column_config.TextColumn("Tier", width="medium"),
                                "Rec Yds/Gm": st.column_config.NumberColumn("Rec Yds/Gm", format="%.1f"),
                                "Rec TDs/Gm": st.column_config.NumberColumn("Rec TDs/Gm", format="%.2f"),
                                "Targets/Gm": st.column_config.NumberColumn("Targets/Gm", format="%.1f"),
                                "Target Share %": st.column_config.NumberColumn("Target Share %", format="%.1f", help="Percentage of team targets"),
                                "Avg Yds/Game": st.column_config.NumberColumn("Avg Yds/Game", format="%.1f"),
                                "Median Rec Yds": st.column_config.NumberColumn("Median Rec Yds", format="%.1f"),
                                "Median Tgts": st.column_config.NumberColumn("Median Tgts", format="%.1f"),
                                "Total Targets": st.column_config.NumberColumn("Total Targets", format="%.0f"),
                                "Total Receptions": st.column_config.NumberColumn("Total Receptions", format="%.0f"),
                                "Last 3 Avg Tgts": st.column_config.NumberColumn("Last 3 Avg Tgts", format="%.1f"),
                                "Last 3 Avg Rec": st.column_config.NumberColumn("Last 3 Avg Rec", format="%.1f"),
                                "Rec TDs": st.column_config.NumberColumn("Rec TDs", format="%.0f"),
                                "Def Rec Yds": st.column_config.NumberColumn("Def Rec Yds", format="%.1f", help="Rec yards allowed to WRs per game"),
                                "Def Rec TDs": st.column_config.NumberColumn("Def Rec TDs", format="%.0f"),
                                "Def Rec Rank": st.column_config.NumberColumn("Def Rec Rank", format="%.0f"),
                                "Projected Yds": st.column_config.NumberColumn("Projected Yds", format="%.1f"),
                                "Games": st.column_config.NumberColumn("Games", format="%.1f")
                            }
                        )

                        # Show storylines/recommendations in expandable section
                        with st.expander("📊 View Detailed WR Recommendations"):
                            for _, wr in wr_df.iterrows():
                                st.markdown(f"**{wr['Player']} ({wr['Team']}) vs {wr['Opponent']}** - Score: {wr['WR Score']}")
                                st.markdown(f"_{wr['Recommendation']}_")
                                st.markdown("---")
                    else:
                        st.info("No WR data available for this week")

                # Skill Players Combined Tab
                with proj_tabs[3]:
                    if not projections.get('SKILL', pd.DataFrame()).empty:
                        st.markdown("##### ⭐ Top Skill Position Players - Comprehensive Matchup Analysis")
                        st.caption("Multi-factor scoring combining player production (yards, TDs, touches) with defensive matchup quality. RBs and WRs ranked by their respective comprehensive scores.")

                        skill_df = projections['SKILL'].copy()

                        # Add unified Score and Position columns for sorting
                        skill_df['Score'] = skill_df.apply(lambda row: row.get('RB Score', row.get('WR Score', 0)), axis=1)
                        skill_df['Position'] = skill_df['Team'].apply(lambda x: 'RB' if '(RB)' in x else ('WR' if '(WR)' in x else 'TE'))

                        # Sort by Score and limit to top 60
                        skill_df = skill_df.sort_values('Score', ascending=False).head(60)

                        # Define tier-based color coding function (matching QB/RB/WR)
                        def get_tier_color(tier):
                            tier_colors = {
                                'ELITE SMASH': 'background-color: #006400; color: white;',  # Dark green
                                'GREAT PLAY': 'background-color: #228B22; color: white;',   # Forest green
                                'FAVORABLE': 'background-color: #32CD32; color: black;',    # Lime green
                                'SOLID': 'background-color: #90EE90; color: black;',        # Light green
                                'BALANCED': 'background-color: #FFFFE0; color: black;',     # Light yellow
                                'TOUGH': 'background-color: #FFB6C1; color: black;',        # Light pink
                                'AVOID': 'background-color: #FF69B4; color: white;'         # Hot pink
                            }
                            return tier_colors.get(tier, '')

                        def style_skill_tier(row):
                            tier = row.get('Tier', '')
                            color = get_tier_color(tier)
                            return [color] * len(row) if color else [''] * len(row)

                        # Select columns to display
                        display_cols = ['Player', 'Team', 'Opponent', 'Score', 'Tier']

                        # Add RB-specific or WR-specific columns based on position
                        for col in ['Rush Yds/Gm', 'Rec Yds/Gm', 'Rush TDs/Gm', 'Rec TDs/Gm', 'Total TDs/Gm', 'Touches/Gm', 'Targets/Gm', 'Target Share %', 'Recommendation']:
                            if col in skill_df.columns:
                                display_cols.append(col)

                        # Filter to display_cols that exist
                        display_cols = [col for col in display_cols if col in skill_df.columns]
                        display_df = skill_df[display_cols].copy()

                        styled_df = display_df.style.apply(style_skill_tier, axis=1)

                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Player": st.column_config.TextColumn("Player", width="medium"),
                                "Team": st.column_config.TextColumn("Team", width="small"),
                                "Opponent": st.column_config.TextColumn("Opp", width="small"),
                                "Score": st.column_config.NumberColumn(
                                    "Score",
                                    help="Comprehensive matchup score (0-100): Combines production metrics + defensive matchup quality",
                                    format="%.1f",
                                    width="small"
                                ),
                                "Tier": st.column_config.TextColumn("Tier", width="medium"),
                                "Rush Yds/Gm": st.column_config.NumberColumn("Rush Yds/Gm", format="%.1f", width="small"),
                                "Rec Yds/Gm": st.column_config.NumberColumn("Rec Yds/Gm", format="%.1f", width="small"),
                                "Rush TDs/Gm": st.column_config.NumberColumn("Rush TDs/Gm", format="%.2f", width="small"),
                                "Rec TDs/Gm": st.column_config.NumberColumn("Rec TDs/Gm", format="%.2f", width="small"),
                                "Total TDs/Gm": st.column_config.NumberColumn("Total TDs/Gm", format="%.2f", width="small"),
                                "Touches/Gm": st.column_config.NumberColumn("Touches/Gm", format="%.1f", width="small"),
                                "Targets/Gm": st.column_config.NumberColumn("Targets/Gm", format="%.1f", width="small"),
                                "Target Share %": st.column_config.NumberColumn("Target %", format="%.1f", width="small"),
                                "Recommendation": st.column_config.TextColumn("Recommendation", width="large")
                            }
                        )

                        # Expandable section for detailed recommendations
                        with st.expander("📋 View Detailed Skill Player Recommendations", expanded=False):
                            st.markdown("### Complete Matchup Analysis")

                            for idx, row in skill_df.iterrows():
                                if pd.notna(row.get('Recommendation')):
                                    position = row['Position']
                                    score = row['Score']
                                    tier = row.get('Tier', 'N/A')

                                    st.markdown(f"**{row['Player']}** ({position}) - {row['Team']} vs {row['Opponent']}")
                                    st.markdown(f"*Score: {score:.1f} | Tier: {tier}*")
                                    st.markdown(f"> {row['Recommendation']}")
                                    st.divider()
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

                # ========== AIR YARDS / YAC MATCHUP ANALYSIS ==========
                st.divider()
                with st.expander("🎯 **Air Yards vs YAC Matchup Analysis** - Identify WR/TE Opportunities", expanded=False):
                    st.markdown("""
                    This analysis shows which teams' offensive passing tendencies match up well against their opponent's defensive vulnerabilities.
                    Look for teams with vertical passing attacks facing defenses that allow yards after catch (YAC).
                    """)

                    # Get upcoming games for this week
                    upcoming_games_df = df[['Home Team', 'Away Team']].copy()
                    upcoming_games_df.columns = ['home_team', 'away_team']

                    render_air_yac_matchup_chart(selected_season, selected_week, upcoming_games_df)

                # ========== QB PRESSURE MATCHUP ANALYSIS ==========
                st.divider()
                with st.expander("⚡ **QB Pressure Matchup Analysis** - Identify Protection Concerns", expanded=False):
                    st.markdown("""
                    This analysis reveals quarterback protection vulnerabilities by comparing QB pressure rates against opposing defensive pass rush strength.
                    **DANGER ZONE** matchups indicate QBs who struggle under pressure facing aggressive defenses.
                    """)

                    # Use same upcoming games dataframe
                    render_qb_pressure_matchup_chart(selected_season, selected_week, upcoming_games_df)

                # ========== RUSHING TD EFFICIENCY MATCHUP ANALYSIS ==========
                st.divider()
                with st.expander("🏈 **Rushing TD Efficiency Matchup** - Identify RB Touchdown Opportunities", expanded=False):
                    st.markdown("""
                    Match team rushing TD scoring rates vs defensive TD prevention (API data: `rushing_tds`).
                    Look for **EXPLOIT** and **SMASH SPOT** matchups where high-scoring offenses face TD-vulnerable defenses.
                    """)

                    # Use same upcoming games dataframe
                    render_rushing_td_matchup_chart(selected_season, selected_week, upcoming_games_df)

                # ========== PLAYER RUSH TD VS DEFENSE MATCHUP ==========
                st.divider()
                with st.expander("🏃‍♂️ **Player Rush TD vs Defense** - Individual TD Opportunities", expanded=False):
                    st.markdown("""
                    Analyze individual player rushing TD rates vs defensive vulnerability.
                    Identifies **TD SMASH** opportunities where prolific scorers face generous defenses.
                    """)

                    # Use same upcoming games dataframe
                    render_player_rush_td_matchup_chart(selected_season, selected_week, upcoming_games_df)

                # ========== RB RUSHING YARDS MATCHUP ANALYSIS ==========
                st.divider()
                with st.expander("🏃 **RB Rushing Yards Matchup** - Identify Volume Opportunities", expanded=False):
                    st.markdown("""
                    Match RB rushing yards vs defensive rushing yards allowed (API data: `rushing_yards`).
                    Look for **CEILING EXPLOSION** and **VOLUME SPIKE** matchups for yardage upside.
                    """)

                    # Use same upcoming games dataframe
                    render_rushing_yards_matchup_chart(selected_season, selected_week, upcoming_games_df)

                # ========== RB PASS-CATCHING MATCHUP ANALYSIS ==========
                st.divider()
                with st.expander("🎯 **RB Pass-Catching Matchup** - Identify PPR RB Value", expanded=False):
                    st.markdown("""
                    Identify pass-catching RBs in favorable PPR matchups vs defenses vulnerable to RB targets.
                    Look for **PPR SMASH** and **CHECKDOWN CITY** matchups for PPR league plays.
                    """)

                    # Use same upcoming games dataframe
                    render_rb_receiving_matchup_chart(selected_season, selected_week, upcoming_games_df)

                # ========== QB PASSING TD MATCHUP ANALYSIS ==========
                st.divider()
                with st.expander("🎯 **QB Passing TD Matchup** - Identify QB Touchdown Opportunities", expanded=False):
                    st.markdown("""
                    Match QB passing TD production vs defensive TD prevention (API data: `passing_tds`).
                    Look for **TD SMASH SPOT** and **SNEAKY VALUE** matchups for QB streaming.
                    """)

                    render_passing_td_matchup_chart(selected_season, selected_week, upcoming_games_df)

                # ========== QB PASSING YARDS MATCHUP ANALYSIS ==========
                st.divider()
                with st.expander("🚀 **QB Passing Yards Matchup** - Identify Volume Opportunities", expanded=False):
                    st.markdown("""
                    Match QB passing yards vs defensive yards allowed (API data: `passing_yards`).
                    Look for **CEILING EXPLOSION** and **VOLUME SPIKE** matchups for yardage upside.
                    """)

                    render_passing_yards_matchup_chart(selected_season, selected_week, upcoming_games_df)

                # ========== QB TD RATE VS DEFENSIVE INT RATE ==========
                st.divider()
                with st.expander("⚡ **QB Efficiency vs INT Risk** - Risk-Reward Analysis", expanded=False):
                    st.markdown("""
                    Match QB TD efficiency vs defensive INT creation (API: `passing_tds`/attempts vs `def_interceptions`).
                    Look for **HIGH CEILING LOW RISK** plays and avoid **DANGER ZONE** matchups.
                    """)

                    render_qb_td_rate_vs_int_chart(selected_season, selected_week, upcoming_games_df)

                # ========== QB EFFICIENCY MATCHUP (COMPOSITE) ==========
                st.divider()
                with st.expander("💎 **QB Composite Efficiency** - Multi-Factor Matchup Analysis", expanded=False):
                    st.markdown("""
                    Composite QB efficiency (yards + INT rate) vs defensive passing efficiency (yards allowed + INTs + sacks).
                    Look for **SAFE CEILING SPOT** and **VOLUME OPPORTUNITY** matchups while avoiding **DEFENSIVE WALL** games.
                    """)

                    render_qb_efficiency_matchup_chart(selected_season, selected_week, upcoming_games_df)

                # ========== COMPREHENSIVE MATCHUP RECOMMENDATIONS ==========
                st.divider()
                st.header("📊 Comprehensive Matchup Recommendations")
                st.markdown("""
                Synthesizes all 8 matchup storylines into actionable position-specific insights.
                - **🟢 ELITE:** Must-start with top-tier upside
                - **🟡 FAVORABLE:** Good play with solid floor
                - **⚪ NEUTRAL:** Game script dependent
                - **🔴 AVOID:** Bench or pivot to better matchup
                """)

                # Get upcoming games for matchup recommendations
                upcoming_games_df = df[['Home Team', 'Away Team']].copy()
                upcoming_games_df.columns = ['home_team', 'away_team']

                render_matchup_recommendations_table(selected_season, selected_week, upcoming_games_df)

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
            "Team PPG Home vs Away",
            "Team Plays per Game vs Points per Game",
            "Defense Yards Allowed (Pass vs Rush)",
            "Air Yards vs YAC Matchup Analysis",
            "QB Pressure Matchup Analysis",
            "Rushing TD Efficiency Matchup Analysis",
            "RB Rushing Yards Matchup Analysis",
            "RB Pass-Catching Matchup Analysis",
            "QB Passing TD Matchup Analysis",
            "QB Passing Yards Matchup Analysis",
            "QB TD Rate vs Defensive INT Rate",
            "QB Composite Efficiency Matchup",
            "Power Rating vs Offensive Yards",
            "RB Efficiency (Rush Yards vs TDs)",
            "RB Yards per Carry",
            "Skill Player Total Yards vs Touches",
            "WR Efficiency (Targets vs Yards per Route Run)",
            "QB Passing TDs vs Interceptions",
            "QB Passing Yards vs Attempts",
            "QB Rush Yards vs Pass Yards"
        ]
    )

    st.divider()

    if chart_type == "Team Offense Efficiency (Yards vs Points)":
        render_team_offense_efficiency_chart(season, week)
    elif chart_type == "Team Balance (Points Scored vs Allowed)":
        render_team_balance_chart(season, week)
    elif chart_type == "Team PPG Home vs Away":
        render_team_ppg_home_away_chart(season, week)
    elif chart_type == "Team Plays per Game vs Points per Game":
        render_team_plays_ppg_chart(season, week)
    elif chart_type == "Defense Yards Allowed (Pass vs Rush)":
        render_defense_yards_allowed_chart(season, week)
    elif chart_type == "Air Yards vs YAC Matchup Analysis":
        render_air_yac_matchup_chart(season, week, upcoming_games=None)
    elif chart_type == "QB Pressure Matchup Analysis":
        render_qb_pressure_matchup_chart(season, week, upcoming_games=None)
    elif chart_type == "Rushing TD Efficiency Matchup Analysis":
        render_rushing_td_matchup_chart(season, week, upcoming_games=None)
    elif chart_type == "RB Rushing Yards Matchup Analysis":
        render_rushing_yards_matchup_chart(season, week, upcoming_games=None)
    elif chart_type == "RB Pass-Catching Matchup Analysis":
        render_rb_receiving_matchup_chart(season, week, upcoming_games=None)
    elif chart_type == "QB Passing TD Matchup Analysis":
        render_passing_td_matchup_chart(season, week, upcoming_games=None)
    elif chart_type == "QB Passing Yards Matchup Analysis":
        render_passing_yards_matchup_chart(season, week, upcoming_games=None)
    elif chart_type == "QB TD Rate vs Defensive INT Rate":
        render_qb_td_rate_vs_int_chart(season, week, upcoming_games=None)
    elif chart_type == "QB Composite Efficiency Matchup":
        render_qb_efficiency_matchup_chart(season, week, upcoming_games=None)
    elif chart_type == "Power Rating vs Offensive Yards":
        render_power_rating_yards_chart(season, week)
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
    elif chart_type == "QB Rush Yards vs Pass Yards":
        render_qb_rush_vs_pass_yards_chart(season, week)


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
    - **Bottom-left quadrant:** Elite defense (low rush & pass yards allowed)
    - **Bottom-right:** Weak against run, strong against pass
    - **Top-left:** Strong against run, weak against pass
    - **Top-right:** Struggling defense overall (high yards allowed)
    """)

    try:
        week_filter = f"AND g.week <= {week}" if week else ""

        # Get defensive stats by aggregating opponent offensive yards
        # For each game, calculate what each team allowed
        sql_defense = f"""
        SELECT
            defense_team,
            COUNT(DISTINCT game_id) as games,
            SUM(pass_yds_allowed) as total_pass_yds_allowed,
            SUM(rush_yds_allowed) as total_rush_yds_allowed
        FROM (
            -- Home team defense (allowed away team offense)
            SELECT
                g.game_id,
                g.home_team_abbr as defense_team,
                SUM(CASE WHEN pb.pass_yds > 0 THEN pb.pass_yds ELSE 0 END) as pass_yds_allowed,
                SUM(CASE WHEN pb.rush_yds > 0 THEN pb.rush_yds ELSE 0 END) as rush_yds_allowed
            FROM games g
            JOIN player_box_score pb ON g.season = pb.season AND g.week = pb.week AND pb.team = g.away_team_abbr
            WHERE g.season = ?
            {week_filter}
            GROUP BY g.game_id, g.home_team_abbr

            UNION ALL

            -- Away team defense (allowed home team offense)
            SELECT
                g.game_id,
                g.away_team_abbr as defense_team,
                SUM(CASE WHEN pb.pass_yds > 0 THEN pb.pass_yds ELSE 0 END) as pass_yds_allowed,
                SUM(CASE WHEN pb.rush_yds > 0 THEN pb.rush_yds ELSE 0 END) as rush_yds_allowed
            FROM games g
            JOIN player_box_score pb ON g.season = pb.season AND g.week = pb.week AND pb.team = g.home_team_abbr
            WHERE g.season = ?
            {week_filter}
            GROUP BY g.game_id, g.away_team_abbr
        )
        GROUP BY defense_team
        """

        df = query(sql_defense, (season, season))

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

        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        with st.expander("📋 View Defensive Stats Table"):
            display_df = df[['team_abbr', 'avg_pass_yds_allowed', 'avg_rush_yds_allowed', 'total_yds_allowed']].copy()
            display_df.columns = ['Team', 'Pass Yds/G', 'Rush Yds/G', 'Total Yds/G']
            display_df = display_df.round(1).sort_values('Total Yds/G', ascending=True)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error generating chart: {e}")


def render_air_yac_matchup_chart(season: Optional[int], week: Optional[int], upcoming_games=None):
    """Chart: Air Yards vs YAC Matchup Analysis for Upcoming Games"""
    st.subheader("🎯 Air Yards vs YAC Matchup Analysis")
    st.markdown("""
    **Goal:** Identify favorable/unfavorable passing matchups based on offensive air yards tendencies vs defensive YAC vulnerabilities.
    - **Top-right quadrant:** 🎯 Explosive potential (vertical offense vs YAC-leaky defense)
    - **Top-left:** 🔄 Mismatch (underneath offense vs air-prone defense)
    - **Bottom-right:** 🛡️ Tough matchup (vertical offense vs tight coverage)
    - **Bottom-left:** 🚧 Limited upside (underneath offense vs YAC-stingy defense)
    """)

    if not season:
        st.warning("No season data available.")
        return

    try:
        # Calculate Air/YAC stats for all teams
        air_yac_df = calculate_air_yac_matchup_stats(season, week)

        if air_yac_df.empty:
            st.info("No Air Yards/YAC data available for selected filters")
            return

        # If upcoming games provided, create matchup-specific data
        if upcoming_games is not None and not upcoming_games.empty:
            matchup_data = []

            for _, game in upcoming_games.iterrows():
                home_team = game['home_team']
                away_team = game['away_team']

                # Get home team offense vs away team defense
                home_stats = air_yac_df[air_yac_df['team'] == home_team]
                away_defense_stats = air_yac_df[air_yac_df['team'] == away_team]

                if not home_stats.empty and not away_defense_stats.empty:
                    home_off_air_share = home_stats.iloc[0]['offense_air_share']
                    away_def_yac_share = away_defense_stats.iloc[0]['defense_yac_share']
                    away_def_air_share = away_defense_stats.iloc[0]['defense_air_share']
                    home_off_yac_share = home_stats.iloc[0]['offense_yac_share']

                    storyline, description = generate_air_yac_storyline(
                        home_off_air_share, away_def_yac_share, away_def_air_share, home_off_yac_share
                    )

                    matchup_data.append({
                        'team': home_team,
                        'opponent': away_team,
                        'location': 'Home',
                        'air_share': home_off_air_share,
                        'opp_yac_share': away_def_yac_share,
                        'storyline': storyline,
                        'description': description,
                        'projected_rec_yds': home_stats.iloc[0]['offense_passing_yards']
                    })

                # Get away team offense vs home team defense
                away_stats = air_yac_df[air_yac_df['team'] == away_team]
                home_defense_stats = air_yac_df[air_yac_df['team'] == home_team]

                if not away_stats.empty and not home_defense_stats.empty:
                    away_off_air_share = away_stats.iloc[0]['offense_air_share']
                    home_def_yac_share = home_defense_stats.iloc[0]['defense_yac_share']
                    home_def_air_share = home_defense_stats.iloc[0]['defense_air_share']
                    away_off_yac_share = away_stats.iloc[0]['offense_yac_share']

                    storyline, description = generate_air_yac_storyline(
                        away_off_air_share, home_def_yac_share, home_def_air_share, away_off_yac_share
                    )

                    matchup_data.append({
                        'team': away_team,
                        'opponent': home_team,
                        'location': 'Away',
                        'air_share': away_off_air_share,
                        'opp_yac_share': home_def_yac_share,
                        'storyline': storyline,
                        'description': description,
                        'projected_rec_yds': away_stats.iloc[0]['offense_passing_yards']
                    })

            matchup_df = pd.DataFrame(matchup_data)
        else:
            # Show all teams (not matchup-specific)
            matchup_df = air_yac_df.copy()
            matchup_df['air_share'] = matchup_df['offense_air_share']
            matchup_df['opp_yac_share'] = matchup_df['defense_yac_share']
            matchup_df['projected_rec_yds'] = matchup_df['offense_passing_yards']
            matchup_df = matchup_df.rename(columns={'team': 'team'})

        if matchup_df.empty:
            st.info("No matchup data available")
            return

        # Create scatter plot
        fig = go.Figure()

        # Add invisible markers for hover functionality
        fig.add_trace(go.Scatter(
            x=matchup_df['air_share'],
            y=matchup_df['opp_yac_share'],
            mode='markers',
            marker=dict(
                size=1,
                color=matchup_df['projected_rec_yds'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Proj Pass Yds"),
                opacity=0
            ),
            text=matchup_df['team'] + ('' if 'opponent' not in matchup_df.columns else ' vs ' + matchup_df['opponent']),
            customdata=matchup_df[['projected_rec_yds'] + (['storyline'] if 'storyline' in matchup_df.columns else [])].values if 'storyline' in matchup_df.columns else matchup_df[['projected_rec_yds']].values,
            hovertemplate='<b>%{text}</b><br>' +
                         'Offense Air Share: %{x:.1f}%<br>' +
                         'Defense YAC Share Allowed: %{y:.1f}%<br>' +
                         'Projected Receiving Yds/G: %{customdata[0]:.1f}<br>' +
                         ('%{customdata[1]}<br>' if 'storyline' in matchup_df.columns else '') +
                         '<extra></extra>',
            showlegend=False
        ))

        # Add league average lines
        avg_air_share = matchup_df['air_share'].mean()
        avg_yac_share = matchup_df['opp_yac_share'].mean()

        fig.add_hline(y=avg_yac_share, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="Avg Defense YAC%", annotation_position="right")
        fig.add_vline(x=avg_air_share, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="Avg Offense Air%", annotation_position="top")

        # Add quadrant labels
        fig.add_annotation(x=matchup_df['air_share'].max() * 0.95, y=matchup_df['opp_yac_share'].max() * 0.95,
                          text="🎯 Explosive Potential", showarrow=False, font=dict(size=12, color="green"))
        fig.add_annotation(x=matchup_df['air_share'].min() * 1.05, y=matchup_df['opp_yac_share'].max() * 0.95,
                          text="🔄 Favorable Mismatch", showarrow=False, font=dict(size=12, color="blue"))
        fig.add_annotation(x=matchup_df['air_share'].max() * 0.95, y=matchup_df['opp_yac_share'].min() * 1.05,
                          text="🛡️ Tough Matchup", showarrow=False, font=dict(size=12, color="orange"))
        fig.add_annotation(x=matchup_df['air_share'].min() * 1.05, y=matchup_df['opp_yac_share'].min() * 1.05,
                          text="🚧 Limited Upside", showarrow=False, font=dict(size=12, color="red"))

        # Calculate logo sizes based on data range
        x_range = matchup_df['air_share'].max() - matchup_df['air_share'].min()
        y_range = matchup_df['opp_yac_share'].max() - matchup_df['opp_yac_share'].min()
        logo_size_x = max(x_range * 0.06, 2)
        logo_size_y = max(y_range * 0.06, 2)

        # Build layout with team logo images
        layout_images = []
        for idx, row in matchup_df.iterrows():
            layout_images.append(dict(
                source=get_team_logo_url(row['team']),
                xref="x",
                yref="y",
                x=row['air_share'],
                y=row['opp_yac_share'],
                sizex=logo_size_x,
                sizey=logo_size_y,
                xanchor="center",
                yanchor="middle",
                layer="above",
                opacity=0.9
            ))

        fig.update_layout(
            title=f"Air Yards vs YAC Matchup Analysis ({season} Season{f', Week {week}' if week else ''})",
            xaxis_title="Offense Air Share (%)",
            yaxis_title="Defense YAC Share Allowed (%)",
            height=600,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            images=layout_images
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show matchup storylines table
        if 'storyline' in matchup_df.columns:
            with st.expander("📋 View Matchup Storylines"):
                display_df = matchup_df[['team', 'opponent', 'location', 'air_share', 'opp_yac_share',
                                        'storyline', 'description', 'projected_rec_yds']].copy()
                display_df.columns = ['Team', 'Opponent', 'Location', 'Off Air %', 'Def YAC %',
                                     'Storyline', 'Description', 'Proj Pass Yds']
                display_df = display_df.round({'Off Air %': 1, 'Def YAC %': 1, 'Proj Pass Yds': 1})
                display_df = display_df.sort_values('Proj Pass Yds', ascending=False)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Highlight top storylines
                st.markdown("### 🔥 Key Matchups This Week")
                explosive = display_df[display_df['Storyline'].str.contains('Vertical vs YAC')]
                if not explosive.empty:
                    st.success(f"**Explosive Potential:** {len(explosive)} matchup(s) with vertical offense vs YAC-leaky defense")
                    for _, row in explosive.head(3).iterrows():
                        st.markdown(f"- **{row['Team']} vs {row['Opponent']}**: {row['Description']}")

        else:
            # Show general team stats
            with st.expander("📋 View Air/YAC Stats Table"):
                display_df = air_yac_df[['team', 'offense_air_share', 'offense_yac_share',
                                         'defense_air_share', 'defense_yac_share']].copy()
                display_df.columns = ['Team', 'Off Air %', 'Off YAC %', 'Def Air %', 'Def YAC %']
                display_df = display_df.round(1).sort_values('Off Air %', ascending=False)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error generating Air/YAC matchup chart: {e}")


def render_qb_pressure_matchup_chart(season: Optional[int], week: Optional[int], upcoming_games=None):
    """Chart: QB Pressure/Sack Matchup Analysis for Upcoming Games"""
    st.subheader("⚡ QB Pressure Matchup Analysis")
    st.markdown("""
    **Goal:** Identify QB protection concerns based on QB pressure vulnerability vs defensive pass rush strength.
    - **Top-right quadrant:** 🚨 DANGER ZONE (vulnerable QB vs aggressive pass rush)
    - **Top-left:** ⚔️ Tough test (good protection vs strong rush)
    - **Bottom-right:** 🎯 Exploitable (vulnerable QB vs weak rush)
    - **Bottom-left:** ✅ Clean pocket expected
    """)

    if not season:
        st.warning("No season data available.")
        return

    try:
        # Calculate QB and defensive pressure stats
        qb_stats = calculate_qb_pressure_stats(season, week)
        def_stats = calculate_defense_pressure_stats(season, week)

        if qb_stats.empty or def_stats.empty:
            st.info("No QB pressure data available for selected filters. PFR advanced stats may be limited.")
            return

        # If upcoming games provided, create matchup-specific data
        if upcoming_games is not None and not upcoming_games.empty:
            matchup_data = []

            # Get starting QBs from schedules table for upcoming games
            for _, game in upcoming_games.iterrows():
                home_team = game['home_team']
                away_team = game['away_team']

                # Get home team QB stats
                home_qbs = qb_stats[qb_stats['team'] == home_team]
                if not home_qbs.empty:
                    # Take the QB with most games played
                    home_qb = home_qbs.sort_values('games', ascending=False).iloc[0]

                    # Get away team defense stats
                    away_def = def_stats[def_stats['team'] == away_team]

                    if not away_def.empty:
                        qb_pressure_rate = home_qb['pressure_rate']
                        def_pressure_rate = away_def.iloc[0]['pressures_per_game']
                        sacks_per_pressure = home_qb['sacks_per_pressure']

                        storyline, description = generate_qb_pressure_storyline(
                            qb_pressure_rate, def_pressure_rate, sacks_per_pressure
                        )

                        matchup_data.append({
                            'qb': home_qb['player'],
                            'team': home_team,
                            'opponent': away_team,
                            'location': 'Home',
                            'qb_pressure_rate': qb_pressure_rate,
                            'def_pressure_rate': def_pressure_rate,
                            'sacks_per_pressure': sacks_per_pressure,
                            'storyline': storyline,
                            'description': description,
                            'projected_pass_yds': 250.0  # Placeholder - could calculate actual projection
                        })

                # Get away team QB stats
                away_qbs = qb_stats[qb_stats['team'] == away_team]
                if not away_qbs.empty:
                    # Take the QB with most games played
                    away_qb = away_qbs.sort_values('games', ascending=False).iloc[0]

                    # Get home team defense stats
                    home_def = def_stats[def_stats['team'] == home_team]

                    if not home_def.empty:
                        qb_pressure_rate = away_qb['pressure_rate']
                        def_pressure_rate = home_def.iloc[0]['pressures_per_game']
                        sacks_per_pressure = away_qb['sacks_per_pressure']

                        storyline, description = generate_qb_pressure_storyline(
                            qb_pressure_rate, def_pressure_rate, sacks_per_pressure
                        )

                        matchup_data.append({
                            'qb': away_qb['player'],
                            'team': away_team,
                            'opponent': home_team,
                            'location': 'Away',
                            'qb_pressure_rate': qb_pressure_rate,
                            'def_pressure_rate': def_pressure_rate,
                            'sacks_per_pressure': sacks_per_pressure,
                            'storyline': storyline,
                            'description': description,
                            'projected_pass_yds': 250.0  # Placeholder
                        })

            matchup_df = pd.DataFrame(matchup_data)
        else:
            # Show all QBs (not matchup-specific)
            matchup_df = qb_stats.copy()
            matchup_df['qb'] = matchup_df['player']
            matchup_df['qb_pressure_rate'] = matchup_df['pressure_rate']
            # For general view, use league average defensive pressure
            avg_def_pressure = def_stats['pressures_per_game'].mean()
            matchup_df['def_pressure_rate'] = avg_def_pressure
            matchup_df['projected_pass_yds'] = 250.0

        if matchup_df.empty:
            st.info("No QB matchup data available")
            return

        # Create scatter plot
        fig = go.Figure()

        # Add invisible markers for hover functionality
        hover_text = matchup_df['qb'] if 'opponent' not in matchup_df.columns else matchup_df['qb'] + ' (' + matchup_df['team'] + ') vs ' + matchup_df['opponent']

        fig.add_trace(go.Scatter(
            x=matchup_df['qb_pressure_rate'],
            y=matchup_df['def_pressure_rate'],
            mode='markers',
            marker=dict(
                size=matchup_df['sacks_per_pressure'] if 'sacks_per_pressure' in matchup_df.columns else 10,  # Bubble size = sack vulnerability
                sizemode='diameter',
                sizeref=2,
                sizemin=4,
                color=matchup_df['projected_pass_yds'] if 'projected_pass_yds' in matchup_df.columns else matchup_df['pressure_rate'],
                colorscale='RdYlGn_r',  # Reversed: red=high pressure, green=low
                showscale=True,
                colorbar=dict(title="Proj Pass Yds"),
                opacity=0.6,
                line=dict(width=2, color='white')
            ),
            text=hover_text,
            customdata=matchup_df[['qb_pressure_rate', 'def_pressure_rate', 'sacks_per_pressure'] +
                                  (['storyline'] if 'storyline' in matchup_df.columns else [])].values if 'storyline' in matchup_df.columns else matchup_df[['qb_pressure_rate', 'def_pressure_rate', 'sacks_per_pressure']].values,
            hovertemplate='<b>%{text}</b><br>' +
                         'QB Pressure Rate: %{x:.1f}%<br>' +
                         'Opp Defense Pressures/Gm: %{y:.1f}<br>' +
                         'Sacks per Pressure: %{customdata[2]:.1f}%<br>' +
                         ('%{customdata[3]}<br>' if 'storyline' in matchup_df.columns else '') +
                         '<extra></extra>',
            showlegend=False
        ))

        # Add league average lines
        avg_qb_pressure = matchup_df['qb_pressure_rate'].mean()
        avg_def_pressure = matchup_df['def_pressure_rate'].mean()

        fig.add_hline(y=avg_def_pressure, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="Avg Def Pressure/Gm", annotation_position="right")
        fig.add_vline(x=avg_qb_pressure, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="Avg QB Pressure%", annotation_position="top")

        # Add quadrant labels
        fig.add_annotation(x=matchup_df['qb_pressure_rate'].max() * 0.95, y=matchup_df['def_pressure_rate'].max() * 0.95,
                          text="🚨 DANGER ZONE", showarrow=False, font=dict(size=14, color="red"))
        fig.add_annotation(x=matchup_df['qb_pressure_rate'].min() * 1.05, y=matchup_df['def_pressure_rate'].max() * 0.95,
                          text="⚔️ Tough Test", showarrow=False, font=dict(size=14, color="orange"))
        fig.add_annotation(x=matchup_df['qb_pressure_rate'].max() * 0.95, y=matchup_df['def_pressure_rate'].min() * 1.05,
                          text="🎯 Exploitable", showarrow=False, font=dict(size=14, color="blue"))
        fig.add_annotation(x=matchup_df['qb_pressure_rate'].min() * 1.05, y=matchup_df['def_pressure_rate'].min() * 1.05,
                          text="✅ Clean Pocket", showarrow=False, font=dict(size=14, color="green"))

        # Calculate logo sizes based on data range
        x_range = matchup_df['qb_pressure_rate'].max() - matchup_df['qb_pressure_rate'].min()
        y_range = matchup_df['def_pressure_rate'].max() - matchup_df['def_pressure_rate'].min()
        logo_size_x = max(x_range * 0.06, 1.5)  # At least 1.5% wide
        logo_size_y = max(y_range * 0.06, 0.5)  # At least 0.5 pressures tall

        # Build layout with team logo images
        layout_images = []
        for idx, row in matchup_df.iterrows():
            layout_images.append(dict(
                source=get_team_logo_url(row['team']),
                xref="x",
                yref="y",
                x=row['qb_pressure_rate'],
                y=row['def_pressure_rate'],
                sizex=logo_size_x,
                sizey=logo_size_y,
                xanchor="center",
                yanchor="middle",
                layer="above",
                opacity=0.9
            ))

        fig.update_layout(
            title=f"QB Pressure Matchup Analysis ({season} Season{f', Week {week}' if week else ''})",
            xaxis_title="QB Pressure Rate (%)",
            yaxis_title="Opposing Defense Pressures per Game",
            height=600,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            images=layout_images
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show matchup storylines table
        if 'storyline' in matchup_df.columns:
            with st.expander("📋 View QB Pressure Matchup Details"):
                display_df = matchup_df[['qb', 'team', 'opponent', 'location', 'qb_pressure_rate',
                                        'def_pressure_rate', 'sacks_per_pressure', 'storyline', 'description']].copy()
                display_df.columns = ['QB', 'Team', 'Opponent', 'Location', 'QB Press %',
                                     'Def Press/Gm', 'Sacks/Press %', 'Storyline', 'Description']
                display_df = display_df.round({'QB Press %': 1, 'Def Press/Gm': 1, 'Sacks/Press %': 1})
                display_df = display_df.sort_values('QB Press %', ascending=False)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Highlight danger zone matchups
                st.markdown("### 🚨 Watch List: Pressure Concerns")
                danger = display_df[display_df['Storyline'].str.contains('DANGER')]
                if not danger.empty:
                    st.error(f"**{len(danger)} Danger Zone Matchup(s):** QB struggles with pressure facing aggressive defense")
                    for _, row in danger.head(3).iterrows():
                        st.markdown(f"- **{row['QB']} ({row['Team']}) vs {row['Opponent']}**: {row['Description']}")
                else:
                    st.success("No extreme danger zone matchups this week")

                # Highlight clean pocket opportunities
                clean = display_df[display_df['Storyline'].str.contains('Clean Pocket')]
                if not clean.empty:
                    st.success(f"**{len(clean)} Clean Pocket Opportunity(ies):** QB should have time to throw")
                    for _, row in clean.head(3).iterrows():
                        st.markdown(f"- **{row['QB']} ({row['Team']}) vs {row['Opponent']}**: {row['Description']}")

        else:
            # Show general QB stats
            with st.expander("📋 View QB Pressure Stats"):
                display_df = qb_stats[['player', 'team', 'pressure_rate', 'sacks_per_pressure',
                                      'pressures_per_game', 'sacks_per_game']].copy()
                display_df.columns = ['QB', 'Team', 'Pressure %', 'Sacks/Press %', 'Pressures/Gm', 'Sacks/Gm']
                display_df = display_df.round(1).sort_values('Pressure %', ascending=False)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error generating QB pressure matchup chart: {e}")
        import traceback
        st.error(traceback.format_exc())


def render_rushing_td_matchup_chart(season: Optional[int], week: Optional[int], upcoming_games=None):
    """Chart: Rushing TD Efficiency Matchup Analysis"""
    st.subheader("🏈 Rushing TD Efficiency Matchup Analysis")
    st.markdown("""
    **Goal:** Match team rushing TD scoring rates vs defensive TD vulnerability (API data: `rushing_tds` per game).
    - **Top-right quadrant:** 🎯 EXPLOIT (high TD rate vs TD-prone defense)
    - **Top-left:** 💎 VALUE (low TD rate vs TD-prone defense - breakout potential)
    - **Bottom-right:** 🛡️ TOUGH (high TD rate vs stingy defense)
    - **Bottom-left:** ⚖️ NEUTRAL (balanced matchup)
    """)

    if not season:
        st.warning("No season data available.")
        return

    try:
        # Calculate rushing TD matchup stats
        td_stats = calculate_rushing_td_matchup_stats(season, week)

        if td_stats.empty:
            st.info("No rushing TD data available for selected filters.")
            return

        # If upcoming games provided, create matchup-specific data
        if upcoming_games is not None and not upcoming_games.empty:
            matchup_data = []

            for _, game in upcoming_games.iterrows():
                home_team = game['home_team']
                away_team = game['away_team']

                # Home team offense vs Away team defense
                home_stats = td_stats[td_stats['team'] == home_team]
                away_def = td_stats[td_stats['team'] == away_team]

                if not home_stats.empty and not away_def.empty:
                    offense_rush_tds = home_stats.iloc[0]['offense_rush_tds']
                    defense_rush_tds_allowed = away_def.iloc[0]['defense_rush_tds_allowed']

                    storyline, description = generate_rushing_td_storyline(
                        offense_rush_tds, defense_rush_tds_allowed
                    )

                    matchup_data.append({
                        'team': home_team,
                        'opponent': away_team,
                        'location': 'Home',
                        'offense_rush_tds': offense_rush_tds,
                        'defense_rush_tds_allowed': defense_rush_tds_allowed,
                        'storyline': storyline,
                        'description': description
                    })

                # Away team offense vs Home team defense
                away_stats = td_stats[td_stats['team'] == away_team]
                home_def = td_stats[td_stats['team'] == home_team]

                if not away_stats.empty and not home_def.empty:
                    offense_rush_tds = away_stats.iloc[0]['offense_rush_tds']
                    defense_rush_tds_allowed = home_def.iloc[0]['defense_rush_tds_allowed']

                    storyline, description = generate_rushing_td_storyline(
                        offense_rush_tds, defense_rush_tds_allowed
                    )

                    matchup_data.append({
                        'team': away_team,
                        'opponent': home_team,
                        'location': 'Away',
                        'offense_rush_tds': offense_rush_tds,
                        'defense_rush_tds_allowed': defense_rush_tds_allowed,
                        'storyline': storyline,
                        'description': description
                    })

            matchup_df = pd.DataFrame(matchup_data)
        else:
            # Show all teams (not matchup-specific)
            matchup_df = td_stats.copy()
            # Use league average for defensive TD allowed
            avg_def_tds = td_stats['defense_rush_tds_allowed'].mean()
            matchup_df['opponent'] = 'Avg'
            matchup_df['location'] = 'N/A'

        if matchup_df.empty:
            st.info("No rushing TD matchup data available")
            return

        # Create scatter plot
        fig = go.Figure()

        # Add invisible markers for hover functionality
        hover_text = matchup_df['team'] if 'opponent' not in matchup_df.columns or matchup_df['opponent'].iloc[0] == 'Avg' else matchup_df['team'] + ' vs ' + matchup_df['opponent']

        fig.add_trace(go.Scatter(
            x=matchup_df['offense_rush_tds'],
            y=matchup_df['defense_rush_tds_allowed'],
            mode='markers',
            marker=dict(
                size=15,
                color=matchup_df['offense_rush_tds'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Off Rush TDs/Gm"),
                opacity=0.6,
                line=dict(width=2, color='white')
            ),
            text=hover_text,
            customdata=matchup_df[['offense_rush_tds', 'defense_rush_tds_allowed'] +
                                  (['storyline'] if 'storyline' in matchup_df.columns else [])].values if 'storyline' in matchup_df.columns else matchup_df[['offense_rush_tds', 'defense_rush_tds_allowed']].values,
            hovertemplate='<b>%{text}</b><br>' +
                         'Offense Rush TDs/Gm: %{x:.2f}<br>' +
                         'Def Rush TDs Allowed/Gm: %{y:.2f}<br>' +
                         ('%{customdata[2]}<br>' if 'storyline' in matchup_df.columns else '') +
                         '<extra></extra>',
            showlegend=False
        ))

        # Add league average lines
        avg_off_tds = matchup_df['offense_rush_tds'].mean()
        avg_def_tds = matchup_df['defense_rush_tds_allowed'].mean()

        fig.add_hline(y=avg_def_tds, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="Avg Def TDs Allowed", annotation_position="right")
        fig.add_vline(x=avg_off_tds, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="Avg Off TDs", annotation_position="top")

        # Add quadrant labels
        fig.add_annotation(x=matchup_df['offense_rush_tds'].max() * 0.95, y=matchup_df['defense_rush_tds_allowed'].max() * 0.95,
                          text="🎯 EXPLOIT", showarrow=False, font=dict(size=14, color="green"))
        fig.add_annotation(x=matchup_df['offense_rush_tds'].min() * 1.05, y=matchup_df['defense_rush_tds_allowed'].max() * 0.95,
                          text="💎 BREAKOUT", showarrow=False, font=dict(size=14, color="blue"))
        fig.add_annotation(x=matchup_df['offense_rush_tds'].max() * 0.95, y=matchup_df['defense_rush_tds_allowed'].min() * 1.05,
                          text="🛡️ TOUGH", showarrow=False, font=dict(size=14, color="orange"))
        fig.add_annotation(x=matchup_df['offense_rush_tds'].min() * 1.05, y=matchup_df['defense_rush_tds_allowed'].min() * 1.05,
                          text="⚖️ NEUTRAL", showarrow=False, font=dict(size=14, color="gray"))

        # Calculate logo sizes based on data range
        x_range = matchup_df['offense_rush_tds'].max() - matchup_df['offense_rush_tds'].min()
        y_range = matchup_df['defense_rush_tds_allowed'].max() - matchup_df['defense_rush_tds_allowed'].min()
        logo_size_x = max(x_range * 0.08, 0.1)
        logo_size_y = max(y_range * 0.08, 0.1)

        # Build layout with team logo images
        layout_images = []
        for idx, row in matchup_df.iterrows():
            layout_images.append(dict(
                source=get_team_logo_url(row['team']),
                xref="x",
                yref="y",
                x=row['offense_rush_tds'],
                y=row['defense_rush_tds_allowed'],
                sizex=logo_size_x,
                sizey=logo_size_y,
                xanchor="center",
                yanchor="middle",
                layer="above",
                opacity=0.9
            ))

        fig.update_layout(
            title=f"Rushing TD Efficiency Matchup ({season} Season{f', Week {week}' if week else ''})",
            xaxis_title="Offense Rushing TDs per Game",
            yaxis_title="Defense Rushing TDs Allowed per Game",
            height=600,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            images=layout_images
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show matchup storylines table
        if 'storyline' in matchup_df.columns:
            with st.expander("📋 View Rushing TD Matchup Details"):
                display_df = matchup_df[['team', 'opponent', 'location', 'offense_rush_tds',
                                        'defense_rush_tds_allowed', 'storyline', 'description']].copy()
                display_df.columns = ['Team', 'Opponent', 'Location', 'Off Rush TDs/Gm',
                                     'Def TDs Allowed/Gm', 'Storyline', 'Description']
                display_df = display_df.round({'Off Rush TDs/Gm': 2, 'Def TDs Allowed/Gm': 2})
                display_df = display_df.sort_values('Off Rush TDs/Gm', ascending=False)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Highlight EXPLOIT matchups
                exploit = display_df[display_df['Storyline'].str.contains('EXPLOIT|SMASH')]
                if not exploit.empty:
                    st.success(f"**{len(exploit)} Favorable TD Matchup(s):** High RB touchdown upside")
                    for _, row in exploit.head(3).iterrows():
                        st.markdown(f"- **{row['Team']} vs {row['Opponent']}**: {row['Description']}")
                else:
                    st.info("No standout TD exploit matchups this week")

                # Highlight TOUGH matchups
                tough = display_df[display_df['Storyline'].str.contains('TOUGH|YARDAGE')]
                if not tough.empty:
                    st.warning(f"**{len(tough)} Tough TD Matchup(s):** Limited scoring potential")
                    for _, row in tough.head(3).iterrows():
                        st.markdown(f"- **{row['Team']} vs {row['Opponent']}**: {row['Description']}")

    except Exception as e:
        st.error(f"Error generating rushing TD matchup chart: {e}")
        import traceback
        st.error(traceback.format_exc())


def render_player_rush_td_matchup_chart(season: Optional[int], week: Optional[int], upcoming_games=None):
    """Chart: Player Rush TD vs Defense Analysis"""
    st.subheader("🏃‍♂️ Player Rush TD vs Defense Matchup")
    st.markdown("""
    **Goal:** Identify individual players with high rushing TD probability based on their scoring rate vs defensive vulnerability.
    - **Top-right quadrant:** 🎯 TD SMASH (elite scorer vs generous defense)
    - **Top-left:** 💎 BREAKOUT (emerging scorer vs generous defense)
    - **Bottom-right:** ⚔️ CHALLENGE (elite scorer vs stingy defense)
    - **Bottom-left:** ⚠️ TOUGH (limited scorer vs stingy defense)
    """)

    if not season:
        st.warning("No season data available.")
        return

    try:
        # Calculate player rush TD stats
        player_stats = calculate_player_rush_td_vs_defense_stats(season, week)

        if player_stats.empty:
            st.info("No player rushing TD data available for selected filters.")
            return

        # If upcoming games provided, create matchup-specific data
        if upcoming_games is not None and not upcoming_games.empty:
            matchup_data = []

            # Get team to opponent mapping
            team_opponent = {}
            for _, game in upcoming_games.iterrows():
                team_opponent[game['home_team']] = game['away_team']
                team_opponent[game['away_team']] = game['home_team']

            # Filter players whose teams are playing
            for _, player_row in player_stats.iterrows():
                team = player_row['team']
                if team in team_opponent:
                    opponent = team_opponent[team]

                    # Get opponent's defensive stats
                    opponent_def_stats = player_stats[player_stats['team'] == opponent]
                    if not opponent_def_stats.empty:
                        opp_def_rush_tds = opponent_def_stats.iloc[0]['defense_rush_tds_allowed']
                    else:
                        # Use league average if opponent stats not available
                        opp_def_rush_tds = player_stats['defense_rush_tds_allowed'].mean()

                    storyline, description = generate_player_rush_td_storyline(
                        player_row['player_rush_tds_per_game'],
                        opp_def_rush_tds,
                        player_row['player'],
                        player_row['position']
                    )

                    matchup_data.append({
                        'player': player_row['player'],
                        'position': player_row['position'],
                        'team': team,
                        'opponent': opponent,
                        'player_rush_tds_per_game': player_row['player_rush_tds_per_game'],
                        'defense_rush_tds_allowed': opp_def_rush_tds,
                        'total_tds': player_row['player_rush_tds'],
                        'games': player_row['player_games'],
                        'storyline': storyline,
                        'description': description
                    })

            matchup_df = pd.DataFrame(matchup_data)
        else:
            # Show all players (use their team's defensive stats as baseline)
            matchup_df = player_stats.copy()
            matchup_df['opponent'] = 'Avg'

        if matchup_df.empty:
            st.info("No player TD matchup data available")
            return

        # Create scatter plot
        fig = go.Figure()

        # Add scatter points with player names
        hover_text = matchup_df['player'] + ' (' + matchup_df['position'] + ')' + '<br>' + matchup_df['team']
        if 'opponent' in matchup_df.columns and matchup_df['opponent'].iloc[0] != 'Avg':
            hover_text = hover_text + ' vs ' + matchup_df['opponent']

        fig.add_trace(go.Scatter(
            x=matchup_df['player_rush_tds_per_game'],
            y=matchup_df['defense_rush_tds_allowed'],
            mode='markers+text',
            marker=dict(
                size=12,
                color=matchup_df['player_rush_tds_per_game'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Player<br>TDs/Gm"),
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=matchup_df['player'].str.split().str[-1],  # Last name only
            textposition='top center',
            textfont=dict(size=8, color='black'),
            customdata=matchup_df[['player', 'position', 'team', 'player_rush_tds_per_game', 'defense_rush_tds_allowed'] +
                                  (['storyline'] if 'storyline' in matchup_df.columns else [])].values,
            hovertemplate='<b>%{customdata[0]} (%{customdata[1]})</b><br>' +
                         'Team: %{customdata[2]}<br>' +
                         'Player Rush TDs/Gm: %{x:.2f}<br>' +
                         'Opp Def Rush TDs Allowed/Gm: %{y:.2f}<br>' +
                         ('%{customdata[5]}<br>' if 'storyline' in matchup_df.columns else '') +
                         '<extra></extra>',
            showlegend=False
        ))

        # Add league average lines
        avg_player_tds = matchup_df['player_rush_tds_per_game'].mean()
        avg_def_tds = matchup_df['defense_rush_tds_allowed'].mean()

        fig.add_hline(y=avg_def_tds, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="Avg Def TDs Allowed", annotation_position="right")
        fig.add_vline(x=avg_player_tds, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="Avg Player TDs", annotation_position="top")

        # Add quadrant labels
        fig.add_annotation(x=matchup_df['player_rush_tds_per_game'].max() * 0.9,
                          y=matchup_df['defense_rush_tds_allowed'].max() * 0.95,
                          text="🎯 TD SMASH", showarrow=False, font=dict(size=12, color="green"))
        fig.add_annotation(x=matchup_df['player_rush_tds_per_game'].min() * 1.1,
                          y=matchup_df['defense_rush_tds_allowed'].max() * 0.95,
                          text="💎 BREAKOUT", showarrow=False, font=dict(size=12, color="blue"))
        fig.add_annotation(x=matchup_df['player_rush_tds_per_game'].max() * 0.9,
                          y=matchup_df['defense_rush_tds_allowed'].min() * 1.05,
                          text="⚔️ CHALLENGE", showarrow=False, font=dict(size=12, color="orange"))
        fig.add_annotation(x=matchup_df['player_rush_tds_per_game'].min() * 1.1,
                          y=matchup_df['defense_rush_tds_allowed'].min() * 1.05,
                          text="⚠️ TOUGH", showarrow=False, font=dict(size=12, color="red"))

        fig.update_layout(
            title=f"Player Rush TD vs Defense ({season} Season{f', Week {week}' if week else ''})",
            xaxis_title="Player Rushing TDs per Game",
            yaxis_title="Opponent Defense Rush TDs Allowed per Game",
            height=700,
            hovermode='closest',
            plot_bgcolor='rgba(240,240,240,0.3)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show matchup storylines table
        if 'storyline' in matchup_df.columns:
            with st.expander("📋 View Player TD Matchup Details"):
                display_df = matchup_df[['player', 'position', 'team', 'opponent',
                                        'player_rush_tds_per_game', 'defense_rush_tds_allowed',
                                        'total_tds', 'games', 'storyline', 'description']].copy()
                display_df.columns = ['Player', 'Pos', 'Team', 'Opponent',
                                     'Player TDs/Gm', 'Opp Def TDs Allowed/Gm',
                                     'Total TDs', 'Games', 'Storyline', 'Description']
                display_df = display_df.round({'Player TDs/Gm': 2, 'Opp Def TDs Allowed/Gm': 2})
                display_df = display_df.sort_values('Player TDs/Gm', ascending=False)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Highlight SMASH matchups
                smash = display_df[display_df['Storyline'].str.contains('SMASH|GREAT|FAVORABLE', na=False)]
                if not smash.empty:
                    st.success(f"**{len(smash)} Favorable TD Matchup(s):** High TD probability")
                    for _, row in smash.head(5).iterrows():
                        st.markdown(f"- **{row['Player']} ({row['Pos']})**: {row['Description']}")
                else:
                    st.info("No standout TD smash matchups this week")

                # Highlight AVOID matchups
                avoid = display_df[display_df['Storyline'].str.contains('AVOID|TOUGH', na=False)]
                if not avoid.empty:
                    st.warning(f"**{len(avoid)} Tough TD Matchup(s):** Limited TD upside")
                    for _, row in avoid.head(3).iterrows():
                        st.markdown(f"- **{row['Player']} ({row['Pos']})**: {row['Description']}")

    except Exception as e:
        st.error(f"Error generating player rush TD matchup chart: {e}")
        import traceback
        st.error(traceback.format_exc())


def render_rushing_yards_matchup_chart(season: Optional[int], week: Optional[int], upcoming_games=None):
    """Chart: RB Rushing Yards Matchup Analysis"""
    st.subheader("🏃 RB Rushing Yards Matchup Analysis")
    st.markdown("""
    **Goal:** Match RB rushing yards vs defensive rushing yards allowed (API data: `rushing_yards` per game)
    - **Top-right quadrant:** 🚀 CEILING EXPLOSION (elite ground game vs run funnel)
    - **Top-left:** 📈 VOLUME SPIKE (pass-heavy vs run funnel)
    - **Bottom-right:** 💪 GRIND IT OUT (elite ground game vs elite run defense)
    - **Bottom-left:** ⚠️ LOW VOLUME (pass-heavy vs elite run defense)
    """)

    if not season:
        st.warning("No season data available.")
        return

    try:
        # Calculate stats
        stats = calculate_rushing_yards_matchup_stats(season, week)

        if stats.empty:
            st.info("No data available for selected filters.")
            return

        # If upcoming games provided, create matchup-specific data
        if upcoming_games is not None and not upcoming_games.empty:
            matchup_data = []

            for _, game in upcoming_games.iterrows():
                home_team = game['home_team']
                away_team = game['away_team']

                # Home team offense vs Away team defense
                home_stats = stats[stats['team'] == home_team]
                away_def = stats[stats['team'] == away_team]

                if not home_stats.empty and not away_def.empty:
                    offense_val = home_stats.iloc[0]['offense_rush_yards']
                    defense_val = away_def.iloc[0]['defense_rush_yards_allowed']

                    storyline, description = generate_rushing_yards_storyline(
                        offense_val, defense_val
                    )

                    matchup_data.append({
                        'team': home_team,
                        'opponent': away_team,
                        'location': 'Home',
                        'offense_rush_yards': offense_val,
                        'defense_rush_yards_allowed': defense_val,
                        'storyline': storyline,
                        'description': description
                    })

                # Away team offense vs Home team defense
                away_stats = stats[stats['team'] == away_team]
                home_def = stats[stats['team'] == home_team]

                if not away_stats.empty and not home_def.empty:
                    offense_val = away_stats.iloc[0]['offense_rush_yards']
                    defense_val = home_def.iloc[0]['defense_rush_yards_allowed']

                    storyline, description = generate_rushing_yards_storyline(
                        offense_val, defense_val
                    )

                    matchup_data.append({
                        'team': away_team,
                        'opponent': home_team,
                        'location': 'Away',
                        'offense_rush_yards': offense_val,
                        'defense_rush_yards_allowed': defense_val,
                        'storyline': storyline,
                        'description': description
                    })

            matchup_df = pd.DataFrame(matchup_data)
        else:
            matchup_df = stats.copy()
            matchup_df['opponent'] = 'Avg'
            matchup_df['location'] = 'N/A'

        if matchup_df.empty:
            st.info("No matchup data available")
            return

        # Create scatter plot
        fig = go.Figure()

        hover_text = matchup_df['team'] if 'opponent' not in matchup_df.columns or matchup_df['opponent'].iloc[0] == 'Avg' else matchup_df['team'] + ' vs ' + matchup_df['opponent']

        fig.add_trace(go.Scatter(
            x=matchup_df['offense_rush_yards'],
            y=matchup_df['defense_rush_yards_allowed'],
            mode='markers',
            marker=dict(
                size=15,
                color=matchup_df['offense_rush_yards'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Rushing Yards per Game")
            ),
            text=hover_text,
            hovertemplate='<b>%{text}</b><br>Rushing Yards per Game: %{x}<br>Rushing Yards Allowed per Game: %{y}<extra></extra>',
            showlegend=False
        ))

        # Add team logos
        for _, row in matchup_df.iterrows():
            team = row['team']
            logo_url = get_team_logo_url(team)
            if logo_url:
                fig.add_layout_image(
                    dict(
                        source=logo_url,
                        xref="x",
                        yref="y",
                        x=row['offense_rush_yards'],
                        y=row['defense_rush_yards_allowed'],
                        sizex=0.15 * (matchup_df['offense_rush_yards'].max() - matchup_df['offense_rush_yards'].min()),
                        sizey=0.15 * (matchup_df['defense_rush_yards_allowed'].max() - matchup_df['defense_rush_yards_allowed'].min()),
                        xanchor="center",
                        yanchor="middle",
                        sizing="contain",
                        opacity=0.8,
                        layer="above"
                    )
                )

        # Add league average lines
        avg_offense = matchup_df['offense_rush_yards'].mean()
        avg_defense = matchup_df['defense_rush_yards_allowed'].mean()

        fig.add_hline(y=avg_defense, line_dash="dash", line_color="gray", annotation_text="Avg Defense")
        fig.add_vline(x=avg_offense, line_dash="dash", line_color="gray", annotation_text="Avg Offense")

        fig.update_layout(
            title=f"RB Rushing Yards Matchup Analysis ({season} Season{f', Week {week}' if week else ''})",
            xaxis_title="Rushing Yards per Game",
            yaxis_title="Rushing Yards Allowed per Game",
            height=600,
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show matchup storylines table
        if 'storyline' in matchup_df.columns:
            st.subheader("Matchup Storylines")
            display_df = matchup_df[['team', 'opponent', 'location', 'offense_rush_yards', 'defense_rush_yards_allowed', 'storyline', 'description']].copy()
            display_df.columns = ['Team', 'Opponent', 'Location', 'Rush Yards', 'Yards Allowed', 'Storyline', 'Description']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error generating chart: {e}")
        import traceback
        st.error(traceback.format_exc())


def render_rb_receiving_matchup_chart(season: Optional[int], week: Optional[int], upcoming_games=None):
    """Chart: RB Pass-Catching Exploitation Matchup Analysis"""
    st.subheader("🎯 RB Pass-Catching Matchup Analysis")
    st.markdown("""
    **Goal:** Identify pass-catching RBs in favorable PPR matchups vs defenses vulnerable to RB targets.
    - **Top-right quadrant:** 🎯 PPR SMASH (high-target RB vs RB-vulnerable defense)
    - **Top-left:** 💰 OPPORTUNITY (low-target team vs vulnerable coverage - volume spike)
    - **Bottom-right:** 🚫 AVOID (high-target RB vs lockdown LB coverage)
    - **Bottom-left:** ⚖️ NEUTRAL (balanced matchup)
    """)

    if not season:
        st.warning("No season data available.")
        return

    try:
        # Calculate RB receiving matchup stats
        rb_stats = calculate_rb_receiving_matchup_stats(season, week)

        if rb_stats.empty:
            st.info("No RB receiving data available for selected filters.")
            return

        # If upcoming games provided, create matchup-specific data
        if upcoming_games is not None and not upcoming_games.empty:
            matchup_data = []

            for _, game in upcoming_games.iterrows():
                home_team = game['home_team']
                away_team = game['away_team']

                # Home team RBs vs Away team defense
                home_stats = rb_stats[rb_stats['team'] == home_team]
                away_def = rb_stats[rb_stats['team'] == away_team]

                if not home_stats.empty and not away_def.empty:
                    rb_targets_per_game = home_stats.iloc[0]['rb_targets_per_game']
                    defense_rec_to_rb = away_def.iloc[0]['defense_rec_to_rb']

                    storyline, description = generate_rb_receiving_storyline(
                        rb_targets_per_game, defense_rec_to_rb
                    )

                    matchup_data.append({
                        'team': home_team,
                        'opponent': away_team,
                        'location': 'Home',
                        'rb_targets_per_game': rb_targets_per_game,
                        'defense_rec_to_rb': defense_rec_to_rb,
                        'storyline': storyline,
                        'description': description
                    })

                # Away team RBs vs Home team defense
                away_stats = rb_stats[rb_stats['team'] == away_team]
                home_def = rb_stats[rb_stats['team'] == home_team]

                if not away_stats.empty and not home_def.empty:
                    rb_targets_per_game = away_stats.iloc[0]['rb_targets_per_game']
                    defense_rec_to_rb = home_def.iloc[0]['defense_rec_to_rb']

                    storyline, description = generate_rb_receiving_storyline(
                        rb_targets_per_game, defense_rec_to_rb
                    )

                    matchup_data.append({
                        'team': away_team,
                        'opponent': home_team,
                        'location': 'Away',
                        'rb_targets_per_game': rb_targets_per_game,
                        'defense_rec_to_rb': defense_rec_to_rb,
                        'storyline': storyline,
                        'description': description
                    })

            matchup_df = pd.DataFrame(matchup_data)
        else:
            # Show all teams (not matchup-specific)
            matchup_df = rb_stats.copy()
            avg_def_rec = rb_stats['defense_rec_to_rb'].mean()
            matchup_df['opponent'] = 'Avg'
            matchup_df['location'] = 'N/A'

        if matchup_df.empty:
            st.info("No RB receiving matchup data available")
            return

        # Create scatter plot
        fig = go.Figure()

        # Add invisible markers for hover functionality
        hover_text = matchup_df['team'] if 'opponent' not in matchup_df.columns or matchup_df['opponent'].iloc[0] == 'Avg' else matchup_df['team'] + ' vs ' + matchup_df['opponent']

        fig.add_trace(go.Scatter(
            x=matchup_df['rb_targets_per_game'],
            y=matchup_df['defense_rec_to_rb'],
            mode='markers',
            marker=dict(
                size=15,
                color=matchup_df['rb_targets_per_game'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="RB Targets/Gm"),
                opacity=0.6,
                line=dict(width=2, color='white')
            ),
            text=hover_text,
            customdata=matchup_df[['rb_targets_per_game', 'defense_rec_to_rb'] +
                                  (['storyline'] if 'storyline' in matchup_df.columns else [])].values if 'storyline' in matchup_df.columns else matchup_df[['rb_targets_per_game', 'defense_rec_to_rb']].values,
            hovertemplate='<b>%{text}</b><br>' +
                         'RB Targets/Gm: %{x:.1f}<br>' +
                         'Def Rec Yds to RBs: %{y:.1f}<br>' +
                         ('%{customdata[2]}<br>' if 'storyline' in matchup_df.columns else '') +
                         '<extra></extra>',
            showlegend=False
        ))

        # Add league average lines
        avg_rb_targets = matchup_df['rb_targets_per_game'].mean()
        avg_def_rec = matchup_df['defense_rec_to_rb'].mean()

        fig.add_hline(y=avg_def_rec, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="Avg Def Rec to RBs", annotation_position="right")
        fig.add_vline(x=avg_rb_targets, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="Avg RB Targets", annotation_position="top")

        # Add quadrant labels
        fig.add_annotation(x=matchup_df['rb_targets_per_game'].max() * 0.95, y=matchup_df['defense_rec_to_rb'].max() * 0.95,
                          text="🎯 PPR SMASH", showarrow=False, font=dict(size=14, color="green"))
        fig.add_annotation(x=matchup_df['rb_targets_per_game'].min() * 1.05, y=matchup_df['defense_rec_to_rb'].max() * 0.95,
                          text="💰 OPPORTUNITY", showarrow=False, font=dict(size=14, color="blue"))
        fig.add_annotation(x=matchup_df['rb_targets_per_game'].max() * 0.95, y=matchup_df['defense_rec_to_rb'].min() * 1.05,
                          text="🚫 AVOID", showarrow=False, font=dict(size=14, color="red"))
        fig.add_annotation(x=matchup_df['rb_targets_per_game'].min() * 1.05, y=matchup_df['defense_rec_to_rb'].min() * 1.05,
                          text="⚖️ NEUTRAL", showarrow=False, font=dict(size=14, color="gray"))

        # Calculate logo sizes
        x_range = matchup_df['rb_targets_per_game'].max() - matchup_df['rb_targets_per_game'].min()
        y_range = matchup_df['defense_rec_to_rb'].max() - matchup_df['defense_rec_to_rb'].min()
        logo_size_x = max(x_range * 0.08, 0.5)
        logo_size_y = max(y_range * 0.08, 2.0)

        # Build layout with team logo images
        layout_images = []
        for idx, row in matchup_df.iterrows():
            layout_images.append(dict(
                source=get_team_logo_url(row['team']),
                xref="x",
                yref="y",
                x=row['rb_targets_per_game'],
                y=row['defense_rec_to_rb'],
                sizex=logo_size_x,
                sizey=logo_size_y,
                xanchor="center",
                yanchor="middle",
                layer="above",
                opacity=0.9
            ))

        fig.update_layout(
            title=f"RB Pass-Catching Matchup Analysis ({season} Season{f', Week {week}' if week else ''})",
            xaxis_title="RB Targets per Game",
            yaxis_title="Defense Rec Yards to RBs (per game)",
            height=600,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            images=layout_images
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show matchup storylines table
        if 'storyline' in matchup_df.columns:
            with st.expander("📋 View RB Receiving Matchup Details"):
                display_df = matchup_df[['team', 'opponent', 'location', 'rb_targets_per_game',
                                        'defense_rec_to_rb', 'storyline', 'description']].copy()
                display_df.columns = ['Team', 'Opponent', 'Location', 'RB Targets/Gm',
                                     'Def Rec Yds to RBs', 'Storyline', 'Description']
                display_df = display_df.round({'RB Targets/Gm': 1, 'Def Rec Yds to RBs': 1})
                display_df = display_df.sort_values('RB Targets/Gm', ascending=False)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Highlight PPR SMASH matchups
                smash = display_df[display_df['Storyline'].str.contains('PPR SMASH|CHECKDOWN')]
                if not smash.empty:
                    st.success(f"**{len(smash)} PPR Smash Matchup(s):** Pass-catching RBs in favorable spots")
                    for _, row in smash.head(3).iterrows():
                        st.markdown(f"- **{row['Team']} vs {row['Opponent']}**: {row['Description']}")
                else:
                    st.info("No standout PPR RB matchups this week")

                # Highlight AVOID matchups
                avoid = display_df[display_df['Storyline'].str.contains('AVOID|LIMITED')]
                if not avoid.empty:
                    st.warning(f"**{len(avoid)} Tough Coverage Matchup(s):** RBs face elite LB coverage")
                    for _, row in avoid.head(3).iterrows():
                        st.markdown(f"- **{row['Team']} vs {row['Opponent']}**: {row['Description']}")

    except Exception as e:
        st.error(f"Error generating RB receiving matchup chart: {e}")
        import traceback
        st.error(traceback.format_exc())



def render_passing_td_matchup_chart(season: Optional[int], week: Optional[int], upcoming_games=None):
    """Chart: QB Passing TD Matchup Analysis"""
    st.subheader("🎯 QB Passing TD Matchup Analysis")
    st.markdown("""
    **Goal:** Match QB passing TD rates vs defensive TD prevention (API data: `passing_tds` per game)
    - **Top-right quadrant:** 🎯 TD SMASH SPOT (high TD production vs TD-prone defense)
    - **Top-left:** 💎 SNEAKY VALUE (game manager vs TD-prone defense)
    - **Bottom-right:** ⚔️ TOUGH TEST (elite QB vs lockdown secondary)
    - **Bottom-left:** ⚖️ BALANCED (balanced matchup)
    """)

    if not season:
        st.warning("No season data available.")
        return

    try:
        # Calculate stats
        stats = calculate_passing_td_matchup_stats(season, week)

        if stats.empty:
            st.info("No data available for selected filters.")
            return

        # If upcoming games provided, create matchup-specific data
        if upcoming_games is not None and not upcoming_games.empty:
            matchup_data = []

            for _, game in upcoming_games.iterrows():
                home_team = game['home_team']
                away_team = game['away_team']

                # Home team offense vs Away team defense
                home_stats = stats[stats['team'] == home_team]
                away_def = stats[stats['team'] == away_team]

                if not home_stats.empty and not away_def.empty:
                    offense_val = home_stats.iloc[0]['offense_pass_tds']
                    defense_val = away_def.iloc[0]['defense_pass_tds_allowed']

                    storyline, description = generate_passing_td_storyline(
                        offense_val, defense_val
                    )

                    matchup_data.append({
                        'team': home_team,
                        'opponent': away_team,
                        'location': 'Home',
                        'offense_pass_tds': offense_val,
                        'defense_pass_tds_allowed': defense_val,
                        'storyline': storyline,
                        'description': description
                    })

                # Away team offense vs Home team defense
                away_stats = stats[stats['team'] == away_team]
                home_def = stats[stats['team'] == home_team]

                if not away_stats.empty and not home_def.empty:
                    offense_val = away_stats.iloc[0]['offense_pass_tds']
                    defense_val = home_def.iloc[0]['defense_pass_tds_allowed']

                    storyline, description = generate_passing_td_storyline(
                        offense_val, defense_val
                    )

                    matchup_data.append({
                        'team': away_team,
                        'opponent': home_team,
                        'location': 'Away',
                        'offense_pass_tds': offense_val,
                        'defense_pass_tds_allowed': defense_val,
                        'storyline': storyline,
                        'description': description
                    })

            matchup_df = pd.DataFrame(matchup_data)
        else:
            matchup_df = stats.copy()
            matchup_df['opponent'] = 'Avg'
            matchup_df['location'] = 'N/A'

        if matchup_df.empty:
            st.info("No matchup data available")
            return

        # Create scatter plot
        fig = go.Figure()

        hover_text = matchup_df['team'] if 'opponent' not in matchup_df.columns or matchup_df['opponent'].iloc[0] == 'Avg' else matchup_df['team'] + ' vs ' + matchup_df['opponent']

        fig.add_trace(go.Scatter(
            x=matchup_df['offense_pass_tds'],
            y=matchup_df['defense_pass_tds_allowed'],
            mode='markers',
            marker=dict(
                size=15,
                color=matchup_df['offense_pass_tds'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Passing TDs per Game")
            ),
            text=hover_text,
            hovertemplate='<b>%{text}</b><br>Passing TDs per Game: %{x}<br>Passing TDs Allowed per Game: %{y}<extra></extra>',
            showlegend=False
        ))

        # Add team logos
        for _, row in matchup_df.iterrows():
            team = row['team']
            logo_url = get_team_logo_url(team)
            if logo_url:
                fig.add_layout_image(
                    dict(
                        source=logo_url,
                        xref="x",
                        yref="y",
                        x=row['offense_pass_tds'],
                        y=row['defense_pass_tds_allowed'],
                        sizex=0.15 * (matchup_df['offense_pass_tds'].max() - matchup_df['offense_pass_tds'].min()),
                        sizey=0.15 * (matchup_df['defense_pass_tds_allowed'].max() - matchup_df['defense_pass_tds_allowed'].min()),
                        xanchor="center",
                        yanchor="middle",
                        sizing="contain",
                        opacity=0.8,
                        layer="above"
                    )
                )

        # Add league average lines
        avg_offense = matchup_df['offense_pass_tds'].mean()
        avg_defense = matchup_df['defense_pass_tds_allowed'].mean()

        fig.add_hline(y=avg_defense, line_dash="dash", line_color="gray", annotation_text="Avg Defense")
        fig.add_vline(x=avg_offense, line_dash="dash", line_color="gray", annotation_text="Avg Offense")

        fig.update_layout(
            title=f"QB Passing TD Matchup Analysis ({season} Season{f', Week {week}' if week else ''})",
            xaxis_title="Passing TDs per Game",
            yaxis_title="Passing TDs Allowed per Game",
            height=600,
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show matchup storylines table
        if 'storyline' in matchup_df.columns:
            st.subheader("Matchup Storylines")
            display_df = matchup_df[['team', 'opponent', 'location', 'offense_pass_tds', 'defense_pass_tds_allowed', 'storyline', 'description']].copy()
            display_df.columns = ['Team', 'Opponent', 'Location', 'Pass TDs', 'TDs Allowed', 'Storyline', 'Description']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error generating chart: {e}")
        import traceback
        st.error(traceback.format_exc())


def render_passing_yards_matchup_chart(season: Optional[int], week: Optional[int], upcoming_games=None):
    """Chart: QB Passing Yards Matchup Analysis"""
    st.subheader("🚀 QB Passing Yards Matchup Analysis")
    st.markdown("""
    **Goal:** Match QB passing yards vs defensive passing yards allowed (API data: `passing_yards` per game)
    - **Top-right quadrant:** 🚀 CEILING EXPLOSION (air raid vs pass funnel)
    - **Top-left:** 📈 VOLUME SPIKE (run-first vs pass funnel)
    - **Bottom-right:** 💪 GRIND IT OUT (air raid vs elite coverage)
    - **Bottom-left:** ⚠️ LOW VOLUME (run-first vs elite coverage)
    """)

    if not season:
        st.warning("No season data available.")
        return

    try:
        # Calculate stats
        stats = calculate_passing_yards_matchup_stats(season, week)

        if stats.empty:
            st.info("No data available for selected filters.")
            return

        # If upcoming games provided, create matchup-specific data
        if upcoming_games is not None and not upcoming_games.empty:
            matchup_data = []

            for _, game in upcoming_games.iterrows():
                home_team = game['home_team']
                away_team = game['away_team']

                # Home team offense vs Away team defense
                home_stats = stats[stats['team'] == home_team]
                away_def = stats[stats['team'] == away_team]

                if not home_stats.empty and not away_def.empty:
                    offense_val = home_stats.iloc[0]['offense_pass_yards']
                    defense_val = away_def.iloc[0]['defense_pass_yards_allowed']

                    storyline, description = generate_passing_yards_storyline(
                        offense_val, defense_val
                    )

                    matchup_data.append({
                        'team': home_team,
                        'opponent': away_team,
                        'location': 'Home',
                        'offense_pass_yards': offense_val,
                        'defense_pass_yards_allowed': defense_val,
                        'storyline': storyline,
                        'description': description
                    })

                # Away team offense vs Home team defense
                away_stats = stats[stats['team'] == away_team]
                home_def = stats[stats['team'] == home_team]

                if not away_stats.empty and not home_def.empty:
                    offense_val = away_stats.iloc[0]['offense_pass_yards']
                    defense_val = home_def.iloc[0]['defense_pass_yards_allowed']

                    storyline, description = generate_passing_yards_storyline(
                        offense_val, defense_val
                    )

                    matchup_data.append({
                        'team': away_team,
                        'opponent': home_team,
                        'location': 'Away',
                        'offense_pass_yards': offense_val,
                        'defense_pass_yards_allowed': defense_val,
                        'storyline': storyline,
                        'description': description
                    })

            matchup_df = pd.DataFrame(matchup_data)
        else:
            matchup_df = stats.copy()
            matchup_df['opponent'] = 'Avg'
            matchup_df['location'] = 'N/A'

        if matchup_df.empty:
            st.info("No matchup data available")
            return

        # Create scatter plot
        fig = go.Figure()

        hover_text = matchup_df['team'] if 'opponent' not in matchup_df.columns or matchup_df['opponent'].iloc[0] == 'Avg' else matchup_df['team'] + ' vs ' + matchup_df['opponent']

        fig.add_trace(go.Scatter(
            x=matchup_df['offense_pass_yards'],
            y=matchup_df['defense_pass_yards_allowed'],
            mode='markers',
            marker=dict(
                size=15,
                color=matchup_df['offense_pass_yards'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Passing Yards per Game")
            ),
            text=hover_text,
            hovertemplate='<b>%{text}</b><br>Passing Yards per Game: %{x}<br>Passing Yards Allowed per Game: %{y}<extra></extra>',
            showlegend=False
        ))

        # Add team logos
        for _, row in matchup_df.iterrows():
            team = row['team']
            logo_url = get_team_logo_url(team)
            if logo_url:
                fig.add_layout_image(
                    dict(
                        source=logo_url,
                        xref="x",
                        yref="y",
                        x=row['offense_pass_yards'],
                        y=row['defense_pass_yards_allowed'],
                        sizex=0.15 * (matchup_df['offense_pass_yards'].max() - matchup_df['offense_pass_yards'].min()),
                        sizey=0.15 * (matchup_df['defense_pass_yards_allowed'].max() - matchup_df['defense_pass_yards_allowed'].min()),
                        xanchor="center",
                        yanchor="middle",
                        sizing="contain",
                        opacity=0.8,
                        layer="above"
                    )
                )

        # Add league average lines
        avg_offense = matchup_df['offense_pass_yards'].mean()
        avg_defense = matchup_df['defense_pass_yards_allowed'].mean()

        fig.add_hline(y=avg_defense, line_dash="dash", line_color="gray", annotation_text="Avg Defense")
        fig.add_vline(x=avg_offense, line_dash="dash", line_color="gray", annotation_text="Avg Offense")

        fig.update_layout(
            title=f"QB Passing Yards Matchup Analysis ({season} Season{f', Week {week}' if week else ''})",
            xaxis_title="Passing Yards per Game",
            yaxis_title="Passing Yards Allowed per Game",
            height=600,
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show matchup storylines table
        if 'storyline' in matchup_df.columns:
            st.subheader("Matchup Storylines")
            display_df = matchup_df[['team', 'opponent', 'location', 'offense_pass_yards', 'defense_pass_yards_allowed', 'storyline', 'description']].copy()
            display_df.columns = ['Team', 'Opponent', 'Location', 'Pass Yards', 'Yards Allowed', 'Storyline', 'Description']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error generating chart: {e}")
        import traceback
        st.error(traceback.format_exc())


def render_qb_td_rate_vs_int_chart(season: Optional[int], week: Optional[int], upcoming_games=None):
    """Chart: QB TD Rate vs Defensive INT Rate"""
    st.subheader("⚡ QB TD Rate vs Defensive INT Rate")
    st.markdown("""
    **Goal:** Match QB TD efficiency vs defensive INT creation (API: `passing_tds`/attempts vs `def_interceptions`)
    - **Top-right quadrant:** ⚡ HIGH CEILING LOW RISK (elite efficiency vs passive coverage)
    - **Top-left:** 🎲 BOOM-BUST (elite efficiency vs ball-hawk defense)
    - **Bottom-right:** 💎 SOLID PLAY (elite efficiency vs average coverage)
    - **Bottom-left:** ⚖️ BALANCED (average efficiency)
    """)

    if not season:
        st.warning("No season data available.")
        return

    try:
        # Calculate stats
        stats = calculate_qb_td_rate_vs_int_stats(season, week)

        if stats.empty:
            st.info("No data available for selected filters.")
            return

        # If upcoming games provided, create matchup-specific data
        if upcoming_games is not None and not upcoming_games.empty:
            matchup_data = []

            for _, game in upcoming_games.iterrows():
                home_team = game['home_team']
                away_team = game['away_team']

                # Home team offense vs Away team defense
                home_stats = stats[stats['team'] == home_team]
                away_def = stats[stats['team'] == away_team]

                if not home_stats.empty and not away_def.empty:
                    offense_val = home_stats.iloc[0]['qb_td_rate']
                    defense_val = away_def.iloc[0]['defense_ints_per_game']

                    storyline, description = generate_qb_td_rate_vs_int_storyline(
                        offense_val, defense_val
                    )

                    matchup_data.append({
                        'team': home_team,
                        'opponent': away_team,
                        'location': 'Home',
                        'qb_td_rate': offense_val,
                        'defense_ints_per_game': defense_val,
                        'storyline': storyline,
                        'description': description
                    })

                # Away team offense vs Home team defense
                away_stats = stats[stats['team'] == away_team]
                home_def = stats[stats['team'] == home_team]

                if not away_stats.empty and not home_def.empty:
                    offense_val = away_stats.iloc[0]['qb_td_rate']
                    defense_val = home_def.iloc[0]['defense_ints_per_game']

                    storyline, description = generate_qb_td_rate_vs_int_storyline(
                        offense_val, defense_val
                    )

                    matchup_data.append({
                        'team': away_team,
                        'opponent': home_team,
                        'location': 'Away',
                        'qb_td_rate': offense_val,
                        'defense_ints_per_game': defense_val,
                        'storyline': storyline,
                        'description': description
                    })

            matchup_df = pd.DataFrame(matchup_data)
        else:
            matchup_df = stats.copy()
            matchup_df['opponent'] = 'Avg'
            matchup_df['location'] = 'N/A'

        if matchup_df.empty:
            st.info("No matchup data available")
            return

        # Create scatter plot
        fig = go.Figure()

        hover_text = matchup_df['team'] if 'opponent' not in matchup_df.columns or matchup_df['opponent'].iloc[0] == 'Avg' else matchup_df['team'] + ' vs ' + matchup_df['opponent']

        fig.add_trace(go.Scatter(
            x=matchup_df['qb_td_rate'],
            y=matchup_df['defense_ints_per_game'],
            mode='markers',
            marker=dict(
                size=15,
                color=matchup_df['qb_td_rate'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="QB TD Rate (%)")
            ),
            text=hover_text,
            hovertemplate='<b>%{text}</b><br>QB TD Rate (%): %{x}<br>Defensive INTs per Game: %{y}<extra></extra>',
            showlegend=False
        ))

        # Add team logos
        for _, row in matchup_df.iterrows():
            team = row['team']
            logo_url = get_team_logo_url(team)
            if logo_url:
                fig.add_layout_image(
                    dict(
                        source=logo_url,
                        xref="x",
                        yref="y",
                        x=row['qb_td_rate'],
                        y=row['defense_ints_per_game'],
                        sizex=0.15 * (matchup_df['qb_td_rate'].max() - matchup_df['qb_td_rate'].min()),
                        sizey=0.15 * (matchup_df['defense_ints_per_game'].max() - matchup_df['defense_ints_per_game'].min()),
                        xanchor="center",
                        yanchor="middle",
                        sizing="contain",
                        opacity=0.8,
                        layer="above"
                    )
                )

        # Add league average lines
        avg_offense = matchup_df['qb_td_rate'].mean()
        avg_defense = matchup_df['defense_ints_per_game'].mean()

        fig.add_hline(y=avg_defense, line_dash="dash", line_color="gray", annotation_text="Avg Defense")
        fig.add_vline(x=avg_offense, line_dash="dash", line_color="gray", annotation_text="Avg Offense")

        fig.update_layout(
            title=f"QB TD Rate vs Defensive INT Rate ({season} Season{f', Week {week}' if week else ''})",
            xaxis_title="QB TD Rate (%)",
            yaxis_title="Defensive INTs per Game",
            height=600,
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show matchup storylines table
        if 'storyline' in matchup_df.columns:
            st.subheader("Matchup Storylines")
            display_df = matchup_df[['team', 'opponent', 'location', 'qb_td_rate', 'defense_ints_per_game', 'storyline', 'description']].copy()
            display_df.columns = ['Team', 'Opponent', 'Location', 'TD Rate %', 'INTs/Gm', 'Storyline', 'Description']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error generating chart: {e}")
        import traceback
        st.error(traceback.format_exc())


def render_qb_efficiency_matchup_chart(season: Optional[int], week: Optional[int], upcoming_games=None):
    """Chart: QB Efficiency Matchup (Composite)"""
    st.subheader("💎 QB Efficiency Matchup (Composite)")
    st.markdown("""
    **Goal:** Composite QB efficiency (yards + INT rate) vs defensive passing efficiency (yards + INTs + sacks)
    - **Top-right quadrant:** 💎 SAFE CEILING SPOT (elite efficient vs generous defense)
    - **Top-left:** ✅ VOLUME OPPORTUNITY (safe manager vs generous defense)
    - **Bottom-right:** 🔥 FAVORABLE (elite QB vs average defense)
    - **Bottom-left:** 🛡️ DEFENSIVE WALL (vs dominant defense)
    """)

    if not season:
        st.warning("No season data available.")
        return

    try:
        # Calculate stats
        stats = calculate_qb_efficiency_matchup_stats(season, week)

        if stats.empty:
            st.info("No data available for selected filters.")
            return

        # If upcoming games provided, create matchup-specific data
        if upcoming_games is not None and not upcoming_games.empty:
            matchup_data = []

            for _, game in upcoming_games.iterrows():
                home_team = game['home_team']
                away_team = game['away_team']

                # Home team offense vs Away team defense
                home_stats = stats[stats['team'] == home_team]
                away_def = stats[stats['team'] == away_team]

                if not home_stats.empty and not away_def.empty:
                    offense_val = home_stats.iloc[0]['qb_efficiency_score']
                    defense_val = away_def.iloc[0]['defense_efficiency_score']

                    storyline, description = generate_qb_efficiency_storyline(
                        home_stats.iloc[0]['qb_pass_yards_per_game'],
                        home_stats.iloc[0]['qb_int_rate'],
                        away_def.iloc[0]['defense_pass_yards_allowed'],
                        away_def.iloc[0]['defense_ints_per_game'],
                        away_def.iloc[0]['defense_sacks_per_game']
                    )

                    matchup_data.append({
                        'team': home_team,
                        'opponent': away_team,
                        'location': 'Home',
                        'qb_efficiency_score': offense_val,
                        'defense_efficiency_score': defense_val,
                        'storyline': storyline,
                        'description': description
                    })

                # Away team offense vs Home team defense
                away_stats = stats[stats['team'] == away_team]
                home_def = stats[stats['team'] == home_team]

                if not away_stats.empty and not home_def.empty:
                    offense_val = away_stats.iloc[0]['qb_efficiency_score']
                    defense_val = home_def.iloc[0]['defense_efficiency_score']

                    storyline, description = generate_qb_efficiency_storyline(
                        away_stats.iloc[0]['qb_pass_yards_per_game'],
                        away_stats.iloc[0]['qb_int_rate'],
                        home_def.iloc[0]['defense_pass_yards_allowed'],
                        home_def.iloc[0]['defense_ints_per_game'],
                        home_def.iloc[0]['defense_sacks_per_game']
                    )

                    matchup_data.append({
                        'team': away_team,
                        'opponent': home_team,
                        'location': 'Away',
                        'qb_efficiency_score': offense_val,
                        'defense_efficiency_score': defense_val,
                        'storyline': storyline,
                        'description': description
                    })

            matchup_df = pd.DataFrame(matchup_data)
        else:
            matchup_df = stats.copy()
            matchup_df['opponent'] = 'Avg'
            matchup_df['location'] = 'N/A'

        if matchup_df.empty:
            st.info("No matchup data available")
            return

        # Create scatter plot
        fig = go.Figure()

        hover_text = matchup_df['team'] if 'opponent' not in matchup_df.columns or matchup_df['opponent'].iloc[0] == 'Avg' else matchup_df['team'] + ' vs ' + matchup_df['opponent']

        fig.add_trace(go.Scatter(
            x=matchup_df['qb_efficiency_score'],
            y=matchup_df['defense_efficiency_score'],
            mode='markers',
            marker=dict(
                size=15,
                color=matchup_df['qb_efficiency_score'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="QB Efficiency Score")
            ),
            text=hover_text,
            hovertemplate='<b>%{text}</b><br>QB Efficiency Score: %{x}<br>Defensive Efficiency Score: %{y}<extra></extra>',
            showlegend=False
        ))

        # Add team logos
        for _, row in matchup_df.iterrows():
            team = row['team']
            logo_url = get_team_logo_url(team)
            if logo_url:
                fig.add_layout_image(
                    dict(
                        source=logo_url,
                        xref="x",
                        yref="y",
                        x=row['qb_efficiency_score'],
                        y=row['defense_efficiency_score'],
                        sizex=0.15 * (matchup_df['qb_efficiency_score'].max() - matchup_df['qb_efficiency_score'].min()),
                        sizey=0.15 * (matchup_df['defense_efficiency_score'].max() - matchup_df['defense_efficiency_score'].min()),
                        xanchor="center",
                        yanchor="middle",
                        sizing="contain",
                        opacity=0.8,
                        layer="above"
                    )
                )

        # Add league average lines
        avg_offense = matchup_df['qb_efficiency_score'].mean()
        avg_defense = matchup_df['defense_efficiency_score'].mean()

        fig.add_hline(y=avg_defense, line_dash="dash", line_color="gray", annotation_text="Avg Defense")
        fig.add_vline(x=avg_offense, line_dash="dash", line_color="gray", annotation_text="Avg Offense")

        fig.update_layout(
            title=f"QB Efficiency Matchup (Composite) ({season} Season{f', Week {week}' if week else ''})",
            xaxis_title="QB Efficiency Score",
            yaxis_title="Defensive Efficiency Score",
            height=600,
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show matchup storylines table
        if 'storyline' in matchup_df.columns:
            st.subheader("Matchup Storylines")
            display_df = matchup_df[['team', 'opponent', 'location', 'qb_efficiency_score', 'defense_efficiency_score', 'storyline', 'description']].copy()
            display_df.columns = ['Team', 'Opponent', 'Location', 'QB Eff', 'Def Eff', 'Storyline', 'Description']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error generating chart: {e}")
        import traceback
        st.error(traceback.format_exc())



def render_power_rating_yards_chart(season: Optional[int], week: Optional[int]):
    """Chart: Power Rating vs Offensive Yards per Game."""
    st.subheader("⚡ Power Rating vs Offensive Yards")
    st.markdown("""
    **Goal:** Identify teams that produce yards and their overall team strength.
    - **Top-right quadrant:** Elite teams (high power rating, high offensive yards)
    - **Top-left:** Strong overall but lower offensive production
    - **Bottom-right:** High offensive yards but lower overall team strength
    - **Bottom-left:** Struggling teams overall
    """)

    if not season:
        st.warning("No season data available.")
        return

    try:
        # Get all teams
        teams_query = f"SELECT DISTINCT home_team_abbr as team FROM games WHERE season={season} ORDER BY home_team_abbr"
        teams_df = query(teams_query)

        if teams_df.empty:
            st.warning("No teams found for this season.")
            return

        all_teams = teams_df['team'].unique()

        # Calculate power ratings for each team using the same method as Power Rankings view
        st.info(f"Calculating power ratings for {len(all_teams)} teams...")

        # Determine the week to use for calculation
        if week:
            calc_week = week
        else:
            max_week_query = f"SELECT MAX(week) as max_week FROM games WHERE season={season}"
            max_week_df = query(max_week_query)
            calc_week = int(max_week_df['max_week'].iloc[0]) if not max_week_df.empty else 1

        # Step 1: Calculate league statistics
        league_stats = calculate_league_statistics(season, calc_week, list(all_teams))

        # Step 2: Calculate baseline power ratings for all teams (for SOS calculation)
        all_team_powers = {}
        for team in all_teams:
            try:
                power = calculate_team_power_rating(team, season, calc_week, all_team_powers=None, league_stats=league_stats)
                all_team_powers[team] = power
            except:
                all_team_powers[team] = 0.0

        # Step 3: Calculate quality margin-adjusted league stats
        quality_margins = []
        for team in all_teams:
            try:
                qm = calculate_quality_victory_margin(team, season, calc_week, all_team_powers)
                quality_margins.append(qm.get('quality_margin_per_game', 0))
            except:
                quality_margins.append(0)

        if len(quality_margins) > 1:
            import statistics
            league_stats['quality_margin'] = {
                'mean': statistics.mean(quality_margins),
                'std': statistics.stdev(quality_margins) if len(quality_margins) > 1 else 1.0
            }

        # Step 4: Calculate final power ratings with quality margin adjustments
        final_team_powers = {}
        for team in all_teams:
            try:
                final_power = calculate_team_power_rating(team, season, calc_week, all_team_powers, league_stats)
                final_team_powers[team] = final_power
            except:
                final_team_powers[team] = 0.0

        all_team_powers = final_team_powers

        # Get offensive yards for each team
        week_filter = f"AND week <= {week}" if week else ""

        sql_yards = f"""
        SELECT
            team_abbr,
            COUNT(DISTINCT game_id) as games,
            AVG(yards_total) as avg_yards,
            SUM(yards_total) as total_yards
        FROM team_game_summary
        WHERE season = ?
        {week_filter}
        GROUP BY team_abbr
        """

        df_yards = query(sql_yards, (season,))

        if df_yards.empty:
            st.info("No offensive data available for selected filters")
            return

        # Combine power ratings with yards
        chart_data = []
        for team in all_teams:
            power_rating = all_team_powers.get(team, 50.0)
            yards_row = df_yards[df_yards['team_abbr'] == team]

            if not yards_row.empty:
                avg_yards = yards_row['avg_yards'].iloc[0]
                chart_data.append({
                    'team_abbr': team,
                    'power_rating': power_rating,
                    'avg_yards': avg_yards
                })

        df = pd.DataFrame(chart_data)

        if df.empty:
            st.info("No data available to display")
            return

        # Create scatter plot
        fig = go.Figure()

        # Add invisible markers for hover functionality
        fig.add_trace(go.Scatter(
            x=df['avg_yards'],
            y=df['power_rating'],
            mode='markers',
            marker=dict(
                size=1,
                color=df['power_rating'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Power Rating"),
                opacity=0
            ),
            text=df['team_abbr'],
            customdata=df[['power_rating', 'avg_yards']],
            hovertemplate='<b>%{text}</b><br>' +
                         'Power Rating: %{customdata[0]:.1f}<br>' +
                         'Offensive Yards/Game: %{customdata[1]:.1f}<br>' +
                         '<extra></extra>',
            showlegend=False
        ))

        # Add league average lines
        avg_power = df['power_rating'].mean()
        avg_yards = df['avg_yards'].mean()

        fig.add_hline(y=avg_power, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="League Avg Power", annotation_position="right")
        fig.add_vline(x=avg_yards, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="League Avg Yards", annotation_position="top")

        # Calculate dynamic sizing for team logos
        x_range = df['avg_yards'].max() - df['avg_yards'].min()
        y_range = df['power_rating'].max() - df['power_rating'].min()

        logo_size_x = max(x_range * 0.06, 10)  # At least 10 yards wide
        logo_size_y = max(y_range * 0.06, 3)   # At least 3 rating points tall

        # Build layout with team logo images
        layout_images = []
        for idx, row in df.iterrows():
            layout_images.append(dict(
                source=get_team_logo_url(row['team_abbr']),
                xref="x",
                yref="y",
                x=row['avg_yards'],
                y=row['power_rating'],
                sizex=logo_size_x,
                sizey=logo_size_y,
                xanchor="center",
                yanchor="middle",
                layer="above",
                opacity=0.9
            ))

        fig.update_layout(
            title=f"Power Rating vs Offensive Yards per Game ({season} Season)",
            xaxis_title="Offensive Yards per Game",
            yaxis_title="Power Rating (1-100 scale, 50 = average)",
            height=600,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            images=layout_images
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        with st.expander("📋 View Power Rating & Yards Table"):
            display_df = df[['team_abbr', 'power_rating', 'avg_yards']].copy()
            display_df.columns = ['Team', 'Power Rating', 'Yards/Game']
            display_df = display_df.round(1).sort_values('Power Rating', ascending=False)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error generating chart: {e}")
        import traceback
        st.error(traceback.format_exc())


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
            AVG(pb.rush_td) as avg_tds_per_game,
            CAST(SUM(pb.rush_yds) AS FLOAT) / NULLIF(SUM(pb.rush_att), 0) as yards_per_carry
        FROM player_box_score pb
        JOIN games g ON pb.game_id = g.game_id
        WHERE g.season = ?
        {week_filter}
        AND pb.rush_att > 0
        GROUP BY pb.player, pb.team
        HAVING total_attempts >= ?
        ORDER BY avg_yards_per_game DESC
        """

        df = query(sql_rb, (season, min_touches))

        if df.empty:
            st.info(f"No running backs with at least {min_touches} attempts found")
            return

        # Create scatter plot
        fig = go.Figure()

        # Add invisible markers for hover functionality
        fig.add_trace(go.Scatter(
            x=df['avg_yards_per_game'],
            y=df['avg_tds_per_game'],
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
            customdata=df[['player', 'team', 'total_attempts', 'yards_per_carry', 'games', 'total_yards', 'total_tds']],
            hovertemplate='<b>%{customdata[0]}</b> (%{customdata[1]})<br>' +
                         'Avg Rush Yards/Game: %{x:.1f}<br>' +
                         'Avg Rush TDs/Game: %{y:.2f}<br>' +
                         'Games Played: %{customdata[4]}<br>' +
                         'Total Yards: %{customdata[5]}<br>' +
                         'Total TDs: %{customdata[6]}<br>' +
                         'Attempts: %{customdata[2]}<br>' +
                         'Yards/Carry: %{customdata[3]:.2f}<br>' +
                         '<extra></extra>',
            showlegend=False
        ))

        # Calculate dynamic sizing based on data ranges
        x_range = df['avg_yards_per_game'].max() - df['avg_yards_per_game'].min()
        y_range = df['avg_tds_per_game'].max() - df['avg_tds_per_game'].min()

        # Use fixed size that works well for player headshots (adjusted for per-game stats)
        headshot_size_x = max(x_range * 0.04, 3)  # Smaller since using per-game yards
        headshot_size_y = max(y_range * 0.15, 0.05)  # Smaller since using per-game TDs

        # Build layout with player headshot images
        layout_images = []
        for idx, row in df.iterrows():
            layout_images.append(dict(
                source=get_player_headshot_url(row['player'], row['team']),
                xref="x",
                yref="y",
                x=row['avg_yards_per_game'],
                y=row['avg_tds_per_game'],
                sizex=headshot_size_x,
                sizey=headshot_size_y,
                xanchor="center",
                yanchor="middle",
                layer="above",
                opacity=0.9
            ))

        fig.update_layout(
            title=f"Rushing Yards vs Touchdowns per Game ({season} Season)",
            xaxis_title="Avg Rushing Yards per Game",
            yaxis_title="Avg Rushing Touchdowns per Game",
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


def render_qb_rush_vs_pass_yards_chart(season: Optional[int], week: Optional[int]):
    """Chart: QB Median Rush Yards vs Median Pass Yards."""
    st.subheader("🏃 QB Rush Yards vs Pass Yards (Medians)")
    st.markdown("*Shows QB dual-threat capability. Higher rush yards indicate mobile QBs.*")

    if not season:
        st.warning("No season data available.")
        return

    # Minimum attempts filter
    min_attempts = st.slider("Minimum Pass Attempts", 10, 300, 50, 10, key="qb_rush_pass_min_att")

    # Build query to get QB stats
    week_filter = f"AND g.week <= {week}" if week else ""

    sql_qb = f"""
    SELECT
        pb.player,
        pb.team,
        pb.pass_yds,
        pb.rush_yds,
        pb.pass_att
    FROM player_box_score pb
    JOIN games g ON pb.game_id = g.game_id
    WHERE g.season = ?
    {week_filter}
    AND pb.pass_att > 0
    """

    try:
        df = query(sql_qb, (season,))

        if df.empty:
            st.info("No QB data available.")
            return

        # Calculate medians per QB
        qb_medians = df.groupby(['player', 'team']).agg({
            'pass_yds': 'median',
            'rush_yds': 'median',
            'pass_att': ['sum', 'count']
        }).reset_index()

        qb_medians.columns = ['player', 'team', 'median_pass_yds', 'median_rush_yds', 'total_pass_att', 'games']

        # Filter by minimum attempts
        qb_medians = qb_medians[qb_medians['total_pass_att'] >= min_attempts]

        if qb_medians.empty:
            st.info(f"No QB data available with at least {min_attempts} attempts.")
            return

        import plotly.express as px

        # X = Median Pass Yards, Y = Median Rush Yards
        fig = px.scatter(
            qb_medians,
            x='median_pass_yds',
            y='median_rush_yds',
            hover_data=['player', 'team', 'games'],
            text='player',
            labels={
                'median_pass_yds': 'Median Passing Yards',
                'median_rush_yds': 'Median Rushing Yards'
            },
            title=f"QB Median Rush Yards vs Median Pass Yards ({season})",
            color='median_rush_yds',
            color_continuous_scale='Reds'
        )
        fig.update_traces(textposition='top center', marker=dict(size=12))
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        with st.expander("📋 View QB Stats Table"):
            display_df = qb_medians[['player', 'team', 'games', 'median_pass_yds', 'median_rush_yds', 'total_pass_att']].copy()
            display_df.columns = ['Player', 'Team', 'Games', 'Med Pass Yds', 'Med Rush Yds', 'Total Pass Att']
            st.dataframe(display_df.sort_values('Med Rush Yds', ascending=False), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error generating chart: {e}")


def render_team_ppg_home_away_chart(season: Optional[int], week: Optional[int]):
    """Chart: NFL Teams PPG at Home vs PPG Away."""
    st.subheader("🏠 Team PPG Home vs Away")
    st.markdown("*Shows team performance based on location. Teams above the diagonal perform better at home.*")

    if not season:
        st.warning("No season data available.")
        return

    # Build query to get team scoring by location
    week_filter = f"AND week <= {week}" if week else ""

    sql_teams = f"""
    SELECT
        team,
        location,
        AVG(avg_ppg) as avg_ppg
    FROM (
        SELECT
            home_team_abbr as team,
            'home' as location,
            home_score as avg_ppg
        FROM games
        WHERE season = ?
        {week_filter}
        AND home_score IS NOT NULL

        UNION ALL

        SELECT
            away_team_abbr as team,
            'away' as location,
            away_score as avg_ppg
        FROM games
        WHERE season = ?
        {week_filter}
        AND away_score IS NOT NULL
    )
    GROUP BY team, location
    """

    try:
        df = query(sql_teams, (season, season))

        if df.empty:
            st.info("No team scoring data available.")
            return

        # Pivot to get home and away PPG for each team
        pivot_df = df.pivot(index='team', columns='location', values='avg_ppg').reset_index()

        # Ensure both home and away columns exist
        if 'home' not in pivot_df.columns or 'away' not in pivot_df.columns:
            st.info("Insufficient home/away data for teams.")
            return

        pivot_df = pivot_df.dropna()

        if pivot_df.empty:
            st.info("No complete home/away data available.")
            return

        # Calculate difference for color coding
        pivot_df['home_advantage'] = pivot_df['home'] - pivot_df['away']

        # Create scatter plot with invisible markers for hover functionality
        fig = go.Figure()

        # Add invisible markers for hover functionality
        fig.add_trace(go.Scatter(
            x=pivot_df['away'],
            y=pivot_df['home'],
            mode='markers',
            marker=dict(
                size=1,
                color=pivot_df['home_advantage'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Home Adv"),
                opacity=0
            ),
            text=pivot_df['team'],
            customdata=pivot_df[['team', 'home_advantage']],
            hovertemplate='<b>%{customdata[0]}</b><br>' +
                         'PPG Away: %{x:.1f}<br>' +
                         'PPG Home: %{y:.1f}<br>' +
                         'Home Advantage: %{customdata[1]:.1f}<br>' +
                         '<extra></extra>',
            showlegend=False
        ))

        # Add diagonal line (equal performance)
        max_ppg = max(pivot_df['home'].max(), pivot_df['away'].max())
        min_ppg = min(pivot_df['home'].min(), pivot_df['away'].min())
        fig.add_trace(go.Scatter(
            x=[min_ppg, max_ppg],
            y=[min_ppg, max_ppg],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            showlegend=False,
            name='Equal Performance'
        ))

        # Build layout with team logo images
        layout_images = []
        for idx, row in pivot_df.iterrows():
            layout_images.append(dict(
                source=get_team_logo_url(row['team']),
                xref="x",
                yref="y",
                x=row['away'],
                y=row['home'],
                sizex=2.0,  # 2.0 PPG wide
                sizey=2.0,  # 2.0 PPG tall
                xanchor="center",
                yanchor="middle",
                layer="above",
                opacity=0.9
            ))

        fig.update_layout(
            title=f"Team Points Per Game: Home vs Away ({season} Season)",
            xaxis_title="PPG Away",
            yaxis_title="PPG Home",
            height=600,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            images=layout_images
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        with st.expander("📋 View Team Stats Table"):
            display_df = pivot_df[['team', 'home', 'away', 'home_advantage']].copy()
            display_df.columns = ['Team', 'PPG Home', 'PPG Away', 'Home Advantage']
            display_df = display_df.round(1)
            st.dataframe(display_df.sort_values('Home Advantage', ascending=False), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error generating chart: {e}")


def render_team_plays_ppg_chart(season: Optional[int], week: Optional[int]):
    """Chart: Team Plays per Game vs Points per Game - Offensive Efficiency."""
    st.subheader("🏈 Team Plays per Game vs Points per Game")
    st.markdown("*Shows offensive efficiency. Teams higher and to the left score more points with fewer plays (more efficient).*")

    if not season:
        st.warning("No season data available.")
        return

    # Build query with week filter
    week_filter = f"AND week <= {week}" if week else ""

    sql_query = f"""
    SELECT
        team_abbr,
        COUNT(DISTINCT game_id) as games,
        AVG(pass_att + rush_att) as avg_plays,
        AVG(points) as avg_points,
        SUM(points) as total_points
    FROM team_game_summary
    WHERE season = ?
    {week_filter}
    AND pass_att IS NOT NULL
    AND rush_att IS NOT NULL
    GROUP BY team_abbr
    HAVING games > 0
    ORDER BY avg_points DESC
    """

    try:
        df = query(sql_query, (season,))

        if df.empty:
            st.info("No team data available.")
            return

        # Calculate efficiency score (points per play)
        df['points_per_play'] = df['avg_points'] / df['avg_plays']

        # Create scatter plot with invisible markers for hover functionality
        fig = go.Figure()

        # Add invisible markers for hover functionality
        fig.add_trace(go.Scatter(
            x=df['avg_plays'],
            y=df['avg_points'],
            mode='markers',
            marker=dict(
                size=1,
                color=df['points_per_play'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Points/Play"),
                opacity=0
            ),
            text=df['team_abbr'],
            customdata=df[['team_abbr', 'games', 'points_per_play']],
            hovertemplate='<b>%{customdata[0]}</b><br>' +
                         'Plays/Game: %{x:.1f}<br>' +
                         'Points/Game: %{y:.1f}<br>' +
                         'Points/Play: %{customdata[2]:.3f}<br>' +
                         'Games: %{customdata[1]}<br>' +
                         '<extra></extra>',
            showlegend=False
        ))

        # Add league average lines
        avg_plays = df['avg_plays'].mean()
        avg_points = df['avg_points'].mean()

        fig.add_hline(
            y=avg_points,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text=f"League Avg PPG: {avg_points:.1f}",
            annotation_position="right"
        )
        fig.add_vline(
            x=avg_plays,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text=f"League Avg Plays: {avg_plays:.1f}",
            annotation_position="top"
        )

        # Build layout with team logo images
        layout_images = []
        for idx, row in df.iterrows():
            layout_images.append(dict(
                source=get_team_logo_url(row['team_abbr']),
                xref="x",
                yref="y",
                x=row['avg_plays'],
                y=row['avg_points'],
                sizex=2.5,  # 2.5 plays wide
                sizey=2.5,  # 2.5 points tall
                xanchor="center",
                yanchor="middle",
                layer="above",
                opacity=0.9
            ))

        fig.update_layout(
            title=f"Plays per Game vs Points per Game ({season} Season)",
            xaxis_title="Plays per Game",
            yaxis_title="Points per Game",
            height=600,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            images=layout_images
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        with st.expander("📋 View Team Stats Table"):
            display_df = df[['team_abbr', 'games', 'avg_plays', 'avg_points', 'points_per_play']].copy()
            display_df.columns = ['Team', 'Games', 'Plays/Game', 'PPG', 'Points/Play']
            display_df['Plays/Game'] = display_df['Plays/Game'].round(1)
            display_df['PPG'] = display_df['PPG'].round(1)
            display_df['Points/Play'] = display_df['Points/Play'].round(3)
            st.dataframe(display_df.sort_values('Points/Play', ascending=False), use_container_width=True, hide_index=True)

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
    """Chart: Skill Player Avg Yards (Rush + Rec) vs Avg Touches (Carries + Receptions) per Game."""
    st.subheader("💨 Skill Player Yards vs Touches per Game")
    st.markdown("*Shows offensive involvement and production per game (normalized for games played). X-axis = Avg Yards/Game (Rush + Rec), Y-axis = Avg Touches/Game (Carries + Receptions).*")

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

        # Calculate per-game averages
        df['avg_touches_per_game'] = (df['total_touches'] / df['games']).round(1)
        df['avg_yards_per_game'] = (df['total_yards'] / df['games']).round(1)
        df['avg_tds_per_game'] = (df['total_tds'] / df['games']).round(2)

        # Avoid division by zero
        df['yards_per_touch'] = df.apply(
            lambda row: round(row['total_yards'] / row['total_touches'], 2) if row['total_touches'] > 0 else 0,
            axis=1
        )

        # Filter by minimum touches
        chart_df = df[df['total_touches'] >= min_touches].copy()

        if not chart_df.empty:
            import plotly.express as px

            # X = Avg Yards/Game, Y = Avg Touches/Game, Size = Avg TDs/Game
            fig = px.scatter(
                chart_df,
                x='avg_yards_per_game',
                y='avg_touches_per_game',
                size='avg_tds_per_game',
                hover_data=['player', 'team', 'yards_per_touch', 'games', 'total_yards', 'total_touches', 'total_tds'],
                text='player',
                color='yards_per_touch',
                labels={
                    'avg_yards_per_game': 'Avg Yards/Game (Rush + Rec)',
                    'avg_touches_per_game': 'Avg Touches/Game (Carries + Receptions)',
                    'yards_per_touch': 'Yards/Touch',
                    'avg_tds_per_game': 'Avg TDs/Game'
                },
                title=f"Skill Player Yards vs Touches per Game ({season})",
                color_continuous_scale='Viridis',
                size_max=30
            )
            fig.update_traces(textposition='top center')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Show data table
            with st.expander("📋 View Skill Player Stats Table"):
                display_df = chart_df[['player', 'team', 'games', 'avg_touches_per_game', 'avg_yards_per_game',
                                       'avg_tds_per_game', 'yards_per_touch', 'total_touches',
                                       'total_yards', 'total_tds']].copy()
                display_df.columns = ['Player', 'Team', 'Games', 'Touches/G', 'Yards/G',
                                     'TDs/G', 'Yds/Touch', 'Total Touches', 'Total Yds', 'Total TDs']
                st.dataframe(display_df.sort_values('Yards/G', ascending=False), use_container_width=True, hide_index=True)
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

    elif view == "Historical Trends":
        render_historical_trends()

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

    elif view == "Projection Analytics":
        render_projection_analytics(season, week)

    elif view == "Database Manager":
        render_database_manager()

    elif view == "Transaction Manager":
        render_transaction_manager(season, week)


if __name__ == "__main__":
    main()
