"""
Projection Snapshot Manager for NFL App

Handles saving, loading, and accuracy tracking of projection snapshots.
Stores full matchup projections in GCS and tracks accuracy against actual results.

Author: NFL Fantasy App
"""

import json
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from google.cloud import storage
from google.oauth2 import service_account


class ProjectionSnapshotManager:
    """Manages projection snapshots in Google Cloud Storage and database."""

    def __init__(self, db_path: str, bucket_name: str, service_account_dict: dict):
        """
        Initialize the projection snapshot manager.

        Args:
            db_path: Path to the SQLite database
            bucket_name: GCS bucket name
            service_account_dict: GCS service account credentials
        """
        self.db_path = db_path
        self.bucket_name = bucket_name

        # Initialize GCS client
        try:
            credentials = service_account.Credentials.from_service_account_info(
                service_account_dict
            )
            self.gcs_client = storage.Client(credentials=credentials)
            self.bucket = self.gcs_client.bucket(bucket_name)
            logging.info(f"ProjectionSnapshotManager initialized with bucket: {bucket_name}")
        except Exception as e:
            logging.error(f"Failed to initialize GCS client: {e}")
            self.gcs_client = None
            self.bucket = None

    def create_snapshot(
        self,
        season: int,
        week: int,
        home_team: str,
        away_team: str,
        settings: dict,
        team_projections: dict,
        player_projections: dict,
        qb_projections: dict,
        injury_context: dict
    ) -> Tuple[bool, str]:
        """
        Create a projection snapshot and save to GCS and database.

        Args:
            season: NFL season
            week: Week number
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            settings: Projection settings dict {strategy, high_pace, vegas_total, spread_line}
            team_projections: Dict with team abbreviations as keys, TeamProjection objects as values
            player_projections: Dict with team abbreviations as keys, lists of PlayerProjection objects as values
            qb_projections: Dict with team abbreviations as keys, QBProjection objects as values
            injury_context: Dict with team abbreviations as keys, lists of injured player names as values

        Returns:
            Tuple of (success: bool, message: str or snapshot_id: str)
        """
        if not self.gcs_client:
            return False, "GCS client not initialized"

        try:
            # Generate unique snapshot ID with timestamp
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            snapshot_id = f"{season}-W{week:02d}-{away_team}-vs-{home_team}-{timestamp}"

            # Build comprehensive JSON structure
            snapshot_data = {
                "snapshot_id": snapshot_id,
                "created_at": datetime.now().isoformat(),
                "season": season,
                "week": week,
                "matchup": {
                    "home_team": home_team,
                    "away_team": away_team
                },
                "projection_settings": settings,
                "injury_context": injury_context,
                "team_projections": {},
                "player_projections": {
                    "QB": [],
                    "RB": [],
                    "WR": [],
                    "TE": []
                }
            }

            # Serialize team projections
            for team_abbr, team_proj in team_projections.items():
                snapshot_data["team_projections"][team_abbr] = {
                    "total_plays": team_proj.total_plays,
                    "pass_attempts": team_proj.pass_attempts,
                    "rush_attempts": team_proj.rush_attempts,
                    "pass_rate": team_proj.pass_rate,
                    "rush_yards_anchor": team_proj.rush_yards_anchor,
                    "pass_yards_anchor": team_proj.pass_yards_anchor,
                    "total_yards_anchor": team_proj.total_yards_anchor,
                    "is_home": team_abbr == home_team
                }

            # Serialize QB projections
            for team_abbr, qb_proj in qb_projections.items():
                if qb_proj is not None:
                    snapshot_data["player_projections"]["QB"].append({
                        "player_name": qb_proj.player_name,
                        "team": team_abbr,
                        "opponent": away_team if team_abbr == home_team else home_team,
                        "position": "QB",
                        "is_backup": getattr(qb_proj, 'is_backup', False),
                        "replaced_qb": getattr(qb_proj, 'replaced_qb', ""),
                        "projected_pass_att": qb_proj.projected_pass_att,
                        "projected_completions": qb_proj.projected_completions,
                        "projected_pass_yards": qb_proj.projected_pass_yards,
                        "projected_pass_tds": qb_proj.projected_pass_tds,
                        "projected_interceptions": qb_proj.projected_interceptions,
                        "projected_carries": qb_proj.projected_carries,
                        "projected_rush_yards": qb_proj.projected_rush_yards,
                        "p10_pass_yards": getattr(qb_proj, 'projected_pass_yards_p10', 0),
                        "p90_pass_yards": getattr(qb_proj, 'projected_pass_yards_p90', 0)
                    })

            # Serialize player projections
            for team_abbr, players in player_projections.items():
                opponent = away_team if team_abbr == home_team else home_team

                for player in players:
                    position = player.position
                    player_data = {
                        "player_name": player.player_name,
                        "team": team_abbr,
                        "opponent": opponent,
                        "position": position,
                        "injured": getattr(player, 'injured', False),
                        "projected_carries": player.projected_carries,
                        "projected_targets": player.projected_targets,
                        "projected_receptions": player.projected_receptions,
                        "projected_rush_yards": player.projected_rush_yards,
                        "projected_recv_yards": player.projected_recv_yards,
                        "projected_total_yards": player.projected_total_yards,
                        "p10_total_yards": player.projected_total_yards_p10,
                        "p90_total_yards": player.projected_total_yards_p90,
                        "dvoa_pct": player.dvoa_pct
                    }

                    # Add to appropriate position list
                    if position in ["WR", "TE"]:
                        snapshot_data["player_projections"][position].append(player_data)
                    elif position == "RB":
                        snapshot_data["player_projections"]["RB"].append(player_data)

            # Save to GCS
            gcs_path = f"projection-snapshots/{season}/week-{week:02d}-{away_team}-vs-{home_team}-{timestamp}.json"
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_string(
                data=json.dumps(snapshot_data, indent=2),
                content_type='application/json'
            )

            # Save metadata to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # First, check for existing conflicting records
            cursor.execute("""
                SELECT player_name, team_abbr, opponent_abbr, position
                FROM projection_accuracy
                WHERE season = ? AND week = ?
                AND (team_abbr = ? OR team_abbr = ?)
            """, (season, week, home_team, away_team))
            existing_records = cursor.fetchall()

            if existing_records:
                logging.info(f"Found {len(existing_records)} existing projection records for Week {week} {away_team}/{home_team}")
                for rec in existing_records[:5]:  # Log first 5
                    logging.info(f"  Existing: {rec[0]} ({rec[1]} vs {rec[2]}) - {rec[3]}")

            # Delete ALL projections for these teams in this week (not just specific matchup)
            # This is needed because UNIQUE constraint doesn't include opponent_abbr
            cursor.execute("""
                DELETE FROM projection_accuracy
                WHERE season = ? AND week = ?
                AND (team_abbr = ? OR team_abbr = ?)
            """, (season, week, home_team, away_team))

            deleted_count = cursor.rowcount
            logging.info(f"Deleted {deleted_count} existing projections for {away_team}/{home_team} Week {week}")

            # COMMIT the DELETE before INSERT to ensure it's not rolled back on error
            conn.commit()

            cursor.execute("""
                INSERT INTO projection_snapshots (
                    snapshot_id, season, week, home_team, away_team,
                    gcs_path, projection_settings, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot_id, season, week, home_team, away_team,
                gcs_path, json.dumps(settings), 'pending'
            ))

            # Save team projections to database
            for team_abbr, team_proj in team_projections.items():
                cursor.execute("""
                    INSERT INTO team_projection_accuracy (
                        snapshot_id, team_abbr, projected_plays, projected_total_yards
                    )
                    VALUES (?, ?, ?, ?)
                """, (
                    snapshot_id, team_abbr,
                    team_proj.total_plays,
                    int(team_proj.total_yards_anchor)
                ))

            # Save player projections to projection_accuracy table
            # QB projections
            for team_abbr, qb_proj in qb_projections.items():
                if qb_proj is not None:
                    opponent = away_team if team_abbr == home_team else home_team
                    total_yards = qb_proj.projected_pass_yards + qb_proj.projected_rush_yards

                    try:
                        # Extract QB rushing attempts (using 'projected_carries' attribute)
                        try:
                            rush_attempts = int(qb_proj.projected_carries)
                        except AttributeError as attr_err:
                            # Diagnostic: Show available attributes if error occurs
                            available_attrs = [attr for attr in dir(qb_proj) if not attr.startswith('_')]
                            error_msg = f"QB object missing 'projected_carries'. Available attributes: {available_attrs}"
                            logging.error(error_msg)
                            raise Exception(error_msg)

                        cursor.execute("""
                            INSERT INTO projection_accuracy (
                                player_name, team_abbr, opponent_abbr, season, week,
                                position, projected_yds, snapshot_id,
                                projected_pass_att, projected_completions, projected_pass_yds,
                                projected_pass_tds, projected_interceptions,
                                projected_rush_att, projected_rush_yds, projected_rush_tds
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            qb_proj.player_name, team_abbr, opponent, season, week,
                            "QB", total_yards, snapshot_id,
                            qb_proj.projected_pass_att, qb_proj.projected_completions,
                            qb_proj.projected_pass_yards, qb_proj.projected_pass_tds,
                            qb_proj.projected_interceptions,
                            rush_attempts, qb_proj.projected_rush_yards,
                            qb_proj.projected_rush_tds
                        ))
                    except sqlite3.IntegrityError as e:
                        error_msg = f"UNIQUE constraint for QB {qb_proj.player_name} ({team_abbr}, S{season} W{week}): {e}"
                        logging.error(error_msg)
                        raise Exception(error_msg)

            # Player projections (RB, WR, TE)
            for team_abbr, players in player_projections.items():
                opponent = away_team if team_abbr == home_team else home_team

                # Deduplicate players by (player_name, position) - keep first occurrence
                seen_players = set()
                unique_players = []
                duplicates_found = []

                for player in players:
                    if not getattr(player, 'injured', False):  # Don't save injured players
                        player_key = (player.player_name, player.position)
                        if player_key not in seen_players:
                            seen_players.add(player_key)
                            unique_players.append(player)
                        else:
                            duplicates_found.append(f"{player.position} {player.player_name}")

                if duplicates_found:
                    logging.warning(f"Found {len(duplicates_found)} duplicate players in {team_abbr} projections: {', '.join(duplicates_found[:5])}")

                # Insert deduplicated players
                for player in unique_players:
                    try:
                        cursor.execute("""
                            INSERT INTO projection_accuracy (
                                player_name, team_abbr, opponent_abbr, season, week,
                                position, projected_yds, snapshot_id, in_range,
                                projected_rush_att, projected_rush_yds,
                                projected_targets, projected_receptions, projected_rec_yds
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            player.player_name, team_abbr, opponent, season, week,
                            player.position, player.projected_total_yards,
                            snapshot_id, None,  # Will be calculated when actuals are loaded
                            int(player.projected_carries) if player.projected_carries else None,
                            player.projected_rush_yards if player.projected_rush_yards else None,
                            player.projected_targets,
                            player.projected_receptions,
                            player.projected_recv_yards
                        ))
                    except sqlite3.IntegrityError as e:
                        error_msg = f"UNIQUE constraint for {player.position} {player.player_name} ({team_abbr}, S{season} W{week}): {e}"
                        logging.error(error_msg)
                        raise Exception(error_msg)

            conn.commit()
            conn.close()

            logging.info(f"Projection snapshot created: {snapshot_id}")
            return True, snapshot_id

        except Exception as e:
            error_msg = f"Failed to create projection snapshot: {e}"
            logging.error(error_msg)
            return False, error_msg

    def list_snapshots(
        self,
        season: Optional[int] = None,
        week: Optional[int] = None,
        status: Optional[str] = None
    ) -> pd.DataFrame:
        """
        List all projection snapshots with optional filters.

        Args:
            season: Filter by season
            week: Filter by week
            status: Filter by status ('pending' or 'completed')

        Returns:
            DataFrame with snapshot metadata
        """
        try:
            conn = sqlite3.connect(self.db_path)

            query = "SELECT * FROM projection_snapshots WHERE 1=1"
            params = []

            if season is not None:
                query += " AND season = ?"
                params.append(season)

            if week is not None:
                query += " AND week = ?"
                params.append(week)

            if status is not None:
                query += " AND status = ?"
                params.append(status)

            query += " ORDER BY created_at DESC"

            df = pd.read_sql_query(query, conn, params=params if params else None)
            conn.close()

            return df

        except Exception as e:
            logging.error(f"Error listing snapshots: {e}")
            return pd.DataFrame()

    def delete_snapshots_by_week(self, season: int, week: int) -> Tuple[bool, int]:
        """
        Delete all projection snapshots for a given season/week.

        Removes snapshots from both database and GCS to allow fresh snapshots
        to be created (e.g., after injury updates).

        Args:
            season: Season year
            week: Week number

        Returns:
            Tuple of (success, count_deleted)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all snapshots for this week (to get GCS paths)
            cursor.execute("""
                SELECT snapshot_id, gcs_path
                FROM projection_snapshots
                WHERE season = ? AND week = ?
            """, (season, week))
            snapshots = cursor.fetchall()

            if not snapshots:
                conn.close()
                return True, 0

            # Delete from GCS
            deleted_gcs = 0
            if self.gcs_client:
                for snapshot_id, gcs_path in snapshots:
                    try:
                        blob = self.bucket.blob(gcs_path)
                        if blob.exists():
                            blob.delete()
                            deleted_gcs += 1
                            logging.info(f"Deleted GCS snapshot: {gcs_path}")
                    except Exception as e:
                        logging.warning(f"Failed to delete GCS blob {gcs_path}: {e}")

            # Delete from projection_accuracy table
            cursor.execute("""
                DELETE FROM projection_accuracy
                WHERE season = ? AND week = ?
            """, (season, week))

            # Delete from team_projection_accuracy table
            cursor.execute("""
                DELETE FROM team_projection_accuracy
                WHERE snapshot_id IN (
                    SELECT snapshot_id FROM projection_snapshots
                    WHERE season = ? AND week = ?
                )
            """, (season, week))

            # Delete from projection_snapshots table
            cursor.execute("""
                DELETE FROM projection_snapshots
                WHERE season = ? AND week = ?
            """, (season, week))

            conn.commit()
            conn.close()

            count_deleted = len(snapshots)
            logging.info(f"Deleted {count_deleted} snapshots for Season {season}, Week {week} ({deleted_gcs} from GCS)")
            return True, count_deleted

        except Exception as e:
            error_msg = f"Error deleting snapshots for S{season} W{week}: {e}"
            logging.error(error_msg)
            return False, 0

    def load_snapshot(self, snapshot_id: str) -> Optional[dict]:
        """
        Load a projection snapshot from GCS.

        Args:
            snapshot_id: Unique snapshot identifier

        Returns:
            Snapshot data dictionary or None if not found
        """
        if not self.gcs_client:
            return None

        try:
            # Get GCS path from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT gcs_path FROM projection_snapshots WHERE snapshot_id = ?", (snapshot_id,))
            result = cursor.fetchone()
            conn.close()

            if not result:
                logging.warning(f"Snapshot not found in database: {snapshot_id}")
                return None

            gcs_path = result[0]

            # Download from GCS
            blob = self.bucket.blob(gcs_path)

            if not blob.exists():
                logging.warning(f"Snapshot not found in GCS: {gcs_path}")
                return None

            snapshot_json = blob.download_as_text()
            snapshot = json.loads(snapshot_json)

            logging.info(f"Loaded snapshot: {snapshot_id}")
            return snapshot

        except Exception as e:
            logging.error(f"Error loading snapshot: {e}")
            return None

    def update_actuals(self, snapshot_id: str) -> Tuple[bool, int]:
        """
        Update snapshot with actual game results from player_stats table.

        Args:
            snapshot_id: Snapshot to update

        Returns:
            Tuple of (success: bool, players_updated_count: int)
        """
        try:
            # Load snapshot
            snapshot = self.load_snapshot(snapshot_id)
            if not snapshot:
                return False, 0

            season = snapshot['season']
            week = snapshot['week']

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            players_updated = 0
            players_checked = 0
            players_not_found = []

            # Update QB actuals
            for qb in snapshot['player_projections']['QB']:
                players_checked += 1
                actual_stats = self._get_actual_stats(
                    qb['player_name'],
                    qb['team'],
                    season,
                    week,
                    'QB'
                )

                if actual_stats:
                    # Calculate error
                    projected = qb['projected_pass_yards'] + qb['projected_rush_yards']
                    actual = actual_stats['total_yards']
                    error = actual - projected

                    # Update database with all component stats
                    cursor.execute("""
                        UPDATE projection_accuracy
                        SET actual_yds = ?, variance = ?, game_played = ?,
                            actual_pass_att = ?, actual_completions = ?, actual_pass_yds = ?,
                            actual_pass_tds = ?, actual_interceptions = ?,
                            actual_rush_att = ?, actual_rush_yds = ?
                        WHERE snapshot_id = ? AND player_name = ? AND team_abbr = ?
                    """, (
                        actual, error, actual_stats['game_played'],
                        actual_stats.get('pass_att'), actual_stats.get('completions'), actual_stats.get('pass_yds'),
                        actual_stats.get('pass_tds'), actual_stats.get('interceptions'),
                        actual_stats.get('rush_att'), actual_stats.get('rush_yds'),
                        snapshot_id, qb['player_name'], qb['team']
                    ))
                    players_updated += 1
                else:
                    players_not_found.append(f"QB {qb['player_name']} ({qb['team']})")

            # Update player actuals (RB, WR, TE)
            for position in ['RB', 'WR', 'TE']:
                for player in snapshot['player_projections'][position]:
                    players_checked += 1
                    actual_stats = self._get_actual_stats(
                        player['player_name'],
                        player['team'],
                        season,
                        week,
                        position
                    )

                    if actual_stats:
                        projected = player['projected_total_yards']
                        actual = actual_stats['total_yards']
                        error = actual - projected

                        # Check if actual is within projected range (P10-P90)
                        in_range = player['p10_total_yards'] <= actual <= player['p90_total_yards']

                        cursor.execute("""
                            UPDATE projection_accuracy
                            SET actual_yds = ?, variance = ?, in_range = ?, game_played = ?,
                                actual_rush_att = ?, actual_rush_yds = ?,
                                actual_targets = ?, actual_receptions = ?, actual_rec_yds = ?
                            WHERE snapshot_id = ? AND player_name = ? AND team_abbr = ?
                        """, (
                            actual, error, in_range, actual_stats['game_played'],
                            actual_stats.get('rush_att'), actual_stats.get('rush_yds'),
                            actual_stats.get('targets'), actual_stats.get('receptions'), actual_stats.get('rec_yds'),
                            snapshot_id, player['player_name'], player['team']
                        ))
                        players_updated += 1
                    else:
                        players_not_found.append(f"{position} {player['player_name']} ({player['team']})")

            # Log diagnostic info
            logging.info(f"Update actuals summary for {snapshot_id}:")
            logging.info(f"  Checked: {players_checked} players")
            logging.info(f"  Found: {players_updated} players")
            logging.info(f"  Not found: {len(players_not_found)} players")
            if players_not_found and len(players_not_found) <= 10:
                logging.info(f"  Missing players: {', '.join(players_not_found)}")

            # Only mark snapshot as completed if we actually updated some players
            if players_updated > 0:
                cursor.execute("""
                    UPDATE projection_snapshots
                    SET status = 'completed', game_completed = TRUE
                    WHERE snapshot_id = ?
                """, (snapshot_id,))
                status_msg = f"Updated {players_updated} players - marked as completed"
            else:
                # Keep as pending if no stats were found
                status_msg = f"No player stats found in database yet - keeping as pending"

            conn.commit()
            conn.close()

            logging.info(f"{status_msg} for snapshot {snapshot_id}")
            return True, players_updated

        except Exception as e:
            logging.error(f"Error updating actuals: {e}")
            return False, 0

    def _get_actual_stats(
        self,
        player_name: str,
        team: str,
        season: int,
        week: int,
        position: str
    ) -> Optional[dict]:
        """
        Fetch actual stats from player_stats table.

        Args:
            player_name: Player name
            team: Team abbreviation
            season: Season
            week: Week
            position: Position (QB, RB, WR, TE)

        Returns:
            Dict with total_yards and game_played, or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)

            # Fetch all stats regardless of position
            query = """
                SELECT
                    -- Totals
                    COALESCE(passing_yards, 0) + COALESCE(rushing_yards, 0) + COALESCE(receiving_yards, 0) as total_yards,

                    -- Passing stats
                    attempts as pass_att,
                    completions,
                    passing_yards as pass_yds,
                    passing_tds as pass_tds,
                    interceptions,

                    -- Rushing stats
                    carries as rush_att,
                    rushing_yards as rush_yds,
                    rushing_tds as rush_tds,

                    -- Receiving stats
                    targets,
                    receptions,
                    receiving_yards as rec_yds,
                    receiving_tds as rec_tds

                FROM player_stats
                WHERE player_display_name = ? AND season = ? AND week = ?
                LIMIT 1
            """

            result = pd.read_sql_query(query, conn, params=(player_name, season, week))
            conn.close()

            if result.empty:
                # Player not found - DNP or missing data
                return None

            row = result.iloc[0]

            # Position-specific total yards calculation
            if position == 'QB':
                total_yards = (row['pass_yds'] or 0) + (row['rush_yds'] or 0)
                touches = (row['pass_att'] or 0) + (row['rush_att'] or 0)
            elif position == 'RB':
                total_yards = (row['rush_yds'] or 0) + (row['rec_yds'] or 0)
                touches = (row['rush_att'] or 0) + (row['targets'] or 0)
            else:  # WR, TE
                total_yards = row['rec_yds'] or 0
                touches = row['targets'] or 0

            # Determine if player actually played
            game_played = touches > 0 or total_yards > 0

            return {
                "total_yards": total_yards,
                "game_played": game_played,
                # Passing stats
                "pass_att": row['pass_att'],
                "completions": row['completions'],
                "pass_yds": row['pass_yds'],
                "pass_tds": row['pass_tds'],
                "interceptions": row['interceptions'],
                # Rushing stats
                "rush_att": row['rush_att'],
                "rush_yds": row['rush_yds'],
                "rush_tds": row['rush_tds'],
                # Receiving stats
                "targets": row['targets'],
                "receptions": row['receptions'],
                "rec_yds": row['rec_yds'],
                "rec_tds": row['rec_tds']
            }

        except Exception as e:
            logging.error(f"Error fetching actual stats for {player_name}: {e}")
            return None

    def calculate_accuracy_metrics(
        self,
        season: Optional[int] = None,
        week: Optional[int] = None,
        position: Optional[str] = None
    ) -> dict:
        """
        Calculate comprehensive accuracy metrics for all component stats.

        Args:
            season: Filter by season
            week: Filter by week
            position: Filter by position (if None, calculates for all positions)

        Returns:
            Dict with nested structure:
            {
                'overall': {
                    'total_yds': {'mae': X, 'rmse': X, 'bias': X, 'hit_rate_20': X},
                    'count': X
                },
                'by_position': {
                    'QB': {'passing': {...}, 'rushing': {...}, 'count': X},
                    'RB': {'rushing': {...}, 'receiving': {...}, 'count': X},
                    ...
                }
            }
        """
        try:
            conn = sqlite3.connect(self.db_path)

            # Build base WHERE clause
            where_conditions = ["game_played = TRUE", "actual_yds IS NOT NULL"]
            params = []

            if season is not None:
                where_conditions.append("season = ?")
                params.append(season)

            if week is not None:
                where_conditions.append("week = ?")
                params.append(week)

            if position is not None:
                where_conditions.append("position = ?")
                params.append(position)

            where_clause = " AND ".join(where_conditions)

            # ==================================================================
            # OVERALL METRICS (Total Yards - backwards compatible)
            # ==================================================================
            overall_query = f"""
                SELECT
                    projected_yds,
                    actual_yds,
                    variance,
                    in_range
                FROM projection_accuracy
                WHERE {where_clause}
            """

            df_overall = pd.read_sql_query(overall_query, conn, params=params if params else None)

            if df_overall.empty:
                conn.close()
                return {
                    "overall": {
                        "total_yds": {
                            "mae": 0, "rmse": 0, "bias": 0,
                            "hit_rate_10": 0, "hit_rate_20": 0, "hit_rate_30": 0,
                            "range_accuracy": 0
                        },
                        "count": 0
                    },
                    "by_position": {}
                }

            # Calculate overall total yards metrics
            errors = df_overall['variance'].abs()
            overall_metrics = {
                "overall": {
                    "total_yds": {
                        "mae": float(errors.mean()),
                        "rmse": float(np.sqrt((df_overall['variance'] ** 2).mean())),
                        "bias": float(df_overall['variance'].mean()),
                        "hit_rate_10": float((errors <= 10).sum() / len(errors) * 100),
                        "hit_rate_20": float((errors <= 20).sum() / len(errors) * 100),
                        "hit_rate_30": float((errors <= 30).sum() / len(errors) * 100),
                        "range_accuracy": float(df_overall['in_range'].sum() / len(df_overall) * 100) if 'in_range' in df_overall.columns else 0
                    },
                    "count": len(df_overall)
                },
                "by_position": {}
            }

            # ==================================================================
            # POSITION-SPECIFIC COMPONENT METRICS
            # ==================================================================

            # --- QB METRICS ---
            qb_where = where_clause + " AND position = 'QB' AND actual_pass_att IS NOT NULL"
            qb_params = params.copy()

            qb_query = f"""
                SELECT
                    -- Passing stats
                    AVG(ABS(actual_pass_att - projected_pass_att)) as pass_att_mae,
                    AVG(actual_pass_att - projected_pass_att) as pass_att_bias,
                    AVG(ABS(actual_completions - projected_completions)) as completions_mae,
                    AVG(actual_completions - projected_completions) as completions_bias,
                    AVG(ABS(actual_pass_yds - projected_pass_yds)) as pass_yds_mae,
                    AVG(actual_pass_yds - projected_pass_yds) as pass_yds_bias,
                    AVG(ABS(actual_pass_tds - projected_pass_tds)) as pass_tds_mae,
                    AVG(actual_pass_tds - projected_pass_tds) as pass_tds_bias,
                    AVG(ABS(actual_interceptions - projected_interceptions)) as int_mae,
                    AVG(actual_interceptions - projected_interceptions) as int_bias,

                    -- Rushing stats
                    AVG(ABS(actual_rush_att - projected_rush_att)) as rush_att_mae,
                    AVG(actual_rush_att - projected_rush_att) as rush_att_bias,
                    AVG(ABS(actual_rush_yds - projected_rush_yds)) as rush_yds_mae,
                    AVG(actual_rush_yds - projected_rush_yds) as rush_yds_bias,
                    AVG(ABS(actual_rush_tds - projected_rush_tds)) as rush_tds_mae,
                    AVG(actual_rush_tds - projected_rush_tds) as rush_tds_bias,

                    COUNT(*) as count
                FROM projection_accuracy
                WHERE {qb_where}
            """

            qb_result = pd.read_sql_query(qb_query, conn, params=qb_params if qb_params else None)

            if not qb_result.empty and qb_result['count'].iloc[0] > 0:
                row = qb_result.iloc[0]
                overall_metrics['by_position']['QB'] = {
                    'passing': {
                        'pass_att': {'mae': float(row['pass_att_mae'] or 0), 'bias': float(row['pass_att_bias'] or 0)},
                        'completions': {'mae': float(row['completions_mae'] or 0), 'bias': float(row['completions_bias'] or 0)},
                        'pass_yds': {'mae': float(row['pass_yds_mae'] or 0), 'bias': float(row['pass_yds_bias'] or 0)},
                        'pass_tds': {'mae': float(row['pass_tds_mae'] or 0), 'bias': float(row['pass_tds_bias'] or 0)},
                        'interceptions': {'mae': float(row['int_mae'] or 0), 'bias': float(row['int_bias'] or 0)}
                    },
                    'rushing': {
                        'rush_att': {'mae': float(row['rush_att_mae'] or 0), 'bias': float(row['rush_att_bias'] or 0)},
                        'rush_yds': {'mae': float(row['rush_yds_mae'] or 0), 'bias': float(row['rush_yds_bias'] or 0)},
                        'rush_tds': {'mae': float(row['rush_tds_mae'] or 0), 'bias': float(row['rush_tds_bias'] or 0)}
                    },
                    'count': int(row['count'])
                }

            # --- RB METRICS ---
            rb_where = where_clause + " AND position = 'RB' AND actual_rush_att IS NOT NULL"
            rb_params = params.copy()

            rb_query = f"""
                SELECT
                    -- Rushing stats
                    AVG(ABS(actual_rush_att - projected_rush_att)) as rush_att_mae,
                    AVG(actual_rush_att - projected_rush_att) as rush_att_bias,
                    AVG(ABS(actual_rush_yds - projected_rush_yds)) as rush_yds_mae,
                    AVG(actual_rush_yds - projected_rush_yds) as rush_yds_bias,
                    AVG(ABS(actual_rush_tds - projected_rush_tds)) as rush_tds_mae,
                    AVG(actual_rush_tds - projected_rush_tds) as rush_tds_bias,

                    -- Receiving stats
                    AVG(ABS(actual_targets - projected_targets)) as targets_mae,
                    AVG(actual_targets - projected_targets) as targets_bias,
                    AVG(ABS(actual_receptions - projected_receptions)) as receptions_mae,
                    AVG(actual_receptions - projected_receptions) as receptions_bias,
                    AVG(ABS(actual_rec_yds - projected_rec_yds)) as rec_yds_mae,
                    AVG(actual_rec_yds - projected_rec_yds) as rec_yds_bias,
                    AVG(ABS(actual_rec_tds - projected_rec_tds)) as rec_tds_mae,
                    AVG(actual_rec_tds - projected_rec_tds) as rec_tds_bias,

                    COUNT(*) as count
                FROM projection_accuracy
                WHERE {rb_where}
            """

            rb_result = pd.read_sql_query(rb_query, conn, params=rb_params if rb_params else None)

            if not rb_result.empty and rb_result['count'].iloc[0] > 0:
                row = rb_result.iloc[0]
                overall_metrics['by_position']['RB'] = {
                    'rushing': {
                        'rush_att': {'mae': float(row['rush_att_mae'] or 0), 'bias': float(row['rush_att_bias'] or 0)},
                        'rush_yds': {'mae': float(row['rush_yds_mae'] or 0), 'bias': float(row['rush_yds_bias'] or 0)},
                        'rush_tds': {'mae': float(row['rush_tds_mae'] or 0), 'bias': float(row['rush_tds_bias'] or 0)}
                    },
                    'receiving': {
                        'targets': {'mae': float(row['targets_mae'] or 0), 'bias': float(row['targets_bias'] or 0)},
                        'receptions': {'mae': float(row['receptions_mae'] or 0), 'bias': float(row['receptions_bias'] or 0)},
                        'rec_yds': {'mae': float(row['rec_yds_mae'] or 0), 'bias': float(row['rec_yds_bias'] or 0)},
                        'rec_tds': {'mae': float(row['rec_tds_mae'] or 0), 'bias': float(row['rec_tds_bias'] or 0)}
                    },
                    'count': int(row['count'])
                }

            # --- WR METRICS ---
            wr_where = where_clause + " AND position = 'WR' AND actual_targets IS NOT NULL"
            wr_params = params.copy()

            wr_query = f"""
                SELECT
                    AVG(ABS(actual_targets - projected_targets)) as targets_mae,
                    AVG(actual_targets - projected_targets) as targets_bias,
                    AVG(ABS(actual_receptions - projected_receptions)) as receptions_mae,
                    AVG(actual_receptions - projected_receptions) as receptions_bias,
                    AVG(ABS(actual_rec_yds - projected_rec_yds)) as rec_yds_mae,
                    AVG(actual_rec_yds - projected_rec_yds) as rec_yds_bias,
                    AVG(ABS(actual_rec_tds - projected_rec_tds)) as rec_tds_mae,
                    AVG(actual_rec_tds - projected_rec_tds) as rec_tds_bias,
                    COUNT(*) as count
                FROM projection_accuracy
                WHERE {wr_where}
            """

            wr_result = pd.read_sql_query(wr_query, conn, params=wr_params if wr_params else None)

            if not wr_result.empty and wr_result['count'].iloc[0] > 0:
                row = wr_result.iloc[0]
                overall_metrics['by_position']['WR'] = {
                    'receiving': {
                        'targets': {'mae': float(row['targets_mae'] or 0), 'bias': float(row['targets_bias'] or 0)},
                        'receptions': {'mae': float(row['receptions_mae'] or 0), 'bias': float(row['receptions_bias'] or 0)},
                        'rec_yds': {'mae': float(row['rec_yds_mae'] or 0), 'bias': float(row['rec_yds_bias'] or 0)},
                        'rec_tds': {'mae': float(row['rec_tds_mae'] or 0), 'bias': float(row['rec_tds_bias'] or 0)}
                    },
                    'count': int(row['count'])
                }

            # --- TE METRICS ---
            te_where = where_clause + " AND position = 'TE' AND actual_targets IS NOT NULL"
            te_params = params.copy()

            te_query = f"""
                SELECT
                    AVG(ABS(actual_targets - projected_targets)) as targets_mae,
                    AVG(actual_targets - projected_targets) as targets_bias,
                    AVG(ABS(actual_receptions - projected_receptions)) as receptions_mae,
                    AVG(actual_receptions - projected_receptions) as receptions_bias,
                    AVG(ABS(actual_rec_yds - projected_rec_yds)) as rec_yds_mae,
                    AVG(actual_rec_yds - projected_rec_yds) as rec_yds_bias,
                    AVG(ABS(actual_rec_tds - projected_rec_tds)) as rec_tds_mae,
                    AVG(actual_rec_tds - projected_rec_tds) as rec_tds_bias,
                    COUNT(*) as count
                FROM projection_accuracy
                WHERE {te_where}
            """

            te_result = pd.read_sql_query(te_query, conn, params=te_params if te_params else None)

            if not te_result.empty and te_result['count'].iloc[0] > 0:
                row = te_result.iloc[0]
                overall_metrics['by_position']['TE'] = {
                    'receiving': {
                        'targets': {'mae': float(row['targets_mae'] or 0), 'bias': float(row['targets_bias'] or 0)},
                        'receptions': {'mae': float(row['receptions_mae'] or 0), 'bias': float(row['receptions_bias'] or 0)},
                        'rec_yds': {'mae': float(row['rec_yds_mae'] or 0), 'bias': float(row['rec_yds_bias'] or 0)},
                        'rec_tds': {'mae': float(row['rec_tds_mae'] or 0), 'bias': float(row['rec_tds_bias'] or 0)}
                    },
                    'count': int(row['count'])
                }

            conn.close()
            return overall_metrics

        except Exception as e:
            logging.error(f"Error calculating accuracy metrics: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                "overall": {
                    "total_yds": {
                        "mae": 0, "rmse": 0, "bias": 0,
                        "hit_rate_10": 0, "hit_rate_20": 0, "hit_rate_30": 0,
                        "range_accuracy": 0
                    },
                    "count": 0
                },
                "by_position": {}
            }

    def get_snapshot_summary(self, snapshot_id: str) -> Optional[dict]:
        """
        Get comprehensive summary of a snapshot including accuracy breakdown.

        Args:
            snapshot_id: Snapshot to summarize

        Returns:
            Summary dictionary with metadata, projections, and accuracy
        """
        try:
            snapshot = self.load_snapshot(snapshot_id)
            if not snapshot:
                return None

            # Get accuracy metrics for this specific snapshot
            conn = sqlite3.connect(self.db_path)

            query = """
                SELECT position,
                       AVG(ABS(variance)) as mae,
                       AVG(variance) as bias,
                       COUNT(*) as count
                FROM projection_accuracy
                WHERE snapshot_id = ? AND game_played = TRUE
                GROUP BY position
            """

            accuracy_by_position = pd.read_sql_query(query, conn, params=(snapshot_id,))
            conn.close()

            return {
                "metadata": {
                    "snapshot_id": snapshot_id,
                    "season": snapshot['season'],
                    "week": snapshot['week'],
                    "matchup": snapshot['matchup'],
                    "created_at": snapshot['created_at']
                },
                "accuracy_by_position": accuracy_by_position.to_dict('records'),
                "projection_count": {
                    "QB": len(snapshot['player_projections']['QB']),
                    "RB": len(snapshot['player_projections']['RB']),
                    "WR": len(snapshot['player_projections']['WR']),
                    "TE": len(snapshot['player_projections']['TE'])
                }
            }

        except Exception as e:
            logging.error(f"Error getting snapshot summary: {e}")
            return None
