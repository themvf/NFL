"""
Injury Snapshot Manager for NFL App

Handles weekly snapshots of injury data to Google Cloud Storage.
Provides backup, versioning, and restore capabilities.

Author: NFL Fantasy App
"""

import json
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account


class InjurySnapshotManager:
    """Manages injury data snapshots in Google Cloud Storage."""

    def __init__(self, db_path: str, bucket_name: str, service_account_dict: dict):
        """
        Initialize the snapshot manager.

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
            logging.info(f"InjurySnapshotManager initialized with bucket: {bucket_name}")
        except Exception as e:
            logging.error(f"Failed to initialize GCS client: {e}")
            self.gcs_client = None
            self.bucket = None

    def get_all_injuries(self) -> List[Dict]:
        """
        Fetch all current injuries from the database.

        Returns:
            List of injury dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT
                    injury_id,
                    player_name,
                    team_abbr,
                    season,
                    injury_type,
                    start_week,
                    end_week,
                    injury_description,
                    created_at,
                    updated_at
                FROM player_injuries
                ORDER BY updated_at DESC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()

            # Convert to list of dicts
            injuries = df.to_dict('records')
            return injuries
        except Exception as e:
            logging.error(f"Error fetching injuries: {e}")
            return []

    def create_snapshot(self, season: int, week: int, description: str = "") -> Tuple[bool, str]:
        """
        Create a snapshot of current injury data and upload to GCS.

        Args:
            season: NFL season (e.g., 2025)
            week: Week number (1-18)
            description: Optional description for this snapshot

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.gcs_client:
            return False, "GCS client not initialized"

        try:
            # Get all current injuries
            injuries = self.get_all_injuries()

            # Create snapshot data
            snapshot = {
                "snapshot_date": datetime.now().isoformat(),
                "season": season,
                "week": week,
                "description": description,
                "injuries": injuries,
                "metadata": {
                    "total_injuries": len(injuries),
                    "teams_affected": list(set([inj['team_abbr'] for inj in injuries])),
                    "injury_types": self._count_by_type(injuries)
                }
            }

            # Upload to GCS
            # Path: injury-snapshots/{season}/week-{week}-injuries.json
            blob_path = f"injury-snapshots/{season}/week-{week}-injuries.json"
            blob = self.bucket.blob(blob_path)

            # Upload as JSON
            blob.upload_from_string(
                data=json.dumps(snapshot, indent=2),
                content_type='application/json'
            )

            # Also update "current" snapshot
            current_blob = self.bucket.blob(f"injury-snapshots/{season}/current-injuries.json")
            current_blob.upload_from_string(
                data=json.dumps(snapshot, indent=2),
                content_type='application/json'
            )

            logging.info(f"Snapshot created: {blob_path}")
            return True, f"Snapshot saved to {blob_path}"

        except Exception as e:
            error_msg = f"Failed to create snapshot: {e}"
            logging.error(error_msg)
            return False, error_msg

    def list_snapshots(self, season: int) -> List[Dict]:
        """
        List all available snapshots for a season.

        Args:
            season: NFL season

        Returns:
            List of snapshot metadata dicts
        """
        if not self.gcs_client:
            return []

        try:
            prefix = f"injury-snapshots/{season}/"
            blobs = self.bucket.list_blobs(prefix=prefix)

            snapshots = []
            for blob in blobs:
                # Skip the "current" file
                if "current-injuries.json" in blob.name:
                    continue

                # Extract week number from filename
                filename = blob.name.split('/')[-1]  # e.g., "week-17-injuries.json"
                if filename.startswith("week-") and filename.endswith("-injuries.json"):
                    week_str = filename.replace("week-", "").replace("-injuries.json", "")
                    try:
                        week = int(week_str)
                        snapshots.append({
                            "season": season,
                            "week": week,
                            "blob_name": blob.name,
                            "created": blob.time_created,
                            "size_bytes": blob.size
                        })
                    except ValueError:
                        continue

            # Sort by week
            snapshots.sort(key=lambda x: x['week'])
            return snapshots

        except Exception as e:
            logging.error(f"Error listing snapshots: {e}")
            return []

    def load_snapshot(self, season: int, week: int) -> Optional[Dict]:
        """
        Load a specific snapshot from GCS.

        Args:
            season: NFL season
            week: Week number

        Returns:
            Snapshot dictionary or None if not found
        """
        if not self.gcs_client:
            return None

        try:
            blob_path = f"injury-snapshots/{season}/week-{week}-injuries.json"
            blob = self.bucket.blob(blob_path)

            if not blob.exists():
                logging.warning(f"Snapshot not found: {blob_path}")
                return None

            # Download and parse JSON
            snapshot_json = blob.download_as_text()
            snapshot = json.loads(snapshot_json)

            logging.info(f"Loaded snapshot: {blob_path}")
            return snapshot

        except Exception as e:
            logging.error(f"Error loading snapshot: {e}")
            return None

    def restore_snapshot(self, season: int, week: int) -> Tuple[bool, str]:
        """
        Restore injury data from a snapshot.

        WARNING: This will DELETE current injuries and replace with snapshot data!

        Args:
            season: NFL season
            week: Week number

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Load the snapshot
            snapshot = self.load_snapshot(season, week)
            if not snapshot:
                return False, f"Snapshot not found for Season {season}, Week {week}"

            injuries = snapshot.get('injuries', [])

            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Clear existing injuries for this season
            cursor.execute("DELETE FROM player_injuries WHERE season = ?", (season,))
            logging.info(f"Cleared existing injuries for season {season}")

            # Insert snapshot injuries
            inserted = 0
            for injury in injuries:
                cursor.execute("""
                    INSERT INTO player_injuries (
                        player_name, team_abbr, season, injury_type,
                        start_week, end_week, injury_description,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    injury['player_name'],
                    injury['team_abbr'],
                    injury['season'],
                    injury['injury_type'],
                    injury['start_week'],
                    injury['end_week'],
                    injury.get('injury_description', ''),
                    injury.get('created_at', datetime.now().isoformat()),
                    datetime.now().isoformat()  # Update the updated_at timestamp
                ))
                inserted += 1

            conn.commit()
            conn.close()

            success_msg = f"Restored {inserted} injuries from Season {season}, Week {week} snapshot"
            logging.info(success_msg)
            return True, success_msg

        except Exception as e:
            error_msg = f"Error restoring snapshot: {e}"
            logging.error(error_msg)
            return False, error_msg

    def compare_snapshots(self, season: int, week1: int, week2: int) -> Dict:
        """
        Compare two snapshots to see what changed.

        Args:
            season: NFL season
            week1: First week number
            week2: Second week number

        Returns:
            Dictionary with additions, removals, and modifications
        """
        snap1 = self.load_snapshot(season, week1)
        snap2 = self.load_snapshot(season, week2)

        if not snap1 or not snap2:
            return {
                "error": "One or both snapshots not found",
                "additions": [],
                "removals": [],
                "modifications": []
            }

        # Create lookup dictionaries by player key
        injuries1 = {
            f"{inj['player_name']}_{inj['team_abbr']}": inj
            for inj in snap1['injuries']
        }
        injuries2 = {
            f"{inj['player_name']}_{inj['team_abbr']}": inj
            for inj in snap2['injuries']
        }

        # Find additions (in week2 but not week1)
        additions = [
            injuries2[key] for key in injuries2
            if key not in injuries1
        ]

        # Find removals (in week1 but not week2)
        removals = [
            injuries1[key] for key in injuries1
            if key not in injuries2
        ]

        # Find modifications
        modifications = []
        for key in injuries1:
            if key in injuries2:
                inj1 = injuries1[key]
                inj2 = injuries2[key]

                # Check if anything changed
                if (inj1['injury_type'] != inj2['injury_type'] or
                    inj1.get('start_week') != inj2.get('start_week') or
                    inj1.get('end_week') != inj2.get('end_week')):
                    modifications.append({
                        "player": key,
                        "before": inj1,
                        "after": inj2
                    })

        return {
            "additions": additions,
            "removals": removals,
            "modifications": modifications
        }

    def _count_by_type(self, injuries: List[Dict]) -> Dict[str, int]:
        """Count injuries by type."""
        counts = {}
        for injury in injuries:
            injury_type = injury.get('injury_type', 'UNKNOWN')
            counts[injury_type] = counts.get(injury_type, 0) + 1
        return counts
