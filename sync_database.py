#!/usr/bin/env python3
"""
Database GCS Sync Helper for Weekly Updates

Usage:
  Before scraping: python sync_database.py download
  After scraping:  python sync_database.py upload
"""

import sys
from pathlib import Path
from google.cloud import storage
from google.oauth2 import service_account
import json

# Configuration (update these to match your setup)
DB_PATH = Path(__file__).parent / "data" / "pfr.db"
SECRETS_FILE = Path(__file__).parent / ".streamlit" / "secrets.toml"

def get_gcs_config():
    """Read GCS configuration from Streamlit secrets file."""
    try:
        import toml
        secrets = toml.load(SECRETS_FILE)
        return {
            'bucket_name': secrets.get('gcs_bucket_name'),
            'service_account': secrets.get('gcs_service_account')
        }
    except Exception as e:
        print(f"Error reading secrets: {e}")
        print(f"Make sure {SECRETS_FILE} exists with gcs_bucket_name and gcs_service_account")
        sys.exit(1)

def get_gcs_client(service_account_dict):
    """Initialize GCS client."""
    credentials = service_account.Credentials.from_service_account_info(service_account_dict)
    return storage.Client(
        credentials=credentials,
        project=service_account_dict.get("project_id")
    )

def download_db():
    """Download database from GCS (preserves injuries/notes added through app)."""
    print("📥 Downloading database from GCS...")

    config = get_gcs_config()
    if not config['bucket_name']:
        print("❌ GCS bucket name not configured")
        sys.exit(1)

    try:
        client = get_gcs_client(config['service_account'])
        bucket = client.bucket(config['bucket_name'])
        blob = bucket.blob("pfr.db")

        # Create data directory if needed
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Download
        blob.download_to_filename(str(DB_PATH))

        file_size = DB_PATH.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ Successfully downloaded pfr.db ({file_size:.1f} MB)")
        print(f"   Location: {DB_PATH}")
        print(f"\n💡 Now run your scraping scripts to update the database")

    except Exception as e:
        print(f"❌ Failed to download: {e}")
        sys.exit(1)

def upload_db():
    """Upload database to GCS after scraping."""
    print("📤 Uploading database to GCS...")

    if not DB_PATH.exists():
        print(f"❌ Database not found at {DB_PATH}")
        print("   Make sure you ran your scraping scripts first")
        sys.exit(1)

    config = get_gcs_config()
    if not config['bucket_name']:
        print("❌ GCS bucket name not configured")
        sys.exit(1)

    try:
        client = get_gcs_client(config['service_account'])
        bucket = client.bucket(config['bucket_name'])
        blob = bucket.blob("pfr.db")

        # Upload
        blob.upload_from_filename(str(DB_PATH))

        file_size = DB_PATH.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ Successfully uploaded pfr.db ({file_size:.1f} MB)")
        print(f"   Bucket: {config['bucket_name']}/pfr.db")
        print(f"\n💡 Streamlit Cloud will download this database on next restart")

    except Exception as e:
        print(f"❌ Failed to upload: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['download', 'upload']:
        print(__doc__)
        print("\nExamples:")
        print("  python sync_database.py download   # Before running scraping scripts")
        print("  python sync_database.py upload     # After running scraping scripts")
        sys.exit(1)

    action = sys.argv[1]

    if action == 'download':
        download_db()
    else:
        upload_db()

if __name__ == "__main__":
    main()
