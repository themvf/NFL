"""
Quick test script to verify Projection Snapshot functionality.
Run this to test database and GCS integration before using the UI.
"""

import sys
from pathlib import Path
import sqlite3

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from google.cloud import storage
    from google.oauth2 import service_account
    import projection_snapshot_manager as psm
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have all dependencies installed:")
    print("  pip install google-cloud-storage pandas")
    sys.exit(1)


def test_database_tables():
    """Test that database tables exist and are properly structured."""
    print("\n🔍 Testing Database Tables...")

    try:
        db_path = Path(__file__).parent / "data" / "nfl_merged.db"
        if not db_path.exists():
            print(f"❌ Database not found at: {db_path}")
            return False

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check projection_snapshots table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='projection_snapshots'")
        if cursor.fetchone():
            print("  ✅ projection_snapshots table exists")

            # Check columns
            cursor.execute("PRAGMA table_info(projection_snapshots)")
            columns = [row[1] for row in cursor.fetchall()]
            required_cols = ['snapshot_id', 'season', 'week', 'home_team', 'away_team', 'gcs_path', 'status']
            missing = [col for col in required_cols if col not in columns]
            if missing:
                print(f"  ⚠️ Missing columns: {missing}")
            else:
                print(f"  ✅ All required columns present ({len(columns)} total)")
        else:
            print("  ❌ projection_snapshots table not found")
            conn.close()
            return False

        # Check team_projection_accuracy table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='team_projection_accuracy'")
        if cursor.fetchone():
            print("  ✅ team_projection_accuracy table exists")
        else:
            print("  ❌ team_projection_accuracy table not found")
            conn.close()
            return False

        # Check projection_accuracy table extensions
        cursor.execute("PRAGMA table_info(projection_accuracy)")
        columns = [row[1] for row in cursor.fetchall()]
        new_cols = ['snapshot_id', 'in_range', 'game_played', 'projected_p10', 'projected_p90']
        existing_new_cols = [col for col in new_cols if col in columns]
        if len(existing_new_cols) == len(new_cols):
            print(f"  ✅ projection_accuracy extended with {len(new_cols)} new columns")
        else:
            print(f"  ⚠️ projection_accuracy has {len(existing_new_cols)}/{len(new_cols)} new columns")

        conn.close()
        return True

    except Exception as e:
        print(f"  ❌ Database test failed: {e}")
        return False


def test_gcs_connection():
    """Test basic GCS connectivity."""
    print("\n🔍 Testing GCS Connection...")

    try:
        # Load secrets
        secrets_path = Path.home() / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            import toml
            secrets = toml.load(secrets_path)
            bucket_name = secrets.get('gcs_bucket_name')
            service_account_dict = secrets.get('gcs_service_account')
        else:
            print(f"  ❌ No secrets.toml found at: {secrets_path}")
            print("  Create one with your GCS credentials or run via Streamlit")
            return False

        if not bucket_name:
            print("  ❌ No gcs_bucket_name in secrets")
            return False

        # Test connection
        credentials = service_account.Credentials.from_service_account_info(
            service_account_dict
        )
        client = storage.Client(credentials=credentials)
        bucket = client.bucket(bucket_name)

        # Try to list blobs in projection-snapshots folder
        blobs = list(bucket.list_blobs(prefix="projection-snapshots/", max_results=5))
        print(f"  ✅ Successfully connected to bucket: {bucket_name}")
        print(f"     Found {len(blobs)} existing projection snapshot files")

        return True

    except Exception as e:
        print(f"  ❌ GCS connection failed: {e}")
        return False


def test_snapshot_manager():
    """Test snapshot manager initialization."""
    print("\n🔍 Testing Projection Snapshot Manager...")

    try:
        secrets_path = Path.home() / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            import toml
            secrets = toml.load(secrets_path)
            bucket_name = secrets.get('gcs_bucket_name')
            service_account_dict = secrets.get('gcs_service_account')
        else:
            print("  ❌ No secrets.toml found")
            return False

        # Initialize manager
        db_path = Path(__file__).parent / "data" / "nfl_merged.db"

        manager = psm.ProjectionSnapshotManager(
            db_path=str(db_path),
            bucket_name=bucket_name,
            service_account_dict=service_account_dict
        )

        print("  ✅ Snapshot manager initialized successfully")

        # Test listing snapshots
        snapshots = manager.list_snapshots(2025)
        print(f"     Found {len(snapshots)} snapshots for 2025 season")

        if not snapshots.empty:
            print("\n     Available snapshots (first 5):")
            for idx, snap in snapshots.head().iterrows():
                matchup = f"{snap['away_team']} @ {snap['home_team']}"
                print(f"     - Week {snap['week']}: {matchup} ({snap['status']})")

        # Test calculate_accuracy_metrics (should work even with no data)
        metrics = manager.calculate_accuracy_metrics(season=2025)
        if metrics:
            print(f"\n     Accuracy metrics calculated:")
            print(f"     - Total projections: {metrics.get('total_projections', 0)}")
            if metrics.get('total_projections', 0) > 0:
                print(f"     - MAE: {metrics.get('mae', 0):.1f} yards")
                print(f"     - RMSE: {metrics.get('rmse', 0):.1f} yards")
                print(f"     - Bias: {metrics.get('bias', 0):+.1f} yards")

        return True

    except Exception as e:
        print(f"  ❌ Snapshot manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 Projection Snapshot System Test")
    print("=" * 60)

    tests_passed = 0
    tests_total = 3

    # Test 1: Database Tables
    if test_database_tables():
        tests_passed += 1

    # Test 2: GCS Connection
    if test_gcs_connection():
        tests_passed += 1

    # Test 3: Snapshot Manager
    if test_snapshot_manager():
        tests_passed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{tests_total} passed")
    print("=" * 60)

    if tests_passed == tests_total:
        print("✅ All tests passed! Projection snapshot system is ready to use.")
        print("\nNext steps:")
        print("1. Run your Streamlit app: streamlit run pfr_viewer.py")
        print("2. Navigate to Team Comparison → View 2")
        print("3. Select a matchup and click 📸 Save Projection")
        print("4. After games complete, go to Transaction Manager → Injury Management → 🎯 Projection Accuracy")
        print("5. Click 🔄 Update Actuals to fetch real stats")
        print("6. View accuracy metrics!")
    else:
        print("❌ Some tests failed. Check the errors above.")
        print("\nCommon fixes:")
        print("1. Make sure secrets.toml exists in ~/.streamlit/")
        print("2. Verify GCS credentials are correct")
        print("3. Check that bucket name is correct")
        print("4. Ensure database file exists")
        print("5. Run pfr_viewer.py at least once to create new tables")

    return tests_passed == tests_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
