"""
Quick test script to verify GCS snapshot functionality.
Run this to test connectivity before using the UI.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import streamlit as st
    from google.cloud import storage
    from google.oauth2 import service_account
    import injury_snapshot_manager as ism
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have all dependencies installed:")
    print("  pip install google-cloud-storage streamlit pandas")
    sys.exit(1)

def test_gcs_connection():
    """Test basic GCS connectivity."""
    print("\n🔍 Testing GCS Connection...")

    # Load secrets
    try:
        # Try to load from Streamlit secrets (works if streamlit is initialized)
        secrets_path = Path.home() / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            import toml
            secrets = toml.load(secrets_path)
            bucket_name = secrets.get('gcs_bucket_name')
            service_account_dict = secrets.get('gcs_service_account')
        else:
            print("❌ No secrets.toml found at:", secrets_path)
            print("Create one with your GCS credentials or run via Streamlit")
            return False

    except Exception as e:
        print(f"❌ Error loading secrets: {e}")
        return False

    if not bucket_name:
        print("❌ No gcs_bucket_name in secrets")
        return False

    # Test connection
    try:
        credentials = service_account.Credentials.from_service_account_info(
            service_account_dict
        )
        client = storage.Client(credentials=credentials)
        bucket = client.bucket(bucket_name)

        # Try to list blobs
        blobs = list(bucket.list_blobs(prefix="injury-snapshots/", max_results=5))
        print(f"✅ Successfully connected to bucket: {bucket_name}")
        print(f"   Found {len(blobs)} existing snapshot files")

        return True

    except Exception as e:
        print(f"❌ GCS connection failed: {e}")
        return False


def test_snapshot_manager():
    """Test snapshot manager initialization."""
    print("\n🔍 Testing Snapshot Manager...")

    try:
        secrets_path = Path.home() / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            import toml
            secrets = toml.load(secrets_path)
            bucket_name = secrets.get('gcs_bucket_name')
            service_account_dict = secrets.get('gcs_service_account')
        else:
            print("❌ No secrets.toml found")
            return False

        # Initialize manager
        db_path = Path(__file__).parent / "data" / "nfl_merged.db"

        manager = ism.InjurySnapshotManager(
            db_path=str(db_path),
            bucket_name=bucket_name,
            service_account_dict=service_account_dict
        )

        print("✅ Snapshot manager initialized successfully")

        # Test listing snapshots
        snapshots = manager.list_snapshots(2025)
        print(f"   Found {len(snapshots)} snapshots for 2025 season")

        if snapshots:
            print("\n   Available snapshots:")
            for snap in snapshots[:5]:  # Show first 5
                print(f"   - Week {snap['week']}: {snap['created'].strftime('%Y-%m-%d %H:%M')}")

        return True

    except Exception as e:
        print(f"❌ Snapshot manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 GCS Snapshot System Test")
    print("=" * 60)

    tests_passed = 0
    tests_total = 2

    # Test 1: GCS Connection
    if test_gcs_connection():
        tests_passed += 1

    # Test 2: Snapshot Manager
    if test_snapshot_manager():
        tests_passed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{tests_total} passed")
    print("=" * 60)

    if tests_passed == tests_total:
        print("✅ All tests passed! Snapshot system is ready to use.")
        print("\nNext steps:")
        print("1. Run your Streamlit app: streamlit run pfr_viewer.py")
        print("2. Navigate to Injury Management → 📸 Snapshots")
        print("3. Create your first snapshot!")
    else:
        print("❌ Some tests failed. Check the errors above.")
        print("\nCommon fixes:")
        print("1. Make sure secrets.toml exists in ~/.streamlit/")
        print("2. Verify GCS credentials are correct")
        print("3. Check that bucket name is correct")
        print("4. Ensure database file exists")

    return tests_passed == tests_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
