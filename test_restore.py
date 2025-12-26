"""
Test restore process to see what's actually happening.
"""
import sys
from pathlib import Path
import sqlite3

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, str(Path(__file__).parent))

try:
    import injury_snapshot_manager as ism
    print("✅ Import successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

db_path = Path(__file__).parent / "data" / "nfl_merged.db"

# Load GCS credentials
try:
    secrets_path = Path.home() / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        import toml
        secrets = toml.load(secrets_path)
        bucket_name = secrets.get('gcs_bucket_name')
        service_account_dict = secrets.get('gcs_service_account')
    else:
        print(f"❌ No secrets.toml found")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error loading secrets: {e}")
    sys.exit(1)

print("=" * 60)
print("🧪 Test Restore Process")
print("=" * 60)

# Check database BEFORE restore
print("\n📊 BEFORE Restore:")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM player_injuries WHERE season = 2025")
before_count = cursor.fetchone()[0]
print(f"  Injuries in database: {before_count}")

# Show what's there
cursor.execute("SELECT player_name, team_abbr FROM player_injuries WHERE season = 2025")
before_injuries = cursor.fetchall()
for player, team in before_injuries:
    print(f"    - {player} ({team})")
conn.close()

# Initialize manager
print("\n🔧 Initializing snapshot manager...")
manager = ism.InjurySnapshotManager(
    db_path=str(db_path),
    bucket_name=bucket_name,
    service_account_dict=service_account_dict
)

# Perform restore
print("\n🔄 Restoring Week 17 snapshot...")
success, message = manager.restore_snapshot(2025, 17)

if success:
    print(f"✅ {message}")
else:
    print(f"❌ {message}")
    sys.exit(1)

# Check database AFTER restore
print("\n📊 AFTER Restore:")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM player_injuries WHERE season = 2025")
after_count = cursor.fetchone()[0]
print(f"  Injuries in database: {after_count}")

# Show what's there now
cursor.execute("SELECT player_name, team_abbr FROM player_injuries WHERE season = 2025 ORDER BY player_name")
after_injuries = cursor.fetchall()
print(f"\n  All {after_count} injuries:")
for player, team in after_injuries:
    print(f"    - {player} ({team})")

conn.close()

print("\n" + "=" * 60)
print(f"📈 Result: {before_count} → {after_count} injuries")
print("=" * 60)

if after_count < 30:
    print("\n⚠️  WARNING: Expected ~38 injuries but only got", after_count)
    print("Something is preventing the full restore!")
else:
    print("\n✅ Restore successful!")
