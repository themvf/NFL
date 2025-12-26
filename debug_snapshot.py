"""
Debug script to inspect snapshot contents and restore process.
"""
import sys
from pathlib import Path
import sqlite3
import json

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import injury_snapshot_manager as ism
    from google.cloud import storage
    from google.oauth2 import service_account
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Load GCS credentials
try:
    secrets_path = Path.home() / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        import toml
        secrets = toml.load(secrets_path)
        bucket_name = secrets.get('gcs_bucket_name')
        service_account_dict = secrets.get('gcs_service_account')
    else:
        print(f"❌ No secrets.toml found at: {secrets_path}")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error loading secrets: {e}")
    sys.exit(1)

db_path = Path(__file__).parent / "data" / "nfl_merged.db"

print("=" * 60)
print("🔍 Snapshot Debug Analysis")
print("=" * 60)

# Initialize manager
manager = ism.InjurySnapshotManager(
    db_path=str(db_path),
    bucket_name=bucket_name,
    service_account_dict=service_account_dict
)

# List all snapshots
print("\n📋 Available Snapshots:")
snapshots = manager.list_snapshots(2025)
if snapshots:
    for snap in snapshots:
        print(f"  - Week {snap['week']}: Created {snap['created']}, Size: {snap['size_bytes']} bytes")
else:
    print("  No snapshots found for 2025")
    sys.exit(1)

# Ask user which week to debug
week_to_debug = 17  # Default to week 17
print(f"\n🔍 Debugging Week {week_to_debug} snapshot...")

# Load the snapshot
snapshot = manager.load_snapshot(2025, week_to_debug)

if not snapshot:
    print(f"❌ Could not load Week {week_to_debug} snapshot")
    sys.exit(1)

print(f"\n📦 Snapshot Contents:")
print(f"  Snapshot Date: {snapshot.get('snapshot_date')}")
print(f"  Season: {snapshot.get('season')}")
print(f"  Week: {snapshot.get('week')}")
print(f"  Description: {snapshot.get('description', 'N/A')}")

# Check metadata
metadata = snapshot.get('metadata', {})
print(f"\n📊 Metadata:")
print(f"  Total Injuries: {metadata.get('total_injuries', 0)}")
print(f"  Teams Affected: {len(metadata.get('teams_affected', []))}")

teams_affected = metadata.get('teams_affected', [])
if teams_affected:
    print(f"  Teams: {', '.join(sorted(teams_affected))}")

injury_types = metadata.get('injury_types', {})
if injury_types:
    print(f"\n  Injury Types:")
    for itype, count in sorted(injury_types.items(), key=lambda x: -x[1]):
        print(f"    - {itype}: {count}")

# Check actual injuries list
injuries = snapshot.get('injuries', [])
print(f"\n🏥 Injuries in Snapshot: {len(injuries)} total")

if len(injuries) != metadata.get('total_injuries', 0):
    print(f"⚠️  WARNING: Mismatch between metadata ({metadata.get('total_injuries')}) and actual count ({len(injuries)})")

# Group by team
injuries_by_team = {}
for inj in injuries:
    team = inj.get('team_abbr', 'UNKNOWN')
    if team not in injuries_by_team:
        injuries_by_team[team] = []
    injuries_by_team[team].append(inj)

print(f"\n📋 Injuries by Team:")
for team in sorted(injuries_by_team.keys()):
    team_injuries = injuries_by_team[team]
    print(f"\n  {team} ({len(team_injuries)} injuries):")
    for inj in team_injuries[:5]:  # Show first 5 per team
        print(f"    - {inj['player_name']}: {inj['injury_type']} (Weeks {inj.get('start_week')}-{inj.get('end_week')})")
    if len(team_injuries) > 5:
        print(f"    ... and {len(team_injuries) - 5} more")

# Check current database state BEFORE restore
print(f"\n💾 Current Database State (BEFORE restore):")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM player_injuries WHERE season = 2025")
current_count = cursor.fetchone()[0]
print(f"  Current injuries in DB for 2025: {current_count}")

# Show what would be deleted
cursor.execute("""
    SELECT player_name, team_abbr, injury_type
    FROM player_injuries
    WHERE season = 2025
""")
current_injuries = cursor.fetchall()
if current_injuries:
    print(f"\n  Current injuries that will be DELETED:")
    for inj in current_injuries:
        print(f"    - {inj[0]} ({inj[1]}): {inj[2]}")
conn.close()

# Simulate restore (show what would be inserted)
print(f"\n🔄 Simulating Restore Process:")
print(f"  Step 1: DELETE FROM player_injuries WHERE season = 2025")
print(f"           → Would delete {current_count} records")
print(f"\n  Step 2: INSERT {len(injuries)} injuries from snapshot")

# Check for potential issues
print(f"\n🔍 Checking for Potential Issues:")

# Check if any injuries are missing required fields
missing_fields = []
for idx, inj in enumerate(injuries):
    required = ['player_name', 'team_abbr', 'season', 'injury_type']
    missing = [f for f in required if not inj.get(f)]
    if missing:
        missing_fields.append((idx, inj.get('player_name', 'UNKNOWN'), missing))

if missing_fields:
    print(f"  ⚠️  Found {len(missing_fields)} injuries with missing required fields:")
    for idx, name, fields in missing_fields[:5]:
        print(f"    - Injury #{idx}: {name} missing {fields}")
else:
    print(f"  ✅ All injuries have required fields")

# Check for duplicates within the snapshot
player_keys = [(inj['player_name'], inj['team_abbr'], inj['season']) for inj in injuries]
unique_keys = set(player_keys)
if len(player_keys) != len(unique_keys):
    duplicates_in_snapshot = len(player_keys) - len(unique_keys)
    print(f"  ⚠️  Found {duplicates_in_snapshot} duplicate player-team-season combinations in snapshot!")

    # Find which ones are duplicated
    from collections import Counter
    counts = Counter(player_keys)
    dups = [(k, v) for k, v in counts.items() if v > 1]
    print(f"    Duplicates:")
    for key, count in dups[:5]:
        print(f"      - {key[0]} ({key[1]}, {key[2]}): {count} copies")
else:
    print(f"  ✅ No duplicates in snapshot (all {len(unique_keys)} are unique)")

# Summary
print(f"\n" + "=" * 60)
print(f"📊 Summary")
print(f"=" * 60)
print(f"  Snapshot contains: {len(injuries)} injuries")
print(f"  Unique players: {len(unique_keys)}")
print(f"  Current DB has: {current_count} injuries for 2025")
print(f"  After restore, DB should have: {len(unique_keys)} injuries")
print(f"=" * 60)

print("\n💡 To actually perform restore:")
print("   1. Go to Streamlit app")
print("   2. Transaction Manager → Injury Management → Snapshots")
print("   3. Click Restore on Week 17")
print(f"   4. Should restore {len(unique_keys)} unique injuries")
