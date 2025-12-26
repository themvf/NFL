"""
Migration script to fix player_injuries table schema.

This script:
1. Creates a new table with the correct schema (including UNIQUE constraint)
2. Migrates existing data, removing duplicates
3. Replaces the old table

Run this script ONCE to fix the schema issue.
"""
import sqlite3
from pathlib import Path
import sys

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

db_path = Path(__file__).parent / "data" / "nfl_merged.db"

if not db_path.exists():
    print(f"❌ Database not found at: {db_path}")
    sys.exit(1)

print("=" * 60)
print("🔧 Player Injuries Schema Migration")
print("=" * 60)

# Backup warning
print("\n⚠️  WARNING: This will modify the player_injuries table!")
print("Make sure you have a backup of your database before proceeding.")
response = input("\nType 'YES' to continue: ")
if response != 'YES':
    print("Migration cancelled.")
    sys.exit(0)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Step 1: Count current injuries
    cursor.execute("SELECT COUNT(*) FROM player_injuries")
    total_before = cursor.fetchone()[0]
    print(f"\n📊 Current injuries in database: {total_before}")

    # Step 2: Check for duplicates
    cursor.execute("""
        SELECT player_name, team_abbr, season, COUNT(*) as count
        FROM player_injuries
        GROUP BY player_name, team_abbr, season
        HAVING count > 1
    """)
    duplicates = cursor.fetchall()

    if duplicates:
        print(f"\n⚠️  Found {len(duplicates)} sets of duplicates:")
        for dup in duplicates[:10]:  # Show first 10
            print(f"  - {dup[0]} ({dup[1]}, {dup[2]}): {dup[3]} copies")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more")
    else:
        print("\n✅ No duplicates found")

    # Step 3: Create new table with correct schema
    print("\n🔨 Creating new table with correct schema...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_injuries_new (
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
    print("  ✅ New table created")

    # Step 4: Migrate data, keeping only most recent record for each (player, team, season)
    print("\n📦 Migrating data (removing duplicates)...")
    cursor.execute("""
        INSERT INTO player_injuries_new (
            player_name, team_abbr, season, injury_type,
            start_week, end_week, injury_description,
            created_at, updated_at
        )
        SELECT
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
        GROUP BY player_name, team_abbr, season
        HAVING injury_id = MAX(injury_id)
    """)
    migrated = cursor.rowcount
    print(f"  ✅ Migrated {migrated} unique injuries")

    # Step 5: Replace old table with new one
    print("\n🔄 Replacing old table...")
    cursor.execute("DROP TABLE player_injuries")
    cursor.execute("ALTER TABLE player_injuries_new RENAME TO player_injuries")
    print("  ✅ Table replaced")

    # Step 6: Verify
    cursor.execute("SELECT COUNT(*) FROM player_injuries")
    total_after = cursor.fetchone()[0]
    print(f"\n✅ Migration complete!")
    print(f"   Before: {total_before} injuries")
    print(f"   After:  {total_after} injuries")
    print(f"   Removed: {total_before - total_after} duplicates")

    # Commit changes
    conn.commit()
    print("\n💾 Changes committed to database")

    # Verify schema
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='player_injuries'")
    schema = cursor.fetchone()
    if 'UNIQUE' in schema[0]:
        print("✅ UNIQUE constraint verified")
    else:
        print("❌ Warning: UNIQUE constraint not found in schema")

except Exception as e:
    print(f"\n❌ Error during migration: {e}")
    conn.rollback()
    print("⏪ Changes rolled back")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    conn.close()

print("\n" + "=" * 60)
print("✅ Migration successful!")
print("=" * 60)
print("\nNext steps:")
print("1. Verify injuries look correct in Active Injuries tab")
print("2. Restore your Week 17 snapshot again")
print("3. All 39 injuries should now appear correctly")
