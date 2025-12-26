"""
Check what projection records exist in the database.
"""
import sqlite3
from pathlib import Path
import sys

# Configure UTF-8 encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

db_path = Path(__file__).parent / "data" / "nfl_merged.db"

if not db_path.exists():
    print(f"❌ Database not found at: {db_path}")
    sys.exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=" * 60)
print("🔍 Projection Records Diagnostic")
print("=" * 60)

# Check if projection_accuracy table has the new columns
print("\n📋 Table Schema:")
cursor.execute("PRAGMA table_info(projection_accuracy)")
columns = cursor.fetchall()
print(f"Columns in projection_accuracy:")
for col in columns:
    print(f"  - {col[1]} ({col[2]})")

has_snapshot_id = any(col[1] == 'snapshot_id' for col in columns)
if has_snapshot_id:
    print("\n✅ snapshot_id column exists")
else:
    print("\n❌ snapshot_id column MISSING - run app to add it")
    sys.exit(1)

# Count total projection records
cursor.execute("SELECT COUNT(*) FROM projection_accuracy")
total = cursor.fetchone()[0]
print(f"\n📊 Total projection records: {total}")

if total == 0:
    print("✅ No projection records - fresh start!")
    sys.exit(0)

# Show records by season/week
cursor.execute("""
    SELECT season, week, COUNT(*) as count
    FROM projection_accuracy
    GROUP BY season, week
    ORDER BY season DESC, week DESC
""")
print("\n📅 Records by Season/Week:")
for row in cursor.fetchall():
    print(f"  Season {row[0]}, Week {row[1]}: {row[2]} projections")

# Show records by team
cursor.execute("""
    SELECT team_abbr, opponent_abbr, season, week, COUNT(*) as count
    FROM projection_accuracy
    GROUP BY team_abbr, opponent_abbr, season, week
    ORDER BY season DESC, week DESC
""")
print("\n🏈 Records by Matchup:")
for row in cursor.fetchall():
    print(f"  {row[0]} vs {row[1]} (S{row[2]} W{row[3]}): {row[4]} players")

# Show sample player records
cursor.execute("""
    SELECT player_name, team_abbr, opponent_abbr, season, week, position
    FROM projection_accuracy
    ORDER BY season DESC, week DESC
    LIMIT 10
""")
print("\n👥 Sample Player Records (first 10):")
for row in cursor.fetchall():
    print(f"  {row[0]} ({row[1]}): {row[5]} vs {row[2]} - S{row[3]} W{row[4]}")

# Check for snapshot_id population
cursor.execute("""
    SELECT COUNT(*) FROM projection_accuracy WHERE snapshot_id IS NOT NULL
""")
with_snapshot = cursor.fetchone()[0]
print(f"\n📸 Records with snapshot_id: {with_snapshot} / {total}")

conn.close()

print("\n" + "=" * 60)
print("💡 To clear all projection records:")
print("   DELETE FROM projection_accuracy;")
print("=" * 60)
