"""
Manually add missing columns to projection_accuracy table.
Run this once to fix the schema.
"""
import sqlite3
from pathlib import Path
import sys

# Configure UTF-8 encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

db_path = Path(__file__).parent / "data" / "nfl_merged.db"

if not db_path.exists():
    print(f"❌ Database not found at: {db_path}")
    sys.exit(1)

print("=" * 60)
print("🔧 Adding Missing Columns to projection_accuracy")
print("=" * 60)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check existing columns
cursor.execute("PRAGMA table_info(projection_accuracy)")
existing_columns = [row[1] for row in cursor.fetchall()]
print(f"\n📋 Existing columns: {len(existing_columns)}")

# Columns to add
new_columns = {
    'snapshot_id': 'TEXT',
    'in_range': 'BOOLEAN',
    'game_played': 'BOOLEAN DEFAULT TRUE',
    'projected_p10': 'REAL',
    'projected_p90': 'REAL'
}

added = 0
skipped = 0

for col_name, col_type in new_columns.items():
    if col_name in existing_columns:
        print(f"  ⏭️  {col_name} - already exists")
        skipped += 1
    else:
        try:
            cursor.execute(f"ALTER TABLE projection_accuracy ADD COLUMN {col_name} {col_type}")
            print(f"  ✅ {col_name} - added")
            added += 1
        except Exception as e:
            print(f"  ❌ {col_name} - error: {e}")

conn.commit()
conn.close()

print(f"\n✅ Added {added} new columns, {skipped} already existed")
print("=" * 60)
print("Now try saving a projection snapshot in the app!")
