"""Check player_injuries table schema."""
import sqlite3
from pathlib import Path

db_path = Path(__file__).parent / "data" / "nfl_merged.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get table schema
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='player_injuries'")
schema = cursor.fetchone()
if schema:
    print("player_injuries table schema:")
    print(schema[0])
else:
    print("Table not found")

# Get all indexes on the table
cursor.execute("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='player_injuries'")
indexes = cursor.fetchall()
print("\nIndexes:")
for idx in indexes:
    if idx[0]:
        print(idx[0])

conn.close()
