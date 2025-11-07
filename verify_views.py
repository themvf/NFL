"""Verify all compatibility views were created successfully"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "nfl_merged.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("="*60)
print("VIEW VERIFICATION")
print("="*60)

# Get all views
cursor.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name")
views = [row[0] for row in cursor.fetchall()]

print(f"\nTotal views: {len(views)}\n")
print("All views:")
for view in views:
    print(f"  - {view}")

# Check critical views with row counts
print("\n" + "="*60)
print("CRITICAL VIEW ROW COUNTS")
print("="*60)

critical_views = [
    'games',
    'team_game_summary',
    'player_box_score',
    'rushing_leaders',
    'receiving_leaders',
    'passing_leaders',
    'touchdown_leaders',
    'touchdown_scorers',
    'first_td_game_leaders',
    'box_score_summary'
]

for view_name in critical_views:
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {view_name}")
        count = cursor.fetchone()[0]
        status = "OK" if count > 0 else "EMPTY"
        print(f"  [{status:5}] {view_name:30} {count:>6} rows")
    except Exception as e:
        print(f"  [ERROR] {view_name:30} {str(e)}")

conn.close()

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
