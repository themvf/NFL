"""
Quick diagnostic script to check injury records in database.
"""
import sqlite3
from pathlib import Path

db_path = Path(__file__).parent / "data" / "nfl_merged.db"

if not db_path.exists():
    print(f"Database not found at: {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Count total injuries
cursor.execute("SELECT COUNT(*) FROM player_injuries")
total = cursor.fetchone()[0]
print(f"\nTotal injuries in database: {total}")

# Count by season
cursor.execute("""
    SELECT season, COUNT(*) as count
    FROM player_injuries
    GROUP BY season
    ORDER BY season DESC
""")
print("\nInjuries by season:")
for row in cursor.fetchall():
    print(f"  Season {row[0]}: {row[1]} injuries")

# Count by team (top 10)
cursor.execute("""
    SELECT team_abbr, COUNT(*) as count
    FROM player_injuries
    GROUP BY team_abbr
    ORDER BY count DESC
    LIMIT 10
""")
print("\nTop 10 teams by injury count:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} injuries")

# Count by injury type
cursor.execute("""
    SELECT injury_type, COUNT(*) as count
    FROM player_injuries
    GROUP BY injury_type
    ORDER BY count DESC
""")
print("\nInjuries by type:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} injuries")

# Show all injuries for a specific season (change this to match your restored snapshot)
check_season = 2025
cursor.execute("""
    SELECT player_name, team_abbr, injury_type, start_week, end_week
    FROM player_injuries
    WHERE season = ?
    ORDER BY team_abbr, player_name
""", (check_season,))

print(f"\nAll injuries for {check_season} season:")
rows = cursor.fetchall()
print(f"Total count: {len(rows)}\n")
for row in rows:
    print(f"  {row[1]:4s} | {row[0]:30s} | {row[2]:15s} | Weeks {row[3]}-{row[4]}")

conn.close()
