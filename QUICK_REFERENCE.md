# NFL Database - Quick Reference

**Quick lookup for common schema mistakes** 🚨

---

## ⚠️ Common Column Name Errors

### Receiving Stats (player_stats table)
| ❌ WRONG | ✅ CORRECT |
|---------|----------|
| `rec_air_yds` | `receiving_air_yards` |
| `rec_yac` | `receiving_yards_after_catch` |
| `rec_yds` | `receiving_yards` |
| `rec` | `receptions` |
| `rec_td` | `receiving_tds` |

### Rushing Stats (player_stats table)
| ❌ WRONG | ✅ CORRECT |
|---------|----------|
| `rush_yds` | `rushing_yards` |
| `rush_att` | `carries` |
| `rush_td` | `rushing_tds` |

### Passing Stats (player_stats table)
| ❌ WRONG | ✅ CORRECT |
|---------|----------|
| `pass_yds` | `passing_yards` |
| `pass_att` | `attempts` |
| `pass_comp` | `completions` |
| `pass_td` | `passing_tds` |
| `pass_int` | `passing_interceptions` |

---

## 🔑 Key Join Patterns

### ❌ WRONG: Joining player_stats on game_id
```sql
-- This FAILS - player_stats has no game_id column!
SELECT ps.*, s.location
FROM player_stats ps
JOIN games g ON ps.game_id = g.game_id  -- ERROR!
```

### ✅ CORRECT: Use schedules with composite keys
```sql
SELECT ps.*,
    CASE
        WHEN ps.team = s.home_team THEN 'home'
        WHEN ps.team = s.away_team THEN 'away'
    END as location
FROM player_stats ps
LEFT JOIN schedules s
    ON ps.season = s.season
    AND ps.week = s.week
    AND (ps.team = s.home_team OR ps.team = s.away_team)
```

---

## 📊 Aggregation Patterns

### ❌ WRONG: Per-player averaging
```sql
-- Averages individual player stats (too low!)
SELECT AVG(receiving_yards) as avg_yards_allowed
FROM player_stats
WHERE opponent_team = 'KC' AND targets >= 4
```
**Result:** ~60 yards (individual player average)

### ✅ CORRECT: Per-game totals then average
```sql
-- Sum per game, then average (correct!)
SELECT AVG(game_yards) as avg_yards_allowed
FROM (
    SELECT week, SUM(receiving_yards) as game_yards
    FROM player_stats
    WHERE opponent_team = 'KC' AND targets >= 4
    GROUP BY week
) per_game
```
**Result:** ~200+ yards (realistic team total)

---

## 💯 Percentage Fields

**PFR tables store percentages as decimals (0.0 - 1.0)**

### Example: Drop percentage
```python
# From database
drop_pct = 0.091  # This is 9.1%

# ❌ Wrong display
print(f"{drop_pct}%")  # Shows "0.091%"

# ✅ Correct display
print(f"{drop_pct * 100:.1f}%")  # Shows "9.1%"
```

### Percentage columns in PFR tables:
- `passing_drop_pct`
- `passing_bad_throw_pct`
- `times_pressured_pct`
- `def_completion_pct`
- `def_missed_tackle_pct`

---

## 🗂️ Table Sources Quick Lookup

| Table | Source | Use For |
|-------|--------|---------|
| `player_stats` | NFLVerse API | Weekly player stats |
| `pfr_advstats_pass_week` | PFR (scraped) | QB pressure, drops, throw quality |
| `pfr_advstats_rec_week` | PFR (scraped) | Broken tackles, drops |
| `pfr_advstats_rush_week` | PFR (scraped) | Yards before/after contact |
| `pfr_advstats_def_week` | PFR (scraped) | Coverage and pressure metrics |
| `schedules` | NFLVerse API | Game info, home/away |
| `team_stats_week` | NFLVerse API | Team totals by week |

---

## 🔍 When in Doubt

1. **Check actual schema:**
   ```bash
   python explore_schema.py
   ```

2. **Query column names:**
   ```sql
   PRAGMA table_info(player_stats);
   ```

3. **Consult full documentation:**
   See `DATA_DICTIONARY.md` for complete details

---

**Last Updated:** 2025-11-11
