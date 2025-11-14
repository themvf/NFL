# NFL Database Data Dictionary

**Database:** `nfl_merged.db`
**Last Updated:** 2025-11-14
**Purpose:** Comprehensive reference for all data types, column names, and sources

---

## 📊 Data Sources Overview

| Source | Tables | Description |
|--------|--------|-------------|
| **NFLVerse API** | player_stats, schedules, team_stats, team_stats_week, rosters, injuries, pfr_advstats_* | Official NFL statistical data from nflverse-data repository (includes PFR-format advanced stats) |
| **Custom Tracking** | player_injuries, ingest_metadata, merge_metadata, upcoming_games, projection_accuracy | Application-specific data management |
| **Play-by-Play** | plays, play_participants | Detailed play-level data |

---

## 🗂️ Table Descriptions

### 1. player_stats
**Source:** NFLVerse API
**Rows:** 20,347
**Description:** Weekly player statistics including passing, rushing, receiving, defense, and kicking stats

#### Key Columns:
| Column Name | Data Type | Definition |
|-------------|-----------|------------|
| `player_id` | TEXT | NFLVerse unique player identifier |
| `player_name` | TEXT | Player's full name (Last, First format) |
| `player_display_name` | TEXT | Player's display name (First Last format) |
| `season` | INTEGER | NFL season year (e.g., 2025) |
| `week` | INTEGER | Week number within season (1-18 for regular season) |
| `season_type` | TEXT | REG (regular season), POST (playoffs), PRE (preseason) |
| `team` | TEXT | Player's team abbreviation (3 letters) |
| `opponent_team` | TEXT | Opponent team abbreviation |

#### Passing Stats:
| Column Name | Data Type | Definition |
|-------------|-----------|------------|
| `completions` | INTEGER | Number of completed passes |
| `attempts` | INTEGER | Total pass attempts |
| `passing_yards` | INTEGER | Total passing yards |
| `passing_tds` | INTEGER | Passing touchdowns |
| `passing_interceptions` | INTEGER | Interceptions thrown |
| `sacks_suffered` | INTEGER | Times sacked |
| `sack_yards_lost` | INTEGER | Yards lost on sacks |
| `passing_air_yards` | INTEGER | Total air yards (distance ball travels in air) |
| `passing_yards_after_catch` | INTEGER | Yards gained after catch (YAC) |
| `passing_first_downs` | INTEGER | First downs via passing |
| `passing_epa` | REAL | Expected Points Added on pass plays |
| `pacr` | REAL | Passing Air Conversion Ratio (yards/air_yards) |

#### Rushing Stats:
| Column Name | Data Type | Definition |
|-------------|-----------|------------|
| `carries` | INTEGER | Number of rushing attempts |
| `rushing_yards` | INTEGER | Total rushing yards |
| `rushing_tds` | INTEGER | Rushing touchdowns |
| `rushing_fumbles` | INTEGER | Fumbles on rush attempts |
| `rushing_fumbles_lost` | INTEGER | Fumbles lost on rushes |
| `rushing_first_downs` | INTEGER | First downs via rushing |
| `rushing_epa` | REAL | Expected Points Added on rush plays |

#### Receiving Stats:
| Column Name | Data Type | Definition |
|-------------|-----------|------------|
| `receptions` | INTEGER | Number of receptions (catches) |
| `targets` | INTEGER | Number of times targeted |
| `receiving_yards` | INTEGER | Total receiving yards |
| `receiving_tds` | INTEGER | Receiving touchdowns |
| `receiving_air_yards` | INTEGER | **Total air yards on targets** (distance ball travels before catch) |
| `receiving_yards_after_catch` | INTEGER | **Yards gained after catch** (YAC) |
| `receiving_first_downs` | INTEGER | First downs via receiving |
| `receiving_epa` | REAL | Expected Points Added on receptions |
| `racr` | REAL | Receiver Air Conversion Ratio |
| `target_share` | REAL | % of team targets (0.0-1.0) |
| `air_yards_share` | REAL | % of team air yards (0.0-1.0) |
| `wopr` | REAL | Weighted Opportunity Rating |

> ⚠️ **Important:** Use `receiving_air_yards` and `receiving_yards_after_catch` (NOT `rec_air_yds` or `rec_yac`)

#### Defensive Stats:
| Column Name | Data Type | Definition |
|-------------|-----------|------------|
| `def_tackles_solo` | INTEGER | Solo tackles |
| `def_tackles_with_assist` | INTEGER | Tackles with assistance |
| `def_tackle_assists` | INTEGER | Assisted tackles |
| `def_tackles_for_loss` | INTEGER | Tackles behind line of scrimmage |
| `def_sacks` | REAL | QB sacks (can be fractional for assists) |
| `def_qb_hits` | INTEGER | QB hits |
| `def_interceptions` | INTEGER | Interceptions |
| `def_pass_defended` | INTEGER | Passes defensed |
| `def_tds` | INTEGER | Defensive touchdowns |

#### Fantasy Stats:
| Column Name | Data Type | Definition |
|-------------|-----------|------------|
| `fantasy_points` | REAL | Standard fantasy points |
| `fantasy_points_ppr` | REAL | PPR (Point Per Reception) fantasy points |

---

### 2. pfr_advstats_pass_week
**Source:** NFLVerse API (via `nfl.load_pfr_advstats()`)
**Rows:** 320
**Description:** Advanced passing metrics including pressure stats, drops, and throw quality (PFR-format data from nflverse API)

| Column Name | Data Type | Definition |
|-------------|-----------|------------|
| `game_id` | TEXT | NFLVerse game identifier (YYYY_WW_AWAY_HOME) |
| `pfr_game_id` | TEXT | PFR-specific game identifier |
| `season` | INTEGER | NFL season year |
| `week` | INTEGER | Week number |
| `team` | TEXT | Team abbreviation |
| `opponent` | TEXT | Opponent team abbreviation |
| `pfr_player_name` | TEXT | Player name (PFR format) |
| `pfr_player_id` | TEXT | PFR unique player identifier |
| `passing_drops` | REAL | **Number of drops by receivers** (raw count) |
| `passing_drop_pct` | REAL | Percentage of passes dropped (0.0-1.0) |
| `passing_bad_throws` | REAL | Number of bad throws |
| `passing_bad_throw_pct` | REAL | Percentage of bad throws (0.0-1.0) |
| `times_sacked` | REAL | Number of times sacked |
| `times_blitzed` | REAL | Number of times blitzed |
| `times_hurried` | REAL | Number of hurried throws |
| `times_hit` | REAL | Number of times hit |
| `times_pressured` | REAL | Total pressures |
| `times_pressured_pct` | REAL | Pressure rate (0.0-1.0) |

> ⚠️ **Important:** `passing_drops` is the raw count, `passing_drop_pct` is the percentage (0.0-1.0)

---

### 3. pfr_advstats_rec_week
**Source:** NFLVerse API (via `nfl.load_pfr_advstats()`)
**Rows:** 2,139
**Description:** Advanced receiving metrics including broken tackles and drops (PFR-format data from nflverse API)

| Column Name | Data Type | Definition |
|-------------|-----------|------------|
| `game_id` | TEXT | NFLVerse game identifier |
| `pfr_player_name` | TEXT | Player name (PFR format) |
| `rushing_broken_tackles` | REAL | Broken tackles on rushes |
| `receiving_broken_tackles` | REAL | Broken tackles on receptions |
| `receiving_drop` | REAL | Number of drops by receiver |
| `receiving_drop_pct` | REAL | Drop percentage (0.0-1.0) |
| `receiving_int` | REAL | Interceptions on targets |
| `receiving_rat` | REAL | Passer rating when targeted |

---

### 4. pfr_advstats_rush_week
**Source:** NFLVerse API (via `nfl.load_pfr_advstats()`)
**Rows:** 1,107
**Description:** Advanced rushing metrics including yards before/after contact (PFR-format data from nflverse API)

| Column Name | Data Type | Definition |
|-------------|-----------|------------|
| `game_id` | TEXT | NFLVerse game identifier |
| `pfr_player_name` | TEXT | Player name (PFR format) |
| `carries` | REAL | Number of rushing attempts |
| `rushing_yards_before_contact` | REAL | Total yards before first contact |
| `rushing_yards_before_contact_avg` | REAL | Average yards before contact per carry |
| `rushing_yards_after_contact` | REAL | Total yards after first contact |
| `rushing_yards_after_contact_avg` | REAL | Average yards after contact per carry |
| `rushing_broken_tackles` | REAL | Number of broken tackles on rushes |

---

### 5. pfr_advstats_def_week
**Source:** NFLVerse API (via `nfl.load_pfr_advstats()`)
**Rows:** 3,789
**Description:** Advanced defensive coverage and pressure metrics (PFR-format data from nflverse API)

| Column Name | Data Type | Definition |
|-------------|-----------|------------|
| `game_id` | TEXT | NFLVerse game identifier |
| `pfr_player_name` | TEXT | Player name (PFR format) |
| `def_ints` | REAL | Interceptions |
| `def_targets` | REAL | Times targeted in coverage |
| `def_completions_allowed` | REAL | Completions allowed |
| `def_completion_pct` | REAL | Completion % allowed (0.0-1.0) |
| `def_yards_allowed` | REAL | Yards allowed in coverage |
| `def_yards_allowed_per_cmp` | REAL | Yards per completion allowed |
| `def_yards_allowed_per_tgt` | REAL | Yards per target allowed |
| `def_receiving_td_allowed` | REAL | TDs allowed in coverage |
| `def_passer_rating_allowed` | REAL | Passer rating when targeted |
| `def_adot` | REAL | Average depth of target allowed |
| `def_air_yards_completed` | REAL | Air yards on completions allowed |
| `def_yards_after_catch` | REAL | YAC allowed |
| `def_times_blitzed` | REAL | Number of blitzes |
| `def_times_hurried` | REAL | QB hurries |
| `def_times_hitqb` | REAL | QB hits |
| `def_sacks` | REAL | Sacks |
| `def_pressures` | REAL | Total pressures |
| `def_tackles_combined` | REAL | Total tackles |
| `def_missed_tackles` | REAL | Missed tackles |
| `def_missed_tackle_pct` | REAL | Missed tackle % (0.0-1.0) |

---

### 6. team_stats_week
**Source:** NFLVerse API
**Rows:** 640
**Description:** Weekly team-level statistics

| Column Name | Data Type | Definition |
|-------------|-----------|------------|
| `team` | TEXT | Team abbreviation |
| `season` | INTEGER | NFL season year |
| `week` | INTEGER | Week number |
| `points` | INTEGER | Points scored |
| `total_yards` | INTEGER | Total offensive yards |
| `turnovers` | INTEGER | Total turnovers |
| `penalties` | INTEGER | Number of penalties |
| `penalty_yards` | INTEGER | Penalty yards |
| `time_of_possession` | TEXT | Time of possession (MM:SS) |

---

### 7. schedules
**Source:** NFLVerse API
**Rows:** 320
**Description:** Game schedule information

| Column Name | Data Type | Definition |
|-------------|-----------|------------|
| `game_id` | TEXT | NFLVerse game identifier (YYYY_WW_AWAY_HOME) |
| `season` | INTEGER | NFL season year |
| `week` | INTEGER | Week number |
| `game_type` | TEXT | REG, POST, or PRE |
| `gameday` | TEXT | Game date (YYYY-MM-DD) |
| `weekday` | TEXT | Day of week |
| `gametime` | TEXT | Game time (HH:MM) |
| `away_team` | TEXT | Away team abbreviation |
| `away_score` | INTEGER | Away team final score |
| `home_team` | TEXT | Home team abbreviation |
| `home_score` | INTEGER | Home team final score |
| `location` | TEXT | Stadium location |
| `stadium` | TEXT | Stadium name |

> ⚠️ **Important:** Use `schedules` table (NOT `games` table) for home/away lookups. Join on week/season/team.

---

### 8. injuries
**Source:** NFLVerse API
**Rows:** 6,215
**Description:** Official injury reports

| Column Name | Data Type | Definition |
|-------------|-----------|------------|
| `season` | INTEGER | NFL season year |
| `week` | INTEGER | Week number |
| `team` | TEXT | Team abbreviation |
| `gsis_id` | TEXT | NFL GSIS player ID |
| `full_name` | TEXT | Player's full name |
| `position` | TEXT | Player position |
| `report_primary_injury` | TEXT | Primary injury designation |
| `report_status` | TEXT | Game status (Out, Questionable, Doubtful, etc.) |
| `practice_status` | TEXT | Practice participation level |
| `date_modified` | TIMESTAMP | Last update timestamp |

---

### 9. player_injuries
**Source:** Custom tracking
**Rows:** 15
**Description:** Manually tracked injury data for custom analysis

| Column Name | Data Type | Definition |
|-------------|-----------|------------|
| `injury_id` | INTEGER | Unique injury record ID |
| `player_name` | TEXT | Player name |
| `team_abbr` | TEXT | Team abbreviation |
| `season` | INTEGER | Season year |
| `injury_type` | TEXT | OUT, IR, DOUBTFUL, QUESTIONABLE |
| `start_week` | INTEGER | Week injury started |
| `end_week` | INTEGER | Expected return week |
| `injury_description` | TEXT | Additional notes |

---

### 10. plays
**Source:** NFLVerse API
**Rows:** 54,396
**Description:** Play-by-play data for each game

| Column Name | Data Type | Definition |
|-------------|-----------|------------|
| `game_id` | TEXT | NFLVerse game identifier |
| `play_index` | INTEGER | Sequential play number in game |
| `quarter` | INTEGER | Quarter (1-4, 5 for OT) |
| `time` | TEXT | Time remaining in quarter (MM:SS) |
| `down` | INTEGER | Down (1-4) |
| `yards_to_go` | INTEGER | Yards to first down |
| `yardline` | TEXT | Field position |
| `play_type` | TEXT | pass, run, kickoff, punt, etc. |
| `yards_gained` | INTEGER | Yards gained on play |
| `desc` | TEXT | Play description |

---

### 11. play_participants
**Source:** NFLVerse API
**Rows:** 120,500
**Description:** Player involvement in each play

| Column Name | Data Type | Definition |
|-------------|-----------|------------|
| `game_id` | TEXT | NFLVerse game identifier |
| `play_index` | INTEGER | Play number within game |
| `player_href` | TEXT | PFR player URL identifier |
| `player_name` | TEXT | Player name |
| `role` | TEXT | passer, rusher, receiver, tackler, etc. |

---

## 🔑 Key Relationships

### Joining Tables

**Player Stats + PFR Advanced Stats:**
```sql
SELECT ps.*, pap.*
FROM player_stats ps
LEFT JOIN pfr_advstats_pass_week pap
    ON ps.season = pap.season
    AND ps.week = pap.week
    AND ps.team = pap.team
    -- Note: Player name matching may require fuzzy logic
```

**Player Stats + Schedule (for home/away):**
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

**Team Stats + Schedules:**
```sql
SELECT ts.*, s.opponent, s.gameday
FROM team_stats_week ts
LEFT JOIN schedules s
    ON ts.season = s.season
    AND ts.week = s.week
    AND (ts.team = s.home_team OR ts.team = s.away_team)
```

---

## ⚠️ Common Pitfalls

### 1. Column Name Mismatches
❌ **Wrong:** `rec_air_yds`, `rec_yac`
✅ **Correct:** `receiving_air_yards`, `receiving_yards_after_catch`

### 2. Percentage Formats
- **PFR tables:** Store as decimals (0.0-1.0)
- **Display:** Multiply by 100 for percentage display

```python
# Example: Drop percentage display
drop_pct = 0.091  # From database
display = f"{drop_pct * 100:.1f}%"  # Shows "9.1%"
```

### 3. Missing game_id Column
- **player_stats** table does NOT have a `game_id` column
- Use composite keys: `season`, `week`, `team`, `opponent_team`

### 4. Home/Away Determination
- Don't try to merge on `game_id` from player_stats (doesn't exist)
- Use `schedules` table with composite key matching

### 5. Aggregation Level
- **Per-player averages:** AVG(stat_column)
- **Per-game totals:** AVG(SUM(stat_column) GROUP BY week)

```sql
-- ❌ Wrong: Averages individual player stats
SELECT AVG(receiving_yards) FROM player_stats WHERE opponent_team = 'KC'

-- ✅ Correct: Averages per-game team totals
SELECT AVG(game_yards) FROM (
    SELECT week, SUM(receiving_yards) as game_yards
    FROM player_stats
    WHERE opponent_team = 'KC'
    GROUP BY week
)
```

---

## 📝 Data Refresh Schedule

- **NFLVerse data:** Refreshed automatically via API calls
- **PFR scraped data:** Updated as needed
- **Merge metadata:** Tracks last refresh timestamp

Check `merge_metadata` table for last update time:
```sql
SELECT * FROM merge_metadata ORDER BY last_refresh_timestamp DESC LIMIT 1;
```

---

## 🛠️ Maintenance Notes

### Adding New Columns
1. Verify actual column name in database using: `PRAGMA table_info(table_name)`
2. Update this dictionary
3. Update any calculation functions in `pfr_viewer.py`
4. Test with actual data queries

### Schema Changes
- Log all changes in Git commit messages
- Update DATA_DICTIONARY.md
- Test affected calculations
- Push to GitHub for Streamlit Cloud deployment

---

## 📚 Additional Resources

- **NFLVerse Documentation:** https://nflverse.com
- **Pro Football Reference:** https://www.pro-football-reference.com
- **NFL GSIS Data Dictionary:** https://operations.nfl.com/stats-central/

---

**Document Version:** 1.0
**Created:** 2025-11-11
**Last Modified:** 2025-11-11
