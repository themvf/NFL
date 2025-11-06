# Database Merger Approach

## Overview

This NFL analytics application uses a **merged database architecture** that combines play-by-play data from Pro Football Reference with advanced statistics and schedules from NFLverse. This approach provides comprehensive NFL data while maintaining a single, unified database for the application.

### Why Merge Databases?

**Problem**: Two valuable but incompatible data sources:
- **Pro Football Reference (PFR)**: Detailed play-by-play data (20,303 plays) - NOT available in NFLverse
- **NFLverse**: Comprehensive schedules, rosters, advanced stats, betting odds - NOT available in PFR

**Solution**: Merge both into a single database (`nfl_merged.db`) that contains:
- ✅ Play-by-play granularity from PFR
- ✅ Advanced statistics from NFLverse
- ✅ Complete schedules with betting data
- ✅ Player rosters with multiple ID systems
- ✅ Defensive statistics
- ✅ User custom data (notes, projections, injuries)

---

## Database Inventory

### 1. pfr.db - Pro Football Reference Data

**Location**: `data/pfr.db`
**Size**: ~11.6 MB
**Update Frequency**: Manual (weekly)
**Data Source**: Custom web scraping from Pro Football Reference

**Contains**:
- **plays** (20,303 rows) - Play-by-play data with down, distance, yards gained
- **play_participants** - Player involvement in each play
- **player_box_score** - Traditional player statistics
- **team_game_summary** - Basic team statistics per game
- **user_notes** - Custom user annotations
- **player_injuries** - User-tracked injury data
- **player_transactions** - User-tracked roster moves
- **projection_accuracy** - Custom projection tracking

**Team Abbreviation Format**: PFR 3-letter codes
- Examples: GNB (Green Bay), KAN (Kansas City), NOR (New Orleans)

**Key Characteristic**: Contains play-by-play data NOT available anywhere else

---

### 2. nflverse.sqlite - NFLverse Official Data

**Location**: `C:\Docs\_AI Python Projects\NFL Data NFLVerse\NFL-Data\data\nflverse.sqlite`
**Size**: ~3.0 MB
**Update Frequency**: Daily (automated via GitHub Actions)
**Data Source**: NFLverse API (nflreadpy library)

**Contains**:
- **schedules** (272 games) - Complete season schedule with scores, betting odds
- **rosters** (3,100 players) - Player rosters with multiple ID systems
- **injuries** (6,215 entries) - Official NFL injury reports
- **team_stats_week** - Weekly team offensive & defensive statistics
- **pfr_advstats_pass_week** - Advanced passing stats (drops, pressure, blitzes)
- **pfr_advstats_rush_week** - Advanced rushing stats (yards before/after contact)
- **pfr_advstats_rec_week** - Advanced receiving stats (broken tackles, targets)
- **pfr_advstats_def_week** - Individual defensive player statistics

**Team Abbreviation Format**: Standard NFL 2-3 letter codes
- Examples: GB (Green Bay), KC (Kansas City), NO (New Orleans)

**Key Characteristics**:
- Automatically updated daily at 10:00 UTC
- Official NFL data, highly reliable
- Includes betting lines (spreads, moneylines, totals)
- Weather data (temperature, wind)
- Stadium information

---

### 3. nfl_merged.db - Unified Database (THIS IS WHAT THE APP USES)

**Location**: `data/nfl_merged.db`
**Size**: ~7.8 MB
**Update Frequency**: Manual refresh (via script or Streamlit UI)
**Data Source**: Merged from pfr.db + nflverse.sqlite

**Contains ALL tables from both sources**:
- All pfr.db tables (with converted team abbreviations)
- All nflverse.sqlite tables
- team_abbreviation_mapping - PFR ↔ NFLverse conversion table
- game_id_mapping (view) - Cross-reference game IDs between formats
- games (view) - Compatibility view mapping schedules to old format

**Team Abbreviation Format**: NFLverse standard (GB, KC, NO, etc.)

**Key Characteristics**:
- Single database for application simplicity
- Preserves custom user data (notes, projections)
- Uses NFLverse team abbreviations as standard
- Can be refreshed without losing custom data

---

## Merge Strategy

### Initial Merge Process

The initial merge is performed by `merge_databases.py`:

1. **Copy NFLverse as Base**
   ```
   nflverse.sqlite → nfl_merged.db
   ```

2. **Import PFR Tables**
   - plays (with team abbreviation conversion)
   - play_participants
   - user_notes
   - player_injuries
   - player_transactions
   - upcoming_games (empty table for future use)

3. **Create Supporting Tables**
   - team_abbreviation_mapping (8 team conversions)
   - projection_accuracy (custom projection tracking)

4. **Create Compatibility Views**
   - games (maps schedules table to old schema)
   - game_id_mapping (cross-reference between formats)

### Team Abbreviation Conversion

**8 teams have different abbreviations** between PFR and NFLverse:

| PFR (Old) | NFLverse (New) | Team Name |
|-----------|----------------|-----------|
| GNB | GB | Green Bay Packers |
| KAN | KC | Kansas City Chiefs |
| LAR | LA | Los Angeles Rams |
| LVR | LV | Las Vegas Raiders |
| NOR | NO | New Orleans Saints |
| NWE | NE | New England Patriots |
| SFO | SF | San Francisco 49ers |
| TAM | TB | Tampa Bay Buccaneers |

**24 teams remain unchanged**: ARI, ATL, BAL, BUF, CAR, CHI, CIN, CLE, DAL, DEN, DET, HOU, IND, JAX, LAC, MIA, MIN, NYG, NYJ, PHI, PIT, SEA, TEN, WAS

**Conversion Process**:
- All team references in imported PFR tables converted to NFLverse format
- Mapping table stores conversions for queries
- Application uses NFLverse abbreviations throughout

---

## Update Workflow

### Data Freshness Challenge

**Problem**: NFLverse data updates daily, but merged database becomes stale.

**What changes frequently** (needs refresh):
- schedules - Game scores added as games complete (DAILY during season)
- injuries - Injury reports change constantly (DAILY)
- rosters - Player signings/releases (WEEKLY)
- team_stats_week - New game statistics (WEEKLY)
- pfr_advstats_* - Advanced stats calculated post-game (WEEKLY)

**What changes rarely** (does NOT need refresh):
- plays - Only when you scrape new play-by-play data (WEEKLY/MANUAL)
- play_participants - Tied to plays table
- user_notes - Only when you add notes
- player_injuries - Only when you track injuries
- projection_accuracy - Only when you save projections

### Recommended Refresh Strategy

**Use incremental refresh**: Update only NFLverse tables, preserve PFR and custom data

**Advantages**:
- ✅ Fast (3 MB vs 7.8 MB)
- ✅ Safe (preserves custom data)
- ✅ Simple (one script)
- ✅ Can run from Streamlit UI

**When to refresh**:
- After NFLverse data updates (daily/weekly as needed)
- Before analyzing current week's games
- When you notice stale data (old scores, missing games)

---

## Scripts Reference

### merge_databases.py - Initial Merge Script

**Purpose**: Create nfl_merged.db from scratch

**Usage**:
```bash
python merge_databases.py
```

**What it does**:
1. Copies nflverse.sqlite as base
2. Imports pfr.db tables with team abbreviation conversion
3. Creates mapping tables and views
4. Verifies data integrity

**When to use**:
- First-time setup
- Complete database rebuild needed
- Testing merge process

**Time**: ~30-60 seconds

**⚠️ WARNING**: Deletes existing nfl_merged.db - any custom data added directly to merged DB will be lost

---

### refresh_merged_db.py - Incremental Refresh Script

**Purpose**: Update only NFLverse tables in merged database

**Usage**:
```bash
python refresh_merged_db.py
```

**Or from Streamlit**:
1. Navigate to "Database Manager"
2. Click "Refresh NFLverse Data" tab
3. Click "Refresh Now" button

**What it does**:
1. Creates backup (nfl_merged.db.backup)
2. Deletes old NFLverse data for current season
3. Copies fresh data from nflverse.sqlite
4. Preserves all PFR tables and custom data
5. Updates refresh timestamp

**Tables refreshed**:
- schedules
- rosters
- injuries
- team_stats_week
- pfr_advstats_pass_week
- pfr_advstats_rush_week
- pfr_advstats_rec_week
- pfr_advstats_def_week

**Tables preserved**:
- plays
- play_participants
- user_notes
- player_injuries
- player_transactions
- projection_accuracy

**Time**: ~3-5 seconds

**⚠️ SAFE**: Preserves custom data, creates automatic backup

---

### convert_team_abbrs.py - Team Abbreviation Converter

**Purpose**: Convert team abbreviations in code files

**Usage**:
```bash
python convert_team_abbrs.py
```

**What it does**:
- Scans pfr_viewer.py for old team abbreviations
- Replaces PFR format with NFLverse format
- Shows count of replacements made

**When to use**:
- After adding new queries with hardcoded team abbreviations
- Migrating old code to new abbreviation standard

**Time**: <1 second

---

## Detailed Refresh Workflow

### Step-by-Step: Refreshing Merged Database

#### Option A: Command Line

```bash
# 1. Navigate to project directory
cd "C:\Docs\_AI Python Projects\Cursor Projects\NFL - Copy"

# 2. Ensure NFLverse data is updated
# (You handle this separately - NFLverse repo updates daily)

# 3. Run refresh script
python refresh_merged_db.py

# Output example:
# ==========================================
# NFL MERGED DATABASE REFRESH
# ==========================================
#
# Creating backup...
# ✓ Backup saved: nfl_merged.db.backup
#
# Refreshing NFLverse tables for 2025 season...
#   schedules............... 272 rows updated
#   rosters................ 3,100 rows updated
#   injuries............... 6,215 rows updated
#   team_stats_week......... 270 rows updated
#   ...
#
# ✓ Refresh complete! Total updated: 15,511 rows

# 4. Commit updated database
git add data/nfl_merged.db
git commit -m "chore: refresh NFLverse data"
git push

# 5. Streamlit Cloud auto-deploys with fresh data
```

#### Option B: Streamlit UI (Recommended)

```
1. Open Streamlit app
2. Navigate to "Database Manager" in sidebar
3. Go to "Refresh NFLverse Data" tab
4. Click "Refresh Now" button
5. Watch progress bar and confirmation
6. Database automatically updated!
7. Commit changes via Git (manual or automated)
```

---

## Database Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                              │
├─────────────────────────────────┬───────────────────────────┤
│  pfr.db (Custom Scraping)       │  nflverse.sqlite (API)    │
│  - Play-by-play (20K plays)     │  - Schedules (272 games)  │
│  - User notes                   │  - Rosters (3.1K players) │
│  - Custom tracking              │  - Advanced stats         │
│  Update: Weekly (manual)        │  Update: Daily (auto)     │
│  Team format: GNB, KAN, NOR     │  Team format: GB, KC, NO  │
└─────────────────────────────────┴───────────────────────────┘
                          │
                          │ merge_databases.py
                          │ (initial merge)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              nfl_merged.db (UNIFIED DATABASE)                │
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────┐        │
│  │  PFR Tables          │  │  NFLverse Tables     │        │
│  │  (Preserved)         │  │  (Refreshable)       │        │
│  ├──────────────────────┤  ├──────────────────────┤        │
│  │ • plays              │  │ • schedules          │        │
│  │ • play_participants  │  │ • rosters            │        │
│  │ • user_notes         │  │ • injuries           │        │
│  │ • player_injuries    │  │ • team_stats_week    │        │
│  │ • player_transactions│  │ • pfr_advstats_*     │        │
│  │ • projection_accuracy│  │                      │        │
│  └──────────────────────┘  └──────────────────────┘        │
│                                                              │
│  Team Format: NFLverse standard (GB, KC, NO, etc.)          │
│  Size: 7.8 MB                                               │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ refresh_merged_db.py
                          │ (incremental refresh)
                          ▼
                  Updates NFLverse tables only
                  Preserves PFR tables intact
```

---

## Troubleshooting

### Issue: "no such table: schedules"

**Cause**: Application pointing to old pfr.db instead of nfl_merged.db

**Solution**:
```python
# Check pfr_viewer.py line 29
DB_PATH = Path(__file__).parent / "data" / "nfl_merged.db"  # ✓ Correct
# NOT:
DB_PATH = Path(__file__).parent / "data" / "pfr.db"  # ✗ Wrong
```

### Issue: "Team abbreviation not found"

**Cause**: Query using old PFR abbreviations (GNB, KAN, etc.)

**Solution**:
```bash
# Run conversion script
python convert_team_abbrs.py

# Or manually update:
# GNB → GB, KAN → KC, NOR → NO, etc.
```

### Issue: "Merged database is stale"

**Cause**: NFLverse data updated but merged DB not refreshed

**Solution**:
```bash
# Refresh via command line
python refresh_merged_db.py

# Or via Streamlit UI:
# Database Manager → Refresh NFLverse Data → Refresh Now
```

### Issue: "Refresh script fails mid-update"

**Cause**: Database locked, schema mismatch, or connection error

**Solution**:
```bash
# Restore from automatic backup
cp data/nfl_merged.db.backup data/nfl_merged.db

# Or restore from Git
git checkout HEAD -- data/nfl_merged.db

# Check error message for specific issue
```

### Issue: "Custom data disappeared after merge"

**Cause**: Used merge_databases.py instead of refresh_merged_db.py

**Prevention**:
- Use `refresh_merged_db.py` for updates (preserves custom data)
- Use `merge_databases.py` only for initial setup

**Recovery**:
```bash
# Restore from backup
cp data/nfl_merged.db.backup data/nfl_merged.db

# Or restore from Git history
git log --all -- data/nfl_merged.db
git checkout <commit-hash> -- data/nfl_merged.db
```

---

## Best Practices

### ✅ DO

- **Use refresh_merged_db.py** for regular updates
- **Commit merged DB to Git** after refreshes
- **Check refresh status** before analysis (Database Manager UI)
- **Keep backups** of important custom data
- **Test locally** before pushing to Streamlit Cloud
- **Document custom queries** that use team abbreviations

### ❌ DON'T

- **Don't use merge_databases.py** for regular updates (loses custom data)
- **Don't add data directly** to nfl_merged.db (add to pfr.db instead)
- **Don't mix team abbreviations** in queries (use NFLverse format)
- **Don't skip backups** before manual database operations
- **Don't commit both** pfr.db AND nfl_merged.db (too large, redundant)

---

## Future Enhancements

### Potential Improvements

1. **Automated Refresh via GitHub Actions**
   - Daily workflow refreshes merged DB
   - Auto-commits to repository
   - Streamlit Cloud auto-deploys

2. **Streamlit Cloud Integration**
   - Download latest nflverse.sqlite from NFLverse repo
   - Refresh on-demand via UI button
   - Upload refreshed DB to Google Cloud Storage

3. **Multi-Season Support**
   - Refresh specific seasons (2024, 2025, etc.)
   - Historical data preservation
   - Season selector in UI

4. **Advanced Validation**
   - Compare row counts vs expected
   - Detect schema changes
   - Alert on data anomalies

5. **Incremental PFR Updates**
   - Merge new play-by-play data without full rebuild
   - Preserve existing plays
   - Add only new games

---

## References

### Related Documentation

- **MIGRATION_SUMMARY.md** - Complete migration history and details
- **README.md** - Project overview and setup instructions
- **merge_databases.py** - Initial merge script with inline documentation
- **refresh_merged_db.py** - Refresh script with inline documentation

### External Resources

- [NFLverse Documentation](https://nflverse.nflverse.com/)
- [nflreadpy Library](https://github.com/nflverse/nflreadpy)
- [Pro Football Reference](https://www.pro-football-reference.com/)

### Database Schemas

To view complete table schemas:

```bash
# View nfl_merged.db schema
python -c "import sqlite3; conn = sqlite3.connect('data/nfl_merged.db'); \
  cursor = conn.cursor(); cursor.execute('PRAGMA table_info(schedules)'); \
  print('\\n'.join([f'{row[1]} ({row[2]})' for row in cursor.fetchall()]))"
```

---

## Version History

- **v1.0** (2025-11-05): Initial database merge
  - Merged pfr.db + nflverse.sqlite
  - Created compatibility views
  - Converted team abbreviations

- **v1.1** (2025-11-05): Added refresh capability
  - Created refresh_merged_db.py
  - Added Database Manager UI
  - Implemented backup/restore

---

## Contact & Support

For issues related to:
- **Database merge**: Check this document
- **Refresh scripts**: See inline script documentation
- **NFLverse data**: Visit [NFLverse GitHub](https://github.com/nflverse)
- **Application errors**: Check Streamlit logs

---

**Last Updated**: November 5, 2025
**Maintained By**: Database merger scripts and documentation
**Version**: 1.1
