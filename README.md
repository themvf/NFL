# NFL Data Browser – First TD Play-by-Play (PBP) Notes

## Summary
The app primarily uses locally ingested, partitioned Parquet data under `data/processed/`. Most sections (e.g., Team Overview/Comparison, Players by Position) rely on `schedule/` and `player_week/` and work fully offline.

The two "First TD" sections ("First TD Grid" and "First TD") require play-by-play (PBP) data to determine the first offensive touchdown in each game. Historically these sections pulled PBP via `nfl_data_py.import_pbp_data([season])` from the internet.

Recently, two issues surfaced when requesting 2025 PBP:
- Remote PBP endpoint can return HTTP 404 before the provider publishes data
- A bug in `nfl_data_py` references an undefined `Error` class, and in some cases leads to `UnboundLocalError: cannot access local variable 'plays'` internally

## Current Fix
We implemented a robust, production-friendly approach that keeps the original logic:

1) Local-first PBP loader
- Attempts to read local PBP Parquet files if present:
  - `data/processed/pbp/season={YYYY}/week=*/pbp.parquet`
- If found, the app uses this local data and avoids the internet entirely

2) Patched remote fallback
- If local PBP is not available, the loader falls back to `nfl_data_py.import_pbp_data([season])`
- At runtime, we inject a missing `Error` class into the module to avoid the library’s `name 'Error' is not defined` crash
- We catch library exceptions (including `UnboundLocalError`) and return an empty DataFrame gracefully so the UI shows a helpful message instead of crashing

This keeps the original First TD logic intact while making the data source resilient and user-friendly.

3) Last-resort remote parquet
- If the library call fails (e.g., HTTP 404), we attempt known nflverse parquet URLs directly. If those are also unavailable, the UI shows a fallback view based on `player_week` so the page always renders something useful.

4) Preserved fallbacks (no manual steps)
- For "First TD Grid": when PBP is unavailable, we render a TD-by-week grid from `player_week` (RB/WR/TE only; QBs excluded). This is not a true "first TD" order but provides actionable information.
- For "First TD": when PBP is unavailable, we list weekly TD scorers (RB/WR/TE only) for the selected week. If the selected team has none, we still show league-wide results so the table isn't blank.

## Recommended Workflow
- Prefer local data for reliability and speed
- If you need First TD sections for a season where remote PBP isn’t published yet, ingest PBP locally and store it under `data/processed/pbp/season=YYYY/week=K/pbp.parquet`
- If you do not maintain a PBP ingest pipeline yet, the app will still operate and the First TD sections will degrade gracefully when remote PBP is unavailable

We will preserve these fallbacks permanently so production never breaks when upstream PBP is delayed.

## Weekly Data Updates (No Manual Steps)

1) Ingest schedule and player-week
- Use the existing CLI (examples):
  - `python scripts/run_ingest.py --season 2025 --all-weeks`
  - Or weekly: `python scripts/run_ingest.py --season 2025 --weeks 1-4`
- Output lands in `data/processed/schedule/` and `data/processed/player_week/`

2) Ingest (or auto-cache) play-by-play (PBP)
- When available upstream, the app will attempt to fetch PBP and will automatically cache/use it locally the next time you run an ingest step that saves PBP under:
  - `data/processed/pbp/season=YYYY/week=*/pbp.parquet`
- Until then, the app will:
  - Prefer local PBP if present
  - Fall back to remote via `nfl_data_py`
  - Try known nflverse parquet URLs
  - Finally, present TD fallbacks from `player_week` so views still render

3) App behavior the next week
- Simply re-run the ingest for the new week. All pages will automatically include the new weeks with no code changes.

Notes
- All tables in the app are native Streamlit tables (sortable). First TD fallbacks show RB/WR/TE only; QBs appear only when they rush or catch a TD.

## Future Enhancements (Optional)
- Add a dedicated PBP ingest step alongside `schedule` and `player_week` so all views work offline for any season once ingested
- Consider caching remote PBP on first fetch (when available) to `data/processed/pbp/season=YYYY/`

## What Changed in Code
- Introduced a helper to load PBP with local-first and patched-remote fallback
- Wired the helper into the two First TD sections
- Preserved the original TD identification and display logic

No changes were made to the aggregated player/position tables or predictions; those continue to use local Parquet inputs as before.

## Data Sources and Update Flow (First TD specifics)

The app prefers local Parquet and falls back to upstream sources only when needed.

Local directories (preferred)
- Schedule: `data/processed/schedule/season={YEAR}/week=*/schedule.parquet`
- Player-week: `data/processed/player_week/season={YEAR}/week=*/player_week.parquet`
- Play-by-play (PBP): `data/processed/pbp/season={YEAR}/week=*/pbp.parquet`

Remote fallbacks currently used by the app for PBP
1) `nfl_data_py.import_pbp_data([season])` (season parquet)
2) Known nflverse/nflfastR season parquet URLs (season-level):
   - `https://github.com/nflverse/nflfastR-data/releases/download/pbp/play_by_play_{SEASON}.parquet`
   - `https://github.com/nflverse/nflfastR-data/releases/download/play_by_play_{SEASON}/play_by_play_{SEASON}.parquet`
   - `https://raw.githubusercontent.com/nflverse/nflfastR-data/master/data/play_by_play_{SEASON}.parquet`

Why Week 3 may appear in other pages but not in First TD
- Skill/Team pages use `player_week` and `schedule`, which are already local for 2025.
- First TD needs PBP; if the season parquet above is not yet updated/published, First TD stays on fallback.

Additional upstream to review for PBP
- The `nfldata` repo (nflverse) hosts partitioned data, potentially including PBP by season/week: `https://github.com/nflverse/nfldata`. We plan to extend the loader to also probe partitioned PBP there so First TD can update sooner when weekly parquet is available. See [nfldata](https://github.com/nflverse/nfldata).

Buttons on the Data Updates view
- Check Remote: probes the same season parquet used by `nfl_data_py` and reports availability.
- Ingest PBP: once available upstream, writes local `data/processed/pbp/season={YEAR}/week={WEEK}/pbp.parquet` and First TD switches to true ordering on refresh.
- Refresh Coverage: re-scans local coverage for schedule/player_week/pbp.

## Weekly Update Playbook (What to do every week)

1) Ingest schedule and player-week (local)
- Run your existing ingest for the latest week. This populates:
  - `data/processed/schedule/season={YEAR}/week={WEEK}/schedule.parquet`
  - `data/processed/player_week/season={YEAR}/week={WEEK}/player_week.parquet`
- The Overview/Comparison/Skill grids will immediately include the new week.

2) Get Play-by-Play (PBP) for First TD pages
- Preferred (season parquet via nfl-data-py):
  - When published upstream, run `nfl.cache_pbp([YEAR], alt_path=<your_cache>)` and the app can load from cache.
  - Docs: [nfl-data-py](https://pypi.org/project/nfl-data-py/)
- Alternative (when the season parquet lags): probe weekly partitioned PBP under nflverse’s `nfldata` repo and ingest that week only into your local PBP:
  - Expected layout: `data/pbp/season={YEAR}/week={WEEK}/pbp.parquet`
  - Example raw path (verify existence in your browser first):
    - `https://raw.githubusercontent.com/nflverse/nfldata/master/data/pbp/season=2025/week=4/pbp.parquet`
  - Repo reference: [nfldata](https://github.com/nflverse/nfldata)

3) Use the app’s Data Updates view
- Check Remote: confirms PBP availability (season parquet today; see Notes below).
- Ingest PBP: downloads the week and writes it locally to `data/processed/pbp/season={YEAR}/week={WEEK}/pbp.parquet`.
- Refresh Coverage: re-scans local folders and shows the weeks now present.

Notes about sources and timing
- Season parquet (single file per year) sometimes publishes after the weekly partition does. That’s why Skill/Team pages (player_week) can show Week N while First TD (PBP) cannot yet.
- To minimize waiting, prefer the weekly partition under `nfldata` when the season parquet isn’t updated yet.

## If Week 4 isn’t appearing (checklist)
- Open View → Data Updates and confirm `schedule_weeks` and `player_week_weeks` contain 4. If not, rerun your weekly ingest.
- For First TD pages, confirm `pbp_weeks` contains 4. If not:
  1) Try “Check Remote” and “Ingest PBP” again later (season parquet route).
  2) Alternative: directly ingest weekly PBP from `nfldata` (partitioned path). Once written to `data/processed/pbp/season=2025/week=4/`, First TD switches to true ordering on refresh.

## Why we used these tools
- `nfl-data-py` is the canonical Python client for nflverse data and supports PBP caching for a whole season. See [nfl-data-py](https://pypi.org/project/nfl-data-py/).
- `nfldata` provides partitioned parquet (by season/week), which often becomes available sooner. See [nfldata](https://github.com/nflverse/nfldata).
- The combination lets you update First TD as soon as a per‑week parquet exists, then later rely on the season parquet + cache for stable, fast access.


