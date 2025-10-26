from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

try:
    import nfl_data_py as nfl
except Exception as e:
    raise SystemExit(f"Failed to import nfl_data_py: {e}")


class IngestConfig(BaseModel):
    seasons: List[int] = Field(default_factory=list)
    weeks: Optional[List[int]] = None
    data_dir: Path = Field(default_factory=lambda: Path("data/processed"))
    as_of: Optional[datetime] = None


@dataclass
class IngestResult:
    schedules_rows: int
    player_week_rows: int
    injuries_rows: int


def _write_partitioned(
    df: pd.DataFrame,
    base: Path,
    table: str,
    season_col: str = "season",
    week_col: Optional[str] = "week",
) -> None:
    base = Path(base) / table
    base.mkdir(parents=True, exist_ok=True)

    if df.empty:
        logger.warning(f"No rows for {table}; skipping write.")
        return

    required = [season_col] + ([week_col] if week_col else [])
    for col in required:
        if col and col not in df.columns:
            raise ValueError(f"Missing required column {col} in {table}")

    group_cols = [season_col] + ([week_col] if week_col else [])
    for keys, part in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        season = keys[0]
        week = keys[1] if week_col else None
        subdir = base / f"season={season}"
        if week_col:
            week_val = "na" if pd.isna(week) else int(week)
            subdir = subdir / f"week={week_val}"
        subdir.mkdir(parents=True, exist_ok=True)
        out = subdir / f"{table}.parquet"
        part.to_parquet(out, index=False)
        logger.info(f"Wrote {len(part):,} rows to {out}")


def _fetch_weekly_safe(seasons: List[int]) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for season in seasons:
        try:
            df = nfl.import_weekly_data([season])
            parts.append(df)
        except Exception as e:
            logger.error(f"Failed to fetch weekly data for season {season}: {e}")
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _filter_weeks(df: pd.DataFrame, weeks: Optional[List[int]]) -> pd.DataFrame:
    if not weeks or df.empty or "week" not in df.columns:
        return df
    return df[df["week"].isin(weeks)].copy()


def fetch_and_store(cfg: IngestConfig) -> IngestResult:
    seasons = cfg.seasons or nfl.see_seasons()[-2:]
    logger.info(f"Fetching data for seasons: {seasons}")

    try:
        schedules = nfl.import_schedules(seasons)
    except Exception as e:
        logger.error(f"Failed to fetch schedules for {seasons}: {e}")
        schedules = pd.DataFrame()

    player_week = _fetch_weekly_safe(seasons)

    try:
        injuries = nfl.import_injuries(seasons)
    except Exception as e:
        logger.error(f"Failed to fetch injuries for {seasons}: {e}")
        injuries = pd.DataFrame()

    # Optional week filtering for partial seasons (e.g., only weeks 1-4)
    schedules = _filter_weeks(schedules, cfg.weeks)
    player_week = _filter_weeks(player_week, cfg.weeks)
    injuries = _filter_weeks(injuries, cfg.weeks)

    for df in (schedules, player_week, injuries):
        df.columns = [c.lower() for c in df.columns]

    _write_partitioned(schedules, cfg.data_dir, "schedule", "season", "week")
    _write_partitioned(player_week, cfg.data_dir, "player_week", "season", "week")
    _write_partitioned(injuries, cfg.data_dir, "injuries", "season", "week")

    return IngestResult(
        schedules_rows=len(schedules),
        player_week_rows=len(player_week),
        injuries_rows=len(injuries),
    )
