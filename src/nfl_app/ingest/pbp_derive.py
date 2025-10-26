from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

try:
    import nfl_data_py as nfl
except Exception as e:
    raise SystemExit(f"Failed to import nfl_data_py: {e}")


@dataclass
class DeriveConfig:
    season: int
    weeks: Optional[List[int]]
    out_dir: Path


def _write_partitioned(df: pd.DataFrame, base: Path, table: str) -> None:
    base = Path(base) / table
    base.mkdir(parents=True, exist_ok=True)
    if df.empty:
        logger.warning(f"No rows for {table}; skipping write.")
        return
    for (season, week), part in df.groupby(["season", "week"], dropna=False):
        subdir = base / f"season={int(season)}" / f"week={int(week)}"
        subdir.mkdir(parents=True, exist_ok=True)
        out = subdir / f"{table}.parquet"
        part.to_parquet(out, index=False)
        logger.info(f"Wrote {len(part):,} rows to {out}")


def _safe_sum(df: pd.DataFrame, group_cols: list, cols: list) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    if not present:
        return df.groupby(group_cols, as_index=False).size().rename(columns={"size": "plays"})
    return df.groupby(group_cols, as_index=False)[present].sum()


def derive_player_week_from_pbp(cfg: DeriveConfig) -> pd.DataFrame:
    # Pull one season and filter weeks early
    pbp = nfl.import_pbp_data([cfg.season])
    if cfg.weeks:
        pbp = pbp[pbp["week"].isin(cfg.weeks)]

    # Normalize columns to lowercase for safety
    pbp.columns = [c.lower() for c in pbp.columns]

    # Passing aggregates
    pass_df = pbp[pbp.get("pass_attempt", 0) == 1].copy()
    # Identify targets via presence of receiver id on pass attempts
    pass_df["is_target"] = pass_df["receiver_player_id"].notna().astype(int)
    pass_agg = _safe_sum(
        pass_df,
        ["season", "week", "posteam", "passer_player_id", "passer_player_name"],
        [
            "pass_attempt",
            "complete_pass",
            "passing_yards",
            "interception",
            "sack",
            "sack_yards",
            "pass_touchdown",
            "is_target",
        ],
    )
    if not pass_agg.empty:
        pass_agg = pass_agg.rename(
            columns={
                "posteam": "team",
                "passer_player_id": "player_id",
                "passer_player_name": "player_name",
                "complete_pass": "completions",
                "passing_yards": "pass_yards",
                "interception": "interceptions",
                "sack": "sacks",
                "sack_yards": "sack_yards",
                "pass_touchdown": "pass_td",
                "is_target": "targets",
                "pass_attempt": "attempts",
            }
        )
        pass_agg["role"] = "QB"

    # Rushing aggregates
    rush_df = pbp[pbp.get("rush_attempt", 0) == 1].copy()
    rush_agg = _safe_sum(
        rush_df,
        ["season", "week", "posteam", "rusher_player_id", "rusher_player_name"],
        ["rush_attempt", "rushing_yards", "rush_touchdown", "fumble"]
    )
    if not rush_agg.empty:
        rush_agg = rush_agg.rename(
            columns={
                "posteam": "team",
                "rusher_player_id": "player_id",
                "rusher_player_name": "player_name",
                "rush_attempt": "carries",
                "rushing_yards": "rush_yards",
                "rush_touchdown": "rush_td",
                "fumble": "fumbles",
            }
        )
        rush_agg["role"] = "RB"

    # Receiving aggregates: use presence of receiver id; receptions from complete_pass
    rec_df = pbp[pbp["receiver_player_id"].notna()].copy()
    # Count targets at pass level; receptions where complete_pass==1
    rec_df["receptions"] = (rec_df.get("complete_pass", 0) == 1).astype(int)
    rec_df["targets"] = 1
    # Receiving yards equals passing_yards when receiver present and completed; on incompletions, yards 0
    rec_df["rec_yards"] = rec_df.get("passing_yards", 0).fillna(0)
    rec_df["rec_td"] = rec_df.get("pass_touchdown", 0)
    rec_agg = _safe_sum(
        rec_df,
        ["season", "week", "posteam", "receiver_player_id", "receiver_player_name"],
        ["targets", "receptions", "rec_yards", "rec_td"]
    )
    if not rec_agg.empty:
        rec_agg = rec_agg.rename(
            columns={
                "posteam": "team",
                "receiver_player_id": "player_id",
                "receiver_player_name": "player_name",
            }
        )
        rec_agg["role"] = "WR/TE"

    frames = [df for df in [pass_agg, rush_agg, rec_agg] if not df.empty]
    if not frames:
        logger.warning("No derived player-week rows from PBP.")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    # Ensure dtypes
    for col in ["season", "week"]:
        out[col] = out[col].astype("int64")
    # Write partitioned as player_week to align with app
    _write_partitioned(out, cfg.out_dir, "player_week")
    return out



