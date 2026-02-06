from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import re

import numpy as np
import pandas as pd
import streamlit as st
import sqlite3


DATA_BASE = Path("data/processed").resolve()
DB_PATH = Path("data/nfl_merged.db").resolve()

TEAM_ALIASES = {
    "JAC": "JAX",
    "JAX": "JAX",
    "LAR": "LA",
    "LA": "LA",
    "LAC": "LAC",
    "LVR": "LV",
    "LV": "LV",
    "WSH": "WAS",
    "WAS": "WAS",
    "GNB": "GB",
    "GB": "GB",
    "KAN": "KC",
    "KC": "KC",
    "SFO": "SF",
    "SF": "SF",
    "TAM": "TB",
    "TB": "TB",
    "NWE": "NE",
    "NE": "NE",
    "NOR": "NO",
    "NO": "NO",
}

NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_team(team: str) -> str:
    if team is None:
        return ""
    t = str(team).strip().upper()
    return TEAM_ALIASES.get(t, t)


def normalize_name(name: str) -> str:
    if name is None:
        return ""
    txt = str(name).lower().strip()
    txt = re.sub(r"[^a-z0-9 ]+", " ", txt)
    parts = [p for p in txt.split() if p and p not in NAME_SUFFIXES]
    return "".join(parts)


def split_name_tokens(name: str) -> Tuple[str, str]:
    if name is None:
        return "", ""
    txt = str(name).lower().strip()
    txt = re.sub(r"[^a-z0-9 ]+", " ", txt)
    parts = [p for p in txt.split() if p and p not in NAME_SUFFIXES]
    if not parts:
        return "", ""
    return parts[0], parts[-1]


def _pick_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    col_set = {c.lower(): c for c in columns}
    for c in candidates:
        if c in columns:
            return c
        c_lower = c.lower()
        if c_lower in col_set:
            return col_set[c_lower]
    return None


def _list_available_weeks(table: str, season: int) -> List[int]:
    season_dir = DATA_BASE / table / f"season={season}"
    if not season_dir.exists():
        return []
    weeks: List[int] = []
    for p in season_dir.glob("week=*"):
        wk = p.name.split("=", 1)[1]
        if wk.isdigit():
            weeks.append(int(wk))
    return sorted(set(weeks))


def _read_player_week(season: int, weeks: Sequence[int]) -> pd.DataFrame:
    if not weeks:
        return pd.DataFrame()
    cols = [
        "player_name",
        "player_display_name",
        "position",
        "recent_team",
        "team",
        "season",
        "week",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "passing_2pt_conversions",
        "rushing_yards",
        "rushing_tds",
        "rushing_2pt_conversions",
        "receiving_yards",
        "receiving_tds",
        "receiving_2pt_conversions",
        "receptions",
        "special_teams_tds",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
        "sack_fumbles_lost",
        "fantasy_points",
        "fantasy_points_ppr",
    ]
    parts = []
    for wk in weeks:
        path = DATA_BASE / f"player_week/season={season}/week={wk}/player_week.parquet"
        if path.exists():
            try:
                parts.append(pd.read_parquet(path, columns=cols))
            except Exception:
                parts.append(pd.read_parquet(path))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    return df


def _compute_dk_points(df: pd.DataFrame) -> pd.Series:
    def col_any(names: Sequence[str]) -> pd.Series:
        for name in names:
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
        return pd.Series([0.0] * len(df), index=df.index)

    passing_yards = col_any(["passing_yards"])
    passing_tds = col_any(["passing_tds"])
    interceptions = col_any(["interceptions", "passing_interceptions"])
    pass_2pt = col_any(["passing_2pt_conversions"])

    rushing_yards = col_any(["rushing_yards"])
    rushing_tds = col_any(["rushing_tds"])
    rush_2pt = col_any(["rushing_2pt_conversions"])

    receiving_yards = col_any(["receiving_yards"])
    receiving_tds = col_any(["receiving_tds"])
    rec_2pt = col_any(["receiving_2pt_conversions"])
    receptions = col_any(["receptions"])

    special_tds = col_any(["special_teams_tds"])

    fumbles_lost = (
        col_any(["rushing_fumbles_lost"])
        + col_any(["receiving_fumbles_lost"])
        + col_any(["sack_fumbles_lost"])
    )

    pass_bonus = (passing_yards >= 300).astype(float) * 3.0
    rush_bonus = (rushing_yards >= 100).astype(float) * 3.0
    rec_bonus = (receiving_yards >= 100).astype(float) * 3.0

    return (
        passing_yards * 0.04
        + passing_tds * 4.0
        + pass_bonus
        - interceptions
        + rushing_yards * 0.1
        + rushing_tds * 6.0
        + rush_bonus
        + receiving_yards * 0.1
        + receiving_tds * 6.0
        + rec_bonus
        + receptions
        + special_tds * 6.0
        + (pass_2pt + rush_2pt + rec_2pt) * 2.0
        - fumbles_lost
    )


def _read_player_stats_db(season: int, weeks: Sequence[int]) -> pd.DataFrame:
    if not DB_PATH.exists() or not weeks:
        return pd.DataFrame()
    cols = [
        "player_display_name",
        "player_name",
        "position",
        "team",
        "season",
        "week",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "passing_2pt_conversions",
        "rushing_yards",
        "rushing_tds",
        "rushing_2pt_conversions",
        "receiving_yards",
        "receiving_tds",
        "receiving_2pt_conversions",
        "receptions",
        "special_teams_tds",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
        "sack_fumbles_lost",
    ]
    conn = sqlite3.connect(str(DB_PATH))
    placeholder = ",".join("?" for _ in weeks)
    query = f"""
        SELECT {",".join(cols)}
        FROM player_stats
        WHERE season = ?
          AND week IN ({placeholder})
          AND season_type IN ('REG', 'POST')
    """
    params = [season] + list(weeks)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    df.columns = [c.lower() for c in df.columns]
    return df


def _list_db_weeks(season: int) -> List[int]:
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT week FROM player_stats WHERE season = ? ORDER BY week", (season,))
    weeks = [int(r[0]) for r in cur.fetchall() if r[0] is not None]
    conn.close()
    return weeks


def _load_histories(
    season: int,
    week: int,
    recent_weeks: int,
    use_full_season: bool,
    source: str,
) -> pd.DataFrame:
    if source == "db":
        weeks_all = _list_db_weeks(season)
    else:
        weeks_all = _list_available_weeks("player_week", season)
    if not weeks_all:
        return pd.DataFrame()

    if use_full_season:
        weeks = [w for w in weeks_all if w <= week]
    else:
        past = [w for w in weeks_all if w <= week]
        weeks = past[-recent_weeks:] if past else []

    if not weeks:
        return pd.DataFrame()

    if source == "db":
        df = _read_player_stats_db(season, weeks)
    else:
        df = _read_player_week(season, weeks)
    if df.empty:
        return pd.DataFrame()

    df["dk_points"] = _compute_dk_points(df)
    name_col = "player_display_name" if "player_display_name" in df.columns else "player_name"
    team_col = "recent_team" if "recent_team" in df.columns else "team"
    df["name_key"] = df[name_col].astype(str).map(normalize_name)
    df["team_key"] = df[team_col].astype(str).map(normalize_team)
    if "position" not in df.columns:
        df["position"] = ""
    df = df[df["name_key"] != ""]
    return df[["name_key", "team_key", "position", "dk_points"]].copy()


def _simulate_lineups(
    lineups: List[pd.DataFrame],
    pool: pd.DataFrame,
    history_df: pd.DataFrame,
    sims: int,
    top_pct: float,
    force_kdst_dk_avg: bool,
    replace_zero_with_dk_avg: bool,
) -> Tuple[List[float], List[float]]:
    rng = np.random.default_rng(42)

    hist_map: Dict[Tuple[str, str], List[float]] = {}
    pos_map: Dict[str, List[float]] = {}
    league_hist: List[float] = []
    if not history_df.empty:
        history_df["dk_points"] = pd.to_numeric(history_df["dk_points"], errors="coerce").fillna(0.0)
        league_hist = history_df["dk_points"].tolist()
        for (name_key, team_key), group in history_df.groupby(["name_key", "team_key"]):
            hist_map[(name_key, team_key)] = group["dk_points"].tolist()
        for pos, group in history_df.groupby("position"):
            pos_map[str(pos).upper()] = group["dk_points"].tolist()

    pool_index = pool.set_index("player_key")

    def sample_player(base_id: str) -> np.ndarray:
        row = pool_index.loc[base_id]
        pos = str(row["position"]).upper()
        mean = float(row.get("proj_points", 0.0))
        dk_avg = float(row.get("avg_points", 0.0))

        is_kdst = pos in {"K", "DST", "D/ST", "DEF"}
        if force_kdst_dk_avg:
            if replace_zero_with_dk_avg:
                if dk_avg > 0 and mean <= 0:
                    mean = dk_avg
            else:
                if dk_avg > 0:
                    mean = dk_avg
            if is_kdst:
                std = 3.0 if pos == "K" else 4.5
                samples = rng.normal(mean, std, size=sims)
                return np.clip(samples, -5, None)

        key = (row["name_key"], row["team"])
        hist = hist_map.get(key, [])
        if len(hist) >= 3:
            return rng.choice(hist, size=sims, replace=True)
        pos_hist = pos_map.get(pos, [])
        if len(pos_hist) >= 10:
            return rng.choice(pos_hist, size=sims, replace=True)
        if league_hist:
            return rng.choice(league_hist, size=sims, replace=True)
        std = max(3.0, mean * 0.5)
        samples = rng.normal(mean, std, size=sims)
        return np.clip(samples, -5, None)

    metrics: List[float] = []
    means: List[float] = []

    top_n = max(1, int(math.ceil(sims * (top_pct / 100.0))))

    for lineup in lineups:
        scores = np.zeros(sims)
        for _, row in lineup.iterrows():
            samples = sample_player(row["base_id"])
            if row["is_captain"]:
                scores += samples * 1.5
            else:
                scores += samples
        means.append(float(np.mean(scores)))
        top_scores = np.partition(scores, -top_n)[-top_n:]
        metrics.append(float(np.mean(top_scores)))

    return metrics, means


def _build_stat_projections(
    season: int,
    week: int,
    recent_weeks: int,
    use_full_season: bool,
    source: str,
) -> Dict[str, pd.DataFrame]:
    if source == "db":
        weeks_all = _list_db_weeks(season)
    else:
        weeks_all = _list_available_weeks("player_week", season)
    if not weeks_all:
        return {}

    if use_full_season:
        weeks = [w for w in weeks_all if w <= week]
    else:
        past = [w for w in weeks_all if w <= week]
        weeks = past[-recent_weeks:] if past else []

    if source == "db":
        df = _read_player_stats_db(season, weeks)
    else:
        df = _read_player_week(season, weeks)
    if df.empty:
        return {}

    df["dk_points"] = _compute_dk_points(df)
    name_col = "player_display_name" if "player_display_name" in df.columns else "player_name"
    team_col = "recent_team" if "recent_team" in df.columns else "team"

    df["full_name"] = df[name_col].astype(str)
    df["name_key"] = df["full_name"].map(normalize_name)
    df["team_key"] = df[team_col].astype(str).map(normalize_team)
    tokens = df["full_name"].map(split_name_tokens)
    df["first_name"] = tokens.map(lambda x: x[0])
    df["last_name"] = tokens.map(lambda x: x[1])
    df["last_key"] = df["last_name"].map(normalize_name)
    df["first_last_key"] = (df["first_name"].fillna("") + " " + df["last_name"].fillna("")).map(normalize_name)
    df["first_initial_last_key"] = (
        df["first_name"].str[:1].fillna("") + " " + df["last_name"].fillna("")
    ).map(normalize_name)
    df = df[df["name_key"] != ""]

    agg_name_team = df.groupby(["name_key", "team_key"], as_index=False).agg(
        avg_dk_points=("dk_points", "mean"),
        games=("dk_points", "size"),
    )

    unique_players = df[["name_key", "first_last_key", "first_initial_last_key", "last_key", "team_key"]].drop_duplicates()
    counts_first_last = unique_players.groupby(["first_last_key", "team_key"]).size().reset_index(name="count")
    counts_first_initial_last = (
        unique_players.groupby(["first_initial_last_key", "team_key"]).size().reset_index(name="count")
    )
    counts_last = unique_players.groupby(["last_key", "team_key"]).size().reset_index(name="count")
    counts_name = unique_players.groupby(["name_key"]).size().reset_index(name="count")

    agg_first_last = df.groupby(["first_last_key", "team_key"], as_index=False).agg(
        avg_dk_points=("dk_points", "mean"),
        games=("dk_points", "size"),
    )
    agg_first_last = agg_first_last.merge(
        counts_first_last[counts_first_last["count"] == 1][["first_last_key", "team_key"]],
        on=["first_last_key", "team_key"],
        how="inner",
    )

    agg_first_initial_last = df.groupby(["first_initial_last_key", "team_key"], as_index=False).agg(
        avg_dk_points=("dk_points", "mean"),
        games=("dk_points", "size"),
    )
    agg_first_initial_last = agg_first_initial_last.merge(
        counts_first_initial_last[counts_first_initial_last["count"] == 1][["first_initial_last_key", "team_key"]],
        on=["first_initial_last_key", "team_key"],
        how="inner",
    )

    agg_last = df.groupby(["last_key", "team_key"], as_index=False).agg(
        avg_dk_points=("dk_points", "mean"),
        games=("dk_points", "size"),
    )
    agg_last = agg_last.merge(
        counts_last[counts_last["count"] == 1][["last_key", "team_key"]],
        on=["last_key", "team_key"],
        how="inner",
    )

    agg_name_unique = df.groupby(["name_key"], as_index=False).agg(
        avg_dk_points=("dk_points", "mean"),
        games=("dk_points", "size"),
    )
    agg_name_unique = agg_name_unique.merge(
        counts_name[counts_name["count"] == 1][["name_key"]],
        on="name_key",
        how="inner",
    )

    return {
        "by_name_team": agg_name_team,
        "by_first_last_team": agg_first_last,
        "by_first_initial_last_team": agg_first_initial_last,
        "by_last_team": agg_last,
        "by_name_unique": agg_name_unique,
    }


def _load_local_injuries(season: int, week: int) -> Tuple[pd.DataFrame, Optional[int]]:
    base = DATA_BASE / "injuries" / f"season={season}"
    if not base.exists():
        return pd.DataFrame(), None
    weeks = _list_available_weeks("injuries", season)
    if not weeks:
        return pd.DataFrame(), None
    use_week = week if week in weeks else max(weeks)
    path = base / f"week={use_week}/injuries.parquet"
    if not path.exists():
        return pd.DataFrame(), None
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    return df, use_week


def _load_live_injuries(season: int) -> pd.DataFrame:
    try:
        import nfl_data_py as nfl  # type: ignore
    except Exception:
        return pd.DataFrame()
    try:
        df = nfl.import_injuries([season])
        if df is None:
            return pd.DataFrame()
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def _prepare_player_pool(
    dk_df: pd.DataFrame,
    name_col: str,
    id_col: Optional[str],
    pos_col: Optional[str],
    roster_col: Optional[str],
    team_col: Optional[str],
    salary_col: str,
    avg_col: Optional[str],
    game_col: Optional[str],
) -> pd.DataFrame:
    df = dk_df.copy()
    df.columns = [c.strip() for c in df.columns]

    df["player_name"] = df[name_col].astype(str).str.strip()
    if id_col:
        df["dk_id"] = df[id_col].astype(str).str.strip()
    else:
        df["dk_id"] = ""
    df["position"] = df[pos_col].astype(str).str.strip() if pos_col else ""
    df["team"] = df[team_col].astype(str).str.strip() if team_col else ""
    df["team"] = df["team"].map(normalize_team)
    df["salary"] = pd.to_numeric(df[salary_col], errors="coerce").fillna(0).astype(int)
    if avg_col and avg_col in df.columns:
        df["avg_points"] = pd.to_numeric(df[avg_col], errors="coerce").fillna(0.0)
    else:
        df["avg_points"] = 0.0
    if roster_col and roster_col in df.columns:
        df["roster_position"] = df[roster_col].astype(str).str.upper().str.strip()
    else:
        df["roster_position"] = ""
    if game_col and game_col in df.columns:
        df["game_info"] = df[game_col].astype(str)
    else:
        df["game_info"] = ""

    if df["roster_position"].str.contains("CPT").any():
        flex_only = df[df["roster_position"].str.contains("FLEX")]
        if not flex_only.empty:
            df = flex_only

    df["name_key"] = df["player_name"].map(normalize_name)
    tokens = df["player_name"].map(split_name_tokens)
    df["first_name"] = tokens.map(lambda x: x[0])
    df["last_name"] = tokens.map(lambda x: x[1])
    df["last_key"] = df["last_name"].map(normalize_name)
    df["first_last_key"] = (df["first_name"].fillna("") + " " + df["last_name"].fillna("")).map(normalize_name)
    df["first_initial_last_key"] = (
        df["first_name"].str[:1].fillna("") + " " + df["last_name"].fillna("")
    ).map(normalize_name)
    df["player_key"] = df["team"].astype(str) + "|" + df["player_name"] + "|" + df["position"].astype(str)

    agg = (
        df.sort_values("salary", ascending=True)
        .groupby("player_key", as_index=False)
        .agg(
            player_name=("player_name", "first"),
            dk_id=("dk_id", "first"),
            position=("position", "first"),
            team=("team", "first"),
            salary=("salary", "min"),
            avg_points=("avg_points", "max"),
            game_info=("game_info", "first"),
            name_key=("name_key", "first"),
            first_name=("first_name", "first"),
            last_name=("last_name", "first"),
            last_key=("last_key", "first"),
            first_last_key=("first_last_key", "first"),
            first_initial_last_key=("first_initial_last_key", "first"),
        )
    )

    if (agg["dk_id"].astype(str) != "").any():
        agg["dk_name_id"] = agg["player_name"] + " (" + agg["dk_id"].astype(str) + ")"
    else:
        agg["dk_name_id"] = agg["player_name"]

    agg["display_name"] = agg["player_name"] + " (" + agg["position"].astype(str) + ", " + agg["team"].astype(str) + ")"
    return agg


def _apply_injury_exclusions(
    pool: pd.DataFrame,
    season: int,
    week: int,
    exclude_statuses: Sequence[str],
    use_live: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[int]]:
    if use_live:
        inj = _load_live_injuries(season)
        injury_week = None
    else:
        inj, injury_week = _load_local_injuries(season, week)

    if inj.empty:
        return pool, pd.DataFrame(), injury_week

    status_col = "report_status" if "report_status" in inj.columns else "practice_status"
    if status_col not in inj.columns:
        return pool, pd.DataFrame(), injury_week

    name_col = "full_name" if "full_name" in inj.columns else "player_name"
    team_col = "team" if "team" in inj.columns else "club"

    inj = inj.copy()
    inj["status"] = inj[status_col].astype(str).str.upper().str.strip()
    inj["name_key"] = inj[name_col].astype(str).map(normalize_name)
    inj["team_key"] = inj[team_col].astype(str).map(normalize_team)

    excluded = inj[inj["status"].isin([s.upper() for s in exclude_statuses])]
    if excluded.empty:
        return pool, pd.DataFrame(), injury_week

    merged = pool.merge(
        excluded[["name_key", "team_key", "status"]],
        left_on=["name_key", "team"],
        right_on=["name_key", "team_key"],
        how="left",
    )
    injured = merged[merged["status"].notna()].copy()
    filtered = merged[merged["status"].isna()].drop(columns=["team_key", "status"])
    return filtered, injured, injury_week


def _build_optimizer_pool(pool: pd.DataFrame) -> pd.DataFrame:
    base = pool.copy()
    base["base_id"] = base["player_key"]
    base["proj_points"] = pd.to_numeric(base["proj_points"], errors="coerce").fillna(0.0)
    base["salary"] = pd.to_numeric(base["salary"], errors="coerce").fillna(0).astype(int)

    flex = base.copy()
    flex["roster"] = "FLEX"
    flex["is_captain"] = False
    flex["salary"] = flex["salary"]
    flex["proj"] = flex["proj_points"]

    cpt = base.copy()
    cpt["roster"] = "CPT"
    cpt["is_captain"] = True
    cpt["salary"] = (cpt["salary"] * 1.5 / 50).round().astype(int) * 50
    cpt["proj"] = cpt["proj_points"] * 1.5

    all_rows = pd.concat([cpt, flex], ignore_index=True)
    all_rows["row_id"] = np.arange(len(all_rows))
    return all_rows


def _fill_from_map(
    pool: pd.DataFrame,
    map_df: pd.DataFrame,
    left_keys: List[str],
    right_keys: List[str],
    label: str,
) -> pd.DataFrame:
    if map_df is None or map_df.empty:
        return pool
    missing = pool["proj_points"].isna()
    if not missing.any():
        return pool
    subset = pool.loc[missing].copy()
    subset = subset.join(map_df.set_index(right_keys), on=left_keys)
    fill_mask = subset["avg_dk_points"].notna()
    if fill_mask.any():
        pool.loc[subset.index[fill_mask], "proj_points"] = subset.loc[fill_mask, "avg_dk_points"].values
        pool.loc[subset.index[fill_mask], "proj_source"] = label
        if "games" in subset.columns:
            pool.loc[subset.index[fill_mask], "games_used"] = subset.loc[fill_mask, "games"].values
    return pool


def _solve_showdown_lineup(
    pool: pd.DataFrame,
    salary_cap: int,
    min_team_players: int,
    must_include: Sequence[str],
    exclude: Sequence[str],
    lock_captain: Optional[str],
    lock_flex: Sequence[str],
    allowed_captains: Optional[Sequence[str]],
    avoid_lineups: Sequence[Sequence[int]],
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        import pulp  # type: ignore
    except Exception:
        return None, "PuLP not installed. Add pulp to requirements to enable optimizer."

    df = pool.copy()
    if exclude:
        df = df[~df["base_id"].isin(exclude)]
    if allowed_captains is not None:
        mask = df["is_captain"] & (~df["base_id"].isin(allowed_captains))
        df = df[~mask]
    if df.empty:
        return None, "No players left after exclusions."

    df = df.set_index("row_id", drop=False)

    x = {rid: pulp.LpVariable(f"x_{rid}", cat="Binary") for rid in df.index}
    prob = pulp.LpProblem("showdown", pulp.LpMaximize)
    prob += pulp.lpSum(x[rid] * df.loc[rid, "proj_adj"] for rid in df.index)

    prob += pulp.lpSum(x[rid] * df.loc[rid, "salary"] for rid in df.index) <= int(salary_cap)

    prob += pulp.lpSum(x[rid] for rid in df.index if df.loc[rid, "is_captain"]) == 1
    prob += pulp.lpSum(x[rid] for rid in df.index if not df.loc[rid, "is_captain"]) == 5

    for base_id, rows in df.groupby("base_id").groups.items():
        prob += pulp.lpSum(x[rid] for rid in rows) <= 1

    teams = [t for t in sorted(df["team"].dropna().unique().tolist()) if t]
    if len(teams) == 2 and min_team_players > 0:
        for t in teams:
            rows = df.index[df["team"] == t].tolist()
            if rows:
                prob += pulp.lpSum(x[rid] for rid in rows) >= int(min_team_players)

    for base_id in must_include:
        rows = df.index[df["base_id"] == base_id].tolist()
        if not rows:
            return None, f"Must-include player missing from pool: {base_id}"
        prob += pulp.lpSum(x[rid] for rid in rows) == 1

    if lock_captain:
        rows = df.index[(df["base_id"] == lock_captain) & (df["is_captain"])].tolist()
        if not rows:
            return None, "Locked captain not available in pool."
        prob += x[rows[0]] == 1

    for base_id in lock_flex:
        rows = df.index[(df["base_id"] == base_id) & (~df["is_captain"])].tolist()
        if not rows:
            return None, f"Locked flex not available in pool: {base_id}"
        prob += x[rows[0]] == 1

    for prev in avoid_lineups:
        rows = [rid for rid in prev if rid in x]
        if rows:
            prob += pulp.lpSum(x[rid] for rid in rows) <= 5

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        return None, "Optimizer failed to find a valid lineup."

    chosen = df[df.index.map(lambda rid: x[rid].value() == 1)].copy()
    return chosen, None


def _generate_lineups(
    pool: pd.DataFrame,
    num_lineups: int,
    salary_cap: int,
    min_team_players: int,
    exposure_limits: Dict[str, Tuple[int, int]],
    exclude: Sequence[str],
    lock_captain: Optional[str],
    lock_flex: Sequence[str],
    allowed_captains: Optional[Sequence[str]],
    randomness: float,
    enforce_unique: bool,
) -> Tuple[List[pd.DataFrame], List[str]]:
    rng = np.random.default_rng(42)
    counts: Dict[str, int] = {bid: 0 for bid in pool["base_id"].unique().tolist()}
    min_counts = {bid: exposure_limits.get(bid, (0, num_lineups))[0] for bid in counts}
    max_counts = {bid: exposure_limits.get(bid, (0, num_lineups))[1] for bid in counts}

    lineups: List[pd.DataFrame] = []
    errors: List[str] = []
    previous_lineups: List[List[int]] = []

    for i in range(num_lineups):
        remaining = num_lineups - i
        forced = [bid for bid, min_ct in min_counts.items() if min_ct - counts.get(bid, 0) >= remaining]
        maxed = [bid for bid, max_ct in max_counts.items() if counts.get(bid, 0) >= max_ct]

        df = pool.copy()
        if randomness > 0:
            noise = rng.normal(1.0, randomness, size=len(df))
            noise = np.clip(noise, 0.5, 1.5)
            df["proj_adj"] = df["proj"] * noise
        else:
            df["proj_adj"] = df["proj"]

        chosen, err = _solve_showdown_lineup(
            df,
            salary_cap=salary_cap,
            min_team_players=min_team_players,
            must_include=forced,
            exclude=list(set(exclude) | set(maxed)),
            lock_captain=lock_captain,
            lock_flex=lock_flex,
            allowed_captains=allowed_captains,
            avoid_lineups=previous_lineups if enforce_unique else [],
        )
        if chosen is None:
            errors.append(err or "Unknown optimizer failure.")
            break

        lineups.append(chosen)
        previous_lineups.append(chosen["row_id"].tolist())
        for bid in chosen["base_id"].unique().tolist():
            counts[bid] = counts.get(bid, 0) + 1

    return lineups, errors


def render_showdown_generator(season: int, week: int) -> None:
    st.header("DFS Showdown Generator (DraftKings)")
    st.caption("Upload a DraftKings Showdown salaries CSV to build projections and generate lineups.")

    salary_file = st.file_uploader("DK salaries CSV", type=["csv"], key="dk_salaries")
    entry_file = st.file_uploader("Optional: DK entries template CSV", type=["csv"], key="dk_entries")

    if not salary_file:
        st.info("Upload a DK salaries CSV to get started.")
        return

    try:
        dk_df = pd.read_csv(salary_file)
    except Exception as exc:
        st.error(f"Failed to read CSV: {exc}")
        return

    if dk_df.empty:
        st.warning("The uploaded CSV is empty.")
        return

    cols = list(dk_df.columns)
    name_col = _pick_column(cols, ["Name", "Name + ID", "Player", "PlayerName"])
    id_col = _pick_column(cols, ["ID", "Player ID", "PlayerID"])
    pos_col = _pick_column(cols, ["Position", "Pos"])
    roster_col = _pick_column(cols, ["Roster Position", "RosterPosition"])
    team_col = _pick_column(cols, ["TeamAbbrev", "Team", "TeamAbbr", "Team Abbrev"])
    salary_col = _pick_column(cols, ["Salary"])
    avg_col = _pick_column(cols, ["AvgPointsPerGame", "Avg Points", "AvgPoints"])
    game_col = _pick_column(cols, ["Game Info", "GameInfo", "Game"])

    with st.expander("Column mapping", expanded=False):
        name_col = st.selectbox("Player name column", cols, index=cols.index(name_col) if name_col in cols else 0)
        id_col = st.selectbox("Player ID column (optional)", [""] + cols, index=(cols.index(id_col) + 1) if id_col in cols else 0)
        pos_col = st.selectbox("Position column (optional)", [""] + cols, index=(cols.index(pos_col) + 1) if pos_col in cols else 0)
        roster_col = st.selectbox("Roster position column (optional)", [""] + cols, index=(cols.index(roster_col) + 1) if roster_col in cols else 0)
        team_col = st.selectbox("Team column (optional)", [""] + cols, index=(cols.index(team_col) + 1) if team_col in cols else 0)
        salary_col = st.selectbox("Salary column", cols, index=cols.index(salary_col) if salary_col in cols else 0)
        avg_col = st.selectbox("Avg points column (optional)", [""] + cols, index=(cols.index(avg_col) + 1) if avg_col in cols else 0)
        game_col = st.selectbox("Game info column (optional)", [""] + cols, index=(cols.index(game_col) + 1) if game_col in cols else 0)

    if not name_col or not salary_col:
        st.error("Name and Salary columns are required.")
        return

    pool = _prepare_player_pool(
        dk_df,
        name_col=name_col,
        id_col=id_col or None,
        pos_col=pos_col or None,
        roster_col=roster_col or None,
        team_col=team_col or None,
        salary_col=salary_col,
        avg_col=avg_col or None,
        game_col=game_col or None,
    )

    if pool.empty:
        st.warning("No players found after parsing the CSV.")
        return

    teams = sorted([t for t in pool["team"].dropna().unique().tolist() if t])
    if len(teams) > 2:
        st.warning("More than two teams detected. Showdown requires a single game slate.")

    team_filter = st.multiselect("Teams in slate", options=teams, default=teams)
    if team_filter:
        pool = pool[pool["team"].isin(team_filter)]
        pool = pool.reset_index(drop=True)

    st.subheader("Projections")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_full_season = st.checkbox("Use full season data", value=False)
    with col2:
        recent_weeks = st.number_input("Recent weeks", min_value=1, max_value=10, value=4, step=1)
    with col3:
        fallback_to_dk_avg = st.checkbox("Fallback to DK AvgPointsPerGame", value=True)

    src_col1, src_col2 = st.columns(2)
    with src_col1:
        projection_source = st.selectbox(
            "Projection source",
            options=["DB (player_stats)", "Parquet (player_week)"],
            index=0,
        )
    with src_col2:
        if projection_source.startswith("DB") and not DB_PATH.exists():
            st.warning("DB not found; falling back to Parquet.")
            projection_source = "Parquet (player_week)"

    source_key = "db" if projection_source.startswith("DB") else "parquet"

    min_games_col1, min_games_col2 = st.columns(2)
    with min_games_col1:
        min_games = st.number_input("Min games for projection", min_value=1, max_value=10, value=3, step=1)
    with min_games_col2:
        use_dk_if_low_games = st.checkbox("Use DK Avg if games < min", value=True)

    kdst_col1, kdst_col2 = st.columns(2)
    with kdst_col1:
        force_kdst_dk_avg = st.checkbox("Force DK Avg for K/DST", value=True)
    with kdst_col2:
        replace_zero_with_dk_avg = st.checkbox("Replace zero proj with DK Avg", value=True)

    own_col1, own_col2, own_col3 = st.columns(3)
    with own_col1:
        ownership_method = st.selectbox(
            "Ownership projection",
            options=["None", "Proxy (My Proj + DK Avg)", "Proxy (My Proj + Salary)", "Proxy (DK Avg + Salary)"],
            index=1,
        )
    with own_col2:
        ownership_floor = st.slider("Ownership floor %", min_value=0, max_value=20, value=2, step=1)
    with own_col3:
        ownership_cap = st.slider("Ownership cap %", min_value=10, max_value=90, value=60, step=5)

    weeks_available = _list_db_weeks(season) if source_key == "db" else _list_available_weeks("player_week", season)
    max_week_available = max(weeks_available) if weeks_available else week
    effective_week = min(week, max_week_available) if week else max_week_available
    if week and max_week_available and week > max_week_available:
        st.warning(f"Selected week {week} not available in {projection_source}. Using week {max_week_available}.")

    stat_maps: Dict[str, pd.DataFrame] = {}
    try:
        stat_maps = _build_stat_projections(season, effective_week, int(recent_weeks), use_full_season, source_key)
    except Exception:
        stat_maps = {}

    pool["proj_points"] = np.nan
    pool["proj_source"] = ""
    pool["games_used"] = np.nan

    if stat_maps:
        pool = _fill_from_map(
            pool,
            stat_maps.get("by_name_team", pd.DataFrame()),
            left_keys=["name_key", "team"],
            right_keys=["name_key", "team_key"],
            label="stats:exact",
        )
        pool = _fill_from_map(
            pool,
            stat_maps.get("by_first_last_team", pd.DataFrame()),
            left_keys=["first_last_key", "team"],
            right_keys=["first_last_key", "team_key"],
            label="stats:first+last",
        )
        pool = _fill_from_map(
            pool,
            stat_maps.get("by_first_initial_last_team", pd.DataFrame()),
            left_keys=["first_initial_last_key", "team"],
            right_keys=["first_initial_last_key", "team_key"],
            label="stats:init+last",
        )
        pool = _fill_from_map(
            pool,
            stat_maps.get("by_last_team", pd.DataFrame()),
            left_keys=["last_key", "team"],
            right_keys=["last_key", "team_key"],
            label="stats:last",
        )
        pool = _fill_from_map(
            pool,
            stat_maps.get("by_name_unique", pd.DataFrame()),
            left_keys=["name_key"],
            right_keys=["name_key"],
            label="stats:unique",
        )

        if use_dk_if_low_games and min_games > 1:
            low_games = pool["games_used"].fillna(0).astype(int) < int(min_games)
            dk_ok = pool["avg_points"] > 0
            pool.loc[low_games & dk_ok, "proj_points"] = pool.loc[low_games & dk_ok, "avg_points"]
            pool.loc[low_games & dk_ok, "proj_source"] = "dk_avg_low_games"

    if fallback_to_dk_avg:
        missing = pool["proj_points"].isna()
        pool.loc[missing, "proj_points"] = pool.loc[missing, "avg_points"]
        pool.loc[missing & (pool["avg_points"] > 0), "proj_source"] = "dk_avg"

    if force_kdst_dk_avg:
        is_kdst = pool["position"].astype(str).str.upper().isin(["K", "DST", "D/ST", "DEF"])
        if replace_zero_with_dk_avg:
            kdst_mask = is_kdst & (pool["avg_points"] > 0) & (pool["proj_points"].fillna(0) <= 0)
        else:
            kdst_mask = is_kdst & (pool["avg_points"] > 0)
        pool.loc[kdst_mask, "proj_points"] = pool.loc[kdst_mask, "avg_points"]
        pool.loc[kdst_mask, "proj_source"] = "dk_avg_kdst"

    pool["proj_points"] = pool["proj_points"].fillna(0.0)

    if ownership_method != "None":
        if ownership_cap < ownership_floor:
            ownership_cap = ownership_floor
        proj = pool["proj_points"].astype(float)
        dk_avg = pool["avg_points"].astype(float)
        salary = pool["salary"].replace(0, np.nan).astype(float)
        if ownership_method == "Proxy (My Proj + DK Avg)":
            base = proj + dk_avg
        elif ownership_method.startswith("Proxy (My Proj"):
            base = proj
        else:
            base = dk_avg
        value = (base / (salary / 1000.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if base.std(ddof=0) == 0:
            score = base
        elif value.std(ddof=0) == 0:
            score = (base - base.mean()) / base.std(ddof=0)
        else:
            z_base = (base - base.mean()) / base.std(ddof=0)
            z_value = (value - value.mean()) / value.std(ddof=0)
            score = 0.7 * z_base + 0.3 * z_value
        ranks = score.rank(pct=True).fillna(0.5)
        pool["own_proj"] = (ownership_floor + ranks * (ownership_cap - ownership_floor)).round(1)
        zero_mask = (pool["proj_points"].fillna(0) <= 0) & (pool["avg_points"].fillna(0) <= 0)
        pool.loc[zero_mask, "own_proj"] = 0.0
    else:
        pool["own_proj"] = np.nan

    proj_display = pool[
        ["display_name", "team", "position", "salary", "avg_points", "proj_points", "games_used", "own_proj", "proj_source"]
    ].copy()
    proj_display = proj_display.rename(
        columns={
            "display_name": "Player",
            "avg_points": "DK Avg",
            "proj_points": "My Proj",
            "games_used": "Games",
            "own_proj": "Own %",
            "proj_source": "Source",
        }
    )
    edited = st.data_editor(
        proj_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "DK Avg": st.column_config.NumberColumn("DK Avg", format="%.2f"),
            "My Proj": st.column_config.NumberColumn("My Proj", format="%.2f"),
            "Games": st.column_config.NumberColumn("Games", format="%.0f"),
            "Own %": st.column_config.NumberColumn("Own %", format="%.1f"),
        },
        disabled=["Player", "team", "position", "salary", "DK Avg", "Games", "Source"],
        key="showdown_proj_editor",
    )
    pool = pool.merge(
        edited[["Player", "My Proj", "Own %"]], left_on="display_name", right_on="Player", how="left"
    )
    pool["proj_points"] = pool["My Proj"].fillna(pool["proj_points"])
    pool["own_proj"] = pool["Own %"].fillna(pool.get("own_proj"))
    pool = pool.drop(columns=["Player", "My Proj", "Own %"])

    with st.expander("Matching diagnostics", expanded=False):
        total = len(pool)
        counts = pool["proj_source"].value_counts().to_dict()
        st.write({
            "total_players": total,
            "stats_exact": counts.get("stats:exact", 0),
            "stats_first_last": counts.get("stats:first+last", 0),
            "stats_init_last": counts.get("stats:init+last", 0),
            "stats_last": counts.get("stats:last", 0),
            "stats_unique": counts.get("stats:unique", 0),
            "dk_avg": counts.get("dk_avg", 0),
            "dk_avg_low_games": counts.get("dk_avg_low_games", 0),
            "dk_avg_kdst": counts.get("dk_avg_kdst", 0),
            "zero_projection": int((pool["proj_points"] <= 0).sum()),
        })
        st.caption(f"Projection source: {projection_source} | Week used: {effective_week}")
        fallback = pool[pool["proj_source"].isin(["dk_avg", "dk_avg_kdst", "dk_avg_low_games", ""])]
        if not fallback.empty:
            st.caption("Players using DK AvgPointsPerGame or no match:")
            st.dataframe(
                fallback[
                    ["display_name", "team", "position", "avg_points", "proj_points", "games_used", "own_proj", "proj_source"]
                ].rename(
                    columns={
                        "display_name": "Player",
                        "avg_points": "DK Avg",
                        "proj_points": "My Proj",
                        "games_used": "Games",
                        "own_proj": "Own %",
                        "proj_source": "Source",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Injury Filters")
    use_live_injuries = st.checkbox("Use live injuries (if available)", value=False)
    status_options = ["Out", "IR", "PUP", "Suspended", "DNP", "Doubtful", "Questionable"]
    exclude_statuses = st.multiselect(
        "Exclude injury statuses",
        options=status_options,
        default=["Out", "IR", "PUP", "Suspended", "DNP", "Doubtful"],
    )

    filtered_pool, injured, injury_week = _apply_injury_exclusions(
        pool,
        season=season,
        week=week,
        exclude_statuses=exclude_statuses,
        use_live=use_live_injuries,
    )
    if injury_week:
        st.caption(f"Using local injury data for week {injury_week}.")
    if not injured.empty:
        st.warning(f"Excluded {len(injured)} injured players.")
        st.dataframe(injured[["player_name", "team", "position", "status"]].rename(columns={"player_name": "Player"}), use_container_width=True)

    pool = filtered_pool.copy().reset_index(drop=True)
    pool["proj_points"] = pd.to_numeric(pool["proj_points"], errors="coerce").fillna(0.0)

    st.subheader("Lineup Controls")
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        num_lineups = st.number_input("Lineups to generate", min_value=1, max_value=150, value=20, step=1)
    with ctrl2:
        salary_cap = st.number_input("Salary cap", min_value=1, max_value=100000, value=50000, step=500)
    with ctrl3:
        randomness = st.slider("Projection randomness", min_value=0.0, max_value=0.30, value=0.05, step=0.01)

    enforce_unique = st.checkbox("Enforce unique lineups", value=True)

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        use_simulation = st.checkbox("Use simulation scoring (GPP)", value=False)
    with sim_col2:
        sim_count = st.number_input("Simulations", min_value=200, max_value=5000, value=1000, step=100)
    with sim_col3:
        top_pct = st.slider("Top % average", min_value=1, max_value=20, value=5, step=1)

    player_labels = pool["display_name"].tolist()
    label_to_id = dict(zip(pool["display_name"], pool["player_key"]))

    exclude_players = st.multiselect("Exclude players", options=player_labels, default=[])
    lock_captain_label = st.selectbox("Lock captain (optional)", options=[""] + player_labels, index=0)
    lock_flex_labels = st.multiselect("Lock flex players (optional)", options=player_labels, default=[])

    restrict_cpt = st.checkbox("Restrict CPT to QB/RB/WR/TE", value=False)
    if restrict_cpt:
        cpt_labels = pool[
            pool["position"].astype(str).str.upper().isin(["QB", "RB", "WR", "TE"])
        ]["display_name"].tolist()
    else:
        cpt_labels = player_labels
    captain_pool_labels = st.multiselect("Captain pool (optional)", options=cpt_labels, default=cpt_labels)

    st.subheader("Exposure Limits")
    exposure_players = st.multiselect("Set exposure limits for players", options=player_labels, default=[])
    exposure_limits: Dict[str, Tuple[int, int]] = {}
    if exposure_players:
        exp_df = pd.DataFrame({"Player": exposure_players, "Min%": 0, "Max%": 100})
        exp_df = st.data_editor(exp_df, use_container_width=True, hide_index=True, key="showdown_exposure_editor")
        for _, row in exp_df.iterrows():
            base_id = label_to_id.get(row["Player"])
            if not base_id:
                continue
            min_ct = math.ceil(float(row["Min%"]) / 100.0 * int(num_lineups))
            max_ct = math.floor(float(row["Max%"]) / 100.0 * int(num_lineups))
            exposure_limits[base_id] = (min_ct, max_ct)

    if st.button("Generate lineups", key="showdown_generate"):
        exclude_ids = [label_to_id[x] for x in exclude_players if x in label_to_id]
        lock_captain_id = label_to_id.get(lock_captain_label) if lock_captain_label else None
        lock_flex_ids = [label_to_id[x] for x in lock_flex_labels if x in label_to_id]
        allowed_captains = [label_to_id[x] for x in captain_pool_labels if x in label_to_id]

        invalid_exposures = []
        for label, base_id in label_to_id.items():
            if base_id in exposure_limits:
                min_ct, max_ct = exposure_limits[base_id]
                if min_ct > max_ct:
                    invalid_exposures.append(label)
        if invalid_exposures:
            st.error(f"Exposure limits invalid (min > max) for: {', '.join(invalid_exposures)}")
            return

        if lock_captain_id and lock_captain_id in lock_flex_ids:
            st.error("A player cannot be locked as both captain and flex.")
            return

        for locked_id in [lock_captain_id] + lock_flex_ids:
            if not locked_id:
                continue
            min_ct, max_ct = exposure_limits.get(locked_id, (0, int(num_lineups)))
            if max_ct < int(num_lineups):
                st.error("Locked players must have max exposure of 100%.")
                return

        opt_pool = _build_optimizer_pool(pool)

        lineups, errors = _generate_lineups(
            opt_pool,
            num_lineups=int(num_lineups),
            salary_cap=int(salary_cap),
            min_team_players=1,
            exposure_limits=exposure_limits,
            exclude=exclude_ids,
            lock_captain=lock_captain_id,
            lock_flex=lock_flex_ids,
            allowed_captains=allowed_captains,
            randomness=float(randomness),
            enforce_unique=enforce_unique,
        )

        if errors:
            st.error(errors[0])
            return
        if not lineups:
            st.error("No lineups generated.")
            return

        lineup_rows = []
        for idx, lineup in enumerate(lineups, start=1):
            cpt_row = lineup[lineup["is_captain"]].iloc[0]
            flex_rows = lineup[~lineup["is_captain"]].sort_values("proj", ascending=False)
            flex_names = flex_rows["dk_name_id"].tolist()
            total_salary = int(lineup["salary"].sum())
            total_proj = float(lineup["proj"].sum())
            lineup_rows.append(
                {
                    "Lineup": idx,
                    "CPT": cpt_row["dk_name_id"],
                    "FLEX1": flex_names[0],
                    "FLEX2": flex_names[1],
                    "FLEX3": flex_names[2],
                    "FLEX4": flex_names[3],
                    "FLEX5": flex_names[4],
                    "Salary": total_salary,
                    "Proj": round(total_proj, 2),
                }
            )

        lineup_df = pd.DataFrame(lineup_rows)
        if use_simulation:
            history_df = _load_histories(
                season=season,
                week=effective_week,
                recent_weeks=int(recent_weeks),
                use_full_season=use_full_season,
                source=source_key,
            )
            sim_metrics, sim_means = _simulate_lineups(
                lineups,
                pool,
                history_df,
                sims=int(sim_count),
                top_pct=float(top_pct),
                force_kdst_dk_avg=force_kdst_dk_avg,
                replace_zero_with_dk_avg=replace_zero_with_dk_avg,
            )
            lineup_df["Sim Mean"] = [round(x, 2) for x in sim_means]
            lineup_df[f"Top {int(top_pct)}% Avg"] = [round(x, 2) for x in sim_metrics]
            lineup_df = lineup_df.sort_values(f"Top {int(top_pct)}% Avg", ascending=False)
        st.subheader("Generated Lineups")
        st.dataframe(lineup_df, use_container_width=True, hide_index=True)

        all_selected = pd.concat(lineups, ignore_index=True)
        exposure_counts = all_selected.groupby("base_id")["base_id"].count().reset_index(name="Count")
        exposure_counts["Exposure%"] = (exposure_counts["Count"] / float(num_lineups) * 100).round(1)
        exposure_view = pool[["player_key", "display_name"]].drop_duplicates().merge(
            exposure_counts, left_on="player_key", right_on="base_id", how="right"
        )
        exposure_view = exposure_view.sort_values("Exposure%", ascending=False)
        st.subheader("Exposure Summary")
        st.dataframe(
            exposure_view[["display_name", "Count", "Exposure%"]].rename(columns={"display_name": "Player"}),
            use_container_width=True,
            hide_index=True,
        )

        upload_cols = ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]
        upload_values = lineup_df[["CPT", "FLEX1", "FLEX2", "FLEX3", "FLEX4", "FLEX5"]].values.tolist()
        lineup_upload = pd.DataFrame(upload_values, columns=upload_cols)

        if entry_file is not None:
            try:
                entries = pd.read_csv(entry_file)
                cpt_cols = [c for c in entries.columns if c.upper() == "CPT"]
                flex_cols = [c for c in entries.columns if c.upper().startswith("FLEX")]
                entry_cols = cpt_cols + flex_cols
                if len(entry_cols) >= 6:
                    limit = min(len(entries), len(lineup_upload))
                    for i, col in enumerate(entry_cols[:6]):
                        entries.loc[: limit - 1, col] = lineup_upload.iloc[:limit, i].values
                    lineup_upload = entries
                else:
                    st.warning("Entries template missing lineup columns. Using basic upload format.")
            except Exception as exc:
                st.warning(f"Failed to read entries template: {exc}")

        st.download_button(
            "Download DK lineups CSV",
            data=lineup_upload.to_csv(index=False),
            file_name="dk_showdown_lineups.csv",
            mime="text/csv",
        )
