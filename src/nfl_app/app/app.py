from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable, List, Optional
import subprocess

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


DATA_BASE = Path("data/processed").resolve()


def _list_available_seasons(table: str) -> List[int]:
    base = DATA_BASE / table
    if not base.exists():
        return []
    seasons = []
    for p in base.glob("season=*"):
        try:
            seasons.append(int(p.name.split("=", 1)[1]))
        except Exception:
            pass
    return sorted(set(seasons))


def _list_available_weeks(table: str, season: int) -> List[int]:
    season_dir = DATA_BASE / table / f"season={season}"
    if not season_dir.exists():
        return []
    weeks = []
    for p in season_dir.glob("week=*"):
        wk = p.name.split("=", 1)[1]
        if wk.isdigit():
            weeks.append(int(wk))
    return sorted(set(weeks))


def _read_first(path_glob: str, nrows: Optional[int] = None) -> pd.DataFrame:
    matches = glob.glob(path_glob)
    if not matches:
        return pd.DataFrame()
    return pd.read_parquet(matches[0], columns=None) if nrows is None else pd.read_parquet(matches[0]).head(nrows)


def _read_all(path_glob: str, limit_files: Optional[int] = None) -> pd.DataFrame:
    matches = sorted(glob.glob(path_glob))
    if not matches:
        return pd.DataFrame()
    if limit_files:
        matches = matches[:limit_files]
    parts = [pd.read_parquet(p) for p in matches]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _pick_team_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "recent_team",
        "team",
        "team_abbr",
        "player_team",
        "club",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _load_pbp_local_first(season: int, week: Optional[int] = None) -> tuple[pd.DataFrame, Optional[str]]:
    """Load PBP for a season.

    1) Try local Parquet under data/processed/pbp/season=YYYY/week=*/pbp.parquet
    2) Fall back to nfl_data_py.import_pbp_data([season]) with a runtime patch
    Returns (pbp_df, error_message_or_None)
    """
    # Attempt local first
    local_glob = str(DATA_BASE / f"pbp/season={season}/week=*/pbp.parquet")
    pbp_local = _read_all(local_glob)
    if not pbp_local.empty:
        pbp_local.columns = [c.lower() for c in pbp_local.columns]
        if week is not None and "week" in pbp_local.columns:
            pbp_local = pbp_local[pbp_local["week"] <= week]
        return pbp_local, None

    # Remote fallback with runtime patch
    try:
        import nfl_data_py as nfl
        # Patch undefined Error reference if missing in the library
        if not hasattr(nfl, "Error"):
            nfl.Error = Exception
        pbp_remote = nfl.import_pbp_data([season])
        pbp_remote.columns = [c.lower() for c in pbp_remote.columns]
        if week is not None and "week" in pbp_remote.columns:
            pbp_remote = pbp_remote[pbp_remote["week"] <= week]
        return pbp_remote, None
    except Exception as e:
        # Attempt direct URLs to nflverse parquet as a final fallback
        try:
            candidates = [
                f"https://github.com/nflverse/nflfastR-data/releases/download/pbp/play_by_play_{season}.parquet",
                f"https://github.com/nflverse/nflfastR-data/releases/download/play_by_play_{season}/play_by_play_{season}.parquet",
                f"https://raw.githubusercontent.com/nflverse/nflfastR-data/master/data/play_by_play_{season}.parquet",
            ]
            last_err: Optional[str] = None
            for url in candidates:
                try:
                    df = pd.read_parquet(url)
                    df.columns = [c.lower() for c in df.columns]
                    if week is not None and "week" in df.columns:
                        df = df[df["week"] <= week]
                    return df, None
                except Exception as ee:
                    last_err = f"{type(ee).__name__}: {ee}"
            return pd.DataFrame(), (last_err or f"{type(e).__name__}: {e}")
        except Exception as ee2:
            return pd.DataFrame(), f"{type(e).__name__}: {e} | fallback: {type(ee2).__name__}: {ee2}"


def _remote_pbp_available(season: int, weeks: Optional[List[int]] = None) -> tuple[bool, str]:
    """Check if remote PBP is available for a season (and optionally specific weeks)."""
    try:
        import nfl_data_py as nfl
        if not hasattr(nfl, "Error"):
            nfl.Error = Exception
        df = nfl.import_pbp_data([season])
        if df is None or len(df) == 0:
            return False, "Remote returned empty PBP"
        df.columns = [c.lower() for c in df.columns]
        if weeks is not None and "week" in df.columns:
            have = sorted(set(int(w) for w in pd.to_numeric(df["week"], errors="coerce").dropna().astype(int)))
            missing = [w for w in weeks if w not in have]
            return (len(missing) == 0, f"Weeks present: {have}; missing: {missing}")
        return True, f"Rows: {len(df)}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _ingest_pbp_local(season: int, weeks: List[int]) -> tuple[bool, str]:
    """Fetch PBP remotely and write partitioned Parquet locally by week.
    Returns (success, message).
    """
    try:
        import nfl_data_py as nfl
        if not hasattr(nfl, "Error"):
            nfl.Error = Exception
        df = nfl.import_pbp_data([season])
        if df is None or len(df) == 0:
            return False, "Remote returned empty PBP"
        df.columns = [c.lower() for c in df.columns]
        if "week" not in df.columns:
            return False, "No 'week' column in PBP"
        written = []
        for wk in weeks:
            part = df[pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int) == int(wk)]
            if part.empty:
                continue
            out_path = DATA_BASE / f"pbp/season={season}/week={wk}/pbp.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            part.to_parquet(out_path, index=False)
            written.append(wk)
        if not written:
            return False, "No matching weeks found to write"
        return True, f"Written weeks: {sorted(written)}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _safe_unique(series: pd.Series) -> List[str]:
    vals = sorted(v for v in series.dropna().unique().tolist() if isinstance(v, (str, int)))
    return [str(v) for v in vals]


def _teams_from_schedule(season: Optional[int]) -> List[str]:
    if not season:
        return []
    df = _read_all(str(DATA_BASE / f"schedule/season={season}/week=*/schedule.parquet"))
    if df.empty:
        return []
    candidates_home = ["home_team", "home_team_abbr", "home"]
    candidates_away = ["away_team", "away_team_abbr", "away"]
    home = next((c for c in candidates_home if c in df.columns), None)
    away = next((c for c in candidates_away if c in df.columns), None)
    if not home or not away:
        return []
    teams = sorted(set(pd.concat([df[home], df[away]], ignore_index=True).dropna().astype(str).unique().tolist()))
    return teams


st.set_page_config(page_title="NFL Data Browser", page_icon="🏈", layout="wide")
st.title("NFL Data Browser (ingested Parquet)")
st.caption(f"Base directory: {DATA_BASE}")

colA, colB, colC, colD = st.columns([1.2, 1, 1, 1.2])

with colA:
    seasons_sched = _list_available_seasons("schedule")
    seasons_plyr = _list_available_seasons("player_week")
    seasons_inj = _list_available_seasons("injuries")
    seasons_any = sorted(set(seasons_sched + seasons_plyr + seasons_inj))
    season = st.selectbox("Season", seasons_any, index=len(seasons_any) - 1 if seasons_any else 0)

with colB:
    weeks_sched = _list_available_weeks("schedule", season) if season else []
    weeks_plyr = _list_available_weeks("player_week", season) if season else []
    weeks_inj = _list_available_weeks("injuries", season) if season else []
    weeks_any = sorted(set(weeks_sched + weeks_plyr + weeks_inj))
    week = st.selectbox("Week", weeks_any, index=0 if weeks_any else 0)

with colC:
    # Try player_week first; if empty, fallback to schedule home/away teams
    sample = _read_all(str(DATA_BASE / f"player_week/season={season}/week=*/player_week.parquet"), limit_files=3)
    team_col = _pick_team_column(sample) if not sample.empty else None
    teams = _safe_unique(sample[team_col]) if team_col else _teams_from_schedule(season)
    team = st.selectbox("Team", teams, index=0 if teams else 0)

with colD:
    view = st.selectbox(
        "View",
        [
            "Overview",
            "Schedule",
            "Scores",
            "Team Overview",
            "Team Comparison",
            "Analytics Chart",
            "Skill Yards Grid",
            "Skill TDs Grid",
            "First TD Grid",
            "First TD",
            "Data Updates",
            "Player Week",
            "Team Leaders",
        ],
        index=0,
    )

st.divider()

@st.cache_data(show_spinner=False)
def get_team_metrics(season: int, week: Optional[int]) -> pd.DataFrame:
    """Compute per-team metrics through a given week.

    Returns one row per team with columns used across Overview/Comparison pages.
    """
    # Load schedule and filter to completed games
    sched = _read_all(str(DATA_BASE / f"schedule/season={season}/week=*/schedule.parquet"))
    if "week" in sched.columns and week:
        sched = sched[sched["week"] <= week]
    if sched.empty:
        return pd.DataFrame()
    h_col = _pick_col(sched, ["home_team", "home_team_abbr", "home"]) 
    a_col = _pick_col(sched, ["away_team", "away_team_abbr", "away"]) 
    hs_col = _pick_col(sched, ["home_score", "home_points", "home_pts"]) 
    as_col = _pick_col(sched, ["away_score", "away_points", "away_pts"]) 
    if not all([h_col, a_col, hs_col, as_col]):
        return pd.DataFrame()
    played = sched[sched[hs_col].notna() & sched[as_col].notna()].copy()
    # Scores & wins
    h = played[[h_col, hs_col, as_col]].copy(); h.columns = ["team", "points_for", "points_against"]; h["is_home"] = True
    a = played[[a_col, as_col, hs_col]].copy(); a.columns = ["team", "points_for", "points_against"]; a["is_home"] = False
    long = pd.concat([h, a], ignore_index=True)
    long["win"] = (pd.to_numeric(long["points_for"], errors="coerce") > pd.to_numeric(long["points_against"], errors="coerce")).astype(int)
    long["game"] = 1
    agg = long.groupby("team", as_index=False).agg(
        games_played=("game", "sum"),
        wins=("win", "sum"),
        points_for=("points_for", "sum"),
        points_against=("points_against", "sum"),
    )
    wins_home = (long[long["is_home"]].groupby("team", as_index=False)["win"].sum().rename(columns={"win": "wins_home"}))
    wins_away = (long[~long["is_home"]].groupby("team", as_index=False)["win"].sum().rename(columns={"win": "wins_away"}))
    agg = agg.merge(wins_home, on="team", how="left").merge(wins_away, on="team", how="left")
    for c in ["wins_home", "wins_away"]:
        if c in agg.columns:
            agg[c] = agg[c].fillna(0).astype(int)

    # Player-week aggregation for offense totals
    pw = _read_all(str(DATA_BASE / f"player_week/season={season}/week=*/player_week.parquet"))
    if not pw.empty:
        if "week" in pw.columns and week:
            pw = pw[pw["week"] <= week]
        t_col = _pick_team_column(pw)
        pass_y = _pick_col(pw, ["passing_yards", "pass_yards", "pass_yds"]) 
        rush_y = _pick_col(pw, ["rushing_yards", "rush_yards", "rush_yds"]) 
        pass_td = _pick_col(pw, ["passing_tds", "pass_td", "pass_touchdown"]) 
        rush_td = _pick_col(pw, ["rushing_tds", "rush_td", "rush_touchdown"]) 
        ints_thrown = _pick_col(pw, ["interceptions", "int", "interceptions_thrown"]) 
        sacks_taken = _pick_col(pw, ["sacks", "sack"]) 
        keep_cols = [c for c in [t_col, pass_y, rush_y, pass_td, rush_td, ints_thrown, sacks_taken] if c]
        off = pw[keep_cols].copy()
        off_agg = off.groupby(t_col, as_index=False).sum(numeric_only=True)
        off_agg = off_agg.rename(columns={
            t_col: "team",
            pass_y or "": "off_pass_yards",
            rush_y or "": "off_rush_yards",
            pass_td or "": "off_pass_tds",
            rush_td or "": "off_rush_tds",
            ints_thrown or "": "opp_ints_thrown_src",
            sacks_taken or "": "opp_sacks_taken_src",
        })
        agg = agg.merge(off_agg, on="team", how="left")

        # Defensive allowed and defensive counting stats via opponent mapping
        if "week" in pw.columns:
            # team-week offense
            tw_keep = [c for c in [t_col, "week", pass_y, rush_y, ints_thrown, sacks_taken] if c]
            if tw_keep:
                tw = pw[tw_keep].groupby([t_col, "week"], as_index=False).sum(numeric_only=True)
                tw = tw.rename(columns={t_col: "team"})
                map_df = sched[[h_col, a_col, "week"]].dropna(subset=["week"]).copy()
                # home offense becomes away defense
                home_join = tw.merge(map_df, left_on=["team", "week"], right_on=[h_col, "week"], how="inner")
                home_join["def_team"] = home_join[a_col]
                # away offense becomes home defense
                away_join = tw.merge(map_df, left_on=["team", "week"], right_on=[a_col, "week"], how="inner")
                away_join["def_team"] = away_join[h_col]
                opp_long = pd.concat([home_join, away_join], ignore_index=True)
                def_cols = {}
                if pass_y: def_cols[pass_y] = "def_pass_yards_allowed"
                if rush_y: def_cols[rush_y] = "def_rush_yards_allowed"
                if ints_thrown: def_cols[ints_thrown] = "def_interceptions"
                if sacks_taken: def_cols[sacks_taken] = "def_sacks"
                pick = ["def_team"] + [c for c in def_cols.keys()]
                if pick:
                    d_agg = opp_long[pick].groupby("def_team", as_index=False).sum(numeric_only=True)
                    d_agg = d_agg.rename(columns={"def_team": "team", **def_cols})
                    agg = agg.merge(d_agg, on="team", how="left")

    # PBP-derived attempts, sacks, yards/play, completions, penalties, scoring plays
    try:
        import nfl_data_py as nfl
        pbp = nfl.import_pbp_data([season])
        pbp.columns = [c.lower() for c in pbp.columns]
        if "week" in pbp.columns and week:
            pbp = pbp[pbp["week"] <= week]
        played_ids = set(sched[sched[hs_col].notna() & sched[as_col].notna()].get("game_id", pd.Series(dtype=str))) if "game_id" in sched.columns else None
        if played_ids:
            pbp = pbp[pbp["game_id"].isin(played_ids)]
        for c in ["pass_attempt", "rush_attempt", "sack", "complete_pass", "pass_touchdown", "rush_touchdown"]:
            pbp[c] = pd.to_numeric(pbp.get(c, 0), errors="coerce").fillna(0).astype(int)
        pbp["yards_gained"] = pd.to_numeric(pbp.get("yards_gained", 0), errors="coerce").fillna(0)
        pbp["fg_made"] = (pbp.get("field_goal_result", "").astype(str).str.lower() == "made").astype(int)
        pbp["off_td"] = ((pbp["pass_touchdown"] == 1) | (pbp["rush_touchdown"] == 1)).astype(int)
        team_key = _pick_col(pbp, ["posteam", "offense_team"]) or "posteam"
        plays = pbp.groupby(team_key, as_index=False)[["pass_attempt", "rush_attempt", "sack", "complete_pass", "yards_gained", "off_td", "fg_made"]].sum()
        plays = plays.rename(columns={
            team_key: "team",
            "pass_attempt": "pass_attempts",
            "rush_attempt": "rush_attempts",
            "sack": "times_sacked",
            "complete_pass": "completions",
            "yards_gained": "yards_gained_sum",
            "off_td": "td_plays",
            "fg_made": "fg_made_plays",
        })
        agg = agg.merge(plays, on="team", how="left")
        # Penalties
        pbp["penalty"] = pd.to_numeric(pbp.get("penalty", 0), errors="coerce").fillna(0).astype(int)
        pen_team = _pick_col(pbp, ["penalty_team"]) or None
        if pen_team:
            pens = pbp[pbp["penalty"] == 1].groupby(pen_team, as_index=False)["penalty"].sum().rename(columns={pen_team: "team", "penalty": "penalties_committed"})
            agg = agg.merge(pens, on="team", how="left")
    except Exception:
        pass

    agg["points_diff"] = agg["points_for"] - agg["points_against"]
    # Derived
    if set(["pass_attempts", "rush_attempts", "times_sacked"]).issubset(agg.columns):
        agg["total_off_plays"] = agg[["pass_attempts", "rush_attempts", "times_sacked"]].fillna(0).sum(axis=1)
    if "yards_gained_sum" in agg.columns:
        agg["yards_per_play"] = (agg["yards_gained_sum"].fillna(0) / agg["total_off_plays"].where(agg["total_off_plays"]>0, 1)).round(2)
    if "completions" in agg.columns and "games_played" in agg.columns:
        agg["avg_completions"] = (agg["completions"].fillna(0) / agg["games_played"].where(agg["games_played"]>0, 1)).round(1)
    if "off_rush_yards" in agg.columns and "rush_attempts" in agg.columns:
        agg["rush_yards_per_att"] = (agg["off_rush_yards"].fillna(0) / agg["rush_attempts"].where(agg["rush_attempts"]>0, 1)).round(2)
    if "penalties_committed" in agg.columns and "games_played" in agg.columns:
        agg["avg_penalties_committed"] = (agg["penalties_committed"].fillna(0) / agg["games_played"].where(agg["games_played"]>0, 1)).round(1)
    if set(["td_plays", "fg_made_plays", "total_off_plays"]).issubset(agg.columns):
        agg["scoring_play_pct"] = ((agg["td_plays"].fillna(0) + agg["fg_made_plays"].fillna(0)) / agg["total_off_plays"].where(agg["total_off_plays"]>0, 1) * 100).round(1)
    return agg


def render_tornado(subset: pd.DataFrame, team_a: str, team_b: str, keys_with_labels: list[tuple[str, str]]) -> None:
    """Render a mirrored horizontal bar (tornado) chart comparing two teams.

    Falls back with diagnostics if Plotly fails or no metrics are available.
    """
    st.caption("Tornado: attempting Plotly…")
    # keep keys that exist
    keys = [(k, lbl) for (k, lbl) in keys_with_labels if k in subset.columns]
    missing = [k for k, _ in keys_with_labels if k not in subset.columns]
    if missing:
        st.caption(f"Missing metrics (not plotted): {', '.join(missing)}")
    st.write("Tornado metrics:", [k for k, _ in keys])
    if not keys:
        st.warning("No comparable metrics available for tornado.")
        return

    # robust numeric conversion
    def _num(x) -> float:
        try:
            v = pd.to_numeric(x, errors="coerce")
            return float(v) if pd.notna(v) else 0.0
        except Exception:
            return 0.0

    labels = [lbl for (_, lbl) in keys]
    a_vals = [_num(subset.loc[team_a, k]) for (k, _) in keys]
    b_vals = [_num(subset.loc[team_b, k]) for (k, _) in keys]

    try:
        import plotly.graph_objects as go
        max_abs = float(max([abs(v) for v in a_vals + b_vals] + [1]))
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=labels, x=[-v for v in a_vals], name=team_a, orientation="h",
            text=[f"{v:.2f}" for v in a_vals], textposition="inside",
            insidetextanchor="middle", cliponaxis=False, offsetgroup="A",
        ))
        fig.add_trace(go.Bar(
            y=labels, x=b_vals, name=team_b, orientation="h",
            text=[f"{v:.2f}" for v in b_vals], textposition="inside",
            insidetextanchor="middle", cliponaxis=False, offsetgroup="B",
        ))
        fig.update_layout(
            barmode="relative",
            height=380 + 22*len(labels),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.08, x=0.0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            uniformtext_minsize=10, uniformtext_mode="hide",
        )
        fig.update_xaxes(
            range=[-max_abs*1.2, max_abs*1.2],
            zeroline=True, zerolinewidth=2, showgrid=True, tickmode="auto",
        )
        fig.update_yaxes(categoryorder="array", categoryarray=labels, title="")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Tornado plot error: {e}")
        # Altair fallback: mirrored bars
        try:
            import altair as alt
            data = []
            for (k, lbl), av, bv in zip(keys, a_vals, b_vals):
                data.append({"metric": lbl, "team": team_a, "value": -av})
                data.append({"metric": lbl, "team": team_b, "value": bv})
            df_alt = pd.DataFrame(data)
            max_abs = float(max(abs(df_alt["value"].max()), abs(df_alt["value"].min()), 1))
            chart = (
                alt.Chart(df_alt)
                .mark_bar()
                .encode(
                    y=alt.Y("metric:N", sort=None, title=""),
                    x=alt.X("value:Q", scale=alt.Scale(domain=[-max_abs*1.2, max_abs*1.2]), title=""),
                    color=alt.Color("team:N", legend=alt.Legend(orient="top")),
                    tooltip=["team", "metric", alt.Tooltip("value:Q", format=".2f")],
                )
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            # final fallback simple bars per metric
            for k, lbl in keys:
                chart_df = pd.DataFrame({"team": [team_a, team_b], lbl: [subset.loc[team_a, k], subset.loc[team_b, k]]}).set_index("team")
                st.bar_chart(chart_df)

if view == "Overview":
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Schedule")
        df = _read_first(str(DATA_BASE / f"schedule/season={season}/week={week}/schedule.parquet"))
        st.write(df.head(20) if not df.empty else "No data")
    with c2:
        st.subheader("Player Week")
        df = _read_first(str(DATA_BASE / f"player_week/season={season}/week={week}/player_week.parquet"))
        if df.empty:
            st.info("No player-week data available for this selection. For 2025, weekly players may not be published yet.")
        else:
            st.write(df.head(20))
    with c3:
        st.subheader("Injuries")
        df = _read_first(str(DATA_BASE / f"injuries/season={season}/week={week}/injuries.parquet"))
        st.write(df.head(20) if not df.empty else "No data")

elif view == "Schedule":
    st.subheader(f"Schedule {season} Week {week}")
    df = _read_first(str(DATA_BASE / f"schedule/season={season}/week={week}/schedule.parquet"))
    st.dataframe(df if not df.empty else pd.DataFrame())

elif view == "Scores":
    st.subheader(f"Scores {season} Week {week}")
    df = _read_first(str(DATA_BASE / f"schedule/season={season}/week={week}/schedule.parquet"))
    if df.empty:
        st.info("No schedule data available for this selection.")
    else:
        home_col = _pick_col(df, ["home_team", "home_team_abbr", "home"])
        away_col = _pick_col(df, ["away_team", "away_team_abbr", "away"])
        hscore_col = _pick_col(df, ["home_score", "home_points", "home_pts"]) 
        ascore_col = _pick_col(df, ["away_score", "away_points", "away_pts"]) 
        if not all([home_col, away_col, hscore_col, ascore_col]):
            st.dataframe(df)
        else:
            tmp = df[[home_col, away_col, hscore_col, ascore_col]].copy()
            tmp.columns = ["home_team", "away_team", "home_score", "away_score"]
            tmp["home_score"] = pd.to_numeric(tmp["home_score"], errors="coerce")
            tmp["away_score"] = pd.to_numeric(tmp["away_score"], errors="coerce")
            tmp["total_points"] = tmp["home_score"] + tmp["away_score"]
            tmp["point_diff"] = tmp["home_score"] - tmp["away_score"]
            def winner(row):
                if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
                    return None
                if row["home_score"] > row["away_score"]:
                    return row["home_team"]
                if row["home_score"] < row["away_score"]:
                    return row["away_team"]
                return "TIE"
            tmp["winner"] = tmp.apply(winner, axis=1)
            tmp["matchup"] = tmp["away_team"] + " @ " + tmp["home_team"]
            tmp = tmp[["matchup", "home_team", "home_score", "away_team", "away_score", "total_points", "point_diff", "winner"]]
            st.dataframe(tmp, hide_index=True)

elif view == "Team Overview":
    st.subheader(f"Team Overview {season}{' through Week ' + str(week) if week else ''}")
    # Load schedule and filter to completed games only (both scores present)
    sched = _read_all(str(DATA_BASE / f"schedule/season={season}/week=*/schedule.parquet"))
    if sched.empty:
        st.info("No schedule data available for this season.")
    else:
        if "week" in sched.columns and week:
            sched = sched[sched["week"] <= week]
        h_col = _pick_col(sched, ["home_team", "home_team_abbr", "home"]) 
        a_col = _pick_col(sched, ["away_team", "away_team_abbr", "away"]) 
        hs_col = _pick_col(sched, ["home_score", "home_points", "home_pts"]) 
        as_col = _pick_col(sched, ["away_score", "away_points", "away_pts"]) 
        if not all([h_col, a_col, hs_col, as_col]):
            st.dataframe(sched)
        else:
            # Completed games only
            played = sched[sched[hs_col].notna() & sched[as_col].notna()].copy()
            # Per-team long table for scores and wins
            h = played[[h_col, hs_col, as_col]].copy(); h.columns = ["team", "points_for", "points_against"]; h["is_home"] = True
            a = played[[a_col, as_col, hs_col]].copy(); a.columns = ["team", "points_for", "points_against"]; a["is_home"] = False
            long = pd.concat([h, a], ignore_index=True)
            long["win"] = (pd.to_numeric(long["points_for"], errors="coerce") > pd.to_numeric(long["points_against"], errors="coerce")).astype(int)
            long["game"] = 1
            base = long.groupby("team", as_index=False).agg(
                games_played=("game", "sum"),
                wins=("win", "sum"),
                points_for=("points_for", "sum"),
                points_against=("points_against", "sum"),
            )
            wins_home = (long[long["is_home"]].groupby("team", as_index=False)["win"].sum().rename(columns={"win": "wins_home"}))
            wins_away = (long[~long["is_home"]].groupby("team", as_index=False)["win"].sum().rename(columns={"win": "wins_away"}))
            agg = base.merge(wins_home, on="team", how="left").merge(wins_away, on="team", how="left")
            for c in ["wins_home", "wins_away"]:
                if c in agg.columns:
                    agg[c] = agg[c].fillna(0).astype(int)

            # Build offense aggregates from player_week (season, <=week)
            pw = _read_all(str(DATA_BASE / f"player_week/season={season}/week=*/player_week.parquet"))
            if not pw.empty:
                if "week" in pw.columns and week:
                    pw = pw[pw["week"] <= week]
                t_col = _pick_team_column(pw)
                pass_y = _pick_col(pw, ["passing_yards", "pass_yards", "pass_yds"]) 
                rush_y = _pick_col(pw, ["rushing_yards", "rush_yards", "rush_yds"]) 
                pass_td = _pick_col(pw, ["passing_tds", "pass_td", "pass_touchdown"]) 
                rush_td = _pick_col(pw, ["rushing_tds", "rush_td", "rush_touchdown"]) 
                ints_thrown = _pick_col(pw, ["interceptions", "int", "interceptions_thrown"]) 
                sacks_taken = _pick_col(pw, ["sacks", "sack"]) 
                keep_cols = [c for c in [t_col, pass_y, rush_y, pass_td, rush_td, ints_thrown, sacks_taken] if c]
                off = pw[keep_cols].copy()
                # Sum by team
                off_agg = off.groupby(t_col, as_index=False).sum(numeric_only=True)
                off_agg = off_agg.rename(columns={
                    t_col: "team",
                    pass_y or "": "off_pass_yards",
                    rush_y or "": "off_rush_yards",
                    pass_td or "": "off_pass_tds",
                    rush_td or "": "off_rush_tds",
                    ints_thrown or "": "opp_ints_thrown_src",
                    sacks_taken or "": "opp_sacks_taken_src",
                })
                agg = agg.merge(off_agg, on="team", how="left")

                # Defense allowed and defensive stats from opponent offense via schedule mapping
                # Team-week offense
                if "week" in pw.columns:
                    tw_keep = [c for c in [t_col, "week", pass_y, rush_y, ints_thrown, sacks_taken] if c]
                    tw = pw[tw_keep].groupby([t_col, "week"], as_index=False).sum(numeric_only=True)
                    tw = tw.rename(columns={t_col: "team"})
                    # Map to opponent per week
                    map_df = sched[[h_col, a_col, "week"]].dropna(subset=["week"]).copy()
                    # Join for home offense to away defense
                    home_join = tw.merge(map_df, left_on=["team", "week"], right_on=[h_col, "week"], how="inner")
                    home_join["def_team"] = home_join[a_col]
                    away_join = tw.merge(map_df, left_on=["team", "week"], right_on=[a_col, "week"], how="inner")
                    away_join["def_team"] = away_join[h_col]
                    opp_long = pd.concat([home_join, away_join], ignore_index=True)
                    # Sum opponent offense by defensive team
                    def_cols = {}
                    if pass_y: def_cols[pass_y] = "def_pass_yards_allowed"
                    if rush_y: def_cols[rush_y] = "def_rush_yards_allowed"
                    if ints_thrown: def_cols[ints_thrown] = "def_interceptions"
                    if sacks_taken: def_cols[sacks_taken] = "def_sacks"
                    pick = ["def_team"] + [c for c in def_cols.keys()]
                    if pick:
                        d_agg = opp_long[pick].groupby("def_team", as_index=False).sum(numeric_only=True)
                        d_agg = d_agg.rename(columns={"def_team": "team", **def_cols})
                        agg = agg.merge(d_agg, on="team", how="left")

            # Offensive attempts and sacks from play-by-play (completed games only)
            try:
                import nfl_data_py as nfl
                pbp = nfl.import_pbp_data([season])
                pbp.columns = [c.lower() for c in pbp.columns]
                if "week" in pbp.columns and week:
                    pbp = pbp[pbp["week"] <= week]
                # Limit to completed games
                if "game_id" in pbp.columns and "game_id" in sched.columns:
                    played_ids = set(sched[sched[hs_col].notna() & sched[as_col].notna()]["game_id"]) if "game_id" in sched.columns else None
                    if played_ids:
                        pbp = pbp[pbp["game_id"].isin(played_ids)]
                # Normalize flags
                for c in ["pass_attempt", "rush_attempt", "sack", "complete_pass", "pass_touchdown", "rush_touchdown"]:
                    if c in pbp.columns:
                        pbp[c] = pd.to_numeric(pbp[c], errors="coerce").fillna(0).astype(int)
                    else:
                        pbp[c] = 0
                # yards_gained numeric
                if "yards_gained" in pbp.columns:
                    pbp["yards_gained"] = pd.to_numeric(pbp["yards_gained"], errors="coerce").fillna(0)
                else:
                    pbp["yards_gained"] = 0
                # Offensive scoring plays: rushing/receiving TDs or FG made
                if "field_goal_result" in pbp.columns:
                    pbp["fg_made"] = (pbp["field_goal_result"].astype(str).str.lower() == "made").astype(int)
                else:
                    pbp["fg_made"] = 0
                pbp["off_td"] = ((pbp["pass_touchdown"] == 1) | (pbp["rush_touchdown"] == 1)).astype(int)
                team_key = _pick_col(pbp, ["posteam", "offense_team"]) or "posteam"
                plays = pbp.groupby(team_key, as_index=False)[["pass_attempt", "rush_attempt", "sack", "complete_pass", "yards_gained", "off_td", "fg_made"]].sum()
                plays = plays.rename(columns={
                    team_key: "team",
                    "pass_attempt": "pass_attempts",
                    "rush_attempt": "rush_attempts",
                    "sack": "times_sacked",
                    "complete_pass": "completions",
                    "yards_gained": "yards_gained_sum",
                    "off_td": "td_plays",
                    "fg_made": "fg_made_plays",
                })
                agg = agg.merge(plays, on="team", how="left")
                # Penalties committed: by penalty_team
                if "penalty" in pbp.columns:
                    pbp["penalty"] = pd.to_numeric(pbp["penalty"], errors="coerce").fillna(0).astype(int)
                else:
                    pbp["penalty"] = 0
                pen_team = _pick_col(pbp, ["penalty_team"]) or None
                if pen_team:
                    pens = pbp[pbp["penalty"] == 1].groupby(pen_team, as_index=False)["penalty"].sum().rename(columns={pen_team: "team", "penalty": "penalties_committed"})
                    agg = agg.merge(pens, on="team", how="left")
            except Exception:
                pass

            agg["points_diff"] = agg["points_for"] - agg["points_against"]
            # Derived rates/averages
            if set(["pass_attempts", "rush_attempts", "times_sacked"]).issubset(agg.columns):
                agg["total_off_plays"] = agg[["pass_attempts", "rush_attempts", "times_sacked"]].fillna(0).sum(axis=1)
            if "yards_gained_sum" in agg.columns:
                agg["yards_per_play"] = (agg["yards_gained_sum"].fillna(0) / agg["total_off_plays"].where(agg["total_off_plays"]>0, 1)).round(2)
            if "completions" in agg.columns:
                agg["avg_completions"] = (agg["completions"].fillna(0) / agg["games_played"].where(agg["games_played"]>0, 1)).round(1)
            if "off_rush_yards" in agg.columns and "rush_attempts" in agg.columns:
                agg["rush_yards_per_att"] = (agg["off_rush_yards"].fillna(0) / agg["rush_attempts"].where(agg["rush_attempts"]>0, 1)).round(2)
            if "penalties_committed" in agg.columns:
                agg["avg_penalties_committed"] = (agg["penalties_committed"].fillna(0) / agg["games_played"].where(agg["games_played"]>0, 1)).round(1)
            if set(["td_plays", "fg_made_plays", "total_off_plays"]).issubset(agg.columns):
                agg["scoring_play_pct"] = ((agg["td_plays"].fillna(0) + agg["fg_made_plays"].fillna(0)) / agg["total_off_plays"].where(agg["total_off_plays"]>0, 1) * 100).round(1)
            # New: attempt averages per game up to selected week
            if "pass_attempts" in agg.columns:
                agg["avg_pass_attempts"] = (agg["pass_attempts"].fillna(0) / agg["games_played"].where(agg["games_played"]>0, 1)).round(1)
            if "rush_attempts" in agg.columns:
                agg["avg_rush_attempts"] = (agg["rush_attempts"].fillna(0) / agg["games_played"].where(agg["games_played"]>0, 1)).round(1)
            # Order columns
            order = [
                "team", "games_played", "wins", "wins_home", "wins_away",
                "points_for", "points_against", "points_diff",
                "pass_attempts", "rush_attempts", "times_sacked",
                "avg_pass_attempts", "avg_rush_attempts",
                "yards_per_play", "avg_completions", "rush_yards_per_att", "avg_penalties_committed", "scoring_play_pct",
                "off_pass_yards", "off_rush_yards", "off_pass_tds", "off_rush_tds",
                "def_pass_yards_allowed", "def_rush_yards_allowed", "def_interceptions", "def_sacks",
            ]
            cols = [c for c in order if c in agg.columns]
            agg = agg[cols].fillna(0)
            # Sort by wins then point diff
            sort_cols = [c for c in ["wins", "points_diff"] if c in agg.columns]
            agg = agg.sort_values(sort_cols, ascending=[False, False]) if sort_cols else agg
            try:
                styler = agg.style.background_gradient(cmap="YlGnBu")
                st.dataframe(styler, use_container_width=True)
            except Exception:
                st.dataframe(agg, use_container_width=True)

elif view == "Team Comparison":
    st.subheader(f"Team Comparison {season}{' through Week ' + str(week) if week else ''}")
    
    # Use cached team metrics
    metrics_df = get_team_metrics(season, week)
    if metrics_df.empty:
        st.info("No team metrics available for this season/week.")
    else:
        # Build schedule context for played_ids (for pbp filtering below)
        sched = _read_all(str(DATA_BASE / f"schedule/season={season}/week=*/schedule.parquet"))
        if "week" in sched.columns and week:
            sched = sched[sched["week"] <= week]
        h_col = _pick_col(sched, ["home_team", "home_team_abbr", "home"]) 
        a_col = _pick_col(sched, ["away_team", "away_team_abbr", "away"]) 
        hs_col = _pick_col(sched, ["home_score", "home_points", "home_pts"]) 
        as_col = _pick_col(sched, ["away_score", "away_points", "away_pts"]) 
        if not all([h_col, a_col, hs_col, as_col]):
            st.info("Schedule missing required columns for comparison.")
        else:
            agg = metrics_df.copy()

            # PBP-derived attempts, sacks, yards/play, etc. (align with Team Overview)
            try:
                import nfl_data_py as nfl
                pbp = nfl.import_pbp_data([season])
                pbp.columns = [c.lower() for c in pbp.columns]
                if "week" in pbp.columns and week:
                    pbp = pbp[pbp["week"] <= week]
                played_ids = set(sched[sched[hs_col].notna() & sched[as_col].notna()].get("game_id", pd.Series(dtype=str))) if "game_id" in sched.columns else None
                if played_ids:
                    pbp = pbp[pbp["game_id"].isin(played_ids)]
                for c in ["pass_attempt", "rush_attempt", "sack", "complete_pass", "pass_touchdown", "rush_touchdown"]:
                    pbp[c] = pd.to_numeric(pbp.get(c, 0), errors="coerce").fillna(0).astype(int)
                pbp["yards_gained"] = pd.to_numeric(pbp.get("yards_gained", 0), errors="coerce").fillna(0)
                pbp["fg_made"] = (pbp.get("field_goal_result", "").astype(str).str.lower() == "made").astype(int)
                pbp["off_td"] = ((pbp["pass_touchdown"] == 1) | (pbp["rush_touchdown"] == 1)).astype(int)
                team_key = _pick_col(pbp, ["posteam", "offense_team"]) or "posteam"
                plays = pbp.groupby(team_key, as_index=False)[["pass_attempt", "rush_attempt", "sack", "complete_pass", "yards_gained", "off_td", "fg_made"]].sum()
                plays = plays.rename(columns={
                    team_key: "team",
                    "pass_attempt": "pass_attempts",
                    "rush_attempt": "rush_attempts",
                    "sack": "times_sacked",
                    "complete_pass": "completions",
                    "yards_gained": "yards_gained_sum",
                    "off_td": "td_plays",
                    "fg_made": "fg_made_plays",
                })
                agg = agg.merge(plays, on="team", how="left")
            except Exception:
                pass

            # Derivations
            if set(["pass_attempts", "rush_attempts", "times_sacked"]).issubset(agg.columns):
                agg["total_off_plays"] = agg[["pass_attempts", "rush_attempts", "times_sacked"]].fillna(0).sum(axis=1)
            if "yards_gained_sum" in agg.columns:
                agg["yards_per_play"] = (agg["yards_gained_sum"].fillna(0) / agg["total_off_plays"].where(agg["total_off_plays"]>0, 1)).round(2)
            if "completions" in agg.columns and "games_played" in agg.columns:
                agg["avg_completions"] = (agg["completions"].fillna(0) / agg["games_played"].where(agg["games_played"]>0, 1)).round(1)
            if "off_rush_yards" in agg.columns and "rush_attempts" in agg.columns:
                agg["rush_yards_per_att"] = (agg["off_rush_yards"].fillna(0) / agg["rush_attempts"].where(agg["rush_attempts"]>0, 1)).round(2)

            teams = sorted(agg["team"].dropna().astype(str).unique().tolist())
            c1, c2 = st.columns(2)
            with c1:
                team_a = st.selectbox("Team A", teams, index=0 if teams else 0, key="cmp_a")
            with c2:
                team_b = st.selectbox("Team B", teams, index=1 if len(teams)>1 else 0, key="cmp_b")

            if teams and team_a in agg["team"].values and team_b in agg["team"].values:
                subset = agg[agg["team"].isin([team_a, team_b])].set_index("team").fillna(0)

                # Quick delta banner
                banner_metrics = [
                    ("points_for", "Points For"),
                    ("points_diff", "Point Diff"),
                    ("yards_per_play", "Yards/Play"),
                    ("rush_yards_per_att", "Rush Yds/Att"),
                ]
                cols = st.columns(len(banner_metrics))
                for (key, label), col in zip(banner_metrics, cols):
                    if key in subset.columns:
                        delta = float(subset.loc[team_a, key]) - float(subset.loc[team_b, key])
                        col.metric(label=label, value=f"{subset.loc[team_a, key]:.2f}", delta=f"{delta:+.2f} vs {team_b}")

                st.markdown("---")

                # Simple comparison table with delta
                metrics = [
                    "games_played","wins","wins_home","wins_away",
                    "points_for","points_against","points_diff",
                    "yards_per_play","rush_yards_per_att","avg_completions",
                    "pass_attempts","rush_attempts","times_sacked",
                    "off_pass_yards","off_rush_yards","off_pass_tds","off_rush_tds",
                    "def_pass_yards_allowed","def_rush_yards_allowed","def_interceptions","def_sacks",
                    "avg_penalties_committed","scoring_play_pct",
                ]
                present = [m for m in metrics if m in subset.columns]
                show = subset[present].T
                show.columns = [team_a, team_b]
                # numeric-safe delta and formatting
                for col in [team_a, team_b]:
                    show[col] = pd.to_numeric(show[col], errors="coerce")
                show["Δ (A-B)"] = (show[team_a].fillna(0) - show[team_b].fillna(0))
                try:
                    styler = (
                        show.style
                        .format({team_a: "{:.2f}", team_b: "{:.2f}", "Δ (A-B)": "{:+.2f}"})
                        .set_table_styles([
                            {
                                "selector": "th.col_heading, th.row_heading, td",
                                "props": [
                                    ("padding", "2px 4px"),
                                    ("font-size", "11px"),
                                    ("min-width", "56px"),
                                    ("width", "56px"),
                                    ("max-width", "70px"),
                                ],
                            }
                        ])
                    )
                    st.dataframe(styler, use_container_width=True)
                except Exception:
                    st.dataframe(show, use_container_width=True)

                # Matchup Predictions Section
                st.markdown("---")
                st.markdown("## 🎯 Matchup Predictions")
                st.markdown("Expected player performance based on team defensive stats and player averages")
                
                # Calculate per-game averages for defensive stats
                def calc_per_game_def(team_data, games_played):
                    """Calculate per-game defensive stats"""
                    if games_played == 0:
                        return {}
                    return {
                        'pass_yards_allowed_pg': team_data.get('def_pass_yards_allowed', 0) / games_played,
                        'rush_yards_allowed_pg': team_data.get('def_rush_yards_allowed', 0) / games_played,
                        'sacks_pg': team_data.get('def_sacks', 0) / games_played,
                        'interceptions_pg': team_data.get('def_interceptions', 0) / games_played,
                        # Note: We'll calculate TD predictions based on league averages since we don't have direct defensive TD stats
                        'offensive_pass_tds_pg': team_data.get('off_pass_tds', 0) / games_played,
                        'offensive_rush_tds_pg': team_data.get('off_rush_tds', 0) / games_played,
                    }
                
                # Get defensive stats per game for both teams
                team_a_data = subset.loc[team_a].to_dict()
                team_b_data = subset.loc[team_b].to_dict()
                
                team_a_games = team_a_data.get('games_played', 1)
                team_b_games = team_b_data.get('games_played', 1)
                
                team_a_def = calc_per_game_def(team_a_data, team_a_games)
                team_b_def = calc_per_game_def(team_b_data, team_b_games)
                
                # Display prediction tables
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### {team_a} Offensive Predictions")
                    st.markdown(f"*vs {team_b} Defense*")
                    
                    pred_data_a = []
                    
                    # QB Predictions
                    if team_b_def['pass_yards_allowed_pg'] > 0:
                        pred_data_a.append({
                            'Position': '🏈 QB Passing',
                            'Team Defense Allows': f"{team_b_def['pass_yards_allowed_pg']:.0f} yds/game",
                            'Matchup': '🟢 Favorable' if team_b_def['pass_yards_allowed_pg'] > 250 else '🟡 Average' if team_b_def['pass_yards_allowed_pg'] > 220 else '🔴 Tough'
                        })
                    
                    # Passing TD Predictions
                    if team_b_def['offensive_pass_tds_pg'] > 0:
                        # Use league average as baseline (roughly 1.5 pass TDs per game)
                        league_avg_pass_tds = 1.5
                        td_prediction = "High" if team_b_def['offensive_pass_tds_pg'] < league_avg_pass_tds else "Average" if team_b_def['offensive_pass_tds_pg'] < 2.0 else "Low"
                        pred_data_a.append({
                            'Position': '🎯 Passing TDs',
                            'Team Defense Allows': f"{team_b_def['offensive_pass_tds_pg']:.1f} TDs/game",
                            'Matchup': f'🟢 {td_prediction} TD Potential' if td_prediction == "High" else f'🟡 {td_prediction} TD Potential' if td_prediction == "Average" else f'🔴 {td_prediction} TD Potential'
                        })
                    
                    # RB Predictions  
                    if team_b_def['rush_yards_allowed_pg'] > 0:
                        pred_data_a.append({
                            'Position': '🏃 RB Rushing',
                            'Team Defense Allows': f"{team_b_def['rush_yards_allowed_pg']:.0f} yds/game",
                            'Matchup': '🟢 Favorable' if team_b_def['rush_yards_allowed_pg'] > 120 else '🟡 Average' if team_b_def['rush_yards_allowed_pg'] > 100 else '🔴 Tough'
                        })
                    
                    # Rushing TD Predictions
                    if team_b_def['offensive_rush_tds_pg'] > 0:
                        # Use league average as baseline (roughly 1.0 rush TDs per game)
                        league_avg_rush_tds = 1.0
                        td_prediction = "High" if team_b_def['offensive_rush_tds_pg'] < league_avg_rush_tds else "Average" if team_b_def['offensive_rush_tds_pg'] < 1.5 else "Low"
                        pred_data_a.append({
                            'Position': '💪 Rushing TDs',
                            'Team Defense Allows': f"{team_b_def['offensive_rush_tds_pg']:.1f} TDs/game",
                            'Matchup': f'🟢 {td_prediction} TD Potential' if td_prediction == "High" else f'🟡 {td_prediction} TD Potential' if td_prediction == "Average" else f'🔴 {td_prediction} TD Potential'
                        })
                    
                    # Pass Rush Pressure
                    if team_b_def['sacks_pg'] > 0:
                        pred_data_a.append({
                            'Position': '⚡ Pass Protection',
                            'Team Defense Allows': f"{team_b_def['sacks_pg']:.1f} sacks/game",
                            'Matchup': '🔴 Tough' if team_b_def['sacks_pg'] > 2.5 else '🟡 Average' if team_b_def['sacks_pg'] > 1.5 else '🟢 Favorable'
                        })
                    
                    if pred_data_a:
                        pred_df_a = pd.DataFrame(pred_data_a)
                        st.dataframe(pred_df_a, hide_index=True, use_container_width=True)
                
                with col2:
                    st.markdown(f"### {team_b} Offensive Predictions")
                    st.markdown(f"*vs {team_a} Defense*")
                    
                    pred_data_b = []
                    
                    # QB Predictions
                    if team_a_def['pass_yards_allowed_pg'] > 0:
                        pred_data_b.append({
                            'Position': '🏈 QB Passing',
                            'Team Defense Allows': f"{team_a_def['pass_yards_allowed_pg']:.0f} yds/game",
                            'Matchup': '🟢 Favorable' if team_a_def['pass_yards_allowed_pg'] > 250 else '🟡 Average' if team_a_def['pass_yards_allowed_pg'] > 220 else '🔴 Tough'
                        })
                    
                    # Passing TD Predictions
                    if team_a_def['offensive_pass_tds_pg'] > 0:
                        league_avg_pass_tds = 1.5
                        td_prediction = "High" if team_a_def['offensive_pass_tds_pg'] < league_avg_pass_tds else "Average" if team_a_def['offensive_pass_tds_pg'] < 2.0 else "Low"
                        pred_data_b.append({
                            'Position': '🎯 Passing TDs',
                            'Team Defense Allows': f"{team_a_def['offensive_pass_tds_pg']:.1f} TDs/game",
                            'Matchup': f'🟢 {td_prediction} TD Potential' if td_prediction == "High" else f'🟡 {td_prediction} TD Potential' if td_prediction == "Average" else f'🔴 {td_prediction} TD Potential'
                        })
                    
                    # RB Predictions
                    if team_a_def['rush_yards_allowed_pg'] > 0:
                        pred_data_b.append({
                            'Position': '🏃 RB Rushing',
                            'Team Defense Allows': f"{team_a_def['rush_yards_allowed_pg']:.0f} yds/game",
                            'Matchup': '🟢 Favorable' if team_a_def['rush_yards_allowed_pg'] > 120 else '🟡 Average' if team_a_def['rush_yards_allowed_pg'] > 100 else '🔴 Tough'
                        })
                    
                    # Rushing TD Predictions
                    if team_a_def['offensive_rush_tds_pg'] > 0:
                        league_avg_rush_tds = 1.0
                        td_prediction = "High" if team_a_def['offensive_rush_tds_pg'] < league_avg_rush_tds else "Average" if team_a_def['offensive_rush_tds_pg'] < 1.5 else "Low"
                        pred_data_b.append({
                            'Position': '💪 Rushing TDs',
                            'Team Defense Allows': f"{team_a_def['offensive_rush_tds_pg']:.1f} TDs/game",
                            'Matchup': f'🟢 {td_prediction} TD Potential' if td_prediction == "High" else f'🟡 {td_prediction} TD Potential' if td_prediction == "Average" else f'🔴 {td_prediction} TD Potential'
                        })
                    
                    # Pass Rush Pressure
                    if team_a_def['sacks_pg'] > 0:
                        pred_data_b.append({
                            'Position': '⚡ Pass Protection',
                            'Team Defense Allows': f"{team_a_def['sacks_pg']:.1f} sacks/game",
                            'Matchup': '🔴 Tough' if team_a_def['sacks_pg'] > 2.5 else '🟡 Average' if team_a_def['sacks_pg'] > 1.5 else '🟢 Favorable'
                        })
                    
                    if pred_data_b:
                        pred_df_b = pd.DataFrame(pred_data_b)
                        st.dataframe(pred_df_b, hide_index=True, use_container_width=True)

                # Per-position player data for both teams
                st.markdown("---")
                st.markdown("## Players by Position Analysis")
                st.markdown("Detailed stats broken down by position with totals, medians, and advanced metrics")
                
                pw_all = _read_all(str(DATA_BASE / f"player_week/season={season}/week=*/player_week.parquet"))
                if pw_all.empty:
                    st.info("No player-week data available to derive player tables.")
                else:
                    if "week" in pw_all.columns and week:
                        pw_all = pw_all[pw_all["week"] <= week]

                    # Identify columns robustly
                    tcol = _pick_team_column(pw_all)
                    name_col = _pick_col(pw_all, ["full_name", "player_display_name", "player_name", "name"]) or "player_id"
                    pos_col = _pick_col(pw_all, ["position", "pos", "player_position", "role"]) or "position"
                    pass_y = _pick_col(pw_all, ["passing_yards", "pass_yards", "pass_yds"]) or "passing_yards"
                    rush_y = _pick_col(pw_all, ["rushing_yards", "rush_yards", "rush_yds"]) or "rushing_yards"
                    rec_y = _pick_col(pw_all, ["receiving_yards", "rec_yards", "rec_yds"]) or "receiving_yards"
                    pass_td_col = _pick_col(pw_all, ["passing_tds", "pass_td", "pass_touchdown"]) 
                    rush_td_col = _pick_col(pw_all, ["rushing_tds", "rush_td", "rush_touchdown"]) 
                    rec_td_col = _pick_col(pw_all, ["receiving_tds", "receiving_touchdowns", "rec_td", "rec_tds"])
                    rush_att_col = _pick_col(pw_all, ["carries", "rushing_attempts", "rush_attempts", "rush_att"])
                    rec_col = _pick_col(pw_all, ["receptions", "rec", "catches"])
                    targets_col = _pick_col(pw_all, ["targets", "tgt"]) 

                    # Normalize numeric
                    for c in [pass_y, rush_y, rec_y, pass_td_col, rush_td_col, rush_att_col, rec_col, targets_col, "week"]:
                        if c and c in pw_all.columns:
                            pw_all[c] = pd.to_numeric(pw_all[c], errors="coerce")

                    # Limit to the two teams first for efficiency
                    if tcol:
                        pw_teams = pw_all[pw_all[tcol].isin([team_a, team_b])].copy()
                    else:
                        pw_teams = pw_all.copy()

                    # Base renamed subset used for role filters
                    keep = [name_col, tcol, pos_col, pass_y, rush_y, rec_y, "week"]
                    keep = [c for c in keep if c and c in pw_teams.columns]
                    pw = pw_teams[keep].rename(columns={
                        name_col: "player", tcol or "team": "team", pos_col or "pos": "pos",
                        pass_y if pass_y in keep else "": "pass_yards",
                        rush_y if rush_y in keep else "": "rush_yards",
                        rec_y if rec_y in keep else "": "rec_yards",
                    })

                    # Robust helper: compute per-player per-week totals on the RAW df, then median across weeks
                    def median_by_week_raw(value_col: str, label: str, role_regex: str) -> pd.DataFrame:
                        if not value_col or value_col not in pw_teams.columns:
                            return pd.DataFrame(columns=["team", "player", label])
                        cols_needed = [tcol, name_col, pos_col, "week", value_col]
                        if not all(c and c in pw_teams.columns for c in cols_needed):
                            return pd.DataFrame(columns=["team", "player", label])
                        
                        src = pw_teams[cols_needed].copy()
                        # Filter by position
                        src = src[src[pos_col].astype(str).str.upper().str.contains(role_regex, na=False)]
                        if src.empty:
                            return pd.DataFrame(columns=["team", "player", label])
                            
                        src["week"] = pd.to_numeric(src["week"], errors="coerce")
                        src[value_col] = pd.to_numeric(src[value_col], errors="coerce").fillna(0)
                        
                        # Sum by player-week, then median across weeks per player
                        by_week = (
                            src.dropna(subset=["week"]) 
                               .groupby([tcol, name_col, "week"], as_index=False)[value_col]
                              .sum()
                        )
                        if by_week.empty:
                            return pd.DataFrame(columns=["team", "player", label])
                            
                        med = by_week.groupby([tcol, name_col], as_index=False)[value_col].median()
                        med = med.rename(columns={tcol: "team", name_col: "player", value_col: label})
                        return med

                    # QB table: totals, medians, and TDs
                    qb_src = pw[pw["pos"].astype(str).str.upper().str.contains("QB", na=False)].copy()
                    qb_tot = qb_src.groupby(["team", "player"], as_index=False)["pass_yards"].sum() if "pass_yards" in qb_src.columns else pd.DataFrame(columns=["team", "player", "pass_yards"])
                    
                    # Get medians  
                    qb_med_pass = median_by_week_raw(pass_y, "median_pass_yards", r"QB")
                    
                    # Get passing TDs - filter QB players from raw data and sum TDs
                    qb_td_data = pd.DataFrame(columns=["team", "player", "pass_tds"])
                    if tcol and name_col and pos_col and pass_td_col and pass_td_col in pw_teams.columns:
                        qb_raw = pw_teams[pw_teams[pos_col].astype(str).str.upper().str.contains("QB", na=False)].copy()
                        if not qb_raw.empty:
                            qb_td_data = qb_raw[[tcol, name_col, pass_td_col]].rename(columns={tcol: "team", name_col: "player", pass_td_col: "pass_tds"})
                            qb_td_data = qb_td_data.groupby(["team", "player"], as_index=False).sum(numeric_only=True)

                    # Merge all QB data
                    qb = qb_tot.copy()
                    if not qb_med_pass.empty:
                        qb = qb.merge(qb_med_pass, on=["team", "player"], how="left")
                    if not qb_td_data.empty:
                        qb = qb.merge(qb_td_data, on=["team", "player"], how="left")

                    # RB table: totals, medians, and TDs
                    rb_src = pw[pw["pos"].astype(str).str.upper().str.contains("RB", na=False)].copy()
                    rb_tot = rb_src.groupby(["team", "player"], as_index=False)["rush_yards"].sum() if "rush_yards" in rb_src.columns else pd.DataFrame(columns=["team", "player", "rush_yards"])
                    
                    # Get medians and TDs
                    rb_med = median_by_week_raw(rush_y, "median_rush_yards", r"RB")
                    
                    # Get RB rushing attempts and TDs
                    rb_stats_data = pd.DataFrame(columns=["team", "player", "rush_tds", "carries"])
                    if tcol and name_col and pos_col:
                        rb_raw = pw_teams[pw_teams[pos_col].astype(str).str.upper().str.contains("RB", na=False)].copy()
                        if not rb_raw.empty:
                            stat_cols = [tcol, name_col]
                            rename_map = {tcol: "team", name_col: "player"}
                            if rush_td_col and rush_td_col in rb_raw.columns:
                                stat_cols.append(rush_td_col)
                                rename_map[rush_td_col] = "rush_tds"
                            if rush_att_col and rush_att_col in rb_raw.columns:
                                stat_cols.append(rush_att_col)
                                rename_map[rush_att_col] = "carries"
                            
                            if len(stat_cols) > 2:  # More than just team and player
                                rb_stats_data = rb_raw[stat_cols].rename(columns=rename_map)
                                rb_stats_data = rb_stats_data.groupby(["team", "player"], as_index=False).sum(numeric_only=True)

                    # Merge all RB data
                    rb = rb_tot.copy()
                    if not rb_med.empty:
                        rb = rb.merge(rb_med, on=["team", "player"], how="left")
                    if not rb_stats_data.empty:
                        rb = rb.merge(rb_stats_data, on=["team", "player"], how="left")
                    
                    # Calculate yards per carry
                    if "rush_yards" in rb.columns and "carries" in rb.columns:
                        rb["yards_per_carry"] = (rb["rush_yards"] / rb["carries"].where(rb["carries"] > 0, 1)).round(2)

                    # WR/TE table: totals, medians, and TDs
                    wr_src = pw[pw["pos"].astype(str).str.upper().str.contains("WR|TE", na=False)].copy()
                    wr_tot = wr_src.groupby(["team", "player"], as_index=False)["rec_yards"].sum() if "rec_yards" in wr_src.columns else pd.DataFrame(columns=["team", "player", "rec_yards"])
                    
                    # Get medians and TDs
                    wr_med = median_by_week_raw(rec_y, "median_rec_yards", r"WR|TE")
                    
                    # Get WR/TE receiving stats (TDs, receptions, targets)
                    wr_stats_data = pd.DataFrame(columns=["team", "player", "rec_tds", "receptions", "targets"])
                    if tcol and name_col and pos_col:
                        wr_raw = pw_teams[pw_teams[pos_col].astype(str).str.upper().str.contains("WR|TE", na=False)].copy()
                        if not wr_raw.empty:
                            stat_cols = [tcol, name_col]
                            rename_map = {tcol: "team", name_col: "player"}
                            if rec_td_col and rec_td_col in wr_raw.columns:
                                stat_cols.append(rec_td_col)
                                rename_map[rec_td_col] = "rec_tds"
                            if rec_col and rec_col in wr_raw.columns:
                                stat_cols.append(rec_col)
                                rename_map[rec_col] = "receptions"
                            if targets_col and targets_col in wr_raw.columns:
                                stat_cols.append(targets_col)
                                rename_map[targets_col] = "targets"
                            
                            if len(stat_cols) > 2:  # More than just team and player
                                wr_stats_data = wr_raw[stat_cols].rename(columns=rename_map)
                                wr_stats_data = wr_stats_data.groupby(["team", "player"], as_index=False).sum(numeric_only=True)

                    # Merge all WR/TE data
                    wr = wr_tot.copy()
                    if not wr_med.empty:
                        wr = wr.merge(wr_med, on=["team", "player"], how="left")
                    if not wr_stats_data.empty:
                        wr = wr.merge(wr_stats_data, on=["team", "player"], how="left")
                    
                    # Calculate derived stats
                    if "receptions" in wr.columns and "targets" in wr.columns:
                        wr["catch_rate"] = (wr["receptions"] / wr["targets"].where(wr["targets"] > 0, 1) * 100).round(1)
                    if "rec_yards" in wr.columns and "receptions" in wr.columns:
                        wr["yards_per_reception"] = (wr["rec_yards"] / wr["receptions"].where(wr["receptions"] > 0, 1)).round(1)


                    # ===== QB SECTION =====
                    st.markdown("---")
                    st.markdown("### 🏈 QUARTERBACKS - Passing Stats")
                    st.markdown("*Total yards, median per game, and touchdown stats*")
                    qb_cols = ["team", "player", "pass_yards", "median_pass_yards", "pass_tds"]
                    for col in qb_cols:
                        if col not in qb.columns:
                            qb[col] = 0
                    qb_display = qb[qb_cols].fillna(0).round(1)
                    st.dataframe(qb_display, width="stretch", hide_index=True)
                    st.write("")  # Add spacing
                    
                    # ===== RB SECTION =====
                    st.markdown("---")
                    st.markdown("### 🏃 RUNNING BACKS - Rushing Stats") 
                    st.markdown("*Total yards, median per game, attempts, and efficiency*")
                    rb_cols = ["team", "player", "rush_yards", "median_rush_yards", "carries", "yards_per_carry", "rush_tds"]
                    for col in rb_cols:
                        if col not in rb.columns:
                            rb[col] = 0
                    rb_display = rb[rb_cols].fillna(0).round(1)
                    st.dataframe(rb_display, width="stretch", hide_index=True)
                    st.write("")  # Add spacing
                    
                    # ===== WR/TE SECTION =====
                    st.markdown("---")
                    st.markdown("### 🙌 RECEIVERS - Receiving Stats")
                    st.markdown("*Total yards, median per game, targets, and efficiency*")
                    wr_cols = ["team", "player", "rec_yards", "median_rec_yards", "receptions", "targets", "catch_rate", "rec_tds", "yards_per_reception"]
                    for col in wr_cols:
                        if col not in wr.columns:
                            wr[col] = 0
                    wr_display = wr[wr_cols].fillna(0).round(1)
                    st.dataframe(wr_display, width="stretch", hide_index=True)

                # League leaderboards (median yards) up to selected week
                st.markdown("---")
                st.markdown("### League Leaderboards (Median Yards up to Week)")
                pw_all2 = _read_all(str(DATA_BASE / f"player_week/season={season}/week=*/player_week.parquet"))
                if pw_all2.empty:
                    st.info("No data for leaderboards.")
                else:
                    if "week" in pw_all2.columns and week:
                        pw_all2 = pw_all2[pw_all2["week"] <= week]
                    name_col = _pick_col(pw_all2, ["full_name", "player_display_name", "player_name", "name"]) or "player_id"
                    pos_col = _pick_col(pw_all2, ["position", "pos", "player_position", "role"]) or "position"
                    pass_y = _pick_col(pw_all2, ["passing_yards", "pass_yards", "pass_yds"]) or "passing_yards"
                    rush_y = _pick_col(pw_all2, ["rushing_yards", "rush_yards", "rush_yds"]) or "rushing_yards"
                    rec_y = _pick_col(pw_all2, ["receiving_yards", "rec_yards", "rec_yds"]) or "receiving_yards"
                    df = pw_all2[[name_col, pos_col, pass_y, rush_y, rec_y]].rename(columns={
                        name_col: "player", pos_col: "pos",
                        pass_y: "pass_yards", rush_y: "rush_yards", rec_y: "rec_yards"
                    })
                    # Median aggregations
                    qb_med = df[df["pos"].astype(str).str.upper().str.contains("QB")].groupby("player", as_index=False)["pass_yards"].median().rename(columns={"pass_yards":"median_pass_yards"}).sort_values("median_pass_yards", ascending=False).head(10)
                    rb_med = df[df["pos"].astype(str).str.upper().str.contains("RB")].groupby("player", as_index=False)["rush_yards"].median().rename(columns={"rush_yards":"median_rush_yards"}).sort_values("median_rush_yards", ascending=False).head(10)
                    wr_med = df[df["pos"].astype(str).str.upper().str.contains("WR|TE")].groupby("player", as_index=False)["rec_yards"].median().rename(columns={"rec_yards":"median_rec_yards"}).sort_values("median_rec_yards", ascending=False).head(10)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.write("QB median passing yards")
                        st.dataframe(qb_med, use_container_width=True)
                    with c2:
                        st.write("RB median rushing yards")
                        st.dataframe(rb_med, use_container_width=True)
                    with c3:
                        st.write("WR/TE median receiving yards")
                        st.dataframe(wr_med, use_container_width=True)
elif view == "Analytics Chart":
    st.subheader(f"QB Analytics Charts — {season}{' through Week ' + str(week) if week else ''}")

    # Load player week data
    pw_df = _read_all(str(DATA_BASE / f"player_week/season={season}/week=*/player_week.parquet"))
    if pw_df.empty:
        st.info("No player-week data available for this season.")
    else:
        # Filter by week if specified
        if "week" in pw_df.columns and week:
            pw_df = pw_df[pw_df["week"] <= week]

        # Identify column names robustly
        name_col = _pick_col(pw_df, ["full_name", "player_display_name", "player_name", "name"]) or "player_id"
        pos_col = _pick_col(pw_df, ["position", "pos", "player_position", "role"]) or "position"
        team_col = _pick_team_column(pw_df)
        pass_y = _pick_col(pw_df, ["passing_yards", "pass_yards", "pass_yds"])
        pass_td = _pick_col(pw_df, ["passing_tds", "pass_td", "pass_touchdown"])
        pass_att = _pick_col(pw_df, ["pass_attempts", "passing_attempts", "attempts"])
        ints_thrown = _pick_col(pw_df, ["interceptions", "int", "interceptions_thrown"])

        # Filter for QBs only
        qb_df = pw_df[pw_df[pos_col].astype(str).str.upper().str.contains("QB", na=False)].copy()

        if qb_df.empty:
            st.info("No QB data available for the selected season/week.")
        else:
            # Prepare columns for analysis
            cols_to_keep = [name_col, team_col, pass_y, pass_td, pass_att, ints_thrown]
            cols_to_keep = [c for c in cols_to_keep if c and c in qb_df.columns]

            qb_data = qb_df[cols_to_keep].copy()

            # Rename columns for easier access
            rename_map = {name_col: "player"}
            if team_col:
                rename_map[team_col] = "team"
            if pass_y:
                rename_map[pass_y] = "passing_yards"
            if pass_td:
                rename_map[pass_td] = "passing_tds"
            if pass_att:
                rename_map[pass_att] = "pass_attempts"
            if ints_thrown:
                rename_map[ints_thrown] = "interceptions"

            qb_data = qb_data.rename(columns=rename_map)

            # Convert to numeric
            for col in ["passing_yards", "passing_tds", "pass_attempts", "interceptions"]:
                if col in qb_data.columns:
                    qb_data[col] = pd.to_numeric(qb_data[col], errors="coerce").fillna(0)

            # Aggregate by player (sum across all weeks)
            agg_cols = ["player"]
            if "team" in qb_data.columns:
                agg_cols.append("team")

            numeric_cols = [c for c in ["passing_yards", "passing_tds", "pass_attempts", "interceptions"] if c in qb_data.columns]
            if numeric_cols:
                qb_agg = qb_data.groupby(agg_cols, as_index=False)[numeric_cols].sum()
            else:
                st.warning("Required QB statistics columns not found in the data.")
                qb_agg = pd.DataFrame()

            if not qb_agg.empty:
                # Chart 1: QB Passing Touchdowns vs. Interceptions
                if "passing_tds" in qb_agg.columns and "interceptions" in qb_agg.columns:
                    st.markdown("### QB Passing Touchdowns vs. Interceptions")
                    st.markdown("*Shows the TD-to-INT ratio for QBs. Top-left quadrant represents QBs with high TDs and low INTs.*")

                    try:
                        import plotly.express as px

                        # Filter out QBs with 0 TDs and 0 INTs (likely minimal playing time)
                        chart_df = qb_agg[(qb_agg["passing_tds"] > 0) | (qb_agg["interceptions"] > 0)].copy()

                        if not chart_df.empty:
                            fig = px.scatter(
                                chart_df,
                                x="interceptions",
                                y="passing_tds",
                                hover_data=["player"] + (["team"] if "team" in chart_df.columns else []),
                                text="player",
                                labels={"interceptions": "Interceptions", "passing_tds": "Passing Touchdowns"},
                                title="QB Passing TDs vs Interceptions"
                            )
                            fig.update_traces(textposition="top center", marker=dict(size=12))
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No QB data available with passing TDs or interceptions.")
                    except ImportError:
                        st.warning("Plotly is required for interactive charts. Install with: pip install plotly")

                st.markdown("---")

                # Chart 2: QB Passing Yards vs. Attempts
                if "passing_yards" in qb_agg.columns and "pass_attempts" in qb_agg.columns:
                    st.markdown("### QB Passing Yards vs. Attempts")
                    st.markdown("*Shows passing efficiency. Steeper slopes indicate higher yards per attempt.*")

                    try:
                        import plotly.express as px

                        # Filter out QBs with very few attempts (< 10 total)
                        chart_df = qb_agg[qb_agg["pass_attempts"] >= 10].copy()

                        if not chart_df.empty:
                            # Calculate yards per attempt for color coding
                            chart_df["yards_per_attempt"] = (chart_df["passing_yards"] / chart_df["pass_attempts"]).round(2)

                            fig = px.scatter(
                                chart_df,
                                x="pass_attempts",
                                y="passing_yards",
                                hover_data=["player", "yards_per_attempt"] + (["team"] if "team" in chart_df.columns else []),
                                text="player",
                                color="yards_per_attempt",
                                labels={
                                    "pass_attempts": "Pass Attempts",
                                    "passing_yards": "Passing Yards",
                                    "yards_per_attempt": "Yards/Attempt"
                                },
                                title="QB Passing Yards vs Attempts",
                                color_continuous_scale="Viridis"
                            )
                            fig.update_traces(textposition="top center", marker=dict(size=12))
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No QB data available with sufficient pass attempts (minimum 10).")
                    except ImportError:
                        st.warning("Plotly is required for interactive charts. Install with: pip install plotly")

elif view == "Skill Yards Grid":
    st.subheader(f"RB/WR/TE Combined Rush + Receiving Yards — {season}{' (<= Week ' + str(week) + ')' if week else ''}")
    df = _read_all(str(DATA_BASE / f"player_week/season={season}/week=*/player_week.parquet"))
    if df.empty:
        st.info("No player-week data available for this season.")
    else:
        # Optional cutoff by selected week
        if "week" in df.columns and week:
            df = df[df["week"] <= week]

        # Identify columns
        name_col = _pick_col(df, ["full_name", "player_display_name", "player_name", "name", "player"]) or (
            "player_id" if "player_id" in df.columns else None
        )
        team_col = _pick_team_column(df)
        pos_col = _pick_col(df, ["position", "pos", "player_position", "role"])  # 'role' exists in derived data
        rush_col = _pick_col(df, ["rushing_yards", "rush_yards", "rush_yds"])  
        rec_col = _pick_col(df, ["receiving_yards", "rec_yards", "rec_yds"])

        # For this view we ignore Team filter: show all players league-wide

        # Filter to skill positions
        if pos_col and pos_col in df.columns:
            df = df[df[pos_col].astype(str).isin(["RB", "WR", "TE", "WR/TE"])]

        # Compute combined yards, guard against missing columns
        for c in [rush_col, rec_col]:
            if c and c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        combined = (
            (df[rush_col] if rush_col in df.columns else 0)
            + (df[rec_col] if rec_col in df.columns else 0)
        )
        df = df.assign(combined_yards=combined)

        # Always try to enrich to full display names when player_id exists
        if "player_id" in df.columns:
            try:
                import nfl_data_py as nfl
                players = nfl.import_players()
                players.columns = [c.lower() for c in players.columns]
                id_col = _pick_col(players, ["gsis_id", "player_id", "nfl_id"]) or "gsis_id"
                disp_col = _pick_col(players, ["player_display_name", "display_name", "full_name"]) or "player_display_name"
                if id_col in players.columns and disp_col in players.columns:
                    name_map = players.set_index(id_col)[disp_col].dropna().to_dict()
                    df["display_name"] = df["player_id"].map(name_map)
                    # Prefer mapped display_name, fallback to existing name_col or player_id
                    df["display_name"] = df["display_name"].fillna(df.get(name_col, df["player_id"]))
                    name_col = "display_name"
            except Exception:
                # If lookup fails, keep existing name_col
                pass

        if df.empty or "week" not in df.columns or not name_col or name_col not in df.columns or not team_col or team_col not in df.columns:
            st.info("Required columns not present to build the grid.")
        else:
            # Ensure week numeric and aggregate robustly
            df["week"] = pd.to_numeric(df["week"], errors="coerce")
            tmp = df[[name_col, team_col, "week", "combined_yards"]].dropna(subset=["week"]).copy()
            tmp["week"] = tmp["week"].astype(int)
            grouped = tmp.groupby([name_col, team_col, "week"], as_index=False)["combined_yards"].sum()
            # Build matrix without pivot/unstack to avoid pandas version quirks
            row_keys = list(zip(grouped[name_col].astype(str), grouped[team_col].astype(str)))
            weeks = grouped["week"].astype(int).tolist()
            values = grouped["combined_yards"].astype(int).tolist()
            data: dict[tuple[str, str], dict[int, int]] = {}
            for key, w, v in zip(row_keys, weeks, values):
                if key not in data:
                    data[key] = {}
                data[key][w] = data[key].get(w, 0) + int(v)
            grid = pd.DataFrame.from_dict(data, orient="index").fillna(0).astype(int)
            # Sort week columns ascending
            grid = grid.reindex(sorted(grid.columns), axis=1)
            # Move player/team from index to columns and compute Total
            grid.index = pd.MultiIndex.from_tuples(grid.index, names=["player", "team"])
            grid = grid.reset_index()
            week_cols = [c for c in grid.columns if isinstance(c, int)]
            grid["Total"] = grid[week_cols].sum(axis=1).astype(int) if week_cols else 0
            # Reorder columns: Player, Team, Total, weeks...
            grid = grid[["player", "team", "Total"] + week_cols]
            # Sortable native dataframe
            st.dataframe(grid, use_container_width=True)

elif view == "First TD Grid":
    st.subheader(f"First Offensive TD Scorers (Grid) — {season}{' (<= Week ' + str(week) + ')' if week else ''}")
    pbp, pbp_err = _load_pbp_local_first(season, week)
    if pbp_err:
        st.info(f"Unable to load play-by-play: {pbp_err}")
    if not pbp.empty:
        if week is not None and "week" in pbp.columns:
            pbp = pbp[pbp["week"] <= week]

        # Normalize flags
        for c in ["touchdown", "rush_touchdown", "pass_touchdown", "defensive_td", "return_touchdown"]:
            if c in pbp.columns:
                pbp[c] = pd.to_numeric(pbp[c], errors="coerce").fillna(0).astype(int)
            else:
                pbp[c] = 0

        td = pbp[
            (pbp["touchdown"] == 1)
            & ((pbp["rush_touchdown"] == 1) | (pbp["pass_touchdown"] == 1))
            & (pbp["defensive_td"] == 0)
            & (pbp["return_touchdown"] == 0)
        ].copy()

        # Identify scorer id/name and team
        if "td_player_id" in td.columns:
            td["scorer_id"] = td["td_player_id"]
        else:
            td["scorer_id"] = td["rusher_player_id"].where(td["rush_touchdown"] == 1, td.get("receiver_player_id"))
        if "td_player_name" in td.columns:
            td["scorer_name"] = td["td_player_name"]
        else:
            td["scorer_name"] = td["rusher_player_name"].where(td["rush_touchdown"] == 1, td.get("receiver_player_name"))

        team_src = _pick_col(td, ["posteam", "offense_team", "td_team"]) or "posteam"
        td["team"] = td.get(team_src, None)

        # Sort within game to get first TD
        sort_cols = ["game_id"]
        if "play_id" in td.columns:
            sort_cols.append("play_id")
            td = td.sort_values(sort_cols)
        elif "game_seconds_remaining" in td.columns:
            td = td.sort_values(["game_id", "game_seconds_remaining"], ascending=[True, False])
        first_td = td.groupby("game_id", as_index=False).head(1)

        # Map to full display names when possible
        try:
            if "scorer_id" in first_td.columns:
                players = nfl.import_players()
                players.columns = [c.lower() for c in players.columns]
                id_col = _pick_col(players, ["gsis_id", "player_id", "nfl_id"]) or "gsis_id"
                disp_col = _pick_col(players, ["player_display_name", "display_name", "full_name"]) or "player_display_name"
                name_map = players.set_index(id_col)[disp_col].dropna().to_dict()
                first_td["scorer_name"] = first_td["scorer_id"].map(name_map).fillna(first_td["scorer_name"])
        except Exception:
            pass

        # Build grid: rows=(player, team), columns=week, value=1 if first TD in that game-week
        if first_td.empty or "week" not in first_td.columns:
            st.info("No first TD data available for this selection.")
        else:
            first_td["week"] = pd.to_numeric(first_td["week"], errors="coerce").fillna(0).astype(int)
            row_keys = list(zip(first_td["scorer_name"].astype(str), first_td["team"].astype(str)))
            weeks = first_td["week"].tolist()
            data: dict[tuple[str, str], dict[int, int]] = {}
            for key, w in zip(row_keys, weeks):
                if key not in data:
                    data[key] = {}
                data[key][w] = 1  # first TD indicator
            grid = pd.DataFrame.from_dict(data, orient="index").fillna(0).astype(int)
            grid = grid.reindex(sorted(grid.columns), axis=1)
            grid.index = pd.MultiIndex.from_tuples(grid.index, names=["player", "team"])
            grid = grid.reset_index()
            week_cols = [c for c in grid.columns if isinstance(c, int)]
            grid["Total"] = grid[week_cols].sum(axis=1).astype(int) if week_cols else 0
            grid = grid[["player", "team", "Total"] + week_cols]
            # Interactive controls: team filter and sortable table
            try:
                teams_opt = ["All"] + sorted(grid["team"].astype(str).unique().tolist())
            except Exception:
                teams_opt = ["All"]
            sel_team = st.selectbox("Filter team (First TD Grid)", teams_opt, index=0, key="ftg_team_filter_pbp")
            if sel_team != "All" and "team" in grid.columns:
                grid = grid[grid["team"].astype(str) == sel_team]
            st.dataframe(grid, use_container_width=True)
    else:
        st.info("No play-by-play available locally or remotely for this selection.")
        # Fallback: show TD scorers grid (not first TD order) using player_week
        df = _read_all(str(DATA_BASE / f"player_week/season={season}/week=*/player_week.parquet"))
        if df.empty:
            st.info("No player-week data available for fallback.")
        else:
            if week and "week" in df.columns:
                df = df[df["week"] <= week]
            name_col = _pick_col(df, ["full_name", "player_display_name", "player_name", "name", "player"]) or (
                "player_id" if "player_id" in df.columns else None
            )
            team_col = _pick_team_column(df)
            pos_col = _pick_col(df, ["position", "pos", "player_position", "role"])
            rush_td_col = _pick_col(df, ["rushing_tds", "rushing_touchdowns", "rush_td"])  
            rec_td_col = _pick_col(df, ["receiving_tds", "receiving_touchdowns", "rec_td", "rec_tds"])
            if not all([name_col, team_col, "week" in df.columns]):
                st.info("Fallback requires player name, team, and week columns.")
            else:
                # Include RB/WR/TE. Include QBs only if they have rushing or receiving TDs
                for c in [rush_td_col, rec_td_col]:
                    if c and c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
                if pos_col and pos_col in df.columns:
                    pos_up = df[pos_col].astype(str).str.upper()
                    has_rb_wr_te = pos_up.isin(["RB", "WR", "TE", "WR/TE"])
                    qb_rush_rec = (pos_up == "QB") & (
                        (df.get(rush_td_col, 0) > 0) | (df.get(rec_td_col, 0) > 0)
                    )
                    df = df[has_rb_wr_te | qb_rush_rec]
                td_any = (
                    (df[rush_td_col] if rush_td_col in df.columns else 0)
                    + (df[rec_td_col] if rec_td_col in df.columns else 0)
                )
                tmp = df.assign(td_any=td_any)
                tmp = tmp[[name_col, team_col, "week", "td_any"]].dropna(subset=["week"]).copy()
                tmp["week"] = pd.to_numeric(tmp["week"], errors="coerce").fillna(0).astype(int)
                # Binary indicator if player scored any TD that week
                tmp["td_any"] = (pd.to_numeric(tmp["td_any"], errors="coerce").fillna(0) > 0).astype(int)
                # Build grid similar to First TD Grid (but 'any TD')
                row_keys = list(zip(tmp[name_col].astype(str), tmp[team_col].astype(str)))
                weeks = tmp["week"].tolist()
                vals = tmp["td_any"].tolist()
                data: dict[tuple[str, str], dict[int, int]] = {}
                for key, w, v in zip(row_keys, weeks, vals):
                    if key not in data:
                        data[key] = {}
                    data[key][w] = max(int(v), data[key].get(w, 0))
                grid = pd.DataFrame.from_dict(data, orient="index").fillna(0).astype(int)
                grid = grid.reindex(sorted(grid.columns), axis=1)
                grid.index = pd.MultiIndex.from_tuples(grid.index, names=["player", "team"])
                grid = grid.reset_index()
                week_cols = [c for c in grid.columns if isinstance(c, int)]
                grid["Total Weeks with TD"] = grid[week_cols].sum(axis=1).astype(int) if week_cols else 0
                grid = grid[["player", "team", "Total Weeks with TD"] + week_cols]
                st.caption("Fallback view (TD scored by week; NOT first TD order)")
                teams_opt = ["All"] + sorted(grid["team"].astype(str).unique().tolist())
                sel_team = st.selectbox("Filter team (TD-by-week)", teams_opt, index=0, key="ftg_team_filter_fb")
                if sel_team != "All":
                    grid = grid[grid["team"].astype(str) == sel_team]
                st.dataframe(grid, use_container_width=True)

elif view == "First TD":
    st.subheader(f"First Offensive TD Scorer — {season}{(' Week ' + str(week)) if week else ''}")
    pbp, pbp_err = _load_pbp_local_first(season, week)
    if pbp_err:
        st.info(f"Unable to load play-by-play: {pbp_err}")
    if not pbp.empty:
        if week is not None and "week" in pbp.columns:
            pbp = pbp[pbp["week"] == week]

        # Normalize flags
        for c in ["touchdown", "rush_touchdown", "pass_touchdown", "defensive_td", "return_touchdown"]:
            if c in pbp.columns:
                pbp[c] = pd.to_numeric(pbp[c], errors="coerce").fillna(0).astype(int)
            else:
                pbp[c] = 0

        td = pbp[
            (pbp["touchdown"] == 1)
            & ((pbp["rush_touchdown"] == 1) | (pbp["pass_touchdown"] == 1))
            & (pbp["defensive_td"] == 0)
            & (pbp["return_touchdown"] == 0)
        ].copy()

        # Determine scorer fields
        scorer_id = None
        if "td_player_id" in td.columns:
            scorer_id = td["td_player_id"]
        else:
            scorer_id = td["rusher_player_id"].where(td["rush_touchdown"] == 1, td["receiver_player_id"])
        td["scorer_id"] = scorer_id

        if "td_player_name" in td.columns:
            td["scorer_name"] = td["td_player_name"]
        else:
            td["scorer_name"] = td["rusher_player_name"].where(td["rush_touchdown"] == 1, td.get("receiver_player_name"))

        td["td_type"] = td["rush_touchdown"].map({1: "Rush", 0: None}).fillna(
            td["pass_touchdown"].map({1: "Receiving"})
        ).fillna("TD")

        # Chronology: use play_id within game, as pbp rows are ordered; safeguard with game_seconds_remaining if present
        sort_cols = ["game_id"]
        if "play_id" in td.columns:
            sort_cols.append("play_id")
        elif "game_seconds_remaining" in td.columns:
            # earliest has larger remaining seconds? Sort descending remaining
            td = td.sort_values(["game_id", "game_seconds_remaining"], ascending=[True, False])
        td = td.sort_values(sort_cols)
        first_td = td.groupby("game_id", as_index=False).head(1)

        # Map names to full display names when possible
        try:
            if "scorer_id" in first_td.columns:
                players = nfl.import_players()
                players.columns = [c.lower() for c in players.columns]
                id_col = _pick_col(players, ["gsis_id", "player_id", "nfl_id"]) or "gsis_id"
                disp_col = _pick_col(players, ["player_display_name", "display_name", "full_name"]) or "player_display_name"
                name_map = players.set_index(id_col)[disp_col].dropna().to_dict()
                first_td["scorer_name"] = first_td["scorer_id"].map(name_map).fillna(first_td["scorer_name"])
        except Exception:
            pass

        # Merge matchup from schedule parquet
        sched = _read_first(str(DATA_BASE / f"schedule/season={season}/week={week}/schedule.parquet")) if week else _read_all(str(DATA_BASE / f"schedule/season={season}/week=*/schedule.parquet"))
        if not sched.empty:
            sched.columns = [c.lower() for c in sched.columns]
            home = _pick_col(sched, ["home_team", "home_team_abbr", "home"]) or _pick_col(sched, list(sched.columns))
            away = _pick_col(sched, ["away_team", "away_team_abbr", "away"]) or _pick_col(sched, list(sched.columns))
            key = _pick_col(sched, ["game_id"]) or "game_id"
            pick_cols = [c for c in [key, home, away, "week"] if c in sched.columns]
            sched_small = sched[pick_cols].copy()
            rename_map = {key: "game_id"}
            if home in sched_small.columns:
                rename_map[home] = "home_team"
            if away in sched_small.columns:
                rename_map[away] = "away_team"
            sched_small = sched_small.rename(columns=rename_map)
            out = first_td.merge(sched_small, on="game_id", how="left")
        else:
            out = first_td.copy()

        # Harmonize column names after merge
        home_out = _pick_col(out, ["home_team", "home_team_abbr", "home"]) 
        away_out = _pick_col(out, ["away_team", "away_team_abbr", "away"]) 
        rename_final = {}
        if home_out and home_out != "home_team":
            rename_final[home_out] = "home_team"
        if away_out and away_out != "away_team":
            rename_final[away_out] = "away_team"
        if rename_final:
            out = out.rename(columns=rename_final)

        # Select output columns
        cols = []
        if "week" in out.columns:
            cols.append("week")
        if "away_team" in out.columns and "home_team" in out.columns:
            cols += ["away_team", "home_team"]
        cols += ["scorer_name"]
        team_src = _pick_col(out, ["team", "posteam", "offense_team", "td_team"]) 
        if team_src:
            if team_src != "team":
                out = out.rename(columns={team_src: "team"})
            cols += ["team"]
        cols += ["td_type"]
        if "qtr" in out.columns:
            cols.append("qtr")
        if "clock" in out.columns:
            cols.append("clock")
        if "desc" in out.columns:
            cols.append("desc")
        out = out[cols] if cols else out
        teams_opt = ["All"] + sorted(out.get("team", pd.Series([], dtype=str)).dropna().astype(str).unique().tolist())
        sel_team = st.selectbox("Filter team (First TD)", teams_opt, index=0, key="ft_team_filter_pbp")
        if sel_team != "All" and "team" in out.columns:
            out = out[out["team"].astype(str) == sel_team]
        st.dataframe(out, use_container_width=True)
    else:
        st.info("No play-by-play available locally or remotely for this selection.")
        # Fallback: show TD scorers for the week (not first) using player_week
        df = _read_all(str(DATA_BASE / f"player_week/season={season}/week=*/player_week.parquet"))
        if df.empty:
            st.info("No player-week data available for fallback.")
        else:
            if week and "week" in df.columns:
                df = df[df["week"] == week]
            name_col = _pick_col(df, ["full_name", "player_display_name", "player_name", "name", "player"]) or (
                "player_id" if "player_id" in df.columns else None
            )
            team_col = _pick_team_column(df)
            pos_col = _pick_col(df, ["position", "pos", "player_position", "role"])  
            rush_td_col = _pick_col(df, ["rushing_tds", "rushing_touchdowns", "rush_td"])  
            rec_td_col = _pick_col(df, ["receiving_tds", "receiving_touchdowns", "rec_td", "rec_tds"])
            if not all([name_col, team_col]):
                st.info("Fallback requires player name and team columns.")
            else:
                # Include RB/WR/TE. Include QBs only if they have rushing or receiving TDs
                for c in [rush_td_col, rec_td_col]:
                    if c and c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
                if pos_col and pos_col in df.columns:
                    pos_up = df[pos_col].astype(str).str.upper()
                    has_rb_wr_te = pos_up.isin(["RB", "WR", "TE", "WR/TE"])
                    qb_rush_rec = (pos_up == "QB") & (
                        (df.get(rush_td_col, 0) > 0) | (df.get(rec_td_col, 0) > 0)
                    )
                    df = df[has_rb_wr_te | qb_rush_rec]
                total_tds = (
                    (df[rush_td_col] if rush_td_col in df.columns else 0)
                    + (df[rec_td_col] if rec_td_col in df.columns else 0)
                )
                out = (
                    df.assign(total_tds=total_tds)[[name_col, team_col, "total_tds"]]
                      .groupby([name_col, team_col], as_index=False)["total_tds"].sum()
                      .sort_values("total_tds", ascending=False)
                )
                out = out.rename(columns={name_col: "player", team_col: "team"})
                st.caption("Fallback view (TD scorers for selected week; NOT first TD)")
                teams_opt = ["All"] + sorted(out["team"].astype(str).unique().tolist())
                sel_team = st.selectbox("Filter team (TD scorers)", teams_opt, index=0, key="ft_team_filter_fb")
                if sel_team != "All":
                    out = out[out["team"].astype(str) == sel_team]
                st.dataframe(out, use_container_width=True)

elif view == "Skill TDs Grid":
    st.subheader(f"RB/WR/TE Receiving + Rushing TDs — {season}{' (<= Week ' + str(week) + ')' if week else ''}")
    df = _read_all(str(DATA_BASE / f"player_week/season={season}/week=*/player_week.parquet"))
    if df.empty:
        st.info("No player-week data available for this season.")
    else:
        if "week" in df.columns and week:
            df = df[df["week"] <= week]

        # Identify columns
        name_col = _pick_col(df, ["full_name", "player_display_name", "player_name", "name", "player"]) or (
            "player_id" if "player_id" in df.columns else None
        )
        team_col = _pick_team_column(df)
        pos_col = _pick_col(df, ["position", "pos", "player_position", "role"])  
        rush_td_col = _pick_col(df, ["rushing_tds", "rushing_touchdowns", "rush_td"])  
        rec_td_col = _pick_col(df, ["receiving_tds", "receiving_touchdowns", "rec_td", "rec_tds"])

        # Enrich full names via players table when possible
        if "player_id" in df.columns:
            try:
                import nfl_data_py as nfl
                players = nfl.import_players()
                players.columns = [c.lower() for c in players.columns]
                id_col = _pick_col(players, ["gsis_id", "player_id", "nfl_id"]) or "gsis_id"
                disp_col = _pick_col(players, ["player_display_name", "display_name", "full_name"]) or "player_display_name"
                if id_col in players.columns and disp_col in players.columns:
                    name_map = players.set_index(id_col)[disp_col].dropna().to_dict()
                    df["display_name"] = df["player_id"].map(name_map)
                    df["display_name"] = df["display_name"].fillna(df.get(name_col, df["player_id"]))
                    name_col = "display_name"
            except Exception:
                pass

        # Filter to skill positions
        if pos_col and pos_col in df.columns:
            df = df[df[pos_col].astype(str).isin(["RB", "WR", "TE", "WR/TE"])]

        # Compute combined TDs
        for c in [rush_td_col, rec_td_col]:
            if c and c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        combined = (
            (df[rush_td_col] if rush_td_col in df.columns else 0)
            + (df[rec_td_col] if rec_td_col in df.columns else 0)
        )
        df = df.assign(combined_tds=combined)

        if df.empty or "week" not in df.columns or not name_col or name_col not in df.columns or not team_col or team_col not in df.columns:
            st.info("Required columns not present to build the grid.")
        else:
            df["week"] = pd.to_numeric(df["week"], errors="coerce")
            tmp = df[[name_col, team_col, "week", "combined_tds"]].dropna(subset=["week"]).copy()
            tmp["week"] = tmp["week"].astype(int)
            grouped = tmp.groupby([name_col, team_col, "week"], as_index=False)["combined_tds"].sum()
            # Build matrix
            row_keys = list(zip(grouped[name_col].astype(str), grouped[team_col].astype(str)))
            weeks = grouped["week"].astype(int).tolist()
            values = grouped["combined_tds"].astype(int).tolist()
            data: dict[tuple[str, str], dict[int, int]] = {}
            for key, w, v in zip(row_keys, weeks, values):
                if key not in data:
                    data[key] = {}
                data[key][w] = data[key].get(w, 0) + int(v)
            grid = pd.DataFrame.from_dict(data, orient="index").fillna(0).astype(int)
            grid = grid.reindex(sorted(grid.columns), axis=1)
            grid.index = pd.MultiIndex.from_tuples(grid.index, names=["player", "team"])
            grid = grid.reset_index()
            week_cols = [c for c in grid.columns if isinstance(c, int)]
            grid["Total"] = grid[week_cols].sum(axis=1).astype(int) if week_cols else 0
            grid = grid[["player", "team", "Total"] + week_cols]
            st.dataframe(grid, use_container_width=True)

elif view == "Player Week":
    st.subheader(f"Player-Week {season} Week {week}{' - ' + team if team else ''}")
    df = _read_first(str(DATA_BASE / f"player_week/season={season}/week={week}/player_week.parquet"))
    if team and not df.empty:
        tc = _pick_team_column(df)
        if tc:
            df = df[df[tc] == team]
    if df.empty:
        st.info("No player-week data for the chosen season/week. Try a different season or use the Schedule view.")
    else:
        st.dataframe(df)

elif view == "Team Leaders":
    st.subheader(f"Team Leaders {team} — {season}")
    # Load all weeks for the season for the selected team (limit files for responsiveness if needed)
    df = _read_all(str(DATA_BASE / f"player_week/season={season}/week=*/player_week.parquet"), limit_files=None)
    if df.empty:
        st.info("No player-week data found for this season. For 2025, weekly players may not be published yet.")
    else:
        tc = _pick_team_column(df)
        if tc and team:
            df = df[df[tc] == team]

        # Try multiple common column names for key stats
        stat_map = {
            "Passing Yards": ["passing_yards", "pass_yards", "pass_yds", "yards_gained_air"],
            "Rushing Yards": ["rushing_yards", "rush_yards", "rush_yds"],
            "Receiving Yards": ["receiving_yards", "rec_yards", "rec_yds", "yards_after_catch"],
        }
        name_cols = [c for c in ["player_name", "name", "player", "full_name"] if c in df.columns]
        if not name_cols:
            st.dataframe(df.head(50))
        else:
            name_col = name_cols[0]
            cols_layout = st.columns(3)
            for (label, candidates), col in zip(stat_map.items(), cols_layout):
                with col:
                    stat_col = next((c for c in candidates if c in df.columns), None)
                    if not stat_col:
                        st.write(f"{label}: N/A")
                        continue
                    top = (
                        df[[name_col, stat_col]]
                        .groupby(name_col, as_index=False)[stat_col]
                        .sum()
                        .sort_values(stat_col, ascending=False)
                        .head(10)
                    )
                    st.write(label)
                    st.dataframe(top, hide_index=True)


elif view == "Data Updates":
    st.subheader("Data Update Status")
    st.caption("When each dataset updates, where it comes from, and your current coverage.")

    seasons_sched = _list_available_seasons("schedule")
    seasons_pw = _list_available_seasons("player_week")
    seasons_pbp = _list_available_seasons("pbp")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Schedule**")
        st.write("Source: nflverse schedule (via ingest). Updates weekly on game completion.")
        st.write(f"Local seasons: {seasons_sched}")
        wks = _list_available_weeks("schedule", season) if season else []
        st.write(f"Weeks for {season}: {wks}")
    with col2:
        st.markdown("**Player Week**")
        st.write("Source: derived from play-by-play or nfl_data_py weekly players (ingest). Updates after each week.")
        st.write(f"Local seasons: {seasons_pw}")
        wks = _list_available_weeks("player_week", season) if season else []
        st.write(f"Weeks for {season}: {wks}")
    with col3:
        st.markdown("**Play-by-Play (PBP)**")
        st.write("Source: nflverse/nflfastR (remote) cached locally when available. Needed for true First TD.")
        st.write(f"Local seasons: {seasons_pbp}")
        wks = _list_available_weeks("pbp", season) if season else []
        st.write(f"Weeks for {season}: {wks}")

    st.markdown("---")
    st.markdown("### Current Week Coverage")
    cur_sched_weeks = _list_available_weeks("schedule", season)
    cur_pw_weeks = _list_available_weeks("player_week", season)
    cur_pbp_weeks = _list_available_weeks("pbp", season)
    st.write({
        "season": season,
        "schedule_weeks": cur_sched_weeks,
        "player_week_weeks": cur_pw_weeks,
        "pbp_weeks": cur_pbp_weeks,
    })

    st.markdown("---")
    st.markdown("### Actions")
    wk_options = sorted(set(cur_sched_weeks + cur_pw_weeks + cur_pbp_weeks))
    default_wk = wk_options[-1] if wk_options else None
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("**Check Remote PBP Availability**")
        target_week = st.number_input("Week to check", min_value=1, max_value=22, value=(default_wk or 1), step=1, key="upd_check_week")
        if st.button("Check Remote", key="btn_check_remote"):
            ok, msg = _remote_pbp_available(season, [int(target_week)])
            st.info(f"Available: {ok} — {msg}")
    with colB:
        st.markdown("**Ingest PBP Locally (when available)**")
        ingest_week = st.number_input("Week to ingest", min_value=1, max_value=22, value=(default_wk or 1), step=1, key="upd_ingest_week")
        if st.button("Ingest PBP", key="btn_ingest_pbp"):
            ok, msg = _ingest_pbp_local(season, [int(ingest_week)])
            st.info(f"Ingest result: {ok} — {msg}")
    with colC:
        st.markdown("**Refresh Local Coverage**")
        if st.button("Refresh Coverage", key="btn_refresh_coverage"):
            cur_sched_weeks = _list_available_weeks("schedule", season)
            cur_pw_weeks = _list_available_weeks("player_week", season)
            cur_pbp_weeks = _list_available_weeks("pbp", season)
            st.write({
                "season": season,
                "schedule_weeks": cur_sched_weeks,
                "player_week_weeks": cur_pw_weeks,
                "pbp_weeks": cur_pbp_weeks,
            })

    st.markdown("---")
    st.markdown("### How and When Data Updates")
    st.write("- Schedule & Player Week: available immediately after weekly ingest runs.")
    st.write("- Play-by-Play (PBP): available when nflverse publishes; app auto-falls back until then.")
    st.write("- First TD views require PBP. Until PBP is published, the fallbacks show RB/WR/TE TD information from player_week.")

