# Player Impact Integration Guide

## Summary

This guide explains how to integrate the Player Impact Analysis feature into the NFL Streamlit app.

## Files Created

1. **`player_impact_analytics.py`** - Backend analytics module (✅ Complete)
   - `get_significant_players()` - Get list of fantasy-relevant players
   - `calculate_team_impact()` - Team performance with/without player
   - `calculate_teammate_redistribution()` - **DFS VALUE FINDER** - Which teammates benefit when star is out
   - `calculate_opponent_impact()` - How opponents perform differently

## Integration Steps

### Step 1: Add "Player Impact" to View Dropdown

In `src/nfl_app/app/app.py`, find the View selectbox (around line 228):

```python
view = st.selectbox(
    "View",
    [
        "Overview",
        "Schedule",
        "Scores",
        # ... existing views ...
        "Team Leaders",
        "Player Impact",  # ← ADD THIS
    ],
    index=0,
)
```

### Step 2: Import Player Impact Module

At the top of `app.py`, add:

```python
import sqlite3
from pathlib import Path

# Add this import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import player_impact_analytics as pia
```

### Step 3: Add Database Connection Helper

After the existing helper functions:

```python
@st.cache_resource
def get_db_connection():
    """Get cached SQLite database connection."""
    db_path = Path(__file__).parent.parent.parent.parent / "nfl_stats.db"
    if not db_path.exists():
        st.error(f"Database not found: {db_path}")
        st.info("Run `python nfl_to_sqlite.py` to create the database")
        return None
    return sqlite3.connect(str(db_path), check_same_thread=False)
```

### Step 4: Add Player Impact View Handler

At the end of `app.py` (before or after other view handlers), add:

```python
# ========================================
# PLAYER IMPACT VIEW
# ========================================
elif view == "Player Impact":
    st.header("🚑 Player Impact Analysis")
    st.markdown(
        "Analyze how player absences affect team performance and teammate opportunities. "
        "**Perfect for identifying DFS value plays when stars are ruled out!**"
    )

    conn = get_db_connection()
    if not conn:
        st.stop()

    # Player selection controls
    impact_col1, impact_col2, impact_col3 = st.columns(3)

    with impact_col1:
        impact_season = st.number_input("Season", min_value=2023, max_value=2025, value=season, step=1)

    with impact_col2:
        impact_season_type = st.selectbox("Season Type", ["REG", "POST", "PRE"], index=0)

    with impact_col3:
        min_fantasy_pts = st.slider("Min Fantasy PPG", 5.0, 20.0, 10.0, 0.5)

    # Get significant players
    with st.spinner("Loading players..."):
        players_df = pia.get_significant_players(
            conn,
            season=impact_season,
            season_type=impact_season_type,
            min_games=5,
            min_avg_fantasy_pts=min_fantasy_pts
        )

    if players_df.empty:
        st.warning("No players found matching criteria")
        st.stop()

    # Player selector
    player_options = {
        f"{row['player_name']} ({row['position']}, {row['team']}) - {row['avg_fantasy_pts']:.1f} PPG": row['player_id']
        for _, row in players_df.iterrows()
    }

    selected_display = st.selectbox(
        "👤 Select Player to Analyze",
        options=list(player_options.keys()),
        help="Choose a player to see how their absence affects teammates"
    )

    selected_player_id = player_options[selected_display]
    selected_player_name = selected_display.split(" (")[0]

    st.divider()

    # Get absence data
    weeks_with, weeks_without = pia.get_player_absences(conn, selected_player_id, impact_season, impact_season_type)

    # Show absence info
    absence_col1, absence_col2, absence_col3 = st.columns(3)
    with absence_col1:
        st.metric("Games Played", len(weeks_with))
    with absence_col2:
        st.metric("Games Missed", len(weeks_without))
    with absence_col3:
        st.metric("Total Team Games", len(weeks_with) + len(weeks_without))

    st.caption(f"**Weeks Played:** {', '.join(map(str, weeks_with)) if weeks_with else 'None'}")
    st.caption(f"**Weeks Missed:** {', '.join(map(str, weeks_without)) if weeks_without else 'None'}")

    if len(weeks_without) == 0:
        st.info(
            f"💪 **{selected_player_name} is an iron man!**\n\n"
            f"This player hasn't missed any games in {impact_season}. "
            "Select a different player who has missed games to see teammate impact analysis."
        )
        st.stop()

    st.divider()

    # Create analysis tabs
    tab1, tab2 = st.tabs(["🚑 Teammate Impact (DFS Value!)", "🏈 Team Performance"])

    # TAB 1: TEAMMATE IMPACT (DFS GOLD!)
    with tab1:
        st.markdown(f"### When {selected_player_name} is OUT, which teammates benefit?")
        st.markdown(
            "This analysis identifies **DFS value plays** - teammates who get more opportunities "
            "and score more fantasy points when the star player is absent."
        )

        with st.spinner(f"Analyzing teammate redistribution..."):
            teammate_impacts = pia.calculate_teammate_redistribution(
                conn,
                selected_player_id,
                season=impact_season,
                season_type=impact_season_type,
                min_games=1
            )

        if not teammate_impacts:
            st.warning(
                f"Not enough data to analyze teammate impact. "
                f"{selected_player_name} needs to have missed at least 1 game for comparison."
            )
        else:
            st.success(f"✅ Found {len(teammate_impacts)} teammates with impact data")

            # Show top DFS value plays
            st.markdown("#### 🎯 Top DFS Value Plays")
            st.caption("Teammates who perform BETTER when this player is out")

            top_impacts = teammate_impacts[:10]  # Top 10

            impact_data = []
            for tm in top_impacts:
                # Color indicator based on DFS value score
                if tm.dfs_value_score >= 70:
                    indicator = "🟢🟢"  # Excellent value
                elif tm.dfs_value_score >= 60:
                    indicator = "🟢"  # Good value
                elif tm.dfs_value_score >= 50:
                    indicator = "🟡"  # Moderate value
                else:
                    indicator = "🔴"  # Avoid

                impact_data.append({
                    "": indicator,
                    "Player": f"{tm.teammate_name} ({tm.position})",
                    "PPG With": f"{tm.avg_fantasy_pts_with:.1f}",
                    "PPG Without": f"{tm.avg_fantasy_pts_without:.1f}",
                    "Δ Pts": f"{tm.fantasy_pts_delta:+.1f}",
                    "Δ Tgts": f"{tm.targets_delta:+.1f}" if tm.position in ('WR', 'TE') else "-",
                    "Δ Carries": f"{tm.carries_delta:+.1f}" if tm.position == 'RB' else "-",
                    "Sample": f"{tm.games_together}/{tm.games_apart}",
                    "DFS Score": f"{tm.dfs_value_score:.0f}/100"
                })

            impact_df = pd.DataFrame(impact_data)
            st.dataframe(impact_df, use_container_width=True, hide_index=True)

            st.caption(
                "🟢🟢 Excellent DFS value (70+) | 🟢 Good value (60-70) | 🟡 Moderate (50-60) | 🔴 Avoid (<50)\n\n"
                "**Sample** = Games together / Games apart | **DFS Score** = Overall value rating (0-100)"
            )

            # Highlight best value play
            if teammate_impacts:
                best_teammate = teammate_impacts[0]
                if best_teammate.dfs_value_score >= 60 and best_teammate.fantasy_pts_delta >= 3:
                    st.success(
                        f"💰 **Top DFS Play:** {best_teammate.teammate_name} ({best_teammate.position})\n\n"
                        f"When {selected_player_name} is OUT, {best_teammate.teammate_name} averages "
                        f"**{best_teammate.avg_fantasy_pts_without:.1f} fantasy PPG** "
                        f"(vs {best_teammate.avg_fantasy_pts_with:.1f} PPG when playing together). "
                        f"That's a **{best_teammate.fantasy_pts_delta:+.1f} point boost!**\n\n"
                        f"🎯 DFS Value Score: **{best_teammate.dfs_value_score:.0f}/100**"
                    )

                    if best_teammate.position in ('WR', 'TE') and best_teammate.targets_delta >= 2:
                        st.info(
                            f"📊 **Target Share Increase:** {best_teammate.teammate_name} sees "
                            f"**{best_teammate.targets_delta:+.1f} more targets per game** "
                            f"({best_teammate.avg_targets_without:.1f} vs {best_teammate.avg_targets_with:.1f}) "
                            f"when {selected_player_name} is out."
                        )
                    elif best_teammate.position == 'RB' and best_teammate.carries_delta >= 3:
                        st.info(
                            f"📊 **Carry Share Increase:** {best_teammate.teammate_name} gets "
                            f"**{best_teammate.carries_delta:+.1f} more carries per game** "
                            f"({best_teammate.avg_carries_without:.1f} vs {best_teammate.avg_carries_with:.1f}) "
                            f"when {selected_player_name} is out."
                        )

    # TAB 2: TEAM PERFORMANCE
    with tab2:
        st.markdown(f"### How does the team perform without {selected_player_name}?")

        with st.spinner("Calculating team impact..."):
            team_impact = pia.calculate_team_impact(
                conn,
                selected_player_id,
                selected_player_name,
                season=impact_season,
                season_type=impact_season_type
            )

        if not team_impact:
            st.warning("Not enough data for team impact analysis")
        else:
            # Team stats comparison
            team_col1, team_col2, team_col3 = st.columns(3)

            with team_col1:
                st.metric(
                    "Team PPG (With Player)",
                    f"{team_impact.team_avg_pts_with:.1f}",
                    f"{team_impact.pts_delta:+.1f}"
                )

            with team_col2:
                st.metric(
                    "Team Pass Yds (With)",
                    f"{team_impact.team_avg_pass_yds_with:.0f}",
                    f"{team_impact.pass_yds_delta:+.0f}"
                )

            with team_col3:
                st.metric(
                    "Team Rush Yds (With)",
                    f"{team_impact.team_avg_rush_yds_with:.0f}",
                    f"{team_impact.rush_yds_delta:+.0f}"
                )

            st.divider()

            # Impact interpretation
            if team_impact.pts_delta < -5:
                st.error(
                    f"🔴 **Major Impact:** {team_impact.team} scores **{abs(team_impact.pts_delta):.1f} fewer points** "
                    f"per game without {selected_player_name}. This player is critical to the offense!"
                )
            elif team_impact.pts_delta < -2:
                st.warning(
                    f"🟡 **Significant Impact:** {team_impact.team} scores **{abs(team_impact.pts_delta):.1f} fewer points** "
                    f"per game without {selected_player_name}."
                )
            elif team_impact.pts_delta > 2:
                st.info(
                    f"🟢 **Positive Impact:** Surprisingly, {team_impact.team} scores **{team_impact.pts_delta:+.1f} more points** "
                    f"per game without {selected_player_name}. Small sample size or scheme change?"
                )
            else:
                st.info(
                    f"➖ **Minimal Impact:** {team_impact.team} performs similarly with or without {selected_player_name} "
                    f"({team_impact.pts_delta:+.1f} PPG difference)."
                )
```

## Testing Locally

```bash
cd "C:\Docs\_AI Python Projects\NFL"

# Run migration if not done
python nfl_to_sqlite.py

# Run Streamlit app
streamlit run src/nfl_app/app/app.py

# Select "Player Impact" from View dropdown
# Try DeAndre Carter (WR, CHI) - he has absences
# See Keenan Allen gets +6.1 fantasy pts when Carter is out!
```

## What Users Will See

1. **View Dropdown** - Select "Player Impact"
2. **Player Selector** - Choose from fantasy-relevant players
3. **Absence Info** - See which weeks player missed
4. **Teammate Impact Tab** - **DFS GOLD!**
   - Table showing which teammates benefit when star is out
   - Fantasy point increases
   - Target/carry share increases
   - DFS Value Score (0-100)
   - Best value play highlighted
5. **Team Performance Tab** - Team scoring with/without player

## Example Output

```
Top DFS Play: Keenan Allen (WR)

When DeAndre Carter is OUT, Keenan Allen averages 13.5 fantasy PPG
(vs 7.4 PPG when playing together). That's a +6.1 point boost!

DFS Value Score: 90/100

Target Share Increase: Keenan Allen sees +2.3 more targets per game
(9.8 vs 7.5) when DeAndre Carter is out.
```

## Next Steps

1. ✅ Backend analytics module created
2. ⏳ Integrate into app.py (follow steps above)
3. ⏳ Test locally
4. ⏳ Commit and push to GitHub
5. ⏳ Deploy to Streamlit Cloud

Ready to integrate into the app!
