# Projection Ranges & High-Pace Mode - Implementation Spec

## Phase 1: Projection Ranges (P10/P50/P90)

### Goal
Show uncertainty for all projections using percentile-based ranges derived from statistical volatility.

### Data Model Changes

#### Add to PlayerProjection dataclass:
```python
# Current (median/P50)
projected_carries: float
projected_ypc: float
projected_rush_yards: float

# Add ranges
projected_carries_p10: float
projected_carries_p90: float
projected_ypc_p10: float
projected_ypc_p90: float
projected_rush_yards_p10: float
projected_rush_yards_p90: float

# Same for receiving
projected_targets_p10: float
projected_targets_p90: float
projected_ypt_p10: float
projected_ypt_p90: float
projected_recv_yards_p10: float
projected_recv_yards_p90: float
```

#### Add to QBProjection dataclass:
```python
# Current
projected_pass_att: int
projected_pass_yards: float

# Add ranges
projected_pass_att_p10: int
projected_pass_att_p90: int
projected_pass_yards_p10: float
projected_pass_yards_p90: float
```

### Calculation Method

#### Volume Volatility (Carries, Targets)
```python
def calculate_volume_volatility(player_name, team, season, week, stat='carries'):
    """
    Calculate coefficient of variation (CV) for player volume.

    CV = std_dev / mean

    Returns:
        float: CV for recent games (last 4-8 weeks)
    """
    # Get recent stats
    recent_stats = get_player_recent_stats(player_name, team, season, week, lookback=6)

    # Calculate CV
    mean_vol = recent_stats[stat].mean()
    std_vol = recent_stats[stat].std()
    cv = std_vol / mean_vol if mean_vol > 0 else 0.3  # default 30% CV

    # Typical RB carry CV: 0.15-0.35 (15-35% volatility)
    # Typical WR target CV: 0.25-0.45 (higher volatility)

    return cv
```

#### Efficiency Volatility (YPC, YPT)
```python
def calculate_efficiency_volatility(player_name, team, season, week, stat='ypc'):
    """
    Calculate CV for player efficiency.

    Returns:
        float: CV for recent games
    """
    # Get recent stats
    recent_stats = get_player_recent_stats(player_name, team, season, week, lookback=6)

    # Calculate YPC or YPT
    if stat == 'ypc':
        efficiency = recent_stats['rushing_yards'] / recent_stats['carries']
    else:  # ypt
        efficiency = recent_stats['receiving_yards'] / recent_stats['targets']

    mean_eff = efficiency.mean()
    std_eff = efficiency.std()
    cv = std_eff / mean_eff if mean_eff > 0 else 0.25

    # Typical YPC CV: 0.20-0.40
    # Typical YPT CV: 0.15-0.30

    return cv
```

#### Apply Ranges to Projections
```python
def apply_projection_ranges(player_proj: PlayerProjection, vol_cv: float, eff_cv: float):
    """
    Calculate P10 and P90 bounds using normal distribution approximation.

    For normal distribution:
    - P10 ≈ mean - 1.28 * std
    - P90 ≈ mean + 1.28 * std

    Since CV = std/mean, then std = CV * mean
    """
    # Volume ranges (carries or targets)
    if player_proj.position == 'RB':
        # Carries
        std_carries = vol_cv * player_proj.projected_carries
        player_proj.projected_carries_p10 = max(0, player_proj.projected_carries - 1.28 * std_carries)
        player_proj.projected_carries_p90 = player_proj.projected_carries + 1.28 * std_carries

        # YPC
        std_ypc = eff_cv * player_proj.projected_ypc
        player_proj.projected_ypc_p10 = max(2.0, player_proj.projected_ypc - 1.28 * std_ypc)
        player_proj.projected_ypc_p90 = min(10.0, player_proj.projected_ypc + 1.28 * std_ypc)

        # Rush yards (compound: carries × YPC)
        # Variance of product: Var(X*Y) ≈ E[X]²Var[Y] + E[Y]²Var[X]
        # Simplified: use combined CV
        combined_cv = math.sqrt(vol_cv**2 + eff_cv**2)
        std_rush_yds = combined_cv * player_proj.projected_rush_yards
        player_proj.projected_rush_yards_p10 = max(0, player_proj.projected_rush_yards - 1.28 * std_rush_yds)
        player_proj.projected_rush_yards_p90 = player_proj.projected_rush_yards + 1.28 * std_rush_yds

    # Same logic for receiving (targets, YPT, receiving yards)
    if player_proj.position in ['WR', 'TE'] or player_proj.projected_targets > 0:
        std_targets = vol_cv * player_proj.projected_targets
        player_proj.projected_targets_p10 = max(0, player_proj.projected_targets - 1.28 * std_targets)
        player_proj.projected_targets_p90 = player_proj.projected_targets + 1.28 * std_targets

        std_ypt = eff_cv * player_proj.projected_ypt
        player_proj.projected_ypt_p10 = max(4.0, player_proj.projected_ypt - 1.28 * std_ypt)
        player_proj.projected_ypt_p90 = min(15.0, player_proj.projected_ypt + 1.28 * std_ypt)

        combined_cv = math.sqrt(vol_cv**2 + eff_cv**2)
        std_recv_yds = combined_cv * player_proj.projected_recv_yards
        player_proj.projected_recv_yards_p10 = max(0, player_proj.projected_recv_yards - 1.28 * std_recv_yds)
        player_proj.projected_recv_yards_p90 = player_proj.projected_recv_yards + 1.28 * std_recv_yds
```

### UI Display

#### Player Table with Ranges:
```
| Player          | Proj Yds  | Range (P10-P90) | Volatility |
|-----------------|-----------|-----------------|------------|
| David Montgomery| 72.0      | 52 - 92         | Medium     |
| Jahmyr Gibbs    | 48.0      | 30 - 66         | High       |
```

#### Expanded View:
```
David Montgomery (DET RB)
  Carries:    15.0  (12.5 - 17.5)
  YPC:        4.8   (3.9 - 5.7)
  Rush Yds:   72.0  (52 - 92)      ◄── 80% confidence band
  DVOA:       +12.5%

  Volatility: Medium (CV=28%)

  📊 Interpretation:
  - 10% chance of < 52 yards
  - 50% chance of ~72 yards (median)
  - 10% chance of > 92 yards
```

---

## Phase 2: High-Pace / High-Variance Toggle

### Goal
Allow users to signal expectation of fast-paced, high-scoring game with data-driven conditional boosts.

### UI Component
```python
# In Team Comparison view, below strategy selector
high_pace = st.checkbox(
    "🏃 High-Pace Game",
    value=False,
    help="Select if expecting faster tempo, more plays, or shootout conditions"
)
```

### Boost Logic (Conditional & Capped)

#### Layer 1: Total Plays
```python
def predict_team_plays(team, opponent, season, week, vegas_total, spread_line, is_home, high_pace=False):
    # ... existing logic ...

    if high_pace:
        # Boost baseline by 3 plays
        baseline_plays += 3

        # Increase cap from 75 to 82
        plays_cap = 82
    else:
        plays_cap = 75

    TeamPlays = clamp(baseline_plays + vegas_adj + spread_adj, 55, plays_cap)
    return TeamPlays
```

#### Layer 2: Pass Rate
```python
def split_plays_pass_rush(team, season, week, total_plays, spread_line, is_home, high_pace=False):
    # ... existing logic ...

    # Conditional boost: only if close game (spread ≤ 4)
    if high_pace and abs(spread_line) <= 4:
        game_pass_rate += 0.02  # +2 percentage points

    game_pass_rate = clamp(game_pass_rate, 0.45, 0.75)

    TeamPassAtt = round(total_plays * game_pass_rate)
    TeamRushAtt = total_plays - TeamPassAtt
    return TeamPassAtt, TeamRushAtt
```

#### Layer 4/5: Efficiency
```python
def compute_team_anchors(team, opponent, season, week, total_plays, pass_rate, strategy='neutral', high_pace=False):
    # ... existing logic ...

    # Conditional boost: only if both offenses above league average
    team_off_ypp = get_team_offensive_ypp(team, season, week)
    opp_off_ypp = get_team_offensive_ypp(opponent, season, week)
    league_avg_ypp = get_league_avg_ypp(season, week)

    both_offenses_good = (team_off_ypp > league_avg_ypp) and (opp_off_ypp > league_avg_ypp)

    if high_pace and both_offenses_good:
        # +3% efficiency boost (less defensive pressure in shootouts)
        expected_ypp *= 1.03

    TeamTotalYards_anchor = expected_ypp * total_plays
    # ... rest of logic ...
```

### Expected Impact Example

**Rams @ Seahawks with High-Pace Toggle:**

| Metric | Base | High-Pace | Actual | Delta |
|--------|------|-----------|--------|-------|
| Total Plays | 65 | 68 (+3) | 88 | Still -20 |
| Pass Att | 37 | 40 (+3) | 49 | -9 |
| Pass Yards | 278 | 304 (+26) | 457 | -153 |

**Improvement**: 26 yards closer (179 → 153 yard miss)

**Key**: Still can't fully predict extreme outliers, but reduces miss by ~15%.

---

## Implementation Order

### Step 1: Add Range Fields to Dataclasses
- Update `PlayerProjection` in `closed_projection_engine.py`
- Update `QBProjection` in `closed_projection_engine.py`
- Add default values (P10=P50, P90=P50) initially

### Step 2: Implement Volatility Calculation
- Add `calculate_volume_volatility()` function
- Add `calculate_efficiency_volatility()` function
- Add `apply_projection_ranges()` function

### Step 3: Integrate Ranges into Projection Pipeline
- Call volatility functions in `allocate_rushing_volume()`
- Call volatility functions in `allocate_passing_volume()`
- Apply ranges before reconciliation (ranges scale proportionally)

### Step 4: Update UI (pfr_viewer.py)
- Add ranges to player tables
- Add expandable detail views
- Add volatility indicators

### Step 5: Add High-Pace Toggle
- Add checkbox to UI
- Pass `high_pace` parameter through all layers
- Implement conditional boosts in Layers 1, 2, 4/5

### Step 6: Testing & Calibration
- Test on historical games (Rams-Seahawks, etc.)
- Verify ranges contain ~80% of actual results
- Adjust CV defaults if needed

---

## Data Requirements

### New Helper Functions:
```python
def get_player_recent_stats(player_name, team, season, week, lookback=6):
    """Get player's last N games for volatility calculation."""
    # Query player_stats table
    # Return DataFrame with carries, targets, yards, etc.

def get_league_avg_ypp(season, week):
    """Get league-average yards per play for reference."""
    # Query team_stats_week
    # Calculate median YPP across all teams

def get_team_offensive_ypp(team, season, week):
    """Get team's offensive efficiency (yards per play)."""
    # Query team_stats_week
    # Return team's average YPP in recent games
```

---

## Success Metrics

### Phase 1 (Ranges):
- ✅ P10-P90 band contains 80% of actual results
- ✅ Volatility CV matches historical patterns (RB: 15-35%, WR: 25-45%)
- ✅ User feedback: "Ranges help me understand uncertainty"

### Phase 2 (High-Pace):
- ✅ High-pace games (Vegas 48+) improve by 10-20 yards on average
- ✅ Non-high-pace games unaffected (no false positives)
- ✅ Boosts are conditional (don't apply blindly)

---

## Notes

- Ranges are applied **within reconciliation** (conserved totals remain conserved)
- High-pace boosts are **data-driven** (require specific game conditions)
- Everything still **sums up perfectly** (conservation laws maintained)
- No "magic outlier button" - user makes informed decision based on Vegas/matchup
