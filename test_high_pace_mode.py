"""
Test high-pace mode on Rams @ Seahawks Week 16 2024 game.

Actual game stats:
- Rams: 88 plays, 49 pass attempts, 457 pass yards (9.33 YPA)
- Seahawks: High scoring shootout, 75 total points

This game was severely under-projected without high-pace mode.
"""

import closed_projection_engine as cpe

def test_rams_seahawks():
    """Compare projections with and without high-pace mode."""

    # Game details
    away_team = "LA"  # Rams
    home_team = "SEA"  # Seahawks
    season = 2025  # Note: NFL labels 2024-2025 season as 2025
    week = 16
    vegas_total = 42.5  # Actual Vegas total (if available)
    spread_line = 3.5   # Seahawks favored by 3.5

    print("=" * 80)
    print("RAMS @ SEAHAWKS WEEK 16 - HIGH-PACE MODE TEST")
    print("=" * 80)
    print(f"\nActual Game Stats (Rams):")
    print(f"  Total Plays: 88")
    print(f"  Pass Attempts: 49")
    print(f"  Pass Yards: 457 (9.33 YPA)")
    print(f"  Total Points in Game: 75 (extreme outlier)")
    print()

    # --- Test 1: Without high-pace mode ---
    print("-" * 80)
    print("PROJECTION WITHOUT HIGH-PACE MODE")
    print("-" * 80)

    away_proj_base, away_players_base, away_qb_base, home_proj_base, home_players_base, home_qb_base = cpe.project_matchup(
        away_team=away_team,
        home_team=home_team,
        season=season,
        week=week,
        vegas_total=vegas_total,
        spread_line=spread_line,
        strategy='neutral',
        high_pace=False
    )

    print(f"\nRams Team Projection (Base):")
    print(f"  Total Plays: {away_proj_base.total_plays} (Actual: 88, Miss: {88 - away_proj_base.total_plays})")
    print(f"  Pass Attempts: {away_proj_base.pass_attempts} (Actual: 49, Miss: {49 - away_proj_base.pass_attempts})")
    print(f"  Pass Rate: {away_proj_base.pass_rate:.1%}")
    print(f"  Pass Yards Anchor: {away_proj_base.pass_yards_anchor:.1f} (Actual: 457, Miss: {457 - away_proj_base.pass_yards_anchor:.1f})")

    if away_qb_base:
        print(f"\nMatthew Stafford (Base):")
        print(f"  Pass Attempts: {away_qb_base.projected_pass_att}")
        print(f"  Pass Yards: {away_qb_base.projected_pass_yards:.1f}")
        print(f"  YPA: {away_qb_base.projected_ypa:.2f} (Actual: 9.33, Miss: {9.33 - away_qb_base.projected_ypa:.2f})")

    # --- Test 2: With high-pace mode ---
    print("\n" + "-" * 80)
    print("PROJECTION WITH HIGH-PACE MODE")
    print("-" * 80)

    away_proj_hp, away_players_hp, away_qb_hp, home_proj_hp, home_players_hp, home_qb_hp = cpe.project_matchup(
        away_team=away_team,
        home_team=home_team,
        season=season,
        week=week,
        vegas_total=vegas_total,
        spread_line=spread_line,
        strategy='neutral',
        high_pace=True  # Enable high-pace mode
    )

    print(f"\nRams Team Projection (High-Pace):")
    print(f"  Total Plays: {away_proj_hp.total_plays} (Actual: 88, Miss: {88 - away_proj_hp.total_plays})")
    print(f"  Pass Attempts: {away_proj_hp.pass_attempts} (Actual: 49, Miss: {49 - away_proj_hp.pass_attempts})")
    print(f"  Pass Rate: {away_proj_hp.pass_rate:.1%}")
    print(f"  Pass Yards Anchor: {away_proj_hp.pass_yards_anchor:.1f} (Actual: 457, Miss: {457 - away_proj_hp.pass_yards_anchor:.1f})")

    if away_qb_hp:
        print(f"\nMatthew Stafford (High-Pace):")
        print(f"  Pass Attempts: {away_qb_hp.projected_pass_att}")
        print(f"  Pass Yards: {away_qb_hp.projected_pass_yards:.1f}")
        print(f"  YPA: {away_qb_hp.projected_ypa:.2f} (Actual: 9.33, Miss: {9.33 - away_qb_hp.projected_ypa:.2f})")

    # --- Compare improvements ---
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 80)

    plays_improvement = (away_proj_hp.total_plays - away_proj_base.total_plays)
    pass_att_improvement = (away_proj_hp.pass_attempts - away_proj_base.pass_attempts)
    pass_yards_improvement = (away_proj_hp.pass_yards_anchor - away_proj_base.pass_yards_anchor)

    base_miss = abs(457 - away_proj_base.pass_yards_anchor)
    hp_miss = abs(457 - away_proj_hp.pass_yards_anchor)
    miss_reduction = base_miss - hp_miss
    miss_reduction_pct = (miss_reduction / base_miss) * 100 if base_miss > 0 else 0

    print(f"\nBoosts Applied:")
    print(f"  Total Plays: +{plays_improvement} plays")
    print(f"  Pass Attempts: +{pass_att_improvement} attempts")
    print(f"  Pass Yards: +{pass_yards_improvement:.1f} yards")

    print(f"\nAccuracy Improvement:")
    print(f"  Base Miss: {base_miss:.1f} yards")
    print(f"  High-Pace Miss: {hp_miss:.1f} yards")
    print(f"  Reduction: {miss_reduction:.1f} yards ({miss_reduction_pct:.1f}% improvement)")

    print(f"\nConditional Boosts Check:")
    print(f"  Spread: {spread_line} (Close game: {abs(spread_line) <= 4})")
    print(f"  Layer 2 pass rate boost applied: {abs(spread_line) <= 4}")

    # Check if both offenses are good (for Layer 4/5 boost)
    rams_ypp = (away_proj_base.total_yards_anchor / away_proj_base.total_plays) if away_proj_base.total_plays > 0 else 0
    seahawks_ypp = (home_proj_base.total_yards_anchor / home_proj_base.total_plays) if home_proj_base.total_plays > 0 else 0
    league_avg_ypp = 5.5  # Approximate

    both_good = (rams_ypp > league_avg_ypp) and (seahawks_ypp > league_avg_ypp)
    print(f"  Rams YPP: {rams_ypp:.2f}, Seahawks YPP: {seahawks_ypp:.2f}, League Avg: {league_avg_ypp:.2f}")
    print(f"  Layer 4/5 efficiency boost applied: {both_good}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if miss_reduction > 0:
        print(f"SUCCESS: High-pace mode IMPROVED projection by {miss_reduction:.1f} yards ({miss_reduction_pct:.1f}%)")
        print(f"   Still under-projects by {hp_miss:.1f} yards, but better than base {base_miss:.1f} yard miss")
        print(f"   Note: Game was extreme outlier (88 plays vs {away_proj_hp.total_plays} cap) - cannot fully predict")
    else:
        print(f"WARNING: High-pace mode did not improve projection (may be due to game conditions)")

    print()

if __name__ == "__main__":
    test_rams_seahawks()
