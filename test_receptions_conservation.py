"""
Test Phase 3: Receptions Tracking and QB-Receiver Conservation Law

Verifies that QB completions = sum(receiver receptions)
"""

import closed_projection_engine as cpe

def test_receptions_conservation():
    """Test QB completions conservation on a sample matchup."""

    # Test on LA Rams @ SEA Seahawks Week 16 2025
    away_team = "LA"  # Rams
    home_team = "SEA"  # Seahawks
    season = 2025
    week = 16

    print("=" * 80)
    print("PHASE 3: RECEPTIONS CONSERVATION TEST")
    print("=" * 80)
    print(f"\nMatchup: {away_team} @ {home_team}, Week {week}")

    # Run projection
    away_proj, away_players, away_qb, home_proj, home_players, home_qb = cpe.project_matchup(
        away_team=away_team,
        home_team=home_team,
        season=season,
        week=week,
        vegas_total=42.5,
        spread_line=3.5,
        strategy='neutral',
        high_pace=False
    )

    print("\n" + "-" * 80)
    print(f"{away_team} (AWAY) - RECEPTIONS CONSERVATION CHECK")
    print("-" * 80)

    # Calculate sum of all receiver receptions
    total_receptions = sum(p.projected_receptions for p in away_players)

    if away_qb:
        print(f"\n{away_qb.player_name} (QB):")
        print(f"  Pass Attempts: {away_qb.projected_pass_att}")
        print(f"  Completions: {away_qb.projected_completions:.1f}")
        print(f"  Completion %: {away_qb.projected_completion_pct:.1%}")

        print(f"\nReceiver Receptions Breakdown:")
        for p in sorted(away_players, key=lambda x: x.projected_receptions, reverse=True)[:8]:
            if p.projected_receptions > 0:
                catch_rate_display = p.projected_catch_rate * 100
                print(f"  {p.player_name:20s} ({p.position}): "
                      f"{p.projected_targets:4.1f} tgt -> {p.projected_receptions:4.1f} rec "
                      f"({catch_rate_display:.0f}% CR) = {p.projected_recv_yards:5.1f} yds "
                      f"({p.projected_ypr:.1f} YPR)")

        print(f"\n{'='*60}")
        print(f"CONSERVATION LAW CHECK:")
        print(f"{'='*60}")
        print(f"QB Completions:        {away_qb.projected_completions:6.1f}")
        print(f"Sum(Rec Receptions):   {total_receptions:6.1f}")
        print(f"Difference:            {abs(away_qb.projected_completions - total_receptions):6.3f}")

        if abs(away_qb.projected_completions - total_receptions) < 0.1:
            print(f"\nSUCCESS: QB completions = sum(receiver receptions)")
        else:
            print(f"\nFAILED: Conservation law violated!")
    else:
        print("No QB projection available")

    # Same check for home team
    print("\n" + "-" * 80)
    print(f"{home_team} (HOME) - RECEPTIONS CONSERVATION CHECK")
    print("-" * 80)

    total_receptions_home = sum(p.projected_receptions for p in home_players)

    if home_qb:
        print(f"\n{home_qb.player_name} (QB):")
        print(f"  Pass Attempts: {home_qb.projected_pass_att}")
        print(f"  Completions: {home_qb.projected_completions:.1f}")
        print(f"  Completion %: {home_qb.projected_completion_pct:.1%}")

        print(f"\nReceiver Receptions Breakdown:")
        for p in sorted(home_players, key=lambda x: x.projected_receptions, reverse=True)[:8]:
            if p.projected_receptions > 0:
                catch_rate_display = p.projected_catch_rate * 100
                print(f"  {p.player_name:20s} ({p.position}): "
                      f"{p.projected_targets:4.1f} tgt -> {p.projected_receptions:4.1f} rec "
                      f"({catch_rate_display:.0f}% CR) = {p.projected_recv_yards:5.1f} yds "
                      f"({p.projected_ypr:.1f} YPR)")

        print(f"\n{'='*60}")
        print(f"CONSERVATION LAW CHECK:")
        print(f"{'='*60}")
        print(f"QB Completions:        {home_qb.projected_completions:6.1f}")
        print(f"Sum(Rec Receptions):   {total_receptions_home:6.1f}")
        print(f"Difference:            {abs(home_qb.projected_completions - total_receptions_home):6.3f}")

        if abs(home_qb.projected_completions - total_receptions_home) < 0.1:
            print(f"\n✓ SUCCESS: QB completions = sum(receiver receptions)")
        else:
            print(f"\n✗ FAILED: Conservation law violated!")
    else:
        print("No QB projection available")

    print("\n" + "=" * 80)
    print("PHASE 3 TEST COMPLETE")
    print("=" * 80)
    print()

if __name__ == "__main__":
    test_receptions_conservation()
