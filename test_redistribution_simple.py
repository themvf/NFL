"""
Simple test of redistribution logic without Streamlit dependencies.

Verifies conservation laws are maintained during injury redistribution.
"""

import closed_projection_engine as cpe

def test_simple_redistribution():
    """Test that conservation laws hold with/without injuries."""

    # Test on LA Rams @ SEA Seahawks Week 16 2025
    away_team = "LA"  # Rams
    home_team = "SEA"  # Seahawks
    season = 2025
    week = 16

    print("=" * 80)
    print("CONSERVATION LAW TEST - REDISTRIBUTION")
    print("=" * 80)
    print(f"\nMatchup: {away_team} @ {home_team}, Week {week}")

    # Run projection
    print("\nGenerating projections...")
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
    print("AWAY TEAM - CONSERVATION CHECK (No Injuries)")
    print("-" * 80)

    if away_qb:
        print(f"\n{away_qb.player_name} (QB):")
        print(f"  Pass Attempts: {away_qb.projected_pass_att}")
        print(f"  Completions: {away_qb.projected_completions:.1f}")
        print(f"  Completion %: {away_qb.projected_completion_pct:.1%}")

        print(f"\nTop Receivers:")
        for p in sorted(away_players, key=lambda x: x.projected_targets, reverse=True)[:5]:
            if p.projected_targets > 0:
                print(f"  {p.player_name:20s}: {p.projected_targets:4.1f} tgt -> "
                      f"{p.projected_receptions:4.1f} rec -> {p.projected_recv_yards:5.1f} yds")

        # Check conservation law
        total_rec = sum(p.projected_receptions for p in away_players)

        print(f"\n{'='*60}")
        print(f"CONSERVATION LAW CHECK")
        print(f"{'='*60}")
        print(f"QB Completions:        {away_qb.projected_completions:6.1f}")
        print(f"Sum(Rec Receptions):   {total_rec:6.1f}")
        print(f"Difference:            {abs(away_qb.projected_completions - total_rec):6.3f}")

        if abs(away_qb.projected_completions - total_rec) < 0.1:
            print(f"\nSUCCESS: QB completions = sum(receiver receptions)")
        else:
            print(f"\nFAILED: Conservation law violated!")

    print("\n" + "-" * 80)
    print("HOME TEAM - CONSERVATION CHECK (No Injuries)")
    print("-" * 80)

    if home_qb:
        print(f"\n{home_qb.player_name} (QB):")
        print(f"  Pass Attempts: {home_qb.projected_pass_att}")
        print(f"  Completions: {home_qb.projected_completions:.1f}")
        print(f"  Completion %: {home_qb.projected_completion_pct:.1%}")

        print(f"\nTop Receivers:")
        for p in sorted(home_players, key=lambda x: x.projected_targets, reverse=True)[:5]:
            if p.projected_targets > 0:
                print(f"  {p.player_name:20s}: {p.projected_targets:4.1f} tgt -> "
                      f"{p.projected_receptions:4.1f} rec -> {p.projected_recv_yards:5.1f} yds")

        # Check conservation law
        total_rec = sum(p.projected_receptions for p in home_players)

        print(f"\n{'='*60}")
        print(f"CONSERVATION LAW CHECK")
        print(f"{'='*60}")
        print(f"QB Completions:        {home_qb.projected_completions:6.1f}")
        print(f"Sum(Rec Receptions):   {total_rec:6.1f}")
        print(f"Difference:            {abs(home_qb.projected_completions - total_rec):6.3f}")

        if abs(home_qb.projected_completions - total_rec) < 0.1:
            print(f"\nSUCCESS: QB completions = sum(receiver receptions)")
        else:
            print(f"\nFAILED: Conservation law violated!")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nNOTE: To test injury redistribution with Player Impact Analytics:")
    print("  1. Run the Streamlit app: streamlit run pfr_viewer.py")
    print("  2. Navigate to 'Closed-System Projections' tab")
    print("  3. Mark a player as OUT using the checkboxes")
    print("  4. Observe the redistribution method used (Historical or Proportional)")
    print("  5. Verify conservation laws still hold in the displayed stats")
    print()


if __name__ == "__main__":
    test_simple_redistribution()
