"""
Script to convert team abbreviations in pfr_viewer.py from pfr format to NFLverse format
"""

import re
from pathlib import Path
import sys

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Team abbreviation mapping: pfr format → nflverse format
TEAM_MAPPING = {
    'GNB': 'GB',   # Green Bay Packers
    'KAN': 'KC',   # Kansas City Chiefs
    'LAR': 'LA',   # Los Angeles Rams
    'LVR': 'LV',   # Las Vegas Raiders
    'NOR': 'NO',   # New Orleans Saints
    'NWE': 'NE',   # New England Patriots
    'SFO': 'SF',   # San Francisco 49ers
    'TAM': 'TB',   # Tampa Bay Buccaneers
}

def convert_file(file_path):
    """Convert team abbreviations in a file"""
    print(f"Processing {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    total_replacements = 0

    for old_abbr, new_abbr in TEAM_MAPPING.items():
        # Find all occurrences with context
        # Match patterns like: 'GNB', "GNB", team='GNB', team="GNB", IN ('GNB',
        patterns = [
            (f"'{old_abbr}'", f"'{new_abbr}'"),
            (f'"{old_abbr}"', f'"{new_abbr}"'),
            (f"= '{old_abbr}'", f"= '{new_abbr}'"),
            (f'= "{old_abbr}"', f'= "{new_abbr}"'),
            (f"('{old_abbr}'", f"('{new_abbr}'"),
            (f'("{old_abbr}"', f'("{new_abbr}"'),
        ]

        for old_pattern, new_pattern in patterns:
            count = content.count(old_pattern)
            if count > 0:
                content = content.replace(old_pattern, new_pattern)
                total_replacements += count
                print(f"  Replaced {count} occurrences of {old_pattern} → {new_pattern}")

    if content != original_content:
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n✓ Total replacements: {total_replacements}")
        return total_replacements
    else:
        print("  No changes needed")
        return 0

def main():
    """Main conversion process"""
    print("="*60)
    print("TEAM ABBREVIATION CONVERTER")
    print("="*60)
    print("\nConverting from pfr format to NFLverse format:")
    for old, new in TEAM_MAPPING.items():
        print(f"  {old} → {new}")
    print()

    # Convert pfr_viewer.py
    viewer_path = Path(r"C:\Docs\_AI Python Projects\Cursor Projects\NFL - Copy\pfr_viewer.py")
    count = convert_file(viewer_path)

    print("\n" + "="*60)
    print(f"✓ CONVERSION COMPLETE - {count} total replacements")
    print("="*60)

if __name__ == "__main__":
    main()
