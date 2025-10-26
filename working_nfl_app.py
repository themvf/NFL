from flask import Flask, request, render_template_string
import pandas as pd
from pathlib import Path
import glob

app = Flask(__name__)
DATA_BASE = Path("data/processed").resolve()

def load_basic_data(season, week, team_a, team_b):
    """Load basic player data for comparison."""
    try:
        # Try to load player-week data
        path = str(DATA_BASE / f"player_week/season={season}/week={week}/player_week.parquet")
        if Path(path).exists():
            df = pd.read_parquet(path)
            
            # Find team column
            team_col = None
            for col in ["recent_team", "team", "team_abbr", "player_team", "club"]:
                if col in df.columns:
                    team_col = col
                    break
            
            if team_col:
                # Filter to selected teams
                df = df[df[team_col].isin([team_a, team_b])]
                
                # Get key columns
                display_cols = [team_col]
                
                # Add player name
                name_col = None
                for col in ["player_name", "full_name", "name", "player"]:
                    if col in df.columns:
                        name_col = col
                        display_cols.append(col)
                        break
                
                # Add position
                pos_col = None
                for col in ["position", "pos", "role"]:
                    if col in df.columns:
                        pos_col = col
                        display_cols.append(col)
                        break
                
                # Add stats
                stat_cols = ["passing_yards", "rushing_yards", "receiving_yards"]
                for col in stat_cols:
                    if col in df.columns:
                        display_cols.append(col)
                
                if len(display_cols) > 1:
                    result = df[display_cols].head(20)
                    return result.to_dict('records')
        
        return []
    except Exception as e:
        return [{"error": str(e)}]

@app.route('/')
def index():
    season = int(request.args.get('season', 2025))
    week = int(request.args.get('week', 3))
    team_a = request.args.get('team_a', 'ARI')
    team_b = request.args.get('team_b', 'ATL')
    
    data = load_basic_data(season, week, team_a, team_b)
    
    return render_template_string(TEMPLATE, 
                                  season=season, week=week, team_a=team_a, team_b=team_b,
                                  player_data=data)

TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>NFL Data Browser - Working Version</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .controls { background: #f0f0f0; padding: 15px; margin-bottom: 20px; }
        .controls select, .controls button { margin: 5px; padding: 5px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .error { color: red; font-weight: bold; }
    </style>
</head>
<body>
    <h1>NFL Data Browser - Working Version</h1>
    
    <div class="controls">
        <form method="get">
            <label>Season:</label>
            <select name="season">
                <option value="2023" {% if season == 2023 %}selected{% endif %}>2023</option>
                <option value="2024" {% if season == 2024 %}selected{% endif %}>2024</option>
                <option value="2025" {% if season == 2025 %}selected{% endif %}>2025</option>
            </select>
            
            <label>Week:</label>
            <select name="week">
                {% for w in range(1, 19) %}
                <option value="{{ w }}" {% if week == w %}selected{% endif %}>{{ w }}</option>
                {% endfor %}
            </select>
            
            <label>Team A:</label>
            <select name="team_a">
                {% for team in ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"] %}
                <option value="{{ team }}" {% if team_a == team %}selected{% endif %}>{{ team }}</option>
                {% endfor %}
            </select>
            
            <label>Team B:</label>
            <select name="team_b">
                {% for team in ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"] %}
                <option value="{{ team }}" {% if team_b == team %}selected{% endif %}>{{ team }}</option>
                {% endfor %}
            </select>
            
            <button type="submit">Load Data</button>
        </form>
    </div>
    
    <h2>Player Data: {{ team_a }} vs {{ team_b }} (Season {{ season }}, Week {{ week }})</h2>
    
    {% if player_data %}
        {% if player_data[0].get('error') %}
            <div class="error">Error: {{ player_data[0].error }}</div>
        {% else %}
            <table>
                <tr>
                    {% for key in player_data[0].keys() %}
                    <th>{{ key }}</th>
                    {% endfor %}
                </tr>
                {% for player in player_data %}
                <tr>
                    {% for value in player.values() %}
                    <td>{{ value }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </table>
        {% endif %}
    {% else %}
        <p>No data found for the selected criteria.</p>
        <p>Available data directory: {{ "data/processed" }}</p>
    {% endif %}
</body>
</html>
'''

if __name__ == '__main__':
    print("=" * 50)
    print("NFL DATA BROWSER - WORKING VERSION")
    print("Open: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
