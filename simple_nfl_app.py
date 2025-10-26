import sys
sys.path.append('src')

import gradio as gr
import pandas as pd
from pathlib import Path
import glob

DATA_BASE = Path("data/processed").resolve()

def load_player_data(season, week, team):
    """Load player data for a specific season/week/team."""
    try:
        # Load player-week data
        path = str(DATA_BASE / f"player_week/season={season}/week={week}/player_week.parquet")
        if Path(path).exists():
            df = pd.read_parquet(path)
            # Filter by team if specified
            team_cols = ["recent_team", "team", "team_abbr", "player_team", "club"]
            team_col = None
            for col in team_cols:
                if col in df.columns:
                    team_col = col
                    break
            
            if team_col and team != "All":
                df = df[df[team_col] == team]
            
            if df.empty:
                return "No data found for the selected criteria."
            
            # Display first 10 rows with key columns
            display_cols = []
            if "player_name" in df.columns:
                display_cols.append("player_name")
            elif "name" in df.columns:
                display_cols.append("name")
            
            if team_col:
                display_cols.append(team_col)
                
            stat_cols = ["passing_yards", "rushing_yards", "receiving_yards"]
            for col in stat_cols:
                if col in df.columns:
                    display_cols.append(col)
            
            if display_cols:
                result_df = df[display_cols].head(10)
                return result_df.to_html(index=False)
            else:
                return f"Data found but no recognizable columns. Columns: {list(df.columns[:10])}"
        else:
            return f"No data file found at: {path}"
    except Exception as e:
        return f"Error loading data: {str(e)}"

# Get available seasons and teams
seasons = [2023, 2024, 2025]
weeks = list(range(1, 19))
teams = ["All", "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"]

# Create Gradio interface
with gr.Blocks(title="NFL Data Viewer") as app:
    gr.Markdown("# NFL Player Data Viewer")
    gr.Markdown("Simple viewer for NFL player-week data")
    
    with gr.Row():
        season_input = gr.Dropdown(choices=seasons, value=2025, label="Season")
        week_input = gr.Dropdown(choices=weeks, value=3, label="Week") 
        team_input = gr.Dropdown(choices=teams, value="ARI", label="Team")
    
    load_btn = gr.Button("Load Data", variant="primary")
    output = gr.HTML()
    
    load_btn.click(
        load_player_data,
        inputs=[season_input, week_input, team_input],
        outputs=[output]
    )

if __name__ == "__main__":
    print("Starting NFL Data Viewer...")
    print("This will open in your web browser at http://127.0.0.1:7860")
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )
