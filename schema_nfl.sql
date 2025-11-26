-- NFL SQLite Database Schema
-- Designed to support prediction tracking, player impact analysis, and DFS recommendations

-- =============================================================================
-- CORE DATA TABLES (migrated from Parquet)
-- =============================================================================

-- Player weekly statistics (primary table for analysis)
CREATE TABLE IF NOT EXISTS player_week_stats (
    -- Identifiers
    player_id TEXT NOT NULL,
    player_name TEXT,
    player_display_name TEXT,
    position TEXT,
    position_group TEXT,
    headshot_url TEXT,
    recent_team TEXT,

    -- Game context
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    season_type TEXT,  -- REG, POST, PRE
    opponent_team TEXT,

    -- Passing stats (QB)
    completions INTEGER DEFAULT 0,
    attempts INTEGER DEFAULT 0,
    passing_yards REAL DEFAULT 0,
    passing_tds INTEGER DEFAULT 0,
    interceptions REAL DEFAULT 0,
    sacks REAL DEFAULT 0,
    sack_yards REAL DEFAULT 0,
    sack_fumbles INTEGER DEFAULT 0,
    sack_fumbles_lost INTEGER DEFAULT 0,
    passing_air_yards REAL,
    passing_yards_after_catch REAL,
    passing_first_downs REAL,
    passing_epa REAL,
    passing_2pt_conversions INTEGER DEFAULT 0,
    pacr REAL,  -- Passing Air Conversion Ratio
    dakota REAL,  -- Dakota (QB efficiency metric)

    -- Rushing stats (RB, QB, etc.)
    carries INTEGER DEFAULT 0,
    rushing_yards REAL DEFAULT 0,
    rushing_tds INTEGER DEFAULT 0,
    rushing_fumbles REAL DEFAULT 0,
    rushing_fumbles_lost REAL DEFAULT 0,
    rushing_first_downs REAL,
    rushing_epa REAL,
    rushing_2pt_conversions INTEGER DEFAULT 0,

    -- Receiving stats (WR, TE, RB)
    receptions INTEGER DEFAULT 0,
    targets INTEGER DEFAULT 0,
    receiving_yards REAL DEFAULT 0,
    receiving_tds INTEGER DEFAULT 0,
    receiving_fumbles REAL DEFAULT 0,
    receiving_fumbles_lost REAL DEFAULT 0,
    receiving_air_yards REAL,
    receiving_yards_after_catch REAL,
    receiving_first_downs REAL,
    receiving_epa REAL,
    receiving_2pt_conversions INTEGER DEFAULT 0,
    racr REAL,  -- Receiver Air Conversion Ratio
    target_share REAL,
    air_yards_share REAL,
    wopr REAL,  -- Weighted Opportunity Rating

    -- Special teams
    special_teams_tds REAL DEFAULT 0,

    -- Fantasy points
    fantasy_points REAL DEFAULT 0,
    fantasy_points_ppr REAL DEFAULT 0,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (player_id, season, week, season_type)
);

-- Game schedule with matchup details
CREATE TABLE IF NOT EXISTS schedule (
    -- Identifiers
    game_id TEXT PRIMARY KEY,
    season INTEGER NOT NULL,
    game_type TEXT NOT NULL,  -- REG, POST, PRE
    week INTEGER NOT NULL,

    -- Game details
    gameday TEXT NOT NULL,  -- YYYY-MM-DD
    weekday TEXT NOT NULL,
    gametime TEXT,

    -- Teams & scores
    away_team TEXT NOT NULL,
    away_score REAL,
    home_team TEXT NOT NULL,
    home_score REAL,
    location TEXT,
    result REAL,  -- Point differential from home perspective
    total REAL,
    overtime REAL,

    -- External IDs
    old_game_id INTEGER,
    gsis REAL,
    nfl_detail_id TEXT,
    pfr TEXT,
    pff REAL,
    espn INTEGER,
    ftn REAL,

    -- Team rest
    away_rest INTEGER,
    home_rest INTEGER,

    -- Vegas lines (critical for projections)
    away_moneyline REAL,
    home_moneyline REAL,
    spread_line REAL,
    away_spread_odds REAL,
    home_spread_odds REAL,
    total_line REAL,
    under_odds REAL,
    over_odds REAL,

    -- Matchup factors
    div_game INTEGER,  -- 1 if divisional game

    -- Weather & venue (major impact on fantasy)
    roof TEXT,  -- dome, outdoors, closed, open
    surface TEXT,  -- grass, turf
    temp REAL,
    wind REAL,

    -- Personnel
    away_qb_id TEXT,
    home_qb_id TEXT,
    away_qb_name TEXT,
    home_qb_name TEXT,
    away_coach TEXT,
    home_coach TEXT,
    referee TEXT,

    -- Stadium
    stadium_id TEXT,
    stadium TEXT,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Injury reports (critical for DFS value identification)
CREATE TABLE IF NOT EXISTS injuries (
    -- Identifiers
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_type TEXT NOT NULL,
    team TEXT NOT NULL,
    gsis_id TEXT NOT NULL,  -- Player ID

    -- Player info
    position TEXT,
    full_name TEXT NOT NULL,
    first_name TEXT,
    last_name TEXT,

    -- Official injury report
    report_primary_injury TEXT,
    report_secondary_injury TEXT,
    report_status TEXT,  -- Out, Doubtful, Questionable, Probable

    -- Practice participation
    practice_primary_injury TEXT,
    practice_secondary_injury TEXT,
    practice_status TEXT,  -- Full, Limited, Did Not Participate

    -- Metadata
    date_modified TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (gsis_id, season, week, game_type, team)
);

-- =============================================================================
-- PREDICTION TRACKING TABLES (new functionality)
-- =============================================================================

-- Player projections and actual results
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Player & game context
    player_id TEXT NOT NULL,
    player_name TEXT NOT NULL,
    position TEXT NOT NULL,
    team TEXT NOT NULL,
    opponent TEXT NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    season_type TEXT NOT NULL DEFAULT 'REG',
    game_id TEXT,

    -- Projected stats (position-specific)
    proj_fantasy_points REAL,
    proj_fantasy_points_ppr REAL,
    proj_confidence REAL,  -- 0-100 confidence score
    proj_floor REAL,  -- 10th percentile
    proj_ceiling REAL,  -- 90th percentile

    -- QB projections
    proj_pass_yards REAL,
    proj_pass_tds REAL,
    proj_interceptions REAL,
    proj_rush_yards REAL,
    proj_rush_tds REAL,

    -- RB/WR/TE projections
    proj_carries REAL,
    proj_targets REAL,
    proj_receptions REAL,
    proj_rec_yards REAL,
    proj_rec_tds REAL,

    -- DFS metrics
    dfs_score REAL,  -- Overall DFS value score (0-100)
    dfs_grade TEXT,  -- A+, A, B+, B, C+, C, D, F
    dfs_salary REAL,  -- DraftKings/FanDuel salary
    dfs_value REAL,  -- Points per $1000
    dfs_ownership REAL,  -- Projected ownership %

    -- Projection factors (for transparency)
    matchup_quality REAL,  -- 0-100 (opponent defense ranking)
    recent_form REAL,  -- Last 3-5 games trend
    weather_impact REAL,  -- Weather adjustment factor
    injury_risk REAL,  -- 0-100 injury concern

    -- Actual results (filled after game)
    actual_fantasy_points REAL,
    actual_fantasy_points_ppr REAL,
    actual_pass_yards REAL,
    actual_pass_tds REAL,
    actual_interceptions REAL,
    actual_rush_yards REAL,
    actual_rush_tds REAL,
    actual_carries REAL,
    actual_targets REAL,
    actual_receptions REAL,
    actual_rec_yards REAL,
    actual_rec_tds REAL,

    -- Accuracy metrics (calculated after game)
    prediction_error REAL,  -- abs(actual - projected)
    prediction_error_pct REAL,  -- (actual - projected) / projected
    accuracy_grade TEXT,  -- How accurate was projection

    -- Metadata
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    game_date TEXT,
    result_updated_at TIMESTAMP,
    notes TEXT,

    FOREIGN KEY (player_id, season, week, season_type)
        REFERENCES player_week_stats(player_id, season, week, season_type),
    FOREIGN KEY (game_id) REFERENCES schedule(game_id)
);

-- =============================================================================
-- INDEXES for performance
-- =============================================================================

-- Player week stats indexes
CREATE INDEX IF NOT EXISTS idx_player_week_player ON player_week_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_player_week_season_week ON player_week_stats(season, week);
CREATE INDEX IF NOT EXISTS idx_player_week_team ON player_week_stats(recent_team);
CREATE INDEX IF NOT EXISTS idx_player_week_position ON player_week_stats(position);
CREATE INDEX IF NOT EXISTS idx_player_week_composite ON player_week_stats(player_id, season, season_type);

-- Schedule indexes
CREATE INDEX IF NOT EXISTS idx_schedule_season_week ON schedule(season, week);
CREATE INDEX IF NOT EXISTS idx_schedule_teams ON schedule(away_team, home_team);
CREATE INDEX IF NOT EXISTS idx_schedule_gameday ON schedule(gameday);

-- Injuries indexes
CREATE INDEX IF NOT EXISTS idx_injuries_player ON injuries(gsis_id);
CREATE INDEX IF NOT EXISTS idx_injuries_team ON injuries(team, season, week);
CREATE INDEX IF NOT EXISTS idx_injuries_status ON injuries(report_status);

-- Predictions indexes
CREATE INDEX IF NOT EXISTS idx_predictions_player ON predictions(player_id);
CREATE INDEX IF NOT EXISTS idx_predictions_week ON predictions(season, week);
CREATE INDEX IF NOT EXISTS idx_predictions_position ON predictions(position);
CREATE INDEX IF NOT EXISTS idx_predictions_dfs_score ON predictions(dfs_score DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_accuracy ON predictions(prediction_error);

-- =============================================================================
-- VIEWS for common queries
-- =============================================================================

-- Top DFS values for current week
CREATE VIEW IF NOT EXISTS v_top_dfs_values AS
SELECT
    p.player_name,
    p.position,
    p.team,
    p.opponent,
    p.proj_fantasy_points_ppr as proj_points,
    p.dfs_score,
    p.dfs_grade,
    p.dfs_value as points_per_1k,
    p.proj_confidence as confidence,
    s.gameday,
    s.roof,
    s.temp,
    s.wind,
    s.total_line as game_total
FROM predictions p
LEFT JOIN schedule s ON p.game_id = s.game_id
WHERE p.actual_fantasy_points IS NULL  -- Only未completed games
ORDER BY p.dfs_score DESC;

-- Player season averages
CREATE VIEW IF NOT EXISTS v_player_season_averages AS
SELECT
    player_id,
    player_display_name,
    position,
    recent_team,
    season,
    season_type,
    COUNT(*) as games_played,
    AVG(fantasy_points) as avg_fantasy_points,
    AVG(fantasy_points_ppr) as avg_fantasy_points_ppr,
    AVG(targets) as avg_targets,
    AVG(carries) as avg_carries,
    AVG(passing_yards) as avg_pass_yards,
    AVG(rushing_yards) as avg_rush_yards,
    AVG(receiving_yards) as avg_rec_yards,
    SUM(passing_tds + rushing_tds + receiving_tds) as total_tds
FROM player_week_stats
WHERE season_type = 'REG'
GROUP BY player_id, season, season_type;

-- Prediction accuracy by position
CREATE VIEW IF NOT EXISTS v_prediction_accuracy AS
SELECT
    position,
    COUNT(*) as total_predictions,
    AVG(prediction_error) as avg_error,
    AVG(prediction_error_pct) as avg_error_pct,
    AVG(CASE WHEN ABS(prediction_error_pct) < 0.10 THEN 1.0 ELSE 0.0 END) as pct_within_10,
    AVG(CASE WHEN ABS(prediction_error_pct) < 0.20 THEN 1.0 ELSE 0.0 END) as pct_within_20
FROM predictions
WHERE actual_fantasy_points IS NOT NULL
GROUP BY position;
