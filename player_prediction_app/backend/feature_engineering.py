import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Keywords indicating Did Not Play
DNP_KEYWORDS = {'Inactive', 'Did Not Play', 'Did Not Dress', 'Not With Team', ''}
SHOOTING_STATS = {
    'FG%': 'FGA',   # Field Goal %
    '3P%': '3PA',   # 3-Point %
    '2P%': '2PA',   # 2-Point %
    'FT%': 'FTA'    # Free Throw %
}
def convert_mp(mp):
    minutes, seconds = mp.split(':')
    return int(minutes) + int(seconds) / 60


def preprocess_player_df(df: pd.DataFrame, team_stats_df: pd.DataFrame, player_name: str) -> pd.DataFrame:
    """
    Raw CSV → merged, cleaned DataFrame ready for feature engineering.
    - Parses dates, converts MP string→float, filters out DNPs.
    - Merges opponent team defensive stats.
    """
    df = df.copy()
    df["Player"] = player_name
    merged = df.merge(team_stats_df, left_on='Opp', right_on='Team', how='left')
    # Parse date
    merged['Date'] = pd.to_datetime(merged['Date'], errors='coerce')
    
    # Flag home games
    merged['Home'] = merged['Unnamed: 5'].apply(lambda x: 1 if pd.isna(x) or x == '' else 0)
    merged = merged[merged['MP'].notna() & (merged['MP'] != '') & (merged["MP"] != 'Inactive') & (merged["MP"] != 'Did Not Play') & (merged["MP"] != 'Did Not Dress') & (merged["MP"] != 'Not With Team')]
    # Convert MP to minutes float
    merged['MP'] = merged['MP'].apply(lambda x: convert_mp(x) if isinstance(x, str) else 0)
    # player_career_avg = merged.groupby('Player')[['FG%', '3P%', 'FT%']].mean()
    # merged = merged.set_index('Player').fillna(player_career_avg).reset_index()
    # Numeric coercion
    merged['PTS'] = pd.to_numeric(merged['PTS'], errors='coerce').fillna(0)
    merged['FGA'] = pd.to_numeric(merged['FGA'], errors='coerce').fillna(0)

    merged.drop(columns=['Rk', 'Gcar', 'Gtm', 'Team_x', 'Unnamed: 5', 'Opp', 'Result',
        'GmSc', '+/-', 'Unnamed: 0', 'Team_y', ], inplace=True)
    # # Filter out Did Not Play
    # df = df[~df['MP'].isin([0])]
    for pct_col, attempts_col in SHOOTING_STATS.items():
        # Set percentage to 0 if attempts = 0 (no shots → 0% success)
        merged.loc[merged[attempts_col] == 0, pct_col] = 0
        
        # Force-convert to numeric (strings/empty → NaN)
        merged[pct_col] = pd.to_numeric(merged[pct_col], errors='coerce')
        
        # Fill remaining NaN (missing/invalid) with 0
        merged[pct_col] = merged[pct_col].fillna(0)
    # Merge opponent defensive stats
    # assert not merged.isna().any().any(), "NaNs detected!"
    # assert not np.isinf(merged.select_dtypes(include=np.number)).any().any(), "Infs detected!"
    return merged


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes preprocessed DataFrame (with team stats merged) and adds rolling, trend,
    efficiency and contextual features.
    """
    out = df.sort_values(['Player', 'Date']).copy()

    # Rolling aggregates
    out['PTS_last_5_avg'] = out['PTS'].rolling(5, min_periods=1).mean()
    out['MP_last_5_avg']  = out['MP'].rolling(5, min_periods=1).mean()

    # Trend (slope) helper
    def _slope(x):
        if len(x) >= 2:
            return np.polyfit(np.arange(len(x)), x, 1)[0]
        return 0

    out['PTS_trend_5'] = out['PTS_last_5_avg'].rolling(5, min_periods=1).apply(_slope, raw=False)

    # Volatility
    out['PTS_vol_5'] = out['PTS'].rolling(5, min_periods=1).std().fillna(0)

    # Efficiency: PTS per minute (5-game ewm)
    raw_eff = out['PTS'] / out['MP'].replace(0, np.nan)
    out['PTS_per_min'] = raw_eff.ewm(span=5, adjust=False).mean().fillna(0)

    # Opponent adjustment: league avg DRtg vs opp
    league_avg = out['DRtg'].mean()
    out['def_adj'] = out['PTS_last_5_avg'] * (league_avg / out['DRtg'])

    # Days of rest
    last_played = out.groupby('Player')['Date'].shift(1)
    out['Days_of_rest'] = (out['Date'] - last_played).dt.days.fillna(0)

    return out


def get_train_test_splits(feat_df: pd.DataFrame):
    """
    From feature-engineered DataFrame, split into X, X_test, X_train, y_train, y_test
    using a fixed 80/20 split.
    """
    # Define feature columns
    features = [
        'Home', 'Pace', 'eFG%_y', 'TOV%', 'DRB%',
        'PTS_last_5_avg', 'MP_last_5_avg', 'PTS_trend_5',
        'PTS_vol_5', 'PTS_per_min', 'def_adj', 'Days_of_rest'
    ]
    y = feat_df['PTS']
    X = feat_df[features]
    return train_test_split(X, y, test_size=0.2, random_state=42)

FEATURE_COLUMNS = [
    'Home', 'Pace', 'eFG%_y', 'TOV%', 'DRB%',
    'PTS_last_5_avg', 'MP_last_5_avg', 'PTS_trend_5',
    'PTS_vol_5', 'PTS_per_min', 'def_adj', 'Days_of_rest'
]

def prep_game(
    opponent: str,
    raw_player_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    player_name,
    home
) -> pd.DataFrame:
    def safe_polyfit(window: np.ndarray) -> float:
        if len(window) < 2:
            return 0.0
        try:
            return np.polyfit(np.arange(len(window)), window, 1)[0]
        except np.linalg.LinAlgError:
            return 0.0
    """
    Build the feature‐vector for a player's NEXT game vs `opponent`.
    We:
      1. Preprocess and merge historical logs.
      2. Engineer rolling/contextual features on history.
      3. Pull off the last N games to compute “next‐game” statistics.
      4. Fetch opponent stats and assemble a new 1×len(FEATURE_COLUMNS) row.
    """
    is_home_game = 1 if home == "Home" else 0
  
    # 1) Grab the last 10 real games (or fewer if <10 exist)
    last_n = raw_player_df.tail(10)
    # 2) Compute “next” rolling features from last_n
    pts_avg   = last_n['PTS'].rolling(10, min_periods=1).mean().iloc[-1]
    mp_avg    = last_n['MP'].rolling(10, min_periods=1).mean().iloc[-1]
    
    
    trend_5 = (
        last_n['PTS_last_5_avg']
        .rolling(5, min_periods=1)
        .apply(safe_polyfit, raw=False)
        .iloc[-1]
    )
    vol_5     = last_n['PTS'].rolling(5, min_periods=1).std().shift(1).fillna(0).iloc[-1]
    # ewm efficiency
    raw_eff   = last_n['PTS'] / last_n['MP'].replace(0, 1e-6)
    eff_ewm   = raw_eff.ewm(span=5, adjust=False).mean().iloc[-1]
    # days of rest
    played    = raw_player_df.loc[raw_player_df['MP']>0, 'Date']
    days_rest = (raw_player_df['Date'].max() - played.iloc[-2]).days if len(played) >= 2 else 0

    # 3) Opponent stats
    opp = team_stats_df.loc[team_stats_df['Team']==opponent].iloc[0]
    league_avg = team_stats_df['DRtg'].mean()
    def_adj    = pts_avg * (league_avg / opp['DRtg'])

    # 4) Build your one-row DataFrame
    row = {
        'Home':           is_home_game,               # or 1 if you know it’s home
        'Pace':           opp['Pace'],
        'eFG%_y':         opp['eFG%'],
        'TOV%':           opp['TOV%'],
        'DRB%':           opp['DRB%'],
        'PTS_last_5_avg': pts_avg,
        'MP_last_5_avg':  mp_avg,
        'PTS_trend_5':    trend_5,
        'PTS_vol_5':      vol_5,
        'PTS_per_min':    eff_ewm,
        'def_adj':        def_adj,
        'Days_of_rest':   days_rest
    }

    # 5) Return as a 1×N DataFrame in the exact order the model expects
    return pd.DataFrame([row])[FEATURE_COLUMNS]